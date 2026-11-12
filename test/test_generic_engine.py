"""G4 data integration into the shared scheduler, not a second engine."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from simulator import shared_engine as e
from simulator import gen5_tables as g
from simulator.shared_engine_io import draw_rows,player_view,save_state,load_state
from simulator.rollout import rollout
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step

COMPILED=jax.jit(e.advance)


def same(a,b):
    for x,y in zip(jax.tree.leaves(jax.device_get(a)),jax.tree.leaves(jax.device_get(b)),strict=True):
        np.testing.assert_array_equal(x,y)


class GenericEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('GEN5_NATIVE_CASES')
        cls.pack=json.loads(Path(path).read_text() if path else subprocess.check_output(
            ['node','reference/shared-engine-cases.cjs','--batch','generic'],text=True,timeout=120))
        cls.cases=cls.pack['cases']

    def test_native_initialization_and_complete_traces(self):
        full=0
        for case in self.cases:
            with self.subTest(case=case['id']):
                state,trace=initialize_case(case)
                self.assertEqual(draw_rows(trace),case['initializationDraws'])
                assert_state(self,state,case['initial'],case['id']+' initial')
                histories={side:list(case['initialEvents'][f'p{side+1}']) for side in range(2)}
                for side in range(2):self.assertEqual(player_view(state,side)['events'],histories[side])
                for n,step in enumerate(case['steps']):
                    result=COMPILED(state,choices_array(state,step['choices']))
                    assert_step(self,state,result,step,f"{case['id']}/{n}");state=result.state
                    for side in range(2):
                        histories[side].extend(step['events'][f'p{side+1}'])
                        self.assertEqual(player_view(state,side)['events'],histories[side])
                if case['setup'].get('fullBattle'):
                    full+=1;self.assertGreaterEqual(int(state.winner),0)
        self.assertEqual(full,4)

    def test_directed_eager_and_mixed_batches(self):
        states=[];actions=[];results=[]
        for case in self.cases:
            if case['setup'].get('fullBattle'):continue
            state,_=initialize_case(case)
            for n,step in enumerate(case['steps']):
                action=choices_array(state,step['choices']);result=COMPILED(state,action)
                if case['id'] in ('G4-data-hidden-power','G4-data-entry-Charizard') and n==0:
                    with jax.disable_jit():eager=e.advance(state,action)
                    same(eager,result)
                states.append(state);actions.append(action);results.append(result);state=result.state
        run=jax.jit(jax.vmap(e.advance))
        for size in (1,8,32):
            for start in range(0,len(states),size):
                ids=[min(start+n,len(states)-1) for n in range(size)]
                batch=jax.tree.map(lambda *xs:jnp.stack(xs),*(states[n] for n in ids))
                result=run(batch,jnp.stack([actions[n] for n in ids]))
                same(result,jax.tree.map(lambda *xs:jnp.stack(xs),*(results[n] for n in ids)))

    def test_unsupported_callbacks_preflight_without_rng(self):
        base={'species':'Mew','moves':['Surf'],'ability':'Synchronize'}
        for replacement in ({'ability':'Magic Guard'},{'moves':['Sleep Talk']},{'item':'Black Sludge'},
                            {'moves':['Psychic']},{'moves':['False Swipe']},{'species':'Darmanitan-Zen'}):
            result=e.make_gen5_state([[base|replacement],[base]],seed=[1,2,3,4])
            self.assertEqual(int(result.code),e.UNSUPPORTED,replacement)
            self.assertEqual(int(result.trace.dcount),0)
            self.assertEqual(int(result.trace.ecount),0)
            self.assertEqual(result.state.rng.tolist(),[1,2,3,4])
            attempted=COMPILED(result.state,e.i32([0,0]))
            self.assertEqual(int(attempted.code),e.UNSUPPORTED)
            same(result.state,attempted.state)

    def test_schema_migration_saved_continuation_and_rollouts(self):
        manifest=json.loads(Path('rules/shared-engine-v12.json').read_text())
        self.assertEqual(manifest['schema_version'],e.SCHEMA)
        self.assertEqual(manifest['reference_commit'],g.PIN)
        self.assertEqual(e.SPECIES[:len(e.LEGACY_SPECIES)],e.LEGACY_SPECIES)
        self.assertEqual(e.MOVE_NAMES[:len(e.LEGACY_MOVES)],e.LEGACY_MOVES)
        self.assertEqual(e.ITEMS[:len(e.LEGACY_ITEMS)],e.LEGACY_ITEMS)
        self.assertEqual(g.DATA_SHA256,json.loads(Path('rules/gen5-reachable-v1.json').read_text())['dataSha256'])
        case=next(c for c in self.cases if c['setup'].get('fullBattle'))
        start,_=initialize_case(case);state=start;plan=[];middle=None
        for n,step in enumerate(case['steps']):
            action=choices_array(state,step['choices']);plan.append(action)
            if n==len(case['steps'])//2:middle=state
            state=COMPILED(state,action).state
        plan=jnp.stack(plan)
        for snapshot in (start,middle,state):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'state.npz';save_state(path,snapshot);restored=load_state(path)
            same(snapshot,restored)
        lean=jax.jit(rollout)(start,plan)
        full=jax.jit(lambda s,p:rollout(s,p,retain=('trace','states','codes')))(start,plan)
        same(lean.state,state);same(full.state,state)
        self.assertTrue(bool(lean.terminal));self.assertFalse(bool(lean.failed))
        token=jnp.stack((state.generation,state.episode))
        reset=jax.jit(e.reset)(state,start,token)
        self.assertEqual(int(reset.code),e.OK)
        same(reset.state,start._replace(generation=state.generation+1))
        # Frozen schema 1/2 fields migrate explicitly, including inferred
        # Speed EVs and the original stable entity IDs/hidden commitments.
        legacy=e.make_state([[{'species':'Snorlax','moves':['Tackle'],'speed_evs':252}],
                             [{'species':'Mew','moves':['Tackle']}]]).state
        pending=COMPILED(legacy,e.i32([0,-1])).state
        fields={name:np.asarray(getattr(pending,name)) for name in e.State._fields[:e.State._fields.index('ivs')]}
        for schema in (1,2):
            old=dict(fields)
            if schema==1:old.pop('hazards')
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'legacy.npz';np.savez(path,**old)
                with self.assertRaises(ValueError):load_state(path)
                migrated=load_state(path,schema=schema)
            same(pending,migrated)
            same(COMPILED(pending,e.i32([-1,0])),COMPILED(migrated,e.i32([-1,0])))

    def test_complete_battles_in_streamed_heterogeneous_batches(self):
        cases=[c for c in self.cases if c['setup'].get('fullBattle')]
        starts=[];plans=[];expected=[]
        for case in cases:
            state,_=initialize_case(case);starts.append(state);actions=[];results=[]
            for step in case['steps']:
                action=choices_array(state,step['choices']);result=COMPILED(state,action)
                actions.append(action);results.append(result);state=result.state
            plans.append(actions);expected.append(results)
        run=jax.jit(jax.vmap(e.advance))
        length=max(map(len,plans))
        for size in (1,8,32):
            lanes=[n%len(cases) for n in range(size)]
            batch=jax.tree.map(lambda *xs:jnp.stack(xs),*(starts[n] for n in lanes))
            for chunk in range(0,length,16):
                for boundary in range(chunk,min(length,chunk+16)):
                    actions=jnp.stack([plans[n][boundary] if boundary<len(plans[n]) else e.i32([-1,-1]) for n in lanes])
                    result=run(batch,actions)
                    targets=[]
                    for n in lanes:
                        if boundary<len(expected[n]):targets.append(expected[n][boundary])
                        else:targets.append(e.Result(expected[n][-1].state,e.i32(e.TERMINAL),e.empty_trace()))
                    same(result,jax.tree.map(lambda *xs:jnp.stack(xs),*targets));batch=result.state


if __name__=='__main__':unittest.main()

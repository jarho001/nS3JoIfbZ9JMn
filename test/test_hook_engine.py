"""Independent native ability/item integration, not full-format acceptance."""
import json
from functools import partial
import tempfile
import os
from pathlib import Path
import subprocess
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from simulator import shared_engine as e
from simulator.shared_engine_io import draw_rows,player_view,save_state,load_state
from simulator.rollout import rollout
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step,assert_placement

COMPILED=jax.jit(e.advance)
LEAN=jax.jit(rollout)
FULL=jax.jit(partial(rollout,retain=('trace','codes')))


def same(a,b):
    assert_placement((a,b))
    a,b=jax.device_get((a,b))
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)


class HookEngineTest(unittest.TestCase):
    def tearDown(self):
        # Avoid retaining several large executable families in the 8 GiB cloud.
        jax.clear_caches()

    @classmethod
    def setUpClass(cls):
        path=os.environ.get('HOOK_NATIVE_CASES')
        raw=Path(path).read_text() if path else subprocess.check_output(
            ['node','reference/shared-engine-cases.cjs','--batch','hooks'],text=True,timeout=180)
        cls.cases=json.loads(raw)['cases']

    def test_native_states_draws_requests_and_histories(self):
        self.assertEqual(len(self.cases),70)
        manifest=json.loads(Path('rules/shared-engine-v13.json').read_text())
        self.assertEqual(manifest['schema_version'],e.SCHEMA)
        self.assertEqual(manifest['draw_contexts'],list(e.DRAW_CONTEXTS))
        for case in self.cases:
            s,trace=initialize_case(case)
            self.assertEqual(draw_rows(trace),case['initializationDraws'],case['id']+' init draws')
            assert_state(self,s,case['initial'],case['id']+' initial')
            histories=[list(case['initialEvents'][f'p{k+1}']) for k in range(2)]
            for side in range(2):self.assertEqual(player_view(s,side)['events'],histories[side],case['id']+' init history')
            for n,step in enumerate(case['steps']):
                self.assertTrue(all(a['accepted'] for a in step['accepted']))
                r=COMPILED(s,choices_array(s,step['choices']))
                assert_step(self,s,r,step,f"{case['id']}/{n}");s=r.state
                for side in range(2):
                    histories[side].extend(step['events'][f'p{side+1}'])
                    self.assertEqual(player_view(s,side)['events'],histories[side])
            if case['setup'].get('fullBattle'):self.assertGreaterEqual(int(s.winner),0)

    def test_mixed_eager_compiled_batches_and_saved_states(self):
        names={'G4-hooks-intimidate-tied','G4-hooks-weather-dual-tied','G4-hooks-weather-initial-speed',
               'G4-hooks-choice-last-pp-protect','G4-hooks-choice-sleep-status','G4-hooks-life-orb-terminal',
               'G4-hooks-life-orb-target-bench','G4-hooks-life-orb-struggle','G4-hooks-life-orb-pivot-ko',
               'G4-hooks-eviolite-sand','G4-hooks-damage-handler-interaction','G4-hooks-choice-all-pp-empty'}
        inputs=[];outputs=[]
        for case in self.cases:
            if case['id'] not in names:continue
            s,trace=initialize_case(case)
            if case['id'] in {'G4-hooks-intimidate-tied','G4-hooks-weather-dual-tied','G4-hooks-weather-initial-speed'}:
                with jax.disable_jit():eager_start,eager_trace=initialize_case(case)
                same(eager_start,s);same(eager_trace,trace)
            for step in case['steps']:
                action=choices_array(s,step['choices']);r=COMPILED(s,action)
                with jax.disable_jit():eager=e.advance(s,action)
                same(eager,r)
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'state.npz';save_state(path,s);restored=load_state(path)
                same(restored,s);same(COMPILED(restored,action),r)
                inputs.append((s,action));outputs.append(r);s=r.state
        stack=lambda xs:jax.tree.map(lambda *ys:jnp.stack(ys),*xs)
        jax.clear_caches()
        for size in (1,8,32):
            indices=[i%len(inputs) for i in range(size)]
            result=jax.jit(jax.vmap(e.advance))(stack([inputs[i][0] for i in indices]),jnp.stack([inputs[i][1] for i in indices]))
            same(result,stack([outputs[i] for i in indices]));jax.clear_caches()

    def test_complete_rollouts_continuation_and_reset(self):
        length=max(len(c['steps']) for c in self.cases if c['setup'].get('fullBattle'))+1
        for case in self.cases:
            if not case['setup'].get('fullBattle'):continue
            start,_=initialize_case(case);s=start;plan=[]
            for step in case['steps']:
                a=choices_array(s,step['choices']);plan.append(a);s=COMPILED(s,a).state
            plan=jnp.pad(jnp.stack(plan),((0,length-len(plan)),(0,0)),constant_values=-1)
            lean=LEAN(start,plan);full=FULL(start,plan)
            same(lean.state,s);same(full.state,s)
            self.assertTrue(bool(lean.terminal));self.assertFalse(bool(lean.failed))
            split=len(plan)//2;prefix=LEAN(start,plan[:split])
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'continuation.npz';save_state(path,prefix.state);saved=load_state(path)
            same(LEAN(saved,plan[split:]).state,s)
            reset=jax.jit(e.reset)(s,start,jnp.stack((s.generation,s.episode)))
            self.assertEqual(int(reset.code),e.OK);same(reset.state,start._replace(generation=s.generation+1))

    def test_atomic_lock_metadata_and_lossless_schema5_migration(self):
        case=next(c for c in self.cases if c['id']=='G4-hooks-item-ChoiceScarf');start,_=initialize_case(case)
        s=COMPILED(start,choices_array(start,case['steps'][0]['choices'])).state
        for bad in (s._replace(item=s.item.at[0,0].set(0)),
                    s._replace(volatile_state=s.volatile_state.at[0,0,3,4].set(1)),
                    s._replace(volatile_state=s.volatile_state.at[0,0,3,5].set(0)),
                    s._replace(volatile_state=s.volatile_state.at[0,0,3,1].set(s.effect_order))):
            r=COMPILED(bad,e.i32([0,0]));self.assertEqual(int(r.code),e.UNSUPPORTED);same(r.state,bad)
            self.assertEqual(int(r.trace.dcount),0);self.assertEqual(int(r.trace.ecount),0)
        from simulator import gen5_tables as g
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'schema5.npz'
            def old_snapshot(state):
                fields={k:np.asarray(getattr(state,k)) for k in e.State._fields[:e.State._fields.index('move_history')]}
                fields['volatile_state']=fields['volatile_state'][:,:,:3,:6]
                np.savez(path,schema=5,data_sha256=np.asarray(g.DATA_SHA256),**fields)
            old_snapshot(s)
            with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(path)
            # Schema5 already represented field/status source and timer state.
            raw=Path(os.environ['FIELD_NATIVE_CASES']).read_text() if os.environ.get('FIELD_NATIVE_CASES') else subprocess.check_output(['node','reference/shared-engine-cases.cjs','--batch','field'],text=True,timeout=180)
            case=next(c for c in json.loads(raw)['cases'] if c['id']=='G4-field-hazard-source-order')
            s,_=initialize_case(case)
            for step in case['steps']:
                s=COMPILED(s,choices_array(s,step['choices'])).state
            old_snapshot(s)
            with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(path)
            same(load_state(path,replayed_state=s),s)

if __name__=='__main__':unittest.main()

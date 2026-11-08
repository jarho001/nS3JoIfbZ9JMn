"""Bounded CPU correctness, independent native six-member hazard traces."""
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
from simulator.rollout import rollout
from simulator.shared_engine_io import draw_rows, player_view
from test.shared_engine_support import initialize_case, choices_array, assert_state, assert_step

COMPILED = jax.jit(e.advance)
BATCHED = jax.jit(jax.vmap(e.advance))

def same(test,a,b):
    a,b=jax.device_get((a,b))
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        np.testing.assert_array_equal(x,y)

class RosterHazardsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('G4_NATIVE_CASES')
        raw=Path(path).read_text() if path else subprocess.check_output(
            ['node','reference/shared-engine-cases.cjs','--batch','g4'],text=True,timeout=120)
        cls.pack=json.loads(raw);cls.cases=cls.pack['cases']

    def test_manifest_and_catalog(self):
        manifest=json.loads(Path('rules/shared-engine-v2.json').read_text())
        self.assertEqual(manifest['schema_version'],2)
        self.assertEqual(manifest['moves'],[m for m in e.LEGACY_MOVES[1:] if m!='Struggle'])
        self.assertEqual(manifest['storage']['occupied_per_side'],[1,6])
        self.assertEqual(self.pack['pin'],manifest['reference_commit'])
        self.assertEqual(sum(bool(c['setup'].get('fullBattle')) for c in self.cases),4)
        for c in self.cases:
            self.assertTrue(all(a['accepted'] for t in c['steps'] for a in t['accepted']),c['id'])

    def test_native_boundaries_and_complete_battles(self):
        for case in self.cases:
            with self.subTest(case=case['id']):
                s,trace=initialize_case(case)
                self.assertEqual(draw_rows(trace),case['initializationDraws'])
                assert_state(self,s,case['initial'],case['id']+' initial')
                histories={side:list(case['initialEvents'][f'p{side+1}']) for side in range(2)}
                for n,step in enumerate(case['steps']):
                    before=s;choices=choices_array(s,step['choices']);r=COMPILED(s,choices)
                    assert_step(self,s,r,step,f"{case['id']}/{n}");s=r.state
                    if case['id'] in ('G4-pivot-entry-ko','G4-entry-ko-chain','G4-spin-ko','G4-place-Spikes'):
                        with jax.disable_jit():eager=e.advance(before,choices)
                        same(self,eager,r)
                        with tempfile.TemporaryDirectory() as tmp:
                            path=Path(tmp)/'state.npz';np.savez(path,**dict(zip(e.State._fields,jax.device_get(before),strict=True)))
                            with np.load(path,allow_pickle=False) as saved:restored=e.State(*(jnp.asarray(saved[k]) for k in e.State._fields))
                        same(self,COMPILED(restored,choices),r)
                    for side in range(2):
                        histories[side].extend(step['events'][f'p{side+1}'])
                        self.assertEqual(player_view(s,side)['events'],histories[side])
                if case['setup'].get('fullBattle'):
                    self.assertGreaterEqual(int(s.winner),0)
                    self.assertLess(int(s.hcount),e.HISTORY)

    def test_mixed_batch_rollout_continuation_and_reset(self):
        selected=[c for c in self.cases if c['setup'].get('fullBattle')]
        starts=[];finals=[];plans=[]
        for c in selected:
            s,_=initialize_case(c);starts.append(s);plan=[]
            for step in c['steps']:
                a=choices_array(s,step['choices']);plan.append(a);s=COMPILED(s,a).state
            finals.append(s);plans.append(jnp.stack(plan))
        length=max(len(p) for p in plans)+2
        plans=jnp.stack([jnp.pad(p,((0,length-len(p)),(0,0)),constant_values=-1) for p in plans])
        batch=jax.tree.map(lambda *xs:jnp.stack(xs),*starts)
        lean=jax.jit(jax.vmap(lambda s,p:rollout(s,p)))(batch,plans)
        expected=jax.tree.map(lambda *xs:jnp.stack(xs),*finals)
        same(self,lean.state,expected)
        self.assertTrue(np.all(np.asarray(lean.terminal)))
        self.assertFalse(np.any(np.asarray(lean.failed)))
        # One uninterrupted battle equals a saved/resumed scan; hazard order and
        # pending U-turn state survive the exact on-disk State representation.
        first=jax.jit(lambda s,p:rollout(s,p))(starts[0],plans[0,:8])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.npz';np.savez(path,**dict(zip(e.State._fields,jax.device_get(first.state))))
            with np.load(path,allow_pickle=False) as saved:restored=e.State(*(jnp.asarray(saved[k]) for k in e.State._fields))
        resumed=jax.jit(lambda s,p:rollout(s,p))(restored,plans[0,8:])
        same(self,resumed.state,finals[0])
        full=jax.jit(lambda s,p:rollout(s,p,retain=('trace','states','codes')))(starts[0],plans[0])
        same(self,full.state,finals[0])
        reset=jax.jit(e.reset)(finals[0],starts[1],jnp.stack((finals[0].generation,finals[0].episode)))
        self.assertEqual(int(reset.code),e.OK)
        same(self,reset.state,starts[1]._replace(generation=finals[0].generation+1))
        # Mixed placement/removal/entry KO/pending and terminal/error lanes.
        states=[];actions=[];outputs=[]
        for c in self.cases:
            if c['setup'].get('fullBattle'):continue
            s,_=initialize_case(c)
            for n,step in enumerate(c['steps']):
                a=choices_array(s,step['choices']);r=COMPILED(s,a)
                states.append(s);actions.append(a);outputs.append(r);s=r.state
        states.extend([finals[0],starts[0]._replace(hazards=e.i32([[0,0,1],[0,0,0]]))])
        actions.extend([e.i32([0,0]),e.i32([0,0])]);outputs.extend([COMPILED(states[-2],actions[-2]),COMPILED(states[-1],actions[-1])])
        batched=BATCHED(jax.tree.map(lambda *xs:jnp.stack(xs),*states),jnp.stack(actions))
        same(self,batched,jax.tree.map(lambda *xs:jnp.stack(xs),*outputs))
        self.assertEqual(int(outputs[-1].code),e.UNSUPPORTED)
        same(self,outputs[-1].state,states[-1])

    def test_domain_and_hazard_validation(self):
        desc={'species':'Mew','moves':['Spikes']}
        for size in (0,7):
            with self.assertRaises(ValueError):e.make_state([[desc]*size,[desc]])
        for move in ('Struggle','Toxic Spikes'):
            with self.assertRaises(ValueError):e.make_state([[{'species':'Mew','moves':[move]}],[desc]])
        s,_=initialize_case(self.cases[0])
        for row in ([2,0,1],[0,4,2],[0,0,1],[1,0,2],[1,1,0]):
            bad=s._replace(hazards=s.hazards.at[0].set(e.i32(row)))
            r=COMPILED(bad,e.i32([0,0]));self.assertEqual(int(r.code),e.UNSUPPORTED);same(self,r.state,bad)
        with self.assertRaises(ValueError):e.advance(s._replace(hazards=jnp.zeros((2,2),jnp.int32)),e.i32([0,0]))

if __name__=='__main__':unittest.main()

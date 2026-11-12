"""Independent pinned recharge requests, timers and execution/lifecycle parity."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from simulator import shared_engine as e
from simulator.shared_engine_io import draw_rows,player_view,save_state,load_state
from simulator.rollout import rollout
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step,assert_placement

STEP=jax.jit(e.advance)
LEAN=jax.jit(rollout)
FULL=jax.jit(partial(rollout,retain=('trace','codes')))


def same(a,b):
    assert_placement((a,b))
    for x,y in zip(jax.tree.leaves(jax.device_get(a)),jax.tree.leaves(jax.device_get(b)),strict=True):np.testing.assert_array_equal(x,y)


class RechargeEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('RECHARGE_NATIVE_CASES')
        raw=Path(path).read_text() if path else subprocess.check_output(['node','reference/shared-engine-cases.cjs','--batch','recharge'],text=True,timeout=180)
        cls.cases=json.loads(raw)['cases']

    def tearDown(self):jax.clear_caches()

    def test_native_requests_effects_draws_and_histories(self):
        self.assertEqual(len(self.cases),34)
        manifest=json.loads(Path('rules/shared-engine-v22.json').read_text())
        self.assertEqual(manifest['schema_version'],e.SCHEMA)
        self.assertEqual(manifest['draw_contexts'],list(e.DRAW_CONTEXTS))
        for case in self.cases:
            s,trace=initialize_case(case)
            self.assertEqual(draw_rows(trace),case['initializationDraws'],case['id'])
            assert_state(self,s,case['initial'],case['id']+' initial')
            histories=[list(case['initialEvents'][f'p{k+1}']) for k in range(2)]
            for n,step in enumerate(case['steps']):
                r=STEP(s,choices_array(s,step['choices']));label=f"{case['id']}/{n}"
                if any(not a['accepted'] for a in step['accepted']):
                    self.assertEqual(int(r.code),e.RETRY if any(cmd.startswith('switch ') for cmd in step['choices'].values()) else e.INVALID,label);same(r.state,s)
                    self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
                    assert_state(self,s,step['after'],label)
                else:assert_step(self,s,r,step,label)
                s=r.state
                for side in range(2):
                    histories[side].extend(step['events'][f'p{side+1}']);self.assertEqual(player_view(s,side)['events'],histories[side],label)
            if case['setup'].get('fullBattle'):self.assertGreaterEqual(int(s.winner),0)

    def test_eager_jit_mixed_batches_and_saved_locked_requests(self):
        names={'G4-recharge-slot2-pending-undo','G4-recharge-Spore','G4-recharge-contact-1','G4-recharge-target-ko','G4-recharge-last-pp','G4-recharge-ChoiceScarf','G4-recharge-terminal','G4-recharge-frozen-recharge','G4-recharge-both-recharge','G4-recharge-reject-switch'}
        inputs=[];outputs=[]
        for case in self.cases:
            if case['id'] not in names:continue
            s,_=initialize_case(case)
            for step in case['steps']:
                a=choices_array(s,step['choices']);r=STEP(s,a)
                with jax.disable_jit():same(e.advance(s,a),r)
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/'state.npz';save_state(p,s);restored=load_state(p)
                same(s,restored);same(STEP(restored,a),r)
                inputs.append((s,a));outputs.append(r);s=r.state
            if int(s.winner)>=0:
                a=e.i32([-1,-1]);r=STEP(s,a);self.assertEqual(int(r.code),e.TERMINAL)
                inputs.append((s,a));outputs.append(r)
        stack=lambda xs:jax.tree.map(lambda *ys:jnp.stack(ys),*xs)
        jax.clear_caches()
        for size in (1,8,32):
            ix=[i%len(inputs)for i in range(size)]
            actual=jax.jit(jax.vmap(e.advance))(stack([inputs[i][0]for i in ix]),jnp.stack([inputs[i][1]for i in ix]))
            same(actual,stack([outputs[i]for i in ix]));jax.clear_caches()

    def test_complete_games_full_lean_continuation_and_reset(self):
        for case in self.cases:
            if not case['setup'].get('fullBattle'):continue
            start,_=initialize_case(case);s=start;plan=[];split=None
            for step in case['steps']:
                a=choices_array(s,step['choices']);plan.append(a);s=STEP(s,a).state
                if split is None and bool(jnp.any(e.phase_ops.locked(s))):split=len(plan)
            self.assertIsNotNone(split)
            plan=jnp.concatenate((jnp.stack(plan),jnp.full((1,2),-1,jnp.int32)),axis=0)
            lean=LEAN(start,plan);full=FULL(start,plan);same(lean.state,s);same(full.state,s)
            self.assertTrue(bool(lean.terminal));self.assertFalse(bool(lean.failed));self.assertFalse(bool(lean.truncated))
            prefix=LEAN(start,plan[:split])
            self.assertTrue(bool(jnp.any(e.phase_ops.locked(prefix.state))))
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'saved.npz';save_state(p,prefix.state);saved=load_state(p)
            same(LEAN(saved,plan[split:]).state,s)
            reset=jax.jit(e.reset)(s,start,jnp.stack((s.generation,s.episode)))
            self.assertEqual(int(reset.code),e.OK);same(reset.state,start._replace(generation=s.generation+1))

    def test_atomic_corruption_and_schema7_continuation(self):
        case=next(c for c in self.cases if c['id']=='G4-recharge-slot2-pending-undo')
        s,_=initialize_case(case);s=STEP(s,choices_array(s,case['steps'][0]['choices'])).state
        for field,value in [(4,0),(5,0),(1,int(s.effect_order)),(2,1),(3,1)]:
            bad=s._replace(volatile_state=s.volatile_state.at[0,0,4,field].set(value))
            r=STEP(bad,e.i32([0,0]));self.assertEqual(int(r.code),e.UNSUPPORTED);same(r.state,bad)
            self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        full=s._replace(hcount=e.i32(e.HISTORY));r=STEP(full,e.i32([0,0]))
        self.assertEqual(int(r.code),e.CAPACITY);same(r.state,full)
        self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        stale=STEP(s,e.i32([0,0]),e.i32([-1,-1]))
        self.assertEqual(int(stale.code),e.STALE);same(stale.state,s)
        # Old accepted states cannot have recharge capabilities. Ordinary advanced
        # states preserve exact continuation; fabricated old recharge is rejected.
        from simulator import gen5_tables as g
        normal=e.make_state([[{'species':'Mew','moves':['Tackle']}],[{'species':'Snorlax','moves':['Pound']}]],[0,2,3,4]).state
        normal=STEP(normal,e.i32([0,0])).state
        for state,accepted in [(normal,True),(s,False)]:
            fields={k:np.asarray(v)[:,:,:4,:6] if k=='volatile_state' else np.asarray(v)[:,:10]if k=='history'else np.asarray(v) for k,v in zip(e.State._fields,state) if k in e.State._fields[:e.State._fields.index('move_history')]}
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'old.npz';np.savez(p,schema=7,data_sha256=np.asarray(g.DATA_SHA256),**fields)
                if accepted:
                    with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(p)
                    restored=load_state(p,replayed_state=normal);same(restored,normal);same(STEP(restored,e.i32([0,0])),STEP(normal,e.i32([0,0])))
                else:
                    with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(p)

if __name__=='__main__':unittest.main()

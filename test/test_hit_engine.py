"""Native differential coverage for declarative hits, healing and reactive damage.

This family checkpoint does not accept full-format conformance. Native phase
investigations for locks/recharge and causal counters remain separately pending.
"""
import json
import os
from functools import partial
from pathlib import Path
import subprocess
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from simulator import shared_engine as e
from simulator import gen5_engine_tables as tables
from simulator.shared_engine_io import draw_rows,player_view,save_state,load_state
from simulator.rollout import rollout
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step,assert_placement

STEP=jax.jit(e.advance)
LEAN=jax.jit(rollout)
FULL=jax.jit(partial(rollout,retain=('trace','codes')))


def same(a,b):
    assert_placement((a,b))
    for x,y in zip(jax.tree.leaves(jax.device_get(a)),jax.tree.leaves(jax.device_get(b)),strict=True):np.testing.assert_array_equal(x,y)


class HitEngineTest(unittest.TestCase):
    def tearDown(self):jax.clear_caches()

    @classmethod
    def setUpClass(cls):
        path=os.environ.get('HIT_NATIVE_CASES')
        raw=Path(path).read_text() if path else subprocess.check_output(['node','reference/shared-engine-cases.cjs','--batch','hits'],text=True,timeout=180)
        cls.cases=json.loads(raw)['cases']

    def test_native_state_rng_requests_histories_and_outcomes(self):
        self.assertEqual(len(self.cases),97)
        manifest=json.loads(Path('rules/shared-engine-v22.json').read_text())
        self.assertEqual(manifest['schema_version'],e.SCHEMA)
        self.assertEqual(manifest['draw_contexts'],list(e.DRAW_CONTEXTS))
        for case in self.cases:
            s,trace=initialize_case(case)
            self.assertEqual(draw_rows(trace),case['initializationDraws'],case['id']+' initialization draws')
            assert_state(self,s,case['initial'],case['id']+' initial')
            histories=[list(case['initialEvents'][f'p{k+1}']) for k in range(2)]
            for side in range(2):self.assertEqual(player_view(s,side)['events'],histories[side])
            for n,step in enumerate(case['steps']):
                self.assertTrue(all(a['accepted'] for a in step['accepted']))
                r=STEP(s,choices_array(s,step['choices']));assert_step(self,s,r,step,f"{case['id']}/{n}");s=r.state
                for side in range(2):
                    histories[side].extend(step['events'][f'p{side+1}']);self.assertEqual(player_view(s,side)['events'],histories[side],case['id'])
            if case['setup'].get('fullBattle'):self.assertGreaterEqual(int(s.winner),0)

    def test_eager_jit_mixed_batches_and_serialization(self):
        names={'G4-phase-count-2','G4-phase-count-3','G4-phase-count-4','G4-phase-count-5',
            'G4-hit-contact-stop-33','G4-hit-contact-terminal-multihit-1','G4-hit-recoil-double-ko',
            'G4-hit-frozen-target-fire-contact','G4-hit-frozen-target-fire-terminal','G4-hit-spin-contact-1','G4-hit-spin-contact-100','G4-hit-defrost-miss','G4-hit-scald-choice-frozen','G4-hit-scald-frozen-confusion-blocked','G4-hit-secondary-multihit-Synchronize','G4-hit-struggle-contact-orb','G4-hit-struggle-contact-double-ko','G4-hit-explosion-double-ko','G4-hit-explosion-miss','G4-hit-LiquidOoze-GigaDrain','G4-hit-boost-weather-screen','G4-hit-paralysis-tied-contact'}
        inputs=[];outputs=[]
        for case in self.cases:
            if case['id'] not in names:continue
            s,_=initialize_case(case)
            for step in case['steps']:
                a=choices_array(s,step['choices']);r=STEP(s,a)
                with jax.disable_jit():eager=e.advance(s,a)
                same(eager,r)
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'state.npz';save_state(path,s);restored=load_state(path)
                same(restored,s);same(STEP(restored,a),r)
                inputs.append((s,a));outputs.append(r);s=r.state
        stack=lambda xs:jax.tree.map(lambda *ys:jnp.stack(ys),*xs)
        jax.clear_caches()
        for size in (1,8,32):
            ix=[i%len(inputs) for i in range(size)]
            result=jax.jit(jax.vmap(e.advance))(stack([inputs[i][0] for i in ix]),jnp.stack([inputs[i][1] for i in ix]))
            same(result,stack([outputs[i]for i in ix]));jax.clear_caches()

    def test_complete_games_full_lean_continuation_and_reset(self):
        for case in self.cases:
            if not case['setup'].get('fullBattle'):continue
            start,_=initialize_case(case);s=start;plan=[]
            for step in case['steps']:
                a=choices_array(s,step['choices']);plan.append(a);s=STEP(s,a).state
            plan=jnp.concatenate((jnp.stack(plan),jnp.full((1,2),-1,jnp.int32)),axis=0)
            lean=LEAN(start,plan);full=FULL(start,plan)
            same(lean.state,s);same(full.state,s)
            self.assertTrue(bool(lean.terminal));self.assertFalse(bool(lean.failed));self.assertFalse(bool(lean.truncated))
            split=len(plan)//2;prefix=LEAN(start,plan[:split])
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'saved.npz';save_state(path,prefix.state);saved=load_state(path)
            same(LEAN(saved,plan[split:]).state,s)
            reset=jax.jit(e.reset)(s,start,jnp.stack((s.generation,s.episode)))
            self.assertEqual(int(reset.code),e.OK);same(reset.state,start._replace(generation=s.generation+1))

    def test_atomic_unsupported_and_schema6_continuation(self):
        case=next(c for c in self.cases if c['id']=='G4-phase-count-2');s,_=initialize_case(case)
        for move in ('triplekick','batonpass','transform','sleeptalk'):
            bad=s._replace(moves=s.moves.at[0,0,0].set(tables.MOVE_IDS[move]),pp=s.pp.at[0,0,0].set(e.PP[tables.MOVE_IDS[move]]))
            r=STEP(bad,e.i32([0,0]));self.assertEqual(int(r.code),e.UNSUPPORTED);same(r.state,bad)
            self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        s=e.make_state([[{'species':'Mew','moves':['Tackle']}],[{'species':'Snorlax','moves':['Pound']}]],[0,2,3,4]).state
        s=STEP(s,e.i32([0,0])).state
        # Schema6 uses exactly the same state layout and canonical IDs. Its
        # histories predate these new event/draw IDs and remain lossless.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'old.npz'
            from simulator import gen5_tables as g
            np.savez(path,schema=6,data_sha256=np.asarray(g.DATA_SHA256),**{k:(np.asarray(getattr(s,k))[:,:10] if k=='history' else np.asarray(getattr(s,k)))[:,:,:4,:6] if k=='volatile_state' else (np.asarray(getattr(s,k))[:,:10] if k=='history' else np.asarray(getattr(s,k))) for k in e.State._fields[:e.State._fields.index('move_history')]})
            with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(path)
            restored=load_state(path,replayed_state=s);same(restored,s);same(STEP(restored,e.i32([0,0])),STEP(s,e.i32([0,0])))

if __name__=='__main__':unittest.main()

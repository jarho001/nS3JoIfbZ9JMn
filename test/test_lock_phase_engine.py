"""Pinned causal-state and move-phase integration; full G4 remains separate."""
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
from simulator.shared_engine_io import draw_rows,player_view,save_state,load_state
from simulator.rollout import rollout
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step,assert_placement

STEP=jax.jit(e.advance)
LEAN=jax.jit(rollout)
FULL=jax.jit(partial(rollout,retain=('trace','codes')))


def same(a,b):
    assert_placement((a,b))
    for x,y in zip(jax.tree.leaves(jax.device_get(a)),jax.tree.leaves(jax.device_get(b)),strict=True):np.testing.assert_array_equal(x,y)


class LockPhaseEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('LOCK_NATIVE_CASES')
        raw=Path(path).read_text() if path else subprocess.check_output(['node','reference/shared-engine-cases.cjs','--batch','locks'],text=True,timeout=180)
        cls.cases=json.loads(raw)['cases']

    def tearDown(self):jax.clear_caches()

    def test_native_causal_state_effects_draws_requests_and_histories(self):
        self.assertEqual(len(self.cases),73)
        self.assertEqual(sum(len(c['steps'])for c in self.cases),371)
        manifest=json.loads(Path('rules/shared-engine-v25.json').read_text())
        self.assertEqual(manifest['schema_version'],e.SCHEMA)
        self.assertEqual(manifest['draw_contexts'],list(e.DRAW_CONTEXTS))
        for case in self.cases:
            s,trace=initialize_case(case)
            self.assertEqual(draw_rows(trace),case['initializationDraws'],case['id'])
            assert_state(self,s,case['initial'],case['id']+' initial')
            histories=[list(case['initialEvents'][f'p{k+1}'])for k in range(2)]
            for n,step in enumerate(case['steps']):
                r=STEP(s,choices_array(s,step['choices']));label=f"{case['id']}/{n}"
                if any(not a['accepted']for a in step['accepted']):
                    self.assertEqual(len(step['accepted']),1,label+' rejection episode must isolate the command')
                    self.assertEqual(int(r.code),e.RETRY if any(cmd.startswith('switch ') for cmd in step['choices'].values()) else e.INVALID,label);same(r.state,s)
                    self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
                    assert_state(self,r.state,step['after'],label)
                else:assert_step(self,s,r,step,label)
                s=r.state
                for side in range(2):
                    histories[side].extend(step['events'][f'p{side+1}']);self.assertEqual(player_view(s,side)['events'],histories[side],label)
            if case['setup'].get('fullBattle'):self.assertGreaterEqual(int(s.winner),0)

    def test_eager_jit_mixed_batches_and_saved_phase_states(self):
        names={'G4-phase-rampage-Outrage','G4-phase-charge-Fly','G4-lock-sky-flinch',
          'G4-lock-disable-queued','G4-lock-encore-override','G4-lock-pending-charge',
          'G4-lock-torment-rampage','G4-lock-terminal','G4-lock-reject-switch','G4-lock-rampage-residual-ko','G4-lock-charge-SolarBeam-herb','G4-lock-solar-technician-rain','G4-lock-counter-Dive'}
        inputs=[];outputs=[]
        for case in self.cases:
            if case['id'] not in names:continue
            s,_=initialize_case(case)
            for step in case['steps']:
                a=choices_array(s,step['choices']);r=STEP(s,a)
                with jax.disable_jit():same(e.advance(s,a),r)
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/'phase.npz';save_state(p,s);restored=load_state(p)
                same(restored,s);same(STEP(restored,a),r)
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

    def test_complete_full_lean_games_locked_continuation_and_reset(self):
        games=[c for c in self.cases if c['setup'].get('fullBattle')]
        length=max(len(c['steps'])for c in games)+1
        for case in games:
            start,_=initialize_case(case);s=start;plan=[];split=None
            for step in case['steps']:
                a=choices_array(s,step['choices']);plan.append(a);s=STEP(s,a).state
                if split is None and bool(jnp.any(e.phase_ops.locked(s))):split=len(plan)
            self.assertIsNotNone(split)
            plan=jnp.pad(jnp.stack(plan),((0,length-len(plan)),(0,0)),constant_values=-1)
            lean=LEAN(start,plan);full=FULL(start,plan);same(lean.state,s);same(full.state,s)
            self.assertTrue(bool(lean.terminal));self.assertFalse(bool(lean.failed));self.assertFalse(bool(lean.truncated))
            prefix=LEAN(start,plan[:split]);self.assertTrue(bool(jnp.any(e.phase_ops.locked(prefix.state))))
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'saved.npz';save_state(p,prefix.state);saved=load_state(p)
            same(LEAN(saved,plan[split:]).state,s)
            reset=jax.jit(e.reset)(s,start,jnp.stack((s.generation,s.episode)))
            self.assertEqual(int(reset.code),e.OK);same(reset.state,start._replace(generation=s.generation+1))

    def test_causal_metadata_capacity_privacy_and_explicit_legacy_replay(self):
        case=next(c for c in self.cases if c['id']=='G4-phase-rampage-Outrage');s,_=initialize_case(case)
        s=STEP(s,choices_array(s,case['steps'][0]['choices'])).state
        for k,value in [(e.tracking.RESULT,9),(e.tracking.ATTACK_COUNT,-1),(e.tracking.USED_SLOTS,16)]:
            bad=s._replace(action_state=s.action_state.at[0,0,k].set(value));r=STEP(bad,e.i32([0,0]))
            self.assertEqual(int(r.code),e.UNSUPPORTED);same(r.state,bad)
            self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        bad=s._replace(volatile_state=s.volatile_state.at[0,0,5,6].set(e.phase_ops.SOURCE_ITEM))
        r=STEP(bad,e.i32([0,0]));self.assertEqual(int(r.code),e.UNSUPPORTED);same(r.state,bad)
        row=e.i32([0,0,e.TACKLE,1,0,e.tracking.NUMBER,1])
        full=s._replace(attacked_by=s.attacked_by.at[1,s.active[1]].set(jnp.tile(row,(e.tracking.ATTACK_RECORDS,1))),
          action_state=s.action_state.at[1,s.active[1],e.tracking.ATTACK_COUNT].set(e.tracking.ATTACK_RECORDS))
        r=STEP(full,e.i32([0,0]));self.assertEqual(int(r.code),e.CAPACITY);same(r.state,full)
        self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        token=e.i32([-1,-1]);r=STEP(s,e.i32([0,0]),token);self.assertEqual(int(r.code),e.STALE);same(r.state,s)
        # New hidden arrays never enter either recipient's request/event view.
        for side in range(2):self.assertEqual(set(player_view(s,side)),{'request','events'})
        from simulator import gen5_tables as g
        # Replay a genuine schema8-compatible recharge state, retaining every
        # recorded old field and supplying the formerly absent causal state.
        source=e.make_gen5_state([[{'species':'Mew','moves':['Hyper Beam'],'ability':'Synchronize','evs':{'spe':252}}],
                                  [{'species':'Mew','moves':['Recover'],'ability':'Synchronize'}]],[1,2,3,4]).state
        source=STEP(source,e.i32([0,0])).state
        fields={k:(np.asarray(getattr(source,k))[:,:10] if k=='history' else np.asarray(getattr(source,k))) for k in e.State._fields[:e.State._fields.index('move_history')]}
        fields['volatile_state']=fields['volatile_state'][:,:,:5,:6]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'schema8.npz';np.savez(p,schema=8,data_sha256=np.asarray(g.DATA_SHA256),**fields)
            with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(p)
            same(load_state(p,replayed_state=source),source)
            different=source._replace(rng=source.rng.at[0].add(1))
            with self.assertRaisesRegex(ValueError,'mismatch'):load_state(p,replayed_state=different)
            same(STEP(load_state(p,replayed_state=source),e.i32([0,0])),STEP(source,e.i32([0,0])))

if __name__=='__main__':unittest.main()

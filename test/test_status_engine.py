"""Independent native duration/status parity through the shared JAX engine."""
import json
from functools import partial
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from simulator import shared_engine as e
from simulator.shared_engine_io import player_view,draw_rows,save_state,load_state
from simulator.rollout import rollout
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step

COMPILED=jax.jit(e.advance)
LEAN=jax.jit(rollout)
FULL=jax.jit(partial(rollout,retain=('trace','codes')))


def same(a,b):
    a,b=jax.device_get((a,b))
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)


class StatusEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('STATUS_NATIVE_CASES')
        raw=Path(path).read_text() if path else subprocess.check_output(
            ['node','reference/shared-engine-cases.cjs','--batch','status'],text=True,timeout=180)
        cls.cases=json.loads(raw)['cases']

    def test_native_states_draws_requests_and_histories(self):
        self.assertEqual(len(self.cases),55)
        manifest=json.loads(Path('rules/shared-engine-v15.json').read_text())
        self.assertEqual(manifest['schema_version'],e.SCHEMA)
        self.assertEqual(manifest['draw_contexts'],list(e.DRAW_CONTEXTS))
        contexts={d['semanticContext'] for c in self.cases for step in c['steps'] for d in step['draws']}
        self.assertTrue(set(e.DRAW_CONTEXTS[12:18])<=contexts)
        for case in self.cases:
            with self.subTest(case=case['id']):
                s,trace=initialize_case(case)
                self.assertEqual(draw_rows(trace),case['initializationDraws'])
                assert_state(self,s,case['initial'],case['id']+' initial')
                histories=[list(case['initialEvents'][f'p{k+1}']) for k in range(2)]
                for n,step in enumerate(case['steps']):
                    self.assertTrue(all(a['accepted'] for a in step['accepted']))
                    r=COMPILED(s,choices_array(s,step['choices']))
                    assert_step(self,s,r,step,f"{case['id']}/{n}");s=r.state
                    for side in range(2):
                        histories[side].extend(step['events'][f'p{side+1}'])
                        self.assertEqual(player_view(s,side)['events'],histories[side])
                if case['setup'].get('fullBattle'):self.assertGreaterEqual(int(s.winner),0)

    def test_mixed_eager_compiled_batches_and_saved_states(self):
        names={'G4-status-rest-lum','G4-status-confusion-paralysis','G4-status-sleep-switch-reset',
               'G4-status-confusion-self-ko','G4-status-fire-thaw-0','G4-status-secondary-ko'}
        inputs=[];outputs=[]
        for case in self.cases:
            if case['id'] not in names:continue
            s,_=initialize_case(case)
            for step in case['steps']:
                action=choices_array(s,step['choices']);r=COMPILED(s,action)
                with jax.disable_jit():eager=e.advance(s,action)
                same(eager,r)
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'state.npz';save_state(path,s);restored=load_state(path)
                same(restored,s);same(COMPILED(restored,action),r)
                inputs.append((s,action));outputs.append(r);s=r.state
        stack=lambda xs:jax.tree.map(lambda *ys:jnp.stack(ys),*xs)
        for size in (1,8,32):
            indices=[i%len(inputs) for i in range(size)]
            result=jax.jit(jax.vmap(e.advance))(stack([inputs[i][0] for i in indices]),jnp.stack([inputs[i][1] for i in indices]))
            same(result,stack([outputs[i] for i in indices]))

    def test_complete_rollouts_continuation_and_reset(self):
        length=max(len(c['steps']) for c in self.cases if c['setup'].get('fullBattle'))+1
        for case in self.cases:
            if not case['setup'].get('fullBattle'):continue
            start,_=initialize_case(case);s=start;plan=[]
            for step in case['steps']:
                a=choices_array(s,step['choices']);plan.append(a);s=COMPILED(s,a).state
            plan=jnp.pad(jnp.stack(plan),((0,length-len(plan)),(0,0)),constant_values=-1)
            lean=LEAN(start,plan)
            full=FULL(start,plan)
            same(lean.state,s);same(full.state,s)
            self.assertTrue(bool(lean.terminal));self.assertFalse(bool(lean.failed))
            split=len(plan)//2
            prefix=LEAN(start,plan[:split])
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'continuation.npz';save_state(path,prefix.state);saved=load_state(path)
            resumed=LEAN(saved,plan[split:]);same(resumed.state,s)
            reset=jax.jit(e.reset)(s,start,jnp.stack((s.generation,s.episode)))
            self.assertEqual(int(reset.code),e.OK);same(reset.state,start._replace(generation=s.generation+1))

    def test_frozen_schema3_requires_explicit_rehydration_for_advanced_states(self):
        c=next(c for c in self.cases if c['id']=='G4-status-rest-lum');s,_=initialize_case(c)
        # A stored old state cannot encode retained empty-status Rest timers.
        s=COMPILED(s,choices_array(s,c['steps'][0]['choices'])).state
        fields={k:np.asarray(getattr(s,k)) for k in e.State._fields[:e.State._fields.index('effect_order')]}
        from simulator import gen5_tables as g
        for field,value in [('status_state',s.status_state.at[0,0,5].set(999)),
                            ('volatile_state',s.volatile_state.at[0,0,2,2].set(9))]:
            bad=s._replace(**{field:value});r=COMPILED(bad,e.i32([0,0]))
            self.assertEqual(int(r.code),e.UNSUPPORTED);same(r.state,bad)
            self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'schema3.npz';np.savez(path,schema=3,data_sha256=np.asarray(g.DATA_SHA256),**fields)
            with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(path)

if __name__=='__main__':unittest.main()

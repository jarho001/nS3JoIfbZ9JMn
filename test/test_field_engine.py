"""Independent native weather, screen, stage and side-condition contracts."""
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
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step

COMPILED=jax.jit(e.advance)
LEAN=jax.jit(rollout)
FULL=jax.jit(partial(rollout,retain=('trace','codes')))


def same(a,b):
    a,b=jax.device_get((a,b))
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)


class FieldEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('FIELD_NATIVE_CASES')
        raw=Path(path).read_text() if path else subprocess.check_output(
            ['node','reference/shared-engine-cases.cjs','--batch','field'],text=True,timeout=180)
        cls.cases=json.loads(raw)['cases']

    def test_native_states_draws_requests_and_histories(self):
        self.assertEqual(len(self.cases),40)
        manifest=json.loads(Path('rules/shared-engine-v6.json').read_text())
        self.assertEqual(manifest['schema_version'],e.SCHEMA)
        self.assertEqual(manifest['draw_contexts'],list(e.DRAW_CONTEXTS))
        contexts={d['semanticContext'] for c in self.cases for step in c['steps'] for d in step['draws']}
        self.assertTrue(set(e.DRAW_CONTEXTS[18:])<=contexts)
        for case in self.cases:
            s,trace=initialize_case(case)
            self.assertEqual(draw_rows(trace),case['initializationDraws'],case['id'])
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
        names={'G4-field-weather-tied','G4-field-weather-double-ko','G4-field-screen-minimum',
               'G4-field-native-order-ShellSmash','G4-field-status-residuals','G4-field-stages-critical',
               'G4-field-confusion-boost','G4-field-critical-negative-attack','G4-field-four-screen-handlers',
               'G4-field-sun-status-immunity-metadata'}
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
            lean=LEAN(start,plan);full=FULL(start,plan)
            same(lean.state,s);same(full.state,s)
            self.assertTrue(bool(lean.terminal));self.assertFalse(bool(lean.failed))
            split=len(plan)//2;prefix=LEAN(start,plan[:split])
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'continuation.npz';save_state(path,prefix.state);saved=load_state(path)
            same(LEAN(saved,plan[split:]).state,s)
            reset=jax.jit(e.reset)(s,start,jnp.stack((s.generation,s.episode)))
            self.assertEqual(int(reset.code),e.OK);same(reset.state,start._replace(generation=s.generation+1))

    def test_rejects_invalid_sources_and_lossy_hazard_migration(self):
        case=next(c for c in self.cases if c['id']=='G4-field-hazard-source-order')
        start,_=initialize_case(case);s=COMPILED(start,choices_array(start,case['steps'][0]['choices'])).state
        for bad in (s._replace(hazard_meta=s.hazard_meta.at[1,0,1].set(99)),
                    s._replace(weather_state=e.i32([1,0,0,0,4,0])),
                    s._replace(screens=s.screens.at[0,0].set(e.i32([4,0,99,e.field_tables.SCREEN_MOVES[0],0])))):
            r=COMPILED(bad,e.i32([0,0]));self.assertEqual(int(r.code),e.UNSUPPORTED);same(r.state,bad)
            self.assertEqual(int(r.trace.dcount),0);self.assertEqual(int(r.trace.ecount),0)
        from simulator import gen5_tables as g
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'schema4.npz'
            fields={k:np.asarray(getattr(s,k)) for k in e.State._fields[:e.State._fields.index('weather_state')]}
            fields['volatile_state']=fields['volatile_state'][:,:,:3,:]
            np.savez(path,schema=4,data_sha256=np.asarray(g.DATA_SHA256),**fields)
            with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(path)
            # Schema 4's non-hazard status histories encode the rule-target flag.
            case=next(c for c in self.cases if c['id']=='G4-field-status-residuals');s,_=initialize_case(case)
            fields={k:np.asarray(getattr(s,k)) for k in e.State._fields[:e.State._fields.index('weather_state')]}
            fields['volatile_state']=fields['volatile_state'][:,:,:3,:]
            np.savez(path,schema=4,data_sha256=np.asarray(g.DATA_SHA256),**fields)
            same(load_state(path),s)

if __name__=='__main__':unittest.main()

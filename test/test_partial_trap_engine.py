"""Focused pinned partial-trapping parity; broad execution matrix is deferred."""
import gzip,json,os,tempfile,unittest
from pathlib import Path
import jax
import numpy as np
from simulator import shared_engine as e,journal
from simulator.shared_engine_io import draw_rows,player_view,private_packets,save_state,load_state
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step
STEP=jax.jit(e.advance)

class PartialTrapEngineTest(unittest.TestCase):
    def test_native_timers_residuals_requests_rng_histories_and_persistence(self):
        path=os.environ.get('PARTIAL_NATIVE_CASES')
        raw=Path(path).read_bytes()if path else gzip.decompress(Path('test-logs/g4-cpu-conformance/partial-trap.json.gz').read_bytes())
        cases=json.loads(raw)['cases'];self.assertEqual(len(cases),27)
        self.assertEqual(sum(len(c['steps'])for c in cases),105)
        def same(a,b):
            for x,y in zip(jax.tree.leaves(jax.device_get(a)),jax.tree.leaves(jax.device_get(b)),strict=True):np.testing.assert_array_equal(x,y)
        for c in cases:
            s,t=initialize_case(c);assert_state(self,s,c['initial'],c['id']+' initial');self.assertEqual(draw_rows(t),c['initializationDraws'])
            histories=[list(c['initialEvents'][f'p{k+1}'])for k in range(2)]
            for i,n in enumerate(c['steps']):
                r=STEP(s,choices_array(s,n['choices']));label=f"{c['id']}/{i}"
                if any(not a['accepted']for a in n['accepted']):
                    self.assertEqual(int(r.code),e.RETRY,label);same(r.state,s);self.assertEqual(draw_rows(r.trace),[]);self.assertEqual(int(r.trace.ecount),0)
                    self.assertEqual(sum([private_packets(r.state,r.trace,k)for k in range(2)],[]),n['privatePackets'],label)
                else:assert_step(self,s,r,n,label)
                s=r.state
                for k in range(2):
                    histories[k].extend(n['events'][f'p{k+1}']);self.assertEqual(player_view(s,k)['events'],histories[k],label)
                if i==0 and c['id']=='G4-partial-trap-expiry-ordinary':
                    with tempfile.TemporaryDirectory()as tmp:
                        p=Path(tmp)/'live.npz';save_state(p,s);same(load_state(p),s)
                        cp=journal.from_state(s);journal.save(p,cp);same(journal.load(p).core,cp.core)
            if c['id']=='G4-partial-trap-substitute-blocks':
                # Schema12 accepted Substitute state has21 volatile slots.
                with tempfile.TemporaryDirectory()as tmp:
                    p=Path(tmp)/'old.npz';fields={k:np.asarray(v)for k,v in zip(e.State._fields,s)};fields['volatile_state']=fields['volatile_state'][:,:,:21,:];fields['action_state']=fields['action_state'][:,:,:21]
                    np.savez(p,schema=12,data_sha256=np.asarray(journal.tables.DATA_SHA256),**fields);same(load_state(p),s);same(load_state(p,replayed_state=s),s)
                    cp=journal.from_state(s);fields={k:np.asarray(v)for k,v in zip(e.CoreState._fields,cp.core)};fields['volatile_state']=fields['volatile_state'][:,:,:21,:];fields['action_state']=fields['action_state'][:,:,:21]
                    meta={'journal_schema':3,'data_sha256':journal.tables.DATA_SHA256,'pages':[{'rows':page.rows,'recipients':page.recipients}for page in cp.pages]};payload=json.dumps(meta,sort_keys=True,separators=(',',':'))
                    np.savez(p,metadata=np.asarray(payload),journal_sha256=np.asarray(journal._checksum(payload,fields)),**fields)
                    with self.assertRaisesRegex(ValueError,'replay/provenance'):journal.load(p)
                    same(journal.load(p,replayed=cp).core,cp.core)
        jax.clear_caches()

    def test_deferred_eager_mixed_rollout_and_reset(self):
        # Consolidated milestone/return-home gate; not a routine family check.
        import jax.numpy as jp
        from simulator.rollout import rollout
        path=os.environ.get('PARTIAL_NATIVE_CASES')
        raw=Path(path).read_bytes()if path else gzip.decompress(Path('test-logs/g4-cpu-conformance/partial-trap.json.gz').read_bytes())
        cases=json.loads(raw)['cases'];inputs=[];outputs=[]
        def same(a,b):
            for x,y in zip(jax.tree.leaves(jax.device_get(a)),jax.tree.leaves(jax.device_get(b)),strict=True):np.testing.assert_array_equal(x,y)
        for c in cases:
            if c['id']not in {'G4-partial-trap-expiry-ordinary','G4-partial-trap-source-pivot','G4-partial-trap-known-trap-retry-mirror'}:continue
            start,_=initialize_case(c);s=start;plan=[]
            for n in c['steps']:
                a=choices_array(s,n['choices']);r=STEP(s,a);plan.append(a)
                with jax.disable_jit():same(e.advance(s,a),r)
                inputs.append((s,a));outputs.append(r);s=r.state
            plan=jp.stack(plan);lean=jax.jit(rollout)(start,plan)
            full=jax.jit(lambda s,p:rollout(s,p,retain=('trace','codes')))(start,plan)
            same(lean.state,s);same(full.state,s);self.assertFalse(bool(lean.failed))
            reset=jax.jit(e.reset)(s,start,jp.stack((s.generation,s.episode)))
            self.assertEqual(int(reset.code),e.OK);same(reset.state,start._replace(generation=s.generation+1))
        stack=lambda xs:jax.tree.map(lambda *v:jp.stack(v),*xs)
        jax.clear_caches()
        for size in (1,8,32):
            ix=[i%len(inputs)for i in range(size)]
            actual=jax.jit(jax.vmap(e.advance))(stack([inputs[i][0]for i in ix]),jp.stack([inputs[i][1]for i in ix]))
            same(actual,stack([outputs[i]for i in ix]));jax.clear_caches()

if __name__=='__main__':unittest.main()

"""Pinned trapping truth, private retries and independent seat/queue histories."""
import gzip,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import jax
import jax.numpy as jp
import numpy as np
from simulator import shared_engine as e,journal as journal
from simulator.shared_engine_io import draw_rows,private_packets,player_view,save_state,load_state
from simulator.rollout import rollout
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step,assert_placement
STEP=jax.jit(e.advance)
CORE=jax.jit(e.advance_core)

def same(a,b):
    assert_placement((a,b))
    for x,y in zip(jax.tree.leaves(jax.device_get(a)),jax.tree.leaves(jax.device_get(b)),strict=True):np.testing.assert_array_equal(x,y)

class TrappingEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('TRAP_NATIVE_CASES')
        raw=Path(path).read_text()if path else gzip.decompress(Path('test-logs/g4-cpu-conformance/trapping.json.gz').read_bytes())
        cls.cases=json.loads(raw)['cases']
    def tearDown(self):jax.clear_caches()

    def test_native_requests_feedback_rng_causal_state_and_histories(self):
        self.assertEqual(len(self.cases),31);self.assertEqual(sum(len(c['steps'])for c in self.cases),215)
        for c in self.cases:
            s,t=initialize_case(c);assert_state(self,s,c['initial'],c['id']+' initial');self.assertEqual(draw_rows(t),c['initializationDraws'])
            histories=[list(c['initialEvents'][f'p{i+1}'])for i in range(2)]
            for i,n in enumerate(c['steps']):
                before=s;r=STEP(s,choices_array(s,n['choices']));label=f"{c['id']}/{i}"
                if any(not a['accepted']for a in n['accepted']):
                    self.assertEqual(int(r.code),e.RETRY,label);assert_state(self,r.state,n['after'],label)
                    self.assertEqual(draw_rows(r.trace),[]);self.assertEqual(int(r.trace.ecount),0)
                    self.assertEqual(sum([private_packets(r.state,r.trace,k)for k in range(2)],[]),n['privatePackets'],label+' exact private retry feedback')
                else:
                    assert_step(self,s,r,n,label)
                    self.assertTrue(bool(jp.all(r.trace.private_errors==0)))
                    # Native choose() flushes ordinary requests once its log
                    # exceeds500 rows. Canonical boundary requests remain exact;
                    # this direct oracle is not G1 BattleStream delivery timing.
                    for packet in n['privatePackets']:
                        recipient,payload=packet.split('\n',1);self.assertTrue(payload.startswith('|request|'),label)
                        request=json.loads(payload[len('|request|'):]);request.pop('update',None)
                        self.assertEqual(request,n['after']['sides'][int(recipient[1])-1]['request'],label+' flushed canonical request')
                s=r.state
                for k in range(2):
                    histories[k].extend(n['events'][f'p{k+1}']);self.assertEqual(player_view(s,k)['events'],histories[k],label)
            if c['setup'].get('fullBattle'):self.assertGreaterEqual(int(s.winner),0)

    def test_eager_jit_mixed_private_states_and_saved_continuation(self):
        names={'G4-trap-shadow-opponent-pending','G4-trap-query-frisk-undo','G4-trap-magnet-disclose-mirror','G4-trap-query-frisk-item-mirror','G4-trap-pivot-escapes'}
        inputs=[];outputs=[]
        for c in self.cases:
            if c['id']not in names:continue
            s,_=initialize_case(c)
            for n in c['steps']:
                a=choices_array(s,n['choices']);r=STEP(s,a)
                with jax.disable_jit():same(e.advance(s,a),r)
                with tempfile.TemporaryDirectory()as tmp:
                    p=Path(tmp)/'state.npz';save_state(p,s);restored=load_state(p)
                same(s,restored);same(STEP(restored,a),r)
                inputs.append((s,a));outputs.append(r);s=r.state
        stack=lambda xs:jax.tree.map(lambda *v:jp.stack(v),*xs)
        jax.clear_caches()
        for size in (1,8,32):
            indices=[n%len(inputs)for n in range(size)]
            actual=jax.jit(jax.vmap(e.advance))(stack([inputs[n][0]for n in indices]),jp.stack([inputs[n][1]for n in indices]))
            same(actual,stack([outputs[n]for n in indices]));jax.clear_caches()

    def test_complete_core_journal_full_lean_retry_rollout_and_reset(self):
        for c in [c for c in self.cases if c['setup'].get('fullBattle')or c['id']=='G4-trap-shadow-opponent-pending']:
            full,_=initialize_case(c);checkpoint=journal.from_state(full);start=checkpoint.core;plan=[]
            for n in c['steps']:
                a=choices_array(checkpoint.core,n['choices']);plan.append(a)
                checkpoint,r=journal.advance(checkpoint,a,step=CORE)
            plan=jp.stack(plan)
            lean=jax.jit(rollout)(start,plan)
            retained=jax.jit(lambda s,p:rollout(s,p,retain=('trace','codes')))(start,plan)
            same(lean.state,checkpoint.core);same(retained.state,checkpoint.core);self.assertFalse(bool(lean.failed))
            for side in range(2):
                expected=list(c['initialEvents'][f'p{side+1}'])
                for n in c['steps']:expected.extend(n['events'][f'p{side+1}'])
                self.assertEqual(journal.player_view(checkpoint,side)['events'],expected)
            if c['setup'].get('fullBattle'):
                self.assertTrue(bool(lean.terminal))
                reset=jax.jit(e.reset)(checkpoint.core,start,jp.stack((checkpoint.core.generation,checkpoint.core.episode)))
                self.assertEqual(int(reset.code),e.OK)
            else:self.assertIn(e.RETRY,np.asarray(retained.outputs['codes']).tolist())
            jax.clear_caches()

    def test_stale_invalid_privacy_atomic_save_and_legacy_provenance(self):
        c=next(c for c in self.cases if c['id']=='G4-trap-shadow-opponent-pending');s,_=initialize_case(c)
        s=STEP(s,choices_array(s,c['steps'][0]['choices'])).state
        old_token=jp.stack((s.generation,s.episode));r=STEP(s,e.i32([5,-1]));self.assertEqual(int(r.code),e.RETRY)
        self.assertEqual(int(r.state.pending[1]),int(s.pending[1]));self.assertGreater(int(r.state.episode),int(s.episode));same(r.state.rng,s.rng)
        self.assertEqual(private_packets(r.state,r.trace,1),[])
        for side in range(2):self.assertEqual(set(player_view(r.state,side)),{'request','events'})
        stale=STEP(r.state,e.i32([0,-1]),old_token);self.assertEqual(int(stale.code),e.STALE);same(stale.state,r.state)
        for choices in (e.i32([5,-3]),e.i32([5,0]),e.i32([99,-1])):
            invalid=STEP(s,choices);self.assertEqual(int(invalid.code),e.INVALID);same(invalid.state,s);self.assertTrue(bool(jp.all(invalid.trace.private_errors==0)))
        bad=s._replace(request_state=s.request_state.at[0,0].set(2));invalid=STEP(bad,e.i32([0,-1]));self.assertEqual(int(invalid.code),e.UNSUPPORTED);same(invalid.state,bad)
        cp=journal.from_state(s);cp=journal.append(cp,CORE(cp.core,e.i32([5,-1])))
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'journal.npz';journal.save(path,cp);same(journal.load(path).core,cp.core)
            original=path.read_bytes()
            with patch.object(journal.np,'savez',side_effect=OSError('simulated storage failure')):
                with self.assertRaises(OSError):journal.save(path,cp)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(len(list(Path(tmp).iterdir())),1)
            # Old journals lack private state: require independently matching
            # replay and preserve every recorded core field and journal page.
            names=tuple(n for n in e.CoreState._fields if n!='request_state')
            fields={n:np.asarray(getattr(cp.core,n))[:,:,:20,:] if n=='volatile_state' else np.asarray(getattr(cp.core,n))for n in names}
            meta={'journal_schema':1,'data_sha256':journal.tables.DATA_SHA256,'pages':[{'rows':p.rows,'recipients':p.recipients}for p in cp.pages]}
            payload=json.dumps(meta,sort_keys=True,separators=(',',':'));np.savez(path,metadata=np.asarray(payload),journal_sha256=np.asarray(journal._checksum(payload,fields,names)),**fields)
            with self.assertRaisesRegex(ValueError,'replay/provenance'):journal.load(path)
            same(journal.load(path,replayed=cp).core,cp.core)
            wrong=journal.Checkpoint(cp.core._replace(rng=cp.core.rng.at[0].add(1)),cp.pages)
            with self.assertRaisesRegex(ValueError,'mismatch'):journal.load(path,replayed=wrong)
            legacy=Path(tmp)/'legacy.npz';old={n:np.asarray(getattr(s,n))[:,:,:20,:] if n=='volatile_state' else np.asarray(getattr(s,n))for n in e.State._fields if n!='request_state'}
            np.savez(legacy,schema=10,data_sha256=np.asarray(journal.tables.DATA_SHA256),**old)
            with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(legacy)
            same(load_state(legacy,replayed_state=s),s)

if __name__=='__main__':unittest.main()

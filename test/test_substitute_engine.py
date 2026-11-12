"""Pinned Gen5 primary interception, causal bookkeeping and lossless continuation."""
import gzip,json,os,tempfile,unittest
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from simulator import shared_engine as e,journal
from simulator.shared_engine_io import draw_rows,player_view,save_state,load_state
from simulator.rollout import rollout
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step,assert_placement
STEP=jax.jit(e.advance)
CORE=jax.jit(e.advance_core)

def same(a,b):
    assert_placement((a,b))
    for x,y in zip(jax.tree.leaves(jax.device_get(a)),jax.tree.leaves(jax.device_get(b)),strict=True):np.testing.assert_array_equal(x,y)

class SubstituteEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('SUB_NATIVE_CASES')
        raw=Path(path).read_bytes()if path else gzip.decompress(Path('test-logs/g4-cpu-conformance/substitute.json.gz').read_bytes())
        cls.cases=json.loads(raw)['cases']
    def tearDown(self):jax.clear_caches()

    def test_native_state_requests_rng_effects_and_recipient_histories(self):
        self.assertEqual(len(self.cases),51);self.assertEqual(sum(len(c['steps'])for c in self.cases),184)
        for c in self.cases:
            s,t=initialize_case(c);assert_state(self,s,c['initial'],c['id']+' initial');self.assertEqual(draw_rows(t),c['initializationDraws'])
            histories=[list(c['initialEvents'][f'p{k+1}'])for k in range(2)]
            for i,n in enumerate(c['steps']):
                r=STEP(s,choices_array(s,n['choices']));label=f"{c['id']}/{i}"
                self.assertTrue(all(a['accepted']for a in n['accepted']),label)
                assert_step(self,s,r,n,label);s=r.state
                for k in range(2):
                    histories[k].extend(n['events'][f'p{k+1}']);self.assertEqual(player_view(s,k)['events'],histories[k],label)
                for packet in n['privatePackets']:
                    recipient,payload=packet.split('\n',1);self.assertTrue(payload.startswith('|request|'),label)
                    req=json.loads(payload[len('|request|'):]);req.pop('update',None)
                    self.assertEqual(req,n['after']['sides'][int(recipient[1])-1]['request'],label)
            if c['setup'].get('fullBattle'):self.assertGreaterEqual(int(s.winner),0)

    def test_eager_jit_mixed_batches_save_and_resume(self):
        names={'G4-substitute-break-multihit','G4-substitute-drain-ooze','G4-substitute-recoil-ko-mirror','G4-substitute-ThunderWave','G4-substitute-pivot'}
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
            ix=[n%len(inputs)for n in range(size)]
            actual=jax.jit(jax.vmap(e.advance))(stack([inputs[n][0]for n in ix]),jp.stack([inputs[n][1]for n in ix]))
            same(actual,stack([outputs[n]for n in ix]));jax.clear_caches()

    def test_complete_games_core_journal_rollout_reset(self):
        for c in self.cases:
            if not c['setup'].get('fullBattle'):continue
            s,_=initialize_case(c);cp=journal.from_state(s);start=cp.core;plan=[]
            for i,n in enumerate(c['steps']):
                a=choices_array(cp.core,n['choices']);plan.append(a);cp,r=journal.advance(cp,a,step=CORE)
                assert_state(self,cp.core,n['after'],c['id']+f'/core{i}')
                if i==1:
                    with tempfile.TemporaryDirectory()as tmp:
                        p=Path(tmp)/'journal.npz';journal.save(p,cp);cp=journal.load(p)
            plan=jp.stack(plan)
            lean=jax.jit(rollout)(start,plan);full=jax.jit(lambda s,p:rollout(s,p,retain=('trace','codes')))(start,plan)
            same(lean.state,cp.core);same(full.state,cp.core);self.assertTrue(bool(lean.terminal));self.assertFalse(bool(lean.failed));self.assertFalse(bool(lean.truncated))
            for k in range(2):
                expected=list(c['initialEvents'][f'p{k+1}'])
                for n in c['steps']:expected.extend(n['events'][f'p{k+1}'])
                self.assertEqual(journal.player_view(cp,k)['events'],expected)
            reset=jax.jit(e.reset)(cp.core,start,jp.stack((cp.core.generation,cp.core.episode)))
            self.assertEqual(int(reset.code),e.OK);same(reset.state,start._replace(generation=cp.core.generation+1))
            jax.clear_caches()

    def test_atomic_metadata_rejection_and_old_schema_provenance(self):
        c=next(c for c in self.cases if c['id']=='G4-substitute-Pound')
        start,_=initialize_case(c);s=STEP(start,choices_array(start,c['steps'][0]['choices'])).state
        for col,value in [(7,0),(7,int(s.stats[0,0,0]//4)+1),(8,1),(4,0),(2,1),(3,1),(1,int(s.effect_order))]:
            bad=s._replace(volatile_state=s.volatile_state.at[0,0,20,col].set(value));r=STEP(bad,e.i32([0,0]))
            self.assertEqual(int(r.code),e.UNSUPPORTED);same(r.state,bad);self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        for state,a,token,code in [(s,e.i32([99,0]),None,e.INVALID),(s,e.i32([0,0]),e.i32([-1,-1]),e.STALE),(s._replace(hcount=e.i32(e.HISTORY)),e.i32([0,0]),None,e.CAPACITY)]:
            r=STEP(state,a,token);self.assertEqual(int(r.code),code);same(r.state,state);self.assertEqual(int(r.trace.ecount),0);self.assertEqual(int(r.trace.dcount),0)
        # Schema11 cannot contain a Substitute slot; safe extension preserves all
        # recorded values. Current states cannot masquerade as old saves.
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'live-journal.npz';live=journal.from_state(s);journal.save(p,live);restored=journal.load(p);same(restored.core,live.core)
            a=choices_array(s,c['steps'][1]['choices']);next_cp,r=journal.advance(restored,a,step=CORE);same(r.state,e.core_state(STEP(s,a).state))
            for side in range(2):self.assertEqual(journal.player_view(next_cp,side),player_view(STEP(s,a).state,side))
            p=Path(tmp)/'old.npz';fields={name:np.asarray(getattr(start,name))for name in e.State._fields if name!='item_provenance'};fields['volatile_state']=fields['volatile_state'][:,:,:20,:];fields['action_state']=fields['action_state'][:,:,:21]
            np.savez(p,schema=11,data_sha256=np.asarray(journal.tables.DATA_SHA256),**fields)
            same(load_state(p),start);same(load_state(p,replayed_state=start),start)
            with self.assertRaisesRegex(ValueError,'mismatch'):load_state(p,replayed_state=s)
            fields['volatile_state']=np.asarray(start.volatile_state);np.savez(p,schema=11,data_sha256=np.asarray(journal.tables.DATA_SHA256),**fields)
            with self.assertRaisesRegex(ValueError,'volatile shape'):load_state(p)
            cp=journal.from_state(start);fields={name:np.asarray(getattr(cp.core,name))for name in e.CoreState._fields if name!='item_provenance'};fields['volatile_state']=fields['volatile_state'][:,:,:20,:];fields['action_state']=fields['action_state'][:,:,:21]
            meta={'journal_schema':2,'data_sha256':journal.tables.DATA_SHA256,'pages':[{'rows':page.rows,'recipients':page.recipients}for page in cp.pages]}
            payload=json.dumps(meta,sort_keys=True,separators=(',',':'));np.savez(p,metadata=np.asarray(payload),journal_sha256=np.asarray(journal._checksum(payload,fields,tuple(fields))),**fields)
            with self.assertRaisesRegex(ValueError,'replay/provenance'):journal.load(p)
            same(journal.load(p,replayed=cp).core,cp.core)
            with self.assertRaisesRegex(ValueError,'mismatch'):journal.load(p,replayed=journal.from_state(s))

if __name__=='__main__':unittest.main()

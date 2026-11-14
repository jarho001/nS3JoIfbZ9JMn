"""Grouped native Counter and Mirror Coat parity through one shared executable; broad gates deferred."""
import gzip,json,os,unittest,tempfile,time
from pathlib import Path
import jax,numpy as np
from simulator import shared_engine as e,journal
from simulator.shared_engine_io import draw_rows,player_view,private_packets,save_state,load_state
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step
STEP=jax.jit(e.advance)
def assert_legacy_checkpoint(s):
    self=unittest.TestCase()
    legacy=Path('test-logs/g4-cpu-conformance/retaliation-schema66-state.npz')
    restored=load_state(legacy)
    for a,b in zip(s,restored):np.testing.assert_array_equal(a,b)
    for a,b in zip(s,load_state(legacy,replayed_state=s)):np.testing.assert_array_equal(a,b)
    oldjournal=Path('test-logs/g4-cpu-conformance/retaliation-journal18.npz')
    with self.assertRaisesRegex(ValueError,'replay/provenance'):journal.load(oldjournal)
    cp=journal.from_state(s);migrated=journal.load(oldjournal,replayed=cp)
    for a,b in zip(cp.core,migrated.core):np.testing.assert_array_equal(a,b)
    self.assertEqual(cp.pages,migrated.pages)
    wrong=s._replace(hp=s.hp.at[0,0].add(-1))
    with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(legacy,replayed_state=wrong)
    with self.assertRaisesRegex(ValueError,'replay/provenance'):journal.load(oldjournal,replayed=journal.from_state(wrong))
    with tempfile.TemporaryDirectory()as tmp:
        p=Path(tmp)/'unsafe-old.npz'
        with np.load(legacy,allow_pickle=False)as old:fields={k:old[k]for k in old.files}
        fields['moves']=fields['moves'].copy();fields['moves'][0,0,0]=e.retaliation_ops.COUNTER
        np.savez(p,**fields)
        with self.assertRaisesRegex(ValueError,'replay/provenance'):load_state(p)
    return restored

class RetaliationEngineTest(unittest.TestCase):
    def test_native_retaliation(self):
        self.assertGreaterEqual(e.SCHEMA,67);self.assertGreaterEqual(journal.CORE_SCHEMA,19)
        path=os.environ.get('RETALIATION_NATIVE_CASES');raw=Path(path).read_bytes()if path else gzip.decompress(Path('test-logs/g4-cpu-conformance/retaliation.json.gz').read_bytes())
        cases=json.loads(raw)['cases'];self.assertEqual(len(cases),52)
        self.assertEqual(sum(len(c['steps'])for c in cases),70)
        print('cache settings: '+json.dumps({k:os.environ.get(k)for k in ('JAX_ENABLE_COMPILATION_CACHE','JAX_COMPILATION_CACHE_DIR')}),flush=True)
        legacy=next(c for c in cases if c['id']=='G4-retaliation-legacy-pressure');legacy_s,_=initialize_case(legacy)
        restored_legacy=assert_legacy_checkpoint(legacy_s)
        compiled=None;replay_started=time.monotonic();compile_seconds=0.0
        for c in cases:
            s,t=initialize_case(c);assert_state(self,s,c['initial'],c['id']+' initial');self.assertEqual(draw_rows(t),c['initializationDraws'])
            if c['id']=='G4-retaliation-legacy-pressure':s=restored_legacy
            histories=[list(c['initialEvents'][f'p{k+1}'])for k in range(2)]
            for i,n in enumerate(c['steps']):
                actions=choices_array(s,n['choices'])
                if compiled is None:
                    started=time.monotonic();compiled=STEP.lower(s,actions).compile();compile_seconds=time.monotonic()-started
                    print(f"advance lower/compile: {compile_seconds:.3f}s",flush=True)
                r=compiled(s,actions);label=f"{c['id']}/{i}"
                if any(not a['accepted']for a in n['accepted']):
                    self.assertEqual(int(r.code),e.RETRY,label);assert_state(self,r.state,n['after'],label)
                    self.assertEqual(sum([private_packets(r.state,r.trace,k)for k in range(2)],[]),n['privatePackets'],label)
                else:assert_step(self,s,r,n,label)
                s=r.state
                self.assertEqual(int(s.call_depth),0,label+' drained frame depth');self.assertFalse(bool(np.any(s.call_frames)),label+' drained frames')
                if c['id'] in ('G4-retaliation-counter','G4-retaliation-counter-pivot-mew','G4-retaliation-mirrorcoat-queued-fail','G4-retaliation-counter-terminal','G4-retaliation-mirrorcoat-blank-terminal') and i==0:
                    with tempfile.TemporaryDirectory()as tmp:
                        p=Path(tmp)/'retaliation.npz';save_state(p,s);restored=load_state(p)
                        for a,b in zip(s,restored):np.testing.assert_array_equal(a,b)
                        s=restored
                        cp=journal.from_state(s);journal.save(p,cp);saved=journal.load(p)
                        for a,b in zip(cp.core,saved.core):np.testing.assert_array_equal(a,b)
                        self.assertEqual(cp.pages,saved.pages)
                for k in range(2):histories[k].extend(n['events'][f'p{k+1}']);self.assertEqual(player_view(s,k)['events'],histories[k],label)
        print(f"fixture initialization/replay/assertions: {time.monotonic()-replay_started-compile_seconds:.3f}s",flush=True)
        jax.clear_caches()
if __name__=='__main__':unittest.main()

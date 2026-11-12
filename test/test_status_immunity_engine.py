"""Focused native base/effective Transform/Imposter parity; broad gates deferred."""
import gzip,json,os,unittest,tempfile
from pathlib import Path
import jax,numpy as np
from simulator import shared_engine as e,journal
from simulator.shared_engine_io import draw_rows,player_view,private_packets,save_state,load_state
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step
STEP=jax.jit(e.advance)
class StatusImmunityEngineTest(unittest.TestCase):
    def test_native_status_immunity_updates_and_acquisition(self):
        path=os.environ.get('STATUS_IMMUNITY_NATIVE_CASES');raw=Path(path).read_bytes()if path else gzip.decompress(Path('test-logs/g4-cpu-conformance/status-immunity.json.gz').read_bytes())
        cases=json.loads(raw)['cases'];self.assertEqual(len(cases),18)
        for c in cases:
            s,t=initialize_case(c);assert_state(self,s,c['initial'],c['id']+' initial');self.assertEqual(draw_rows(t),c['initializationDraws'])
            histories=[list(c['initialEvents'][f'p{k+1}'])for k in range(2)]
            for i,n in enumerate(c['steps']):
                r=STEP(s,choices_array(s,n['choices']));label=f"{c['id']}/{i}"
                if any(not a['accepted']for a in n['accepted']):
                    self.assertEqual(int(r.code),e.RETRY,label);assert_state(self,r.state,n['after'],label)
                    self.assertEqual(sum([private_packets(r.state,r.trace,k)for k in range(2)],[]),n['privatePackets'],label)
                else:assert_step(self,s,r,n,label)
                s=r.state
                if c['id']=='G4-status-immunity-acquire-Limber-par' and i==0:
                    with tempfile.TemporaryDirectory()as tmp:
                        p=Path(tmp)/'status-immunity.npz';save_state(p,s);restored=load_state(p)
                        for a,b in zip(s,restored):np.testing.assert_array_equal(a,b)
                        cp=journal.from_state(s);journal.save(p,cp);saved=journal.load(p)
                        for a,b in zip(cp.core,saved.core):np.testing.assert_array_equal(a,b)
                for k in range(2):histories[k].extend(n['events'][f'p{k+1}']);self.assertEqual(player_view(s,k)['events'],histories[k],label)
        jax.clear_caches()
if __name__=='__main__':unittest.main()

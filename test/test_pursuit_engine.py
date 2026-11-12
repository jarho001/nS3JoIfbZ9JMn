"""Focused native pursuit parity; broader execution gates remain deferred."""
import gzip,json,os,unittest,tempfile
from pathlib import Path
import jax
import numpy as np
from simulator import shared_engine as e,journal
from simulator.shared_engine_io import draw_rows,player_view,save_state,load_state
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step
STEP=jax.jit(e.advance)
class PursuitEngineTest(unittest.TestCase):
    def test_native_switch_pivot_interception_flags_rng_requests_and_histories(self):
        path=os.environ.get('PURSUIT_NATIVE_CASES')
        raw=Path(path).read_bytes()if path else gzip.decompress(Path('test-logs/g4-cpu-conformance/pursuit.json.gz').read_bytes())
        cases=json.loads(raw)['cases'];self.assertEqual(len(cases),23)
        for c in cases:
            s,t=initialize_case(c);assert_state(self,s,c['initial'],c['id']+' initial');self.assertEqual(draw_rows(t),c['initializationDraws'])
            histories=[list(c['initialEvents'][f'p{k+1}'])for k in range(2)]
            for i,n in enumerate(c['steps']):
                r=STEP(s,choices_array(s,n['choices']));label=f"{c['id']}/{i}"
                self.assertTrue(all(a['accepted']for a in n['accepted']),label)
                assert_step(self,s,r,n,label);self.assertFalse(bool(r.trace.force_switch.any()));s=r.state
                if c['id']=='G4-pursuit-pivot-slower' and i==0:
                    with tempfile.TemporaryDirectory()as tmp:
                        path=Path(tmp)/'saved.npz';save_state(path,s);saved=load_state(path)
                        for a,b in zip(s,saved):np.testing.assert_array_equal(a,b)
                        cp=journal.from_state(s);journal.save(path,cp);restored=journal.load(path)
                        for a,b in zip(cp.core,restored.core):np.testing.assert_array_equal(a,b)
                for k in range(2):histories[k].extend(n['events'][f'p{k+1}']);self.assertEqual(player_view(s,k)['events'],histories[k],label)
        jax.clear_caches()
if __name__=='__main__':unittest.main()

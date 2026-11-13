"""Guard shared Update holder order when Trace's callback body is looped."""
import gzip,json,unittest
from pathlib import Path
import jax
from simulator import shared_engine as e
from simulator.shared_engine_io import draw_rows,player_view
from test.shared_engine_support import initialize_case,choices_array,assert_state,assert_step

class UpdateOrderTest(unittest.TestCase):
    def test_native_both_holders_cure_after_skill_swap(self):
        cases=json.loads(gzip.decompress(Path('test-logs/g4-cpu-conformance/status-immunity.json.gz').read_bytes()))['cases']
        case=next(c for c in cases if c['id']=='G4-status-immunity-both-cure')
        s,t=initialize_case(case)
        assert_state(self,s,case['initial'],case['id']+' initial')
        self.assertEqual(draw_rows(t),case['initializationDraws'])
        histories=[list(case['initialEvents'][f'p{k+1}']) for k in range(2)]
        step=jax.jit(e.advance)
        for i,n in enumerate(case['steps']):
            r=step(s,choices_array(s,n['choices']));label=f"{case['id']}/{i}"
            assert_step(self,s,r,n,label);s=r.state
            for k in range(2):
                histories[k].extend(n['events'][f'p{k+1}'])
                self.assertEqual(player_view(s,k)['events'],histories[k],label)
        jax.clear_caches()

if __name__=='__main__':unittest.main()

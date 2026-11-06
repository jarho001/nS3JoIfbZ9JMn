import unittest
import jax
from simulator import shared_engine as e
from simulator.shared_engine_io import draw_rows
from test.shared_engine_support import native_cases,initialize_case,assert_state,choices_array,assert_step

COMPILED=jax.jit(e.advance)

class SharedEngineCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=[c for c in native_cases() if not c['setup'].get('fullBattle')]

    def test_native_initialization_and_boundaries(self):
        for case in self.cases:
            with self.subTest(case=case['id']):
                s,trace=initialize_case(case)
                self.assertEqual(draw_rows(trace),case['initializationDraws'],'native initialization draws')
                assert_state(self,s,case['initial'],case['id']+' initial')
                for n,step in enumerate(case['steps']):
                    before=s;r=COMPILED(s,choices_array(s,step['choices']))
                    assert_step(self,before,r,step,f"{case['id']}/{n+1}");s=r.state

if __name__=='__main__':unittest.main()

import unittest
import jax
import numpy as np
from simulator import shared_engine as e
from simulator.shared_engine_io import player_view
from test.shared_engine_support import native_cases,initialize_case,assert_state,choices_array,assert_step
from test.test_shared_engine_cases import COMPILED

class SharedEngineBattlesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=[c for c in native_cases() if c['setup'].get('fullBattle')]

    def test_complete_three_member_native_battles(self):
        self.assertGreaterEqual(len(self.cases),4,'empty/incomplete full-battle coverage')
        families=set();seeds=set();total=0
        for case in self.cases:
            with self.subTest(case=case['id']):
                self.assertEqual([len(case['setup'][key]) for key in ('p1','p2')],[3,3])
                families.add(case['setup']['family']);seeds.add(case['setup']['seed'])
                s,_=initialize_case(case)
                assert_state(self,s,case['initial'],case['id']+' initial')
                histories={side:list(case['initialEvents'][f'p{side+1}']) for side in range(2)}
                for n,step in enumerate(case['steps']):
                    before=s;r=COMPILED(s,choices_array(s,step['choices']))
                    assert_step(self,before,r,step,f"{case['id']}/{n+1}");s=r.state;total+=1
                    for side in range(2):
                        histories[side].extend(step['events'][f'p{side+1}'])
                        self.assertEqual(player_view(s,side)['events'],histories[side],case['id']+' complete recipient history')
                self.assertGreaterEqual(int(s.winner),0,'a turn cap is not terminal evidence')
                self.assertFalse(np.any(e.legal_actions(s)))
                self.assertEqual(np.asarray(e.rewards(s)).tolist(),[1,-1] if int(s.winner)==1 else [-1,1] if int(s.winner)==2 else [0,0])
                self.assertEqual(player_view(s,0)['request'],None)
                # Absorbing terminal submissions cannot consume randomness.
                r=COMPILED(s,e.i32([0,0]));self.assertEqual(int(r.code),e.TERMINAL)
                for a,b in zip(jax.tree.leaves(s),jax.tree.leaves(r.state)):np.testing.assert_array_equal(a,b)
        self.assertEqual(families,{'direct','status','pivot','exhaustion'})
        self.assertGreaterEqual(len(seeds),2)
        print(f'Native complete-battle comparison: {len(self.cases)} battles, {total} boundaries, families={sorted(families)}',flush=True)

if __name__=='__main__':unittest.main()

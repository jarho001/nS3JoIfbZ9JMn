import unittest
import json
from pathlib import Path
import jax
from simulator import shared_engine as e
from simulator.shared_engine_io import draw_rows,player_view
from test.shared_engine_support import native_cases,initialize_case,assert_state,choices_array,assert_step

COMPILED=jax.jit(e.advance)

class SharedEngineCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=[c for c in native_cases() if not c['setup'].get('fullBattle')]

    def test_manifest_and_nonempty_family_coverage(self):
        manifest=json.loads((Path(__file__).resolve().parents[1]/'rules/shared-engine-v1.json').read_text())
        self.assertEqual(manifest['moves'],list(e.MOVE_NAMES[1:e.PROTECT+1]))
        self.assertEqual(manifest['trace']['context'],list(e.DRAW_CONTEXTS[:12]))
        self.assertEqual(list(manifest['species']),list(e.LEGACY_SPECIES[1:]))
        self.assertEqual(manifest['items'],list(e.LEGACY_ITEMS))
        self.assertEqual([manifest['storage'][k] for k in ('queue_rows','events_per_advance','draws_per_advance','history_events')],[e.QUEUE,e.EVENTS,e.DRAWS,e.HISTORY])
        self.assertGreaterEqual(len(self.cases),80)
        ids={c['id'] for c in self.cases}
        for name in ('legacy-pound-tied','wisp-hit','toxic-type-immunity','pivot-resume','forced-replacement','struggle-double-ko','struggle-protected','protect-success-256','M03-last-tackle','M04-residual-ko','privacy-hp-a','privacy-b'):
            self.assertIn(name,ids)
        self.assertEqual(set().union(*(set(d['semanticContext'] for d in step['draws']) for c in self.cases for step in c['steps'])),set(e.DRAW_CONTEXTS[1:12]))

    def test_native_initialization_and_boundaries(self):
        for case in self.cases:
            with self.subTest(case=case['id']):
                s,trace=initialize_case(case)
                self.assertEqual(draw_rows(trace),case['initializationDraws'],'native initialization draws')
                assert_state(self,s,case['initial'],case['id']+' initial')
                for side in range(2):self.assertEqual(player_view(s,side)['events'],case['initialEvents'][f'p{side+1}'],case['id']+' initialization view')
                histories={side:list(case['initialEvents'][f'p{side+1}']) for side in range(2)}
                for n,step in enumerate(case['steps']):
                    before=s;r=COMPILED(s,choices_array(s,step['choices']))
                    assert_step(self,before,r,step,f"{case['id']}/{n+1}");s=r.state
                    for side in range(2):
                        histories[side].extend(step['events'][f'p{side+1}'])
                        self.assertEqual(player_view(s,side)['events'],histories[side],case['id']+' complete recipient history')

if __name__=='__main__':unittest.main()

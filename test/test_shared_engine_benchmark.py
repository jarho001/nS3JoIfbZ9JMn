"""Validate benchmark rollout/reset accounting against independent native plans."""
import unittest
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch,Mock

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.shared_engine import plans, rollout
from benchmarks import shared_engine_campaign
from simulator import shared_engine as e
from test.shared_engine_support import native_cases, assert_step, assert_placement


class SharedEngineBenchmarkTest(unittest.TestCase):
    def test_native_producer_failure_cannot_be_successful_empty_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'config.json';out=root/'evidence'
            config.write_text(json.dumps({'schemaVersion':1,'cases':[{}],'targetBoundaries':100000}))
            process=Mock();process.stdout=io.StringIO('');process.wait.return_value=7;process.poll.return_value=7
            argv=['campaign','--config',str(config),'--output',str(out),'--platform',jax.default_backend()]
            with patch('sys.argv',argv),patch.object(shared_engine_campaign,'identity',return_value={'test':'producer failure'}),patch.object(shared_engine_campaign.subprocess,'Popen',return_value=process):
                with self.assertRaisesRegex(RuntimeError,'native producer exit 7'):
                    shared_engine_campaign.main()
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['status'],'incomplete_native_runner')
            self.assertEqual(summary['starts'],0)
            self.assertEqual(summary['completed'],0)

    def test_materialized_scan_repeats_only_after_native_terminal_and_reset(self):
        cases=native_cases();case=next(c for c in cases if c['id']=='M03-duel-accuracy')
        states,plan,length,initial,indices,_=plans(cases,'ordinary',1)
        size=len(case['steps']);chunk=size*2+1
        result=jax.jit(rollout(chunk))(states,indices,jnp.zeros(8,jnp.int32),initial,plan,length)
        assert_placement(result)
        before=jax.tree.map(lambda x:x[0],states);generation=0
        for boundary in range(chunk):
            if int(before.winner)>=0:
                generation+=1;before=jax.tree.map(lambda x:x[0],initial)._replace(generation=e.i32(generation))
            actual=jax.tree.map(lambda x:x[boundary,0],result[1])
            assert_step(self,before,actual,case['steps'][boundary%size],f'benchmark/{boundary}')
            self.assertEqual(int(actual.state.generation),generation)
            before=actual.state
        counts=np.asarray(result[0][2])
        self.assertEqual(counts[:6].tolist(),[chunk,2*chunk,2,2,0,0])
        self.assertEqual(int(result[0][1][0]),1)
        self.assertEqual(int(counts[6]),int(np.sum(result[1].trace.ecount)))
        self.assertEqual(int(counts[7]),int(np.sum(result[1].trace.dcount)))


if __name__=='__main__':unittest.main()

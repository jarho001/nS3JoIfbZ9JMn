"""Handoff failure/retention checks; no JAX import, GPU, or battle execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import g4_gpu_handoff as h


class GpuHandoffTest(unittest.TestCase):
    def test_native_failure_retains_failed_summary_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'evidence'
            with patch.object(h,'run',side_effect=RuntimeError('simulated native failure')):
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    code=h.main(['--prepare-only','--output',str(out)])
            self.assertEqual(code,1)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['status'],'failed')
            self.assertIn('simulated native failure',summary['error'])
            self.assertFalse(summary['full_G4_accepted'])
            with tarfile.open(out.with_name('evidence.tar.gz')) as archive:
                self.assertIsNotNone(archive.getmember('evidence/summary.json'))

    def test_cache_and_cold_options_override_inherited_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'cache';cache.mkdir()
            marker=cache/'keep';marker.write_text('cached executable')
            for cold in (False,True):
                out=Path(tmp)/str(cold);seen=[]
                def stop_after_environment(command,log,env,timeout,records):
                    seen.append(env.copy())
                    raise RuntimeError('stop before native execution')
                argv=['--prepare-only','--output',str(out)]+(['--cold']if cold else ['--cache-dir',str(cache)])
                with self.subTest(cold=cold),patch.dict(os.environ,JAX_ENABLE_COMPILATION_CACHE='false',JAX_COMPILATION_CACHE_DIR='inherited'),patch.object(h,'run',side_effect=stop_after_environment):
                    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(h.main(argv),1)
                    settings=json.loads((out/'summary.json').read_text())['compilation_cache']
                    self.assertEqual(settings['JAX_ENABLE_COMPILATION_CACHE'],'false'if cold else 'true')
                    if cold:
                        self.assertNotIn('JAX_COMPILATION_CACHE_DIR',seen[0])
                        self.assertNotIn('JAX_COMPILATION_CACHE_DIR',settings)
                    else:
                        self.assertEqual(settings['JAX_COMPILATION_CACHE_DIR'],str(cache))
                        self.assertEqual(seen[0]['JAX_COMPILATION_CACHE_DIR'],str(cache))
                    self.assertEqual(marker.read_text(),'cached executable')

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'evidence';out.mkdir();marker=out/'keep.txt';marker.write_text('original')
            with self.assertRaises(FileExistsError):h.main(['--prepare-only','--output',str(out)])
            self.assertEqual(marker.read_text(),'original')

    def test_timeout_stops_child_and_records_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=[]
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(subprocess.TimeoutExpired):
                h.run([sys.executable,'-c','import time; time.sleep(30)'],Path(tmp)/'timeout.log',
                      {},0.1,records)
            self.assertNotEqual(records[0]['exit_code'],0)
            self.assertEqual(records[0]['timeout_seconds'],0.1)


if __name__=='__main__':unittest.main()

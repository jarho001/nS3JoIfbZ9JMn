"""Handoff failure/retention checks; no JAX import, GPU, or battle execution."""
import contextlib
import io
import json
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

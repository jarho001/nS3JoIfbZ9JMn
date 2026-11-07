"""Portable evidence loading contracts; not executed in the implementation task."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import g3_fixtures as f


class G3FixturesTest(unittest.TestCase):
    def test_historical_absolute_path_and_line_endings_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "simulator/example.py"
            path.parent.mkdir()
            lf = b"value = 1\nother = 2\n"
            path.write_bytes(lf)
            recorded = hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()
            with patch.object(f, "ROOT", root):
                self.assertEqual(f.source_path("/workspace/simulator/example.py"), path)
                verified = f.verify_source("/workspace/simulator/example.py", recorded)
                self.assertEqual(verified["match"], "line-endings-only")
                self.assertEqual(verified["lf_sha256"], hashlib.sha256(lf).hexdigest())
                path.write_bytes(b"value = 3\nother = 2\n")
                with self.assertRaises(ValueError):
                    f.verify_source("simulator/example.py", recorded)

    def test_source_paths_cannot_escape_repository_roots(self):
        for path in ("../simulator/example.py", "/etc/passwd", "unrecognized/example.py"):
            with self.assertRaises(ValueError):
                f.source_path(path)

    def test_saved_pack_is_complete_and_uses_the_current_pin(self):
        pack = f.frozen_pack()
        self.assertEqual(tuple(c["id"] for c in pack["cases"]), f.CASE_IDS)
        self.assertTrue(all(c["steps"][-1]["after"]["terminal"] for c in pack["cases"]))


if __name__ == "__main__":
    unittest.main()

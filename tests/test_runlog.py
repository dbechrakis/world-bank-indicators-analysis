import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from wb_indicators.runlog import tee_stdout

ROOT = Path(__file__).resolve().parents[1]


class TeeTests(unittest.TestCase):
    def test_output_reaches_both_places_and_stdout_is_restored(self):
        original = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "log.txt"
                with tee_stdout(path):
                    print("hello")
                self.assertIs(sys.stdout, captured)
                self.assertEqual(path.read_text(), "hello\n")
        finally:
            sys.stdout = original
        self.assertEqual(captured.getvalue(), "hello\n")

    def test_stdout_is_restored_when_the_block_raises(self):
        original = sys.stdout
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                with tee_stdout(Path(folder) / "log.txt"):
                    raise ValueError("boom")
        self.assertIs(sys.stdout, original)


class ScriptExitCodeTests(unittest.TestCase):
    def test_a_missing_data_file_fails_the_run(self):
        with tempfile.TemporaryDirectory() as folder:
            empty = Path(folder) / "empty"
            empty.mkdir()
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "world_bank_analysis.py"),
                 "--data-dir", str(empty), "--output-dir", str(Path(folder) / "out")],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 1)
            self.assertIn("CRITICAL ERROR", result.stderr)
            self.assertTrue((Path(folder) / "out" / "project_output.txt").exists())


if __name__ == "__main__":
    unittest.main()

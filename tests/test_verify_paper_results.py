import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_paper_results.py"
EVIDENCE = ROOT / "pess_2026_rl_luce" / "evidence"


class VerifyPaperResultsTests(unittest.TestCase):
    def test_headline_results_match_committed_evidence(self):
        if not EVIDENCE.exists():
            self.fail(
                "pess_2026_rl_luce/evidence/ is missing - the paper's result "
                "evidence must ship with the repository"
            )
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True, text=True, cwd=ROOT,
        )
        self.assertEqual(
            result.returncode, 0,
            f"verify_paper_results.py failed:\n{result.stdout}\n{result.stderr}",
        )
        self.assertIn("All headline results match", result.stdout)


if __name__ == "__main__":
    unittest.main()

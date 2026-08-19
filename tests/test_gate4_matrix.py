import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Gate4MatrixTests(unittest.TestCase):
    def test_exact_seeded_affected_matrix(self):
        matrix = json.loads((ROOT / "gate4" / "matrix.json").read_text())
        self.assertEqual(matrix["shared"]["training_seed"], 0)
        self.assertFalse(matrix["shared"]["deterministic"])
        actual = {
            (
                run["tag"], run["mode"], run["window_W"], run["alpha"],
                run["false_positive_penalty"], run["correct_reward"],
            )
            for run in matrix["runs"]
        }
        expected = {
            ("phasor_W48", "phasor", 48, 0.5, -100.0, 5.0),
            ("combined_W48", "combined", 48, 0.5, -100.0, 5.0),
            ("phasor_W96", "phasor", 96, 0.5, -100.0, 5.0),
            ("combined_W96", "combined", 96, 0.5, -100.0, 5.0),
            ("combined_W48_fp10", "combined", 48, 0.5, -10.0, 5.0),
            ("combined_W48_fp200", "combined", 48, 0.5, -200.0, 5.0),
            ("combined_W48_fp100", "combined", 48, 0.5, -100.0, 50.0),
            ("combined_W48_alpha0.1", "combined", 48, 0.1, -100.0, 5.0),
            ("combined_W48_alpha0.9", "combined", 48, 0.9, -100.0, 5.0),
        }
        self.assertEqual(actual, expected)

    def test_posthoc_gamma_run_is_not_in_the_prespecified_matrix(self):
        matrix = json.loads((ROOT / "gate4" / "matrix.json").read_text())
        tags = {run["tag"] for run in matrix["runs"]}
        self.assertNotIn("combined_W48_gamma099", tags)
        self.assertEqual(matrix["shared"]["gamma"], 0.95)

    def test_posthoc_gamma_config_inherits_combined_w48_defaults(self):
        matrix = json.loads((ROOT / "gate4" / "matrix.json").read_text())
        posthoc = json.loads(
            (ROOT / "gate4" / "posthoc_gamma099.json").read_text()
        )
        self.assertEqual(posthoc["protocol"], "exploratory post-hoc")
        self.assertEqual(len(posthoc["runs"]), 1)

        # shared settings inherit the prespecified defaults except gamma
        shared_base = dict(matrix["shared"])
        shared_posthoc = dict(posthoc["shared"])
        self.assertEqual(shared_posthoc.pop("gamma"), 0.99)
        self.assertEqual(shared_base.pop("gamma"), 0.95)
        self.assertEqual(shared_posthoc, shared_base)

        # the single run matches the default combined_W48 configuration
        base_run = next(
            run for run in matrix["runs"] if run["tag"] == "combined_W48"
        )
        run = posthoc["runs"][0]
        self.assertEqual(run["tag"], "combined_W48_gamma099")
        for key in ("mode", "window_W", "alpha",
                    "false_positive_penalty", "correct_reward"):
            self.assertEqual(run[key], base_run[key])


if __name__ == "__main__":
    unittest.main()

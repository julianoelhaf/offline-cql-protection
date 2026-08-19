import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_event_type_tables():
    """Import run_ablation._event_type_tables without its heavy dependencies."""
    with mock.patch.dict(sys.modules, {
        "torch": mock.MagicMock(),
        "tqdm": mock.MagicMock(),
        "matplotlib": mock.MagicMock(),
        "matplotlib.pyplot": mock.MagicMock(),
        "rl_protection.constants": mock.MagicMock(),
        "rl_protection.expert": mock.MagicMock(),
        "rl_protection.models": mock.MagicMock(),
        "rl_protection.reward": mock.MagicMock(),
    }):
        sys.modules.pop("run_ablation", None)
        import run_ablation
        table_fn = run_ablation._event_type_tables
    # Drop the partially mocked module so other tests import a clean copy.
    sys.modules.pop("run_ablation", None)
    return table_fn


class EventTypeBreakdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.event_type_tables = staticmethod(_load_event_type_tables())

    @staticmethod
    def _frame(rows):
        return pd.DataFrame(
            rows,
            columns=["tag", "event_type", "sim_idx", "is_fault",
                     "TP", "FN", "FP", "TN"],
        )

    def test_f1_uses_fn_not_tn_in_denominator(self):
        # Large TN would corrupt F1 if the denominator regressed to
        # 2*TP + FP + TN instead of 2*TP + FP + FN.
        full = self._frame([
            ("run", "flt_x", 0, True, 80, 20, 10, 1_000_000),
        ])
        fault, _ = self.event_type_tables(full)
        expected_f1 = 2 * 80 / (2 * 80 + 10 + 20)
        self.assertAlmostEqual(fault["f1"].iloc[0], expected_f1, places=12)
        # Guard against the old bug: with TN in the denominator the value
        # would collapse towards zero.
        self.assertGreater(fault["f1"].iloc[0], 0.5)

    def test_fault_metrics_match_hand_computed_values(self):
        full = self._frame([
            ("run", "flt_a", 0, True, 30, 10, 5, 100),
            ("run", "flt_a", 1, True, 20, 40, 15, 200),
            ("run", "flt_b", 2, True, 50, 0, 0, 300),
        ])
        fault, nonfault = self.event_type_tables(full)
        self.assertTrue(nonfault.empty)

        flt_a = fault[fault["event_type"] == "flt_a"].iloc[0]
        # Aggregated over the two flt_a episodes: TP=50, FN=50, FP=20, TN=300.
        self.assertAlmostEqual(flt_a["recall"], 50 / (50 + 50), places=12)
        self.assertAlmostEqual(flt_a["f1"], 2 * 50 / (2 * 50 + 20 + 50), places=12)

        flt_b = fault[fault["event_type"] == "flt_b"].iloc[0]
        self.assertAlmostEqual(flt_b["recall"], 1.0, places=12)
        self.assertAlmostEqual(flt_b["f1"], 1.0, places=12)

    def test_nonfault_fpr(self):
        full = self._frame([
            ("run", "switch_x", 0, False, 0, 0, 25, 75),
        ])
        fault, nonfault = self.event_type_tables(full)
        self.assertTrue(fault.empty)
        self.assertAlmostEqual(nonfault["fpr"].iloc[0], 25 / (25 + 75), places=12)

    def test_zero_denominators_yield_nan(self):
        full = self._frame([
            ("run", "flt_empty", 0, True, 0, 0, 0, 0),
            ("run", "switch_empty", 1, False, 0, 0, 0, 0),
        ])
        fault, nonfault = self.event_type_tables(full)
        self.assertTrue(np.isnan(fault["recall"].iloc[0]))
        self.assertTrue(np.isnan(fault["f1"].iloc[0]))
        self.assertTrue(np.isnan(nonfault["fpr"].iloc[0]))


if __name__ == "__main__":
    unittest.main()

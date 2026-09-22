import unittest
from pathlib import Path

from pess_2026_rl_luce.figure_scripts.make_terminal_action_examples import (
    EXPECTED_COUNTS,
    build_figure_data,
    load_context,
    select_examples,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT / "outputs" / "leakage_free" / "gate4_seedonly_3779240"
    / "runs" / "combined_W48"
)


@unittest.skipUnless(
    (RUN / "raw_predictions.npz").is_file(),
    "seeded Gate 4 predictions are not available locally",
)
class TerminalActionExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = load_context(
            RUN,
            ROOT / "labels" / "val_indices.npz",
            ROOT / "labels" / "settings.csv",
            ROOT / "rl_protection" / "constants.py",
        )
        cls.fault, cls.nonfault = select_examples(cls.context)
        cls.frame = build_figure_data(cls.context, cls.fault, cls.nonfault)

    def test_current_aggregate_and_deterministic_selection(self):
        self.assertEqual(self.context.counts, EXPECTED_COUNTS)
        self.assertEqual(self.fault["simulation_id"], 655)
        self.assertEqual(self.nonfault["simulation_id"], 4449)

    def test_terminal_examples(self):
        self.assertEqual(self.fault["outcome"], "correct_first_trip")
        self.assertTrue(self.fault["later_wrong_indices"])
        self.assertEqual(self.nonfault["outcome"], "false_first_trip")
        self.assertGreater(self.nonfault["n_nuisance_steps"], 0)

    def test_paper_figure_episode_details(self):
        sample_rate = self.context.constants["SAMPLE_RATE"]

        # Fault example (sim 655): correct first line-trip action at ~+0.10 ms
        # after fault onset, with later wrong-line predictions that do not
        # change the terminal outcome.
        fault_delay_ms = (
            (self.fault["first_trip_index"] - self.fault["onset_index"])
            / sample_rate * 1000
        )
        self.assertAlmostEqual(fault_delay_ms, 0.10, places=2)
        self.assertTrue(all(
            index > self.fault["first_trip_index"]
            for index in self.fault["later_wrong_indices"]
        ))
        self.assertEqual(self.fault["outcome"], "correct_first_trip")

        # Non-fault example (sim 4449): the first non-wait action is a
        # nuisance trip at ~+0.42 ms after the switching event.
        nonfault_delay_ms = (
            (self.nonfault["first_trip_index"] - self.nonfault["onset_index"])
            / sample_rate * 1000
        )
        self.assertAlmostEqual(nonfault_delay_ms, 0.42, places=2)
        self.assertEqual(self.nonfault["outcome"], "false_first_trip")

        # Event metadata matches the paper captions.
        settings = self.context.settings
        self.assertEqual(
            settings.loc[655, "events/event_type"], "flt_1phg_shc")
        self.assertEqual(settings.loc[655, "events/event_target"], "MainLn3-8")
        self.assertEqual(
            settings.loc[4449, "events/event_target"], "MainBus8")
        self.assertFalse(
            settings.loc[4449, "events/event_type"].startswith("flt_"))

    def test_post_trip_predictions_are_retained_and_marked(self):
        for example in (self.fault, self.nonfault):
            rows = self.frame[
                self.frame["simulation_id"] == example["simulation_id"]
            ].reset_index(drop=True)
            first = int(rows.index[rows["is_first_trip"]][0])
            self.assertEqual(len(rows), self.context.actions.shape[1])
            self.assertFalse(rows.loc[:first, "ignored_after_first_trip"].any())
            self.assertTrue(rows.loc[first + 1:, "ignored_after_first_trip"].all())


if __name__ == "__main__":
    unittest.main()

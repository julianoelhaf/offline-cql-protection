import unittest

import numpy as np

from gate5.first_trip import classify_episode


class FirstTripTests(unittest.TestCase):
    def classify(self, actions, is_fault=True, expert=3):
        return classify_episode(
            np.asarray(actions),
            t_start=958,
            is_fault=is_fault,
            expert_action=expert,
            wait_action=15,
            fault_onset=960,
            sample_rate=9600,
            n_steps=4800,
        )

    def test_terminal_first_trip_outcomes_and_times(self):
        premature = self.classify([15, 3, 3])
        self.assertEqual(premature["outcome"], "premature_trip")
        self.assertAlmostEqual(premature["relative_delay_ms"], -1 / 9.6)

        correct = self.classify([15, 15, 3, 4])
        self.assertEqual(correct["outcome"], "correct_first_trip")
        self.assertAlmostEqual(correct["relative_delay_ms"], 0.0)

        wrong = self.classify([15, 15, 4, 3])
        self.assertEqual(wrong["outcome"], "wrong_relay_first_trip")
        self.assertEqual(wrong["first_action"], 4)

        false_trip = self.classify([15, 15, 4], is_fault=False, expert=15)
        self.assertEqual(false_trip["outcome"], "false_first_trip")

        censored = self.classify([15, 15, 15])
        self.assertEqual(censored["outcome"], "no_trip")
        self.assertTrue(censored["censored"])
        self.assertEqual(censored["censor_index"], 4799)


if __name__ == "__main__":
    unittest.main()

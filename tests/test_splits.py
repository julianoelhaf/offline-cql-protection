import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"


def read_ids(name: str) -> list[int]:
    with open(SPLITS / name, newline="") as handle:
        reader = csv.DictReader(handle)
        return [int(row["sim_idx"]) for row in reader]


class SplitManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.development = read_ids("development_episode_ids.csv")
        cls.optimization = read_ids("optimization_episode_ids.csv")
        cls.monitoring = read_ids("monitoring_episode_ids.csv")
        cls.evaluation = read_ids("evaluation_episode_ids.csv")
        with open(SPLITS / "episodes.csv", newline="") as handle:
            cls.episodes = list(csv.DictReader(handle))

    def test_partition_sizes(self):
        self.assertEqual(len(self.development), 4282)
        self.assertEqual(len(self.optimization), 3853)
        self.assertEqual(len(self.monitoring), 429)
        self.assertEqual(len(self.evaluation), 225)

    def test_no_episode_crosses_a_partition_boundary(self):
        development = set(self.development)
        optimization = set(self.optimization)
        monitoring = set(self.monitoring)
        evaluation = set(self.evaluation)

        # No duplicate ids within any file.
        self.assertEqual(len(development), len(self.development))
        self.assertEqual(len(optimization), len(self.optimization))
        self.assertEqual(len(monitoring), len(self.monitoring))
        self.assertEqual(len(evaluation), len(self.evaluation))

        self.assertEqual(optimization & monitoring, set())
        self.assertEqual(development & evaluation, set())
        self.assertEqual(optimization | monitoring, development)

    def test_ids_cover_exactly_0_to_4506(self):
        union = set(self.development) | set(self.evaluation)
        self.assertEqual(union, set(range(4507)))

    def test_episode_metadata_is_consistent_with_id_files(self):
        self.assertEqual(len(self.episodes), 4507)
        by_partition = {"optimization": set(), "monitoring": set(),
                        "evaluation": set()}
        n_fault = 0
        eval_fault = 0
        for row in self.episodes:
            sim_idx = int(row["sim_idx"])
            partition = row["partition"]
            self.assertIn(partition, by_partition)
            by_partition[partition].add(sim_idx)
            is_fault = row["is_fault"] == "True"
            self.assertEqual(is_fault, row["event_type"].startswith("flt_"))
            n_fault += is_fault
            if partition == "evaluation":
                eval_fault += is_fault

        self.assertEqual(by_partition["optimization"], set(self.optimization))
        self.assertEqual(by_partition["monitoring"], set(self.monitoring))
        self.assertEqual(by_partition["evaluation"], set(self.evaluation))

        self.assertEqual(n_fault, 4353)
        self.assertEqual(len(self.episodes) - n_fault, 154)
        self.assertEqual(eval_fault, 214)
        self.assertEqual(len(self.evaluation) - eval_fault, 11)

    def test_paper_figure_episodes_are_in_the_evaluation_set(self):
        evaluation = set(self.evaluation)
        self.assertIn(655, evaluation)
        self.assertIn(4449, evaluation)
        meta = {int(row["sim_idx"]): row for row in self.episodes}
        self.assertEqual(meta[655]["event_type"], "flt_1phg_shc")
        self.assertEqual(meta[655]["event_target"], "MainLn3-8")
        self.assertEqual(meta[655]["is_fault"], "True")
        self.assertEqual(meta[4449]["is_fault"], "False")
        self.assertEqual(meta[4449]["event_target"], "MainBus8")

    def test_manifest_matches_frozen_label_artifacts_when_present(self):
        labels = ROOT / "labels"
        val_pt = labels / "validation" / "val_labels.pt"
        if not val_pt.exists():
            self.skipTest("local label artifacts not available")
        import torch

        val = torch.load(val_pt, weights_only=False)
        # Exact order match: the manifest preserves the frozen storage order
        # used by raw_predictions.npz and the figure pipeline.
        self.assertEqual([int(i) for i in val["indices"]], self.evaluation)
        train_pt = labels / "train_labels.pt"
        if train_pt.exists():
            train = torch.load(train_pt, weights_only=False)
            self.assertEqual(sorted(int(i) for i in train["indices"]),
                             sorted(self.development))


if __name__ == "__main__":
    unittest.main()

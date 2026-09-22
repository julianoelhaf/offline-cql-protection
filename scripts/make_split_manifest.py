"""Regenerate the committed episode-split manifest in splits/.

Maintainer tool. It reads the frozen label artifacts (the same files whose
SHA-256 values are pinned in gate4/matrix.json) and writes the lightweight,
text-based split manifest that ships with the public repository:

  splits/development_episode_ids.csv   4,282 episodes (optimization + monitoring)
  splits/optimization_episode_ids.csv  3,853 episodes used for gradient updates
  splits/monitoring_episode_ids.csv      429 episodes used for internal monitoring
  splits/evaluation_episode_ids.csv      225 held-out evaluation episodes
  splits/episodes.csv                  per-episode metadata for all 4,507 episodes

The optimization/monitoring partition replicates, bit for bit, the episode-level
split performed at training time by rl_protection.dataset.build_dataloader
(train_frac=0.9, split_seed=0): a seed-0 NumPy permutation over the unique
episode ids present in the (warm-up-filtered) transition table. The script
asserts that the warm-up filter never removes a whole episode, so the split is
identical for every (mode, window) configuration in the study.

Usage:
  python scripts/make_split_manifest.py [--labels-dir labels] [--output-dir splits]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Mirrors rl_protection/constants.py (not imported: that module requires the
# raw dataset to be present at import time).
CYCLE_LEN = 192

# Mirrors rl_protection/dataset.py::build_dataloader defaults, pinned in
# gate4/matrix.json ("split_seed": 0).
TRAIN_FRAC = 0.9
SPLIT_SEED = 0

# Every (mode, window) configuration in the prespecified matrix and the
# post-hoc run. The warm-up filter depends on these; the resulting episode
# split must not.
MODE_WINDOW_GRID = [
    ("phasor", 48), ("phasor", 96),
    ("combined", 48), ("combined", 96),
]

EXPECTED = {
    "development": 4282,
    "optimization": 3853,
    "monitoring": 429,
    "evaluation": 225,
    "total": 4507,
    "fault_total": 4353,
    "nonfault_total": 154,
    "evaluation_fault": 214,
    "evaluation_nonfault": 11,
}


def load_indices(pt_path: Path) -> np.ndarray:
    import torch

    data = torch.load(pt_path, weights_only=False)
    return np.asarray(data["indices"], dtype=np.int64)


def monitoring_split(transitions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the episode-level split from dataset.build_dataloader."""
    reference = None
    for mode, window in MODE_WINDOW_GRID:
        if mode in ("phasor", "combined"):
            keep = transitions[:, 1] >= CYCLE_LEN - 1 + window
        else:
            keep = np.ones(len(transitions), dtype=bool)
        unique_sims = np.unique(transitions[keep, 0])
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(len(unique_sims))
        n_train = int(TRAIN_FRAC * len(unique_sims))
        split = (
            np.sort(unique_sims[perm[:n_train]]),
            np.sort(unique_sims[perm[n_train:]]),
        )
        if reference is None:
            reference = split
        elif not (np.array_equal(reference[0], split[0])
                  and np.array_equal(reference[1], split[1])):
            raise AssertionError(
                f"warm-up filter changed the episode split for {mode} W={window}; "
                "the split is no longer configuration-independent"
            )
    return reference


def write_id_csv(path: Path, ids: np.ndarray, sort: bool = True) -> None:
    ordered = np.sort(ids) if sort else np.asarray(ids)
    pd.DataFrame({"sim_idx": ordered.astype(int)}).to_csv(path, index=False)
    print(f"  wrote {path} ({len(ids)} episodes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=REPO / "labels")
    parser.add_argument("--output-dir", type=Path, default=REPO / "splits")
    args = parser.parse_args()

    labels = args.labels_dir
    dev_ids = load_indices(labels / "train_labels.pt")
    eval_ids = load_indices(labels / "validation" / "val_labels.pt")
    transitions = np.load(labels / "transitions.npz")["transitions"]
    settings = pd.read_csv(labels / "settings.csv", sep=";")

    opt_ids, mon_ids = monitoring_split(transitions)

    dev_set, eval_set = set(dev_ids.tolist()), set(eval_ids.tolist())
    opt_set, mon_set = set(opt_ids.tolist()), set(mon_ids.tolist())

    # ── Structural invariants ────────────────────────────────────────────────
    assert len(dev_set) == EXPECTED["development"], len(dev_set)
    assert len(eval_set) == EXPECTED["evaluation"], len(eval_set)
    assert len(opt_set) == EXPECTED["optimization"], len(opt_set)
    assert len(mon_set) == EXPECTED["monitoring"], len(mon_set)
    assert opt_set & mon_set == set(), "optimization/monitoring overlap"
    assert dev_set & eval_set == set(), "development/evaluation overlap"
    assert opt_set | mon_set == dev_set, "optimization ∪ monitoring != development"
    assert dev_set | eval_set == set(range(EXPECTED["total"])), (
        "episode ids are not exactly 0..4506"
    )

    # ── Metadata invariants ──────────────────────────────────────────────────
    meta = settings.set_index("general/sim_idx")
    missing = (dev_set | eval_set) - set(meta.index.tolist())
    assert not missing, f"settings.csv is missing sim ids: {sorted(missing)[:5]}"

    def partition_of(sim_idx: int) -> str:
        if sim_idx in eval_set:
            return "evaluation"
        return "optimization" if sim_idx in opt_set else "monitoring"

    rows = []
    for sim_idx in range(EXPECTED["total"]):
        event_type = meta.loc[sim_idx, "events/event_type"]
        rows.append({
            "sim_idx": sim_idx,
            "partition": partition_of(sim_idx),
            "event_type": event_type,
            "event_target": meta.loc[sim_idx, "events/event_target"],
            "is_fault": event_type.startswith("flt_"),
        })
    episodes = pd.DataFrame(rows)

    n_fault = int(episodes["is_fault"].sum())
    assert n_fault == EXPECTED["fault_total"], n_fault
    assert len(episodes) - n_fault == EXPECTED["nonfault_total"]
    eval_rows = episodes[episodes["partition"] == "evaluation"]
    assert int(eval_rows["is_fault"].sum()) == EXPECTED["evaluation_fault"]
    assert int((~eval_rows["is_fault"]).sum()) == EXPECTED["evaluation_nonfault"]

    # ── Write manifest ───────────────────────────────────────────────────────
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_id_csv(out / "development_episode_ids.csv", dev_ids)
    write_id_csv(out / "optimization_episode_ids.csv", opt_ids)
    write_id_csv(out / "monitoring_episode_ids.csv", mon_ids)
    # The evaluation ids keep their frozen storage order: raw_predictions.npz
    # and the figure pipeline store episodes in exactly this order.
    write_id_csv(out / "evaluation_episode_ids.csv", eval_ids, sort=False)
    episodes.to_csv(out / "episodes.csv", index=False)
    print(f"  wrote {out / 'episodes.csv'} ({len(episodes)} episodes)")
    print("All split invariants hold.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"SPLIT MANIFEST ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

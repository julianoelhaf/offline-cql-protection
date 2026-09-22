# Episode split manifest

This directory freezes the exact episode partition used in the paper, as plain
simulation IDs. It is the single public source of truth for which EvEMTBench
simulation belongs to which partition.

## Files

| File | Episodes | Meaning |
| --- | ---: | --- |
| `development_episode_ids.csv` | 4,282 | All episodes available during development (optimization + internal monitoring). |
| `optimization_episode_ids.csv` | 3,853 | Development episodes used for gradient updates. |
| `monitoring_episode_ids.csv` | 429 | Development episodes used only for internal monitoring during training. |
| `evaluation_episode_ids.csv` | 225 | Held-out evaluation set (214 fault + 11 non-fault episodes). All paper results are reported on these episodes. Row order is the frozen storage order used by `raw_predictions.npz` and the figure pipeline — do not sort it. |
| `episodes.csv` | 4,507 | One row per episode: `sim_idx`, `partition`, `event_type`, `event_target`, `is_fault`. |

Invariants (asserted by `tests/test_splits.py` and by
`scripts/make_split_manifest.py`):

- `len(development) == 4282`, `len(optimization) == 3853`,
  `len(monitoring) == 429`, `len(evaluation) == 225`
- `optimization ∩ monitoring == {}` and `development ∩ evaluation == {}`
- `optimization ∪ monitoring == development`
- `development ∪ evaluation` is exactly the simulation IDs `0–4506`

## How the partition was produced

- The development/evaluation partition is the frozen project split shipped in
  `labels/train_labels.pt` / `labels/validation/val_labels.pt` (their SHA-256
  values are pinned in `gate4/matrix.json`).
- The optimization/monitoring partition replicates the episode-level split
  performed at training time by
  `rl_protection.dataset.build_dataloader(train_frac=0.9, split_seed=0)`:
  a seed-0 NumPy permutation over the sorted development episode IDs. The
  split is identical for every (representation, window) configuration in the
  study; `scripts/make_split_manifest.py` asserts this.

## Dataset composition

The experiments use exactly **4,507 simulations with IDs 0–4506** from
EvEMTBench: **4,353 fault episodes and 154 non-fault episodes**.

The current cleaned EvEMTBench metadata contains 4,509 simulations (156
non-fault episodes). The two additional records, IDs **4507 and 4508**, are
non-fault `switch_ibr_trip` events that were **not used** in this project —
neither for training nor for evaluation. `scripts/prepare_paper_data.py`
excludes them explicitly.

# Offline Reinforcement Learning for Distribution-Grid Protection

Official code release for the PESS 2026 paper **"Offline Reinforcement
Learning for Distribution-Grid Protection."**

**Julian Oelhaf**\*†, **Alexander Luce**\*, Christian Bergler, Andreas Maier,
and Siming Bayer

\* These authors contributed equally to this work. † Corresponding author.

Pattern Recognition Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg ·
Ostbayerische Technische Hochschule Amberg-Weiden

[![CI](https://github.com/julianoelhaf/offline-cql-protection/actions/workflows/ci.yml/badge.svg)](https://github.com/julianoelhaf/offline-cql-protection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository studies line-selective protection tripping from fixed
trajectories of a realistically simulated CIGRE medium-voltage grid. A
convolutional Q-network is trained with conservative Q-learning (CQL) using
causal phasor and impedance features, optionally combined with raw waveforms.

> **Scope.** The experiments use static simulated trajectories. They do not
> model closed-loop grid interaction, and the security evaluation contains
> only 11 non-fault episodes in the held-out evaluation set. The code and
> reported results must not be interpreted as evidence of deployment
> readiness.

## Highlights

- Causal voltage-current phasor and impedance features: every decision uses
  only present and preceding samples.
- A fixed, seeded, **prespecified nine-run sensitivity matrix** covering
  representations, observation windows, reward variants, and CQL weights —
  plus one clearly separated **exploratory post-hoc** γ=0.99 run.
- Two complementary evaluations on the held-out evaluation set: dense
  per-timestep classification and terminal first-trip behavior.
- Reproducibility checks for input hashes, code provenance, RNG-complete
  checkpoints, finite outputs, and immutable result directories.
- Committed episode-split manifest ([`splits/`](splits/)) and committed result
  evidence ([`pess_2026_rl_luce/evidence/`](pess_2026_rl_luce/evidence/)) so
  the paper's headline numbers can be verified from a fresh clone without a
  GPU or the dataset.

## Headline results (held-out evaluation set)

The held-out evaluation set contains 225 episodes: 214 fault and 11 non-fault
episodes. Per-timestep metrics:

| Study | Configuration | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|
| Prespecified (default) | Combined, W=48, α=0.5 | 0.9946 | 0.9347 | 0.9637 | 2.05% |
| Prespecified (best dense) | Combined, W=48, α=0.9 | 0.9993 | 0.9496 | **0.9738** | 0.28% |
| Exploratory post-hoc | Combined, W=48, γ=0.99 | 0.9997 | 0.9422 | 0.9701 | 0.10% |

Terminal first-trip behavior of the prespecified default (combined, W=48,
α=0.5):

| Evaluation | Result |
|---|---:|
| Correct line tripped first (fault episodes) | **210/214 (98.13%)** |
| Wrong line tripped first / no trip | 1/214 / 3/214 |
| Non-fault episodes with a false trip | **8/11 (72.73%)** |
| Correct-trip latency, median / p95 | 0.104 / 1.667 ms |

The terminal results are deliberately reported beside the strong per-timestep
scores: a single nuisance trip determines the outcome of a non-fault episode,
even when most individual wait decisions are correct.

Verify all of these against the committed evidence with:

```bash
python scripts/verify_paper_results.py
```

## Dataset

The experiments use the public **EvEMTBench** dataset (Kordowich et al.,
2026), available from the [FAU Data Cloud](https://data.fau.de/share/0e8d60feb7e65616c60aab78b93db77053275da53fd894bf5b75fc5e9ee7dfbf/).
The raw data is **not** redistributed in this repository.

This project uses exactly **4,507 simulations with IDs 0–4506**:

| Composition | Episodes |
|---|---:|
| Fault episodes | 4,353 |
| Non-fault episodes | 154 |
| **Total** | **4,507** |

The current cleaned EvEMTBench metadata contains 4,509 simulations (156
non-fault episodes). The two additional records, **IDs 4507 and 4508**, are
non-fault `switch_ibr_trip` events that were **not used** in this project.
`scripts/prepare_paper_data.py` excludes them explicitly; do not extend the
experiments to those IDs.

## Episode splits

The exact partition used in the paper is committed as plain simulation-ID
lists in [`splits/`](splits/) (see [`splits/README.md`](splits/README.md)):

| Partition | Episodes |
|---|---:|
| Development | 4,282 |
| — Optimization (gradient updates) | 3,853 |
| — Internal monitoring | 429 |
| Held-out evaluation | 225 (214 fault / 11 non-fault) |

No episode crosses a partition boundary; `tests/test_splits.py` asserts the
counts, disjointness, and that the union is exactly IDs 0–4506. The term
"validation" appears in some internal file and variable names
(`val_labels.pt`, `validation_trajectories.csv`, …); those files describe the
**held-out evaluation set**. The 429-episode internal-monitoring partition is
used only for monitoring during training.

## Repository structure

```text
rl_protection/     preprocessing, features, models, rewards, training, evaluation
gate4/             prespecified nine-run matrix, post-hoc γ=0.99 config, Slurm runner
gate5/             terminal first-trip evaluator
scripts/           data preparation, split manifest, and result verification tools
splits/            committed episode-split manifest (IDs only)
pess_2026_rl_luce/
  evidence/        committed final evaluation artifacts backing the paper tables
  figure_scripts/  generator for the paper's terminal-action TikZ panels
tests/             causal-feature, matrix, split, resume, evaluation, and figure tests
run_ablation.py    training and dense evaluation routines used by Gate 4
requirements.txt   versions recorded for the final experiment environment
```

Large source data, processed arrays, labels, checkpoints, and generated
results are intentionally excluded from Git.

## Installation

```bash
git clone https://github.com/julianoelhaf/offline-cql-protection.git
cd offline-cql-protection

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The final Gate 4 runs used PyTorch `2.6.0+cu124` on NVIDIA A100 MIG instances.
Install the PyTorch build appropriate for the local CUDA stack when GPU
training is required.

Run the code-only checks after installation:

```bash
python -m unittest discover -s tests -v
```

Tests requiring the external experiment artifacts skip automatically when
those artifacts are absent. The split and paper-result tests run from a fresh
clone.

GitHub Actions runs this same command on every push and pull request, on
Python 3.12 against the pinned `requirements.txt`, with no GPU and no dataset
(`.github/workflows/ci.yml`). It also parses the Slurm job scripts and guards
against site-internal paths reappearing.

## Reproduction workflow

End to end, a reproduction consists of:

1. Clone this repository and install the requirements (above).
2. Obtain the public EvEMTBench dataset (link above).
3. Prepare the exact 4,507-episode project subset:
   `scripts/prepare_paper_data.py` (below).
4. The committed splits in `splits/` are reconstructed into the label files by
   step 3; `python -m rl_protection.preprocess all` builds the feature arrays.
5. Run the nine prespecified experiments (`sbatch --array=0-8%4` or per-tag
   `gate4/run_one.py`).
6. Run the separate exploratory post-hoc γ=0.99 experiment.
7. Dense per-timestep metrics are produced per run (`metrics.json`,
   `validation_trajectories.csv`).
8. Evaluate terminal first-trip metrics with `gate5/first_trip.py`.
9. Generate the paper figure panels with
   `pess_2026_rl_luce/figure_scripts/make_terminal_action_examples.py`.
10. Verify the headline results with `scripts/verify_paper_results.py`.

### Step 3 - prepare the project dataset

```bash
python scripts/prepare_paper_data.py \
  --data-root /path/to/evemtbench \
  --split-dir splits/ \
  --output-dir prepared/
```

The script validates the EvEMTBench metadata against the committed split
manifest and fails loudly if simulations 0–4506 are missing, if unexpected IDs
are present, if event labels/targets differ from the frozen project metadata,
or if the split counts do not match the paper. It writes
`prepared/labels/{settings.csv, train_labels.pt, validation/val_labels.pt,
val_indices.npz}` and arranges the raw episode CSVs into `prepared/data/` and
`prepared/data/validation/` (hardlinks by default; `--raw-transfer copy`
copies, `--raw-transfer none` skips the raw files).

Then point the pipeline at the prepared tree and build the feature arrays:

```bash
export POWER_GRID_DATA_DIR="$PWD/prepared/data"
ln -s "$PWD/prepared/labels" labels   # or copy the directory
python -m rl_protection.preprocess all
```

The authoritative experiment matrix in
[`gate4/matrix.json`](gate4/matrix.json) records the expected episode counts
and SHA-256 manifests. `PHASOR_SHA256_LEDGER` must point to the checksum
ledger for the corrected phasor arrays.

## Reproducing the prespecified nine-run matrix

The nine prespecified configurations share the same split, training seed,
30-epoch stopping point, and held-out evaluation episodes. On Slurm:

```bash
export PROJECT="$PWD"
export GATE4_ENV="/path/to/venv"
export GATE4_OUTPUT="/path/to/new/output-root"
export POWER_GRID_DATA_DIR="/path/to/prepared/data"
export PHASOR_SHA256_LEDGER="/path/to/phasor_sha256.txt"

sbatch --array=0-8%4 gate4/gate4.sbatch
```

Each task:

1. verifies the input manifests;
2. records the resolved configuration, source hashes, environment, and command;
3. trains and retains all 30 checkpoints;
4. exports dense evaluation metrics, trajectories, actions, and Q-values;
5. rejects missing, malformed, or non-finite outputs; and
6. writes a completion marker and output checksum ledger.

Output directories are immutable by default. Use a new output root for a new
attempt; failed attempts should be preserved rather than overwritten.

Without Slurm, a single configuration runs with:

```bash
python gate4/run_one.py combined_W48 \
  --matrix gate4/matrix.json \
  --output-root /path/to/new/output-root/runs
```

## Exploratory post-hoc γ=0.99 experiment

The paper additionally reports **one exploratory post-hoc run** (combined,
W=48, α=0.5, γ=0.99, seed 0, 30 epochs). It was executed **after** the
prespecified matrix was frozen and completed, and is therefore kept in a
separate configuration file —
[`gate4/posthoc_gamma099.json`](gate4/posthoc_gamma099.json) — which inherits
the default combined-W48 setup and changes only `gamma = 0.99`. It is **not**
part of the `--array=0-8` matrix above and must not be treated as
prespecified.

```bash
python gate4/run_one.py combined_W48_gamma099 \
  --matrix gate4/posthoc_gamma099.json \
  --output-root /path/to/new/posthoc-output-root/runs
```

(On Slurm, submit it as a single job with the same environment variables as
the matrix; do not extend the array.)

### Reproducibility boundary

The reported runs use training seed `0`, seeded data loading, and checkpoints
containing Python, NumPy, CPU, CUDA, and DataLoader RNG states. Strict
deterministic CUDA kernels are intentionally disabled because they were
prohibitively slow. Seed reproducibility is supported, but bitwise identity
across different hardware or CUDA stacks is not claimed.

`gate4/run_one.py` refuses to train unless the label files match the SHA-256
manifests frozen in the matrix files. `scripts/prepare_paper_data.py` prints
the hashes of the files it regenerates and compares them against the
manifests; byte-identical regeneration has been confirmed with the pinned
package versions, but is not guaranteed under other torch/pandas versions.

## Terminal first-trip evaluation

After all nine prediction archives exist:

```bash
python gate5/first_trip.py \
  --runs-root /path/to/output-root/runs \
  --output /path/to/new/first-trip-output \
  --settings labels/settings.csv \
  --matrix gate4/matrix.json \
  --training-commit "$(git rev-parse HEAD)"
```

For every episode, the first action other than wait is terminal and all later
actions are ignored (they still count toward the dense per-timestep metrics).
The evaluator reports the line-trip outcome per fault episode — correct
faulted line, wrong line, or no trip (in the code: `correct_first_trip`,
`wrong_relay_first_trip`, `no_trip`) — plus non-fault false trips, Wilson 95%
confidence intervals, and correct-trip latency. The median-latency interval
uses a fixed-seed 10,000-sample bootstrap.

## Reproducing the paper figure

The plot-only TikZ panels for the representative terminal-action examples
(fault simulation 655, non-fault simulation 4449) are generated directly from
the default `combined_W48` prediction archive:

```bash
python pess_2026_rl_luce/figure_scripts/make_terminal_action_examples.py \
  --run-dir /path/to/output-root/runs/combined_W48 \
  --split labels/val_indices.npz
```

Before selecting the two fixed episodes, the script verifies the run
configuration, final checkpoint, evaluation split, and aggregate first-trip
counts against the paper. Generated tables and renderings remain excluded
from Git.

## Verifying the paper's headline results

The final evaluation artifacts behind the paper's tables are committed under
[`pess_2026_rl_luce/evidence/`](pess_2026_rl_luce/evidence/) (dense metric
summaries, per-episode first-trip outcomes for all nine runs, and the
post-hoc γ=0.99 comparison). A fresh clone can verify every headline value
without a GPU or the dataset:

```bash
python scripts/verify_paper_results.py
python -m unittest tests.test_splits tests.test_verify_paper_results -v
```

[`pess_2026_rl_luce/traceability.csv`](pess_2026_rl_luce/traceability.csv)
maps each number printed in the manuscript to its evidence file.

## Citation

Bibliographic metadata will be updated after publication. Until then, cite the
manuscript as:

```bibtex
@misc{oelhaf2026offline,
  title  = {Offline Reinforcement Learning for Distribution-Grid Protection},
  author = {Oelhaf, Julian and Luce, Alexander and Bergler, Christian and
            Maier, Andreas and Bayer, Siming},
  year   = {2026},
  url    = {https://github.com/julianoelhaf/offline-cql-protection},
  note   = {Manuscript submitted for publication. Julian Oelhaf and
            Alexander Luce contributed equally.}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

This project is released under the MIT License; see [`LICENSE`](LICENSE).
The EvEMTBench dataset is distributed separately under its own terms.

## Contact

For questions about the code or experiments, contact
[Julian Oelhaf](mailto:julian.oelhaf@fau.de).

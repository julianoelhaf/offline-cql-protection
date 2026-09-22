"""Prepare the exact 4,507-episode project dataset from public EvEMTBench data.

Reconstructs every derived label artifact the experiment pipeline expects,
using only (1) the public EvEMTBench download and (2) the committed split
manifest in splits/. It fails loudly whenever the inputs deviate from the
frozen project dataset.

  python scripts/prepare_paper_data.py \
      --data-root <EVEMTBENCH_ROOT> \
      --split-dir splits/ \
      --output-dir prepared/

Outputs (under --output-dir):

  labels/settings.csv              exact 4,507-row project metadata subset
  labels/train_labels.pt           {tensor, columns, indices} for 4,282 dev episodes
  labels/validation/val_labels.pt  {tensor, columns, indices} for 225 eval episodes
  labels/val_indices.npz           evaluation ids in the frozen storage order
  data/result*.csv                 raw development episodes  (hardlink or copy)
  data/validation/result*.csv      raw evaluation episodes   (hardlink or copy)

Afterwards point the pipeline at the prepared tree, e.g.:

  export POWER_GRID_DATA_DIR=<output-dir>/data
  # place/link <output-dir>/labels at <repo>/labels
  python -m rl_protection.preprocess all

Hard failure modes (by design):
  - any of simulations 0-4506 missing from the metadata or the raw files
  - unexpected simulation ids other than the documented unused 4507/4508
  - event types/targets that differ from the committed project metadata
  - split counts that do not match the paper (4282/3853/429/225)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

N_EPISODES = 4507          # project simulations, ids 0..4506
KNOWN_EXTRA_IDS = {4507, 4508}  # cleaned EvEMTBench extras, unused here

EXPECTED_COUNTS = {
    "development": 4282,
    "optimization": 3853,
    "monitoring": 429,
    "evaluation": 225,
}

# One-hot label layout used by the frozen project label files.
LABEL_COLUMNS = [
    "is_fault", "is_no_fault",
    "flt_1phg_shc", "flt_1phg_shc_w_arc",
    "flt_1phg_hif", "flt_1phg_hif_w_arc",
    "flt_1phg_incipient", "flt_1phg_incipient_w_arc",
    "flt_2ph_shc", "flt_2phg_shc", "flt_3ph_shc",
]


class PreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path: Path) -> list[int]:
    frame = pd.read_csv(path)
    return [int(v) for v in frame["sim_idx"]]


def find_settings_csv(data_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise PreparationError(f"--settings does not exist: {explicit}")
        return explicit
    candidates = sorted(data_root.rglob("settings*.csv"),
                        key=lambda p: len(p.parts))
    if not candidates:
        raise PreparationError(
            f"no settings*.csv found under {data_root}; "
            "pass --settings <path> explicitly"
        )
    return candidates[0]


def index_result_files(data_root: Path) -> dict[int, Path]:
    """Map sim_idx -> raw result CSV path found under the data root."""
    mapping: dict[int, Path] = {}
    for path in data_root.rglob("result*.csv"):
        stem = path.stem
        suffix = stem.removeprefix("result")
        if not suffix.isdigit():
            continue
        sim_idx = int(suffix)
        # Prefer the shallower path when duplicates exist.
        if sim_idx not in mapping or len(path.parts) < len(mapping[sim_idx].parts):
            mapping[sim_idx] = path
    return mapping


def validate_metadata(settings: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    if "general/sim_idx" not in settings.columns:
        raise PreparationError(
            "metadata has no 'general/sim_idx' column - wrong file or separator?"
        )
    ids = set(int(v) for v in settings["general/sim_idx"])
    required = set(range(N_EPISODES))
    missing = sorted(required - ids)
    if missing:
        raise PreparationError(
            f"metadata is missing {len(missing)} project simulations, "
            f"first missing ids: {missing[:10]}"
        )
    extras = ids - required
    unknown_extras = sorted(extras - KNOWN_EXTRA_IDS)
    if unknown_extras:
        raise PreparationError(
            f"metadata contains unexpected simulation ids: {unknown_extras[:10]}"
        )
    if extras:
        extra_rows = settings[settings["general/sim_idx"].isin(sorted(extras))]
        extra_types = set(extra_rows["events/event_type"])
        print(f"  note: ignoring unused simulations {sorted(extras)} "
              f"(event types: {sorted(extra_types)}) - they are not part of "
              "this project's 4,507-episode dataset")
        if extra_types - {"switch_ibr_trip"}:
            raise PreparationError(
                "the extra simulations are documented as non-fault "
                f"switch_ibr_trip events, but found {sorted(extra_types)}"
            )

    subset = settings[settings["general/sim_idx"] < N_EPISODES].copy()
    subset = subset.sort_values("general/sim_idx").reset_index(drop=True)

    expected = episodes.set_index("sim_idx")
    got_types = dict(zip(subset["general/sim_idx"], subset["events/event_type"]))
    got_targets = dict(zip(subset["general/sim_idx"], subset["events/event_target"]))
    mismatches = []
    for sim_idx in range(N_EPISODES):
        if got_types[sim_idx] != expected.loc[sim_idx, "event_type"] or \
                got_targets[sim_idx] != expected.loc[sim_idx, "event_target"]:
            mismatches.append(sim_idx)
    if mismatches:
        raise PreparationError(
            f"event labels/targets differ from the expected project metadata "
            f"for {len(mismatches)} simulations, first ids: {mismatches[:10]}"
        )
    return subset


def build_label_dict(ids: list[int], episodes: pd.DataFrame) -> dict:
    import torch

    event_type = dict(zip(episodes["sim_idx"], episodes["event_type"]))
    col_index = {name: i for i, name in enumerate(LABEL_COLUMNS)}
    tensor = torch.zeros((len(ids), len(LABEL_COLUMNS)), dtype=torch.float32)
    for row, sim_idx in enumerate(ids):
        etype = event_type[sim_idx]
        flag = "is_fault" if etype.startswith("flt_") else "is_no_fault"
        tensor[row, col_index[flag]] = 1.0
        if etype in col_index:
            tensor[row, col_index[etype]] = 1.0
    return {"tensor": tensor, "columns": list(LABEL_COLUMNS), "indices": list(ids)}


def place_raw_files(ids: list[int], sources: dict[int, Path], dest: Path,
                    transfer: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for sim_idx in ids:
        target = dest / f"result{sim_idx}.csv"
        if target.exists():
            continue
        source = sources[sim_idx]
        if transfer == "link":
            try:
                target.hardlink_to(source)
            except OSError as exc:
                raise PreparationError(
                    f"hardlinking {source} -> {target} failed ({exc}); "
                    "rerun with --raw-transfer copy (needs ~30 GB) or "
                    "--raw-transfer none"
                ) from exc
        else:
            shutil.copy2(source, target)


def compare_manifest_hashes(produced: dict[str, Path]) -> None:
    matrix_path = REPO / "gate4" / "matrix.json"
    if not matrix_path.exists():
        return
    manifests = json.loads(matrix_path.read_text(encoding="utf-8"))
    expected = manifests["shared"]["input_manifests"]
    print("\nSHA-256 comparison against gate4/matrix.json input_manifests:")
    any_mismatch = False
    for key, path in produced.items():
        digest = sha256(path)
        status = "MATCH" if digest == expected.get(key) else "DIFFERS"
        any_mismatch |= status == "DIFFERS"
        print(f"  {key:18s} {digest}  [{status}]")
    if any_mismatch:
        print(
            "  Content was validated above, but at least one file is not "
            "byte-identical to the frozen original (torch/pandas "
            "serialization is only bit-stable in the pinned environment). "
            "gate4/run_one.py verifies these hashes before training and "
            "will refuse regenerated inputs; see README 'Reproducibility "
            "boundary'."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="root of the public EvEMTBench download")
    parser.add_argument("--split-dir", type=Path, default=REPO / "splits")
    parser.add_argument("--output-dir", type=Path, default=REPO / "prepared")
    parser.add_argument("--settings", type=Path, default=None,
                        help="explicit path to the EvEMTBench metadata CSV "
                             "(default: search --data-root)")
    parser.add_argument("--raw-transfer", choices=["link", "copy", "none"],
                        default="link",
                        help="how to place raw result CSVs into the prepared "
                             "layout (default: hardlink)")
    args = parser.parse_args()

    if not args.data_root.exists():
        raise PreparationError(f"--data-root does not exist: {args.data_root}")

    # ── Load and validate the committed split manifest ───────────────────────
    dev_ids = read_ids(args.split_dir / "development_episode_ids.csv")
    opt_ids = read_ids(args.split_dir / "optimization_episode_ids.csv")
    mon_ids = read_ids(args.split_dir / "monitoring_episode_ids.csv")
    eval_ids = read_ids(args.split_dir / "evaluation_episode_ids.csv")
    episodes = pd.read_csv(args.split_dir / "episodes.csv")

    counts = {
        "development": len(dev_ids),
        "optimization": len(opt_ids),
        "monitoring": len(mon_ids),
        "evaluation": len(eval_ids),
    }
    if counts != EXPECTED_COUNTS:
        raise PreparationError(
            f"split manifest counts {counts} do not match the paper "
            f"{EXPECTED_COUNTS}"
        )
    if set(opt_ids) | set(mon_ids) != set(dev_ids):
        raise PreparationError("optimization ∪ monitoring != development")
    if set(dev_ids) & set(eval_ids):
        raise PreparationError("development and evaluation sets overlap")
    if set(dev_ids) | set(eval_ids) != set(range(N_EPISODES)):
        raise PreparationError("split ids are not exactly 0..4506")

    # ── Validate EvEMTBench metadata against the frozen project metadata ─────
    settings_path = find_settings_csv(args.data_root, args.settings)
    print(f"Metadata: {settings_path}")
    settings = pd.read_csv(settings_path, sep=";")
    if settings.shape[1] == 1:  # not semicolon-separated after all
        settings = pd.read_csv(settings_path)
    subset = validate_metadata(settings, episodes)
    print(f"  validated {len(subset)} project simulations (ids 0-4506)")

    # ── Locate raw episode files ─────────────────────────────────────────────
    if args.raw_transfer != "none":
        sources = index_result_files(args.data_root)
        missing_raw = sorted(set(range(N_EPISODES)) - set(sources))
        if missing_raw:
            raise PreparationError(
                f"{len(missing_raw)} raw result CSVs are missing under "
                f"{args.data_root}, first missing ids: {missing_raw[:10]}"
            )

    # ── Write the prepared layout ────────────────────────────────────────────
    out = args.output_dir
    labels_dir = out / "labels"
    (labels_dir / "validation").mkdir(parents=True, exist_ok=True)

    settings_out = labels_dir / "settings.csv"
    if len(settings) == N_EPISODES and int(settings["general/sim_idx"].max()) == N_EPISODES - 1:
        # Already the exact project subset: copy verbatim to preserve bytes.
        shutil.copyfile(settings_path, settings_out)
    else:
        subset.to_csv(settings_out, sep=";", index=False)
    print(f"  wrote {settings_out}")

    import torch

    train_out = labels_dir / "train_labels.pt"
    torch.save(build_label_dict(dev_ids, episodes), train_out)
    print(f"  wrote {train_out} ({len(dev_ids)} episodes)")

    val_out = labels_dir / "validation" / "val_labels.pt"
    torch.save(build_label_dict(eval_ids, episodes), val_out)
    print(f"  wrote {val_out} ({len(eval_ids)} episodes, frozen order)")

    val_idx_out = labels_dir / "val_indices.npz"
    np.savez(val_idx_out, indices=np.asarray(eval_ids, dtype=np.int32))
    print(f"  wrote {val_idx_out}")

    if args.raw_transfer != "none":
        place_raw_files(dev_ids, sources, out / "data", args.raw_transfer)
        print(f"  placed {len(dev_ids)} development episodes in {out / 'data'}")
        place_raw_files(eval_ids, sources, out / "data" / "validation",
                        args.raw_transfer)
        print(f"  placed {len(eval_ids)} evaluation episodes in "
              f"{out / 'data' / 'validation'}")

    compare_manifest_hashes({
        "settings_csv": settings_out,
        "train_labels": train_out,
        "validation_labels": val_out,
    })
    print("\nDone. Next: set POWER_GRID_DATA_DIR to the prepared data "
          "directory, place the prepared labels at <repo>/labels, and run "
          "'python -m rl_protection.preprocess all'.")


if __name__ == "__main__":
    try:
        main()
    except PreparationError as exc:
        print(f"PREPARATION ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

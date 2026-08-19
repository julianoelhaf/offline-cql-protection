"""Run one committed Gate 4 configuration end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_ablation
from rl_protection.constants import CYCLE_LEN, N_STEPS
from rl_protection.models import build_model


SOURCE_FILES = [
    "rl_protection/preprocess.py",
    "rl_protection/dataset.py",
    "rl_protection/models.py",
    "rl_protection/train_cql.py",
    "run_ablation.py",
    "gate4/matrix.json",
    "gate4/run_one.py",
    "gate4/gate4.sbatch",
    "gate4/smoke_repro.py",
    "gate4/smoke.sbatch",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(matrix_path: Path, tag: str) -> dict:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    run = next((item for item in matrix["runs"] if item["tag"] == tag), None)
    if run is None:
        raise ValueError(f"unknown Gate 4 tag: {tag}")
    return {**matrix["shared"], **run}


def verify_inputs(config: dict) -> None:
    expected = config["input_manifests"]
    label_files = {
        "settings_csv": ROOT / "labels" / "settings.csv",
        "train_labels": ROOT / "labels" / "train_labels.pt",
        "transitions": ROOT / "labels" / "transitions.npz",
        "validation_labels": ROOT / "labels" / "validation" / "val_labels.pt",
    }
    for name, path in label_files.items():
        if sha256(path) != expected[name]:
            raise RuntimeError(f"input hash mismatch: {path}")

    phasor_ledger = Path(os.environ["PHASOR_SHA256_LEDGER"])
    if sha256(phasor_ledger) != expected["corrected_phasor_sha256_ledger"]:
        raise RuntimeError("corrected phasor ledger hash mismatch")

    counts = {
        "raw_train": len(list((ROOT / "data_npy" / "train").glob("result*.npy"))),
        "raw_validation": len(list((ROOT / "data_npy" / "val").glob("result*.npy"))),
        "phasor_train": len(list(
            (ROOT / "data_npy" / "phasor" / "train").glob("result*.npy")
        )),
        "phasor_validation": len(list(
            (ROOT / "data_npy" / "phasor" / "val").glob("result*.npy")
        )),
    }
    if counts != {
        "raw_train": 4282,
        "raw_validation": 225,
        "phasor_train": 4282,
        "phasor_validation": 225,
    }:
        raise RuntimeError(f"input file-count mismatch: {counts}")


def write_provenance(run_dir: Path, config: dict, matrix_path: Path) -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=True)
    provenance = {
        "commit": commit,
        "matrix_sha256": sha256(matrix_path),
        "command": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm": {
            key: value for key, value in os.environ.items()
            if key.startswith("SLURM_")
        },
        "source_sha256": {
            name: sha256(ROOT / name) for name in SOURCE_FILES
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    (run_dir / "pip-freeze.txt").write_text(freeze, encoding="utf-8")


def last_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def train(run_dir: Path, config: dict, resume: bool) -> None:
    command = [
        sys.executable, "-m", "rl_protection.train_cql",
        "--mode", config["mode"],
        "--W", str(config["window_W"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["learning_rate"]),
        "--alpha", str(config["alpha"]),
        "--gamma", str(config["gamma"]),
        "--tau", str(config["tau"]),
        "--batch", str(config["batch_size"]),
        "--workers", str(config["workers"]),
        "--fp-penalty", str(config["false_positive_penalty"]),
        "--correct-reward", str(config["correct_reward"]),
        "--seed", str(config["training_seed"]),
        "--save-dir", str(run_dir),
    ]
    if config["deterministic"]:
        command.append("--deterministic")
    checkpoint = last_checkpoint(run_dir)
    if resume:
        if checkpoint is None:
            raise RuntimeError("--resume requested but no checkpoint exists")
        command += ["--resume", str(checkpoint)]
    subprocess.run(command, cwd=ROOT, check=True)


def export_predictions(run_root: Path, config: dict) -> None:
    tag = config["tag"]
    run_dir = run_root / tag
    checkpoint = torch.load(
        run_dir / f"epoch_{config['epochs']:02d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = build_model(config["mode"], config["window_W"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    val_labels = torch.load(
        run_ablation.ROOT / "labels" / "validation" / "val_labels.pt",
        weights_only=False,
    )
    indices = np.asarray(val_labels["indices"], dtype=np.int32)
    window = config["window_W"]
    t_start = CYCLE_LEN - 1 + window
    n_eval = N_STEPS - t_start
    actions = np.empty((len(indices), n_eval), dtype=np.int16)
    q_values = np.empty((len(indices), n_eval, 16), dtype=np.float32)

    with torch.no_grad():
        for episode, sim_idx in enumerate(indices):
            array = run_ablation._load_obs(int(sim_idx), config["mode"])
            if array is None:
                raise FileNotFoundError(f"missing validation episode {sim_idx}")
            offset = t_start - window + 1
            for start in range(0, n_eval, 1024):
                stop = min(start + 1024, n_eval)
                batch = np.stack([
                    array[offset + step:offset + step + window]
                    for step in range(start, stop)
                ])
                q = model(torch.from_numpy(batch).to(device)).cpu().numpy()
                if not np.isfinite(q).all():
                    raise RuntimeError(f"non-finite Q values for episode {sim_idx}")
                q_values[episode, start:stop] = q
                actions[episode, start:stop] = q.argmax(axis=1)

    np.savez_compressed(
        run_dir / "raw_predictions.npz",
        sim_idx=indices,
        t_start=np.int32(t_start),
        actions=actions,
        q_values=q_values,
    )
    pd.DataFrame({
        "sim_idx": indices,
        "n_predictions": n_eval,
        "first_action": actions[:, 0],
    }).to_csv(run_dir / "prediction_inventory.csv", index=False)


def validate_and_finalize(run_root: Path, config: dict) -> None:
    tag = config["tag"]
    run_dir = run_root / tag
    run_ablation.CKPT_ROOT = run_root
    result = run_ablation.validate_run(
        tag=tag,
        mode=config["mode"],
        W=config["window_W"],
    )
    if result is None:
        raise RuntimeError(f"{tag} validation produced no result")
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    export_predictions(run_root, config)

    checkpoints = sorted(run_dir.glob("epoch_*.pt"))
    assert len(checkpoints) == config["epochs"]
    final = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    assert final["epoch"] == config["epochs"]
    assert final["seed"] == config["training_seed"]
    assert final["deterministic"] is config["deterministic"]
    for values in final["history"].values():
        assert len(values) == config["epochs"]
        assert np.isfinite(values).all()

    predictions = np.load(run_dir / "raw_predictions.npz")
    expected_steps = N_STEPS - (CYCLE_LEN - 1 + config["window_W"])
    assert predictions["actions"].shape == (225, expected_steps)
    assert predictions["q_values"].shape == (225, expected_steps, 16)
    assert np.isfinite(predictions["q_values"]).all()
    assert len(pd.read_csv(run_dir / "validation_trajectories.csv")) == 225

    excluded = {"output_sha256.txt", "completed_at.txt"}
    ledger = [
        f"{sha256(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name not in excluded
    ]
    (run_dir / "output_sha256.txt").write_text(
        "\n".join(ledger) + "\n", encoding="utf-8"
    )
    (run_dir / "completed_at.txt").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--matrix", type=Path, default=ROOT / "gate4" / "matrix.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.matrix, args.tag)
    verify_inputs(config)
    run_dir = args.output_root / args.tag
    if run_dir.exists() and not (args.resume or args.postprocess_only):
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (args.resume or args.postprocess_only):
        write_provenance(run_dir, config, args.matrix)
    if not args.postprocess_only:
        train(run_dir, config, args.resume)
    validate_and_finalize(args.output_root, config)


if __name__ == "__main__":
    main()

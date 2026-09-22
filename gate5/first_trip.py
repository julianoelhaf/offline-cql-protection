"""Approved post-hoc first-trip evaluation for Gate 4 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def classify_episode(
    actions: np.ndarray,
    *,
    t_start: int,
    is_fault: bool,
    expert_action: int,
    wait_action: int,
    fault_onset: int,
    sample_rate: float,
    n_steps: int,
) -> dict:
    trips = np.flatnonzero(actions != wait_action)
    record = {
        "first_trip_index": None,
        "first_trip_time_ms": None,
        "relative_delay_ms": None,
        "first_action": None,
        "censored": not len(trips),
        "censor_index": n_steps - 1 if not len(trips) else None,
        "censor_time_ms": (n_steps - 1) / sample_rate * 1000 if not len(trips) else None,
    }
    if not len(trips):
        record["outcome"] = "no_trip"
        return record

    offset = int(trips[0])
    first_index = t_start + offset
    first_action = int(actions[offset])
    record.update({
        "first_trip_index": first_index,
        "first_trip_time_ms": first_index / sample_rate * 1000,
        "first_action": first_action,
        "censored": False,
    })
    if not is_fault:
        record["outcome"] = "false_first_trip"
    else:
        record["relative_delay_ms"] = (first_index - fault_onset) / sample_rate * 1000
        if first_index < fault_onset:
            record["outcome"] = "premature_trip"
        elif first_action == expert_action:
            record["outcome"] = "correct_first_trip"
        else:
            record["outcome"] = "wrong_relay_first_trip"
    return record


def wilson(count: int, total: int) -> dict:
    if total == 0:
        return {"count": count, "total": total, "rate": None, "ci95": [None, None]}
    z = 1.959963984540054
    p = count / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"count": count, "total": total, "rate": p, "ci95": [centre - half, centre + half]}


def summarize(episodes: pd.DataFrame, bootstrap_seed: int, bootstrap_samples: int) -> dict:
    fault = episodes[episodes["is_fault"]]
    nonfault = episodes[~episodes["is_fault"]]

    def rate(frame: pd.DataFrame, outcome: str) -> dict:
        return wilson(int((frame["outcome"] == outcome).sum()), len(frame))

    delays = fault.loc[
        fault["outcome"] == "correct_first_trip", "relative_delay_ms"
    ].to_numpy(dtype=float)
    latency = {
        "n": int(len(delays)),
        "median_ms": None,
        "q25_ms": None,
        "q75_ms": None,
        "p95_ms": None,
        "median_bootstrap_ci95_ms": [None, None],
    }
    if len(delays):
        rng = np.random.default_rng(bootstrap_seed)
        medians = np.median(
            rng.choice(delays, size=(bootstrap_samples, len(delays)), replace=True),
            axis=1,
        )
        latency.update({
            "median_ms": float(np.median(delays)),
            "q25_ms": float(np.percentile(delays, 25)),
            "q75_ms": float(np.percentile(delays, 75)),
            "p95_ms": float(np.percentile(delays, 95)),
            "median_bootstrap_ci95_ms": [
                float(np.percentile(medians, 2.5)),
                float(np.percentile(medians, 97.5)),
            ],
        })

    return {
        "episodes": int(len(episodes)),
        "fault_episodes": int(len(fault)),
        "nonfault_episodes": int(len(nonfault)),
        "fault_outcomes": {
            name: rate(fault, name)
            for name in (
                "correct_first_trip",
                "premature_trip",
                "wrong_relay_first_trip",
                "no_trip",
            )
        },
        "nonfault_outcomes": {
            "false_first_trip": rate(nonfault, "false_first_trip"),
            "no_trip": rate(nonfault, "no_trip"),
        },
        "correct_first_trip_latency": latency,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--training-commit", required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")

    from rl_protection.constants import FAULT_ONSET, N_STEPS, SAMPLE_RATE, WAIT
    from rl_protection.expert import get_expert_action

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    tags = [run["tag"] for run in matrix["runs"]]
    settings = pd.read_csv(args.settings, sep=";").set_index("general/sim_idx")
    evaluation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.run(["git", "diff", "--quiet"]).returncode:
        raise RuntimeError("tracked working tree is dirty")

    args.output.mkdir(parents=True)
    combined_rows = []
    source_hashes = {}
    for tag in tags:
        prediction_path = args.runs_root / tag / "raw_predictions.npz"
        source_hashes[tag] = sha256(prediction_path)
        with np.load(prediction_path) as predictions:
            sim_indices = predictions["sim_idx"].astype(int)
            t_start = int(predictions["t_start"])
            actions = predictions["actions"]
            if actions.shape[0] != len(sim_indices) or len(sim_indices) != 225:
                raise ValueError(f"unexpected prediction shape for {tag}: {actions.shape}")

            records = []
            for sim_idx, episode_actions in zip(sim_indices, actions):
                row = settings.loc[sim_idx]
                event_type = str(row["events/event_type"])
                event_target = str(row["events/event_target"])
                is_fault = event_type.startswith("flt_")
                expert_action = int(get_expert_action(event_type, event_target))
                records.append({
                    "run_tag": tag,
                    "sim_idx": int(sim_idx),
                    "event_type": event_type,
                    "event_target": event_target,
                    "is_fault": is_fault,
                    "t_start": t_start,
                    "expert_action": expert_action,
                    **classify_episode(
                        episode_actions,
                        t_start=t_start,
                        is_fault=is_fault,
                        expert_action=expert_action,
                        wait_action=WAIT,
                        fault_onset=FAULT_ONSET,
                        sample_rate=SAMPLE_RATE,
                        n_steps=N_STEPS,
                    ),
                })

        episodes = pd.DataFrame(records)
        summary = summarize(episodes, bootstrap_seed=0, bootstrap_samples=10_000)
        summary["tag"] = tag
        run_output = args.output / tag
        run_output.mkdir()
        episodes.to_csv(run_output / "first_trip_episodes.csv", index=False)
        (run_output / "first_trip_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

        fault = summary["fault_outcomes"]
        nonfault = summary["nonfault_outcomes"]
        latency = summary["correct_first_trip_latency"]
        combined_row = {
            "tag": tag,
            "episodes": summary["episodes"],
            "fault_episodes": summary["fault_episodes"],
            "nonfault_episodes": summary["nonfault_episodes"],
            **{f"latency_{key}": value for key, value in latency.items()
               if key != "median_bootstrap_ci95_ms"},
            "latency_median_bootstrap_ci95_low_ms":
                latency["median_bootstrap_ci95_ms"][0],
            "latency_median_bootstrap_ci95_high_ms":
                latency["median_bootstrap_ci95_ms"][1],
        }
        for prefix, outcomes in (("fault", fault), ("nonfault", nonfault)):
            for name, values in outcomes.items():
                stem = f"{prefix}_{name}"
                combined_row.update({
                    f"{stem}_count": values["count"],
                    f"{stem}_total": values["total"],
                    f"{stem}_rate": values["rate"],
                    f"{stem}_ci95_low": values["ci95"][0],
                    f"{stem}_ci95_high": values["ci95"][1],
                })
        combined_rows.append(combined_row)

    pd.DataFrame(combined_rows).to_csv(
        args.output / "first_trip_summary.csv", index=False
    )
    (args.output / "pip-freeze.txt").write_text(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ),
        encoding="utf-8",
    )
    definition = {
        "command": sys.argv,
        "evaluation_commit": evaluation_commit,
        "evaluator_sha256": sha256(Path(__file__)),
        "training_commit": args.training_commit,
        "matrix_sha256": sha256(args.matrix),
        "settings_sha256": sha256(args.settings),
        "raw_prediction_sha256": source_hashes,
        "wait_action": WAIT,
        "trip_actions": list(range(WAIT)),
        "fault_onset_index": FAULT_ONSET,
        "sample_rate_hz": SAMPLE_RATE,
        "sample_period_ms": 1000 / SAMPLE_RATE,
        "episode_final_index": N_STEPS - 1,
        "latency_formula": "(first_trip_index - fault_onset_index) / sample_rate_hz * 1000",
        "outcomes": {
            "fault": [
                "premature_trip",
                "correct_first_trip",
                "wrong_relay_first_trip",
                "no_trip",
            ],
            "nonfault": ["false_first_trip", "no_trip"],
        },
        "post_first_trip_actions_ignored": True,
        "no_trip_right_censored": True,
        "bootstrap": {"samples": 10_000, "seed": 0},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "slurm": {
                key: value for key, value in os.environ.items()
                if key.startswith("SLURM_")
            },
        },
    }
    (args.output / "definition.json").write_text(
        json.dumps(definition, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    ledger = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "output_sha256.txt":
            ledger.append(f"{sha256(path)}  {path.relative_to(args.output).as_posix()}")
    (args.output / "output_sha256.txt").write_text(
        "\n".join(ledger) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

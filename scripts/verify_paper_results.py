"""Verify the manuscript's headline results against the committed evidence.

Reads the frozen evaluation artifacts in pess_2026_rl_luce/evidence/ and
asserts every headline value printed in the paper. The expected values below
exist ONLY in this script, as assertions against independently generated
artifacts - they are never fed into the evaluation pipeline itself.

  python scripts/verify_paper_results.py

Exit code 0 means every check passed; any mismatch is reported and the script
exits non-zero. If an artifact disagrees with the paper, that is a finding to
report - do not edit the artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "pess_2026_rl_luce" / "evidence"

FAILURES: list[str] = []


def check(label: str, actual, expected) -> None:
    ok = actual == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {actual!r}"
          + ("" if ok else f" (expected {expected!r})"))
    if not ok:
        FAILURES.append(label)


def check_close(label: str, actual: float, expected: float,
                decimals: int) -> None:
    check(label, round(float(actual), decimals), expected)


def dense_row(tag: str) -> pd.Series:
    frame = pd.read_csv(EVIDENCE / "corrected_validation_summary.csv")
    return frame[frame["tag"] == tag].iloc[0]


def first_trip_row(tag: str) -> pd.Series:
    frame = pd.read_csv(EVIDENCE / "first_trip_summary.csv")
    return frame[frame["tag"] == tag].iloc[0]


def check_dense(name: str, row, precision: float, recall: float,
                f1: float, fpr_percent: float) -> None:
    print(f"\n{name} - dense per-timestep metrics")
    check_close("precision", row["precision"], precision, 4)
    check_close("recall", row["recall"], recall, 4)
    check_close("F1", row["f1"], f1, 4)
    check_close("FPR [%]", 100 * float(row["fpr"]), fpr_percent, 2)

    # Internal consistency: the stored ratios must follow from the stored
    # raw confusion counts.
    if all(key in row for key in ("TP", "FN", "FP", "TN")):
        tp, fn = int(row["TP"]), int(row["FN"])
        fp, tn = int(row["FP"]), int(row["TN"])
        check_close("precision from counts", tp / (tp + fp), precision, 4)
        check_close("recall from counts", tp / (tp + fn), recall, 4)
        p, r = tp / (tp + fp), tp / (tp + fn)
        check_close("F1 from counts", 2 * p * r / (p + r), f1, 4)
        check_close("FPR from counts [%]", 100 * fp / (fp + tn),
                    fpr_percent, 2)


def main() -> int:
    # ── Prespecified default: combined, W=48, alpha=0.5 ─────────────────────
    row = dense_row("combined_W48")
    check_dense("Prespecified default combined_W48 (alpha=0.5)", row,
                precision=0.9946, recall=0.9347, f1=0.9637, fpr_percent=2.05)

    ft = first_trip_row("combined_W48")
    print("\nPrespecified default combined_W48 - terminal first-trip outcomes")
    check("fault episodes", int(ft["fault_episodes"]), 214)
    check("correct first trip", int(ft["fault_correct_first_trip_count"]), 210)
    check("wrong line", int(ft["fault_wrong_relay_first_trip_count"]), 1)
    check("no trip (fault)", int(ft["fault_no_trip_count"]), 3)
    check("non-fault episodes", int(ft["nonfault_episodes"]), 11)
    check("false trips", int(ft["nonfault_false_first_trip_count"]), 8)
    check("no trip (non-fault)", int(ft["nonfault_no_trip_count"]), 3)

    # ── Best prespecified dense run: combined, W=48, alpha=0.9 ──────────────
    row = dense_row("combined_W48_alpha0.9")
    check_dense("Best prespecified dense run combined_W48_alpha0.9", row,
                precision=0.9993, recall=0.9496, f1=0.9738, fpr_percent=0.28)

    # ── Exploratory post-hoc: combined, W=48, gamma=0.99 ────────────────────
    comparison = json.loads(
        (EVIDENCE / "posthoc_gamma099_comparison.json").read_text()
    )
    dense = comparison["dense"]
    posthoc = pd.Series({
        "precision": dense["precision"]["candidate_gamma099"],
        "recall": dense["recall"]["candidate_gamma099"],
        "f1": dense["f1"]["candidate_gamma099"],
        "fpr": dense["fpr"]["candidate_gamma099"],
        "TP": dense["TP"]["candidate_gamma099"],
        "FN": dense["FN"]["candidate_gamma099"],
        "FP": dense["FP"]["candidate_gamma099"],
        "TN": dense["TN"]["candidate_gamma099"],
    })
    check_dense("Exploratory post-hoc combined_W48_gamma099", posthoc,
                precision=0.9997, recall=0.9422, f1=0.9701, fpr_percent=0.10)

    # The comparison baseline must equal the prespecified default.
    print("\nPost-hoc comparison baseline consistency")
    base = dense_row("combined_W48")
    for key in ("precision", "recall", "f1", "fpr"):
        # 12 decimals: CSV and JSON parsers may disagree in the last ulp.
        check_close(f"baseline {key} equals prespecified default",
                    float(dense[key]["baseline_gamma095"]),
                    round(float(base[key]), 12), 12)

    summary = json.loads(
        (EVIDENCE / "posthoc_gamma099_first_trip_summary.json").read_text()
    )
    print("\nExploratory post-hoc combined_W48_gamma099 - terminal outcomes")
    check("fault episodes", int(summary["fault_episodes"]), 214)
    check("correct first trip",
          int(summary["fault_outcomes"]["correct_first_trip"]["count"]), 211)
    check("wrong line",
          int(summary["fault_outcomes"]["wrong_relay_first_trip"]["count"]), 1)
    check("no trip (fault)",
          int(summary["fault_outcomes"]["no_trip"]["count"]), 2)
    check("non-fault episodes", int(summary["nonfault_episodes"]), 11)
    check("false trips",
          int(summary["nonfault_outcomes"]["false_first_trip"]["count"]), 6)
    check("no trip (non-fault)",
          int(summary["nonfault_outcomes"]["no_trip"]["count"]), 5)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} headline check(s) FAILED:")
        for label in FAILURES:
            print(f"  - {label}")
        return 1
    print("All headline results match the committed evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

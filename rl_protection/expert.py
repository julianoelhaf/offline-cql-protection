"""
expert.py — Expert action labeling and transition table construction.

get_expert_action(event_type, event_target) -> int
    Returns the optimal relay action for a given simulation episode.

build_transitions(settings_df, train_indices) -> np.ndarray
    Returns the full transition index table as a (N, 5) int32 array with columns:
    [sim_idx, t_idx, action, is_fault, expert_action]
"""

import numpy as np
import pandas as pd

from .constants import (
    LINE_TO_ACTION, BUS_TO_UPSTREAM_LINE, WAIT,
    FAULT_ONSET, CYCLE_LEN, N_STEPS, N_ACTIONS,
)


def get_expert_action(event_type: str, event_target: str) -> int:
    """
    Return the expert relay action (integer 0–15) for one simulation episode.

    - Fault on a line  → trip that line.
    - Fault on a bus   → trip the upstream (infeed) line for that bus.
    - Switch event     → WAIT (15); no trip should happen.
    """
    if not event_type.startswith('flt_'):
        return WAIT

    if event_target in LINE_TO_ACTION:
        return LINE_TO_ACTION[event_target]

    if event_target in BUS_TO_UPSTREAM_LINE:
        upstream = BUS_TO_UPSTREAM_LINE[event_target]
        return LINE_TO_ACTION[upstream]

    # Unknown target — default to WAIT and print a warning
    print(f'[expert] WARNING: unknown event_target "{event_target}", defaulting to WAIT')
    return WAIT


def build_transitions(settings_df: pd.DataFrame, train_indices: list,
                      n_wrong_per_step: int = 1, seed: int = 0) -> np.ndarray:
    """
    Build the offline transition index table for all training episodes.

    Each row: [sim_idx, t_idx, action, is_fault, expert_action]  (all int32)
    t_idx is the END of the observation window (W steps ending at t_idx).

    For fault episodes, post-fault timesteps include:
      - 1 correct-relay row   (action == expert_action)
      - n_wrong_per_step wrong-relay rows (random subset of the other 14 relays)
    With n_wrong_per_step=1, the wrong-relay share of the dataset is ~36%.
    With n_wrong_per_step=14, every wrong relay is included for every step
    (~89% wrong-relay rows — a strong "don't trip" bias).

    Counterfactual sampling is deterministic given `seed`.

    Window ranges (expressed as t_end = window end step):
      Pre-fault  : [CYCLE_LEN-1,                  FAULT_ONSET-1]
                   Window is entirely before the fault — expert always WAIT.
      Dense onset: [FAULT_ONSET,                   FAULT_ONSET + CYCLE_LEN//2]
                   Fault onset moves from last step to mid-window — every step.
      Sparse tail: [FAULT_ONSET + CYCLE_LEN//2+1,  N_STEPS-1]
                   Fault fully established in window — sampled at sparse_step.

    Per fault episode (with n_wrong_per_step=1, default):
      pre-fault ≈ 110 rows | correct ≈ 146 rows | wrong ≈ 146 rows  (≈ 36% wrong)

    Switch episodes contribute additional WAIT rows.

    Returns np.ndarray of shape (N, 5), dtype int32.
    """
    rows = []
    rng  = np.random.default_rng(seed)

    # ── Range boundaries (t_end) ──────────────────────────────────────────
    pre_start    = CYCLE_LEN - 1                      # first complete window
    pre_end      = FAULT_ONSET - 1                    # [191, 959]

    dense_start  = FAULT_ONSET                        # [960, 1056]
    dense_end    = FAULT_ONSET + CYCLE_LEN // 2

    sparse_start = dense_end + 1                      # [1057, N_STEPS-1]
    sparse_end   = N_STEPS - 1

    # ── Step sizes (in unique timesteps; row counts depend on n_wrong_per_step) ──
    N_dense      = dense_end - dense_start + 1                               # 97
    pre_step     = max(1, (pre_end - pre_start + 1) // N_dense)              # ~7
    sparse_step  = max(1, (sparse_end - sparse_start + 1) // (N_dense // 2)) # ~77

    # All trip actions (0–14), used as the pool for counterfactual wrong-relay rows
    all_trips = np.arange(N_ACTIONS - 1, dtype=np.int64)  # [0, 1, ..., 14]

    def add_fault_step(sim_idx: int, t: int, expert_action: int):
        """Emit 1 correct row + n_wrong_per_step random-wrong rows."""
        rows.append([sim_idx, t, expert_action, 1, expert_action])
        wrong_pool = all_trips[all_trips != expert_action]
        k = min(n_wrong_per_step, len(wrong_pool))
        if k > 0:
            sampled = rng.choice(wrong_pool, size=k, replace=False)
            for a in sampled:
                rows.append([sim_idx, t, int(a), 1, expert_action])

    for sim_idx in train_indices:
        row           = settings_df.loc[settings_df['general/sim_idx'] == sim_idx].iloc[0]
        event_type    = row['events/event_type']
        event_target  = row['events/event_target']
        expert_action = get_expert_action(event_type, event_target)
        is_fault      = int(event_type.startswith('flt_'))

        # Pre-fault: uniformly sampled, expert always WAIT
        for t in range(pre_start, pre_end + 1, pre_step):
            rows.append([sim_idx, t, WAIT, 0, WAIT])

        if is_fault:
            # Dense onset zone: every step, fault active
            for t in range(dense_start, dense_end + 1):
                add_fault_step(sim_idx, t, expert_action)
            # Sparse tail: fault established, sparsely sampled
            for t in range(sparse_start, sparse_end + 1, sparse_step):
                add_fault_step(sim_idx, t, expert_action)
        else:
            # Switch episode: sparse over full post-fault range, always WAIT
            for t in range(dense_start, sparse_end + 1, sparse_step):
                rows.append([sim_idx, t, WAIT, 0, WAIT])

    return np.array(rows, dtype=np.int32)

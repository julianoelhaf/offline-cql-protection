"""
evaluate.py — Full-episode evaluation and metrics for the relay selection agent.

run_episode(model, sim_idx, settings_df, mode, W) -> dict
    Simulates one episode: steps through post-fault period until trip or timeout.
    Returns a result dict with trip action, delay, correctness, etc.

evaluate_all(model, val_indices, settings_df, mode, W) -> pd.DataFrame
    Runs run_episode for all validation episodes; returns a DataFrame of results.

print_metrics(results_df)
    Prints the main metric table.

Usage:
  python -m rl_protection.evaluate --ckpt checkpoints/bc_phasor_W192.pt --mode phasor --W 192
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from .constants import (
    NPY_VAL_DIR, PHASOR_VAL_DIR,
    SETTINGS_CSV, LABELS_DIR,
    FAULT_ONSET, CYCLE_LEN, N_STEPS, DT, WAIT, N_ACTIONS,
    LINE_COL_INDICES, N_SIGNAL_COLS, N_PHASOR_COLS,
)
from .expert  import get_expert_action
from .dataset import load_window, extract_features
from .models  import build_model


MAX_POST_FAULT_STEPS = 5 * CYCLE_LEN   # 960 steps = 100 ms


def load_val_mmaps(val_indices: list):
    """
    Open all validation .npy files as memory maps.
    Returns (raw_mmaps, phasor_mmaps) dicts keyed by sim_idx.
    """
    raw_mmaps    = {}
    phasor_mmaps = {}
    for sim_idx in val_indices:
        raw_path    = os.path.join(NPY_VAL_DIR,    f'result{sim_idx}.npy')
        phasor_path = os.path.join(PHASOR_VAL_DIR, f'result{sim_idx}.npy')
        if os.path.exists(raw_path):
            raw_mmaps[sim_idx]    = np.load(raw_path,    mmap_mode='r')
        if os.path.exists(phasor_path):
            phasor_mmaps[sim_idx] = np.load(phasor_path, mmap_mode='r')
    return raw_mmaps, phasor_mmaps


def get_obs(sim_idx: int, t: int, W: int, mode: str,
            raw_mmaps: dict, phasor_mmaps: dict) -> torch.Tensor:
    """Build a single observation tensor of shape (1, W, C)."""
    raw_mmap    = raw_mmaps.get(sim_idx)
    phasor_mmap = phasor_mmaps.get(sim_idx)

    raw_w    = load_window(raw_mmap,    t, W) if raw_mmap    is not None else np.zeros((W, N_SIGNAL_COLS),  np.float32)
    phasor_w = load_window(phasor_mmap, t, W) if phasor_mmap is not None else np.zeros((W, N_PHASOR_COLS),  np.float32)

    feat = extract_features(raw_w, phasor_w, mode)   # (W, C)
    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)   # (1, W, C)


def run_episode(model: torch.nn.Module, sim_idx: int, meta: dict,
                mode: str, W: int,
                raw_mmaps: dict, phasor_mmaps: dict) -> dict:
    """
    Step through a single episode starting at FAULT_ONSET.
    Stop when the agent takes a non-WAIT action or MAX_POST_FAULT_STEPS elapse.

    meta keys: event_type, event_target
    """
    event_type   = meta['events/event_type']
    event_target = meta['events/event_target']
    expert_action = get_expert_action(event_type, event_target)
    is_fault      = event_type.startswith('flt_')

    model.eval()
    tripped     = False
    agent_action = None
    delay_steps  = None

    with torch.no_grad():
        for delta in range(MAX_POST_FAULT_STEPS):
            t   = FAULT_ONSET + delta
            obs = get_obs(sim_idx, t, W, mode, raw_mmaps, phasor_mmaps)
            q   = model(obs)                        # (1, N_ACTIONS)
            act = int(q.argmax(dim=1).item())

            if act != WAIT:
                tripped      = True
                agent_action = act
                delay_steps  = delta
                break

    delay_ms = delay_steps * DT * 1000.0 if delay_steps is not None else None

    return {
        'sim_idx':       sim_idx,
        'event_type':    event_type,
        'event_target':  event_target,
        'is_fault':      is_fault,
        'expert_action': expert_action,
        'agent_action':  agent_action,
        'tripped':       tripped,
        'correct':       tripped and (agent_action == expert_action),
        'wrong_relay':   tripped and (agent_action != expert_action) and is_fault,
        'false_pos':     tripped and (not is_fault),
        'missed':        (not tripped) and is_fault,
        'delay_ms':      delay_ms,
    }


def evaluate_all(model, val_indices: list, settings_df: pd.DataFrame,
                 mode: str, W: int) -> pd.DataFrame:
    """Run full-episode evaluation for all validation episodes."""
    raw_mmaps, phasor_mmaps = load_val_mmaps(val_indices)
    results = []
    for sim_idx in val_indices:
        row = settings_df.loc[settings_df['general/sim_idx'] == sim_idx].iloc[0]
        meta = row.to_dict()
        result = run_episode(model, sim_idx, meta, mode, W, raw_mmaps, phasor_mmaps)
        results.append(result)
    return pd.DataFrame(results)


def print_metrics(df: pd.DataFrame):
    """Print the main evaluation metrics table."""
    fault_df  = df[df['is_fault']]
    switch_df = df[~df['is_fault']]

    all_trips   = df['tripped'].sum()
    correct     = df['correct'].sum()
    wrong       = df['wrong_relay'].sum()
    false_pos   = df['false_pos'].sum()
    missed      = df['missed'].sum()

    trip_acc    = correct / all_trips if all_trips else float('nan')
    fpr         = false_pos / len(switch_df) if len(switch_df) else float('nan')
    selectivity = correct / (correct + wrong + false_pos) if (correct + wrong + false_pos) else float('nan')
    mfr         = missed / len(fault_df) if len(fault_df) else float('nan')

    delays = df.loc[df['correct'], 'delay_ms'].dropna()
    p50    = delays.quantile(0.50) if len(delays) else float('nan')
    p95    = delays.quantile(0.95) if len(delays) else float('nan')

    print('─' * 50)
    print(f'Episodes:       {len(df)} total  ({len(fault_df)} fault, {len(switch_df)} switch)')
    print(f'Trip accuracy:  {trip_acc:.3f}  ({correct}/{all_trips} trips correct)')
    print(f'Trip delay:     p50={p50:.1f} ms   p95={p95:.1f} ms')
    print(f'False pos rate: {fpr:.3f}  ({false_pos}/{len(switch_df)} switch eps)')
    print(f'Selectivity:    {selectivity:.3f}')
    print(f'Missed faults:  {mfr:.3f}  ({missed}/{len(fault_df)} fault eps)')
    print('─' * 50)

    # Per fault-type breakdown
    print('\nPer fault type:')
    for ft, sub in fault_df.groupby('event_type'):
        c = sub['correct'].sum()
        n = len(sub)
        d = sub.loc[sub['correct'], 'delay_ms'].dropna()
        d50 = d.quantile(0.50) if len(d) else float('nan')
        print(f'  {ft:<30s}  acc={c/n:.3f}  delay_p50={d50:.1f} ms  n={n}')

    # Per fault location
    if 'event_flt_target_line_location' in fault_df.columns:
        print('\nPer fault location:')
        for loc, sub in fault_df.groupby('events/event_flt_target_line_location'):
            c = sub['correct'].sum()
            n = len(sub)
            print(f'  {loc:5}%  acc={c/n:.3f}  n={n}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',    required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--mode',    default='phasor', choices=['raw', 'raw_v2', 'phasor', 'combined'])
    parser.add_argument('--W',       type=int, default=192)
    args = parser.parse_args()

    import torch

    # Load settings and validation indices
    settings_df = pd.read_csv(SETTINGS_CSV, sep=';')
    import torch as _torch
    val_data    = _torch.load(
        os.path.join(LABELS_DIR, 'validation', 'val_labels.pt'),
        weights_only=False,
    )
    val_indices = val_data['indices']

    # Load model
    model = build_model(args.mode, args.W)
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))

    # Evaluate
    results_df = evaluate_all(model, val_indices, settings_df, args.mode, args.W)
    print_metrics(results_df)

    # Save results CSV
    out_path = args.ckpt.replace('.pt', '_eval.csv')
    results_df.to_csv(out_path, index=False)
    print(f'\nFull results saved to {out_path}')

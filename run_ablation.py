"""
run_ablation.py — Ablation study runner for the relay-selection CQL agent.

Runs all configurations defined in BLOCKS below, saves per-epoch checkpoints to
  ablation_study/<tag>/epoch_XX.pt
writes a summary CSV to
  ablation_study/results.csv
and after each run validates on the full validation set, saving
  ablation_study/<tag>/validation_trajectories.csv
  ablation_study/<tag>/reward_violin.png

After all blocks complete, a combined ROC diagram is saved to
  ablation_study/roc_all_runs.png

Usage:
  python run_ablation.py                   # run all blocks
  python run_ablation.py --block A         # only Block A (mode × window)
  python run_ablation.py --block B         # only Block B (reward shaping)
  python run_ablation.py --block C         # only Block C (CQL alpha)
  python run_ablation.py --best phasor_192 # set best config tag for B/C

Blocks:
  A  — input representation × window size  (9 CQL + 3 BC = 12 runs)
  B  — reward shaping at best Block-A config   (3 CQL runs)
  C  — CQL alpha at best Block-A config        (2 CQL runs)

After Block A completes, inspect results.csv to pick the best (mode, W) config,
then re-run with --block B --best <tag>  and  --block C --best <tag>.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import tqdm

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl_protection.constants import (
    N_STEPS, FAULT_ONSET, WAIT, CYCLE_LEN,
    LINE_COL_INDICES, NPY_VAL_DIR, PHASOR_VAL_DIR,
)
from rl_protection.models import build_model
from rl_protection.expert import get_expert_action
from rl_protection.reward import compute_reward

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent
CKPT_ROOT = ROOT / 'ablation_study'
RESULTS   = CKPT_ROOT / 'results.csv'

TRAIN_CQL = [sys.executable, '-m', 'rl_protection.train_cql']
TRAIN_BC  = [sys.executable, '-m', 'rl_protection.train_bc']

EPOCHS_CQL = 30
EPOCHS_BC  = 20

COMMON_CQL = ['--epochs', str(EPOCHS_CQL), '--workers', '0']
COMMON_BC  = ['--epochs', str(EPOCHS_BC),  '--workers', '0']

PLOT_DTYPE = ".pdf"

# ── Block definitions ──────────────────────────────────────────────────────────

BLOCK_A_CQL = [
    {'tag': f'{mode}_W{W}', 'mode': mode, 'W': W}
    for W    in [48, 96, 192]
    for mode in ['raw', 'raw_v2']
] + [
    {'tag': f'{mode}_W256', 'mode': mode, 'W': 256} for mode in ['raw', 'raw_v2']
] + [
    {'tag': f'{mode}_W{W}', 'mode': mode, 'W': W}
    for W    in [48, 96]
    for mode in ['phasor', 'combined']
]


BLOCK_A_BC = [
    {'tag': f'bc_{mode}_W48', 'mode': mode, 'W': 48}
    for mode in ['raw', 'phasor', 'combined']
]

# Block B and C are filled in at runtime based on --best
BLOCK_B_TEMPLATES = [
    {'tag_suffix': 'fp10',   'fp_penalty': -10.0,  'correct_reward': 5.0},
    {'tag_suffix': 'fp200',  'fp_penalty': -200.0, 'correct_reward': 5.0},
    {'tag_suffix': 'fp100',  'fp_penalty': -100.0, 'correct_reward': 50.0}, 
]

BLOCK_C_TEMPLATES = [
    {'tag_suffix': 'alpha0.1', 'alpha': 0.1},
    {'tag_suffix': 'alpha0.9', 'alpha': 0.9},
    # alpha=0.5 is already covered by the Block A default
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run(cmd: list[str], tag: str) -> int:
    print(f'\n{"─"*60}')
    print(f'[{datetime.now():%H:%M:%S}]  {tag}')
    print(' '.join(cmd))
    print('─' * 60, flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def ckpt_dir(tag: str) -> str:
    return str(CKPT_ROOT / tag)


def last_checkpoint(tag: str) -> str | None:
    """Return path to the latest epoch checkpoint in a run's directory, or None."""
    d = CKPT_ROOT / tag
    pts = sorted(d.glob('epoch_*.pt')) if d.exists() else []
    return str(pts[-1]) if pts else None


def checkpoint_epoch(ckpt_path: str) -> int:
    """Extract the epoch number from a checkpoint filename like 'epoch_15.pt'."""
    return int(Path(ckpt_path).stem.split('_')[-1])


def is_run_complete(tag: str, target_epochs: int) -> bool:
    """True iff the final epoch checkpoint exists for this tag."""
    return (CKPT_ROOT / tag / f'epoch_{target_epochs:02d}.pt').exists()


def append_result(row: dict):
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS.exists()
    with open(RESULTS, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_cql(tag: str, mode: str, W: int, alpha: float = 0.5,
            fp_penalty: float = -100.0, correct_reward: float = 5.0,
            seed: int | None = None, deterministic: bool = False,
            extra: list[str] | None = None):
    save = ckpt_dir(tag)

    if is_run_complete(tag, EPOCHS_CQL):
        print(f'[skip]   CQL  {tag}: complete (epoch {EPOCHS_CQL}/{EPOCHS_CQL})')
        return

    cmd = TRAIN_CQL + COMMON_CQL + [
        '--mode',           mode,
        '--W',              str(W),
        '--alpha',          str(alpha),
        '--fp-penalty',     str(fp_penalty),
        '--correct-reward', str(correct_reward),
        '--save-dir',       save,
    ] + (extra or [])
    if seed is not None:
        cmd += ['--seed', str(seed)]
    if deterministic:
        cmd += ['--deterministic']

    last = last_checkpoint(tag)
    if last:
        done_epoch = checkpoint_epoch(last)
        cmd += ['--resume', last]
        print(f'[resume] CQL  {tag}: epoch {done_epoch}/{EPOCHS_CQL} done — continuing from epoch {done_epoch + 1}')
    else:
        print(f'[start]  CQL  {tag}: training from scratch (target {EPOCHS_CQL} epochs)')

    rc = run(cmd, f'CQL  {tag}')
    append_result({'block': 'CQL', 'tag': tag, 'mode': mode, 'W': W,
                   'alpha': alpha, 'fp_penalty': fp_penalty,
                   'correct_reward': correct_reward, 'seed': seed,
                   'deterministic': deterministic, 'exit_code': rc})


def run_bc(tag: str, mode: str, W: int):
    save = ckpt_dir(tag)

    if is_run_complete(tag, EPOCHS_BC):
        print(f'[skip]   BC   {tag}: complete (epoch {EPOCHS_BC}/{EPOCHS_BC})')
        return

    cmd = TRAIN_BC + COMMON_BC + [
        '--mode',     mode,
        '--W',        str(W),
        '--save-dir', save,
    ]

    last = last_checkpoint(tag)
    if last:
        done_epoch = checkpoint_epoch(last)
        cmd += ['--resume', last]
        print(f'[resume] BC   {tag}: epoch {done_epoch}/{EPOCHS_BC} done — continuing from epoch {done_epoch + 1}')
    else:
        print(f'[start]  BC   {tag}: training from scratch (target {EPOCHS_BC} epochs)')

    rc = run(cmd, f'BC   {tag}')
    append_result({'block': 'BC', 'tag': tag, 'mode': mode, 'W': W,
                   'alpha': '', 'fp_penalty': '', 'correct_reward': '', 'exit_code': rc})


# ── Validation ────────────────────────────────────────────────────────────────

def _load_obs(sim_idx: int, mode: str) -> np.ndarray | None:
    """Load and concatenate signal arrays for one episode depending on mode."""
    if mode in ('raw', 'raw_v2', 'combined'):
        raw_path = os.path.join(NPY_VAL_DIR, f'result{sim_idx}.npy')
        if not os.path.exists(raw_path):
            return None
        raw_arr = np.load(raw_path).astype(np.float32)[:, LINE_COL_INDICES]

    if mode in ('phasor', 'combined'):
        phasor_path = os.path.join(PHASOR_VAL_DIR, f'result{sim_idx}.npy')
        if not os.path.exists(phasor_path):
            return None
        phasor_arr = np.load(phasor_path).astype(np.float32)

    if mode == 'raw' or mode == "raw_v2":
        return raw_arr
    if mode == 'phasor':
        return phasor_arr
    return np.concatenate([raw_arr, phasor_arr], axis=1)


def _aggregate_metrics(tag: str, mode: str, W: int, trajectories: pd.DataFrame) -> dict:
    """Sample-level TP/FN/FP/TN aggregation → metric dict for ROC plotting."""
    TP = int(trajectories['TP'].sum())
    FN = int(trajectories['FN'].sum())
    FP = int(trajectories['FP'].sum())
    TN = int(trajectories['TN'].sum())

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tpr       = recall
    fpr       = FP / (FP + TN) if (FP + TN) else 0.0

    return {
        'tag': tag, 'mode': mode, 'W': W,
        'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
        'precision': precision, 'recall': recall, 'f1': f1,
        'tpr': tpr, 'fpr': fpr,
        'n_trajectories': len(trajectories),
    }


def validate_run(tag: str, mode: str, W: int) -> dict | None:
    """
    Load the last checkpoint for tag, evaluate every validation trajectory,
    compute sample-level TP/FP/TN/FN, save per-trajectory CSV and violin plot.
    Returns aggregate metric dict (with 'tag', 'tpr', 'fpr', …) or None.

    If validation_trajectories.csv already exists for this tag, the per-trajectory
    table is loaded from disk and only the aggregate metrics are recomputed
    (delete the CSV manually to force re-evaluation, e.g. after changing the
    reward function).
    """
    csv_path = CKPT_ROOT / tag / 'validation_trajectories.csv'
    if csv_path.exists():
        print(f'[skip] validate {tag}: {csv_path} already present')
        trajectories = pd.read_csv(csv_path)
        result = _aggregate_metrics(tag, mode, W, trajectories)
        tpr, fpr, f1 = result['tpr'], result['fpr'], result['f1']
        out_dir = CKPT_ROOT / tag
        ckpt_path = last_checkpoint(tag)
        ckpt_data = torch.load(ckpt_path, weights_only=True) if ckpt_path else None
        print(f"[{tag}]  TP={result['TP']}  FN={result['FN']}  FP={result['FP']}  TN={result['TN']}  "
              f"TPR={tpr:.4f}  FPR={fpr:.4f}  F1={f1:.4f}")
    else:

        ckpt_path = last_checkpoint(tag)
        if ckpt_path is None:
            print(f'[validate] No checkpoint found for {tag}, skipping.')
            return None

        print(f'\n[validate] {tag}  ←  {ckpt_path}')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        q_net  = build_model(mode, W)
        ckpt_data = torch.load(ckpt_path, weights_only=True)
        state     = ckpt_data.get('model_state_dict', ckpt_data)
        q_net.load_state_dict(state)
        q_net.to(device).eval()

        settings   = pd.read_csv(ROOT / 'labels' / 'settings.csv', sep=';')
        val_labels = torch.load(ROOT / 'labels' / 'validation' / 'val_labels.pt',
                                weights_only=False)
        val_indices = val_labels['indices']

        # Skip phasor warm-up region (CYCLE_LEN-1+W). Otherwise the first ~W eval
        # windows are out-of-distribution (excluded from training by the dataset
        # warm-up filter) and produce spurious non-WAIT outputs → FP penalties.
        t_start         = (CYCLE_LEN - 1 + W) if mode in ('phasor', 'combined') else (W - 1)
        n_eval          = N_STEPS - t_start
        t_rel_arr       = np.arange(t_start, N_STEPS) - FAULT_ONSET
        post_fault_step = t_rel_arr >= 0
        EVAL_BATCH      = 1024

        def episode_actions(arr: np.ndarray) -> np.ndarray:
            actions = np.empty(n_eval, dtype=np.int64)
            offset  = t_start - W + 1   # first valid window-start index
            with torch.no_grad():
                for i in range(0, n_eval, EVAL_BATCH):
                    j     = min(i + EVAL_BATCH, n_eval)
                    batch = np.stack([arr[offset + k : offset + k + W] for k in range(i, j)])
                    q     = q_net(torch.from_numpy(batch).to(device))
                    actions[i:j] = q.argmax(dim=1).cpu().numpy()
            return actions

        def traj_reward(actions: np.ndarray, expert_ep: int, is_fault: bool) -> float:
            return float(sum(
                compute_reward(int(actions[i]), expert_ep, int(t_rel_arr[i]), is_fault)
                for i in range(n_eval)
            ))

        records = []
        for sim_idx in tqdm.tqdm(val_indices, desc=f'[validate] {tag}'):
            row          = settings.loc[settings['general/sim_idx'] == sim_idx].iloc[0]
            event_type   = row['events/event_type']
            event_target = row['events/event_target']
            is_fault_ep  = event_type.startswith('flt_')
            expert_ep    = get_expert_action(event_type, event_target)

            arr = _load_obs(sim_idx, mode)
            if arr is None:
                continue

            actions    = episode_actions(arr)
            agent_trip = actions != WAIT

            expert_trip_mask = post_fault_step if is_fault_ep else np.zeros(n_eval, dtype=bool)
            correct_line     = (actions == expert_ep)

            tp = int(np.sum( expert_trip_mask &  correct_line))
            fn = int(np.sum( expert_trip_mask & ~correct_line))
            fp = int(np.sum(~expert_trip_mask &  agent_trip))
            tn = int(np.sum(~expert_trip_mask & ~agent_trip))

            expert_a = np.full(n_eval, WAIT, dtype=np.int64)
            if is_fault_ep:
                expert_a[post_fault_step] = expert_ep

            agent_r = traj_reward(actions,  expert_ep, is_fault_ep)
            max_r   = traj_reward(expert_a, expert_ep, is_fault_ep)
            norm_r  = agent_r / max_r if max_r != 0 else float('nan')

            records.append({
                'sim_idx'     : int(sim_idx),
                'event_type'  : event_type,
                'is_fault'    : is_fault_ep,
                'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
                'agent_reward': agent_r,
                'max_reward'  : max_r,
                'norm_reward' : norm_r,
            })

        trajectories = pd.DataFrame(records)
        out_dir      = CKPT_ROOT / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / 'validation_trajectories.csv'
        trajectories.to_csv(csv_path, index=False)

        result = _aggregate_metrics(tag, mode, W, trajectories)
        tpr, fpr, f1 = result['tpr'], result['fpr'], result['f1']
        print(f"[{tag}]  TP={result['TP']}  FN={result['FN']}  FP={result['FP']}  TN={result['TN']}  "
            f'TPR={tpr:.4f}  FPR={fpr:.4f}  F1={f1:.4f}')

    # ── Violin plot ───────────────────────────────────────────────────────────
    rng      = np.random.default_rng(0)
    groups   = ['Non-fault', 'Fault']
    raw_data = [
        trajectories.loc[~trajectories['is_fault'], 'agent_reward'].values,
        trajectories.loc[ trajectories['is_fault'], 'agent_reward'].values,
    ]
    # violinplot requires at least 2 points; fall back to boxplot for tiny groups
    positions = [i for i, d in enumerate(raw_data) if len(d) >= 2]
    plot_data = [raw_data[i] for i in positions]

    fig, ax = plt.subplots(figsize=(6, 5), layout="compressed")
    if plot_data:
        parts = ax.violinplot(plot_data, positions=positions,
                              showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightgrey')
            pc.set_edgecolor('grey')
            pc.set_alpha(0.8)
        for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if key in parts:
                parts[key].set_color('grey')
    for pos, data in zip(positions, plot_data):
        ax.scatter(pos + rng.uniform(-0.05, 0.05, len(data)), data,
                   s=12, alpha=0.5, zorder=3)
    # single-point groups as horizontal lines
    for i, data in enumerate(raw_data):
        if len(data) == 1:
            ax.axhline(data[0], color='C0', lw=1.5, alpha=0.7)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f'{g}\n(n={len(d)})' for g, d in zip(groups, raw_data)])
    ax.set_ylabel('Agent cumulative reward')
    # ax.set_title(f'{tag}\nTPR={tpr:.3f}  FPR={fpr:.3f}  F1={f1:.3f}')
    ax.grid(True, axis='y', alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    fig_path = out_dir / f'{tag}_reward_violin{PLOT_DTYPE}'
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f'[{tag}]  Violin plot → {fig_path}')

    # ── Training-loss curves ──────────────────────────────────────────────────
    history = ckpt_data.get('history') if isinstance(ckpt_data, dict) else None
    if history and history.get('train_td'):
        epochs = np.arange(1, len(history['train_td']) + 1)
        fig, axes = plt.subplots(2, 1, figsize=(4, 4), layout="compressed")

        ax = axes[0]
        ax.plot(epochs, history['train_td'], label='train', color='C0')
        if history.get('test_td'):
            ax.plot(epochs, history['test_td'], label='test', color='C1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('TD loss')
        ax.set_title('TD loss')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.plot(epochs, history['train_cql'], label='train', color='C0')
        if history.get('test_cql'):
            ax.plot(epochs, history['test_cql'], label='test', color='C1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CQL penalty')
        ax.set_title('CQL penalty')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # fig.suptitle(f'{tag} — training curves', fontsize=13)
        loss_path = out_dir / f'{tag}_training_curves{PLOT_DTYPE}'
        fig.savefig(loss_path, dpi=120)
        plt.close(fig)
        print(f'[{tag}]  Training curves → {loss_path}')

    return result


# ── ROC diagram ───────────────────────────────────────────────────────────────

def plot_roc(val_results: list[dict], out_path: Path):
    """
    Scatter each run as a labelled point in ROC space (FPR, TPR).
    One marker per run; colour-coded by W (window size) where available.
    """
    if not val_results:
        print('[ROC] No validation results — skipping ROC plot.')
        return

    df  = pd.DataFrame(val_results)
    fig, ax = plt.subplots(figsize=(5, 4), layout="compressed")

    # colour by window size
    w_vals  = sorted(df['W'].unique())
    palette = plt.cm.tab10.colors
    w_color = {w: palette[i % len(palette)] for i, w in enumerate(w_vals)}

    for _, row in df.iterrows():
        color = w_color.get(row['W'], 'gray')
        ax.scatter(row['fpr'], row['tpr'], color=color, s=80, zorder=3,
                   label=f"W={row['W']}")
        ax.annotate(row['tag'], (row['fpr'], row['tpr']),
                    textcoords='offset points', xytext=(5, 3), fontsize=7)

    # diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='Random')

    # deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc='lower right', fontsize=8)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title('ROC parameter study')
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[ROC]  Saved → {out_path}')


def plot_roc_zoomed(val_results: list[dict], out_path: Path):
    """
    Scatter each run as a labelled point in ROC space (FPR, TPR).
    One marker per run; colour-coded by W (window size) where available.
    """
    if not val_results:
        print('[ROC] No validation results — skipping ROC plot.')
        return

    df  = pd.DataFrame(val_results)
    fig, ax = plt.subplots(figsize=(3, 3), layout="compressed")

    # colour by window size
    w_vals  = sorted(df['W'].unique())
    palette = plt.cm.tab10.colors
    w_color = {w: palette[i % len(palette)] for i, w in enumerate(w_vals)}

    for _, row in df.iterrows():
        color = w_color.get(row['W'], 'gray')
        ax.scatter(row['fpr'], row['tpr'], color=color, s=80, zorder=3,
                   label=f"W={row['W']}")
        ax.annotate(row['tag'], (row['fpr'], row['tpr']),
                    textcoords='offset points', xytext=(5, 3), fontsize=7)

    # diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='Random')

    # deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc='lower right', fontsize=8)

    ax.set_xlim(-0.02, 0.22)
    ax.set_ylim(0.87, 0.96)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title('ROC parameter study')
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[ROC]  Saved → {out_path}')


# ── Per-event-type & error-composition diagnostics ────────────────────────────

def _load_trajectories(tag: str) -> pd.DataFrame | None:
    p = CKPT_ROOT / tag / 'validation_trajectories.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['tag'] = tag
    return df


def _draw_heatmap(piv: pd.DataFrame, title: str, out_path: Path, cmap: str):
    if piv.empty:
        print(f'[heatmap] {title}: empty pivot, skipping.')
        return
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * piv.shape[1] + 3),
                                    max(3, 0.45 * piv.shape[0] + 2)),
                           layout='compressed')
    im = ax.imshow(piv.values, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color='white' if v < 0.5 else 'black', fontsize=7)
    fig.colorbar(im, ax=ax, label=title)
    # ax.set_title(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[heatmap]  Saved → {out_path}')


def _event_type_tables(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-episode TP/FN/FP/TN rows into per-(tag, event_type) metric
    tables. Returns (fault, nonfault) DataFrames with recall/f1 columns on the
    fault table and an fpr column on the non-fault table.
    """
    agg = full.groupby(['tag', 'event_type'], as_index=False).agg(
        TP=('TP', 'sum'), FN=('FN', 'sum'),
        FP=('FP', 'sum'), TN=('TN', 'sum'),
        is_fault=('is_fault', 'first'),
        n=('sim_idx', 'count'),
    )

    fault    = agg[ agg['is_fault']].copy()
    nonfault = agg[~agg['is_fault']].copy()

    fault['recall'] = fault['TP'] / (fault['TP'] + fault['FN']).replace(0, np.nan)
    fault['f1'] = 2*fault['TP'] / (2 * fault['TP'] + fault['FP'] + fault['FN']).replace(0, np.nan)

    nonfault['fpr'] = nonfault['FP'] / (nonfault['FP'] + nonfault['TN']).replace(0, np.nan)

    return fault, nonfault


def plot_event_type_breakdown(tags: list[str], out_dir: Path):
    """
    Aggregate sample-level TP/FN/FP/TN per (tag, event_type) from cached
    validation_trajectories.csv files and draw two heatmaps:

      recall_by_event_type{ext} — recall (TPR) per fault event_type × tag
      fpr_by_event_type{ext}    — FPR per non-fault event_type × tag

    Faults are rows where event_type starts with 'flt_'. For non-fault types
    TPR is undefined (no positives), so we report FPR instead.
    """
    frames = [_load_trajectories(t) for t in tags]
    frames = [f for f in frames if f is not None]
    if not frames:
        print('[event-heatmap] No trajectory CSVs found; skipping.')
        return
    full = pd.concat(frames, ignore_index=True)

    fault, nonfault = _event_type_tables(full)

    xlsx_path = out_dir / 'event_type_breakdown.xlsx'
    with pd.ExcelWriter(xlsx_path) as xw:
        fault.to_excel(xw, sheet_name='fault', index=False)
        nonfault.to_excel(xw, sheet_name='nonfault', index=False)
    print(f'[event-heatmap]  Tables  → {xlsx_path}')

    recall_fault_piv = fault.pivot(index='event_type', columns='tag', values='recall')
    f1_fault_piv = fault.pivot(index='event_type', columns='tag', values='f1')

    fpr_nonfault_piv = nonfault.pivot(index='event_type', columns='tag', values='fpr')

    _draw_heatmap(recall_fault_piv, 'Recall (TPR) by fault type',
                  out_dir / f'fault_episode_recall_by_event_type{PLOT_DTYPE}', 'viridis')
    _draw_heatmap(f1_fault_piv, 'F1-score by fault type',
                  out_dir / f'fault_episode_f1_by_event_type{PLOT_DTYPE}', 'viridis')
    
    _draw_heatmap(fpr_nonfault_piv, 'FPR by non-fault event type',
                  out_dir / f'nonfault_episode_fpr_by_event_type{PLOT_DTYPE}', 'viridis')



# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=['A', 'B', 'C'], default=None,
                        help='Which block to run (default: all)')
    parser.add_argument('--best', default='combined_W48',
                        help='Best Block-A tag to use as base for B and C (default: combined_W48)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Training seed passed to every CQL run')
    parser.add_argument('--deterministic', action='store_true',
                        help='Require deterministic Torch/CUDA algorithms')
    args = parser.parse_args()

    run_all     = args.block is None
    val_results = []

    # ── Block A ───────────────────────────────────────────────────────────────
    if run_all or args.block == 'A':
        print('\n══ Block A: input representation × window size ══')

        for cfg in BLOCK_A_CQL:
            run_cql(**cfg, seed=args.seed, deterministic=args.deterministic)
            result = validate_run(**cfg)
            if result:
                val_results.append(result)

        # for cfg in BLOCK_A_BC:
        #     run_bc(**cfg)
        #     result = validate_run(**cfg)
        #     if result:
        #         val_results.append(result)


    # Parse a Block-A tag like 'raw_v2_W192' into (mode='raw_v2', W=192).
    # Splits on the right-most '_W' so multi-underscore modes (raw_v2) work.
    def _parse_base_tag(tag: str) -> tuple[str, int]:
        mode, sep, w_str = tag.rpartition('_W')
        if not sep:
            raise ValueError(f"--best tag {tag!r} must contain '_W<int>' (e.g. 'raw_v2_W192')")
        return mode, int(w_str)

    # ── Block B ───────────────────────────────────────────────────────────────
    if run_all or args.block == 'B':
        print(f'\n══ Block B: reward shaping  (base: {args.best}) ══')

        base_tag            = args.best
        base_mode, base_W   = _parse_base_tag(base_tag)

        for tmpl in BLOCK_B_TEMPLATES:
            tag = f'{base_tag}_{tmpl["tag_suffix"]}'
            run_cql(tag=tag, mode=base_mode, W=base_W,
                    fp_penalty=tmpl['fp_penalty'],
                    correct_reward=tmpl['correct_reward'],
                    seed=args.seed, deterministic=args.deterministic)
            result = validate_run(tag=tag, mode=base_mode, W=base_W)
            if result:
                val_results.append(result)

    # ── Block C ───────────────────────────────────────────────────────────────
    if run_all or args.block == 'C':
        print(f'\n══ Block C: CQL alpha  (base: {args.best}) ══')

        base_tag            = args.best
        base_mode, base_W   = _parse_base_tag(base_tag)

        for tmpl in BLOCK_C_TEMPLATES:
            tag = f'{base_tag}_{tmpl["tag_suffix"]}'
            run_cql(tag=tag, mode=base_mode, W=base_W, alpha=tmpl['alpha'],
                    seed=args.seed, deterministic=args.deterministic)
            result = validate_run(tag=tag, mode=base_mode, W=base_W)
            if result:
                val_results.append(result)

    # ── Final ROC diagram ──────────────────────────────────────────────────────
    if val_results:
        val_df_path = CKPT_ROOT / 'validation_summary.csv'
        pd.DataFrame(val_results).to_csv(val_df_path, index=False)
        print(f'\nValidation summary → {val_df_path}')
        plot_roc(val_results, CKPT_ROOT / f'roc_all_runs{PLOT_DTYPE}')
        plot_roc_zoomed(val_results, CKPT_ROOT / f'roc_zoomed_all_runs{PLOT_DTYPE}')

        tags = [r['tag'] for r in val_results]
        plot_event_type_breakdown(tags, CKPT_ROOT)

    print(f'\nDone. Results log: {RESULTS}')


if __name__ == '__main__':
    main()

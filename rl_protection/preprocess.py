"""
preprocess.py — One-time data preparation pipeline. Run stages in order:

  python -m rl_protection.preprocess stage1   # CSV → float16 .npy  (~15 min)
  python -m rl_protection.preprocess stage2   # .npy → phasor .npy  (~2 hrs)
  python -m rl_protection.preprocess stage3   # build transitions.npz

Each stage is safe to re-run; existing output files are skipped by default.
Pass --overwrite to force regeneration.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import scipy.signal
from tqdm import tqdm

from .constants import (
    DATA_DIR, VAL_DATA_DIR,
    NPY_TRAIN_DIR, NPY_VAL_DIR,
    PHASOR_TRAIN_DIR, PHASOR_VAL_DIR,
    SETTINGS_CSV, TRANSITIONS_NPZ,
    N_STEPS, N_SIGNAL_COLS, N_PHASOR_COLS,
    N_LINE_COLS, LINE_COL_INDICES,
    GRID_FREQ, SAMPLE_RATE, CYCLE_LEN,
)
from .expert import get_expert_action, build_transitions


# ── Stage 1: CSV → float16 .npy ───────────────────────────────────────────────

def load_csv(path: str) -> np.ndarray:
    """
    Load one simulation CSV and return a float32 array of shape (N_STEPS, N_SIGNAL_COLS).

    CSV format:
      - Row 0 (pandas index 0): unit labels  ('b:tnow in s', 'c:Isec:A in A', ...)
      - Rows 1 onwards: numeric data
      - Column 0: time (dropped)
      - Columns 1–276: signals
    """
    df = pd.read_csv(path, low_memory=False)
    # Drop header row (units) and time column; convert to float32
    arr = df.iloc[1:, 1:].astype(np.float32).values
    return arr   # shape (N_STEPS, N_SIGNAL_COLS)


def run_stage1(overwrite: bool = False):
    """Convert every CSV in data/ and data/validation/ to float16 .npy."""
    pairs = [
        (DATA_DIR,     NPY_TRAIN_DIR),
        (VAL_DATA_DIR, NPY_VAL_DIR),
    ]
    for src_dir, dst_dir in pairs:
        csv_files = sorted(glob.glob(os.path.join(src_dir, 'result*.csv')))
        if not csv_files:
            print(f'Stage 1: no CSV files found in {src_dir}', flush=True)
            continue

        # Count how many .npy files already exist
        already_done = sum(
            1 for p in csv_files
            if os.path.exists(os.path.join(dst_dir, os.path.basename(p).replace('.csv', '.npy')))
        )
        todo = len(csv_files) - already_done if not overwrite else len(csv_files)
        print(
            f'Stage 1: {src_dir}\n'
            f'  Total={len(csv_files)}  already_converted={already_done}  to_process={todo}',
            flush=True,
        )

        n_done = 0
        errors = []
        for csv_path in csv_files:
            name     = os.path.basename(csv_path).replace('.csv', '.npy')
            out_path = os.path.join(dst_dir, name)
            if not overwrite and os.path.exists(out_path):
                continue   # skip — already converted
            try:
                arr = load_csv(csv_path)
                np.save(out_path, arr.astype(np.float16))
            except Exception as e:
                errors.append((csv_path, str(e)))
                print(f'  ERROR on {csv_path}: {e}', flush=True)
                continue
            n_done += 1
            if n_done % 100 == 0:
                print(f'  converted {n_done}/{todo}', flush=True)

        print(f'  Stage 1 done: {n_done} converted, {len(errors)} errors, {already_done} skipped (already existed)', flush=True)


# ── Stage 2: .npy → phasor .npy ──────────────────────────────────────────────

def sliding_phasor(x: np.ndarray) -> np.ndarray:
    """
    Full-cycle sliding DFT: estimate the complex phasor at GRID_FREQ Hz.

    Adapted from sandkasten.ipynb.

    x     : 1-D float array of length N_STEPS
    returns: complex64 array of shape (N_STEPS,)
             entries 0 .. CYCLE_LEN-2 are invalid (warm-up); set to NaN.
    """
    N = CYCLE_LEN   # samples per cycle
    # FIR coefficients: e^(j*2*pi*k/N) for k = 0..N-1, normalised by N/2
    k    = np.arange(N, dtype=np.float32)
    h    = (2.0 / N) * np.exp(-1j * 2 * np.pi * GRID_FREQ / SAMPLE_RATE * k)

    # Complex FIR filter via lfilter (split real/imag for speed)
    y = scipy.signal.lfilter(h.real, [1.0], x) + 1j * scipy.signal.lfilter(h.imag, [1.0], x)
    # Warm-up: first N-1 samples are unreliable
    y[:N - 1] = np.nan
    return y.astype(np.complex64)


def impedance_phasor(u_phasor: np.ndarray, i_phasor: np.ndarray,
                     i_threshold: float = 0.005,
                     i_peak: float | np.ndarray | None = None) -> np.ndarray:
    """
    Compute complex impedance Z = V / I from two phasor arrays.

    Adapted from sandkasten.ipynb.
    Entries where |I| < i_threshold * i_peak are set to NaN (division noise).

    `i_peak` is the reference current magnitude used to set the masking threshold.
    If None, falls back to the per-phase peak `np.nanmax(|i_phasor|)`. For balanced
    three-phase processing, pass `max(|Ia|, |Ib|, |Ic|)` so all three phases share
    the same threshold and don't pick up phase-asymmetric quirks (a healthy phase
    has a small peak and would otherwise keep noisy R/X near nominal load).

    Returns complex64 array of shape (N_STEPS,).
    """
    if i_peak is None:
        i_peak = np.nanmax(np.abs(i_phasor))
    mask = np.abs(i_phasor) < i_threshold * i_peak
    z    = np.where(mask, np.nan + 0j, u_phasor / (i_phasor + 1e-12))
    return z.astype(np.complex64)


def compute_phasors(arr: np.ndarray) -> np.ndarray:
    """
    Compute three-phase phasor/impedance features for line cubicles only.

    arr     : float32 array of shape (N_STEPS, N_SIGNAL_COLS)
              Columns are grouped per device in blocks of 6: Ia, Ib, Ic, Va, Vb, Vc
    returns : float32 array of shape (N_STEPS, N_PHASOR_COLS)
              For each of 29 line cubicles, all three phases:
                [|Va|, |Ia|, Ra, Xa, |Vb|, |Ib|, Rb, Xb, |Vc|, |Ic|, Rc, Xc]
              → 29 × 12 = 348 features.
              NaN entries (warm-up / low-current) are replaced with 0.
    """
    n_cubicles = N_LINE_COLS // 6   # 29
    out = np.zeros((N_STEPS, n_cubicles * 12), dtype=np.float32)

    for c in range(n_cubicles):
        # Six absolute column indices for this cubicle: [Ia, Ib, Ic, Va, Vb, Vc]
        cols = LINE_COL_INDICES[c * 6 : (c + 1) * 6]

        # Compute V and I phasors for all three phases first
        v_phs = [sliding_phasor(arr[:, cols[p + 3]].astype(np.float64)) for p in range(3)]
        i_phs = [sliding_phasor(arr[:, cols[p]    ].astype(np.float64)) for p in range(3)]

        # Causal joint peak |I| across the three phases through each timestep.
        i_peak_cubicle = np.maximum.accumulate(np.maximum.reduce([
            np.nan_to_num(np.abs(i_ph), nan=0.0) for i_ph in i_phs
        ]))

        for ph in range(3):
            v_ph = v_phs[ph]
            i_ph = i_phs[ph]
            z_ph = impedance_phasor(v_ph, i_ph, i_peak=i_peak_cubicle)

            base = c * 12 + ph * 4
            out[:, base + 0] = np.nan_to_num(np.abs(v_ph), nan=0.0)    # |V|
            out[:, base + 1] = np.nan_to_num(np.abs(i_ph), nan=0.0)    # |I|
            out[:, base + 2] = np.nan_to_num(z_ph.real,    nan=0.0)    # R
            out[:, base + 3] = np.nan_to_num(z_ph.imag,    nan=0.0)    # X

    return out


def run_stage2(overwrite: bool = False):
    """Compute phasor features for all .npy files and save to phasor subdirectory."""
    pairs = [
        (NPY_TRAIN_DIR, PHASOR_TRAIN_DIR),
        (NPY_VAL_DIR,   PHASOR_VAL_DIR),
    ]
    for src_dir, dst_dir in pairs:
        npy_files = sorted(glob.glob(os.path.join(src_dir, 'result*.npy')))
        if not npy_files:
            print(f'Stage 2: no .npy files found in {src_dir} — run stage1 first', flush=True)
            continue

        already_done = sum(
            1 for p in npy_files
            if os.path.exists(os.path.join(dst_dir, os.path.basename(p)))
        )
        todo = len(npy_files) - already_done if not overwrite else len(npy_files)
        print(
            f'Stage 2: {src_dir}\n'
            f'  Total={len(npy_files)}  already_done={already_done}  to_process={todo}',
            flush=True,
        )

        n_done = 0
        errors = []
        pending = [
            p for p in npy_files
            if overwrite or not os.path.exists(os.path.join(dst_dir, os.path.basename(p)))
        ]
        for npy_path in tqdm(pending, desc=f'stage2 {os.path.basename(src_dir)}', unit='ep'):
            out_path = os.path.join(dst_dir, os.path.basename(npy_path))
            try:
                arr     = np.load(npy_path).astype(np.float32)
                phasors = compute_phasors(arr)
                # Clip to float16 range before casting — large impedances (Z = V/I when I
                # is very small) can exceed float16 max (~65504) and produce inf/nan.
                np.save(out_path, np.clip(phasors, -65504.0, 65504.0).astype(np.float16))
            except Exception as e:
                errors.append((npy_path, str(e)))
                tqdm.write(f'  ERROR on {npy_path}: {e}')
                continue
            n_done += 1

        print(f'  Stage 2 done: {n_done} computed, {len(errors)} errors, {already_done} skipped (already existed)', flush=True)


# ── Stage 3: Build transition index table ─────────────────────────────────────

def run_stage3(overwrite: bool = False):
    """Build and save transitions.npz from settings.csv and train_labels.pt."""
    import torch

    if not overwrite and os.path.exists(TRANSITIONS_NPZ):
        print('Stage 3: transitions.npz already exists. Pass --overwrite to rebuild.')
        return

    settings = pd.read_csv(SETTINGS_CSV, sep=';')

    # Load train indices from the existing label file
    label_data    = torch.load(
        os.path.join(os.path.dirname(SETTINGS_CSV), 'train_labels.pt'),
        weights_only=False,
    )
    train_indices = label_data['indices']   # list of sim_idx values

    print(f'Stage 3: building transitions for {len(train_indices)} training episodes...')
    transitions = build_transitions(settings, train_indices)
    print(f'  Total transitions: {len(transitions):,}')

    np.savez(TRANSITIONS_NPZ, transitions=transitions)
    print(f'  Saved to {TRANSITIONS_NPZ}')

    # Also save val indices for reference
    val_data    = torch.load(
        os.path.join(os.path.dirname(SETTINGS_CSV), 'validation', 'val_labels.pt'),
        weights_only=False,
    )
    val_indices = val_data['indices']
    val_path    = os.path.join(os.path.dirname(TRANSITIONS_NPZ), 'val_indices.npz')
    np.savez(val_path, indices=np.array(val_indices, dtype=np.int32))
    print(f'  Val indices saved to {val_path}')


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['stage1', 'stage2', 'stage3', 'all'])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.stage in ('stage1', 'all'):
        run_stage1(args.overwrite)
    if args.stage in ('stage2', 'all'):
        run_stage2(args.overwrite)
    if args.stage in ('stage3', 'all'):
        run_stage3(args.overwrite)

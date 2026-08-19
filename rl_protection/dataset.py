"""
dataset.py — PyTorch Dataset and DataLoader for offline RL training.

Memory strategy:
  All .npy files are loaded into RAM as float32 arrays at init time.
  Column slicing (raw mode) is also done once at load time.
  Rewards and done flags are precomputed for all transitions.
  This keeps __getitem__ to a single array slice + torch.from_numpy.

Usage:
  from rl_protection.dataset import build_dataloader
  loader = build_dataloader(transitions, mode='phasor', W=192, batch_size=256)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .constants import (
    NPY_TRAIN_DIR, PHASOR_TRAIN_DIR,
    LINE_COL_INDICES, N_STEPS,
    FAULT_ONSET, WAIT, CYCLE_LEN,
)
from .reward import compute_reward


def _load_window(arr: np.ndarray, t: int, W: int) -> np.ndarray:
    """Extract a (W, C) window ending at step t. Zero-pads the left if t < W-1."""
    start = t - W + 1
    if start >= 0:
        return arr[start : t + 1]

    pad_len   = -start
    available = arr[0 : t + 1]
    pad       = np.zeros((pad_len, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([pad, available], axis=0)


# Public aliases used by evaluate.py
def load_window(arr: np.ndarray, t: int, W: int) -> np.ndarray:
    return _load_window(arr, t, W)


def extract_features(raw_w: np.ndarray, phasor_w: np.ndarray, mode: str) -> np.ndarray:
    """Select and concatenate observation columns for the given mode.

    raw_w    : (W, N_SIGNAL_COLS) float32
    phasor_w : (W, N_PHASOR_COLS) float32
    Returns  : (W, C) float32
    """
    if mode == 'raw' or mode == 'raw_v2':
        return raw_w[:, LINE_COL_INDICES]
    if mode == 'phasor':
        return phasor_w
    # combined
    return np.concatenate([raw_w[:, LINE_COL_INDICES], phasor_w], axis=1)


class GridDataset(Dataset):
    """
    Offline RL dataset for the power grid relay selection task.

    Each item is a tuple:
      (obs, next_obs, action, reward, done, t, sim_idx)
    where obs/next_obs are float32 tensors of shape (W, C).

    Optimisations vs. the naive per-item approach:
      - Arrays are loaded once as float32 with columns pre-sliced for the mode.
      - Rewards and done flags are vectorised at init (no per-item Python call).
      - __getitem__ is a thin slice + torch.from_numpy (zero-copy).
    """

    def __init__(
        self,
        transitions: np.ndarray,
        npy_dir: str,
        phasor_dir: str,
        mode: str = 'phasor',
        W: int = 192,
        sim_idx_to_filename: dict = None,
        false_positive_penalty: float = -100.0,
        correct_reward: float = 5.0,
        wrong_relay_penalty: float = -100.0,
        wait_penalty: float = 0.0,
    ):
        # Drop transitions whose observation window would overlap the phasor
        # warm-up region. sliding_phasor sets the first CYCLE_LEN-1 samples to
        # NaN (later replaced with 0 in compute_phasors), so the first window
        # with no warm-up contamination ends at t = CYCLE_LEN - 1 + W.
        # Raw mode is unaffected; phasor and combined need filtering.
        if mode in ('phasor', 'combined'):
            min_t  = CYCLE_LEN - 1 + W
            keep   = transitions[:, 1] >= min_t
            n_drop = int((~keep).sum())
            if n_drop:
                transitions = transitions[keep]
                print(f'GridDataset: dropped {n_drop} pre-fault transitions in {mode} '
                      f'mode (phasor warm-up filter, W={W}, min_t={min_t})')

        self.transitions = transitions
        self.mode        = mode
        self.W           = W

        # Build sim_idx → filename mapping
        npy_files = sorted(glob.glob(os.path.join(npy_dir, 'result*.npy')))
        if sim_idx_to_filename is None:
            sim_idx_to_filename = {}
            for path in npy_files:
                stem    = os.path.basename(path).replace('.npy', '')
                sim_idx = int(stem.replace('result', ''))
                sim_idx_to_filename[sim_idx] = stem

        # Load arrays into RAM as float32, pre-sliced for the mode so that
        # __getitem__ only needs a contiguous row slice (no fancy indexing).
        self.arrays = {}
        for sim_idx, stem in sim_idx_to_filename.items():
            if mode in ('raw', 'raw_v2', 'combined'):
                raw_path = os.path.join(npy_dir, stem + '.npy')
                if os.path.exists(raw_path):
                    raw = np.load(raw_path).astype(np.float32)[:, LINE_COL_INDICES]
                else:
                    raw = None
            else:
                raw = None

            if mode in ('phasor', 'combined'):
                phasor_path = os.path.join(phasor_dir, stem + '.npy')
                if os.path.exists(phasor_path):
                    phasor = np.load(phasor_path).astype(np.float32)
                else:
                    phasor = None
            else:
                phasor = None

            if mode == 'raw' or mode == 'raw_v2':
                arr = raw
            elif mode == 'phasor':
                arr = phasor
            else:  # combined
                if raw is not None and phasor is not None:
                    arr = np.concatenate([raw, phasor], axis=1)
                else:
                    arr = raw if raw is not None else phasor

            if arr is not None:
                # Ensure contiguous C-order for fast slicing
                self.arrays[sim_idx] = np.ascontiguousarray(arr)

        # Determine observation width from any loaded array
        self._n_cols = next(iter(self.arrays.values())).shape[1]

        # Precompute rewards by delegating to compute_reward — single source of truth
        # with rl_protection.reward, so dataset and notebooks can never drift.
        actions        = transitions[:, 2].astype(np.int64)
        is_fault       = transitions[:, 3].astype(np.bool_)
        expert_actions = transitions[:, 4].astype(np.int64)
        t_rel          = transitions[:, 1].astype(np.int64) - FAULT_ONSET

        n = len(transitions)
        self.rewards = np.fromiter(
            (compute_reward(int(actions[i]), int(expert_actions[i]),
                            int(t_rel[i]),  bool(is_fault[i]),
                            false_positive_penalty=false_positive_penalty,
                            correct_reward=correct_reward,
                            wrong_relay_penalty=wrong_relay_penalty,
                            wait_penalty=wait_penalty)
             for i in range(n)),
            dtype=np.float32,
            count=n,
        )
        self.dones   = (actions != WAIT).astype(np.float32)
        self.actions = actions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int):
        sim_idx, t = int(self.transitions[idx, 0]), int(self.transitions[idx, 1])

        arr = self.arrays.get(sim_idx)

        if arr is not None:
            obs      = _load_window(arr, t, self.W)
            next_obs = _load_window(arr, min(t + 1, N_STEPS - 1), self.W)
        else:
            obs      = np.zeros((self.W, self._n_cols), dtype=np.float32)
            next_obs = obs

        return (
            torch.from_numpy(obs),
            torch.from_numpy(next_obs),
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            t,
            sim_idx,
        )


def build_dataloader(
    transitions: np.ndarray,
    mode: str = 'phasor',
    W: int = 192,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    npy_dir: str = NPY_TRAIN_DIR,
    phasor_dir: str = PHASOR_TRAIN_DIR,
    false_positive_penalty: float = -100.0,
    correct_reward: float = 5.0,
    wrong_relay_penalty: float = -100.0,
    wait_penalty: float = 0.0,
    train_frac: float = 0.9,
    split_seed: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test DataLoaders with an episode-level split.

    Episodes (sim_idx values) are partitioned into train and test groups; every
    transition for a given episode goes entirely to one side or the other. This
    makes the test loss/accuracy a real held-out-episode estimator.
    """

    dataset = GridDataset(
        transitions=transitions,
        npy_dir=npy_dir,
        phasor_dir=phasor_dir,
        mode=mode,
        W=W,
        false_positive_penalty=false_positive_penalty,
        correct_reward=correct_reward,
        wrong_relay_penalty=wrong_relay_penalty,
        wait_penalty=wait_penalty,
    )

    # Episode-level split. Use the dataset's (possibly warm-up-filtered)
    # transitions array so subset indices line up with __getitem__.
    sim_ids       = dataset.transitions[:, 0]
    unique_sims   = np.unique(sim_ids)
    rng           = np.random.default_rng(split_seed)
    perm          = rng.permutation(len(unique_sims))
    n_train_eps   = int(train_frac * len(unique_sims))
    train_sims    = unique_sims[perm[:n_train_eps]]
    train_sim_set = set(train_sims.tolist())

    train_mask  = np.isin(sim_ids, train_sims)
    train_idx   = np.flatnonzero( train_mask).tolist()
    test_idx    = np.flatnonzero(~train_mask).tolist()

    print(f'Episode-level split (seed={split_seed}): '
          f'{n_train_eps} train / {len(unique_sims) - n_train_eps} test episodes  |  '
          f'{len(train_idx)} train / {len(test_idx)} test transitions')

    train_dataset = Subset(dataset, train_idx)
    test_dataset  = Subset(dataset, test_idx)

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=use_pin,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=use_pin,
    )
    return (train_loader, test_loader)

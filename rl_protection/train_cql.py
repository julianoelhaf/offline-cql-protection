"""
train_cql.py — Conservative Q-Learning (CQL) for offline relay selection.

Loss = TD loss (Bellman backup) + alpha * CQL penalty.
The CQL penalty discourages overestimating Q-values for actions not seen in the
offline data, which prevents the agent from learning to take unvalidated actions.

Usage:
  python -m rl_protection.train_cql --mode phasor --W 192 --epochs 30 --alpha 0.5
"""

from __future__ import annotations

import os
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

from .constants import TRANSITIONS_NPZ, N_ACTIONS
from .dataset   import build_dataloader
from .models    import build_model


def seed_everything(seed: int, deterministic: bool = False) -> torch.Generator:
    """Seed every RNG used by training and return the DataLoader generator."""
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def cql_loss(q_all: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    CQL penalty: logsumexp over all actions minus Q at the data action.
    Encourages Q-values to be lower for unseen actions.

    Takes pre-computed q_all (from the TD-loss forward pass) to avoid a redundant forward.
    """
    q_data   = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)           # (batch,)
    penalty  = torch.logsumexp(q_all, dim=1) - q_data                    # (batch,)
    return penalty.mean()


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak averaging: target ← tau * source + (1-tau) * target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def train_cql(
    q_net: nn.Module,
    train_loader,
    test_loader,
    n_epochs: int,
    lr: float = 6e-4,
    gamma: float = 0.95,
    alpha: float = 0.5,
    tau: float = 0.005,
    save_dir: str | None = None,
    device: torch.device | str = 'cpu',
    plot_cb=None,
    resume: str | None = None,
    seed: int | None = None,
    deterministic: bool = False,
):
    """
    CQL training loop.

    q_net       : the Q-network (modified in place, should already be on `device`)
    train_loader: DataLoader returning (obs, next_obs, action, reward, done, ...)
    test_loader : DataLoader for evaluation
    n_epochs    : number of passes over the data
    lr          : Adam learning rate
    gamma       : discount factor
    alpha       : CQL penalty weight
    tau         : soft update rate for target network
    save_dir    : directory for per-epoch checkpoints (epoch_01.pt, epoch_02.pt, ...)
    device      : torch device to use for training
    plot_cb     : optional callback called as plot_cb(history) after each epoch
    resume      : path to a checkpoint to resume from (restores model, optimizer, history)

    Returns
    -------
    history : dict with keys 'train_td', 'train_cql', 'test_td', 'test_cql'
              each a list of per-epoch mean values.
    """
    dev = torch.device(device)
    q_net.to(dev)

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    history = {'train_td': [], 'train_cql': [], 'test_td': [], 'test_cql': []}
    start_epoch = 1

    # Restore from checkpoint BEFORE torch.compile (keys don't have _orig_mod. prefix)
    ckpt = None
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=dev, weights_only=True)
        state = ckpt.get('model_state_dict', ckpt)
        q_net.load_state_dict(state)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'history' in ckpt:
            history = ckpt['history']
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from {resume} — continuing at epoch {start_epoch}')

    # Auto-tune conv algorithms for the fixed input shape
    if dev.type == 'cuda' and not deterministic:
        torch.backends.cudnn.benchmark = True

    target_net = copy.deepcopy(q_net)
    if ckpt and 'target_model_state_dict' in ckpt:
        target_net.load_state_dict(ckpt['target_model_state_dict'])
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    if ckpt and 'rng_state' in ckpt:
        rng = ckpt['rng_state']
        random.setstate(rng['python'])
        numpy_rng = rng['numpy']
        np.random.set_state((
            numpy_rng['bit_generator'],
            np.asarray(numpy_rng['state'], dtype=np.uint32),
            numpy_rng['position'],
            numpy_rng['has_gauss'],
            numpy_rng['cached_gaussian'],
        ))
        torch.set_rng_state(rng['torch'].cpu())
        if dev.type == 'cuda' and rng.get('cuda'):
            torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
        if train_loader.generator is not None and rng.get('dataloader') is not None:
            train_loader.generator.set_state(rng['dataloader'].cpu())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    epoch_bar = trange(start_epoch, n_epochs + 1, desc='Epochs')
    for epoch in epoch_bar:
        # Accumulate on-GPU tensors to avoid per-batch CPU-GPU sync via .item().
        total_td  = torch.zeros((), device=dev)
        total_cql = torch.zeros((), device=dev)
        total_n   = 0

        total_test_td  = torch.zeros((), device=dev)
        total_test_cql = torch.zeros((), device=dev)
        total_test_n   = 0

        def compute_td_cql(obs, next_obs, actions, rewards, dones):
            # ── Bellman target (no gradient) ───────────────────────────────
            with torch.no_grad():
                q_next   = target_net(next_obs).max(dim=1).values
                q_target = rewards + gamma * (1.0 - dones) * q_next

            # ── Single forward pass shared by TD loss and CQL penalty ──────
            q_all  = q_net(obs)
            q_pred = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            td     = F.mse_loss(q_pred, q_target)
            cql    = cql_loss(q_all, actions)

            return td, cql

        q_net.train()
        for obs, next_obs, actions, rewards, dones, *_ in tqdm(train_loader, desc=f'Epoch {epoch} [train]', leave=False):
            obs, next_obs = obs.to(dev, non_blocking=True), next_obs.to(dev, non_blocking=True)
            actions, rewards, dones = actions.to(dev, non_blocking=True), rewards.to(dev, non_blocking=True), dones.to(dev, non_blocking=True)

            td, cql = compute_td_cql(obs, next_obs, actions, rewards, dones)

            loss = td + alpha * cql

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

            # Polyak update of the target network — every step, not every epoch.
            soft_update(target_net, q_net, tau)

            bs         = len(actions)
            total_td  += td.detach()  * bs
            total_cql += cql.detach() * bs
            total_n   += bs

        q_net.eval()
        with torch.no_grad():
            for obs, next_obs, actions, rewards, dones, *_ in tqdm(test_loader, desc=f'Epoch {epoch} [test]', leave=False):
                obs, next_obs = obs.to(dev, non_blocking=True), next_obs.to(dev, non_blocking=True)
                actions, rewards, dones = actions.to(dev, non_blocking=True), rewards.to(dev, non_blocking=True), dones.to(dev, non_blocking=True)

                td, cql = compute_td_cql(obs, next_obs, actions, rewards, dones)

                bs              = len(actions)
                total_test_td  += td  * bs
                total_test_cql += cql * bs
                total_test_n   += bs

        # Sync once per epoch: .item() forces a single CPU-GPU transfer here.
        train_td_avg  = (total_td  / total_n).item()
        train_cql_avg = (total_cql / total_n).item()
        test_td_avg   = (total_test_td  / total_test_n).item()
        test_cql_avg  = (total_test_cql / total_test_n).item()

        history['train_td'].append(train_td_avg)
        history['train_cql'].append(train_cql_avg)
        history['test_td'].append(test_td_avg)
        history['test_cql'].append(test_cql_avg)

        epoch_bar.set_postfix(
            train_td=f'{train_td_avg:.4f}',
            train_cql=f'{train_cql_avg:.4f}',
            test_td=f'{test_td_avg:.4f}',
            test_cql=f'{test_cql_avg:.4f}',
        )

        # Save checkpoint after every epoch (atomic write to avoid corruption)
        if save_dir:
            ckpt_path = os.path.join(save_dir, f'epoch_{epoch:02d}.pt')
            tmp_path  = ckpt_path + '.tmp'
            numpy_rng = np.random.get_state()
            torch.save({
                'epoch': epoch,
                'model_state_dict': q_net.state_dict(),
                'target_model_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'seed': seed,
                'deterministic': deterministic,
                'rng_state': {
                    'python': random.getstate(),
                    'numpy': {
                        'bit_generator': numpy_rng[0],
                        'state': numpy_rng[1].tolist(),
                        'position': numpy_rng[2],
                        'has_gauss': numpy_rng[3],
                        'cached_gaussian': numpy_rng[4],
                    },
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if dev.type == 'cuda' else [],
                    'dataloader': (
                        train_loader.generator.get_state()
                        if train_loader.generator is not None else None
                    ),
                },
            }, tmp_path)
            os.replace(tmp_path, ckpt_path)

        # Live plot callback
        if plot_cb is not None:
            plot_cb(history)

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   default='raw', choices=['raw', 'raw_v2', 'phasor', 'combined'])
    parser.add_argument('--W',      type=int,   default=192)
    parser.add_argument('--epochs', type=int,   default=30)
    parser.add_argument('--lr',     type=float, default=1e-3)
    parser.add_argument('--alpha',  type=float, default=0.5,
                        help='CQL penalty weight; try 0.1 / 0.5 / 1.0')
    parser.add_argument('--gamma',  type=float, default=0.95)
    parser.add_argument('--tau',    type=float, default=0.005)
    parser.add_argument('--batch',  type=int,   default=1024)
    parser.add_argument('--workers',type=int,   default=0)
    parser.add_argument('--ckpt',   default=None,
                        help='Optional path to a BC checkpoint to warm-start from')
    parser.add_argument('--resume', default=None,
                        help='Path to an epoch_XX.pt checkpoint to resume from (model + optimizer + history)')
    parser.add_argument('--device', default=None,
                        help='torch device (e.g. cuda, cpu); auto-detects if omitted')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed Python, NumPy, Torch, CUDA, and DataLoader RNGs')
    parser.add_argument('--deterministic', action='store_true',
                        help='Require deterministic Torch/CUDA algorithms')
    parser.add_argument('--fp-penalty',      type=float, default=-100.0,
                        help='Reward for tripping on a non-fault episode (default: -100)')
    parser.add_argument('--correct-reward',  type=float, default=5.0,
                        help='Reward for selecting the correct relay (default: 5.0)')
    parser.add_argument('--save-dir', default=None,
                        help='Checkpoint directory (overrides default path)')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {args.device}')
    loader_generator = (
        seed_everything(args.seed, args.deterministic)
        if args.seed is not None else None
    )

    # Load transition table
    data        = np.load(TRANSITIONS_NPZ)
    transitions = data['transitions']
    print(f'--- Loading {len(transitions):,} transitions ---')

    # DataLoader
    train_loader, test_loader = build_dataloader(
        transitions,
        mode=args.mode,
        W=args.W,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        false_positive_penalty=args.fp_penalty,
        correct_reward=args.correct_reward,
        generator=loader_generator,
    )
    print("--- Transitions loaded ---")
    # Model
    q_net = build_model(args.mode, args.W)
    if args.ckpt and os.path.exists(args.ckpt):
        ckpt_data = torch.load(args.ckpt, weights_only=True)
        state = ckpt_data.get('model_state_dict', ckpt_data)
        q_net.load_state_dict(state)
        print(f'Loaded warm-start checkpoint from {args.ckpt}')

    n_params = sum(p.numel() for p in q_net.parameters())
    print(f'Model params: {n_params:,}')

    # Train
    save_dir = args.save_dir or os.path.join(
        os.path.dirname(__file__), '..', 'checkpoints',
        f'cql_{args.mode}_W{args.W}_a{args.alpha}',
    )

    train_cql(
        q_net, train_loader, test_loader,
        n_epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        alpha=args.alpha,
        tau=args.tau,
        save_dir=save_dir,
        device=args.device,
        resume=args.resume,
        seed=args.seed,
        deterministic=args.deterministic,
    )

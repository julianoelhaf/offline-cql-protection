"""
models.py — Neural network architectures for relay selection.

make_cnn(in_channels, W)     → 1D-CNN, RF=13      
make_cnn_v2(in_channels, W)  → dilated 1D-CNN     
make_combined(in_channels_raw, in_channels_phasor, W) → 2x dilated 1D-CNN     

All models:
  - Input:  float32 tensor of shape (batch, W, C)  [batch-first, time-second, channel-last]
  - Output: float32 tensor of shape (batch, 16)    [Q-values or logits for 16 actions]
"""

import torch
import torch.nn as nn

from .constants import N_ACTIONS, N_LINE_COLS, N_PHASOR_COLS




def make_cnn(in_channels: int, W: int) -> nn.Module:
    """
    1D-CNN for raw waveform input.

    Input:  (batch, W, in_channels)
    Permuted internally to (batch, in_channels, W) for Conv1d.

    Architecture: 3 conv layers with strided downsampling → global avg pool → 2 FC layers.
    """

    class ChannelLayerNorm(nn.Module):
        """LayerNorm across channels per timestep for (batch, C, W) tensors."""
        def __init__(self, C):
            super().__init__()
            self.ln = nn.LayerNorm(C)

        def forward(self, x):
            return self.ln(x.transpose(1, 2)).transpose(1, 2)

    class CNN1d(nn.Module):
        def __init__(self):
            super().__init__()
            # Per-channel z-score with running stats (preserves cross-time magnitude differences,
            self.input_norm = nn.BatchNorm1d(in_channels)
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, 64,  kernel_size=5, padding=2),
                ChannelLayerNorm(64),
                nn.ReLU(),
                nn.Conv1d(64,          128, kernel_size=5, stride=2, padding=2),
                ChannelLayerNorm(128),
                nn.ReLU(),
                nn.Conv1d(128,         128, kernel_size=3, stride=2, padding=1),
                ChannelLayerNorm(128),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)   # (batch, 128, 1)
            self.fc   = nn.Sequential(
                nn.Flatten(),                      # (batch, 128)
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, N_ACTIONS),
            )

        def forward(self, x):
            # x: (batch, W, C) -> (batch, C, W) for BN1d / Conv1d
            x = x.permute(0, 2, 1)
            x = self.input_norm(x)
            x = self.conv(x)
            x = self.pool(x)
            return self.fc(x)

    return CNN1d()


class ChannelLayerNorm(nn.Module):
    """LayerNorm across channels per timestep for (batch, C, W) tensors."""
    def __init__(self, C):
        super().__init__()
        self.ln = nn.LayerNorm(C)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

def _block(c_in: int, c_out: int, dilation: int) -> nn.Sequential:
    k = 7
    pad = (k - 1) // 2 * dilation
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=k, dilation=dilation, padding=pad),
        ChannelLayerNorm(c_out),
        nn.ReLU(),
    )



def make_cnn_v2(in_channels: int, W: int) -> nn.Module:
    """
    Dilated 1D-CNN for raw waveform input — designed so the receptive field
    covers a full 50 Hz cycle (192 samples) with margin.

    Stack of 4 conv layers, all kernel=7, dilations [1, 3, 9, 27], stride=1.
    Receptive field per output sample:
        RF = 1 + 6*1 + 6*3 + 6*9 + 6*27 = 241 samples (≈ 25 ms ≈ 1.25 cycles)

    Padding is set to (kernel-1)//2 * dilation so each layer preserves the
    temporal length, then a global average pool collapses time. 
    """

    class CNN1dDilated(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.BatchNorm1d(in_channels)
            self.conv = nn.Sequential(
                _block(in_channels, 64,  dilation=1),    # RF =   7
                _block(64,          128, dilation=3),    # RF =  25
                _block(128,         128, dilation=9),    # RF =  79
                _block(128,         128, dilation=27),   # RF = 241
            )
            self.pool = nn.AdaptiveAvgPool1d(1)          # (batch, 128, 1)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, N_ACTIONS),
            )

        def forward(self, x):
            # x: (batch, W, C) -> (batch, C, W) for BN1d / Conv1d
            x = x.permute(0, 2, 1)
            x = self.input_norm(x)
            x = self.conv(x)
            x = self.pool(x)
            return self.fc(x)

    return CNN1dDilated()

def make_combined(in_channels_raw: int, in_channels_phasor: int, W: int) -> nn.Module:
    """
    Two-branch network for 'combined' mode.

    Branch A: dilated 1D-CNN (RF=241) over the raw waveform channels.
    Branch B: dilated 1D-CNN (RF=241) over the phasor / impedance channels.
    Each branch produces a 128-dim pooled feature vector. The two are
    concatenated (→ 256) and passed through a 3-layer FC head to N_ACTIONS.

    Channel layout in the input tensor follows rl_protection.dataset (combined
    mode concatenates raw then phasor along the channel axis):

        x[:, :, :in_channels_raw]                                → raw signals
        x[:, :, in_channels_raw:in_channels_raw+in_channels_phasor] → phasor feats

    """

    class ChannelLayerNorm(nn.Module):
        """LayerNorm across channels per timestep for (batch, C, W) tensors."""
        def __init__(self, C):
            super().__init__()
            self.ln = nn.LayerNorm(C)

        def forward(self, x):
            return self.ln(x.transpose(1, 2)).transpose(1, 2)

    def _block(c_in: int, c_out: int, dilation: int) -> nn.Sequential:
        k = 7
        pad = (k - 1) // 2 * dilation
        return nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, dilation=dilation, padding=pad),
            ChannelLayerNorm(c_out),
            nn.ReLU(),
        )

    class DilatedBranch(nn.Module):
        """Dilated CNN trunk producing a 128-dim pooled feature vector."""
        out_dim = 128

        def __init__(self, in_channels: int):
            super().__init__()
            self.input_norm = nn.BatchNorm1d(in_channels)
            self.conv = nn.Sequential(
                _block(in_channels, 64,  dilation=1),    # RF =   7
                _block(64,          128, dilation=3),    # RF =  25
                _block(128,         128, dilation=9),    # RF =  79
                _block(128,         128, dilation=27),   # RF = 241
            )
            self.pool = nn.AdaptiveAvgPool1d(1)          # (B, 128, 1)

        def forward(self, x):
            # x: (B, W, C) -> (B, C, W)
            x = x.permute(0, 2, 1)
            x = self.input_norm(x)
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)                 # (B, 128)
            return x

    class CombinedTwoBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_raw    = in_channels_raw
            self.n_phasor = in_channels_phasor
            self.raw_branch    = DilatedBranch(in_channels_raw)
            self.phasor_branch = DilatedBranch(in_channels_phasor)
            feat_dim = self.raw_branch.out_dim + self.phasor_branch.out_dim  # 256
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Linear(128, 64),       nn.LayerNorm(64),  nn.ReLU(),
                nn.Linear(64, N_ACTIONS),
            )

        def forward(self, x):
            # x: (B, W, n_raw + n_phasor)
            raw_x = x[:, :, : self.n_raw]
            phs_x = x[:, :, self.n_raw : self.n_raw + self.n_phasor]
            f_raw = self.raw_branch(raw_x)
            f_phs = self.phasor_branch(phs_x)
            return self.fc(torch.cat([f_raw, f_phs], dim=1))

    return CombinedTwoBranch()


def build_model(mode: str, W: int) -> nn.Module:
    """
    Convenience function: return the right model for a given observation mode.

    mode: 'raw', 'raw_v2', 'phasor', or 'combined'
    W   : window size in steps
    """

    match mode:
        case "raw":
            return make_cnn(in_channels=N_LINE_COLS, W=W)
        case "raw_v2":
            return make_cnn_v2(in_channels=N_LINE_COLS, W=W)
        case "phasor":
            return make_cnn_v2(in_channels=N_PHASOR_COLS, W=W)
        case "combined":
            return make_combined(in_channels_raw=N_LINE_COLS,
                                 in_channels_phasor=N_PHASOR_COLS, W=W)
        case _:
            raise ValueError(
                f"{mode!r} not a recognised mode — use 'raw', 'raw_v2', 'phasor', or 'combined'."
            )

"""Causal multi-scale history encoder used by the planned F1/F2 models."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _LeftPaddedDepthwiseConv1d(nn.Module):
    """Depthwise Conv1d with explicit left-only padding and fixed output length."""

    def __init__(self, channels: int, *, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError(f"Expected [B,C,L], got {tuple(x.shape)}")
        output = self.conv(F.pad(x, (self.left_padding, 0)))
        if output.shape != x.shape:
            raise RuntimeError(
                "Causal depthwise convolution changed shape: "
                f"{tuple(x.shape)} -> {tuple(output.shape)}"
            )
        return output


class CausalMultiScaleHistoryBranch(nn.Module):
    """Two frozen-specification causal receptive fields over VSN history.

    Input and output both have shape ``[B, L, H]``.  The branch never receives
    decoder positions; causality is supplied by explicit left-only padding.
    """

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.short = _LeftPaddedDepthwiseConv1d(
            self.hidden_size, kernel_size=3, dilation=1
        )
        self.medium = _LeftPaddedDepthwiseConv1d(
            self.hidden_size, kernel_size=3, dilation=4
        )
        self.pointwise = nn.Conv1d(
            2 * self.hidden_size, self.hidden_size, kernel_size=1, bias=True
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, encoder_vsn: torch.Tensor) -> torch.Tensor:
        if encoder_vsn.ndim != 3 or encoder_vsn.shape[-1] != self.hidden_size:
            raise RuntimeError(
                "Expected encoder VSN [B,L,hidden_size], got "
                f"{tuple(encoder_vsn.shape)} for hidden_size={self.hidden_size}"
            )
        channels_first = encoder_vsn.transpose(1, 2)
        short = self.short(channels_first)
        medium = self.medium(channels_first)
        mixed = self.pointwise(torch.cat((short, medium), dim=1)).transpose(1, 2)
        output = self.norm(self.dropout(mixed) + encoder_vsn)
        if output.shape != encoder_vsn.shape:
            raise RuntimeError("History branch output shape contract failed")
        return output

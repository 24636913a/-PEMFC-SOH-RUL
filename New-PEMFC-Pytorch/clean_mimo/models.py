"""Small, deliberately plain models for Phase 1."""
from __future__ import annotations

import torch
from torch import nn


class MIMOLSTM(nn.Module):
    """One-layer causal LSTM followed by one linear H-dimensional head."""

    def __init__(self, input_size: int, horizon: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])

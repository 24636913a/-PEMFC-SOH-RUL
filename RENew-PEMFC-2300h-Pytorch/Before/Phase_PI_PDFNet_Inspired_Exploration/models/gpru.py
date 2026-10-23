"""Paper Eq. (9) gated periodic recurrent unit."""
from __future__ import annotations

import torch
from torch import nn


class GatedPeriodicRecurrentUnit(nn.Module):
    """GPRU with paper-explicit variable-specific `[N,D]` biases."""

    def __init__(self, variables: int, dimension: int) -> None:
        super().__init__()
        self.variables, self.dimension = variables, dimension
        self.W_r = nn.Parameter(torch.empty(2 * dimension, dimension))
        self.W_u = nn.Parameter(torch.empty(2 * dimension, dimension))
        self.W_o = nn.Parameter(torch.empty(2 * dimension, dimension))
        self.b_r = nn.Parameter(torch.zeros(variables, dimension))
        self.b_u = nn.Parameter(torch.zeros(variables, dimension))
        self.b_o = nn.Parameter(torch.zeros(variables, dimension))
        for weight in (self.W_r, self.W_u, self.W_o):
            nn.init.xavier_uniform_(weight)

    def forward(self, signal: torch.Tensor, previous_hidden: torch.Tensor) -> torch.Tensor:
        expected = (self.variables, self.dimension)
        if signal.ndim not in {2, 3} or tuple(signal.shape[-2:]) != expected or signal.shape != previous_hidden.shape:
            raise ValueError(f"GPRU expects signal/hidden [...,{expected[0]},{expected[1]}]")
        joined = torch.cat([signal, previous_hidden], dim=-1)
        reset = torch.sigmoid(joined @ self.W_r + self.b_r)
        update = torch.sigmoid(joined @ self.W_u + self.b_u)
        candidate = torch.tanh(torch.cat([signal, reset * previous_hidden], dim=-1) @ self.W_o + self.b_o)
        return update * previous_hidden + (1.0 - update) * candidate

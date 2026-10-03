"""Frozen Cycle-Aligned Latent Health State recurrence and observation map."""

from __future__ import annotations

import math

import torch
from torch import nn


class CHMModel(nn.Module):
    """Exact V2 architecture: Linear(d,16) -> SiLU -> Linear(16,1)."""

    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dimension, 16), nn.SiLU(), nn.Linear(16, 1))
        self.eta = nn.Parameter(torch.tensor(math.log(0.995 / 0.005), dtype=torch.float64))
        self.a_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.double()

    def latent_trajectory(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or len(inputs) < 1:
            raise ValueError("CHM inputs must be a non-empty [HOU, feature] tensor.")
        decrement = torch.exp(-1e-3 * torch.nn.functional.softplus(self.network(inputs).squeeze(-1)))
        # The first observed state is H_1.  For j >= 2 the exact frozen
        # recurrence is a cumulative product of decrements[1:].  Vectorizing
        # it removes Python-loop overhead without changing the model, loss,
        # chronology, or floating-point operation order within each product.
        factors = torch.cat((torch.ones(1, dtype=inputs.dtype, device=inputs.device), decrement[1:]))
        return torch.sigmoid(self.eta) * torch.cumprod(factors, dim=0)

    def observation_calibration(self, retention: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.a_raw) * retention + self.b

"""Equation-literal PDFNet Eq. (8) variable-axis attention."""
from __future__ import annotations

import math

import torch
from torch import nn


class VariableAxisAttention(nn.Module):
    """Use Eq. (8) literally: nodes/variables, not periods, are attended."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.W_Q = nn.Linear(dimension, dimension, bias=False)
        self.W_K = nn.Linear(dimension, dimension, bias=False)
        self.W_V = nn.Linear(dimension, dimension, bias=False)

    def project_value(self, values: torch.Tensor) -> torch.Tensor:
        return self.W_V(values)

    def forward(self, representation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if representation.ndim not in {2, 3} or representation.shape[-1] != self.dimension:
            raise ValueError("Expected [N,D] or independent-batch [B,N,D] branch representation")
        query, key, value = self.W_Q(representation), self.W_K(representation), self.W_V(representation)
        attention = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(self.dimension), dim=-1)
        return attention, value, attention @ value

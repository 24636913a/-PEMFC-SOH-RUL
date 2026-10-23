"""Preregistered Eq. (12)–(13) two-hop bidirectional graph propagation."""
from __future__ import annotations

import torch
from torch import nn


class BidirectionalMixHop(nn.Module):
    """The A/Aᵀ equal mix is explicitly an exploratory bridge."""

    def __init__(self, dimension: int, hop_depth: int = 2, beta: float = 0.5) -> None:
        super().__init__()
        if hop_depth != 2 or beta != 0.5:
            raise ValueError("Phase PI freezes hop_depth=2 and beta=.5")
        self.dimension, self.hop_depth, self.beta = dimension, hop_depth, beta
        self.weights = nn.Parameter(torch.empty(hop_depth + 1, dimension, dimension))
        nn.init.xavier_uniform_(self.weights)

    def _path(self, adjacency: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim not in {2, 3} or signal.ndim not in {2, 3}:
            raise ValueError("Mix-hop expects [N,N]/[N,D] or independent batched tensors")
        states = [signal]
        current = signal
        for hop in range(1, self.hop_depth + 1):
            current = self.beta * signal + (1.0 - self.beta) * (torch.matrix_power(adjacency, hop) @ current)
            states.append(current)
        return sum(state @ self.weights[index] for index, state in enumerate(states))

    def forward(self, adjacency: torch.Tensor, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direct = self._path(adjacency, signal)
        reverse = self._path(adjacency.transpose(-2, -1), signal)
        return (direct + reverse) / 2.0, direct, reverse

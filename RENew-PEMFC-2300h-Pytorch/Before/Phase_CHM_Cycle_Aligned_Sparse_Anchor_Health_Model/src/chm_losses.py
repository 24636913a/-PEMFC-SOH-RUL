"""Frozen mean-normalized CHM losses."""

from __future__ import annotations

from typing import Iterable

import torch
from torch.nn import functional as F


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denominator = weights.sum()
    if not torch.isfinite(denominator) or float(denominator.detach().cpu()) <= 0:
        raise ValueError("Loss normalization denominator invalid.")
    return (values * weights).sum() / denominator


def observation_loss(states: torch.Tensor, calibrated_retention: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return weighted_mean(F.huber_loss(states, calibrated_retention, delta=0.001, reduction="none"), weights)


def anchor_predictions(states: torch.Tensor, index_groups: Iterable[tuple[int, int, int]]) -> torch.Tensor:
    values = [torch.median(states[torch.tensor(indices, device=states.device)]) for indices in index_groups]
    return torch.stack(values)


def anchor_loss(predictions: torch.Tensor, targets: torch.Tensor, reliability_weights: torch.Tensor) -> torch.Tensor:
    return weighted_mean((predictions - targets).square(), reliability_weights)


def prepare_smoothness_design(triples: tuple[tuple[int, int, int, int, int, int], ...], complete_weights: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze legal triple indices and weights once for a complete fit."""
    indices: list[tuple[int, int, int]] = []
    weights: list[float] = []
    for left, middle, right, c_left, c_middle, c_right in triples:
        weight = min(complete_weights[c_left], complete_weights[c_middle], complete_weights[c_right])
        if weight > 0:
            indices.append((left, middle, right))
            weights.append(float(weight))
    if not indices:
        raise ValueError("INVALID_REPLICATE_SMOOTHNESS_WEIGHT_ZERO")
    return torch.tensor(indices, dtype=torch.long), torch.tensor(weights, dtype=torch.float64)


def smoothness_loss(states: torch.Tensor, design: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Mean weighted second differences for the already frozen legal triples."""
    indices, weights = design
    values = states[indices.to(device=states.device)]
    second_differences = (values[:, 2] - 2 * values[:, 1] + values[:, 0]).square()
    return weighted_mean(second_differences, weights.to(dtype=states.dtype, device=states.device))

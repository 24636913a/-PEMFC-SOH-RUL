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


def smoothness_loss(states: torch.Tensor, triples: tuple[tuple[int, int, int, int, int, int], ...], complete_weights: dict[int, int]) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    weights: list[float] = []
    for left, middle, right, c_left, c_middle, c_right in triples:
        weight = min(complete_weights[c_left], complete_weights[c_middle], complete_weights[c_right])
        if weight > 0:
            terms.append((states[right] - 2 * states[middle] + states[left]).square())
            weights.append(float(weight))
    if not terms:
        raise ValueError("INVALID_REPLICATE_SMOOTHNESS_WEIGHT_ZERO")
    stacked = torch.stack(terms)
    return weighted_mean(stacked, torch.tensor(weights, dtype=states.dtype, device=states.device))

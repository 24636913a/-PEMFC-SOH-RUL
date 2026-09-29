"""Embedding-domain period selection and masked right-padding utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PeriodSelection:
    periods: torch.Tensor
    frequencies: torch.Tensor
    amplitudes: torch.Tensor


def half_up_period(length: int, frequency: int) -> int:
    """Frozen integer form of paper Eq. (4): clip(floor(D/f + .5), 2, D)."""
    if length < 2 or frequency < 1:
        raise ValueError("period mapping requires length>=2 and positive frequency")
    return max(2, min(length, math.floor(length / frequency + 0.5)))


def select_distinct_periods(embedded: torch.Tensor, k: int = 2) -> PeriodSelection:
    """Per-sample, variable-mean rFFT selection with DC removal and deduplication."""
    if embedded.ndim != 3:
        raise ValueError(f"Expected [B,N,D], got {tuple(embedded.shape)}")
    batch, _variables, dimension = embedded.shape
    if k != 2 or dimension < 4:
        raise ValueError("Phase PI freezes k=2 and needs at least four embedding positions")
    amplitudes = torch.fft.rfft(embedded, dim=-1).abs().mean(dim=1)
    amplitudes = amplitudes.clone()
    amplitudes[:, 0] = -torch.inf  # never select the DC component
    ranked = torch.argsort(amplitudes, dim=-1, descending=True)
    periods, frequencies = [], []
    for b in range(batch):
        row_periods: list[int] = []
        row_frequencies: list[int] = []
        for candidate in ranked[b].detach().cpu().tolist():
            frequency = int(candidate)
            if frequency == 0:
                continue
            period = half_up_period(dimension, frequency)
            if period not in row_periods:
                row_periods.append(period)
                row_frequencies.append(frequency)
            if len(row_periods) == k:
                break
        if len(row_periods) != k:
            raise RuntimeError("Could not select two distinct non-DC integer periods")
        periods.append(row_periods)
        frequencies.append(row_frequencies)
    return PeriodSelection(
        periods=torch.tensor(periods, dtype=torch.long, device=embedded.device),
        frequencies=torch.tensor(frequencies, dtype=torch.long, device=embedded.device),
        amplitudes=amplitudes,
    )


def select_inner_patch_length(segment: torch.Tensor, valid_mask: torch.Tensor) -> tuple[int, int]:
    """Frozen bridge for Eq. (6): use p_outer/f_inner rather than D/f_inner."""
    if segment.ndim != 2 or valid_mask.ndim != 1 or segment.shape[-1] != valid_mask.numel():
        raise ValueError("inner period inputs must be [N,p] and [p]")
    outer_period = int(segment.shape[-1])
    if outer_period < 2:
        raise ValueError("outer period must be at least two")
    masked = segment * valid_mask.to(dtype=segment.dtype).unsqueeze(0)
    amplitude = torch.fft.rfft(masked, dim=-1).abs().mean(dim=0)
    amplitude = amplitude.clone()
    amplitude[0] = -torch.inf
    frequency = int(torch.argmax(amplitude).item())
    if frequency <= 0:
        raise RuntimeError("inner rFFT selected DC despite firewall")
    return half_up_period(outer_period, frequency), frequency


def right_pad_and_segment(values: torch.Tensor, valid_mask: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-zero pad `[N,L]` and return `[segments,N,length]`, `[segments,length]`."""
    if values.ndim != 2 or valid_mask.ndim != 1 or values.shape[-1] != valid_mask.numel():
        raise ValueError("values/mask shape mismatch")
    if length < 1:
        raise ValueError("segment length must be positive")
    original_length = int(values.shape[-1])
    segment_count = int(math.ceil(original_length / length))
    total = segment_count * length
    padding = total - original_length
    padded_values = F.pad(values, (0, padding))
    padded_mask = F.pad(valid_mask.to(dtype=torch.bool), (0, padding), value=False)
    segments = padded_values.reshape(values.shape[0], segment_count, length).permute(1, 0, 2).contiguous()
    masks = padded_mask.reshape(segment_count, length)
    return segments, masks


def right_pad_and_segment_batched(values: torch.Tensor, valid_mask: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent-batch equivalent of :func:`right_pad_and_segment`.

    ``valid_mask`` is shared because all members of a period group have the
    same source length.  No statistic is ever computed across the batch axis.
    """
    if values.ndim != 3 or valid_mask.ndim != 1 or values.shape[-1] != valid_mask.numel():
        raise ValueError("batched values/mask shape mismatch")
    if length < 1:
        raise ValueError("segment length must be positive")
    batch, variables, original_length = values.shape
    segment_count = int(math.ceil(original_length / length))
    total = segment_count * length
    padding = total - original_length
    padded_values = F.pad(values, (0, padding))
    padded_mask = F.pad(valid_mask.to(dtype=torch.bool), (0, padding), value=False)
    segments = padded_values.reshape(batch, variables, segment_count, length).permute(0, 2, 1, 3).contiguous()
    masks = padded_mask.reshape(segment_count, length)
    return segments, masks


def right_pad_and_segment_per_mask(values: torch.Tensor, valid_masks: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch segmentation when each independent sample has its own mask.

    This supports a final partial outer period without treating its padded
    positions as observations.  It is still purely elementwise along batches.
    """
    if values.ndim != 3 or valid_masks.ndim != 2 or tuple(valid_masks.shape) != (values.shape[0], values.shape[-1]):
        raise ValueError("per-mask batch values/masks shape mismatch")
    if length < 1:
        raise ValueError("segment length must be positive")
    batch, variables, original_length = values.shape
    segment_count = int(math.ceil(original_length / length))
    total = segment_count * length
    padding = total - original_length
    padded_values = F.pad(values, (0, padding))
    padded_masks = F.pad(valid_masks.to(dtype=torch.bool), (0, padding), value=False)
    segments = padded_values.reshape(batch, variables, segment_count, length).permute(0, 2, 1, 3).contiguous()
    masks = padded_masks.reshape(batch, segment_count, length)
    return segments, masks


def mask_positions(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set non-observed right-padding positions to zero after convolution."""
    if mask.ndim == 1:
        if values.shape[-1] != mask.numel():
            raise ValueError("mask length does not match temporal dimension")
        shaped = mask.to(dtype=values.dtype).reshape(*([1] * (values.ndim - 1)), -1)
    elif values.ndim == 3 and mask.ndim == 2 and tuple(mask.shape) == (values.shape[0], values.shape[-1]):
        shaped = mask.to(dtype=values.dtype).unsqueeze(1)
    else:
        raise ValueError("mask shape does not match values")
    return values * shaped

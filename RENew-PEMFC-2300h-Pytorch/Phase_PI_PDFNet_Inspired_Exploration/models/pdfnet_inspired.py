"""Preregistered PDFNet-inspired exploratory architecture.

This is intentionally not named or represented as an Official PDFNet
implementation.  Each ``CORE_EXPLORATORY_BRIDGE`` below closes an unspecified
paper connection and is frozen in the Phase PI assumptions YAML.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .embedding_periods import PeriodSelection, mask_positions, right_pad_and_segment_batched, right_pad_and_segment_per_mask, select_distinct_periods
from .gated_graph import GatedDynamicGraph
from .gpru import GatedPeriodicRecurrentUnit
from .mixhop import BidirectionalMixHop
from .variable_attention import VariableAxisAttention


class PDFNetInspiredExploratory(nn.Module):
    """Frozen Raw-F2 → all-variable absolute-voltage forecaster for Phase PI."""

    def __init__(self, dimension: int = 128, variables: int = 14, lookback: int = 48, horizon: int = 24) -> None:
        super().__init__()
        if dimension not in {128, 64}:
            raise ValueError("Only preregistered D=128 or pre-score resource fallback D=64 is allowed")
        if (variables, lookback, horizon) != (14, 48, 24):
            raise ValueError("Phase PI freezes 14 variables, 48 history steps, and 24 forecast steps")
        self.dimension, self.variables, self.lookback, self.horizon = dimension, variables, lookback, horizon
        self.embedding = nn.Sequential(nn.Linear(lookback, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.shared_conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
        self.variable_attention = VariableAxisAttention(dimension)
        self.gpru = GatedPeriodicRecurrentUnit(variables, dimension)
        self.dynamic_graph = GatedDynamicGraph(variables, dimension)
        self.mixhop = BidirectionalMixHop(dimension, hop_depth=2, beta=0.5)
        self.branch_summary = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension), nn.LayerNorm(dimension))
        # CORE_EXPLORATORY_BRIDGE: the only multi-period fusion parameters.
        self.fusion_logits = nn.Parameter(torch.zeros(2))
        self.forecast_head = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, horizon))

    def _conv(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Apply the same Conv1D to every variable, branch, segment, and patch."""
        if values.ndim == 2:
            if values.shape[0] != self.variables:
                raise ValueError("shared Conv1D expects [14,length]")
            convolved = self.shared_conv(values.unsqueeze(1)).squeeze(1)
        elif values.ndim == 3:
            batch, variables, length = values.shape
            if variables != self.variables:
                raise ValueError("shared Conv1D expects [B,14,length]")
            convolved = self.shared_conv(values.reshape(batch * variables, 1, length)).squeeze(1).reshape(batch, variables, length)
        else:
            raise ValueError("shared Conv1D expects [14,length] or [B,14,length]")
        return mask_positions(convolved, valid_mask)

    def _outer_attention_representation(self, outer_segments: torch.Tensor, outer_masks: torch.Tensor) -> torch.Tensor:
        """Eq. (7) across segments then Eq. (8)'s concat/truncate/valid-fill P_i."""
        batch, segment_count, variables, period = outer_segments.shape
        flat = outer_segments.reshape(batch * segment_count, variables, period)
        repeated_masks = outer_masks.unsqueeze(0).expand(batch, -1, -1).reshape(batch * segment_count, period)
        conv = self._conv(flat, repeated_masks).reshape(batch, segment_count, variables, period)
        representation = conv.permute(0, 2, 1, 3).reshape(batch, variables, segment_count * period)[..., : self.dimension]
        return representation

    @staticmethod
    def _inner_periods(segment: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample inner rFFT selection; this never aggregates batch spectra."""
        outer_period = segment.shape[-1]
        if valid_mask.ndim == 1:
            mask = valid_mask.to(segment.dtype).reshape(1, 1, -1)
        elif valid_mask.ndim == 2 and tuple(valid_mask.shape) == (segment.shape[0], segment.shape[-1]):
            mask = valid_mask.to(segment.dtype).unsqueeze(1)
        else:
            raise ValueError("inner valid mask shape mismatch")
        amplitude = torch.fft.rfft(segment * mask, dim=-1).abs().mean(dim=1)
        amplitude[:, 0] = -torch.inf
        frequency = torch.argmax(amplitude, dim=-1)
        if bool((frequency <= 0).any()):
            raise RuntimeError("Inner rFFT selected DC despite firewall")
        patch_length = torch.floor(outer_period / frequency.to(torch.float32) + .5).to(torch.long).clamp(2, outer_period)
        return patch_length, frequency

    def _branch_batched(self, embedded: torch.Tensor, outer_period: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """One period group with mathematically independent batch members.

        This is a performance-only batching of the frozen sample-local
        recurrence.  Groups share an outer period but never pool features,
        spectra, graphs, hidden states, losses, or summaries across samples.
        """
        batch = embedded.shape[0]
        original_mask = torch.ones(self.dimension, dtype=torch.bool, device=embedded.device)
        outer_segments, outer_masks = right_pad_and_segment_batched(embedded, original_mask, outer_period)
        representation = self._outer_attention_representation(outer_segments, outer_masks)
        attention, value, _attention_value = self.variable_attention(representation)
        initial_hidden = attention @ value  # Paper literal H^{j,0}=EV.
        # Outer periods have no recurrent connection.  Flattening [B,S] into a
        # work axis is therefore an exact parallelization, not a cross-sample
        # aggregation.  It removes thousands of tiny GPU launches at D=128.
        outer_count = outer_segments.shape[1]
        flat_segments = outer_segments.reshape(batch * outer_count, self.variables, outer_period)
        flat_masks = outer_masks.unsqueeze(0).expand(batch, -1, -1).reshape(batch * outer_count, outer_period)
        flat_attention = attention.unsqueeze(1).expand(-1, outer_count, -1, -1).reshape(batch * outer_count, self.variables, self.variables)
        flat_value = value.unsqueeze(1).expand(-1, outer_count, -1, -1).reshape(batch * outer_count, self.variables, self.dimension)
        flat_initial = initial_hidden.unsqueeze(1).expand(-1, outer_count, -1, -1).reshape(batch * outer_count, self.variables, self.dimension)
        inner_periods, frequencies = self._inner_periods(flat_segments, flat_masks)
        inter_flat = torch.zeros_like(flat_initial)
        first_graph_flat = torch.zeros(batch * outer_count, self.variables, self.variables, dtype=embedded.dtype, device=embedded.device)
        patch_count_flat = torch.zeros(batch * outer_count, dtype=torch.long, device=embedded.device)
        for inner_period in torch.unique(inner_periods, sorted=True).detach().cpu().tolist():
            indexes = torch.nonzero(inner_periods.eq(int(inner_period)), as_tuple=False).flatten()
            segment_group = flat_segments.index_select(0, indexes)
            mask_group = flat_masks.index_select(0, indexes)
            attention_group = flat_attention.index_select(0, indexes)
            value_group = flat_value.index_select(0, indexes)
            hidden = flat_initial.index_select(0, indexes)  # reset for every original outer segment
            patches, patch_masks = right_pad_and_segment_per_mask(segment_group, mask_group, int(inner_period))
            intra_sum = torch.zeros_like(hidden)
            intra_count = torch.zeros(len(indexes), 1, 1, dtype=embedded.dtype, device=embedded.device)
            for patch_index in range(patches.shape[1]):
                valid_local = torch.nonzero(patch_masks[:, patch_index].any(dim=-1), as_tuple=False).flatten()
                if not len(valid_local):
                    continue
                patch = patches[:, patch_index].index_select(0, valid_local)
                patch_mask = patch_masks[:, patch_index].index_select(0, valid_local)
                current_attention = attention_group.index_select(0, valid_local)
                current_value = value_group.index_select(0, valid_local)
                current_hidden = hidden.index_select(0, valid_local)
                patch_conv = self._conv(patch, patch_mask)
                # CORE_EXPLORATORY_BRIDGE: patch → Eq. (8) value space → GPRU.
                patch_pad = F.pad(patch_conv, (0, self.dimension - patch_conv.shape[-1]))
                signal = current_attention @ self.variable_attention.project_value(patch_pad)
                current_hidden = self.gpru(signal, current_hidden)
                hidden = hidden.index_copy(0, valid_local, current_hidden)
                adjacency, _tau0, _tau1 = self.dynamic_graph(current_attention, current_value, current_hidden)
                output, _direct, _reverse = self.mixhop(adjacency, signal)
                intra_sum = intra_sum.index_add(0, valid_local, output)
                intra_count = intra_count.index_add(0, valid_local, torch.ones(len(valid_local), 1, 1, dtype=embedded.dtype, device=embedded.device))
                if patch_index == 0:
                    first_graph_flat = first_graph_flat.index_copy(0, indexes.index_select(0, valid_local), adjacency)
            if bool((intra_count == 0).any()):
                raise RuntimeError("A valid outer period produced no valid inner patch")
            intra = intra_sum / intra_count  # exact validity-mask mean
            hidden = self.gpru(intra, hidden)  # Paper Eq. (14) period-level reuse.
            adjacency, _tau0, _tau1 = self.dynamic_graph(attention_group, value_group, hidden)
            inter, _direct, _reverse = self.mixhop(adjacency, intra)
            inter_flat = inter_flat.index_copy(0, indexes, inter)
            patch_count_flat = patch_count_flat.index_copy(0, indexes, intra_count[:, 0, 0].to(torch.long))
        if bool((patch_count_flat == 0).any()):
            raise RuntimeError("Period branch did not process every valid outer segment")
        branch = self.branch_summary(inter_flat.reshape(batch, outer_count, self.variables, self.dimension).mean(dim=1))
        patch_counts = patch_count_flat.reshape(batch, outer_count).detach().cpu().tolist()
        first_frequencies = frequencies.reshape(batch, outer_count)[:, 0]
        first_graph = first_graph_flat.reshape(batch, outer_count, self.variables, self.variables)[:, 0]
        return branch, {
            "outer_period": outer_period,
            "outer_segment_count": [outer_count] * batch,
            "inner_patch_counts": patch_counts,
            "inner_frequency_first_outer": first_frequencies,
            "first_dynamic_graph": first_graph,
        }

    def forward(self, x: torch.Tensor, return_audit: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.lookback, self.variables):
            raise ValueError(f"Expected [B,{self.lookback},{self.variables}], got {tuple(x.shape)}")
        embedded = self.embedding(x.transpose(1, 2))
        selection: PeriodSelection = select_distinct_periods(embedded, k=2)
        batch = x.shape[0]
        branch_tensors: list[torch.Tensor] = []
        graph_tensors: list[torch.Tensor] = []
        outer_segment_counts: list[list[int | None]] = [[None, None] for _ in range(batch)]
        patch_count_rows: list[list[list[int] | None]] = [[None, None] for _ in range(batch)]
        frequency_rows: list[list[int | None]] = [[None, None] for _ in range(batch)]
        for branch_index in range(2):
            branch_tensor = torch.zeros(batch, self.variables, self.dimension, dtype=embedded.dtype, device=embedded.device)
            graph_tensor = torch.zeros(batch, self.variables, self.variables, dtype=embedded.dtype, device=embedded.device)
            for outer_period in torch.unique(selection.periods[:, branch_index], sorted=True).detach().cpu().tolist():
                indexes = torch.nonzero(selection.periods[:, branch_index].eq(int(outer_period)), as_tuple=False).flatten()
                branch, details = self._branch_batched(embedded.index_select(0, indexes), int(outer_period))
                branch_tensor = branch_tensor.index_copy(0, indexes, branch)
                graph_tensor = graph_tensor.index_copy(0, indexes, details["first_dynamic_graph"])
                for local, global_index in enumerate(indexes.detach().cpu().tolist()):
                    outer_segment_counts[global_index][branch_index] = int(details["outer_segment_count"][local])
                    patch_count_rows[global_index][branch_index] = list(details["inner_patch_counts"][local])
                    frequency_rows[global_index][branch_index] = int(details["inner_frequency_first_outer"][local].item())
            branch_tensors.append(branch_tensor)
            graph_tensors.append(graph_tensor)
        branch_tensor = torch.stack(branch_tensors, dim=1)
        if return_audit and branch_tensor.requires_grad:
            branch_tensor.retain_grad()
        fusion_weights = torch.softmax(self.fusion_logits, dim=0)
        fused = (branch_tensor * fusion_weights.reshape(1, 2, 1, 1)).sum(dim=1)
        output = self.forecast_head(fused).transpose(1, 2)
        if tuple(output.shape[1:]) != (self.horizon, self.variables):
            raise RuntimeError("Forecast head violates frozen output contract")
        if not return_audit:
            return output
        return output, {
            "periods": selection.periods.detach(),
            "frequencies": selection.frequencies.detach(),
            "outer_segment_counts": outer_segment_counts,
            "inner_patch_counts": patch_count_rows,
            "inner_frequencies_first_outer": frequency_rows,
            "first_dynamic_graphs": torch.stack(graph_tensors, dim=1),
            "fusion_weights": fusion_weights.detach(),
            "branch_tensor": branch_tensor,
        }

"""Strict-history TFT adapters used only by Phase_TFT_Targeted_Reconsideration.

The implementation retains TFT's continuous variable-selection, gated residual,
local recurrent encoder/decoder, causal attention and gated output path.  The
decoder has no observed future inputs: it receives a fixed all-zero token.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE = SCRIPT_DIR.parent
XLSTM_SOURCE = PHASE / "sources" / "xlstm_official_v2_0_4"


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, 2 * output_size)
        self.skip = nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.fc2(self.dropout(F.elu(self.fc1(x))))
        value, gate = self.gate(transformed).chunk(2, dim=-1)
        return self.norm(self.skip(x) + value * torch.sigmoid(gate))


class ContinuousVariableSelection(nn.Module):
    """TFT-style selection over fourteen scalar continuous F2 variables."""
    def __init__(self, feature_count: int, embedding_size: int, hidden_size: int, grn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.embeddings = nn.ModuleList([nn.Linear(1, embedding_size) for _ in range(feature_count)])
        self.feature_grns = nn.ModuleList(
            [GatedResidualNetwork(embedding_size, grn_hidden, hidden_size, dropout) for _ in range(feature_count)]
        )
        self.weight_grn = GatedResidualNetwork(feature_count, grn_hidden, feature_count, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.feature_count:
            raise RuntimeError(f"Variable selection expected [B,T,{self.feature_count}], got {tuple(x.shape)}")
        candidates = [grn(embed(x[..., index:index + 1])) for index, (embed, grn) in enumerate(zip(self.embeddings, self.feature_grns))]
        stacked = torch.stack(candidates, dim=-2)
        weights = torch.softmax(self.weight_grn(x), dim=-1).unsqueeze(-1)
        return torch.sum(stacked * weights, dim=-2)


def _load_native_xlstm() -> tuple[Any, Any, Any, Any]:
    """Import the frozen official implementation without triggering CUDA compilation.

    Version 2.0.4 imports its CUDA extension helper at module import time even
    for the official pure-PyTorch ``backend='vanilla'`` sLSTM path.  The host
    has an NVIDIA runtime but no CUDA toolkit.  Temporarily reporting CUDA as
    unavailable only while importing skips that helper; it does not change the
    selected runtime device or the native cell equations.
    """
    if not XLSTM_SOURCE.is_dir():
        raise RuntimeError(f"Frozen NX-AI source checkout is absent: {XLSTM_SOURCE}")
    if str(XLSTM_SOURCE) not in sys.path:
        sys.path.insert(0, str(XLSTM_SOURCE))
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
    try:
        from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig
        from xlstm.blocks.slstm.layer import sLSTMLayer, sLSTMLayerConfig
    finally:
        torch.cuda.is_available = original  # type: ignore[method-assign]
    return sLSTMLayer, sLSTMLayerConfig, mLSTMLayer, mLSTMLayerConfig


class NativeRecurrentPath(nn.Module):
    """One TFT local recurrent encoder/decoder with a genuine state handoff."""
    def __init__(self, kind: str, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.kind = kind
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        if kind == "LSTM":
            self.layer = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        elif kind == "sLSTM":
            s_layer, s_config, _, _ = _load_native_xlstm()
            self.layer = s_layer(
                s_config(
                    embedding_dim=hidden_size,
                    num_heads=4,
                    conv1d_kernel_size=0,
                    dropout=dropout,
                    backend="vanilla",
                    dtype="float32",
                    enable_automatic_mixed_precision=False,
                )
            )
        elif kind == "mLSTM":
            _, _, m_layer, m_config = _load_native_xlstm()
            self.layer = m_layer(
                m_config(
                    embedding_dim=hidden_size,
                    num_heads=4,
                    conv1d_kernel_size=1,
                    qkv_proj_blocksize=4,
                    proj_factor=1.0,
                    context_length=72,
                    dropout=dropout,
                )
            )
        else:
            raise RuntimeError(f"Unsupported recurrent replacement: {kind}")

    @staticmethod
    def _require_token_shape(x: torch.Tensor, hidden_size: int) -> None:
        if x.ndim != 3 or x.shape[-1] != hidden_size:
            raise RuntimeError(f"Recurrent path expected [B,T,{hidden_size}], got {tuple(x.shape)}")

    def _mlstm_run(self, tokens: torch.Tensor, state: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        output: list[torch.Tensor] = []
        current = {} if state is None else state
        for step in range(tokens.shape[1]):
            # The frozen official CausalConv1d.step updates its provided state
            # buffer in place. Clone each carried native state before the next
            # official step so autograd retains the preceding graph version.
            # This preserves the actual native values and state handoff rather
            # than replacing them with an output-derived surrogate.
            step_state = {
                key: tuple(value.clone() for value in values)
                for key, values in current.items()
            }
            value, current = self.layer.step(tokens[:, step:step + 1], **step_state)
            output.append(value)
        return torch.cat(output, dim=1), current

    def forward(self, encoder_tokens: torch.Tensor, decoder_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        self._require_token_shape(encoder_tokens, self.hidden_size)
        self._require_token_shape(decoder_tokens, self.hidden_size)
        if self.kind == "LSTM":
            encoder_output, native_state = self.layer(encoder_tokens)
            decoder_output, _ = self.layer(decoder_tokens, native_state)
            return self.dropout(encoder_output), self.dropout(decoder_output), "torch.nn.LSTM_(h,c)_handoff"
        if self.kind == "sLSTM":
            encoder_output, native_state = self.layer(encoder_tokens, return_last_state=True)
            decoder_output, _ = self.layer(
                decoder_tokens,
                conv_state=native_state["conv_state"],
                slstm_state=native_state["slstm_state"],
                return_last_state=True,
            )
            return self.dropout(encoder_output), self.dropout(decoder_output), "NX-AI_sLSTMLayer_(conv_state,slstm_state)_handoff"
        encoder_output, native_state = self._mlstm_run(encoder_tokens)
        decoder_output, _ = self._mlstm_run(decoder_tokens, native_state)
        return self.dropout(encoder_output), self.dropout(decoder_output), "NX-AI_mLSTMLayer_(conv_state,mlstm_state)_handoff"


class StrictHistoryTFT(nn.Module):
    """A compact TFT adaptation with no future covariate interface."""
    def __init__(
        self,
        recurrent_kind: str,
        feature_count: int = 14,
        hidden_size: int = 32,
        embedding_size: int = 16,
        grn_hidden: int = 32,
        attention_heads: int = 4,
        dropout: float = 0.1,
        forecast_steps: int = 24,
    ) -> None:
        super().__init__()
        if hidden_size % attention_heads:
            raise RuntimeError("hidden_size must be divisible by attention heads")
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.forecast_steps = forecast_steps
        self.recurrent_kind = recurrent_kind
        self.variable_selection = ContinuousVariableSelection(feature_count, embedding_size, hidden_size, grn_hidden, dropout)
        self.recurrent = NativeRecurrentPath(recurrent_kind, hidden_size, dropout)
        self.post_recurrent_gate = GatedResidualNetwork(hidden_size, grn_hidden, hidden_size, dropout)
        self.attention = nn.MultiheadAttention(hidden_size, attention_heads, dropout=dropout, batch_first=True)
        self.post_attention_gate = GatedResidualNetwork(hidden_size, grn_hidden, hidden_size, dropout)
        self.positionwise_grn = GatedResidualNetwork(hidden_size, grn_hidden, hidden_size, dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

    def decoder_zero_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.forecast_steps, self.hidden_size, device=device, dtype=dtype)

    def causal_attention_mask(self, device: torch.device) -> torch.Tensor:
        total = 48 + self.forecast_steps
        return torch.triu(torch.ones(total, total, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (48, self.feature_count):
            raise RuntimeError(f"Strict-history TFT expected [B,48,{self.feature_count}], got {tuple(x.shape)}")
        encoder_selected = self.variable_selection(x)
        decoder_tokens = self.decoder_zero_tokens(x.shape[0], x.device, x.dtype)
        encoder_recurrent, decoder_recurrent, _ = self.recurrent(encoder_selected, decoder_tokens)
        temporal = torch.cat([encoder_recurrent + encoder_selected, decoder_recurrent], dim=1)
        temporal = self.post_recurrent_gate(temporal)
        attended, _ = self.attention(temporal, temporal, temporal, attn_mask=self.causal_attention_mask(x.device), need_weights=False)
        temporal = self.post_attention_gate(temporal + attended)
        temporal = self.positionwise_grn(temporal)
        absolute_scaled = self.output_layer(temporal[:, -self.forecast_steps:]).squeeze(-1)
        if absolute_scaled.shape != (x.shape[0], self.forecast_steps):
            raise RuntimeError("Absolute standardized voltage head contract failed")
        return absolute_scaled

    def native_state_contract(self) -> str:
        if self.recurrent_kind == "LSTM":
            return "torch.nn.LSTM_(h,c)_handoff"
        if self.recurrent_kind == "sLSTM":
            return "NX-AI_sLSTMLayer_(conv_state,slstm_state)_handoff"
        return "NX-AI_mLSTMLayer_(conv_state,mlstm_state)_handoff"


def create_model(kind: str, hidden_size: int) -> StrictHistoryTFT:
    return StrictHistoryTFT(recurrent_kind=kind, hidden_size=hidden_size)


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))

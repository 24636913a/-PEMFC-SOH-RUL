"""Small, explicit model definitions for the health-anchor pilot.

The TimeXer class wraps the project's vendored TimeXer implementation in a
history-only setting: all exogenous channels are historical measurements and
the endogenous voltage channel remains the last channel required by TimeXer.
It deliberately receives no future covariates.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class LSTMDynamicEncoder(nn.Module):
    """One-layer LSTM encoder pre-trained on a short dynamic-voltage task."""

    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dynamic_head = nn.Linear(hidden_size, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic_head(self.encode(x)).squeeze(-1)


def _load_timexer_module(vendor_root: Path):
    """Load TimeXer without altering the historical implementation."""

    root = str(vendor_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    module_path = vendor_root / "models" / "TimeXer.py"
    spec = importlib.util.spec_from_file_location("health_anchor_vendor_timexer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import vendored TimeXer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimeXerHistoryEncoder(nn.Module):
    """TimeXer encoder with a scalar dynamic-task head.

    This is not the original long-horizon TimeXer prediction head.  The
    architecture reuses its patch embedding and self/cross-attention encoder,
    then attaches a scalar head so the frozen encoder can support sparse
    health-anchor calibration.  The distinction is reported by the runner.
    """

    def __init__(self, vendor_root: Path, input_size: int, seq_len: int = 48) -> None:
        super().__init__()
        module = _load_timexer_module(vendor_root)
        cfg = SimpleNamespace(
            task_name="long_term_forecast", features="MS", seq_len=seq_len,
            pred_len=1, use_norm=False, patch_len=8, enc_in=input_size,
            c_out=1, d_model=32, n_heads=4, e_layers=1, d_ff=64,
            dropout=0.05, factor=3, embed="fixed", freq="h", activation="gelu",
        )
        self.vendor_model = module.Model(cfg)
        self.dynamic_head = nn.Linear(cfg.d_model, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # TimeXer MS mode requires endogenous channel last.  The runner makes
        # voltage the first selected input, hence this role-only reorder.
        ordered = torch.cat([x[:, :, 1:], x[:, :, :1]], dim=2)
        endogenous = ordered[:, :, -1].unsqueeze(-1).permute(0, 2, 1)
        patch_tokens, n_vars = self.vendor_model.en_embedding(endogenous)
        exogenous_tokens = self.vendor_model.ex_embedding(ordered[:, :, :-1], None)
        encoded = self.vendor_model.encoder(patch_tokens, exogenous_tokens)
        encoded = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1])
        # The final token is TimeXer's global token after cross-attention.
        return encoded[:, 0, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic_head(self.encode(x)).squeeze(-1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

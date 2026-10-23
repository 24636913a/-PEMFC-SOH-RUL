"""Local history-only dynamic encoders for Phase C.

Both encoders are used only on historical causal load-cycle features.  They
never receive a future operating condition or a future polarization curve.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class LSTMDynamicEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dynamic_head = nn.Linear(hidden_size, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic_head(self.encode(x)).squeeze(-1)


def _load_vendor(vendor_root: Path):
    root = str(vendor_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    source = vendor_root / "models" / "TimeXer.py"
    spec = importlib.util.spec_from_file_location("phase_c_vendor_timexer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import vendored TimeXer: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimeXerHistoryEncoder(nn.Module):
    """Vendored TimeXer patch/cross-attention encoder with a scalar task head.

    This is a documented history-only adaptation: TimeXer's original forecast
    head is not used, and all exogenous channels are historical observations.
    """

    def __init__(self, vendor_root: Path, input_size: int, seq_len: int = 48) -> None:
        super().__init__()
        module = _load_vendor(vendor_root)
        cfg = SimpleNamespace(
            task_name="long_term_forecast", features="MS", seq_len=seq_len,
            pred_len=1, use_norm=False, patch_len=8, enc_in=input_size,
            c_out=1, d_model=32, n_heads=4, e_layers=1, d_ff=64,
            dropout=0.05, factor=3, embed="fixed", freq="h", activation="gelu",
        )
        self.vendor_model = module.Model(cfg)
        self.dynamic_head = nn.Linear(cfg.d_model, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # The dynamic table stores voltage first; MS TimeXer requires it last.
        ordered = torch.cat([x[:, :, 1:], x[:, :, :1]], dim=2)
        endogenous = ordered[:, :, -1].unsqueeze(-1).permute(0, 2, 1)
        patch_tokens, n_vars = self.vendor_model.en_embedding(endogenous)
        historical_exogenous = self.vendor_model.ex_embedding(ordered[:, :, :-1], None)
        encoded = self.vendor_model.encoder(patch_tokens, historical_exogenous)
        encoded = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1])
        return encoded[:, 0, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic_head(self.encode(x)).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

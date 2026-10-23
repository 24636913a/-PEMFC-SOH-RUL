"""F2 regime-aware fusion, deliberately limited to the F1 fusion operation."""
from __future__ import annotations

import torch
from torch import nn

# Importing the parent first installs the frozen baseline-model path used by
# this project-local extension; direct module imports then remain reliable.
from .dualpath_strict_history_tft import DualPathStrictHistoryTFT
from canonical_tft import RAW_NAMES
from .frozen_feature_schema import RAW_F2_FEATURE_NAMES


REGIME_FEATURE_NAMES = (
    "load_transition_count",
    "zero_current_fraction",
    "actual_current_density_A_cm2_std",
    "actual_current_A_std",
    "actual_current_density_A_cm2_mean",
)
class RegimeFusionGate(nn.Module):
    """The frozen 5→8→1 historical-regime gate."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(5, 8)
        self.activation = nn.ELU()
        self.output = nn.Linear(8, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, regime_context: torch.Tensor) -> torch.Tensor:
        if regime_context.ndim != 2 or regime_context.shape[1] != 5:
            raise RuntimeError(f"Expected [B,5] regime context, got {tuple(regime_context.shape)}")
        return torch.sigmoid(self.output(self.activation(self.hidden(regime_context))))


class RegimeAwareDualPathTFT(DualPathStrictHistoryTFT):
    """F2: F1 with only its fixed encoder fusion replaced by a regime gate."""

    regime_feature_names = REGIME_FEATURE_NAMES

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.dual_path_enabled:
            raise RuntimeError("F2 requires the enabled F1 convolutional branch")
        if tuple(RAW_NAMES) != tuple(f"raw_f2_{i:02d}" for i in range(14)):
            raise RuntimeError("Frozen Raw F2 schema identity mismatch")
        self.regime_indices = tuple(RAW_F2_FEATURE_NAMES.index(name) for name in REGIME_FEATURE_NAMES)
        self.regime_gate = RegimeFusionGate()

    def regime_context(self, encoder_cont: torch.Tensor) -> torch.Tensor:
        if encoder_cont.ndim != 3 or encoder_cont.shape[-1] < 14:
            raise RuntimeError("Expected scaled/masked encoder continuous tensor with 14 Raw F2 slots")
        index = torch.as_tensor(self.regime_indices, dtype=torch.long, device=encoder_cont.device)
        return encoder_cont.index_select(dim=-1, index=index).mean(dim=1)

    def regime_weight(self, encoder_cont: torch.Tensor) -> torch.Tensor:
        return self.regime_gate(self.regime_context(encoder_cont))

    def fuse_encoder(
        self,
        encoder_lstm: torch.Tensor,
        encoder_conv: torch.Tensor,
        encoder_cont: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.regime_weight(encoder_cont).view(-1, 1, 1)
        return gate * encoder_lstm + (1.0 - gate) * encoder_conv

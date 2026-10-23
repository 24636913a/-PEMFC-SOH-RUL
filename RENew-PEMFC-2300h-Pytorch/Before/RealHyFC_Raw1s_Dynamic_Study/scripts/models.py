"""Read-only native F2 adapter for direct raw-second DeltaV forecasting.

The full historical TFT/F2 model classes are compiled verbatim from their
audited source, without executing the historical module's imports, training,
data, checkpoint, or output code.  Only the input schema and temporal query
dictionary are adapted.  Loss and target scaling belong to the shared runner.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# In particular, do not create __pycache__ entries in historical directories.
sys.dont_write_bytecode = True

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer


STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parent
NATIVE_SOURCE = PROJECT / "RealHyFC_F2_Main.py"
NATIVE_SOURCE_SHA256 = "64cde85cac939806641921083bcf2c2e5354ee2bcd19dc4315949459305660f9"
NATIVE_CLASS_NAMES = (
    "CanonicalStrictHistoryTFT",
    "_LeftPaddedDepthwiseConv1d",
    "CausalMultiScaleHistoryBranch",
    "DualPathStrictHistoryTFT",
    "RegimeFusionGate",
    "RegimeAwareDualPathTFT",
)
MODEL_PARAMETERS = {
    "hidden_size": 64,
    "hidden_continuous_size": 32,
    "attention_head_size": 4,
    "lstm_layers": 1,
    "dropout": 0.1,
}
FEATURE_NAMES = (
    "mean10", "std10", "range10", "min10", "max10", "I_act",
    "I_act_trailing30s_std", "I_act_trailing30s_ptp", "j_act",
    "j_act_trailing30s_std", "j_act_trailing30s_ptp",
    "I_change_past1s", "operation_point_change_past1s", "zero_current_flag",
    "I_set", "I_set_internal", "T_Si_A", "T_Si_C", "T_Si_CL", "T_So_CL",
    "p_Si_A", "p_Si_C", "operation_point", "OpHrs2",
)
REGIME_INDICES = (11, 13, 9, 6, 8)


@lru_cache(maxsize=None)
def _native_classes(n_features: int) -> SimpleNamespace:
    """Load only original class definitions into an isolated, read-only namespace."""
    source_bytes = NATIVE_SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != NATIVE_SOURCE_SHA256:
        raise RuntimeError("Historical native F2 source SHA256 differs from audited authority")
    tree = ast.parse(source_bytes.decode("utf-8-sig"), filename=str(NATIVE_SOURCE))
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name in NATIVE_CLASS_NAMES]
    if tuple(node.name for node in selected) != NATIVE_CLASS_NAMES:
        raise RuntimeError("Historical native F2 class inventory mismatch")
    literal_names = {"FEATURES", "REGIME_FEATURE_NAMES"}
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in literal_names:
                    literals[target.id] = ast.literal_eval(node.value)
    original_features = tuple(literals["FEATURES"])
    original_regime_names = tuple(literals["REGIME_FEATURE_NAMES"])
    native_indices = tuple(original_features.index(name) for name in original_regime_names)
    if len(original_features) != 14 or native_indices != REGIME_INDICES:
        raise RuntimeError("Historical regime semantics no longer match the raw-second schema")
    # RAW_NAMES remains the original 14-slot identity used by F2's assertion;
    # X_REALS describes all input slots seen by the unchanged TFT constructor.
    namespace = {
        "__name__": "raw1s_readonly_native_f2",
        "__file__": str(NATIVE_SOURCE),
        "Any": Any, "torch": torch, "nn": nn, "F": F,
        "TemporalFusionTransformer": TemporalFusionTransformer,
        "RAW_NAMES": [f"raw_f2_{i:02d}" for i in range(14)],
        "RELATIVE_STEP": "relative_forecast_step",
        "X_REALS": [*[f"raw_f2_{i:02d}" for i in range(n_features)],
                    "relative_forecast_step"],
        "RAW_F2_FEATURE_NAMES": original_features,
        "REGIME_FEATURE_NAMES": original_regime_names,
    }
    executable = ast.Module(body=selected, type_ignores=[])
    exec(compile(executable, str(NATIVE_SOURCE), "exec"), namespace)
    # The source class implementations above are entirely unmodified.
    return SimpleNamespace(**{name: namespace[name] for name in NATIVE_CLASS_NAMES})


def _validate_dimensions(n_features: int, history: int, horizon: int) -> None:
    if any(not isinstance(value, int) or value <= 0
           for value in (n_features, history, horizon)):
        raise ValueError("n_features, history and horizon must be positive integers")


class Raw1sNativeF2(nn.Module):
    """Full native TFT + causal CNN + unchanged historical regime fusion."""

    def __init__(self, n_features: int, history: int, horizon: int) -> None:
        super().__init__()
        _validate_dimensions(n_features, history, horizon)
        if n_features != len(FEATURE_NAMES):
            raise ValueError("Frozen raw-second F2 protocol requires the documented 24-feature order")
        self.n_features = n_features
        self.history = history
        self.horizon = horizon
        original = _native_classes(n_features)
        self.native = original.RegimeAwareDualPathTFT(**MODEL_PARAMETERS)
        # This hparam is constructor metadata, not a learned positional tensor.
        # Native forward obtains all actual lengths from the input dictionary.
        self.native.hparams.max_encoder_length = history
        if tuple(self.native.regime_indices) != REGIME_INDICES:
            raise RuntimeError("Native regime indices changed")

    def to(self, *args: Any, **kwargs: Any) -> Raw1sNativeF2:
        """Keep Lightning's cached native device/dtype aligned with its tensors."""
        super().to(*args, **kwargs)
        # A plain nn.Module parent recursively applies tensor conversions but
        # does not call a Lightning child's .to(), which also updates the
        # child's cached .device/.dtype used by the untouched native forward.
        self.native.to(*args, **kwargs)
        return self

    def cuda(self, device: Any = None) -> Raw1sNativeF2:
        super().cuda(device=device)
        self.native.to(device=next(self.native.parameters()).device)
        return self

    def cpu(self) -> Raw1sNativeF2:
        super().cpu()
        self.native.to(device=torch.device("cpu"))
        return self

    def build_input(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.history, self.n_features):
            raise ValueError(f"Expected [B,{self.history},{self.n_features}], got {tuple(x.shape)}")
        batch, device, dtype = x.shape[0], x.device, x.dtype
        relative_history = torch.arange(
            1 - self.history, 1, device=device, dtype=dtype,
        ).view(1, self.history, 1).expand(batch, -1, -1) / float(self.horizon)
        relative_future = torch.arange(
            1, self.horizon + 1, device=device, dtype=dtype,
        ).view(1, self.horizon, 1).expand(batch, -1, -1) / float(self.horizon)
        return {
            "encoder_cont": torch.cat((x, relative_history), dim=-1),
            "decoder_cont": torch.cat((
                torch.zeros((batch, self.horizon, self.n_features), device=device, dtype=dtype),
                relative_future,
            ), dim=-1),
            "encoder_cat": torch.empty((batch, self.history, 0), device=device, dtype=torch.long),
            "decoder_cat": torch.empty((batch, self.horizon, 0), device=device, dtype=torch.long),
            "encoder_lengths": torch.full((batch,), self.history, device=device, dtype=torch.long),
            "decoder_lengths": torch.full((batch,), self.horizon, device=device, dtype=torch.long),
            "target_scale": torch.zeros((batch, 2), device=device, dtype=dtype),
        }

    def forward_with_details(self, x: torch.Tensor) -> Any:
        return self.native(self.build_input(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.forward_with_details(x)["prediction"].squeeze(-1)
        if prediction.shape != (x.shape[0], self.horizon):
            raise RuntimeError("Raw-second native F2 direct multi-step shape mismatch")
        return prediction


class Raw1sLSTM(nn.Module):
    """Basic direct LSTM: the native F2 LSTM width/layer count, common inputs."""

    def __init__(self, n_features: int, history: int, horizon: int) -> None:
        super().__init__()
        _validate_dimensions(n_features, history, horizon)
        self.n_features, self.history, self.horizon = n_features, history, horizon
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=MODEL_PARAMETERS["hidden_size"],
            num_layers=MODEL_PARAMETERS["lstm_layers"], batch_first=True,
            dropout=0.0,  # As in the historical fair baseline, one layer has no inter-layer dropout.
        )
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_PARAMETERS["hidden_size"]),
            nn.Linear(MODEL_PARAMETERS["hidden_size"], horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.history, self.n_features):
            raise ValueError(f"Expected [B,{self.history},{self.n_features}], got {tuple(x.shape)}")
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


def create_model(kind: str, n_features: int, history: int, horizon: int) -> nn.Module:
    """Return an untrained CPU model; caller sets Seed202 before construction."""
    constructors = {"LSTM": Raw1sLSTM, "F2": Raw1sNativeF2}
    try:
        constructor = constructors[kind.upper()]
    except (KeyError, AttributeError) as error:
        raise ValueError("kind must be LSTM or F2") from error
    return constructor(n_features, history, horizon)


def source_identity() -> dict[str, Any]:
    """Provenance without model training, historical imports or output writes."""
    content = NATIVE_SOURCE.read_bytes()
    tree = ast.parse(content.decode("utf-8-sig"))
    classes = {
        node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        for node in tree.body if isinstance(node, ast.ClassDef)
        and node.name in NATIVE_CLASS_NAMES
    }
    return {
        "native_source": str(NATIVE_SOURCE.relative_to(PROJECT.parent)),
        "native_source_sha256": hashlib.sha256(content).hexdigest(),
        "native_class_ast_sha256": classes,
        "native_source_unchanged": hashlib.sha256(content).hexdigest() == NATIVE_SOURCE_SHA256,
        "constructor_parameters": dict(MODEL_PARAMETERS),
        "feature_order": list(FEATURE_NAMES),
        "regime_indices": list(REGIME_INDICES),
        "regime_context": "mean over encoder history of standardized channels [11,13,9,6,8]",
        "regime_gate": "unchanged Linear(5,8) -> ELU -> Linear(8,1) -> sigmoid",
        "target": "direct per-future-second standardized DeltaV; external shared MSE",
        "historical_module_execution": False,
        "native_class_body_modification": False,
        "adapter_changes": [
            "24 real encoder channels instead of historical 14 aggregated channels",
            "history length from caller; frozen first experiment is 300 true seconds",
            "direct decoder query length from caller; 30 then 120 true seconds",
            "only known relative forecast step enters decoder; future real values are zero placeholders",
            "relative coordinates scaled by horizon, matching historical relative-step convention",
            "first 14 feature semantics are explicitly redefined causally at raw-second resolution",
            "no Top10 masking and no EWMA; identical shared raw-second inputs to both models",
        ],
        "preserved_modules": [
            "encoder/decoder variable-selection networks", "native encoder and decoder LSTM",
            "causal depthwise CNN k3/d1 and k3/d4 plus pointwise mixing and residual LayerNorm",
            "native post-LSTM gates and add-norm", "native 5-to-8-to-1 regime fusion gate",
            "static context and static enrichment", "interpretable multi-head causal attention",
            "position-wise GRN and pre-output gate norm", "native scalar decoder output layer",
        ],
        "lstm_baseline": {
            "hidden_size": 64, "lstm_layers": 1, "inter_layer_dropout": 0.0,
            "head": "LayerNorm(64) -> Linear(64,horizon), same as historical fair baseline",
            "parameter_matching_claim": False,
            "fairness": "same width/layers, inputs, Train-only scaler, origins, target and shared training protocol",
        },
    }

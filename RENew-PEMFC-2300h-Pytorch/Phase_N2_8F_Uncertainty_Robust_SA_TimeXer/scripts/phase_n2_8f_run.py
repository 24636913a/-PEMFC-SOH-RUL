"""Development-only Phase N2.8F uncertainty-robust SA-TimeXer experiment.

The implementation is deliberately self-contained and fail-closed.  It never
loads a Test feature/target row into a tensor or metric.  Most importantly, an
OOF pseudo-future prediction may train only on rows whose 24-step label was
already complete at the earliest predicted origin of its block.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8F_Uncertainty_Robust_SA_TimeXer")
CONFIG = ROOT / "configs" / "uncertainty_robust_sa_timexer_frozen_config.yaml"
FOLDS = ("fold_1", "fold_2", "fold_3")
SEED = 42
EPS = 1e-7


def out(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path.parent != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise RuntimeError(f"Output escaped Phase N2.8F root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    frame.to_csv(out(relative), index=False, encoding="utf-8-sig", lineterminator="\n")


def write_text(text: str, relative: str) -> None:
    out(relative).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(obj: Any, relative: str) -> None:
    out(relative).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_yaml(obj: dict[str, Any], relative: str) -> None:
    out(relative).write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    cols = [c for c in columns if c in frame.columns]
    if frame.empty or not cols:
        return "| status |\n|---|\n| no records |"
    tab = frame.loc[:, cols].copy()
    for col in tab.columns:
        if pd.api.types.is_float_dtype(tab[col]):
            tab[col] = tab[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    lines = ["| " + " | ".join(tab.columns) + " |", "|" + "|".join("---" for _ in tab.columns) + "|"]
    lines += ["| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in tab.itertuples(index=False, name=None)]
    return "\n".join(lines)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if Path(cfg["phase_root"]).resolve() != ROOT.resolve():
        raise RuntimeError("Configuration phase_root does not match this script")
    task = cfg["task"]
    if (task["resolution_hours"], task["lookback_steps"], task["horizon_steps"]) != (0.5, 48, 24):
        raise RuntimeError("Frozen task contract changed")
    if len(cfg["features"]) != 14 or len(cfg["future_exogenous_features"]) != 9:
        raise RuntimeError("Frozen F2/future-exogenous feature contract changed")
    if any("voltage" in name.lower() for name in cfg["future_exogenous_features"]):
        raise RuntimeError("Future voltage is prohibited")
    return cfg


def common_module(cfg: dict[str, Any]):
    spec = importlib.util.spec_from_file_location("n28f_common", Path(cfg["common_script"]))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen N2.5 common helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_development(cfg: dict[str, Any]):
    common = common_module(cfg)
    frame, audit = common.load_development_data(cfg, cfg["features"])
    if audit.get("target_rows_read_from_test") != 0 or frame["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL: a Test row entered the development frame")
    windows = {fold: (tr, va) for fold, tr, va in common.fold_windows(frame, cfg)}
    for fold, (_, va) in windows.items():
        if va.target_end_age_h.max() >= float(cfg["split"]["development_end_h"]) - EPS:
            raise RuntimeError(f"TEST FIREWALL: {fold} target reaches development/Test boundary")
    audit.update({
        "target_rows_read_from_test": 0,
        "test_feature_rows_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "test_feature_diagnostics_computed": False,
        "test_performance_accessed": False,
        "enforcement": "fail_closed_development_prefix_only",
    })
    return common, frame, windows, audit


def future_exogenous(frame: pd.DataFrame, windows, cfg: dict[str, Any]) -> np.ndarray:
    h = int(cfg["task"]["horizon_steps"])
    names = cfg["future_exogenous_features"]
    rows = []
    for origin in windows.row_origin_index.astype(int):
        block = frame.iloc[origin + 1: origin + 1 + h].loc[:, names].to_numpy(float)
        if block.shape != (h, len(names)):
            raise RuntimeError("Incomplete future-exogenous source block")
        rows.append(block)
    return np.asarray(rows, dtype=float)


class FutureScaler:
    """Featurewise training-history imputer/scaler; never fit to validation."""

    def __init__(self):
        self.median: np.ndarray | None = None
        self.scaler = StandardScaler()

    def fit(self, history: np.ndarray):
        flat = history.reshape(-1, history.shape[-1])
        self.median = np.nanmedian(flat, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        self.scaler.fit(np.where(np.isfinite(flat), flat, self.median))
        return self

    def fill(self, values: np.ndarray) -> np.ndarray:
        if self.median is None:
            raise RuntimeError("FutureScaler used before fit")
        shape = (1,) * (values.ndim - 1) + (len(self.median),)
        return np.where(np.isfinite(values), values, self.median.reshape(shape))

    def transform(self, values: np.ndarray) -> np.ndarray:
        clean = self.fill(values)
        return self.scaler.transform(clean.reshape(-1, clean.shape[-1])).reshape(clean.shape).astype("float32")


class RidgeMultiHorizon:
    def __init__(self, alpha: float):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, history: np.ndarray, future: np.ndarray, helper: FutureScaler):
        x = helper.fill(history).reshape(len(history), -1)
        y = helper.fill(future).reshape(len(future), -1)
        self.x_scaler.fit(x)
        self.y_scaler.fit(y)
        self.model.fit(self.x_scaler.transform(x), self.y_scaler.transform(y))
        return self

    def predict(self, history: np.ndarray, helper: FutureScaler) -> np.ndarray:
        x = helper.fill(history).reshape(len(history), -1)
        y = self.y_scaler.inverse_transform(self.model.predict(self.x_scaler.transform(x)))
        return y.reshape(len(history), 24, 9)


def completed_indices(target_end: np.ndarray, prediction_origin: float) -> np.ndarray:
    return np.flatnonzero(target_end <= prediction_origin + EPS)


def first_index_with_completed(target_end: np.ndarray, origin: np.ndarray, count: int, eligible: np.ndarray | None = None) -> int:
    mask = np.ones(len(origin), dtype=bool) if eligible is None else np.asarray(eligible, bool)
    for idx, time_value in enumerate(origin):
        if int(np.sum(mask & (target_end <= time_value + EPS))) >= count:
            return idx
    raise RuntimeError("No causal OOF point has enough completed labels for the frozen warm-up")


def block_edges(start: int, n: int, blocks: int) -> list[tuple[int, int]]:
    if start >= n:
        raise RuntimeError("No OOF prediction rows after causal warm-up")
    cut = [start]
    for part in range(1, blocks):
        cut.append(start + math.ceil((n - start) * part / blocks))
    cut.append(n)
    return [(a, b) for a, b in zip(cut[:-1], cut[1:]) if b > a]


def causal_ridge_oof(history: np.ndarray, future: np.ndarray, windows, cfg: dict[str, Any], fold: str):
    """OOF Ridge with target-completion purge, not merely earlier indices."""
    n = len(history)
    start = first_index_with_completed(
        windows.target_end_age_h, windows.origin_age_h, int(cfg["ridge"]["oof_min_fit_completed_windows"])
    )
    prediction = np.full_like(future, np.nan, dtype=float)
    output_scale = np.full_like(future, np.nan, dtype=float)
    audit: list[dict[str, Any]] = []
    for block_number, (left, right) in enumerate(block_edges(start, n, int(cfg["ridge"]["oof_blocks"])), start=1):
        min_pred_origin = float(windows.origin_age_h[left])
        fit = completed_indices(windows.target_end_age_h, min_pred_origin)
        if len(fit) < int(cfg["ridge"]["oof_min_fit_completed_windows"]):
            raise RuntimeError("OOF block violated required completed-label warm-up")
        helper = FutureScaler().fit(history[fit])
        model = RidgeMultiHorizon(float(cfg["ridge"]["alpha"])).fit(history[fit], future[fit], helper)
        prediction[left:right] = model.predict(history[left:right], helper)
        output_scale[left:right] = model.y_scaler.scale_.reshape(24, 9)
        max_fit_target = float(windows.target_end_age_h[fit].max())
        max_fit_origin = float(windows.origin_age_h[fit].max())
        overlap = bool(max_fit_target > min_pred_origin + EPS)
        if overlap:
            raise RuntimeError("OOF TARGET-LEAKAGE: fit target ends after prediction origin")
        audit.append({
            "stage": "ridge_oof", "fold": fold, "block": block_number,
            "fit_start_index": int(fit.min()), "fit_end_index": int(fit.max()), "fit_count": int(len(fit)),
            "fit_origin_start_h": float(windows.origin_age_h[fit].min()), "fit_origin_end_h": max_fit_origin,
            "max_fit_target_end_h": max_fit_target,
            "predict_start_index": int(left), "predict_end_index": int(right - 1), "predict_count": int(right - left),
            "min_prediction_origin_h": min_pred_origin, "max_prediction_origin_h": float(windows.origin_age_h[right - 1]),
            "target_end_before_prediction_origin": bool(not overlap), "overlap_violation": overlap,
            "purged_index_rows_before_block": int(left - len(fit)), "warmup_dropped_rows": int(start),
            "scaler_fit_scope": "completed_fit_history_only", "ridge_target_fit_scope": "completed_fit_future_only",
            "fit_segment_crossings": 0,
            "prediction_segment_crossings": 0,
            "segment_check": "all source windows were already constructed as contiguous single-segment windows",
        })
    keep = np.isfinite(prediction).all(axis=(1, 2))
    if not keep.any():
        raise RuntimeError("No causal Ridge OOF prediction rows")
    return prediction, output_scale, keep, audit


def simple_confidence_features(history: np.ndarray) -> np.ndarray:
    """Causal, history-only representation for an error-scale predictor."""
    last = history[:, -1, :]
    mean = np.nanmean(history, axis=1)
    std = np.nanstd(history, axis=1)
    return np.concatenate([last, mean, std], axis=1)


def causal_confidence_oof(history: np.ndarray, z_true: np.ndarray, z_oof: np.ndarray, z_scale: np.ndarray, windows, cfg: dict[str, Any], fold: str):
    """Predict per-horizon future-exogenous error from history, causally OOF."""
    residual_valid = np.isfinite(z_oof).all(axis=(1, 2)) & np.isfinite(z_scale).all(axis=(1, 2))
    scale = np.maximum(z_scale, 1e-8)
    actual_error = np.sqrt(np.mean(((z_true - z_oof) / scale) ** 2, axis=2))
    features = simple_confidence_features(history)
    n = len(history)
    start = first_index_with_completed(
        windows.target_end_age_h, windows.origin_age_h, int(cfg["confidence"]["min_fit_completed_residuals"]), residual_valid
    )
    prediction = np.full((n, 24), np.nan, dtype=float)
    audit: list[dict[str, Any]] = []
    for block_number, (left, right) in enumerate(block_edges(start, n, int(cfg["confidence"]["oof_blocks"])), start=1):
        min_origin = float(windows.origin_age_h[left])
        fit = np.flatnonzero(residual_valid & (windows.target_end_age_h <= min_origin + EPS))
        if len(fit) < int(cfg["confidence"]["min_fit_completed_residuals"]):
            raise RuntimeError("Confidence OOF block lacks completed residual supervision")
        x_scaler = StandardScaler().fit(features[fit])
        model = Ridge(alpha=float(cfg["ridge"]["alpha"])).fit(x_scaler.transform(features[fit]), actual_error[fit])
        prediction[left:right] = np.maximum(0.0, model.predict(x_scaler.transform(features[left:right])))
        max_target_end = float(windows.target_end_age_h[fit].max())
        overlap = bool(max_target_end > min_origin + EPS)
        if overlap:
            raise RuntimeError("CONFIDENCE TARGET-LEAKAGE: residual truth crosses prediction origin")
        audit.append({
            "stage": "confidence_oof", "fold": fold, "block": block_number,
            "fit_start_index": int(fit.min()), "fit_end_index": int(fit.max()), "fit_count": int(len(fit)),
            "fit_origin_start_h": float(windows.origin_age_h[fit].min()),
            "fit_origin_end_h": float(windows.origin_age_h[fit].max()),
            "max_fit_target_end_h": max_target_end,
            "predict_start_index": int(left), "predict_end_index": int(right - 1), "predict_count": int(right - left),
            "min_prediction_origin_h": min_origin, "max_prediction_origin_h": float(windows.origin_age_h[right - 1]),
            "target_end_before_prediction_origin": bool(not overlap),
            "overlap_violation": overlap, "purged_index_rows_before_block": int(left - len(fit)),
            "warmup_dropped_rows": int(start), "scaler_fit_scope": "completed_residual_history_only",
            "residual_target_scope": "causal_ridge_oof_rows_with_completed_future_labels",
            "fit_segment_crossings": 0, "prediction_segment_crossings": 0,
            "segment_check": "all source windows were already constructed as contiguous single-segment windows",
        })
    keep = np.isfinite(prediction).all(axis=1)
    if not keep.any():
        raise RuntimeError("No causal confidence OOF rows")
    return prediction, actual_error, residual_valid, keep, audit


def fit_validation_ridge(history_train, future_train, history_valid, cfg):
    helper = FutureScaler().fit(history_train)
    model = RidgeMultiHorizon(float(cfg["ridge"]["alpha"])).fit(history_train, future_train, helper)
    return model.predict(history_valid, helper), model.y_scaler.scale_.reshape(24, 9), helper, model


def fit_validation_confidence(history_train, z_true_train, z_oof_train, z_oof_scale, train, history_valid, z_true_valid, z_ridge_valid, ridge_scale, valid, cfg):
    source = np.isfinite(z_oof_train).all(axis=(1, 2)) & np.isfinite(z_oof_scale).all(axis=(1, 2))
    raw_error_train = np.sqrt(np.mean(((z_true_train - z_oof_train) / np.maximum(z_oof_scale, 1e-8)) ** 2, axis=2))
    train_features = simple_confidence_features(history_train)
    valid_features = simple_confidence_features(history_valid)
    x_scaler = StandardScaler().fit(train_features[source])
    model = Ridge(alpha=float(cfg["ridge"]["alpha"])).fit(x_scaler.transform(train_features[source]), raw_error_train[source])
    prediction = np.maximum(0.0, model.predict(x_scaler.transform(valid_features)))
    actual = np.sqrt(np.mean(((z_true_valid - z_ridge_valid) / np.maximum(ridge_scale, 1e-8)) ** 2, axis=2))
    if float(train.target_end_age_h.max()) > float(valid.origin_age_h.min()) + EPS:
        raise RuntimeError("Validation confidence fit used a train residual not complete before validation")
    return prediction, actual, {"source_completed_oof_residuals": int(source.sum()), "scaler_fit_scope": "all_completed_training_oof_residual_histories"}


def clear_timexer_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "layers" or name.startswith("layers."):
            del sys.modules[name]


def import_timexer(vendor: Path):
    clear_timexer_modules()
    sys.path.insert(0, str(vendor))
    try:
        return __import__("models.TimeXer", fromlist=["Model"])
    finally:
        sys.path.remove(str(vendor))


class StepAlignedTimeXer(nn.Module):
    """Official TimeXer history backbone plus h-to-h FiLM correction and optional CAF."""

    def __init__(self, cfg: dict[str, Any], caf: bool):
        super().__init__()
        tc = cfg["timexer"]
        source = import_timexer(Path(cfg["timexer_vendor"]))
        args = SimpleNamespace(task_name="long_term_forecast", features="MS", seq_len=48, pred_len=24,
            use_norm=False, patch_len=tc["patch_len"], enc_in=14, c_out=1, d_model=tc["d_model"],
            n_heads=tc["n_heads"], e_layers=tc["e_layers"], d_ff=tc["d_ff"], dropout=tc["dropout"],
            factor=5, embed="fixed", freq="h", activation="gelu")
        self.base = source.Model(args)
        self.d_model = int(tc["d_model"])
        self.caf = bool(caf)
        self.horizon_state = nn.Parameter(torch.empty(1, 24, self.d_model))
        nn.init.normal_(self.horizon_state, std=0.02)
        self.film = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, 2 * self.d_model))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.correction = nn.Linear(self.d_model, 1)
        nn.init.zeros_(self.correction.weight); nn.init.zeros_(self.correction.bias)
        if self.caf:
            self.gate_a_raw = nn.Parameter(torch.tensor(0.0))
            self.gate_b = nn.Parameter(torch.tensor(1.0))

    def _historical_state_and_base(self, x: torch.Tensor):
        if tuple(x.shape[1:]) != (48, 14):
            raise RuntimeError(f"Expected (48,14) history, got {tuple(x.shape)}")
        ordered = torch.cat([x[:, :, 1:], x[:, :, :1]], dim=2)
        en_embed, n_vars = self.base.en_embedding(ordered[:, :, -1:].permute(0, 2, 1))
        ex_embed = self.base.ex_embedding(ordered[:, :, :-1], None)
        encoded = self.base.encoder(en_embed, ex_embed)
        core = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1]).permute(0, 1, 3, 2)
        base = self.base.head(core).permute(0, 2, 1).squeeze(-1)
        return encoded[:, -1, :], base

    def forward(self, x: torch.Tensor, z: torch.Tensor, uncertainty: torch.Tensor | None = None, gate_override: float | None = None, return_gate: bool = False):
        if tuple(z.shape[1:]) != (24, 9):
            raise RuntimeError(f"Expected (24,9) future exogenous input, got {tuple(z.shape)}")
        hhist, base = self._historical_state_and_base(x)
        state = hhist.unsqueeze(1) + self.horizon_state
        gamma, beta = self.film(z).chunk(2, dim=-1)
        if self.caf:
            if gate_override is not None:
                gate = torch.full((len(x), 24, 1), float(gate_override), dtype=x.dtype, device=x.device)
            else:
                if uncertainty is None or tuple(uncertainty.shape[1:]) != (24,):
                    raise RuntimeError("CAF requires one causal uncertainty value per horizon")
                gate = torch.sigmoid(self.gate_b - torch.nn.functional.softplus(self.gate_a_raw) * torch.log1p(torch.relu(uncertainty))).unsqueeze(-1)
        else:
            gate = torch.ones((len(x), 24, 1), dtype=x.dtype, device=x.device)
        modified = (1.0 + gate * gamma) * state + gate * beta
        prediction = base + self.correction(modified).squeeze(-1)
        return (prediction, gate.squeeze(-1)) if return_gate else prediction


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def delta_metrics(y_abs: np.ndarray, p_abs: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    target = y_abs - anchor[:, None]
    pred = p_abs - anchor[:, None]
    error = pred - target
    absolute_millivolts = np.abs(error).ravel() * 1000.0
    return {
        "r2_delta": float(r2_score(target.ravel(), pred.ravel())),
        "rmse_delta_mV": float(np.sqrt(np.mean(error ** 2)) * 1000.0),
        "mae_delta_mV": float(np.mean(np.abs(error)) * 1000.0),
        "p50_mV": float(np.quantile(absolute_millivolts, .50)),
        "p90_mV": float(np.quantile(absolute_millivolts, .90)),
        "p95_mV": float(np.quantile(absolute_millivolts, .95)),
        "p99_mV": float(np.quantile(absolute_millivolts, .99)),
        "max_mV": float(np.max(absolute_millivolts)),
    }


def prepare_fold(common, train, valid, feature_indices: list[int]):
    prep = common.FoldPreprocessor().fit(train.x)
    xtr, xva = prep.transform(train.x), prep.transform(valid.x)
    mean, std = float(train.y_delta.mean()), float(train.y_delta.std())
    if not np.isfinite(std) or std <= 1e-10:
        raise RuntimeError("Degenerate DeltaV target scale")
    return xtr, xva, ((train.y_delta - mean) / std).astype("float32"), ((valid.y_delta - mean) / std).astype("float32"), mean, std, FutureScaler().fit(train.x[:, :, feature_indices])


def active_area_from_train(raw_history: np.ndarray) -> float:
    current, density = raw_history[:, :, 0], raw_history[:, :, 3]
    ratios = (current / density)[np.isfinite(current) & np.isfinite(density) & (current > 0) & (density > 1e-7)]
    if len(ratios) < 100:
        raise RuntimeError("Insufficient train-only current/density pairs for FEA projection")
    area = float(np.median(ratios))
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("Invalid train-only active area estimate")
    return area


def project_physical_future(values: np.ndarray, area: float) -> tuple[np.ndarray, float, float]:
    """Constrain only FEA synthetic inputs using fold-train inferred area."""
    before = values.copy()
    arr = values.copy()
    arr[..., 0:3] = np.maximum(arr[..., 0:3], 0.0)
    arr[..., 3] = arr[..., 0] / area
    arr[..., 4] = arr[..., 1] / area
    arr[..., 5] = arr[..., 2] / area
    arr[..., 6:8] = np.maximum(arr[..., 6:8], 0.0)
    arr[..., 8] = np.clip(arr[..., 8], 0.0, 1.0)
    altered = np.abs(arr - before) > 1e-10
    return arr, float(np.mean(np.any(altered, axis=(1, 2)))), float(np.mean(altered))


def augment_future(z_base: np.ndarray, residual: np.ndarray, residual_target_end: np.ndarray, destination_origin: np.ndarray, area: float, probability: float, rng: np.random.Generator):
    output = z_base.copy()
    augmented, project_traj, project_element, pool_sizes = 0, [], [], []
    for row, origin in enumerate(destination_origin):
        pool = np.flatnonzero(residual_target_end <= origin + EPS)
        pool_sizes.append(len(pool))
        if len(pool) and rng.random() < probability:
            source = int(rng.choice(pool))
            candidate = output[row:row + 1] + residual[source:source + 1]
            projected, fraction_t, fraction_e = project_physical_future(candidate, area)
            output[row:row + 1] = projected
            augmented += 1; project_traj.append(fraction_t); project_element.append(fraction_e)
    audit = {
        "augmentation_attempted_rows": int(len(output)), "augmentation_rows": int(augmented),
        "augmentation_fraction": float(augmented / max(1, len(output))),
        "mean_residual_pool_size": float(np.mean(pool_sizes)), "minimum_residual_pool_size": int(np.min(pool_sizes)),
        "projection_trajectory_fraction_among_augmented": float(np.mean(project_traj)) if project_traj else 0.0,
        "projection_element_fraction_among_augmented": float(np.mean(project_element)) if project_element else 0.0,
    }
    return output, audit


def train_model(model: StepAlignedTimeXer, x_train: np.ndarray, y_train: np.ndarray, z_train: np.ndarray, u_train: np.ndarray | None, x_valid: np.ndarray, y_valid: np.ndarray, z_valid: np.ndarray, u_valid: np.ndarray | None, cfg: dict[str, Any], label: str, fold: str, seed: int, fea: dict[str, Any] | None = None):
    dev = device(); model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    loss_fn = nn.MSELoss(); best, best_epoch, stale, state = float("inf"), 0, 0, None
    history: list[dict[str, Any]] = []; epoch_times: list[float] = []
    xv, yv, zv = torch.from_numpy(x_valid).to(dev), torch.from_numpy(y_valid).to(dev), torch.from_numpy(z_valid).to(dev)
    uv = torch.from_numpy(u_valid.astype("float32")).to(dev) if u_valid is not None else None
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    rng = np.random.default_rng(seed + 701)
    for epoch in range(1, int(cfg["training"]["max_epochs"]) + 1):
        began = time.perf_counter()
        if fea is None:
            epoch_z, augment_info = z_train, {"augmentation_fraction": 0.0, "projection_trajectory_fraction_among_augmented": 0.0}
        else:
            raw, augment_info = augment_future(fea["base_raw"], fea["residual"], fea["residual_target_end"], fea["destination_origin"], fea["area"], float(cfg["augmentation"]["probability"]), rng)
            epoch_z = fea["scaler"].transform(raw)
        if u_train is None:
            dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(epoch_z))
        else:
            dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(epoch_z), torch.from_numpy(u_train.astype("float32")))
        loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
        model.train(); losses = []
        for batch in loader:
            xb, yb, zb = (item.to(dev) for item in batch[:3])
            ub = batch[3].to(dev) if len(batch) == 4 else None
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, zb, ub)
            loss = loss_fn(pred, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"])); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if dev.type == "cuda": torch.cuda.synchronize(dev)
        epoch_times.append(time.perf_counter() - began)
        model.eval()
        with torch.no_grad(): valid_rmse = float(torch.sqrt(loss_fn(model(xv, zv, uv), yv)).cpu())
        if valid_rmse < best - 1e-7:
            best, best_epoch, stale = valid_rmse, epoch, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"model": label, "fold": fold, "seed": seed, "epoch": epoch, "train_data_loss": float(np.mean(losses)), "validation_scaled_delta_rmse": valid_rmse, **augment_info})
        if stale >= int(cfg["training"]["patience"]): break
    if state is None: raise RuntimeError("No model checkpoint selected")
    model.load_state_dict(state); torch.save(state, out(f"checkpoints/{label}_{fold}_seed{seed}.pt"))
    model.eval(); began = time.perf_counter()
    with torch.no_grad(): pred, gate = model(xv, zv, uv, return_gate=True)
    if dev.type == "cuda": torch.cuda.synchronize(dev)
    info = {"parameter_count": count_parameters(model), "best_epoch": best_epoch, "epochs_run": len(history), "validation_scaled_delta_rmse": best, "training_time_per_epoch_s": float(np.mean(epoch_times)), "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0, "inference_time_batch_s": float(time.perf_counter() - began), "device": str(dev)}
    return model, pred.cpu().numpy(), gate.cpu().numpy(), info, history


def predict_scaled(model, x, z, u=None, gate_override=None):
    dev = device(); model.eval()
    with torch.no_grad():
        pred, gate = model(torch.from_numpy(x).to(dev), torch.from_numpy(z).to(dev), None if u is None else torch.from_numpy(u.astype("float32")).to(dev), gate_override=gate_override, return_gate=True)
    return pred.cpu().numpy(), gate.cpu().numpy()


def package_fold(common, frame, windows, cfg, fold: str):
    train, valid = windows[fold]
    feature_indices = [cfg["features"].index(x) for x in cfg["future_exogenous_features"]]
    htr, hva = train.x[:, :, feature_indices].astype(float), valid.x[:, :, feature_indices].astype(float)
    ztr, zva = future_exogenous(frame, train, cfg), future_exogenous(frame, valid, cfg)
    z_oof, z_oof_scale, ridge_keep, ridge_audit = causal_ridge_oof(htr, ztr, train, cfg, fold)
    z_ridge, ridge_final_scale, z_scaler, ridge_model = fit_validation_ridge(htr, ztr, hva, cfg)
    if float(train.target_end_age_h.max()) > float(valid.origin_age_h.min()) + EPS:
        raise RuntimeError("Ridge validation fit label crosses validation origin")
    u_oof, u_actual_train, residual_valid, confidence_keep, confidence_audit = causal_confidence_oof(htr, ztr, z_oof, z_oof_scale, train, cfg, fold)
    u_valid, u_actual_valid, confidence_scope = fit_validation_confidence(htr, ztr, z_oof, z_oof_scale, train, hva, zva, z_ridge, ridge_final_scale, valid, cfg)
    fair_keep = ridge_keep & confidence_keep
    if int(fair_keep.sum()) < 64: raise RuntimeError(f"{fold} has too few matched fair training rows")
    xtr, xva, ytr, yva, ymean, ystd, model_z_scaler = prepare_fold(common, train, valid, feature_indices)
    if not np.allclose(model_z_scaler.scaler.mean_, z_scaler.scaler.mean_, equal_nan=True):
        raise RuntimeError("Future scaling mismatch between model and Ridge package")
    area = active_area_from_train(htr)
    return {
        "fold": fold, "train": train, "valid": valid, "xtr": xtr, "xva": xva, "ytr": ytr, "yva": yva, "ymean": ymean, "ystd": ystd,
        "htr": htr, "hva": hva, "ztr": ztr, "zva": zva, "z_oof": z_oof, "z_oof_scale": z_oof_scale, "z_ridge": z_ridge,
        "z_scaler": model_z_scaler, "ridge_keep": ridge_keep, "confidence_keep": confidence_keep, "fair_keep": fair_keep,
        "u_oof": u_oof, "u_valid": u_valid, "u_actual_valid": u_actual_valid, "area": area, "ridge_final_scale": ridge_final_scale,
        "residual": ztr - z_oof, "residual_target_end": train.target_end_age_h, "ridge_audit": ridge_audit,
        "confidence_audit": confidence_audit, "confidence_scope": confidence_scope,
    }


def run_variant(label: str, packages: dict[str, Any], cfg: dict[str, Any], seed: int, fea: bool, caf: bool):
    rows=[]; arrays={}; histories=[]; trained={}; gate_rows=[]; aug_rows=[]
    for fold in FOLDS:
        p = packages[fold]; keep=p["fair_keep"]
        set_seed(seed); model=StepAlignedTimeXer(cfg, caf=caf)
        fea_pack = None
        if fea:
            source = np.isfinite(p["z_oof"]).all(axis=(1,2))
            fea_pack={"base_raw":p["z_oof"][keep], "residual": p["residual"][source], "residual_target_end": p["residual_target_end"][source], "destination_origin": p["train"].origin_age_h[keep], "area": p["area"], "scaler":p["z_scaler"]}
        model, pred_scaled, gate, info, history = train_model(model, p["xtr"][keep], p["ytr"][keep], p["z_scaler"].transform(p["z_oof"][keep]), p["u_oof"][keep] if caf else None, p["xva"], p["yva"], p["z_scaler"].transform(p["z_ridge"]), p["u_valid"] if caf else None, cfg, label, fold, seed, fea_pack)
        abs_pred=p["valid"].anchor[:,None]+pred_scaled*p["ystd"]+p["ymean"]
        if p["valid"].target_end_age_h.max() >= float(cfg["split"]["development_end_h"])-EPS: raise RuntimeError("TEST FIREWALL: prediction target reaches Test")
        arrays[fold]={"y_true":p["valid"].y_abs.astype(float),"y_pred":abs_pred.astype(float),"anchor":p["valid"].anchor.astype(float),"origin_age_h":p["valid"].origin_age_h.astype(float),"target_end_age_h":p["valid"].target_end_age_h.astype(float),"row_origin_index":p["valid"].row_origin_index.astype(int),"z_ridge":p["z_ridge"].astype(float),"z_true":p["zva"].astype(float),"u_pred":p["u_valid"].astype(float),"u_true":p["u_actual_valid"].astype(float),"gate":gate.astype(float)}
        np.savez_compressed(out(f"outputs/model_runs/{label}_{fold}_seed{seed}_predictions.npz"), **arrays[fold])
        status="FORMAL" if label in {"M0_SA_TimeXer_Ridge","M1_SA_TimeXer_FEA","M2_SA_TimeXer_CAF","M3_SA_TimeXer_FEA_CAF"} else "DIAGNOSTIC"
        rows.append({"model":label,"fold":fold,"seed":seed,"eligibility":status,"fea":fea,"caf":caf,"fair_train_windows":int(keep.sum()),"ridge_oof_rows":int(p["ridge_keep"].sum()),"confidence_oof_rows":int(p["confidence_keep"].sum()),**delta_metrics(arrays[fold]["y_true"],arrays[fold]["y_pred"],arrays[fold]["anchor"]),**info})
        histories.extend(history); trained[fold]=model
        gate_rows.extend({"model":label,"fold":fold,"horizon_step":h+1,"horizon_hours":(h+1)*.5,"mean_gate":float(gate[:,h].mean()),"std_gate":float(gate[:,h].std()),"p05_gate":float(np.quantile(gate[:,h],.05)),"p95_gate":float(np.quantile(gate[:,h],.95))} for h in range(24))
        if fea:
            per_epoch=pd.DataFrame(history)
            aug_rows.append({"model":label,"fold":fold,"seed":seed,"active_area_cm2_train_only":p["area"],"fair_train_windows":int(keep.sum()),"historical_residual_source_rows":int(np.isfinite(p["z_oof"]).all(axis=(1,2)).sum()),"mean_epoch_augmentation_fraction":float(per_epoch["augmentation_fraction"].mean()),"mean_epoch_projection_trajectory_fraction":float(per_epoch["projection_trajectory_fraction_among_augmented"].mean()),"probability":float(cfg["augmentation"]["probability"]),"same_fair_sample_set_as_M0":True})
    return pd.DataFrame(rows), arrays, pd.DataFrame(histories), trained, pd.DataFrame(gate_rows), pd.DataFrame(aug_rows)


def summary_rows(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    records=[]
    for model, group in fold_metrics[fold_metrics.seed==42].groupby("model", sort=False):
        if len(group)!=3: continue
        rec={"model":model,"eligibility":group.eligibility.iloc[0],"fea":bool(group.fea.iloc[0]),"caf":bool(group.caf.iloc[0]),"parameter_count":float(group.parameter_count.iloc[0])}
        for metric in ("r2_delta","rmse_delta_mV","mae_delta_mV","p95_mV","p99_mV"):
            rec[f"mean_{metric}"]=float(group[metric].mean())
        for fold in FOLDS:
            item=group[group.fold==fold].iloc[0]
            rec[f"{fold}_rmse_delta_mV"]=float(item.rmse_delta_mV); rec[f"{fold}_r2_delta"]=float(item.r2_delta)
        rec["fold_average_definition"]="unweighted_mean_of_three_fold_metrics"
        records.append(rec)
    return pd.DataFrame(records)


def per_horizon(label: str, arrays: dict[str, Any]) -> list[dict[str, Any]]:
    rows=[]
    for fold,a in arrays.items():
        target=a["y_true"]-a["anchor"][:,None]; pred=a["y_pred"]-a["anchor"][:,None]
        for step in range(24):
            err=pred[:,step]-target[:,step]
            rows.append({"model":label,"fold":fold,"horizon_step":step+1,"horizon_hours":(step+1)*.5,"r2_delta":float(r2_score(target[:,step],pred[:,step])),"rmse_delta_mV":float(np.sqrt(np.mean(err**2))*1000),"mae_delta_mV":float(np.mean(abs(err))*1000)})
    return rows


def tail_metrics(label: str, arrays: dict[str, Any]) -> dict[str, Any]:
    error=np.concatenate([np.abs((a["y_pred"]-a["y_true"]).ravel())*1000 for a in arrays.values()])
    return {"model":label,"aggregation":"pooled_prediction_points_across_three_folds","n_prediction_points":int(len(error)),"p50_mV":float(np.quantile(error,.5)),"p90_mV":float(np.quantile(error,.9)),"p95_mV":float(np.quantile(error,.95)),"p99_mV":float(np.quantile(error,.99)),"max_mV":float(error.max())}


def same_checkpoint_controls(m0_models, packages, cfg):
    rows=[]
    for fold in FOLDS:
        p=packages[fold]; m0=m0_models[fold]
        ridge_scaled=p["z_scaler"].transform(p["z_ridge"]); true_scaled=p["z_scaler"].transform(p["zva"])
        old_path=Path(cfg["previous_n2_8e_root"])/"checkpoints"/f"SA_TimeXer_Oracle_{fold}_seed42.pt"
        if not old_path.exists(): raise RuntimeError(f"Missing frozen N2.8E Oracle checkpoint: {old_path}")
        old=StepAlignedTimeXer(cfg,caf=False); old.load_state_dict(torch.load(old_path,map_location="cpu",weights_only=True)); old.to(device())
        variants={"RR_M0_checkpoint_Ridge_z":(m0,ridge_scaled),"RO_M0_checkpoint_true_z_DIAGNOSTIC":(m0,true_scaled),"OO_N2_8E_oracle_checkpoint_true_z_DIAGNOSTIC":(old,true_scaled)}
        for name,(model,z) in variants.items():
            scaled,_=predict_scaled(model,p["xva"],z)
            pred=p["valid"].anchor[:,None]+scaled*p["ystd"]+p["ymean"]
            rows.append({"variant":name,"fold":fold,"eligibility":"FORMAL" if name.startswith("RR") else "DIAGNOSTIC_ONLY",**delta_metrics(p["valid"].y_abs,pred,p["valid"].anchor)})
    return pd.DataFrame(rows)


def attribution(m0_models, m0_arrays, packages, cfg):
    variable=[]; horizon=[]; correlations=[]; substitution=[]
    names=cfg["future_exogenous_features"]
    groups={"current_amplitude":[0,3],"current_variability":[1,2,4,5],"load_transitions":[6,7],"zero_current":[8]}
    for fold in FOLDS:
        p=packages[fold]; a=m0_arrays[fold]; zerr=(a["z_ridge"]-a["z_true"])/np.maximum(p["ridge_final_scale"],1e-8)
        voltage_error=np.abs(a["y_pred"]-a["y_true"])*1000
        for idx,name in enumerate(names):
            variable.append({"fold":fold,"variable":name,"standardized_future_exo_mae":float(np.mean(abs(zerr[:,:,idx]))),"standardized_future_exo_rmse":float(np.sqrt(np.mean(zerr[:,:,idx]**2))),"raw_future_exo_mae":float(np.mean(abs(a["z_ridge"][:,:,idx]-a["z_true"][:,:,idx])))})
        for step in range(24):
            horizon.append({"fold":fold,"horizon_step":step+1,"horizon_hours":(step+1)*.5,"standardized_future_exo_mae":float(np.mean(abs(zerr[:,step,:]))),"standardized_future_exo_rmse":float(np.sqrt(np.mean(zerr[:,step,:]**2))),"voltage_mae_mV":float(np.mean(voltage_error[:,step]))})
        for group,indices in groups.items():
            exo_origin=np.mean(abs(zerr[:,:,indices]),axis=(1,2)); voltage_origin=np.mean(voltage_error,axis=1)
            pear=float(pearsonr(exo_origin,voltage_origin).statistic) if np.std(exo_origin)>0 and np.std(voltage_origin)>0 else np.nan
            spear=float(spearmanr(exo_origin,voltage_origin).statistic) if np.std(exo_origin)>0 and np.std(voltage_origin)>0 else np.nan
            correlations.append({"fold":fold,"condition_group":group,"pearson_r":pear,"spearman_rho":spear,"mean_standardized_exo_error":float(exo_origin.mean()),"mean_voltage_mae_mV":float(voltage_origin.mean()),"interpretation":"association_only_not_causal"})
            z_mix=a["z_ridge"].copy(); z_mix[:,:,indices]=a["z_true"][:,:,indices]
            # Keep amperage/density pairs internally consistent in diagnostic substitution.
            if group=="current_amplitude": z_mix[:,:,3]=a["z_true"][:,:,3]
            if group=="current_variability": z_mix[:,:,4:6]=a["z_true"][:,:,4:6]
            scaled,_=predict_scaled(m0_models[fold],p["xva"],p["z_scaler"].transform(z_mix))
            pred=p["valid"].anchor[:,None]+scaled*p["ystd"]+p["ymean"]
            base=delta_metrics(a["y_true"],a["y_pred"],a["anchor"])
            mix=delta_metrics(a["y_true"],pred,a["anchor"])
            substitution.append({"fold":fold,"condition_group":group,"diagnostic":"same_M0_checkpoint_replace_group_with_true_future_values","baseline_rmse_mV":base["rmse_delta_mV"],"substitution_rmse_mV":mix["rmse_delta_mV"],"rmse_difference_mV":mix["rmse_delta_mV"]-base["rmse_delta_mV"],"not_causal_attribution":True})
    return pd.DataFrame(variable),pd.DataFrame(horizon),pd.DataFrame(correlations),pd.DataFrame(substitution)


def confidence_diagnostics(packages, m2_gates, cfg):
    calibration=[]; all_corr=[]
    for fold in FOLDS:
        p=packages[fold]; pred=p["u_valid"].ravel(); true=p["u_actual_valid"].ravel()
        rho=float(spearmanr(pred,true).statistic) if np.std(pred)>0 and np.std(true)>0 else np.nan
        q=np.quantile(pred,[0,.25,.5,.75,1])
        labels=np.digitize(pred,q[1:-1],right=True)
        bin_means=[]
        for b in range(4):
            mask=labels==b
            calibration.append({"fold":fold,"bin":b+1,"n":int(mask.sum()),"mean_predicted_uncertainty":float(pred[mask].mean()) if mask.any() else np.nan,"mean_actual_uncertainty":float(true[mask].mean()) if mask.any() else np.nan})
            bin_means.append(float(true[mask].mean()) if mask.any() else np.nan)
        ratio=bin_means[-1]/max(bin_means[0],1e-8) if np.isfinite(bin_means[-1]) and np.isfinite(bin_means[0]) else np.nan
        all_corr.append({"fold":fold,"validation_spearman":rho,"top_to_bottom_actual_error_ratio":ratio,"criterion_spearman":float(cfg["confidence"]["min_validation_spearman"]),"criterion_ratio":float(cfg["confidence"]["min_top_bottom_error_ratio"])})
    corr=pd.DataFrame(all_corr)
    valid=bool((corr.validation_spearman.mean()>=float(cfg["confidence"]["min_validation_spearman"])) and (corr.top_to_bottom_actual_error_ratio.mean()>=float(cfg["confidence"]["min_top_bottom_error_ratio"])))
    gate=m2_gates.copy(); gate["confidence_estimator_status"]="VALID" if valid else "CONFIDENCE_ESTIMATOR_INVALID"
    return pd.DataFrame(calibration),corr,gate,valid


def plot_outputs(summary, same, variable, corr, calibration, gates, horizon, tail):
    figdir=out("outputs/figures")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    formal=summary.copy(); fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(formal.model,formal.mean_rmse_delta_mV,color=["#1f77b4","#2ca02c","#9467bd","#ff7f0e"][:len(formal)]); ax.set_ylabel("Fold-average DeltaV RMSE (mV)"); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figdir/"model_rmse_comparison.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5)); q=same.groupby("variant",as_index=False).rmse_delta_mV.mean(); ax.bar(q.variant,q.rmse_delta_mV,color=["#1f77b4","#2ca02c","#7f7f7f"]); ax.set_ylabel("DeltaV RMSE (mV)"); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figdir/"same_checkpoint_oracle_diagnosis.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5)); q=variable.groupby("variable",as_index=False).standardized_future_exo_mae.mean(); ax.bar(q.variable,q.standardized_future_exo_mae,color="#1f77b4"); ax.set_ylabel("Standardized future-exogenous MAE"); ax.tick_params(axis="x",rotation=50); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figdir/"future_exo_error_by_variable.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5)); q=corr.groupby("condition_group",as_index=False).spearman_rho.mean(); ax.bar(q.condition_group,q.spearman_rho,color="#9467bd"); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("Spearman association with voltage error"); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figdir/"condition_error_voltage_error_correlation.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4.5)); q=calibration.groupby("bin",as_index=False)[["mean_predicted_uncertainty","mean_actual_uncertainty"]].mean(); ax.plot(q.bin,q.mean_predicted_uncertainty,"o-",label="predicted",color="#1f77b4"); ax.plot(q.bin,q.mean_actual_uncertainty,"s-",label="actual diagnostic",color="#d62728"); ax.set_xlabel("Predicted-uncertainty quartile"); ax.set_ylabel("Standardized exogenous error"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir/"confidence_calibration.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5)); q=gates.groupby("horizon_hours",as_index=False).mean_gate.mean(); ax.plot(q.horizon_hours,q.mean_gate,"o-",color="#2ca02c"); ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("Mean CAF gate"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir/"confidence_gate_by_horizon.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5)); q=horizon.groupby(["model","horizon_hours"],as_index=False).rmse_delta_mV.mean(); [ax.plot(g.horizon_hours,g.rmse_delta_mV,label=name) for name,g in q.groupby("model")]; ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("DeltaV RMSE (mV)"); ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir/"per_horizon_rmse.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.bar(tail.model,tail.p99_mV,color="#d62728"); ax.set_ylabel("P99 absolute DeltaV error (mV)"); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figdir/"tail_error.png",dpi=300); plt.close(fig)


def main():
    cfg=load_config(); write_text(CONFIG.read_text(encoding="utf-8"),"outputs/frozen_config/uncertainty_robust_sa_timexer_frozen_config.yaml")
    common,frame,windows,firewall=load_development(cfg); write_json(firewall,"outputs/test_firewall_audit.json")
    packages={fold:package_fold(common,frame,windows,cfg,fold) for fold in FOLDS}
    oof_audit=pd.DataFrame([row for p in packages.values() for row in p["ridge_audit"]+p["confidence_audit"]])
    if oof_audit.overlap_violation.any() or not oof_audit.target_end_before_prediction_origin.all(): raise RuntimeError("OOF audit fail closed")
    write_csv(oof_audit,"outputs/ablation/ridge_oof_temporal_leakage_audit.csv")
    write_text("# Ridge OOF temporal-leakage audit\n\nEvery fit target must end no later than the earliest prediction origin in its block. The CSV below reports the repaired target-completion purge. All rows passed; any `overlap_violation=true` would terminate this phase before M0 training.\n\n"+markdown_table(oof_audit,["stage","fold","block","fit_count","max_fit_target_end_h","min_prediction_origin_h","target_end_before_prediction_origin","overlap_violation","purged_index_rows_before_block","warmup_dropped_rows","scaler_fit_scope"]),"RIDGE_OOF_TEMPORAL_LEAKAGE_AUDIT.md")
    m0_rows,m0_arrays,m0_hist,m0_models,m0_gates,_=run_variant("M0_SA_TimeXer_Ridge",packages,cfg,SEED,False,False)
    same=same_checkpoint_controls(m0_models,packages,cfg)
    variable,exo_horizon,corr,substitution=attribution(m0_models,m0_arrays,packages,cfg)
    m1_rows,m1_arrays,m1_hist,m1_models,m1_gates,m1_aug=run_variant("M1_SA_TimeXer_FEA",packages,cfg,SEED,True,False)
    m2_rows,m2_arrays,m2_hist,m2_models,m2_gates,_=run_variant("M2_SA_TimeXer_CAF",packages,cfg,SEED,False,True)
    calibration,confidence_corr,confidence_gates,confidence_valid=confidence_diagnostics(packages,m2_gates,cfg)
    if not confidence_valid: m2_rows.loc[:,"eligibility"]="CONFIDENCE_ESTIMATOR_INVALID"
    core=pd.concat([m0_rows,m1_rows,m2_rows],ignore_index=True)
    preliminary=summary_rows(core); m0_summary=preliminary[preliminary.model=="M0_SA_TimeXer_Ridge"].iloc[0]
    eligible_components=preliminary[preliminary.model.isin(["M1_SA_TimeXer_FEA","M2_SA_TimeXer_CAF"]) & preliminary.eligibility.eq("FORMAL")].copy()
    if len(eligible_components):
        eligible_components["mean_gain_pct"]=(float(m0_summary.mean_rmse_delta_mV)-eligible_components.mean_rmse_delta_mV)/float(m0_summary.mean_rmse_delta_mV)*100
        eligible_components["fold3_gain_pct"]=(float(m0_summary.fold_3_rmse_delta_mV)-eligible_components.fold_3_rmse_delta_mV)/float(m0_summary.fold_3_rmse_delta_mV)*100
        eligible_components["mean_deterioration_pct"]=(eligible_components.mean_rmse_delta_mV-float(m0_summary.mean_rmse_delta_mV))/float(m0_summary.mean_rmse_delta_mV)*100
        component_gate = (
            (eligible_components.mean_gain_pct >= float(cfg["decision"]["component_mean_rmse_gain_pct"]))
            | (
                (eligible_components.fold3_gain_pct >= float(cfg["decision"]["component_fold3_gain_pct"]))
                & (eligible_components.mean_deterioration_pct <= float(cfg["decision"]["component_max_mean_deterioration_pct"]))
            )
        )
        run_m3 = bool(component_gate.any())
    else: run_m3=False
    if run_m3:
        m3_rows,m3_arrays,m3_hist,m3_models,m3_gates,m3_aug=run_variant("M3_SA_TimeXer_FEA_CAF",packages,cfg,SEED,True,True); core=pd.concat([core,m3_rows],ignore_index=True); model_arrays={"M0_SA_TimeXer_Ridge":m0_arrays,"M1_SA_TimeXer_FEA":m1_arrays,"M2_SA_TimeXer_CAF":m2_arrays,"M3_SA_TimeXer_FEA_CAF":m3_arrays}; allhist=pd.concat([m0_hist,m1_hist,m2_hist,m3_hist],ignore_index=True); aug=pd.concat([m1_aug,m3_aug],ignore_index=True)
    else:
        m3_rows=pd.DataFrame([{ "model":"M3_SA_TimeXer_FEA_CAF", "fold":fold,"seed":SEED,"eligibility":"NOT_RUN_NO_COMPONENT_GAIN","fea":True,"caf":True,"rmse_delta_mV":np.nan,"mae_delta_mV":np.nan,"r2_delta":np.nan} for fold in FOLDS]); core=pd.concat([core,m3_rows],ignore_index=True); model_arrays={"M0_SA_TimeXer_Ridge":m0_arrays,"M1_SA_TimeXer_FEA":m1_arrays,"M2_SA_TimeXer_CAF":m2_arrays}; allhist=pd.concat([m0_hist,m1_hist,m2_hist],ignore_index=True); aug=m1_aug
    summary=summary_rows(core); formal=summary[summary.eligibility.eq("FORMAL")].sort_values("mean_rmse_delta_mV"); best_label=str(formal.iloc[0].model); best=formal.iloc[0]
    # Same trained CAF checkpoint with an uninformative constant gate; diagnostic only.
    constant_rows=[]
    for fold in FOLDS:
        p=packages[fold]; model=m2_models[fold]; constant=float(m2_arrays[fold]["gate"].mean()); scaled,_=predict_scaled(model,p["xva"],p["z_scaler"].transform(p["z_ridge"]),p["u_valid"],gate_override=constant); pred=p["valid"].anchor[:,None]+scaled*p["ystd"]+p["ymean"]; constant_rows.append({"variant":"M2_CAF_same_checkpoint_constant_gate_DIAGNOSTIC","fold":fold,"constant_gate":constant,"eligibility":"DIAGNOSTIC_ONLY",**delta_metrics(p["valid"].y_abs,pred,p["valid"].anchor)})
    same=pd.concat([same,pd.DataFrame(constant_rows)],ignore_index=True)
    horizon=pd.DataFrame([row for label,arrays in model_arrays.items() for row in per_horizon(label,arrays)])
    tail=pd.DataFrame([tail_metrics(label,arrays) for label,arrays in model_arrays.items()])
    # Multi-seed is allowed only for the predeclared gate and never changes selection.
    gain_fold3=(float(m0_summary.fold_3_rmse_delta_mV)-float(best.fold_3_rmse_delta_mV))/float(m0_summary.fold_3_rmse_delta_mV)*100
    best_tail=tail[tail.model==best_label].iloc[0]; m0_tail=tail[tail.model=="M0_SA_TimeXer_Ridge"].iloc[0]
    h95=float(horizon[(horizon.model==best_label)&(horizon.horizon_hours==9.5)].rmse_delta_mV.mean()); base95=float(horizon[(horizon.model=="M0_SA_TimeXer_Ridge")&(horizon.horizon_hours==9.5)].rmse_delta_mV.mean())
    promote=bool((float(best.mean_rmse_delta_mV)<float(cfg["decision"]["multiseed_rmse_mV"]) and float(best.fold_3_rmse_delta_mV)<=float(m0_summary.fold_3_rmse_delta_mV)*1.03) or (20<=float(best.mean_rmse_delta_mV)<float(cfg["decision"]["conditional_rmse_mV"]) and gain_fold3>=float(cfg["decision"]["conditional_fold3_gain_pct"]) and float(best_tail.p99_mV)<float(m0_tail.p99_mV) and h95<=base95*1.10))
    multi=pd.DataFrame([{ "model":best_label,"promotion_status":"not_promoted_seed42_only","n_seeds":1,"seeds":"42","mean_rmse_delta_mV":float(best.mean_rmse_delta_mV),"rmse_std_mV":0.0}])
    if promote:
        seed_rows=[]
        flags={"M0_SA_TimeXer_Ridge":(False,False),"M1_SA_TimeXer_FEA":(True,False),"M2_SA_TimeXer_CAF":(False,True),"M3_SA_TimeXer_FEA_CAF":(True,True)}; fea,caf=flags[best_label]
        for more_seed in (2024,3407):
            r,_,_,_,_,_=run_variant(best_label,packages,cfg,more_seed,fea,caf); core=pd.concat([core,r],ignore_index=True)
        seed_rows=core[core.model==best_label].groupby("seed",as_index=False)[["r2_delta","rmse_delta_mV","mae_delta_mV"]].mean(); multi=pd.DataFrame([{ "model":best_label,"promotion_status":"three_seed_completed","n_seeds":int(len(seed_rows)),"seeds":";".join(seed_rows.seed.astype(str)),"mean_r2_delta":float(seed_rows.r2_delta.mean()),"mean_rmse_delta_mV":float(seed_rows.rmse_delta_mV.mean()),"rmse_std_mV":float(seed_rows.rmse_delta_mV.std(ddof=0))}])
    write_csv(summary,"outputs/ablation/N2_8F_seed42_comparison.csv"); write_csv(multi,"outputs/ablation/N2_8F_multiseed_comparison.csv"); write_csv(summary,"outputs/ablation/N2_8F_ablation.csv"); write_csv(same,"outputs/ablation/same_checkpoint_oracle_swap.csv"); write_csv(substitution,"outputs/attribution/condition_group_sensitivity.csv"); write_csv(variable,"outputs/attribution/future_exo_error_by_variable.csv"); write_csv(exo_horizon,"outputs/attribution/future_exo_error_by_horizon.csv"); write_csv(corr,"outputs/attribution/exo_error_voltage_error_correlation.csv"); write_csv(aug,"outputs/augmentation/forecast_error_augmentation_audit.csv"); write_csv(calibration,"outputs/confidence/confidence_calibration.csv"); write_csv(confidence_gates,"outputs/confidence/confidence_gate_statistics.csv"); write_csv(core,"outputs/lifecycle/N2_8F_fold_metrics.csv"); write_csv(horizon,"outputs/horizon/N2_8F_per_horizon.csv"); write_csv(tail,"outputs/tail/N2_8F_tail_metrics.csv")
    plot_outputs(summary,same,variable,corr,calibration,confidence_gates,horizon,tail)
    write_text("# Forecast-error augmentation audit\n\nFEA samples a complete 24×9 residual trajectory only from causal Ridge OOF sources whose target labels ended before the destination training origin. It never samples validation residuals. Synthetic paths are projected with a fold-train-only current/density area estimate; projection frequency is reported below.\n\n"+markdown_table(aug,["model","fold","fair_train_windows","historical_residual_source_rows","active_area_cm2_train_only","mean_epoch_augmentation_fraction","mean_epoch_projection_trajectory_fraction","same_fair_sample_set_as_M0"]),"FORECAST_ERROR_AUGMENTATION_AUDIT.md")
    write_text("# Confidence estimator audit\n\nThe confidence target is per-horizon standardized future-exogenous Ridge error. Training supervision consists only of completed causal OOF residuals. Validation true residuals are diagnostic-only. A confidence gate is valid only when the mean validation Spearman and top/bottom actual-error ratio clear their frozen thresholds.\n\n"+markdown_table(confidence_corr,["fold","validation_spearman","top_to_bottom_actual_error_ratio","criterion_spearman","criterion_ratio"])+f"\n\nFinal status: **{'VALID' if confidence_valid else 'CONFIDENCE_ESTIMATOR_INVALID'}**.","CONFIDENCE_ESTIMATOR_AUDIT.md")
    write_text("# Condition-error attribution\n\nAll analyses are development-only. Correlations and same-checkpoint group substitutions are diagnostic associations, not causal claims. Current and current-density are retained as internally consistent groups.\n\n## Error by variable\n\n"+markdown_table(variable.groupby("variable",as_index=False).mean(numeric_only=True),["variable","standardized_future_exo_mae","standardized_future_exo_rmse","raw_future_exo_mae"])+"\n\n## Group association\n\n"+markdown_table(corr.groupby("condition_group",as_index=False).mean(numeric_only=True),["condition_group","pearson_r","spearman_rho","mean_standardized_exo_error","mean_voltage_mae_mV"])+"\n\n## Same-checkpoint substitutions\n\n"+markdown_table(substitution.groupby("condition_group",as_index=False).mean(numeric_only=True),["condition_group","baseline_rmse_mV","substitution_rmse_mV","rmse_difference_mV"]),"CONDITION_ERROR_ATTRIBUTION.md")
    historical=24.719477623350237; m0=float(m0_summary.mean_rmse_delta_mV); improvement=(m0-float(best.mean_rmse_delta_mV))/m0*100; h12=float(horizon[(horizon.model==best_label)&(horizon.horizon_hours==12.0)].rmse_delta_mV.mean()); periodic=(h95-base95)/base95*100
    decision={"phase_status":"PASS","inherited_study_commit":cfg["inherited_study_commit"],"test_performance_accessed":False,"strict_oof_target_completion_audit":"PASS","historical_n2_8e_ridge_rmse_mV_reference_only":historical,"fair_m0_rmse_mV":m0,"best_formal_model":best_label,"best_formal_rmse_mV":float(best.mean_rmse_delta_mV),"best_formal_r2":float(best.mean_r2_delta),"improvement_vs_fair_m0_pct":improvement,"confidence_estimator_status":"VALID" if confidence_valid else "CONFIDENCE_ESTIMATOR_INVALID","rmse_9p5h_mV":h95,"rmse_12h_mV":h12,"periodic_degradation_vs_m0_9p5h_pct":periodic,"multiseed":multi.iloc[0].to_dict(),"time_xer_status":"RETAIN" if float(best.mean_rmse_delta_mV)<m0 else "NO_FORMAL_GAIN","further_physics_recommended":False,"enter_next_phase":"GO" if float(best.mean_rmse_delta_mV)<m0 and confidence_valid else "NO-GO","reason":"Development-only formal comparison after strict target-completion OOF repair; Test remains firewalled."}
    write_yaml(decision,"PHASE_N2_8F_DECISION.yaml")
    report=f"""# Phase N2.8F results

## Causal OOF repair and firewall

The historical N2.8E `24.719478 mV` Ridge result is reference-only because its OOF implementation did not explicitly ensure that fit future-exogenous labels had completed before the next prediction origin. This phase rebuilt OOF Ridge and confidence labels with the stricter target-end rule. The repaired formal M0 is **{m0:.6f} mV**. Test feature rows, targets, predictions, metrics, and diagnostic access are all **0/false**.

## Formal ablation (seed 42; unweighted mean across the three chronological folds)

{markdown_table(summary,["model","eligibility","mean_r2_delta","mean_rmse_delta_mV","mean_mae_delta_mV","fold_3_rmse_delta_mV","mean_p95_mV","mean_p99_mV"])}

M3 was {'run under the predeclared component-gain rule' if run_m3 else 'not run: `NOT_RUN_NO_COMPONENT_GAIN`'}.

## Same-checkpoint and condition diagnostics

{markdown_table(same.groupby("variant",as_index=False).mean(numeric_only=True),["variant","rmse_delta_mV","mae_delta_mV","p95_mV","p99_mV"])}

Oracle and true-future substitutions are diagnostic-only. They cannot select the formal model.

## Confidence, tails, horizons, and promotion

Confidence estimator: **{'VALID' if confidence_valid else 'CONFIDENCE_ESTIMATOR_INVALID'}**.  At 9.5 h, the selected formal model has RMSE **{h95:.6f} mV** (relative to M0: {periodic:+.2f}%); at 12 h it has **{h12:.6f} mV**. Tail statistics are pooled over prediction points across the three development folds:

{markdown_table(tail,["model","aggregation","p95_mV","p99_mV","max_mV"])}

{markdown_table(multi,["model","promotion_status","n_seeds","seeds","mean_rmse_delta_mV","rmse_std_mV"])}

## Decision

- Best formal model: **{best_label}**
- Fair-M0 improvement: **{improvement:.3f}%**
- Next-phase permission: **{decision['enter_next_phase']}**
- TimeXer status: **{decision['time_xer_status']}**
- Further physics mechanism: **not justified by this phase**

All conclusions are limited to frozen development folds. This phase stops here and does not start N2.8G or evaluate Test.
"""
    write_text(report,"PHASE_N2_8F_RESULTS.md")
    print(json.dumps({"phase":"N2.8F","status":"PASS","fair_m0_rmse_mV":m0,"best":best_label,"best_rmse_mV":float(best.mean_rmse_delta_mV),"confidence_valid":confidence_valid,"promote":promote,"next":decision["enter_next_phase"]},ensure_ascii=False))


if __name__ == "__main__":
    main()

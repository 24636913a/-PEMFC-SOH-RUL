"""Frozen Development-only protocol for the final RealHyFC backbone screen.

This module never scans the full causal CSV. It reads exactly the registered
prefix and rejects any attempt to extend the data or the protocol.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml


sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE = SCRIPT_DIR.parent
PROJECT = PHASE.parent
REPO = PROJECT.parent
CONFIG_PATH = PHASE / "configs" / "PREREGISTERED_PROTOCOL.yaml"
CONFIG_SHA_PATH = PHASE / "configs" / "PREREGISTERED_PROTOCOL.sha256"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def phase_path(*parts: str) -> Path:
    path = PHASE.joinpath(*parts)
    resolved = path.resolve()
    if resolved != PHASE.resolve() and PHASE.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing path outside Phase BS: {path}")
    return path


def write_json(payload: Any, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_text(text: str, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return path


def load_config() -> dict[str, Any]:
    raw = CONFIG_PATH.read_bytes()
    expected_line = CONFIG_SHA_PATH.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256_bytes(raw)
    if observed != expected_line:
        raise RuntimeError("PREREGISTERED_PROTOCOL SHA256 mismatch")
    cfg = yaml.safe_load(raw)
    if cfg.get("status") != "FROZEN_BEFORE_DEVELOPMENT_RESULTS":
        raise RuntimeError("Protocol is not frozen")
    current = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if current != str(cfg["base_commit"]):
        raise RuntimeError(f"Repository HEAD drift: expected {cfg['base_commit']}, got {current}")
    if cfg["task"]["input_shape"] != ["batch", 48, 14]:
        raise RuntimeError("Frozen model input shape changed")
    if len(cfg["data_firewall"]["f2_raw_feature_order"]) != 14:
        raise RuntimeError("F2 feature count changed")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _data_columns(cfg: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(cfg["data_firewall"]["required_columns"]))


def read_development_prefix(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only registered rows 0..3006; no global split/Test scan is allowed."""
    data_cfg = cfg["data_firewall"]
    nrows = int(data_cfg["pandas_read_csv_nrows"])
    if nrows != 3007:
        raise RuntimeError("Frozen CSV prefix length changed")
    source = PROJECT / data_cfg["csv"]
    if not source.is_file():
        raise FileNotFoundError(source)
    prefix = pd.read_csv(source, usecols=_data_columns(cfg), nrows=nrows)
    if len(prefix) != int(data_cfg["expected_prefix_rows"]):
        raise RuntimeError(f"Prefix row count mismatch: {len(prefix)}")
    end_col = "bin_end_ophrs2_h"
    start_col = "bin_start_ophrs2_h"
    development_end = float(cfg["development_end_h_exclusive"])
    development = prefix.loc[prefix[end_col].to_numpy(float) < development_end].copy()
    if len(development) != int(data_cfg["expected_development_rows_after_end_filter"]):
        raise RuntimeError(f"Development row count mismatch: {len(development)}")
    development.reset_index(drop=True, inplace=True)
    starts = development[start_col].to_numpy(float)
    if not np.all(np.diff(starts) > 0):
        raise RuntimeError("Development prefix is not strictly time ordered")
    if np.any(development[end_col].to_numpy(float) >= development_end):
        raise RuntimeError("Development end boundary violation")
    audit = {
        "csv": str(source),
        "pandas_read_csv_nrows": nrows,
        "prefix_rows_read": len(prefix),
        "development_rows_retained": len(development),
        "rows_after_registered_prefix_read": 0,
        "test_rows_read": 0,
        "test_features_read": 0,
        "test_targets_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "development_end_h_exclusive": development_end,
    }
    return development, audit


@dataclass(frozen=True)
class WindowSet:
    x: np.ndarray
    y_delta_V: np.ndarray
    anchor_V: np.ndarray
    origin_age_h: np.ndarray
    origin_datetime: np.ndarray
    target_end_age_h: np.ndarray
    origin_row: np.ndarray
    target_row: np.ndarray
    target_datetime: np.ndarray

    def take(self, indexes: np.ndarray) -> "WindowSet":
        return WindowSet(
            x=self.x[indexes], y_delta_V=self.y_delta_V[indexes], anchor_V=self.anchor_V[indexes],
            origin_age_h=self.origin_age_h[indexes], origin_datetime=self.origin_datetime[indexes],
            target_end_age_h=self.target_end_age_h[indexes], origin_row=self.origin_row[indexes],
            target_row=self.target_row[indexes], target_datetime=self.target_datetime[indexes],
        )


def build_windows(frame: pd.DataFrame, cfg: dict[str, Any], region_start_h: float, region_end_h: float) -> WindowSet:
    features = list(cfg["data_firewall"]["f2_raw_feature_order"])
    task = cfg["task"]
    lookback, horizon, resolution = int(task["lookback_steps"]), int(task["forecast_steps"]), float(task["resolution_h"])
    starts = frame["bin_start_ophrs2_h"].to_numpy(float)
    ends = frame["bin_end_ophrs2_h"].to_numpy(float)
    voltage = frame["cell_voltage_mean"].to_numpy(float)
    values = frame[features].to_numpy(float)
    valid = frame["bin_valid_for_windows"].astype(bool).to_numpy()
    segment = frame["temporal_segment_id"].to_numpy()
    times = frame["last_datetime"].astype(str).to_numpy()
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    anchors: list[float] = []
    ages: list[float] = []
    origin_times: list[str] = []
    target_ends: list[float] = []
    origins: list[int] = []
    target_rows: list[np.ndarray] = []
    target_times: list[np.ndarray] = []
    for origin in range(lookback - 1, len(frame) - horizon):
        first, last = origin - lookback + 1, origin + horizon
        if starts[first] < region_start_h - 1e-9 or ends[last] > region_end_h + 1e-9:
            continue
        block = slice(first, last + 1)
        if not valid[block].all() or np.any(segment[block] != segment[origin]):
            continue
        if not np.allclose(np.diff(starts[block]), resolution, atol=1e-6, rtol=0):
            continue
        if not np.isfinite(voltage[first:last + 1]).all():
            continue
        anchor = voltage[origin]
        future = voltage[origin + 1:last + 1]
        x_rows.append(values[first:origin + 1])
        y_rows.append(future - anchor)
        anchors.append(anchor)
        ages.append(ends[origin])
        origin_times.append(times[origin])
        target_ends.append(ends[last])
        origins.append(origin)
        target_rows.append(np.arange(origin + 1, last + 1, dtype=np.int64))
        target_times.append(times[origin + 1:last + 1])
    if not x_rows:
        raise RuntimeError(f"No legal windows in {region_start_h}..{region_end_h}")
    result = WindowSet(
        x=np.asarray(x_rows, dtype=np.float32), y_delta_V=np.asarray(y_rows, dtype=np.float32),
        anchor_V=np.asarray(anchors, dtype=np.float32), origin_age_h=np.asarray(ages, dtype=np.float64),
        origin_datetime=np.asarray(origin_times), target_end_age_h=np.asarray(target_ends, dtype=np.float64),
        origin_row=np.asarray(origins, dtype=np.int64), target_row=np.asarray(target_rows, dtype=np.int64),
        target_datetime=np.asarray(target_times),
    )
    if np.any(result.target_end_age_h >= float(cfg["development_end_h_exclusive"])):
        raise RuntimeError("Test firewall: legal target reaches Development boundary")
    return result


def build_outer_folds(frame: pd.DataFrame, cfg: dict[str, Any]) -> list[tuple[str, WindowSet, WindowSet]]:
    folds: list[tuple[str, WindowSet, WindowSet]] = []
    for spec in cfg["folds"]:
        train = build_windows(frame, cfg, float(spec["train_start_h"]), float(spec["train_end_h"]))
        validation = build_windows(frame, cfg, float(spec["validation_start_h"]), float(spec["validation_end_h"]))
        if set(train.origin_row) & set(validation.origin_row):
            raise RuntimeError(f"{spec['fold']}: outer origins overlap")
        folds.append((str(spec["fold"]), train, validation))
    return folds


def split_inner_fit_stopper(train: WindowSet, fraction: float) -> tuple[WindowSet, WindowSet]:
    if not (0 < fraction < 0.5):
        raise RuntimeError("Invalid frozen stopper fraction")
    n_stop = int(math.ceil(len(train.origin_row) * fraction))
    stop_idx = np.arange(len(train.origin_row) - n_stop, len(train.origin_row), dtype=int)
    stop_targets = set(train.target_row[stop_idx].reshape(-1).tolist())
    candidate = np.arange(0, len(train.origin_row) - n_stop, dtype=int)
    fit_idx = np.asarray([i for i in candidate if not (set(train.target_row[i].tolist()) & stop_targets)], dtype=int)
    if not len(fit_idx):
        raise RuntimeError("No fit origins remain after stopper target isolation")
    fit, stopper = train.take(fit_idx), train.take(stop_idx)
    if set(fit.target_row.reshape(-1).tolist()) & set(stopper.target_row.reshape(-1).tolist()):
        raise RuntimeError("Inner target-row overlap firewall failed")
    if not np.all(fit.origin_row < stopper.origin_row.min()):
        raise RuntimeError("Inner stopper is not chronologically last")
    return fit, stopper


class FoldPreprocessor:
    def __init__(self) -> None:
        self.medians: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "FoldPreprocessor":
        flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        self.medians = np.nanmedian(flat, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        flat = np.where(np.isfinite(flat), flat, self.medians)
        self.mean = flat.mean(axis=0)
        self.scale = flat.std(axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.medians is None or self.mean is None or self.scale is None:
            raise RuntimeError("Scaler used before train-only fit")
        flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        flat = np.where(np.isfinite(flat), flat, self.medians)
        return ((flat - self.mean) / self.scale).reshape(x.shape).astype(np.float32)

    @property
    def voltage_scale(self) -> float:
        if self.scale is None:
            raise RuntimeError("Scaler used before fit")
        return float(self.scale[0])


def standardized_delta_target(windows: WindowSet, scaler: FoldPreprocessor) -> np.ndarray:
    return (windows.y_delta_V / scaler.voltage_scale).astype(np.float32)


def delta_from_standardized_absolute(pred_abs_scaled: np.ndarray | torch.Tensor, x_scaled: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return pred_abs_scaled - x_scaled[:, -1, 0:1]


def delta_metrics(y_true_V: np.ndarray, y_pred_V: np.ndarray, target_row: np.ndarray) -> dict[str, float]:
    truth, pred = np.asarray(y_true_V, float), np.asarray(y_pred_V, float)
    if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1] != 24:
        raise RuntimeError("DeltaV metric shape mismatch")
    error = pred - truth
    flat = error.reshape(-1)
    mse = float(np.mean(flat ** 2))
    denominator = float(np.sum((truth.reshape(-1) - truth.mean()) ** 2))
    point = pd.DataFrame({"target_row": target_row.reshape(-1), "se_mV2": (flat * 1000.0) ** 2})
    target_equal_mse_mV2 = float(point.groupby("target_row", sort=False).se_mV2.mean().mean())
    absolute_mV = abs(flat) * 1000.0
    return {
        "rmse_mV": math.sqrt(mse) * 1000.0,
        "mae_mV": float(np.mean(absolute_mV)),
        "r2": 1.0 - float(np.sum(flat ** 2)) / denominator if denominator > 0 else math.nan,
        "endpoint_rmse_mV": float(np.sqrt(np.mean(error[:, -1] ** 2)) * 1000.0),
        "p95_abs_error_mV": float(np.quantile(absolute_mV, 0.95)),
        "p99_abs_error_mV": float(np.quantile(absolute_mV, 0.99)),
        "max_abs_error_mV": float(np.max(absolute_mV)),
        "target_time_equal_rmse_mV": math.sqrt(target_equal_mse_mV2),
        "query_count": int(flat.size),
        "unique_target_row_count": int(point.target_row.nunique()),
    }


def summarize_seed(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "seed", "fold", "rmse_mV", "mae_mV", "r2", "endpoint_rmse_mV", "p95_abs_error_mV", "p99_abs_error_mV", "max_abs_error_mV", "target_time_equal_rmse_mV"}
    if required - set(rows.columns):
        raise RuntimeError("Metric summary input incomplete")
    out = rows.groupby(["model", "seed"], as_index=False).agg(
        formal_RMSE_mV=("rmse_mV", "mean"), MAE_mV=("mae_mV", "mean"), R2=("r2", "mean"),
        endpoint_RMSE_mV=("endpoint_rmse_mV", "mean"), P95_mV=("p95_abs_error_mV", "mean"),
        P99_mV=("p99_abs_error_mV", "mean"), max_abs_error_mV=("max_abs_error_mV", "max"),
        target_time_equal_RMSE_mV=("target_time_equal_rmse_mV", "mean"),
        Fold3_RMSE_mV=("rmse_mV", lambda s: float(s[rows.loc[s.index, "fold"].eq("fold_3")].iloc[0])),
        Fold3_target_time_equal_RMSE_mV=("target_time_equal_rmse_mV", lambda s: float(s[rows.loc[s.index, "fold"].eq("fold_3")].iloc[0])),
        parameter_count=("parameter_count", "max"), peak_gpu_memory_MiB=("peak_gpu_memory_MiB", "max"),
        total_inner_plus_refit_time_s=("total_inner_plus_refit_time_s", "sum"),
    )
    return out


def canonical_source_hash(repo_root: Path, commit: str, relative: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), "show", f"{commit}:{relative}"], stderr=subprocess.STDOUT)


def canonical_source_sha256(repo_root: Path, commit: str, relative: str) -> str:
    return sha256_bytes(canonical_source_hash(repo_root, commit, relative))


def source_fingerprint_audit(cfg: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model, source in cfg["official_sources"].items():
        runtime_root = PHASE / source["vendor_root"]
        runtime_is_git_root = (runtime_root / ".git").exists()
        git_root = runtime_root if runtime_is_git_root else runtime_root.parent
        prefix = "" if runtime_is_git_root else runtime_root.name + "/"
        observed_commit = subprocess.check_output(["git", "-C", str(git_root), "rev-parse", "HEAD"], text=True).strip()
        for relative, expected in source["canonical_git_blob_sha256"].items():
            git_relative = prefix + relative
            observed = canonical_source_sha256(git_root, str(source["commit"]), git_relative)
            records.append({"model": model, "repository": source["repository"], "expected_commit": source["commit"], "observed_commit": observed_commit, "license": source["license"], "relative_path": relative, "expected_canonical_sha256": expected, "observed_canonical_sha256": observed, "pass": observed_commit == str(source["commit"]) and observed == str(expected)})
    result = pd.DataFrame(records)
    if not result["pass"].all():
        raise RuntimeError("Official source fingerprint mismatch")
    return result


def _clear_absolute_model_packages() -> None:
    for name in list(sys.modules):
        if name in {"models", "layers", "model"} or name.startswith(("models.", "layers.", "model.")):
            del sys.modules[name]


class TimesNetVoltageAdapter(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        source = cfg["official_sources"]["TimesNet"]["config"]
        root = PHASE / cfg["official_sources"]["TimesNet"]["vendor_root"]
        _clear_absolute_model_packages(); sys.path.insert(0, str(root))
        try:
            official = importlib.import_module("models.TimesNet")
        finally:
            sys.path.remove(str(root))
        self.official_model = official.Model(SimpleNamespace(label_len=0, num_class=0, **source))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.official_model(x, None, None, None)
        if out.shape != (x.shape[0], 24, 14):
            raise RuntimeError(f"TimesNet output shape {tuple(out.shape)}")
        return out[:, :, 0], out.new_zeros(())


class SimpleTMVoltageAdapter(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        source = cfg["official_sources"]["SimpleTM"]["config"]
        root = PHASE / cfg["official_sources"]["SimpleTM"]["vendor_root"]
        _clear_absolute_model_packages(); sys.path.insert(0, str(root))
        try:
            official = importlib.import_module("model.SimpleTM")
        finally:
            sys.path.remove(str(root))
        self.official_model = official.Model(SimpleNamespace(**source))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.official_model(x, None, None, None)
        if out.shape != (x.shape[0], 24, 14):
            raise RuntimeError(f"SimpleTM output shape {tuple(out.shape)}")
        return out[:, :, 0], out.new_zeros(())


class DUETVoltageAdapter(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        source = cfg["official_sources"]["DUET"]["config"]
        root = PHASE / "vendor" / "DUET_official"
        sys.path.insert(0, str(root))
        try:
            official = importlib.import_module("ts_benchmark.baselines.duet.models.duet_model")
        finally:
            sys.path.remove(str(root))
        self.official_model = official.DUETModel(SimpleNamespace(**source))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, importance_loss = self.official_model(x)
        if out.shape != (x.shape[0], 24, 14):
            raise RuntimeError(f"DUET output shape {tuple(out.shape)}")
        if importance_loss.ndim > 0:
            importance_loss = importance_loss.mean()
        return out[:, :, 0], importance_loss


class ModernTCNVoltageAdapter(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        source = cfg["official_sources"]["ModernTCN"]["config"]
        root = PHASE / cfg["official_sources"]["ModernTCN"]["vendor_root"]
        _clear_absolute_model_packages(); sys.path.insert(0, str(root))
        try:
            official = importlib.import_module("models.ModernTCN")
        finally:
            sys.path.remove(str(root))
        self.official_model = official.ModernTCN(
            patch_size=int(source["patch_size"]), patch_stride=int(source["patch_stride"]), stem_ratio=int(source["stem_ratio"]),
            downsample_ratio=int(source["downsample_ratio"]), ffn_ratio=int(source["ffn_ratio"]), num_blocks=list(source["num_blocks"]),
            large_size=list(source["large_size"]), small_size=list(source["small_size"]), dims=list(source["dims"]), dw_dims=list(source["dw_dims"]),
            nvars=int(source["nvars"]), small_kernel_merged=bool(source["small_kernel_merged"]), backbone_dropout=float(source["backbone_dropout"]),
            head_dropout=float(source["head_dropout"]), use_multi_scale=False, revin=bool(source["revin"]), affine=bool(source["affine"]),
            subtract_last=bool(source["subtract_last"]), freq="h", seq_len=int(source["seq_len"]), c_in=int(source["nvars"]),
            individual=bool(source["individual"]), target_window=int(source["target_window"]),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.official_model(x.transpose(1, 2), None)
        if out.shape != (x.shape[0], 14, 24):
            raise RuntimeError(f"ModernTCN output shape {tuple(out.shape)}")
        return out[:, 0, :], out.new_zeros(())


def create_model(model: str, cfg: dict[str, Any]) -> nn.Module:
    factories = {"TimesNet": TimesNetVoltageAdapter, "SimpleTM": SimpleTMVoltageAdapter, "DUET": DUETVoltageAdapter, "ModernTCN": ModernTCNVoltageAdapter}
    if model not in factories:
        raise RuntimeError(f"Frozen candidate list violation: {model}")
    return factories[model](cfg)


def model_delta_prediction(model: nn.Module, x_scaled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    absolute_scaled, auxiliary = model(x_scaled)
    delta_scaled = delta_from_standardized_absolute(absolute_scaled, x_scaled)
    if delta_scaled.shape != (x_scaled.shape[0], 24):
        raise RuntimeError("Voltage DeltaV adapter shape mismatch")
    return delta_scaled, auxiliary


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

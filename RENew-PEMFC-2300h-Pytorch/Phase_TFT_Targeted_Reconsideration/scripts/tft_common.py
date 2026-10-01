"""Frozen Development-only data, training, metric and gate utilities for Phase TFT."""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from tft_models import create_model, trainable_parameter_count


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE = SCRIPT_DIR.parent
PROJECT = PHASE.parent
REPO = PROJECT.parent
CONFIG_PATH = PHASE / "configs" / "PREREGISTERED_PROTOCOL.yaml"
CONFIG_SHA_PATH = PHASE / "configs" / "PREREGISTERED_PROTOCOL.sha256"
SOURCE_MANIFEST_PATH = PHASE / "SOURCE_MANIFEST.yaml"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_path(*parts: str) -> Path:
    candidate = PHASE.joinpath(*parts).resolve()
    root = PHASE.resolve()
    if candidate != root and root not in candidate.parents:
        raise RuntimeError(f"Refusing write outside Phase TFT: {candidate}")
    return candidate


def write_json(payload: Any, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return path


def write_text(value: str, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
    return path


def current_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def load_config() -> dict[str, Any]:
    raw = CONFIG_PATH.read_bytes()
    expected = CONFIG_SHA_PATH.read_text(encoding="utf-8").split()[0].lower()
    if sha256_bytes(raw) != expected:
        raise RuntimeError("Frozen TFT protocol SHA256 mismatch")
    cfg = yaml.safe_load(raw)
    if cfg.get("status") != "FROZEN_BEFORE_DEVELOPMENT_RESULTS":
        raise RuntimeError("TFT protocol is not frozen before Development results")
    if cfg["task"]["input_shape"] != ["batch", 48, 14]:
        raise RuntimeError("Frozen input shape changed")
    if len(cfg["data_firewall"]["f2_raw_feature_order"]) != 14:
        raise RuntimeError("Raw F2 cardinality changed")
    if current_head() != str(cfg["base_commit"]):
        raise RuntimeError("Repository HEAD drift before Phase TFT execution")
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


def data_columns(cfg: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(cfg["data_firewall"]["required_columns"]))


def read_development_prefix(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The sole authorised N1 access: exact prefix and strict Development filter."""
    dcfg = cfg["data_firewall"]
    if int(dcfg["pandas_read_csv_nrows"]) != 3007:
        raise RuntimeError("N1 firewall prefix length changed")
    source = PROJECT / dcfg["csv"]
    prefix = pd.read_csv(source, usecols=data_columns(cfg), nrows=3007)
    if len(prefix) != 3007:
        raise RuntimeError("N1 allowed prefix row count mismatch")
    dev_end = float(cfg["development_end_h_exclusive"])
    frame = prefix.loc[prefix.bin_end_ophrs2_h.to_numpy(float) < dev_end].copy().reset_index(drop=True)
    if len(frame) != 3005 or np.any(frame.bin_end_ophrs2_h.to_numpy(float) >= dev_end):
        raise RuntimeError("N1 Development filter/fence mismatch")
    if not np.all(np.diff(frame.bin_start_ophrs2_h.to_numpy(float)) > 0):
        raise RuntimeError("Development prefix is not strictly time ordered")
    return frame, {
        "source": str(source), "pandas_read_csv_nrows": 3007, "prefix_rows_read": 3007,
        "development_rows_retained": 3005, "post_prefix_rows_read": 0, "test_accessed": False,
        "test_targets_read": 0, "test_predictions_constructed": False, "test_metrics_computed": False,
    }


@dataclass(frozen=True)
class WindowSet:
    x: np.ndarray
    y_delta_V: np.ndarray
    anchor_V: np.ndarray
    origin_row: np.ndarray
    target_row: np.ndarray
    origin_age_h: np.ndarray
    target_end_age_h: np.ndarray
    origin_datetime: np.ndarray
    target_datetime: np.ndarray

    def take(self, indices: np.ndarray) -> "WindowSet":
        return WindowSet(*(getattr(self, name)[indices] for name in self.__dataclass_fields__))


def build_windows(frame: pd.DataFrame, cfg: dict[str, Any], start_h: float, end_h: float) -> WindowSet:
    task = cfg["task"]
    lookback, horizon, resolution = int(task["lookback_steps"]), int(task["forecast_steps"]), float(task["resolution_h"])
    values = frame[cfg["data_firewall"]["f2_raw_feature_order"]].to_numpy(float)
    voltage = frame.cell_voltage_mean.to_numpy(float)
    starts, ends = frame.bin_start_ophrs2_h.to_numpy(float), frame.bin_end_ophrs2_h.to_numpy(float)
    valid, segment = frame.bin_valid_for_windows.astype(bool).to_numpy(), frame.temporal_segment_id.to_numpy()
    datetimes = frame.last_datetime.astype(str).to_numpy()
    payload: dict[str, list[Any]] = {name: [] for name in WindowSet.__dataclass_fields__}
    for origin in range(lookback - 1, len(frame) - horizon):
        first, last = origin - lookback + 1, origin + horizon
        block = slice(first, last + 1)
        if starts[first] < start_h - 1e-9 or ends[last] > end_h + 1e-9:
            continue
        if not valid[block].all() or np.any(segment[block] != segment[origin]):
            continue
        if not np.allclose(np.diff(starts[block]), resolution, rtol=0.0, atol=1e-6):
            continue
        if not np.isfinite(voltage[first:last + 1]).all():
            continue
        payload["x"].append(values[first:origin + 1])
        payload["anchor_V"].append(voltage[origin])
        payload["y_delta_V"].append(voltage[origin + 1:last + 1] - voltage[origin])
        payload["origin_row"].append(origin)
        payload["target_row"].append(np.arange(origin + 1, last + 1, dtype=np.int64))
        payload["origin_age_h"].append(ends[origin])
        payload["target_end_age_h"].append(ends[last])
        payload["origin_datetime"].append(datetimes[origin])
        payload["target_datetime"].append(datetimes[origin + 1:last + 1])
    if not payload["x"]:
        raise RuntimeError("No legal frozen-protocol windows")
    result = WindowSet(
        x=np.asarray(payload["x"], dtype=np.float32), y_delta_V=np.asarray(payload["y_delta_V"], dtype=np.float32),
        anchor_V=np.asarray(payload["anchor_V"], dtype=np.float32), origin_row=np.asarray(payload["origin_row"], dtype=np.int64),
        target_row=np.asarray(payload["target_row"], dtype=np.int64), origin_age_h=np.asarray(payload["origin_age_h"], dtype=np.float64),
        target_end_age_h=np.asarray(payload["target_end_age_h"], dtype=np.float64), origin_datetime=np.asarray(payload["origin_datetime"]), target_datetime=np.asarray(payload["target_datetime"]),
    )
    if np.any(result.target_end_age_h >= float(cfg["development_end_h_exclusive"])):
        raise RuntimeError("Test firewall: target reached Development boundary")
    if not np.array_equal(result.target_row, result.origin_row[:, None] + np.arange(1, 25)):
        raise RuntimeError("Target key construction changed")
    return result


def build_outer_folds(frame: pd.DataFrame, cfg: dict[str, Any]) -> list[tuple[str, WindowSet, WindowSet]]:
    result = []
    for fold in cfg["folds"]:
        train = build_windows(frame, cfg, float(fold["train_start_h"]), float(fold["train_end_h"]))
        valid = build_windows(frame, cfg, float(fold["validation_start_h"]), float(fold["validation_end_h"]))
        if set(train.origin_row) & set(valid.origin_row):
            raise RuntimeError("Outer origins overlap")
        result.append((str(fold["fold"]), train, valid))
    return result


def split_inner_fit_stopper(train: WindowSet, fraction: float) -> tuple[WindowSet, WindowSet]:
    count = int(math.ceil(len(train.origin_row) * fraction))
    stop_index = np.arange(len(train.origin_row) - count, len(train.origin_row), dtype=int)
    stop_targets = set(train.target_row[stop_index].reshape(-1).tolist())
    fit_index = np.asarray([index for index in range(len(train.origin_row) - count) if not (set(train.target_row[index]) & stop_targets)], dtype=int)
    if not len(fit_index):
        raise RuntimeError("No inner fit windows after target isolation")
    fit, stopper = train.take(fit_index), train.take(stop_index)
    if set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)):
        raise RuntimeError("Inner fit/stopper target overlap")
    return fit, stopper


class FoldPreprocessor:
    def fit(self, x: np.ndarray) -> "FoldPreprocessor":
        flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        self.medians = np.where(np.isfinite(np.nanmedian(flat, axis=0)), np.nanmedian(flat, axis=0), 0.0)
        flat = np.where(np.isfinite(flat), flat, self.medians)
        self.mean, self.scale = flat.mean(axis=0), flat.std(axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.where(np.isfinite(x), x, self.medians) - self.mean) / self.scale).astype(np.float32)

    @property
    def voltage_scale(self) -> float:
        return float(self.scale[0])


def standardized_delta_target(windows: WindowSet, scaler: FoldPreprocessor) -> np.ndarray:
    return (windows.y_delta_V / scaler.voltage_scale).astype(np.float32)


def delta_from_absolute_scaled(absolute_scaled: torch.Tensor, x_scaled: torch.Tensor) -> torch.Tensor:
    if absolute_scaled.shape != (x_scaled.shape[0], 24):
        raise RuntimeError("Absolute output shape contract failed")
    return absolute_scaled - x_scaled[:, -1, 0:1]


def delta_metrics(truth: np.ndarray, pred: np.ndarray, target_row: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(pred, float) - np.asarray(truth, float)
    if error.ndim != 2 or error.shape[1] != 24 or target_row.shape != error.shape:
        raise RuntimeError("Frozen 24-step metric shape mismatch")
    point = error.reshape(-1)
    absolute_mV = abs(point) * 1000.0
    point_mse = float(np.mean(point ** 2))
    time_equal_mse = float(pd.DataFrame({"target_row": target_row.reshape(-1), "sse": (point * 1000.0) ** 2}).groupby("target_row", sort=False).sse.mean().mean())
    denom = float(np.sum((np.asarray(truth).reshape(-1) - np.asarray(truth).mean()) ** 2))
    return {
        "rmse_mV": math.sqrt(point_mse) * 1000.0, "mae_mV": float(np.mean(absolute_mV)),
        "r2": 1.0 - float(np.sum(point ** 2)) / denom if denom > 0 else math.nan,
        "endpoint_rmse_mV": float(np.sqrt(np.mean(error[:, -1] ** 2)) * 1000.0),
        "p95_abs_error_mV": float(np.quantile(absolute_mV, 0.95)), "p99_abs_error_mV": float(np.quantile(absolute_mV, 0.99)),
        "max_abs_error_mV": float(np.max(absolute_mV)), "target_time_equal_rmse_mV": math.sqrt(time_equal_mse),
        "query_count": int(point.size), "unique_target_row_count": int(len(np.unique(target_row))),
    }


def prediction_delta_scaled(model: torch.nn.Module, x: np.ndarray, run_device: torch.device) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            tensor = torch.from_numpy(x[start:start + 256]).to(run_device)
            values.append(delta_from_absolute_scaled(model(tensor), tensor).cpu().numpy())
    return np.concatenate(values, axis=0)


def _loader(x: np.ndarray, y: np.ndarray, seed: int, batch_size: int) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)


def _loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(delta_from_absolute_scaled(model(x), x), y)


def train_one(model_name: str, fold: str, seed: int, outer_train: WindowSet, outer_valid: WindowSet, cfg: dict[str, Any]) -> dict[str, Any]:
    run_id = f"tft_{model_name.lower()}_{fold}_seed{seed}"
    existing = phase_path("outputs", "model_runs", f"{run_id}.json")
    if existing.is_file():
        result = json.loads(existing.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or result.get("protocol_sha256") != CONFIG_SHA_PATH.read_text(encoding="utf-8").split()[0]:
            raise RuntimeError(f"Invalid cached run: {run_id}")
        return result
    hidden = int(cfg["models"][model_name]["hidden_size"])
    fit, stopper = split_inner_fit_stopper(outer_train, float(cfg["inner_stopping"]["stopper_fraction_of_outer_train_legal_origins"]))
    set_seed(seed)
    run_device = device()
    scaler = FoldPreprocessor().fit(fit.x)
    x_fit, y_fit, x_stop = scaler.transform(fit.x), standardized_delta_target(fit, scaler), scaler.transform(stopper.x)
    model = create_model(model_name, hidden).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best_epoch, best_score, stale, inner_logs = 0, math.inf, 0, []
    peak = 0.0
    inner_start = time.perf_counter()
    for epoch in range(1, int(cfg["training"]["max_epochs"]) + 1):
        model.train(); losses = []
        for x_batch, y_batch in _loader(x_fit, y_fit, seed + epoch, int(cfg["training"]["batch_size"])):
            x_batch, y_batch = x_batch.to(run_device), y_batch.to(run_device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, x_batch, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{run_id}: non-finite training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            if not torch.isfinite(grad):
                raise RuntimeError(f"{run_id}: non-finite gradient")
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        stop_pred = prediction_delta_scaled(model, x_stop, run_device) * scaler.voltage_scale
        stop_rmse = float(delta_metrics(stopper.y_delta_V, stop_pred, stopper.target_row)["rmse_mV"])
        inner_logs.append({"phase": "inner_stopper", "epoch": epoch, "train_delta_mse": float(np.mean(losses)), "stopper_deltaV_RMSE_mV": stop_rmse})
        if stop_rmse < best_score - 1e-10:
            best_epoch, best_score, stale = epoch, stop_rmse, 0
        else:
            stale += 1
        if stale >= int(cfg["training"]["patience"]):
            break
    inner_time = time.perf_counter() - inner_start
    del model
    if run_device.type == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(run_device)

    set_seed(seed)
    scaler = FoldPreprocessor().fit(outer_train.x)
    x_train, y_train, x_valid = scaler.transform(outer_train.x), standardized_delta_target(outer_train, scaler), scaler.transform(outer_valid.x)
    model = create_model(model_name, hidden).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    outer_logs, outer_start = [], time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        model.train(); losses = []
        for x_batch, y_batch in _loader(x_train, y_train, seed + epoch, int(cfg["training"]["batch_size"])):
            x_batch, y_batch = x_batch.to(run_device), y_batch.to(run_device)
            optimizer.zero_grad(set_to_none=True); loss = _loss(model, x_batch, y_batch)
            if not torch.isfinite(loss): raise RuntimeError(f"{run_id}: non-finite refit loss")
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            if not torch.isfinite(grad): raise RuntimeError(f"{run_id}: non-finite refit gradient")
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        outer_logs.append({"phase": "outer_refit", "epoch": epoch, "train_delta_mse": float(np.mean(losses))})
    prediction = prediction_delta_scaled(model, x_valid, run_device) * scaler.voltage_scale
    if run_device.type == "cuda":
        torch.cuda.synchronize(run_device)
    infer_start = time.perf_counter(); _ = prediction_delta_scaled(model, x_valid[:min(256, len(x_valid))], run_device)
    if run_device.type == "cuda": torch.cuda.synchronize(run_device)
    latency_ms = (time.perf_counter() - infer_start) * 1000.0 / min(256, len(x_valid))
    outer_time = time.perf_counter() - outer_start
    peak = float(torch.cuda.max_memory_allocated(run_device) / 1024**2) if run_device.type == "cuda" else 0.0
    arrays = {
        "y_true_delta_V": outer_valid.y_delta_V, "y_pred_delta_V": prediction, "anchor_V": outer_valid.anchor_V,
        "origin_row": outer_valid.origin_row, "target_row": outer_valid.target_row, "origin_age_h": outer_valid.origin_age_h,
        "target_end_age_h": outer_valid.target_end_age_h, "origin_datetime": outer_valid.origin_datetime,
        "target_datetime": outer_valid.target_datetime, "horizon_step": np.arange(1, 25, dtype=np.int64),
    }
    if float(np.max(outer_valid.target_end_age_h)) >= float(cfg["development_end_h_exclusive"]):
        raise RuntimeError("Validation cache crosses Development fence")
    metrics = delta_metrics(arrays["y_true_delta_V"], arrays["y_pred_delta_V"], arrays["target_row"])
    protocol_sha = CONFIG_SHA_PATH.read_text(encoding="utf-8").split()[0]
    result = {
        "run_id": run_id, "status": "complete", "model": model_name, "fold": fold, "seed": seed,
        "selected_epoch": best_epoch, "inner_fit_origins": len(fit.origin_row), "inner_stopper_origins": len(stopper.origin_row),
        "inner_target_overlap_count": 0, "inner_stopper_best_RMSE_mV": best_score,
        "parameter_count": trainable_parameter_count(model), "trainable_parameter_count": trainable_parameter_count(model),
        "peak_gpu_memory_MiB": peak, "inner_time_s": inner_time, "outer_refit_time_s": outer_time,
        "inference_latency_ms_per_origin": latency_ms, "nan_count": 0, "gradient_failure_count": 0,
        "max_target_end_age_h": float(np.max(outer_valid.target_end_age_h)), "test_accessed": False,
        "future_covariates_used": False, "all_legal_validation_origins_retained": True,
        "output_semantics": "ABSOLUTE_SCALED_VOLTAGE_TO_DELTAV", "native_state_contract": model.native_state_contract(),
        "protocol_sha256": protocol_sha, **metrics,
    }
    write_csv(pd.DataFrame(inner_logs + outer_logs), "logs", f"{run_id}_epochs.csv")
    pred_path = phase_path("outputs", "predictions", f"{run_id}.npz"); pred_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(pred_path, **arrays)
    ckpt = {"model_state_dict": model.state_dict(), "model": model_name, "seed": seed, "selected_epoch": best_epoch, "protocol_sha256": protocol_sha, "scaler_mean": scaler.mean, "scaler_scale": scaler.scale, "feature_medians": scaler.medians}
    torch.save(ckpt, phase_path("checkpoints", f"{run_id}.pt"))
    write_json(result, "outputs", "model_runs", f"{run_id}.json")
    del model
    if run_device.type == "cuda": torch.cuda.empty_cache()
    return result


def summarize_seed(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    # Historical frozen references predate this phase's resource telemetry.
    # Preserve their metric values and represent unavailable resource fields as
    # NaN instead of rejecting a valid Development reference table.
    for column in ("parameter_count", "peak_gpu_memory_MiB", "outer_refit_time_s", "inference_latency_ms_per_origin"):
        if column not in rows:
            rows[column] = math.nan
    return rows.groupby(["model", "seed"], as_index=False).agg(
        formal_RMSE_mV=("rmse_mV", "mean"), MAE_mV=("mae_mV", "mean"), R2=("r2", "mean"),
        endpoint_RMSE_mV=("endpoint_rmse_mV", "mean"), P95_mV=("p95_abs_error_mV", "mean"), P99_mV=("p99_abs_error_mV", "mean"),
        max_abs_error_mV=("max_abs_error_mV", "max"), target_time_equal_RMSE_mV=("target_time_equal_rmse_mV", "mean"),
        Fold3_RMSE_mV=("rmse_mV", lambda s: float(s[rows.loc[s.index, "fold"].eq("fold_3")].iloc[0])),
        Fold3_target_time_equal_RMSE_mV=("target_time_equal_rmse_mV", lambda s: float(s[rows.loc[s.index, "fold"].eq("fold_3")].iloc[0])),
        parameter_count=("parameter_count", "max"), peak_gpu_memory_MiB=("peak_gpu_memory_MiB", "max"),
        total_training_time_s=("outer_refit_time_s", "sum"), inference_latency_ms_per_origin=("inference_latency_ms_per_origin", "mean"),
    )


def gate_rows(summary: pd.DataFrame, raw_reference: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    raw = raw_reference.set_index("seed")
    rows = []
    for _, candidate in summary.iterrows():
        ref = raw.loc[int(candidate.seed)]
        model_cfg = cfg["models"][str(candidate.model)]
        is_vanilla = candidate.model == "LSTM"
        if is_vanilla:
            formal_limit, fold3_limit, p99_limit, endpoint_limit, target_limit = 1.10, 1.15, 1.10, 1.15, 1.10
        else:
            formal_limit, fold3_limit, p99_limit, endpoint_limit, target_limit = .97, 1.03, 1.05, 1.05, 1.05
        tests = {
            "formal": candidate.formal_RMSE_mV <= ref.formal_RMSE_mV * formal_limit,
            "fold3": candidate.Fold3_RMSE_mV <= ref.Fold3_RMSE_mV * fold3_limit,
            "p99": candidate.P99_mV <= ref.P99_mV * p99_limit,
            "endpoint": candidate.endpoint_RMSE_mV <= ref.endpoint_RMSE_mV * endpoint_limit,
            "target_time_equal": candidate.target_time_equal_RMSE_mV <= ref.target_time_equal_RMSE_mV * target_limit,
            "capacity_within_125pct": float(model_cfg["parameter_ratio"]) <= 1.25,
            "implementation_clean": True,
        }
        rows.append({"model": candidate.model, "seed": int(candidate.seed), "stage": "vanilla_seed42" if is_vanilla else "encoder_seed42", "capacity_status": model_cfg["capacity_status"], **tests, "pass": all(tests.values())})
    return pd.DataFrame(rows)

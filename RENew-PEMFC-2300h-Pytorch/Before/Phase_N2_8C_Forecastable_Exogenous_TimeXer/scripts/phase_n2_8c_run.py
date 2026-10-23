"""Development-only Phase N2.8C forecastable-exogenous TimeXer test.

The script is fail-closed: it reads F2/targets only from the development
prefix, evaluates two causal future-exogenous forecasters, then applies the
Oracle gate before any forecastable-exogenous TimeXer training.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import shutil
import sys
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
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8C_Forecastable_Exogenous_TimeXer")
CFG_PATH = ROOT / "configs" / "timexer_future_exo_frozen_config.yaml"
FOLDS = ("fold_1", "fold_2", "fold_3")


def ensure(path: Path) -> Path:
    path = path.resolve()
    root = ROOT.resolve()
    if path.parent != root and root not in path.parents:
        raise RuntimeError(f"Refusing an output outside Phase N2.8C: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    frame.to_csv(ensure(ROOT / relative), index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(value: Any, relative: str) -> None:
    ensure(ROOT / relative).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_yaml(value: dict[str, Any], relative: str) -> None:
    ensure(ROOT / relative).write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_text(value: str, relative: str) -> None:
    ensure(ROOT / relative).write_text(value.rstrip() + "\n", encoding="utf-8")


def table_markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "| status |\n|---|\n| no rows |"
    view = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def config() -> dict[str, Any]:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    if Path(cfg["phase_root"]).resolve() != ROOT.resolve():
        raise RuntimeError("Frozen N2.8C configuration points outside this phase")
    task = cfg["task"]
    if (task["resolution_hours"], task["lookback_steps"], task["horizon_steps"]) != (0.5, 48, 24):
        raise RuntimeError("Frozen task specification mismatch")
    features = cfg["features"]
    if len(features) != 14 or features[0] != "cell_voltage_mean":
        raise RuntimeError("F2 must remain exactly the frozen 14-column group")
    exo = cfg["future_exogenous_features"]
    if len(exo) != 9 or any(name not in features for name in exo):
        raise RuntimeError("Future-exogenous feature contract mismatch")
    if set(exo) & {"cell_voltage_mean", "cell_voltage_std", "cell_voltage_range", "cell_voltage_min", "cell_voltage_max"}:
        raise RuntimeError("Future voltage-state input is forbidden")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_common(cfg: dict[str, Any]):
    source = Path(cfg["common_script"])
    spec = importlib.util.spec_from_file_location("n28c_common", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen shared development loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_windows(cfg: dict[str, Any]):
    common = load_common(cfg)
    frame, loader_audit = common.load_development_data(cfg, cfg["features"])
    if loader_audit.get("target_rows_read_from_test") != 0 or frame["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL VIOLATION: Test entered N2.8C frame")
    windows = {fold: (train, validation) for fold, train, validation in common.fold_windows(frame, cfg)}
    for fold, (train, validation) in windows.items():
        if len(train.x) == 0 or len(validation.x) == 0:
            raise RuntimeError(f"No legal {fold} windows")
        if float(validation.target_end_age_h.max()) >= float(cfg["split"]["development_end_h"]):
            raise RuntimeError(f"TEST FIREWALL VIOLATION: validation target enters Test in {fold}")
    firewall = {
        **loader_audit,
        "target_rows_read_from_test": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "test_feature_diagnostics_computed": False,
        "test_performance_accessed": False,
        "enforcement": "fail_closed_development_prefix_only",
    }
    return common, frame, windows, firewall


def future_exogenous(frame: pd.DataFrame, windows: Any, cfg: dict[str, Any]) -> np.ndarray:
    exogenous = cfg["future_exogenous_features"]
    horizon = int(cfg["task"]["horizon_steps"])
    sequences: list[np.ndarray] = []
    for origin in windows.row_origin_index.astype(int):
        future = frame.iloc[origin + 1: origin + 1 + horizon].loc[:, exogenous].to_numpy(dtype=float)
        if future.shape != (horizon, len(exogenous)):
            raise RuntimeError("Future exogenous target incomplete")
        sequences.append(future)
    return np.asarray(sequences, dtype=np.float64)


class FutureExogenousScaler:
    """Fold-local median imputation and standardization for the nine exogenous variables."""

    def __init__(self) -> None:
        self.medians: np.ndarray | None = None
        self.scaler = StandardScaler()

    def fit(self, history: np.ndarray) -> "FutureExogenousScaler":
        flat = history.reshape(-1, history.shape[-1]).astype(np.float64)
        self.medians = np.nanmedian(flat, axis=0)
        repaired = np.where(np.isfinite(flat), flat, self.medians)
        self.scaler.fit(repaired)
        return self

    def repair(self, values: np.ndarray) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("Exogenous scaler has not been fitted")
        return np.where(np.isfinite(values), values, self.medians.reshape((1,) * (values.ndim - 1) + (-1,)))

    def transform(self, values: np.ndarray) -> np.ndarray:
        repaired = self.repair(values)
        flat = repaired.reshape(-1, repaired.shape[-1])
        return self.scaler.transform(flat).reshape(repaired.shape).astype(np.float32)


class RidgeMultiHorizonExogenous:
    """Fixed-alpha, joint 48x9 -> 24x9 Ridge forecaster fitted per fold."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)

    def fit(self, history_raw: np.ndarray, future_raw: np.ndarray, scaler: FutureExogenousScaler) -> "RidgeMultiHorizonExogenous":
        x = scaler.repair(history_raw).reshape(len(history_raw), -1)
        y = scaler.repair(future_raw).reshape(len(future_raw), -1)
        self.input_scaler.fit(x)
        self.output_scaler.fit(y)
        self.model.fit(self.input_scaler.transform(x), self.output_scaler.transform(y))
        return self

    def predict(self, history_raw: np.ndarray, scaler: FutureExogenousScaler) -> np.ndarray:
        x = scaler.repair(history_raw).reshape(len(history_raw), -1)
        y = self.output_scaler.inverse_transform(self.model.predict(self.input_scaler.transform(x)))
        return y.reshape(len(history_raw), 24, -1).astype(np.float64)


def cycle_repeat(history_raw: np.ndarray, period: int, horizon: int) -> np.ndarray:
    """Causal repeated cycle: for h>P, repeat the most recent legal P-position."""
    if history_raw.ndim != 3 or history_raw.shape[1] != 48:
        raise RuntimeError("CycleRepeat requires 48-step historical exogenous input")
    rows = []
    for h in range(1, horizon + 1):
        lag = period * math.ceil(h / period)
        relative_source = h - lag  # always <= 0; source is historical/current only
        source_index = (history_raw.shape[1] - 1) + relative_source
        if source_index < 0 or source_index >= history_raw.shape[1]:
            raise RuntimeError("CycleRepeat attempted an invalid causal source index")
        rows.append(history_raw[:, source_index, :])
    return np.stack(rows, axis=1).astype(np.float64)


def exogenous_metric_rows(model: str, fold: str, truth: np.ndarray, prediction: np.ndarray, features: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        y_true = truth[:, :, index].reshape(-1)
        y_pred = prediction[:, :, index].reshape(-1)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if finite.sum() < 2:
            rmse = mae = r2 = float("nan")
        else:
            rmse = float(np.sqrt(mean_squared_error(y_true[finite], y_pred[finite])))
            mae = float(mean_absolute_error(y_true[finite], y_pred[finite]))
            r2 = float(r2_score(y_true[finite], y_pred[finite]))
        rows.append({"model": model, "fold": fold, "variable": feature, "rmse": rmse, "mae": mae, "r2": r2, "n_values": int(finite.sum())})
    return rows


def load_timexer_module(vendor: Path):
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "layers" or name.startswith("layers."):
            del sys.modules[name]
    sys.path.insert(0, str(vendor))
    try:
        return __import__("models.TimeXer", fromlist=["Model"])
    finally:
        sys.path.remove(str(vendor))


class FutureExoTimeXerAdapter(nn.Module):
    """Official TimeXer plus the one permitted PEMFC future-exogenous adapter.

    Official TimeXer does not consume x_dec/x_mark_dec on its forecast path. This
    adapter maps nine 24-step future exogenous trajectories to 48-step context
    token traces and injects them through the existing official x_mark_enc input.
    """

    def __init__(self, vendor: Path, cfg: dict[str, Any]) -> None:
        super().__init__()
        module = load_timexer_module(vendor)
        tc = cfg["timexer"]
        official_cfg = SimpleNamespace(
            task_name="long_term_forecast", features="MS", seq_len=48, pred_len=24,
            use_norm=bool(tc["use_norm"]), patch_len=int(tc["patch_len"]), enc_in=14, c_out=1,
            d_model=int(tc["d_model"]), n_heads=int(tc["n_heads"]), e_layers=int(tc["e_layers"]),
            d_ff=int(tc["d_ff"]), dropout=float(tc["dropout"]), factor=5, embed="fixed", freq="h", activation="gelu",
        )
        self.model = module.Model(official_cfg)
        self.future_exogenous_adapter = nn.Linear(24, 48)

    def forward(self, x_history: torch.Tensor, z_future: torch.Tensor) -> torch.Tensor:
        if x_history.ndim != 3 or z_future.ndim != 3:
            raise RuntimeError("TimeXer adapter expects [batch,time,feature] tensors")
        if x_history.shape[1:] != (48, 14) or z_future.shape[1:] != (24, 9):
            raise RuntimeError("TimeXer adapter received a frozen-task shape mismatch")
        # Official MS mode requires its endogenous target in the last channel.
        ordered = torch.cat([x_history[:, :, 1:], x_history[:, :, :1]], dim=2)
        future_context = self.future_exogenous_adapter(z_future.transpose(1, 2)).transpose(1, 2)
        return self.model(ordered, future_context, None, None).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def train_one(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, z_train: np.ndarray,
              x_validation: np.ndarray, y_validation: np.ndarray, z_validation: np.ndarray,
              cfg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    train_cfg = cfg["training"]
    run_device = device()
    model = model.to(run_device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(z_train)),
        batch_size=int(train_cfg["batch_size"]), shuffle=True, drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    criterion = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    epochs_run = 0
    x_val = torch.from_numpy(x_validation).to(run_device)
    y_val = torch.from_numpy(y_validation).to(run_device)
    z_val = torch.from_numpy(z_validation).to(run_device)
    for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
        model.train()
        for x_batch, y_batch, z_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch.to(run_device), z_batch.to(run_device))
            loss = criterion(prediction, y_batch.to(run_device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(x_val, z_val)
            validation_rmse = float(torch.sqrt(criterion(validation_prediction, y_val)).item())
        epochs_run = epoch
        if validation_rmse < best_rmse - 1e-7:
            best_rmse = validation_rmse
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(train_cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(x_val, z_val).detach().cpu().numpy().astype(np.float64)
    return {
        "parameter_count": count_parameters(model), "best_epoch": best_epoch, "epochs_run": epochs_run,
        "device": str(run_device), "validation_scaled_delta_rmse": best_rmse,
    }, prediction


def delta_metrics(y_true_abs: np.ndarray, y_pred_abs: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    true_delta = np.asarray(y_true_abs, dtype=float) - np.asarray(anchor, dtype=float)[:, None]
    pred_delta = np.asarray(y_pred_abs, dtype=float) - np.asarray(anchor, dtype=float)[:, None]
    error = pred_delta - true_delta
    truth = true_delta.reshape(-1)
    estimate = pred_delta.reshape(-1)
    abs_error_mV = np.abs(error).reshape(-1) * 1000.0
    return {
        "r2_delta": float(r2_score(truth, estimate)),
        "rmse_delta_mV": float(np.sqrt(np.mean(error ** 2)) * 1000.0),
        "mae_delta_mV": float(np.mean(np.abs(error)) * 1000.0),
        "p95_mV": float(np.quantile(abs_error_mV, 0.95)),
        "p99_mV": float(np.quantile(abs_error_mV, 0.99)),
    }


def scale_fold(common: Any, train: Any, validation: Any, exo_indices: list[int]):
    preprocessor = common.FoldPreprocessor().fit(train.x)
    x_train = preprocessor.transform(train.x)
    x_validation = preprocessor.transform(validation.x)
    y_mean = float(np.mean(train.y_delta))
    y_std = float(np.std(train.y_delta))
    if not np.isfinite(y_std) or y_std <= 1e-10:
        raise RuntimeError("Degenerate train target scale")
    y_train = ((train.y_delta - y_mean) / y_std).astype(np.float32)
    y_validation = ((validation.y_delta - y_mean) / y_std).astype(np.float32)
    z_scaler = FutureExogenousScaler().fit(train.x[:, :, exo_indices])
    return preprocessor, x_train, x_validation, y_train, y_validation, y_mean, y_std, z_scaler


def run_timexer_variant(label: str, z_by_fold: dict[str, tuple[np.ndarray, np.ndarray]], common: Any,
                        windows: dict[str, tuple[Any, Any]], cfg: dict[str, Any], seed: int,
                        source: str, eligibility: str) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    vendor = Path(cfg["timexer"]["vendor_root"])
    exo_indices = [cfg["features"].index(name) for name in cfg["future_exogenous_features"]]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for fold in FOLDS:
        train, validation = windows[fold]
        preprocessor, x_train, x_val, y_train, y_val, y_mean, y_std, z_scaler = scale_fold(common, train, validation, exo_indices)
        z_train_raw, z_val_raw = z_by_fold[fold]
        z_train = z_scaler.transform(z_train_raw)
        z_val = z_scaler.transform(z_val_raw)
        set_seed(seed)
        model = FutureExoTimeXerAdapter(vendor, cfg)
        train_info, prediction_scaled = train_one(model, x_train, y_train, z_train, x_val, y_val, z_val, cfg)
        y_pred_abs = validation.anchor[:, None] + prediction_scaled * y_std + y_mean
        array = {
            "y_true": validation.y_abs.astype(np.float64), "y_pred": y_pred_abs.astype(np.float64),
            "anchor": validation.anchor.astype(np.float64), "target_end_age_h": validation.target_end_age_h.astype(np.float64),
            "row_origin_index": validation.row_origin_index.astype(np.int64), "z_future_raw": z_val_raw.astype(np.float64),
        }
        if float(array["target_end_age_h"].max()) >= float(cfg["split"]["development_end_h"]):
            raise RuntimeError("TEST FIREWALL VIOLATION in TimeXer prediction")
        np.savez_compressed(ensure(ROOT / f"outputs/model_runs/{label.lower().replace('-', '_')}_{fold}_seed{seed}_predictions.npz"), **array)
        rows.append({"model": label, "source": source, "eligibility": eligibility, "fold": fold, "seed": seed,
                     **delta_metrics(array["y_true"], array["y_pred"], array["anchor"]), **train_info})
        arrays[fold] = array
    return pd.DataFrame(rows), arrays


def legacy_original(cfg: dict[str, Any], windows: dict[str, tuple[Any, Any]]) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    root = Path(cfg["phase_n2_8a_root"]) / "outputs" / "model_screening"
    arrays: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        array = dict(np.load(root / f"timexer_official_{fold}_seed42_predictions.npz", allow_pickle=False))
        required = {"y_true", "y_pred", "anchor", "target_end_age_h", "row_origin_index"}
        if not required.issubset(array):
            raise RuntimeError("Frozen original TimeXer artifact is malformed")
        validation = windows[fold][1]
        if not np.array_equal(array["row_origin_index"], validation.row_origin_index) or not np.array_equal(array["y_true"], validation.y_abs):
            raise RuntimeError("Frozen original TimeXer windows do not match the N2.8C protocol")
        if float(array["target_end_age_h"].max()) >= float(cfg["split"]["development_end_h"]):
            raise RuntimeError("TEST FIREWALL VIOLATION in frozen original TimeXer")
        arrays[fold] = array
        rows.append({"model": "TimeXer-Original", "source": "frozen N2.8A official TimeXer seed42 output", "eligibility": "reference_only",
                     "fold": fold, "seed": 42, **delta_metrics(array["y_true"], array["y_pred"], array["anchor"]),
                     "parameter_count": np.nan, "best_epoch": "frozen", "epochs_run": np.nan, "device": "frozen", "validation_scaled_delta_rmse": np.nan})
    return pd.DataFrame(rows), arrays


def aggregate_model(rows: pd.DataFrame, status: str = "complete") -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    numeric = ["r2_delta", "rmse_delta_mV", "mae_delta_mV", "p95_mV", "p99_mV"]
    grouped = rows.groupby(["model", "source", "eligibility"], as_index=False)[numeric].mean()
    grouped = grouped.rename(columns={name: f"mean_{name}" for name in numeric})
    fold3 = rows.loc[rows["fold"].eq("fold_3"), ["model", "rmse_delta_mV", "p95_mV", "p99_mV"]].rename(columns={
        "rmse_delta_mV": "fold3_rmse_delta_mV", "p95_mV": "fold3_p95_mV", "p99_mV": "fold3_p99_mV"})
    fold1 = rows.loc[rows["fold"].eq("fold_1"), ["model", "rmse_delta_mV"]].rename(columns={"rmse_delta_mV": "fold1_rmse_delta_mV"})
    grouped = grouped.merge(fold3, on="model", how="left").merge(fold1, on="model", how="left")
    grouped["lifecycle_ratio"] = grouped["fold3_rmse_delta_mV"] / grouped["fold1_rmse_delta_mV"]
    grouped["status"] = status
    return grouped


def per_horizon_rows(model: str, arrays: dict[str, dict[str, np.ndarray]], cfg: dict[str, Any], eligibility: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, array in arrays.items():
        true_delta = array["y_true"] - array["anchor"][:, None]
        pred_delta = array["y_pred"] - array["anchor"][:, None]
        for step in range(true_delta.shape[1]):
            truth = true_delta[:, step]
            prediction = pred_delta[:, step]
            err = prediction - truth
            rows.append({"model": model, "eligibility": eligibility, "fold": fold, "horizon_step": step + 1,
                         "horizon_hours": (step + 1) * float(cfg["task"]["resolution_hours"]),
                         "r2_delta": float(r2_score(truth, prediction)),
                         "rmse_delta_mV": float(np.sqrt(np.mean(err ** 2)) * 1000.0),
                         "mae_delta_mV": float(np.mean(np.abs(err)) * 1000.0),
                         "is_9p5h": bool(step + 1 == 19), "is_12h": bool(step + 1 == 24)})
    return rows


def exo_error_analysis(model: str, arrays: dict[str, dict[str, np.ndarray]], truth_by_fold: dict[str, np.ndarray],
                       history_by_fold: dict[str, np.ndarray], exo_scaler_by_fold: dict[str, FutureExogenousScaler]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    all_origins: list[dict[str, Any]] = []
    for fold, array in arrays.items():
        truth = truth_by_fold[fold]
        predicted_z = array["z_future_raw"]
        scale = exo_scaler_by_fold[fold].scaler.scale_.reshape(1, 1, -1)
        z_error = np.sqrt(np.mean(((predicted_z - truth) / scale) ** 2, axis=(1, 2)))
        true_delta = array["y_true"] - array["anchor"][:, None]
        pred_delta = array["y_pred"] - array["anchor"][:, None]
        voltage_rmse = np.sqrt(np.mean((pred_delta - true_delta) ** 2, axis=1)) * 1000.0
        for index, (e_z, e_v) in enumerate(zip(z_error, voltage_rmse, strict=True)):
            row = {"model": model, "fold": fold, "origin_index": int(array["row_origin_index"][index]),
                   "exo_normalized_rmse": float(e_z), "voltage_rmse_delta_mV": float(e_v)}
            detail_rows.append(row)
            all_origins.append({**row, "array_index": index})
        if len(z_error) >= 3 and np.std(z_error) > 0 and np.std(voltage_rmse) > 0:
            pearson = float(pearsonr(z_error, voltage_rmse).statistic)
            spearman = float(spearmanr(z_error, voltage_rmse).statistic)
        else:
            pearson = spearman = float("nan")
        correlation_rows.append({"model": model, "fold": fold, "n_origins": len(z_error), "pearson": pearson, "spearman": spearman})
    detail = pd.DataFrame(detail_rows)
    for fold in FOLDS:
        subset = detail.loc[detail["fold"].eq(fold)].copy()
        if subset.empty:
            continue
        q50, q90 = subset["exo_normalized_rmse"].quantile([0.5, 0.9]).tolist()
        array = arrays[fold]
        true_delta = array["y_true"] - array["anchor"][:, None]
        pred_delta = array["y_pred"] - array["anchor"][:, None]
        error = (pred_delta - true_delta) * 1000.0
        for label, mask in (("Low_50pct", subset["exo_normalized_rmse"] <= q50),
                            ("Mid_40pct", (subset["exo_normalized_rmse"] > q50) & (subset["exo_normalized_rmse"] <= q90)),
                            ("Top_10pct", subset["exo_normalized_rmse"] > q90)):
            indices = np.flatnonzero(mask.to_numpy())
            values = error[indices].reshape(-1)
            if len(values) == 0:
                metrics = {"rmse_delta_mV": np.nan, "mae_delta_mV": np.nan, "p95_mV": np.nan, "p99_mV": np.nan}
            else:
                absolute = np.abs(values)
                metrics = {"rmse_delta_mV": float(np.sqrt(np.mean(values ** 2))), "mae_delta_mV": float(np.mean(absolute)),
                           "p95_mV": float(np.quantile(absolute, 0.95)), "p99_mV": float(np.quantile(absolute, 0.99))}
            tail_rows.append({"model": model, "fold": fold, "stratum": label, "n_origins": int(len(indices)), **metrics})
    return detail, pd.DataFrame(correlation_rows), pd.DataFrame(tail_rows)


def plot_outputs(exo_summary: pd.DataFrame, gain: pd.DataFrame, comparison: pd.DataFrame,
                 horizon: pd.DataFrame, origin_detail: pd.DataFrame, tail: pd.DataFrame) -> None:
    figures = ensure(ROOT / "outputs/figures/.keep").parent
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    for metric, filename, title in (("r2_mean", "future_exogenous_r2_by_variable.png", "Future exogenous R² by variable"),
                                    ("rmse_mean", "future_exogenous_rmse_by_variable.png", "Future exogenous RMSE by variable")):
        pivot = exo_summary.pivot(index="variable", columns="model", values=metric)
        ax = pivot.plot(kind="bar", figsize=(10, 4))
        ax.set_title(title); ax.set_xlabel("Variable"); ax.tick_params(axis="x", rotation=55); ax.legend(title="Forecaster")
        plt.tight_layout(); plt.savefig(figures / filename); plt.close()
    if not gain.empty:
        ax = gain.set_index("model")["gain_pct"].plot(kind="bar", figsize=(6, 4), color="#4472C4")
        ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel("Exogenous gain (%)"); ax.set_title("TimeXer exogenous gain")
        plt.tight_layout(); plt.savefig(figures / "timexer_exogenous_gain.png"); plt.close()
    valid = comparison.loc[comparison["status"].eq("complete")].copy()
    if not valid.empty:
        for column, filename, title in (("mean_r2_delta", "timexer_r2_delta_comparison.png", "TimeXer ΔV R² comparison"),
                                        ("mean_rmse_delta_mV", "timexer_rmse_delta_comparison.png", "TimeXer ΔV RMSE comparison")):
            ax = valid.set_index("model")[column].plot(kind="bar", figsize=(7, 4), color="#5B9BD5")
            ax.set_title(title); ax.tick_params(axis="x", rotation=35)
            plt.tight_layout(); plt.savefig(figures / filename); plt.close()
        oracle_forecast = valid.loc[valid["model"].isin(["TimeXer-OracleFutureExo", "TimeXer-CycleExo", "TimeXer-RidgeExo"])]
        if not oracle_forecast.empty:
            ax = oracle_forecast.set_index("model")["mean_rmse_delta_mV"].plot(kind="bar", figsize=(6, 4), color="#70AD47")
            ax.set_title("Oracle versus forecastable future exogenous input"); ax.set_ylabel("Mean RMSEΔV (mV)")
            plt.tight_layout(); plt.savefig(figures / "oracle_vs_forecastable_exo.png"); plt.close()
    if not horizon.empty and horizon["r2_delta"].notna().any():
        for column, filename, title in (("r2_delta", "per_horizon_r2_delta.png", "R²ΔV by forecast horizon"),
                                        ("rmse_delta_mV", "per_horizon_rmse_delta.png", "RMSEΔV by forecast horizon")):
            view = horizon.groupby(["model", "horizon_hours"], as_index=False)[column].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            for name, part in view.groupby("model"):
                ax.plot(part["horizon_hours"], part[column], marker="o", label=name)
            ax.axvline(9.5, color="gray", linestyle="--", linewidth=0.8); ax.axvline(12, color="gray", linestyle=":", linewidth=0.8)
            ax.set_title(title); ax.set_xlabel("Horizon (h)"); ax.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(figures / filename); plt.close()
    if not origin_detail.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, part in origin_detail.groupby("model"):
            ax.scatter(part["exo_normalized_rmse"], part["voltage_rmse_delta_mV"], s=12, alpha=0.55, label=name)
        ax.set_xlabel("Future exogenous normalized RMSE"); ax.set_ylabel("Origin ΔV RMSE (mV)"); ax.legend(fontsize=7)
        ax.set_title("Exogenous error versus voltage error")
        plt.tight_layout(); plt.savefig(figures / "exo_error_vs_voltage_error.png"); plt.close()
    if not tail.empty and tail["rmse_delta_mV"].notna().any():
        pivot = tail.groupby(["model", "stratum"], as_index=False)["rmse_delta_mV"].mean().pivot(index="stratum", columns="model", values="rmse_delta_mV")
        ax = pivot.plot(kind="bar", figsize=(7, 4)); ax.set_ylabel("RMSEΔV (mV)"); ax.set_title("Tail error by exogenous-error stratum")
        plt.tight_layout(); plt.savefig(figures / "tail_error_vs_exo_error.png"); plt.close()


def main() -> None:
    cfg = config()
    shutil.copy2(CFG_PATH, ensure(ROOT / "outputs/frozen_config/timexer_future_exo_frozen_config.yaml"))
    common, frame, windows, firewall = load_windows(cfg)
    write_json(firewall, "outputs/test_firewall_audit.json")
    exo_indices = [cfg["features"].index(name) for name in cfg["future_exogenous_features"]]
    raw_history: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    raw_future: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    future_quality_rows: list[dict[str, Any]] = []
    g_predictions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {"CycleRepeat": {}, "RidgeMultiHorizon": {}}
    z_scalers: dict[str, FutureExogenousScaler] = {}
    for fold in FOLDS:
        train, validation = windows[fold]
        history_train = train.x[:, :, exo_indices].astype(np.float64)
        history_val = validation.x[:, :, exo_indices].astype(np.float64)
        future_train = future_exogenous(frame, train, cfg)
        future_val = future_exogenous(frame, validation, cfg)
        raw_history[fold] = (history_train, history_val)
        raw_future[fold] = (future_train, future_val)
        z_scaler = FutureExogenousScaler().fit(history_train)
        z_scalers[fold] = z_scaler
        cycle_train = cycle_repeat(z_scaler.repair(history_train), int(cfg["future_exogenous"]["cycle_period_steps"]), int(cfg["task"]["horizon_steps"]))
        cycle_val = cycle_repeat(z_scaler.repair(history_val), int(cfg["future_exogenous"]["cycle_period_steps"]), int(cfg["task"]["horizon_steps"]))
        ridge = RidgeMultiHorizonExogenous(float(cfg["future_exogenous"]["ridge_alpha"])).fit(history_train, future_train, z_scaler)
        ridge_train = ridge.predict(history_train, z_scaler)
        ridge_val = ridge.predict(history_val, z_scaler)
        g_predictions["CycleRepeat"][fold] = (cycle_train, cycle_val)
        g_predictions["RidgeMultiHorizon"][fold] = (ridge_train, ridge_val)
        future_quality_rows.extend(exogenous_metric_rows("CycleRepeat", fold, future_val, cycle_val, cfg["future_exogenous_features"]))
        future_quality_rows.extend(exogenous_metric_rows("RidgeMultiHorizon", fold, future_val, ridge_val, cfg["future_exogenous_features"]))
    quality = pd.DataFrame(future_quality_rows)
    quality_summary = quality.groupby(["model", "variable"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"), folds=("fold", "nunique"), n_values=("n_values", "sum"))
    write_csv(quality, "outputs/exogenous_forecasting/future_exogenous_forecast_quality.csv")
    write_csv(quality_summary, "outputs/exogenous_forecasting/future_exogenous_forecast_summary.csv")

    original_rows, original_arrays = legacy_original(cfg, windows)
    oracle_z = {fold: raw_future[fold] for fold in FOLDS}
    oracle_rows, oracle_arrays = run_timexer_variant(
        "TimeXer-OracleFutureExo", oracle_z, common, windows, cfg, 42,
        "official TimeXer plus PEMFC future-exo x_mark_enc adapter using true development-validation future exogenous",
        "DIAGNOSTIC_ORACLE_ONLY")
    oracle_summary = aggregate_model(oracle_rows).iloc[0]
    oracle_rmse = float(oracle_summary["mean_rmse_delta_mV"])
    terminal_gate = oracle_rmse > float(cfg["thresholds"]["oracle_terminal_stop_rmse_mV"])

    all_rows = [original_rows, oracle_rows]
    all_arrays: dict[str, tuple[dict[str, dict[str, np.ndarray]], str]] = {
        "TimeXer-Original": (original_arrays, "reference_only"),
        "TimeXer-OracleFutureExo": (oracle_arrays, "DIAGNOSTIC_ORACLE_ONLY"),
    }
    skipped_models: list[str] = []
    if not terminal_gate:
        for model, forecaster, source in (
            ("TimeXer-CycleExo", "CycleRepeat", "future exogenous forecasted causally by CycleRepeat P=19"),
            ("TimeXer-RidgeExo", "RidgeMultiHorizon", "future exogenous forecasted causally by fold-local fixed-alpha multi-output Ridge"),
        ):
            z = {fold: g_predictions[forecaster][fold] for fold in FOLDS}
            rows, arrays = run_timexer_variant(model, z, common, windows, cfg, 42, source, "FORMAL_CANDIDATE")
            all_rows.append(rows)
            all_arrays[model] = (arrays, "FORMAL_CANDIDATE")
    else:
        skipped_models = ["TimeXer-CycleExo", "TimeXer-RidgeExo"]
    foldwise = pd.concat(all_rows, ignore_index=True)
    summaries = [aggregate_model(rows) for rows in all_rows]
    comparison = pd.concat(summaries, ignore_index=True)
    for model in skipped_models:
        comparison = pd.concat([comparison, pd.DataFrame([{
            "model": model, "source": "not run: Oracle terminal-stop gate", "eligibility": "FORMAL_CANDIDATE",
            "mean_r2_delta": np.nan, "mean_rmse_delta_mV": np.nan, "mean_mae_delta_mV": np.nan,
            "mean_p95_mV": np.nan, "mean_p99_mV": np.nan, "fold3_rmse_delta_mV": np.nan,
            "fold3_p95_mV": np.nan, "fold3_p99_mV": np.nan, "fold1_rmse_delta_mV": np.nan,
            "lifecycle_ratio": np.nan, "status": "not_run_oracle_terminal_stop"}])], ignore_index=True)

    original_rmse = float(comparison.loc[comparison["model"].eq("TimeXer-Original"), "mean_rmse_delta_mV"].iloc[0])
    gain_rows = []
    for model in ("TimeXer-CycleExo", "TimeXer-RidgeExo", "TimeXer-OracleFutureExo"):
        matched = comparison.loc[comparison["model"].eq(model)].iloc[0]
        rmse = matched["mean_rmse_delta_mV"]
        gain = float((original_rmse - rmse) / original_rmse * 100.0) if pd.notna(rmse) else np.nan
        gain_rows.append({"model": model, "eligibility": matched["eligibility"], "mean_rmse_delta_mV": rmse,
                          "original_mean_rmse_delta_mV": original_rmse, "gain_pct": gain, "status": matched["status"]})
    gain = pd.DataFrame(gain_rows)

    horizon_rows: list[dict[str, Any]] = []
    for model, (arrays, eligibility) in all_arrays.items():
        horizon_rows.extend(per_horizon_rows(model, arrays, cfg, eligibility))
    horizon = pd.DataFrame(horizon_rows)

    detail_frames: list[pd.DataFrame] = []
    corr_frames: list[pd.DataFrame] = []
    tail_frames: list[pd.DataFrame] = []
    for model, forecaster in (("TimeXer-CycleExo", "CycleRepeat"), ("TimeXer-RidgeExo", "RidgeMultiHorizon")):
        if model not in all_arrays:
            continue
        arrays = all_arrays[model][0]
        truths = {fold: raw_future[fold][1] for fold in FOLDS}
        histories = {fold: raw_history[fold][1] for fold in FOLDS}
        detail, correlations, tail = exo_error_analysis(model, arrays, truths, histories, z_scalers)
        detail_frames.append(detail); corr_frames.append(correlations); tail_frames.append(tail)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(columns=["model", "fold", "origin_index", "exo_normalized_rmse", "voltage_rmse_delta_mV"])
    correlations = pd.concat(corr_frames, ignore_index=True) if corr_frames else pd.DataFrame(columns=["model", "fold", "n_origins", "pearson", "spearman"])
    tail = pd.concat(tail_frames, ignore_index=True) if tail_frames else pd.DataFrame(columns=["model", "fold", "stratum", "n_origins", "rmse_delta_mV", "mae_delta_mV", "p95_mV", "p99_mV"])
    if correlations.empty:
        correlations = pd.DataFrame([{"model": "not_run", "fold": "all", "n_origins": 0, "pearson": np.nan, "spearman": np.nan,
                                      "status": "not_run_oracle_terminal_stop"}])
    else:
        correlations["status"] = "complete"
    if tail.empty:
        tail = pd.DataFrame([{"model": "not_run", "fold": "all", "stratum": "not_run_oracle_terminal_stop", "n_origins": 0,
                              "rmse_delta_mV": np.nan, "mae_delta_mV": np.nan, "p95_mV": np.nan, "p99_mV": np.nan}])

    formal = comparison.loc[(comparison["eligibility"].eq("FORMAL_CANDIDATE")) & comparison["mean_rmse_delta_mV"].notna()].copy()
    promote = formal.loc[formal["mean_rmse_delta_mV"] <= float(cfg["thresholds"]["three_seed_promotion_rmse_mV"])]
    three_seed_rows: list[pd.DataFrame] = []
    if not promote.empty:
        for model in promote["model"].tolist():
            forecaster = "CycleRepeat" if model.endswith("CycleExo") else "RidgeMultiHorizon"
            z = {fold: g_predictions[forecaster][fold] for fold in FOLDS}
            for seed in (2024, 3407):
                rows, _ = run_timexer_variant(model, z, common, windows, cfg, seed,
                                              "same formal model, additional mandatory seed", "FORMAL_CANDIDATE")
                three_seed_rows.append(rows)
        foldwise = pd.concat([foldwise, *three_seed_rows], ignore_index=True)
    multi_rows: list[dict[str, Any]] = []
    for model in ["TimeXer-CycleExo", "TimeXer-RidgeExo"]:
        subset = foldwise.loc[foldwise["model"].eq(model)]
        if subset.empty:
            multi_rows.append({"model": model, "multiseed_status": "not_run_oracle_terminal_stop", "n_seeds": 0, "seeds": "", "mean_r2_delta": np.nan,
                               "mean_rmse_delta_mV": np.nan, "rmse_delta_std_mV": np.nan})
        elif subset["seed"].nunique() == 1:
            averaged = subset.groupby("seed", as_index=False)[["r2_delta", "rmse_delta_mV"]].mean().iloc[0]
            multi_rows.append({"model": model, "multiseed_status": "not_promoted_seed42_only", "n_seeds": 1, "seeds": "42",
                               "mean_r2_delta": float(averaged["r2_delta"]), "mean_rmse_delta_mV": float(averaged["rmse_delta_mV"]), "rmse_delta_std_mV": 0.0})
        else:
            by_seed = subset.groupby("seed", as_index=False)[["r2_delta", "rmse_delta_mV"]].mean()
            multi_rows.append({"model": model, "multiseed_status": "promoted_three_seed", "n_seeds": int(by_seed.seed.nunique()),
                               "seeds": ";".join(map(str, by_seed.seed.astype(int).tolist())), "mean_r2_delta": float(by_seed.r2_delta.mean()),
                               "mean_rmse_delta_mV": float(by_seed.rmse_delta_mV.mean()), "rmse_delta_std_mV": float(by_seed.rmse_delta_mV.std(ddof=0))})
    multiseed = pd.DataFrame(multi_rows)

    write_csv(foldwise, "outputs/timexer_screening/foldwise_timexer_future_exo_metrics.csv")
    write_csv(oracle_rows, "outputs/oracle_analysis/timexer_oracle_future_exo_seed42.csv")
    if "TimeXer-CycleExo" in all_arrays:
        write_csv(foldwise.loc[foldwise["model"].eq("TimeXer-CycleExo")], "outputs/timexer_screening/timexer_cycle_exo_seed42.csv")
    else:
        write_csv(pd.DataFrame([{"model": "TimeXer-CycleExo", "status": "not_run_oracle_terminal_stop", "reason": "Oracle RMSEDeltaV exceeded 20 mV"}]),
                  "outputs/timexer_screening/timexer_cycle_exo_seed42.csv")
    if "TimeXer-RidgeExo" in all_arrays:
        write_csv(foldwise.loc[foldwise["model"].eq("TimeXer-RidgeExo")], "outputs/timexer_screening/timexer_ridge_exo_seed42.csv")
    else:
        write_csv(pd.DataFrame([{"model": "TimeXer-RidgeExo", "status": "not_run_oracle_terminal_stop", "reason": "Oracle RMSEDeltaV exceeded 20 mV"}]),
                  "outputs/timexer_screening/timexer_ridge_exo_seed42.csv")
    write_csv(comparison, "outputs/timexer_screening/N2_8C_model_comparison.csv")
    write_csv(multiseed, "outputs/timexer_screening/N2_8C_multiseed_comparison.csv")
    write_csv(gain, "outputs/timexer_screening/timexer_exogenous_gain.csv")
    write_csv(horizon, "outputs/per_horizon/timexer_future_exo_per_horizon.csv")
    write_csv(correlations, "outputs/error_correlation/exo_error_vs_voltage_error.csv")
    write_csv(tail, "outputs/tail_analysis/exo_error_stratified_voltage_metrics.csv")

    plot_outputs(quality_summary, gain, comparison, horizon, detail, tail)
    if terminal_gate:
        status = "TERMINAL_STOP"
        scientific = "Oracle future exogenous context did not reduce development RMSEΔV below 20 mV. Missing forecastable future exogenous input is not a sufficient explanation for TimeXer failure on this task."
        next_phase = "Lifecycle-Aware Robust TimesNet; do not continue the TimeXer route."
    elif formal.empty:
        status = "TERMINAL_STOP"
        scientific = "No formal forecastable-exogenous TimeXer candidate completed after the Oracle stage."
        next_phase = "Lifecycle-Aware Robust TimesNet."
    else:
        best = formal.sort_values("mean_rmse_delta_mV").iloc[0]
        best_rmse = float(best["mean_rmse_delta_mV"])
        if best_rmse <= float(cfg["thresholds"]["three_seed_promotion_rmse_mV"]):
            status = "REVIVED"
            scientific = "Forecastable future exogenous context restored TimeXer to the pre-registered competitive threshold."
            next_phase = "PEMFC-specific Forecastable-Exogenous TimeXer, subject to frozen three-seed confirmation."
        elif best_rmse < float(cfg["thresholds"]["structural_recovery_rmse_mV"]):
            status = "PROMISING_BUT_LIMITED"
            scientific = "Future exogenous context materially helps TimeXer, but it does not meet the TimesNet-competitive threshold."
            next_phase = "Retain TimeXer only as a discussion/control result; Lifecycle-Aware Robust TimesNet remains the main direction."
        else:
            status = "TERMINAL_STOP"
            scientific = "Forecastable future exogenous context did not meet the structural-recovery threshold."
            next_phase = "Lifecycle-Aware Robust TimesNet; do not continue the TimeXer route."
    completed_formal = formal.sort_values("mean_rmse_delta_mV").iloc[0] if not formal.empty else None
    corr_summary = correlations.loc[correlations["status"].eq("complete")].groupby("model", as_index=False)[["pearson", "spearman"]].mean() if "status" in correlations else pd.DataFrame()
    best_corr = corr_summary.sort_values("model").iloc[0] if not corr_summary.empty else None
    def scalar_or_none(value: Any) -> float | None:
        return None if value is None or pd.isna(value) else float(value)

    def summary_value(model: str, column: str) -> float | None:
        values = comparison.loc[comparison.model.eq(model), column]
        return None if values.empty else scalar_or_none(values.iloc[0])

    decision = {
        "phase_status": "PASS", "hypothesis": {"timexer_failure_due_to_missing_future_exogenous": not terminal_gate},
        "original_timexer": {"r2_delta": float(comparison.loc[comparison.model.eq("TimeXer-Original"), "mean_r2_delta"].iloc[0]), "rmse_delta_mV": original_rmse},
        "oracle_future_exo": {"r2_delta": float(oracle_summary["mean_r2_delta"]), "rmse_delta_mV": oracle_rmse,
                              "gain_pct": float(gain.loc[gain.model.eq("TimeXer-OracleFutureExo"), "gain_pct"].iloc[0])},
        "cycle_exo": {"r2_delta": summary_value("TimeXer-CycleExo", "mean_r2_delta"),
                      "rmse_delta_mV": summary_value("TimeXer-CycleExo", "mean_rmse_delta_mV"),
                      "gain_pct": scalar_or_none(gain.loc[gain.model.eq("TimeXer-CycleExo"), "gain_pct"].iloc[0])},
        "ridge_exo": {"r2_delta": summary_value("TimeXer-RidgeExo", "mean_r2_delta"),
                      "rmse_delta_mV": summary_value("TimeXer-RidgeExo", "mean_rmse_delta_mV"),
                      "gain_pct": scalar_or_none(gain.loc[gain.model.eq("TimeXer-RidgeExo"), "gain_pct"].iloc[0])},
        "best_forecastable_exogenous_method": None if completed_formal is None else completed_formal.model,
        "best_forecastable_timexer_r2_delta": None if completed_formal is None else float(completed_formal.mean_r2_delta),
        "best_forecastable_timexer_rmse_delta_mV": None if completed_formal is None else float(completed_formal.mean_rmse_delta_mV),
        "fold3_rmse_delta_mV": None if completed_formal is None else float(completed_formal.fold3_rmse_delta_mV),
        "p95_mV": None if completed_formal is None else float(completed_formal.mean_p95_mV),
        "p99_mV": None if completed_formal is None else float(completed_formal.mean_p99_mV),
        "exo_error_voltage_error_pearson": None if best_corr is None else float(best_corr.pearson),
        "exo_error_voltage_error_spearman": None if best_corr is None else float(best_corr.spearman),
        "three_seed_promoted": bool(not promote.empty), "timexer_status": status,
        "scientific_conclusion": scientific, "next_phase": next_phase, "test_performance_accessed": False,
    }
    write_yaml(decision, "PHASE_N2_8C_DECISION.yaml")
    audit = """# TimeXer future-exogenous adapter audit

## Pinned implementation

The base implementation is `thuml/TimeXer` at commit `76011909357972bd55a27adba2e1be994d81b327`, read only from the previously frozen vendor copy.

## Native-interface finding

The official `Model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)` implementation uses `x_enc` and `x_mark_enc`. It does not consume `x_dec` or `x_mark_dec`; therefore official TimeXer has no native future-exogenous forecasting interface in this path.

## PEMFC-specific adapter

`FutureExoTimeXerAdapter` leaves the official TimeXer encoder, self/cross attention, endogenous patching, historical exogenous embedding, and forecasting head untouched. The adapter applies one shared linear map from 24 future steps to 48 encoder-context steps independently for each of the nine permitted predicted future exogenous variables. The resulting 48-step traces are passed to the official `x_mark_enc` input, which the official `DataEmbedding_inverted` code already turns into exogenous tokens. This is a PEMFC-specific adapter, not native official future-exogenous behavior.

## Causality

Oracle uses true future exogenous values only for Development Validation and is labelled `DIAGNOSTIC_ORACLE_ONLY`. CycleRepeat and Ridge use future exogenous values predicted causally from history. Ridge preprocessing, imputation, and fitting are Train-only and fold-local. No Test values enter any stage.
"""
    write_text(audit, "TIMEXER_FUTURE_EXO_ADAPTER_AUDIT.md")
    quality_best = quality_summary.sort_values("r2_mean", ascending=False).iloc[0]
    quality_worst = quality_summary.sort_values("r2_mean", ascending=True).iloc[0]
    horizon_summary = horizon.groupby(["model", "horizon_hours"], as_index=False)[["r2_delta", "rmse_delta_mV", "mae_delta_mV"]].mean() if not horizon.empty else pd.DataFrame()
    def horizon_line(hours: float) -> str:
        part = horizon_summary.loc[np.isclose(horizon_summary["horizon_hours"], hours)] if not horizon_summary.empty else pd.DataFrame()
        return table_markdown(part, ["model", "horizon_hours", "r2_delta", "rmse_delta_mV", "mae_delta_mV"])
    report = f"""# Phase N2.8C results

## Firewall and frozen task

All work used only the development prefix of the frozen 0.5 h / 48-step / 24-step / F2 / DeltaV task. Test target rows read: 0. No Test forecast, metric, feature diagnostic, or model-selection result was constructed.

## Q1: Can missing future exogenous context explain TimeXer failure?

{scientific}

## Q2–Q5: Future exogenous forecast quality

{table_markdown(quality_summary, ["model", "variable", "rmse_mean", "mae_mean", "r2_mean", "folds"])}

Best mean R²: `{quality_best.model}` / `{quality_best.variable}` = {quality_best.r2_mean:.6f}. Worst mean R²: `{quality_worst.model}` / `{quality_worst.variable}` = {quality_worst.r2_mean:.6f}.

## Q6–Q11: TimeXer comparison and exogenous gain

{table_markdown(comparison, ["model", "eligibility", "mean_r2_delta", "mean_rmse_delta_mV", "mean_mae_delta_mV", "fold3_rmse_delta_mV", "mean_p95_mV", "mean_p99_mV", "status"])}

{table_markdown(gain, ["model", "eligibility", "mean_rmse_delta_mV", "gain_pct", "status"])}

Oracle is `DIAGNOSTIC_ORACLE_ONLY` and never participates in formal ranking or three-seed promotion.

## Q12–Q13: Exogenous error versus voltage error

{table_markdown(correlations, ["model", "fold", "n_origins", "pearson", "spearman", "status"])}

{table_markdown(tail, ["model", "fold", "stratum", "n_origins", "rmse_delta_mV", "mae_delta_mV", "p95_mV", "p99_mV"])}

## Q14–Q16: Fold3 and key horizons

Fold3 information is in the comparison table. At 9.5 h:

{horizon_line(9.5)}

At 12 h:

{horizon_line(12.0)}

## Q17–Q20: Pre-registered decision

- Oracle RMSEDeltaV = {oracle_rmse:.6f} mV. Oracle terminal gate (>20 mV): `{terminal_gate}`.
- Formal structural recovery (<20 mV): `{bool(not formal.empty and float(formal.mean_rmse_delta_mV.min()) < 20.0)}`.
- Three-seed threshold (<=11.9 mV) reached: `{bool(not promote.empty)}`.
- Final TimeXer status: **{status}**.
- Next-phase recommendation: **{next_phase}**

## Frozen artifacts

- `PHASE_N2_8C_DECISION.yaml`
- `TIMEXER_FUTURE_EXO_ADAPTER_AUDIT.md`
- development-only CSV artifacts under `outputs/`
"""
    write_text(report, "PHASE_N2_8C_RESULTS.md")
    print(json.dumps({"phase_status": "PASS", "oracle_rmse_delta_mV": oracle_rmse, "terminal_gate": terminal_gate,
                      "timexer_status": status, "three_seed_promoted": bool(not promote.empty)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

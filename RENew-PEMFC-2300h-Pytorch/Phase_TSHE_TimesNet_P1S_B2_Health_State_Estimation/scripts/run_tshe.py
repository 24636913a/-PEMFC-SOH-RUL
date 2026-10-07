"""Leakage-safe TimesNet health-state estimation with frozen P1S-B2 supervision.

The only learned model in this phase is the already frozen TimesNet encoder plus
a scalar temporal-mean-pooling regression head.  Source artifacts are read
only; every newly created artifact stays inside Phase TSHE.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml


sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
CONFIG_PATH = PHASE / "configs" / "TSHE_FROZEN_TASK_CONFIG.yaml"
OUT = PHASE / "outputs"
FIG = PHASE / "figures"
REPORT = PHASE / "reports"
CHECKPOINTS = PHASE / "checkpoints"
BS_PHASE = PROJECT / "Phase_BS_Final_Backbone_Screening"
BS_PROTOCOL = BS_PHASE / "configs" / "PREREGISTERED_PROTOCOL.yaml"
BS_PROTOCOL_SHA = BS_PHASE / "configs" / "PREREGISTERED_PROTOCOL.sha256"
CAUSAL = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
P1S = PROJECT / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs" / "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv"
CYCLE_CATALOG = PROJECT / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit" / "outputs" / "RAW_CYCLE_CATALOG.csv"
B2_MODELS = PROJECT / "Phase_AHC_TV_One_Time_Late_Life_Test_Protocol" / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml"
VENDOR = BS_PHASE / "vendor" / "TimesNet_official"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral": "#CFCECE",
}
FEATURES = [
    "cell_voltage_mean", "cell_voltage_std", "cell_voltage_range", "cell_voltage_min", "cell_voltage_max",
    "actual_current_A_mean", "actual_current_A_std", "actual_current_A_peak_to_peak",
    "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std",
    "actual_current_density_A_cm2_peak_to_peak", "load_transition_count", "operation_point_transitions",
    "zero_current_fraction",
]
FOLDS = (
    ("fold_1", 72.788690, 946.393214, 946.393214, 1164.794345),
    ("fold_2", 72.788690, 1164.794345, 1164.794345, 1383.195476),
    ("fold_3", 72.788690, 1383.195476, 1383.195476, 1819.997738),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    return "| " + " | ".join(shown.columns) + " |\n|" + "|".join(["---"] * len(shown.columns)) + "|\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runtime_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_frozen_authority() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_head = "85d0d6027de5949ff1ad13177a1041d64d54dd0a"
    observed_head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if observed_head != expected_head:
        raise RuntimeError(f"TSHE base HEAD mismatch: {observed_head}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("base_commit") != expected_head or cfg.get("backbone") != "TimesNet":
        raise RuntimeError("TSHE frozen task configuration mismatch")
    if list(cfg["input"]["feature_order"]) != FEATURES or cfg["input"]["shape"] != [48, 14]:
        raise RuntimeError("Frozen Raw F2 input contract mismatch")
    bs_raw = BS_PROTOCOL.read_bytes()
    bs_expected = BS_PROTOCOL_SHA.read_text(encoding="utf-8").split()[0].lower()
    if hashlib.sha256(bs_raw).hexdigest() != bs_expected:
        raise RuntimeError("Frozen BS protocol SHA256 mismatch")
    bs = yaml.safe_load(bs_raw)
    official = bs["official_sources"]["TimesNet"]["config"]
    frozen_expected = {
        "seq_len": 48, "enc_in": 14, "d_model": 32, "d_ff": 64, "e_layers": 1,
        "top_k": 2, "num_kernels": 3, "dropout": 0.1, "embed": "fixed", "freq": "h",
    }
    if {key: official.get(key) for key in frozen_expected} != frozen_expected:
        raise RuntimeError("TimesNet backbone configuration drift")
    training = bs["training"]
    expected_training = {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 1e-5,
                         "batch_size": 64, "max_epochs": 300, "patience": 30,
                         "gradient_clip_norm": 1.0, "seeds": [42, 2024, 3407]}
    if {key: training.get(key) for key in expected_training} != expected_training:
        raise RuntimeError("Frozen TimesNet training contract drift")
    b2 = yaml.safe_load(B2_MODELS.read_text(encoding="utf-8"))
    p1s_affine = b2["P1S_affine"]
    if not math.isclose(float(p1s_affine["slope"]), float(cfg["target"]["slope"]), abs_tol=1e-15):
        raise RuntimeError("Frozen B2 slope mismatch")
    if not math.isclose(float(p1s_affine["intercept"]), float(cfg["target"]["intercept"]), abs_tol=1e-15):
        raise RuntimeError("Frozen B2 intercept mismatch")
    required = (CAUSAL, P1S, CYCLE_CATALOG, B2_MODELS, VENDOR / "models" / "TimesNet.py")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("A required frozen TSHE input artifact is unavailable")
    return cfg, bs, b2


@dataclass
class Standardizer:
    mean: np.ndarray | float | None = None
    scale: np.ndarray | float | None = None

    def fit(self, data: np.ndarray) -> "Standardizer":
        value = np.asarray(data, dtype=np.float64)
        self.mean = value.mean(axis=tuple(range(value.ndim - 1))) if value.ndim > 1 else float(value.mean())
        self.scale = value.std(axis=tuple(range(value.ndim - 1))) if value.ndim > 1 else float(value.std())
        self.scale = np.where(np.asarray(self.scale) > 1e-8, self.scale, 1.0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Attempted to transform before train-only scaler fit")
        return ((np.asarray(data, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Attempted to inverse-transform before train-only scaler fit")
        return (np.asarray(data, dtype=np.float64) * self.scale + self.mean).astype(np.float64)


class TimesNetHealth(nn.Module):
    """Frozen official TimesBlock/embedding encoder with the sole permitted scalar head."""

    def __init__(self, official_config: dict[str, Any]) -> None:
        super().__init__()
        for name in list(sys.modules):
            if name == "models" or name.startswith("models.") or name == "layers" or name.startswith("layers."):
                del sys.modules[name]
        sys.path.insert(0, str(VENDOR))
        try:
            official = importlib.import_module("models.TimesNet")
            embed = importlib.import_module("layers.Embed")
        finally:
            sys.path.remove(str(VENDOR))
        # pred_len=0 removes only the old forecasting contract; the official encoder body remains unchanged.
        args = SimpleNamespace(**({**official_config, "label_len": 0, "num_class": 0, "c_out": 14,
                                   "task_name": "health_regression", "pred_len": 0}))
        self.seq_len = int(args.seq_len)
        self.enc_embedding = embed.DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.blocks = nn.ModuleList([official.TimesBlock(args) for _ in range(args.e_layers)])
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.health_head = nn.Linear(args.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (48, 14):
            raise RuntimeError(f"TimesNet health input shape violation: {tuple(x.shape)}")
        means = x.mean(dim=1, keepdim=True).detach()
        normalized = x.sub(means)
        stdev = torch.sqrt(torch.var(normalized, dim=1, keepdim=True, unbiased=False) + 1e-5)
        encoded = normalized.div(stdev)
        encoded = self.enc_embedding(encoded, None)
        for block in self.blocks:
            encoded = self.layer_norm(block(encoded))
        return self.health_head(encoded.mean(dim=1)).squeeze(-1)


def event_membership(age: float) -> str:
    labels: list[str] = []
    for name, train_start, train_end, val_start, val_end in FOLDS:
        if train_start <= age < train_end:
            labels.append(f"{name}_train")
        if val_start <= age < val_end:
            labels.append(f"{name}_validation")
    if 1819.997738 <= age <= 2256.8:
        labels.append("posthoc_late_life_holdout")
    return "|".join(labels) if labels else "outside_TSHE_ranges"


def build_event_dataset(slope: float, intercept: float) -> tuple[pd.DataFrame, list[np.ndarray]]:
    columns = ["bin_start_ophrs2_h", "bin_end_ophrs2_h", "bin_valid_for_windows", "temporal_segment_id", *FEATURES]
    causal = pd.read_csv(CAUSAL, usecols=columns).reset_index(names="causal_row")
    if not causal["bin_start_ophrs2_h"].is_monotonic_increasing:
        raise RuntimeError("Causal feature table is not chronologically sorted")
    p1s = pd.read_csv(P1S)
    p1s = p1s.loc[(p1s["interval_kind"] == "COMPLETE_CYCLE_INTERVAL") & (p1s["stable_window"] == "middle_50")].copy()
    p1s["raw_cycle_id"] = p1s["raw_cycle_id"].astype(int)
    if p1s["raw_cycle_id"].duplicated().any():
        raise RuntimeError("Primary P1S complete-cycle labels are not unique per raw cycle")
    catalog = pd.read_csv(CYCLE_CATALOG, usecols=["raw_cycle_id", "start_ophrs2_h", "end_ophrs2_h", "complete_cycle"])
    catalog["raw_cycle_id"] = catalog["raw_cycle_id"].astype(int)
    p1s = p1s.merge(catalog, on="raw_cycle_id", how="left", validate="one_to_one")
    if p1s[["start_ophrs2_h", "end_ophrs2_h"]].isna().any().any() or not p1s["complete_cycle"].astype(bool).all():
        raise RuntimeError("Target P1S labels do not map one-to-one to complete raw cycles")
    records: list[dict[str, Any]] = []
    legal_x: list[np.ndarray] = []
    for target in p1s.sort_values("measurement_end_ophrs2_h").itertuples(index=False):
        target_start = float(target.start_ophrs2_h)
        earlier = causal.index[causal["bin_end_ophrs2_h"].to_numpy(float) < target_start]
        base = {
            "cycle_interval_id": str(target.cycle_interval_id),
            "raw_cycle_id": int(target.raw_cycle_id),
            "measurement_end_ophrs2_h": float(target.measurement_end_ophrs2_h),
            "target_cycle_start_ophrs2_h": target_start,
            "target_cycle_end_ophrs2_h": float(target.end_ophrs2_h),
            "target_P1S_retention": float(target.P1S_operating_voltage_retention),
            "target_SOH_P1S_B2": float(slope * float(target.P1S_operating_voltage_retention) + intercept),
            "fold_membership": event_membership(float(target.measurement_end_ophrs2_h)),
            "stable_window": "middle_50",
            "interval_kind": "COMPLETE_CYCLE_INTERVAL",
        }
        reason = ""
        start_index = end_index = None
        if len(earlier) == 0:
            reason = "NO_CAUSAL_BIN_BEFORE_TARGET_CYCLE_START"
        else:
            end_index = int(earlier[-1])
            start_index = end_index - 47
            if start_index < 0:
                reason = "INSUFFICIENT_48_STEP_HISTORY"
            else:
                window = causal.iloc[start_index:end_index + 1]
                reasons: list[str] = []
                if not window["bin_valid_for_windows"].astype(bool).all():
                    reasons.append("INVALID_BIN")
                if window["temporal_segment_id"].nunique() != 1:
                    reasons.append("CROSSES_TEMPORAL_SEGMENT")
                if not np.allclose(np.diff(window["bin_start_ophrs2_h"].to_numpy(float)), 0.5, rtol=0, atol=1e-6):
                    reasons.append("NONCONTIGUOUS_0P5H")
                if not np.isfinite(window[FEATURES].to_numpy(float)).all():
                    reasons.append("NONFINITE_RAW_F2")
                if float(window["bin_end_ophrs2_h"].iloc[-1]) >= target_start:
                    reasons.append("TARGET_CYCLE_ISOLATION_VIOLATION")
                reason = "|".join(reasons)
                if not reason:
                    legal_x.append(window[FEATURES].to_numpy(dtype=np.float32))
        if start_index is None or start_index < 0:
            input_start = math.nan
            input_end = math.nan
            segment = math.nan
        else:
            row_window = causal.iloc[start_index:end_index + 1]
            input_start = float(row_window["bin_start_ophrs2_h"].iloc[0])
            input_end = float(row_window["bin_end_ophrs2_h"].iloc[-1])
            segment = int(row_window["temporal_segment_id"].iloc[0])
        legal = reason == ""
        record = base | {
            "input_start_ophrs2_h": input_start,
            "input_end_ophrs2_h": input_end,
            "input_start_causal_row": int(causal["causal_row"].iloc[start_index]) if start_index is not None and start_index >= 0 else math.nan,
            "input_end_causal_row": int(causal["causal_row"].iloc[end_index]) if end_index is not None else math.nan,
            "temporal_segment_id": segment,
            "window_legal": legal,
            "target_cycle_isolated": bool(legal and input_end < target_start),
            "exclusion_reason": reason if reason else "",
        }
        if legal and not record["target_cycle_isolated"]:
            raise RuntimeError("Target-cycle isolation unexpectedly failed")
        records.append(record)
    frame = pd.DataFrame(records).sort_values("measurement_end_ophrs2_h").reset_index(drop=True)
    if len(legal_x) != int(frame["window_legal"].sum()):
        raise RuntimeError("Legal feature-window count does not match event table")
    if int(frame.loc[frame.window_legal, "target_cycle_isolated"].sum()) != int(frame.window_legal.sum()):
        raise RuntimeError("A legal health event violates target-cycle isolation")
    return frame, legal_x


def metric_record(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(truth) != len(pred) or not len(truth) or not np.isfinite(pred).all():
        raise RuntimeError("Invalid scalar health metric inputs")
    error = pred - truth
    centered = float(np.sum((truth - truth.mean()) ** 2))
    spearman = pd.Series(truth).corr(pd.Series(pred), method="spearman")
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": float(1.0 - np.sum(error ** 2) / centered) if centered > 0 else math.nan,
        "Spearman": float(spearman) if pd.notna(spearman) else math.nan,
        "MAE_percentage_points": float(np.mean(np.abs(error)) * 100.0),
        "RMSE_percentage_points": float(np.sqrt(np.mean(error ** 2)) * 100.0),
    }


def fit_linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.asarray(x_train, dtype=float), np.ones(len(x_train), dtype=float)])
    coef, *_ = np.linalg.lstsq(design, np.asarray(y_train, dtype=float), rcond=None)
    return (coef[0] * np.asarray(x_eval, dtype=float) + coef[1]).astype(float)


def data_loader(x: np.ndarray, y: np.ndarray, seed: int) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=64, shuffle=True,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, x: np.ndarray, y: np.ndarray, seed: int, epoch: int, dev: torch.device) -> float:
    model.train()
    losses: list[float] = []
    for bx, by in data_loader(x, y, seed + epoch):
        bx, by = bx.to(dev), by.to(dev)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(bx), by)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite standardized SOH loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise RuntimeError("Non-finite gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def predict_standardized(model: nn.Module, x: np.ndarray, dev: torch.device) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            batches.append(model(torch.from_numpy(x[start:start + 256]).to(dev)).detach().cpu().numpy())
    return np.concatenate(batches)


def select_epoch(x: np.ndarray, y: np.ndarray, outer_train_idx: np.ndarray, seed: int, official: dict[str, Any]) -> tuple[int, pd.DataFrame]:
    ordered = np.asarray(outer_train_idx, dtype=int)
    stopper_count = int(math.ceil(len(ordered) * 0.15))
    fit_idx, stopper_idx = ordered[:-stopper_count], ordered[-stopper_count:]
    if len(fit_idx) < 2 or len(stopper_idx) < 1 or not np.all(fit_idx < stopper_idx.min()):
        raise RuntimeError("Chronological 85/15 inner stopper cannot be formed")
    x_scaler = Standardizer().fit(x[fit_idx])
    y_scaler = Standardizer().fit(y[fit_idx])
    x_fit, x_stop = x_scaler.transform(x[fit_idx]), x_scaler.transform(x[stopper_idx])
    y_fit = y_scaler.transform(y[fit_idx])
    set_seed(seed)
    dev = runtime_device()
    model = TimesNetHealth(official).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_epoch, best_mse, stale = 0, math.inf, 0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, 301):
        train_loss = train_epoch(model, optimizer, x_fit, y_fit, seed, epoch, dev)
        stop_prediction = predict_standardized(model, x_stop, dev)
        stop_mse = float(np.mean((stop_prediction - y_scaler.transform(y[stopper_idx])) ** 2))
        logs.append({"stage": "inner_stopper", "seed": seed, "epoch": epoch, "fit_count": len(fit_idx),
                     "stopper_count": len(stopper_idx), "train_standardized_SOH_MSE": train_loss,
                     "stopper_standardized_SOH_MSE": stop_mse})
        if stop_mse < best_mse - 1e-12:
            best_epoch, best_mse, stale = epoch, stop_mse, 0
        else:
            stale += 1
        if stale >= 30:
            break
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if best_epoch == 0:
        raise RuntimeError("No valid TimesNet inner-stopper epoch")
    return best_epoch, pd.DataFrame(logs)


def full_refit_predict(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray, seed: int,
                       epochs: int, official: dict[str, Any], checkpoint_name: str | None = None) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    x_scaler = Standardizer().fit(x[train_idx])
    y_scaler = Standardizer().fit(y[train_idx])
    x_train = x_scaler.transform(x[train_idx])
    y_train = y_scaler.transform(y[train_idx])
    set_seed(seed)
    dev = runtime_device()
    model = TimesNetHealth(official).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    logs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, optimizer, x_train, y_train, seed, epoch, dev)
        logs.append({"stage": "full_outer_refit", "seed": seed, "epoch": epoch, "train_count": len(train_idx),
                     "train_standardized_SOH_MSE": train_loss})
    prediction = np.empty(0, dtype=float) if len(eval_idx) == 0 else y_scaler.inverse(predict_standardized(model, x_scaler.transform(x[eval_idx]), dev))
    metadata = {
        "selected_epoch": int(epochs), "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "train_seconds": float(time.perf_counter() - started), "device": str(dev),
        "input_scaler_mean": np.asarray(x_scaler.mean).tolist(), "input_scaler_scale": np.asarray(x_scaler.scale).tolist(),
        "target_scaler_mean": float(np.asarray(y_scaler.mean)), "target_scaler_scale": float(np.asarray(y_scaler.scale)),
    }
    if checkpoint_name:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "metadata": metadata, "seed": seed, "epochs": epochs}, CHECKPOINTS / checkpoint_name)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, metadata, pd.DataFrame(logs)


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 300,
        "legend.frameon": False,
    })


def create_figures(predictions: pd.DataFrame, baselines: pd.DataFrame, late: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    apply_style()
    # Fig 1: all held-out development events and seed-specific predictions.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    truth = predictions.drop_duplicates(["fold", "cycle_interval_id"]).sort_values("age")
    ax.plot(truth["age"], truth["true_SOH_proxy"], color="black", linewidth=1.7, label="P1S-B2 proxy")
    for seed, group in predictions.groupby("seed", sort=True):
        group = group.sort_values("age")
        ax.scatter(group["age"], group["predicted_SOH"], s=18, alpha=.65, label=f"TimesNet seed {seed}")
    ax.set(xlabel="Target measurement-end operating age (h)", ylabel="P1S-B2 calibrated SOH proxy",
           title="Development held-out SOH-proxy estimation")
    ax.grid(alpha=.22); ax.legend(ncol=2, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig1_Development_SOH_Estimation.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    times_mae = predictions.groupby(["seed", "fold"], as_index=False).agg(MAE=("error", lambda values: float(np.mean(np.abs(values)))))
    values = [times_mae["MAE"].mean(), baselines["Age_MAE"].mean(), baselines["LastVoltage_MAE"].mean()]
    errors = [times_mae["MAE"].std(ddof=0), 0.0, 0.0]
    bars = ax.bar(["TimesNet\n(3 seeds)", "Age-only", "Last voltage"], np.asarray(values) * 100,
                  yerr=np.asarray(errors) * 100, capsize=4,
                  color=[PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["teal"]])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value * 100:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="MAE (percentage points)", title="Development MAE: TimesNet versus deterministic baselines")
    ax.grid(axis="y", alpha=.22); fig.tight_layout(); fig.savefig(FIG / "Fig2_TimesNet_vs_Baselines_MAE.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if late.empty or late.get("status", pd.Series(dtype=str)).eq("NOT_IDENTIFIABLE_NO_LEGAL_COMPLETE_P1S_EVENTS").all():
        ax.axis("off")
        ax.text(.5, .55, "Late-life temporal holdout not identifiable", ha="center", va="center", fontsize=15, weight="bold")
        ax.text(.5, .40, "No frozen COMPLETE_CYCLE_INTERVAL P1S labels exist at\n1819.997738–2256.8 h; no prediction quality was computed.", ha="center", va="center")
    else:
        late_mean = late.groupby(["age", "cycle_interval_id"], as_index=False).agg(true_SOH_proxy=("true_SOH_proxy", "first"), predicted_SOH=("predicted_SOH", "mean"))
        ax.plot(late_mean.age, late_mean.true_SOH_proxy, color="black", label="P1S-B2 proxy")
        ax.plot(late_mean.age, late_mean.predicted_SOH, color=PALETTE["blue_main"], label="TimesNet seed mean")
        ax.set(xlabel="Target measurement-end operating age (h)", ylabel="P1S-B2 calibrated SOH proxy")
        ax.legend()
    ax.set_title("Post-hoc late-life temporal holdout")
    fig.tight_layout(); fig.savefig(FIG / "Fig3_Late_Life_Temporal_Holdout.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for seed, group in predictions.groupby("seed", sort=True):
        ax.scatter(group["age"], group["error"] * 100, s=22, alpha=.65, label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Target measurement-end operating age (h)", ylabel="Prediction residual (percentage points)",
           title="Development TimesNet residuals versus operating age")
    ax.grid(alpha=.22); ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig4_Prediction_Residual_vs_Age.png"); plt.close(fig)


def report_text(summary: dict[str, Any], events: pd.DataFrame, fold_results: pd.DataFrame, baselines: pd.DataFrame,
                late: pd.DataFrame, source_hashes: dict[str, str]) -> str:
    count = int(events.window_legal.sum())
    legal = events.loc[events.window_legal]
    late_status = str(late.iloc[0]["status"]) if not late.empty and "status" in late else "NOT_RUN"
    mean_by_seed = fold_results.groupby("seed", as_index=False).agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), R2=("R2", "mean"), Spearman=("Spearman", "mean"))
    return f"""# TimesNet P1S-B2 health-state estimation

## 1. Research objective

This phase tests whether the frozen TimesNet encoder can map a causal 24 h Raw F2 operating-history window to the next available reference-condition health-state label. It does not reopen backbone selection.

## 2. P1S+B2 is supervision, not the final model

`P1S + frozen B2` is a **P1S-B2 calibrated SOH proxy** used solely as health supervision. It is not an absolute ground-truth SOH and it is not the final predictive model. **TimesNet is the main time-series model.**

## 3. SOH-proxy definition

For the PAA primary `middle_50` complete-cycle P1S measurement, the target is `SOH_P1S_B2 = 1.1988114938324186 × R_P1S − 0.20903624631182127`. The affine coefficients are read from the frozen AHC-TV development-model registry and were never refit here.

## 4. Leakage-safe event construction

The event audit contains {len(events)} primary complete-cycle labels, of which **{count}** have a legal 48-bin history. For every legal event, `input_end_ophrs2_h < target_cycle_start_ophrs2_h` is true, so no target-cycle P1S dwell or target P1S measurement enters the input. The model receives only the frozen ordered 14 Raw F2 variables; it receives neither P1S values, the proxy target, cycle identifiers, future conditions, nor OpHrs2.

## 5. TimesNet architecture

The frozen official TimesNet DataEmbedding, FFT period discovery, TimesBlock, and encoder layer normalization are retained (`48 × 14`, `d_model=32`, `d_ff=64`, one layer, `top_k=2`, three kernels, dropout 0.1). The only task change is `encoded.mean(dim=1) → Linear(32, 1)`. There is no attention head, MLP stack, recurrent unit, new Transformer, or alternative backbone.

## 6. Expanding-fold protocol

The three frozen chronological Development folds were used. Input and target standardizers were fit only to the appropriate training population; epoch selection used the final chronological 15% inner stopper and each outer result used a same-seed reinitialization refit on the full outer train set. Training was standardized SOH MSE with fixed AdamW settings and seeds 42, 2024, and 3407.

## 7. Simple baselines

Age-only is a train-only affine regression on OpHrs2 and is a comparator only; operating age is prohibited from the TimesNet input. Last-voltage-only is a train-only affine regression on the final historical `cell_voltage_mean` in each window.

## 8. Development results

{markdown_table(fold_results[["seed", "fold", "n_train", "n_validation", "MAE", "RMSE", "R2", "Spearman"]], 6)}

Baseline results:

{markdown_table(baselines[["fold", "Age_MAE", "Age_RMSE", "Age_R2", "LastVoltage_MAE", "LastVoltage_RMSE", "LastVoltage_R2"]], 6)}

## 9. Seed stability

{markdown_table(mean_by_seed, 6)}

The reported TimesNet three-fold aggregate is the unweighted mean across all nine seed-fold results; no best seed was selected.

## 10. Late-life temporal holdout

Status: **{late_status}**. The frozen P1S primary complete-cycle table ends at {legal.measurement_end_ophrs2_h.max():.5f} h. Consequently, it contains no eligible complete-cycle P1S-B2 proxy labels in the predeclared 1819.997738–2256.8 h post-hoc interval. No late-life error metric was invented, and no 2000/2300 polarization marker was used for model selection or tuning.

## 11. Does temporal information improve health estimation?

Gate A compares TimesNet with the age-only comparator; Gate B compares it with last-voltage-only. Passing Gate A supports information beyond an age trend, and passing Gate B supports information beyond merely the final historical voltage. The gates are frozen before examining the late-life interval.

## 12. Limitations

There are only sparse P1S-derived labels, the target is a calibrated proxy rather than direct ground truth, and late-life complete-cycle labels are absent from the frozen table. The post-hoc holdout is therefore not identifiable in this phase. These limitations do not authorize target interpolation or post-hoc label construction.

## 13. RUL task connection

The intended line is `SOH estimation → multi-step SOH trajectory forecasting → threshold-crossing RUL`. Official EOL, 90% EOL, and 95% EOL are unobserved; true RUL labels are unavailable and the dataset is right-censored. No direct RUL model was trained.

## 14. Final decision

**{summary['development_status']}**. Gates: A={summary['gate_A']}, B={summary['gate_B']}, C={summary['gate_C']}, D={summary['gate_D']}. The recommended next decision is `{summary['recommended_next_phase']}`. This phase stops here.

## Provenance

All source artifacts were read-only. Source SHA256 values: `{json.dumps(source_hashes, ensure_ascii=False)}`.
"""


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    cfg, bs, b2 = require_frozen_authority()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    source_hashes = {"causal_features_30min.csv": sha256_file(CAUSAL), "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv": sha256_file(P1S),
                     "RAW_CYCLE_CATALOG.csv": sha256_file(CYCLE_CATALOG), "FROZEN_DEVELOPMENT_MODELS.yaml": sha256_file(B2_MODELS),
                     "PREREGISTERED_PROTOCOL.yaml": sha256_file(BS_PROTOCOL)}
    slope, intercept = float(b2["P1S_affine"]["slope"]), float(b2["P1S_affine"]["intercept"])
    events, legal_windows = build_event_dataset(slope, intercept)
    write_csv(events, OUT / "HEALTH_EVENT_DATASET.csv")
    legal = events.loc[events["window_legal"]].copy().reset_index(drop=True)
    x = np.stack(legal_windows).astype(np.float32)
    y = legal["target_SOH_P1S_B2"].to_numpy(dtype=np.float32)
    if x.shape != (len(legal), 48, 14) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Legal event tensor violates frozen 48x14 / finite target contract")
    official = bs["official_sources"]["TimesNet"]["config"]
    seeds = tuple(int(value) for value in bs["training"]["seeds"])
    all_fold_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    training_logs: list[pd.DataFrame] = []
    selected_epochs: list[int] = []
    for fold, train_start, train_end, val_start, val_end in FOLDS:
        train_idx = np.flatnonzero((legal.measurement_end_ophrs2_h.to_numpy(float) >= train_start) & (legal.measurement_end_ophrs2_h.to_numpy(float) < train_end))
        val_idx = np.flatnonzero((legal.measurement_end_ophrs2_h.to_numpy(float) >= val_start) & (legal.measurement_end_ophrs2_h.to_numpy(float) < val_end))
        if not len(train_idx) or not len(val_idx) or set(train_idx) & set(val_idx):
            raise RuntimeError(f"{fold}: invalid expanding-fold event population")
        if np.max(legal.measurement_end_ophrs2_h.to_numpy(float)[val_idx]) >= 1819.997738:
            raise RuntimeError(f"{fold}: development validation crosses frozen boundary")
        age_pred = fit_linear_predict(legal.measurement_end_ophrs2_h.to_numpy(float)[train_idx], y[train_idx], legal.measurement_end_ophrs2_h.to_numpy(float)[val_idx])
        last_v_pred = fit_linear_predict(x[train_idx, -1, 0], y[train_idx], x[val_idx, -1, 0])
        age_metrics, last_v_metrics = metric_record(y[val_idx], age_pred), metric_record(y[val_idx], last_v_pred)
        baseline_rows.append({"fold": fold, "n_train": len(train_idx), "n_validation": len(val_idx),
                              **{f"Age_{key}": value for key, value in age_metrics.items()},
                              **{f"LastVoltage_{key}": value for key, value in last_v_metrics.items()}})
        for seed in seeds:
            epoch, inner_log = select_epoch(x, y, train_idx, seed, official)
            prediction, metadata, refit_log = full_refit_predict(x, y, train_idx, val_idx, seed, epoch, official,
                                                                   f"timesnet_health_{fold}_seed{seed}.pt")
            metrics = metric_record(y[val_idx], prediction)
            selected_epochs.append(epoch)
            all_fold_rows.append({"seed": seed, "fold": fold, "n_train": len(train_idx), "n_validation": len(val_idx),
                                  "selected_epoch": epoch, "parameter_count": metadata["parameter_count"], **metrics})
            current_inner = inner_log.assign(fold=fold)
            current_refit = refit_log.assign(fold=fold, selected_epoch=epoch)
            training_logs.extend([current_inner, current_refit])
            for index, prediction_value in zip(val_idx, prediction):
                all_prediction_rows.append({"age": float(legal.loc[index, "measurement_end_ophrs2_h"]),
                                            "true_SOH_proxy": float(y[index]), "predicted_SOH": float(prediction_value),
                                            "seed": seed, "fold": fold, "error": float(prediction_value - y[index]),
                                            "cycle_interval_id": str(legal.loc[index, "cycle_interval_id"]),
                                            "raw_cycle_id": int(legal.loc[index, "raw_cycle_id"]),
                                            "input_end_ophrs2_h": float(legal.loc[index, "input_end_ophrs2_h"]),
                                            "target_cycle_start_ophrs2_h": float(legal.loc[index, "target_cycle_start_ophrs2_h"])})
    fold_results = pd.DataFrame(all_fold_rows).sort_values(["seed", "fold"]).reset_index(drop=True)
    baselines = pd.DataFrame(baseline_rows).sort_values("fold").reset_index(drop=True)
    predictions = pd.DataFrame(all_prediction_rows).sort_values(["seed", "fold", "age"]).reset_index(drop=True)
    write_csv(fold_results, OUT / "TIMESNET_SOH_FOLD_RESULTS.csv")
    write_csv(baselines, OUT / "SIMPLE_BASELINE_RESULTS.csv")
    write_csv(predictions, OUT / "TIMESNET_SOH_PREDICTIONS.csv")
    write_csv(pd.concat(training_logs, ignore_index=True), OUT / "TIMESNET_SOH_TRAINING_LOG.csv")

    # All model-selection quantities are now fixed.  The final refit is performed even when the predeclared late target set is empty.
    final_epochs = int(round(float(np.median(selected_epochs))))
    dev_idx = np.flatnonzero(legal.measurement_end_ophrs2_h.to_numpy(float) < 1819.997738)
    late_idx = np.flatnonzero((legal.measurement_end_ophrs2_h.to_numpy(float) >= 1819.997738) & (legal.measurement_end_ophrs2_h.to_numpy(float) <= 2256.8))
    late_rows: list[dict[str, Any]] = []
    final_logs: list[pd.DataFrame] = []
    if len(late_idx):
        for seed in seeds:
            prediction, metadata, final_log = full_refit_predict(x, y, dev_idx, late_idx, seed, final_epochs, official,
                                                                   f"timesnet_health_final_development_seed{seed}.pt")
            final_logs.append(final_log.assign(stage="final_development_refit", seed=seed))
            for index, prediction_value in zip(late_idx, prediction):
                late_rows.append({"status": "POSTHOC_TEMPORAL_HOLDOUT_EVALUATED", "seed": seed,
                                  "cycle_interval_id": str(legal.loc[index, "cycle_interval_id"]), "age": float(legal.loc[index, "measurement_end_ophrs2_h"]),
                                  "true_SOH_proxy": float(y[index]), "predicted_SOH": float(prediction_value), "error": float(prediction_value - y[index]),
                                  "final_training_epochs": final_epochs})
    else:
        for seed in seeds:
            _, _, final_log = full_refit_predict(x, y, dev_idx, np.asarray([], dtype=int), seed, final_epochs, official,
                                                  f"timesnet_health_final_development_seed{seed}.pt")
            final_logs.append(final_log.assign(stage="final_development_refit", seed=seed))
        late_rows.append({"status": "NOT_IDENTIFIABLE_NO_LEGAL_COMPLETE_P1S_EVENTS", "seed": math.nan,
                          "cycle_interval_id": "", "age": math.nan, "true_SOH_proxy": math.nan, "predicted_SOH": math.nan,
                          "error": math.nan, "final_training_epochs": final_epochs,
                          "reason": "Frozen primary P1S COMPLETE_CYCLE_INTERVAL table contains no targets in 1819.997738–2256.8 h."})
    if final_logs:
        write_csv(pd.concat(final_logs, ignore_index=True), OUT / "TIMESNET_SOH_FINAL_REFIT_LOG.csv")
    late = pd.DataFrame(late_rows)
    write_csv(late, OUT / "LATE_LIFE_TEMPORAL_HOLDOUT_RESULTS.csv")

    timesnet_mean_mae = float(fold_results.MAE.mean())
    timesnet_mean_rmse = float(fold_results.RMSE.mean())
    age_mean_mae = float(baselines.Age_MAE.mean())
    last_mean_mae = float(baselines.LastVoltage_MAE.mean())
    fold3 = fold_results.loc[fold_results.fold == "fold_3"]
    gate_a, gate_b = timesnet_mean_mae < age_mean_mae, timesnet_mean_mae < last_mean_mae
    gate_c, gate_d = float(fold3.R2.mean()) > 0.0, float(fold3.Spearman.mean()) > 0.0
    status = "TIMESNET_HEALTH_STATE_ESTIMATION_SUPPORTED" if all((gate_a, gate_b, gate_c, gate_d)) else "TIMESNET_HEALTH_STATE_ESTIMATION_NOT_YET_SUPPORTED"
    next_phase = "TIMESNET_MULTISTEP_SOH_FORECASTING" if status == "TIMESNET_HEALTH_STATE_ESTIMATION_SUPPORTED" else "TIMESNET_MULTITASK_HEALTH_LEARNING"
    late_mae = float(np.mean(np.abs(late.loc[late.status == "POSTHOC_TEMPORAL_HOLDOUT_EVALUATED", "error"]))) if (late.status == "POSTHOC_TEMPORAL_HOLDOUT_EVALUATED").any() else None
    summary = {
        "backbone": "TimesNet", "backbone_search_reopened": False, "input": "48x14 Raw F2",
        "target": "P1S-B2 calibrated SOH proxy", "target_cycle_leakage": False,
        "timesnet_mean_mae": timesnet_mean_mae, "timesnet_mean_rmse": timesnet_mean_rmse,
        "age_baseline_mean_mae": age_mean_mae, "last_voltage_baseline_mean_mae": last_mean_mae,
        "fold3_r2": float(fold3.R2.mean()), "fold3_spearman": float(fold3.Spearman.mean()),
        "gate_A": "PASS" if gate_a else "FAIL", "gate_B": "PASS" if gate_b else "FAIL",
        "gate_C": "PASS" if gate_c else "FAIL", "gate_D": "PASS" if gate_d else "FAIL",
        "development_status": status, "late_life_holdout_mae": late_mae,
        "late_life_holdout_status": str(late.iloc[0]["status"]), "late_life_legal_event_count": int(len(late_idx)),
        "real_rul_labels_available": False, "direct_rul_model_trained": False,
        "recommended_next_phase": next_phase, "final_development_refit_epochs": final_epochs,
        "legal_health_event_samples": int(len(legal)), "target_cycle_isolation": "PASS",
        "P1S_used_as_model_input": False, "OpHrs2_used_as_TimesNet_input": False,
        "source_sha256": source_hashes,
    }
    write_json(summary, OUT / "TSHE_SUMMARY.json")
    create_figures(predictions, baselines, late)
    (REPORT / "TIMESNET_HEALTH_STATE_ESTIMATION_REPORT.md").write_text(report_text(summary, events, fold_results, baselines, late, source_hashes), encoding="utf-8")
    print(json.dumps({"status": status, "legal_health_event_samples": len(legal), "late_life_holdout_status": summary["late_life_holdout_status"],
                      "timesnet_mean_mae": timesnet_mean_mae, "age_baseline_mean_mae": age_mean_mae,
                      "last_voltage_baseline_mean_mae": last_mean_mae}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

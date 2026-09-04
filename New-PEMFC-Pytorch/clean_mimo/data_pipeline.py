"""Strictly causal hourly data pipeline for the clean N=15/H=4 MIMO study.

This module intentionally has no dependency on the legacy TFT experiments.  In
particular, it never creates features from a future target, never backfills,
and never smooths the complete life trajectory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_ROOT = Path(
    r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码"
    r"\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main"
    r"\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData"
)

TIME_COLUMN = "Time (h)"
TARGET_NAME = "Utot_mean"


@dataclass(frozen=True)
class PipelineConfig:
    """Fixed Phase-1 protocol shared unchanged by FC1 and FC2."""

    history_hours: int = 15
    horizon_hours: int = 4
    train_fraction: float = 0.50
    validation_fraction: float = 0.10
    forecasting_resolution_hours: float = 1.0
    target_name: str = TARGET_NAME


@dataclass
class WindowSplit:
    X_raw: np.ndarray
    y_raw: np.ndarray
    origins: np.ndarray
    feature_times: np.ndarray
    target_times: np.ndarray


@dataclass
class PreparedData:
    fc: str
    config: PipelineConfig
    feature_names: List[str]
    hourly: pd.DataFrame
    split_rows: Dict[str, Tuple[int, int]]
    splits: Dict[str, WindowSplit]
    X_scaler: StandardScaler
    y_scaler: StandardScaler

    def transformed(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        item = self.splits[split]
        n, _, f = item.X_raw.shape
        x = self.X_scaler.transform(item.X_raw.reshape(n * self.config.history_hours, f))
        x = x.reshape(n, self.config.history_hours, f).astype(np.float32)
        y = self.y_scaler.transform(item.y_raw.reshape(-1, 1))
        return x, y.reshape(item.y_raw.shape).astype(np.float32)


def configured_data_root() -> Path:
    """Allow a portable override without silently falling back to old outputs."""
    return Path(os.environ.get("PEMFC_IEEE2014_PATH", str(DEFAULT_DATA_ROOT)))


def _csv_paths(data_root: Path, fc: str) -> List[Path]:
    paths = sorted(p for p in data_root.rglob("*.csv") if fc in p.name and "Ageing" in p.name)
    if not paths:
        raise FileNotFoundError(f"No {fc} Ageing CSV files found under: {data_root}")
    return paths


def load_ageing_data(fc: str, data_root: Path | None = None) -> pd.DataFrame:
    """Load only the raw PHM2014 ageing recordings for one stack."""
    if fc not in {"FC1", "FC2"}:
        raise ValueError("fc must be 'FC1' or 'FC2'")
    root = data_root or configured_data_root()
    if not root.exists():
        raise FileNotFoundError(
            f"IEEE PHM2014 data root does not exist: {root}. "
            "Set PEMFC_IEEE2014_PATH to the raw-data directory."
        )
    frames = []
    for path in _csv_paths(root, fc):
        frame = pd.read_csv(path, encoding="ISO-8859-1")
        if TIME_COLUMN not in frame.columns:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"No readable {fc} ageing CSV contains {TIME_COLUMN!r}")
    data = pd.concat(frames, ignore_index=True)
    data[TIME_COLUMN] = pd.to_numeric(data[TIME_COLUMN], errors="coerce")
    data = data.dropna(subset=[TIME_COLUMN]).sort_values(TIME_COLUMN).reset_index(drop=True)
    return data


def _column_starting_with(columns: Iterable[str], token: str) -> str:
    matches = [c for c in columns if c.strip().startswith(token)]
    if len(matches) != 1:
        raise KeyError(f"Expected one column beginning with {token!r}; found {matches}")
    return matches[0]


def _required_raw_columns(columns: Iterable[str]) -> Dict[str, str]:
    """Resolve the 24 PHM2014 sensors by physical meaning, not column index."""
    cols = list(columns)
    required = {
        "U1": "U1 ", "U2": "U2 ", "U3": "U3 ", "U4": "U4 ", "U5": "U5 ",
        "Utot": "Utot ", "J": "J ", "I": "I ",
        "TinH2": "TinH2", "ToutH2": "ToutH2", "TinAIR": "TinAIR", "ToutAIR": "ToutAIR",
        "TinWAT": "TinWAT", "ToutWAT": "ToutWAT", "PinAIR": "PinAIR", "PoutAIR": "PoutAIR",
        "PoutH2": "PoutH2", "PinH2": "PinH2", "DinH2": "DinH2", "DoutH2": "DoutH2",
        "DinAIR": "DinAIR", "DoutAIR": "DoutAIR", "DWAT": "DWAT", "HrAIRFC": "HrAIRFC",
    }
    return {name: _column_starting_with(cols, prefix) for name, prefix in required.items()}


def aggregate_to_hourly(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate [k, k+1) to an observation available at the end of hour k+1.

    Mean statistics retain the hourly operating level.  I_std and I_p2p retain
    within-hour current variability for FC2; their near-zero FC1 values are a
    legitimate property of the static operating condition.
    """
    mapping = _required_raw_columns(raw.columns)
    use_raw = list(mapping.values())
    work = raw[[TIME_COLUMN] + use_raw].copy()
    for col in use_raw:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["hour_id"] = np.floor(work[TIME_COLUMN].to_numpy(dtype=float)).astype("int64")

    # No bfill, interpolation, SG filter, target-derived feature, or sampled first row.
    group = work.groupby("hour_id", sort=True)
    mean = group[use_raw].mean()
    counts = group.size().rename("raw_samples")
    raw_min = group[TIME_COLUMN].min().rename("raw_time_min")
    raw_max = group[TIME_COLUMN].max().rename("raw_time_max")
    i_col = mapping["I"]
    i_std = group[i_col].std(ddof=0).fillna(0.0).rename("I_std")
    i_p2p = (group[i_col].max() - group[i_col].min()).rename("I_p2p")

    hourly = pd.concat([mean, counts, raw_min, raw_max, i_std, i_p2p], axis=1).reset_index()
    hourly["origin_time_h"] = hourly["hour_id"].astype(float) + 1.0
    hourly = hourly.rename(columns={mapping["Utot"]: TARGET_NAME, mapping["I"]: "I_mean"})

    rename = {mapping[name]: f"{name}_mean" for name in mapping if name not in {"Utot", "I"}}
    hourly = hourly.rename(columns=rename)
    feature_names = [
        TARGET_NAME,
        "U1_mean", "U2_mean", "U3_mean", "U4_mean", "U5_mean",
        "J_mean", "I_mean", "I_std", "I_p2p",
        "TinH2_mean", "ToutH2_mean", "TinAIR_mean", "ToutAIR_mean", "TinWAT_mean", "ToutWAT_mean",
        "PinAIR_mean", "PoutAIR_mean", "PoutH2_mean", "PinH2_mean",
        "DinH2_mean", "DoutH2_mean", "DinAIR_mean", "DoutAIR_mean", "DWAT_mean", "HrAIRFC_mean",
    ]

    if list(hourly["hour_id"]) != list(range(int(hourly["hour_id"].iloc[0]), int(hourly["hour_id"].iloc[-1]) + 1)):
        raise ValueError("Hourly bins are not consecutive; do not construct windows across a time gap.")

    # Forward fill uses earlier recorded values only.  Leading unresolved values are dropped.
    hourly[feature_names] = hourly[feature_names].ffill()
    hourly = hourly.dropna(subset=feature_names).reset_index(drop=True)
    return hourly, feature_names


def chronological_row_splits(n_rows: int, config: PipelineConfig) -> Dict[str, Tuple[int, int]]:
    train_end = int(n_rows * config.train_fraction)
    val_end = train_end + int(n_rows * config.validation_fraction)
    splits = {"train": (0, train_end), "validation": (train_end, val_end), "test": (val_end, n_rows)}
    min_rows = config.history_hours + config.horizon_hours
    if any(end - start < min_rows for start, end in splits.values()):
        raise ValueError(f"At least {min_rows} hourly rows are required in every split; got {splits}")
    return splits


def _windows_for_split(
    values: np.ndarray,
    target: np.ndarray,
    origin_times: np.ndarray,
    start: int,
    end: int,
    config: PipelineConfig,
) -> WindowSplit:
    X, y, origins, feature_times, target_times = [], [], [], [], []
    for t in range(start + config.history_hours - 1, end - config.horizon_hours):
        # X[t-14:t] -> [Utot(t+1), ..., Utot(t+4)], all indices within one split.
        x_idx = np.arange(t - config.history_hours + 1, t + 1)
        y_idx = np.arange(t + 1, t + config.horizon_hours + 1)
        if x_idx.min() < start or y_idx.max() >= end:
            raise AssertionError("A window crossed a chronological split boundary")
        X.append(values[x_idx])
        y.append(target[y_idx])
        origins.append(t)
        feature_times.append(origin_times[x_idx])
        target_times.append(origin_times[y_idx])
    if not X:
        raise ValueError("No complete MIMO windows after split-boundary protection")
    return WindowSplit(
        X_raw=np.asarray(X, dtype=np.float32),
        y_raw=np.asarray(y, dtype=np.float32),
        origins=np.asarray(origins, dtype=np.int64),
        feature_times=np.asarray(feature_times, dtype=np.float64),
        target_times=np.asarray(target_times, dtype=np.float64),
    )


def prepare_data(fc: str, data_root: Path | None = None, config: PipelineConfig | None = None) -> PreparedData:
    """Create leakage-safe hourly splits and train-only scalers."""
    config = config or PipelineConfig()
    raw = load_ageing_data(fc, data_root)
    hourly, feature_names = aggregate_to_hourly(raw)
    split_rows = chronological_row_splits(len(hourly), config)
    values = hourly[feature_names].to_numpy(dtype=np.float32)
    target = hourly[config.target_name].to_numpy(dtype=np.float32)
    origin_times = hourly["origin_time_h"].to_numpy(dtype=float)
    splits = {
        name: _windows_for_split(values, target, origin_times, start, end, config)
        for name, (start, end) in split_rows.items()
    }
    # Fit only after chronological splitting and only on raw training windows.
    train = splits["train"]
    X_scaler = StandardScaler().fit(train.X_raw.reshape(-1, len(feature_names)))
    y_scaler = StandardScaler().fit(train.y_raw.reshape(-1, 1))
    return PreparedData(fc, config, feature_names, hourly, split_rows, splits, X_scaler, y_scaler)


def serialisable_pipeline_metadata(data: PreparedData) -> dict:
    return {
        "fc": data.fc,
        "config": asdict(data.config),
        "feature_names": data.feature_names,
        "n_hourly_rows": int(len(data.hourly)),
        "split_rows": {key: list(value) for key, value in data.split_rows.items()},
        "window_counts": {key: int(len(value.X_raw)) for key, value in data.splits.items()},
        "feature_scaler_fit_windows": int(len(data.splits["train"].X_raw)),
        "target_scaler_fit_windows": int(len(data.splits["train"].y_raw)),
        "feature_scaler_mean": data.X_scaler.mean_.tolist(),
        "target_scaler_mean": data.y_scaler.mean_.tolist(),
    }


def save_pipeline_metadata(data: PreparedData, output_path: Path) -> None:
    output_path.write_text(json.dumps(serialisable_pipeline_metadata(data), indent=2), encoding="utf-8")

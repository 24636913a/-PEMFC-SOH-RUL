"""Phase N2 shared data, metrics and fail-closed utilities.

The loader reads Test metadata only.  Target/features are read with ``nrows``
ending before the first Test row, so Phase N2 never constructs Test labels.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PHASE_ROOT / "configs" / "phase_n2_config.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if Path(cfg["phase_n2_root"]).resolve() != PHASE_ROOT.resolve():
        raise RuntimeError("Configured Phase N2 root does not match script location")
    return cfg


def ensure_output(path: Path) -> Path:
    root = PHASE_ROOT.resolve()
    parent = path.parent.resolve()
    if parent != root and root not in parent.parents:
        raise RuntimeError(f"Refusing to write outside Phase N2: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_output(path)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    ensure_output(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    ensure_output(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def development_folds(cfg: dict[str, Any]) -> list[dict[str, float | str]]:
    s = cfg["split"]
    return [
        {"fold": "fold_1", "train_start_h": s["min_h"], "train_end_h": s["fold_1_train_end_h"], "validation_start_h": s["fold_1_train_end_h"], "validation_end_h": s["fold_1_validation_end_h"]},
        {"fold": "fold_2", "train_start_h": s["min_h"], "train_end_h": s["fold_2_train_end_h"], "validation_start_h": s["fold_2_train_end_h"], "validation_end_h": s["fold_2_validation_end_h"]},
        {"fold": "fold_3", "train_start_h": s["min_h"], "train_end_h": s["fold_3_train_end_h"], "validation_start_h": s["fold_3_train_end_h"], "validation_end_h": s["development_end_h"]},
    ]


def data_path(cfg: dict[str, Any], resolution_h: float) -> Path:
    root = Path(cfg["project_root"])
    rel = cfg["data"]["primary_csv"] if math.isclose(resolution_h, 0.5) else cfg["data"]["robustness_csv"]
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_development_data(cfg: dict[str, Any], resolution_h: float, required_columns: Iterable[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only pre-Test feature/target rows, while auditing Test metadata."""
    path = data_path(cfg, resolution_h)
    dcfg = cfg["data"]
    metadata_cols = [
        dcfg["bin_start_column"], dcfg["bin_end_column"], dcfg["datetime_column"],
        "split", dcfg["valid_column"], "coverage_ratio", "gap_before_flag", "gap_after_flag",
    ]
    metadata = pd.read_csv(path, usecols=metadata_cols)
    test_mask = metadata["split"].eq("test")
    first_test_index = int(np.flatnonzero(test_mask.to_numpy())[0]) if test_mask.any() else len(metadata)
    columns = list(dict.fromkeys([*required_columns, *metadata_cols, dcfg["segment_column"], dcfg["ageing_column"]]))
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(columns) - set(header))
    if missing:
        raise KeyError(f"Required columns missing: {missing}")
    development = pd.read_csv(path, usecols=columns, nrows=first_test_index)
    development = development.loc[development[dcfg["bin_end_column"]] <= float(cfg["split"]["development_end_h"]) + 1e-10].copy()
    if development["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL VIOLATION: Test rows entered feature/target frame")
    development.sort_values(dcfg["bin_start_column"], inplace=True)
    development.reset_index(drop=True, inplace=True)
    audit = {
        "resolution_hours": resolution_h,
        "source_file": str(path),
        "schema_column_count": len(header),
        "total_metadata_rows": len(metadata),
        "test_metadata_rows": int(test_mask.sum()),
        "test_metadata_start_h": float(metadata.loc[test_mask, dcfg["bin_start_column"]].min()) if test_mask.any() else None,
        "test_metadata_end_h": float(metadata.loc[test_mask, dcfg["bin_end_column"]].max()) if test_mask.any() else None,
        "development_feature_target_rows": len(development),
        "target_rows_read_from_test": 0,
        "test_predictions_constructed": False,
    }
    return development, audit


@dataclass
class WindowSet:
    x: np.ndarray
    y_delta: np.ndarray
    y_abs: np.ndarray
    anchor: np.ndarray
    origin_age_h: np.ndarray
    origin_datetime: np.ndarray
    target_end_age_h: np.ndarray
    row_origin_index: np.ndarray


def build_windows(
    frame: pd.DataFrame,
    features: list[str],
    lookback_steps: int,
    horizon_steps: int,
    region_start_h: float,
    region_end_h: float,
    resolution_h: float,
    cfg: dict[str, Any],
    allowed_origin_indices: set[int] | None = None,
) -> WindowSet:
    dcfg = cfg["data"]
    start_col, end_col = dcfg["bin_start_column"], dcfg["bin_end_column"]
    voltage = frame[dcfg["target_column"]].to_numpy(float)
    valid = frame[dcfg["valid_column"]].astype(bool).to_numpy()
    segment = frame[dcfg["segment_column"]].to_numpy()
    starts = frame[start_col].to_numpy(float)
    ends = frame[end_col].to_numpy(float)
    datetimes = frame[dcfg["datetime_column"]].astype(str).to_numpy()
    values = frame[features].to_numpy(float)
    x_rows: list[np.ndarray] = []
    y_delta_rows: list[np.ndarray] = []
    y_abs_rows: list[np.ndarray] = []
    anchors: list[float] = []
    ages: list[float] = []
    times: list[str] = []
    target_ends: list[float] = []
    indices: list[int] = []

    for origin in range(lookback_steps - 1, len(frame) - horizon_steps):
        if allowed_origin_indices is not None and origin not in allowed_origin_indices:
            continue
        first = origin - lookback_steps + 1
        last = origin + horizon_steps
        if starts[first] < region_start_h - 1e-9 or ends[last] > region_end_h + 1e-9:
            continue
        block = slice(first, last + 1)
        if not valid[block].all() or np.any(segment[block] != segment[origin]):
            continue
        expected = np.diff(starts[block])
        if not np.allclose(expected, resolution_h, atol=1e-6, rtol=0):
            continue
        if not np.isfinite(voltage[first : last + 1]).all():
            continue
        y_abs = voltage[origin + 1 : last + 1]
        anchor = voltage[origin]
        x_rows.append(values[first : origin + 1])
        y_abs_rows.append(y_abs)
        y_delta_rows.append(y_abs - anchor)
        anchors.append(anchor)
        ages.append(ends[origin])
        times.append(datetimes[origin])
        target_ends.append(ends[last])
        indices.append(origin)
    if not x_rows:
        raise RuntimeError(f"No legal windows for region {region_start_h}-{region_end_h}, L={lookback_steps}, H={horizon_steps}")
    return WindowSet(
        x=np.asarray(x_rows, dtype=np.float32),
        y_delta=np.asarray(y_delta_rows, dtype=np.float32),
        y_abs=np.asarray(y_abs_rows, dtype=np.float32),
        anchor=np.asarray(anchors, dtype=np.float32),
        origin_age_h=np.asarray(ages, dtype=np.float64),
        origin_datetime=np.asarray(times),
        target_end_age_h=np.asarray(target_ends, dtype=np.float64),
        row_origin_index=np.asarray(indices, dtype=np.int64),
    )


class FoldPreprocessor:
    def __init__(self) -> None:
        self.medians: np.ndarray | None = None
        self.scaler = StandardScaler()

    def fit(self, x: np.ndarray) -> "FoldPreprocessor":
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        self.medians = np.nanmedian(flat, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        flat = np.where(np.isfinite(flat), flat, self.medians)
        self.scaler.fit(flat)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("Preprocessor not fitted")
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).astype(np.float64)
        flat = np.where(np.isfinite(flat), flat, self.medians)
        return self.scaler.transform(flat).reshape(shape).astype(np.float32)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, float).reshape(-1)
    yp = np.asarray(y_pred, float).reshape(-1)
    error = yp - yt
    mse = float(np.mean(error**2))
    denom = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "rmse_V": math.sqrt(mse),
        "rmse_mV": math.sqrt(mse) * 1000.0,
        "mae_V": float(np.mean(np.abs(error))),
        "mae_mV": float(np.mean(np.abs(error))) * 1000.0,
        "bias_V": float(np.mean(error)),
        "bias_mV": float(np.mean(error)) * 1000.0,
        "r2": 1.0 - float(np.sum(error**2)) / denom if denom > 0 else math.nan,
    }


def per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, resolution_h: float) -> list[dict[str, float]]:
    return [{"horizon_step": h + 1, "horizon_hours": (h + 1) * resolution_h, **metric_dict(y_true[:, h], y_pred[:, h])} for h in range(y_true.shape[1])]


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def now_seconds() -> float:
    return time.perf_counter()

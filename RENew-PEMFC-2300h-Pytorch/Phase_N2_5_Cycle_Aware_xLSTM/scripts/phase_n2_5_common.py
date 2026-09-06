"""Phase N2.5 shared data, metrics, baselines and fail-closed utilities."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PHASE_ROOT / "configs" / "phase_n2_5_config.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if Path(cfg["phase_n2_5_root"]).resolve() != PHASE_ROOT.resolve():
        raise RuntimeError("Configured Phase N2.5 root does not match script location")
    frozen_path = Path(cfg["project_root"]) / cfg["phase_n2_frozen_config"]
    with frozen_path.open("r", encoding="utf-8") as stream:
        frozen = yaml.safe_load(stream)
    expected = {
        "sampling_resolution_hours": 0.5,
        "feature_group": "F2",
        "lookback_hours": 24,
        "lookback_steps": 48,
        "primary_horizon_hours": 12,
        "horizon_steps": 24,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"Phase N2 frozen protocol mismatch for {key}: {frozen.get(key)!r}")
    features = list(frozen.get("exact_feature_names", []))
    if len(features) != 14 or len(set(features)) != 14:
        raise RuntimeError("Phase N2 frozen F2 feature list is not exactly 14 unique columns")
    cfg["frozen"] = frozen
    cfg["features"] = features
    return cfg


def ensure_output(path: Path) -> Path:
    root = PHASE_ROOT.resolve()
    parent = path.parent.resolve()
    if parent != root and root not in parent.parents:
        raise RuntimeError(f"Refusing to write outside Phase N2.5: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_output(path)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    ensure_output(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(content: str, path: Path) -> None:
    ensure_output(path).write_text(content.rstrip() + "\n", encoding="utf-8")


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
    split = cfg["split"]
    return [
        {
            "fold": "fold_1",
            "train_start_h": split["min_h"],
            "train_end_h": split["fold_1_train_end_h"],
            "validation_start_h": split["fold_1_train_end_h"],
            "validation_end_h": split["fold_1_validation_end_h"],
        },
        {
            "fold": "fold_2",
            "train_start_h": split["min_h"],
            "train_end_h": split["fold_2_train_end_h"],
            "validation_start_h": split["fold_2_train_end_h"],
            "validation_end_h": split["fold_2_validation_end_h"],
        },
        {
            "fold": "fold_3",
            "train_start_h": split["min_h"],
            "train_end_h": split["fold_3_train_end_h"],
            "validation_start_h": split["fold_3_train_end_h"],
            "validation_end_h": split["development_end_h"],
        },
    ]


def load_development_data(cfg: dict[str, Any], required_columns: Iterable[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read target/features only before the first Test row; Test is metadata-only."""
    path = Path(cfg["project_root"]) / cfg["data"]["primary_csv"]
    if not path.is_file():
        raise FileNotFoundError(path)
    dcfg = cfg["data"]
    metadata_columns = [
        dcfg["bin_start_column"],
        dcfg["bin_end_column"],
        dcfg["datetime_column"],
        "split",
        dcfg["valid_column"],
        "coverage_ratio",
        "gap_before_flag",
        "gap_after_flag",
    ]
    metadata = pd.read_csv(path, usecols=metadata_columns)
    test_mask = metadata["split"].eq("test")
    first_test_index = int(np.flatnonzero(test_mask.to_numpy())[0]) if test_mask.any() else len(metadata)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    columns = list(
        dict.fromkeys(
            [
                *required_columns,
                *metadata_columns,
                dcfg["segment_column"],
                dcfg["ageing_column"],
                dcfg["target_column"],
            ]
        )
    )
    missing = sorted(set(columns) - set(header))
    if missing:
        raise KeyError(f"Required columns missing: {missing}")
    development = pd.read_csv(path, usecols=columns, nrows=first_test_index)
    development = development.loc[
        development[dcfg["bin_end_column"]] <= float(cfg["split"]["development_end_h"]) + 1e-10
    ].copy()
    if development["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL VIOLATION: Test entered feature/target frame")
    development.sort_values(dcfg["bin_start_column"], inplace=True)
    development.reset_index(drop=True, inplace=True)
    audit = {
        "resolution_hours": float(cfg["task"]["resolution_hours"]),
        "source_file": str(path),
        "schema_column_count": len(header),
        "total_metadata_rows": len(metadata),
        "test_metadata_rows": int(test_mask.sum()),
        "test_metadata_start_h": float(metadata.loc[test_mask, dcfg["bin_start_column"]].min()),
        "test_metadata_end_h": float(metadata.loc[test_mask, dcfg["bin_end_column"]].max()),
        "development_feature_target_rows": len(development),
        "target_rows_read_from_test": 0,
        "test_predictions_constructed": False,
        "test_performance_accessed": False,
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
) -> WindowSet:
    dcfg = cfg["data"]
    start_column = dcfg["bin_start_column"]
    end_column = dcfg["bin_end_column"]
    voltage = frame[dcfg["target_column"]].to_numpy(float)
    valid = frame[dcfg["valid_column"]].astype(bool).to_numpy()
    segment = frame[dcfg["segment_column"]].to_numpy()
    starts = frame[start_column].to_numpy(float)
    ends = frame[end_column].to_numpy(float)
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
        first = origin - lookback_steps + 1
        last = origin + horizon_steps
        if starts[first] < region_start_h - 1e-9 or ends[last] > region_end_h + 1e-9:
            continue
        block = slice(first, last + 1)
        if not valid[block].all() or np.any(segment[block] != segment[origin]):
            continue
        if not np.allclose(np.diff(starts[block]), resolution_h, atol=1e-6, rtol=0):
            continue
        if not np.isfinite(voltage[first : last + 1]).all():
            continue
        future = voltage[origin + 1 : last + 1]
        anchor = voltage[origin]
        x_rows.append(values[first : origin + 1])
        y_abs_rows.append(future)
        y_delta_rows.append(future - anchor)
        anchors.append(anchor)
        ages.append(ends[origin])
        times.append(datetimes[origin])
        target_ends.append(ends[last])
        indices.append(origin)
    if not x_rows:
        raise RuntimeError(f"No legal windows for {region_start_h}-{region_end_h}")
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


def fold_windows(frame: pd.DataFrame, cfg: dict[str, Any]) -> list[tuple[str, WindowSet, WindowSet]]:
    features = cfg["features"]
    task = cfg["task"]
    output: list[tuple[str, WindowSet, WindowSet]] = []
    for spec in development_folds(cfg):
        train = build_windows(
            frame,
            features,
            int(task["lookback_steps"]),
            int(task["horizon_steps"]),
            float(spec["train_start_h"]),
            float(spec["train_end_h"]),
            float(task["resolution_hours"]),
            cfg,
        )
        validation = build_windows(
            frame,
            features,
            int(task["lookback_steps"]),
            int(task["horizon_steps"]),
            float(spec["validation_start_h"]),
            float(spec["validation_end_h"]),
            float(task["resolution_hours"]),
            cfg,
        )
        output.append((str(spec["fold"]), train, validation))
    return output


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
            raise RuntimeError("Preprocessor has not been fitted")
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).astype(np.float64)
        flat = np.where(np.isfinite(flat), flat, self.medians)
        return self.scaler.transform(flat).reshape(shape).astype(np.float32)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, float).reshape(-1)
    prediction = np.asarray(y_pred, float).reshape(-1)
    error = prediction - truth
    mse = float(np.mean(error**2))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "rmse_V": math.sqrt(mse),
        "rmse_mV": math.sqrt(mse) * 1000.0,
        "mae_V": float(np.mean(np.abs(error))),
        "mae_mV": float(np.mean(np.abs(error))) * 1000.0,
        "bias_V": float(np.mean(error)),
        "bias_mV": float(np.mean(error)) * 1000.0,
        "r2": 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else math.nan,
    }


def per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, resolution_h: float = 0.5) -> list[dict[str, float]]:
    return [
        {
            "horizon_step": step + 1,
            "horizon_hours": (step + 1) * resolution_h,
            **metric_dict(y_true[:, step], y_pred[:, step]),
        }
        for step in range(y_true.shape[1])
    ]


def simple_predictions(train: WindowSet, validation: WindowSet, alpha: float, period_steps: int) -> dict[str, np.ndarray]:
    horizon = validation.y_abs.shape[1]
    persistence = np.repeat(validation.anchor[:, None], horizon, axis=1)
    history = validation.x[:, :, 0].astype(float)
    cycle = np.empty_like(validation.y_abs, dtype=float)
    for index, values in enumerate(history):
        period = values[-period_steps:]
        cycle[index] = period[np.arange(horizon) % period_steps]
    preprocessor = FoldPreprocessor().fit(train.x)
    ridge = Ridge(alpha=alpha).fit(
        preprocessor.transform(train.x).reshape(len(train.x), -1), train.y_delta
    )
    ridge_prediction = validation.anchor[:, None] + ridge.predict(
        preprocessor.transform(validation.x).reshape(len(validation.x), -1)
    )
    return {"Persistence": persistence, "CycleRepeat": cycle, "Ridge": ridge_prediction}


def baseline_denominators(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, float | str]:
    rmses = {name: metric_dict(y_true, prediction)["rmse_V"] for name, prediction in predictions.items()}
    best_model = min(rmses, key=rmses.get)
    return {**rmses, "best": rmses[best_model], "best_model": best_model}


def add_skill(metrics: dict[str, float], denominators: dict[str, float | str]) -> dict[str, float | str]:
    rmse = metrics["rmse_V"]
    return {
        **metrics,
        "skill_persistence": 1.0 - rmse / float(denominators["Persistence"]),
        "skill_cycle": 1.0 - rmse / float(denominators["CycleRepeat"]),
        "skill_best": 1.0 - rmse / float(denominators["best"]),
        "strongest_simple_baseline": str(denominators["best_model"]),
        "strongest_simple_rmse_V": float(denominators["best"]),
    }


def load_phase_n2_lstm_reference(cfg: dict[str, Any], seed: int, fold: str, validation: WindowSet) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n2_root = Path(cfg["phase_n2_root"])
    if seed == 42:
        run_id = f"feature_F2_LSTM_{fold}_seed42"
    else:
        run_id = f"multiseed_LSTM_F2_L24p0h_H12p0h_R0p5h_{fold}_seed{seed}"
    result_path = n2_root / "outputs" / "model_runs" / f"{run_id}.json"
    prediction_path = n2_root / "outputs" / "model_runs" / f"{run_id}_predictions.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(prediction_path, allow_pickle=False))
    if result.get("status") != "complete" or result.get("fold") != fold:
        raise RuntimeError(f"Invalid Phase N2 LSTM reference: {run_id}")
    if not np.array_equal(arrays["row_origin_index"], validation.row_origin_index):
        raise RuntimeError(f"Phase N2 LSTM origins differ in {run_id}")
    if not np.array_equal(arrays["y_true"], validation.y_abs):
        raise RuntimeError(f"Phase N2 LSTM targets differ in {run_id}")
    return result, arrays


def paired_moving_block_bootstrap(
    pairs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    repetitions: int,
    block_length: int,
    seed: int,
    comparison: str,
    candidate: str,
    reference: str,
) -> dict[str, float | int | str]:
    """Bootstrap prediction origins in paired moving blocks within each fold-seed series."""
    generator = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        candidate_sse = 0.0
        reference_sse = 0.0
        count = 0
        for truth, candidate_prediction, reference_prediction in pairs.values():
            n = len(truth)
            blocks = math.ceil(n / block_length)
            starts = generator.integers(0, n, size=blocks)
            indices = np.concatenate(
                [(start + np.arange(block_length)) % n for start in starts]
            )[:n]
            candidate_error = candidate_prediction[indices] - truth[indices]
            reference_error = reference_prediction[indices] - truth[indices]
            candidate_sse += float(np.sum(candidate_error**2))
            reference_sse += float(np.sum(reference_error**2))
            count += int(candidate_error.size)
        deltas[repetition] = math.sqrt(candidate_sse / count) - math.sqrt(reference_sse / count)
    point_candidate_sse = sum(float(np.sum((pc - yt) ** 2)) for yt, pc, _ in pairs.values())
    point_reference_sse = sum(float(np.sum((pr - yt) ** 2)) for yt, _, pr in pairs.values())
    point_count = sum(int(yt.size) for yt, _, _ in pairs.values())
    return {
        "comparison": comparison,
        "candidate": candidate,
        "reference": reference,
        "delta_definition": "RMSE_candidate_minus_RMSE_reference_V",
        "bootstrap_repetitions": repetitions,
        "block_length_steps": block_length,
        "point_delta_rmse_V": math.sqrt(point_candidate_sse / point_count) - math.sqrt(point_reference_sse / point_count),
        "mean_delta_rmse_V": float(deltas.mean()),
        "median_delta_rmse_V": float(np.median(deltas)),
        "ci95_lower_V": float(np.quantile(deltas, 0.025)),
        "ci95_upper_V": float(np.quantile(deltas, 0.975)),
        "probability_candidate_better": float(np.mean(deltas < 0)),
    }


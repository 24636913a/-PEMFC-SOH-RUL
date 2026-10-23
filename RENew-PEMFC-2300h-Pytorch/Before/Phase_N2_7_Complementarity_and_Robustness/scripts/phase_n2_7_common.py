"""Shared frozen protocol, causal filters, metrics, and prediction loaders for Phase N2.7."""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PHASE_ROOT.parent
CONFIG_PATH = PHASE_ROOT / "configs/phase_n2_7_config.yaml"
N25_SCRIPTS = PROJECT_ROOT / "Phase_N2_5_Cycle_Aware_xLSTM/scripts"
if str(N25_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(N25_SCRIPTS))

from phase_n2_5_common import (  # noqa: E402
    FoldPreprocessor,
    WindowSet,
    fold_windows,
    load_development_data,
    metric_dict,
)


FOLDS = ["fold_1", "fold_2", "fold_3"]
SEEDS = [42, 2024, 3407]
MODELS = ["LSTM", "TimesNet"]


def load_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if Path(cfg["phase_n2_7_root"]).resolve() != PHASE_ROOT.resolve():
        raise RuntimeError("Configured Phase N2.7 root does not match script location")
    frozen_path = Path(cfg["project_root"]) / cfg["n2_6r_frozen_config"]
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    expected = {
        "selected_backbone": "LSTM",
        "sampling_resolution_hours": 0.5,
        "feature_group": "F2",
        "lookback_hours": 24,
        "lookback_steps": 48,
        "horizon_hours": 12,
        "horizon_steps": 24,
        "test_performance_accessed": False,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"Inherited Phase N2.6R mismatch for {key}: {frozen.get(key)!r}")
    if list(frozen.get("exact_feature_names", [])) != list(cfg["features"]):
        raise RuntimeError("F2 feature order differs from Phase N2.6R freeze")
    if len(cfg["features"]) != 14 or cfg["features"][0] != "cell_voltage_mean":
        raise RuntimeError("F2 must remain exact 14-column voltage-first protocol")
    cfg["frozen_n2_6r"] = frozen
    return cfg


def ensure_output(path: Path) -> Path:
    root = PHASE_ROOT.resolve()
    parent = path.parent.resolve()
    if parent != root and root not in parent.parents:
        raise RuntimeError(f"Refusing to write outside Phase N2.7: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(ensure_output(path), index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(payload: Any, path: Path) -> None:
    ensure_output(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def load_protocol(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], list[tuple[str, WindowSet, WindowSet]]]:
    frame, inherited = load_development_data(cfg, cfg["features"])
    audit = {
        **inherited,
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "test_shift_diagnostics_computed": False,
        "test_performance_accessed": False,
    }
    if inherited.get("target_rows_read_from_test") != 0 or frame["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL VIOLATION")
    windows = fold_windows(frame, cfg)
    for fold, _, validation in windows:
        if float(validation.target_end_age_h.max()) >= float(cfg["split"]["development_end_h"]):
            raise RuntimeError(f"TEST FIREWALL VIOLATION in {fold}")
    return frame, audit, windows


def verify_prediction_arrays(arrays: dict[str, np.ndarray], cfg: dict[str, Any], label: str) -> None:
    required = {"y_true", "y_pred", "anchor", "origin_age_h", "target_end_age_h", "row_origin_index"}
    if not required.issubset(arrays):
        raise RuntimeError(f"Incomplete predictions: {label}")
    if arrays["y_true"].shape != arrays["y_pred"].shape or arrays["y_true"].shape[1] != 24:
        raise RuntimeError(f"Prediction shape mismatch: {label}")
    if not np.isfinite(arrays["y_true"]).all() or not np.isfinite(arrays["y_pred"]).all():
        raise RuntimeError(f"Non-finite predictions: {label}")
    if float(np.max(arrays["target_end_age_h"])) >= float(cfg["split"]["development_end_h"]):
        raise RuntimeError(f"TEST FIREWALL VIOLATION: {label}")


def load_reference_prediction(
    cfg: dict[str, Any], model: str, seed: int, fold: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    if model == "LSTM":
        if seed == 42:
            source_root = Path(cfg["phase_n2_6r_root"])
            run_id = f"reproduction_LSTM_{fold}_seed42"
            source = "Phase_N2_6R_reproduction"
        else:
            source_root = Path(cfg["phase_n2_root"])
            run_id = f"multiseed_LSTM_F2_L24p0h_H12p0h_R0p5h_{fold}_seed{seed}"
            source = "Phase_N2_frozen_reference"
    elif model == "TimesNet":
        source_root = Path(cfg["phase_n2_6r_root"])
        run_id = (
            f"screening_timesnet_hf_t1_{fold}_seed42"
            if seed == 42
            else f"multiseed_timesnet_hf_t1_{fold}_seed{seed}"
        )
        source = "Phase_N2_6R_high_fidelity"
    else:
        raise KeyError(model)
    result_path = source_root / "outputs/model_runs" / f"{run_id}.json"
    array_path = source_root / "outputs/model_runs" / f"{run_id}_predictions.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(array_path, allow_pickle=False))
    if result.get("status") != "complete":
        raise RuntimeError(f"Incomplete historical run: {run_id}")
    verify_prediction_arrays(arrays, cfg, run_id)
    return result, arrays, source


def load_all_reference_predictions(cfg: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, np.ndarray]]:
    output: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        for fold in FOLDS:
            _, lstm, _ = load_reference_prediction(cfg, "LSTM", seed, fold)
            _, timesnet, _ = load_reference_prediction(cfg, "TimesNet", seed, fold)
            if not np.array_equal(lstm["row_origin_index"], timesnet["row_origin_index"]):
                raise RuntimeError(f"Matched-Origin violation: seed={seed}, {fold}")
            if not np.array_equal(lstm["target_end_age_h"], timesnet["target_end_age_h"]):
                raise RuntimeError(f"Target timestamp violation: seed={seed}, {fold}")
            if not np.array_equal(lstm["y_true"], timesnet["y_true"]):
                raise RuntimeError(f"Target value violation: seed={seed}, {fold}")
            output[("LSTM", seed, fold)] = lstm
            output[("TimesNet", seed, fold)] = timesnet
    return output


def macro_metrics(predictions: dict[tuple[int, str], dict[str, np.ndarray]]) -> dict[str, float]:
    rows = [metric_dict(arrays["y_true"], arrays["y_pred"]) for arrays in predictions.values()]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def causal_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    state = math.nan
    for index, value in enumerate(np.asarray(values, float)):
        if not math.isfinite(value):
            continue
        state = value if not math.isfinite(state) else alpha * value + (1.0 - alpha) * state
        result[index] = state
    return result


def causal_sg(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    source = np.asarray(values, float)
    result = np.full(source.shape, np.nan, dtype=float)
    for index, value in enumerate(source):
        if not math.isfinite(value):
            continue
        first = max(0, index - window + 1)
        indices = np.arange(first, index + 1)
        finite = np.isfinite(source[indices])
        x = indices[finite].astype(float) - float(index)
        y = source[indices][finite]
        degree = min(polyorder, len(y) - 1)
        if degree <= 0:
            result[index] = float(y[-1])
        else:
            result[index] = float(np.polyval(np.polyfit(x, y, degree), 0.0))
    return result


def filtered_frame(frame: pd.DataFrame, cfg: dict[str, Any], preprocessing_name: str) -> pd.DataFrame:
    spec = cfg["preprocessing"][preprocessing_name]
    method = spec["method"]
    output = frame.copy()
    if method == "raw":
        return output
    segment_column = cfg["data"]["segment_column"]
    for _, indices in output.groupby(segment_column, sort=False, dropna=False).groups.items():
        index_array = np.asarray(list(indices), dtype=int)
        for feature in cfg["voltage_state_features"]:
            values = output.loc[index_array, feature].to_numpy(float)
            if method == "causal_ema":
                transformed = causal_ema(values, float(spec["alpha"]))
            elif method == "causal_sg":
                transformed = causal_sg(values, int(spec["window"]), int(spec["polyorder"]))
            else:
                raise KeyError(method)
            output.loc[index_array, feature] = transformed
    return output


def preprocessing_windows(
    frame: pd.DataFrame,
    raw_windows: list[tuple[str, WindowSet, WindowSet]],
    cfg: dict[str, Any],
    preprocessing_name: str,
) -> list[tuple[str, WindowSet, WindowSet]]:
    if preprocessing_name == "P0_raw":
        return raw_windows
    processed = fold_windows(filtered_frame(frame, cfg, preprocessing_name), cfg)
    output: list[tuple[str, WindowSet, WindowSet]] = []
    raw_map = {fold: (train, validation) for fold, train, validation in raw_windows}
    for fold, filtered_train, filtered_validation in processed:
        raw_train, raw_validation = raw_map[fold]
        if not np.array_equal(filtered_train.row_origin_index, raw_train.row_origin_index):
            raise RuntimeError(f"Filtered train origins changed: {preprocessing_name}, {fold}")
        if not np.array_equal(filtered_validation.row_origin_index, raw_validation.row_origin_index):
            raise RuntimeError(f"Filtered validation origins changed: {preprocessing_name}, {fold}")
        train = replace(
            filtered_train,
            y_delta=raw_train.y_delta,
            y_abs=raw_train.y_abs,
            anchor=raw_train.anchor,
            target_end_age_h=raw_train.target_end_age_h,
        )
        validation = replace(
            filtered_validation,
            y_delta=raw_validation.y_delta,
            y_abs=raw_validation.y_abs,
            anchor=raw_validation.anchor,
            target_end_age_h=raw_validation.target_end_age_h,
        )
        output.append((fold, train, validation))
    return output


def window_summary_matrix(
    windows: WindowSet,
    cfg: dict[str, Any],
    medians: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    feature_indices = [cfg["features"].index(name) for name in cfg["shift_summary_features"]]
    values = windows.x[:, :, feature_indices].astype(float)
    if medians is None:
        medians = np.nanmedian(values.reshape(-1, values.shape[-1]), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    values = np.where(np.isfinite(values), values, medians[None, None, :])
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    last = values[:, -1, :]
    time = np.arange(values.shape[1], dtype=float)
    centered = time - time.mean()
    slope = np.einsum("ntl,t->nl", values, centered) / float(np.sum(centered**2))
    matrix = np.concatenate([mean, std, last, slope], axis=1)
    names = []
    for statistic in ["mean", "std", "last", "slope"]:
        names.extend([f"{feature}__{statistic}" for feature in cfg["shift_summary_features"]])
    return matrix, names, medians


def development_stage_bounds(windows: list[tuple[str, WindowSet, WindowSet]]) -> tuple[float, float, float, float]:
    ages = np.concatenate([validation.origin_age_h for _, _, validation in windows])
    lower, upper = float(ages.min()), float(ages.max())
    return lower, lower + (upper - lower) / 3.0, lower + 2.0 * (upper - lower) / 3.0, upper


def stage_labels(ages: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    _, cut1, cut2, _ = bounds
    return np.where(ages < cut1, "Early Development", np.where(ages < cut2, "Middle Development", "Late Development"))

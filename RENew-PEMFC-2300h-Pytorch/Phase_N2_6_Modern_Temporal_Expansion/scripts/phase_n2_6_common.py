"""Phase N2.6 shared protocol, metrics, I/O and fail-closed utilities."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PHASE_ROOT.parent
CONFIG_PATH = PHASE_ROOT / "configs" / "phase_n2_6_config.yaml"
N25_SCRIPT_DIR = PROJECT_ROOT / "Phase_N2_5_Cycle_Aware_xLSTM" / "scripts"
if str(N25_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(N25_SCRIPT_DIR))

from phase_n2_5_common import (  # noqa: E402
    FoldPreprocessor,
    WindowSet,
    add_skill,
    baseline_denominators,
    fold_windows,
    load_development_data,
    load_phase_n2_lstm_reference,
    metric_dict,
    paired_moving_block_bootstrap,
    per_horizon_metrics,
    simple_predictions,
)


def load_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if Path(cfg["phase_n2_6_root"]).resolve() != PHASE_ROOT.resolve():
        raise RuntimeError("Configured Phase N2.6 root does not match script location")
    frozen_path = Path(cfg["project_root"]) / cfg["phase_n2_5_frozen_config"]
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    expected = {
        "sampling_resolution": 0.5,
        "feature_group": "F2",
        "lookback_hours": 24,
        "lookback_steps": 48,
        "horizon_hours": 12,
        "horizon_steps": 24,
        "model": "LSTM",
        "test_performance_accessed": False,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"Phase N2.5 frozen protocol mismatch for {key}: {frozen.get(key)!r}")
    features = list(frozen.get("exact_feature_names", []))
    if len(features) != 14 or len(set(features)) != 14:
        raise RuntimeError("Frozen F2 feature list is not exactly 14 unique columns")
    cfg["frozen"] = frozen
    cfg["features"] = features
    return cfg


def ensure_output(path: Path) -> Path:
    root = PHASE_ROOT.resolve()
    parent = path.parent.resolve()
    if parent != root and root not in parent.parents:
        raise RuntimeError(f"Refusing to write outside Phase N2.6: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_output(path)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    ensure_output(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def candidate_config_names(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["architectures"].keys())


def family_for_config(cfg: dict[str, Any], config_name: str) -> str:
    return str(cfg["architectures"][config_name]["family"])


def add_skill_lstm(
    metrics: dict[str, float],
    lstm_rmse_v: float,
    baseline_denominator: dict[str, float | str],
) -> dict[str, float | str]:
    enriched = add_skill(metrics, baseline_denominator)
    enriched["skill_lstm"] = 1.0 - float(metrics["rmse_V"]) / float(lstm_rmse_v)
    enriched["lstm_rmse_V"] = float(lstm_rmse_v)
    return enriched


def load_development_protocol(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], list[tuple[str, WindowSet, WindowSet]]]:
    frame, audit = load_development_data(cfg, cfg["features"])
    audit["test_metrics_computed"] = False
    if (
        audit.get("target_rows_read_from_test") != 0
        or audit.get("test_predictions_constructed") is not False
        or audit.get("test_performance_accessed") is not False
    ):
        raise RuntimeError("TEST FIREWALL VIOLATION")
    windows = fold_windows(frame, cfg)
    for fold, _, validation in windows:
        if np.any(validation.target_end_age_h >= float(cfg["split"]["development_end_h"])):
            raise RuntimeError(f"Validation target reaches Test boundary in {fold}")
    return frame, audit, windows


def ridge_alpha(cfg: dict[str, Any]) -> float:
    path = Path(cfg["phase_n2_root"]) / "outputs" / "baseline_results" / "ridge_alpha_selection.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["selected_ridge_alpha"])


def baseline_by_fold(
    cfg: dict[str, Any], windows: list[tuple[str, WindowSet, WindowSet]]
) -> dict[str, dict[str, Any]]:
    alpha = ridge_alpha(cfg)
    output: dict[str, dict[str, Any]] = {}
    for fold, train, validation in windows:
        predictions = simple_predictions(train, validation, alpha, int(cfg["task"]["empirical_period_steps"]))
        output[fold] = {
            "predictions": predictions,
            "denominators": baseline_denominators(validation.y_abs, predictions),
        }
    return output


def load_local_run(run_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    result_path = PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}.json"
    prediction_path = PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}_predictions.npz"
    if result_path.is_file() and prediction_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8")), dict(np.load(prediction_path, allow_pickle=False))
    return None


def aggregate_seed_rows(fold_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_summary = fold_rows.groupby(["family", "config_name", "seed"], as_index=False, dropna=False).agg(
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        bias_V=("bias_V", "mean"),
        bias_mV=("bias_mV", "mean"),
        r2=("r2", "mean"),
        skill_best=("skill_best", "mean"),
        skill_lstm=("skill_lstm", "mean"),
        worst_fold_skill_lstm=("skill_lstm", "min"),
        improved_fold_count_vs_lstm=("improved_vs_lstm", "sum"),
        parameter_count=("parameter_count", "first"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        mean_epoch_time_s=("mean_epoch_time_s", "mean"),
        total_training_time_s=("total_training_time_s", "sum"),
    )
    seed_summary.insert(0, "fold", "mean_3fold")
    seed_summary.insert(0, "row_type", "seed_summary")
    model_summary = seed_summary.groupby(["family", "config_name"], as_index=False, dropna=False).agg(
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        rmse_mV_seed_std=("rmse_mV", "std"),
        rmse_mV_seed_min=("rmse_mV", "min"),
        rmse_mV_seed_max=("rmse_mV", "max"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        bias_V=("bias_V", "mean"),
        bias_mV=("bias_mV", "mean"),
        r2=("r2", "mean"),
        skill_best=("skill_best", "mean"),
        mean_skill_lstm=("skill_lstm", "mean"),
        skill_lstm_seed_std=("skill_lstm", "std"),
        worst_fold_skill_lstm=("worst_fold_skill_lstm", "min"),
        improved_fold_seed_count_vs_lstm=("improved_fold_count_vs_lstm", "sum"),
        parameter_count=("parameter_count", "first"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        mean_epoch_time_s=("mean_epoch_time_s", "mean"),
        total_training_time_s=("total_training_time_s", "sum"),
        seed_count=("seed", "count"),
    )
    model_summary["rmse_mV_seed_std"] = model_summary["rmse_mV_seed_std"].fillna(0.0)
    model_summary["skill_lstm_seed_std"] = model_summary["skill_lstm_seed_std"].fillna(0.0)
    model_summary.insert(0, "seed", "all_available")
    model_summary.insert(0, "fold", "all_folds")
    model_summary.insert(0, "row_type", "model_summary")
    return seed_summary, model_summary


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return metric_dict(y_true, y_pred)


def safe_float(value: Any) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"Non-finite value: {value}")
    return converted

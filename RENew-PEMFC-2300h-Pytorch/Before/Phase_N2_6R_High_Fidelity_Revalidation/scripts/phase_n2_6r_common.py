"""Shared frozen protocol and fail-closed utilities for Phase N2.6R."""

from __future__ import annotations

import json
import random
import sys
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
CONFIG_PATH = PHASE_ROOT / "configs" / "phase_n2_6r_config.yaml"
N25_SCRIPT_DIR = PROJECT_ROOT / "Phase_N2_5_Cycle_Aware_xLSTM" / "scripts"
if str(N25_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(N25_SCRIPT_DIR))

from phase_n2_5_common import (  # noqa: E402
    FoldPreprocessor,
    WindowSet,
    fold_windows,
    load_development_data,
    load_phase_n2_lstm_reference,
    metric_dict,
    paired_moving_block_bootstrap,
    per_horizon_metrics,
)


FAMILY_NAMES = {"LSTM", "TimesNet", "ModernTCN"}


def load_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if Path(cfg["phase_n2_6r_root"]).resolve() != PHASE_ROOT.resolve():
        raise RuntimeError("Configured Phase N2.6R root does not match script location")
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
            raise RuntimeError(f"Inherited frozen protocol mismatch for {key}: {frozen.get(key)!r}")
    features = list(frozen.get("exact_feature_names", []))
    if len(features) != 14 or len(set(features)) != 14 or features[0] != "cell_voltage_mean":
        raise RuntimeError("Inherited F2 list is not the exact 14-column voltage-first protocol")
    if cfg["task"]["voltage_feature_index"] != 0:
        raise RuntimeError("Voltage target channel must remain feature index 0")
    cfg["frozen"] = frozen
    cfg["features"] = features
    return cfg


def ensure_output(path: Path) -> Path:
    root = PHASE_ROOT.resolve()
    parent = path.parent.resolve()
    if parent != root and root not in parent.parents:
        raise RuntimeError(f"Refusing to write outside Phase N2.6R: {path}")
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


def model_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def candidate_names(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["architectures"])


def family_for_config(cfg: dict[str, Any], config_name: str) -> str:
    if config_name == "LSTM":
        return "LSTM"
    return str(cfg["architectures"][config_name]["family"])


def load_development_protocol(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[tuple[str, WindowSet, WindowSet]]]:
    frame, inherited_audit = load_development_data(cfg, cfg["features"])
    audit = {
        **inherited_audit,
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_rmse_computed": False,
        "test_r2_computed": False,
        "test_mae_computed": False,
        "test_performance_accessed": False,
    }
    if inherited_audit.get("target_rows_read_from_test") != 0:
        raise RuntimeError("TEST FIREWALL VIOLATION: inherited loader read Test targets")
    windows = fold_windows(frame, cfg)
    development_end = float(cfg["split"]["development_end_h"])
    for fold, _, validation in windows:
        if np.any(validation.target_end_age_h >= development_end):
            raise RuntimeError(f"TEST FIREWALL VIOLATION: {fold} reaches Test")
    return frame, audit, windows


def load_local_run(run_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    result_path = PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}.json"
    prediction_path = PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}_predictions.npz"
    if not (result_path.is_file() and prediction_path.is_file()):
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(prediction_path, allow_pickle=False))
    if result.get("status") != "complete":
        return None
    return result, arrays


def mean_fold_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(["family", "config_name", "seed"], as_index=False).agg(
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        bias_V=("bias_V", "mean"),
        bias_mV=("bias_mV", "mean"),
        r2=("r2", "mean"),
        fold_rmse_std_mV=("rmse_mV", "std"),
        parameter_count=("parameter_count", "first"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        mean_epoch_time_s=("mean_epoch_time_s", "mean"),
        total_training_time_s=("total_training_time_s", "sum"),
    )
    summary.insert(0, "fold", "mean_3fold")
    summary.insert(0, "row_type", "config_summary")
    return summary


def verify_no_test_arrays(arrays: dict[str, np.ndarray], cfg: dict[str, Any], run_id: str) -> None:
    required = {"y_true", "y_pred", "anchor", "origin_age_h", "target_end_age_h", "row_origin_index"}
    if not required.issubset(arrays):
        raise RuntimeError(f"Incomplete prediction artifact for {run_id}")
    if arrays["y_true"].shape != arrays["y_pred"].shape or arrays["y_true"].shape[1] != 24:
        raise RuntimeError(f"Prediction shape mismatch for {run_id}")
    if not np.isfinite(arrays["y_true"]).all() or not np.isfinite(arrays["y_pred"]).all():
        raise RuntimeError(f"Non-finite prediction artifact for {run_id}")
    if float(np.max(arrays["target_end_age_h"])) >= float(cfg["split"]["development_end_h"]):
        raise RuntimeError(f"TEST FIREWALL VIOLATION in {run_id}")

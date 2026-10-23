"""Leakage-safe shared utilities for the frozen T1 Canonical TFT study."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE = SCRIPT_DIR.parent
PROJECT = PHASE.parent
REPO = PROJECT.parent
T0 = PROJECT / "Phase_TFT_Original_Baseline"
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
N1_TABLE = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
PROTOCOL = PHASE / "configs" / "PREREGISTERED_T1_PROTOCOL.yaml"
PROTOCOL_SHA = PHASE / "configs" / "PREREGISTERED_T1_PROTOCOL.sha256"

REQUIRED_COLUMNS = [
    "bin_start_ophrs2_h", "bin_end_ophrs2_h", "last_datetime", "temporal_segment_id", "bin_valid_for_windows",
    "cell_voltage_mean", "cell_voltage_std", "cell_voltage_range", "cell_voltage_min", "cell_voltage_max",
    "actual_current_A_mean", "actual_current_A_std", "actual_current_A_peak_to_peak",
    "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std", "actual_current_density_A_cm2_peak_to_peak",
    "load_transition_count", "operation_point_transitions", "zero_current_fraction",
]


class T1Failure(RuntimeError):
    """Protocol failure; callers must stop rather than repair by selection."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def phase_path(*parts: str) -> Path:
    path = PHASE.joinpath(*parts).resolve()
    if path != PHASE.resolve() and PHASE.resolve() not in path.parents:
        raise T1Failure(f"PHASE_PATH_ESCAPE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(value: Any, *parts: str) -> Path:
    path = phase_path(*parts)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, *parts: str) -> Path:
    path = phase_path(*parts)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return path


def write_text(text: str, *parts: str) -> Path:
    path = phase_path(*parts)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def load_protocol() -> tuple[dict[str, Any], str]:
    if not PROTOCOL_SHA.is_file():
        raise T1Failure("T1_PROTOCOL_SHA_MISSING")
    expected = PROTOCOL_SHA.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256_file(PROTOCOL)
    if expected != observed:
        raise T1Failure("T1_PROTOCOL_HASH_MISMATCH")
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if cfg.get("status") != "FROZEN_BEFORE_SEARCH_AND_OUTER_EVALUATION":
        raise T1Failure("T1_PROTOCOL_NOT_FROZEN")
    # The base must be the exact pre-registration ancestry; a later commit is
    # permitted only because the protocol itself is committed before training.
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", str(cfg["base_commit"]), "HEAD"], check=False).returncode != 0:
        raise T1Failure("T1_BASE_COMMIT_NOT_ANCESTOR")
    if int(cfg["search"]["trials_per_outer_fold"]) != 16:
        raise T1Failure("T1_TRIAL_BUDGET_DRIFT")
    if cfg["task"]["input_shape"] != ["batch", 48, 14] or cfg["task"]["output_shape"] != ["batch", 24]:
        raise T1Failure("T1_TENSOR_CONTRACT_DRIFT")
    return cfg, observed


def audit_t0_authority(cfg: dict[str, Any]) -> dict[str, Any]:
    authority = cfg["frozen_t0_authority"]
    model_path = T0 / authority["canonical_tft_relative_path"]
    t0_protocol = T0 / "configs" / "PREREGISTERED_T0_PROTOCOL.yaml"
    observed_model = sha256_file(model_path)
    observed_protocol = sha256_file(t0_protocol)
    if observed_model != authority["canonical_tft_sha256"]:
        raise T1Failure("T0_CANONICAL_MODEL_MUTATION_DETECTED")
    if observed_protocol != authority["t0_protocol_sha256"]:
        raise T1Failure("T0_PROTOCOL_MUTATION_DETECTED")
    return {"canonical_tft_sha256": observed_model, "t0_protocol_sha256": observed_protocol, "pass": True}


def load_bs_common() -> tuple[dict[str, Any], ModuleType]:
    protocol = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
    sha = BS / "configs" / "PREREGISTERED_PROTOCOL.sha256"
    if sha256_file(protocol) != sha.read_text(encoding="utf-8").split()[0].lower():
        raise T1Failure("BS_PROTOCOL_HASH_MISMATCH")
    cfg = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("t1_bs_common", BS / "scripts" / "backbone_common.py")
    if spec is None or spec.loader is None:
        raise T1Failure("BS_COMMON_IMPORT_FAILURE")
    module = importlib.util.module_from_spec(spec)
    sys.modules["t1_bs_common"] = module
    spec.loader.exec_module(module)
    return cfg, module


def load_canonical_tft() -> tuple[type, Any]:
    source = T0 / "models" / "canonical_tft.py"
    spec = importlib.util.spec_from_file_location("t1_frozen_canonical_tft", source)
    if spec is None or spec.loader is None:
        raise T1Failure("CANONICAL_TFT_IMPORT_FAILURE")
    module = importlib.util.module_from_spec(spec)
    sys.modules["t1_frozen_canonical_tft"] = module
    spec.loader.exec_module(module)
    return module.CanonicalStrictHistoryTFT, module.count_parameters


def safe_read_development(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    firewall = cfg["data_firewall"]
    if not N1_TABLE.is_file() or int(firewall["pandas_read_csv_nrows"]) != 3007:
        raise T1Failure("DEVELOPMENT_READER_CONFIGURATION_INVALID")
    prefix = pd.read_csv(N1_TABLE, usecols=REQUIRED_COLUMNS, nrows=3007)
    if len(prefix) != 3007:
        raise T1Failure("DEVELOPMENT_PREFIX_ROW_COUNT_MISMATCH")
    frame = prefix.loc[prefix["bin_end_ophrs2_h"].to_numpy(float) < 1819.997738].copy().reset_index(drop=True)
    if len(frame) != 3005 or np.any(frame["bin_end_ophrs2_h"].to_numpy(float) >= 1819.997738):
        raise T1Failure("DEVELOPMENT_ROW_COUNT_MISMATCH")
    payload = frame.to_csv(index=False, header=True, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return frame, {
        "pandas_read_csv_nrows": 3007,
        "prefix_rows_read": 3007,
        "post_read_filter": "bin_end_ophrs2_h < 1819.997738",
        "development_rows": 3005,
        "development_prefix_fingerprint_sha256": sha256_bytes(payload),
        "full_causal_file_hash_computed": False,
        "full_causal_file_read_detected": False,
        "test_accessed": False,
        "test_rows_read": 0,
        "test_targets_read": 0,
    }


def build_folds(frame: pd.DataFrame, cfg: dict[str, Any], bs_cfg: dict[str, Any], common: ModuleType) -> list[tuple[str, Any, Any]]:
    folds = common.build_outer_folds(frame, bs_cfg)
    expected = cfg["expected_windows"]
    rows: list[dict[str, Any]] = []
    for name, outer_train, validation in folds:
        fit, stopper = common.split_inner_fit_stopper(outer_train, float(cfg["training"]["chronological_inner_stopper_fraction"]))
        observed = {"outer_train_origins": len(outer_train.origin_row), "outer_validation_origins": len(validation.origin_row), "inner_fit_origins": len(fit.origin_row), "inner_stopper_origins": len(stopper.origin_row)}
        if observed != expected[name]:
            raise T1Failure(f"WINDOW_UNIVERSE_MISMATCH_{name}: {observed}")
        if set(fit.target_row.reshape(-1).tolist()) & set(stopper.target_row.reshape(-1).tolist()):
            raise T1Failure("INNER_TARGET_ROW_ISOLATION_FAIL")
        if np.any(validation.target_end_age_h >= 1819.997738):
            raise T1Failure("TEST_FIREWALL_FAIL")
        rows.append({"fold": name, **observed, "target_row_isolation": True, "all_legal_validation_origins_retained": True, "max_validation_target_age_h": float(validation.target_end_age_h.max())})
    write_csv(pd.DataFrame(rows), "outputs", "T1_WINDOW_MANIFEST.csv")
    return folds


def verify_t0_validation_authority(folds: list[tuple[str, Any, Any]]) -> dict[str, Any]:
    """Confirm current frozen windows reproduce every T0 validation key/truth."""
    path = T0 / "outputs" / "TFT_SEED42_PREDICTIONS.csv"
    historical = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for fold, _, validation in folds:
        expected = pd.DataFrame({"origin_row": np.repeat(validation.origin_row, 24), "target_row": validation.target_row.reshape(-1), "horizon_step": np.tile(np.arange(1, 25), len(validation.origin_row)), "y_true_delta_V": validation.y_delta_V.reshape(-1)})
        actual = historical.loc[historical["fold"].eq(fold), ["origin_row", "target_row", "horizon_step", "y_true_delta_V"]].copy()
        merged = expected.merge(actual, on=["origin_row", "target_row", "horizon_step"], how="outer", suffixes=("_current", "_t0"), indicator=True)
        matches = len(merged) == len(expected) and bool((merged["_merge"] == "both").all()) and bool(np.allclose(merged["y_true_delta_V_current"], merged["y_true_delta_V_t0"], rtol=0.0, atol=1e-12))
        if not matches:
            raise T1Failure(f"T0_VALIDATION_AUTHORITY_MISMATCH_{fold}")
        rows.append({"fold": fold, "expected_queries": len(expected), "t0_queries": len(actual), "origin_target_truth_exact_match": True})
    return {"pass": True, "folds": rows}


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def valid_width_combinations(cfg: dict[str, Any]) -> list[tuple[int, int]]:
    values = [(int(row["hidden_size"]), int(row["hidden_continuous_size"])) for row in cfg["search"]["width_combinations"]]
    if len(set(values)) != len(values) or any(hc > h or h % int(cfg["architecture_fixed"]["attention_head_size"]) for h, hc in values):
        raise T1Failure("INVALID_FROZEN_WIDTH_COMBINATION_TABLE")
    return values


def delta_metrics(common: ModuleType, y_true: np.ndarray, y_pred: np.ndarray, target_row: np.ndarray) -> dict[str, float]:
    result = common.delta_metrics(y_true, y_pred, target_row)
    if not np.isfinite(list(result.values())).all():
        raise T1Failure("NONFINITE_OUTER_METRIC")
    return result

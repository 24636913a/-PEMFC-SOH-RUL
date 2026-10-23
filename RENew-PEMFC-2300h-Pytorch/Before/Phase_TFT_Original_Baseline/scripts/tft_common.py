"""Protocol, firewall, frozen-window, and canonical-TFT helpers for T0."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import platform
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
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
N1_TABLE = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
PROTOCOL = PHASE / "configs" / "PREREGISTERED_T0_PROTOCOL.yaml"
PROTOCOL_SHA = PHASE / "configs" / "PREREGISTERED_T0_PROTOCOL.sha256"

REQUIRED_COLUMNS = [
    "bin_start_ophrs2_h", "bin_end_ophrs2_h", "last_datetime", "temporal_segment_id", "bin_valid_for_windows",
    "cell_voltage_mean", "cell_voltage_std", "cell_voltage_range", "cell_voltage_min", "cell_voltage_max",
    "actual_current_A_mean", "actual_current_A_std", "actual_current_A_peak_to_peak",
    "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std", "actual_current_density_A_cm2_peak_to_peak",
    "load_transition_count", "operation_point_transitions", "zero_current_fraction",
]


class T0Failure(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def phase_path(*parts: str) -> Path:
    path = PHASE.joinpath(*parts).resolve()
    if path != PHASE.resolve() and PHASE.resolve() not in path.parents:
        raise T0Failure(f"PHASE_PATH_ESCAPE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: Any, *parts: str) -> Path:
    path = phase_path(*parts)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    expected = PROTOCOL_SHA.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256_file(PROTOCOL)
    if expected != observed:
        raise T0Failure("T0_PROTOCOL_HASH_MISMATCH")
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if cfg.get("status") != "FROZEN_BEFORE_TRAINING":
        raise T0Failure("T0_PROTOCOL_NOT_FROZEN")
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", str(cfg["base_commit"]), "HEAD"], check=False).returncode != 0:
        raise T0Failure("T0_BASE_COMMIT_NOT_ANCESTOR")
    if cfg["task"]["input_shape"] != ["batch", 48, 14] or len(cfg["task"]["raw_f2_feature_order"]) != 14:
        raise T0Failure("T0_INPUT_CONTRACT_DRIFT")
    return cfg, observed


def load_bs_common() -> tuple[dict[str, Any], ModuleType]:
    bs_protocol = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
    bs_sha = BS / "configs" / "PREREGISTERED_PROTOCOL.sha256"
    if sha256_file(bs_protocol) != bs_sha.read_text(encoding="utf-8").split()[0].lower():
        raise T0Failure("BS_PROTOCOL_HASH_MISMATCH")
    bs_cfg = yaml.safe_load(bs_protocol.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("t0_bs_common", BS / "scripts" / "backbone_common.py")
    if spec is None or spec.loader is None:
        raise T0Failure("BS_COMMON_IMPORT_FAILURE")
    module = importlib.util.module_from_spec(spec)
    sys.modules["t0_bs_common"] = module
    spec.loader.exec_module(module)
    return bs_cfg, module


def safe_read_development(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the legal mixed-table prefix and then retain Development only."""
    firewall = cfg["data_firewall"]
    if int(firewall["pandas_read_csv_nrows"]) != 3007 or not N1_TABLE.is_file():
        raise T0Failure("DEVELOPMENT_READER_CONFIGURATION_INVALID")
    prefix = pd.read_csv(N1_TABLE, usecols=REQUIRED_COLUMNS, nrows=3007)
    if len(prefix) != 3007:
        raise T0Failure("DEVELOPMENT_PREFIX_ROW_COUNT_MISMATCH")
    frame = prefix.loc[prefix["bin_end_ophrs2_h"].to_numpy(float) < 1819.997738].copy().reset_index(drop=True)
    if len(frame) != 3005 or np.any(frame["bin_end_ophrs2_h"].to_numpy(float) >= 1819.997738):
        raise T0Failure("DEVELOPMENT_ROW_COUNT_MISMATCH")
    payload = frame.to_csv(index=False, header=True, lineterminator="\n", float_format="%.17g").encode("utf-8")
    audit = {
        "pandas_read_csv_nrows": 3007,
        "prefix_rows_read": 3007,
        "post_read_filter": "bin_end_ophrs2_h < 1819.997738",
        "development_rows": 3005,
        "rows_after_prefix_read": 0,
        "full_causal_file_hash_computed": False,
        "full_causal_file_read_detected": False,
        "development_prefix_fingerprint_sha256": sha256_bytes(payload),
        "test_accessed": False,
        "test_rows_read": 0,
        "test_targets_read": 0,
    }
    return frame, audit


def build_folds(frame: pd.DataFrame, cfg: dict[str, Any], bs_cfg: dict[str, Any], common: ModuleType) -> list[tuple[str, Any, Any]]:
    folds = common.build_outer_folds(frame, bs_cfg)
    expected = cfg["expected_windows"]
    rows: list[dict[str, Any]] = []
    for name, outer_train, outer_validation in folds:
        fit, stopper = common.split_inner_fit_stopper(outer_train, float(cfg["training"]["chronological_inner_stopper_fraction"]))
        observed = {"outer_train_origins": len(outer_train.origin_row), "outer_validation_origins": len(outer_validation.origin_row), "inner_fit_origins": len(fit.origin_row), "inner_stopper_origins": len(stopper.origin_row)}
        if observed != expected[name]:
            raise T0Failure(f"WINDOW_UNIVERSE_MISMATCH_{name}: {observed}")
        if set(fit.target_row.reshape(-1).tolist()) & set(stopper.target_row.reshape(-1).tolist()):
            raise T0Failure("TARGET_ROW_ISOLATION_FAIL")
        if np.any(outer_validation.target_end_age_h >= 1819.997738):
            raise T0Failure("TEST_FIREWALL_FAIL")
        rows.append({"fold": name, **observed, "target_row_isolation": True, "all_validation_origins_retained": True})
    write_csv(pd.DataFrame(rows), "outputs", "T0_WINDOW_MANIFEST.csv")
    return folds


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def runtime_source_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    import pytorch_forecasting as pf
    from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
    import pytorch_forecasting.models.temporal_fusion_transformer.sub_modules as sm
    paths = {"_tft.py_sha256": Path(inspect.getsourcefile(TemporalFusionTransformer)), "sub_modules.py_sha256": Path(inspect.getsourcefile(sm.InterpretableMultiHeadAttention))}
    expected = cfg["upstream_architecture"]["source_files"]
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != expected:
        raise T0Failure(f"PYTORCH_FORECASTING_SOURCE_HASH_MISMATCH: {observed}")
    return {
        "pytorch_forecasting_version": str(pf.__version__), "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
        "python_version": sys.version.split()[0], "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_paths": {key: str(value) for key, value in paths.items()}, "source_hashes": observed,
    }

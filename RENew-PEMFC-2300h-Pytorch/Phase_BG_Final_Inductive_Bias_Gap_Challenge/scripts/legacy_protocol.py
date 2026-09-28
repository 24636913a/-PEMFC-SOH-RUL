"""Read-only recovery of the frozen Phase BS data contract.

This module intentionally does not call the legacy ``load_config`` function:
that function correctly requires the *old* execution HEAD.  Instead this
module verifies the immutable legacy YAML and SHA-256 itself, rejects drift in
protected BS files, then invokes only the already-verified data/window helpers
from the unchanged legacy module.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
BS_REFERENCE_COMMIT = "68f880908f9fd56d35c3788d7a19a0b2d69b4ecd"
LEGACY_CONFIG = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
LEGACY_SHA = BS / "configs" / "PREREGISTERED_PROTOCOL.sha256"
PROTECTED_BS_PATHS = [
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/configs/PREREGISTERED_PROTOCOL.yaml",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/configs/PREREGISTERED_PROTOCOL.sha256",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/FINAL_DECISION.yaml",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/FINAL_BACKBONE_REPORT.md",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/scripts/backbone_common.py",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/window_manifest.csv",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/formal_vs_target_time_weighting.csv",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/per_fold.csv",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions/bs_timesnet_fold_1_seed42.npz",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions/bs_timesnet_fold_2_seed42.npz",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions/bs_timesnet_fold_3_seed42.npz",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions/bs_timesnet_fold_1_seed2024.npz",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions/bs_timesnet_fold_2_seed2024.npz",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions/bs_timesnet_fold_3_seed2024.npz",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions/bs_timesnet_fold_1_seed3407.npz",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions/bs_timesnet_fold_2_seed3407.npz",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions/bs_timesnet_fold_3_seed3407.npz",
]


def phase_path(*parts: str) -> Path:
    path = PHASE.joinpath(*parts)
    resolved = path.resolve()
    if resolved != PHASE.resolve() and PHASE.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing write outside Phase BG: {path}")
    return path


def write_json(payload: Any, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return path


def write_text(text: str, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _protected_drift() -> list[str]:
    command = ["git", "-C", str(REPO), "diff", "--name-only", f"{BS_REFERENCE_COMMIT}..HEAD", "--", *PROTECTED_BS_PATHS]
    return [line for line in subprocess.check_output(command, text=True).splitlines() if line]


def load_legacy_contract() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Verify immutable BS protocol before importing helper functions."""
    if not LEGACY_CONFIG.is_file() or not LEGACY_SHA.is_file():
        raise RuntimeError("Missing frozen Phase BS protocol")
    expected = LEGACY_SHA.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256_file(LEGACY_CONFIG)
    if observed != expected:
        raise RuntimeError("Phase BS protocol SHA256 mismatch")
    drift = _protected_drift()
    if drift:
        raise RuntimeError(f"PROTOCOL_DRIFT_AUDIT required; protected BS files changed: {drift}")
    cfg = yaml.safe_load(LEGACY_CONFIG.read_bytes())
    if cfg.get("base_commit") != "2083d1e46e830d998abc537fd2d671a3319d1fa5":
        raise RuntimeError("Unexpected Phase BS frozen-base identity")
    if int(cfg["data_firewall"]["pandas_read_csv_nrows"]) != 3007 or int(cfg["data_firewall"]["expected_development_rows_after_end_filter"]) != 3005:
        raise RuntimeError("Phase BS registered-prefix contract drift")
    helper_path = BS / "scripts" / "backbone_common.py"
    spec = importlib.util.spec_from_file_location("phase_bs_readonly_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load verified Phase BS helper module")
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    audit = {
        "bg_head": current_head(), "bs_reference_commit": BS_REFERENCE_COMMIT,
        "legacy_protocol_path": str(LEGACY_CONFIG), "legacy_protocol_sha256": observed,
        "legacy_sha_file": str(LEGACY_SHA), "protected_bs_drift": drift,
        "read_only_helper_path": str(helper_path), "old_load_config_called": False,
        "helper_reuse": ["read_development_prefix", "build_windows", "build_outer_folds", "split_inner_fit_stopper", "FoldPreprocessor", "standardized_delta_target", "delta_metrics"],
    }
    return cfg, helpers, audit


def recover_windows_and_controls() -> tuple[dict[str, Any], Any, pd.DataFrame, list[tuple[str, Any, Any]], pd.DataFrame, dict[str, Any]]:
    """Invoke frozen helpers and require exact agreement with BS windows/caches."""
    cfg, helpers, protocol_audit = load_legacy_contract()
    frame, data_audit = helpers.read_development_prefix(cfg)
    folds = helpers.build_outer_folds(frame, cfg)
    prior_manifest = pd.read_csv(BS / "outputs" / "window_manifest.csv")
    expected = {"fold_1": (1048, 233, 867, 158, 1130.5), "fold_2": (1281, 245, 1065, 193, 1383.0), "fold_3": (1526, 466, 1281, 229, 1819.5)}
    records: list[dict[str, Any]] = []
    for fold, train, valid in folds:
        fit, stopper = helpers.split_inner_fit_stopper(train, .15)
        target_overlap = set(fit.target_row.reshape(-1).tolist()) & set(stopper.target_row.reshape(-1).tolist())
        current_tuple = (len(train.origin_row), len(valid.origin_row), len(fit.origin_row), len(stopper.origin_row), float(valid.target_end_age_h.max()))
        if current_tuple != expected[fold] or target_overlap:
            raise RuntimeError(f"{fold}: recovered windows differ from registered Phase BS identity")
        old_row = prior_manifest.loc[prior_manifest.fold.eq(fold)]
        if len(old_row) != 1:
            raise RuntimeError(f"{fold}: missing or duplicate Phase BS window-manifest row")
        values = old_row.iloc[0]
        manifest_match = (
            int(values.outer_train_origins) == current_tuple[0] and int(values.outer_validation_origins) == current_tuple[1]
            and int(values.inner_fit_origins) == current_tuple[2] and int(values.inner_stopper_origins) == current_tuple[3]
            and int(values.inner_target_overlap_count) == 0 and float(values.validation_max_target_end_age_h) == current_tuple[4]
        )
        if not manifest_match:
            raise RuntimeError(f"{fold}: Phase BS manifest mismatch")
        for seed in (42, 2024, 3407):
            cache = BS / "outputs" / "predictions" / f"bs_timesnet_{fold}_seed{seed}.npz"
            with np.load(cache, allow_pickle=False) as z:
                required = {"y_true_delta_V", "anchor_V", "origin_row", "target_row", "target_end_age_h", "horizon_step"}
                if required - set(z.files):
                    raise RuntimeError(f"{cache.name}: missing frozen control keys")
                key_match = np.array_equal(z["origin_row"], valid.origin_row) and np.array_equal(z["target_row"], valid.target_row)
                truth_match = np.array_equal(z["y_true_delta_V"], valid.y_delta_V)
                anchor_match = np.array_equal(z["anchor_V"], valid.anchor_V)
                target_match = np.array_equal(z["target_end_age_h"], valid.target_end_age_h)
                horizon_match = np.array_equal(z["horizon_step"], np.arange(1, 25, dtype=np.int64))
                if not (key_match and truth_match and anchor_match and target_match and horizon_match):
                    raise RuntimeError(f"{cache.name}: frozen TimesNet control identity mismatch")
                records.append({"fold": fold, "seed": seed, "outer_train_origins": current_tuple[0], "outer_validation_origins": current_tuple[1], "inner_fit_origins": current_tuple[2], "inner_stopper_origins": current_tuple[3], "inner_target_overlap_count": 0, "origin_row_match": key_match, "target_row_match": key_match, "anchor_match": anchor_match, "truth_match": truth_match, "target_end_match": target_match, "horizon_shape": str(z["y_true_delta_V"].shape), "segment_rule_revalidated": True, "max_target_end_age_h": current_tuple[4], "test_accessed": False})
    audit = {**protocol_audit, "data_audit": data_audit, "window_identity_pass": True, "test_accessed": False, "test_feature_rows": 0, "test_target_rows": 0, "test_predictions": 0, "test_metrics": 0}
    return cfg, helpers, frame, folds, pd.DataFrame(records), audit

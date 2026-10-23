"""Read-only exact recovery of the frozen Phase BS Development contract."""
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


PI = Path(__file__).resolve().parents[1]
PROJECT = PI.parent
REPO = PROJECT.parent
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
BG = PROJECT / "Phase_BG_Final_Inductive_Bias_Gap_Challenge"
BASE_COMMIT = "1b59d924bb12468f37debe4ab550c044d60466f7"
ASSUMPTIONS = PI / "configs" / "EXPLORATORY_ARCHITECTURE_ASSUMPTIONS.yaml"
ASSUMPTIONS_SHA = PI / "configs" / "EXPLORATORY_ARCHITECTURE_ASSUMPTIONS.sha256"
GATE = PI / "configs" / "PREREGISTERED_POTENTIAL_GATE.yaml"
GATE_SHA = PI / "configs" / "PREREGISTERED_POTENTIAL_GATE.sha256"
PROTECTED = [
    "参考论文/PDFNet.pdf",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/configs/PREREGISTERED_PROTOCOL.yaml",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/configs/PREREGISTERED_PROTOCOL.sha256",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/scripts/backbone_common.py",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/window_manifest.csv",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/formal_vs_target_time_weighting.csv",
    "RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/outputs/predictions",
    "RENew-PEMFC-2300h-Pytorch/Phase_BG_Final_Inductive_Bias_Gap_Challenge/final/FINAL_DECISION.yaml",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_path(*parts: str) -> Path:
    target = PI.joinpath(*parts).resolve()
    if target != PI.resolve() and PI.resolve() not in target.parents:
        raise RuntimeError(f"Refusing write outside Phase PI: {target}")
    return target


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


def current_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _frozen_yaml(path: Path, checksum: Path, expected_status: str) -> dict[str, Any]:
    expected = checksum.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"Frozen configuration SHA mismatch: {path.name}")
    data = yaml.safe_load(path.read_bytes())
    if data.get("status") != expected_status:
        raise RuntimeError(f"Frozen configuration status mismatch: {path.name}")
    return data


def load_frozen_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    assumptions = _frozen_yaml(ASSUMPTIONS, ASSUMPTIONS_SHA, "FROZEN_BEFORE_ANY_DEVELOPMENT_PERFORMANCE")
    gate = _frozen_yaml(GATE, GATE_SHA, "FROZEN_BEFORE_ANY_DEVELOPMENT_PERFORMANCE")
    if assumptions.get("base_commit") != BASE_COMMIT:
        raise RuntimeError("Phase PI base commit identity mismatch")
    return assumptions, gate


def protected_drift() -> list[str]:
    """Return protected paths changed in commits *or* the working tree.

    Phase PI itself is deliberately uncommitted while it is being assembled, so
    the comparison is restricted to the immutable paper/BS/BG sources.  Using
    ``git diff BASE`` (rather than only ``BASE..HEAD``) also prevents a local
    edit to a historical source from silently changing the protocol.
    """
    committed = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "--name-only", f"{BASE_COMMIT}..HEAD", "--", *PROTECTED],
        text=True,
    ).splitlines()
    worktree = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "--name-only", BASE_COMMIT, "--", *PROTECTED],
        text=True,
    ).splitlines()
    return sorted({line for line in [*committed, *worktree] if line})


def load_phase_bs_contract() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    assumptions, _gate = load_frozen_configs()
    drift = protected_drift()
    if drift:
        raise RuntimeError(f"PROTOCOL_DRIFT_AUDIT required: {drift}")
    legacy_cfg_path = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
    legacy_sha_path = BS / "configs" / "PREREGISTERED_PROTOCOL.sha256"
    expected, observed = legacy_sha_path.read_text(encoding="utf-8").split()[0].lower(), sha256(legacy_cfg_path)
    if observed != expected or observed != assumptions["data_contract"]["protocol_sha256"]:
        raise RuntimeError("Phase BS frozen protocol SHA mismatch")
    legacy_cfg = yaml.safe_load(legacy_cfg_path.read_bytes())
    if legacy_cfg.get("base_commit") != "2083d1e46e830d998abc537fd2d671a3319d1fa5":
        raise RuntimeError("Unexpected Phase BS base identity")
    helper_path = BS / "scripts" / "backbone_common.py"
    spec = importlib.util.spec_from_file_location("phase_pi_readonly_bs_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Phase BS read-only helpers")
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    audit = {
        "phase_pi_head": current_head(), "base_commit": BASE_COMMIT,
        "protected_drift": drift, "phase_bs_protocol_sha256": observed,
        "old_phase_bs_load_config_called": False,
        "paper_sha256": sha256(REPO / assumptions["paper"]["path"]),
        "paper_hash_matches_assumptions": sha256(REPO / assumptions["paper"]["path"]) == assumptions["paper"]["sha256"],
        "test_accessed": False,
    }
    if not audit["paper_hash_matches_assumptions"]:
        raise RuntimeError("PDFNet paper SHA drift")
    return legacy_cfg, helpers, audit


def recover_windows_and_timesnet() -> tuple[dict[str, Any], Any, pd.DataFrame, list[tuple[str, Any, Any]], pd.DataFrame, dict[str, Any]]:
    """Exact registered-prefix recovery plus frozen TimesNet cache identity check."""
    cfg, helpers, protocol = load_phase_bs_contract()
    frame, data_audit = helpers.read_development_prefix(cfg)
    folds = helpers.build_outer_folds(frame, cfg)
    expected = {"fold_1": (1048, 233, 867, 158, 1130.5), "fold_2": (1281, 245, 1065, 193, 1383.0), "fold_3": (1526, 466, 1281, 229, 1819.5)}
    registered = pd.read_csv(BS / "outputs" / "window_manifest.csv")
    records: list[dict[str, Any]] = []
    for fold, train, valid in folds:
        fit, stopper = helpers.split_inner_fit_stopper(train, .15)
        current = (len(train.origin_row), len(valid.origin_row), len(fit.origin_row), len(stopper.origin_row), float(valid.target_end_age_h.max()))
        if current != expected[fold] or set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)):
            raise RuntimeError(f"{fold}: Phase BS window identity mismatch")
        registered_row = registered.loc[registered.fold.eq(fold)]
        if len(registered_row) != 1:
            raise RuntimeError(f"{fold}: invalid registered manifest")
        manifest = registered_row.iloc[0]
        if not (int(manifest.outer_train_origins) == current[0] and int(manifest.outer_validation_origins) == current[1] and int(manifest.inner_fit_origins) == current[2] and int(manifest.inner_stopper_origins) == current[3] and int(manifest.inner_target_overlap_count) == 0 and float(manifest.validation_max_target_end_age_h) == current[4]):
            raise RuntimeError(f"{fold}: registered window manifest mismatch")
        for seed in (42, 2024, 3407):
            cache = BS / "outputs" / "predictions" / f"bs_timesnet_{fold}_seed{seed}.npz"
            with np.load(cache, allow_pickle=False) as z:
                required = {"y_true_delta_V", "anchor_V", "origin_row", "target_row", "target_end_age_h", "horizon_step"}
                if required - set(z.files):
                    raise RuntimeError(f"{cache.name}: required control key missing")
                checks = {
                    "origin_row_match": np.array_equal(z["origin_row"], valid.origin_row),
                    "target_row_match": np.array_equal(z["target_row"], valid.target_row),
                    "truth_match": np.array_equal(z["y_true_delta_V"], valid.y_delta_V),
                    "anchor_match": np.array_equal(z["anchor_V"], valid.anchor_V),
                    "target_end_match": np.array_equal(z["target_end_age_h"], valid.target_end_age_h),
                    "horizon_match": np.array_equal(z["horizon_step"], np.arange(1, 25, dtype=np.int64)),
                }
                if not all(checks.values()):
                    raise RuntimeError(f"{cache.name}: TimesNet cache identity mismatch")
            records.append({"fold": fold, "seed": seed, "outer_train_origins": current[0], "outer_validation_origins": current[1], "inner_fit_origins": current[2], "inner_stopper_origins": current[3], "inner_target_overlap_count": 0, "max_target_end_age_h": current[4], "temporal_segment_rule_revalidated": True, "test_accessed": False, **checks})
    audit = {
        **protocol,
        "raw_prefix_rows_read": int(data_audit["prefix_rows_read"]),
        "development_rows_after_filter": int(data_audit["development_rows_retained"]),
        "boundary_rows_filtered": int(data_audit["prefix_rows_read"] - data_audit["development_rows_retained"]),
        "post_boundary_rows_entered_pipeline": 0,
        "test_accessed": False,
        "test_feature_rows": 0,
        "test_target_rows": 0,
        "test_predictions": 0,
        "test_metrics": 0,
    }
    return cfg, helpers, frame, folds, pd.DataFrame(records), audit


def frozen_timesnet_summary() -> pd.DataFrame:
    table = pd.read_csv(BS / "outputs" / "formal_vs_target_time_weighting.csv")
    result = table.loc[table.model.eq("TimesNet")].copy()
    if set(result.seed) != {42, 2024, 3407}:
        raise RuntimeError("Frozen TimesNet controls are incomplete")
    return result

"""Formal H1 next-contiguous-cycle causal P1S-retention forecasting.

This runner has one intentionally narrow task.  It reconstructs the immutable
PAA P1S measurement envelopes before 1719.31 h, freezes a Development-only
H1 sample authority, and evaluates fixed causal baselines and the frozen-
configuration TimesNet family.  It never opens Test raw data, Test labels, or
the mixed N1 feature file beyond its legally readable prefix.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yaml


sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
CONFIG = PHASE / "configs" / "CPRF_H1_FROZEN_PROTOCOL.yaml"
CONFIG_SHA = PHASE / "configs" / "CPRF_H1_FROZEN_PROTOCOL.sha256"
INPUT = PHASE / "inputs"
OUT = PHASE / "outputs"
REPORT = PHASE / "reports"
FIG = PHASE / "figures"

CTAE = PROJECT / "Phase_CTAE_P1S_Reference_Availability_Erratum"
CTAE_AUDIT = CTAE / "outputs" / "P1S_REFERENCE_AVAILABILITY_AUDIT.csv"
CTAE_SLOPES = CTAE / "outputs" / "POST_REFERENCE_CAUSAL_SLOPE_AVAILABILITY.csv"
PHI = PROJECT / "Phase_PHI_P1S_B2_Health_Index_Consolidation"
PHI_IDENTITY = PHI / "outputs" / "P1S_B2_FULL_HEALTH_TRAJECTORY.csv"
PAA = PROJECT / "Phase_PAA_P1_StateToken_PreAnchor_Alignment"
PAA_SCRIPT = PAA / "scripts" / "run_p1_state_token_preanchor_audit.py"
CSHSE = PROJECT / "Phase_CSHSE_Causal_SameCycle_Health_State_Estimation"
CSHSE_SCRIPT = CSHSE / "scripts" / "run_cshse.py"
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
BS_PROTOCOL = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
BS_VENDOR = BS / "vendor" / "TimesNet_official"
CAUSAL = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"

DEVELOPMENT_BOUNDARY = 1819.997738
PREFIX_NROWS = 3007
DEVELOPMENT_ROWS = 3005
SEEDS = (42, 2024, 3407)
ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
FOLDS = (
    ("fold_1", 72.788690, 946.393214, 946.393214, 1164.794345),
    ("fold_2", 72.788690, 1164.794345, 1164.794345, 1383.195476),
    ("fold_3", 72.788690, 1383.195476, 1383.195476, 1819.997738),
)
FEATURES = [
    "cell_voltage_mean", "cell_voltage_std", "cell_voltage_range", "cell_voltage_min", "cell_voltage_max",
    "actual_current_A_mean", "actual_current_A_std", "actual_current_A_peak_to_peak",
    "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std",
    "actual_current_density_A_cm2_peak_to_peak", "load_transition_count", "operation_point_transitions",
    "zero_current_fraction",
]
CAUSAL_COLUMNS = ["bin_start_ophrs2_h", "bin_end_ophrs2_h", "bin_valid_for_windows", "temporal_segment_id", *FEATURES]
PALETTE = {"navy": "#164B7A", "blue": "#2C7FB8", "orange": "#E07A32", "green": "#2E8B57", "gray": "#767676", "red": "#BC3C3C"}


class ProtocolFailure(RuntimeError):
    """A frozen sample, causal-access, or leakage rule failed closed."""


def ensure_dirs() -> None:
    for path in (INPUT, OUT, REPORT, FIG):
        path.mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256_file(path: Path) -> str:
    if path.resolve() == CAUSAL.resolve():
        raise ProtocolFailure("FULL_CAUSAL_FILE_HASH_FORBIDDEN")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = CONFIG_SHA.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(CONFIG)
    if observed != expected:
        raise ProtocolFailure("CPRF_H1_PROTOCOL_SHA_MISMATCH")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")), observed


def development_prefix_fingerprint(frame: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    """Hash exactly the already-filtered 3005-row legal Development object."""
    ordered = frame.loc[:, CAUSAL_COLUMNS].copy()
    payload = ordered.to_csv(index=False, header=True, lineterminator="\n", float_format="%.17g", na_rep="<NA>").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "name": "DEVELOPMENT_PREFIX_FINGERPRINT",
        "development_prefix_fingerprint_sha256": digest,
        "row_count": int(len(ordered)),
        "column_count": int(len(ordered.columns)),
        "column_order": ordered.columns.tolist(),
        "serialization": {"encoding": "utf-8", "index": False, "header": True, "line_terminator": "LF", "float_format": "%.17g", "na_rep": "<NA>", "row_order": "original_filtered_prefix_order"},
        "legal_read_contract": "pd.read_csv(usecols=CAUSAL_COLUMNS, nrows=3007), then bin_end_ophrs2_h < 1819.997738",
        "not_a_full_source_file_hash": True,
    }
    return digest, manifest


def read_causal_development_prefix() -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """The sole production reader for the mixed Development/Test physical CSV."""
    prefix = pd.read_csv(CAUSAL, usecols=CAUSAL_COLUMNS, nrows=PREFIX_NROWS)
    if len(prefix) != PREFIX_NROWS:
        raise ProtocolFailure("DEVELOPMENT_PREFIX_ROW_COUNT_MISMATCH")
    development = prefix.loc[prefix["bin_end_ophrs2_h"].to_numpy(float) < DEVELOPMENT_BOUNDARY].copy().reset_index(drop=True)
    if len(development) != DEVELOPMENT_ROWS:
        raise ProtocolFailure("DEVELOPMENT_PREFIX_FILTER_ROW_COUNT_MISMATCH")
    if not (development["bin_end_ophrs2_h"].to_numpy(float) < DEVELOPMENT_BOUNDARY).all():
        raise ProtocolFailure("DEVELOPMENT_BOUNDARY_FILTER_FAILED")
    fingerprint, manifest = development_prefix_fingerprint(development)
    return development, fingerprint, manifest


def dynamic_import(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolFailure(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reconstruct_all_causal_p1s_intervals(ctae: pd.DataFrame) -> pd.DataFrame:
    """Replay frozen PAA raw membership up to 1719.31 h; no Test source is opened."""
    paa = dynamic_import(PAA_SCRIPT, "cprf_h1_paa")
    cshse = dynamic_import(CSHSE_SCRIPT, "cprf_h1_cshse_replay")
    events = ctae.loc[ctae["causal_target_eligible"].astype(bool), ["cycle_interval_id"]].copy()
    replay = cshse.raw_reconstruction(paa, events, voltage_tolerance=1e-10)
    if len(replay) != len(events) or not replay["voltage_match_pass"].astype(bool).all() or not replay["dwell_count_pass"].astype(bool).all():
        raise ProtocolFailure("P1S_MEASUREMENT_INTERVAL_RECONSTRUCTION_MISMATCH")
    if replay["raw_test_accessed"].astype(bool).any():
        raise ProtocolFailure("P1S_REPLAY_TEST_RAW_ACCESS")
    if replay["p1s_measurement_end_ophrs2_h"].to_numpy(float).max() >= DEVELOPMENT_BOUNDARY:
        raise ProtocolFailure("P1S_REPLAY_CROSSED_DEVELOPMENT_BOUNDARY")
    return replay


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, float, list[dict[str, Any]]]:
    ctae = pd.read_csv(CTAE_AUDIT)
    slopes = pd.read_csv(CTAE_SLOPES)
    identity = pd.read_csv(PHI_IDENTITY, usecols=["cycle_interval_id", "raw_cycle_id", "cycle_start_ophrs2_h", "cycle_end_ophrs2_h"])
    required_ctae = {"cycle_interval_id", "P1S_voltage_V", "V_ref_V", "causal_target_eligible", "R_P1S_causal", "condition_support_status"}
    required_slopes = {"cycle_interval_id", "R_P1S_causal", "slope_24h", "slope_24h_available", "slope_48h", "slope_48h_available", "slope_96h", "slope_96h_available"}
    if required_ctae - set(ctae) or required_slopes - set(slopes):
        raise ProtocolFailure("CTAE_SOURCE_SCHEMA_MISMATCH")
    ctae = ctae.loc[ctae["causal_target_eligible"].astype(bool)].copy()
    if len(ctae) != 131 or not ctae["V_ref_available_at_event"].astype(bool).all():
        raise ProtocolFailure("CTAE_CAUSAL_TARGET_AUTHORITY_MISMATCH")
    reference_values = ctae["V_ref_V"].dropna().unique()
    if len(reference_values) != 1:
        raise ProtocolFailure("CAUSAL_REFERENCE_VOLTAGE_AMBIGUOUS")
    vref = float(reference_values[0])
    ctae = ctae.merge(identity, on="cycle_interval_id", how="left", validate="one_to_one")
    ctae = ctae.merge(slopes, on=["cycle_interval_id", "R_P1S_causal"], how="left", validate="one_to_one")
    if ctae["raw_cycle_id"].isna().any() or len(ctae) != 131:
        raise ProtocolFailure("P1_RAW_CYCLE_IDENTITY_MISMATCH")
    ctae["raw_cycle_id"] = ctae["raw_cycle_id"].astype(int)
    if ctae["raw_cycle_id"].duplicated().any():
        raise ProtocolFailure("DUPLICATE_CAUSAL_RAW_CYCLE_ID")
    sources = [
        {"path": str(CTAE_AUDIT.relative_to(REPO)).replace("\\", "/"), "role": "causal_raw_retention_target_and_support", "sha256": sha256_file(CTAE_AUDIT)},
        {"path": str(CTAE_SLOPES.relative_to(REPO)).replace("\\", "/"), "role": "causal_origin_slopes", "sha256": sha256_file(CTAE_SLOPES)},
        {"path": str(PHI_IDENTITY.relative_to(REPO)).replace("\\", "/"), "role": "raw_cycle_identity_only_no_health_label_used", "sha256": sha256_file(PHI_IDENTITY)},
        {"path": str(PAA_SCRIPT.relative_to(REPO)).replace("\\", "/"), "role": "frozen_P1S_raw_measurement_membership_replay", "sha256": sha256_file(PAA_SCRIPT)},
        {"path": str(CSHSE_SCRIPT.relative_to(REPO)).replace("\\", "/"), "role": "reused_PAA_envelope_replay_implementation", "sha256": sha256_file(CSHSE_SCRIPT)},
        {"path": str(BS_PROTOCOL.relative_to(REPO)).replace("\\", "/"), "role": "TimesNet_architecture_training_authority", "sha256": sha256_file(BS_PROTOCOL)},
    ]
    return ctae.sort_values("raw_cycle_id").reset_index(drop=True), slopes, vref, sources


def choose_latest_legal_window(causal: pd.DataFrame, origin_end_h: float, target_start_h: float) -> tuple[pd.DataFrame | None, str]:
    candidates = np.flatnonzero(causal["bin_end_ophrs2_h"].to_numpy(float) <= origin_end_h + 1e-9)
    for end in candidates[::-1]:
        start = int(end) - 47
        if start < 0:
            continue
        window = causal.iloc[start:end + 1]
        if len(window) != 48 or not window["bin_valid_for_windows"].astype(bool).all():
            continue
        if window["temporal_segment_id"].nunique() != 1:
            continue
        if not np.allclose(np.diff(window["bin_start_ophrs2_h"].to_numpy(float)), 0.5, atol=1e-6, rtol=0.0):
            continue
        if not np.isfinite(window[FEATURES].to_numpy(float)).all():
            continue
        if float(window["bin_end_ophrs2_h"].iloc[-1]) > origin_end_h + 1e-9:
            continue
        if not (window["bin_end_ophrs2_h"].to_numpy(float) < target_start_h - 1e-9).all():
            raise ProtocolFailure("TARGET_P1S_MEASUREMENT_LEAKAGE")
        return window.copy(), "LEGAL"
    return None, "NO_LEGAL_48_BIN_CAUSAL_HISTORY"


def construct_h1_samples(ctae: pd.DataFrame, replay: pd.DataFrame, causal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    replay_cols = ["cycle_interval_id", "p1s_measurement_start_ophrs2_h", "p1s_measurement_end_ophrs2_h", "P1S_voltage_frozen_V", "P1S_voltage_reconstructed_V", "P1S_voltage_difference_V", "p1s_dwell_count", "p1s_raw_sample_count"]
    table = ctae.merge(replay[replay_cols], on="cycle_interval_id", how="left", validate="one_to_one")
    if table["p1s_measurement_start_ophrs2_h"].isna().any():
        missing = table.loc[table["p1s_measurement_start_ophrs2_h"].isna(), "cycle_interval_id"].tolist()
        raise ProtocolFailure(f"MISSING_RAW_P1S_INTERVAL_FOR_CAUSAL_CYCLE:{missing}")
    if not np.allclose(table["P1S_voltage_V"].to_numpy(float), table["P1S_voltage_reconstructed_V"].to_numpy(float), atol=1e-10, rtol=0.0):
        raise ProtocolFailure("CTAE_AND_PAA_P1S_VOLTAGE_MISMATCH")
    table = table.sort_values("raw_cycle_id").reset_index(drop=True)
    by_raw = {int(row.raw_cycle_id): row for row in table.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    windows: list[np.ndarray] = []
    for origin_raw, origin in by_raw.items():
        target_raw = origin_raw + 1
        target = by_raw.get(target_raw)
        base = {"origin_cycle_interval_id": str(origin.cycle_interval_id), "origin_raw_cycle_id": origin_raw, "required_target_raw_cycle_id": target_raw}
        if target is None:
            exclusions.append({**base, "exclusion_reason": "NEXT_RAW_CYCLE_NOT_CAUSAL_P1S_ELIGIBLE"})
            continue
        window, status = choose_latest_legal_window(causal, float(origin.p1s_measurement_end_ophrs2_h), float(target.p1s_measurement_start_ophrs2_h))
        if window is None:
            exclusions.append({**base, "target_cycle_interval_id": str(target.cycle_interval_id), "exclusion_reason": status})
            continue
        input_start = float(window["bin_start_ophrs2_h"].iloc[0])
        input_end = float(window["bin_end_ophrs2_h"].iloc[-1])
        lead = float(target.p1s_measurement_end_ophrs2_h - origin.p1s_measurement_end_ophrs2_h)
        if lead <= 0 or input_end > float(origin.p1s_measurement_end_ophrs2_h) + 1e-9 or input_end >= float(target.p1s_measurement_start_ophrs2_h) - 1e-9:
            raise ProtocolFailure("H1_TIMELINE_CONTRACT_FAILED")
        rows.append({
            **base,
            "target_cycle_interval_id": str(target.cycle_interval_id), "target_raw_cycle_id": target_raw,
            "origin_p1s_measurement_start_ophrs2_h": float(origin.p1s_measurement_start_ophrs2_h),
            "origin_p1s_measurement_end_ophrs2_h": float(origin.p1s_measurement_end_ophrs2_h),
            "target_p1s_measurement_start_ophrs2_h": float(target.p1s_measurement_start_ophrs2_h),
            "target_p1s_measurement_end_ophrs2_h": float(target.p1s_measurement_end_ophrs2_h),
            "origin_R_P1S_causal": float(origin.R_P1S_causal), "target_R_P1S_causal": float(target.R_P1S_causal),
            "lead_to_target_end_h": lead,
            "input_start_ophrs2_h": input_start, "input_end_ophrs2_h": input_end,
            "input_start_bin_index": int(window.index[0]), "input_end_bin_index": int(window.index[-1]),
            "temporal_segment_id": int(window["temporal_segment_id"].iloc[0]),
            "origin_slope_24h": float(origin.slope_24h) if bool(origin.slope_24h_available) else np.nan,
            "origin_slope_48h": float(origin.slope_48h) if bool(origin.slope_48h_available) else np.nan,
            "origin_slope_96h": float(origin.slope_96h) if bool(origin.slope_96h_available) else np.nan,
            "origin_slope_24h_available": bool(origin.slope_24h_available),
            "origin_slope_48h_available": bool(origin.slope_48h_available),
            "origin_slope_96h_available": bool(origin.slope_96h_available),
            "target_condition_support_status": str(target.condition_support_status),
            "target_measurement_excluded": True, "future_target_skipped": False,
        })
        windows.append(window[FEATURES].to_numpy(np.float32))
    samples = pd.DataFrame(rows).sort_values("target_p1s_measurement_end_ophrs2_h").reset_index(drop=True)
    excluded = pd.DataFrame(exclusions).sort_values(["origin_raw_cycle_id"]).reset_index(drop=True)
    if samples.empty:
        raise ProtocolFailure("NO_LEGAL_H1_SAMPLES")
    if not (samples["target_raw_cycle_id"].to_numpy(int) == samples["origin_raw_cycle_id"].to_numpy(int) + 1).all():
        raise ProtocolFailure("H1_CONTIGUITY_VIOLATION")
    if samples["target_p1s_measurement_end_ophrs2_h"].to_numpy(float).max() >= DEVELOPMENT_BOUNDARY:
        raise ProtocolFailure("H1_TARGET_CROSSED_DEVELOPMENT_BOUNDARY")
    if not (samples["input_end_ophrs2_h"].to_numpy(float) <= samples["origin_p1s_measurement_end_ophrs2_h"].to_numpy(float) + 1e-9).all():
        raise ProtocolFailure("ORIGIN_CUTOFF_VIOLATION")
    if not (samples["input_end_ophrs2_h"].to_numpy(float) < samples["target_p1s_measurement_start_ophrs2_h"].to_numpy(float) - 1e-9).all():
        raise ProtocolFailure("TARGET_MEASUREMENT_EXCLUSION_VIOLATION")
    tensor = np.stack(windows).astype(np.float32)
    if tensor.shape != (len(samples), 48, 14) or not np.isfinite(tensor).all():
        raise ProtocolFailure("H1_INPUT_TENSOR_CONTRACT_FAILED")
    return samples, excluded, tensor


def fold_expand(samples: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold, train_lo, train_hi, val_lo, val_hi in FOLDS:
        for role, lo, hi in (("train", train_lo, train_hi), ("validation", val_lo, val_hi)):
            mask = (samples["target_p1s_measurement_start_ophrs2_h"].to_numpy(float) >= lo) & (samples["target_p1s_measurement_end_ophrs2_h"].to_numpy(float) < hi)
            for sample_index in np.flatnonzero(mask):
                records.append({"fold": fold, "role": role, "sample_index": int(sample_index), **samples.iloc[int(sample_index)].to_dict()})
    authority = pd.DataFrame(records).sort_values(["fold", "role", "target_p1s_measurement_end_ophrs2_h"]).reset_index(drop=True)
    if authority.empty or authority.duplicated(["fold", "role", "origin_raw_cycle_id", "target_raw_cycle_id"]).any():
        raise ProtocolFailure("H1_FOLD_AUTHORITY_DUPLICATE_OR_EMPTY")
    expected_by_fold = {fold: (train_lo, train_hi, val_lo, val_hi) for fold, train_lo, train_hi, val_lo, val_hi in FOLDS}
    for row in authority.itertuples(index=False):
        train_lo, train_hi, val_lo, val_hi = expected_by_fold[str(row.fold)]
        lo, hi = (train_lo, train_hi) if row.role == "train" else (val_lo, val_hi)
        if not (float(row.target_p1s_measurement_start_ophrs2_h) >= lo and float(row.target_p1s_measurement_end_ophrs2_h) < hi):
            raise ProtocolFailure("TARGET_INTERVAL_FOLD_ASSIGNMENT_VIOLATION")
    return authority


def save_stage_a(samples: pd.DataFrame, excluded: pd.DataFrame, authority: pd.DataFrame, replay: pd.DataFrame, fingerprint: str, manifest: dict[str, Any], sources: list[dict[str, Any]], protocol_sha: str) -> dict[str, Any]:
    authority_path = INPUT / "H1_FROZEN_SAMPLE_AUTHORITY.csv"
    write_csv(authority, authority_path)
    write_csv(excluded, INPUT / "H1_SAMPLE_EXCLUSION_LOG.csv")
    write_csv(replay, OUT / "P1S_RAW_MEASUREMENT_INTERVAL_RECONSTRUCTION.csv")
    write_json(manifest, OUT / "DEVELOPMENT_PREFIX_FINGERPRINT_MANIFEST.json")
    authority_sha = sha256_file(authority_path)
    (INPUT / "H1_FROZEN_SAMPLE_AUTHORITY.sha256").write_text(f"{authority_sha}  H1_FROZEN_SAMPLE_AUTHORITY.csv\n", encoding="utf-8")
    checks = {
        "protocol_sha256": protocol_sha, "sample_authority_sha256": authority_sha,
        "development_prefix_fingerprint_sha256": fingerprint,
        "all_target_raw_cycle_ids_equal_origin_plus_one": bool((samples.target_raw_cycle_id.to_numpy(int) == samples.origin_raw_cycle_id.to_numpy(int) + 1).all()),
        "target_measurement_excluded_from_inputs": bool((samples.input_end_ophrs2_h.to_numpy(float) < samples.target_p1s_measurement_start_ophrs2_h.to_numpy(float) - 1e-9).all()),
        "origin_cutoff_obeyed": bool((samples.input_end_ophrs2_h.to_numpy(float) <= samples.origin_p1s_measurement_end_ophrs2_h.to_numpy(float) + 1e-9).all()),
        "all_targets_before_development_boundary": bool(samples.target_p1s_measurement_end_ophrs2_h.to_numpy(float).max() < DEVELOPMENT_BOUNDARY),
        "PAA_raw_replay_voltage_match": bool(replay.voltage_match_pass.astype(bool).all()),
        "PAA_raw_replay_dwell_count_match": bool(replay.dwell_count_pass.astype(bool).all()),
        "PAA_raw_replay_test_accessed": False,
        "PAA_raw_replay_test_not_accessed": True,
        "test_accessed": False,
        "test_not_accessed": True,
        "source_manifest": sources,
    }
    required_passes = ("all_target_raw_cycle_ids_equal_origin_plus_one", "target_measurement_excluded_from_inputs", "origin_cutoff_obeyed", "all_targets_before_development_boundary", "PAA_raw_replay_voltage_match", "PAA_raw_replay_dwell_count_match", "PAA_raw_replay_test_not_accessed", "test_not_accessed")
    if not all(bool(checks[key]) for key in required_passes):
        raise ProtocolFailure("STAGE_A_AUTHORITY_CHECK_FAILED")
    return checks


def fold_positions(authority: pd.DataFrame, fold: str) -> tuple[np.ndarray, np.ndarray]:
    current = authority.loc[authority.fold.eq(fold)].copy()
    train = current.loc[current.role.eq("train"), "sample_index"].to_numpy(int)
    valid = current.loc[current.role.eq("validation"), "sample_index"].to_numpy(int)
    if not len(train) or not len(valid) or train.max(initial=-1) < 0:
        raise ProtocolFailure(f"EMPTY_FOLD:{fold}")
    return train[np.argsort(train)], valid[np.argsort(valid)]


def metric_record(y: np.ndarray, pred: np.ndarray, vref: float) -> dict[str, float]:
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    if len(y) != len(pred) or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ProtocolFailure("INVALID_METRIC_INPUT")
    err = pred - y; absolute_mv = np.abs(err) * vref * 1000.0
    sse = float(np.sum(err ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    rho = float(pd.Series(y).corr(pd.Series(pred), method="spearman")) if len(y) > 1 else math.nan
    return {"n": int(len(y)), "MAE_R": float(np.mean(np.abs(err))), "RMSE_R": float(np.sqrt(np.mean(err ** 2))),
            "MAE_mV": float(np.mean(absolute_mv)), "RMSE_mV": float(np.sqrt(np.mean((err * vref * 1000.0) ** 2))),
            "R2": float(1.0 - sse / sst) if sst > 0 else math.nan, "Spearman": rho,
            "P95_absolute_error_mV": float(np.quantile(absolute_mv, .95)), "max_absolute_error_mV": float(absolute_mv.max())}


def select_ridge_alpha(features: np.ndarray, target: np.ndarray, train_idx: np.ndarray) -> tuple[float, pd.DataFrame]:
    stopper_n = int(math.ceil(len(train_idx) * .15))
    fit_idx, stopper_idx = train_idx[:-stopper_n], train_idx[-stopper_n:]
    if len(fit_idx) < 2 or not len(stopper_idx) or not np.all(fit_idx < stopper_idx.min()):
        raise ProtocolFailure("RIDGE_CHRONOLOGICAL_INNER_STOPPER_UNAVAILABLE")
    rows = []
    for alpha in ALPHAS:
        scaler = StandardScaler().fit(features[fit_idx])
        model = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(features[fit_idx]), target[fit_idx])
        pred = model.predict(scaler.transform(features[stopper_idx]))
        rows.append({"alpha": alpha, "inner_fit_count": len(fit_idx), "inner_stopper_count": len(stopper_idx), "inner_chronological_MAE_R": float(np.mean(np.abs(pred - target[stopper_idx])))})
    grid = pd.DataFrame(rows).sort_values(["inner_chronological_MAE_R", "alpha"], kind="stable").reset_index(drop=True)
    return float(grid.iloc[0].alpha), grid


def evaluate_baselines(samples: pd.DataFrame, authority: pd.DataFrame, vref: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    selection: list[pd.DataFrame] = []
    state_features = samples[["origin_R_P1S_causal", "origin_slope_24h", "origin_slope_48h", "origin_slope_96h", "lead_to_target_end_h"]].to_numpy(float)
    full_slope = samples[["origin_slope_24h_available", "origin_slope_48h_available", "origin_slope_96h_available"]].astype(bool).all(axis=1).to_numpy()
    for fold, *_ in FOLDS:
        train, valid = fold_positions(authority, fold)
        y = samples["target_R_P1S_causal"].to_numpy(float)
        persist = samples.loc[valid, "origin_R_P1S_causal"].to_numpy(float)
        record = metric_record(y[valid], persist, vref)
        rows.append({"fold": fold, "model": "PERSISTENCE", "scope": "all_valid_H1_samples", "n_train": len(train), "n_validation": len(valid), **record})
        for idx, pred in zip(valid, persist):
            predictions.append(prediction_row(samples.iloc[idx], fold, "PERSISTENCE", y[idx], pred, "all_valid_H1_samples"))
        for horizon in (24, 48, 96):
            available = samples[f"origin_slope_{horizon}h_available"].astype(bool).to_numpy()
            idx = valid[available[valid]]
            if not len(idx):
                continue
            pred = samples.loc[idx, "origin_R_P1S_causal"].to_numpy(float) + samples.loc[idx, f"origin_slope_{horizon}h"].to_numpy(float) * samples.loc[idx, "lead_to_target_end_h"].to_numpy(float)
            label = f"CAUSAL_LINEAR_{horizon}H"
            rows.append({"fold": fold, "model": label, "scope": f"origin_{horizon}h_slope_available", "n_train": int(full_slope[train].sum()), "n_validation": len(idx), **metric_record(y[idx], pred, vref)})
            for item, value in zip(idx, pred):
                predictions.append(prediction_row(samples.iloc[item], fold, label, y[item], value, f"origin_{horizon}h_slope_available"))
        rtrain = train[full_slope[train]]; rvalid = valid[full_slope[valid]]
        if len(rtrain) < 4 or not len(rvalid):
            raise ProtocolFailure(f"RIDGE_CAUSAL_STATE_SCOPE_UNAVAILABLE:{fold}")
        alpha, grid = select_ridge_alpha(state_features, y, rtrain)
        grid.insert(0, "fold", fold); grid.insert(1, "model", "RIDGE_CAUSAL_STATE"); selection.append(grid)
        scaler = StandardScaler().fit(state_features[rtrain])
        ridge = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(state_features[rtrain]), y[rtrain])
        pred = ridge.predict(scaler.transform(state_features[rvalid]))
        rows.append({"fold": fold, "model": "RIDGE_CAUSAL_STATE", "scope": "all_24_48_96h_origin_slopes_available", "n_train": len(rtrain), "n_validation": len(rvalid), "selected_alpha": alpha, **metric_record(y[rvalid], pred, vref)})
        for item, value in zip(rvalid, pred):
            predictions.append(prediction_row(samples.iloc[item], fold, "RIDGE_CAUSAL_STATE", y[item], value, "all_24_48_96h_origin_slopes_available"))
    return pd.DataFrame(rows), pd.DataFrame(predictions), pd.concat(selection, ignore_index=True)


def prediction_row(sample: pd.Series, fold: str, model: str, y: float, pred: float, scope: str, seed: int | None = None) -> dict[str, Any]:
    return {"fold": fold, "model": model, "seed": seed, "scope": scope,
            "origin_cycle_interval_id": sample.origin_cycle_interval_id, "target_cycle_interval_id": sample.target_cycle_interval_id,
            "origin_raw_cycle_id": int(sample.origin_raw_cycle_id), "target_raw_cycle_id": int(sample.target_raw_cycle_id),
            "target_p1s_measurement_end_ophrs2_h": float(sample.target_p1s_measurement_end_ophrs2_h),
            "target_condition_support_status": sample.target_condition_support_status,
            "true_R_P1S_causal": float(y), "predicted_R_P1S_causal": float(pred), "error_R": float(pred - y), "absolute_error_R": float(abs(pred - y)),
            "target_measurement_excluded": True, "future_target_skipped": False}


def import_timesnet_components() -> tuple[Any, Any]:
    sys.path.insert(0, str(BS_VENDOR))
    try:
        module = __import__("models.TimesNet", fromlist=["TimesBlock"])
        embed = __import__("layers.Embed", fromlist=["DataEmbedding"])
        return module.TimesBlock, embed.DataEmbedding
    finally:
        sys.path.remove(str(BS_VENDOR))


class TimesNetRetention(torch.nn.Module):
    """Frozen TimesNet body configuration with only a scalar retention head."""
    def __init__(self, TimesBlock: Any, DataEmbedding: Any) -> None:
        super().__init__()
        cfg = SimpleNamespace(seq_len=48, pred_len=24, top_k=2, d_model=32, d_ff=64, num_kernels=3,
                              e_layers=1, enc_in=14, embed="fixed", freq="h", dropout=.1)
        self.seq_len, self.pred_len = cfg.seq_len, cfg.pred_len
        self.enc_embedding = DataEmbedding(cfg.enc_in, cfg.d_model, cfg.embed, cfg.freq, cfg.dropout)
        self.predict_linear = torch.nn.Linear(cfg.seq_len, cfg.seq_len + cfg.pred_len)
        self.blocks = torch.nn.ModuleList([TimesBlock(cfg) for _ in range(cfg.e_layers)])
        self.layer_norm = torch.nn.LayerNorm(cfg.d_model)
        self.health_head = torch.nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = x.mean(1, keepdim=True).detach()
        normalized = (x - means) / torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        latent = self.enc_embedding(normalized, None)
        latent = self.predict_linear(latent.permute(0, 2, 1)).permute(0, 2, 1)
        for block in self.blocks:
            latent = self.layer_norm(block(latent))
        return self.health_head(latent.mean(dim=1)).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class TrainResult:
    prediction: np.ndarray
    selected_epoch: int
    inner_best_loss: float
    input_scaler_mean: np.ndarray
    target_mean: float
    target_std: float


def scale_windows(train_windows: np.ndarray, eval_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(train_windows.reshape(-1, train_windows.shape[-1]))
    return scaler.transform(train_windows.reshape(-1, train_windows.shape[-1])).reshape(train_windows.shape).astype(np.float32), scaler.transform(eval_windows.reshape(-1, eval_windows.shape[-1])).reshape(eval_windows.shape).astype(np.float32), scaler


def train_epochs(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, seed: int, x_stop: np.ndarray | None = None, y_stop: np.ndarray | None = None, patience: int | None = None) -> tuple[dict[str, Any], int, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict()); best_loss = math.inf; best_epoch = 1; stale = 0
    x_tensor = torch.from_numpy(x); y_tensor = torch.from_numpy(y)
    stopper_x = torch.from_numpy(x_stop).to(device) if x_stop is not None else None
    stopper_y = torch.from_numpy(y_stop).to(device) if y_stop is not None else None
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_tensor))
        for begin in range(0, len(permutation), 64):
            take = permutation[begin:begin + 64]
            xb = x_tensor[take].to(device); yb = y_tensor[take].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if stopper_x is None:
            continue
        model.eval()
        with torch.no_grad():
            current = float(loss_fn(model(stopper_x), stopper_y).item())
        if current < best_loss - 1e-12:
            best_loss, best_epoch, stale, best_state = current, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if patience is not None and stale >= patience:
                break
    if stopper_x is None:
        best_state = copy.deepcopy(model.state_dict()); best_epoch = epochs; best_loss = math.nan
    return best_state, best_epoch, best_loss


def run_timesnet_fold(windows: np.ndarray, target: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray, seed: int, components: tuple[Any, Any]) -> TrainResult:
    stopper_n = int(math.ceil(len(train_idx) * .15))
    fit_idx, stopper_idx = train_idx[:-stopper_n], train_idx[-stopper_n:]
    if len(fit_idx) < 2 or not len(stopper_idx) or not np.all(fit_idx < stopper_idx.min()):
        raise ProtocolFailure("TIMESNET_CHRONOLOGICAL_INNER_STOPPER_UNAVAILABLE")
    target_mean_inner = float(target[fit_idx].mean()); target_std_inner = float(target[fit_idx].std(ddof=0))
    if target_std_inner <= 0:
        raise ProtocolFailure("TIMESNET_INNER_TARGET_ZERO_VARIANCE")
    x_fit, x_stop, _ = scale_windows(windows[fit_idx], windows[stopper_idx])
    y_fit = ((target[fit_idx] - target_mean_inner) / target_std_inner).astype(np.float32)
    y_stop = ((target[stopper_idx] - target_mean_inner) / target_std_inner).astype(np.float32)
    TimesBlock, DataEmbedding = components
    set_seed(seed)
    selection_model = TimesNetRetention(TimesBlock, DataEmbedding)
    state, best_epoch, best_loss = train_epochs(selection_model, x_fit, y_fit, 300, seed, x_stop, y_stop, patience=30)
    del selection_model
    target_mean = float(target[train_idx].mean()); target_std = float(target[train_idx].std(ddof=0))
    if target_std <= 0:
        raise ProtocolFailure("TIMESNET_OUTER_TARGET_ZERO_VARIANCE")
    x_train, x_valid, input_scaler = scale_windows(windows[train_idx], windows[valid_idx])
    y_train = ((target[train_idx] - target_mean) / target_std).astype(np.float32)
    set_seed(seed)
    final_model = TimesNetRetention(TimesBlock, DataEmbedding)
    final_state, _, _ = train_epochs(final_model, x_train, y_train, max(1, best_epoch), seed, None, None, None)
    final_model.load_state_dict(final_state); final_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.to(device)
    with torch.no_grad():
        standardized = final_model(torch.from_numpy(x_valid).to(device)).cpu().numpy()
    pred = standardized * target_std + target_mean
    return TrainResult(prediction=pred.astype(float), selected_epoch=int(best_epoch), inner_best_loss=float(best_loss), input_scaler_mean=input_scaler.mean_.copy(), target_mean=target_mean, target_std=target_std)


def evaluate_timesnet(samples: pd.DataFrame, authority: pd.DataFrame, windows: np.ndarray, vref: float, ridge_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = import_timesnet_components()
    target = samples["target_R_P1S_causal"].to_numpy(float)
    rows: list[dict[str, Any]] = []; predictions: list[dict[str, Any]] = []; training: list[dict[str, Any]] = []
    ridge_keys = set(zip(ridge_predictions.fold, ridge_predictions.target_cycle_interval_id))
    for fold, *_ in FOLDS:
        train, valid = fold_positions(authority, fold)
        matched = np.asarray([idx for idx in valid if (fold, samples.iloc[int(idx)].target_cycle_interval_id) in ridge_keys], dtype=int)
        if not len(matched):
            raise ProtocolFailure(f"TIMESNET_RIDGE_MATCHED_SCOPE_EMPTY:{fold}")
        for seed in SEEDS:
            result = run_timesnet_fold(windows, target, train, valid, seed, components)
            metrics = metric_record(target[valid], result.prediction, vref)
            mapping = {int(idx): float(pred) for idx, pred in zip(valid, result.prediction)}
            matched_pred = np.asarray([mapping[int(idx)] for idx in matched])
            matched_metrics = metric_record(target[matched], matched_pred, vref)
            rows.append({"fold": fold, "seed": seed, "n_train": len(train), "n_validation": len(valid), "n_ridge_matched_validation": len(matched),
                         "selected_epoch": result.selected_epoch, "inner_best_standardized_MSE": result.inner_best_loss,
                         "input_scaler_fit_population": "full_outer_train_windows_only", "target_scaler_fit_population": "full_outer_train_targets_only",
                         "TimesNet_parameters_updated": int(sum(p.numel() for p in TimesNetRetention(*components).parameters())),
                         **metrics, **{f"ridge_matched_{key}": value for key, value in matched_metrics.items()}})
            training.append({"fold": fold, "seed": seed, "selected_epoch": result.selected_epoch, "inner_best_standardized_MSE": result.inner_best_loss,
                             "outer_train_target_mean_R": result.target_mean, "outer_train_target_std_R": result.target_std,
                             "input_scaler_mean_l2": float(np.linalg.norm(result.input_scaler_mean)), "refit_reinitialized_same_seed": True})
            for idx, pred in mapping.items():
                predictions.append(prediction_row(samples.iloc[idx], fold, "TIMESNET", target[idx], pred, "all_valid_H1_samples", seed=seed))
    return pd.DataFrame(rows), pd.DataFrame(predictions), pd.DataFrame(training)


def pooled_metric(frame: pd.DataFrame, vref: float) -> dict[str, float]:
    return metric_record(frame.true_R_P1S_causal.to_numpy(float), frame.predicted_R_P1S_causal.to_numpy(float), vref)


def make_counts(samples: pd.DataFrame, exclusions: pd.DataFrame, authority: pd.DataFrame) -> pd.DataFrame:
    rows = [{"scope": "unique_H1_candidates", "n": len(samples), "detail": "contiguous causal P1S origin-target pairs with legal Raw F2 history"},
            {"scope": "excluded_candidate_origins", "n": len(exclusions), "detail": "no next causal target or no legal 48-bin causal history"},
            {"scope": "all_fold_expanded_authority_rows", "n": len(authority), "detail": "frozen fold/role sample rows"}]
    for fold, *_ in FOLDS:
        unit = authority.loc[authority.fold.eq(fold)]
        for role in ("train", "validation"):
            rows.append({"scope": f"{fold}_{role}", "n": int(unit.role.eq(role).sum()), "detail": "target interval entirely within frozen role bounds"})
    return pd.DataFrame(rows)


def make_stratified(times_pred: pd.DataFrame, baseline_pred: pd.DataFrame, vref: float) -> pd.DataFrame:
    average = times_pred.groupby(["fold", "target_cycle_interval_id"], as_index=False).agg(
        true_R_P1S_causal=("true_R_P1S_causal", "first"), predicted_R_P1S_causal=("predicted_R_P1S_causal", "mean"),
        target_condition_support_status=("target_condition_support_status", "first"))
    average["model"] = "TIMESNET_seed_mean"
    use = pd.concat([average, baseline_pred], ignore_index=True, sort=False)
    rows = []
    for (model, support), group in use.groupby(["model", "target_condition_support_status"], dropna=False):
        metrics = metric_record(group.true_R_P1S_causal.to_numpy(float), group.predicted_R_P1S_causal.to_numpy(float), vref)
        rows.append({"model": model, "target_condition_support_status": support, **metrics, "posthoc_only_not_used_for_selection": True})
    return pd.DataFrame(rows).sort_values(["model", "target_condition_support_status"]).reset_index(drop=True)


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=.6, alpha=.7)
    ax.tick_params(labelsize=8)


def figures(samples: pd.DataFrame, baseline_pred: pd.DataFrame, times_pred: pd.DataFrame, baseline_metrics: pd.DataFrame, time_metrics: pd.DataFrame, stratified: pd.DataFrame, vref: float) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150, "savefig.dpi": 300})
    mean_time = times_pred.groupby(["fold", "target_cycle_interval_id", "target_p1s_measurement_end_ophrs2_h"], as_index=False).agg(true_R=("true_R_P1S_causal", "first"), pred_R=("predicted_R_P1S_causal", "mean"))
    persistence = baseline_pred.loc[baseline_pred.model.eq("PERSISTENCE"), ["fold", "target_cycle_interval_id", "predicted_R_P1S_causal"]].rename(columns={"predicted_R_P1S_causal": "persistence_R"})
    ridge = baseline_pred.loc[baseline_pred.model.eq("RIDGE_CAUSAL_STATE"), ["fold", "target_cycle_interval_id", "predicted_R_P1S_causal"]].rename(columns={"predicted_R_P1S_causal": "ridge_R"})
    plot = mean_time.merge(persistence, on=["fold", "target_cycle_interval_id"], how="left").merge(ridge, on=["fold", "target_cycle_interval_id"], how="left").sort_values("target_p1s_measurement_end_ophrs2_h")
    fig, ax = plt.subplots(figsize=(8.2, 3.4)); x = plot.target_p1s_measurement_end_ophrs2_h
    ax.plot(x, plot.true_R, color=PALETTE["navy"], marker="o", ms=3, lw=1.5, label="Causal P1S retention target")
    ax.plot(x, plot.persistence_R, color=PALETTE["gray"], marker=".", ms=4, lw=1, label="Persistence")
    ax.plot(x, plot.ridge_R, color=PALETTE["orange"], marker=".", ms=4, lw=1, label="Ridge causal state")
    ax.plot(x, plot.pred_R, color=PALETTE["blue"], marker=".", ms=4, lw=1, label="TimesNet (three-seed mean)")
    for _, _, _, boundary, _ in FOLDS[:-1]: ax.axvline(boundary, color="#999999", ls="--", lw=.8)
    ax.set(xlabel="Target P1S measurement end, operating hours", ylabel="Causal P1S retention R", title="H1 contiguous-cycle Development predictions")
    style_axes(ax); ax.legend(frameon=False, ncol=2, fontsize=7); fig.tight_layout(); fig.savefig(FIG / "Fig1_H1_Chronological_Target_and_Predictions.png"); plt.close(fig)
    fold_labels = [item[0] for item in FOLDS]
    models = [("PERSISTENCE", "Persistence", PALETTE["gray"]), ("RIDGE_CAUSAL_STATE", "Ridge causal state", PALETTE["orange"]), ("TIMESNET", "TimesNet seed mean", PALETTE["blue"])]
    fig, ax = plt.subplots(figsize=(7.6, 3.5)); xpos = np.arange(3); width = .23
    for k, (key, label, color) in enumerate(models):
        if key == "TIMESNET": values = [float(time_metrics.loc[time_metrics.fold.eq(fold), "MAE_mV"].mean()) for fold in fold_labels]
        else: values = [float(baseline_metrics.loc[(baseline_metrics.fold.eq(fold)) & (baseline_metrics.model.eq(key)), "MAE_mV"].iloc[0]) for fold in fold_labels]
        ax.bar(xpos + (k - 1) * width, values, width, label=label, color=color)
    ax.set(xticks=xpos, xticklabels=fold_labels, ylabel="MAE (mV equivalent)", title="H1 error by frozen Development fold")
    style_axes(ax); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig2_H1_Error_By_Fold.png"); plt.close(fig)
    relevant = stratified.loc[stratified.model.isin(["TIMESNET_seed_mean", "PERSISTENCE", "RIDGE_CAUSAL_STATE"])].copy()
    supports = sorted(relevant.target_condition_support_status.dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(7.6, 3.5)); xpos = np.arange(len(supports)); width = .23
    labels = [("PERSISTENCE", "Persistence", PALETTE["gray"]), ("RIDGE_CAUSAL_STATE", "Ridge causal state", PALETTE["orange"]), ("TIMESNET_seed_mean", "TimesNet seed mean", PALETTE["blue"])]
    for k, (key, label, color) in enumerate(labels):
        values = []
        for support in supports:
            match = relevant.loc[(relevant.model.eq(key)) & (relevant.target_condition_support_status.eq(support)), "MAE_mV"]
            values.append(float(match.iloc[0]) if len(match) else math.nan)
        ax.bar(xpos + (k - 1) * width, values, width, label=label, color=color)
    ax.set(xticks=xpos, xticklabels=[item.replace("_", " ") for item in supports], ylabel="MAE (mV equivalent)", title="Post-hoc error by target condition-support stratum")
    style_axes(ax); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig3_H1_Error_By_Target_Support.png"); plt.close(fig)


def md_table(frame: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    view = frame.loc[:, [c for c in columns if c in frame.columns]].head(n).copy()
    for col in view.select_dtypes(include=[np.number]).columns: view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    headers = [str(column).replace("|", "\\|") for column in view.columns]
    records = []
    for values in view.itertuples(index=False, name=None):
        records.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join(records)


def build_report(protocol_sha: str, stage_a: dict[str, Any], counts: pd.DataFrame, baseline_metrics: pd.DataFrame, time_metrics: pd.DataFrame, summary: dict[str, Any], vref: float) -> None:
    seed_summary = time_metrics.groupby("seed", as_index=False).agg(equal_fold_mean_MAE_mV=("MAE_mV", "mean"), equal_fold_mean_RMSE_mV=("RMSE_mV", "mean"))
    report = f"""# Phase CPRF — H1 causal P1S-retention forecasting

## Decision

**{summary['final_decision']}**

This formal Development-only H1 task forecasts the next *contiguous* raw complete-cycle target. Every authority row satisfies `target_raw_cycle_id = origin_raw_cycle_id + 1`; a future legal P1S cycle was never skipped to replace a missing adjacent target. The target is raw `R_P1S_causal = V_P1S / V_ref`, with `V_ref = {vref:.12g} V`; global B2 and all smoothed/trend health labels were excluded from fitting and model selection.

## Frozen sample authority and leakage controls

Protocol SHA-256: `{protocol_sha}`.  The causal feature source was read only with `pd.read_csv(..., nrows=3007)`, then filtered to the 3005-row Development prefix. Its legal-object fingerprint is `{stage_a['development_prefix_fingerprint_sha256']}`; it is explicitly **not** a hash of the mixed physical source file.

PAA middle-50 membership was replayed for all 131 causal-target-eligible complete cycles through the pre-1719.31 h raw source universe. Each reconstructed voltage and dwell count matched the frozen PAA authority. The origin input is the latest legal 48 x 14 Raw F2 window ending at or before the origin's actual P1S measurement end. Every bin ends strictly before the target P1S measurement begins. No target P1S measurement, future covariate, operating age, B2 value, or target support status entered any model input.

{md_table(counts, ['scope', 'n', 'detail'])}

## Baselines

`PERSISTENCE` is available on every frozen H1 sample. `CAUSAL_LINEAR_96H` is the predeclared physical slope baseline; 24 h and 48 h slope variants are sensitivity rows only. `RIDGE_CAUSAL_STATE` uses exactly `[R_origin, slope_24h, slope_48h, slope_96h, lead_to_target_end_h]`, with a train-only embedding scaler and chronological inner alpha choice. Any Ridge scope difference is preserved rather than hidden.

{md_table(baseline_metrics, ['fold', 'model', 'scope', 'n_validation', 'MAE_mV', 'RMSE_mV', 'R2', 'Spearman'])}

## TimesNet H1 model

The model retains the official TimesNet FFT period discovery, TimesBlock, 48-to-72 temporal projection, and frozen backbone-screening hyperparameters. Only the forecasting output contract was replaced by temporal mean pooling followed by a single `Linear(32,1)` scalar retention head. Each fold and seed uses chronological 85/15 inner stopping, then same-seed reinitialisation and full-outer-train refitting for the selected epoch. The network was trained normally; no checkpoint or encoder was treated as a frozen pretrained model.

{md_table(time_metrics, ['fold', 'seed', 'n_train', 'n_validation', 'selected_epoch', 'MAE_mV', 'RMSE_mV', 'R2', 'Spearman', 'ridge_matched_MAE_mV'])}

Per-seed three-fold equal summaries:

{md_table(seed_summary, list(seed_summary.columns))}

## Primary score and gates

The formal score is the arithmetic mean of fold MAE values in mV-equivalent retention error. Pooled results are descriptive only because folds are intentionally equally weighted. Gate B compares TimesNet and Ridge only on the explicit all-slope-available Ridge scope. Condition-support strata are post-hoc reporting groups, not sampling or selection criteria.

* Gate A (≥5% below Persistence): **{summary['gate_A']}**
* Gate B (≥3% below matched Ridge): **{summary['gate_B']}**
* Gate C (Fold 3 ≤105% Persistence): **{summary['gate_C']}**
* Gate D (at least two seeds strictly beat Persistence): **{summary['gate_D']}**

## Interpretation and stop

{summary['interpretation']}

This result concerns a causal proxy target and next-contiguous-cycle H1 prediction, not independently validated SOH or RUL. No Test target, Test raw region, 2000/2300 polarization target, future health-trajectory model, H3/H6 model, calibration model, or post-hoc architecture change was executed. The phase stops here.
"""
    (REPORT / "CPRF_H1_CAUSAL_P1S_RETENTION_FORECASTING_REPORT.md").write_text(report, encoding="utf-8")


def run() -> dict[str, Any]:
    ensure_dirs()
    protocol, protocol_sha = load_protocol()
    causal, fingerprint, fp_manifest = read_causal_development_prefix()
    ctae, _, vref, sources = load_sources()
    replay = reconstruct_all_causal_p1s_intervals(ctae)
    samples, exclusions, windows = construct_h1_samples(ctae, replay, causal)
    authority = fold_expand(samples)
    stage_a = save_stage_a(samples, exclusions, authority, replay, fingerprint, fp_manifest, sources, protocol_sha)
    write_csv(make_counts(samples, exclusions, authority), OUT / "H1_SAMPLE_COUNTS.csv")
    baseline_metrics, baseline_predictions, ridge_selection = evaluate_baselines(samples, authority, vref)
    write_csv(baseline_metrics, OUT / "H1_BASELINE_METRICS.csv")
    write_csv(baseline_predictions, OUT / "H1_BASELINE_PREDICTIONS.csv")
    write_csv(ridge_selection, OUT / "H1_RIDGE_CAUSAL_STATE_SELECTION.csv")
    time_metrics, time_predictions, training = evaluate_timesnet(samples, authority, windows, vref, baseline_predictions.loc[baseline_predictions.model.eq("RIDGE_CAUSAL_STATE")])
    write_csv(time_metrics, OUT / "H1_TIMESNET_SEED_RESULTS.csv")
    write_csv(time_predictions, OUT / "H1_TIMESNET_PREDICTIONS.csv")
    write_csv(training, OUT / "H1_TIMESNET_TRAINING_SELECTION.csv")
    stratified = make_stratified(time_predictions, baseline_predictions, vref)
    write_csv(stratified, OUT / "H1_STRATIFIED_RESULTS.csv")
    counts = pd.read_csv(OUT / "H1_SAMPLE_COUNTS.csv")
    persistence = baseline_metrics.loc[baseline_metrics.model.eq("PERSISTENCE")].sort_values("fold")
    ridge = baseline_metrics.loc[baseline_metrics.model.eq("RIDGE_CAUSAL_STATE")].sort_values("fold")
    equal_time_by_seed = time_metrics.groupby("seed")["MAE_mV"].mean()
    time_equal = float(equal_time_by_seed.mean())
    persistence_equal = float(persistence.MAE_mV.mean())
    ridge_equal = float(ridge.MAE_mV.mean())
    matched_time_equal = float(time_metrics.groupby("seed")["ridge_matched_MAE_mV"].mean().mean())
    fold3_time = float(time_metrics.loc[time_metrics.fold.eq("fold_3"), "MAE_mV"].mean())
    fold3_persistence = float(persistence.loc[persistence.fold.eq("fold_3"), "MAE_mV"].iloc[0])
    gate_a = time_equal <= .95 * persistence_equal
    gate_b = matched_time_equal <= .97 * ridge_equal
    gate_c = fold3_time <= 1.05 * fold3_persistence
    gate_d = int((equal_time_by_seed < persistence_equal).sum()) >= 2
    status = "CAUSAL_P1S_H1_FORECASTING_SUPPORTED" if all((gate_a, gate_b, gate_c, gate_d)) else "COMPLEX_H1_MODEL_NOT_JUSTIFIED_OVER_SIMPLE_CAUSAL_BASELINES"
    interpretation = ("The fixed TimesNet configuration met every preregistered H1 gate against the causal baselines." if status.endswith("SUPPORTED") else
                      "The fixed TimesNet configuration did not meet all preregistered H1 gates. Under the frozen protocol, this does not justify a more complex H1 model; the next methodological decision must precede any H3/H6 work.")
    pooled_time = pooled_metric(time_predictions, vref)
    summary: dict[str, Any] = {
        "protocol_version": "H1", "base_commit": git_head(), "protocol_sha256": protocol_sha,
        "target_definition": "R_P1S_causal_target = V_P1S(target) / causal V_ref", "reference_voltage_V": vref,
        "horizon": "H1_next_contiguous_raw_complete_cycle", "lead_time_definition": "target_P1S_measurement_end - origin_P1S_measurement_end",
        "TimesNet_configuration_source": str(BS_PROTOCOL.relative_to(REPO)).replace("\\", "/"),
        "TimesNet_official_body": True, "TimesNet_normal_training_performed": True,
        "development_prefix_rows": DEVELOPMENT_ROWS, "development_prefix_fingerprint_sha256": fingerprint,
        "raw_P1S_replay_max_h": float(replay.p1s_measurement_end_ophrs2_h.max()),
        "legal_unique_H1_samples": int(len(samples)), "frozen_fold_authority_rows": int(len(authority)),
        "persistence_equal_fold_mean_MAE_mV": persistence_equal, "ridge_equal_fold_mean_MAE_mV": ridge_equal,
        "timesnet_three_seed_equal_fold_mean_MAE_mV": time_equal, "timesnet_matched_ridge_scope_equal_fold_mean_MAE_mV": matched_time_equal,
        "timesnet_pooled_descriptive": pooled_time,
        "gate_A": "PASS" if gate_a else "FAIL", "gate_B": "PASS" if gate_b else "FAIL", "gate_C": "PASS" if gate_c else "FAIL", "gate_D": "PASS" if gate_d else "FAIL",
        "final_decision": status, "interpretation": interpretation,
        "global_B2_used_for_training_or_selection": False, "smoothed_targets_used": False, "future_target_skipped": False,
        "test_accessed": False, "test_raw_region_accessed": False, "test_labels_accessed": False,
        "future_health_forecasting_trained": False, "RUL_trained": False, "H3_H6_trained": False,
        "condition_support_used_only_posthoc": True,
    }
    write_json(summary, OUT / "CPRF_H1_SUMMARY.json")
    figures(samples, baseline_predictions, time_predictions, baseline_metrics, time_metrics, stratified, vref)
    build_report(protocol_sha, stage_a, counts, baseline_metrics, time_metrics, summary, vref)
    return summary


if __name__ == "__main__":
    try:
        final = run()
        print(json.dumps({"status": final["final_decision"], "timesnet_MAE_mV": final["timesnet_three_seed_equal_fold_mean_MAE_mV"], "test_accessed": False}, ensure_ascii=False))
    except ProtocolFailure as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "reason": str(exc), "test_accessed": False}, ensure_ascii=False))
        raise

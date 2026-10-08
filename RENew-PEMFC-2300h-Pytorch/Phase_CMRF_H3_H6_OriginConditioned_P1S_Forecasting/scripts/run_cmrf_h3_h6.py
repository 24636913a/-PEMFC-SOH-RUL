"""Development-only H3/H6 origin-conditioned raw P1S-retention forecasting.

The mixed N1 causal table is accessed only through its frozen 3007-row
prefix.  All horizon membership is physical (`target_raw_cycle_id=origin+H`)
and all target P1S values come from pre-existing, causal, Development-only
authorities.  The target measurement, timing and conditions never enter model
inputs.  H6 is the preregistered primary horizon; H3 is secondary.
"""
from __future__ import annotations

import copy
import hashlib
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
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True

PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
CONFIG = PHASE / "configs" / "CMRF_H3_H6_FROZEN_PROTOCOL.yaml"
CONFIG_SHA = PHASE / "configs" / "CMRF_H3_H6_FROZEN_PROTOCOL.sha256"
INPUT = PHASE / "inputs"
OUT = PHASE / "outputs"
REPORT = PHASE / "reports"
FIG = PHASE / "figures"

CTAE = PROJECT / "Phase_CTAE_P1S_Reference_Availability_Erratum"
CTAE_AUDIT = CTAE / "outputs" / "P1S_REFERENCE_AVAILABILITY_AUDIT.csv"
PHI = PROJECT / "Phase_PHI_P1S_B2_Health_Index_Consolidation"
PHI_IDENTITY = PHI / "outputs" / "P1S_B2_FULL_HEALTH_TRAJECTORY.csv"
RCR = PROJECT / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit"
RAW_CATALOG = RCR / "outputs" / "RAW_CYCLE_CATALOG.csv"
H1 = PROJECT / "Phase_CPRF_H1_Causal_P1S_Retention_Forecasting"
H1_RECON = H1 / "outputs" / "P1S_RAW_MEASUREMENT_INTERVAL_RECONSTRUCTION.csv"
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
BS_PROTOCOL = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
BS_VENDOR = BS / "vendor" / "TimesNet_official"
CAUSAL = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"

BASE_COMMIT = "aeb755136aa895d0e1dcba41538ce13e38a881f5"
DEVELOPMENT_BOUNDARY = 1819.997738
PREFIX_NROWS = 3007
DEVELOPMENT_ROWS = 3005
VREF = 0.71383023635
HORIZONS = (3, 6)
PRIMARY_HORIZON = 6
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
    """Raised whenever a frozen data, chronology, or leakage contract fails."""


@dataclass
class TrainResult:
    predicted_delta: np.ndarray
    selected_epoch: int
    inner_best_loss: float
    parameter_count: int


def ensure_dirs() -> None:
    for path in (INPUT, OUT, REPORT, FIG):
        path.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Hash normal frozen artifacts; whole-file hashing of mixed Causal CSV is forbidden."""
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
        raise ProtocolFailure("CMRF_PROTOCOL_SHA_MISMATCH")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")), observed


def development_prefix_fingerprint(frame: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    ordered = frame.loc[:, CAUSAL_COLUMNS].copy()
    payload = ordered.to_csv(index=False, header=True, lineterminator="\n", float_format="%.17g", na_rep="<NA>").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return digest, {
        "name": "DEVELOPMENT_PREFIX_FINGERPRINT",
        "development_prefix_fingerprint_sha256": digest,
        "row_count": int(len(ordered)), "column_count": int(len(ordered.columns)), "column_order": ordered.columns.tolist(),
        "serialization": {"encoding": "utf-8", "index": False, "header": True, "line_terminator": "LF", "float_format": "%.17g", "na_rep": "<NA>", "row_order": "original_filtered_prefix_order"},
        "legal_read_contract": "pd.read_csv(usecols=CAUSAL_COLUMNS, nrows=3007), then bin_end_ophrs2_h < 1819.997738",
        "not_a_full_source_file_hash": True,
    }


def read_causal_development_prefix() -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """The only permitted reader for the mixed Development/Test causal feature file."""
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


def source_manifest(fingerprint: str) -> list[dict[str, Any]]:
    entries = [
        (CTAE_AUDIT, "causal_raw_retention_target_and_support"),
        (PHI_IDENTITY, "Development_P1S_to_raw_cycle_identity_only"),
        (RAW_CATALOG, "Development_only_raw_cycle_completeness_catalog"),
        (H1_RECON, "frozen_matched_PAA_P1S_measurement_envelopes"),
        (BS_PROTOCOL, "frozen_TimesNet_architecture_and_training_authority"),
        (CONFIG, "CMRF_frozen_protocol"),
    ]
    rows = [{"path": str(path.relative_to(REPO)).replace("\\", "/"), "role": role, "sha256": sha256_file(path)} for path, role in entries]
    rows.append({"path": str(CAUSAL.relative_to(REPO)).replace("\\", "/"), "role": "mixed_causal_feature_source_prefix_only", "development_prefix_fingerprint_sha256": fingerprint, "full_file_hash_computed": False})
    return rows


def load_development_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only frozen Development artifacts and prove their endpoint is pre-Test."""
    ctae = pd.read_csv(CTAE_AUDIT)
    identity = pd.read_csv(PHI_IDENTITY, usecols=["cycle_interval_id", "raw_cycle_id", "cycle_start_ophrs2_h", "cycle_end_ophrs2_h"])
    replay = pd.read_csv(H1_RECON)
    catalog = pd.read_csv(RAW_CATALOG)
    needed = {"cycle_interval_id", "P1S_voltage_V", "V_ref_V", "causal_target_eligible", "R_P1S_causal", "condition_support_status"}
    if needed - set(ctae):
        raise ProtocolFailure("CTAE_SCHEMA_MISMATCH")
    if {"raw_cycle_id", "complete_cycle", "start_ophrs2_h", "end_ophrs2_h"} - set(catalog):
        raise ProtocolFailure("RAW_CATALOG_SCHEMA_MISMATCH")
    if float(catalog["end_ophrs2_h"].max()) >= DEVELOPMENT_BOUNDARY:
        raise ProtocolFailure("RAW_CATALOG_IS_NOT_DEVELOPMENT_ONLY")
    if int(catalog["raw_cycle_id"].max()) != 156 or catalog["raw_cycle_id"].duplicated().any():
        raise ProtocolFailure("RAW_CATALOG_IDENTITY_MISMATCH")
    replay_needed = {"cycle_interval_id", "p1s_measurement_start_ophrs2_h", "p1s_measurement_end_ophrs2_h", "P1S_voltage_reconstructed_V", "voltage_match_pass", "dwell_count_pass", "raw_test_accessed"}
    if replay_needed - set(replay):
        raise ProtocolFailure("H1_P1S_REPLAY_SCHEMA_MISMATCH")
    if not replay["voltage_match_pass"].astype(bool).all() or not replay["dwell_count_pass"].astype(bool).all() or replay["raw_test_accessed"].astype(bool).any():
        raise ProtocolFailure("H1_P1S_REPLAY_INTEGRITY_FAILURE")
    causal = ctae.loc[ctae["causal_target_eligible"].astype(bool)].copy()
    if len(causal) != 131 or not causal["V_ref_available_at_event"].astype(bool).all():
        raise ProtocolFailure("CAUSAL_P1S_TARGET_AUTHORITY_MISMATCH")
    if not np.allclose(causal["V_ref_V"].to_numpy(float), VREF, atol=1e-12, rtol=0.0):
        raise ProtocolFailure("REFERENCE_VOLTAGE_MISMATCH")
    causal = causal.merge(identity, on="cycle_interval_id", how="left", validate="one_to_one")
    causal = causal.merge(replay[list(replay_needed)], on="cycle_interval_id", how="left", validate="one_to_one")
    if causal["raw_cycle_id"].isna().any() or causal["p1s_measurement_end_ophrs2_h"].isna().any():
        raise ProtocolFailure("CAUSAL_P1S_IDENTITY_OR_ENVELOPE_MISSING")
    causal["raw_cycle_id"] = causal["raw_cycle_id"].astype(int)
    if causal["raw_cycle_id"].duplicated().any() or causal["p1s_measurement_end_ophrs2_h"].to_numpy(float).max() >= DEVELOPMENT_BOUNDARY:
        raise ProtocolFailure("CAUSAL_P1S_DEVELOPMENT_BOUNDARY_FAILURE")
    if not np.allclose(causal["P1S_voltage_V"].to_numpy(float), causal["P1S_voltage_reconstructed_V"].to_numpy(float), atol=1e-10, rtol=0.0):
        raise ProtocolFailure("FROZEN_AND_RECONSTRUCTED_P1S_MISMATCH")
    return causal.sort_values("raw_cycle_id").reset_index(drop=True), catalog.sort_values("raw_cycle_id").reset_index(drop=True), replay


def choose_origin_window(causal: pd.DataFrame, origin_measurement_end_h: float, target_measurement_start_h: float) -> tuple[pd.DataFrame | None, str]:
    candidates = np.flatnonzero(causal["bin_end_ophrs2_h"].to_numpy(float) <= origin_measurement_end_h + 1e-9)
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
        input_end = float(window["bin_end_ophrs2_h"].iloc[-1])
        if input_end > origin_measurement_end_h + 1e-9:
            raise ProtocolFailure("ORIGIN_INPUT_CUTOFF_VIOLATION")
        if input_end >= target_measurement_start_h - 1e-9:
            raise ProtocolFailure("TARGET_P1S_MEASUREMENT_LEAKAGE")
        return window.copy(), "LEGAL"
    return None, "NO_LEGAL_48_BIN_ORIGIN_HISTORY"


def physical_continuity(catalog_by_raw: dict[int, Any], origin_raw: int, horizon: int) -> tuple[bool, str]:
    for raw_id in range(origin_raw, origin_raw + horizon + 1):
        row = catalog_by_raw.get(raw_id)
        if row is None:
            return False, f"PHYSICAL_HORIZON_CONTINUITY_FAIL_MISSING_RAW_{raw_id}"
        if not bool(row.complete_cycle):
            return False, f"PHYSICAL_HORIZON_CONTINUITY_FAIL_INCOMPLETE_RAW_{raw_id}"
    return True, "PHYSICAL_HORIZON_CONTINUITY_PASS"


def construct_horizon_samples(horizon: int, p1: pd.DataFrame, catalog: pd.DataFrame, causal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    by_raw = {int(row.raw_cycle_id): row for row in p1.itertuples(index=False)}
    catalog_by_raw = {int(row.raw_cycle_id): row for row in catalog.itertuples(index=False)}
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    tensors: list[np.ndarray] = []
    for origin_raw, origin in by_raw.items():
        target_raw = origin_raw + horizon
        base = {"horizon": f"H{horizon}", "origin_cycle_interval_id": str(origin.cycle_interval_id), "origin_raw_cycle_id": origin_raw, "required_target_raw_cycle_id": target_raw}
        continuous, continuity_reason = physical_continuity(catalog_by_raw, origin_raw, horizon)
        if not continuous:
            excluded.append({**base, "exclusion_reason": continuity_reason})
            continue
        target = by_raw.get(target_raw)
        if target is None:
            excluded.append({**base, "exclusion_reason": "TARGET_RAW_CYCLE_NOT_CAUSAL_P1S_ELIGIBLE"})
            continue
        window, status = choose_origin_window(causal, float(origin.p1s_measurement_end_ophrs2_h), float(target.p1s_measurement_start_ophrs2_h))
        if window is None:
            excluded.append({**base, "target_cycle_interval_id": str(target.cycle_interval_id), "exclusion_reason": status})
            continue
        input_start = float(window["bin_start_ophrs2_h"].iloc[0]); input_end = float(window["bin_end_ophrs2_h"].iloc[-1])
        origin_r, target_r = float(origin.R_P1S_causal), float(target.R_P1S_causal)
        if not (input_end <= float(origin.p1s_measurement_end_ophrs2_h) + 1e-9 and input_end < float(target.p1s_measurement_start_ophrs2_h) - 1e-9):
            raise ProtocolFailure("ORIGIN_OR_TARGET_ISOLATION_FAILURE")
        records.append({
            **base, "physical_continuity_status": continuity_reason,
            "target_cycle_interval_id": str(target.cycle_interval_id), "target_raw_cycle_id": target_raw,
            "origin_p1s_measurement_start_ophrs2_h": float(origin.p1s_measurement_start_ophrs2_h),
            "origin_p1s_measurement_end_ophrs2_h": float(origin.p1s_measurement_end_ophrs2_h),
            "target_p1s_measurement_start_ophrs2_h": float(target.p1s_measurement_start_ophrs2_h),
            "target_p1s_measurement_end_ophrs2_h": float(target.p1s_measurement_end_ophrs2_h),
            "origin_R_P1S_causal": origin_r, "target_R_P1S_causal": target_r, "target_DeltaR": target_r - origin_r,
            "input_start_ophrs2_h": input_start, "input_end_ophrs2_h": input_end,
            "input_start_bin_index": int(window.index[0]), "input_end_bin_index": int(window.index[-1]),
            "temporal_segment_id": int(window["temporal_segment_id"].iloc[0]),
            "target_condition_support_status": str(target.condition_support_status),
            "target_measurement_excluded": True, "future_target_skipped": False,
        })
        tensors.append(window[FEATURES].to_numpy(np.float32))
    samples = pd.DataFrame(records).sort_values("target_p1s_measurement_end_ophrs2_h").reset_index(drop=True)
    exclusions = pd.DataFrame(excluded).sort_values(["origin_raw_cycle_id", "required_target_raw_cycle_id"]).reset_index(drop=True)
    if samples.empty:
        raise ProtocolFailure(f"NO_LEGAL_H{horizon}_SAMPLES")
    if not (samples["target_raw_cycle_id"].to_numpy(int) == samples["origin_raw_cycle_id"].to_numpy(int) + horizon).all():
        raise ProtocolFailure("RAW_HORIZON_IDENTITY_VIOLATION")
    if not samples["physical_continuity_status"].eq("PHYSICAL_HORIZON_CONTINUITY_PASS").all():
        raise ProtocolFailure("PHYSICAL_CONTINUITY_RECORD_FAILURE")
    if samples["target_p1s_measurement_end_ophrs2_h"].to_numpy(float).max() >= DEVELOPMENT_BOUNDARY:
        raise ProtocolFailure("TARGET_CROSSED_DEVELOPMENT_BOUNDARY")
    tensor = np.stack(tensors).astype(np.float32)
    if tensor.shape != (len(samples), 48, len(FEATURES)) or not np.isfinite(tensor).all():
        raise ProtocolFailure("INPUT_TENSOR_CONTRACT_FAILURE")
    return samples, exclusions, tensor


def fold_expand(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, tr_lo, tr_hi, va_lo, va_hi in FOLDS:
        for role, lo, hi in (("train", tr_lo, tr_hi), ("validation", va_lo, va_hi)):
            use = samples.loc[(samples["target_p1s_measurement_start_ophrs2_h"].to_numpy(float) >= lo) & (samples["target_p1s_measurement_end_ophrs2_h"].to_numpy(float) < hi)]
            for sample_index, row in use.iterrows():
                rows.append({"fold": fold, "role": role, "sample_index": int(sample_index), **row.to_dict()})
    authority = pd.DataFrame(rows).sort_values(["fold", "role", "target_p1s_measurement_end_ophrs2_h"]).reset_index(drop=True)
    if authority.empty or authority.duplicated(["fold", "role", "origin_raw_cycle_id", "target_raw_cycle_id"]).any():
        raise ProtocolFailure("FOLD_AUTHORITY_EMPTY_OR_DUPLICATE")
    for fold, tr_lo, tr_hi, va_lo, va_hi in FOLDS:
        unit = authority.loc[authority.fold.eq(fold)]
        if not unit.role.eq("train").any() or not unit.role.eq("validation").any():
            raise ProtocolFailure(f"EMPTY_FOLD_{fold}")
        for row in unit.itertuples(index=False):
            lo, hi = (tr_lo, tr_hi) if row.role == "train" else (va_lo, va_hi)
            if not (float(row.target_p1s_measurement_start_ophrs2_h) >= lo and float(row.target_p1s_measurement_end_ophrs2_h) < hi):
                raise ProtocolFailure("TARGET_INTERVAL_FOLD_ASSIGNMENT_VIOLATION")
    return authority


def fold_positions(authority: pd.DataFrame, fold: str) -> tuple[np.ndarray, np.ndarray]:
    rows = authority.loc[authority.fold.eq(fold)]
    train = rows.loc[rows.role.eq("train"), "sample_index"].to_numpy(int)
    valid = rows.loc[rows.role.eq("validation"), "sample_index"].to_numpy(int)
    if len(train) < 6 or not len(valid):
        raise ProtocolFailure(f"INSUFFICIENT_FOLD_SAMPLES_{fold}")
    return np.sort(train), np.sort(valid)


def metric_record(y_delta: np.ndarray, pred_delta: np.ndarray, origin_r: np.ndarray) -> dict[str, float]:
    y_delta, pred_delta, origin_r = (np.asarray(v, dtype=float) for v in (y_delta, pred_delta, origin_r))
    if not len(y_delta) or len(y_delta) != len(pred_delta) or not np.isfinite(y_delta).all() or not np.isfinite(pred_delta).all():
        raise ProtocolFailure("INVALID_METRIC_INPUT")
    err = pred_delta - y_delta
    y_r = origin_r + y_delta; pred_r = origin_r + pred_delta
    sse = float(np.square(err).sum()); sst = float(np.square(y_r - y_r.mean()).sum())
    rho = float(pd.Series(y_r).corr(pd.Series(pred_r), method="spearman")) if len(y_r) > 1 else math.nan
    mv = err * VREF * 1000.0
    return {
        "n": int(len(y_delta)), "DeltaR_MAE": float(np.mean(np.abs(err))), "DeltaR_RMSE": float(np.sqrt(np.mean(np.square(err)))),
        "Retention_MAE": float(np.mean(np.abs(pred_r - y_r))), "Retention_RMSE": float(np.sqrt(np.mean(np.square(pred_r - y_r)))),
        "MAE_mV": float(np.mean(np.abs(mv))), "RMSE_mV": float(np.sqrt(np.mean(np.square(mv)))),
        "R2_retention": float(1.0 - sse / sst) if sst > 0 else math.nan, "Spearman_retention": rho,
        "P95_absolute_error_mV": float(np.quantile(np.abs(mv), 0.95)), "max_absolute_error_mV": float(np.max(np.abs(mv))),
    }


def summary_features(windows: np.ndarray, origin_r: np.ndarray) -> np.ndarray:
    t = np.arange(windows.shape[1], dtype=float) * 0.5
    centered = t - t.mean(); denom = float(np.square(centered).sum())
    means = windows.mean(axis=1); stds = windows.std(axis=1); lasts = windows[:, -1, :]
    slopes = ((windows - means[:, None, :]) * centered[None, :, None]).sum(axis=1) / denom
    return np.column_stack([origin_r, means, stds, lasts, slopes])


def select_ridge_alpha(features: np.ndarray, target: np.ndarray, train: np.ndarray) -> tuple[float, pd.DataFrame]:
    stopper_n = int(math.ceil(len(train) * 0.15)); fit, stop = train[:-stopper_n], train[-stopper_n:]
    if len(fit) < 3 or not len(stop) or not np.all(fit < stop.min()):
        raise ProtocolFailure("RIDGE_CHRONOLOGICAL_STOPPER_UNAVAILABLE")
    rows = []
    for alpha in ALPHAS:
        scaler = StandardScaler().fit(features[fit])
        model = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(features[fit]), target[fit])
        pred = model.predict(scaler.transform(features[stop]))
        rows.append({"alpha": alpha, "inner_fit_count": len(fit), "inner_stopper_count": len(stop), "inner_chronological_DeltaR_MAE": float(np.mean(np.abs(pred - target[stop])))})
    table = pd.DataFrame(rows).sort_values(["inner_chronological_DeltaR_MAE", "alpha"], kind="stable").reset_index(drop=True)
    return float(table.iloc[0].alpha), table


def prediction_rows(samples: pd.DataFrame, indices: np.ndarray, prediction_delta: np.ndarray, fold: str, model: str, scope: str, seed: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, pred in zip(indices, prediction_delta):
        row = samples.iloc[int(idx)]
        true_delta, origin = float(row.target_DeltaR), float(row.origin_R_P1S_causal)
        rows.append({
            "fold": fold, "model": model, "seed": seed, "scope": scope,
            "origin_cycle_interval_id": row.origin_cycle_interval_id, "target_cycle_interval_id": row.target_cycle_interval_id,
            "origin_raw_cycle_id": int(row.origin_raw_cycle_id), "target_raw_cycle_id": int(row.target_raw_cycle_id),
            "target_p1s_measurement_end_ophrs2_h": float(row.target_p1s_measurement_end_ophrs2_h),
            "target_condition_support_status": row.target_condition_support_status,
            "origin_R_P1S_causal": origin, "true_DeltaR": true_delta, "predicted_DeltaR": float(pred),
            "true_R_P1S_causal": origin + true_delta, "predicted_R_P1S_causal": origin + float(pred),
            "absolute_error_mV": float(abs(pred - true_delta) * VREF * 1000.0),
            "target_measurement_excluded": True, "future_target_skipped": False,
        })
    return rows


def evaluate_baselines(samples: pd.DataFrame, authority: pd.DataFrame, windows: np.ndarray, p1_by_raw: dict[int, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = samples.target_DeltaR.to_numpy(float); origin_r = samples.origin_R_P1S_causal.to_numpy(float)
    features = summary_features(windows, origin_r)
    rows: list[dict[str, Any]] = []; preds: list[dict[str, Any]] = []; selection: list[pd.DataFrame] = []
    for fold, *_ in FOLDS:
        train, valid = fold_positions(authority, fold)
        persistence = np.zeros(len(valid), dtype=float)
        rows.append({"fold": fold, "model": "PERSISTENCE", "scope": "all_validation_samples", "n_train": len(train), "n_validation": len(valid), **metric_record(target[valid], persistence, origin_r[valid])})
        preds.extend(prediction_rows(samples, valid, persistence, fold, "PERSISTENCE", "all_validation_samples"))
        drift_indices, drift_pred = [], []
        for idx in valid:
            raw = int(samples.iloc[int(idx)].origin_raw_cycle_id) - int(samples.iloc[int(idx)].required_target_raw_cycle_id - samples.iloc[int(idx)].origin_raw_cycle_id)
            previous = p1_by_raw.get(raw)
            if previous is not None:
                drift_indices.append(int(idx)); drift_pred.append(float(samples.iloc[int(idx)].origin_R_P1S_causal - previous.R_P1S_causal))
        if drift_indices:
            d_idx = np.asarray(drift_indices, dtype=int); d_pred = np.asarray(drift_pred, dtype=float)
            rows.append({"fold": fold, "model": "PREVIOUS_H_CYCLE_DRIFT", "scope": "matched_previous_H_cycle_causal_P1S_available", "n_train": len(train), "n_validation": len(d_idx), **metric_record(target[d_idx], d_pred, origin_r[d_idx])})
            preds.extend(prediction_rows(samples, d_idx, d_pred, fold, "PREVIOUS_H_CYCLE_DRIFT", "matched_previous_H_cycle_causal_P1S_available"))
        alpha, grid = select_ridge_alpha(features, target, train)
        grid.insert(0, "fold", fold); grid.insert(1, "model", "RIDGE_MATCHED_CONTEXT"); selection.append(grid)
        scaler = StandardScaler().fit(features[train]); ridge = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(features[train]), target[train])
        pred = ridge.predict(scaler.transform(features[valid]))
        rows.append({"fold": fold, "model": "RIDGE_MATCHED_CONTEXT", "scope": "all_validation_samples", "n_train": len(train), "n_validation": len(valid), "selected_alpha": alpha, **metric_record(target[valid], pred, origin_r[valid])})
        preds.extend(prediction_rows(samples, valid, pred, fold, "RIDGE_MATCHED_CONTEXT", "all_validation_samples"))
    return pd.DataFrame(rows), pd.DataFrame(preds), pd.concat(selection, ignore_index=True)


def import_timesnet_components() -> tuple[Any, Any]:
    sys.path.insert(0, str(BS_VENDOR))
    try:
        block_module = __import__("models.TimesNet", fromlist=["TimesBlock"])
        embed_module = __import__("layers.Embed", fromlist=["DataEmbedding"])
        return block_module.TimesBlock, embed_module.DataEmbedding
    finally:
        sys.path.remove(str(BS_VENDOR))


class OriginConditionedTimesNet(torch.nn.Module):
    """Official TimesNet body plus the preregistered scalar origin-retention context head."""
    def __init__(self, times_block: Any, data_embedding: Any) -> None:
        super().__init__()
        cfg = SimpleNamespace(seq_len=48, pred_len=24, top_k=2, d_model=32, d_ff=64, num_kernels=3, e_layers=1, enc_in=14, embed="fixed", freq="h", dropout=0.1)
        self.enc_embedding = data_embedding(cfg.enc_in, cfg.d_model, cfg.embed, cfg.freq, cfg.dropout)
        self.predict_linear = torch.nn.Linear(cfg.seq_len, cfg.seq_len + cfg.pred_len)
        self.blocks = torch.nn.ModuleList([times_block(cfg) for _ in range(cfg.e_layers)])
        self.layer_norm = torch.nn.LayerNorm(cfg.d_model)
        self.delta_head = torch.nn.Linear(cfg.d_model + 1, 1)

    def forward(self, x: torch.Tensor, origin_retention_standardized: torch.Tensor) -> torch.Tensor:
        means = x.mean(1, keepdim=True).detach()
        normalized = (x - means) / torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        latent = self.enc_embedding(normalized, None)
        latent = self.predict_linear(latent.permute(0, 2, 1)).permute(0, 2, 1)
        for block in self.blocks:
            latent = self.layer_norm(block(latent))
        pooled = latent.mean(dim=1)
        return self.delta_head(torch.cat([pooled, origin_retention_standardized.reshape(-1, 1)], dim=1)).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def scale_windows(train_windows: np.ndarray, other_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_windows.reshape(-1, train_windows.shape[-1]))
    return (scaler.transform(train_windows.reshape(-1, train_windows.shape[-1])).reshape(train_windows.shape).astype(np.float32),
            scaler.transform(other_windows.reshape(-1, other_windows.shape[-1])).reshape(other_windows.shape).astype(np.float32))


def train_epochs(model: torch.nn.Module, x: np.ndarray, r: np.ndarray, y: np.ndarray, epochs: int, seed: int, x_stop: np.ndarray | None = None, r_stop: np.ndarray | None = None, y_stop: np.ndarray | None = None, patience: int | None = None) -> tuple[dict[str, Any], int, float]:
    set_seed(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5); loss_fn = torch.nn.MSELoss()
    x_tensor, r_tensor, y_tensor = torch.from_numpy(x), torch.from_numpy(r), torch.from_numpy(y)
    stop_x = torch.from_numpy(x_stop).to(device) if x_stop is not None else None
    stop_r = torch.from_numpy(r_stop).to(device) if r_stop is not None else None
    stop_y = torch.from_numpy(y_stop).to(device) if y_stop is not None else None
    best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), math.inf, 1, 0
    for epoch in range(1, epochs + 1):
        model.train(); permutation = torch.randperm(len(x_tensor))
        for begin in range(0, len(permutation), 64):
            take = permutation[begin: begin + 64]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_tensor[take].to(device), r_tensor[take].to(device)), y_tensor[take].to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if stop_x is None:
            continue
        model.eval()
        with torch.no_grad(): current = float(loss_fn(model(stop_x, stop_r), stop_y).item())
        if current < best_loss - 1e-12:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), current, epoch, 0
        else:
            stale += 1
            if patience is not None and stale >= patience:
                break
    if stop_x is None:
        best_state, best_epoch, best_loss = copy.deepcopy(model.state_dict()), epochs, math.nan
    return best_state, best_epoch, best_loss


def run_timesnet_fold(windows: np.ndarray, origin_r: np.ndarray, target_delta: np.ndarray, train: np.ndarray, valid: np.ndarray, seed: int, components: tuple[Any, Any]) -> TrainResult:
    stopper_n = int(math.ceil(len(train) * 0.15)); fit, stop = train[:-stopper_n], train[-stopper_n:]
    if len(fit) < 3 or not len(stop) or not np.all(fit < stop.min()):
        raise ProtocolFailure("TIMESNET_CHRONOLOGICAL_STOPPER_UNAVAILABLE")
    delta_mean_in, delta_std_in = float(target_delta[fit].mean()), float(target_delta[fit].std(ddof=0))
    r_mean_in, r_std_in = float(origin_r[fit].mean()), float(origin_r[fit].std(ddof=0))
    if delta_std_in <= 0 or r_std_in <= 0: raise ProtocolFailure("TIMESNET_INNER_SCALER_ZERO_VARIANCE")
    x_fit, x_stop = scale_windows(windows[fit], windows[stop])
    r_fit = ((origin_r[fit] - r_mean_in) / r_std_in).astype(np.float32); r_stop = ((origin_r[stop] - r_mean_in) / r_std_in).astype(np.float32)
    y_fit = ((target_delta[fit] - delta_mean_in) / delta_std_in).astype(np.float32); y_stop = ((target_delta[stop] - delta_mean_in) / delta_std_in).astype(np.float32)
    times_block, data_embedding = components; set_seed(seed)
    selection = OriginConditionedTimesNet(times_block, data_embedding)
    _, best_epoch, best_loss = train_epochs(selection, x_fit, r_fit, y_fit, 300, seed, x_stop, r_stop, y_stop, 30)
    del selection
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    delta_mean, delta_std = float(target_delta[train].mean()), float(target_delta[train].std(ddof=0))
    r_mean, r_std = float(origin_r[train].mean()), float(origin_r[train].std(ddof=0))
    if delta_std <= 0 or r_std <= 0: raise ProtocolFailure("TIMESNET_OUTER_SCALER_ZERO_VARIANCE")
    x_train, x_valid = scale_windows(windows[train], windows[valid])
    r_train = ((origin_r[train] - r_mean) / r_std).astype(np.float32); r_valid = ((origin_r[valid] - r_mean) / r_std).astype(np.float32)
    y_train = ((target_delta[train] - delta_mean) / delta_std).astype(np.float32)
    set_seed(seed); final = OriginConditionedTimesNet(times_block, data_embedding)
    state, _, _ = train_epochs(final, x_train, r_train, y_train, max(1, best_epoch), seed)
    final.load_state_dict(state); final.eval(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); final.to(device)
    with torch.no_grad(): pred_std = final(torch.from_numpy(x_valid).to(device), torch.from_numpy(r_valid).to(device)).detach().cpu().numpy()
    parameter_count = sum(param.numel() for param in final.parameters())
    del final
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return TrainResult(predicted_delta=(pred_std * delta_std + delta_mean).astype(float), selected_epoch=int(best_epoch), inner_best_loss=float(best_loss), parameter_count=int(parameter_count))


def evaluate_timesnet(samples: pd.DataFrame, authority: pd.DataFrame, windows: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = import_timesnet_components(); target = samples.target_DeltaR.to_numpy(float); origin_r = samples.origin_R_P1S_causal.to_numpy(float)
    rows: list[dict[str, Any]] = []; pred_rows: list[dict[str, Any]] = []; training: list[dict[str, Any]] = []
    for fold, *_ in FOLDS:
        train, valid = fold_positions(authority, fold)
        for seed in SEEDS:
            result = run_timesnet_fold(windows, origin_r, target, train, valid, seed, components)
            rows.append({"fold": fold, "seed": seed, "n_train": len(train), "n_validation": len(valid), "selected_epoch": result.selected_epoch, "inner_best_standardized_MSE": result.inner_best_loss, "TimesNet_parameters_updated": result.parameter_count, "input_scaler_fit_population": "outer_train_windows_only", "origin_retention_scaler_fit_population": "outer_train_only", "target_delta_scaler_fit_population": "outer_train_only", **metric_record(target[valid], result.predicted_delta, origin_r[valid])})
            pred_rows.extend(prediction_rows(samples, valid, result.predicted_delta, fold, "TIMESNET_ORIGIN_CONDITIONED", "all_validation_samples", seed))
            training.append({"fold": fold, "seed": seed, "selected_epoch": result.selected_epoch, "inner_best_standardized_MSE": result.inner_best_loss, "refit_reinitialised_same_seed": True})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows), pd.DataFrame(training)


def counts_table(horizon: int, samples: pd.DataFrame, excluded: pd.DataFrame, authority: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"horizon": f"H{horizon}", "scope": "legal_unique_physical_horizon_candidates", "n": len(samples), "detail": "target raw cycle equals origin raw cycle plus H; all intervening raw cycles complete"},
        {"horizon": f"H{horizon}", "scope": "excluded_origin_candidates", "n": len(excluded), "detail": "physical continuity, causal target eligibility, or legal-origin-window exclusion"},
        {"horizon": f"H{horizon}", "scope": "frozen_fold_expanded_authority_rows", "n": len(authority), "detail": "chronological train/validation rows"},
    ]
    for fold, *_ in FOLDS:
        for role in ("train", "validation"):
            rows.append({"horizon": f"H{horizon}", "scope": f"{fold}_{role}", "n": int(((authority.fold == fold) & (authority.role == role)).sum()), "detail": "target P1S interval fully inside frozen role range"})
    return pd.DataFrame(rows)


def stratified_results(baseline_pred: pd.DataFrame, times_pred: pd.DataFrame) -> pd.DataFrame:
    time_mean = times_pred.groupby(["fold", "target_cycle_interval_id"], as_index=False).agg(
        true_DeltaR=("true_DeltaR", "first"), predicted_DeltaR=("predicted_DeltaR", "mean"), origin_R_P1S_causal=("origin_R_P1S_causal", "first"), target_condition_support_status=("target_condition_support_status", "first"))
    time_mean["model"] = "TIMESNET_three_seed_mean"
    base = baseline_pred.loc[baseline_pred.model.isin(["PERSISTENCE", "RIDGE_MATCHED_CONTEXT"]), ["fold", "model", "true_DeltaR", "predicted_DeltaR", "origin_R_P1S_causal", "target_condition_support_status"]]
    joined = pd.concat([base, time_mean], ignore_index=True, sort=False)
    rows = []
    for (model, support), group in joined.groupby(["model", "target_condition_support_status"], dropna=False):
        rows.append({"model": model, "target_condition_support_status": support, "posthoc_only_not_used_for_selection": True, **metric_record(group.true_DeltaR, group.predicted_DeltaR, group.origin_R_P1S_causal)})
    return pd.DataFrame(rows).sort_values(["model", "target_condition_support_status"]).reset_index(drop=True)


def horizon_summary(horizon: int, baseline: pd.DataFrame, times: pd.DataFrame) -> dict[str, Any]:
    persistence = baseline.loc[baseline.model.eq("PERSISTENCE")].sort_values("fold")
    ridge = baseline.loc[baseline.model.eq("RIDGE_MATCHED_CONTEXT")].sort_values("fold")
    by_seed = times.groupby("seed", as_index=False).agg(equal_fold_mean_MAE_mV=("MAE_mV", "mean"), equal_fold_mean_RMSE_mV=("RMSE_mV", "mean"))
    time_equal = float(by_seed.equal_fold_mean_MAE_mV.mean()); persistence_equal = float(persistence.MAE_mV.mean()); ridge_equal = float(ridge.MAE_mV.mean())
    f3_time = float(times.loc[times.fold.eq("fold_3"), "MAE_mV"].mean()); f3_persist = float(persistence.loc[persistence.fold.eq("fold_3"), "MAE_mV"].iloc[0])
    gate_a = time_equal <= .95 * persistence_equal
    gate_b = time_equal <= .97 * ridge_equal
    gate_c = f3_time <= 1.05 * f3_persist
    gate_d = int((by_seed.equal_fold_mean_MAE_mV < persistence_equal).sum()) >= 2
    if horizon == PRIMARY_HORIZON:
        if all((gate_a, gate_b, gate_c, gate_d)):
            status = "H6_ORIGIN_CONDITIONED_TIMESNET_SUPPORTED"
        elif ridge_equal < persistence_equal:
            status = "SIMPLE_CAUSAL_CONTEXT_MODEL_SUFFICIENT"
        else:
            status = "MEDIUM_HORIZON_RETENTION_PREDICTABILITY_NOT_ESTABLISHED"
    else:
        status = "H3_SUPPORTIVE_SIGNAL" if all((gate_a, gate_b, gate_c, gate_d)) else ("H3_MIXED" if any((gate_a, gate_b, gate_c, gate_d)) else "H3_NOT_SUPPORTED")
    gap = (time_equal - ridge_equal) / ridge_equal if ridge_equal > 0 else math.nan
    emd = bool(all((gate_a, gate_b, gate_c, gate_d)) or (gate_a and gate_c and gate_d and gap <= .03)) if horizon == PRIMARY_HORIZON else False
    return {
        "horizon": f"H{horizon}", "persistence_equal_fold_mean_MAE_mV": persistence_equal, "ridge_equal_fold_mean_MAE_mV": ridge_equal,
        "timesnet_three_seed_equal_fold_mean_MAE_mV": time_equal, "timesnet_equal_fold_MAE_by_seed_mV": dict(zip(by_seed.seed.astype(str), by_seed.equal_fold_mean_MAE_mV)),
        "fold3_timesnet_three_seed_mean_MAE_mV": f3_time, "fold3_persistence_MAE_mV": f3_persist,
        "timesnet_vs_ridge_relative_gap": gap, "gate_A": "PASS" if gate_a else "FAIL", "gate_B": "PASS" if gate_b else "FAIL", "gate_C": "PASS" if gate_c else "FAIL", "gate_D": "PASS" if gate_d else "FAIL",
        "status": status, "EMD_followup_authorized": emd,
    }


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#d9d9d9", linewidth=.6, alpha=.7); ax.tick_params(labelsize=8)


def make_figures(payload: dict[int, dict[str, Any]]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "figure.dpi": 150, "savefig.dpi": 300})
    h6 = payload[6]; h3 = payload[3]
    tm = h6["times_pred"].groupby(["fold", "target_cycle_interval_id", "target_p1s_measurement_end_ophrs2_h"], as_index=False).agg(true_R=("true_R_P1S_causal", "first"), pred_R=("predicted_R_P1S_causal", "mean"))
    base = h6["baseline_pred"]
    for key, label in (("PERSISTENCE", "persistence_R"), ("RIDGE_MATCHED_CONTEXT", "ridge_R")):
        next_frame = base.loc[base.model.eq(key), ["fold", "target_cycle_interval_id", "predicted_R_P1S_causal"]].rename(columns={"predicted_R_P1S_causal": label})
        tm = tm.merge(next_frame, on=["fold", "target_cycle_interval_id"], how="left")
    tm = tm.sort_values("target_p1s_measurement_end_ophrs2_h")
    fig, ax = plt.subplots(figsize=(8.2, 3.5)); x = tm.target_p1s_measurement_end_ophrs2_h
    ax.plot(x, tm.true_R, color=PALETTE["navy"], marker="o", ms=3, lw=1.5, label="Target raw P1S retention")
    ax.plot(x, tm.persistence_R, color=PALETTE["gray"], marker=".", ms=4, lw=1, label="Persistence")
    ax.plot(x, tm.ridge_R, color=PALETTE["orange"], marker=".", ms=4, lw=1, label="Ridge matched context")
    ax.plot(x, tm.pred_R, color=PALETTE["blue"], marker=".", ms=4, lw=1, label="TimesNet (three-seed mean)")
    for _, _, _, boundary, _ in FOLDS[:-1]: ax.axvline(boundary, color="#999999", ls="--", lw=.8)
    ax.set(xlabel="H6 target P1S measurement end (operating h)", ylabel="Raw P1S retention R", title="H6 chronological Development predictions")
    style_axes(ax); ax.legend(frameon=False, ncol=2, fontsize=7); fig.tight_layout(); fig.savefig(FIG / "Fig1_H6_Chronological_Predictions.png"); plt.close(fig)
    models = [("PERSISTENCE", "Persistence", PALETTE["gray"]), ("RIDGE_MATCHED_CONTEXT", "Ridge", PALETTE["orange"]), ("TIMESNET", "TimesNet", PALETTE["blue"])]
    fig, ax = plt.subplots(figsize=(7.6, 3.5)); xs = np.arange(3); width = .23
    for k, (key, label, color) in enumerate(models):
        values = [float(h6["times"].loc[h6["times"].fold.eq(fold), "MAE_mV"].mean()) if key == "TIMESNET" else float(h6["baseline"].loc[(h6["baseline"].fold.eq(fold)) & (h6["baseline"].model.eq(key)), "MAE_mV"].iloc[0]) for fold, *_ in FOLDS]
        ax.bar(xs + (k - 1) * width, values, width, label=label, color=color)
    ax.set(xticks=xs, xticklabels=[x[0] for x in FOLDS], ylabel="MAE (mV equivalent)", title="H6 error by frozen Development fold")
    style_axes(ax); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig2_H6_Error_By_Fold.png"); plt.close(fig)
    strat = h6["stratified"]; support = sorted(strat.target_condition_support_status.dropna().unique()); fig, ax = plt.subplots(figsize=(7.6, 3.5)); xs = np.arange(len(support)); width=.23
    for k, (key, label, color) in enumerate([("PERSISTENCE", "Persistence", PALETTE["gray"]), ("RIDGE_MATCHED_CONTEXT", "Ridge", PALETTE["orange"]), ("TIMESNET_three_seed_mean", "TimesNet", PALETTE["blue"])]):
        values=[]
        for s in support:
            q=strat.loc[(strat.model.eq(key)) & (strat.target_condition_support_status.eq(s)), "MAE_mV"]
            values.append(float(q.iloc[0]) if len(q) else math.nan)
        ax.bar(xs+(k-1)*width, values, width, color=color, label=label)
    ax.set(xticks=xs, xticklabels=[s.replace("_", " ") for s in support], ylabel="MAE (mV equivalent)", title="H6 post-hoc error by target support")
    style_axes(ax); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig3_H6_Error_By_Support.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6,3.5)); xs=np.arange(2); width=.23
    for k, (key,label,color) in enumerate(models):
        vals=[]
        for obj in (h3,h6):
            vals.append(float(obj["summary"]["timesnet_three_seed_equal_fold_mean_MAE_mV"]) if key=="TIMESNET" else float(obj["summary"]["persistence_equal_fold_mean_MAE_mV"] if key=="PERSISTENCE" else obj["summary"]["ridge_equal_fold_mean_MAE_mV"]))
        ax.bar(xs+(k-1)*width, vals, width, color=color, label=label)
    ax.set(xticks=xs, xticklabels=["H3 secondary", "H6 primary"], ylabel="Equal-fold mean MAE (mV equivalent)", title="Fixed-scope H3 versus H6 skill")
    style_axes(ax); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig4_H3_vs_H6_Skill.png"); plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    view = frame.loc[:, [col for col in columns if col in frame.columns]].head(n).copy()
    for col in view.select_dtypes(include=[np.number]).columns: view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
    header = "| " + " | ".join(view.columns) + " |\n|" + "|".join(["---"] * len(view.columns)) + "|"
    body = "\n".join("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in view.itertuples(index=False, name=None))
    return header + ("\n" + body if body else "")


def build_report(protocol_sha: str, all_payload: dict[int, dict[str, Any]], prefix_fingerprint: str, source_rows: list[dict[str, Any]]) -> None:
    h3, h6 = all_payload[3], all_payload[6]
    report = f"""# Phase CMRF — H3/H6 origin-conditioned P1S-retention forecasting

## Registered question and decision

H6 is the preregistered **primary** horizon and H3 is a secondary descriptive horizon.  Both horizons are defined by the raw physical identity `target_raw_cycle_id = origin_raw_cycle_id + H`; the target is never substituted with the H-th later P1S observation.  Every raw cycle from the origin through target was required to exist in the frozen Development-only raw-cycle catalog and to be marked complete.

**H6 decision: {h6['summary']['status']}**

**H3 secondary status: {h3['summary']['status']}**

The target is `DeltaR_H = R_target - R_origin`, where raw causal P1S retention is `R = V_P1S / {VREF:.12g}`.  Predictions are reconstructed as `R_origin + DeltaR_hat_H`.  No B2-calibrated health proxy, global calibration, smoothed health trajectory, EMD/VMD feature, target timestamp, actual target lead, future condition, target support label, operating age, or target P1S measurement was an input.

## Development-only data boundary

The mixed causal feature source was read exactly through `pd.read_csv(..., nrows=3007)` and filtered by `bin_end_ophrs2_h < 1819.997738`, yielding 3005 Development rows.  Its legally accessible object fingerprint is `{prefix_fingerprint}`; it is **not** a full physical-file hash.  The pre-existing raw-cycle catalog has maximum endpoint 1811.41 h and was checked as a Development-only artifact before it was used.  No Test raw region, Test label, or late-life target was opened.

## Frozen sample authority

{markdown_table(pd.concat([h3['counts'], h6['counts']], ignore_index=True), ['horizon','scope','n','detail'], 20)}

## Baselines and model

Persistence predicts `DeltaR=0`.  The previous-H-cycle drift baseline is evaluated only where the required past causal P1S value exists.  Ridge receives only `R_origin` and the frozen mean/std/last/linear-slope summaries for each of the 14 legal 48-bin origin-history variables.  Its alpha is selected on the chronological final 15% of each outer training set.

TimesNet retains the frozen official body and backbone configuration.  The only task head is the registered temporal mean pooling of the 32-dimensional latent concatenated with the train-standardized observed origin retention, followed by `Linear(33,1)` for standardized `DeltaR`.  H3 and H6 were trained independently.  There is no architecture search or EMD follow-up in this phase.

### H6 primary metrics

{markdown_table(h6['baseline'], ['fold','model','scope','n_validation','MAE_mV','RMSE_mV','R2_retention','Spearman_retention'])}

{markdown_table(h6['times'], ['fold','seed','n_train','n_validation','selected_epoch','MAE_mV','RMSE_mV','R2_retention','Spearman_retention'])}

### H3 secondary metrics

{markdown_table(h3['baseline'], ['fold','model','scope','n_validation','MAE_mV','RMSE_mV','R2_retention','Spearman_retention'])}

{markdown_table(h3['times'], ['fold','seed','n_train','n_validation','selected_epoch','MAE_mV','RMSE_mV','R2_retention','Spearman_retention'])}

## H6 gates

* Gate A (at least 5% below Persistence): **{h6['summary']['gate_A']}**
* Gate B (at least 3% below Ridge on matched context): **{h6['summary']['gate_B']}**
* Gate C (Fold 3 within 105% of Persistence): **{h6['summary']['gate_C']}**
* Gate D (at least two seeds beat Persistence): **{h6['summary']['gate_D']}**

H6 equal-fold mean MAE is {h6['summary']['timesnet_three_seed_equal_fold_mean_MAE_mV']:.4f} mV equivalent, versus {h6['summary']['persistence_equal_fold_mean_MAE_mV']:.4f} for Persistence and {h6['summary']['ridge_equal_fold_mean_MAE_mV']:.4f} for Ridge.  EMD follow-up authorization under the registered rule is **{h6['summary']['EMD_followup_authorized']}**; no follow-up was executed here.

## Interpretation boundary and stop

This study evaluates a causal raw P1S-retention proxy, not independently validated SOH or RUL.  H1 is not retrained or reinterpreted: it used the next physical cycle; this phase uses fixed H3/H6 physical horizons.  Post-hoc support strata only describe the frozen sample authority and never control sample inclusion, alpha selection, seed selection, or gate outcomes.  The phase stops after these two registered horizons; it does not automatically create an EMD, RUL, calibration, Test, or new-backbone experiment.

## Source provenance

{markdown_table(pd.DataFrame(source_rows), ['path','role','sha256','development_prefix_fingerprint_sha256'], 20)}
"""
    (REPORT / "H3_H6_ORIGIN_CONDITIONED_FORECASTING_REPORT.md").write_text(report, encoding="utf-8")


def run() -> dict[str, Any]:
    ensure_dirs(); _, protocol_sha = load_protocol()
    if git_head() != BASE_COMMIT:
        raise ProtocolFailure("BASE_COMMIT_CHANGED")
    causal, fingerprint, fp_manifest = read_causal_development_prefix()
    p1, catalog, replay = load_development_sources()
    sources = source_manifest(fingerprint)
    write_json(fp_manifest, OUT / "DEVELOPMENT_PREFIX_FINGERPRINT_MANIFEST.json")
    write_json({"sources": sources, "test_accessed": False, "full_causal_file_hash_computed": False}, OUT / "CMRF_SOURCE_MANIFEST.json")
    p1_by_raw = {int(row.raw_cycle_id): row for row in p1.itertuples(index=False)}
    payload: dict[int, dict[str, Any]] = {}
    for h in HORIZONS:
        samples, exclusions, windows = construct_horizon_samples(h, p1, catalog, causal)
        authority = fold_expand(samples)
        authority_path = INPUT / f"H{h}_FROZEN_SAMPLE_AUTHORITY.csv"; write_csv(authority, authority_path)
        authority_hash = sha256_file(authority_path); (INPUT / f"H{h}_FROZEN_SAMPLE_AUTHORITY.sha256").write_text(f"{authority_hash}  H{h}_FROZEN_SAMPLE_AUTHORITY.csv\n", encoding="utf-8")
        write_csv(exclusions, INPUT / f"H{h}_SAMPLE_EXCLUSION_LOG.csv")
        counts = counts_table(h, samples, exclusions, authority); write_csv(counts, OUT / f"H{h}_SAMPLE_COUNTS.csv")
        baseline, baseline_pred, ridge_select = evaluate_baselines(samples, authority, windows, p1_by_raw)
        times, times_pred, training = evaluate_timesnet(samples, authority, windows)
        stratified = stratified_results(baseline_pred, times_pred)
        write_csv(baseline, OUT / f"H{h}_BASELINE_RESULTS.csv"); write_csv(times, OUT / f"H{h}_TIMESNET_SEED_RESULTS.csv")
        write_csv(pd.concat([baseline_pred, times_pred], ignore_index=True, sort=False), OUT / f"H{h}_PREDICTIONS.csv")
        write_csv(stratified, OUT / f"H{h}_SUPPORT_STRATIFIED_RESULTS.csv"); write_csv(ridge_select, OUT / f"H{h}_RIDGE_SELECTION.csv"); write_csv(training, OUT / f"H{h}_TIMESNET_TRAINING.csv")
        payload[h] = {"samples": samples, "authority": authority, "counts": counts, "baseline": baseline, "baseline_pred": baseline_pred, "times": times, "times_pred": times_pred, "stratified": stratified, "summary": horizon_summary(h, baseline, times)}
    make_figures(payload)
    h3, h6 = payload[3]["summary"], payload[6]["summary"]
    final_status = h6["status"]
    summary = {
        "phase": "CMRF_H3_H6_OriginConditioned_P1S_Forecasting", "base_commit": BASE_COMMIT, "protocol_sha256": protocol_sha,
        "primary_horizon": "H6", "secondary_horizon": "H3", "target": "DeltaR_H = R_target - R_origin; R_hat_target = R_origin + DeltaR_hat_H", "reference_voltage_V": VREF,
        "development_prefix_rows": DEVELOPMENT_ROWS, "development_prefix_fingerprint_sha256": fingerprint,
        "H3": h3, "H6": h6, "H6_final_status": final_status,
        "H1_reinterpreted_or_retrained": False, "H3_H6_models_independently_trained": True, "test_accessed": False, "test_raw_region_accessed": False, "test_labels_accessed": False,
        "B2_used": False, "EMD_used": False, "operating_age_input": False, "future_actual_lead_input": False, "future_condition_input": False,
        "EMD_followup_executed": False, "RUL_trained": False,
    }
    write_json(summary, OUT / "CMRF_SUMMARY.json")
    build_report(protocol_sha, payload, fingerprint, sources)
    return summary


if __name__ == "__main__":
    try:
        outcome = run()
        print(json.dumps({"status": outcome["H6_final_status"], "primary": "H6", "test_accessed": False}, ensure_ascii=False))
    except ProtocolFailure as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "reason": str(exc), "test_accessed": False}, ensure_ascii=False))
        raise

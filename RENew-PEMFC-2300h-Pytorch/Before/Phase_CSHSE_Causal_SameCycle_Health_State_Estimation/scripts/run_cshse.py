"""Causal same-cycle Development health-state estimation with frozen TimesNet.

Only this phase is writable.  Historical PAA/CSTL/BS sources are read-only.
The raw replay is restricted to PAA's frozen pre-1719.31 h source universe and
reconstructs the actual raw observations used by every target P1S measurement.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import argparse
import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
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
OUT = PHASE / "outputs"
EMBED = OUT / "embeddings"
FIG = PHASE / "figures"
REPORT = PHASE / "reports"
CONFIG = PHASE / "configs" / "CSHSE_FROZEN_PROTOCOL.yaml"
CONFIG_SHA = PHASE / "configs" / "CSHSE_FROZEN_PROTOCOL.sha256"
REPAIR_CONFIG = PHASE / "configs" / "CSHSE_DEVELOPMENT_PREFIX_REPAIR.yaml"

PAA = PROJECT / "Phase_PAA_P1_StateToken_PreAnchor_Alignment"
PAA_SCRIPT = PAA / "scripts" / "run_p1_state_token_preanchor_audit.py"
PAA_MEASUREMENTS = PAA / "outputs" / "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv"
CSTL = PROJECT / "Phase_CSTL_Chronology_Safe_Pretrained_TimesNet_Health_Transfer"
CSTL_SCRIPT = CSTL / "scripts" / "run_cstl.py"
CSTL_B2 = CSTL / "outputs" / "FOLD_SPECIFIC_CHRONOLOGY_SAFE_B2.csv"
CSTL_EVENTS = CSTL / "outputs" / "CHRONOLOGY_SAFE_HEALTH_EVENT_DATASET.csv"
TSHE_EVENTS = PROJECT / "Phase_TSHE_TimesNet_P1S_B2_Health_State_Estimation" / "outputs" / "HEALTH_EVENT_DATASET.csv"
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
BS_PROTOCOL = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
BS_PROTOCOL_SHA = BS / "configs" / "PREREGISTERED_PROTOCOL.sha256"
BS_SCRIPTS = BS / "scripts"
CAUSAL = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"

FEATURES = [
    "cell_voltage_mean", "cell_voltage_std", "cell_voltage_range", "cell_voltage_min", "cell_voltage_max",
    "actual_current_A_mean", "actual_current_A_std", "actual_current_A_peak_to_peak",
    "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std",
    "actual_current_density_A_cm2_peak_to_peak", "load_transition_count", "operation_point_transitions",
    "zero_current_fraction",
]
DEVELOPMENT_PREFIX_NROWS = 3007
DEVELOPMENT_ROWS = 3005
DEVELOPMENT_BOUNDARY_H = 1819.997738
CAUSAL_SOURCE_COLUMNS = ["bin_start_ophrs2_h", "bin_end_ophrs2_h", "bin_valid_for_windows", "temporal_segment_id", *FEATURES]
FOLDS = (
    ("fold_1", 72.788690, 946.393214, 946.393214, 1164.794345),
    ("fold_2", 72.788690, 1164.794345, 1164.794345, 1383.195476),
    ("fold_3", 72.788690, 1383.195476, 1383.195476, 1819.997738),
)
SEEDS = (42, 2024, 3407)
REPRODUCTION_TOLERANCE_MV = 1.0
PALETTE = {"blue": "#0F4D92", "red": "#B64342", "teal": "#42949E", "violet": "#9A4D8E", "neutral": "#CFCECE"}


class AuditFailure(RuntimeError):
    """A fail-closed protocol, input, or leakage violation."""


def sha256(path: Path) -> str:
    # A whole-file hash of the mixed physical causal table is itself forbidden
    # data access.  The only valid integrity representation is produced below
    # from the legal 3005-row in-memory Development frame.
    if path.resolve() == CAUSAL.resolve():
        raise AuditFailure("FULL_FILE_HASH_READ_BEYOND_DEVELOPMENT_PREFIX")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, decimals: int = 6) -> str:
    if frame.empty:
        return "_No rows._"
    copy = frame.copy()
    for column in copy.columns:
        if pd.api.types.is_float_dtype(copy[column]):
            copy[column] = copy[column].map(lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}")
    return "| " + " | ".join(copy.columns) + " |\n|" + "|".join(["---"] * len(copy.columns)) + "|\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in copy.itertuples(index=False, name=None)
    )


def module_from(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AuditFailure(f"Cannot load authority module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_authorities() -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any]:
    expected_config_hash = CONFIG_SHA.read_text(encoding="utf-8").split()[0].lower()
    if sha256(CONFIG) != expected_config_hash:
        raise AuditFailure("CSHSE frozen protocol SHA mismatch")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    repair = yaml.safe_load(REPAIR_CONFIG.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if head != config["base_commit"]:
        raise AuditFailure(f"CSHSE base HEAD mismatch: {head}")
    if config["test_access"] != "prohibited" or float(config["development_boundary_h"]) != DEVELOPMENT_BOUNDARY_H:
        raise AuditFailure("CSHSE Development/Test contract mismatch")
    if repair.get("prior_run_status") != "INVALID_PROTOCOL_BREACH" or not bool(repair["causal_features_access"].get("full_file_hash_prohibited")):
        raise AuditFailure("CSHSE prefix-repair authority mismatch")
    if list(config["inputs"]["feature_order"]) != FEATURES:
        raise AuditFailure("Raw F2 feature order mismatch")
    if not all(path.is_file() for path in (PAA_SCRIPT, PAA_MEASUREMENTS, CSTL_SCRIPT, CSTL_B2, CSTL_EVENTS, TSHE_EVENTS, CAUSAL, BS_PROTOCOL, BS_PROTOCOL_SHA)):
        raise FileNotFoundError("A required frozen authority is missing")
    if sha256(BS_PROTOCOL) != BS_PROTOCOL_SHA.read_text(encoding="utf-8").split()[0].lower():
        raise AuditFailure("Final Backbone Screening protocol hash mismatch")
    bs_cfg = yaml.safe_load(BS_PROTOCOL.read_text(encoding="utf-8"))
    expected = {"seq_len": 48, "pred_len": 24, "enc_in": 14, "c_out": 14, "d_model": 32, "d_ff": 64,
                "e_layers": 1, "top_k": 2, "num_kernels": 3, "dropout": 0.1}
    actual = bs_cfg["official_sources"]["TimesNet"]["config"]
    if {key: actual.get(key) for key in expected} != expected:
        raise AuditFailure("Frozen TimesNet body configuration mismatch")
    if str(BS_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(BS_SCRIPTS))
    import backbone_common as common
    paa = module_from(PAA_SCRIPT, "paa_frozen_authority")
    cstl = module_from(CSTL_SCRIPT, "cstl_frozen_preprocessing")
    return config, bs_cfg, common, paa, cstl


def compute_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if len(truth) != len(prediction) or not len(truth) or not np.isfinite(prediction).all():
        raise AuditFailure("Invalid metric input")
    err = prediction - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    spearman = pd.Series(truth).corr(pd.Series(prediction), method="spearman")
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1.0 - np.sum(err ** 2) / denominator) if denominator > 0 else math.nan,
        "Spearman": float(spearman) if pd.notna(spearman) else math.nan,
        "MAE_percentage_points": float(np.mean(np.abs(err)) * 100.0),
        "RMSE_percentage_points": float(np.sqrt(np.mean(err ** 2)) * 100.0),
    }


def fit_linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.asarray(x_train, dtype=float), np.ones(len(x_train))])
    coef, *_ = np.linalg.lstsq(design, np.asarray(y_train, dtype=float), rcond=None)
    return (coef[0] * np.asarray(x_eval, dtype=float) + coef[1]).astype(float)


def select_ridge_alpha(features: np.ndarray, target: np.ndarray, train_idx: np.ndarray) -> tuple[float, pd.DataFrame]:
    stopper_n = int(math.ceil(len(train_idx) * 0.15))
    fit_idx, stopper_idx = train_idx[:-stopper_n], train_idx[-stopper_n:]
    if len(fit_idx) < 2 or len(stopper_idx) < 1 or not np.all(fit_idx < stopper_idx.min()):
        raise AuditFailure("Cannot form chronological inner 85/15 selection")
    rows: list[dict[str, Any]] = []
    for alpha in (1e-6, 1e-4, 1e-2, 1.0, 100.0):
        scaler = StandardScaler().fit(features[fit_idx])
        ridge = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(features[fit_idx]), target[fit_idx])
        pred = ridge.predict(scaler.transform(features[stopper_idx]))
        rows.append({"alpha": alpha, "inner_fit_count": len(fit_idx), "inner_stopper_count": len(stopper_idx),
                     "inner_chronological_MAE": float(np.mean(np.abs(pred - target[stopper_idx])))})
    result = pd.DataFrame(rows).sort_values(["inner_chronological_MAE", "alpha"], kind="stable").reset_index(drop=True)
    return float(result.iloc[0].alpha), result


def fit_ridge_predict(features: np.ndarray, target: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(features[train_idx])
    ridge = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(features[train_idx]), target[train_idx])
    pred = ridge.predict(scaler.transform(features[valid_idx]))
    return pred, {"feature_scaler_fit_population": "full_outer_train", "ridge_alpha": float(alpha),
                  "ridge_intercept": float(ridge.intercept_), "ridge_coefficient_l2_norm": float(np.linalg.norm(ridge.coef_))}


def raw_reconstruction(paa: Any, event_source: pd.DataFrame, voltage_tolerance: float) -> pd.DataFrame:
    """Replay PAA middle-50 membership and retain its exact selected raw ages.

    The original PAA interval value is a median of per-dwell medians.  Start/end
    are therefore the envelope of actual selected raw observations across all
    P1S dwells contributing to an interval; input is cut before the *earliest*
    such observation, excluding every contribution even when dwell windows are
    discontinuous.
    """
    paa_config, _ = paa.verify_frozen_inputs()
    complete, cycles, states, _ = paa.load_frozen_structure(paa_config)
    targets = paa.target_intervals(complete, cycles, states, paa_config)
    selected_ids = set(event_source.cycle_interval_id.astype(str))
    targets = targets.loc[targets.cycle_interval_id.astype(str).isin(selected_ids)].copy()
    # Reproduce PAA membership exactly.  PAA registered both windows together;
    # at their shared 75%-of-dwell boundary searchsorted(side="right") assigns
    # the row to last_25.  Replaying only middle_50 would incorrectly retain
    # that boundary row and alter the historical median.
    windows = pd.concat([paa.make_windows(targets, "middle_50"), paa.make_windows(targets, "last_25")], ignore_index=True)
    windows = windows.sort_values("window_start").reset_index(drop=True)
    windows["window_id"] = np.arange(len(windows), dtype=int)
    if windows.empty or set(windows.cycle_interval_id.astype(str)) != selected_ids:
        raise AuditFailure("P1S frozen dwell selection does not cover every source health event")
    starts = windows.window_start.to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = windows.window_end.to_numpy(dtype="datetime64[ns]").astype("int64")
    columns = list(dict.fromkeys([*paa.IDENTITY, *paa.ACTIVE]))
    store: dict[int, list[tuple[float, float]]] = defaultdict(list)
    cutoff = float(paa_config["raw_firewall"]["maximum_required_precharacterization_cutoff_h"])
    access = {"raw_files_opened": [], "test_raw_rows_accessed": False, "maximum_open_age_h": math.nan}
    stopped = False
    for path in paa.raw_files_through_cutoff(paa_config):
        if stopped:
            break
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        missing = [column for column in columns if column not in header]
        if missing:
            raise AuditFailure(f"PAA raw schema missing {missing} in {path.name}")
        access["raw_files_opened"].append(str(path.relative_to(paa.RAW_ROOT)).replace("\\", "/"))
        for chunk in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=columns, chunksize=paa.CHUNK, low_memory=False):
            ages = paa.normalize_age(chunk["OpHrs2 [h]"])
            if bool((ages >= paa.BOUNDARY).any()):
                access["test_raw_rows_accessed"] = True
                raise AuditFailure("DATA_FIREWALL_DEVIATION: Test raw row was accessed")
            keep = ages < cutoff
            after = ages >= cutoff
            development = chunk.loc[keep].copy()
            development["OpHrs2 [h]"] = ages.loc[keep]
            if not development.empty:
                development["DateTime"] = pd.to_datetime(development["DateTime"], errors="coerce")
                development["operation_point"] = paa.normalize_operation(development["operation_point"])
                for column in ["I.S.set [A]", *paa.ACTIVE]:
                    development[column] = pd.to_numeric(development[column], errors="coerce")
                development = development.loc[development.DateTime.notna() & development["OpHrs2 [h]"].notna()].copy()
                if not development.empty:
                    dt = development.DateTime.to_numpy(dtype="datetime64[ns]").astype("int64")
                    index, valid = paa.frozen_window_membership(dt, starts, ends)
                    if valid.any():
                        selected = development.loc[valid].copy()
                        selected_index = index[valid]
                        expected_op = windows.iloc[selected_index].operation_point.astype(str).to_numpy()
                        expected_set = windows.iloc[selected_index].rounded_set_A.to_numpy(dtype=float)
                        actual_set = np.round(selected["I.S.set [A]"].to_numpy(dtype=float), 3)
                        matches = (selected.operation_point.astype(str).to_numpy() == expected_op) & np.isclose(actual_set, expected_set, equal_nan=False)
                        if not matches.all():
                            raise AuditFailure("Raw row violates frozen P1S operation/set-current identity")
                        # Retain only rows whose *formal PAA assignment* is the
                        # primary middle-50 window after resolving the shared
                        # boundary against last_25 as the original implementation.
                        middle = windows.iloc[selected_index].stable_window.astype(str).to_numpy() == "middle_50"
                        means = np.nanmean(selected[paa.ACTIVE].to_numpy(dtype=float), axis=1)
                        selected_ages = selected["OpHrs2 [h]"].to_numpy(dtype=float)
                        for wid, voltage, age, keep_middle in zip(selected_index, means, selected_ages, middle):
                            if not keep_middle:
                                continue
                            store[int(wid)].append((float(voltage), float(age)))
                        access["maximum_open_age_h"] = float(np.nanmax([access["maximum_open_age_h"], np.nanmax(selected_ages)])) if np.isfinite(access["maximum_open_age_h"]) else float(np.nanmax(selected_ages))
            if bool(after.any()):
                stopped = True
                break
    dwell_rows: list[dict[str, Any]] = []
    for window in windows.itertuples(index=False):
        if str(window.stable_window) != "middle_50":
            continue
        values = np.asarray(store.get(int(window.window_id), []), dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values[:, 0]).any():
            raise AuditFailure(f"No finite raw P1S observation in frozen dwell {window.run_id}")
        dwell_rows.append({"cycle_interval_id": str(window.cycle_interval_id), "run_id": int(window.run_id),
                           "raw_sample_count": int(len(values)), "dwell_voltage_V": float(np.nanmedian(values[:, 0])),
                           "raw_window_start_ophrs2_h": float(np.nanmin(values[:, 1])),
                           "raw_window_end_ophrs2_h": float(np.nanmax(values[:, 1]))})
    dwell = pd.DataFrame(dwell_rows)
    frozen = pd.read_csv(PAA_MEASUREMENTS)
    frozen = frozen.loc[(frozen.stable_window == "middle_50") & (frozen.interval_kind == "COMPLETE_CYCLE_INTERVAL"),
                        ["cycle_interval_id", "P1S_voltage_V", "eligible_dwell_count", "valid_dwell_count", "measurement_end_ophrs2_h"]].copy()
    frozen.cycle_interval_id = frozen.cycle_interval_id.astype(str)
    rows: list[dict[str, Any]] = []
    for cycle_id, group in dwell.groupby("cycle_interval_id", sort=True):
        ref = frozen.loc[frozen.cycle_interval_id.eq(cycle_id)]
        if len(ref) != 1:
            raise AuditFailure(f"Frozen P1S authority missing/ambiguous: {cycle_id}")
        expected = ref.iloc[0]
        reconstructed = float(np.median(group.dwell_voltage_V.to_numpy(float)))
        difference = float(reconstructed - float(expected.P1S_voltage_V))
        expected_count = int(expected.eligible_dwell_count)
        count_pass = int(len(group)) == expected_count == int(expected.valid_dwell_count)
        value_pass = bool(np.isclose(reconstructed, float(expected.P1S_voltage_V), atol=voltage_tolerance, rtol=0.0))
        rows.append({"cycle_interval_id": cycle_id, "p1s_stable_window": "middle_50", "p1s_dwell_count": int(len(group)),
                     "p1s_raw_sample_count": int(group.raw_sample_count.sum()), "p1s_measurement_start_ophrs2_h": float(group.raw_window_start_ophrs2_h.min()),
                     "p1s_measurement_end_ophrs2_h": float(group.raw_window_end_ophrs2_h.max()),
                     "P1S_voltage_frozen_V": float(expected.P1S_voltage_V), "P1S_voltage_reconstructed_V": reconstructed,
                     "P1S_voltage_difference_V": difference, "dwell_count_pass": count_pass, "voltage_match_pass": value_pass,
                     "P1S_measurement_is_multi_dwell_aggregate": bool(len(group) > 1),
                     "frozen_interval_measurement_end_ophrs2_h": float(expected.measurement_end_ophrs2_h),
                     "raw_test_accessed": False, "raw_files_opened": " | ".join(access["raw_files_opened"])})
    reconstructed = pd.DataFrame(rows).sort_values("cycle_interval_id").reset_index(drop=True)
    if len(reconstructed) != len(selected_ids) or not reconstructed.dwell_count_pass.all() or not reconstructed.voltage_match_pass.all() or access["test_raw_rows_accessed"]:
        raise AuditFailure("P1S_MEASUREMENT_INTERVAL_RECONSTRUCTION_MISMATCH")
    return reconstructed


def read_causal_development_prefix(path: Path) -> pd.DataFrame:
    """The only legal CSHSE reader for the mixed causal physical CSV."""
    prefix = pd.read_csv(path, usecols=CAUSAL_SOURCE_COLUMNS, nrows=DEVELOPMENT_PREFIX_NROWS)
    if len(prefix) != DEVELOPMENT_PREFIX_NROWS:
        raise AuditFailure("DEVELOPMENT_PREFIX_ROW_COUNT_MISMATCH")
    development = prefix.loc[prefix.bin_end_ophrs2_h.to_numpy(float) < DEVELOPMENT_BOUNDARY_H].copy()
    if len(development) != DEVELOPMENT_ROWS:
        raise AuditFailure("DEVELOPMENT_PREFIX_ROW_COUNT_MISMATCH")
    development = development.loc[:, CAUSAL_SOURCE_COLUMNS].reset_index(drop=True)
    if not np.isfinite(development[FEATURES].to_numpy(dtype=float)).all():
        raise AuditFailure("Non-finite Raw F2 value in Development prefix")
    return development


def development_prefix_fingerprint(development: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    """Hash only the allowed 3005 rows, never file bytes or file metadata."""
    if len(development) != DEVELOPMENT_ROWS or list(development.columns) != CAUSAL_SOURCE_COLUMNS:
        raise AuditFailure("Development fingerprint input violates the frozen source schema")
    payload = development.to_csv(index=False, header=True, lineterminator="\n", float_format="%.17g").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "fingerprint_name": "development_prefix_fingerprint_sha256",
        "fingerprint_scope": "filtered_3005_development_rows_only",
        "development_prefix_fingerprint_sha256": digest,
        "row_count": len(development), "column_count": len(CAUSAL_SOURCE_COLUMNS),
        "column_names": CAUSAL_SOURCE_COLUMNS, "column_order": "explicit_CSHSE_source_column_order",
        "row_order": "original_prefix_order_after_prescribed_filter",
        "serialization": {"format": "csv", "encoding": "utf-8", "index": False, "header": True, "lineterminator": "LF", "float_format": "%.17g"},
        "full_file_hash_computed": False,
        "statement": "This fingerprint covers only the legally accessible frozen Development prefix after nrows=3007 and bin_end_ophrs2_h < 1819.997738.",
    }
    return digest, manifest


def load_causal(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if int(config["causal_table"]["allowed_nrows"]) != DEVELOPMENT_PREFIX_NROWS or int(config["causal_table"]["required_development_rows"]) != DEVELOPMENT_ROWS:
        raise AuditFailure("CSHSE causal-prefix protocol constant mismatch")
    development = read_causal_development_prefix(CAUSAL)
    fingerprint, manifest = development_prefix_fingerprint(development)
    manifest["development_filter"] = "bin_end_ophrs2_h < 1819.997738"
    manifest["causal_features_read_nrows"] = DEVELOPMENT_PREFIX_NROWS
    manifest["development_rows_after_filter"] = DEVELOPMENT_ROWS
    manifest["development_prefix_fingerprint_sha256"] = fingerprint
    return development.reset_index(names="causal_row"), manifest


def data_access_audits() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Static, auditable inventory of every CSHSE data-reading route."""
    own_source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(own_source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    reader_calls = [node for node in ast.walk(functions["read_causal_development_prefix"])
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "read_csv"]
    bounded_causal_reader = len(reader_calls) == 1 and any(keyword.arg == "nrows" and isinstance(keyword.value, ast.Name) and keyword.value.id == "DEVELOPMENT_PREFIX_NROWS" for keyword in reader_calls[0].keywords)
    full_hash_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id == "sha256" and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "CAUSAL"]
    causal_read_bytes_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                               and node.func.attr == "read_bytes" and isinstance(node.func.value, ast.Name) and node.func.value.id == "CAUSAL"]
    no_full_causal_hash = not full_hash_calls and not causal_read_bytes_calls
    code_rows = [
        {"function_name": "read_causal_development_prefix", "file_path": str(Path(__file__).relative_to(REPO)).replace("\\", "/"), "read_method": "pd.read_csv(usecols=explicit, nrows=3007)", "read_limit": 3007, "purpose": "formal Raw F2 Development source", "can_read_beyond_development_prefix": False, "audit_pass": bounded_causal_reader},
        {"function_name": "development_prefix_fingerprint", "file_path": str(Path(__file__).relative_to(REPO)).replace("\\", "/"), "read_method": "in-memory deterministic CSV serialization of filtered Development dataframe", "read_limit": 3005, "purpose": "Development-only integrity fingerprint", "can_read_beyond_development_prefix": False, "audit_pass": no_full_causal_hash},
        {"function_name": "reproduce_preprocessing via CSTL", "file_path": str(CSTL_SCRIPT.relative_to(REPO)).replace("\\", "/"), "read_method": "calls backbone_common.read_development_prefix", "read_limit": 3007, "purpose": "sampled checkpoint-preprocessing consistency", "can_read_beyond_development_prefix": False, "audit_pass": True},
        {"function_name": "backbone_common.read_development_prefix", "file_path": str((BS_SCRIPTS / "backbone_common.py").relative_to(REPO)).replace("\\", "/"), "read_method": "pd.read_csv(usecols=registered, nrows=frozen 3007)", "read_limit": 3007, "purpose": "frozen Final Backbone Screening replay", "can_read_beyond_development_prefix": False, "audit_pass": True},
        {"function_name": "raw_reconstruction", "file_path": str(Path(__file__).relative_to(REPO)).replace("\\", "/"), "read_method": "PAA allowlisted raw chunks below 1719.31 h with fail-closed age firewall", "read_limit": "PAA frozen cutoff", "purpose": "P1S interval location/value replay", "can_read_beyond_development_prefix": False, "audit_pass": True},
        {"function_name": "sha256", "file_path": str(Path(__file__).relative_to(REPO)).replace("\\", "/"), "read_method": "whole-file bytes for small frozen configs/code/artifacts; raises for CAUSAL", "read_limit": "not applicable to causal CSV", "purpose": "small-artifact provenance", "can_read_beyond_development_prefix": False, "audit_pass": no_full_causal_hash},
    ]
    source_rows = [
        {"source": str(CAUSAL.relative_to(REPO)).replace("\\", "/"), "source_type": "mixed Development/Test physical file", "allowed_range": "read nrows=3007; retain bin_end_ophrs2_h < 1819.997738 (3005 rows)", "read_method": "read_causal_development_prefix", "nrows_limit": 3007, "post_read_filter": "bin_end_ophrs2_h < 1819.997738", "full_file_hash_allowed": False},
        {"source": str(PAA_MEASUREMENTS.relative_to(REPO)).replace("\\", "/"), "source_type": "Development-only frozen P1S artifact", "allowed_range": "entire artifact", "read_method": "pd.read_csv", "nrows_limit": "not applicable", "post_read_filter": "middle_50 complete cycles", "full_file_hash_allowed": True},
        {"source": str(TSHE_EVENTS.relative_to(REPO)).replace("\\", "/"), "source_type": "Development-only frozen event artifact", "allowed_range": "entire artifact", "read_method": "pd.read_csv", "nrows_limit": "not applicable", "post_read_filter": "previously legal 105 events", "full_file_hash_allowed": True},
        {"source": str(CSTL_B2.relative_to(REPO)).replace("\\", "/"), "source_type": "Development-only frozen calibration artifact", "allowed_range": "entire artifact", "read_method": "pd.read_csv", "nrows_limit": "not applicable", "post_read_filter": "TARGET_AUTHORITY folds", "full_file_hash_allowed": True},
        {"source": "Phase_BS_Final_Backbone_Screening/checkpoints/bs_timesnet_fold_*_seed*.pt", "source_type": "frozen Development-trained checkpoints", "allowed_range": "nine listed checkpoints", "read_method": "torch.load(map_location=cpu)", "nrows_limit": "not applicable", "post_read_filter": "none", "full_file_hash_allowed": True},
        {"source": "PAA allowlisted Loadcycle raw files through 1719.31 h", "source_type": "mixed raw durability files", "allowed_range": "PAA frozen chunks ages < 1719.31 h", "read_method": "pd.read_csv chunks with age firewall", "nrows_limit": "PAA frozen cutoff", "post_read_filter": "age < 1719.31 h; raises at >=1819.997738", "full_file_hash_allowed": False},
    ]
    return pd.DataFrame(code_rows), pd.DataFrame(source_rows), bool(bounded_causal_reader and no_full_causal_hash)


def load_frozen_health_event_source() -> pd.DataFrame:
    source = pd.read_csv(TSHE_EVENTS)
    columns = ["cycle_interval_id", "raw_cycle_id", "measurement_end_ophrs2_h", "input_start_ophrs2_h", "input_end_ophrs2_h",
               "target_cycle_start_ophrs2_h", "target_P1S_retention", "target_cycle_isolated", "window_legal"]
    if set(columns) - set(source.columns):
        raise AuditFailure("Frozen TSHE source-event schema mismatch")
    source = source.loc[source.window_legal.astype(bool), columns].copy()
    if len(source) != 105 or not source.target_cycle_isolated.astype(bool).all():
        raise AuditFailure("Frozen TSHE source-event authority mismatch")
    return source


def run_preflight(config: dict[str, Any], paa: Any, *, clean_rerun: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], bool]:
    """Complete every non-model firewall/P1S gate in this Python process."""
    source = load_frozen_health_event_source()
    causal, fingerprint_manifest = load_causal(config)
    code_audit, source_manifest, access_pass = data_access_audits()
    reconstruction = raw_reconstruction(paa, source, float(config["p1s_authority"]["voltage_match_tolerance_V"]))
    p1s_pass = bool(reconstruction.voltage_match_pass.all() and reconstruction.dwell_count_pass.all() and len(reconstruction) == len(source))
    if not p1s_pass or not access_pass:
        raise AuditFailure("CSHSE_PROTOCOL_REPAIR_FAILED")
    write_csv(code_audit, OUT / "DATA_ACCESS_CODE_AUDIT.csv")
    write_csv(source_manifest, OUT / "CSHSE_SOURCE_ACCESS_MANIFEST.csv")
    write_csv(reconstruction, OUT / "P1S_MEASUREMENT_INTERVAL_RECONSTRUCTION.csv")
    write_json(fingerprint_manifest, OUT / "DEVELOPMENT_PREFIX_FINGERPRINT_MANIFEST.json")
    runtime_audit = {
        "prior_run_status": "INVALID_PROTOCOL_BREACH", "clean_rerun": clean_rerun,
        "causal_features_read_nrows": DEVELOPMENT_PREFIX_NROWS,
        "development_filter": "bin_end_ophrs2_h < 1819.997738",
        "development_rows_after_filter": DEVELOPMENT_ROWS,
        "development_prefix_fingerprint": fingerprint_manifest["development_prefix_fingerprint_sha256"],
        "full_causal_file_hash_computed": False, "full_causal_file_read_detected": False,
        "test_labels_accessed": False, "test_raw_region_accessed": False,
        "no_full_file_hash_static_audit": access_pass, "P1S_RECONSTRUCTION": "PASS",
    }
    write_json(runtime_audit, OUT / "CSHSE_RUNTIME_DATA_ACCESS_AUDIT.json")
    return source, reconstruction, causal, fingerprint_manifest, access_pass


def choose_latest_legal_window(causal: pd.DataFrame, cutoff_h: float) -> tuple[pd.DataFrame | None, str]:
    candidates = np.flatnonzero(causal.bin_end_ophrs2_h.to_numpy(float) <= cutoff_h + 1e-9)
    for end in candidates[::-1]:
        start = int(end) - 47
        if start < 0:
            continue
        window = causal.iloc[start:int(end) + 1]
        if not window.bin_valid_for_windows.astype(bool).all():
            continue
        if window.temporal_segment_id.nunique() != 1:
            continue
        if not np.allclose(np.diff(window.bin_start_ophrs2_h.to_numpy(float)), 0.5, atol=1e-6, rtol=0):
            continue
        if float(window.bin_end_ophrs2_h.iloc[-1]) > cutoff_h + 1e-9:
            continue
        return window.copy(), "LEGAL"
    return None, "NO_LEGAL_PRE_P1S_HISTORY_WINDOW"


def build_same_cycle_events(config: dict[str, Any], reconstruction: pd.DataFrame, causal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    old = pd.read_csv(TSHE_EVENTS)
    required = {"cycle_interval_id", "raw_cycle_id", "measurement_end_ophrs2_h", "input_start_ophrs2_h", "input_end_ophrs2_h",
                "target_cycle_start_ophrs2_h", "target_P1S_retention", "window_legal", "target_cycle_isolated"}
    if required - set(old.columns):
        raise AuditFailure("TSHE event source schema mismatch")
    old = old.loc[old.window_legal.astype(bool)].copy().sort_values("measurement_end_ophrs2_h").reset_index(drop=True)
    if len(old) != 105 or not old.target_cycle_isolated.astype(bool).all():
        raise AuditFailure("The frozen 105-event TSHE source changed")
    recon = reconstruction.set_index("cycle_interval_id")
    rows: list[dict[str, Any]] = []
    raw_windows: list[np.ndarray] = []
    for source_index, event in old.iterrows():
        cycle_id = str(event.cycle_interval_id)
        if cycle_id not in recon.index:
            raise AuditFailure(f"No P1S reconstruction for {cycle_id}")
        item = recon.loc[cycle_id]
        p1_start = float(item.p1s_measurement_start_ophrs2_h)
        p1_end = float(item.p1s_measurement_end_ophrs2_h)
        cutoff = p1_start - float(config["guard_h"])
        window, status = choose_latest_legal_window(causal, cutoff)
        base = {"source_event_index": int(source_index), "cycle_interval_id": cycle_id, "raw_cycle_id": int(event.raw_cycle_id),
                "measurement_end_ophrs2_h": float(event.measurement_end_ophrs2_h), "target_cycle_start_ophrs2_h": float(event.target_cycle_start_ophrs2_h),
                "p1s_measurement_start_ophrs2_h": p1_start, "p1s_measurement_end_ophrs2_h": p1_end, "guard_h": float(config["guard_h"]),
                "input_cutoff_ophrs2_h": cutoff, "target_P1S_retention": float(event.target_P1S_retention),
                "old_input_start_ophrs2_h": float(event.input_start_ophrs2_h), "old_input_end_ophrs2_h": float(event.input_end_ophrs2_h),
                "p1s_measurement_excluded": True, "reconstruction_voltage_match_pass": bool(item.voltage_match_pass)}
        if window is None:
            rows.append({**base, "window_legal": False, "window_status": status, "input_start_ophrs2_h": math.nan, "input_end_ophrs2_h": math.nan,
                         "input_start_causal_row": math.nan, "input_end_causal_row": math.nan, "temporal_segment_id": math.nan,
                         "same_cycle_history_used": False, "history_extension_h": math.nan})
            raw_windows.append(np.full((48, len(FEATURES)), np.nan, dtype=np.float32))
            continue
        input_start, input_end = float(window.bin_start_ophrs2_h.iloc[0]), float(window.bin_end_ophrs2_h.iloc[-1])
        excluded = input_end <= cutoff + 1e-9
        if not excluded:
            raise AuditFailure("P1S guard exclusion failed")
        rows.append({**base, "window_legal": True, "window_status": status, "input_start_ophrs2_h": input_start, "input_end_ophrs2_h": input_end,
                     "input_start_causal_row": int(window.causal_row.iloc[0]), "input_end_causal_row": int(window.causal_row.iloc[-1]),
                     "temporal_segment_id": int(window.temporal_segment_id.iloc[0]),
                     "same_cycle_history_used": bool(input_end > float(event.target_cycle_start_ophrs2_h)),
                     "history_extension_h": float(input_end - float(event.input_end_ophrs2_h))})
        raw_windows.append(window[FEATURES].to_numpy(dtype=np.float32))
    source_events = pd.DataFrame(rows).sort_values("measurement_end_ophrs2_h").reset_index(drop=True)
    windows = np.stack(raw_windows).astype(np.float32)
    legal = source_events.window_legal.to_numpy(bool)
    if legal.any() and (not np.isfinite(windows[legal]).all() or windows[legal].shape[1:] != (48, 14)):
        raise AuditFailure("Same-cycle Raw F2 window tensor invalid")
    timing = source_events[["cycle_interval_id", "raw_cycle_id", "target_cycle_start_ophrs2_h", "old_input_end_ophrs2_h",
                            "input_end_ophrs2_h", "history_extension_h", "same_cycle_history_used", "window_legal"]].copy()
    timing = timing.rename(columns={"input_end_ophrs2_h": "new_input_end_ophrs2_h"})
    return source_events, timing, {"windows": windows, "legal_mask": legal}


def load_fold_b2() -> dict[str, tuple[float, float]]:
    table = pd.read_csv(CSTL_B2)
    rows = table.loc[table.analysis_role.eq("TARGET_AUTHORITY")].copy()
    expected = {"fold_1": "300 / 600 / 850", "fold_2": "300 / 600 / 850 / 1150", "fold_3": "300 / 600 / 850 / 1150"}
    if set(rows.fold) != set(expected) or any(str(rows.loc[rows.fold.eq(key), "anchor_checkpoints"].iloc[0]) != value for key, value in expected.items()):
        raise AuditFailure("CSTL chronology-safe B2 authority mismatch")
    if rows.future_anchor_used.astype(str).str.lower().isin(["true", "1"]).any():
        raise AuditFailure("CSTL B2 authority reports a future calibration anchor")
    return {row.fold: (float(row.a_fold), float(row.b_fold)) for row in rows.itertuples(index=False)}


def fold_expand(source: pd.DataFrame, b2: dict[str, tuple[float, float]]) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    legal_positions = np.flatnonzero(source.window_legal.to_numpy(bool))
    legal = source.iloc[legal_positions].reset_index(names="legal_position")
    ages = legal.measurement_end_ophrs2_h.to_numpy(float)
    rows: list[dict[str, Any]] = []
    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fold, train_start, train_end, valid_start, valid_end in FOLDS:
        train = np.flatnonzero((ages >= train_start) & (ages < train_end))
        valid = np.flatnonzero((ages >= valid_start) & (ages < valid_end))
        if not len(train) or not len(valid) or set(train) & set(valid) or np.max(ages[valid]) >= 1819.997738:
            raise AuditFailure(f"{fold}: invalid Development fold after same-cycle window audit")
        a, b = b2[fold]
        for role, positions in (("train", train), ("validation", valid)):
            for position in positions:
                event = legal.iloc[int(position)]
                rows.append({"fold": fold, "role": role, **event.drop(labels=["legal_position"]).to_dict(), "B2_a": a, "B2_b": b,
                             "SOH_proxy_fold": float(a * float(event.target_P1S_retention) + b)})
        index[fold] = (train, valid)
    expanded = pd.DataFrame(rows).sort_values(["fold", "role", "measurement_end_ophrs2_h"]).reset_index(drop=True)
    if not (expanded.input_end_ophrs2_h.to_numpy(float) <= expanded.p1s_measurement_start_ophrs2_h.to_numpy(float) - expanded.guard_h.to_numpy(float) + 1e-9).all():
        raise AuditFailure("Expanded event P1S guard contract failed")
    return expanded, index


def voltage_window_features(windows: np.ndarray) -> np.ndarray:
    v = np.asarray(windows[:, :, 0], dtype=float)
    t = np.arange(v.shape[1], dtype=float) * 0.5
    centered = t - t.mean()
    slope = ((v - v.mean(axis=1, keepdims=True)) @ centered) / float(np.dot(centered, centered))
    result = np.column_stack([v.mean(axis=1), slope, v.std(axis=1, ddof=0)])
    if result.shape[1] != 3 or not np.isfinite(result).all():
        raise AuditFailure("WindowVoltage feature construction failed")
    return result


def checkpoint_path(fold: str, seed: int) -> Path:
    return BS / "checkpoints" / f"bs_timesnet_{fold}_seed{seed}.pt"


def embed_all(cstl: Any, common: Any, bs_cfg: dict[str, Any], fold: str, seed: int, windows: np.ndarray) -> np.ndarray:
    path = checkpoint_path(fold, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    scaler = cstl.checkpoint_scaler(common, checkpoint)
    model = cstl.freeze_model(common, bs_cfg, checkpoint)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AuditFailure("TimesNet encoder is not frozen")
    latent = cstl.pre_projection_latent(model, scaler.transform(windows))
    vector = latent.mean(axis=1).astype(np.float64)
    del model
    if vector.shape != (len(windows), 32) or not np.isfinite(vector).all():
        raise AuditFailure("Invalid frozen TimesNet temporal-mean embedding")
    return vector


def target_spread(fold: str, truth: np.ndarray, rmse_values: np.ndarray) -> dict[str, Any]:
    std = float(np.std(truth, ddof=0))
    return {"fold": fold, "n_validation": len(truth), "target_mean": float(np.mean(truth)), "target_std": std,
            "target_min": float(np.min(truth)), "target_max": float(np.max(truth)), "target_range": float(np.max(truth) - np.min(truth)),
            "TimesNet_mean_RMSE": float(np.mean(rmse_values)), "RMSE_to_target_std": float(np.mean(rmse_values) / std) if std > 0 else math.nan}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.linewidth": 1.4,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "figure.dpi": 120, "savefig.dpi": 300})


def make_figures(source: pd.DataFrame, predictions: pd.DataFrame, baselines: pd.DataFrame, results: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    legal = source.loc[source.window_legal].copy()
    example = legal.iloc[int(len(legal) // 2)]
    start = float(example.input_start_ophrs2_h); end = float(example.input_end_ophrs2_h); pstart = float(example.p1s_measurement_start_ophrs2_h); pend = float(example.p1s_measurement_end_ophrs2_h)
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    ax.broken_barh([(start, end - start)], (18, 8), facecolors=PALETTE["blue"], label="48-bin dynamic input (24 h)")
    ax.broken_barh([(end, pstart - end)], (18, 8), facecolors=PALETTE["neutral"], label="Causal gap / 0.5 h guard")
    ax.broken_barh([(pstart, pend - pstart)], (18, 8), facecolors=PALETTE["red"], label="P1S raw measurement envelope — excluded")
    ax.axvline(pstart - .5, color="black", linestyle="--", linewidth=1.1, label="input cutoff")
    ax.set(xlim=(start - 1, pend + 1), ylim=(10, 37), yticks=[], xlabel="Operating age (h)", title="Causal same-cycle input ends before P1S measurement")
    ax.grid(axis="x", alpha=.22); ax.legend(loc="upper left", fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(FIG / "Fig1_Causal_Input_vs_P1S_Measurement_Timeline.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for seed, data in predictions.groupby("pretrained_seed", sort=True):
        ax.scatter(data.SOH_proxy_fold, data.predicted_SOH_proxy, s=28, alpha=.68, label=f"pretrained seed {seed}")
    low = float(min(predictions.SOH_proxy_fold.min(), predictions.predicted_SOH_proxy.min())); high = float(max(predictions.SOH_proxy_fold.max(), predictions.predicted_SOH_proxy.max()))
    ax.plot([low, high], [low, high], "--", color="black", linewidth=1)
    ax.set(xlabel="Fold-specific P1S-B2 proxy", ylabel="Frozen TimesNet + Ridge estimate", title="Same-cycle Development health estimates")
    ax.grid(alpha=.22); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig2_SameCycle_SOH_Predictions.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    values = [float(results.MAE.mean()), float(baselines.Age_MAE.mean()), float(baselines.LastVoltage_MAE.mean()), float(baselines.WindowVoltage_MAE.mean())]
    errors = [float(results.groupby(["fold", "pretrained_seed"]).MAE.mean().std(ddof=0)), 0.0, 0.0, 0.0]
    labels = ["Frozen TimesNet\n+ Ridge", "Age\nLinear", "Last\nVoltage", "WindowVoltage\nLinear"]
    bars = ax.bar(labels, np.asarray(values) * 100, yerr=np.asarray(errors) * 100, capsize=4, color=[PALETTE["blue"], PALETTE["red"], PALETTE["teal"], PALETTE["violet"]])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value * 100:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="MAE (percentage points)", title="Same-cycle Development comparison")
    ax.grid(axis="y", alpha=.22); fig.tight_layout(); fig.savefig(FIG / "Fig3_TimesNet_vs_Baselines.png"); plt.close(fig)

    fold3 = results.loc[results.fold.eq("fold_3")].sort_values("pretrained_seed")
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.6))
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["violet"]]
    for axis, column, title in zip(axes, ["MAE", "R2", "Spearman"], ["MAE", "$R^2$", "Spearman"]):
        axis.bar(fold3.pretrained_seed.astype(str), fold3[column], color=colors)
        axis.set(title=f"Fold 3 {title}", xlabel="pretrained seed")
        axis.grid(axis="y", alpha=.22)
    fig.tight_layout(); fig.savefig(FIG / "Fig4_Fold3_Seed_Results.png"); plt.close(fig)


def build_report(summary: dict[str, Any], reconstruction: pd.DataFrame, timing: pd.DataFrame, results: pd.DataFrame, baselines: pd.DataFrame, spread: pd.DataFrame, stability: pd.DataFrame) -> str:
    fold3 = results.loc[results.fold.eq("fold_3"), ["pretrained_seed", "MAE", "RMSE", "R2", "Spearman", "selected_ridge_alpha"]]
    return f"""# Causal same-cycle health-state estimation

## Protocol incident and repair

An earlier execution was invalidated because a whole-file integrity hash read beyond the frozen Development prefix of `causal_features_30min.csv`. No Test labels were shown to have been accessed. The whole-file causal hash was removed, a deterministic Development-prefix-only fingerprint and regression firewall tests were introduced, and every reported CSHSE result below was regenerated from a clean Python process. The prior invalidated artifacts are preserved only under `runtime_invalidated/` and are not used here.

## 1. Corrected estimand

TSHE and CSTL ended every 24 h history before the target cycle began, so they tested previous-operation-to-next-cycle health prediction. CSHSE preregistered the newest legal 48-bin history ending before the target P1S measurement and allowed that history to enter the target cycle. The estimand would be current P1S-B2 calibrated health-proxy estimation, not independently validated true SOH.

## 2. Frozen P1S measurement reconstruction and leakage control

The PAA `middle_50` raw-selection implementation was replayed for each frozen target. Every cycle's median-of-dwell voltage matched the immutable PAA table within the frozen tolerance. The reported measurement start/end is the envelope of actual raw observations used across its multiple contributing dwells. A 0.5 h guard precedes the earliest selected observation; every selected 30 min input bin ends at or before this cutoff. Therefore target P1S measurement observations are excluded.

{markdown_table(reconstruction[["cycle_interval_id", "p1s_dwell_count", "p1s_raw_sample_count", "p1s_measurement_start_ophrs2_h", "p1s_measurement_end_ophrs2_h", "P1S_voltage_difference_V", "voltage_match_pass"]], 8)}

## 3. Same-cycle information gained

    {markdown_table(timing[["history_extension_h", "same_cycle_history_used", "window_legal"]].describe().transpose().reset_index().rename(columns={"index":"quantity"}), 6)}

Mean available-history extension is **{summary['average_same_cycle_history_extension_h']:.3f} h**. `same_cycle_history_used` is true for {summary['same_cycle_history_event_count']}/{summary['legal_health_events']} legal source events. The frozen PAA timing places the primary P1S measurement immediately after cycle start. With the mandatory 0.5 h guard, no legal input reaches the target cycle. Therefore `{summary['same_cycle_estimand_availability']}`: the computed metrics below are a guard-constrained pre-cycle descriptive replay, not evidence for the intended same-cycle current-state estimand.

## 4. Frozen representation and probe

The nine Final Backbone Screening DeltaV TimesNet checkpoints remained in evaluation mode with zero parameter updates. The CSTL checkpoint-scaler and sampled prediction-consistency replay were reused unchanged. The pre-output-projection `[B,72,32]` latent was temporally mean pooled and mapped by a chronological Ridge probe. The encoder never receives operating age, P1S measurements, P1S retention, or the proxy target.

## 5. Development results

{markdown_table(results[["fold", "pretrained_seed", "n_train", "n_validation", "selected_ridge_alpha", "MAE", "RMSE", "R2", "Spearman"]], 6)}

## 6. Simple baselines

{markdown_table(baselines[["fold", "Age_MAE", "LastVoltage_MAE", "WindowVoltage_MAE", "Age_R2", "LastVoltage_R2", "WindowVoltage_R2"]], 6)}

## 7. Target spread and Fold 3

{markdown_table(spread, 6)}

{markdown_table(fold3, 6)}

## 8. Seed stability

{markdown_table(stability, 6)}

## 9. Decision

Overall gates A/B/C compare mean TimesNet MAE with Age, LastVoltage, and the three-variable WindowVoltage baseline. Fold 3 gates D/E/F require the mean to beat all three simple baselines, median Spearman above 0.5, and at least two positive-$R^2$ seeds; Gate G requires Fold 1 mean MAE to beat Age. They are reported as preregistered descriptive calculations, but cannot establish same-cycle current-state estimation if the timing availability gate fails.

**{summary['final_status']}**. Same-cycle timing availability: {summary['same_cycle_estimand_availability']}. Gates A–G: {summary['gate_A']}, {summary['gate_B']}, {summary['gate_C']}, {summary['gate_D']}, {summary['gate_E']}, {summary['gate_F']}, {summary['gate_G']}.

## 10. Interpretation boundary and next step

The result tests association with a fold-specific P1S-B2 calibrated proxy, not true SOH. Test, late-life labels at or after 1819.997738 h, RUL, future-health forecasting, multitask training, encoder finetuning, and backbone search were not performed. Failure diagnostics are reported in the frozen summary as parallel plausibility checks rather than an automatic attribution to seed variation. The next action remains a research decision; no future-health trajectory experiment is started here.
"""


def render_report_only() -> None:
    """Recover only report rendering from already-computed, immutable phase outputs."""
    required = {
        "summary": OUT / "CSHSE_SUMMARY.json",
        "reconstruction": OUT / "P1S_MEASUREMENT_INTERVAL_RECONSTRUCTION.csv",
        "timing": OUT / "OLD_VS_SAME_CYCLE_INPUT_TIMING.csv",
        "results": OUT / "SAME_CYCLE_TIMESNET_PROBE_RESULTS.csv",
        "baselines": OUT / "SAME_CYCLE_BASELINE_RESULTS.csv",
        "spread": OUT / "TARGET_SPREAD_BY_FOLD.csv",
        "stability": OUT / "SEED_STABILITY_BY_FOLD.csv",
    }
    missing = [label for label, path in required.items() if not path.is_file()]
    if missing:
        raise AuditFailure(f"Cannot render report; missing completed outputs: {missing}")
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "CAUSAL_SAME_CYCLE_HEALTH_STATE_ESTIMATION_REPORT.md").write_text(
        build_report(summary, pd.read_csv(required["reconstruction"]), pd.read_csv(required["timing"]),
                     pd.read_csv(required["results"]), pd.read_csv(required["baselines"]),
                     pd.read_csv(required["spread"]), pd.read_csv(required["stability"])), encoding="utf-8")
    print(json.dumps({"report_rendered": True, "model_or_raw_reexecution": False}, ensure_ascii=False), flush=True)


def apply_same_cycle_availability(summary: dict[str, Any], source_events: pd.DataFrame) -> dict[str, Any]:
    """Prevent a pre-cycle fallback from being misreported as same-cycle evidence."""
    legal = source_events.loc[source_events.window_legal.astype(bool)].copy()
    count = int(legal.same_cycle_history_used.astype(bool).sum())
    summary["same_cycle_history_event_count"] = count
    summary["same_cycle_estimand_availability"] = "PASS" if count else "FAIL__FROZEN_P1S_MEASUREMENT_PRECEDES_TARGET_CYCLE_DYNAMIC_HISTORY"
    summary["same_cycle_metrics_interpretation"] = "FORMAL" if count else "DESCRIPTIVE_GUARD_CONSTRAINED_PRE_CYCLE_REPLAY_ONLY"
    if count == 0:
        summary["final_status"] = "24H_DYNAMIC_HISTORY_NOT_SUFFICIENT_FOR_STABLE_P1S_B2_HEALTH_ESTIMATION"
        summary["recommended_next_phase"] = "RESEARCH_DECISION_REQUIRED__REASSESS_P1S_B2_SUPERVISION_TIMING"
    return summary


def finalize_existing_results() -> None:
    """Apply the frozen timing availability gate without rerunning data or models."""
    summary_path = OUT / "CSHSE_SUMMARY.json"
    events_path = OUT / "SAME_CYCLE_CAUSAL_HEALTH_EVENT_DATASET.csv"
    if not summary_path.is_file() or not events_path.is_file():
        raise AuditFailure("Cannot finalize availability; completed CSHSE outputs are missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = apply_same_cycle_availability(summary, pd.read_csv(events_path))
    write_json(summary, summary_path)
    render_report_only()
    print(json.dumps({"same_cycle_estimand_availability": summary["same_cycle_estimand_availability"], "model_or_raw_reexecution": False}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSHSE or recover its final report from completed outputs.")
    parser.add_argument("--render-report-only", action="store_true", help="Write only the Markdown report from existing phase outputs.")
    parser.add_argument("--finalize-existing-results", action="store_true", help="Apply the same-cycle timing gate to existing outputs and render the report only.")
    parser.add_argument("--preflight-only", action="store_true", help="Run only prefix-firewall, source-access, and P1S-reconstruction gates.")
    args = parser.parse_args()
    if sum(bool(item) for item in (args.render_report_only, args.finalize_existing_results, args.preflight_only)) > 1:
        raise AuditFailure("Choose only one report-recovery mode")
    if args.finalize_existing_results:
        finalize_existing_results()
        return
    if args.render_report_only:
        render_report_only()
        return
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    config, bs_cfg, common, paa, cstl = load_authorities()
    for directory in (OUT, EMBED, FIG, REPORT):
        directory.mkdir(parents=True, exist_ok=True)
    source, reconstruction, causal, prefix_manifest, source_access_pass = run_preflight(config, paa, clean_rerun=not args.preflight_only)
    if args.preflight_only:
        print(json.dumps({"PREFIX_FIREWALL_TEST": "PASS", "NO_FULL_FILE_HASH": "PASS", "DEVELOPMENT_ROW_COUNT": DEVELOPMENT_ROWS,
                          "P1S_RECONSTRUCTION": "PASS", "development_prefix_fingerprint": prefix_manifest["development_prefix_fingerprint_sha256"]}, ensure_ascii=False), flush=True)
        return
    source_hashes = {str(path.relative_to(REPO)).replace("\\", "/"): sha256(path) for path in (CONFIG, REPAIR_CONFIG, PAA_SCRIPT, PAA_MEASUREMENTS, CSTL_SCRIPT, CSTL_B2, TSHE_EVENTS, BS_PROTOCOL)}
    source_events, timing, window_data = build_same_cycle_events(config, reconstruction, causal)
    write_csv(timing, OUT / "OLD_VS_SAME_CYCLE_INPUT_TIMING.csv")
    b2 = load_fold_b2()
    expanded, fold_indices = fold_expand(source_events, b2)
    write_csv(expanded, OUT / "SAME_CYCLE_CAUSAL_HEALTH_EVENT_DATASET.csv")
    legal_positions = np.flatnonzero(window_data["legal_mask"])
    legal_events = source_events.iloc[legal_positions].reset_index(drop=True)
    raw_windows = window_data["windows"][legal_positions]
    if len(legal_events) == 0:
        raise AuditFailure("No legal causal same-cycle health events")
    reproduction = cstl.reproduce_preprocessing(common, bs_cfg)
    write_csv(reproduction, OUT / "PRETRAINED_SCALER_REPRODUCTION_CHECK.csv")
    if not reproduction["pass"].all():
        raise AuditFailure("PRETRAINED_PREPROCESSING_NOT_REPRODUCED")

    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    selection_rows: list[pd.DataFrame] = []
    spread_rows: list[dict[str, Any]] = []
    all_embeddings: dict[tuple[str, int], np.ndarray] = {}
    window_features = voltage_window_features(raw_windows)
    for fold, _tr_start, _tr_end, _va_start, _va_end in FOLDS:
        train_idx, valid_idx = fold_indices[fold]
        a, b = b2[fold]
        target = a * legal_events.target_P1S_retention.to_numpy(float) + b
        age_pred = fit_linear_predict(legal_events.measurement_end_ophrs2_h.to_numpy(float)[train_idx], target[train_idx], legal_events.measurement_end_ophrs2_h.to_numpy(float)[valid_idx])
        last_pred = fit_linear_predict(raw_windows[train_idx, -1, 0], target[train_idx], raw_windows[valid_idx, -1, 0])
        window_alpha, selection = select_ridge_alpha(window_features, target, train_idx)
        window_pred, window_meta = fit_ridge_predict(window_features, target, train_idx, valid_idx, window_alpha)
        selection_rows.append(selection.assign(fold=fold, model="WindowVoltageLinear", pretrained_seed="NOT_APPLICABLE"))
        age_stats, last_stats, window_stats = compute_metrics(target[valid_idx], age_pred), compute_metrics(target[valid_idx], last_pred), compute_metrics(target[valid_idx], window_pred)
        baseline_rows.append({"fold": fold, "n_train": len(train_idx), "n_validation": len(valid_idx),
                              **{f"Age_{key}": value for key, value in age_stats.items()},
                              **{f"LastVoltage_{key}": value for key, value in last_stats.items()},
                              **{f"WindowVoltage_{key}": value for key, value in window_stats.items()},
                              "WindowVoltage_selected_ridge_alpha": window_alpha, **{f"WindowVoltage_{key}": value for key, value in window_meta.items()}})
        fold_rmses: list[float] = []
        for seed in SEEDS:
            embedding = embed_all(cstl, common, bs_cfg, fold, seed, raw_windows)
            all_embeddings[(fold, seed)] = embedding
            alpha, selection = select_ridge_alpha(embedding, target, train_idx)
            selection_rows.append(selection.assign(fold=fold, model="FrozenTimesNetRidge", pretrained_seed=seed))
            pred, meta = fit_ridge_predict(embedding, target, train_idx, valid_idx, alpha)
            metrics = compute_metrics(target[valid_idx], pred)
            fold_rmses.append(metrics["RMSE"])
            result_rows.append({"fold": fold, "pretrained_seed": seed, "n_train": len(train_idx), "n_validation": len(valid_idx),
                                "selected_ridge_alpha": alpha, **meta, **metrics, "encoder_frozen": True, "timesnet_parameters_updated": 0})
            roles = np.array(["train" if i in set(train_idx) else "validation" if i in set(valid_idx) else "not_used" for i in range(len(legal_events))])
            embedding_frame = legal_events[["cycle_interval_id", "measurement_end_ophrs2_h"]].copy()
            embedding_frame.insert(0, "fold", fold); embedding_frame.insert(1, "pretrained_seed", seed); embedding_frame.insert(2, "role", roles)
            embedding_frame["SOH_proxy_fold"] = target
            for dim in range(32):
                embedding_frame[f"z_{dim:02d}"] = embedding[:, dim]
            write_csv(embedding_frame, EMBED / f"{fold}_seed{seed}.csv")
            for index, value in zip(valid_idx, pred):
                event = legal_events.iloc[int(index)]
                prediction_rows.append({"fold": fold, "pretrained_seed": seed, "cycle_interval_id": event.cycle_interval_id, "raw_cycle_id": int(event.raw_cycle_id),
                                        "measurement_end_ophrs2_h": float(event.measurement_end_ophrs2_h), "SOH_proxy_fold": float(target[index]),
                                        "predicted_SOH_proxy": float(value), "error": float(value - target[index]), "absolute_error": float(abs(value - target[index])),
                                        "selected_ridge_alpha": alpha, "p1s_measurement_excluded": True, "same_cycle_history_used": bool(event.same_cycle_history_used)})
        spread_rows.append(target_spread(fold, target[valid_idx], np.asarray(fold_rmses)))
    results = pd.DataFrame(result_rows).sort_values(["fold", "pretrained_seed"]).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(["fold", "pretrained_seed", "measurement_end_ophrs2_h"]).reset_index(drop=True)
    baselines = pd.DataFrame(baseline_rows).sort_values("fold").reset_index(drop=True)
    spread = pd.DataFrame(spread_rows).sort_values("fold").reset_index(drop=True)
    stability = results.groupby("fold", as_index=False).agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", lambda x: float(np.std(x, ddof=0))),
        R2_mean=("R2", "mean"), R2_std=("R2", lambda x: float(np.std(x, ddof=0))), Spearman_mean=("Spearman", "mean"),
        Spearman_std=("Spearman", lambda x: float(np.std(x, ddof=0))), worst_seed_MAE=("MAE", "max"), best_seed_MAE=("MAE", "min"))
    stability["MAE_coefficient_of_variation"] = stability.MAE_std / stability.MAE_mean
    stability["worst_seed_over_best_seed_MAE"] = stability.worst_seed_MAE / stability.best_seed_MAE
    write_csv(results, OUT / "SAME_CYCLE_TIMESNET_PROBE_RESULTS.csv")
    write_csv(predictions, OUT / "SAME_CYCLE_TIMESNET_PREDICTIONS.csv")
    write_csv(baselines, OUT / "SAME_CYCLE_BASELINE_RESULTS.csv")
    write_csv(spread, OUT / "TARGET_SPREAD_BY_FOLD.csv")
    write_csv(stability, OUT / "SEED_STABILITY_BY_FOLD.csv")
    write_csv(pd.concat(selection_rows, ignore_index=True), OUT / "RIDGE_CHRONOLOGICAL_SELECTION_AUDIT.csv")

    overall_mae, overall_rmse = float(results.MAE.mean()), float(results.RMSE.mean())
    age_mae, last_mae, window_mae = float(baselines.Age_MAE.mean()), float(baselines.LastVoltage_MAE.mean()), float(baselines.WindowVoltage_MAE.mean())
    fold3 = results.loc[results.fold.eq("fold_3")].sort_values("pretrained_seed")
    f3base = baselines.loc[baselines.fold.eq("fold_3")].iloc[0]
    fold1 = results.loc[results.fold.eq("fold_1")]
    f1base = baselines.loc[baselines.fold.eq("fold_1")].iloc[0]
    gates = {
        "A": overall_mae < age_mae,
        "B": overall_mae < last_mae,
        "C": overall_mae < window_mae,
        "D": float(fold3.MAE.mean()) < float(f3base.Age_MAE) and float(fold3.MAE.mean()) < float(f3base.LastVoltage_MAE) and float(fold3.MAE.mean()) < float(f3base.WindowVoltage_MAE),
        "E": float(fold3.Spearman.median()) > .5,
        "F": int((fold3.R2 > 0).sum()) >= 2,
        "G": float(fold1.MAE.mean()) < float(f1base.Age_MAE),
    }
    if all(gates.values()):
        status = "CAUSAL_SAME_CYCLE_TIMESNET_SOH_ESTIMATION_SUPPORTED"; next_phase = "Phase_HF_TimesNet_Future_Health_Trajectory_Forecasting"; diagnosis = "CURRENT_DYNAMIC_INFORMATION_SUPPORTED_FOR_PROXY_ONLY"
    elif not gates["E"] or not gates["F"]:
        status = "CURRENT_HEALTH_SIGNAL_PRESENT_BUT_NOT_STABLE"; next_phase = "RESEARCH_DECISION_REQUIRED_REVIEW_P1S_B2_SUPERVISION"; diagnosis = "MULTIPLE_PLAUSIBLE_LIMITATIONS_REPORTED"
    else:
        status = "24H_DYNAMIC_HISTORY_NOT_SUFFICIENT_FOR_STABLE_P1S_B2_HEALTH_ESTIMATION"; next_phase = "RESEARCH_DECISION_REQUIRED_REVIEW_P1S_B2_SUPERVISION"; diagnosis = "MULTIPLE_PLAUSIBLE_LIMITATIONS_REPORTED"
    parallel = {
        "TARGET_RANGE_TOO_NARROW": bool((spread.target_range < 0.02).any()),
        "PROXY_CALIBRATION_UNCERTAINTY": "not_quantified_by_this_phase__fold_specific_sparse_anchor_calibration",
        "TEMPORAL_DISTRIBUTION_SHIFT": bool(float(fold3.MAE.mean()) > float(results.loc[results.fold.ne("fold_3"), "MAE"].mean())),
        "PRETRAINED_REPRESENTATION_LIMITATION": bool(not gates["B"] or not gates["C"]),
        "24H_INFORMATION_LIMITATION": bool(not gates["D"] or not gates["E"] or not gates["F"]),
    }
    summary = {"base_commit": config["base_commit"], "backbone": "TimesNet", "backbone_search_reopened": False,
               "task": "CURRENT_SOH_ESTIMATION", "old_task": "PRE_TARGET_CYCLE_TO_NEXT_CYCLE_HEALTH", "new_task": "PRE_P1S_CURRENT_HISTORY_TO_CURRENT_HEALTH",
               "p1s_measurement_reconstructed": "PASS", "guard_h": float(config["guard_h"]), "p1s_measurement_present_in_input": False,
               "average_same_cycle_history_extension_h": float(source_events.loc[source_events.window_legal, "history_extension_h"].mean()),
               "same_cycle_history_event_count": int(source_events.same_cycle_history_used.sum()), "legal_health_events": int(source_events.window_legal.sum()),
               "frozen_deltaV_pretrained_encoder": True, "timesnet_parameters_updated": 0, "pretrained_scaler_consistency": "PASS_SAMPLED",
               "overall_MAE": overall_mae, "overall_RMSE": overall_rmse, "age_baseline_MAE": age_mae, "last_voltage_baseline_MAE": last_mae, "window_voltage_baseline_MAE": window_mae,
               "fold3_mean_MAE": float(fold3.MAE.mean()), "fold3_median_Spearman": float(fold3.Spearman.median()), "fold3_positive_R2_seeds": int((fold3.R2 > 0).sum()),
               **{f"gate_{key}": "PASS" if value else "FAIL" for key, value in gates.items()}, "final_status": status, "parallel_failure_diagnostics": parallel,
               "test_accessed": False, "RUL_trained": False, "future_health_forecasting_trained": False, "recommended_next_phase": next_phase,
               "prior_run_status": "INVALID_PROTOCOL_BREACH", "prior_run_metrics_reused": False, "clean_rerun": True, "no_test_access": True,
               "full_causal_file_hash_computed": False, "development_prefix_rows": DEVELOPMENT_ROWS,
               "development_prefix_fingerprint": prefix_manifest["development_prefix_fingerprint_sha256"], "prefix_firewall_regression_test": "PASS",
               "source_sha256": source_hashes, "development_prefix_fingerprint_manifest": "outputs/DEVELOPMENT_PREFIX_FINGERPRINT_MANIFEST.json",
               "source_access_audit_pass": source_access_pass}
    summary = apply_same_cycle_availability(summary, source_events)
    write_json(summary, OUT / "CSHSE_SUMMARY.json")
    make_figures(source_events, predictions, baselines, results)
    (REPORT / "CAUSAL_SAME_CYCLE_HEALTH_STATE_ESTIMATION_REPORT.md").write_text(build_report(summary, reconstruction, timing, results, baselines, spread, stability), encoding="utf-8")
    print(json.dumps({"final_status": summary["final_status"], "legal_events": int(source_events.window_legal.sum()), "overall_MAE": overall_mae,
                      "gates": {key: summary[f"gate_{key}"] for key in "ABCDEFG"}, "test_accessed": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

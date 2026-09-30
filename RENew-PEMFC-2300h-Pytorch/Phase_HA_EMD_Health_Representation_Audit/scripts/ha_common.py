"""Shared, read-only guards and deterministic helpers for Phase HA."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_COMMIT = "6efb15bb03af1a8999ccf0b4ff2f75b13f9cd10a"
DEVELOPMENT_END_H = 1819.997738
CONTEXT_STEPS = 48
RESOLUTION_H = 0.5
RAW_PREFIX_ROWS = 3007
EXPECTED_DEVELOPMENT_ROWS = 3005
HA_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = HA_DIR.parent
DF_DIR = REPO_DIR / "Phase_DF_Leakage_Safe_Decomposition_Features"
N1_TABLE = REPO_DIR / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
BS_PROTOCOL = REPO_DIR / "Phase_BS_Final_Backbone_Screening" / "configs" / "PREREGISTERED_PROTOCOL.yaml"

SAFE_N1_COLUMNS = [
    "bin_end_ophrs2_h", "last_datetime", "temporal_segment_id", "cell_voltage_mean",
    "cell_voltage_std", "cell_voltage_range", "actual_current_A_mean",
    "actual_current_A_std", "actual_current_A_peak_to_peak",
    "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std",
    "actual_current_density_A_cm2_peak_to_peak", "load_transition_count",
    "operation_point_transitions", "zero_current_fraction", "startup_flag", "shutdown_flag",
    "gap_before_flag", "gap_before_duration_s", "gap_after_flag",
]
REQUIRED_DF_FILES = [
    "configs/PREREGISTERED_DF_PROTOCOL.yaml",
    "configs/PREREGISTERED_DF_PROTOCOL.sha256",
    "configs/RESOLVED_VMD_PARAMETERS.yaml",
    "scripts/decomposition_features.py",
    "scripts/run_stage0.py",
    "scripts/run_stage1.py",
    "docs/CAUSALITY_CONTRACT.md",
    "docs/FEATURE_DICTIONARY.md",
    "outputs/CACHE_PROVENANCE_MANIFEST.csv",
    "outputs/ENDPOINT_AUDIT.csv",
    "outputs/STAGE0_SUMMARY.json",
    "outputs/LOCAL_ARTIFACT_MANIFEST_AUDIT.json",
    "final/FINAL_DECISION.yaml",
    "final/FINAL_RESEARCH_REPORT.md",
]


class AuditFailure(RuntimeError):
    """Fail closed before any endpoint statistics are calculated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()


def git_branch() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_DIR, text=True).strip()


def ensure_base_commit() -> None:
    if git_head() != BASE_COMMIT:
        raise AuditFailure(f"HEAD mismatch: expected {BASE_COMMIT}, found {git_head()}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_development_prefix() -> tuple[pd.DataFrame, dict]:
    """Read only the frozen N1 prefix and only non-target causal columns."""
    if not N1_TABLE.exists():
        raise AuditFailure(f"missing N1 causal table: {N1_TABLE}")
    frame = pd.read_csv(N1_TABLE, usecols=SAFE_N1_COLUMNS, nrows=RAW_PREFIX_ROWS)
    if len(frame) != RAW_PREFIX_ROWS:
        raise AuditFailure(f"expected {RAW_PREFIX_ROWS} prefix rows, found {len(frame)}")
    frame.index.name = "origin_row"
    development = frame.loc[frame["bin_end_ophrs2_h"] < DEVELOPMENT_END_H].copy()
    if len(development) != EXPECTED_DEVELOPMENT_ROWS:
        raise AuditFailure(f"expected {EXPECTED_DEVELOPMENT_ROWS} Development rows, found {len(development)}")
    if len(frame) - len(development) != 2:
        raise AuditFailure("exactly two boundary/post-boundary prefix rows were required")
    if (development["bin_end_ophrs2_h"] >= DEVELOPMENT_END_H).any():
        raise AuditFailure("post-boundary age entered Development frame")
    audit = {
        "source_relative_path": str(N1_TABLE.relative_to(REPO_DIR)).replace("\\", "/"),
        "pandas_read_csv_nrows": RAW_PREFIX_ROWS,
        "raw_prefix_rows_read": len(frame),
        "development_rows_after_filter": len(development),
        "boundary_rows_filtered": len(frame) - len(development),
        "post_boundary_rows_entered_pipeline": int((development["bin_end_ophrs2_h"] >= DEVELOPMENT_END_H).sum()),
        "test_accessed": False,
        "future_target_loaded": False,
        "loaded_columns": SAFE_N1_COLUMNS,
    }
    return development, audit


def source_hashes() -> dict[str, str]:
    results: dict[str, str] = {}
    for relative in REQUIRED_DF_FILES:
        path = DF_DIR / relative
        if not path.exists():
            raise AuditFailure(f"missing frozen Phase DF source: {relative}")
        results[f"Phase_DF_Leakage_Safe_Decomposition_Features/{relative}"] = sha256_file(path)
    if not BS_PROTOCOL.exists():
        raise AuditFailure("missing frozen Phase BS protocol")
    results["Phase_BS_Final_Backbone_Screening/configs/PREREGISTERED_PROTOCOL.yaml"] = sha256_file(BS_PROTOCOL)
    results[str(N1_TABLE.relative_to(REPO_DIR)).replace("\\", "/")] = sha256_file(N1_TABLE)
    return results


def validate_df_status() -> dict:
    text = (DF_DIR / "final" / "FINAL_DECISION.yaml").read_text(encoding="utf-8")
    required = ("EMD_SPECIFIC_ADVANCE", "TIMESNET_FREEZE", "CLOSED")
    missing = [item for item in required if item not in text]
    if missing:
        raise AuditFailure(f"Phase DF inherited status missing: {missing}")
    return {"EMD_SPECIFIC_ADVANCE": True, "TIMESNET_FREEZE": True, "BACKBONE_SEARCH_CLOSED": True}


def read_manifest() -> pd.DataFrame:
    path = DF_DIR / "outputs" / "CACHE_PROVENANCE_MANIFEST.csv"
    manifest = pd.read_csv(path)
    needed = {"origin_row", "origin_age_h", "temporal_segment_id", "method", "cache_path", "cache_sha256", "input_voltage_sha256", "method_parameter_sha256", "feature_shape", "fold_membership"}
    missing = needed.difference(manifest.columns)
    if missing:
        raise AuditFailure(f"cache manifest columns missing: {sorted(missing)}")
    return manifest


def cache_universe(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    simple = manifest.loc[manifest["method"].eq("SimpleTrend")].copy()
    emd = manifest.loc[manifest["method"].eq("EMD")].copy()
    for label, table in (("SimpleTrend", simple), ("EMD", emd)):
        if table["origin_row"].duplicated().any():
            raise AuditFailure(f"duplicate {label} origin in cache manifest")
        if not table["feature_shape"].eq("[48,2]").all():
            raise AuditFailure(f"non-[48,2] {label} cache found")
    joined = simple.merge(emd, on="origin_row", suffixes=("_simple", "_emd"), validate="one_to_one")
    for column in ("origin_age_h", "temporal_segment_id", "input_voltage_sha256", "fold_membership"):
        a, b = f"{column}_simple", f"{column}_emd"
        if not (joined[a].astype(str).eq(joined[b].astype(str))).all():
            raise AuditFailure(f"SimpleTrend/EMD provenance mismatch for {column}")
    if len(joined) != 1992:
        raise AuditFailure(f"expected 1992 common legal cache origins, found {len(joined)}")
    return simple, emd, joined


def p33_p67_from_fold1_training(development: pd.DataFrame, common: pd.DataFrame) -> tuple[float, float, int]:
    # Frozen outer-train boundary from Phase BS.  Cache universe prevents target construction.
    ages = common[["origin_row", "origin_age_h_simple"]].rename(columns={"origin_age_h_simple": "origin_age_h"})
    merged = ages.merge(development[["actual_current_density_A_cm2_mean"]], left_on="origin_row", right_index=True, validate="one_to_one")
    fold1 = merged.loc[(merged.origin_age_h >= 72.788690) & (merged.origin_age_h < 946.393214)]
    if len(fold1) != 1048:
        raise AuditFailure(f"expected 1048 cached Fold1 outer-train origins, found {len(fold1)}")
    positive = fold1.loc[fold1.actual_current_density_A_cm2_mean > 0, "actual_current_density_A_cm2_mean"].to_numpy(float)
    if len(positive) < 10:
        raise AuditFailure("insufficient positive-current Fold1 origins for frozen strata")
    return float(np.quantile(positive, .33)), float(np.quantile(positive, .67)), int(len(positive))


def load_protocol() -> dict:
    path = HA_DIR / "configs" / "PREREGISTERED_HA_PROTOCOL.yaml"
    digest_path = HA_DIR / "configs" / "PREREGISTERED_HA_PROTOCOL.sha256"
    if not path.exists() or not digest_path.exists():
        raise AuditFailure("Phase HA protocol/hash has not been frozen")
    recorded = digest_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(path)
    if actual != recorded:
        raise AuditFailure("Phase HA protocol SHA256 mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_protocol_sources(protocol: dict) -> None:
    for relative, expected in protocol["frozen_source_hashes"].items():
        actual = sha256_file(REPO_DIR / relative)
        if actual != expected:
            raise AuditFailure(f"frozen source hash mismatch: {relative}")


def load_npz_checked(row: pd.Series, expected_method_parameter: str) -> np.ndarray:
    cache_path = Path(row.cache_path)
    if not cache_path.exists():
        raise AuditFailure(f"missing local cache: {cache_path}")
    if sha256_file(cache_path) != row.cache_sha256:
        raise AuditFailure(f"cache SHA mismatch: {cache_path.name}")
    with np.load(cache_path, allow_pickle=False) as archive:
        required = {"features", "origin_row", "input_voltage_sha256", "method_parameter_sha256"}
        if not required.issubset(archive.files):
            raise AuditFailure(f"missing cache arrays: {cache_path.name}")
        features = archive["features"]
        origin = int(archive["origin_row"].item())
        input_hash = str(archive["input_voltage_sha256"].item())
        param_hash = str(archive["method_parameter_sha256"].item())
    if origin != int(row.origin_row) or input_hash != str(row.input_voltage_sha256):
        raise AuditFailure(f"cache provenance mismatch: {cache_path.name}")
    if param_hash != expected_method_parameter or param_hash != str(row.method_parameter_sha256):
        raise AuditFailure(f"cache parameter mismatch: {cache_path.name}")
    if features.shape != (48, 2) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise AuditFailure(f"invalid cache feature array: {cache_path.name}")
    return features


def consecutive_runs(frame: pd.DataFrame, n_origins: int) -> list[pd.DataFrame]:
    """Non-overlapping rows that are exactly 0.5 h and same segment/third."""
    pieces: list[pd.DataFrame] = []
    for _, subset in frame.groupby(["chronological_third", "temporal_segment_id"], sort=True):
        subset = subset.sort_values("origin_row")
        start = 0
        while start + n_origins <= len(subset):
            candidate = subset.iloc[start:start + n_origins]
            rows_ok = np.all(np.diff(candidate.origin_row.to_numpy()) == 1)
            age_ok = np.allclose(np.diff(candidate.origin_age_h.to_numpy(float)), RESOLUTION_H, atol=1e-9)
            if rows_ok and age_ok:
                pieces.append(candidate)
                start += n_origins
            else:
                start += 1
    return pieces


def chronological_thirds(ages: pd.Series) -> tuple[float, float]:
    low, high = float(ages.min()), float(ages.max())
    return low + (high-low)/3, low + 2*(high-low)/3


def assign_stratum(value: float, p33: float, p67: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return "ZERO_CURRENT"
    if value <= p33:
        return "LOW"
    if value <= p67:
        return "MID"
    return "HIGH"


def quantile_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: np.nan for key in ("signed_mean", "signed_median", "abs_median", "p75_abs", "p90_abs", "p95_abs", "p99_abs", "max_abs", "n")}
    absolute = np.abs(array)
    return {
        "signed_mean": float(array.mean()), "signed_median": float(np.median(array)),
        "abs_median": float(np.median(absolute)), "p75_abs": float(np.quantile(absolute, .75)),
        "p90_abs": float(np.quantile(absolute, .90)), "p95_abs": float(np.quantile(absolute, .95)),
        "p99_abs": float(np.quantile(absolute, .99)), "max_abs": float(absolute.max()), "n": int(len(array)),
    }


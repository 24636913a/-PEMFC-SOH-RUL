"""Frozen AHC-TV alignment and pre-unblinding reader contracts.

This module deliberately contains no Test path registry and does not open any
real Test file.  A later manually authorized executor may import these
functions only after it has verified the frozen input authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
CONDITIONS = (
    "current_density",
    "anode_inlet_temperature_C",
    "anode_inlet_pressure_kPag",
    "anode_inlet_RH_pct",
    "anode_hydrogen_flow_nl_min",
    "anode_stoichiometry",
    "cathode_stoichiometry",
)
METADATA_COLUMNS = ("DateTime",)
TARGET_COLUMNS = (
    "j.S.act [A/cm2]",
    "U1 [V]", "U2 [V]", "U3 [V]", "U4 [V]", "U5 [V]",
    "U7 [V]", "U8 [V]", "U9 [V]", "U10 [V]", "U11 [V]",
)
TAINTED_N1_DIR = (REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_N1_Data_Foundation" / "outputs" / "soh_anchor_analysis").resolve()


class AlignmentContractError(RuntimeError):
    """A frozen alignment or unblinding requirement did not hold."""


class HistoricalTestDerivedInputRejected(AlignmentContractError):
    """A forbidden historical N1 Test-derived path was proposed as an input."""


class TargetUnblindingBlocked(AlignmentContractError):
    """Reader T prerequisites were not satisfied before any target read."""


@dataclass(frozen=True)
class AlignmentResult:
    status: str
    selected: pd.DataFrame
    support_count: int
    retention_median: float
    legal_interval_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(relative_path: str) -> Path:
    return (REPO / relative_path).resolve()


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def assert_runtime_input_allowed(path: Path) -> Path:
    """Reject historical N1 outputs that may embed late-life Test targets."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(TAINTED_N1_DIR):
        raise HistoricalTestDerivedInputRejected(
            "HISTORICAL_TEST_DERIVED_INPUT_REJECTED: " + str(resolved)
        )
    return resolved


def _sidecar_hash(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_SHA256_SIDECAR_MISSING")
    token = sidecar.read_text(encoding="utf-8").strip().split()
    if not token:
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_SHA256_SIDECAR_EMPTY")
    return token[0].lower()


def verify_prediction_artifact(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_ARTIFACT_MISSING")
    if _sidecar_hash(path) != sha256_file(path):
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_SHA256_MISMATCH")
    table = pd.read_csv(path)
    if len(table) != 2 or set(pd.to_numeric(table["checkpoint"], errors="coerce")) != {2000, 2300}:
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_ARTIFACT_NOT_TWO_CHECKPOINTS")


def read_characterization_metadata_only(
    path: Path,
    read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    """Reader M: request only DateTime and return no target-capable columns."""
    data = read_csv(assert_runtime_input_allowed(Path(path)), sep=";", encoding="utf-8-sig", usecols=list(METADATA_COLUMNS))
    if tuple(data.columns) != METADATA_COLUMNS:
        raise AlignmentContractError("METADATA_READER_COLUMN_ISOLATION_FAILED")
    parsed = pd.to_datetime(data["DateTime"], errors="coerce")
    if parsed.notna().sum() < 1:
        raise AlignmentContractError("TEST_CHARACTERIZATION_TIME_UNRESOLVED")
    return pd.DataFrame({"DateTime": parsed.dropna().reset_index(drop=True)})


def characterization_start_from_metadata(metadata: pd.DataFrame) -> pd.Timestamp:
    if tuple(metadata.columns) != METADATA_COLUMNS:
        raise AlignmentContractError("METADATA_READER_COLUMN_ISOLATION_FAILED")
    values = pd.to_datetime(metadata["DateTime"], errors="coerce").dropna()
    if values.empty:
        raise AlignmentContractError("TEST_CHARACTERIZATION_TIME_UNRESOLVED")
    return pd.Timestamp(values.min())


def _authority_hashes_pass(phase: Path) -> bool:
    authority = read_yaml(phase / "configs" / "AHC_TV_FROZEN_INPUT_AUTHORITY.yaml")
    return all(
        repo_path(item["path"]).exists() and sha256_file(repo_path(item["path"])) == item["sha256"]
        for item in authority["frozen_hashes"]
    )


def verify_reader_t_prerequisites(
    frozen_prediction_path: Path,
    phase: Path = PHASE,
) -> None:
    """Fail closed before Reader T can open any target-capable source column."""
    phase = Path(phase)
    verify_prediction_artifact(frozen_prediction_path)
    audit_path = phase / "outputs" / "AHC_TV_PREUNBLINDING_CONTRACT_AUDIT.json"
    if not audit_path.exists():
        raise TargetUnblindingBlocked("PREUNBLINDING_CONTRACT_AUDIT_MISSING")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    static_ok = (
        audit.get("final_status") == "AHC_TV_EXECUTION_CONTRACT_FROZEN_AWAITING_MANUAL_TEST_OPENING_DECISION"
        and audit.get("original_protocol_sha256_pass") is True
        and audit.get("execution_contract_sha256_pass") is True
        and audit.get("Development_alignment_replay") == "6/6 PASS"
        and all(audit.get("frozen_alignment_input_hashes", {}).values())
        and all(audit.get("frozen_input_authority_hashes", {}).values())
        and _authority_hashes_pass(phase)
    )
    if not static_ok:
        raise TargetUnblindingBlocked("PREUNBLINDING_CONTRACT_PREREQUISITE_FAILED")
    prediction = pd.read_csv(frozen_prediction_path)
    required = ("checkpoint", "joint_support_count", "selected_interval_1", "selected_interval_2", "selected_interval_3")
    if any(name not in prediction.columns for name in required):
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_ALIGNMENT_EVIDENCE_MISSING")
    legal = (
        len(prediction) == 2
        and set(pd.to_numeric(prediction["checkpoint"], errors="coerce")) == {2000, 2300}
        and pd.to_numeric(prediction["joint_support_count"], errors="coerce").ge(2).all()
        and prediction[["selected_interval_1", "selected_interval_2", "selected_interval_3"]].notna().all(axis=None)
    )
    if not legal:
        raise TargetUnblindingBlocked("BOTH_TEST_ALIGNMENTS_NOT_LEGAL")


def read_targets_after_prediction_freeze(
    polarization_path: Path,
    frozen_prediction_path: Path,
    read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    """Reader T; checks the prediction artifact before opening target columns."""
    verify_reader_t_prerequisites(frozen_prediction_path)
    return read_csv(assert_runtime_input_allowed(Path(polarization_path)), sep=";", encoding="utf-8-sig", usecols=list(TARGET_COLUMNS))


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def _require_columns(data: pd.DataFrame, required: Iterable[str]) -> None:
    absent = [name for name in required if name not in data.columns]
    if absent:
        raise AlignmentContractError("ALIGNMENT_INPUT_COLUMNS_MISSING: " + ",".join(absent))


def align_latest_three_intervals(
    intervals: pd.DataFrame,
    characterization_start: pd.Timestamp,
    raw_cutoff_ophrs2_h: float,
    support_bounds: dict[str, tuple[float, float] | list[float]],
) -> AlignmentResult:
    """Apply the frozen legal-first, select-first, support-second semantics."""
    required = [
        "cycle_interval_id", "interval_kind", "measurement_end_datetime",
        "measurement_end_ophrs2_h", "P1S_operating_voltage_retention",
        "has_complete_middle50_dwell", "cycle_start_anchor_present", *CONDITIONS,
    ]
    _require_columns(intervals, required)
    cutoff_time = pd.Timestamp(characterization_start)
    if pd.isna(cutoff_time) or not np.isfinite(float(raw_cutoff_ophrs2_h)):
        raise AlignmentContractError("TEST_CHARACTERIZATION_TIME_UNRESOLVED")
    data = intervals.copy()
    data["_end_time"] = pd.to_datetime(data["measurement_end_datetime"], errors="coerce")
    data["_end_age"] = pd.to_numeric(data["measurement_end_ophrs2_h"], errors="coerce")
    data["_retention"] = pd.to_numeric(data["P1S_operating_voltage_retention"], errors="coerce")
    partial = data["interval_kind"].eq("PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL")
    legal = (
        _as_bool(data["has_complete_middle50_dwell"])
        & (~partial | _as_bool(data["cycle_start_anchor_present"]))
        & data["_end_time"].notna()
        & (data["_end_time"] < cutoff_time)
        & data["_end_age"].between(float(raw_cutoff_ophrs2_h) - 48.0, float(raw_cutoff_ophrs2_h), inclusive="both")
        & data["_retention"].notna()
        & data[list(CONDITIONS)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    )
    legal_data = data.loc[legal].copy()
    legal_data = legal_data.sort_values(
        ["_end_time", "_end_age", "cycle_interval_id"],
        ascending=[False, False, True], kind="mergesort",
    ).drop_duplicates("cycle_interval_id", keep="first")
    selected = legal_data.head(3).copy()
    if len(selected) != 3:
        return AlignmentResult("TEST_ALIGNMENT_INCONCLUSIVE", selected, 0, float("nan"), len(legal_data))
    in_support = pd.Series(True, index=selected.index)
    for variable in CONDITIONS:
        if variable not in support_bounds:
            raise AlignmentContractError("FROZEN_SUPPORT_DIMENSION_MISSING: " + variable)
        lower, upper = (float(value) for value in support_bounds[variable])
        value = pd.to_numeric(selected[variable], errors="coerce")
        in_support &= value.between(lower, upper, inclusive="both")
    selected["in_frozen_joint_common_support"] = in_support.astype(bool)
    support_count = int(in_support.sum())
    status = "ALIGNED" if support_count >= 2 else "TEST_ALIGNMENT_INCONCLUSIVE"
    return AlignmentResult(status, selected, support_count, float(selected["_retention"].median()), len(legal_data))


def _intervals_for_development_replay(phase: Path) -> pd.DataFrame:
    frozen = read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    replay = frozen["development_replay_inputs"]
    complete = pd.read_csv(repo_path(replay["P1S_complete_cycle_measurements_path"]))
    partial = pd.read_csv(repo_path(replay["preanchor_partial_intervals_path"]))
    data = pd.concat([complete, partial], ignore_index=True, sort=False)
    data = data.loc[data["stable_window"].eq("middle_50")].copy()
    data["has_complete_middle50_dwell"] = (
        pd.to_numeric(data["eligible_dwell_count"], errors="coerce").gt(0)
        & pd.to_numeric(data["valid_dwell_count"], errors="coerce").gt(0)
        & pd.to_numeric(data["P1S_operating_voltage_retention"], errors="coerce").notna()
    )
    data["cycle_start_anchor_present"] = True
    return data


def replay_development_alignment(phase: Path = PHASE, tolerance: float = 1e-12) -> dict:
    """Replay all six Development anchors from frozen PAA interval artifacts only."""
    phase = Path(phase)
    frozen = read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    paa = read_yaml(repo_path(frozen["PAA_protocol_path"]))
    alignment = pd.read_csv(repo_path(frozen["development_replay_inputs"]["P1S_anchor_alignment_path"]))
    data = _intervals_for_development_replay(phase)
    bounds = frozen["frozen_development_joint_common_support"]
    summaries = alignment.loc[alignment["record_type"].eq("anchor_summary")].copy()
    results: list[dict] = []
    anchors = paa["partial_interval_characterization_metadata"]["anchors"]
    for meta in anchors:
        checkpoint = int(meta["checkpoint_h"])
        expected = summaries.loc[pd.to_numeric(summaries["checkpoint_h"], errors="coerce").eq(checkpoint)]
        if len(expected) != 1:
            raise AlignmentContractError(f"DEVELOPMENT_REPLAY_EXPECTED_ANCHOR_MISSING_{checkpoint}")
        result = align_latest_three_intervals(
            data,
            pd.Timestamp(meta["pre_characterization_cutoff_datetime"]),
            float(meta["pre_characterization_cutoff_ophrs2_h"]),
            bounds,
        )
        expected_ids = str(expected.iloc[0]["selected_cycle_interval_ids"]).split("|")
        observed_ids = result.selected["cycle_interval_id"].astype(str).tolist()
        expected_retention = float(expected.iloc[0]["P1S_anchor_retention"])
        ids_pass = observed_ids == expected_ids
        retention_pass = bool(np.isfinite(result.retention_median) and abs(result.retention_median - expected_retention) <= tolerance)
        support_pass = result.support_count == int(float(expected.iloc[0]["joint_support_interval_count"]))
        results.append({
            "checkpoint": checkpoint,
            "expected_interval_ids": "|".join(expected_ids),
            "replayed_interval_ids": "|".join(observed_ids),
            "expected_retention": expected_retention,
            "replayed_retention": result.retention_median,
            "expected_support_count": int(float(expected.iloc[0]["joint_support_interval_count"])),
            "replayed_support_count": result.support_count,
            "interval_ids_pass": ids_pass,
            "retention_pass_at_1e12": retention_pass,
            "support_count_pass": support_pass,
            "status": result.status,
            "pass": bool(ids_pass and retention_pass and support_pass and result.status == "ALIGNED"),
        })
    passed = sum(int(row["pass"]) for row in results)
    return {"anchors": results, "pass_count": passed, "required": 6, "all_pass": passed == 6}


def verify_frozen_alignment_inputs(phase: Path = PHASE) -> dict[str, bool]:
    """Verify every source referenced by the alignment-input freeze."""
    phase = Path(phase)
    frozen = read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    checks = {
        frozen["PAA_protocol_path"]: frozen["PAA_protocol_sha256"],
        frozen["P1S_reference_file_path"]: frozen["P1S_reference_file_sha256"],
        frozen["P1S_condition_support_file_path"]: frozen["P1S_condition_support_file_sha256"],
        frozen["development_replay_inputs"]["P1S_anchor_alignment_path"]: frozen["development_replay_inputs"]["P1S_anchor_alignment_sha256"],
        frozen["development_replay_inputs"]["P1S_complete_cycle_measurements_path"]: frozen["development_replay_inputs"]["P1S_complete_cycle_measurements_sha256"],
        frozen["development_replay_inputs"]["preanchor_partial_intervals_path"]: frozen["development_replay_inputs"]["preanchor_partial_intervals_sha256"],
        frozen["P1S_alignment_builder_path"]: frozen["P1S_alignment_builder_sha256"],
        frozen["polarization_target_normalization"]["source_artifact_path"]: frozen["polarization_target_normalization"]["source_artifact_sha256"],
    }
    return {path: sha256_file(repo_path(path)) == expected for path, expected in checks.items()}

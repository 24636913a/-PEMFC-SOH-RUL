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
REQUIRED_TEST_AUTHORIZATION = "OPEN_AHC_TV_TEST"
FROZEN_PREDICTION_SCHEMA = PHASE / "schemas" / "FROZEN_TEST_PREDICTIONS_SCHEMA.yaml"
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


def write_sha256_sidecar(path: Path) -> str:
    """Freeze one runtime artifact with its adjacent SHA256 sidecar."""
    path = Path(path)
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def require_manual_test_authorization(token: str) -> None:
    """Require an exact, user-supplied marker before any future Test reader runs."""
    if token != REQUIRED_TEST_AUTHORIZATION:
        raise TargetUnblindingBlocked("MANUAL_TEST_AUTHORIZATION_REQUIRED")


def verify_prediction_artifact(path: Path) -> None:
    """Legacy immutability-only check; it is insufficient for Reader T."""
    path = Path(path)
    if not path.exists():
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_ARTIFACT_MISSING")
    if _sidecar_hash(path) != sha256_file(path):
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_SHA256_MISMATCH")
    table = pd.read_csv(path)
    if len(table) != 2 or set(pd.to_numeric(table["checkpoint"], errors="coerce")) != {2000, 2300}:
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_ARTIFACT_NOT_TWO_CHECKPOINTS")


def _strict_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in {"true", "false", "True", "False"}:
        return value.lower() == "true"
    raise TargetUnblindingBlocked("FROZEN_PREDICTION_SUPPORT_EVIDENCE_INVALID")


def _finite_numeric(table: pd.DataFrame, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    converted: dict[str, np.ndarray] = {}
    for name in names:
        values = pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise TargetUnblindingBlocked("FROZEN_PREDICTION_FINITE_VALUES_REQUIRED")
        converted[name] = values
    return converted


def verify_frozen_prediction_artifact_strict(
    path: Path,
    *,
    phase: Path = PHASE,
) -> pd.DataFrame:
    """Validate an immutable, complete, target-free prediction artifact.

    The schema file is the sole authority for columns and their order. This
    function performs no Test-source read; it inspects only the supplied
    prediction artifact and frozen Development-model parameters.
    """
    phase = Path(phase)
    if not _authority_hashes_pass(phase):
        raise TargetUnblindingBlocked("PREUNBLINDING_CONTRACT_PREREQUISITE_FAILED")
    path = Path(path)
    if not path.exists():
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_ARTIFACT_MISSING")
    if _sidecar_hash(path) != sha256_file(path):
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_SHA256_MISMATCH")
    schema = read_yaml(phase / "schemas" / "FROZEN_TEST_PREDICTIONS_SCHEMA.yaml")
    required_columns = list(schema["required_columns"])
    table = pd.read_csv(path)
    normalized_names = {str(name).strip().lower() for name in table.columns}
    target_like = {
        "target", "y_test", "polarization_target", "actual_health",
        "voltage_anchor", "error", "absolute_error",
    }
    if normalized_names & target_like:
        raise TargetUnblindingBlocked("TARGET_LIKE_COLUMN_IN_PREUNBLINDING_ARTIFACT")
    if list(table.columns) != required_columns:
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_SCHEMA_INVALID")
    row_spec = schema["required_rows"]
    checkpoints = row_spec["checkpoints"]
    if len(table) != int(row_spec["row_count"]) or table["checkpoint"].tolist() != checkpoints:
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_SCHEMA_INVALID")
    numeric = _finite_numeric(
        table,
        (
            "raw_cutoff_ophrs2_h", "R1", "R2", "R3", "P1S_R_median",
            "P1S_affine_prediction", "nominal_age_prediction", "raw_P1S_prediction",
        ),
    )
    intervals = ("selected_interval_1", "selected_interval_2", "selected_interval_3")
    for row in table.itertuples(index=False):
        ids = [getattr(row, name) for name in intervals]
        if any(pd.isna(value) or not str(value).strip() for value in ids) or len(set(map(str, ids))) != 3:
            raise TargetUnblindingBlocked("FROZEN_PREDICTION_INTERVAL_IDS_INVALID")
    support_columns = ("joint_support_1", "joint_support_2", "joint_support_3")
    parsed_support = table.loc[:, support_columns].map(_strict_boolean)
    support_count = pd.to_numeric(table["joint_support_count"], errors="coerce")
    if (
        support_count.isna().any()
        or (~np.isfinite(support_count.to_numpy(dtype=float))).any()
        or (~support_count.isin((2, 3))).any()
        or (~support_count.eq(parsed_support.sum(axis=1))).any()
    ):
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_SUPPORT_EVIDENCE_INVALID")
    datetime_columns = (
        "actual_characterization_start", "raw_cutoff_datetime",
        "measurement_end_datetime_1", "measurement_end_datetime_2", "measurement_end_datetime_3",
    )
    parsed_time = {name: pd.to_datetime(table[name], errors="coerce") for name in datetime_columns}
    if any(values.isna().any() for values in parsed_time.values()):
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_TEMPORAL_EVIDENCE_INVALID")
    start = parsed_time["actual_characterization_start"]
    cutoff = parsed_time["raw_cutoff_datetime"]
    measurement_ends = [parsed_time[name] for name in datetime_columns[2:]]
    if not ((cutoff < start).all() and all((end < start).all() and (end <= cutoff).all() for end in measurement_ends)):
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_TEMPORAL_EVIDENCE_INVALID")
    r_calc = np.median(np.column_stack((numeric["R1"], numeric["R2"], numeric["R3"])), axis=1)
    if not np.all(np.abs(r_calc - numeric["P1S_R_median"]) <= 1e-12):
        raise TargetUnblindingBlocked("FROZEN_PREDICTION_R_MEDIAN_MISMATCH")
    if not np.all(np.abs(numeric["raw_P1S_prediction"] - numeric["P1S_R_median"]) <= 1e-12):
        raise TargetUnblindingBlocked("FROZEN_RAW_P1S_PREDICTION_MISMATCH")
    models = read_yaml(phase / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml")
    affine = models["P1S_affine"]
    expected_p1s = float(affine["slope"]) * numeric["P1S_R_median"] + float(affine["intercept"])
    if not np.all(np.abs(numeric["P1S_affine_prediction"] - expected_p1s) <= 1e-12):
        raise TargetUnblindingBlocked("FROZEN_P1S_PREDICTION_COEFFICIENT_MISMATCH")
    age = models["nominal_age_affine"]
    expected_age = float(age["intercept_alpha"]) + float(age["slope_beta"]) * np.asarray(checkpoints, dtype=float)
    if not np.all(np.abs(numeric["nominal_age_prediction"] - expected_age) <= 1e-12):
        raise TargetUnblindingBlocked("FROZEN_AGE_PREDICTION_COEFFICIENT_MISMATCH")
    return table


def read_characterization_metadata_only(
    path: Path,
    *,
    authorization_token: str,
    read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    """Reader M: request only DateTime and return no target-capable columns."""
    require_manual_test_authorization(authorization_token)
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


def verify_frozen_runtime_authority(phase: Path = PHASE) -> None:
    """Fail closed unless every authority-listed runtime input is immutable."""
    if not _authority_hashes_pass(Path(phase)):
        raise TargetUnblindingBlocked("PREUNBLINDING_CONTRACT_PREREQUISITE_FAILED")


def _verify_stage_p_runtime_evidence(
    frozen_prediction_path: Path,
    artifact: pd.DataFrame,
) -> None:
    """Require Stage P provenance and its completion marker before Reader T.

    This routine deliberately reads only Stage-P-created, target-free runtime
    artifacts.  It is invoked before a Reader T call is permitted to open a
    polarization target column.
    """
    runtime_dir = Path(frozen_prediction_path).parent
    provenance_path = runtime_dir / "FROZEN_TEST_PREDICTION_PROVENANCE.json"
    completion_path = runtime_dir / "STAGE_P_COMPLETION_MARKER.json"
    for path, code in (
        (provenance_path, "STAGE_P_PROVENANCE_MISSING"),
        (completion_path, "STAGE_P_COMPLETION_MARKER_MISSING"),
    ):
        if not path.exists():
            raise TargetUnblindingBlocked(code)
        if _sidecar_hash(path) != sha256_file(path):
            raise TargetUnblindingBlocked(code.replace("MISSING", "SHA256_MISMATCH"))
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise TargetUnblindingBlocked("STAGE_P_RUNTIME_EVIDENCE_INVALID") from error
    expected_sha = sha256_file(Path(frozen_prediction_path))
    if provenance.get("prediction_sha256") != expected_sha:
        raise TargetUnblindingBlocked("STAGE_P_PROVENANCE_PREDICTION_MISMATCH")
    records = provenance.get("checkpoints")
    if not isinstance(records, list) or [record.get("checkpoint") for record in records] != [2000, 2300]:
        raise TargetUnblindingBlocked("STAGE_P_PROVENANCE_CHECKPOINTS_INVALID")
    for row, record in zip(artifact.itertuples(index=False), records):
        if list(record.get("selected_interval_ids", [])) != [
            str(row.selected_interval_1), str(row.selected_interval_2), str(row.selected_interval_3)
        ]:
            raise TargetUnblindingBlocked("STAGE_P_PROVENANCE_INTERVAL_MISMATCH")
    required_completion = {
        "status": "STAGE_P_FROZEN_COMPLETE",
        "authorization_marker_present": True,
        "both_checkpoints_built": True,
        "both_alignments_legal": True,
        "prediction_sha256_frozen": True,
        "provenance_sha256_frozen": True,
        "strict_prediction_verification": "PASS",
        "target_columns_read_so_far": False,
    }
    if any(completion.get(name) != expected for name, expected in required_completion.items()):
        raise TargetUnblindingBlocked("STAGE_P_COMPLETION_MARKER_INVALID")
    if completion.get("prediction_sha256") != expected_sha:
        raise TargetUnblindingBlocked("STAGE_P_COMPLETION_PREDICTION_MISMATCH")
    if completion.get("provenance_sha256") != sha256_file(provenance_path):
        raise TargetUnblindingBlocked("STAGE_P_COMPLETION_PROVENANCE_MISMATCH")


def authorize_and_verify_target_unblinding(
    frozen_prediction_path: Path,
    *,
    authorization_token: str,
    phase: Path = PHASE,
) -> pd.DataFrame:
    """Single target gate: authorization, authority, replay, and strict artifact."""
    require_manual_test_authorization(authorization_token)
    phase = Path(phase)
    verify_frozen_runtime_authority(phase)
    artifact = verify_frozen_prediction_artifact_strict(frozen_prediction_path, phase=phase)
    _verify_stage_p_runtime_evidence(frozen_prediction_path, artifact)
    replay = replay_development_alignment(phase)
    if not replay["all_pass"]:
        raise TargetUnblindingBlocked("DEVELOPMENT_ALIGNMENT_REPLAY_FAILED")
    return artifact


def verify_reader_t_prerequisites(
    frozen_prediction_path: Path,
    *,
    authorization_token: str,
    phase: Path = PHASE,
) -> pd.DataFrame:
    """Compatibility wrapper; Reader T still reaches the single hardened gate."""
    return authorize_and_verify_target_unblinding(
        frozen_prediction_path,
        authorization_token=authorization_token,
        phase=phase,
    )


def read_targets_after_prediction_freeze(
    polarization_path: Path,
    frozen_prediction_path: Path,
    *,
    authorization_token: str,
    read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    """Reader T; checks the prediction artifact before opening target columns."""
    authorize_and_verify_target_unblinding(
        frozen_prediction_path,
        authorization_token=authorization_token,
    )
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

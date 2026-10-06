"""Formal Stage P builder for the future AHC-TV one-shot execution.

Stage P is deliberately target-blind.  It resolves characterization starts
with Reader M, derives P1S intervals directly from authorized raw operating
rows, freezes the two prediction rows, and records enough target-free
provenance for Reader T to verify the freeze later.  This module is not run on
the real Test data during the protocol-freeze phase.
"""
from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from ahc_tv_alignment import (
    CONDITIONS,
    PHASE,
    AlignmentContractError,
    TargetUnblindingBlocked,
    align_latest_three_intervals,
    assert_runtime_input_allowed,
    characterization_start_from_metadata,
    read_characterization_metadata_only,
    read_yaml,
    require_manual_test_authorization,
    sha256_file,
    verify_frozen_prediction_artifact_strict,
    verify_frozen_runtime_authority,
)
from ahc_tv_raw_alignment_core import (
    ACTIVE_CELLS,
    RAW_COLUMNS,
    RawAlignmentCoreError,
    nanmean_active_voltage as _nanmean_active_voltage,
    normalize_operation,
    normalize_raw_operating_frame,
    runtime_p1_intervals,
)
from ahc_tv_one_shot_ledger import write_exclusive_json, write_exclusive_sha256_sidecar, write_exclusive_text


PREDICTION_NAME = "FROZEN_TEST_PREDICTIONS_BEFORE_TARGET_UNBLINDING.csv"
PROVENANCE_NAME = "FROZEN_TEST_PREDICTION_PROVENANCE.json"
COMPLETION_NAME = "STAGE_P_COMPLETION_MARKER.json"


class RuntimeSourceBindingError(AlignmentContractError):
    """A future Test source could not be uniquely bound without inspecting it."""


class StagePError(AlignmentContractError):
    """Target-free Stage P construction did not satisfy frozen requirements."""


def _log(events: list[str] | None, event: str) -> None:
    if events is not None:
        events.append(event)


def _registry(phase: Path) -> dict:
    return read_yaml(Path(phase) / "configs" / "AHC_TV_RUNTIME_SOURCE_REGISTRY.yaml")


def _forbidden_source_name(path: Path, registry: dict) -> bool:
    lowered = Path(path).name.lower()
    return any(token.lower() in lowered for token in registry["forbidden_source_name_tokens"])


def _assert_source_path_allowed(
    path: Path,
    registry: dict,
    *,
    synthetic_test_mode: bool = False,
) -> Path:
    """Validate an identity-only candidate without opening its contents.

    The real Test runtime is constrained to the frozen authoritative root.
    Synthetic tests explicitly opt out so they can exercise the production
    filename-binding functions using temporary paths.
    """
    resolved = assert_runtime_input_allowed(Path(path))
    if not synthetic_test_mode:
        root = Path(registry["local_dataset_root"]["path"]).resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
    return resolved


def _classify_source_candidate(
    path: Path,
    registry: dict,
    *,
    synthetic_test_mode: bool = False,
) -> tuple[Path, bool]:
    """Return a validated path and whether filename policy excludes it.

    Exclusion is deliberately distinct from rejection.  A forbidden candidate
    in an auto-discovered set is ignored; a forbidden sole mapping leaves zero
    nominal matches and therefore still fails closed.
    """
    resolved = _assert_source_path_allowed(
        Path(path), registry, synthetic_test_mode=synthetic_test_mode
    )
    return resolved, _forbidden_source_name(resolved, registry)


def _checkpoint_filename_matches(path: Path, checkpoint: int, registry: dict) -> bool:
    pattern = registry["polarization_checkpoint_sources"][str(int(checkpoint))]["filename_regex"]
    return re.search(pattern, Path(path).name) is not None


def bind_unique_polarization_source(
    checkpoint: int,
    candidates: Iterable[Path],
    *,
    phase: Path = PHASE,
    synthetic_test_mode: bool = False,
) -> Path:
    """Bind a checkpoint by filename identity only; never open source bytes.

    All proposed candidates are classified first.  Forbidden filenames are
    excluded before nominal matching, so an after-Air-Air file cannot make an
    otherwise unique nominal candidate fail.  A sole forbidden or wrong file
    yields zero nominal matches and is rejected below.
    """
    registry = _registry(phase)
    classified = [
        _classify_source_candidate(
            Path(candidate), registry, synthetic_test_mode=synthetic_test_mode
        )
        for candidate in candidates
    ]
    matches = [
        candidate for candidate, excluded in classified
        if not excluded and _checkpoint_filename_matches(candidate, checkpoint, registry)
    ]
    if len(matches) != 1:
        raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
    return matches[0]


def bind_both_polarization_sources(
    sources: Mapping[int, Path] | Iterable[Path], *, phase: Path = PHASE,
    synthetic_test_mode: bool = False,
) -> dict[int, Path]:
    """Return exactly one filename-bound 2000 h and 2300 h source.

    Mappings are checked against their declared checkpoint as well, which
    rejects accidental swapped 2000/2300 paths before any target read.
    """
    if isinstance(sources, Mapping):
        if set(sources) != {2000, 2300}:
            raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
        return {
            checkpoint: bind_unique_polarization_source(
                checkpoint, [Path(sources[checkpoint])], phase=phase,
                synthetic_test_mode=synthetic_test_mode,
            )
            for checkpoint in (2000, 2300)
        }
    candidate_list = [Path(path) for path in sources]
    return {
        checkpoint: bind_unique_polarization_source(
            checkpoint, candidate_list, phase=phase,
            synthetic_test_mode=synthetic_test_mode,
        )
        for checkpoint in (2000, 2300)
    }


def validate_raw_sources(
    raw_sources: Iterable[Path], *, phase: Path = PHASE, synthetic_test_mode: bool = False,
) -> list[Path]:
    """Validate only source identities; raw contents remain unopened here."""
    registry = _registry(phase)
    paths = [
        _assert_source_path_allowed(Path(path), registry, synthetic_test_mode=synthetic_test_mode)
        for path in raw_sources
    ]
    if not paths or any(
        _forbidden_source_name(path, registry)
        or re.fullmatch(registry["raw_operating_source"]["filename_regex"], path.name, re.I) is None
        for path in paths
    ):
        raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
    if len({str(path) for path in paths}) != len(paths):
        raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
    return paths


def _write_json_with_sidecar(path: Path, value: dict) -> str:
    write_exclusive_json(path, value)
    return write_exclusive_sha256_sidecar(path)


def _write_csv_with_sidecar(path: Path, frame: pd.DataFrame, *, sidecar: bool = False) -> str | None:
    # ``to_csv(path)`` overwrites silently, which is prohibited for formal
    # one-shot artifacts.  Serialize first and exclusive-create the final path.
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    write_exclusive_text(path, buffer.getvalue())
    return write_exclusive_sha256_sidecar(path) if sidecar else None


def _normalize_operation(values: pd.Series) -> pd.Series:
    """Compatibility alias; normalization authority is the shared raw core."""
    return normalize_operation(values)


def _read_raw_operating(
    sources: list[Path], *, read_csv: Callable[..., pd.DataFrame] = pd.read_csv
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_index, source in enumerate(sources):
        if not source.exists():
            raise StagePError("TEST_RAW_SOURCE_MISSING")
        header = read_csv(source, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        absent = [name for name in RAW_COLUMNS if name not in header]
        if absent:
            raise StagePError("TEST_RAW_SCHEMA_MISSING_COLUMNS: " + ",".join(absent))
        frame = read_csv(source, sep=";", encoding="utf-8-sig", usecols=list(RAW_COLUMNS), low_memory=False)
        frames.append(normalize_raw_operating_frame(frame, source_index=source_index, source_basename=source.name))
    # The shared core deliberately preserves source operation-point text for
    # RCR identity and applies numeric normalization only during PAA lookup.
    return pd.concat(frames, ignore_index=True, sort=False)


def _runtime_p1_intervals(
    raw_before_start: pd.DataFrame,
    reference_voltage: float,
    *,
    partial_interval_id: str = "PARTIAL_PRE_RUNTIME",
    partial_last_run_id: int | None = None,
    require_development_stratum: bool = False,
    synthetic_test_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delegate all formal raw-to-interval work to the authoritative core."""
    try:
        return runtime_p1_intervals(
            raw_before_start,
            reference_voltage,
            partial_interval_id=partial_interval_id,
            partial_last_run_id=partial_last_run_id,
            require_development_stratum=require_development_stratum,
            synthetic_test_mode=synthetic_test_mode,
        )
    except RawAlignmentCoreError as exc:
        raise StagePError(f"TEST_{exc}") from exc


def _prediction_row(
    checkpoint: int,
    raw: pd.DataFrame,
    characterization_start: pd.Timestamp,
    alignment_inputs: dict,
    models: dict,
    *,
    synthetic_test_mode: bool = False,
    partial_last_run_id: int | None = None,
    require_development_stratum: bool = False,
    operating_age_cutoff_h: float | None = None,
) -> tuple[dict, dict]:
    before = raw.loc[raw["DateTime"] < characterization_start].copy()
    if before.empty:
        raise StagePError("TEST_ALIGNMENT_INCONCLUSIVE")
    intervals, runs = _runtime_p1_intervals(
        before,
        float(alignment_inputs["reference_voltage_V"]),
        partial_interval_id=f"PARTIAL_PRE_{int(checkpoint)}",
        partial_last_run_id=partial_last_run_id,
        require_development_stratum=require_development_stratum,
        synthetic_test_mode=synthetic_test_mode,
    )
    if intervals.empty:
        raise StagePError("TEST_ALIGNMENT_INCONCLUSIVE")
    raw_cutoff = runs.iloc[-1]
    result = align_latest_three_intervals(
        intervals,
        characterization_start,
        float(operating_age_cutoff_h) if operating_age_cutoff_h is not None else float(raw_cutoff.end_ophrs2_h),
        alignment_inputs["frozen_development_joint_common_support"],
    )
    if result.status != "ALIGNED" or len(result.selected) != 3:
        raise StagePError("TEST_ALIGNMENT_INCONCLUSIVE")
    selected = result.selected.reset_index(drop=True)
    r_values = selected["P1S_operating_voltage_retention"].to_numpy(dtype=float)
    r_median = float(np.median(r_values))
    affine = models["P1S_affine"]
    age = models["nominal_age_affine"]
    row = {
        "checkpoint": int(checkpoint),
        "actual_characterization_start": str(characterization_start),
        "raw_cutoff_datetime": str(pd.Timestamp(raw_cutoff.end_datetime)),
        "raw_cutoff_ophrs2_h": float(raw_cutoff.end_ophrs2_h),
        **{f"selected_interval_{index + 1}": str(selected.iloc[index].cycle_interval_id) for index in range(3)},
        **{f"measurement_end_datetime_{index + 1}": str(selected.iloc[index].measurement_end_datetime) for index in range(3)},
        **{f"joint_support_{index + 1}": bool(selected.iloc[index].in_frozen_joint_common_support) for index in range(3)},
        "joint_support_count": int(result.support_count),
        "R1": float(r_values[0]), "R2": float(r_values[1]), "R3": float(r_values[2]),
        "P1S_R_median": r_median,
        "P1S_affine_prediction": float(affine["slope"]) * r_median + float(affine["intercept"]),
        "nominal_age_prediction": float(age["intercept_alpha"]) + float(age["slope_beta"]) * float(checkpoint),
        "raw_P1S_prediction": r_median,
    }
    selected_records = []
    for item in selected.itertuples(index=False):
        selected_records.append({
            "interval_id": str(item.cycle_interval_id), "interval_kind": str(item.interval_kind),
            "raw_cycle_id": int(item.raw_cycle_id), "measurement_end_datetime": str(item.measurement_end_datetime),
            "measurement_end_ophrs2_h": float(item.measurement_end_ophrs2_h),
            "P1S_voltage_V": float(item.P1S_voltage_V),
            "P1S_operating_voltage_retention": float(item.P1S_operating_voltage_retention),
            "eligible_dwell_count": int(item.eligible_dwell_count),
            "raw_contributing_count": int(item.raw_contributing_count),
            "in_frozen_joint_common_support": bool(item.in_frozen_joint_common_support),
            **{condition: float(getattr(item, condition)) for condition in CONDITIONS},
        })
    provenance = {
        "checkpoint": int(checkpoint),
        "characterization_start": str(characterization_start),
        "raw_rows_total": int(len(raw)),
        "eligible_pre_characterization_raw_rows": int(len(before)),
        "latest_pre_characterization_datetime": str(before["DateTime"].max()),
        "raw_cutoff_datetime": str(pd.Timestamp(raw_cutoff.end_datetime)),
        "raw_cutoff_ophrs2_h": float(raw_cutoff.end_ophrs2_h),
        "selected_interval_ids": [record["interval_id"] for record in selected_records],
        "selected_intervals": selected_records,
    }
    return row, provenance


def build_and_freeze_stage_p(
    *,
    raw_sources: Iterable[Path],
    polarization_sources: Mapping[int, Path] | Iterable[Path],
    runtime_dir: Path,
    authorization_token: str,
    phase: Path = PHASE,
    event_log: list[str] | None = None,
    raw_read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
    metadata_read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
    synthetic_test_mode: bool = False,
) -> dict:
    """Execute both target-free Test checkpoints and freeze one 21-column file."""
    require_manual_test_authorization(authorization_token)
    _log(event_log, "AUTHORIZATION_OK")
    phase = Path(phase)
    verify_frozen_runtime_authority(phase)
    raw_paths = validate_raw_sources(raw_sources, phase=phase, synthetic_test_mode=synthetic_test_mode)
    polar_paths = bind_both_polarization_sources(
        polarization_sources, phase=phase, synthetic_test_mode=synthetic_test_mode
    )
    runtime_dir = Path(runtime_dir)
    # The canonical formal directory is created exactly once by this Stage-P
    # builder, after the phase-global ledger has already consumed the opening.
    runtime_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[int, pd.DataFrame] = {}
    starts: dict[int, pd.Timestamp] = {}
    for checkpoint in (2000, 2300):
        metadata[checkpoint] = read_characterization_metadata_only(
            polar_paths[checkpoint], authorization_token=authorization_token, read_csv=metadata_read_csv
        )
        starts[checkpoint] = characterization_start_from_metadata(metadata[checkpoint])
        _log(event_log, f"METADATA_{checkpoint}_READ")
    if not starts[2000] < starts[2300]:
        raise StagePError("TEST_CHARACTERIZATION_TIME_ORDER_INVALID")
    _write_csv_with_sidecar(runtime_dir / "TEST_CHARACTERIZATION_TIME_METADATA.csv", pd.DataFrame([
        {"checkpoint": checkpoint, "source_basename": polar_paths[checkpoint].name,
         "valid_datetime_count": int(len(metadata[checkpoint])),
         "characterization_start_datetime": str(starts[checkpoint]),
         "metadata_columns_read": "DateTime"}
        for checkpoint in (2000, 2300)
    ]))

    _log(event_log, "RAW_STAGE_P_READ")
    raw = _read_raw_operating(raw_paths, read_csv=raw_read_csv)
    alignment_inputs = read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    models = read_yaml(phase / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml")
    rows: list[dict] = []
    provenance_records: list[dict] = []
    for checkpoint in (2000, 2300):
        row, record = _prediction_row(
            checkpoint, raw, starts[checkpoint], alignment_inputs, models,
            synthetic_test_mode=synthetic_test_mode,
        )
        record["metadata_source_basename"] = polar_paths[checkpoint].name
        record["raw_source_basenames"] = [path.name for path in raw_paths]
        rows.append(row); provenance_records.append(record)
    schema = read_yaml(phase / "schemas" / "FROZEN_TEST_PREDICTIONS_SCHEMA.yaml")
    prediction = runtime_dir / PREDICTION_NAME
    _write_csv_with_sidecar(prediction, pd.DataFrame(rows).loc[:, schema["required_columns"]])
    _log(event_log, "PREDICTION_WRITTEN")
    prediction_sha = write_exclusive_sha256_sidecar(prediction)
    _log(event_log, "PREDICTION_HASH_FROZEN")

    provenance = {
        "artifact": prediction.name, "prediction_sha256": prediction_sha,
        "target_columns_read": False, "checkpoints": provenance_records,
        "frozen_reference_voltage_V": float(alignment_inputs["reference_voltage_V"]),
        "reference_source_sha256": alignment_inputs["P1S_reference_file_sha256"],
        "support_source_sha256": alignment_inputs["P1S_condition_support_file_sha256"],
        "alignment_implementation_sha256": alignment_inputs["P1S_alignment_builder_sha256"],
        "raw_sources": [{"basename": path.name, "sha256": sha256_file(path)} for path in raw_paths],
    }
    provenance_path = runtime_dir / PROVENANCE_NAME
    provenance_sha = _write_json_with_sidecar(provenance_path, provenance)
    _log(event_log, "PROVENANCE_WRITTEN")
    _log(event_log, "PROVENANCE_HASH_FROZEN")
    verify_frozen_prediction_artifact_strict(prediction, phase=phase)
    _log(event_log, "STRICT_VERIFY_PASS")
    completion = {
        "status": "STAGE_P_FROZEN_COMPLETE", "authorization_marker_present": True,
        "both_checkpoints_built": True, "both_alignments_legal": True,
        "prediction_sha256_frozen": True, "provenance_sha256_frozen": True,
        "strict_prediction_verification": "PASS", "target_columns_read_so_far": False,
        "prediction_sha256": prediction_sha, "provenance_sha256": provenance_sha,
    }
    completion_path = runtime_dir / COMPLETION_NAME
    _write_json_with_sidecar(completion_path, completion)
    _log(event_log, "STAGE_P_COMPLETION_FROZEN")
    return {
        "prediction_path": prediction, "provenance_path": provenance_path,
        "completion_path": completion_path, "metadata_path": runtime_dir / "TEST_CHARACTERIZATION_TIME_METADATA.csv",
    }

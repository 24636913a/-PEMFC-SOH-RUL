"""Formal Stage P builder for the future AHC-TV one-shot execution.

Stage P is deliberately target-blind.  It resolves characterization starts
with Reader M, derives P1S intervals directly from authorized raw operating
rows, freezes the two prediction rows, and records enough target-free
provenance for Reader T to verify the freeze later.  This module is not run on
the real Test data during the protocol-freeze phase.
"""
from __future__ import annotations

import json
import re
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
    write_sha256_sidecar,
)


ACTIVE_CELLS = (
    "U1 [V]", "U2 [V]", "U3 [V]", "U4 [V]", "U5 [V]",
    "U7 [V]", "U8 [V]", "U9 [V]", "U10 [V]", "U11 [V]",
)
RAW_CONDITIONS = {
    "current_density": "j.S.act [A/cm2]",
    "anode_inlet_temperature_C": "T.Si.A [°C]",
    "anode_inlet_pressure_kPag": "p.Si.A [kPag]",
    "anode_inlet_RH_pct": "RH.Si.A  [%]",
    "anode_hydrogen_flow_nl_min": "F.Si.H.A [nl/min]",
    "anode_stoichiometry": "Stoic.S.A [-]",
    "cathode_stoichiometry": "Stoic.S.C [-]",
}
RAW_IDENTITY = (
    "DateTime", "OpHrs2 [h]", "operation_point", "I.S.set [A]",
    "I.S.act [A]", "j.S.act [A/cm2]",
)
RAW_COLUMNS = tuple(dict.fromkeys((*RAW_IDENTITY, *ACTIVE_CELLS, *RAW_CONDITIONS.values())))
P1_TOKEN = "221||221.200"
ANCHOR_PATTERN = ("15||14.000", "15||0.000")
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


def _assert_source_path_allowed(path: Path, registry: dict) -> Path:
    resolved = assert_runtime_input_allowed(Path(path))
    if _forbidden_source_name(resolved, registry):
        raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
    return resolved


def _checkpoint_filename_matches(path: Path, checkpoint: int, registry: dict) -> bool:
    pattern = registry["polarization_checkpoint_sources"][str(int(checkpoint))]["filename_regex"]
    return re.search(pattern, Path(path).name, flags=re.IGNORECASE) is not None


def bind_unique_polarization_source(
    checkpoint: int,
    candidates: Iterable[Path],
    *,
    phase: Path = PHASE,
) -> Path:
    """Bind a checkpoint by registry/filename only; do not open source bytes."""
    registry = _registry(phase)
    validated = [_assert_source_path_allowed(Path(candidate), registry) for candidate in candidates]
    matches = [
        candidate for candidate in validated
        if _checkpoint_filename_matches(candidate, checkpoint, registry)
        and not _forbidden_source_name(candidate, registry)
    ]
    if len(matches) != 1:
        raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
    return matches[0]


def bind_both_polarization_sources(
    sources: Mapping[int, Path] | Iterable[Path], *, phase: Path = PHASE
) -> dict[int, Path]:
    """Return exactly one filename-bound 2000 h and 2300 h source.

    Mappings are checked against their declared checkpoint as well, which
    rejects accidental swapped 2000/2300 paths before any target read.
    """
    if isinstance(sources, Mapping):
        if set(sources) != {2000, 2300}:
            raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
        return {
            checkpoint: bind_unique_polarization_source(checkpoint, [Path(sources[checkpoint])], phase=phase)
            for checkpoint in (2000, 2300)
        }
    candidate_list = [Path(path) for path in sources]
    return {
        checkpoint: bind_unique_polarization_source(checkpoint, candidate_list, phase=phase)
        for checkpoint in (2000, 2300)
    }


def validate_raw_sources(raw_sources: Iterable[Path], *, phase: Path = PHASE) -> list[Path]:
    """Validate only source identities; raw contents remain unopened here."""
    registry = _registry(phase)
    paths = [_assert_source_path_allowed(Path(path), registry) for path in raw_sources]
    if not paths or any(re.fullmatch(registry["raw_operating_source"]["filename_regex"], path.name, re.I) is None for path in paths):
        raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
    if len({str(path) for path in paths}) != len(paths):
        raise RuntimeSourceBindingError("TEST_SOURCE_BINDING_AMBIGUOUS")
    return paths


def _write_json_with_sidecar(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return write_sha256_sidecar(path)


def _normalize_operation(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = values.astype("string").str.strip()
    result.loc[numeric.notna()] = numeric.loc[numeric.notna()].map(lambda value: f"{float(value):g}")
    return result.astype(str)


def _read_raw_operating(
    sources: list[Path], *, read_csv: Callable[..., pd.DataFrame] = pd.read_csv
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in sources:
        if not source.exists():
            raise StagePError("TEST_RAW_SOURCE_MISSING")
        header = read_csv(source, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        absent = [name for name in RAW_COLUMNS if name not in header]
        if absent:
            raise StagePError("TEST_RAW_SCHEMA_MISSING_COLUMNS: " + ",".join(absent))
        frame = read_csv(source, sep=";", encoding="utf-8-sig", usecols=list(RAW_COLUMNS), low_memory=False).copy()
        frame["_source_basename"] = source.name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["DateTime"] = pd.to_datetime(raw["DateTime"], errors="coerce")
    raw["OpHrs2 [h]"] = pd.to_numeric(raw["OpHrs2 [h]"].astype("string").str.replace(",", "", regex=False), errors="coerce")
    raw["operation_point"] = _normalize_operation(raw["operation_point"])
    for name in ["I.S.set [A]", "I.S.act [A]", *ACTIVE_CELLS, *RAW_CONDITIONS.values()]:
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    raw = raw.loc[raw["DateTime"].notna() & raw["OpHrs2 [h]"].notna()].copy()
    return raw.sort_values(["DateTime", "OpHrs2 [h]", "_source_basename"], kind="mergesort").reset_index(drop=True)


def _state_runs(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data["_token"] = data["operation_point"].astype(str) + "||" + data["I.S.set [A]"].round(3).map(lambda value: f"{value:.3f}" if np.isfinite(value) else "nan")
    discontinuity = data["DateTime"].diff().dt.total_seconds().gt(2.0).fillna(True)
    data["_run"] = (data["_token"].ne(data["_token"].shift()) | discontinuity).cumsum()
    records: list[dict] = []
    for run_id, group in data.groupby("_run", sort=True):
        records.append({
            "run_id": int(run_id),
            "state_token": str(group["_token"].iloc[0]),
            "start_datetime": pd.Timestamp(group["DateTime"].iloc[0]),
            "end_datetime": pd.Timestamp(group["DateTime"].iloc[-1]),
            "start_ophrs2_h": float(group["OpHrs2 [h]"].iloc[0]),
            "end_ophrs2_h": float(group["OpHrs2 [h]"].iloc[-1]),
            "operation_point": str(group["operation_point"].iloc[0]),
            "median_set_current_A": float(group["I.S.set [A]"].median()),
            "duration_wall_s": float((group["DateTime"].iloc[-1] - group["DateTime"].iloc[0]).total_seconds()),
            "row_start": int(group.index.min()),
            "row_end": int(group.index.max()),
        })
    return pd.DataFrame(records)


def _runtime_p1_intervals(raw_before_start: pd.DataFrame, reference_voltage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive anchor-delimited P1S intervals and middle-50 dwell medians."""
    runs = _state_runs(raw_before_start)
    if runs.empty:
        raise StagePError("TEST_ALIGNMENT_INCONCLUSIVE")
    starts = [
        index for index in range(len(runs) - 1)
        if (runs.iloc[index].state_token, runs.iloc[index + 1].state_token) == ANCHOR_PATTERN
    ]
    records: list[dict] = []
    for position, start_idx in enumerate(starts):
        stop_idx = starts[position + 1] if position + 1 < len(starts) else len(runs)
        group = runs.iloc[start_idx:stop_idx]
        p1_runs = group.loc[(group["state_token"] == P1_TOKEN) & (group["duration_wall_s"] >= 120.0)]
        if p1_runs.empty:
            continue
        values: list[dict] = []
        for p1 in p1_runs.itertuples(index=False):
            dwell = raw_before_start.iloc[int(p1.row_start): int(p1.row_end) + 1].copy()
            window_start = p1.start_datetime + pd.to_timedelta(float(p1.duration_wall_s) * 0.25, unit="s")
            window_end = p1.start_datetime + pd.to_timedelta(float(p1.duration_wall_s) * 0.75, unit="s")
            dwell = dwell.loc[dwell["DateTime"].between(window_start, window_end, inclusive="both")]
            dwell = dwell.loc[dwell["I.S.act [A]"].gt(0)].copy()
            active_mean = dwell.loc[:, ACTIVE_CELLS].mean(axis=1, skipna=False)
            record = {
                "P1S_voltage_V": float(active_mean.median()) if active_mean.notna().any() else np.nan,
                "raw_contributing_count": int(active_mean.notna().sum()),
                "measurement_end_datetime": pd.Timestamp(p1.end_datetime),
                "measurement_end_ophrs2_h": float(p1.end_ophrs2_h),
            }
            for canonical, source_name in RAW_CONDITIONS.items():
                record[canonical] = float(dwell[source_name].median()) if dwell[source_name].notna().any() else np.nan
            values.append(record)
        if not values:
            continue
        measurement = pd.DataFrame(values)
        kind = "COMPLETE_CYCLE_INTERVAL" if position + 1 < len(starts) else "PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL"
        records.append({
            "cycle_interval_id": f"RUNTIME_CYCLE_{position + 1:04d}",
            "interval_kind": kind,
            "raw_cycle_id": int(position + 1),
            "cycle_start_datetime": str(group.iloc[0].start_datetime),
            "measurement_end_datetime": str(measurement["measurement_end_datetime"].max()),
            "measurement_end_ophrs2_h": float(measurement["measurement_end_ophrs2_h"].max()),
            "P1S_voltage_V": float(measurement["P1S_voltage_V"].median()),
            "P1S_operating_voltage_retention": float(measurement["P1S_voltage_V"].median() / reference_voltage),
            "has_complete_middle50_dwell": bool(measurement["P1S_voltage_V"].notna().all() and (measurement["raw_contributing_count"] > 0).all()),
            "cycle_start_anchor_present": True,
            "eligible_dwell_count": int(len(measurement)),
            "raw_contributing_count": int(measurement["raw_contributing_count"].sum()),
            **{name: float(measurement[name].median()) for name in CONDITIONS},
        })
    return pd.DataFrame(records), runs


def _prediction_row(
    checkpoint: int,
    raw: pd.DataFrame,
    characterization_start: pd.Timestamp,
    alignment_inputs: dict,
    models: dict,
) -> tuple[dict, dict]:
    before = raw.loc[raw["DateTime"] < characterization_start].copy()
    if before.empty:
        raise StagePError("TEST_ALIGNMENT_INCONCLUSIVE")
    intervals, runs = _runtime_p1_intervals(before, float(alignment_inputs["reference_voltage_V"]))
    if intervals.empty:
        raise StagePError("TEST_ALIGNMENT_INCONCLUSIVE")
    raw_cutoff = runs.iloc[-1]
    result = align_latest_three_intervals(
        intervals,
        characterization_start,
        float(raw_cutoff.end_ophrs2_h),
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
            "eligible_dwell_count": int(item.eligible_dwell_count),
            "raw_contributing_count": int(item.raw_contributing_count),
            "in_frozen_joint_common_support": bool(item.in_frozen_joint_common_support),
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
) -> dict:
    """Execute both target-free Test checkpoints and freeze one 21-column file."""
    require_manual_test_authorization(authorization_token)
    _log(event_log, "AUTHORIZATION_OK")
    phase = Path(phase)
    verify_frozen_runtime_authority(phase)
    raw_paths = validate_raw_sources(raw_sources, phase=phase)
    polar_paths = bind_both_polarization_sources(polarization_sources, phase=phase)
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

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
    pd.DataFrame([
        {"checkpoint": checkpoint, "source_basename": polar_paths[checkpoint].name,
         "valid_datetime_count": int(len(metadata[checkpoint])),
         "characterization_start_datetime": str(starts[checkpoint]),
         "metadata_columns_read": "DateTime"}
        for checkpoint in (2000, 2300)
    ]).to_csv(runtime_dir / "TEST_CHARACTERIZATION_TIME_METADATA.csv", index=False)

    _log(event_log, "RAW_STAGE_P_READ")
    raw = _read_raw_operating(raw_paths, read_csv=raw_read_csv)
    alignment_inputs = read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    models = read_yaml(phase / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml")
    rows: list[dict] = []
    provenance_records: list[dict] = []
    for checkpoint in (2000, 2300):
        row, record = _prediction_row(checkpoint, raw, starts[checkpoint], alignment_inputs, models)
        record["metadata_source_basename"] = polar_paths[checkpoint].name
        record["raw_source_basenames"] = [path.name for path in raw_paths]
        rows.append(row); provenance_records.append(record)
    schema = read_yaml(phase / "schemas" / "FROZEN_TEST_PREDICTIONS_SCHEMA.yaml")
    prediction = runtime_dir / PREDICTION_NAME
    pd.DataFrame(rows).loc[:, schema["required_columns"]].to_csv(prediction, index=False)
    _log(event_log, "PREDICTION_WRITTEN")
    prediction_sha = write_sha256_sidecar(prediction)
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

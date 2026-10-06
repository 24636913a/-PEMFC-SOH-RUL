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
import warnings
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
from ahc_tv_one_shot_ledger import write_exclusive_json, write_exclusive_sha256_sidecar, write_exclusive_text


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
RCR_TEMPLATE_PATH = (
    PHASE.parents[0] / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit"
    / "outputs" / "CYCLE_TEMPLATE.json"
)
RCR_GAP_SECONDS = 5.0
RCR_ZERO_CURRENT_A = 0.1
RCR_LOADED_MIN_H, RCR_LOADED_MAX_H = 8.0, 11.0
RCR_CORE_ORDER_MIN = 0.90


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
    numeric = pd.to_numeric(values, errors="coerce")
    result = values.astype("string").str.strip()
    result.loc[numeric.notna()] = numeric.loc[numeric.notna()].map(lambda value: f"{float(value):g}")
    return result.astype(str)


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
        frame = read_csv(source, sep=";", encoding="utf-8-sig", usecols=list(RAW_COLUMNS), low_memory=False).copy()
        frame["_source_basename"] = source.name
        frame["_source_index"] = source_index
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["DateTime"] = pd.to_datetime(raw["DateTime"], errors="coerce")
    raw["OpHrs2 [h]"] = pd.to_numeric(raw["OpHrs2 [h]"].astype("string").str.replace(",", "", regex=False), errors="coerce")
    raw["operation_point"] = _normalize_operation(raw["operation_point"])
    for name in ["I.S.set [A]", "I.S.act [A]", *ACTIVE_CELLS, *RAW_CONDITIONS.values()]:
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    raw = raw.loc[raw["DateTime"].notna() & raw["OpHrs2 [h]"].notna()].copy()
    # The frozen RCR implementation preserves source ordering and deliberately
    # starts a fresh state sequence at each source-file boundary.
    return raw.reset_index(drop=True)


def _identity_token(operation_point: str, set_current: float) -> str:
    return f"{str(operation_point)}||{round(float(set_current), 3):.3f}"


def _state_runs(raw: pd.DataFrame) -> pd.DataFrame:
    """RCR-equivalent 5 s source-bounded run-length encoding.

    This replaces the previous 2 s/``DataFrame`` grouping shortcut.  State
    identity, time/order discontinuities, zero-current flags, and source-file
    boundaries now have the same semantics as frozen RCR.
    """
    records: list[pd.DataFrame] = []
    run_offset = 0
    for source_order, source in raw.groupby("_source_index", sort=False):
        work = source.copy()
        work["_rounded_set"] = pd.to_numeric(work["I.S.set [A]"], errors="raise").round(3)
        work["_token"] = work["operation_point"].astype(str) + "||" + work["_rounded_set"].map(lambda value: f"{value:.3f}")
        previous_dt = pd.to_datetime(work["DateTime"]).shift()
        previous_age = pd.to_numeric(work["OpHrs2 [h]"], errors="raise").shift()
        delta_s = (pd.to_datetime(work["DateTime"]) - previous_dt).dt.total_seconds()
        work["_discontinuity"] = (
            (delta_s > RCR_GAP_SECONDS)
            | (pd.to_datetime(work["DateTime"]) <= previous_dt)
            | (pd.to_numeric(work["OpHrs2 [h]"], errors="raise") < previous_age)
        ).fillna(False)
        if int(source_order) != 0:
            work.iloc[0, work.columns.get_loc("_discontinuity")] = True
        work["_new_run"] = work["_token"].ne(work["_token"].shift()).fillna(True) | work["_discontinuity"]
        work["_local_run"] = work["_new_run"].cumsum()
        grouped = work.groupby("_local_run", sort=True, observed=True)
        compressed = grouped.agg(
            start_datetime=("DateTime", "first"), end_datetime=("DateTime", "last"),
            start_ophrs2_h=("OpHrs2 [h]", "first"), end_ophrs2_h=("OpHrs2 [h]", "last"),
            operation_point=("operation_point", "first"), rounded_set_A=("_rounded_set", "first"),
            median_set_current_A=("I.S.set [A]", "median"), median_actual_current_A=("I.S.act [A]", "median"),
            median_actual_current_density_A_cm2=("j.S.act [A/cm2]", "median"), raw_row_count=("DateTime", "size"),
            gap_before=("_discontinuity", "first"), row_start=("DateTime", lambda series: int(series.index.min())),
            row_end=("DateTime", lambda series: int(series.index.max())),
        ).reset_index(drop=True)
        compressed["stop_or_zero_current_flag"] = grouped["I.S.act [A]"].apply(lambda values: bool((values.abs() <= RCR_ZERO_CURRENT_A).any())).to_numpy(dtype=int)
        compressed["duration_wall_s"] = (pd.to_datetime(compressed["end_datetime"]) - pd.to_datetime(compressed["start_datetime"])).dt.total_seconds()
        compressed["duration_loaded_s"] = (pd.to_numeric(compressed["end_ophrs2_h"]) - pd.to_numeric(compressed["start_ophrs2_h"])) * 3600.0
        compressed["run_id"] = np.arange(run_offset + 1, run_offset + len(compressed) + 1, dtype=int)
        run_offset += len(compressed)
        compressed["state_token"] = compressed["operation_point"].astype(str) + "||" + compressed["rounded_set_A"].map(lambda value: f"{value:.3f}")
        compressed["set_token"] = compressed["rounded_set_A"].map(lambda value: f"{value:.3f}")
        compressed["source_order"] = int(source_order)
        records.append(compressed)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _lcs_length(left: list[str], right: list[str]) -> int:
    prior = [0] * (len(right) + 1)
    for token_left in left:
        current = [0]
        for index, token_right in enumerate(right, start=1):
            current.append(prior[index - 1] + 1 if token_left == token_right else max(prior[index], current[-1]))
        prior = current
    return prior[-1]


def _load_rcr_template() -> dict[str, Any]:
    if not RCR_TEMPLATE_PATH.exists():
        raise StagePError("FROZEN_RCR_TEMPLATE_MISSING")
    return json.loads(RCR_TEMPLATE_PATH.read_text(encoding="utf-8"))


def _nanmean_active_voltage(frame: pd.DataFrame) -> pd.Series:
    cells = frame.loc[:, ACTIVE_CELLS].to_numpy(dtype=float)
    all_missing = np.isnan(cells).all(axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values = np.nanmean(cells, axis=1)
    values[all_missing] = np.nan
    return pd.Series(values, index=frame.index, dtype=float)


def _measurement_for_p1_runs(raw: pd.DataFrame, p1_runs: pd.DataFrame) -> pd.DataFrame:
    values: list[dict[str, Any]] = []
    for p1 in p1_runs.itertuples(index=False):
        window_start = pd.Timestamp(p1.start_datetime) + pd.to_timedelta(float(p1.duration_wall_s) * .25, unit="s")
        window_end = pd.Timestamp(p1.start_datetime) + pd.to_timedelta(float(p1.duration_wall_s) * .75, unit="s")
        # PAA's frozen window-membership contract is a global DateTime lookup;
        # it intentionally does not add a source-file restriction at packaging
        # boundaries where timestamp regions may overlap.
        dwell = raw.loc[raw["DateTime"].between(window_start, window_end, inclusive="both")].copy()
        # PAA measures a stable run over the inclusive middle-50 window; it
        # does not impose an additional positive-current row filter.
        active_mean = _nanmean_active_voltage(dwell) if not dwell.empty else pd.Series(dtype=float)
        value: dict[str, Any] = {
            "P1S_voltage_V": float(np.nanmedian(active_mean.to_numpy(dtype=float))) if active_mean.notna().any() else np.nan,
            "raw_contributing_count": int(active_mean.notna().sum()),
            "measurement_end_datetime": pd.Timestamp(p1.end_datetime),
            "measurement_end_ophrs2_h": float(p1.end_ophrs2_h),
        }
        for canonical, source_name in RAW_CONDITIONS.items():
            source_values = pd.to_numeric(dwell[source_name], errors="coerce")
            value[canonical] = float(np.nanmedian(source_values.to_numpy(dtype=float))) if source_values.notna().any() else np.nan
        values.append(value)
    return pd.DataFrame(values)


def _runtime_p1_intervals(
    raw_before_start: pd.DataFrame,
    reference_voltage: float,
    *,
    partial_interval_id: str = "PARTIAL_PRE_RUNTIME",
    require_development_stratum: bool = False,
    synthetic_test_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build canonical RCR-template-qualified complete/partial P1S intervals."""
    runs = _state_runs(raw_before_start)
    if runs.empty:
        raise StagePError("TEST_ALIGNMENT_INCONCLUSIVE")
    starts = [index for index in range(len(runs) - 1) if (runs.iloc[index].state_token, runs.iloc[index + 1].state_token) == ANCHOR_PATTERN]
    if not starts:
        raise StagePError("TEST_ALIGNMENT_INCONCLUSIVE")
    template = _load_rcr_template()
    t_set = list(template["template_set_tokens"])
    t_operation = list(template["template_positive_operation_tokens"])
    records: list[dict[str, Any]] = []
    for raw_cycle_id, (left, right) in enumerate(zip(starts[:-1], starts[1:]), start=1):
        cycle = runs.iloc[left:right].copy()
        end_anchor = runs.iloc[right]
        loaded_h = float(end_anchor.start_ophrs2_h - cycle.iloc[0].start_ophrs2_h)
        positive = cycle.loc[cycle["stop_or_zero_current_flag"].eq(0)].copy()
        op_fraction = _lcs_length(t_operation, positive["operation_point"].astype(str).tolist()) / len(t_operation)
        complete = bool(RCR_LOADED_MIN_H <= loaded_h <= RCR_LOADED_MAX_H and op_fraction >= RCR_CORE_ORDER_MIN)
        if require_development_stratum:
            start_age, end_age = float(cycle.iloc[0].start_ophrs2_h), float(end_anchor.start_ophrs2_h)
            complete &= bool((72.788690 <= start_age < 655.191706 and end_age < 655.191706) or (655.191706 <= start_age < 1237.594722 and end_age < 1237.594722) or (1237.594722 <= start_age < 1819.997738 and end_age < 1819.997738))
        if not complete and not synthetic_test_mode:
            continue
        p1 = cycle.loc[(cycle["state_token"].eq(P1_TOKEN)) & (cycle["duration_wall_s"].ge(120.0)) & (cycle["stop_or_zero_current_flag"].eq(0))]
        measurement = _measurement_for_p1_runs(raw_before_start, p1)
        if measurement.empty:
            continue
        records.append({
            "cycle_interval_id": f"COMPLETE_{raw_cycle_id:03d}", "interval_kind": "COMPLETE_CYCLE_INTERVAL",
            "raw_cycle_id": raw_cycle_id, "cycle_start_datetime": str(cycle.iloc[0].start_datetime),
            "measurement_end_datetime": str(measurement["measurement_end_datetime"].max()),
            "measurement_end_ophrs2_h": float(measurement["measurement_end_ophrs2_h"].max()),
            "P1S_voltage_V": float(np.nanmedian(measurement["P1S_voltage_V"].to_numpy(dtype=float))),
            "P1S_operating_voltage_retention": float(np.nanmedian(measurement["P1S_voltage_V"].to_numpy(dtype=float)) / reference_voltage),
            "has_complete_middle50_dwell": bool(measurement["P1S_voltage_V"].notna().all() and (measurement["raw_contributing_count"] > 0).all()),
            "cycle_start_anchor_present": True, "eligible_dwell_count": int(len(measurement)),
            "raw_contributing_count": int(measurement["raw_contributing_count"].sum()),
            **{name: float(np.nanmedian(measurement[name].to_numpy(dtype=float))) for name in CONDITIONS},
        })
    partial_start = starts[-1]
    partial = runs.iloc[partial_start:].copy()
    p1 = partial.loc[(partial["state_token"].eq(P1_TOKEN)) & (partial["duration_wall_s"].ge(120.0)) & (partial["stop_or_zero_current_flag"].eq(0))]
    measurement = _measurement_for_p1_runs(raw_before_start, p1)
    if not measurement.empty:
        partial_id = len(starts)
        records.append({
            "cycle_interval_id": partial_interval_id, "interval_kind": "PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL",
            "raw_cycle_id": partial_id, "cycle_start_datetime": str(partial.iloc[0].start_datetime),
            "measurement_end_datetime": str(measurement["measurement_end_datetime"].max()),
            "measurement_end_ophrs2_h": float(measurement["measurement_end_ophrs2_h"].max()),
            "P1S_voltage_V": float(np.nanmedian(measurement["P1S_voltage_V"].to_numpy(dtype=float))),
            "P1S_operating_voltage_retention": float(np.nanmedian(measurement["P1S_voltage_V"].to_numpy(dtype=float)) / reference_voltage),
            "has_complete_middle50_dwell": bool(measurement["P1S_voltage_V"].notna().all() and (measurement["raw_contributing_count"] > 0).all()),
            "cycle_start_anchor_present": bool((partial.iloc[0].state_token, partial.iloc[1].state_token) == ANCHOR_PATTERN) if len(partial) > 1 else False,
            "eligible_dwell_count": int(len(measurement)), "raw_contributing_count": int(measurement["raw_contributing_count"].sum()),
            **{name: float(np.nanmedian(measurement[name].to_numpy(dtype=float))) for name in CONDITIONS},
        })
    return pd.DataFrame(records), runs


def _prediction_row(
    checkpoint: int,
    raw: pd.DataFrame,
    characterization_start: pd.Timestamp,
    alignment_inputs: dict,
    models: dict,
    *,
    synthetic_test_mode: bool = False,
) -> tuple[dict, dict]:
    before = raw.loc[raw["DateTime"] < characterization_start].copy()
    if before.empty:
        raise StagePError("TEST_ALIGNMENT_INCONCLUSIVE")
    intervals, runs = _runtime_p1_intervals(
        before,
        float(alignment_inputs["reference_voltage_V"]),
        partial_interval_id=f"PARTIAL_PRE_{int(checkpoint)}",
        synthetic_test_mode=synthetic_test_mode,
    )
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
    synthetic_test_mode: bool = False,
) -> dict:
    """Execute both target-free Test checkpoints and freeze one 21-column file."""
    require_manual_test_authorization(authorization_token)
    _log(event_log, "AUTHORIZATION_OK")
    phase = Path(phase)
    verify_frozen_runtime_authority(phase)
    raw_paths = validate_raw_sources(raw_sources, phase=phase)
    polar_paths = bind_both_polarization_sources(polarization_sources, phase=phase)
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

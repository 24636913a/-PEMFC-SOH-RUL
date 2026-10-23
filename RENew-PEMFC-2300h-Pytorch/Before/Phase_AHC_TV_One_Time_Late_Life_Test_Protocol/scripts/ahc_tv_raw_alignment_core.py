"""Authoritative PAA-compatible raw-to-P1S-interval implementation.

This module is intentionally shared by formal Stage P and its Development
replay.  It owns every raw-data semantic that can affect a P1S interval:
source boundaries, discontinuities, RCR qualification, the PAA global window
registry, and active-cell/condition aggregation.  It is target-blind.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ahc_tv_alignment import CONDITIONS, PHASE


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
RCR_TEMPLATE_PATH = (
    PHASE.parents[0] / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit"
    / "outputs" / "CYCLE_TEMPLATE.json"
)
RCR_GAP_SECONDS = 5.0
RCR_ZERO_CURRENT_A = 0.1
RCR_LOADED_MIN_H, RCR_LOADED_MAX_H = 8.0, 11.0
RCR_CORE_ORDER_MIN = 0.90


class RawAlignmentCoreError(RuntimeError):
    """The target-blind, frozen PAA/RCR raw alignment contract failed."""


def normalize_operation(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = values.astype("string").str.strip()
    result.loc[numeric.notna()] = numeric.loc[numeric.notna()].map(lambda value: f"{float(value):g}")
    return result.astype(str)


def normalize_raw_operating_frame(frame: pd.DataFrame, *, source_index: int, source_basename: str) -> pd.DataFrame:
    """Normalize one source without silently filtering measurement candidates."""
    absent = [name for name in RAW_COLUMNS if name not in frame.columns]
    if absent:
        raise RawAlignmentCoreError("RAW_SCHEMA_MISSING_COLUMNS: " + ",".join(absent))
    work = frame.loc[:, RAW_COLUMNS].copy()
    work["_source_basename"] = str(source_basename)
    work["_source_index"] = int(source_index)
    work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
    work["OpHrs2 [h]"] = pd.to_numeric(
        work["OpHrs2 [h]"].astype("string").str.replace(",", "", regex=False), errors="coerce"
    )
    for name in ["I.S.set [A]", "I.S.act [A]", *ACTIVE_CELLS, *RAW_CONDITIONS.values()]:
        work[name] = pd.to_numeric(work[name], errors="coerce")
    # Keep source operation-point text for RCR identity; PAA lookup normalizes
    # it later only when comparing a raw row with an owned P1 window.
    work["operation_point"] = work["operation_point"].astype("string").str.strip()
    return work.loc[work["DateTime"].notna() & work["OpHrs2 [h]"].notna()].reset_index(drop=True)


def read_development_prefix(
    sources: Iterable[Path], *, cutoff_datetime: pd.Timestamp, cutoff_ophrs2_h: float,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Read only the legal Development prefix from explicitly supplied sources."""
    frames: list[pd.DataFrame] = []
    cutoff_datetime = pd.Timestamp(cutoff_datetime)
    for source_index, source in enumerate(Path(path) for path in sources):
        if not source.exists():
            raise RawAlignmentCoreError("DEVELOPMENT_RAW_SOURCE_MISSING")
        header = pd.read_csv(source, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        absent = [name for name in RAW_COLUMNS if name not in header]
        if absent:
            raise RawAlignmentCoreError("DEVELOPMENT_RAW_SCHEMA_MISSING_COLUMNS: " + ",".join(absent))
        reached_cutoff = False
        for chunk in pd.read_csv(source, sep=";", encoding="utf-8-sig", usecols=list(RAW_COLUMNS), chunksize=chunksize, low_memory=False):
            work = normalize_raw_operating_frame(chunk, source_index=source_index, source_basename=source.name)
            before = work.loc[
                (work["DateTime"] < cutoff_datetime) & (work["OpHrs2 [h]"] < float(cutoff_ophrs2_h))
            ].copy()
            if not before.empty:
                frames.append(before)
            if ((work["DateTime"] >= cutoff_datetime) | (work["OpHrs2 [h]"] >= float(cutoff_ophrs2_h))).any():
                reached_cutoff = True
                break
        if reached_cutoff:
            break
    if not frames:
        raise RawAlignmentCoreError("DEVELOPMENT_RAW_PREFIX_EMPTY")
    return pd.concat(frames, ignore_index=True, sort=False)


def _identity_token(operation_point: str, set_current: float) -> str:
    return f"{str(operation_point)}||{round(float(set_current), 3):.3f}"


def state_runs(raw: pd.DataFrame) -> pd.DataFrame:
    """RCR-equivalent source-bounded runs, with all discontinuity semantics."""
    records: list[pd.DataFrame] = []
    run_offset = 0
    source_groups = list(raw.groupby("_source_index", sort=False))
    for source_position, (source_order, source) in enumerate(source_groups):
        work = source.copy()
        finite = (
            work["operation_point"].notna()
            & np.isfinite(pd.to_numeric(work["I.S.set [A]"], errors="coerce"))
            & np.isfinite(pd.to_numeric(work["I.S.act [A]"], errors="coerce"))
            & np.isfinite(pd.to_numeric(work["j.S.act [A/cm2]"], errors="coerce"))
        )
        work = work.loc[finite].copy()
        if work.empty:
            continue
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
        if source_position > 0:
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
        compressed["stop_or_zero_current_flag"] = grouped["I.S.act [A]"].apply(
            lambda values: bool((values.abs() <= RCR_ZERO_CURRENT_A).any())
        ).to_numpy(dtype=int)
        compressed["duration_wall_s"] = (pd.to_datetime(compressed["end_datetime"]) - pd.to_datetime(compressed["start_datetime"])).dt.total_seconds()
        compressed["duration_loaded_s"] = (pd.to_numeric(compressed["end_ophrs2_h"]) - pd.to_numeric(compressed["start_ophrs2_h"])) * 3600.0
        compressed["run_id"] = np.arange(run_offset + 1, run_offset + len(compressed) + 1, dtype=int)
        run_offset += len(compressed)
        compressed["state_token"] = compressed["operation_point"].astype(str) + "||" + compressed["rounded_set_A"].map(lambda value: f"{value:.3f}")
        compressed["set_token"] = compressed["rounded_set_A"].map(lambda value: f"{value:.3f}")
        compressed["source_order"] = int(source_order)
        records.append(compressed)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def lcs_length(left: list[str], right: list[str]) -> int:
    prior = [0] * (len(right) + 1)
    for token_left in left:
        current = [0]
        for index, token_right in enumerate(right, start=1):
            current.append(prior[index - 1] + 1 if token_left == token_right else max(prior[index], current[-1]))
        prior = current
    return prior[-1]


def load_rcr_template() -> dict[str, Any]:
    if not RCR_TEMPLATE_PATH.exists():
        raise RawAlignmentCoreError("FROZEN_RCR_TEMPLATE_MISSING")
    return json.loads(RCR_TEMPLATE_PATH.read_text(encoding="utf-8"))


def nanmean_active_voltage(frame: pd.DataFrame) -> pd.Series:
    cells = frame.loc[:, ACTIVE_CELLS].to_numpy(dtype=float)
    all_missing = np.isnan(cells).all(axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values = np.nanmean(cells, axis=1)
    values[all_missing] = np.nan
    return pd.Series(values, index=frame.index, dtype=float)


def p1_middle50_measurements(raw: pd.DataFrame, runs: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """PAA global middle-50/last-25 registry with global DateTime ownership."""
    eligible = runs.loc[
        runs["state_token"].eq(P1_TOKEN)
        & runs["duration_wall_s"].ge(120.0)
        & runs["stop_or_zero_current_flag"].eq(0)
    ].copy()
    if eligible.empty:
        return {}
    windows: list[dict[str, Any]] = []
    for run in eligible.itertuples(index=False):
        start = pd.Timestamp(run.start_datetime)
        duration = float(run.duration_wall_s)
        windows.extend((
            {"run_id": int(run.run_id), "window_kind": "middle_50", "window_start": start + pd.to_timedelta(duration * .25, unit="s"), "window_end": start + pd.to_timedelta(duration * .75, unit="s"), "state_token": str(run.state_token)},
            {"run_id": int(run.run_id), "window_kind": "last_25", "window_start": start + pd.to_timedelta(duration * .75, unit="s"), "window_end": start + pd.to_timedelta(duration, unit="s"), "state_token": str(run.state_token)},
        ))
    registry = pd.DataFrame(windows).sort_values("window_start", kind="mergesort").reset_index(drop=True)
    registry["window_id"] = np.arange(len(registry), dtype=int)
    starts = registry["window_start"].to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = registry["window_end"].to_numpy(dtype="datetime64[ns]").astype("int64")
    lookup = raw.copy()
    lookup["_normalized_operation"] = normalize_operation(lookup["operation_point"])
    timestamps = pd.to_datetime(lookup["DateTime"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    positions = np.searchsorted(starts, timestamps, side="right") - 1
    valid = (positions >= 0) & (positions < len(registry))
    valid &= timestamps <= ends[np.clip(positions, 0, len(registry) - 1)]
    owned = lookup.loc[valid].copy()
    owned["_window_position"] = positions[valid]
    if not owned.empty:
        expected = registry.iloc[owned["_window_position"].to_numpy()].reset_index(drop=True)
        observed_token = [
            _identity_token(op, current)
            for op, current in zip(owned["_normalized_operation"].tolist(), owned["I.S.set [A]"].tolist())
        ]
        if any(token != expected_token for token, expected_token in zip(observed_token, expected["state_token"].tolist())):
            raise RawAlignmentCoreError("PAA_GLOBAL_WINDOW_OWNERSHIP_IDENTITY_MISMATCH")
    measurements: dict[int, dict[str, Any]] = {}
    for registry_row in registry.loc[registry["window_kind"].eq("middle_50")].itertuples(index=False):
        dwell = owned.loc[owned["_window_position"].eq(int(registry_row.window_id))].copy()
        active = nanmean_active_voltage(dwell) if not dwell.empty else pd.Series(dtype=float)
        result: dict[str, Any] = {
            "P1S_voltage_V": float(np.nanmedian(active.to_numpy(dtype=float))) if active.notna().any() else np.nan,
            "raw_contributing_count": int(active.notna().sum()),
            "measurement_end_datetime": pd.Timestamp(
                eligible.loc[eligible["run_id"].eq(int(registry_row.run_id)), "end_datetime"].iloc[0]
            ),
            "measurement_end_ophrs2_h": float(
                eligible.loc[eligible["run_id"].eq(int(registry_row.run_id)), "end_ophrs2_h"].iloc[0]
            ),
        }
        for canonical, source_name in RAW_CONDITIONS.items():
            values = pd.to_numeric(dwell[source_name], errors="coerce") if not dwell.empty else pd.Series(dtype=float)
            result[canonical] = float(np.nanmedian(values.to_numpy(dtype=float))) if values.notna().any() else np.nan
        measurements[int(registry_row.run_id)] = result
    return measurements


def _development_stratum(start_age: float, end_age: float) -> bool:
    return bool(
        (72.788690 <= start_age < 655.191706 and end_age < 655.191706)
        or (655.191706 <= start_age < 1237.594722 and end_age < 1237.594722)
        or (1237.594722 <= start_age < 1819.997738 and end_age < 1819.997738)
    )


def runtime_p1_intervals(
    raw_before_start: pd.DataFrame, reference_voltage: float, *, partial_interval_id: str = "PARTIAL_PRE_RUNTIME",
    partial_last_run_id: int | None = None, require_development_stratum: bool = False,
    synthetic_test_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all formal complete and partial intervals using the shared core."""
    runs = state_runs(raw_before_start)
    if runs.empty:
        raise RawAlignmentCoreError("ALIGNMENT_INCONCLUSIVE")
    starts = [index for index in range(len(runs) - 1) if (runs.iloc[index].state_token, runs.iloc[index + 1].state_token) == ANCHOR_PATTERN]
    if not starts:
        raise RawAlignmentCoreError("ALIGNMENT_INCONCLUSIVE")
    template = load_rcr_template()
    template_operations = list(template["template_positive_operation_tokens"])
    measurements = p1_middle50_measurements(raw_before_start, runs)
    records: list[dict[str, Any]] = []

    def append_interval(*, interval_id: str, kind: str, raw_cycle_id: int, cycle: pd.DataFrame, end_anchor: pd.Series | None) -> None:
        p1 = cycle.loc[
            cycle["state_token"].eq(P1_TOKEN)
            & cycle["duration_wall_s"].ge(120.0)
            & cycle["stop_or_zero_current_flag"].eq(0)
        ]
        selected = [measurements[int(run_id)] for run_id in p1["run_id"].tolist() if int(run_id) in measurements]
        if not selected:
            return
        measurement = pd.DataFrame(selected)
        end_datetime = pd.to_datetime(measurement["measurement_end_datetime"]).max()
        end_age = float(pd.to_numeric(measurement["measurement_end_ophrs2_h"], errors="raise").max())
        voltage_values = measurement["P1S_voltage_V"].to_numpy(dtype=float)
        records.append({
            "cycle_interval_id": interval_id, "interval_kind": kind, "raw_cycle_id": int(raw_cycle_id),
            "cycle_start_datetime": str(cycle.iloc[0].start_datetime),
            "measurement_end_datetime": str(end_datetime), "measurement_end_ophrs2_h": float(end_age),
            "P1S_voltage_V": float(np.nanmedian(voltage_values)),
            "P1S_operating_voltage_retention": float(np.nanmedian(voltage_values) / float(reference_voltage)),
            "has_complete_middle50_dwell": bool(measurement["P1S_voltage_V"].notna().all() and (measurement["raw_contributing_count"] > 0).all()),
            "cycle_start_anchor_present": True,
            "eligible_dwell_count": int(len(measurement)), "raw_contributing_count": int(measurement["raw_contributing_count"].sum()),
            **{name: float(np.nanmedian(measurement[name].to_numpy(dtype=float))) for name in CONDITIONS},
        })

    for raw_cycle_id, (left, right) in enumerate(zip(starts[:-1], starts[1:]), start=1):
        cycle = runs.iloc[left:right].copy()
        end_anchor = runs.iloc[right]
        loaded_h = float(end_anchor.start_ophrs2_h - cycle.iloc[0].start_ophrs2_h)
        positive = cycle.loc[cycle["stop_or_zero_current_flag"].eq(0)].copy()
        order_fraction = lcs_length(template_operations, positive["operation_point"].astype(str).tolist()) / len(template_operations)
        complete = bool(RCR_LOADED_MIN_H <= loaded_h <= RCR_LOADED_MAX_H and order_fraction >= RCR_CORE_ORDER_MIN)
        if require_development_stratum:
            complete &= _development_stratum(float(cycle.iloc[0].start_ophrs2_h), float(end_anchor.start_ophrs2_h))
        if complete or synthetic_test_mode:
            append_interval(interval_id=f"COMPLETE_{raw_cycle_id:03d}", kind="COMPLETE_CYCLE_INTERVAL", raw_cycle_id=raw_cycle_id, cycle=cycle, end_anchor=end_anchor)

    partial = runs.iloc[starts[-1]:].copy()
    if partial_last_run_id is not None:
        partial = partial.loc[partial["run_id"].le(int(partial_last_run_id))].copy()
    if not partial.empty:
        append_interval(interval_id=partial_interval_id, kind="PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL", raw_cycle_id=len(starts), cycle=partial, end_anchor=None)
    return pd.DataFrame(records), runs

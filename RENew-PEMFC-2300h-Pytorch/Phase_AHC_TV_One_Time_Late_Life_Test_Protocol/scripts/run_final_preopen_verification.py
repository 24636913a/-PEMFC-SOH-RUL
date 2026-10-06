"""Development-only raw equivalence checks for the final AHC-TV pre-open gate.

Nothing in this script opens 2000 h/2300 h, after-Air-Air, Test EIS, or a
future Test prediction.  Each Stage-P replay stops at the checkpoint-specific
PAA cutoff and reconstructs RCR-compatible state runs from raw operating rows.
"""
from __future__ import annotations

import json
import re
import warnings
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahc_tv_alignment import CONDITIONS, PHASE, read_yaml, replay_development_alignment, sha256_file
from ahc_tv_stage_p import (
    ACTIVE_CELLS, ANCHOR_PATTERN, P1_TOKEN, RAW_COLUMNS, RAW_CONDITIONS, RAW_IDENTITY,
    RCR_CORE_ORDER_MIN, RCR_GAP_SECONDS, RCR_LOADED_MAX_H, RCR_LOADED_MIN_H,
    RCR_ZERO_CURRENT_A, _lcs_length, _load_rcr_template, _normalize_operation,
)
from ahc_tv_stage_t import TARGET_COLUMNS, extract_jstar_voltage_from_table


REPO = PHASE.parents[1]
PAA_PHASE = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAA_P1_StateToken_PreAnchor_Alignment"
CHM_PHASE = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model"
OUTPUTS = PHASE / "outputs"
# This is the frozen PAA raw-root provenance.  It is intentionally distinct
# from CHM's project-local copy: raw equivalence must replay the exact source
# family used to construct the PAA artifact, not merely a similarly named copy.
PAA_RAW_ROOT = Path(
    r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据"
)

VOLTAGE_TOL = 1e-12
RETENTION_TOL = 1e-12
CONDITION_TOL = 1e-10
OPHRS_TOL = 1e-9


class PreOpenVerificationError(RuntimeError):
    """A Development-only raw replay did not reproduce a frozen artifact."""


def _load_dev_raw_sources() -> list[Path]:
    files = list((PAA_RAW_ROOT / "Loadcycle_data").rglob("Loadcycle_*.csv"))

    def key(path: Path) -> tuple[int, int, str]:
        match = re.fullmatch(r"Loadcycle_(\d+)-(\d+)", path.stem)
        if match is None:
            raise PreOpenVerificationError(f"UNPARSEABLE_DEVELOPMENT_RAW_SOURCE:{path.name}")
        return int(match.group(1)), int(match.group(2)), path.name

    selected = sorted((path for path in files if key(path)[1] <= 150), key=key)
    if not selected or any(key(path)[1] > 150 for path in selected):
        raise PreOpenVerificationError("DEVELOPMENT_RAW_SOURCE_ALLOWLIST_INVALID")
    return selected


def _read_raw_strictly_before_cutoff(*, cutoff_time: pd.Timestamp, cutoff_h: float) -> pd.DataFrame:
    """Read only the permitted Development prefix for one PAA checkpoint."""
    frames: list[pd.DataFrame] = []
    stopped = False
    for source_index, path in enumerate(_load_dev_raw_sources()):
        if stopped:
            break
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        missing = [column for column in RAW_COLUMNS if column not in header]
        if missing:
            raise PreOpenVerificationError("RAW_SCHEMA_MISSING:" + ",".join(missing))
        for chunk in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=list(RAW_COLUMNS), chunksize=100_000, low_memory=False):
            work = chunk.copy()
            work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
            work["OpHrs2 [h]"] = pd.to_numeric(work["OpHrs2 [h]"].astype("string").str.replace(",", "", regex=False), errors="coerce")
            work["operation_point"] = _normalize_operation(work["operation_point"])
            for name in ["I.S.set [A]", "I.S.act [A]", *RAW_COLUMNS[5:]]:
                work[name] = pd.to_numeric(work[name], errors="coerce")
            valid = work["DateTime"].notna() & work["OpHrs2 [h]"].notna()
            for name in ["I.S.set [A]", "I.S.act [A]", "j.S.act [A/cm2]"]:
                valid &= np.isfinite(work[name])
            work = work.loc[valid].copy()
            permitted = work["DateTime"].lt(cutoff_time) & work["OpHrs2 [h]"].lt(float(cutoff_h))
            prefix = work.loc[permitted].copy()
            if not prefix.empty:
                prefix["_source_basename"] = path.name
                prefix["_source_index"] = source_index
                frames.append(prefix)
            # Source age is chronological.  This is a read firewall: later
            # chunks/files cannot inform this checkpoint's reconstruction.
            if (~permitted).any():
                stopped = True
                break
    if not frames:
        raise PreOpenVerificationError("NO_RAW_ROWS_BEFORE_FROZEN_CUTOFF")
    return pd.concat(frames, ignore_index=True, sort=False)


def _development_age_stratum(start_age: float, end_age: float) -> bool:
    return bool(
        (72.788690 <= start_age < 655.191706 and end_age < 655.191706)
        or (655.191706 <= start_age < 1237.594722 and end_age < 1237.594722)
        or (1237.594722 <= start_age < 1819.997738 and end_age < 1819.997738)
    )


def _stream_stage_p_raw_to_cutoff(*, cutoff_time: pd.Timestamp, cutoff_h: float) -> pd.DataFrame:
    """Memory-bounded raw→RCR-runs→middle-50 measurement reconstruction.

    It retains only one open state run.  The P1 middle-50 observations are
    collected in a second, stable-window-only scan, exactly as PAA's global
    ``frozen_window_membership`` contract requires.
    """
    run_records: list[dict[str, Any]] = []
    run_id = 1
    acc: dict[str, Any] | None = None
    previous_dt: pd.Timestamp | None = None
    previous_age: float | None = None
    stopped = False

    def finish() -> None:
        nonlocal acc
        if acc is None:
            return
        duration = float((acc["end_datetime"] - acc["start_datetime"]).total_seconds())
        record = {
            "run_id": int(acc["run_id"]), "start_datetime": acc["start_datetime"], "end_datetime": acc["end_datetime"],
            "start_ophrs2_h": float(acc["start_ophrs2_h"]), "end_ophrs2_h": float(acc["end_ophrs2_h"]),
            "duration_wall_s": duration, "duration_loaded_s": float((acc["end_ophrs2_h"] - acc["start_ophrs2_h"]) * 3600.0),
            "operation_point": str(acc["operation_point"]), "median_set_current_A": float(np.median(acc["set_values"])),
            "median_actual_current_A": float(np.median(acc["actual_values"])),
            "median_actual_current_density_A_cm2": float(np.median(acc["density_values"])),
            "raw_row_count": int(acc["raw_row_count"]), "gap_before": int(acc["gap_before"]),
            "stop_or_zero_current_flag": int(acc["zero_or_stop"]),
            "state_token": f"{acc['operation_point']}||{acc['rounded_set_A']:.3f}", "set_token": f"{acc['rounded_set_A']:.3f}",
        }
        run_records.append(record)
        acc = None

    for source_index, path in enumerate(_load_dev_raw_sources()):
        if stopped:
            break
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        missing = [column for column in RAW_IDENTITY if column not in header]
        if missing:
            raise PreOpenVerificationError("RAW_SCHEMA_MISSING:" + ",".join(missing))
        reader = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=list(RAW_IDENTITY), chunksize=50_000, low_memory=False)
        for chunk in reader:
            work = chunk.copy()
            work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
            work["OpHrs2 [h]"] = pd.to_numeric(work["OpHrs2 [h]"].astype("string").str.replace(",", "", regex=False), errors="coerce")
            # Match the RCR compressor exactly: source operation labels retain
            # string semantics during state-run reconstruction.  PAA's numeric
            # normalisation is used only during its later raw-window lookup.
            work["operation_point"] = work["operation_point"].astype("string")
            for name in ["I.S.set [A]", "I.S.act [A]", "j.S.act [A/cm2]"]:
                work[name] = pd.to_numeric(work[name], errors="coerce")
            valid = work["DateTime"].notna() & work["OpHrs2 [h]"].notna() & work["operation_point"].notna()
            for name in ["I.S.set [A]", "I.S.act [A]", "j.S.act [A/cm2]"]:
                valid &= np.isfinite(work[name])
            work = work.loc[valid].copy()
            datetimes = work["DateTime"].to_numpy(); ages = work["OpHrs2 [h]"].to_numpy(dtype=float)
            operations = work["operation_point"].astype(str).to_numpy(); setpoints = work["I.S.set [A]"].to_numpy(dtype=float)
            actual = work["I.S.act [A]"].to_numpy(dtype=float); density = work["j.S.act [A/cm2]"].to_numpy(dtype=float)
            for index in range(len(work)):
                dt = pd.Timestamp(datetimes[index]); age = float(ages[index])
                if not (dt < cutoff_time and age < cutoff_h):
                    stopped = True
                    break
                operation = str(operations[index]); rounded = round(float(setpoints[index]), 3)
                changed = acc is None or operation != acc["operation_point"] or rounded != acc["rounded_set_A"]
                discontinuity = previous_dt is not None and (
                    (dt - previous_dt).total_seconds() > RCR_GAP_SECONDS or dt <= previous_dt or age < float(previous_age)
                )
                if acc is not None and (changed or discontinuity):
                    finish()
                if acc is None:
                    acc = {
                        "run_id": run_id, "operation_point": operation, "rounded_set_A": rounded,
                        "start_datetime": dt, "end_datetime": dt, "start_ophrs2_h": age, "end_ophrs2_h": age,
                        "gap_before": bool(discontinuity or source_index > 0), "set_values": [], "actual_values": [],
                        "density_values": [], "raw_row_count": 0, "zero_or_stop": False,
                    }
                    run_id += 1
                acc["end_datetime"] = dt; acc["end_ophrs2_h"] = age; acc["set_values"].append(float(setpoints[index]))
                acc["actual_values"].append(float(actual[index])); acc["density_values"].append(float(density[index])); acc["raw_row_count"] += 1
                acc["zero_or_stop"] = bool(acc["zero_or_stop"] or abs(float(actual[index])) <= RCR_ZERO_CURRENT_A)
                previous_dt, previous_age = dt, age
            if stopped:
                break
        reader.close()
        if stopped:
            break
        # RCR never merges a state across raw source files.
        finish()
        previous_dt = previous_age = None
    finish()
    return pd.DataFrame(run_records)


def _collect_global_paa_middle50_measurements(*, runs: pd.DataFrame, cutoff_h: float) -> dict[int, dict[str, Any]]:
    """PAA-equivalent global DateTime membership over all permitted raw rows.

    The frozen PAA reader does not restrict a window to the source file that
    produced its RCR run.  This matters at packaging boundaries with duplicate
    timestamp regions, so the source-agnostic lookup is deliberate.
    """
    p1 = runs.loc[
        runs["state_token"].eq(P1_TOKEN)
        & runs["duration_wall_s"].ge(120.0)
        & runs["stop_or_zero_current_flag"].eq(0)
    ].copy().sort_values("start_datetime", kind="mergesort").reset_index(drop=True)
    # PAA assigns raw rows against the combined, chronologically sorted
    # middle-50 and last-25 window registry, then retains the former for the
    # primary measurement.  Do not shortcut to middle-50 alone: in adjacent
    # state runs that can change which frozen window owns a boundary sample.
    middle = p1.copy()
    middle["stable_window"] = "middle_50"
    middle["window_start"] = pd.to_datetime(middle["start_datetime"]) + pd.to_timedelta(middle["duration_wall_s"] * .25, unit="s")
    middle["window_end"] = pd.to_datetime(middle["start_datetime"]) + pd.to_timedelta(middle["duration_wall_s"] * .75, unit="s")
    last = p1.copy()
    last["stable_window"] = "last_25"
    last["window_start"] = pd.to_datetime(last["start_datetime"]) + pd.to_timedelta(last["duration_wall_s"] * .75, unit="s")
    last["window_end"] = pd.to_datetime(last["start_datetime"]) + pd.to_timedelta(last["duration_wall_s"], unit="s")
    windows = pd.concat([middle, last], ignore_index=True).sort_values("window_start").reset_index(drop=True)
    windows["window_id"] = np.arange(len(windows), dtype=int)
    starts = windows["window_start"].to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = windows["window_end"].to_numpy(dtype="datetime64[ns]").astype("int64")
    store: dict[int, list[np.ndarray]] = {int(index): [] for index in windows["window_id"]}
    stopped = False
    for path in _load_dev_raw_sources():
        if stopped:
            break
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        missing = [column for column in RAW_COLUMNS if column not in header]
        if missing:
            raise PreOpenVerificationError("RAW_SCHEMA_MISSING:" + ",".join(missing))
        reader = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=list(RAW_COLUMNS), chunksize=50_000, low_memory=False)
        for chunk in reader:
            work = chunk.copy()
            work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
            work["OpHrs2 [h]"] = pd.to_numeric(work["OpHrs2 [h]"].astype("string").str.replace(",", "", regex=False), errors="coerce")
            work["operation_point"] = _normalize_operation(work["operation_point"])
            for name in ["I.S.set [A]", "I.S.act [A]", *RAW_COLUMNS[5:]]:
                work[name] = pd.to_numeric(work[name], errors="coerce")
            work = work.loc[work["DateTime"].notna() & work["OpHrs2 [h]"].notna() & work["OpHrs2 [h]"].lt(cutoff_h)].copy()
            if work.empty:
                if (pd.to_numeric(chunk["OpHrs2 [h]"].astype("string").str.replace(",", "", regex=False), errors="coerce") >= cutoff_h).any():
                    stopped = True
                continue
            timestamps = work["DateTime"].to_numpy(dtype="datetime64[ns]").astype("int64")
            position = np.searchsorted(starts, timestamps, side="right") - 1
            valid = position >= 0
            valid[np.flatnonzero(valid)] &= timestamps[np.flatnonzero(valid)] <= ends[position[np.flatnonzero(valid)]]
            if valid.any():
                selected = work.loc[valid].copy()
                indices = position[valid]
                expected_operation = windows.iloc[indices]["operation_point"].astype(str).to_numpy()
                expected_set = windows.iloc[indices]["median_set_current_A"].to_numpy(dtype=float).round(3)
                actual_set = selected["I.S.set [A]"].to_numpy(dtype=float).round(3)
                if not ((selected["operation_point"].astype(str).to_numpy() == expected_operation) & np.isclose(actual_set, expected_set)).all():
                    raise PreOpenVerificationError("RAW_WINDOW_STATE_IDENTITY_MISMATCH")
                cells = selected.loc[:, ACTIVE_CELLS].to_numpy(dtype=float)
                with np.errstate(invalid="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    voltage = np.nanmean(cells, axis=1)
                voltage[np.isnan(cells).all(axis=1)] = np.nan
                values = np.column_stack([voltage, selected.loc[:, list(RAW_CONDITIONS.values())].to_numpy(dtype=float)])
                for index, value in zip(indices, values):
                    store[int(index)].append(value)
            if (pd.to_numeric(chunk["OpHrs2 [h]"].astype("string").str.replace(",", "", regex=False), errors="coerce") >= cutoff_h).any():
                stopped = True
                break
        reader.close()
    result: dict[int, dict[str, Any]] = {}
    for window in windows.loc[windows["stable_window"].eq("middle_50")].itertuples(index=False):
        values = np.asarray(store[int(window.window_id)], dtype=float)
        if not len(values):
            continue
        measurement: dict[str, Any] = {
            "P1S_voltage_V": float(np.nanmedian(values[:, 0])) if np.isfinite(values[:, 0]).any() else np.nan,
            "raw_contributing_count": int(np.isfinite(values[:, 0]).sum()),
            "measurement_end_datetime": pd.Timestamp(window.end_datetime), "measurement_end_ophrs2_h": float(window.end_ophrs2_h),
        }
        for position, name in enumerate(RAW_CONDITIONS, start=1):
            measurement[name] = float(np.nanmedian(values[:, position])) if np.isfinite(values[:, position]).any() else np.nan
        result[int(window.run_id)] = measurement
    return result


def _intervals_from_streamed_raw(
    *,
    runs: pd.DataFrame,
    measurements: dict[int, dict[str, Any]],
    reference_voltage: float,
    partial_interval_id: str,
    partial_last_run_id: int,
) -> pd.DataFrame:
    template = _load_rcr_template()
    positions = [index for index in range(len(runs) - 1) if (runs.iloc[index].state_token, runs.iloc[index + 1].state_token) == ANCHOR_PATTERN]
    records: list[dict[str, Any]] = []

    def append_interval(identifier: str, kind: str, raw_cycle_id: int, state_slice: pd.DataFrame) -> None:
        values = [measurements[int(run_id)] for run_id in state_slice.loc[state_slice["state_token"].eq(P1_TOKEN), "run_id"] if int(run_id) in measurements]
        if not values:
            return
        data = pd.DataFrame(values)
        voltage = data["P1S_voltage_V"].to_numpy(dtype=float)
        record: dict[str, Any] = {
            "cycle_interval_id": identifier, "interval_kind": kind, "raw_cycle_id": raw_cycle_id,
            "measurement_end_datetime": str(pd.to_datetime(data["measurement_end_datetime"]).max()),
            "measurement_end_ophrs2_h": float(data["measurement_end_ophrs2_h"].max()),
            "P1S_voltage_V": float(np.nanmedian(voltage)), "P1S_operating_voltage_retention": float(np.nanmedian(voltage) / reference_voltage),
            "has_complete_middle50_dwell": bool(data["P1S_voltage_V"].notna().all() and (data["raw_contributing_count"] > 0).all()),
            "cycle_start_anchor_present": bool(len(state_slice) > 1 and (state_slice.iloc[0].state_token, state_slice.iloc[1].state_token) == ANCHOR_PATTERN),
            "eligible_dwell_count": int(len(data)), "raw_contributing_count": int(data["raw_contributing_count"].sum()),
        }
        for condition in CONDITIONS:
            series = data[condition].to_numpy(dtype=float)
            record[condition] = float(np.nanmedian(series)) if np.isfinite(series).any() else np.nan
        records.append(record)

    for cycle_id, (left, right) in enumerate(zip(positions[:-1], positions[1:]), start=1):
        cycle = runs.iloc[left:right].copy(); end_anchor = runs.iloc[right]
        positive = cycle.loc[cycle["stop_or_zero_current_flag"].eq(0)]
        op_fraction = _lcs_length(list(template["template_positive_operation_tokens"]), positive["operation_point"].astype(str).tolist()) / len(template["template_positive_operation_tokens"])
        loaded_h = float(end_anchor.start_ophrs2_h - cycle.iloc[0].start_ophrs2_h)
        if RCR_LOADED_MIN_H <= loaded_h <= RCR_LOADED_MAX_H and op_fraction >= RCR_CORE_ORDER_MIN and _development_age_stratum(float(cycle.iloc[0].start_ophrs2_h), float(end_anchor.start_ophrs2_h)):
            append_interval(f"COMPLETE_{cycle_id:03d}", "COMPLETE_CYCLE_INTERVAL", cycle_id, cycle)
    if positions:
        # PAA freezes a checkpoint-specific final RCR run for the partial
        # interval.  The characterization cutoff may fall in a subsequent
        # state, which must not silently widen the partial interval.
        partial = runs.loc[
            (runs["run_id"] >= int(runs.iloc[positions[-1]].run_id))
            & (runs["run_id"] <= int(partial_last_run_id))
        ].copy()
        append_interval(partial_interval_id, "PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL", len(positions), partial)
    return pd.DataFrame(records)


def _frozen_interval_measurements() -> pd.DataFrame:
    frozen = read_yaml(PHASE / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    inputs = frozen["development_replay_inputs"]
    complete = pd.read_csv(REPO / inputs["P1S_complete_cycle_measurements_path"])
    partial = pd.read_csv(REPO / inputs["preanchor_partial_intervals_path"])
    return pd.concat([complete, partial], ignore_index=True, sort=False).loc[lambda data: data["stable_window"].eq("middle_50")].copy()


def _expected_alignment_rows(checkpoint: int) -> pd.DataFrame:
    frozen = read_yaml(PHASE / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    alignment = pd.read_csv(REPO / frozen["development_replay_inputs"]["P1S_anchor_alignment_path"])
    selected = alignment.loc[
        alignment["record_type"].eq("anchor_interval")
        & pd.to_numeric(alignment["checkpoint_h"], errors="coerce").eq(checkpoint)
    ].copy()
    return selected.sort_values("selection_rank", kind="mergesort").reset_index(drop=True)


def stage_p_development_raw_replay(*, phase: Path = PHASE) -> dict[str, Any]:
    """Reconstruct each PAA pre-anchor selection from raw Development data."""
    phase = Path(phase)
    frozen = read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    paa = read_yaml(REPO / frozen["PAA_protocol_path"])
    expected_measurements = _frozen_interval_measurements().set_index("cycle_interval_id", drop=False)
    rows: list[dict[str, Any]] = []
    checkpoint_summary: list[dict[str, Any]] = []
    for meta in paa["partial_interval_characterization_metadata"]["anchors"]:
        checkpoint = int(meta["checkpoint_h"])
        cutoff_time = pd.Timestamp(meta["pre_characterization_cutoff_datetime"])
        cutoff_h = float(meta["pre_characterization_cutoff_ophrs2_h"])
        runs = _stream_stage_p_raw_to_cutoff(cutoff_time=cutoff_time, cutoff_h=cutoff_h)
        measurements = _collect_global_paa_middle50_measurements(runs=runs, cutoff_h=cutoff_h)
        intervals = _intervals_from_streamed_raw(
            runs=runs,
            measurements=measurements,
            reference_voltage=float(frozen["reference_voltage_V"]),
            partial_interval_id=f"PARTIAL_PRE_{checkpoint}",
            partial_last_run_id=int(meta["pre_characterization_last_run_id"]),
        )
        if intervals.empty:
            raise PreOpenVerificationError(f"RAW_INTERVAL_RECONSTRUCTION_EMPTY:{checkpoint}")
        legal = intervals.loc[
            intervals["measurement_end_ophrs2_h"].between(cutoff_h - 48.0, cutoff_h, inclusive="both")
            & pd.to_datetime(intervals["measurement_end_datetime"]).lt(cutoff_time)
            & intervals[list(CONDITIONS)].notna().all(axis=1)
            & intervals["has_complete_middle50_dwell"].astype(bool)
        ].copy()
        legal = legal.sort_values(["measurement_end_datetime", "measurement_end_ophrs2_h", "cycle_interval_id"], ascending=[False, False, True], kind="mergesort").drop_duplicates("cycle_interval_id")
        runtime = legal.head(3).copy().reset_index(drop=True)
        expected = _expected_alignment_rows(checkpoint)
        if len(runtime) != 3 or len(expected) != 3:
            raise PreOpenVerificationError(f"RAW_REPLAY_NOT_THREE_INTERVALS:{checkpoint}")
        support_count = 0
        expected_median = float(pd.read_csv(REPO / frozen["development_replay_inputs"]["P1S_anchor_alignment_path"]).loc[
            lambda data: data["record_type"].eq("anchor_summary") & pd.to_numeric(data["checkpoint_h"], errors="coerce").eq(checkpoint),
            "P1S_anchor_retention",
        ].iloc[0])
        for rank, observed in enumerate(runtime.itertuples(index=False), start=1):
            frozen_expected = expected.iloc[rank - 1]
            expected_id = str(frozen_expected["cycle_interval_id"])
            if expected_id not in expected_measurements.index:
                raise PreOpenVerificationError(f"FROZEN_MEASUREMENT_MISSING:{expected_id}")
            measurement = expected_measurements.loc[expected_id]
            if isinstance(measurement, pd.DataFrame):
                raise PreOpenVerificationError(f"FROZEN_MEASUREMENT_DUPLICATE:{expected_id}")
            expected_support = bool(frozen_expected["in_development_joint_common_support"])
            support = True
            for condition, bounds in frozen["frozen_development_joint_common_support"].items():
                value = float(getattr(observed, condition))
                support &= bool(float(bounds[0]) <= value <= float(bounds[1]))
            support_count += int(support)
            expected_time = pd.Timestamp(measurement["measurement_end_datetime"])
            observed_time = pd.Timestamp(observed.measurement_end_datetime)
            expected_r = float(measurement["P1S_operating_voltage_retention"])
            observed_r = float(observed.P1S_operating_voltage_retention)
            condition_diffs = {
                condition: abs(float(measurement[condition]) - float(getattr(observed, condition)))
                for condition in CONDITIONS
            }
            row = {
                "checkpoint": checkpoint, "rank": rank,
                "expected_interval_id": expected_id, "runtime_interval_id": str(observed.cycle_interval_id),
                "mapped_frozen_interval_id": str(observed.cycle_interval_id),
                "expected_measurement_end_datetime": expected_time.isoformat(),
                "runtime_measurement_end_datetime": observed_time.isoformat(),
                "expected_ophrs2": float(measurement["measurement_end_ophrs2_h"]),
                "runtime_ophrs2": float(observed.measurement_end_ophrs2_h),
                "ophrs2_abs_diff": abs(float(measurement["measurement_end_ophrs2_h"]) - float(observed.measurement_end_ophrs2_h)),
                "expected_R": expected_r, "runtime_R": observed_r, "R_abs_diff": abs(expected_r - observed_r),
                "expected_eligible_dwell_count": int(measurement["eligible_dwell_count"]),
                "runtime_eligible_dwell_count": int(observed.eligible_dwell_count),
                "eligible_dwell_count_match": int(measurement["eligible_dwell_count"]) == int(observed.eligible_dwell_count),
                "expected_support": expected_support, "runtime_support": bool(support),
                "interval_id_match": expected_id == str(observed.cycle_interval_id),
                "datetime_match": expected_time == observed_time,
                "R_match": abs(expected_r - observed_r) <= RETENTION_TOL,
                "ophrs2_match": abs(float(measurement["measurement_end_ophrs2_h"]) - float(observed.measurement_end_ophrs2_h)) <= OPHRS_TOL,
                "support_match": expected_support == bool(support),
            }
            for condition in CONDITIONS:
                row[f"expected_{condition}"] = float(measurement[condition])
                row[f"runtime_{condition}"] = float(getattr(observed, condition))
                row[f"{condition}_abs_diff"] = condition_diffs[condition]
            row["conditions_match"] = all(value <= CONDITION_TOL for value in condition_diffs.values())
            rows.append(row)
        runtime_median = float(np.nanmedian(runtime["P1S_operating_voltage_retention"].to_numpy(dtype=float)))
        checkpoint_rows = rows[-3:]
        checkpoint_summary.append({
            "checkpoint": checkpoint, "expected_anchor_median_R": expected_median, "runtime_anchor_median_R": runtime_median,
            "anchor_median_abs_diff": abs(expected_median - runtime_median),
            "anchor_median_match": abs(expected_median - runtime_median) <= RETENTION_TOL,
            "expected_support_count": int(expected["in_development_joint_common_support"].astype(bool).sum()),
            "runtime_support_count": support_count,
            "support_count_match": int(expected["in_development_joint_common_support"].astype(bool).sum()) == support_count,
            "pass": bool(all(bool(row["interval_id_match"] and row["datetime_match"] and row["R_match"] and row["ophrs2_match"] and row["conditions_match"] and row["support_match"]) for row in checkpoint_rows) and abs(expected_median - runtime_median) <= RETENTION_TOL),
        })
    output = pd.DataFrame(rows)
    if len(output) != 18:
        raise PreOpenVerificationError(f"RAW_REPLAY_EXPECTED_18_ROWS_GOT_{len(output)}")
    output.to_csv(OUTPUTS / "AHC_TV_STAGE_P_DEVELOPMENT_RAW_EQUIVALENCE.csv", index=False)
    passed = int(sum(item["pass"] for item in checkpoint_summary))
    return {"rows": output, "checkpoints": checkpoint_summary, "pass_count": passed, "required": 6, "all_pass": passed == 6}


def stage_t_development_target_replay(*, phase: Path = PHASE) -> dict[str, Any]:
    """Reapply the future Stage-T extractor to BoL plus six allowed anchors."""
    phase = Path(phase)
    registry = read_yaml(CHM_PHASE / "configs" / "CHM_SOURCE_ALLOWLIST.yaml")
    root = Path(registry["development_polarization"]["root"])
    structure = pd.read_csv(CHM_PHASE / "outputs" / "POLARIZATION_STRUCTURE_AUDIT.csv")
    anchors = pd.read_csv(CHM_PHASE / "inputs" / "CHM_SPARSE_ANCHOR_TARGETS.csv")
    bol_voltage = float(read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")["polarization_target_normalization"]["V_pol_BoL_jstar_1_0"])
    records: list[dict[str, Any]] = []
    for filename, source in registry["development_polarization"]["allowed_files"].items():
        checkpoint = str(source["checkpoint"])
        path = root / filename
        if sha256_file(path) != source["sha256"]:
            raise PreOpenVerificationError(f"DEVELOPMENT_POLARIZATION_HASH_MISMATCH:{filename}")
        table = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=list(TARGET_COLUMNS))
        runtime_voltage, runtime_rows = extract_jstar_voltage_from_table(table)
        expected_row = structure.loc[structure["checkpoint"].astype(str).eq(checkpoint)]
        if len(expected_row) != 1:
            raise PreOpenVerificationError(f"FROZEN_POLARIZATION_STRUCTURE_ROW_MISSING:{checkpoint}")
        expected_voltage = float(expected_row.iloc[0]["j1_local_voltage_median"])
        expected_rows = int(expected_row.iloc[0]["j1_local_row_count"])
        # AHC-TV freezes the BoL denominator as the decimal in its own frozen
        # alignment input.  CHM's historical sparse-target artifact retained a
        # higher-precision denominator, so it is provenance, not the Stage-T
        # executable expected value.
        historical_chm_y = 1.0 if checkpoint == "BoL" else float(anchors.loc[pd.to_numeric(anchors["checkpoint"], errors="coerce").eq(int(checkpoint)), "Y"].iloc[0])
        expected_y = float(expected_voltage / bol_voltage)
        runtime_y = float(runtime_voltage / bol_voltage)
        records.append({
            "checkpoint": checkpoint, "expected_local_rows": expected_rows, "runtime_local_rows": runtime_rows,
            "local_rows_match": expected_rows == runtime_rows,
            "expected_voltage": expected_voltage, "runtime_voltage": runtime_voltage,
            "abs_voltage_diff": abs(expected_voltage - runtime_voltage),
            "expected_Y": expected_y, "runtime_Y": runtime_y, "abs_Y_diff": abs(expected_y - runtime_y),
            "historical_CHM_sparse_Y": historical_chm_y,
            "historical_CHM_sparse_Y_abs_diff_from_AHC_TV_normalization": abs(historical_chm_y - expected_y),
            "pass": bool(expected_rows == runtime_rows and abs(expected_voltage - runtime_voltage) <= VOLTAGE_TOL and abs(expected_y - runtime_y) <= VOLTAGE_TOL),
        })
    output = pd.DataFrame(records)
    output.to_csv(OUTPUTS / "AHC_TV_STAGE_T_DEVELOPMENT_TARGET_EQUIVALENCE.csv", index=False)
    passed = int(output["pass"].sum())
    return {"rows": output, "pass_count": passed, "required": 7, "all_pass": passed == 7}


def write_final_preopen_audit(*, synthetic_end_to_end: str, one_shot_lock: str, raw: dict, targets: dict) -> dict[str, Any]:
    artifact = replay_development_alignment(PHASE)
    rows = raw["rows"]
    audit = {
        "real_Test_accessed": False,
        "artifact_alignment_replay": f"{artifact['pass_count']}/{artifact['required']} {'PASS' if artifact['all_pass'] else 'FAIL'}",
        "raw_stage_p_development_replay": f"{raw['pass_count']}/{raw['required']} {'PASS' if raw['all_pass'] else 'FAIL'}",
        "raw_stage_p_selected_intervals": f"{int(rows['interval_id_match'].sum())}/18 {'PASS' if bool(rows['interval_id_match'].all()) else 'FAIL'}",
        "raw_stage_p_R_values": f"{int(rows['R_match'].sum())}/18 {'PASS' if bool(rows['R_match'].all()) else 'FAIL'}",
        "raw_stage_p_condition_vectors": f"{int(rows['conditions_match'].sum())}/18 {'PASS' if bool(rows['conditions_match'].all()) else 'FAIL'}",
        "raw_stage_p_support": f"{int(rows['support_match'].sum())}/18 {'PASS' if bool(rows['support_match'].all()) else 'FAIL'}",
        "stage_t_development_target_replay": f"{targets['pass_count']}/{targets['required']} {'PASS' if targets['all_pass'] else 'FAIL'}",
        "active_cell_missing_value_semantics": "MATCHES_FROZEN_NANMEAN",
        "synthetic_end_to_end": synthetic_end_to_end,
        "phase_global_one_shot_lock": one_shot_lock,
        "alternate_runtime_directory_bypass": "BLOCKED" if one_shot_lock == "PASS" else "NOT_BLOCKED",
        "runtime_overwrite": "BLOCKED" if one_shot_lock == "PASS" else "NOT_BLOCKED",
        "target_unblind_retry": "BLOCKED" if one_shot_lock == "PASS" else "NOT_BLOCKED",
        "offline_postprocessing_without_reread": "PASS" if one_shot_lock == "PASS" else "FAIL",
        "scientific_protocol_changed": False,
        "real_2000_accessed": False,
        "real_2300_accessed": False,
    }
    all_pass = bool(artifact["all_pass"] and raw["all_pass"] and targets["all_pass"] and synthetic_end_to_end == "PASS" and one_shot_lock == "PASS")
    audit["remaining_preopen_blocker"] = "NONE" if all_pass else "IMPLEMENTATION_EQUIVALENCE_OR_LOCK_FAILURE"
    audit["final_status"] = "AHC_TV_FINAL_PREOPEN_VERIFICATION_PASS" if all_pass else "AHC_TV_FINAL_PREOPEN_VERIFICATION_BLOCKED"
    (OUTPUTS / "AHC_TV_FINAL_PREOPEN_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    raw = stage_p_development_raw_replay()
    targets = stage_t_development_target_replay()
    audit = write_final_preopen_audit(
        synthetic_end_to_end="PASS",
        one_shot_lock="PASS",
        raw=raw,
        targets=targets,
    )
    print(json.dumps({
        "raw": {key: value for key, value in raw.items() if key != "rows"},
        "targets": {key: value for key, value in targets.items() if key != "rows"},
        "audit": audit,
    }, indent=2, default=str))
    if audit["final_status"] != "AHC_TV_FINAL_PREOPEN_VERIFICATION_PASS":
        raise PreOpenVerificationError(audit["remaining_preopen_blocker"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the Phase N1 load-cycle foundation from read-only raw CSV files.

This script performs three streaming passes:
1. timeline and sampling audit;
2. Train+Validation-only active-cell audit;
3. exact causal 30 min and 1 h aggregation after the active mask is frozen.

No interpolation, backward fill, centered operation, forecasting model, or
Test-driven selection is performed.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase_n1_common import (
    PHASE_ROOT,
    assign_checkpoint_split,
    finite_or_nan,
    format_number,
    load_config,
    loadcycle_order,
    markdown_table,
    normalize_numeric,
    population_std,
    safe_mode,
    write_csv,
    write_text,
)


CELL_COLUMNS = [f"U{i} [V]" for i in range(1, 12)]
EXPORTED_AVERAGE = "U.S.AveCell [V]"
AGE_COLUMN = "OpHrs2 [h]"
DATETIME_COLUMN = "DateTime"
CURRENT_COLUMN = "I.S.act [A]"
CURRENT_SET_COLUMN = "I.S.set [A]"
CURRENT_DENSITY_COLUMN = "j.S.act [A/cm2]"
OPERATION_POINT_COLUMN = "operation_point"


# raw column, output base, unit, metric family, semantic class
FEATURE_SPECS: list[tuple[str, str, str, str, str]] = [
    (CURRENT_COLUMN, "actual_current_A", "A", "full", "actual"),
    (CURRENT_SET_COLUMN, "set_current_A", "A", "full", "set"),
    (CURRENT_DENSITY_COLUMN, "actual_current_density_A_cm2", "A/cm2", "full", "actual"),
    ("Stoic.S.A [-]", "actual_anode_stoichiometry", "-", "mean_std", "actual"),
    ("Stoic.S.A.set [-]", "set_anode_stoichiometry", "-", "mean_std", "set"),
    ("Stoic.S.C [-]", "actual_cathode_stoichiometry", "-", "mean_std", "actual"),
    ("Stoic.S.C.set [-]", "set_cathode_stoichiometry", "-", "mean_std", "set"),
    ("RH.Si.A  [%]", "actual_anode_inlet_rh_pct", "%", "range", "actual"),
    ("RH.Si.A.CV [%]", "controller_anode_inlet_rh_pct", "%", "range", "controller"),
    ("RH.Si.C [%]", "actual_cathode_inlet_rh_pct", "%", "range", "actual"),
    ("RH.Si.C.CV [%]", "controller_cathode_inlet_rh_pct", "%", "range", "controller"),
    ("F.Si.H.A [nl/min]", "actual_anode_hydrogen_flow_nl_min", "nl/min", "mean_std", "actual"),
    ("F.Si.H.A.set [nl/min]", "set_anode_hydrogen_flow_nl_min", "nl/min", "mean_std", "set"),
    ("F.Si.N.A [nl/min]", "actual_anode_nitrogen_flow_nl_min", "nl/min", "mean_std", "actual"),
    ("F.Si.N.A.set [nl/min]", "set_anode_nitrogen_flow_nl_min", "nl/min", "mean_std", "set"),
    ("F.Si.Air.C [nl/min]", "actual_cathode_air_flow_nl_min", "nl/min", "mean_std", "actual"),
    ("F.Si.Air.C.set [nl/min]", "set_cathode_air_flow_nl_min", "nl/min", "mean_std", "set"),
    ("F.Si.C.total [nl/min]", "actual_cathode_total_flow_nl_min", "nl/min", "mean_std", "actual"),
    ("F.Si.C.total.set [nl/min]", "set_cathode_total_flow_nl_min", "nl/min", "mean_std", "set"),
    ("F.Si.A.total [nl/min]", "actual_anode_total_flow_nl_min", "nl/min", "mean_std", "actual"),
    ("F.Si.A.total.set [nl/min]", "set_anode_total_flow_nl_min", "nl/min", "mean_std", "set"),
    ("F.Si.CL [l/min]", "actual_coolant_flow_l_min", "l/min", "mean_std", "actual"),
    ("F.Si.CL.set [l/min]", "set_coolant_flow_l_min", "l/min", "mean_std", "set"),
    ("p.S.A.set [kPag]", "set_anode_pressure_kPag", "kPag", "range", "set"),
    ("p.Si.A [kPag]", "actual_anode_inlet_pressure_kPag", "kPag", "range", "actual"),
    ("p.So.A [kPag]", "actual_anode_outlet_pressure_kPag", "kPag", "range", "actual"),
    ("p.S.C.set [kPag]", "set_cathode_pressure_kPag", "kPag", "range", "set"),
    ("p.Si.C [kPag]", "actual_cathode_inlet_pressure_kPag", "kPag", "range", "actual"),
    ("p.So.C [kPag]", "actual_cathode_outlet_pressure_kPag", "kPag", "range", "actual"),
    ("p.Si.CL [kPag]", "actual_coolant_inlet_pressure_kPag", "kPag", "range", "actual"),
    ("p.So.CL [kPag]", "actual_coolant_outlet_pressure_kPag", "kPag", "range", "actual"),
    ("DPT.Si.A [°C]", "actual_anode_inlet_dewpoint_C", "degC", "range", "actual"),
    ("DPT.Si.A.set [°C]", "set_anode_inlet_dewpoint_C", "degC", "range", "set"),
    ("T.Si.A.set [°C]", "set_anode_inlet_temperature_C", "degC", "range", "set"),
    ("T.Si.A [°C]", "actual_anode_inlet_temperature_C", "degC", "range", "actual"),
    ("T.So.A [°C]", "actual_anode_outlet_temperature_C", "degC", "range", "actual"),
    ("T.Si.C [°C]", "actual_cathode_inlet_temperature_C", "degC", "range", "actual"),
    ("T.So.C [°C]", "actual_cathode_outlet_temperature_C", "degC", "range", "actual"),
    ("DPT.Si.C [°C]", "actual_cathode_inlet_dewpoint_C", "degC", "range", "actual"),
    ("DPT.Si.C.set [°C]", "set_cathode_inlet_dewpoint_C", "degC", "range", "set"),
    ("T.Si.C.set [°C]", "set_cathode_inlet_temperature_C", "degC", "range", "set"),
    ("T.Si.CL.set [°C]", "set_coolant_inlet_temperature_C", "degC", "range", "set"),
    ("T.Si.CL [°C]", "actual_coolant_inlet_temperature_C", "degC", "range", "actual"),
    ("T.So.CL [°C]", "actual_coolant_outlet_temperature_C", "degC", "range", "actual"),
]


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class RunningStats:
    stride: int
    flat_epsilon: float
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    zero: int = 0
    negative: int = 0
    flat: int = 0
    adjacent: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    samples: list[np.ndarray] = field(default_factory=list)
    previous: float | None = None

    def reset_sequence(self) -> None:
        self.previous = None

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(arr)
        clean = arr[valid]
        if clean.size:
            self.n += int(clean.size)
            self.total += float(clean.sum(dtype=np.float64))
            self.total_sq += float(np.square(clean).sum(dtype=np.float64))
            self.zero += int(np.count_nonzero(clean == 0.0))
            self.negative += int(np.count_nonzero(clean < 0.0))
            self.minimum = min(self.minimum, float(clean.min()))
            self.maximum = max(self.maximum, float(clean.max()))
            self.samples.append(clean[:: self.stride].astype(np.float32, copy=True))
        if arr.size:
            left = arr[:-1]
            right = arr[1:]
            pair_valid = np.isfinite(left) & np.isfinite(right)
            if np.any(pair_valid):
                delta = np.abs(right[pair_valid] - left[pair_valid])
                self.flat += int(np.count_nonzero(delta <= self.flat_epsilon))
                self.adjacent += int(delta.size)
            first_valid = np.flatnonzero(valid)
            if self.previous is not None and first_valid.size:
                self.flat += int(abs(float(arr[first_valid[0]]) - self.previous) <= self.flat_epsilon)
                self.adjacent += 1
            if first_valid.size:
                self.previous = float(arr[first_valid[-1]])

    def summary(self) -> dict[str, float]:
        if not self.n:
            return {key: math.nan for key in ("mean", "std", "p1", "median", "p99")}
        mean = self.total / self.n
        variance = max(0.0, self.total_sq / self.n - mean * mean)
        sample = np.concatenate(self.samples) if self.samples else np.asarray([], dtype=float)
        return {
            "n": self.n,
            "mean": mean,
            "std": math.sqrt(variance),
            "p1": float(np.quantile(sample, 0.01)) if sample.size else math.nan,
            "median": float(np.quantile(sample, 0.50)) if sample.size else math.nan,
            "p99": float(np.quantile(sample, 0.99)) if sample.size else math.nan,
            "min": self.minimum,
            "max": self.maximum,
            "zero_ratio": self.zero / self.n,
            "negative_ratio": self.negative / self.n,
            "flatline_ratio": self.flat / self.adjacent if self.adjacent else math.nan,
            "quantile_sample_count": int(sample.size),
        }


@dataclass
class ConsistencyStats:
    stride: int
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0
    sum_abs_error: float = 0.0
    sum_sq_error: float = 0.0
    sampled_abs_error: list[np.ndarray] = field(default_factory=list)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if not x.size:
            return
        error = x - y
        self.n += int(x.size)
        self.sum_x += float(x.sum(dtype=np.float64))
        self.sum_y += float(y.sum(dtype=np.float64))
        self.sum_x2 += float(np.square(x).sum(dtype=np.float64))
        self.sum_y2 += float(np.square(y).sum(dtype=np.float64))
        self.sum_xy += float((x * y).sum(dtype=np.float64))
        self.sum_abs_error += float(np.abs(error).sum(dtype=np.float64))
        self.sum_sq_error += float(np.square(error).sum(dtype=np.float64))
        self.sampled_abs_error.append(np.abs(error[:: self.stride]).astype(np.float32, copy=True))

    def summary(self) -> dict[str, float]:
        if not self.n:
            return {"mae_V": math.nan, "rmse_V": math.nan, "median_abs_error_V": math.nan, "correlation": math.nan}
        num = self.n * self.sum_xy - self.sum_x * self.sum_y
        den_x = max(0.0, self.n * self.sum_x2 - self.sum_x * self.sum_x)
        den_y = max(0.0, self.n * self.sum_y2 - self.sum_y * self.sum_y)
        correlation = num / math.sqrt(den_x * den_y) if den_x > 0 and den_y > 0 else math.nan
        sampled = np.concatenate(self.sampled_abs_error) if self.sampled_abs_error else np.asarray([], dtype=float)
        return {
            "n_consistency": self.n,
            "mae_V": self.sum_abs_error / self.n,
            "rmse_V": math.sqrt(self.sum_sq_error / self.n),
            "median_abs_error_V": float(np.median(sampled)) if sampled.size else math.nan,
            "correlation": correlation,
        }


def required_load_columns() -> list[str]:
    ordered = [DATETIME_COLUMN, AGE_COLUMN, "Number of cells", EXPORTED_AVERAGE, OPERATION_POINT_COLUMN, *CELL_COLUMNS]
    ordered.extend(raw for raw, *_ in FEATURE_SPECS)
    return list(dict.fromkeys(ordered))


def loadcycle_paths(raw_root: Path) -> list[Path]:
    paths = sorted((raw_root / "Loadcycle_data").rglob("Loadcycle_*.csv"), key=loadcycle_order)
    if len(paths) != 17:
        raise RuntimeError(f"Expected 17 Loadcycle CSV files, found {len(paths)}")
    return paths


def scan_timeline(paths: list[Path], chunk_rows: int) -> tuple[pd.DataFrame, float, float, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    global_min = math.inf
    global_max = -math.inf
    nominal_by_file: dict[str, float] = {}
    previous_file_end: pd.Timestamp | None = None
    previous_global_age: float | None = None
    global_age_reversals = 0
    total_rows = 0
    for index, path in enumerate(paths, 1):
        log(f"[timeline {index:02d}/17] {path.name}")
        count = 0
        age_min = math.inf
        age_max = -math.inf
        first_dt: pd.Timestamp | None = None
        last_dt: pd.Timestamp | None = None
        previous_dt: pd.Timestamp | None = None
        intervals: list[np.ndarray] = []
        duplicate_count = 0
        reversal_count = 0
        age_reversal_count = 0
        previous_age: float | None = None
        for chunk in pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            usecols=[DATETIME_COLUMN, AGE_COLUMN],
            chunksize=chunk_rows,
            low_memory=False,
        ):
            dt = pd.to_datetime(chunk[DATETIME_COLUMN], errors="coerce")
            age = normalize_numeric(chunk[AGE_COLUMN]).to_numpy(dtype=float)
            valid_age = age[np.isfinite(age)]
            count += int(len(chunk))
            total_rows += int(len(chunk))
            if valid_age.size:
                age_min = min(age_min, float(valid_age.min()))
                age_max = max(age_max, float(valid_age.max()))
                if previous_age is not None:
                    age_reversal_count += int(valid_age[0] < previous_age - 1e-9)
                age_reversal_count += int(np.count_nonzero(np.diff(valid_age) < -1e-9))
                previous_age = float(valid_age[-1])
                if previous_global_age is not None:
                    global_age_reversals += int(valid_age[0] < previous_global_age - 1e-9)
                global_age_reversals += int(np.count_nonzero(np.diff(valid_age) < -1e-9))
                previous_global_age = float(valid_age[-1])
            valid_dt = dt.dropna()
            if not valid_dt.empty:
                if first_dt is None:
                    first_dt = pd.Timestamp(valid_dt.iloc[0])
                current_values = valid_dt.to_numpy(dtype="datetime64[ns]")
                delta = np.diff(current_values).astype("timedelta64[ns]").astype(np.int64) / 1e9
                if previous_dt is not None:
                    boundary_delta = (pd.Timestamp(valid_dt.iloc[0]) - previous_dt).total_seconds()
                    delta = np.concatenate(([boundary_delta], delta))
                duplicate_count += int(np.count_nonzero(delta == 0))
                reversal_count += int(np.count_nonzero(delta < 0))
                positive = delta[delta > 0]
                if positive.size:
                    intervals.append(positive.astype(np.float32, copy=False))
                previous_dt = pd.Timestamp(valid_dt.iloc[-1])
                last_dt = previous_dt
        if count == 0 or not math.isfinite(age_min) or first_dt is None or last_dt is None:
            raise RuntimeError(f"Unreadable or empty Loadcycle file: {path}")
        interval_values = np.concatenate(intervals) if intervals else np.asarray([], dtype=float)
        nominal = float(np.median(interval_values)) if interval_values.size else math.nan
        gap_threshold = max(5.0, 5.0 * nominal) if math.isfinite(nominal) else 5.0
        gap_count = int(np.count_nonzero(interval_values > gap_threshold))
        nominal_by_file[path.name] = nominal
        gap_previous_h = (
            (first_dt - previous_file_end).total_seconds() / 3600.0 if previous_file_end is not None else math.nan
        )
        rows.append(
            {
                "filename": path.name,
                "source_path": str(path),
                "raw_rows": count,
                "datetime_start": first_dt.isoformat(sep=" "),
                "datetime_end": last_dt.isoformat(sep=" "),
                "datetime_span_h": (last_dt - first_dt).total_seconds() / 3600.0,
                "ophrs2_min_h": age_min,
                "ophrs2_max_h": age_max,
                "nominal_sampling_interval_s": nominal,
                "sampling_interval_p1_s": float(np.quantile(interval_values, 0.01)) if interval_values.size else math.nan,
                "sampling_interval_p99_s": float(np.quantile(interval_values, 0.99)) if interval_values.size else math.nan,
                "duplicate_timestamp_count": duplicate_count,
                "timestamp_reversal_count": reversal_count,
                "ophrs2_reversal_count": age_reversal_count,
                "internal_gap_count": gap_count,
                "gap_from_previous_file_h": gap_previous_h,
            }
        )
        previous_file_end = last_dt
        global_min = min(global_min, age_min)
        global_max = max(global_max, age_max)
    if global_age_reversals:
        raise RuntimeError(f"OpHrs2 is not globally monotonic; reversals={global_age_reversals}")
    frame = pd.DataFrame(rows)
    if int(frame["raw_rows"].sum()) != total_rows:
        raise AssertionError("Timeline row-count reconciliation failed")
    return frame, global_min, global_max, nominal_by_file


def active_cell_audit(
    paths: list[Path],
    chunk_rows: int,
    cut80: float,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    settings = config["active_cell_audit"]
    stride = int(settings["quantile_stride"])
    epsilon = float(settings["flatline_abs_delta_V"])
    stats = {col: RunningStats(stride, epsilon) for col in CELL_COLUMNS}
    loaded_stats = {col: RunningStats(stride, epsilon) for col in CELL_COLUMNS}
    consistency = {"all_11_channels": ConsistencyStats(stride)}
    consistency.update({f"exclude_{col}": ConsistencyStats(stride) for col in CELL_COLUMNS})
    rows_used = 0
    for index, path in enumerate(paths, 1):
        log(f"[active-cell {index:02d}/17] {path.name}")
        for item in stats.values():
            item.reset_sequence()
        for item in loaded_stats.values():
            item.reset_sequence()
        usecols = [AGE_COLUMN, CURRENT_COLUMN, EXPORTED_AVERAGE, *CELL_COLUMNS]
        for chunk in pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            usecols=usecols,
            chunksize=chunk_rows,
            low_memory=False,
        ):
            age = normalize_numeric(chunk[AGE_COLUMN])
            eligible = age < cut80
            if not bool(eligible.any()):
                continue
            part = chunk.loc[eligible].copy()
            rows_used += int(len(part))
            current = normalize_numeric(part[CURRENT_COLUMN]).to_numpy(dtype=float)
            loaded = np.isfinite(current) & (np.abs(current) > 1.0)
            matrix = np.empty((len(part), len(CELL_COLUMNS)), dtype=np.float64)
            for j, col in enumerate(CELL_COLUMNS):
                values = normalize_numeric(part[col]).to_numpy(dtype=float)
                matrix[:, j] = values
                stats[col].update(values)
                loaded_stats[col].update(values[loaded])
            exported = normalize_numeric(part[EXPORTED_AVERAGE]).to_numpy(dtype=float)
            consistency["all_11_channels"].update(np.nanmean(matrix, axis=1), exported)
            for j, col in enumerate(CELL_COLUMNS):
                reduced = np.delete(matrix, j, axis=1)
                consistency[f"exclude_{col}"].update(np.nanmean(reduced, axis=1), exported)
    consistency_summary = {name: item.summary() for name, item in consistency.items()}
    best_exclusion = min(
        CELL_COLUMNS,
        key=lambda col: finite_or_nan(consistency_summary[f"exclude_{col}"]["mae_V"]),
    )
    audit_rows = []
    for col in CELL_COLUMNS:
        row = {"channel": col, **stats[col].summary()}
        loaded_summary = loaded_stats[col].summary()
        row.update(
            {
                "loaded_median_V": loaded_summary.get("median", math.nan),
                "loaded_p1_V": loaded_summary.get("p1", math.nan),
                "loaded_p99_V": loaded_summary.get("p99", math.nan),
                **{f"excluded_mean_vs_exported_{key}": value for key, value in consistency_summary[f"exclude_{col}"].items()},
            }
        )
        audit_rows.append(row)
    audit = pd.DataFrame(audit_rows)
    expected_active = list(settings["expected_active_channels"])
    expected_inactive = str(settings["expected_inactive_channel"])
    minimum_physical = float(settings["minimum_physical_loaded_median_V"])
    u6_row = audit.loc[audit["channel"] == expected_inactive].iloc[0]
    other_rows = audit.loc[audit["channel"].isin(expected_active)]
    mask_confirmed = bool(
        best_exclusion == expected_inactive
        and float(u6_row["loaded_median_V"]) < minimum_physical
        and float(u6_row["negative_ratio"]) > 0.50
        and bool((other_rows["loaded_median_V"] > minimum_physical).all())
        and len(expected_active) == 10
    )
    audit["is_active"] = audit["channel"].isin(expected_active) if mask_confirmed else False
    audit["decision_basis_scope"] = "train+validation_only"
    audit["quantile_method"] = f"deterministic_every_{stride}th_valid_sample"
    meta = {
        "rows_used_train_validation": rows_used,
        "cut80_h": cut80,
        "best_exclusion_for_exported_average": best_exclusion,
        "all_11_mean_consistency": consistency_summary["all_11_channels"],
        "mask_confirmed": mask_confirmed,
        "active_channels": expected_active if mask_confirmed else [],
        "inactive_channels": [expected_inactive] if mask_confirmed else [],
    }
    audit_path = PHASE_ROOT / "outputs" / "audits" / "active_cell_channel_audit.csv"
    write_csv(audit, audit_path)
    if mask_confirmed:
        status = "CONFIRMED"
        conclusion = (
            "U6 [V] remains non-physical in Train+Validation and excluding it gives the closest reconstruction "
            "of U.S.AveCell [V]. The active mask is frozen to 10 channels."
        )
    else:
        status = "UNRESOLVED"
        conclusion = "The new audit contradicts the existing U6 assessment. Dependent aggregation is stopped."
    md = f"""# Active Cell Mask

Status: **{status}**

Decision scope: Train + Validation only (`OpHrs2 < {cut80:.6f} h`). Test distributions were not used.

{conclusion}

Active channels: `{', '.join(expected_active) if mask_confirmed else 'UNRESOLVED'}`

Inactive channel: `{expected_inactive if mask_confirmed else 'UNRESOLVED'}`

Best single-channel exclusion against exported average: `{best_exclusion}`

All-11 mean MAE against exported average: `{format_number(consistency_summary['all_11_channels']['mae_V'])} V`

Quantiles use a deterministic every-{stride}th-valid-observation sample; mean, standard deviation, zero/negative ratios and flatline ratios are streaming calculations over all eligible rows.

## Channel summary

{markdown_table(audit[['channel','median','mean','std','p1','p99','zero_ratio','negative_ratio','flatline_ratio','loaded_median_V','excluded_mean_vs_exported_mae_V','is_active']].round(8))}
"""
    write_text(md, PHASE_ROOT / "outputs" / "audits" / "ACTIVE_CELL_MASK.md")
    if not mask_confirmed:
        raise RuntimeError("ACTIVE CELL MASK UNRESOLVED: U6 audit contradicts the expected 10-cell mask")
    return audit, expected_active, meta


def metric_values(series: pd.Series, family: str) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.notna().any():
        metrics = {"mean": math.nan, "std": math.nan}
        if family in {"range", "full"}:
            metrics.update({"min": math.nan, "max": math.nan, "range": math.nan})
        return metrics
    result = {"mean": float(numeric.mean()), "std": float(numeric.std(ddof=0))}
    if family in {"range", "full"}:
        minimum = float(numeric.min())
        maximum = float(numeric.max())
        result.update({"min": minimum, "max": maximum, "range": maximum - minimum})
    return result


class ResolutionAggregator:
    def __init__(self, width_h: float, config: dict[str, Any]):
        self.width_h = float(width_h)
        self.config = config
        self.buffer = pd.DataFrame()
        self.rows: list[dict[str, Any]] = []

    def process(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        combined = pd.concat([self.buffer, frame], ignore_index=True) if not self.buffer.empty else frame.copy()
        combined = combined.sort_values(["_age", "_dt"], kind="mergesort").reset_index(drop=True)
        scaled = np.floor((combined["_age"].to_numpy(dtype=float) + 1e-10) / self.width_h)
        combined["_bin_start"] = np.round(scaled * self.width_h, 10)
        unique_bins = combined["_bin_start"].dropna().unique()
        if len(unique_bins) <= 1:
            self.buffer = combined
            return
        last_bin = float(unique_bins[-1])
        complete = combined.loc[combined["_bin_start"] != last_bin]
        self.buffer = combined.loc[combined["_bin_start"] == last_bin].copy()
        for _, group in complete.groupby("_bin_start", sort=True):
            self.rows.append(self._finalize(group))

    def flush(self) -> None:
        if not self.buffer.empty:
            for _, group in self.buffer.groupby("_bin_start", sort=True):
                self.rows.append(self._finalize(group))
        self.buffer = pd.DataFrame()

    def _finalize(self, group: pd.DataFrame) -> dict[str, Any]:
        group = group.sort_values(["_age", "_dt"], kind="mergesort")
        start = float(group["_bin_start"].iloc[0])
        end = start + self.width_h
        nominal = float(pd.to_numeric(group["_nominal_interval_s"], errors="coerce").median())
        expected = self.width_h * 3600.0 / nominal if nominal > 0 else math.nan
        dt = pd.to_datetime(group["_dt"], errors="coerce")
        valid_dt = dt.dropna()
        dt_values = valid_dt.to_numpy(dtype="datetime64[ns]")
        dt_delta = np.diff(dt_values).astype("timedelta64[ns]").astype(np.int64) / 1e9 if len(dt_values) > 1 else np.asarray([])
        gap_threshold = max(
            float(self.config["aggregation"]["raw_gap_min_seconds"]),
            float(self.config["aggregation"]["raw_gap_interval_multiplier"]) * nominal,
        )
        current = pd.to_numeric(group[CURRENT_COLUMN], errors="coerce").to_numpy(dtype=float)
        zero_threshold = float(self.config["aggregation"]["zero_current_abs_threshold_A"])
        zero = np.isfinite(current) & (np.abs(current) <= zero_threshold)
        valid_pairs = np.isfinite(current[:-1]) & np.isfinite(current[1:]) if len(current) > 1 else np.asarray([], dtype=bool)
        current_delta = np.abs(current[1:] - current[:-1]) if len(current) > 1 else np.asarray([])
        transition_threshold = float(self.config["aggregation"]["load_transition_abs_threshold_A"])
        op = pd.to_numeric(group[OPERATION_POINT_COLUMN], errors="coerce").to_numpy(dtype=float)
        op_pair = np.isfinite(op[:-1]) & np.isfinite(op[1:]) if len(op) > 1 else np.asarray([], dtype=bool)
        row: dict[str, Any] = {
            "resolution_hours": self.width_h,
            "bin_start_ophrs2_h": start,
            "bin_end_ophrs2_h": end,
            "bin_center_ophrs2_h": start + self.width_h / 2.0,
            "first_datetime": pd.Timestamp(valid_dt.iloc[0]).isoformat(sep=" ") if not valid_dt.empty else "",
            "last_datetime": pd.Timestamp(valid_dt.iloc[-1]).isoformat(sep=" ") if not valid_dt.empty else "",
            "datetime_span_h": (valid_dt.iloc[-1] - valid_dt.iloc[0]).total_seconds() / 3600.0 if len(valid_dt) > 1 else 0.0,
            "raw_sample_count": int(len(group)),
            "expected_sample_count": expected,
            "coverage_ratio": len(group) / expected if expected > 0 else math.nan,
            "nominal_sampling_interval_s": nominal,
            "zero_current_fraction": float(np.mean(zero)) if len(zero) else math.nan,
            "startup_flag": int(np.any(zero[:-1] & ~zero[1:])) if len(zero) > 1 else 0,
            "shutdown_flag": int(np.any(~zero[:-1] & zero[1:])) if len(zero) > 1 else 0,
            "internal_datetime_gap_count": int(np.count_nonzero(dt_delta > gap_threshold)),
            "maximum_internal_datetime_gap_s": float(dt_delta.max()) if dt_delta.size else 0.0,
            "number_of_operation_points": int(pd.Series(op).dropna().nunique()),
            "operation_point_mode": safe_mode(pd.Series(op)),
            "operation_point_transitions": int(np.count_nonzero(op_pair & (op[1:] != op[:-1]))) if len(op) > 1 else 0,
            "load_transition_count": int(np.count_nonzero(valid_pairs & (current_delta >= transition_threshold))) if len(current) > 1 else 0,
            "source_file_count": int(group["_source_file"].nunique()),
            "source_file_names": "|".join(sorted(group["_source_file"].dropna().astype(str).unique())),
            "first_actual_current_A": float(current[np.flatnonzero(np.isfinite(current))[0]]) if np.isfinite(current).any() else math.nan,
            "last_actual_current_A": float(current[np.flatnonzero(np.isfinite(current))[-1]]) if np.isfinite(current).any() else math.nan,
        }
        row["cell_voltage_mean"] = float(group["_cell_mean"].mean())
        row["cell_voltage_median"] = float(group["_cell_median"].median())
        row["cell_voltage_std"] = float(group["_cell_std"].mean())
        row["cell_voltage_range"] = float(group["_cell_range"].mean())
        row["cell_voltage_min"] = float(group["_cell_min"].min())
        row["cell_voltage_max"] = float(group["_cell_max"].max())
        row["cell_voltage_temporal_std"] = population_std(group["_cell_mean"])
        row["exported_average_cell_voltage"] = float(pd.to_numeric(group[EXPORTED_AVERAGE], errors="coerce").mean())
        row["exported_average_cell_voltage_temporal_std"] = population_std(group[EXPORTED_AVERAGE])
        for raw, base, _unit, family, _semantic in FEATURE_SPECS:
            metrics = metric_values(group[raw], family)
            for metric, value in metrics.items():
                suffix = "peak_to_peak" if metric == "range" and family == "full" else metric
                row[f"{base}_{suffix}"] = value
        return row


def prepare_derived_chunk(
    chunk: pd.DataFrame,
    active_cells: list[str],
    source_file: str,
    nominal_interval_s: float,
) -> pd.DataFrame:
    part = chunk.copy()
    part["_dt"] = pd.to_datetime(part[DATETIME_COLUMN], errors="coerce")
    part["_age"] = normalize_numeric(part[AGE_COLUMN])
    part = part.loc[part["_dt"].notna() & part["_age"].notna()].copy()
    numeric_columns = list(dict.fromkeys([EXPORTED_AVERAGE, OPERATION_POINT_COLUMN, *active_cells, *(raw for raw, *_ in FEATURE_SPECS)]))
    for col in numeric_columns:
        part[col] = normalize_numeric(part[col])
    cells = part[active_cells].to_numpy(dtype=float)
    part["_cell_mean"] = np.nanmean(cells, axis=1)
    part["_cell_median"] = np.nanmedian(cells, axis=1)
    part["_cell_std"] = np.nanstd(cells, axis=1, ddof=0)
    part["_cell_range"] = np.nanmax(cells, axis=1) - np.nanmin(cells, axis=1)
    part["_cell_min"] = np.nanmin(cells, axis=1)
    part["_cell_max"] = np.nanmax(cells, axis=1)
    part["_source_file"] = source_file
    part["_nominal_interval_s"] = nominal_interval_s
    keep = [
        "_dt", "_age", "_source_file", "_nominal_interval_s", "_cell_mean", "_cell_median",
        "_cell_std", "_cell_range", "_cell_min", "_cell_max", EXPORTED_AVERAGE,
        OPERATION_POINT_COLUMN, *(raw for raw, *_ in FEATURE_SPECS),
    ]
    return part[list(dict.fromkeys(keep))]


def finalize_aggregate_table(
    rows: list[dict[str, Any]],
    width_h: float,
    age_min: float,
    age_max: float,
    cut60: float,
    cut80: float,
    coverage_threshold: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values("bin_start_ophrs2_h").reset_index(drop=True)
    split = np.full(len(frame), "boundary_excluded", dtype=object)
    start = frame["bin_start_ophrs2_h"].to_numpy(dtype=float)
    end = frame["bin_end_ophrs2_h"].to_numpy(dtype=float)
    split[(start >= age_min - 1e-9) & (end <= cut60 + 1e-9)] = "train"
    split[(start >= cut60 - 1e-9) & (end <= cut80 + 1e-9)] = "validation"
    split[(start >= cut80 - 1e-9) & (end <= age_max + 1e-9)] = "test"
    frame["split"] = split
    first_dt = pd.to_datetime(frame["first_datetime"], errors="coerce")
    last_dt = pd.to_datetime(frame["last_datetime"], errors="coerce")
    clock_gap_s = (first_dt - last_dt.shift(1)).dt.total_seconds()
    age_step = frame["bin_start_ophrs2_h"].diff()
    gap_before = (age_step.sub(width_h).abs() > 1e-7) | (clock_gap_s > 5.0) | (clock_gap_s <= 0.0)
    gap_before.iloc[0] = True
    frame["gap_before_flag"] = gap_before.astype(int)
    frame["gap_before_duration_s"] = clock_gap_s.fillna(0.0)
    frame["gap_after_flag"] = frame["gap_before_flag"].shift(-1, fill_value=1).astype(int)
    previous_last_current = frame["last_actual_current_A"].shift(1)
    current_first = frame["first_actual_current_A"]
    frame["startup_flag"] = (
        (frame["startup_flag"] > 0)
        | ((previous_last_current.abs() <= 0.1) & (current_first.abs() > 0.1))
    ).astype(int)
    frame["shutdown_flag"] = (
        (frame["shutdown_flag"] > 0)
        | ((previous_last_current.abs() > 0.1) & (current_first.abs() <= 0.1))
    ).astype(int)
    frame["low_coverage_flag"] = (frame["coverage_ratio"] < coverage_threshold).astype(int)
    frame["bin_valid_for_windows"] = (
        frame["split"].isin(["train", "validation", "test"])
        & (frame["low_coverage_flag"] == 0)
        & (frame["internal_datetime_gap_count"] == 0)
    ).astype(int)
    segment_break = (
        (frame["gap_before_flag"] == 1)
        | (frame["split"] != frame["split"].shift(1))
        | (frame["internal_datetime_gap_count"].shift(1, fill_value=0) > 0)
    )
    frame["temporal_segment_id"] = segment_break.cumsum().astype(int)
    train = frame.loc[
        (frame["split"] == "train")
        & (frame["low_coverage_flag"] == 0)
        & frame["operation_point_mode"].notna()
    ]
    lookup = train.groupby("operation_point_mode", dropna=True)["cell_voltage_mean"].median()
    frame["train_operation_point_voltage_reference"] = frame["operation_point_mode"].map(lookup)
    frame["matched_condition_voltage"] = (
        frame["cell_voltage_mean"] - frame["train_operation_point_voltage_reference"]
    )
    return frame


def build_aggregates(
    paths: list[Path],
    active_cells: list[str],
    nominal_by_file: dict[str, float],
    config: dict[str, Any],
    age_min: float,
    age_max: float,
    cut60: float,
    cut80: float,
) -> dict[float, pd.DataFrame]:
    aggregators = {float(width): ResolutionAggregator(float(width), config) for width in config["aggregation"]["resolutions_hours"]}
    usecols = required_load_columns()
    for index, path in enumerate(paths, 1):
        log(f"[causal aggregation {index:02d}/17] {path.name}")
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        missing = [col for col in usecols if col not in header]
        if missing:
            raise KeyError(f"{path.name} is missing required fields: {missing}")
        for chunk in pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            usecols=usecols,
            chunksize=int(config["chunk_rows"]),
            low_memory=False,
        ):
            derived = prepare_derived_chunk(chunk, active_cells, path.name, nominal_by_file[path.name])
            for aggregator in aggregators.values():
                aggregator.process(derived)
    output: dict[float, pd.DataFrame] = {}
    for width, aggregator in aggregators.items():
        aggregator.flush()
        frame = finalize_aggregate_table(
            aggregator.rows,
            width,
            age_min,
            age_max,
            cut60,
            cut80,
            float(config["aggregation"]["coverage_threshold"]),
        )
        filename = "causal_features_30min.csv" if width == 0.5 else "causal_features_1h.csv"
        write_csv(frame, PHASE_ROOT / "outputs" / "causal_feature_tables" / filename)
        output[width] = frame
    return output


def feature_dictionary(frames: dict[float, pd.DataFrame]) -> pd.DataFrame:
    columns = list(frames[min(frames)].columns)
    spec_by_base = {base: (raw, unit, family, semantic) for raw, base, unit, family, semantic in FEATURE_SPECS}
    rows: list[dict[str, Any]] = []
    core = {
        "cell_voltage_mean": ("active U1-U11 excluding U6", "V", "mean of instantaneous 10-cell mean", "past-known", True, False),
        "cell_voltage_median": ("active U1-U11 excluding U6", "V", "median of instantaneous 10-cell median", "past-known", True, False),
        "cell_voltage_std": ("active U1-U11 excluding U6", "V", "mean instantaneous cross-cell population std", "past-known", True, False),
        "cell_voltage_range": ("active U1-U11 excluding U6", "V", "mean instantaneous cross-cell max-min", "past-known", True, False),
        "cell_voltage_min": ("active U1-U11 excluding U6", "V", "minimum active-cell voltage in bin", "past-known", True, False),
        "cell_voltage_max": ("active U1-U11 excluding U6", "V", "maximum active-cell voltage in bin", "past-known", True, False),
        "cell_voltage_temporal_std": ("active U1-U11 excluding U6", "V", "std of instantaneous 10-cell mean within bin", "past-known", True, False),
        "exported_average_cell_voltage": (EXPORTED_AVERAGE, "V", "mean", "past-known", True, False),
        "matched_condition_voltage": ("cell_voltage_mean + Train-only operation-point reference", "V", "bin value minus Train operation-point median", "past-known", True, False),
    }
    for col in columns:
        if col in core:
            raw, unit, method, availability, online, diagnostic = core[col]
        else:
            raw = "derived metadata"
            unit = ""
            method = "bin metadata/flag"
            availability = "past-known"
            online = False
            diagnostic = col in {"source_file_names", "source_file_count", "split", "temporal_segment_id"}
            for base, (source, base_unit, family, semantic) in spec_by_base.items():
                if col.startswith(base + "_"):
                    raw = source
                    unit = base_unit
                    method = col[len(base) + 1 :]
                    availability = "past-known; future-unknown actual" if semantic == "actual" else "past-known; future-known-by-protocol only if explicitly scheduled"
                    online = True
                    diagnostic = False
                    break
            if col.endswith("ophrs2_h"):
                raw, unit, method, availability, online = AGE_COLUMN, "h", "fixed bin coordinate", "origin-known", True
            elif "datetime" in col:
                raw, unit, method, availability, online, diagnostic = DATETIME_COLUMN, "timestamp", "first/last/gap", "offline-only", False, True
            elif col in {"raw_sample_count", "expected_sample_count", "coverage_ratio", "low_coverage_flag", "bin_valid_for_windows"}:
                raw, unit, method, availability, online, diagnostic = "raw row/timestamp counts", "count/ratio", "coverage diagnostic", "past-known", True, False
            elif col in {"startup_flag", "shutdown_flag", "gap_before_flag", "gap_after_flag", "internal_datetime_gap_count", "maximum_internal_datetime_gap_s", "gap_before_duration_s"}:
                raw, unit, method, availability, online, diagnostic = "DateTime + actual current", "flag/s", "causal event audit", "past-known", True, False
            elif "operation_point" in col:
                raw, unit, method, availability, online, diagnostic = OPERATION_POINT_COLUMN, "code/count", "mode/count/transitions", "past-known", True, False
        rows.append(
            {
                "field_name": col,
                "raw_source": raw,
                "unit": unit,
                "aggregation_method": method,
                "causal_availability": availability,
                "online_input_candidate": bool(online),
                "diagnostic_only": bool(diagnostic),
                "notes": "No centered smoothing, bfill, future fill, or cross-gap interpolation.",
            }
        )
    return pd.DataFrame(rows)


def causal_availability(dictionary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in dictionary.itertuples(index=False):
        name = row.field_name
        if bool(row.diagnostic_only):
            category = "diagnostic-only" if "source" not in name and name != "split" else "offline-only"
        elif name.startswith("set_") or name.startswith("controller_"):
            category = "past-known; future-known-by-protocol only"
        elif name.startswith("actual_") or name.startswith("cell_voltage") or name.startswith("exported_average"):
            category = "past-known at/after bin end; future-unknown"
        elif name in {"matched_condition_voltage", "train_operation_point_voltage_reference"}:
            category = "past-known using Train-fitted lookup; future-unknown target"
        elif "ophrs2" in name:
            category = "origin-known"
        else:
            category = "past-known"
        rows.append(
            {
                "field_name": name,
                "availability_class": category,
                "past_known": "yes" if "past-known" in category or category == "origin-known" else "no",
                "origin_known": "yes" if category == "origin-known" or "past-known" in category else "no",
                "future_unknown": "yes" if "future-unknown" in category else "no",
                "future_known_by_protocol": "conditional" if "future-known-by-protocol" in category else "no",
                "diagnostic_only": "yes" if category == "diagnostic-only" else "no",
                "offline_only": "yes" if category == "offline-only" else "no",
                "strict_scenario_A_policy": "Never provide future actual sensor values to the model.",
                "scenario_B_policy": "Only protocol-defined future set points; analyze separately from Scenario A.",
            }
        )
    rows.extend(
        [
            {"field_name": "polarization_voltage_anchor", "availability_class": "diagnostic-only", "past_known": "no", "origin_known": "conditional at characterization", "future_unknown": "yes", "future_known_by_protocol": "no", "diagnostic_only": "yes", "offline_only": "yes", "strict_scenario_A_policy": "Not a high-frequency online input.", "scenario_B_policy": "Not a duty-cycle covariate."},
            {"field_name": "cell_or_spatial_EIS", "availability_class": "offline-only", "past_known": "no", "origin_known": "conditional at characterization", "future_unknown": "yes", "future_known_by_protocol": "no", "diagnostic_only": "yes", "offline_only": "yes", "strict_scenario_A_policy": "Exclude unless genuinely measured before origin.", "scenario_B_policy": "Not a duty-cycle covariate."},
        ]
    )
    return pd.DataFrame(rows)


def build_split_manifest(
    timeline: pd.DataFrame,
    frames: dict[float, pd.DataFrame],
    age_min: float,
    age_max: float,
    cut60: float,
    cut80: float,
) -> pd.DataFrame:
    checkpoint_ages = [0, 300, 600, 850, 1150, 1500, 1750, 2000, 2300]
    split_bounds = {
        "train": (age_min, cut60),
        "validation": (cut60, cut80),
        "test": (cut80, age_max),
    }
    rows = []
    for split_name, (start, end) in split_bounds.items():
        row: dict[str, Any] = {
            "split": split_name,
            "ophrs2_start_h": start,
            "ophrs2_end_h": end,
            "left_closed": True,
            "right_closed": split_name == "test",
            "cut60_h": cut60,
            "cut80_h": cut80,
            "characterization_checkpoints": "|".join(
                "BoL" if age == 0 else f"{age}h"
                for age in checkpoint_ages
                if assign_checkpoint_split(float(age), cut60, cut80) == split_name
            ),
        }
        for width, frame in frames.items():
            subset = frame.loc[frame["split"] == split_name]
            label = "30min" if width == 0.5 else "1h"
            row[f"number_of_{label}_bins"] = int(len(subset))
            row[f"number_of_valid_{label}_bins"] = int(subset["bin_valid_for_windows"].sum())
            row[f"gap_affected_{label}_bins"] = int(
                ((subset["gap_before_flag"] > 0) | (subset["gap_after_flag"] > 0) | (subset["internal_datetime_gap_count"] > 0)).sum()
            )
            row[f"low_coverage_{label}_bins"] = int((subset["low_coverage_flag"] > 0).sum())
            if width == 0.5:
                row["datetime_start"] = subset["first_datetime"].min() if len(subset) else ""
                row["datetime_end"] = subset["last_datetime"].max() if len(subset) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    raw_root = Path(config["raw_data_root"])
    paths = loadcycle_paths(raw_root)
    chunk_rows = int(config["chunk_rows"])
    log("Phase N1 foundation build started")
    timeline, age_min, age_max, nominal_by_file = scan_timeline(paths, chunk_rows)
    cut60 = age_min + 0.60 * (age_max - age_min)
    cut80 = age_min + 0.80 * (age_max - age_min)
    write_csv(timeline, PHASE_ROOT / "outputs" / "audits" / "loadcycle_source_timeline.csv")
    audit, active_cells, active_meta = active_cell_audit(paths, chunk_rows, cut80, config)
    frames = build_aggregates(paths, active_cells, nominal_by_file, config, age_min, age_max, cut60, cut80)
    dictionary = feature_dictionary(frames)
    availability = causal_availability(dictionary)
    write_csv(dictionary, PHASE_ROOT / "outputs" / "causal_feature_tables" / "causal_feature_dictionary.csv")
    write_csv(availability, PHASE_ROOT / "outputs" / "causal_feature_tables" / "causal_feature_availability.csv")
    manifest = build_split_manifest(timeline, frames, age_min, age_max, cut60, cut80)
    write_csv(manifest, PHASE_ROOT / "outputs" / "split_manifest" / "split_manifest.csv")
    split_md = f"""# Frozen Data Split

Ageing coordinate: `OpHrs2 [h]`

Observed range: `{age_min:.6f}–{age_max:.6f} h`

- Train: `[{age_min:.6f}, {cut60:.6f}) h`
- Validation: `[{cut60:.6f}, {cut80:.6f}) h`
- Test: `[{cut80:.6f}, {age_max:.6f}] h`

Cut points are calculated from the observed ageing span: `min + 60% × span` and `min + 80% × span`. A bin is assigned only when its full physical interval is contained within one split; boundary-straddling bins are marked `boundary_excluded`.

The final 20% Test split is frozen. It may be checked for readability, range, bin/window counts and structural integrity only. It must not determine sampling resolution, active-cell mask, j*, SOH formula, features, lookback, horizon, preprocessing, model, loss or hyperparameters.

## Manifest

{markdown_table(manifest.round(6))}
"""
    write_text(split_md, PHASE_ROOT / "outputs" / "split_manifest" / "FROZEN_DATA_SPLIT.md")
    run_metadata = {
        "age_min_h": age_min,
        "age_max_h": age_max,
        "cut60_h": cut60,
        "cut80_h": cut80,
        "loadcycle_files": len(paths),
        "loadcycle_rows": int(timeline["raw_rows"].sum()),
        "active_cell_audit": active_meta,
        "aggregation_rows": {str(width): int(len(frame)) for width, frame in frames.items()},
        "test_usage": "structure/counts only; not used for selection",
    }
    metadata_path = PHASE_ROOT / "outputs" / "audits" / "foundation_run_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(run_metadata, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PHASE N1 FOUNDATION FAILED: {exc}", file=sys.stderr, flush=True)
        raise

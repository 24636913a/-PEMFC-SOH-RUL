#!/usr/bin/env python
"""Fail-closed legal-window audit for Phase N1.

The audit constructs no model arrays and reads no Test performance metric. It
only counts structurally legal sequences and rejects any window that crosses a
split, time-grid gap, characterization gap, or low-coverage bin.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase_n1_common import PHASE_ROOT, load_config, write_csv


REQUIRED_COLUMNS = {
    "resolution_hours",
    "bin_start_ophrs2_h",
    "bin_end_ophrs2_h",
    "coverage_ratio",
    "low_coverage_flag",
    "internal_datetime_gap_count",
    "gap_before_flag",
    "gap_after_flag",
    "split",
    "cell_voltage_mean",
}
ALLOWED_SPLITS = {"train", "validation", "test", "boundary_excluded"}


def validate_table(frame: pd.DataFrame, width_h: float, source: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"source": str(source), "check": name, "passed": bool(passed), "detail": detail})

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    add("required-columns", not missing, f"missing={missing}")
    if missing:
        return checks
    start = pd.to_numeric(frame["bin_start_ophrs2_h"], errors="coerce").to_numpy(dtype=float)
    end = pd.to_numeric(frame["bin_end_ophrs2_h"], errors="coerce").to_numpy(dtype=float)
    add("finite-bin-coordinates", bool(np.isfinite(start).all() and np.isfinite(end).all()), "All bin coordinates must be finite.")
    add("unique-bin-start", not bool(pd.Series(start).duplicated().any()), "No duplicate bin_start_ophrs2_h values.")
    add("strictly-increasing-bin-start", bool(np.all(np.diff(start) > 0)), "Rows are strictly ordered by OpHrs2.")
    add("correct-bin-width", bool(np.allclose(end - start, width_h, atol=1e-8)), f"Every bin width must equal {width_h} h.")
    resolution = pd.to_numeric(frame["resolution_hours"], errors="coerce").to_numpy(dtype=float)
    add("resolution-column-consistent", bool(np.allclose(resolution, width_h, atol=1e-12)), "Resolution metadata matches file task.")
    unknown_split = sorted(set(frame["split"].dropna().astype(str)) - ALLOWED_SPLITS)
    add("known-split-labels", not unknown_split, f"unknown={unknown_split}")
    add("coverage-nonnegative", bool((pd.to_numeric(frame["coverage_ratio"], errors="coerce") >= 0).all()), "Coverage cannot be negative.")
    return checks


def audit_task(
    frame: pd.DataFrame,
    width_h: float,
    lookback_h: int,
    horizon_h: int,
    coverage_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookback_steps_float = lookback_h / width_h
    horizon_steps_float = horizon_h / width_h
    if not lookback_steps_float.is_integer() or not horizon_steps_float.is_integer():
        raise ValueError("Physical lookback/horizon must map exactly to the selected resolution")
    lookback_steps = int(lookback_steps_float)
    horizon_steps = int(horizon_steps_float)
    n = len(frame)
    split_values = frame["split"].astype(str).to_numpy()
    age = pd.to_numeric(frame["bin_start_ophrs2_h"], errors="coerce").to_numpy(dtype=float)
    coverage = pd.to_numeric(frame["coverage_ratio"], errors="coerce").to_numpy(dtype=float)
    internal_gap = pd.to_numeric(frame["internal_datetime_gap_count"], errors="coerce").fillna(1).to_numpy(dtype=int)
    gap_before = pd.to_numeric(frame["gap_before_flag"], errors="coerce").fillna(1).to_numpy(dtype=int)
    voltage = pd.to_numeric(frame["cell_voltage_mean"], errors="coerce").to_numpy(dtype=float)
    counters = {
        split: {"cross_split": 0, "gap": 0, "coverage": 0, "legal": 0, "causal_failure": 0}
        for split in ("train", "validation", "test")
    }
    rejected_examples: list[dict[str, Any]] = []
    for origin in range(lookback_steps - 1, n - horizon_steps):
        window_start = origin - lookback_steps + 1
        window_end = origin + horizon_steps
        origin_split = split_values[origin]
        if origin_split not in counters:
            continue
        idx = slice(window_start, window_end + 1)
        window_splits = split_values[idx]
        if not np.all(window_splits == origin_split):
            counters[origin_split]["cross_split"] += 1
            continue
        window_age = age[idx]
        exact_grid = bool(np.isfinite(window_age).all() and np.allclose(np.diff(window_age), width_h, atol=1e-7))
        no_recorded_gap = bool(np.all(internal_gap[idx] == 0) and np.all(gap_before[window_start + 1 : window_end + 1] == 0))
        if not exact_grid or not no_recorded_gap:
            counters[origin_split]["gap"] += 1
            continue
        structural_ok = bool(
            np.isfinite(coverage[idx]).all()
            and np.all(coverage[idx] >= coverage_threshold)
            and np.isfinite(voltage[idx]).all()
        )
        if not structural_ok:
            counters[origin_split]["coverage"] += 1
            continue
        input_times = window_age[:lookback_steps]
        target_times = window_age[lookback_steps:]
        causal_ok = bool(
            len(input_times) == lookback_steps
            and len(target_times) == horizon_steps
            and np.all(np.diff(input_times) > 0)
            and np.all(np.diff(target_times) > 0)
            and input_times[-1] < target_times[0]
        )
        if not causal_ok:
            counters[origin_split]["causal_failure"] += 1
            if len(rejected_examples) < 10:
                rejected_examples.append(
                    {
                        "split": origin_split,
                        "origin_row": origin,
                        "lookback_h": lookback_h,
                        "horizon_h": horizon_h,
                        "input_end_h": float(input_times[-1]) if len(input_times) else math.nan,
                        "target_start_h": float(target_times[0]) if len(target_times) else math.nan,
                    }
                )
            continue
        counters[origin_split]["legal"] += 1
    rows = []
    for split in ("train", "validation", "test"):
        split_bins = int(np.count_nonzero(split_values == split))
        naive = max(0, split_bins - lookback_steps - horizon_steps + 1)
        item = counters[split]
        rows.append(
            {
                "sampling_resolution_hours": width_h,
                "lookback_hours": lookback_h,
                "lookback_steps_L": lookback_steps,
                "horizon_hours": horizon_h,
                "horizon_steps_H": horizon_steps,
                "split": split,
                "split_bin_count": split_bins,
                "naive_windows_if_contiguous": naive,
                "windows_lost_split_boundary": item["cross_split"],
                "windows_lost_gap": item["gap"],
                "windows_lost_coverage": item["coverage"],
                "causal_audit_failures": item["causal_failure"],
                "legal_windows": item["legal"],
                "effective_sequence_count": item["legal"],
                "test_usage_policy": "counts/structure only; no target performance inspected" if split == "test" else "eligible for Phase N1 feasibility",
            }
        )
    return rows, rejected_examples


def run_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_config()
    table_dir = PHASE_ROOT / "outputs" / "causal_feature_tables"
    files = {
        0.5: table_dir / "causal_features_30min.csv",
        1.0: table_dir / "causal_features_1h.csv",
    }
    checks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    lookbacks = [int(x) for x in config["temporal"]["lookback_hours"]]
    horizons = [int(x) for x in config["temporal"]["horizon_hours"]]
    additional = [int(x) for x in config["temporal"].get("additional_horizon_hours", [])]
    for width, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        checks.extend(validate_table(frame, width, path))
        if not all(item["passed"] for item in checks if item["source"] == str(path)):
            continue
        for lookback in lookbacks:
            for horizon in [*horizons, *additional]:
                task_rows, task_examples = audit_task(
                    frame,
                    width,
                    lookback,
                    horizon,
                    float(config["aggregation"]["coverage_threshold"]),
                )
                rows.extend(task_rows)
                examples.extend(task_examples)
    summary = pd.DataFrame(rows)
    causal_failures = int(summary["causal_audit_failures"].sum()) if not summary.empty else 1
    passed = bool(checks and all(item["passed"] for item in checks) and causal_failures == 0 and not summary.empty)
    log = {
        "passed": passed,
        "checks": checks,
        "causal_failure_examples": examples,
        "selection_policy": "Train+Validation for task design; Test counts only",
        "coverage_threshold": float(config["aggregation"]["coverage_threshold"]),
        "rows_in_summary": int(len(summary)),
    }
    return summary, log


def main() -> None:
    summary, audit_log = run_audit()
    output = PHASE_ROOT / "outputs" / "temporal_scale_analysis"
    write_csv(summary, output / "window_feasibility_summary.csv")
    log_path = PHASE_ROOT / "outputs" / "audits" / "phase_n1_window_audit_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(audit_log, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit_log["passed"]:
        raise RuntimeError(f"Fail-closed window audit failed; see {log_path}")
    print(json.dumps({"passed": True, "summary_rows": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PHASE N1 WINDOW AUDIT FAILED: {exc}", file=sys.stderr, flush=True)
        raise

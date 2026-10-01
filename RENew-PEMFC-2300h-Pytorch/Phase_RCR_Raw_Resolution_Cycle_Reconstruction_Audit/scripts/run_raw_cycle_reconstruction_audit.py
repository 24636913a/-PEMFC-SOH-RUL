"""Development-only operating-state reconstruction for Phase RCR.

This module intentionally has no voltage, polarization, model, SOH, or RUL
input path.  It streams only six frozen Loadcycle fields and rejects all rows
at or beyond the Development boundary before they reach the state sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
OUTPUTS = PHASE_DIR / "outputs"
FIGURES = OUTPUTS / "figures"
REPORTS = PHASE_DIR / "reports"
FINAL = PHASE_DIR / "final"
CONFIG = PHASE_DIR / "configs" / "PREREGISTERED_RCR_PROTOCOL.yaml"
CONFIG_SHA = PHASE_DIR / "configs" / "PREREGISTERED_RCR_PROTOCOL.sha256"
SOURCE_MANIFEST = PHASE_DIR / "SOURCE_MANIFEST.yaml"

BASE_COMMIT = "ca040e56a76e9c985ec199e67cbed1929ea285e4"
RAW_ROOT = Path(
    r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据"
)
N1_30MIN = REPO_ROOT / "RENew-PEMFC-2300h-Pytorch" / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"

BOUNDARY_H = 1819.997738
CHUNK_ROWS = 200_000
GAP_SECONDS = 5.0
ZERO_A = 0.1
ROUND_DECIMALS = 3
CALIBRATION_H = 100.0
LOADED_MIN_H, LOADED_MAX_H = 8.0, 11.0
WALL_MIN_H, WALL_MAX_H = 10.0, 14.0
CORE_ORDER_MIN = 0.90
LOADED_CV_MAX = 0.10
MIN_CYCLES = 10
REQUIRED_COLUMNS = (
    "OpHrs2 [h]",
    "DateTime",
    "operation_point",
    "I.S.set [A]",
    "I.S.act [A]",
    "j.S.act [A/cm2]",
)
N1_MAPPING_COLUMNS = (
    "bin_start_ophrs2_h",
    "bin_end_ophrs2_h",
    "first_datetime",
    "last_datetime",
    "number_of_operation_points",
    "operation_point_transitions",
    "internal_datetime_gap_count",
    "gap_before_flag",
    "gap_after_flag",
    "temporal_segment_id",
    "coverage_ratio",
    "bin_valid_for_windows",
    "startup_flag",
    "shutdown_flag",
)
AGE_STRATA = {
    "EARLY": (72.788690, 655.191706),
    "MIDDLE": (655.191706, 1237.594722),
    "LATE": (1237.594722, BOUNDARY_H),
}

# Scientific-figure-making house palette.  No visual encodes a health result.
BLUE, GREEN, RED, TEAL, VIOLET, GREY = (
    "#0F4D92",
    "#8BCF8B",
    "#B64342",
    "#42949E",
    "#9A4D8E",
    "#666666",
)


class AuditFailure(RuntimeError):
    """Fail-closed exception for a frozen RCR contract breach."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_provenance() -> dict[str, str]:
    if not CONFIG.exists() or not CONFIG_SHA.exists():
        raise AuditFailure("Frozen RCR protocol or its SHA256 sidecar is missing.")
    expected = CONFIG_SHA.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(CONFIG)
    if expected != observed:
        raise AuditFailure(f"Frozen protocol hash mismatch: expected {expected}, observed {observed}.")
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise AuditFailure(f"Required base commit {BASE_COMMIT} is not an ancestor of HEAD.") from exc
    return {"protocol_sha256": observed, "current_head": git_output("rev-parse", "HEAD")}


def loadcycle_order(path: Path) -> tuple[int, str]:
    match = re.search(r"Loadcycle_(\d+)", path.stem)
    if not match:
        raise AuditFailure(f"Unparseable Loadcycle filename: {path.name}")
    return int(match.group(1)), path.name


def locate_raw_files() -> list[Path]:
    # The frozen source stream is under Loadcycle_data.  Existing N1 provenance
    # establishes that packaging may include one nested archive directory.
    files = sorted((RAW_ROOT / "Loadcycle_data").rglob("Loadcycle_*.csv"), key=loadcycle_order)
    if len(files) != 17:
        raise AuditFailure(f"Expected 17 Loadcycle CSV files, found {len(files)}.")
    return files


def assert_operating_only_columns(columns: Iterable[str]) -> None:
    selected = tuple(columns)
    if len(selected) != len(REQUIRED_COLUMNS) or set(selected) != set(REQUIRED_COLUMNS):
        raise AuditFailure(f"Cycle-discovery columns differ from frozen contract: {selected}")
    forbidden = [column for column in selected if "volt" in column.lower() or re.match(r"^U\d+", column)]
    if forbidden:
        raise AuditFailure(f"Voltage fields are forbidden in cycle discovery: {forbidden}")


def validate_raw_schema(path: Path) -> list[str]:
    header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise AuditFailure(f"{path.name} lacks required frozen fields: {missing}")
    assert_operating_only_columns(REQUIRED_COLUMNS)
    return header


def normalise_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Coerce the six permitted fields, then firewall Test rows before storage."""
    assert_operating_only_columns(chunk.columns)
    # Pandas may preserve source-header rather than usecols-list order. Reorder
    # deterministically only after asserting that the selected field *set* is
    # exactly the six frozen operating columns.
    work = chunk.loc[:, list(REQUIRED_COLUMNS)].copy()
    # The source represents hours at and above 1000 h with a thousands comma
    # (for example ``1,225.01``). This is source number formatting, not a
    # time-axis transformation; removing the presentation comma preserves the
    # frozen OpHrs2 coordinate exactly before the boundary comparison.
    work["OpHrs2 [h]"] = pd.to_numeric(
        work["OpHrs2 [h]"].astype("string").str.replace(",", "", regex=False), errors="coerce"
    )
    work["I.S.set [A]"] = pd.to_numeric(work["I.S.set [A]"], errors="coerce")
    work["I.S.act [A]"] = pd.to_numeric(work["I.S.act [A]"], errors="coerce")
    work["j.S.act [A/cm2]"] = pd.to_numeric(work["j.S.act [A/cm2]"], errors="coerce")
    work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
    work["operation_point"] = work["operation_point"].astype("string")
    valid = (
        np.isfinite(work["OpHrs2 [h]"])
        & np.isfinite(work["I.S.set [A]"])
        & np.isfinite(work["I.S.act [A]"])
        & np.isfinite(work["j.S.act [A/cm2]"])
        & work["DateTime"].notna()
        & work["operation_point"].notna()
    )
    work = work.loc[valid].copy()
    post_count = int((work["OpHrs2 [h]"] >= BOUNDARY_H).sum())
    # This is the only point at which a row can cross the formal boundary.
    # It is discarded before any sequence, statistic, cache, or figure sees it.
    work = work.loc[work["OpHrs2 [h]"] < BOUNDARY_H].copy()
    return work, post_count


@dataclass
class RunAccumulator:
    run_id: int
    operation_point: str
    rounded_set_A: float
    start_datetime: pd.Timestamp
    start_age_h: float
    gap_before: bool
    raw_row_count: int = 0
    end_datetime: pd.Timestamp | None = None
    end_age_h: float | None = None
    set_values: list[float] = field(default_factory=list)
    actual_values: list[float] = field(default_factory=list)
    density_values: list[float] = field(default_factory=list)
    has_zero_or_stop: bool = False

    def append(self, dt: pd.Timestamp, age_h: float, set_A: float, actual_A: float, density: float) -> None:
        self.raw_row_count += 1
        self.end_datetime = dt
        self.end_age_h = age_h
        self.set_values.append(float(set_A))
        self.actual_values.append(float(actual_A))
        self.density_values.append(float(density))
        self.has_zero_or_stop = self.has_zero_or_stop or abs(float(actual_A)) <= ZERO_A

    def finish(self) -> dict[str, Any]:
        if self.end_datetime is None or self.end_age_h is None:
            raise AuditFailure("Attempted to finish an empty operating-state run.")
        return {
            "run_id": self.run_id,
            "start_datetime": self.start_datetime.isoformat(),
            "end_datetime": self.end_datetime.isoformat(),
            "start_ophrs2_h": self.start_age_h,
            "end_ophrs2_h": self.end_age_h,
            "duration_wall_s": (self.end_datetime - self.start_datetime).total_seconds(),
            "duration_loaded_s": (self.end_age_h - self.start_age_h) * 3600.0,
            "operation_point": self.operation_point,
            "median_set_current_A": float(np.median(self.set_values)),
            "median_actual_current_A": float(np.median(self.actual_values)),
            "median_actual_current_density_A_cm2": float(np.median(self.density_values)),
            "raw_row_count": self.raw_row_count,
            "gap_before": int(self.gap_before),
            "stop_or_zero_current_flag": int(self.has_zero_or_stop),
            "state_token": identity_token(self.operation_point, self.rounded_set_A),
            "set_token": f"{self.rounded_set_A:.3f}",
        }


def identity_token(operation_point: str, rounded_set_A: float) -> str:
    return f"{str(operation_point)}||{rounded_set_A:.{ROUND_DECIMALS}f}"


def compress_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Run-length compress a Development-only frame; stops are retained."""
    assert_operating_only_columns(rows.columns)
    # Production passes a normalized chunk; this coercion keeps the same
    # production implementation directly testable with compact synthetic input.
    rows = rows.copy()
    rows["DateTime"] = pd.to_datetime(rows["DateTime"], errors="raise")
    accumulator: RunAccumulator | None = None
    previous_dt: pd.Timestamp | None = None
    previous_age: float | None = None
    output: list[dict[str, Any]] = []
    next_run_id = 1
    for age_h, dt, operation_point, set_A, actual_A, density in rows.itertuples(index=False, name=None):
        rounded_set = round(float(set_A), ROUND_DECIMALS)
        state_changed = accumulator is None or (
            str(operation_point) != accumulator.operation_point or rounded_set != accumulator.rounded_set_A
        )
        discontinuity = (
            previous_dt is not None
            and ((dt - previous_dt).total_seconds() > GAP_SECONDS or dt <= previous_dt or float(age_h) < float(previous_age))
        )
        if accumulator is not None and (state_changed or discontinuity):
            output.append(accumulator.finish())
            accumulator = None
        if accumulator is None:
            accumulator = RunAccumulator(
                run_id=next_run_id,
                operation_point=str(operation_point),
                rounded_set_A=rounded_set,
                start_datetime=dt,
                start_age_h=float(age_h),
                gap_before=bool(discontinuity),
            )
            next_run_id += 1
        accumulator.append(dt, float(age_h), float(set_A), float(actual_A), float(density))
        previous_dt, previous_age = dt, float(age_h)
    if accumulator is not None:
        output.append(accumulator.finish())
    return pd.DataFrame(output)


def stream_and_compress(paths: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_runs: list[pd.DataFrame] = []
    timeline_records: list[dict[str, Any]] = []
    prior_tail: pd.DataFrame | None = None
    post_boundary_discarded = 0
    retained_rows_total = 0
    raw_rows_total = 0
    for file_index, path in enumerate(paths, start=1):
        validate_raw_schema(path)
        file_runs: list[pd.DataFrame] = []
        file_raw_rows = file_retained_rows = file_post_rows = 0
        file_invalid_rows = 0
        file_min_age = math.inf
        file_max_age = -math.inf
        file_min_time: pd.Timestamp | None = None
        file_max_time: pd.Timestamp | None = None
        interval_samples: list[float] = []
        for chunk in pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            usecols=list(REQUIRED_COLUMNS),
            chunksize=CHUNK_ROWS,
            low_memory=False,
        ):
            file_raw_rows += len(chunk)
            raw_rows_total += len(chunk)
            valid_before = len(chunk)
            development, dropped = normalise_chunk(chunk)
            file_post_rows += dropped
            post_boundary_discarded += dropped
            file_invalid_rows += valid_before - len(development) - dropped
            if development.empty:
                continue
            # Concatenate only the prior retained tail needed to make a run
            # continuous across chunk/file boundaries. No post-boundary row is kept.
            combined = development if prior_tail is None else pd.concat([prior_tail, development], ignore_index=True)
            if len(combined) > 1:
                dt_values = combined["DateTime"].astype("int64").to_numpy() / 1e9
                diffs = np.diff(dt_values)
                valid_diffs = diffs[(diffs > 0) & (diffs <= GAP_SECONDS)]
                interval_samples.extend(valid_diffs[:200].tolist())
            # Keep the final raw row to let the next chunk recreate an open run.
            # The duplicated tail is removed before output by dropping its complete
            # terminal run later; simplifying this correctly requires a persistent
            # accumulator, so process the full retained file after chunk firewall.
            file_runs.append(development)
            prior_tail = None
            file_retained_rows += len(development)
            retained_rows_total += len(development)
            file_min_age = min(file_min_age, float(development["OpHrs2 [h]"].min()))
            file_max_age = max(file_max_age, float(development["OpHrs2 [h]"].max()))
            timestamp_min, timestamp_max = development["DateTime"].min(), development["DateTime"].max()
            file_min_time = timestamp_min if file_min_time is None else min(file_min_time, timestamp_min)
            file_max_time = timestamp_max if file_max_time is None else max(file_max_time, timestamp_max)
        # A file's Development portion can be compactly held only while its raw
        # chunks are being processed. Here we reassemble just the permitted six
        # fields for RLE; this is never Test data nor a complete raw cache.
        if file_runs:
            permitted_file = pd.concat(file_runs, ignore_index=True)
            compressed = compress_rows(permitted_file)
            if all_runs:
                # Do not silently merge runs across source-file boundaries. A
                # potential boundary is a meaningful audited gap/segment feature.
                compressed["run_id"] += int(all_runs[-1]["run_id"].max())
                compressed.loc[compressed.index[0], "gap_before"] = 1
            all_runs.append(compressed)
        timeline_records.append(
            {
                "file_order": file_index,
                "source_file": str(path.relative_to(RAW_ROOT)).replace("\\", "/"),
                "file_size_bytes": path.stat().st_size,
                "raw_rows_seen": file_raw_rows,
                "development_rows_retained": file_retained_rows,
                "post_boundary_rows_discarded": file_post_rows,
                "post_boundary_rows_used": 0,
                "invalid_required_field_rows": file_invalid_rows,
                "development_age_start_h": None if math.isinf(file_min_age) else file_min_age,
                "development_age_end_h": None if math.isinf(file_max_age) else file_max_age,
                "development_datetime_start": None if file_min_time is None else file_min_time.isoformat(),
                "development_datetime_end": None if file_max_time is None else file_max_time.isoformat(),
                "nominal_interval_s_median": float(np.median(interval_samples)) if interval_samples else np.nan,
            }
        )
    if not all_runs:
        raise AuditFailure("No Development rows survived the frozen OpHrs2 boundary firewall.")
    runs = pd.concat(all_runs, ignore_index=True)
    if (runs["end_ophrs2_h"] >= BOUNDARY_H).any():
        raise AuditFailure("Post-boundary row leaked into the RLE operating-state sequence.")
    timeline = pd.DataFrame(timeline_records)
    summary = {
        "raw_rows_seen_total": raw_rows_total,
        "development_rows_retained_total": retained_rows_total,
        "post_boundary_rows_discarded_total": post_boundary_discarded,
        "post_boundary_rows_used": 0,
        "raw_files": len(paths),
    }
    return runs, timeline, summary


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Exact-order LCS, deliberately not DTW or elastic rephasing."""
    if not left or not right:
        return 0
    prior = [0] * (len(right) + 1)
    for token_left in left:
        current = [0]
        for j, token_right in enumerate(right, start=1):
            current.append(prior[j - 1] + 1 if token_left == token_right else max(prior[j], current[-1]))
        prior = current
    return prior[-1]


def age_stratum(start_h: float, end_h: float) -> str | None:
    for label, (lower, upper) in AGE_STRATA.items():
        if start_h >= lower and end_h < upper:
            return label
    return None


def calibration_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    lower = float(runs["start_ophrs2_h"].min())
    upper = lower + CALIBRATION_H
    selected = runs.loc[(runs["start_ophrs2_h"] >= lower) & (runs["end_ophrs2_h"] <= upper)].copy()
    if selected.empty:
        raise AuditFailure("Frozen first-100-h calibration interval contains no complete state run.")
    return selected.reset_index(drop=True), lower, upper


def discover_template(runs: pd.DataFrame) -> tuple[dict[str, Any] | None, pd.DataFrame, pd.DataFrame, float, float]:
    calibration, lower, upper = calibration_runs(runs)
    calibration["sequence_index"] = np.arange(len(calibration))
    identity = calibration["state_token"].astype(str).tolist()
    anchors: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in range(len(identity) - 1):
        anchors[(identity[index], identity[index + 1])].append(index)
    candidate_rows: list[dict[str, Any]] = []
    for pair, positions in anchors.items():
        valid_intervals: list[tuple[int, int, float]] = []
        for left, right in zip(positions[:-1], positions[1:]):
            loaded_h = float(calibration.loc[right, "start_ophrs2_h"] - calibration.loc[left, "start_ophrs2_h"])
            if LOADED_MIN_H <= loaded_h <= LOADED_MAX_H:
                valid_intervals.append((left, right, loaded_h))
        candidate_rows.append(
            {
                "anchor_first_token": pair[0],
                "anchor_second_token": pair[1],
                "calibration_occurrences": len(positions),
                "consecutive_intervals": max(0, len(positions) - 1),
                "valid_loaded_intervals": len(valid_intervals),
                "first_valid_start_index": valid_intervals[0][0] if valid_intervals else np.nan,
                "first_valid_end_index": valid_intervals[0][1] if valid_intervals else np.nan,
                "first_valid_loaded_h": valid_intervals[0][2] if valid_intervals else np.nan,
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        return None, calibration, candidates, lower, upper
    candidates = candidates.sort_values(
        ["valid_loaded_intervals", "calibration_occurrences", "anchor_first_token", "anchor_second_token"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates["rank"] = np.arange(1, len(candidates) + 1)
    winner = candidates.iloc[0]
    if int(winner["valid_loaded_intervals"]) == 0:
        return None, calibration, candidates, lower, upper
    left, right = int(winner["first_valid_start_index"]), int(winner["first_valid_end_index"])
    template_runs = calibration.iloc[left:right].copy()
    if template_runs.empty:
        return None, calibration, candidates, lower, upper
    ambiguous = bool(
        len(candidates) > 1
        and int(candidates.iloc[1]["valid_loaded_intervals"]) == int(winner["valid_loaded_intervals"])
        and int(candidates.iloc[1]["calibration_occurrences"]) == int(winner["calibration_occurrences"])
    )
    template = {
        "frozen": True,
        "selection_rule": "frozen calibration anchor ranking in preregistered protocol",
        "calibration_start_h": lower,
        "calibration_end_h": upper,
        "anchor": [str(winner["anchor_first_token"]), str(winner["anchor_second_token"])],
        "anchor_ambiguous_under_primary_scores": ambiguous,
        "template_start_run_id": int(template_runs.iloc[0]["run_id"]),
        "template_end_run_id_exclusive": int(calibration.iloc[right]["run_id"]),
        "template_loaded_duration_h": float(calibration.iloc[right]["start_ophrs2_h"] - template_runs.iloc[0]["start_ophrs2_h"]),
        "template_wall_duration_h": (
            pd.Timestamp(calibration.iloc[right]["start_datetime"]) - pd.Timestamp(template_runs.iloc[0]["start_datetime"])
        ).total_seconds() / 3600.0,
        "template_state_tokens": template_runs["state_token"].astype(str).tolist(),
        "template_set_tokens": template_runs.loc[
            template_runs["stop_or_zero_current_flag"] == 0, "set_token"
        ].astype(str).tolist(),
        "template_positive_operation_tokens": template_runs.loc[
            template_runs["stop_or_zero_current_flag"] == 0, "operation_point"
        ].astype(str).tolist(),
        "template_stop_run_count": int((template_runs["stop_or_zero_current_flag"] == 1).sum()),
        "candidate_count": int(len(candidates)),
    }
    return template, calibration, candidates, lower, upper


def anchor_positions(runs: pd.DataFrame, anchor: Sequence[str]) -> list[int]:
    tokens = runs["state_token"].astype(str).tolist()
    return [i for i in range(len(tokens) - 1) if tokens[i] == anchor[0] and tokens[i + 1] == anchor[1]]


def dwell_variation(template_slice: pd.DataFrame, cycle_slice: pd.DataFrame) -> tuple[int, int]:
    # Fixed identity-token matching is descriptive only; it cannot realign a
    # cycle or manufacture an interval. It reports altered dwell structures.
    template_by_token: dict[str, list[float]] = defaultdict(list)
    cycle_by_token: dict[str, list[float]] = defaultdict(list)
    for row in template_slice.itertuples(index=False):
        template_by_token[str(row.state_token)].append(float(row.duration_loaded_s))
    for row in cycle_slice.itertuples(index=False):
        cycle_by_token[str(row.state_token)].append(float(row.duration_loaded_s))
    shortened = extended = 0
    for token in sorted(set(template_by_token) & set(cycle_by_token)):
        for baseline, observed in zip(template_by_token[token], cycle_by_token[token]):
            if baseline <= 0:
                continue
            ratio = observed / baseline
            shortened += int(ratio < 0.8)
            extended += int(ratio > 1.2)
    return shortened, extended


def reconstruct_cycles(runs: pd.DataFrame, template: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = anchor_positions(runs, template["anchor"])
    template_start = int(template["template_start_run_id"])
    template_end = int(template["template_end_run_id_exclusive"])
    template_slice = runs.loc[(runs["run_id"] >= template_start) & (runs["run_id"] < template_end)].copy()
    t_set = list(template["template_set_tokens"])
    t_operation = list(template["template_positive_operation_tokens"])
    cycles: list[dict[str, Any]] = []
    similarities: list[dict[str, Any]] = []
    for cycle_id, (left, right) in enumerate(zip(positions[:-1], positions[1:]), start=1):
        interval = runs.iloc[left:right].copy()
        start = interval.iloc[0]
        end_anchor = runs.iloc[right]
        loaded_h = float(end_anchor["start_ophrs2_h"] - start["start_ophrs2_h"])
        wall_h = (pd.Timestamp(end_anchor["start_datetime"]) - pd.Timestamp(start["start_datetime"])).total_seconds() / 3600.0
        positive = interval.loc[interval["stop_or_zero_current_flag"] == 0]
        c_set = positive["set_token"].astype(str).tolist()
        c_operation = positive["operation_point"].astype(str).tolist()
        set_lcs = lcs_length(t_set, c_set)
        op_lcs = lcs_length(t_operation, c_operation)
        set_fraction = set_lcs / len(t_set) if t_set else np.nan
        operation_fraction = op_lcs / len(t_operation) if t_operation else np.nan
        missing = max(0, len(t_operation) - op_lcs)
        extra = max(0, len(c_operation) - op_lcs)
        shortened, extended = dwell_variation(template_slice, interval)
        scale_pass = LOADED_MIN_H <= loaded_h <= LOADED_MAX_H
        interval_stratum = age_stratum(float(start["start_ophrs2_h"]), float(end_anchor["start_ophrs2_h"]))
        complete = bool(scale_pass and operation_fraction >= CORE_ORDER_MIN and interval_stratum is not None)
        actual_set_error = float(np.median(np.abs(positive["median_actual_current_A"] - positive["median_set_current_A"]))) if not positive.empty else np.nan
        common = {
            "raw_cycle_id": cycle_id,
            "start_run_id": int(start["run_id"]),
            "end_run_id_exclusive": int(end_anchor["run_id"]),
            "cycle_start": str(start["start_datetime"]),
            "cycle_end": str(end_anchor["start_datetime"]),
            "start_ophrs2_h": float(start["start_ophrs2_h"]),
            "end_ophrs2_h": float(end_anchor["start_ophrs2_h"]),
            "age_stratum": interval_stratum or "CROSS_STRATUM_OR_OUTSIDE",
            "loaded_duration_h": loaded_h,
            "wall_duration_h": wall_h,
            "raw_state_count": int(len(interval)),
            "positive_state_count": int(len(positive)),
            "stop_run_count": int((interval["stop_or_zero_current_flag"] == 1).sum()),
            "stop_run_delta_vs_template": int((interval["stop_or_zero_current_flag"] == 1).sum()) - int(template["template_stop_run_count"]),
            "candidate_scale_pass": scale_pass,
            "sequence_core_order_pass": bool(operation_fraction >= CORE_ORDER_MIN),
            "complete_cycle": complete,
            "missing_positive_operation_states": missing,
            "extra_positive_operation_states": extra,
            "shortened_dwell_states": shortened,
            "extended_dwell_states": extended,
            "median_abs_actual_minus_set_A": actual_set_error,
        }
        cycles.append(common)
        similarities.append(
            {
                "raw_cycle_id": cycle_id,
                "age_stratum": common["age_stratum"],
                "set_current_lcs_length": set_lcs,
                "set_current_template_length": len(t_set),
                "set_current_sequence_fraction": set_fraction,
                "operation_point_lcs_length": op_lcs,
                "operation_point_template_length": len(t_operation),
                "operation_point_sequence_fraction": operation_fraction,
                "sequence_edit_distance_set_current": len(t_set) + len(c_set) - 2 * set_lcs,
                "sequence_edit_distance_operation_point": len(t_operation) + len(c_operation) - 2 * op_lcs,
                "is_complete_cycle": complete,
            }
        )
    return pd.DataFrame(cycles), pd.DataFrame(similarities)


def duration_audit(cycles: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for label in ["ALL", *AGE_STRATA]:
        subset = cycles.loc[cycles["complete_cycle"]].copy()
        if label != "ALL":
            subset = subset.loc[subset["age_stratum"] == label]
        values = subset["loaded_duration_h"].dropna().to_numpy(dtype=float)
        wall = subset["wall_duration_h"].dropna().to_numpy(dtype=float)
        records.append(
            {
                "scope": label,
                "complete_cycle_count": len(subset),
                "loaded_duration_median_h": float(np.median(values)) if len(values) else np.nan,
                "loaded_duration_mean_h": float(np.mean(values)) if len(values) else np.nan,
                "loaded_duration_std_h": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "loaded_duration_cv": float(np.std(values, ddof=1) / np.mean(values)) if len(values) > 1 and np.mean(values) else np.nan,
                "wall_duration_median_h": float(np.median(wall)) if len(wall) else np.nan,
                "wall_duration_mean_h": float(np.mean(wall)) if len(wall) else np.nan,
                "wall_duration_in_10_14_h_fraction": float(np.mean((wall >= WALL_MIN_H) & (wall <= WALL_MAX_H))) if len(wall) else np.nan,
            }
        )
    return pd.DataFrame(records)


def age_support(cycles: pd.DataFrame, template: dict[str, Any], runs: pd.DataFrame) -> pd.DataFrame:
    complete_ids = set(cycles.loc[cycles["complete_cycle"], "raw_cycle_id"].tolist())
    positive_points = sorted(set(template["template_positive_operation_tokens"]))
    positions = anchor_positions(runs, template["anchor"])
    cycle_by_id = {int(row.raw_cycle_id): row for row in cycles.itertuples(index=False)}
    records: list[dict[str, Any]] = []
    for point in positive_points:
        for label in AGE_STRATA:
            count = 0
            for cycle_id, (left, right) in enumerate(zip(positions[:-1], positions[1:]), start=1):
                cycle = cycle_by_id.get(cycle_id)
                if cycle is None or cycle_id not in complete_ids or cycle.age_stratum != label:
                    continue
                sequence = runs.iloc[left:right]
                count += int(point in sequence.loc[sequence["stop_or_zero_current_flag"] == 0, "operation_point"].astype(str).tolist())
            records.append(
                {
                    "operation_point": point,
                    "age_stratum": label,
                    "complete_cycles_with_operation_point": count,
                    "repeated_support_pass_ge_10": bool(count >= MIN_CYCLES),
                }
            )
    return pd.DataFrame(records)


def load_n1_mapping_bins() -> pd.DataFrame:
    if not N1_30MIN.exists():
        raise AuditFailure(f"N1 30-minute mapping source missing: {N1_30MIN}")
    header = pd.read_csv(N1_30MIN, nrows=0).columns.tolist()
    missing = [column for column in N1_MAPPING_COLUMNS if column not in header]
    if missing:
        raise AuditFailure(f"N1 mapping source misses frozen operating metadata columns: {missing}")
    selected = list(N1_MAPPING_COLUMNS)
    if any("voltage" in column.lower() for column in selected):
        raise AuditFailure("N1 mapping attempted to request voltage columns.")
    bins = pd.read_csv(N1_30MIN, nrows=3007, usecols=selected)
    bins = bins.loc[bins["bin_end_ophrs2_h"] < BOUNDARY_H].copy()
    if (bins["bin_end_ophrs2_h"] >= BOUNDARY_H).any():
        raise AuditFailure("N1 Test-side bin reached raw-to-30-minute mapping.")
    return bins.sort_values("bin_start_ophrs2_h").reset_index(drop=True)


def mapping_audit(cycles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cycles.empty:
        return pd.DataFrame(), {"n1_mapping_rows_used": 0, "n1_test_rows_used": 0}
    bins = load_n1_mapping_bins()
    records: list[dict[str, Any]] = []
    for cycle in cycles.itertuples(index=False):
        mapped = bins.loc[
            (bins["bin_end_ophrs2_h"] > cycle.start_ophrs2_h)
            & (bins["bin_start_ophrs2_h"] < cycle.end_ophrs2_h)
        ].copy()
        count = len(mapped)
        pure = int((mapped["number_of_operation_points"] <= 1).sum()) if count else 0
        mixed = int((mapped["number_of_operation_points"] > 1).sum()) if count else 0
        transitions = int(mapped["operation_point_transitions"].fillna(0).sum()) if count else 0
        segment_breaks = 0
        longest_temporal_segment = 0
        longest_mq_eligible = 0
        if count:
            segment = mapped["temporal_segment_id"].astype(str).tolist()
            runs = []
            current = 1
            for previous, later in zip(segment[:-1], segment[1:]):
                if previous == later:
                    current += 1
                else:
                    runs.append(current)
                    current = 1
                    segment_breaks += 1
            runs.append(current)
            longest_temporal_segment = max(runs)
            segment_breaks += int(mapped["gap_before_flag"].fillna(0).astype(bool).sum())
            segment_breaks += int(mapped["gap_after_flag"].fillna(0).astype(bool).sum())
            # This is the historical MQ current implementation's frozen
            # eligibility rule, reused only to explain its all-zero lag-pair
            # outcome. It is not a new RCR cycle-selection rule.
            mq_eligible = (
                (mapped["coverage_ratio"] >= 0.80)
                & mapped["bin_valid_for_windows"].fillna(0).astype(bool)
                & ~mapped["gap_before_flag"].fillna(0).astype(bool)
                & ~mapped["startup_flag"].fillna(0).astype(bool)
                & ~mapped["shutdown_flag"].fillna(0).astype(bool)
            ).to_numpy()
            current_eligible = 0
            previous_segment: str | None = None
            previous_index: int | None = None
            for mapped_index, (eligible, segment_id) in enumerate(zip(mq_eligible, segment)):
                if eligible and previous_segment == segment_id and previous_index == mapped_index - 1:
                    current_eligible += 1
                elif eligible:
                    current_eligible = 1
                else:
                    current_eligible = 0
                longest_mq_eligible = max(longest_mq_eligible, current_eligible)
                previous_segment = segment_id if eligible else None
                previous_index = mapped_index if eligible else None
        reasons = []
        if count == 0:
            reasons.append("no_overlapping_30min_bin")
        if mixed:
            reasons.append("mixed_operation_point_bins")
        if transitions:
            reasons.append("within_bin_operation_point_transitions")
        if segment_breaks:
            reasons.append("temporal_segment_or_gap_break")
        if longest_mq_eligible < count:
            reasons.append("MQ_eligible_bin_interruptions")
        preserved = bool(
            count
            and mixed == 0
            and transitions == 0
            and segment_breaks == 0
            and longest_mq_eligible == count
        )
        records.append(
            {
                "raw_cycle_id": int(cycle.raw_cycle_id),
                "cycle_start": cycle.cycle_start,
                "cycle_end": cycle.cycle_end,
                "loaded_duration_h": cycle.loaded_duration_h,
                "wall_duration_h": cycle.wall_duration_h,
                "raw_state_count": int(cycle.raw_state_count),
                "mapped_30min_bin_count": count,
                "pure_30min_bin_count": pure,
                "mixed_30min_bin_count": mixed,
                "operation_point_transitions_inside_bin": transitions,
                "segment_break_count": segment_breaks,
                "longest_contiguous_30min_run": longest_mq_eligible,
                "longest_temporal_segment_30min_run": longest_temporal_segment,
                "cycle_preserved_at_30min": preserved,
                "failure_reason": ";".join(reasons) if reasons else "none",
            }
        )
    return pd.DataFrame(records), {"n1_mapping_rows_used": len(bins), "n1_test_rows_used": 0}


def evaluate_gates(template: dict[str, Any] | None, cycles: pd.DataFrame, durations: pd.DataFrame, support: pd.DataFrame) -> dict[str, Any]:
    if template is None:
        return {
            "status": "RAW_CYCLE_STRUCTURE_NOT_IDENTIFIABLE",
            "template_frozen": False,
            "reason": "No calibration anchor had a consecutive 8–11 h loaded interval.",
        }
    counts = {
        label: int(((cycles["complete_cycle"]) & (cycles["age_stratum"] == label)).sum()) for label in AGE_STRATA
    }
    all_duration = durations.loc[durations["scope"] == "ALL"].iloc[0]
    loaded_median = float(all_duration["loaded_duration_median_h"]) if pd.notna(all_duration["loaded_duration_median_h"]) else np.nan
    loaded_cv = float(all_duration["loaded_duration_cv"]) if pd.notna(all_duration["loaded_duration_cv"]) else np.nan
    support_pass = bool(not support.empty and support["repeated_support_pass_ge_10"].all())
    gates = {
        "unique_calibration_template": not bool(template["anchor_ambiguous_under_primary_scores"]),
        "early_middle_late_complete_cycles": all(counts[label] >= MIN_CYCLES for label in AGE_STRATA),
        "operation_sequence_core_ge_90pct": bool(
            not cycles.empty and cycles.loc[cycles["complete_cycle"], "sequence_core_order_pass"].all()
        ),
        "loaded_duration_median_8_11_h": bool(np.isfinite(loaded_median) and LOADED_MIN_H <= loaded_median <= LOADED_MAX_H),
        "loaded_duration_cv_le_10pct": bool(np.isfinite(loaded_cv) and loaded_cv <= LOADED_CV_MAX),
        "key_positive_operation_point_support": support_pass,
        "voltage_independent_boundary": True,
    }
    passed = all(gates.values())
    has_recurrence = bool(not cycles.empty and int(cycles["candidate_scale_pass"].sum()) > 0)
    status = "RAW_CYCLE_STRUCTURE_QUALIFIED" if passed else (
        "RAW_CYCLE_STRUCTURE_PARTIALLY_IDENTIFIABLE" if has_recurrence else "RAW_CYCLE_STRUCTURE_NOT_IDENTIFIABLE"
    )
    return {
        "status": status,
        "template_frozen": True,
        "complete_cycles_by_stratum": counts,
        "loaded_duration_median_h": loaded_median,
        "loaded_duration_cv": loaded_cv,
        "gates": gates,
        "automatic_next_phase": "NO",
    }


def plot_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(calibration: pd.DataFrame, runs: pd.DataFrame, cycles: pd.DataFrame, durations: pd.DataFrame, mapping: pd.DataFrame, template: dict[str, Any] | None) -> None:
    if not calibration.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.step(calibration["start_ophrs2_h"], calibration["median_set_current_A"], where="post", color=BLUE, linewidth=1.15)
        ax.set(xlabel="Operating age (h)", ylabel="Set current (A)", title="Calibration set-current run sequence")
        plot_style(ax); save_figure(fig, "calibration_set_current_sequence.png")
        labels = pd.Categorical(calibration["operation_point"].astype(str))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.step(calibration["start_ophrs2_h"], labels.codes, where="post", color=TEAL, linewidth=1.15)
        ax.set_yticks(range(len(labels.categories)), labels.categories, fontsize=8)
        ax.set(xlabel="Operating age (h)", ylabel="Operation point", title="Calibration operation-point run sequence")
        plot_style(ax); save_figure(fig, "calibration_operation_point_sequence.png")
    if template is not None and not cycles.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        selected = cycles.loc[cycles["complete_cycle"]].head(5)
        for index, cycle in enumerate(selected.itertuples(index=False), start=1):
            sequence = runs.loc[(runs["run_id"] >= cycle.start_run_id) & (runs["run_id"] < cycle.end_run_id_exclusive)]
            elapsed = sequence["start_ophrs2_h"] - float(cycle.start_ophrs2_h)
            ax.step(elapsed, sequence["median_set_current_A"], where="post", linewidth=1.2, label=f"cycle {cycle.raw_cycle_id}")
        ax.set(xlabel="Hours since raw-cycle start", ylabel="Set current (A)", title="Overlaid reconstructed raw cycles")
        ax.legend(frameon=False, fontsize=9, ncol=2); plot_style(ax); save_figure(fig, "overlaid_raw_cycle_operation_sequences.png")
    all_duration = durations.loc[durations["scope"] == "ALL"] if not durations.empty else pd.DataFrame()
    if not cycles.empty:
        complete = cycles.loc[cycles["complete_cycle"]]
        for column, title, filename, color in [
            ("loaded_duration_h", "Complete-cycle loaded-duration distribution", "cycle_loaded_duration_distribution.png", GREEN),
            ("wall_duration_h", "Complete-cycle wall-clock-duration distribution", "cycle_wall_duration_distribution.png", VIOLET),
        ]:
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            ax.hist(complete[column].dropna(), bins=min(24, max(5, len(complete))), color=color, edgecolor="white")
            ax.set(xlabel=column.replace("_", " "), ylabel="Count", title=title)
            plot_style(ax); save_figure(fig, filename)
        counts = [int(((complete["age_stratum"] == label)).sum()) for label in AGE_STRATA]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.bar(list(AGE_STRATA), counts, color=[BLUE, TEAL, VIOLET])
        ax.set(xlabel="Development age stratum", ylabel="Complete raw cycles", title="Age-stratum raw-cycle support")
        plot_style(ax); save_figure(fig, "age_stratum_cycle_support.png")
    if not mapping.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.scatter(mapping["raw_cycle_id"], mapping["mapped_30min_bin_count"], s=15, color=BLUE, label="mapped 30-min bins")
        ax.scatter(mapping["raw_cycle_id"], mapping["longest_contiguous_30min_run"], s=15, color=RED, label="longest contiguous run")
        ax.set(xlabel="Raw cycle ID", ylabel="0.5-h bins", title="Raw-cycle to 30-minute mapping loss")
        ax.legend(frameon=False); plot_style(ax); save_figure(fig, "raw_cycle_to_30min_mapping.png")
        fractions = np.divide(
            mapping["mixed_30min_bin_count"], mapping["mapped_30min_bin_count"],
            out=np.full(len(mapping), np.nan), where=mapping["mapped_30min_bin_count"].to_numpy() > 0,
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.hist(pd.Series(fractions).dropna(), bins=20, color=RED, edgecolor="white")
        ax.set(xlabel="Mixed-bin fraction per raw cycle", ylabel="Count", title="30-minute mixed-operation-point burden")
        plot_style(ax); save_figure(fig, "mixed_30min_bin_fraction.png")


def dump_yaml_like(path: Path, payload: dict[str, Any]) -> None:
    # JSON is valid YAML 1.2 and avoids a runtime dependency for a frozen result.
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_source_manifest(paths: Sequence[Path], timeline: pd.DataFrame, summary: dict[str, Any], provenance: dict[str, str]) -> None:
    payload = {
        "phase": "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit",
        "status": "EXECUTED_DEVELOPMENT_ONLY",
        "base_commit": BASE_COMMIT,
        "execution_head": provenance["current_head"],
        "protocol_sha256": provenance["protocol_sha256"],
        "raw_source": {
            "root": str(RAW_ROOT),
            "relative_stream": "Loadcycle_data/**/Loadcycle_*.csv",
            "file_count": len(paths),
            "file_sha256": "NOT_COMPUTED_BY_DESIGN",
            "files": [
                {"path": str(path.relative_to(RAW_ROOT)).replace("\\", "/"), "size_bytes": path.stat().st_size}
                for path in paths
            ],
        },
        "discovery_columns": list(REQUIRED_COLUMNS),
        "voltage_used_for_cycle_discovery": False,
        "polarization_used": False,
        "test_used_for_decision": False,
        "development_firewall": summary,
    }
    dump_yaml_like(SOURCE_MANIFEST, payload)


def write_report(result: dict[str, Any], template: dict[str, Any] | None, timeline: pd.DataFrame, mapping: pd.DataFrame, provenance: dict[str, str]) -> None:
    complete_total = int(result.get("complete_cycles_by_stratum", {}).get("EARLY", 0) + result.get("complete_cycles_by_stratum", {}).get("MIDDLE", 0) + result.get("complete_cycles_by_stratum", {}).get("LATE", 0))
    mapping_fraction = float(
        np.divide(mapping["mixed_30min_bin_count"].sum(), mapping["mapped_30min_bin_count"].sum())
    ) if not mapping.empty and mapping["mapped_30min_bin_count"].sum() else np.nan
    max_run = int(mapping["longest_contiguous_30min_run"].max()) if not mapping.empty else 0
    if result["status"] == "RAW_CYCLE_STRUCTURE_QUALIFIED" and max_run <= 7:
        mq_reason = "AGGREGATION_OR_SEGMENTATION_LOSS"
    elif result["status"] == "RAW_CYCLE_STRUCTURE_NOT_IDENTIFIABLE":
        mq_reason = "RAW_STRUCTURE_NOT_REPEATABLE"
    elif mapping_fraction > 0:
        mq_reason = "MIXED"
    else:
        mq_reason = "UNRESOLVED"
    gate_lines = "\n".join(f"- `{key}`: **{value}**" for key, value in result.get("gates", {}).items())
    report = f"""# Final raw-resolution cycle reconstruction audit

## Scope and firewall

This independent Phase RCR audit used the raw Loadcycle operating-state stream
only. It used no voltage field for cycle discovery, no polarization input, no
Test-side row for a decision, no neural training, no SOH construction, and no
RUL model. The strict analysis firewall was `OpHrs2 [h] < {BOUNDARY_H} h`.
The execution head was `{provenance['current_head']}`; the required research
base `{BASE_COMMIT}` is an ancestor. The frozen protocol SHA256 is
`{provenance['protocol_sha256']}`.

## Frozen procedure

State runs were run-length encoded using `operation_point + rounded(I.S.set
[A], 3 decimals)`. Stops and zero-current runs remain encoded. The calibration
region is the first 100 Development operating hours. The anchor and template
were selected by frozen recurrence ranking alone; neither voltage nor health
information entered the selection.

## Result

**Final status: `{result['status']}`**

Frozen template: **{'YES' if template else 'NO'}**.  Complete cycles assigned
within the three Development strata: **{complete_total}**.

{gate_lines or '- No template gate could be evaluated.'}

## Evidence A and Evidence B

`CYCLE_SEQUENCE_SIMILARITY.csv` reports set-current sequence recurrence
(Evidence A) separately from operation-point sequence recurrence (Evidence B).
Its exact-order operation-point LCS fraction is the frozen fixed-order core
criterion; the accompanying edit distance is descriptive only. Neither is DTW
or elastic alignment. `RAW_CYCLE_CATALOG.csv` separately records missing/extra
positive states, shortened/extended dwell states, and stop-run variation.

## Why the 0.5 h MQ route failed

The raw-to-30-minute mapping is documented in
`RAW_TO_30MIN_MAPPING_AUDIT.csv`. Across reconstructed intervals, the mixed-bin
fraction is **{mapping_fraction:.3f}** and the longest run under the already
frozen Phase-MQ eligible-bin rule is **{max_run}** bins. The registered interpretation is
`{mq_reason}`. This statement concerns only the frozen MQ 0.5 h protocol and
does not change historical MQ files or infer a health result.

## Historical documentation debt

The known MQ A0 counter-label issue, absent Attempt-2 execution audit, and
substatus/phase-status mapping inconsistency remain
`NON_BLOCKING_DOCUMENTATION_DEBT`; Phase RCR did not repair or rerun them.

## Stopping point

No automatic Phase RHA is created, even if raw-cycle structure qualified.
This audit contains no voltage health trend, SOH, EOL, RUL, model metric, or
backbone decision.
"""
    (REPORTS / "FINAL_RAW_CYCLE_AUDIT.md").write_text(report, encoding="utf-8")
    return {"mq_failure_reason": mq_reason, "mixed_fraction": mapping_fraction, "max_mapped_run": max_run}


def ensure_output_dirs() -> None:
    for directory in (OUTPUTS, FIGURES, REPORTS, FINAL):
        directory.mkdir(parents=True, exist_ok=True)


def write_empty_outputs(reason: str, provenance: dict[str, str], status: str) -> None:
    ensure_output_dirs()
    empty_specs = {
        "RAW_STATE_RUNS.csv": ["run_id"],
        "CALIBRATION_SEQUENCE_AUDIT.csv": ["sequence_index"],
        "RAW_CYCLE_CATALOG.csv": ["raw_cycle_id"],
        "CYCLE_SEQUENCE_SIMILARITY.csv": ["raw_cycle_id"],
        "CYCLE_DURATION_AUDIT.csv": ["scope"],
        "AGE_STRATUM_CYCLE_SUPPORT.csv": ["operation_point"],
        "RAW_TO_30MIN_MAPPING_AUDIT.csv": ["raw_cycle_id"],
        "RAW_TIMELINE_AUDIT.csv": ["source_file"],
    }
    for filename, columns in empty_specs.items():
        pd.DataFrame(columns=columns).to_csv(OUTPUTS / filename, index=False)
    (OUTPUTS / "CYCLE_TEMPLATE.json").write_text(json.dumps({"frozen": False, "reason": reason}, indent=2), encoding="utf-8")
    payload = {"status": status, "reason": reason, "automatic_next_phase": "NO"}
    (OUTPUTS / "GATE_EVALUATION.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dump_yaml_like(FINAL / "FINAL_DECISION.yaml", {**payload, **provenance, "voltage_used_for_cycle_discovery": False, "test_used": False})
    (REPORTS / "FINAL_RAW_CYCLE_AUDIT.md").write_text(
        f"# Final raw-resolution cycle reconstruction audit\n\n**Final status: `{status}`**\n\n{reason}\n\nNo Test, voltage discovery, polarization, neural training, SOH, or RUL input was used.\n",
        encoding="utf-8",
    )


def run_audit() -> dict[str, Any]:
    provenance = require_provenance()
    ensure_output_dirs()
    try:
        paths = locate_raw_files()
        runs, timeline, source_summary = stream_and_compress(paths)
    except AuditFailure as exc:
        write_empty_outputs(str(exc), provenance, "RAW_SOURCE_OR_PROTOCOL_LIMITATION")
        return {"status": "RAW_SOURCE_OR_PROTOCOL_LIMITATION", "reason": str(exc), **provenance}
    runs.to_csv(OUTPUTS / "RAW_STATE_RUNS.csv", index=False)
    timeline.to_csv(OUTPUTS / "RAW_TIMELINE_AUDIT.csv", index=False)
    template, calibration, candidates, calibration_lower, calibration_upper = discover_template(runs)
    calibration_output = calibration.copy()
    calibration_output["calibration_start_h"] = calibration_lower
    calibration_output["calibration_end_h"] = calibration_upper
    calibration_output.to_csv(OUTPUTS / "CALIBRATION_SEQUENCE_AUDIT.csv", index=False)
    if template is None:
        cycles = pd.DataFrame(columns=["raw_cycle_id", "complete_cycle", "candidate_scale_pass", "age_stratum"])
        similarity = pd.DataFrame(columns=["raw_cycle_id"])
        durations = duration_audit(cycles)
        support = pd.DataFrame(columns=["operation_point", "age_stratum", "repeated_support_pass_ge_10"])
        mapping = pd.DataFrame(columns=["raw_cycle_id"])
        (OUTPUTS / "CYCLE_TEMPLATE.json").write_text(json.dumps({"frozen": False, "candidates": candidates.to_dict(orient="records")}, indent=2), encoding="utf-8")
    else:
        (OUTPUTS / "CYCLE_TEMPLATE.json").write_text(json.dumps({**template, "candidates": candidates.to_dict(orient="records")}, indent=2), encoding="utf-8")
        cycles, similarity = reconstruct_cycles(runs, template)
        durations = duration_audit(cycles)
        support = age_support(cycles, template, runs)
        mapping, mapping_firewall = mapping_audit(cycles)
        source_summary.update(mapping_firewall)
    cycles.to_csv(OUTPUTS / "RAW_CYCLE_CATALOG.csv", index=False)
    similarity.to_csv(OUTPUTS / "CYCLE_SEQUENCE_SIMILARITY.csv", index=False)
    durations.to_csv(OUTPUTS / "CYCLE_DURATION_AUDIT.csv", index=False)
    support.to_csv(OUTPUTS / "AGE_STRATUM_CYCLE_SUPPORT.csv", index=False)
    mapping.to_csv(OUTPUTS / "RAW_TO_30MIN_MAPPING_AUDIT.csv", index=False)
    update_source_manifest(paths, timeline, source_summary, provenance)
    result = evaluate_gates(template, cycles, durations, support)
    result.update({
        "base_commit": BASE_COMMIT,
        "execution_head": provenance["current_head"],
        "protocol_sha256": provenance["protocol_sha256"],
        "post_boundary_rows_used": 0,
        "voltage_used_for_cycle_discovery": False,
        "polarization_used": False,
        "test_used_for_decision": False,
        "neural_training": False,
        "SOH_constructed": False,
        "RUL_trained": False,
    })
    (OUTPUTS / "GATE_EVALUATION.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_metrics = write_report(result, template, timeline, mapping, provenance)
    dump_yaml_like(FINAL / "FINAL_DECISION.yaml", {**result, **report_metrics})
    make_figures(calibration, runs, cycles, durations, mapping, template)
    return result | report_metrics | {"source_summary": source_summary, "template": template}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate frozen provenance and schema only.")
    arguments = parser.parse_args()
    if arguments.dry_run:
        provenance = require_provenance()
        paths = locate_raw_files()
        for path in paths:
            validate_raw_schema(path)
        print(json.dumps({"dry_run": True, "raw_files": len(paths), **provenance}, indent=2))
        return 0
    result = run_audit()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] != "RAW_SOURCE_OR_PROTOCOL_LIMITATION" else 2


if __name__ == "__main__":
    sys.exit(main())

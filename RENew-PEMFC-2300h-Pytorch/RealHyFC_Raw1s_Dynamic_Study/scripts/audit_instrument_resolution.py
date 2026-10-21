"""Read-only original-CSV audit of observable field-change cadence.

This does not identify a sensor's hidden refresh clock. Exact finite changes in
CSV values are events; unchanged values remain observations, not missingness.
All timing is original DateTime; only within-file 950..1050 ms links are joined.
No cache, model, old experiment, or source file is modified.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parent.parent
OUT = STUDY / "outputs"
FIELDS = [
    "I.S.act [A]", "I.S.set [A]", "I.S.set.internal [A]",
    "j.S.act [A/cm2]", "operation_point", "OpHrs2 [h]",
    "T.Si.A [°C]", "T.Si.C [°C]", "T.Si.CL [°C]", "T.So.CL [°C]",
    "p.Si.A [kPag]", "p.Si.C [kPag]",
]
CHUNK_ROWS = 100_000


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def numeric(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    text = series.astype("string").str.strip().str.replace("\ufeff", "", regex=False)
    both = text.str.contains(",", regex=False, na=False) & text.str.contains(".", regex=False, na=False)
    text = text.mask(both, text.str.replace(",", "", regex=False))
    return pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def quantile_from_hist(hist: collections.Counter, q: float):
    """Exact numpy-compatible linear quantile from integer-millisecond counts."""
    size = sum(hist.values())
    if not size:
        return None
    location = (size - 1) * q
    lower, upper = int(np.floor(location)), int(np.ceil(location))
    cumulative = 0
    low_value = high_value = None
    for value, count in sorted(hist.items()):
        cumulative += count
        if low_value is None and cumulative > lower:
            low_value = value
        if cumulative > upper:
            high_value = value
            break
    return (low_value + (high_value - low_value) * (location - lower)) / 1000.0


class FieldCadence:
    def __init__(self, field: str):
        self.field = field
        self.rows = self.finite_rows = self.pairs = self.changed_pairs = 0
        self.finite_runs = self.change_event_runs = 0
        self.max_hold_ms = 0
        self.last_value = np.nan
        self.last_event_ms = None
        self.last_hold_start_ms = None
        self.intervals_ms = collections.Counter()
        self.change_magnitude = collections.Counter()
        self.negative_changed_pairs = 0

    def update(self, ts: np.ndarray, dt_adjacency: np.ndarray, values: np.ndarray):
        self.rows += len(values)
        finite = np.isfinite(values)
        self.finite_rows += int(finite.sum())
        prev = np.r_[self.last_value, values[:-1]]
        pair = dt_adjacency & finite & np.isfinite(prev)
        changed = pair & (values != prev)
        self.pairs += int(pair.sum())
        self.changed_pairs += int(changed.sum())
        self.negative_changed_pairs += int((changed & (values < prev)).sum())
        self.finite_runs += int((finite & ~pair).sum())

        # Each finite chain uses only links satisfying the DateTime rule. A NaN,
        # file seam, duplicate, or gap resets both event and constant-value runs.
        segment = np.cumsum(~pair, dtype=np.int64)
        event_idx = np.flatnonzero(changed)
        if len(event_idx):
            event_time = ts[event_idx]
            event_segment = segment[event_idx]
            connected = event_segment[1:] == event_segment[:-1]
            intervals = np.diff(event_time)[connected]
            self.change_event_runs += int(1 + (~connected).sum())
            if event_segment[0] == 0 and self.last_event_ms is not None:
                intervals = np.r_[event_time[0] - self.last_event_ms, intervals]
                self.change_event_runs -= 1
            if len(intervals):
                unique, counts = np.unique(intervals, return_counts=True)
                self.intervals_ms.update(dict(zip(unique.tolist(), counts.tolist())))
            if event_segment[-1] == segment[-1]:
                self.last_event_ms = int(event_time[-1])
            elif segment[-1] != 0:
                self.last_event_ms = None
            if self.field in ("OpHrs2 [h]", "operation_point", "I.S.set [A]", "I.S.set.internal [A]"):
                # Rounding only stabilizes histogram labels; exact != above is
                # the definition of change and is never altered by rounding.
                magnitude = np.round(values[changed] - prev[changed], 10)
                unique, counts = np.unique(magnitude, return_counts=True)
                self.change_magnitude.update(dict(zip(unique.tolist(), counts.tolist())))
        elif segment[-1] != 0:
            self.last_event_ms = None

        reset = (~pair) | changed
        starts = np.unique(np.r_[0, np.flatnonzero(reset)])
        ends = np.r_[starts[1:] - 1, len(ts) - 1]
        hold_start_time = ts[starts].copy()
        if pair[0] and not changed[0] and self.last_hold_start_ms is not None:
            hold_start_time[0] = self.last_hold_start_ms
        holds = ts[ends] - hold_start_time
        holds = holds[finite[starts] & finite[ends]]
        if len(holds):
            self.max_hold_ms = max(self.max_hold_ms, int(holds.max()))
        self.last_hold_start_ms = int(hold_start_time[-1]) if finite[-1] else None
        self.last_value = float(values[-1])

    def result(self, file_name: str):
        interval_count = sum(self.intervals_ms.values())
        near_1s = sum(n for ms, n in self.intervals_ms.items() if 950 <= ms <= 1050)
        unit = re.search(r"\[([^]]+)\]", self.field)
        return dict(
            file=file_name, field=self.field,
            value_unit=unit[1] if unit else "dimensionless_state_code",
            interval_unit="s, original DateTime",
            rows=self.rows, finite_rows=self.finite_rows,
            missing_or_nonfinite_rows=self.rows - self.finite_rows,
            valid_contiguous_pairs=self.pairs,
            changed_pairs=self.changed_pairs,
            unchanged_pairs=self.pairs - self.changed_pairs,
            changed_pair_ratio=self.changed_pairs / self.pairs if self.pairs else None,
            unchanged_pair_ratio=(self.pairs - self.changed_pairs) / self.pairs if self.pairs else None,
            finite_contiguous_runs=self.finite_runs,
            observed_change_events=self.changed_pairs,
            change_event_intervals=interval_count,
            censored_first_change_events=self.changed_pairs - interval_count,
            change_interval_min_s=min(self.intervals_ms) / 1000.0 if interval_count else None,
            change_interval_p50_s=quantile_from_hist(self.intervals_ms, 0.5),
            change_interval_p95_s=quantile_from_hist(self.intervals_ms, 0.95),
            change_interval_max_s=max(self.intervals_ms) / 1000.0 if interval_count else None,
            near_1s_change_interval_count=near_1s,
            near_1s_change_interval_ratio=near_1s / interval_count if interval_count else None,
            max_observed_constant_run_duration_s=self.max_hold_ms / 1000.0,
            negative_changed_pairs=self.negative_changed_pairs,
        )

    def merge_counts(self, other):
        for name in ("rows", "finite_rows", "pairs", "changed_pairs", "finite_runs", "change_event_runs", "negative_changed_pairs"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.max_hold_ms = max(self.max_hold_ms, other.max_hold_ms)
        self.intervals_ms.update(other.intervals_ms)
        self.change_magnitude.update(other.change_magnitude)


def verify_algorithm():
    # Events at 1, 3, 4 s yield gaps 2 and 1 s; a real gap and a missing value
    # each censor the next first change. Split at 3 s to exercise chunk seams.
    times = np.array([0, 1000, 2000, 3000, 4000, 8000, 9000, 10000, 11000, 12000], dtype=np.int64)
    values = np.array([1, 2, 2, 3, 4, 5, 6, np.nan, 7, 8], dtype=np.float64)
    expected = FieldCadence("synthetic")
    delta = np.diff(times, prepend=times[0])
    expected.update(times, (delta >= 950) & (delta <= 1050), values)
    split = FieldCadence("synthetic")
    split.update(times[:3], ((delta >= 950) & (delta <= 1050))[:3], values[:3])
    split.update(times[3:], ((delta >= 950) & (delta <= 1050))[3:], values[3:])
    assert expected.result("test") == split.result("test")
    assert expected.changed_pairs == 5
    assert expected.intervals_ms == {1000: 1, 2000: 1}
    assert quantile_from_hist(collections.Counter({1000: 1, 2000: 1}), 0.95) == 1.95


def main():
    start = time.time()
    verify_algorithm()
    csv_path = OUT / "instrument_update_intervals.csv"
    json_path = OUT / "instrument_update_intervals.json"
    if csv_path.exists() or json_path.exists():
        raise RuntimeError("Instrument audit already exists; do not silently overwrite an audit.")
    inventory_path = OUT / "raw_file_inventory.csv"
    inventory = pd.read_csv(inventory_path).to_dict("records")
    assert len(inventory) == 17
    grand = {field: FieldCadence(field) for field in FIELDS}
    summaries = []
    source_checks = []
    total_rows = 0
    for source in inventory:
        path = (REPO / source["path"]).resolve(strict=True)
        if path.name != source["file"] or path.suffix.lower() != ".csv":
            raise RuntimeError("Unexpected source identity")
        before = path.stat()
        with path.open("rb") as stream:
            if stream.read(200).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError("Git LFS pointer is not CSV payload")
        source_hash = digest(path)
        assert source_hash == source["sha256"] == source["lfs_oid"]
        assert before.st_size == source["bytes"] == source["lfs_size"]
        local = {field: FieldCadence(field) for field in FIELDS}
        previous_time = None
        row_count = 0
        dt_breaks = 0
        for frame in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=["DateTime"] + FIELDS, chunksize=CHUNK_ROWS, low_memory=False):
            dates = pd.to_datetime(frame["DateTime"], errors="raise")
            assert not dates.isna().any()
            ts = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64) // 1_000_000
            delta = np.diff(ts, prepend=previous_time if previous_time is not None else ts[0])
            adjacency = (delta >= 950) & (delta <= 1050)
            if previous_time is None:
                adjacency[0] = False
            dt_breaks += int((~adjacency).sum())
            for field in FIELDS:
                local[field].update(ts, adjacency, numeric(frame[field]))
            previous_time = int(ts[-1])
            row_count += len(frame)
        after = path.stat()
        assert row_count == source["rows"], (path.name, row_count, source["rows"])
        assert before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        source_checks.append(dict(
            file=path.name, relative_path=source["path"],
            expected_sha256=source["sha256"], observed_sha256=source_hash,
            sha256_matches_existing_raw_inventory=True,
            size_before=before.st_size, size_after=after.st_size,
            mtime_ns_before=before.st_mtime_ns, mtime_ns_after=after.st_mtime_ns,
            size_and_mtime_unchanged=True,
            rows=row_count, datetime_segments=dt_breaks,
        ))
        for field in FIELDS:
            summaries.append(local[field].result(path.name))
            grand[field].merge_counts(local[field])
        total_rows += row_count
        print(json.dumps(dict(stage="instrument_cadence", file=path.name, rows=row_count, total_rows=total_rows, elapsed_s=round(time.time() - start, 1))), flush=True)

    combined = [grand[field].result("ALL_17_FILES_NO_SEAM_LINKS") for field in FIELDS]
    assert total_rows == int(pd.read_csv(inventory_path)["rows"].sum())
    all_rows = combined + summaries
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    # Independent real N-1-pair recount of the original clocks, including real
    # file-seam pairs for parity with raw_audit (never used as model links).
    # No prepend is allowed: it would create a fictional first observation.
    legacy_audit = json.loads((OUT / "raw_audit.json").read_text(encoding="utf-8"))
    clock_time = np.memmap(STUDY / "cache/time_ms.bin", dtype="<i8", mode="r", shape=(total_rows,))
    clock_age = np.memmap(STUDY / "cache/age_h.bin", dtype="<f8", mode="r", shape=(total_rows,))
    actual_dt = np.diff(clock_time)
    assert len(actual_dt) == total_rows - 1
    actual_counts = dict(
        DateTime_backward_or_duplicate_pairs=int((actual_dt <= 0).sum()),
        near_1second_pairs=int(((actual_dt >= 950) & (actual_dt <= 1050)).sum()),
    )
    del actual_dt
    actual_dage = np.diff(clock_age)
    assert len(actual_dage) == total_rows - 1
    actual_counts.update(
        OpHrs2_frozen_pairs=int((np.abs(actual_dage) < 1e-9).sum()),
        OpHrs2_backward_pairs=int((actual_dage < -1e-7).sum()),
    )
    del actual_dage, clock_time, clock_age
    clock_correction = dict(
        method="Read-only original clock memmaps; np.diff over all actual N rows, no prepend; exactly N-1 actual adjacent pairs. Includes real file seams for raw_audit count parity, not for instrument event continuity.",
        rows=total_rows, real_adjacent_pairs=total_rows - 1,
        previous_recorded_clock_counts=legacy_audit["clock_counts"],
        corrected_actual_clock_counts=actual_counts,
        new_minus_previous={key: actual_counts[key] - legacy_audit["clock_counts"][key] for key in actual_counts},
        diagnosis="The previous audit counted a synthesized first-row previous timestamp and age in clock_counts. Correct metadata only; raw values, cache, valid windows, frozen protocol and thresholds remain unchanged.",
    )
    report = dict(
        status="INSTRUMENT_OBSERVABLE_CHANGE_CADENCE_AUDIT_COMPLETE",
        files=len(inventory), rows=total_rows, fields=FIELDS,
        definitions=dict(
            adjacency="Within one original file; original DateTime difference 950..1050 ms; finite values of this specific field at both ends. All seams and missing values reset event interval chains.",
            change="Exact inequality between parsed float64 original CSV values; no tolerance, smoothing, interpolation or numerical downcast.",
            changed_pair_ratio="Changed finite 950..1050 ms adjacent pairs / all finite 950..1050 ms adjacent pairs; empirical fraction of approximately one-second links showing value change.",
            unchanged_pair_ratio="Equal finite 950..1050 ms adjacent pairs / all finite 950..1050 ms adjacent pairs. An unchanged value is not missing.",
            change_interval="DateTime separation between two consecutive change events in the same finite within-file contiguous chain. First events are left-censored and file/segment-end holds are right-censored.",
            quantiles="Exact linear-interpolated numpy-style quantiles of all eligible event intervals, reconstructed from integer millisecond histograms; not means of per-file quantiles.",
            near_1s_change_interval_ratio="Event intervals in [0.950,1.050] seconds / all eligible consecutive change-event intervals. This denominator differs from changed_pair_ratio.",
            max_observed_constant_run_duration_s="Longest observed same-value run from its first to last timestamp, including boundary-censored runs, never joining a gap or file seam.",
            interpretation="A CSV row each second does not prove a sensor internally refreshes at 1 Hz. No change may represent a held command, constant physical state, limited resolution or slower device refresh; this audit does not disambiguate those causes.",
            state_and_age="operation_point is a dimensionless state code; OpHrs2 is operating hours, not DateTime elapsed seconds. Slow changes, zero increments and quantization remain valid observations, not missing data.",
        ),
        sources=source_checks,
        aggregate=combined,
        clock_count_correction=clock_correction,
        event_interval_histograms_ms={field: [dict(interval_ms=int(ms), count=int(count)) for ms, count in sorted(grand[field].intervals_ms.items())] for field in FIELDS},
        original_value_change_histograms={field: [dict(change_in_original_field_unit=float(value), count=int(count)) for value, count in grand[field].change_magnitude.most_common(30)] for field in FIELDS if grand[field].change_magnitude},
        safeguards=dict(raw_inventory_sha256=digest(inventory_path), all_source_hashes_match_inventory=True, all_source_size_mtime_unchanged=True, no_shared_cache_writes=True, no_model_imports_or_gpu=True, algorithm_chunk_boundary_gap_and_missing_checks_pass=True),
        elapsed_s=time.time() - start,
    )
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], rows=total_rows, fields=len(FIELDS), elapsed_s=round(time.time() - start, 1)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

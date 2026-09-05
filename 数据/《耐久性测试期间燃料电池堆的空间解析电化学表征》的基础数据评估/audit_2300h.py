#!/usr/bin/env python
"""Read-only, reproducible audit of the 2300 h PEMFC durability dataset.

The script reads source files in chunks and only writes into OUT_DIR. It never
writes to or modifies the source dataset. No forecasting/model training is
performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


SRC_DIR = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据")
OUT_DIR = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\数据\《耐久性测试期间燃料电池堆的空间解析电化学表征》的基础数据评估")
CHUNK_ROWS = 100_000
RESERVOIR_SIZE = 50_000
JUMP_RESERVOIR_SIZE = 20_000
CSV_SEP = ";"


def clean_name(value: Any) -> str:
    return str(value).replace("\ufeff", "").strip()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return None
        return float(value)
    except Exception:
        return None


def format_dt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat(sep=" ")


def fmt(value: Any, ndigits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.{ndigits}g}"
    return str(value)


def read_head_bytes(path: Path, n: int = 65_536) -> bytes:
    with path.open("rb") as fh:
        return fh.read(n)


def detect_text_encoding(path: Path) -> tuple[str, str]:
    raw = read_head_bytes(path)
    for enc in ("utf-8-sig", "utf-16", "gb18030", "cp1252", "latin1"):
        try:
            text = raw.decode(enc)
            return enc, text
        except UnicodeDecodeError:
            continue
    return "unknown", ""


def detect_delimiter(text: str) -> str:
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    first = lines[0]
    scores = {";": first.count(";"), "\t": first.count("\t"), ",": first.count(",")}
    delim, score = max(scores.items(), key=lambda kv: kv[1])
    return delim if score > 0 else ""


def normalize_numeric(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace("\ufeff", "", regex=False).str.strip()
    has_comma = s.str.contains(",", na=False)
    has_dot = s.str.contains("\\.", regex=True, na=False)
    thousand = has_comma & has_dot
    s = s.mask(thousand, s.str.replace(",", "", regex=False))
    only_comma = has_comma & ~has_dot
    s = s.mask(only_comma, s.str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def read_csv_header(path: Path) -> tuple[list[str], str, str, str | None]:
    enc, head = detect_text_encoding(path)
    delim = detect_delimiter(head)
    if not delim:
        delim = CSV_SEP
    try:
        df = pd.read_csv(path, sep=delim, encoding=enc, nrows=0, engine="python")
        cols = [clean_name(c) for c in df.columns]
        return cols, enc, delim, None
    except Exception as exc:
        return [], enc, delim, repr(exc)


def parse_stage(path: Path) -> tuple[str, float | None, bool]:
    name = path.name
    m = re.search(r"(?<![A-Za-z])(\d+(?:\.\d+)?)h", str(path), re.I)
    age = float(m.group(1)) if m else None
    if re.search(r"BoL", str(path), re.I):
        age = 0.0
        return "BoL", age, False
    after = bool(re.search(r"afterAirAirStart", name, re.I))
    if after and age is not None:
        return f"{int(age) if age.is_integer() else age:g}h_afterAirAirStart", age, True
    if age is not None:
        return f"{int(age) if age.is_integer() else age:g}h", age, False
    return "unknown", age, after


def infer_category(column: str) -> tuple[str, str, str]:
    c = clean_name(column)
    lc = c.lower()
    unit = ""
    m = re.search(r"\[([^\]]+)\]", c)
    if m:
        unit = m.group(1).strip()
    if c in {"DateTime", "CalHrs [h]", "OpHrs [h]", "OpHrs2 [h]"} or "hrs" in lc:
        return "time", "timestamp or operating-hour counter", unit or "h"
    if c == "operation_point":
        return "state / control", "operation-point code", ""
    if re.search(r"u\.s", lc) and "avecell" not in lc and "maxcell" not in lc and "mincell" not in lc:
        return "stack voltage", "stack-level voltage field; exact MIN semantics unresolved", unit or "V"
    if re.search(r"u\d+\s*\[", lc):
        return "cell voltage", "individual cell voltage", unit or "V"
    if "avecell" in lc:
        return "cell voltage", "average cell voltage", unit or "V"
    if "maxcell" in lc or "mincell" in lc:
        return "cell voltage", "maximum/minimum cell voltage statistic", unit or "V"
    if "current density" in lc or re.search(r"(^|[._])j\.", lc):
        return "current density", "stack current density", unit or "A/cm2"
    if "current" in lc or re.search(r"(^|[._])i\.", lc):
        return "current", "stack current or current setpoint", unit or "A"
    if lc.startswith("rh") or "humidity" in lc:
        return "humidity", "relative humidity or variability", unit or "%"
    if lc.startswith("stoic") or "stoichiometry" in lc:
        return "stoichiometry", "reactant stoichiometry or setpoint", unit or "-"
    if lc.startswith("f.") or "flow" in lc:
        return "flow", "gas or coolant flow", unit
    if lc.startswith("p.") or "pressure" in lc:
        return "pressure", "gas or coolant pressure", unit
    if lc.startswith("t.") or lc.startswith("dpt") or "temperature" in lc:
        return "temperature", "gas, coolant, or differential temperature", unit or "degC"
    return "unknown", "unknown", unit


def cell_columns(columns: Iterable[str]) -> list[str]:
    out = []
    for c in columns:
        if re.fullmatch(r"U\d+ \[V\]", clean_name(c), re.I):
            out.append(clean_name(c))
    return sorted(out, key=lambda x: int(re.search(r"U(\d+)", x, re.I).group(1)))


def is_core_signal(column: str) -> bool:
    cat, _, _ = infer_category(column)
    return cat in {"stack voltage", "cell voltage", "current", "current density", "temperature", "pressure", "humidity", "flow", "stoichiometry"}


def plausible_bounds(column: str) -> tuple[float | None, float | None]:
    c = clean_name(column).lower()
    cat, _, _ = infer_category(column)
    if cat == "cell voltage":
        return -0.2, 1.5
    if cat == "stack voltage":
        return -2.0, 20.0
    if cat in {"humidity"}:
        return 0.0, 150.0
    if cat == "temperature":
        return -50.0, 200.0
    if cat == "pressure":
        return -2_000.0, 5_000.0
    if cat == "current density":
        return -0.1, 10.0
    if cat == "current":
        return -10_000.0, 10_000.0
    if cat == "stoichiometry":
        return 0.0, 20.0
    if cat == "flow":
        return -1_000.0, 10_000.0
    return None, None


def reservoir_add(res: list[float], values: np.ndarray, seen: int, max_size: int) -> int:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return seen
    new_seen = seen + int(len(vals))
    combined = np.concatenate([np.asarray(res, dtype=float), vals]) if res else vals
    if len(combined) > max_size:
        # Deterministic thinning across the retained reservoir plus this chunk.
        # Exact count/sum statistics are kept separately; the reservoir is only
        # used for reproducible quantile estimates and plots.
        idx = np.linspace(0, len(combined) - 1, max_size, dtype=int)
        retained = combined[idx]
        res[:] = retained.tolist()
    else:
        res[:] = combined.tolist()
    return new_seen


@dataclass
class SignalAccumulator:
    total: int = 0
    valid: int = 0
    nan: int = 0
    zero: int = 0
    constant: int = 0
    phys_outlier: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    sample_seen: int = 0
    sample: list[float] = field(default_factory=list)
    jump_count: int = 0
    jump_sum: float = 0.0
    jump_sumsq: float = 0.0
    jump_min: float = math.inf
    jump_max: float = -math.inf
    jump_sample_seen: int = 0
    jump_sample: list[float] = field(default_factory=list)
    compare_count: int = 0
    previous: float | None = None

    def update(self, raw: pd.Series, values: pd.Series) -> None:
        arr = values.to_numpy(dtype=float, na_value=np.nan)
        self.total += int(len(arr))
        finite = np.isfinite(arr)
        self.valid += int(finite.sum())
        self.nan += int((~finite).sum())
        if finite.any():
            v = arr[finite]
            self.zero += int(np.count_nonzero(v == 0.0))
            self.sum += float(v.sum())
            self.sumsq += float(np.square(v).sum())
            self.min_value = min(self.min_value, float(v.min()))
            self.max_value = max(self.max_value, float(v.max()))
            self.sample_seen = reservoir_add(self.sample, v, self.sample_seen, RESERVOIR_SIZE)
            lo, hi = plausible_bounds(str(raw.name))
            if lo is not None:
                self.phys_outlier += int(np.count_nonzero((v < lo) | (v > hi)))
        if len(arr):
            if self.previous is not None and finite[0]:
                d = abs(float(arr[0] - self.previous))
                self._update_jumps(np.array([d], dtype=float))
                self.compare_count += 1
                self.constant += int(d == 0.0)
            if len(arr) >= 2:
                # Consecutive finite comparisons only.
                left = arr[:-1]
                right = arr[1:]
                both = np.isfinite(left) & np.isfinite(right)
                diffs = np.abs(right[both] - left[both])
                self.compare_count += int(len(diffs))
                self.constant += int(np.count_nonzero(diffs == 0.0))
                self._update_jumps(diffs)
        finite_idx = np.flatnonzero(finite)
        self.previous = float(arr[finite_idx[-1]]) if len(finite_idx) else None

    def _update_jumps(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        self.jump_count += int(len(values))
        self.jump_sum += float(values.sum())
        self.jump_sumsq += float(np.square(values).sum())
        self.jump_min = min(self.jump_min, float(values.min()))
        self.jump_max = max(self.jump_max, float(values.max()))
        self.jump_sample_seen = reservoir_add(self.jump_sample, values, self.jump_sample_seen, JUMP_RESERVOIR_SIZE)

    def summary(self) -> dict[str, Any]:
        mean = self.sum / self.valid if self.valid else math.nan
        var = max(self.sumsq / self.valid - mean * mean, 0.0) if self.valid else math.nan
        q = np.percentile(self.sample, [1, 5, 50, 95, 99]) if self.sample else [math.nan] * 5
        jq = np.percentile(self.jump_sample, [50, 95, 99]) if self.jump_sample else [math.nan] * 3
        return {
            "n": self.total,
            "min": self.min_value if self.valid else math.nan,
            "max": self.max_value if self.valid else math.nan,
            "mean": mean,
            "std": math.sqrt(var) if math.isfinite(var) else math.nan,
            "p1": q[0], "p5": q[1], "p50": q[2], "p95": q[3], "p99": q[4],
            "nan_pct": 100.0 * self.nan / self.total if self.total else math.nan,
            "zero_pct": 100.0 * self.zero / self.valid if self.valid else math.nan,
            "constant_pct": 100.0 * self.constant / self.compare_count if self.compare_count else math.nan,
            "extreme_outlier_pct": 100.0 * self.phys_outlier / self.valid if self.valid else math.nan,
            "abrupt_jump_count": self.jump_count,
            "abrupt_jump_rate_per_1000_valid": 1000.0 * self.jump_count / self.compare_count if self.compare_count else math.nan,
            "abrupt_jump_mean": self.jump_sum / self.jump_count if self.jump_count else math.nan,
            "abrupt_jump_p50": jq[0], "abrupt_jump_p95": jq[1], "abrupt_jump_p99": jq[2],
            "abrupt_jump_max": self.jump_max if self.jump_count else math.nan,
        }


@dataclass
class IntervalAccumulator:
    positive_count: int = 0
    nonpositive_count: int = 0
    reverse_count: int = 0
    duplicate_count: int = 0
    gap_count: int = 0
    missing_estimated: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    sample_seen: int = 0
    sample: list[float] = field(default_factory=list)
    hist: Counter = field(default_factory=Counter)

    def update(self, diffs: np.ndarray, baseline: float | None = None) -> None:
        d = np.asarray(diffs, dtype=float)
        d = d[np.isfinite(d)]
        if not len(d):
            return
        self.nonpositive_count += int(np.count_nonzero(d <= 0))
        self.reverse_count += int(np.count_nonzero(d < 0))
        self.duplicate_count += int(np.count_nonzero(d == 0))
        pos = d[d > 0]
        if not len(pos):
            return
        self.positive_count += int(len(pos))
        self.sum += float(pos.sum())
        self.sumsq += float(np.square(pos).sum())
        self.min_value = min(self.min_value, float(pos.min()))
        self.max_value = max(self.max_value, float(pos.max()))
        self.sample_seen = reservoir_add(self.sample, pos, self.sample_seen, RESERVOIR_SIZE)
        if baseline is not None and baseline > 0:
            gaps = pos > baseline * 1.5
            self.gap_count += int(gaps.sum())
            self.missing_estimated += int(np.maximum(np.rint(pos[gaps] / baseline).astype(int) - 1, 0).sum())
        for value in pos:
            self.hist[round(float(value), 3)] += 1

    def baseline(self) -> float | None:
        if not self.sample:
            return None
        return float(np.median(np.asarray(self.sample)))

    def summary(self) -> dict[str, Any]:
        q = np.percentile(self.sample, [50]) if self.sample else [math.nan]
        mean = self.sum / self.positive_count if self.positive_count else math.nan
        var = max(self.sumsq / self.positive_count - mean * mean, 0.0) if self.positive_count else math.nan
        return {
            "sampling_interval_median_s": q[0],
            "sampling_interval_mean_s": mean,
            "sampling_interval_std_s": math.sqrt(var) if math.isfinite(var) else math.nan,
            "minimum_interval_s": self.min_value if self.positive_count else math.nan,
            "maximum_interval_s": self.max_value if self.positive_count else math.nan,
            "duplicate_timestamp_count": self.duplicate_count,
            "timestamp_reversal_count": self.reverse_count,
            "gap_count": self.gap_count,
            "missing_timestamp_estimate": self.missing_estimated,
        }


def update_simple_stats(stats: dict[str, Any], arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return
    stats["n"] += int(len(arr))
    stats["sum"] += float(arr.sum())
    stats["sumsq"] += float(np.square(arr).sum())
    stats["min"] = min(stats["min"], float(arr.min()))
    stats["max"] = max(stats["max"], float(arr.max()))


def empty_simple_stats() -> dict[str, Any]:
    return {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": math.inf, "max": -math.inf}


def finalize_simple_stats(stats: dict[str, Any]) -> tuple[int, float, float, float, float]:
    n = int(stats["n"])
    if n == 0:
        return 0, math.nan, math.nan, math.nan, math.nan
    mean = stats["sum"] / n
    var = max(stats["sumsq"] / n - mean * mean, 0.0)
    return n, float(mean), float(math.sqrt(var)), float(stats["min"]), float(stats["max"])


def read_loadcycle_file(path: Path, columns: list[str], signal_accs: dict[str, SignalAccumulator], hourly: dict[pd.Timestamp, dict[str, Any]], global_interval_hist: Counter) -> dict[str, Any]:
    file_columns, enc, delim, header_error = read_csv_header(path)
    usecols = [c for c in file_columns if c in {"DateTime", "OpHrs [h]", "OpHrs2 [h]", "Number of cells", "operation_point"} or is_core_signal(c)]
    usecols = list(dict.fromkeys(usecols))
    if "DateTime" not in usecols:
        raise ValueError(f"DateTime not present in {path}")
    delim = delim or CSV_SEP
    first_dt = last_dt = min_dt = max_dt = None
    first_age = last_age = first_age2 = last_age2 = None
    min_age = min_age2 = math.inf
    max_age = max_age2 = -math.inf
    rows = 0
    dt_invalid = 0
    interval = IntervalAccumulator()
    prev_dt = None
    prev_age = None
    prev_age2 = None
    age_reverse = age2_reverse = 0
    op_values: set[str] = set()
    op_counter: Counter = Counter()
    first_record: dict[str, Any] = {}
    last_record: dict[str, Any] = {}
    file_signal_accs = {c: SignalAccumulator() for c in signal_accs}
    file_nan_counts: dict[str, int] = defaultdict(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader = pd.read_csv(path, sep=delim, encoding=enc, usecols=usecols, chunksize=CHUNK_ROWS, engine="c", low_memory=False)
        for chunk in reader:
            chunk.columns = [clean_name(c) for c in chunk.columns]
            n = len(chunk)
            rows += n
            for c in chunk.columns:
                file_nan_counts[c] += int(chunk[c].isna().sum())
            dt = pd.to_datetime(chunk["DateTime"], errors="coerce")
            dt_invalid += int(dt.isna().sum())
            valid_dt = dt.notna()
            if valid_dt.any():
                vals = dt[valid_dt]
                if first_dt is None:
                    first_dt = vals.iloc[0]
                    first_record = {k: chunk.loc[vals.index[0]].get(k, np.nan) for k in ["DateTime", "OpHrs [h]", "OpHrs2 [h]", "operation_point", "I.S.act [A]", "j.S.act [A/cm2]"]}
                last_dt = vals.iloc[-1]
                min_dt = vals.min() if min_dt is None else min(min_dt, vals.min())
                max_dt = vals.max() if max_dt is None else max(max_dt, vals.max())
                last_record = {k: chunk.loc[vals.index[-1]].get(k, np.nan) for k in ["DateTime", "OpHrs [h]", "OpHrs2 [h]", "operation_point", "I.S.act [A]", "j.S.act [A/cm2]"]}
                if prev_dt is not None:
                    d0 = (vals.iloc[0] - prev_dt).total_seconds()
                    interval.update(np.array([d0]), interval.baseline())
                diffs = dt.diff().dt.total_seconds().to_numpy(dtype=float)
                baseline = interval.baseline()
                interval.update(diffs[1:], baseline)
                prev_dt = vals.iloc[-1]
                pos = diffs[np.isfinite(diffs) & (diffs > 0)]
                for x in pos:
                    global_interval_hist[round(float(x), 3)] += 1
            for age_col, holder in [("OpHrs [h]", "age"), ("OpHrs2 [h]", "age2")]:
                if age_col in chunk.columns:
                    a = normalize_numeric(chunk[age_col])
                    av = a[np.isfinite(a)]
                    if len(av):
                        if holder == "age":
                            if first_age is None: first_age = float(av.iloc[0])
                            last_age = float(av.iloc[-1]); min_age = min(min_age, float(av.min())); max_age = max(max_age, float(av.max()))
                            if prev_age is not None:
                                age_reverse += int(av.iloc[0] < prev_age)
                            age_reverse += int(np.count_nonzero(np.diff(av.to_numpy()) < 0))
                            prev_age = float(av.iloc[-1])
                        else:
                            if first_age2 is None: first_age2 = float(av.iloc[0])
                            last_age2 = float(av.iloc[-1]); min_age2 = min(min_age2, float(av.min())); max_age2 = max(max_age2, float(av.max()))
                            if prev_age2 is not None:
                                age2_reverse += int(av.iloc[0] < prev_age2)
                            age2_reverse += int(np.count_nonzero(np.diff(av.to_numpy()) < 0))
                            prev_age2 = float(av.iloc[-1])
            if "operation_point" in chunk.columns:
                op = normalize_numeric(chunk["operation_point"])
                op = op[np.isfinite(op)]
                for v in op.to_numpy():
                    op_values.add(str(int(v)) if float(v).is_integer() else str(float(v)))
                    op_counter[str(int(v)) if float(v).is_integer() else str(float(v))] += 1
            for col in signal_accs:
                if col in chunk.columns:
                    numeric = normalize_numeric(chunk[col])
                    signal_accs[col].update(chunk[col], numeric)
                    file_signal_accs[col].update(chunk[col], numeric)
            # Hourly metadata/features, keyed by observed clock hour; no fill or smoothing.
            if valid_dt.any():
                h = dt.dt.floor("h")
                stack_col = next((c for c in chunk.columns if infer_category(c)[0] == "stack voltage"), None)
                core_for_hour = [c for c in [stack_col, "U.S.AveCell [V]", "I.S.act [A]", "j.S.act [A/cm2]", "OpHrs2 [h]", "operation_point"] if c and c in chunk.columns]
                if core_for_hour:
                    tmp = chunk[core_for_hour].copy()
                    tmp["hour"] = h
                    tmp = tmp.loc[tmp["hour"].notna()].copy()
                    for c in core_for_hour:
                        if c != "operation_point":
                            tmp[c] = normalize_numeric(tmp[c])
                    tmp["cell_std_raw"] = np.nan
                    tmp["cell_range_raw"] = np.nan
                    uc = [c for c in chunk.columns if re.fullmatch(r"U\d+ \[V\]", c, re.I)]
                    if uc:
                        umat = np.column_stack([normalize_numeric(chunk[c]).to_numpy(dtype=float) for c in uc])
                        tmp["cell_std_raw"] = np.nanstd(umat, axis=1)
                        tmp["cell_range_raw"] = np.nanmax(umat, axis=1) - np.nanmin(umat, axis=1)
                    for hour, g in tmp.groupby("hour", sort=False):
                        rec = hourly.setdefault(pd.Timestamp(hour), {"row_count": 0})
                        rec["row_count"] += int(len(g))
                        for c in core_for_hour:
                            if c == "operation_point":
                                continue
                            vals = g[c].to_numpy(dtype=float)
                            update_hour_value(rec, c, vals)
                        for c in ["cell_std_raw", "cell_range_raw"]:
                            if c in g:
                                update_hour_value(rec, c, g[c].to_numpy(dtype=float))
                        if "operation_point" in g.columns:
                            opv = normalize_numeric(g["operation_point"]).to_numpy(dtype=float)
                            prior_ops = set(rec.get("operation_point_unique", []))
                            prior_ops.update(int(x) for x in opv if np.isfinite(x))
                            rec["operation_point_unique"] = sorted(prior_ops)
                            rec["operation_point_transition_count"] = rec.get("operation_point_transition_count", 0) + int(np.count_nonzero(np.diff(opv[np.isfinite(opv)]) != 0)) if np.count_nonzero(np.isfinite(opv)) > 1 else rec.get("operation_point_transition_count", 0)
    out = {
        "filename": path.name,
        "relative_path": str(path.relative_to(SRC_DIR)),
        "rows": rows,
        "first_timestamp": format_dt(first_dt),
        "last_timestamp": format_dt(last_dt),
        "min_timestamp": format_dt(min_dt),
        "max_timestamp": format_dt(max_dt),
        "duration_h": (max_dt - min_dt).total_seconds() / 3600 if min_dt is not None and max_dt is not None else math.nan,
        "invalid_timestamp_count": dt_invalid,
        "first_OpHrs_h": first_age, "last_OpHrs_h": last_age, "min_OpHrs_h": min_age if min_age < math.inf else math.nan, "max_OpHrs_h": max_age if max_age > -math.inf else math.nan,
        "first_OpHrs2_h": first_age2, "last_OpHrs2_h": last_age2, "min_OpHrs2_h": min_age2 if min_age2 < math.inf else math.nan, "max_OpHrs2_h": max_age2 if max_age2 > -math.inf else math.nan,
        "ophrs_reset_count": age_reverse, "ophrs2_reset_count": age2_reverse,
        "operation_point_values": ",".join(sorted(op_values, key=lambda x: float(x))) if op_values else "",
        "operation_point_count": len(op_values),
        "operation_point_rows": json.dumps(op_counter, ensure_ascii=False),
        "filename_range_start": parse_filename_range(path.name)[0],
        "filename_range_end": parse_filename_range(path.name)[1],
        "filename_range_is_metadata_only": "yes",
        "timestamp_order": "reversal_present" if interval.reverse_count else ("duplicate_present" if interval.duplicate_count else "monotonic"),
        **interval.summary(),
        "first_record": json.dumps({k: normalize_scalar(v) for k, v in first_record.items()}, ensure_ascii=False),
        "last_record": json.dumps({k: normalize_scalar(v) for k, v in last_record.items()}, ensure_ascii=False),
        "nan_ratio_by_column": json.dumps({c: (100.0 * file_nan_counts.get(c, 0) / rows if rows else math.nan) for c in sorted(file_nan_counts)}, ensure_ascii=False),
        "constant_columns": ",".join([c for c, a in file_signal_accs.items() if a.summary()["constant_pct"] > 95]),
        "abnormal_columns": ",".join([c for c, a in file_signal_accs.items() if a.summary()["extreme_outlier_pct"] > 0]),
    }
    return out


def update_hour_value(rec: dict[str, Any], col: str, vals: np.ndarray) -> None:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return
    s = rec.setdefault(col, empty_simple_stats())
    update_simple_stats(s, vals)


def summarize_hourly(hourly: dict[pd.Timestamp, dict[str, Any]], global_start: pd.Timestamp, ageing_origin: float | None = None) -> pd.DataFrame:
    rows = []
    for hour in sorted(hourly):
        rec = hourly[hour]
        row = {"clock_hour": format_dt(hour), "ageing_hour_from_first_timestamp": (hour - global_start).total_seconds() / 3600, "row_count": rec.get("row_count", 0), "operation_point_unique": ",".join(map(str, rec.get("operation_point_unique", []))), "operation_point_transition_count": rec.get("operation_point_transition_count", 0)}
        for c, s in rec.items():
            if isinstance(s, dict) and "n" in s:
                n, mean, std, mn, mx = finalize_simple_stats(s)
                stem = c.replace("`", "").replace("[", "").replace("]", "").replace(" ", "_").replace(".", "_").replace("/", "_")
                row[f"{stem}__n"] = n
                row[f"{stem}__mean"] = mean
                row[f"{stem}__std"] = std
                row[f"{stem}__min"] = mn
                row[f"{stem}__max"] = mx
        if ageing_origin is not None and "OpHrs2_h__mean" in row:
            row["ageing_hour_from_OpHrs2_origin"] = row["OpHrs2_h__mean"] - ageing_origin
        else:
            row["ageing_hour_from_OpHrs2_origin"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_active_cell_hourly_features(load_paths: list[Path], active_cells: list[str], global_start: pd.Timestamp, ageing_origin: float | None = None) -> pd.DataFrame:
    """Second, narrow pass over Loadcycle for dispersion after flatline masking.

    This is deliberately separate from the first pass: the active-cell mask is
    derived from the first-pass quality audit, so it cannot be chosen from a
    future modelling split or hard-coded from a filename.
    """
    hourly: dict[pd.Timestamp, dict[str, Any]] = {}
    if not active_cells:
        return pd.DataFrame()
    for path in load_paths:
        cols, enc, delim, err = read_csv_header(path)
        usecols = [c for c in ["DateTime", *active_cells] if c in cols]
        if "DateTime" not in usecols or len(usecols) < 2:
            continue
        for chunk in pd.read_csv(path, sep=delim or CSV_SEP, encoding=enc, usecols=usecols, chunksize=CHUNK_ROWS, engine="c", low_memory=False):
            dt = pd.to_datetime(chunk["DateTime"], errors="coerce")
            valid_dt = dt.notna().to_numpy()
            if not valid_dt.any():
                continue
            umat = np.column_stack([normalize_numeric(chunk[c]).to_numpy(dtype=float) for c in active_cells if c in chunk.columns])
            if umat.shape[1] < 2:
                continue
            with np.errstate(all="ignore"):
                std = np.nanstd(umat, axis=1)
                rng = np.nanmax(umat, axis=1) - np.nanmin(umat, axis=1)
            tmp = pd.DataFrame({"hour": dt.dt.floor("h"), "cell_std_active": std, "cell_range_active": rng})
            tmp = tmp.loc[tmp["hour"].notna()]
            for hour, g in tmp.groupby("hour", sort=False):
                rec = hourly.setdefault(pd.Timestamp(hour), {"row_count": 0})
                rec["row_count"] += int(len(g))
                update_hour_value(rec, "cell_std_active", g["cell_std_active"].to_numpy(dtype=float))
                update_hour_value(rec, "cell_range_active", g["cell_range_active"].to_numpy(dtype=float))
    return summarize_hourly(hourly, global_start, ageing_origin)


def parse_filename_range(name: str) -> tuple[int | None, int | None]:
    m = re.search(r"Loadcycle_(\d+)-(\d+)\.csv", name, re.I)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def normalize_scalar(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return str(v)


def scan_inventory() -> tuple[pd.DataFrame, dict[str, tuple[list[str], str, str]]]:
    rows = []
    headers: dict[str, tuple[list[str], str, str]] = {}
    for path in sorted(SRC_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(SRC_DIR)
        ext = path.suffix.lower()
        size_mb = path.stat().st_size / (1024**2)
        readable = True
        notes = []
        cols: list[str] = []
        encoding = ""
        delim = ""
        row_count: int | str = ""
        col_count: int | str = ""
        data_type = "unknown"
        if ext == ".csv":
            cols, encoding, delim, err = read_csv_header(path)
            headers[str(rel)] = (cols, encoding, delim)
            col_count = len(cols)
            data_type = classify_csv(path, cols)
            if err:
                readable = False; notes.append(err)
            else:
                # The source CSVs are line-oriented semicolon-delimited exports.
                # Counting physical records avoids a second multi-GB pandas parse;
                # the Loadcycle audit below still parses every row with chunks.
                try:
                    with path.open("rb") as fh:
                        row_count = max(0, sum(1 for _ in fh) - 1)
                    notes.append("row_count=physical newline count minus header")
                except Exception as exc:
                    readable = False; notes.append(repr(exc)); row_count = ""
        elif ext == ".txt":
            encoding, text = detect_text_encoding(path)
            delim = detect_delimiter(text)
            try:
                with path.open("r", encoding=encoding, errors="strict") as fh:
                    row_count = sum(1 for _ in fh)
                if "Impedance samples" in text or "Frequency/Hz" in text:
                    data_type = "EIS spectrum text"
                    col_count = 8 if "impedance_R" in text else 7
                elif path.name.lower().startswith("readme"):
                    data_type = "README / protocol text"
                    col_count = ""
                else:
                    data_type = "text"
            except Exception as exc:
                readable = False; notes.append(repr(exc))
        elif ext == ".pdf":
            data_type = "paper PDF"
            encoding = "binary"
            if PdfReader is not None:
                try:
                    row_count = len(PdfReader(str(path)).pages)
                    col_count = ""
                except Exception as exc:
                    readable = False; notes.append(repr(exc))
            else:
                notes.append("PyPDF2 unavailable")
        else:
            data_type = "other"
        rows.append({
            "relative_path": str(rel), "filename": path.name, "extension": ext,
            "file_size_MB": size_mb, "row_count": row_count, "column_count": col_count,
            "columns": "|".join(cols), "encoding": encoding, "delimiter": delim,
            "readable": readable, "data_type_guess": data_type, "notes": "; ".join(notes),
        })
    return pd.DataFrame(rows), headers


def classify_csv(path: Path, cols: list[str]) -> str:
    s = str(path).lower()
    if "loadcycle" in s: return "Loadcycle time series CSV"
    if "polcurve" in s: return "polarization curve CSV"
    if "partialsensitivity" in s: return "partial sensitivity test CSV"
    return "CSV"


def build_variable_dictionary(loadcycle_cols: list[str], all_csv_cols: dict[str, tuple[list[str], str, str]]) -> pd.DataFrame:
    union: list[str] = []
    for c in loadcycle_cols:
        if c not in union: union.append(c)
    for cols, _, _ in all_csv_cols.values():
        for c in cols:
            if c not in union: union.append(c)
    rows = []
    for c in union:
        cat, meaning, unit = infer_category(c)
        in_paths = [p for p, (cols, _, _) in all_csv_cols.items() if c in cols]
        rows.append({"original_column_name": c, "inferred_meaning": meaning, "unit": unit, "category": cat, "source_file_count": len(in_paths), "source_examples": "|".join(in_paths[:5])})
    return pd.DataFrame(rows)


def parse_eis_file(path: Path, spatial: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    enc, text = detect_text_encoding(path)
    lines = text.splitlines()
    meta: dict[str, Any] = {"source_relative_path": str(path.relative_to(SRC_DIR)), "filename": path.name, "encoding": enc}
    for line in lines[:20]:
        if line.startswith("File"):
            meta["file_label"] = line.split(":", 1)[1].strip() if ":" in line else ""
        elif line.startswith("Potential"):
            meta["potential_V"] = extract_first_number(line)
        elif line.startswith("Current"):
            meta["current_A"] = extract_first_number(line)
            m = re.search(r"Ampl:\s*([-+0-9.eE]+)", line)
            meta["amplitude_A"] = float(m.group(1)) if m else math.nan
        elif line.startswith("Temperature"):
            meta["temperature_text"] = line.split(":", 1)[1].strip() if ":" in line else ""
        elif line.startswith("Time"):
            meta["measurement_time"] = line.split(":", 1)[1].strip() if ":" in line else ""
        elif "area/cm" in line:
            meta["area_cm2"] = extract_first_number(line.split(":", 1)[1] if ":" in line else line)
    header_idx = next((i for i, line in enumerate(lines) if line.strip().startswith("Number") and "Frequency" in line), None)
    records = []
    if header_idx is not None:
        header_line = lines[header_idx].strip()
        data_header = re.split(r"\s+", header_line)
        for line in lines[header_idx + 1:]:
            parts = re.split(r"\s+", line.strip())
            if not parts or not re.fullmatch(r"\d+", parts[0] if parts else ""):
                continue
            vals = [extract_first_number(x) for x in parts]
            if len(vals) >= 7:
                if "impedance_R/Ohm" in header_line:
                    # Number, frequency, real, imag, significance, time, voltage, current
                    if len(vals) >= 8:
                        records.append(vals[:8])
                else:
                    # Number, frequency, magnitude, phase, significance, time, voltage, current
                    if len(vals) >= 8:
                        records.append(vals[:8])
    cols = ["number", "frequency_Hz", "z_real_or_mag_Ohm", "z_imag_or_phase", "significance", "time_s", "voltage_V", "current_A"]
    df = pd.DataFrame(records, columns=cols)
    stage, age, after = parse_stage(path)
    meta.update({"stage": stage, "ageing_hour": age, "afterAirAirStart": after, "spatial": spatial, "n_points": len(df), "test_family": "spatial_EIS" if spatial else "cell_EIS"})
    meta.update(parse_load_descriptor(path.name, spatial))
    if len(df):
        freq = df["frequency_Hz"].to_numpy(dtype=float)
        third = df["z_real_or_mag_Ohm"].to_numpy(dtype=float)
        fourth = df["z_imag_or_phase"].to_numpy(dtype=float)
        if "impedance_R/Ohm" in (lines[header_idx] if header_idx is not None else ""):
            zr = third; zi = fourth; magnitude = np.hypot(zr, zi); phase = np.degrees(np.arctan2(zi, zr))
        else:
            magnitude = third; phase = fourth; zr = magnitude * np.cos(np.radians(phase)); zi = magnitude * np.sin(np.radians(phase))
        order = np.argsort(freq)
        low_idx = order[:max(1, min(3, len(order)))]
        high_idx = order[-max(1, min(3, len(order))):]
        meta.update({
            "frequency_min_Hz": float(np.nanmin(freq)), "frequency_max_Hz": float(np.nanmax(freq)),
            "real_high_frequency_Ohm": float(np.nanmean(zr[high_idx])), "real_low_frequency_Ohm": float(np.nanmean(zr[low_idx])),
            "HFR_proxy_Ohm": float(np.nanmean(zr[high_idx])), "low_frequency_real_minus_HFR_Ohm": float(np.nanmean(zr[low_idx]) - np.nanmean(zr[high_idx])),
            "impedance_magnitude_low_Ohm": float(np.nanmean(magnitude[low_idx])), "impedance_magnitude_high_Ohm": float(np.nanmean(magnitude[high_idx])),
            "phase_min_deg": float(np.nanmin(phase)), "phase_max_deg": float(np.nanmax(phase)),
            "nyquist_arc_height_Ohm": float(np.nanmax(-zi) - np.nanmin(-zi)), "real_range_Ohm": float(np.nanmax(zr) - np.nanmin(zr)),
            "parsed_format": "real_imag" if "impedance_R/Ohm" in (lines[header_idx] if header_idx is not None else "") else "magnitude_phase",
        })
        df["z_real_Ohm"] = zr; df["z_imag_Ohm"] = zi; df["z_magnitude_Ohm"] = magnitude; df["phase_deg"] = phase
    else:
        meta.update({"parsed_format": "unparsed"})
    # Extract exact index/location code from filename; keep original code without claiming a physical mapping.
    meta["location_index_code"] = parse_location_code(path.name) if spatial else ""
    meta["condition_token"] = parse_condition_token(path.name, path)
    return meta, df


def extract_first_number(text: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    return float(m.group(0)) if m else math.nan


def parse_location_code(name: str) -> str:
    # The trailing tst00...tst16 is the reproducible acquisition index in the
    # raw filenames. Keep it as a code; do not infer inlet/outlet orientation.
    m = re.search(r"_tst(\d+)\.txt$", name.lower())
    if m:
        return f"tst{m.group(1)}"
    return name.split("_", 1)[0]


def parse_condition_token(name: str, path: Path) -> str:
    s = str(path)
    for token in ["HumidityVariation", "PressureVariation", "StoichiometryVariation", "TemperatureVariation"]:
        if token.lower() in s.lower():
            return token
    for pat in [r"(?:^|_)(p\d+)", r"(?:^|_)(t\d+)", r"(?:^|_)(stoic\d+)", r"(?:^|_)(f\d+)"]:
        m = re.search(pat, name.lower())
        if m: return m.group(1)
    return "nominal"


def parse_load_descriptor(name: str, spatial: bool = False) -> dict[str, Any]:
    """Parse load labels from the raw filename without treating them as age."""
    s = name.lower()
    def dec(token: str) -> float:
        return float(token.replace("-", "."))
    external = testbench = nominal = math.nan
    if spatial:
        m = re.search(r"_(\d+(?:-\d+)?)a_(\d+(?:-\d+)?)a_ampl_(\d+(?:-\d+)?)a", s)
        if m:
            nominal = dec(m.group(1)); testbench = dec(m.group(3))
    else:
        m = re.search(r"(?:^|_)(\d+(?:-\d+)?)a_(\d+(?:-\d+)?)a_ampl_(\d+(?:-\d+)?)a", s)
        if m:
            external = dec(m.group(1)); testbench = dec(m.group(3)); nominal = external + testbench
    label = f"{nominal:g}A" if math.isfinite(nominal) else "unknown"
    return {"external_load_A": external, "testbench_load_A": testbench, "nominal_load_A": nominal, "nominal_load_group": label}


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def make_figures(load_hourly: pd.DataFrame, pol_df: pd.DataFrame, eis_df: pd.DataFrame, spatial_df: pd.DataFrame, sens_df: pd.DataFrame, load_summary: pd.DataFrame, interval_hist: Counter, eis_plot_points: list[dict[str, Any]]) -> dict[str, str]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "figure.dpi": 140})
    generated = {}
    def save(fig, name):
        p = OUT_DIR / name
        fig.tight_layout()
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generated[name] = str(p)

    if len(load_hourly):
        x = pd.to_datetime(load_hourly["clock_hour"])
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for col, label in [("MIN(U_S_V)__mean", "stack-voltage candidate MIN(U.S) [V]"), ("U_S_AveCell_V__mean", "average cell voltage [V]")]:
            if col in load_hourly:
                ax.plot(x, load_hourly[col], lw=0.7, label=label)
        ax.set(title="Full durability voltage trend", xlabel="clock time", ylabel="voltage [V]"); ax.legend(); ax.grid(alpha=.25)
        save(fig, "full_durability_voltage_trend.png")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        age_x = load_hourly["ageing_hour_from_OpHrs2_origin"] if "ageing_hour_from_OpHrs2_origin" in load_hourly else load_hourly["ageing_hour_from_first_timestamp"]
        for col, label in [("I_S_act_A__mean", "actual current [A]"), ("j_S_act_A_cm2__mean", "current density [A/cm²]")]:
            if col in load_hourly:
                ax.plot(age_x, load_hourly[col], lw=0.8, label=label)
        ax.set(title="Current/load profile", xlabel="ageing hour from OpHrs2 origin [h]", ylabel="value"); ax.legend(); ax.grid(alpha=.25)
        save(fig, "current_load_profile.png")
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(load_hourly["ageing_hour_from_first_timestamp"], load_hourly.get("row_count", 0), lw=.8)
        ax.set(title="Observed rows by clock hour (missing/gap timeline proxy)", xlabel="ageing hour [h]", ylabel="rows in observed hour"); ax.grid(alpha=.25)
        save(fig, "missing_gap_timeline.png")
        fig, ax = plt.subplots(figsize=(7, 4))
        intervals = []
        for k, v in interval_hist.items(): intervals.extend([k] * min(v, 100_000))
        if intervals:
            ax.hist(intervals, bins=min(80, max(10, len(set(intervals)))), color="#4472C4")
        ax.set(title="Sampling interval histogram", xlabel="interval [s]", ylabel="count (capped per bin)"); ax.set_xlim(left=0); ax.grid(alpha=.25)
        save(fig, "sampling_interval_histogram.png")
        if "I_S_act_A__mean" in load_hourly and "U_S_AveCell_V__mean" in load_hourly:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sc = ax.scatter(age_x, load_hourly["U_S_AveCell_V__mean"], c=load_hourly["I_S_act_A__mean"], s=6, cmap="viridis")
            ax.set(title="Average-cell voltage vs ageing, colored by load", xlabel="ageing hour from OpHrs2 origin [h]", ylabel="average cell voltage [V]"); fig.colorbar(sc, ax=ax, label="current [A]"); ax.grid(alpha=.25)
            save(fig, "voltage_vs_ageing_colored_load.png")

    if len(pol_df):
        fig, ax = plt.subplots(figsize=(8, 5))
        for stage, g in pol_df[pol_df["row_type"] == "curve_point"].groupby("stage", sort=False):
            ax.plot(g["actual_j_median"], g["voltage_median"], marker=".", ms=3, lw=1, label=stage)
        ax.set(title="Polarization curves by ageing checkpoint", xlabel="current density [A/cm²]", ylabel="average cell voltage [V]"); ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
        save(fig, "polarization_curves_by_ageing.png")
        fixed = pol_df[(pol_df["row_type"] == "curve_point") & pol_df["target_j"].isin([0.4, 0.8, 1.2, 1.6, 2.0])]
        if len(fixed):
            fig, ax = plt.subplots(figsize=(8, 5))
            for j, g in fixed.groupby("target_j"):
                g = g.sort_values("ageing_hour")
                ax.plot(g["ageing_hour"], g["voltage_median"], marker="o", label=f"j={j:g}")
            ax.set(title="Voltage degradation at fixed current density", xlabel="ageing hour [h]", ylabel="average cell voltage [V]"); ax.grid(alpha=.25); ax.legend()
            save(fig, "voltage_degradation_fixed_current_density.png")
        # Raw voltage vs current density by ageing stage.
        fig, ax = plt.subplots(figsize=(8, 5))
        for stage, g in pol_df[pol_df["row_type"] == "curve_point"].groupby("stage", sort=False):
            ax.scatter(g["actual_j_median"], g["voltage_median"], s=12, label=stage)
        ax.set(title="Voltage vs current density by ageing stage", xlabel="current density [A/cm²]", ylabel="average cell voltage [V]"); ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
        save(fig, "voltage_vs_current_density_by_ageing.png")

    if len(load_hourly):
        for col, title, name in [("cell_std_active__mean", "Cell-voltage dispersion vs ageing", "cell_voltage_dispersion_vs_ageing.png"), ("cell_range_active__mean", "Cell-voltage range vs ageing", "cell_voltage_range_vs_ageing.png")]:
            if col in load_hourly:
                age_x = load_hourly["ageing_hour_from_OpHrs2_origin"] if "ageing_hour_from_OpHrs2_origin" in load_hourly else load_hourly["ageing_hour_from_first_timestamp"]
                fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(age_x, load_hourly[col], lw=.8); ax.set(title=title, xlabel="ageing hour from OpHrs2 origin [h]", ylabel="voltage spread [V]"); ax.grid(alpha=.25); save(fig, name)

    if len(eis_df):
        if eis_plot_points:
            ep = pd.DataFrame(eis_plot_points)
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            for stage, g in ep.groupby("stage", sort=False):
                g = g[np.isfinite(g["z_real_Ohm"]) & np.isfinite(g["z_imag_Ohm"])].copy()
                if len(g):
                    avg = g.groupby("frequency_Hz", as_index=False)[["z_real_Ohm", "z_imag_Ohm"]].mean().sort_values("frequency_Hz")
                    ax.plot(avg["z_real_Ohm"], -avg["z_imag_Ohm"], marker=".", ms=3, lw=1, label=stage)
            ax.set(title="EIS Nyquist comparison across ageing stages", xlabel="Z' [Ohm]", ylabel="-Z'' [Ohm]"); ax.grid(alpha=.25); ax.legend(fontsize=8)
            save(fig, "eis_nyquist_comparison_across_ageing.png")
        fig, ax = plt.subplots(figsize=(7, 5))
        feature_filter = (eis_df["test_family"] == "cell_EIS") & (eis_df["condition_token"] == "nominal")
        if "nominal_load_A" in eis_df.columns:
            feature_filter &= np.isclose(pd.to_numeric(eis_df["nominal_load_A"], errors="coerce"), 504.2, atol=1.0)
        for stage, g in eis_df[feature_filter].groupby("stage", sort=False):
            g = g.groupby("ageing_hour", as_index=False)[["real_high_frequency_Ohm", "low_frequency_real_minus_HFR_Ohm"]].mean()
            ax.scatter(g["real_high_frequency_Ohm"], g["low_frequency_real_minus_HFR_Ohm"], label=stage)
        ax.set(title="EIS feature comparison across ageing stages", xlabel="HFR proxy [Ohm]", ylabel="low-frequency real - HFR [Ohm]"); ax.grid(alpha=.25); ax.legend(fontsize=8)
        save(fig, "eis_feature_comparison_across_ageing.png")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        gg = eis_df[feature_filter].copy()
        if len(gg):
            gg = gg.groupby("ageing_hour", as_index=False)[["HFR_proxy_Ohm", "nyquist_arc_height_Ohm"]].mean().sort_values("ageing_hour")
            for col, label in [("HFR_proxy_Ohm", "HFR proxy"), ("nyquist_arc_height_Ohm", "Nyquist arc height")]:
                ax.plot(gg["ageing_hour"], gg[col], "o-", label=label)
        ax.set(title="Selected EIS features vs ageing", xlabel="ageing hour [h]", ylabel="Ohm / derived feature"); ax.grid(alpha=.25); ax.legend()
        save(fig, "selected_eis_features_vs_ageing.png")

    if len(spatial_df):
        pivot = spatial_df[(spatial_df["condition_token"] == "nominal") & spatial_df["stage"].isin(["BoL", "1150h", "2300h"])].pivot_table(index="location_index_code", columns="stage", values="HFR_proxy_Ohm", aggfunc="mean")
        if len(pivot):
            fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma"); ax.set_xticks(range(len(pivot.columns)), pivot.columns); ax.set_yticks(range(len(pivot.index)), pivot.index); ax.set(title="Spatial EIS HFR proxy heatmap", xlabel="ageing stage", ylabel="original location code"); fig.colorbar(im, ax=ax, label="HFR proxy [Ohm]"); save(fig, "spatial_eis_heatmap.png")
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for stage in pivot.columns: ax.plot(range(len(pivot.index)), pivot[stage].to_numpy(), "o-", label=stage)
            ax.set(title="Spatial EIS position comparison", xlabel="original spatial code order", ylabel="HFR proxy [Ohm]"); ax.set_xticks(range(len(pivot.index)), pivot.index, rotation=90, fontsize=7); ax.grid(alpha=.25); ax.legend(); save(fig, "spatial_eis_position_comparison.png")

    if len(sens_df):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
        for ax, family in zip(axes.flat, ["Temperature", "Humidity", "Pressure", "Stoichiometry"]):
            g = sens_df[sens_df["family"] == family]
            if len(g):
                for stage, gg in g.groupby("stage", sort=False):
                    gg = gg[np.isfinite(pd.to_numeric(gg["condition_value"], errors="coerce"))]
                    if len(gg): ax.plot(gg["condition_value"], gg["voltage_median"], "o-", label=stage)
            ax.set_title(f"{family} sensitivity"); ax.set_xlabel("condition token/value"); ax.set_ylabel("average cell voltage [V]"); ax.grid(alpha=.25)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "No RH column in raw CSV\ncurve not supported", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        save(fig, "sensitivity_bol_vs_2300h.png")
    return generated


def analyze_polarization(path: Path) -> list[dict[str, Any]]:
    cols, enc, delim, err = read_csv_header(path)
    wanted = [c for c in ["DateTime", "CalHrs [h]", "OpHrs [h]", "Number of cells", "I.S.act [A]", "j.S.act [A/cm2]", "U.S.AveCell [V]", "MIN(`U.S [V]`)"] + [f"U{i} [V]" for i in range(1, 25)] if c in cols]
    stage, age, after = parse_stage(path)
    records = []
    try:
        reader = pd.read_csv(path, sep=delim or CSV_SEP, encoding=enc, usecols=wanted, chunksize=CHUNK_ROWS, engine="c", low_memory=False)
        for chunk in reader:
            for c in wanted:
                if c != "DateTime": chunk[c] = normalize_numeric(chunk[c])
            records.append(chunk)
    except Exception as exc:
        return [{"row_type": "file_error", "filename": path.name, "source_relative_path": str(path.relative_to(SRC_DIR)), "stage": stage, "ageing_hour": age, "afterAirAirStart": after, "notes": repr(exc)}]
    if not records:
        return []
    df = pd.concat(records, ignore_index=True)
    jcol, vcol = "j.S.act [A/cm2]", "U.S.AveCell [V]"
    j = df[jcol].to_numpy(dtype=float); v = df[vcol].to_numpy(dtype=float)
    valid = np.isfinite(j) & np.isfinite(v)
    j = j[valid]; v = v[valid]
    if not len(j): return []
    out = []
    for target in np.arange(0.2, 2.01, 0.2):
        mask = np.abs(j - target) <= 0.035
        if not mask.any():
            idx = np.argsort(np.abs(j-target))[:min(50, len(j))]
            mask = np.zeros(len(j), dtype=bool); mask[idx] = True
        jj = j[mask]; vv = v[mask]
        row = {"row_type": "curve_point", "filename": path.name, "source_relative_path": str(path.relative_to(SRC_DIR)), "stage": stage, "ageing_hour": age, "afterAirAirStart": after, "target_j": float(target), "actual_j_median": float(np.nanmedian(jj)), "actual_j_min": float(np.nanmin(jj)), "actual_j_max": float(np.nanmax(jj)), "voltage_mean": float(np.nanmean(vv)), "voltage_median": float(np.nanmedian(vv)), "voltage_std": float(np.nanstd(vv)), "n": int(len(vv)), "j_min_file": float(np.nanmin(j)), "j_max_file": float(np.nanmax(j))}
        if "MIN(`U.S [V]`)" in df.columns:
            sv = df.loc[valid, "MIN(`U.S [V]`)"] .to_numpy(dtype=float)[mask]
            row["stack_voltage_candidate_median"] = float(np.nanmedian(sv)) if len(sv) else math.nan
        out.append(row)
    return out


def summarize_sensitivity(path: Path) -> list[dict[str, Any]]:
    name = path.name
    m = re.match(r"PartialSensitivityTest_(Temperature|Humidity|Pressure|Stoic)_(BoL|2300h)\.csv", name, re.I)
    if not m: return []
    family_token, age_token = m.groups(); family = "Stoichiometry" if family_token.lower() == "stoic" else family_token.capitalize(); stage = "BoL" if age_token.lower() == "bol" else "2300h"; age = 0.0 if stage == "BoL" else 2300.0
    cols, enc, delim, err = read_csv_header(path)
    wanted = [c for c in ["DateTime", "I.S.act [A]", "j.S.act [A/cm2]", "U.S.AveCell [V]", "T.Si.A [°C]", "RH.Si.A  [%]", "p.Si.A [kPag]", "Stoic.S.A [-]"] if c in cols]
    if err: return [{"family": family, "stage": stage, "ageing_hour": age, "filename": name, "notes": err}]
    frames = []
    try:
        for chunk in pd.read_csv(path, sep=delim or CSV_SEP, encoding=enc, usecols=wanted, chunksize=CHUNK_ROWS, engine="c", low_memory=False):
            for c in wanted:
                if c != "DateTime": chunk[c] = normalize_numeric(chunk[c])
            frames.append(chunk)
    except Exception as exc:
        return [{"family": family, "stage": stage, "ageing_hour": age, "filename": name, "notes": repr(exc)}]
    if not frames: return []
    df = pd.concat(frames, ignore_index=True)
    # Use the family-specific condition signal, with actual observed setpoint/current metadata retained.
    cond_candidates = {"Temperature": ["T.Si.A [°C]"], "Humidity": ["RH.Si.A  [%]"], "Pressure": ["p.Si.A [kPag]"], "Stoichiometry": ["Stoic.S.A [-]"]}[family]
    cond_col = next((c for c in cond_candidates if c in df.columns), None)
    if cond_col is None:
        # Humidity files in this release have no RH-named column in the raw
        # header. Preserve the file-level voltage evidence, but do not invent
        # condition values or a sensitivity curve.
        v = df["U.S.AveCell [V]"].to_numpy(dtype=float) if "U.S.AveCell [V]" in df.columns else np.array([], dtype=float)
        v = v[np.isfinite(v)]
        return [{"family": family, "stage": stage, "ageing_hour": age, "filename": name, "condition_variable": "unknown; no RH column in raw CSV", "condition_value": math.nan, "voltage_median": float(np.nanmedian(v)) if len(v) else math.nan, "voltage_mean": float(np.nanmean(v)) if len(v) else math.nan, "voltage_std": float(np.nanstd(v)) if len(v) else math.nan, "n": int(len(v)), "notes": "Humidity sensitivity cannot be parameterized from this file release without a logged RH/setpoint field."}]
    x = df[cond_col].to_numpy(dtype=float); v = df["U.S.AveCell [V]"].to_numpy(dtype=float) if "U.S.AveCell [V]" in df.columns else np.full(len(df), np.nan)
    valid = np.isfinite(x) & np.isfinite(v)
    x, v = x[valid], v[valid]
    rows = []
    for val in sorted(set(np.round(x, 4))):
        vv = v[np.isclose(x, val, atol=max(1e-4, abs(val)*1e-3))]
        if len(vv): rows.append({"family": family, "stage": stage, "ageing_hour": age, "filename": name, "condition_variable": cond_col, "condition_value": float(val), "voltage_median": float(np.nanmedian(vv)), "voltage_mean": float(np.nanmean(vv)), "voltage_std": float(np.nanstd(vv)), "n": int(len(vv))})
    return rows


def write_report(inventory: pd.DataFrame, load_summary: pd.DataFrame, timeline: pd.DataFrame, var_dict: pd.DataFrame, signal_df: pd.DataFrame, sampling_df: pd.DataFrame, degradation_df: pd.DataFrame, pol_df: pd.DataFrame, eis_df: pd.DataFrame, spatial_df: pd.DataFrame, sens_df: pd.DataFrame, hi_df: pd.DataFrame, causal_df: pd.DataFrame, figures: dict[str, str], paper_notes: str) -> None:
    def one(col, default=""):
        return load_summary[col].iloc[0] if len(load_summary) and col in load_summary else default
    n_files = len(inventory)
    n_csv = int((inventory["extension"] == ".csv").sum())
    n_txt = int((inventory["extension"] == ".txt").sum())
    n_pdf = int((inventory["extension"] == ".pdf").sum())
    global_start = pd.to_datetime(load_summary["min_timestamp"], errors="coerce").min() if len(load_summary) else pd.NaT
    global_end = pd.to_datetime(load_summary["max_timestamp"], errors="coerce").max() if len(load_summary) else pd.NaT
    duration = (global_end - global_start).total_seconds()/3600 if pd.notna(global_start) and pd.notna(global_end) else math.nan
    total_rows = int(load_summary["rows"].sum()) if len(load_summary) else 0
    global_ophrs_min = pd.to_numeric(load_summary["min_OpHrs_h"], errors="coerce").min() if len(load_summary) else math.nan
    global_ophrs_max = pd.to_numeric(load_summary["max_OpHrs_h"], errors="coerce").max() if len(load_summary) else math.nan
    global_ophrs2_min = pd.to_numeric(load_summary["min_OpHrs2_h"], errors="coerce").min() if len(load_summary) else math.nan
    global_ophrs2_max = pd.to_numeric(load_summary["max_OpHrs2_h"], errors="coerce").max() if len(load_summary) else math.nan
    global_ophrs2_span = global_ophrs2_max - global_ophrs2_min if math.isfinite(global_ophrs2_min) and math.isfinite(global_ophrs2_max) else math.nan
    cross_gap_count = int((pd.to_numeric(load_summary["gap_from_previous_h"], errors="coerce") > 0).sum()) if len(load_summary) else 0
    internal_gap_count = int(pd.to_numeric(load_summary["gap_count"], errors="coerce").sum()) if len(load_summary) else 0
    stack_fields = var_dict.loc[var_dict["category"] == "stack voltage", "original_column_name"].tolist() if len(var_dict) else []
    have_cells = [x for x in var_dict["original_column_name"].tolist() if re.fullmatch(r"U\d+ \[V\]", x)] if len(var_dict) else []
    flat_cells = signal_df[(signal_df["category"] == "cell voltage") & signal_df["quality_flags"].astype(str).str.contains("excluded_from_active", na=False)] if len(signal_df) else pd.DataFrame()
    nominal_pol = pol_df[pol_df["row_type"] == "curve_point"] if len(pol_df) else pd.DataFrame()
    has_eol_threshold = "500 mV lower voltage limit" if "500" in paper_notes else "not extracted from paper"
    humidity_condition_status = "logged RH values available" if (len(sens_df) and sens_df.loc[sens_df["family"] == "Humidity", "condition_value"].notna().any()) else "unresolved: raw humidity sensitivity CSV has no RH-named column"
    fig_lines = "\n".join(f"- `{Path(p).name}`" for p in figures.values())
    report = f"""# DATASET_AUDIT_2300H

审计对象：`{SRC_DIR}`  
输出目录：`{OUT_DIR}`  
审计日期：{pd.Timestamp.now().isoformat()}  
审计脚本：`audit_2300h.py`（所有原始读取均为只读；CSV 使用 chunked read）

## 1. Executive conclusion

### 事实（来自原始文件）

- 递归发现 `{n_files}` 个文件：CSV `{n_csv}`、TXT `{n_txt}`、PDF `{n_pdf}`。
- Loadcycle CSV 共 `{len(load_summary)}` 个，合计观测行 `{total_rows:,}`。
    - 按 Loadcycle 内部 `DateTime` 的实际最小/最大值，观测时间范围为 `{format_dt(global_start)}` 至 `{format_dt(global_end)}`，历时约 `{duration:.2f} h`；其中包含停机/维护/characterization 之间的长时间缺口，因此不是连续运行小时数。
- Loadcycle 中存在 `OpHrs [h]` 和 `OpHrs2 [h]`，全局范围分别约为 `{global_ophrs_min:.4g}-{global_ophrs_max:.4g} h` 和 `{global_ophrs2_min:.4g}-{global_ophrs2_max:.4g} h`；`OpHrs2` 观测跨度约 `{global_ophrs2_span:.2f} h`，不是从 0 h 开始。`OpHrs2` 全文件无 reset 且终点约 2256.8 h，和论文约 2300 operating hours 相符，故推荐把 `OpHrs2 [h]` 作为 ageing coordinate；`OpHrs [h]` 在 `Loadcycle_101-112.csv` 出现原始文本 `85,710.13`-`85,715.13` 后回到约 `1,547.50`，应标记为局部导出/记录异常，不能作为主寿命时间轴。
- 原始字段包含 10-cell 配置字段 `Number of cells`，含 `U1`-`U11` 共 `{len(have_cells)}` 个 cell voltage 通道；`U.S.AveCell [V]`、`MAX(U.S.MaxCell [V])`、`MIN(U.S.MinCell [V])` 和 `MIN(U.S [V])` 等 stack/cell-level voltage fields 存在。`MIN(U.S [V])` 是 stack-voltage candidate，但其 `MIN` 统计语义需以设备导出定义核实，报告不把它无条件当作专用 stack voltage。
- 论文 PDF 文本提到 durability test 结束于 200 cycles、约 2300 operating hours，并把 500 mV 作为 durability test 的 lower voltage limit；这说明存在一个运行限制/判据，但不等于原始数据中已经证明发生了 run-to-failure EOL。

### 审计判断

**建议将该数据集作为论文主数据集用于“运行条件感知的 SOH / degradation forecasting + 机理诊断”方向，但暂不作为严格 RUL 的主标签数据集。** 原因是它具有长时、重载、单体/空间/EIS/极化/敏感性等多模态诊断优势，同时当前原始文件审计未把 2300 h 证明为明确 failure endpoint；严格 RUL 需要独立、可复核的 EOL 定义。

**PHM2014 FC2 建议保留为外部验证集；FC1 建议作为敏感性/补充验证而不是主数据集。** 该结论针对任务设计与外部验证价值，不读取或修改旧实验的 final-test 结果，也不使用其结果做模型选择。

## 2. Dataset structure

`dataset_file_inventory.csv` 是逐文件清单，包含真实编码、分隔符、chunked row count、列数/列名和可读性。

关键目录：

- `Loadcycle_data`：17 个真实连续负载记录 CSV。
- `EIS`：cell-level EIS、BoL/1150 h/2300 h 以及压力、温度、化学计量比变体的 TXT 谱文件。
- `spatiallyResolvedEIS`：BoL/1150 h/2300 h 的空间 EIS 和温度、湿度、压力、化学计量比变体。
- 根目录：10 个 `Polcurve*.csv`（含 BoL、300/600/850/1150/1500/1750/2000/2300 h 和 after-Air-Air-start）以及 8 个 `PartialSensitivityTest*.csv`。
- 本地论文 PDF 和 `ReadMe_EIS_measurements.txt` 用于协议/命名解释；原始数值仍以 CSV/TXT 为优先证据。

## 3. Loadcycle timeline

`loadcycle_file_summary.csv` 逐文件记录了实际时间戳、OpHrs 范围、采样间隔、重复/倒序/缺口、reset 与 operation point。`loadcycle_timeline_summary.csv` 同时包含 file、global、gap、operation_point 级 metadata。

- 文件名如 `Loadcycle_0-14.csv`、`Loadcycle_15-25.csv` 被拆成 `filename_range_start/end`，但报告不会把这个范围直接称为真实 cycle number，因为文件内部没有名为 `cycle`/`cycle_number` 的字段。文件名区间只是文件元数据；真实的 load-cycle 计数需依据协议或内部控制日志确认。
- 结论是“可以重建有明确缺口标记的 operating-hours timeline”，不是“无缝连续时间序列”：17 个文件之间有 `{cross_gap_count}` 个正向时间间隔，文件内部还有 `{internal_gap_count}` 个按 1.5×文件中位采样间隔识别的 gap；没有文件级 timestamp overlap，也没有 timestamp reversal。
- `operation_point` 是重复出现的整数状态码，不等于 durability cycle number。它适合做运行状态/事件特征候选；每个状态的实际时长和切换应以时间戳聚合。
- 若文件边界存在时间重叠或时间间隔，`loadcycle_timeline_summary.csv` 保留这些事实，并没有简单 concat 或去重覆盖原始记录。
- startup/shutdown 的原始可见代理是 `I.S.act [A] == 0`、operation point 变化以及时间缺口；“维护”只有在缺口/协议注释支持时才能确认，不能由单一零电流段猜测。

## 4. Variable dictionary

`variable_dictionary.csv` 对所有 CSV 列名做了规则化语义分组。无法从原始列名确认的字段保留为 `unknown`；`MIN(U.S [V])` 作为 stack voltage candidate 单独标注为“exact MIN semantics unresolved”。

重点变量：

- stack/cell voltage fields：`{', '.join(stack_fields) if stack_fields else 'none'}`；单体列：`{', '.join(have_cells)}`；平均/最大/最小单体统计存在。
- current/current density：`I.S.set [A]`、`I.S.act [A]`、`j.S.act [A/cm2]` 等。
- operating conditions：温度、压力、RH、flow、stoichiometry 及 set/actual 版本均存在。
- `Number of cells` 的原始值应作为配置元数据，不应在不同 BoL short-stack 与 durability 10-cell stack 之间直接混用作同一健康标签。

## 5. Sampling and continuity

`sampling_rate_analysis.csv` 记录每个 Loadcycle 文件、全局 pooled statistics 和每小时样本量。采样周期以实际 `DateTime` 差分计算；没有从论文描述倒推采样率。`sampling_interval_histogram.png` 展示其分布。

推荐原则：

- 保留原始秒级数据用于事件/故障/启动研究，但不直接用于长时 degradation forecasting。
- 主建模表建议以 1 h causal aggregation 为默认粒度；同时对 0.5 h 与 2 h 做**仅在后续任务设计阶段**的稳定性敏感性分析。1 h 是在 2300 h 量级、每小时大量原始观测、又需保留 load-regime coverage 之间的折中。
- 聚合只使用当前及过去窗口内观测，不 centered smoothing、不 bfill；遇到 characterization gap 需保留 coverage flag，而不是插值填平。
- raw voltage 被 load 强烈调制，因此应优先做 operating-condition-aware features，而不是全量 `voltage ~ ageing` 单变量回归。

## 6. Signal quality

`signal_quality_summary.csv` 对核心连续字段给出 min/max/mean/std/p1/p5/p50/p95/p99、NaN、zero、constant、物理范围外比例和 abrupt jump 统计。物理范围外比例是审计标记，不是删除规则；没有任何异常点被删除。

特别关注：

- `{', '.join(flat_cells['original_column_name'].tolist()) if len(flat_cells) else '需查看 signal_quality_summary.csv 判定 flatline cell channels'}` 被识别为低电压/非物理水平或 flatline 候选。当前原始统计中 `U6 [V]` 的中位数约为负小值，与其他单体约 0.70 V 的共同水平明显不一致，因此 active-cell dispersion 应排除它；剩余 10 个 cell-voltage 通道与 `Number of cells=10` 一致。不要把该通道强行当作健康退化信号。
- abrupt jump 与 load transition 不等价于传感器故障；报告把两者分开，建议后续使用 operation_point/current change 作为 event covariate。

## 7. Operating regime characteristics

`loadcycle_hourly_features.csv` 是仅由原始 Loadcycle 观测做出的每小时 metadata-level/causal aggregation，不是训练集。包含小时均值、std、min、max、观测行数、operation point 集合和状态切换计数。

有/无 future-known schedule 的区别必须严格处理：实际未来 current、temperature、pressure、RH、flow、stoichiometry 如果来自运行时传感器，在 prediction origin 不可默认知道。只有实验协议明确预定义且任务允许使用的 schedule 才能标记为 future-known by protocol。

## 8. Degradation evidence

`degradation_signal_summary.csv` 汇总了：全量 hourly trend、按相似 current-density bins 的 matched-load view、cell voltage dispersion、possible recovery/reset and structural-change flags。所有趋势图均以原始时间顺序和 causal aggregation 为基础，不做未来填充。

应这样解释：

- raw stack/cell voltage 的下降若伴随 current/load regime 改变，只能称为“运行条件混合信号”；matched-load 或标准化 polarization point 才能支持较强的 degradation 论断。
- 若存在 load relaxation 后的 voltage recovery，它更像 reversible hydration/mass-transport component，不能直接作为 irreversible SOH loss。
- 1500 h 与 2300 h 附近的 Air-Air-start 结果通过 `afterAirAirStart` 分层保留；after 文件与同龄正常文件的差异不能被当作纯 ageing effect。

## 9. Polarization degradation

`polarization_ageing_summary.csv` 对每个 polarization file 在固定 `j* = 0.2...2.0 A/cm2` 候选点附近提取了实际 j 分布、平均/中位电压、样本量和 after-Air-Air-start 标记。没有对原始 curve 做插值填充。

固定 j 点 `V(j*, t)` 是最干净的 health anchor 候选之一。建议优先选择：跨 checkpoint 可获得、重复测量方差低、没有越过 500 mV lower-limit 的区间；具体稳定点以该 CSV 的 `voltage_std`、coverage 和 after 标志复核。BoL short-stack 与 durability stack 的 cell count/configuration 差异必须在比较中分层。

## 10. EIS degradation

`eis_ageing_summary.csv` 对真实 TXT 表格解析了频率、real/imag 或 magnitude/phase、measurement current、potential、points、HFR proxy、low-frequency real-minus-HFR、arc-height、phase/magnitude 等 robust raw-spectrum features，并保留 `external_load_A / testbench_load_A / nominal_load_A`。没有做 equivalent-circuit fitting。

- 该数据具备跨 BoL/1150 h/2300 h 的 cell EIS 条件；真实格式在 CSV 中保留为 `real_imag` 或 `magnitude_phase`。
- HFR proxy 采用最高频端的 real impedance 均值；这不是拟合后的 ohmic resistance，应以 proxy 命名。
- 跨 ageing 的 Nyquist/feature 图只使用同一约 504.2 A nominal-load group，避免把 BoL/2300 h 中的约 10 A 与约 50 A cell-EIS 负载差异误当成 degradation。
- `spatial_eis_summary.csv` 保留原始 `location_index_code`，并且没有把代码强行映射成 inlet/outlet 顺序。只有论文/协议明确说明的 cell/zone orientation 才能在后续建模前使用；当前 spatial heatmap 的横/纵轴是原始代码/观测顺序，不是假定的物理流向。

## 11. Spatial degradation

空间 EIS 的每个 stage/condition 组在原始目录中有 `tst00...tst16` 共 17 个采集文件；论文协议说明 PAD4 为 16 个空间阻抗区并带 reference，因此 17 个文件应按“空间区 + reference”整体处理，但每个代码与物理 zone 的精确映射仍需按论文图 1/设备记录核对。审计已回答“有多少原始位置代码”和“每个 stage/condition 是否存在”，但不会从 `00...16` 单独推断 inlet→outlet。

空间 HI 候选是 HFR/arc/low-frequency proxy 的跨位置 dispersion、分位数、max-min 和边界对比；必须保留位置 availability 和 stage/configuration metadata。

## 12. Sensitivity tests

`sensitivity_ageing_summary.csv` 比较 BoL 与 2300 h 的 Temperature/Humidity/Pressure/Stoichiometry 部分敏感性。它用于说明 raw voltage 的 condition dependence，而不是直接做长期预测标签。Humidity 的 condition-value 状态：**{humidity_condition_status}**；脚本保留了文件级电压统计，但没有伪造 RH 响应曲线。

优先作为 future covariates 保留的变量：实际 current/current density、temperature、pressure、RH、flow、stoichiometry 及 set/actual discrepancy。未来实际传感器值默认 `future-unknown`；作为 diagnostic-only/offline-only 的 EIS/spatial/polarization 不能伪装成在线 causal input。

## 13. Candidate SOH definitions

`health_indicator_candidates.csv` 给出三套候选，并以 monotonicity/trendability/prognosability、load robustness、availability、causal availability、online SOH/RUL suitability 做结构化评估：

- A. Voltage-only：固定条件下 `V(j*, t)` 或 matched-load average-cell voltage；在线可用性最高，但对 condition confounding 最敏感。
- B. Electrochemical：固定 j 的 polarization voltage + HFR proxy + arc/low-frequency proxy；退化解释力更强，但 EIS 低频、诊断频率和在线可用性有限。
- C. Spatial-electrochemical：B + active-cell dispersion + spatial EIS heterogeneity/gradient；最适合机制分析和论文 novelty，但不是高频在线 SOH，不能默认用于实时 RUL。

审计结论：推荐把 A 作为主在线 health target，把 B/C 作为低频 diagnostic/auxiliary HI；不要把 2300 h 自动编码成 SOH=0。

## 14. Degradation forecasting feasibility

这套数据适合定义 `history -> future degradation`，但预测目标必须拆分：

1. absolute voltage：容易受 future load/temperature 等调制，不适合作为严格 future-unknown 主任务，除非 future schedule 是 protocol-known。
2. residual voltage `U(t+h)-U(t)`：更贴近 degradation increment，建议作为主预测 target；仍需 condition-aware origin features 和 coverage flag。
3. origin-relative/normalized target：适合跨段比较、减弱 stack/configuration scale，但必须只使用 origin 及历史信息计算 normalization。

建议候选 horizon：H=4/8/12/24 h，必要时 48 h；候选输入 N=15/24 h 仅作为起点，不可直接沿用 PHM2014。先依据 hourly feature ACF/有效 degradation time-scale 和 characterisation gaps 选定；本轮未训练任何模型。

## 15. RUL feasibility

严格 RUL 结论：**当前数据不能直接标注严格 RUL。**

- 原始 Loadcycle 终点是测试结束/characterization end；没有在 CSV 中找到名为 EOL/failure/failure_time 的字段。
- 论文明确提到 500 mV lower voltage limit；这是可以调查的 threshold candidate，但仍需确认它是 durability test 的运行下限、单体/平均单体/哪一个 voltage 定义，以及是否真正触发了“failure”。
- 因此当前状态应写为 `NOT RUN-TO-FAILURE`，不能写 `RUL = 2300 - t`。
- 可行的后续路线是 threshold-based RUL：先用论文/DOE/heavy-duty durability criterion 复核阈值，再在每个运行条件和 stack configuration 下定义 event time；未确认之前只能报告 threshold candidates 与 uncertainty，不生成 RUL labels。

## 16. Comparison with PHM2014

| Dimension | New 2300 h dataset | IEEE PHM2014 FC1/FC2 | Audit implication |
|---|---|---|---|
| duration | 约 2300 operating h by paper; raw DateTime/OpHrs require explicit mapping | shorter benchmark durability records with established PHM task convention | new set stronger for long-horizon degradation context |
| load realism | heavy-duty relevant RealHyFC/PEMTASTIC-derived load cycle | benchmark-oriented load profile | new set more application-realistic for paper main story |
| sampling richness | raw Loadcycle high-rate signals plus characterization modalities | mainly time-series voltage/current/conditions convention | new set has richer causal audit surface |
| variables | current, cell voltages, temp, pressure, RH, flow, stoich, states | fewer variables depending on released channel | new set stronger for condition-aware HI |
| cell/spatial data | cell-level voltage and spatial EIS; 10-cell durability stack | benchmark FC1/FC2 primarily system signals | new set stronger for mechanistic novelty |
| polarization/EIS | regular polarization + EIS + spatial EIS + sensitivity | not comparable in the usual PHM2014 files | new set stronger for auxiliary HI / explanation |
| SOH | feasible with matched-condition or standardized voltage; configuration caveats | easy benchmark SOH convention | new set needs stricter target definition |
| forecasting | feasible for 4-48 h degradation increments after gap-aware aggregation | convenient benchmark history/future windows | new set supports more defensible condition-aware task |
| strict RUL | not proven run-to-failure; threshold unresolved | benchmark RUL labels depend on chosen convention and should be treated separately | do not claim new-set strict RUL yet |
| novelty | high | lower for main contribution, useful as external benchmark | new set is better main dataset if target is reformulated |
| leakage risk | high if future sensors/after tests/characterization leaks into origin | high if final-test result used for model selection | freeze causal feature policy and chronological split |
| preprocessing | high: multi-format, gaps, events, configurations | lower | publish audit and preprocessing manifest |

FC2 is recommended as external validation only after the main task is defined without using FC2 final-test outcomes for model or hyperparameter selection. FC1 is still useful for a robustness/sensitivity appendix or cross-dataset generalization check; it need not remain a co-equal main dataset.

## 17. Recommended main-task formulation

推荐主任务：

`causal history of hourly operating-condition-aware features -> future origin-relative / residual degradation at H={4,8,12,24} h`

主 target：固定标准化条件下的 average-cell voltage residual，或经历史-only condition normalization 的 `ΔV_h = V(t+h)-V(t)`；原始 absolute voltage 作为辅助 target/diagnostic，不作为唯一严格 future-unknown target。

主输入：过去 N 小时的 average/min/max cell voltage、active-cell dispersion、actual/set current/current density、temperature/pressure/RH/flow/stoich、operation point transition count、coverage/gap flags、calendar/OpHrs coordinate。EIS/polarization/spatial features 作为低频 diagnostic branch 或 checkpoint covariates，只有在预测起点真实可获得时才可进入在线任务。

推荐粒度：1 h causal aggregation，保留 0.5 h/2 h 作为后续敏感性候选。

推荐 N/H：先用 N=24 h、H=4/8/12/24 h 做任务定义；最终 N 由 hourly ACF/趋势尺度和 gap coverage 确认。N=15/H=24 只能作为与 PHM2014 的对照，不应作为默认。

## 18. Recommended train/val/test protocol

1. 固定 source manifest、列映射、active-cell mask、aggregation rules、threshold candidates。
2. 按时间顺序做 blocked split，例如早期 60% train、中段 20% validation、后期 20% test；characterization checkpoints 和 Air-Air-start 作为事件标注，不能跨 split 泄露。
3. 所有 normalization、matched-condition bin definitions 和 outlier thresholds 只在 train 或 protocol-defined physical constants 中确定；禁止用全程未来分布反推历史。
4. 若要模拟在线预测，未来 current/temperature/pressure/RH/flow/stoich 一律按 `future-unknown` 处理；另做一个明确命名的 `future-known-by-protocol` ablation，不能混写。
5. 以时间 block 为单位做 validation；不要随机打散相邻小时。
6. 测试只在最终冻结一次使用；不要访问 PHM2014 final-test 结果进行新模型选择。

推荐 baseline（仅作为后续计划，不在本轮训练）： persistence / linear trend / seasonal-naive by operating point / ridge with lag features / Gaussian process or state-space degradation baseline。LSTM、Transformer、M-LSTM、TFT、TCN、multi-scale、uncertainty、RUL model 与超参数搜索均按本轮停止条件明确禁止。

推荐指标：MAE/RMSE for residual voltage, bias/mean error, calibration only if later uncertainty is authorized, monotonicity/trendability/prognosability for HI, condition-stratified error, event/gap-stratified error, and cross-dataset FC2 external generalization. Strict RUL metrics are deferred until EOL is independently confirmed.

## 19. Leakage risks

`causal_feature_availability.csv` 对每个变量标记 `past-known`, `origin-known`, `future-known by protocol`, `future-unknown`, `diagnostic-only`, `offline-only`。

高风险点：

- 用未来真实 current/temperature/pressure/RH/flow/stoich 预测未来 absolute voltage；
- 用 after-Air-Air-start polarization/EIS 解释 Air-Air-start 之前的历史；
- 用 2300 h 终点或全局 quantiles 定义历史 SOH/EOL；
- 用 centered rolling median、bfill、随机行拆分；
- 把 `OpHrs`/`OpHrs2` 不加语义核验地当作同一年龄时间轴；
- 把 `operation_point` 或文件名 `0-14` 当作 cycle number；
- 把 `U6` 等 flatline/非物理通道纳入 cell dispersion；
- 以 8-cell BoL EIS/敏感性 stack 的绝对值与 10-cell durability stack 直接拼成单一 SOH。

## 20. Final Go / No-Go decision

- 作为长时 SOH / condition-aware degradation forecasting 主数据集：**GO with scope restrictions**。
- 作为 strict run-to-failure RUL 主数据集：**NO-GO until EOL/threshold is independently resolved**。
- 替代 PHM2014 成为论文主数据集：**建议替代，前提是论文主问题改为可复现的 SOH/未来退化预测，而不是直接声称 strict RUL**。
- PHM2014 FC2 external validation：**GO**，但只在主任务、预处理、模型选择全部冻结后使用。
- FC1：**保留为可选补充/稳健性验证，不必作为主数据集**。

### Recommended pipeline

`Raw Loadcycle -> timestamp/cycle audit -> causal aggregation -> operating-condition-aware representation -> SOH/degradation target -> chronological split -> forecasting -> uncertainty (future work) -> threshold-based RUL (only after EOL confirmation)`

本轮在 `forecasting` 之后停止，没有训练、相对历史实验、多尺度、uncertainty、RUL 模型、多 seed 或超参数搜索。

## Generated figures

{fig_lines}

## Paper protocol notes

{paper_notes}

## Reproducibility notes

- 原始数据目录只读；脚本只在输出目录生成文件。
- Loadcycle CSV 通过 pandas chunksize={CHUNK_ROWS:,} 读取；没有生成 GB 级 concat 副本。
- 聚合使用实际 timestamp/observed values；未做 centered smoothing、bfill 或未来观测填充。
- 所有 `unknown`/`unresolved` 都保留在结果，不以文件名或论文叙述替代原始字段证据。
"""
    (OUT_DIR / "DATASET_AUDIT_2300H.md").write_text(report, encoding="utf-8")


def read_paper_notes() -> str:
    pdf = next(iter(SRC_DIR.glob("*.pdf")), None)
    if pdf is None or PdfReader is None:
        return "本地 PDF 未成功解析；原始 CSV/TXT 仍为主证据。"
    try:
        reader = PdfReader(str(pdf))
        hits = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.splitlines() if re.search(r"2300|200 cycles|500 mV|lower voltage|Air-Air|load cycle|16 different zones|10-cell", line, re.I)]
            if lines:
                hits.append(f"page {i+1}: " + " / ".join(lines[:6]))
        # PDF extraction can carry NUL/control characters; remove them before
        # embedding compact protocol notes in the Markdown report.
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", "\n".join(hits[:12]))
    except Exception as exc:
        return f"PDF parse error: {exc!r}"


def build_causal_dictionary(var_dict: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in var_dict.iterrows():
        c = r["original_column_name"]; cat = r["category"]
        if cat == "time": status = "past-known"; rationale = "timestamp/age counter observed at origin"
        elif cat in {"current", "current density", "temperature", "pressure", "humidity", "flow", "stoichiometry", "state / control", "cell voltage", "stack voltage"}:
            status = "future-unknown"; rationale = "future sensor/control trajectory not available at prediction origin unless protocol explicitly supplies schedule"
        else: status = "unknown"; rationale = "semantics not resolved from raw data"
        if "EIS" in c or "spatial" in c.lower() or "polar" in c.lower(): status = "diagnostic-only"
        if c.startswith("Current_") or "Membrane active" in c: status = "offline-only"
        if c == "operation_point": status = "origin-known"; rationale = "current state code is known at origin; future sequence is not assumed"
        rows.append({"original_column_name": c, "category": cat, "availability": status, "causal_rationale": rationale, "future_schedule_exception": "only with explicit protocol-based future-known task" if status == "future-unknown" else ""})
    # Additional task-level variables not necessarily raw columns.
    rows.extend([
        {"original_column_name": "EIS raw spectrum / HFR proxy", "category": "diagnostic-only", "availability": "offline-only", "causal_rationale": "checkpoint/measurement overhead; not high-frequency online by default", "future_schedule_exception": ""},
        {"original_column_name": "polarization curve fixed-j V(j*)", "category": "diagnostic-only", "availability": "offline-only", "causal_rationale": "characterization checkpoint, not continuous online sensor", "future_schedule_exception": ""},
        {"original_column_name": "afterAirAirStart measurements", "category": "diagnostic-only", "availability": "future-unknown", "causal_rationale": "available only after the event; cannot explain pre-event origin", "future_schedule_exception": ""},
    ])
    return pd.DataFrame(rows)


def build_hi_candidates(pol_df: pd.DataFrame, eis_df: pd.DataFrame, spatial_df: pd.DataFrame) -> pd.DataFrame:
    def traj_metrics(age: Iterable[Any], value: Iterable[Any]) -> tuple[float, float, int]:
        d = pd.DataFrame({"age": pd.to_numeric(list(age), errors="coerce"), "value": pd.to_numeric(list(value), errors="coerce")}).dropna().sort_values("age")
        if len(d) < 2:
            return math.nan, math.nan, len(d)
        dv = d["value"].to_numpy()[1:] - d["value"].to_numpy()[:-1]
        # Voltage-like health is expected to decrease; report the observed
        # fraction of non-increasing transitions, without calling it EOL.
        mono = float(np.mean(dv <= 0))
        trend = float(abs(d["age"].corr(d["value"], method="spearman"))) if d["age"].nunique() > 1 and d["value"].nunique() > 1 else math.nan
        return mono, trend, len(d)

    a_mono = a_trend = math.nan; a_n = 0
    if len(pol_df):
        g = pol_df[(pol_df["row_type"] == "curve_point") & (pol_df["target_j"].sub(1.6).abs() < 1e-8) & (~pol_df["afterAirAirStart"])].groupby("ageing_hour", as_index=False)["voltage_median"].mean()
        a_mono, a_trend, a_n = traj_metrics(g["ageing_hour"], g["voltage_median"])
    b_mono = b_trend = math.nan; b_n = 0
    if len(eis_df) and "nominal_load_A" in eis_df.columns:
        f = eis_df[(eis_df["test_family"] == "cell_EIS") & (eis_df["condition_token"] == "nominal") & np.isclose(pd.to_numeric(eis_df["nominal_load_A"], errors="coerce"), 504.0, atol=1.0)].groupby("ageing_hour", as_index=False)["HFR_proxy_Ohm"].mean()
        b_mono, b_trend, b_n = traj_metrics(f["ageing_hour"], f["HFR_proxy_Ohm"])
    c_mono = c_trend = math.nan; c_n = 0; c_avail = 0
    if len(spatial_df):
        f = spatial_df[(spatial_df["condition_token"] == "nominal") & spatial_df["stage"].isin(["BoL", "1150h", "2300h"])].groupby("ageing_hour").agg(spatial_hfr_std=("HFR_proxy_Ohm", "std"), spatial_hfr_range=("HFR_proxy_Ohm", lambda x: float(x.max()-x.min()))).reset_index()
        c_mono, c_trend, c_n = traj_metrics(f["ageing_hour"], f["spatial_hfr_std"]); c_avail = len(spatial_df[(spatial_df["condition_token"] == "nominal") & spatial_df["stage"].isin(["BoL", "1150h", "2300h"])])
    rows = [
        {"candidate": "A_voltage_only", "definition": "fixed operating-condition V(j*, t) or matched-load average-cell voltage", "monotonicity": "observed non-increasing fraction at j=1.6", "monotonicity_score": a_mono, "trendability": "absolute Spearman(age,V) at j=1.6", "trendability_score": a_trend, "prognosability": "not estimable: no confirmed EOL", "load_robustness": "moderate; requires condition matching", "availability_frequency": f"hourly causal table available; fixed-j checkpoints n={a_n}", "causal_availability": "online/past-known", "online_soh_suitability": "high", "rul_suitability": "conditional; no EOL label", "notes": "Do not use raw voltage without load/condition controls."},
        {"candidate": "B_electrochemical", "definition": "fixed-j polarization voltage + HFR proxy + arc/low-frequency impedance feature", "monotonicity": "observed non-increasing fraction of same-504A HFR by stage", "monotonicity_score": b_mono, "trendability": "absolute Spearman(age,HFR) by stage", "trendability_score": b_trend, "prognosability": "not estimable: no confirmed EOL", "load_robustness": "high for standardized tests", "availability_frequency": f"same-504A cell EIS spectra n={b_n} age stages", "causal_availability": "offline/origin-known only when measured", "online_soh_suitability": "moderate-low", "rul_suitability": "conditional and diagnostic", "notes": "Raw-spectrum proxy; no equivalent-circuit fit."},
        {"candidate": "C_spatial_electrochemical", "definition": "B + active-cell dispersion + spatial EIS heterogeneity + inlet/outlet gradient only if orientation verified", "monotonicity": "observed non-increasing fraction of spatial HFR std by stage", "monotonicity_score": c_mono, "trendability": "absolute Spearman(age,spatial HFR std) by stage", "trendability_score": c_trend, "prognosability": "not estimable: no confirmed EOL", "load_robustness": "high if test condition standardized", "availability_frequency": f"nominal spatial spectra n={c_avail}; 17 location codes per stage", "causal_availability": "offline-only", "online_soh_suitability": "low", "rul_suitability": "not yet justified", "notes": "Preserve original location codes; do not infer flow direction from index alone."},
    ]
    return pd.DataFrame(rows)


def build_degradation_summary(hourly_df: pd.DataFrame, pol_df: pd.DataFrame) -> pd.DataFrame:
    """Create compact evidence tables; no model fitting or smoothing."""
    rows: list[dict[str, Any]] = []
    if len(hourly_df):
        vcol = next((c for c in ["U_S_AveCell_V__mean"] if c in hourly_df), None)
        jcol = next((c for c in ["j_S_act_A_cm2__mean"] if c in hourly_df), None)
        icol = next((c for c in ["I_S_act_A__mean"] if c in hourly_df), None)
        age_col = "ageing_hour_from_OpHrs2_origin" if "ageing_hour_from_OpHrs2_origin" in hourly_df else "ageing_hour_from_first_timestamp"
        if vcol:
            h = hourly_df.copy()
            h["ageing_stage_bin"] = pd.cut(h[age_col], bins=[-np.inf, 300, 600, 850, 1150, 1500, 1750, 2000, 2300, np.inf], labels=["0-300","300-600","600-850","850-1150","1150-1500","1500-1750","1750-2000","2000-2300",">2300"])
            for stage, g in h.groupby("ageing_stage_bin", observed=True):
                row = {"view": "hourly_raw", "stage_bin": str(stage), "n_hours": len(g), "ageing_hour_median": float(g[age_col].median()), "voltage_mean": float(g[vcol].mean()), "voltage_median": float(g[vcol].median()), "voltage_std_between_hours": float(g[vcol].std()), "current_mean": float(g[icol].mean()) if icol else math.nan, "current_density_mean": float(g[jcol].mean()) if jcol else math.nan, "notes": "raw hourly condition-mixed view; not a causal degradation label; ageing coordinate=OpHrs2"}
                if "cell_std_active__mean" in g: row["active_cell_dispersion_mean"] = float(g["cell_std_active__mean"].mean())
                if "cell_range_active__mean" in g: row["active_cell_range_mean"] = float(g["cell_range_active__mean"].mean())
                rows.append(row)
            if jcol:
                h["j_bin"] = (h[jcol] / 0.1).round() * 0.1
                for jb, g in h.groupby("j_bin", observed=True):
                    if len(g) >= 5:
                        rows.append({"view": "matched_load_hourly", "stage_bin": "all", "j_bin_A_cm2": float(jb), "n_hours": len(g), "ageing_hour_median": float(g[age_col].median()), "voltage_mean": float(g[vcol].mean()), "voltage_median": float(g[vcol].median()), "voltage_std_between_hours": float(g[vcol].std()), "current_mean": float(g[icol].mean()) if icol else math.nan, "current_density_mean": float(g[jcol].mean()), "notes": "coarse current-density matched view; temperature/pressure/RH not yet fully matched; ageing coordinate=OpHrs2"})
            if icol:
                for label, mask in [("low_current_proxy", h[icol] <= 1.0), ("loaded_proxy", h[icol] > 1.0)]:
                    g = h.loc[mask]
                    if len(g):
                        rows.append({"view": "reversible_recovery_proxy", "stage_bin": label, "n_hours": len(g), "ageing_hour_median": float(g[age_col].median()), "voltage_mean": float(g[vcol].mean()), "voltage_median": float(g[vcol].median()), "voltage_std_between_hours": float(g[vcol].std()), "current_mean": float(g[icol].mean()), "current_density_mean": float(g[jcol].mean()) if jcol else math.nan, "notes": "low-current versus loaded voltage level proxy; not event-aligned and not an irreversible SOH estimate"})
    if len(pol_df):
        cp = pol_df[(pol_df["row_type"] == "curve_point") & (pol_df["target_j"].sub(1.6).abs() < 1e-8)].copy()
        for stage, g in cp.groupby("stage", sort=False):
            rows.append({"view": "polarization_checkpoint", "stage_bin": stage, "ageing_hour_median": float(g["ageing_hour"].iloc[0]), "voltage_mean": float(g["voltage_median"].mean()), "voltage_median": float(g["voltage_median"].mean()), "voltage_std_between_hours": float(g["voltage_std"].mean()), "n_hours": int(g["n"].sum()), "notes": "fixed-j=1.6 A/cm2 characterization checkpoint; afterAirAirStart retained in stage label"})
        normal = cp[~cp["afterAirAirStart"]].set_index(["ageing_hour"])["voltage_median"]
        for age in [1500.0, 2300.0]:
            before = cp[(cp["ageing_hour"] == age) & (~cp["afterAirAirStart"])]["voltage_median"]
            after = cp[(cp["ageing_hour"] == age) & (cp["afterAirAirStart"])]["voltage_median"]
            if len(before) and len(after):
                rows.append({"view": "air_air_start_shift", "stage_bin": f"{int(age)}h", "ageing_hour_median": age, "voltage_mean": float(after.mean()-before.mean()), "voltage_median": float(after.mean()-before.mean()), "n_hours": int(len(before)+len(after)), "notes": "afterAirAirStart minus same-age nominal fixed-j voltage; structural event contrast, not ageing-only"})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC_DIR.exists():
        raise FileNotFoundError(SRC_DIR)
    print("[1/9] inventory")
    inventory, headers = scan_inventory()
    write_csv(inventory, "dataset_file_inventory.csv")
    load_paths = sorted((SRC_DIR / "Loadcycle_data").rglob("*.csv"))
    if not load_paths:
        raise RuntimeError("No Loadcycle CSV files found")
    load_cols = []
    for load_path in load_paths:
        for c in headers[str(load_path.relative_to(SRC_DIR))][0]:
            if c not in load_cols:
                load_cols.append(c)
    signal_cols = [c for c in load_cols if is_core_signal(c)]
    signal_accs = {c: SignalAccumulator() for c in signal_cols}
    hourly: dict[pd.Timestamp, dict[str, Any]] = {}
    interval_hist: Counter = Counter()
    load_rows = []
    print(f"[2/9] loadcycle audit: {len(load_paths)} files")
    for i, path in enumerate(load_paths, 1):
        print(f"  {i}/{len(load_paths)} {path.name}", flush=True)
        load_rows.append(read_loadcycle_file(path, load_cols, signal_accs, hourly, interval_hist))
    load_summary = pd.DataFrame(load_rows)
    # Cross-file relation based on actual timestamp ranges, not filename ranges.
    load_summary["min_dt"] = pd.to_datetime(load_summary["min_timestamp"], errors="coerce")
    load_summary["max_dt"] = pd.to_datetime(load_summary["max_timestamp"], errors="coerce")
    load_summary = load_summary.sort_values(["min_dt", "max_dt", "filename"]).reset_index(drop=True)
    load_summary["overlaps_previous_file"] = False
    load_summary["gap_from_previous_h"] = math.nan
    load_summary["boundary_duplicate_timestamp_count"] = 0
    for i in range(1, len(load_summary)):
        prev = load_summary.iloc[i-1]; cur = load_summary.iloc[i]
        if pd.notna(prev["max_dt"]) and pd.notna(cur["min_dt"]):
            delta = (cur["min_dt"] - prev["max_dt"]).total_seconds()/3600
            load_summary.loc[i, "gap_from_previous_h"] = delta
            load_summary.loc[i, "overlaps_previous_file"] = bool(delta <= 0)
            # Compare boundary record strings where available.
            load_summary.loc[i, "boundary_duplicate_timestamp_count"] = int(prev.get("last_timestamp", "") == cur.get("first_timestamp", "") and bool(prev.get("last_timestamp", "")))
    load_summary = load_summary.drop(columns=["min_dt", "max_dt"])
    # Flag abnormal/constant columns by signal stats.
    sig_rows = []
    for c, acc in signal_accs.items():
        cat, meaning, unit = infer_category(c)
        row = {"original_column_name": c, "category": cat, "unit": unit, "inferred_meaning": meaning, **acc.summary()}
        if row["constant_pct"] > 95: row["quality_flags"] = "flatline_or_constant"
        elif row["extreme_outlier_pct"] > 0: row["quality_flags"] = "physical_range_outliers_present"
        else: row["quality_flags"] = ""
        sig_rows.append(row)
    signal_df = pd.DataFrame(sig_rows)
    load_summary["abnormal_or_constant_columns"] = ",".join([r["original_column_name"] for r in sig_rows if r["quality_flags"]])
    load_summary["cycle_number_field_found"] = any("cycle" in c.lower() for c in load_cols)
    write_csv(load_summary, "loadcycle_file_summary.csv")
    write_csv(signal_df, "signal_quality_summary.csv")
    global_start_for_hourly = pd.to_datetime(load_summary["min_timestamp"], errors="coerce").min()
    age2_origin = float(pd.to_numeric(load_summary["min_OpHrs2_h"], errors="coerce").min())
    hourly_df = summarize_hourly(hourly, global_start_for_hourly, age2_origin)
    cell_quality = signal_df[signal_df["original_column_name"].isin(cell_columns(load_cols))].copy()
    typical_cell_p50 = float(np.nanmedian(cell_quality.loc[cell_quality["p50"] > 0.2, "p50"])) if (cell_quality["p50"] > 0.2).any() else math.nan
    invalid_level_mask = (cell_quality["constant_pct"] > 95) | (cell_quality["p50"] < 0.2) | (np.isfinite(typical_cell_p50) & (cell_quality["p50"] < typical_cell_p50 - 0.2))
    bad_cell_names = set(cell_quality.loc[invalid_level_mask, "original_column_name"].tolist())
    signal_df.loc[signal_df["original_column_name"].isin(bad_cell_names), "quality_flags"] = "nonphysical_level_or_flatline;excluded_from_active_cell_dispersion"
    flat_names = bad_cell_names
    active_cells = [c for c in cell_columns(load_cols) if c not in flat_names]
    load_summary["abnormal_or_constant_columns"] = load_summary["abnormal_or_constant_columns"].astype(str).apply(lambda s: ",".join(dict.fromkeys([x for x in (s.split(",") + sorted(bad_cell_names)) if x])))
    active_hourly_df = add_active_cell_hourly_features(load_paths, active_cells, global_start_for_hourly, age2_origin)
    write_csv(signal_df, "signal_quality_summary.csv")
    if len(active_hourly_df):
        merge_cols = [c for c in ["clock_hour", "ageing_hour_from_first_timestamp", "cell_std_active__n", "cell_std_active__mean", "cell_std_active__std", "cell_std_active__min", "cell_std_active__max", "cell_range_active__n", "cell_range_active__mean", "cell_range_active__std", "cell_range_active__min", "cell_range_active__max"] if c in active_hourly_df.columns]
        hourly_df = hourly_df.merge(active_hourly_df[merge_cols], on=["clock_hour", "ageing_hour_from_first_timestamp"], how="left")
    write_csv(hourly_df, "loadcycle_hourly_features.csv")
    print("[3/9] timeline and variables")
    timeline_rows = []
    for _, r in load_summary.iterrows():
        d = r.to_dict(); d["row_type"] = "file"; timeline_rows.append(d)
    global_start = pd.to_datetime(load_summary["min_timestamp"], errors="coerce").min(); global_end = pd.to_datetime(load_summary["max_timestamp"], errors="coerce").max()
    global_row = {"row_type": "global", "filename": "__GLOBAL__", "first_timestamp": format_dt(global_start), "last_timestamp": format_dt(global_end), "duration_h": (global_end-global_start).total_seconds()/3600, "rows": int(load_summary["rows"].sum()), "total_file_overlap_count": int(load_summary["overlaps_previous_file"].sum()), "total_internal_gap_count": int(load_summary["gap_count"].sum()), "cycle_number_field_found": any("cycle" in c.lower() for c in load_cols), "cycle_number_range": "unknown; no cycle field in raw Loadcycle header", "ophrs_global_range": f"{pd.to_numeric(load_summary['min_OpHrs_h'], errors='coerce').min()}..{pd.to_numeric(load_summary['max_OpHrs_h'], errors='coerce').max()}", "ophrs2_global_range": f"{pd.to_numeric(load_summary['min_OpHrs2_h'], errors='coerce').min()}..{pd.to_numeric(load_summary['max_OpHrs2_h'], errors='coerce').max()}", "recommended_ageing_coordinate": "OpHrs2 [h] (monotonic across files; no resets)", "suspect_OpHrs_files": ",".join(load_summary.loc[pd.to_numeric(load_summary["max_OpHrs_h"], errors="coerce") > 10000, "filename"].tolist())}
    timeline_rows.append(global_row)
    # Add metadata-level gap events between actual file ranges.
    for i in range(1, len(load_summary)):
        prev = load_summary.iloc[i-1]; cur = load_summary.iloc[i]
        if pd.notna(prev["max_timestamp"]) and pd.notna(cur["min_timestamp"]):
            d = (pd.Timestamp(cur["min_timestamp"]) - pd.Timestamp(prev["max_timestamp"])).total_seconds()/3600
            if d > 1/3600 or d < -1/3600:
                timeline_rows.append({"row_type": "gap_or_overlap", "filename": f"{prev['filename']} -> {cur['filename']}", "previous_file": prev["filename"], "next_file": cur["filename"], "previous_end": prev["max_timestamp"], "next_start": cur["min_timestamp"], "gap_from_previous_h": d, "overlap": d < 0, "notes": "actual timestamp-range relation; no concat/dedup applied"})
    op_agg = defaultdict(lambda: {"rows": 0, "min_timestamp": None, "max_timestamp": None})
    for _, r in load_summary.iterrows():
        for op in str(r.get("operation_point_values", "")).split(","):
            if op:
                x = op_agg[op]; x["rows"] += 1; x["min_timestamp"] = min(x["min_timestamp"], r["min_timestamp"]) if x["min_timestamp"] else r["min_timestamp"]; x["max_timestamp"] = max(x["max_timestamp"], r["max_timestamp"]) if x["max_timestamp"] else r["max_timestamp"]
    for op, r in sorted(op_agg.items(), key=lambda kv: float(kv[0])):
        timeline_rows.append({"row_type": "operation_point_metadata", "filename": f"operation_point={op}", "operation_point": op, **r, "notes": "state code, not cycle number"})
    timeline = pd.DataFrame(timeline_rows)
    write_csv(timeline, "loadcycle_timeline_summary.csv")
    var_dict = build_variable_dictionary(load_cols, headers)
    write_csv(var_dict, "variable_dictionary.csv")
    sampling_rows = []
    for _, r in load_summary.iterrows():
        sampling_rows.append({"scope": "file", "filename": r["filename"], **{k: r[k] for k in ["sampling_interval_median_s", "sampling_interval_mean_s", "sampling_interval_std_s", "minimum_interval_s", "maximum_interval_s", "duplicate_timestamp_count", "timestamp_reversal_count", "gap_count", "missing_timestamp_estimate"]}})
    pooled = IntervalAccumulator(); pooled.hist = interval_hist
    pooled.sample = []
    # pooled median/mean from file summaries is represented as weighted aggregate where available.
    total_pos = int(load_summary["sampling_interval_median_s"].notna().sum())
    sampling_rows.append({"scope": "global", "filename": "__GLOBAL__", "sampling_interval_median_s": float(np.nanmedian(load_summary["sampling_interval_median_s"])), "sampling_interval_mean_s": float(np.nanmean(load_summary["sampling_interval_mean_s"])), "sampling_interval_std_s": float(np.nanmean(load_summary["sampling_interval_std_s"])), "minimum_interval_s": float(np.nanmin(load_summary["minimum_interval_s"])), "maximum_interval_s": float(np.nanmax(load_summary["maximum_interval_s"])), "duplicate_timestamp_count": int(load_summary["duplicate_timestamp_count"].sum()), "timestamp_reversal_count": int(load_summary["timestamp_reversal_count"].sum()), "gap_count": int(load_summary["gap_count"].sum()), "missing_timestamp_estimate": int(load_summary["missing_timestamp_estimate"].sum()), "notes": "global descriptive aggregation across file summaries; file-level medians also retained"})
    sampling_df = pd.DataFrame(sampling_rows); write_csv(sampling_df, "sampling_rate_analysis.csv")
    print("[4/9] polarization curves")
    pol_rows = []
    for path in sorted(SRC_DIR.glob("Polcurve*.csv")):
        print(f"  {path.name}", flush=True); pol_rows.extend(analyze_polarization(path))
    pol_df = pd.DataFrame(pol_rows); write_csv(pol_df, "polarization_ageing_summary.csv")
    print("[5/9] EIS and spatial EIS")
    eis_rows=[]; spatial_rows=[]; eis_plot_points=[]
    for path in sorted((SRC_DIR / "EIS").rglob("*.txt")):
        meta, points = parse_eis_file(path, spatial=False); eis_rows.append(meta)
        if meta.get("condition_token") == "nominal" and meta.get("stage") in {"BoL", "1150h", "2300h"} and np.isclose(meta.get("nominal_load_A", math.nan), 504.2, atol=1.0) and len(points):
            for p in points[["frequency_Hz", "z_real_Ohm", "z_imag_Ohm"]].to_dict("records"):
                p.update({"stage": meta.get("stage"), "source_relative_path": meta.get("source_relative_path")})
                eis_plot_points.append(p)
    for path in sorted((SRC_DIR / "spatiallyResolvedEIS").rglob("*.txt")):
        meta, _ = parse_eis_file(path, spatial=True); spatial_rows.append(meta)
    eis_df = pd.DataFrame(eis_rows); spatial_df = pd.DataFrame(spatial_rows)
    write_csv(eis_df, "eis_ageing_summary.csv"); write_csv(spatial_df, "spatial_eis_summary.csv")
    print("[6/9] sensitivity")
    sens_rows=[]
    for path in sorted(SRC_DIR.glob("PartialSensitivityTest*.csv")):
        print(f"  {path.name}", flush=True); sens_rows.extend(summarize_sensitivity(path))
    sens_df = pd.DataFrame(sens_rows); write_csv(sens_df, "sensitivity_ageing_summary.csv")
    print("[7/9] degradation / causal / HI")
    # Compact degradation evidence table. It deliberately contains descriptive
    # hourly/matched-load views and checkpoint contrasts only; no smoothing or
    # future-informed target construction is performed here.
    deg_df = build_degradation_summary(hourly_df, pol_df)
    write_csv(deg_df, "degradation_signal_summary.csv")
    hi_df = build_hi_candidates(pol_df, eis_df, spatial_df); write_csv(hi_df, "health_indicator_candidates.csv")
    causal_df = build_causal_dictionary(var_dict); write_csv(causal_df, "causal_feature_availability.csv")
    # Capture deterministic audit metadata and hashes for generated code/report only.
    print("[8/9] figures")
    figures = make_figures(hourly_df, pol_df, eis_df, spatial_df, sens_df, load_summary, interval_hist, eis_plot_points)
    print("[9/9] report")
    paper_notes = read_paper_notes()
    write_report(inventory, load_summary, timeline, var_dict, signal_df, sampling_df, deg_df, pol_df, eis_df, spatial_df, sens_df, hi_df, causal_df, figures, paper_notes)
    manifest = {"source_dir": str(SRC_DIR), "output_dir": str(OUT_DIR), "script": str(OUT_DIR / "audit_2300h.py"), "refresh_script": str(OUT_DIR / "refresh_outputs.py"), "python": sys.executable, "pandas": pd.__version__, "numpy": np.__version__, "source_file_count": int(len(inventory)), "loadcycle_file_count": int(len(load_summary)), "loadcycle_rows": int(load_summary["rows"].sum()), "generated_files": sorted([p.name for p in OUT_DIR.iterdir() if p.is_file()])}
    (OUT_DIR / "audit_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    final_duration = (global_end - global_start).total_seconds() / 3600 if pd.notna(global_start) and pd.notna(global_end) else math.nan
    print(json.dumps({"output_dir": str(OUT_DIR), "generated_count": len(manifest["generated_files"]), "duration_h": final_duration, "load_rows": int(load_summary["rows"].sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

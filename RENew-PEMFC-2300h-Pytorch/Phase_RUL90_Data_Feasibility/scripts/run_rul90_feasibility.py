#!/usr/bin/env python
"""Two-stage, read-only RealHyFC 90%-threshold feasibility audit.

Stage ``development-freeze`` reads only development-range target values and
persists a reference-condition method.  Stage ``full-descriptive`` refuses to
run until that method exists, then performs a separately labelled full-data
descriptive audit.  It never trains or evaluates a forecasting model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # System Python on this workstation exposes PyPDF2.
    from PyPDF2 import PdfReader


SCRIPT = Path(__file__).resolve()
PHASE = SCRIPT.parent.parent
PROJECT = PHASE.parent
OUTPUTS = PHASE / "outputs"
FIGURES = PHASE / "figures"
METHOD_PATH = OUTPUTS / "METHOD_FREEZE.yaml"
RUN_PATH = PHASE / "RUN_AUDIT.json"
N1 = PROJECT / "Phase_N1_Data_Foundation"
N1_CONFIG = N1 / "configs" / "phase_n1_config.yaml"
N1_SPLIT = N1 / "outputs" / "split_manifest" / "split_manifest.csv"
CHUNK = 200_000
DEV_END_EXPECTED = 1819.997738
ACTIVE_CELLS = [f"U{i} [V]" for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
CELL_CANDIDATES = [f"U{i} [V]" for i in range(1, 12)]
J_CANDIDATES = [0.8, 1.0, 1.2, 1.4, 1.6]
J_TOL = 0.01
LOWER_LIMIT = 0.50


def log(msg: str) -> None:
    print(msg, flush=True)


def numeric(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.replace("\ufeff", "", regex=False).str.strip()
    both = text.str.contains(",", regex=False, na=False) & text.str.contains(".", regex=False, na=False)
    comma_only = text.str.contains(",", regex=False, na=False) & ~text.str.contains(".", regex=False, na=False)
    text = text.mask(both, text.str.replace(",", "", regex=False))
    text = text.mask(comma_only, text.str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")


def prepare_dirs() -> None:
    for directory in (PHASE, OUTPUTS, FIGURES, SCRIPT.parent):
        directory.mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUTS / name
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return path


def write_text(text: str, name: str) -> Path:
    path = PHASE / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NOT_ESTIMABLE"
    try:
        number = float(value)
    except (ValueError, TypeError):
        return str(value)
    return f"{number:.{digits}g}" if math.isfinite(number) else "NOT_ESTIMABLE"


def table(frame: pd.DataFrame, columns: Iterable[str] | None = None, limit: int = 40) -> str:
    use = frame.copy() if columns is None else frame[list(columns)].copy()
    if use.empty:
        return "_无记录。_"
    use = use.head(limit).fillna("").astype(str)
    return "| " + " | ".join(use.columns) + " |\n| " + " | ".join(["---"] * len(use.columns)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in use.itertuples(index=False, name=None))


def load_context() -> dict[str, Any]:
    if not N1_CONFIG.is_file() or not N1_SPLIT.is_file():
        raise FileNotFoundError("缺少正式 Phase_N1 配置或冻结划分清单，不能建立 Test firewall。")
    cfg = yaml.safe_load(N1_CONFIG.read_text(encoding="utf-8"))
    raw = Path(cfg["raw_data_root"])
    if not raw.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在：{raw}")
    split = pd.read_csv(N1_SPLIT)
    required = {"train", "validation", "test"}
    if set(split["split"]) != required:
        raise RuntimeError("冻结划分清单缺少 train/validation/test。")
    dev_end = float(split.loc[split["split"] == "validation", "ophrs2_end_h"].iloc[0])
    test_start = float(split.loc[split["split"] == "test", "ophrs2_start_h"].iloc[0])
    if not math.isclose(dev_end, test_start, abs_tol=1e-9) or not math.isclose(dev_end, DEV_END_EXPECTED, abs_tol=1e-6):
        raise RuntimeError(f"正式 Development 边界不符合预期：{dev_end}")
    return {"raw": raw, "cfg": cfg, "split": split, "dev_end": dev_end}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_header(path: Path) -> tuple[str, str, list[str]]:
    raw = path.read_bytes()[:128_000]
    encoding = "utf-8-sig"
    for candidate in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    line = next((x for x in text.splitlines() if x.strip()), "")
    delimiter = ";" if line.count(";") >= line.count(",") else ","
    columns = next(csv.reader([line], delimiter=delimiter), []) if line else []
    return encoding, delimiter, columns


def count_rows(path: Path, extension: str) -> int | float:
    if extension.lower() not in {".csv", ".txt"}:
        return math.nan
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return max(0, count - 1)


def classify_file(path: Path, columns: list[str]) -> str:
    lower = str(path).lower()
    header = "|".join(columns).lower()
    if "loadcycle" in lower:
        return "Loadcycle durability measurement"
    if "polcurve" in lower:
        return "polarization curve"
    if "spatiallyresolvedeis" in lower:
        return "spatially resolved EIS"
    if "\\eis\\" in lower or "frequency" in header:
        return "EIS or electrochemical spectrum"
    if "sensitivity" in lower:
        return "partial sensitivity test"
    if path.suffix.lower() == ".pdf":
        return "paper / experiment protocol"
    if path.suffix.lower() == ".txt":
        return "text / metadata / spectrum"
    return "unknown"


def inventory(raw: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((p for p in raw.rglob("*") if p.is_file()), key=lambda p: str(p).lower()):
        try:
            encoding, delimiter, columns = decode_header(path) if path.suffix.lower() in {".csv", ".txt"} else ("", "", [])
            rows.append({
                "relative_path": str(path.relative_to(raw)), "filename": path.name,
                "extension": path.suffix.lower(), "file_size_MB": round(path.stat().st_size / 1024**2, 6),
                "row_count": count_rows(path, path.suffix), "column_count": len(columns),
                "columns": " | ".join(columns[:120]), "encoding": encoding, "delimiter": delimiter,
                "readable": True, "data_type": classify_file(path, columns), "notes": "header inspected; raw values not changed",
            })
        except Exception as exc:  # inventory must retain failures as evidence
            rows.append({"relative_path": str(path.relative_to(raw)), "filename": path.name, "extension": path.suffix.lower(), "file_size_MB": round(path.stat().st_size / 1024**2, 6), "row_count": math.nan, "column_count": math.nan, "columns": "", "encoding": "", "delimiter": "", "readable": False, "data_type": "unreadable", "notes": repr(exc)})
    return pd.DataFrame(rows)


def stage_from_path(path: Path) -> tuple[str, float, bool]:
    low = path.stem.lower()
    after = "afterairair" in low
    if "bol" in low:
        return "BoL", 0.0, after
    match = re.search(r"(\d+)h", low)
    if not match:
        raise ValueError(f"无法从文件名解析极化阶段：{path.name}")
    age = float(match.group(1))
    return f"{int(age)}h" + ("_afterAirAirStart" if after else ""), age, after


def selection_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    normal = points[(~points["after_air_air_start"]) & (points["ageing_hour"] < DEV_END_EXPECTED)].copy()
    expected = normal["ageing_hour"].nunique()
    for j, group in normal.groupby("target_jstar_A_cm2"):
        group = group.sort_values("ageing_hour")
        v = group["active_voltage_median_V"].to_numpy(float)
        age = group["ageing_hour"].to_numpy(float)
        delta = np.diff(v)
        abs_move = float(np.abs(delta).sum())
        mono = 1.0 - float(np.clip(delta, 0, None).sum()) / abs_move if abs_move else math.nan
        rho = float(pd.Series(age).corr(pd.Series(v), method="spearman")) if len(group) >= 3 else math.nan
        before = points[(points["target_jstar_A_cm2"] == j) & (points["ageing_hour"] == 1500) & (~points["after_air_air_start"])]
        after = points[(points["target_jstar_A_cm2"] == j) & (points["ageing_hour"] == 1500) & points["after_air_air_start"]]
        air = float(after["active_voltage_median_V"].iloc[0] - before["active_voltage_median_V"].iloc[0]) if len(before) and len(after) else math.nan
        rows.append({"target_jstar_A_cm2": j, "development_checkpoint_count": len(group), "development_checkpoint_total": expected, "coverage": len(group) / expected if expected else math.nan, "voltage_start_V": v[0] if len(v) else math.nan, "voltage_end_V": v[-1] if len(v) else math.nan, "relative_loss": (v[0] - v[-1]) / v[0] if len(v) and v[0] else math.nan, "median_repeat_std_V": float(group["active_voltage_std_V"].median()), "spearman_age_voltage": rho, "monotonicity": mono, "non_increasing_fraction": float(np.mean(delta <= 0)) if len(delta) else math.nan, "air_air_shift_1500h_V": air, "minimum_margin_to_500mV_V": float(np.nanmin(v) - LOWER_LIMIT) if len(v) else math.nan})
    result = pd.DataFrame(rows)
    hard = (result["coverage"] >= 0.90) & (result["development_checkpoint_count"] >= 5) & (result["minimum_margin_to_500mV_V"] >= 0.05) & result["median_repeat_std_V"].notna()
    result["hard_gate_pass"] = hard
    passed = result[hard].copy()
    if not passed.empty:
        def norm(col: str, positive: bool = True) -> pd.Series:
            x = pd.to_numeric(passed[col], errors="coerce")
            lo, hi = x.min(), x.max()
            value = pd.Series(1.0, index=passed.index) if not math.isfinite(hi - lo) or abs(hi - lo) < 1e-15 else (x - lo) / (hi - lo)
            return value if positive else 1 - value
        passed["selection_score"] = pd.concat([norm("coverage"), norm("median_repeat_std_V", False), norm("monotonicity"), norm("spearman_age_voltage", True).abs(), norm("air_air_shift_1500h_V", False).abs(), norm("minimum_margin_to_500mV_V")], axis=1).mean(axis=1)
        result = result.merge(passed[["target_jstar_A_cm2", "selection_score"]], how="left", on="target_jstar_A_cm2")
    else:
        result["selection_score"] = math.nan
    return result.sort_values(["hard_gate_pass", "selection_score", "target_jstar_A_cm2"], ascending=[False, False, True]).reset_index(drop=True)


def analyze_polarization(paths: list[Path], jstars: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    config_cols = ["Number of cells", "Membrane active area [cm2]", "I.S.act [A]", "j.S.act [A/cm2]", "U.S.AveCell [V]", "Stoic.S.A [-]", "Stoic.S.C [-]", "p.So.A [kPag]", "p.So.C [kPag]", "T.Si.A [°C]", "T.Si.C [°C]", "T.Si.CL [°C]"]
    for index, path in enumerate(paths, 1):
        stage, age, after = stage_from_path(path)
        log(f"[polarization {index}/{len(paths)}] {path.name}")
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        required = [x for x in dict.fromkeys([*ACTIVE_CELLS, *config_cols]) if x in header]
        values: dict[str, list[np.ndarray]] = defaultdict(list)
        j_values: dict[float, list[np.ndarray]] = {j: [] for j in jstars}
        active_values: dict[float, list[np.ndarray]] = {j: [] for j in jstars}
        exported_values: dict[float, list[np.ndarray]] = {j: [] for j in jstars}
        diff_values: dict[float, list[np.ndarray]] = {j: [] for j in jstars}
        curve_j: list[np.ndarray] = []; curve_v: list[np.ndarray] = []
        for chunk in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=required, chunksize=CHUNK, low_memory=False):
            for c in required:
                chunk[c] = numeric(chunk[c])
            cells = chunk[[c for c in ACTIVE_CELLS if c in chunk]].to_numpy(float)
            active = np.nanmean(cells, axis=1)
            j = chunk["j.S.act [A/cm2]"].to_numpy(float)
            exported = chunk["U.S.AveCell [V]"].to_numpy(float)
            for c in config_cols:
                if c in chunk:
                    values[c].append(chunk[c].dropna().to_numpy(float)[::100])
            for target in jstars:
                m = np.isfinite(j) & np.isfinite(active) & (np.abs(j - target) <= J_TOL)
                if np.any(m):
                    j_values[target].append(j[m]); active_values[target].append(active[m]); exported_values[target].append(exported[m]); diff_values[target].append(np.abs(active[m] - exported[m]))
            mcurve = np.isfinite(j) & np.isfinite(active) & (j >= 0) & (j <= 1.8)
            if np.any(mcurve):
                curve_j.append(j[mcurve][::100]); curve_v.append(active[mcurve][::100])
        csummary: dict[str, Any] = {"filename": path.name, "stage": stage, "ageing_hour": age, "after_air_air_start": after}
        for c in config_cols:
            arr = np.concatenate(values[c]) if values[c] else np.array([])
            key = re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_")
            csummary[f"{key}_median"] = float(np.nanmedian(arr)) if len(arr) else math.nan
            csummary[f"{key}_min"] = float(np.nanmin(arr)) if len(arr) else math.nan
            csummary[f"{key}_max"] = float(np.nanmax(arr)) if len(arr) else math.nan
        config_rows.append(csummary)
        for target in jstars:
            a = np.concatenate(active_values[target]) if active_values[target] else np.array([])
            e = np.concatenate(exported_values[target]) if exported_values[target] else np.array([])
            j = np.concatenate(j_values[target]) if j_values[target] else np.array([])
            diff = np.concatenate(diff_values[target]) if diff_values[target] else np.array([])
            point_rows.append({"filename": path.name, "stage": stage, "ageing_hour": age, "after_air_air_start": after, "target_jstar_A_cm2": target, "actual_j_median_A_cm2": float(np.nanmedian(j)) if len(j) else math.nan, "active_voltage_median_V": float(np.nanmedian(a)) if len(a) else math.nan, "active_voltage_mean_V": float(np.nanmean(a)) if len(a) else math.nan, "active_voltage_std_V": float(np.nanstd(a)) if len(a) else math.nan, "exported_average_voltage_median_V": float(np.nanmedian(e)) if len(e) else math.nan, "active_vs_exported_mae_V": float(np.nanmean(diff)) if len(diff) else math.nan, "valid_sample_count": int(len(a)), "selection_scope": "development-only" if age < DEV_END_EXPECTED else "full-data-descriptive-only"})
        if curve_j:
            curve = pd.DataFrame({"j": np.concatenate(curve_j), "v": np.concatenate(curve_v)})
            curve["j_bin_A_cm2"] = (curve["j"] / 0.02).round() * 0.02
            grouped = curve.groupby("j_bin_A_cm2")["v"].agg(["median", "count"]).reset_index()
            for row in grouped.itertuples(index=False):
                curve_rows.append({"stage": stage, "ageing_hour": age, "after_air_air_start": after, "j_bin_A_cm2": row.j_bin_A_cm2, "active_voltage_median_V": row.median, "sample_count": int(row.count)})
    return pd.DataFrame(point_rows), pd.DataFrame(config_rows), pd.DataFrame(curve_rows)


class Stats:
    def __init__(self) -> None:
        self.n = self.nan = self.zero = self.abnormal = 0
        self.total = self.total_sq = 0.0; self.minimum = math.inf; self.maximum = -math.inf
        self.sample: list[np.ndarray] = []
    def add(self, values: np.ndarray, abnormal: np.ndarray | None = None) -> None:
        x = np.asarray(values, float); finite = x[np.isfinite(x)]
        self.n += len(x); self.nan += int(np.count_nonzero(~np.isfinite(x)))
        if len(finite):
            self.zero += int(np.count_nonzero(np.isclose(finite, 0.0)))
            self.total += float(finite.sum()); self.total_sq += float(np.square(finite).sum())
            self.minimum = min(self.minimum, float(finite.min())); self.maximum = max(self.maximum, float(finite.max()))
            self.sample.append(finite[::1000])
        if abnormal is not None: self.abnormal += int(np.count_nonzero(abnormal & np.isfinite(x)))
    def row(self, name: str, category: str, unit: str, meaning: str) -> dict[str, Any]:
        nvalid = self.n - self.nan; sample = np.concatenate(self.sample) if self.sample else np.array([])
        mean = self.total / nvalid if nvalid else math.nan
        variance = self.total_sq / nvalid - mean * mean if nvalid else math.nan
        return {"original_field": name, "category": category, "unit": unit, "physical_meaning": meaning, "raw_count": self.n, "missing_count": self.nan, "missing_percent": 100 * self.nan / self.n if self.n else math.nan, "zero_percent": 100 * self.zero / nvalid if nvalid else math.nan, "min": self.minimum if nvalid else math.nan, "max": self.maximum if nvalid else math.nan, "mean": mean, "std": math.sqrt(max(variance, 0)) if math.isfinite(variance) else math.nan, "p1": float(np.nanpercentile(sample, 1)) if len(sample) else math.nan, "p50": float(np.nanpercentile(sample, 50)) if len(sample) else math.nan, "p99": float(np.nanpercentile(sample, 99)) if len(sample) else math.nan, "abnormal_percent": 100 * self.abnormal / nvalid if nvalid else math.nan, "sampled_quantile_count": int(len(sample))}


def signal_dictionary() -> dict[str, tuple[str, str, str]]:
    return {
        "U.S.AveCell [V]": ("average cell voltage", "V", "exported average single-cell voltage"),
        "I.S.act [A]": ("current", "A", "actual stack current"), "j.S.act [A/cm2]": ("current density", "A/cm2", "actual stack current density"),
        "MIN(`U.S [V])": ("stack voltage candidate", "V", "MIN semantic unresolved; not accepted as dedicated stack voltage"),
        "T.Si.A [°C]": ("temperature", "°C", "anode inlet temperature"), "T.Si.C [°C]": ("temperature", "°C", "cathode inlet temperature"),
        "p.So.A [kPag]": ("pressure", "kPag", "anode outlet pressure"), "p.So.C [kPag]": ("pressure", "kPag", "cathode outlet pressure"),
        "RH.Si.A  [%]": ("humidity", "%", "anode inlet relative humidity"), "RH.Si.C [%]": ("humidity", "%", "cathode inlet relative humidity"),
        "F.Si.H.A [nl/min]": ("flow", "nl/min", "anode hydrogen flow"), "F.Si.Air.C [nl/min]": ("flow", "nl/min", "cathode air flow"),
    }


def loadcycle_metadata(paths: list[Path], dev_end: float | None = None) -> tuple[pd.DataFrame, dict[Path, int]]:
    rows = []; dev_rows: dict[Path, int] = {}
    for index, path in enumerate(paths, 1):
        log(f"[timeline {index}/{len(paths)}] {path.name}")
        n = 0; n_dev = 0; first_dt = last_dt = pd.NaT; first_age = last_age = math.nan; min_age = math.inf; max_age = -math.inf; dup = reverse_dt = reverse_age = internal_gap = 0; prev_dt = pd.NaT; prev_age = math.nan; gap_s = []
        for chunk in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=["DateTime", "OpHrs2 [h]"], chunksize=CHUNK, low_memory=False):
            dt = pd.to_datetime(chunk["DateTime"], errors="coerce"); age = numeric(chunk["OpHrs2 [h]"]).to_numpy(float); n += len(chunk); n_dev += int(np.count_nonzero(age < dev_end)) if dev_end is not None else 0
            valid_dt = dt.dropna(); valid_age = age[np.isfinite(age)]
            if len(valid_dt):
                first_dt = valid_dt.iloc[0] if pd.isna(first_dt) else first_dt; last_dt = valid_dt.iloc[-1]
            if len(valid_age):
                first_age = valid_age[0] if not math.isfinite(first_age) else first_age; last_age = valid_age[-1]; min_age = min(min_age, float(valid_age.min())); max_age = max(max_age, float(valid_age.max()))
            dt_arr = dt.to_numpy("datetime64[ns]"); delta = np.diff(dt_arr.astype("int64")) / 1e9
            if pd.notna(prev_dt) and len(dt):
                delta = np.r_[((dt.iloc[0] - prev_dt).total_seconds() if pd.notna(dt.iloc[0]) else np.nan), delta]
            good = np.isfinite(delta); dup += int(np.count_nonzero(delta[good] == 0)); reverse_dt += int(np.count_nonzero(delta[good] < 0)); internal_gap += int(np.count_nonzero(delta[good] > 5)); gap_s.extend(delta[good & (delta > 5)].tolist()); prev_dt = dt.iloc[-1] if len(dt) else prev_dt
            da = np.diff(age); reverse_age += int(np.count_nonzero(da[np.isfinite(da)] < 0)); prev_age = age[-1] if len(age) else prev_age
        rows.append({"filename": path.name, "rows": n, "first_timestamp": str(first_dt), "last_timestamp": str(last_dt), "first_OpHrs2_h": first_age, "last_OpHrs2_h": last_age, "min_OpHrs2_h": min_age if math.isfinite(min_age) else math.nan, "max_OpHrs2_h": max_age if math.isfinite(max_age) else math.nan, "duplicate_timestamp_count": dup, "timestamp_reversal_count": reverse_dt, "ophrs2_reversal_count": reverse_age, "internal_gap_count_dt_gt5s": internal_gap, "largest_internal_gap_s": max(gap_s) if gap_s else 0.0, "development_raw_rows": n_dev if dev_end is not None else math.nan})
        dev_rows[path] = n_dev
    frame = pd.DataFrame(rows).sort_values("first_timestamp").reset_index(drop=True)
    frame["gap_from_previous_file_h"] = math.nan; frame["overlap_previous_file"] = False
    for i in range(1, len(frame)):
        delta = (pd.Timestamp(frame.loc[i, "first_timestamp"]) - pd.Timestamp(frame.loc[i - 1, "last_timestamp"])).total_seconds() / 3600
        frame.loc[i, "gap_from_previous_file_h"] = delta; frame.loc[i, "overlap_previous_file"] = delta <= 0
    return frame, dev_rows


def audit_loadcycle(paths: list[Path], dev_rows: dict[Path, int] | None = None, development_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fields = ["DateTime", "OpHrs2 [h]", "U.S.AveCell [V]", "I.S.act [A]", "j.S.act [A/cm2]", *ACTIVE_CELLS, *[x for x in signal_dictionary() if x not in {"U.S.AveCell [V]", "I.S.act [A]", "j.S.act [A/cm2]"}]]
    stats = {c: Stats() for c in fields if c not in {"DateTime", "OpHrs2 [h]"}}
    bins: dict[int, dict[str, Any]] = {}; events: list[dict[str, Any]] = []; gaps: list[dict[str, Any]] = []; segments: list[dict[str, Any]] = []
    origin = math.nan; previous_dt = pd.NaT; previous_u = previous_i = math.nan
    for file_index, path in enumerate(paths, 1):
        nrows = dev_rows.get(path) if development_only and dev_rows is not None else None
        if development_only and nrows == 0: continue
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist(); use = [c for c in fields if c in header]
        log(f"[{'development' if development_only else 'full'} loadcycle {file_index}/{len(paths)}] {path.name}")
        first_seg = last_seg = None; file_rows = 0
        iterator = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=use, chunksize=CHUNK, nrows=nrows, low_memory=False)
        for chunk in iterator:
            file_rows += len(chunk)
            for c in use:
                if c != "DateTime": chunk[c] = numeric(chunk[c])
            dt = pd.to_datetime(chunk["DateTime"], errors="coerce"); age = chunk["OpHrs2 [h]"].to_numpy(float)
            good_age = age[np.isfinite(age)]
            if len(good_age) and not math.isfinite(origin): origin = float(good_age.min())
            if len(dt.dropna()):
                first_seg = dt.dropna().iloc[0] if first_seg is None else first_seg; last_seg = dt.dropna().iloc[-1]
            active = np.nanmean(chunk[[c for c in ACTIVE_CELLS if c in chunk]].to_numpy(float), axis=1)
            exported = chunk["U.S.AveCell [V]"].to_numpy(float)
            chunk["active_cell_mean_recomputed"] = active
            for c, collector in stats.items():
                if c in chunk:
                    x = chunk[c].to_numpy(float)
                elif c == "active_cell_mean_recomputed":
                    x = active
                else:
                    continue
                abnormal = ((x < -0.05) | (x > 1.2)) if "[V]" in c and c != "MIN(`U.S [V])" else None
                collector.add(x, abnormal)
            # Half-hour aggregation is descriptive; no interpolation or backfill.
            if math.isfinite(origin):
                index = np.floor((age - origin) / 0.5 + 1e-9).astype("int64"); valid = np.isfinite(age) & (index >= 0)
                current = chunk["I.S.act [A]"].to_numpy(float); j = chunk["j.S.act [A/cm2]"].to_numpy(float)
                for b in np.unique(index[valid]):
                    m = valid & (index == b); key = int(b); rec = bins.setdefault(key, {"n": 0, "u_sum": 0.0, "i_sum": 0.0, "j_sum": 0.0, "active_sum": 0.0, "disp_sum": 0.0, "u_min": math.inf, "u_max": -math.inf, "i_min": math.inf, "i_max": -math.inf, "dt_start": None, "dt_end": None})
                    uu = exported[m]; ii = current[m]; jj = j[m]; aa = active[m]; cells = chunk[[c for c in ACTIVE_CELLS if c in chunk]].to_numpy(float)[m]
                    finite_u = uu[np.isfinite(uu)]; finite_i = ii[np.isfinite(ii)]; finite_j = jj[np.isfinite(jj)]; finite_a = aa[np.isfinite(aa)]; disp = np.nanstd(cells, axis=1)
                    rec["n"] += int(m.sum()); rec["u_sum"] += float(np.nansum(uu)); rec["i_sum"] += float(np.nansum(ii)); rec["j_sum"] += float(np.nansum(jj)); rec["active_sum"] += float(np.nansum(aa)); rec["disp_sum"] += float(np.nansum(disp));
                    if len(finite_u): rec["u_min"] = min(rec["u_min"], float(finite_u.min())); rec["u_max"] = max(rec["u_max"], float(finite_u.max()))
                    if len(finite_i): rec["i_min"] = min(rec["i_min"], float(finite_i.min())); rec["i_max"] = max(rec["i_max"], float(finite_i.max()))
                    dtgood = dt[m].dropna()
                    if len(dtgood): rec["dt_start"] = dtgood.iloc[0] if rec["dt_start"] is None else rec["dt_start"]; rec["dt_end"] = dtgood.iloc[-1]
            # Preserve potential voltage discontinuities as flags, never deletions.
            u = exported; i = chunk["I.S.act [A]"].to_numpy(float); dv = np.diff(u); di = np.diff(i)
            if math.isfinite(previous_u): dv = np.r_[u[0] - previous_u, dv]; di = np.r_[i[0] - previous_i, di]
            m = np.isfinite(dv) & (np.abs(dv) >= 0.05)
            for k in np.flatnonzero(m)[:max(0, 500 - len(events))]:
                category = "停机/低负载工况" if abs(i[k]) <= 0.1 else "正常负载响应" if abs(di[k]) >= 5 else "原因不明（保留标记）"
                events.append({"filename": path.name, "timestamp": str(dt.iloc[k]), "age_h": age[k], "delta_average_cell_voltage_V": dv[k], "delta_current_A": di[k], "classification": category})
            dt_ns = dt.to_numpy("datetime64[ns]").astype("int64"); dts = np.diff(dt_ns) / 1e9
            if pd.notna(previous_dt) and len(dt): dts = np.r_[((dt.iloc[0] - previous_dt).total_seconds() if pd.notna(dt.iloc[0]) else np.nan), dts]
            for k in np.flatnonzero(np.isfinite(dts) & (dts > 5))[:100]: gaps.append({"gap_scope": "within_file", "from_file": path.name, "to_file": path.name, "gap_seconds": dts[k], "previous_timestamp": "", "next_timestamp": "", "classification": "timestamp interruption; maintenance cause unresolved"})
            previous_dt = dt.iloc[-1] if len(dt) else previous_dt; previous_u = u[-1] if len(u) else previous_u; previous_i = i[-1] if len(i) else previous_i
        if first_seg is not None: segments.append({"filename": path.name, "rows_read": file_rows, "first_timestamp": str(first_seg), "last_timestamp": str(last_seg), "scope": "development-only" if development_only else "full-data descriptive"})
    label = signal_dictionary(); label["active_cell_mean_recomputed"] = ("average cell voltage", "V", "mean of active U1-U5/U7-U11; U6 excluded")
    stat_rows = [collector.row(c, *label.get(c, ("cell voltage", "V", "individual voltage channel"))) for c, collector in stats.items()]
    stat_df = pd.DataFrame(stat_rows)
    if bins:
        out = []
        for b, rec in sorted(bins.items()):
            n = rec["n"]
            out.append({"bin_start_ophrs2_h": origin + b * .5, "bin_end_ophrs2_h": origin + (b + 1) * .5, "sample_count": n, "coverage_ratio_to_1800": n / 1800, "average_cell_voltage_mean_V": rec["u_sum"] / n, "average_cell_voltage_min_V": rec["u_min"], "average_cell_voltage_max_V": rec["u_max"], "current_mean_A": rec["i_sum"] / n, "current_min_A": rec["i_min"], "current_max_A": rec["i_max"], "current_density_mean_A_cm2": rec["j_sum"] / n, "active_cell_mean_V": rec["active_sum"] / n, "active_cell_dispersion_mean_V": rec["disp_sum"] / n, "datetime_start": str(rec["dt_start"]), "datetime_end": str(rec["dt_end"]), "scope": "development-only" if development_only else "full-data descriptive"})
        bin_df = pd.DataFrame(out)
    else: bin_df = pd.DataFrame()
    return stat_df, bin_df, pd.DataFrame(events), pd.DataFrame(gaps), pd.DataFrame(segments)


def eol_evidence(raw: Path) -> pd.DataFrame:
    terms = re.compile(r"(?i)(\bEOL\b|end of life|failure|lower voltage limit|500\s*mV|durability|degradation threshold|power loss|voltage loss)")
    rows = []
    for path in sorted([p for p in raw.rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]):
        try:
            pages = []
            if path.suffix.lower() == ".pdf":
                pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(PdfReader(path).pages)]
            else:
                pages = [(None, path.read_text(encoding="utf-8", errors="replace"))]
            for page, text in pages:
                for line_no, line in enumerate(text.splitlines(), 1):
                    if terms.search(line):
                        rows.append({"source_file": str(path.relative_to(raw)), "page": page if page is not None else "", "line": line_no, "evidence_text": line.strip()[:900], "interpretation": "evidence only; does not by itself prove official EOL"})
        except Exception as exc:
            rows.append({"source_file": str(path.relative_to(raw)), "page": "", "line": "", "evidence_text": repr(exc), "interpretation": "read failure"})
    return pd.DataFrame(rows)


def timexer_dev_scale(dev_end: float) -> pd.DataFrame:
    path = N1 / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
    if not path.is_file():
        return pd.DataFrame([{"status": "NOT_ESTIMABLE", "reason": "Phase_N1 30-min causal feature table missing"}])
    # The frozen project records this exact first-prefix contract. nrows avoids
    # loading any Test row before the method freeze.
    frame = pd.read_csv(path, nrows=3005)
    if "split" in frame and frame["split"].astype(str).eq("test").any():
        raise RuntimeError("Test firewall breach: a Test row entered Development TimeXer-scale audit")
    age = pd.to_numeric(frame["bin_center_ophrs2_h"], errors="coerce"); voltage = pd.to_numeric(frame["cell_voltage_mean"], errors="coerce")
    segment = frame.get("temporal_segment_id", pd.Series("unknown", index=frame.index)).astype(str)
    valid = np.isfinite(age) & np.isfinite(voltage) & (age < dev_end)
    work = pd.DataFrame({"age": age[valid].to_numpy(), "voltage": voltage[valid].to_numpy(), "segment": segment[valid].to_numpy()}).sort_values("age").reset_index(drop=True)
    rows = []
    for i in range(len(work) - 24):
        j = i + 24
        if work.loc[i, "segment"] != work.loc[j, "segment"] or not math.isclose(float(work.loc[j, "age"] - work.loc[i, "age"]), 12.0, abs_tol=0.11): continue
        rows.append({"origin_age_h": work.loc[i, "age"], "target_age_h": work.loc[j, "age"], "delta_voltage_12h_V": work.loc[j, "voltage"] - work.loc[i, "voltage"], "absolute_delta_voltage_12h_V": abs(work.loc[j, "voltage"] - work.loc[i, "voltage"]), "scope": "development-only; no model prediction"})
    return pd.DataFrame(rows)


def make_figures(meta: pd.DataFrame, half: pd.DataFrame, ref: pd.DataFrame, curves: pd.DataFrame, dev_end: float, timexer: pd.DataFrame) -> list[str]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.alpha": .25})
    names = []
    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout(); fig.savefig(FIGURES / name, dpi=180); plt.close(fig); names.append(name)
    if not meta.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8)); y = np.arange(len(meta)); start = pd.to_datetime(meta["first_timestamp"]); end = pd.to_datetime(meta["last_timestamp"])
        ax.hlines(y, start, end, linewidth=6, color="#2563EB"); ax.set_yticks(y, meta["filename"], fontsize=7); ax.set(title="Raw Loadcycle calendar-time coverage", xlabel="Calendar time", ylabel="source file"); save(fig, "time_coverage.png")
        fig, ax = plt.subplots(figsize=(10, 4.8)); ax.barh(y, pd.to_numeric(meta["max_OpHrs2_h"]) - pd.to_numeric(meta["min_OpHrs2_h"]), left=pd.to_numeric(meta["min_OpHrs2_h"]), color="#0F766E"); ax.set_yticks(y, meta["filename"], fontsize=7); ax.set(title="Observed operating-hour segments", xlabel="OpHrs2 operating hour", ylabel="source file"); ax.axvline(dev_end, color="#B91C1C", linestyle="--", label="frozen Development/Test boundary"); ax.legend(); save(fig, "time_segments.png")
    if not half.empty:
        fig, ax1 = plt.subplots(figsize=(11, 5.3)); x = half["bin_start_ophrs2_h"]; ax1.plot(x, half["average_cell_voltage_mean_V"], color="#2563EB", linewidth=.8, label="average cell voltage"); ax1.set(xlabel="Operating ageing (OpHrs2, h)", ylabel="Average cell voltage (V)"); ax2 = ax1.twinx(); ax2.plot(x, half["current_mean_A"], color="#EA580C", linewidth=.7, alpha=.7, label="current"); ax2.set_ylabel("Current (A)"); ax1.axvline(dev_end, color="#B91C1C", linestyle="--"); ax1.set_title("Full-data descriptive: 0.5 h voltage and current") ; save(fig, "current_voltage_vs_age.png")
        fig, ax = plt.subplots(figsize=(9, 5)); h = half.copy(); h["stage"] = pd.cut(h["bin_start_ophrs2_h"], [-np.inf, 600, 1200, dev_end, np.inf], labels=["early", "mid", "Development tail", "Test / full descriptive"]); data = [g["average_cell_voltage_mean_V"].dropna().to_numpy() for _, g in h.groupby("stage", observed=True)]; labs = [str(k) for k, _ in h.groupby("stage", observed=True)]; ax.boxplot(data, tick_labels=labs, showfliers=False); ax.set(title="0.5 h voltage distribution by ageing stage", ylabel="Average cell voltage (V)"); ax.tick_params(axis="x", rotation=15); save(fig, "voltage_distribution_by_stage.png")
        fig, ax = plt.subplots(figsize=(10, 4.7)); ax.plot(x, half["active_cell_dispersion_mean_V"], linewidth=.8, color="#7C3AED"); ax.axvline(dev_end, color="#B91C1C", linestyle="--"); ax.set(title="Active-cell voltage dispersion (U6 excluded)", xlabel="Operating ageing (h)", ylabel="Cross-cell std (V)"); save(fig, "cell_voltage_dispersion_vs_age.png")
    if not ref.empty:
        normal = ref[~ref["after_air_air_start"]].sort_values("ageing_hour"); dev = normal[normal["ageing_hour"] < dev_end]; test = normal[normal["ageing_hour"] >= dev_end]
        fig, ax = plt.subplots(figsize=(9.5, 5)); ax.errorbar(dev["ageing_hour"], dev["reference_voltage_V"], yerr=dev["active_voltage_std_V"], marker="o", color="#2563EB", label="Development / method selection");
        if len(test): ax.errorbar(test["ageing_hour"], test["reference_voltage_V"], yerr=test["active_voltage_std_V"], marker="s", linestyle="--", color="#B91C1C", label="Test: full descriptive only")
        ax.set(title="Frozen-reference voltage trajectory", xlabel="Nominal polarization ageing checkpoint (h)", ylabel="Active-cell median voltage (V)"); ax.legend(); save(fig, "reference_voltage_vs_age.png")
        fig, ax = plt.subplots(figsize=(9.5, 5)); ax.plot(dev["ageing_hour"], dev["SOH_U"], marker="o", color="#2563EB", label="Development");
        if len(test): ax.plot(test["ageing_hour"], test["SOH_U"], marker="s", linestyle="--", color="#B91C1C", label="Test / descriptive")
        ax.axhline(.90, color="#B91C1C", linestyle="--", label="90% EOL rule"); ax.axhline(.95, color="#D97706", linestyle=":", label="95% auxiliary"); ax.set(title="Sparse reference-condition SOH", xlabel="Operating ageing checkpoint (h)", ylabel="SOH_U = V_ref(t)/V_ref,0"); ax.legend(); save(fig, "soh_vs_age.png")
    if not curves.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.8));
        for stage, group in curves[~curves["after_air_air_start"]].groupby("stage", sort=False): ax.plot(group["j_bin_A_cm2"], group["active_voltage_median_V"], linewidth=1.1, label=stage)
        ax.set(title="Nominal polarization curves by checkpoint", xlabel="Current density (A/cm²)", ylabel="Active-cell median voltage (V)"); ax.legend(ncol=3, fontsize=8); save(fig, "polarization_curves_reference_points.png")
    if not timexer.empty and "absolute_delta_voltage_12h_V" in timexer:
        fig, ax = plt.subplots(figsize=(8, 4.8)); ax.hist(timexer["delta_voltage_12h_V"].dropna() * 1000, bins=50, color="#0F766E", alpha=.85); ax.set(title="Development-only observed 12 h dynamic voltage changes", xlabel="12 h ΔV (mV)", ylabel="count"); save(fig, "timexer_12h_deltaV_development.png")
    return names


def dev_reports(context: dict[str, Any], candidates: pd.DataFrame, points: pd.DataFrame, configs: pd.DataFrame, anchors: pd.DataFrame, timexer: pd.DataFrame, method: dict[str, Any]) -> None:
    primary = method["primary_jstar_A_cm2"]; ref = method["reference_voltage_V"]
    write_text(f"""# Phase RUL90 执行摘要（Development-only）

本文件只使用 `OpHrs2 < {context['dev_end']:.6f} h` 的 Development 信息来冻结参考工况方法。未读取 Test 电压、电流或预测目标。

- 冻结参考工况：名义极化曲线、非 after-Air-Air-start、`j*={primary:g} A/cm²`、10 个 active cell 的中位电压。
- 初始参考：源文件标注为 BoL 的同构 10-cell / 280 cm² 名义极化曲线，`V_ref,0={ref:.9f} V`。
- 这是一条稀疏诊断锚点轨迹，不对 0.5 h 或 1 h 动态电压进行 SOH 插值。
- 90% 阈值已预注册为 `{.9*ref:.9f} V`；95% 仅为辅助敏感性阈值 `{.95*ref:.9f} V`。
""", "00_EXECUTIVE_SUMMARY.md")
    write_text(f"""# 参考工况审计（Reference-condition audit）

## 冻结规则

候选只来自 Development 可见的名义极化曲线：BoL、300、600、850、1150、1500、1750 h；排除同龄 `afterAirAirStart` 文件。候选 `j*` 为 {J_CANDIDATES} A/cm²，容差 ±{J_TOL:g} A/cm²。硬门限要求诊断覆盖率≥90%、至少 5 个 Development checkpoint、距 500 mV lower voltage limit 至少 50 mV，并比较重复离散度、单调性、年龄相关性与 1500 h Air-Air shift。规则在查看任何 Test 电压前保存为 `outputs/METHOD_FREEZE.yaml`。

## 候选证据

{table(candidates.round(7))}

## 配置可比性

Development 名义 polarization 文件的 `Number of cells` 与 `Membrane active area [cm2]` 统计见 `outputs/polarization_configuration.csv`。本指标使用 active 10-cell per-cell mean；`U6` 不进入计算。BoL 文件为源文件标注的 BoL checkpoint，但其内部 `OpHrs` 非零，因此称为“source-labelled BoL reference”，而非未经限定的零使用小时状态。
""", "05_REFERENCE_CONDITION_AUDIT.md")
    write_text(f"""# SOH 构造（State of Health construction）

定义：`SOH_U(t_i) = V_ref(t_i) / V_ref,0`。其中 `V_ref(t_i)` 是已冻结 `j*={primary:g} A/cm²` 的名义极化曲线 active-cell median voltage，`V_ref,0={ref:.9f} V` 是最早合法的 source-labelled BoL reference。

这不是由运行工况瞬时电压构造的 SOH；动态电压的工作点混杂、停机和 load transition 均不被删除或重写。Development 可见原始锚点在 `outputs/reference_voltage_development.csv`，标准化轨迹在 `outputs/soh_trajectory_development.csv`。

{table(anchors[["stage","ageing_hour","reference_voltage_V","SOH_U","valid_sample_count","active_voltage_std_V"]].round(8))}
""", "06_SOH_CONSTRUCTION.md")
    scale = timexer["absolute_delta_voltage_12h_V"].describe().to_frame("value") if "absolute_delta_voltage_12h_V" in timexer else pd.DataFrame()
    write_text(f"""# TimeXer 任务与 RUL 的差距

现有任务是 0.5 h 采样、48 个历史步（24 h）到 24 个未来步（12 h）的动态 `DeltaV`。Development-only 的实际 12 h 动态变化统计（无模型预测）见 `outputs/timexer_12h_signal_scale_dev.csv`：

{table(scale.reset_index(), limit=20)}

该输出不是标准化 `V_ref(t)`，因此不能直接判定 90% EOL。它仍可用于短时动态电压预测；若要研究长期健康，需要以稀疏固定工况 SOH anchor 为监督/校准对象，或显式建立慢时间尺度 health state。禁止把 12 h DeltaV 递推数千小时后当作真实 RUL。
""", "09_TIMEXER_TASK_GAP.md")


def final_reports(context: dict[str, Any], inv: pd.DataFrame, meta: pd.DataFrame, stat: pd.DataFrame, half: pd.DataFrame, events: pd.DataFrame, gaps: pd.DataFrame, segments: pd.DataFrame, evidence: pd.DataFrame, ref: pd.DataFrame, candidates: pd.DataFrame, configs: pd.DataFrame, curve: pd.DataFrame, timexer: pd.DataFrame, decision: dict[str, Any], figures: list[str]) -> None:
    normal = ref[~ref["after_air_air_start"]].sort_values("ageing_hour").copy(); threshold = float(decision["EOL_threshold_voltage_V"])
    first = normal.iloc[0] if len(normal) else pd.Series(dtype=object); last = normal.iloc[-1] if len(normal) else pd.Series(dtype=object)
    write_text(f"""# 数据清单（Data inventory）

递归扫描了 `{len(inv)}` 个实际源文件；清单在 `outputs/data_inventory.csv`。其中可读 `{int(inv['readable'].sum())}` 个、不可读 `{int((~inv['readable']).sum())}` 个。按数据类型汇总：

{table(inv.groupby('data_type', as_index=False).agg(files=('filename','count'), total_MB=('file_size_MB','sum')).round(3))}

原始文件只读取，不写回；大 CSV 均采用 `chunksize={CHUNK}`。
""", "01_DATA_INVENTORY.md")
    file_gaps = meta.loc[pd.to_numeric(meta["gap_from_previous_file_h"], errors="coerce") > 0].copy()
    write_text(f"""# 时间轴审计（Time-axis audit）

主 ageing coordinate 为 `OpHrs2 [h]`，不是 `OpHrs [h]`。全源最小/最大为 `{fmt(meta['min_OpHrs2_h'].min())}` 至 `{fmt(meta['max_OpHrs2_h'].max())}` h，跨度 `{fmt(meta['max_OpHrs2_h'].max()-meta['min_OpHrs2_h'].min())}` h。该数值表示累计运行小时坐标覆盖，不等于 DateTime 日历跨度，也不由记录数×采样间隔推断。

- 文件级时间戳倒序：`{int(meta['timestamp_reversal_count'].sum())}`；重复：`{int(meta['duplicate_timestamp_count'].sum())}`；`OpHrs2` 倒序：`{int(meta['ophrs2_reversal_count'].sum())}`。
- 文件边界正向时间间隔：`{len(file_gaps)}`；文件内部 `DateTime>5 s` 的中断数：`{int(meta['internal_gap_count_dt_gt5s'].sum())}`。
- 这些中断的维护原因仅凭数据不能确认；已经记录而未插值或填平。

## 文件时间段

{table(meta.round(6))}
""", "02_TIME_AXIS_AUDIT.md")
    stack_candidate = stat.loc[stat['category'].eq('stack voltage candidate')]
    write_text(f"""# 信号定义审计（Signal definition audit）

`U.S.AveCell [V]` 是导出的单体平均电压；同时重算 active channels `U1-U5,U7-U11` 的平均值作为交叉检查。`U6 [V]` 按原始源统计表现为非物理/flatline 候选，因此不进入 active-cell SOH 或 dispersion。`MIN(U.S [V])` 仅被标为 stack-voltage candidate；其 MIN 统计语义未由原始字段定义证实。

所有异常仅标记：由 `|ΔU.S.AveCell|≥0.05 V` 产生的跳变根据同秒 `ΔI` 被区分为低负载/停机、正常负载响应或原因不明；未删除任何原始样本。示例记录在 `outputs/voltage_jump_events.csv`。

## 核心字段统计

{table(stat.round(7), limit=80)}

## Stack voltage candidate

{table(stack_candidate.round(7))}
""", "03_SIGNAL_DEFINITION_AUDIT.md")
    excerpt = evidence[evidence['evidence_text'].astype(str).str.contains('500|lower voltage|EOL|end of life|failure', case=False, regex=True, na=False)]
    write_text(f"""# 官方 EOL 审计（Official EOL audit）

`OFFICIAL_EOL_DEFINITION = NOT_FOUND`。原始说明/PDF 中存在 durability lower-voltage-limit 的文字证据（见 `outputs/eol_evidence.csv`），但未发现把 **90% 初始参考电压**定义为官方 EOL、给出官方 EOL time label，或明确说明“2300 h 停止即电池 failure”的结构化原始标签。因此测试结束不能自动等于失效，也不能定义 `RUL=2300-t`。

论文 PDF 第 4 页第 14–20 行说明研究目标是完成超过 2000 operating hours，并在约 200 cycles 后结束 durability test 以做完整 characterization；这支持“预定/研究性结束（EoT）”，而不是已记录 failure。第 8 页第 15–17 行的 500 mV lower voltage limit 出现在 60/50 kPaG 压力下、1.8 A/cm² 的局部 sensitivity measurement 无法继续测量的语境。它是条件特定的运行限值，不是本阶段预注册的 90% reference-voltage EOL，也不能自动证明 run-to-failure。证据摘录：

{table(excerpt[['source_file','page','line','evidence_text','interpretation']], limit=30)}
""", "04_OFFICIAL_EOL_AUDIT.md")
    eol_events = ref[["stage","ageing_hour","after_air_air_start","reference_voltage_V","SOH_U","threshold_eligible","below_90","below_95"]].copy()
    write_text(f"""# 90% EOL 阈值审计

预注册规则：`SOH_EOL=0.90`，即 `V_EOL=0.90×V_ref,0={threshold:.9f} V`。95% 阈值 `{float(decision['EOL95_threshold_voltage_V']):.9f} V` 仅作辅助敏感性检查。判定只使用冻结 j*、同构名义极化曲线、非 after-Air-Air-start 的 reference points；不把任意动态低电压点当作 EOL。

{table(eol_events.round(8), limit=30)}

90% 结论：`EOL_90_OBSERVED = {decision['EOL90_observed']}`；首次 crossing：`{decision['EOL90_first_crossing_hour']}`；持续确认：`{decision['EOL90_sustained_or_confirmed']}`。95%：`EOL95_observed = {decision['EOL95_observed']}`。
""", "07_EOL90_THRESHOLD_AUDIT.md")
    write_text(f"""# RUL 标签可行性

本数据属于 **情况 B：有可追溯的稀疏参考 SOH，但完整观测未提供可确认的 90% EOL**。因此它在 90% 规则下是右删失（right-censored），没有真实点值 RUL 标签，也不能计算真实 EOL RUL 的 MAE/RMSE。

对于每个仍高于阈值的 reference checkpoint，最多可声明下界 `RUL(t) > t_last_valid - t`；其中最终正常 reference point 是 `{fmt(last.get('ageing_hour'))} h`。该下界不是 RUL ground truth。若后续选择 90% 外推，应明确为 `extrapolated RUL`，需要额外 run-to-failure 或独立 EOL 数据验证。
""", "08_RUL_LABEL_FEASIBILITY.md")
    write_text(f"""# 推荐研究方向

唯一推荐主任务：**固定名义工况下的稀疏 reference-condition SOH estimation 与长期健康电压退化预测**，并把 0.5 h / 12 h TimeXer 作为动态电压支路或与慢 health-state 相联系的辅助支路。

不推荐将现有 12 h `DeltaV` 直接循环外推生成数千小时 RUL，也不推荐将本数据写成已具有真实 90% EOL 监督标签的 RUL 数据集。TimeXer 可以继续使用，但预测目标应从单独的未来动态 `DeltaV` 扩展为：过去运行工况 → 当前/未来合法 `V_ref` 或 sparse SOH anchor，并严格区分未来工况未知与 protocol-known scenario。核心创新应是可验证的 fast dynamic response 与 slow reference-condition health state 的因果对齐，而不是堆叠新的网络模块。
""", "10_RECOMMENDED_RESEARCH_DIRECTION.md")
    questions = [
        ("1", "实际运行小时", f"OpHrs2 覆盖 {fmt(decision['data_start_operating_hour'])}–{fmt(decision['data_end_operating_hour'])} h，跨度 {fmt(decision['actual_coverage_hours'])} h。"),
        ("2", "官方 EOL", "NOT_FOUND；仅检出 lower-voltage-limit 文字证据。"),
        ("3", "实验结束是否等于失效", "不能确认，不能把约 2300 h 自动等同 failure。"),
        ("4", "性能诊断点", f"正常名义 polarization reference checkpoints 共 {int((~ref['after_air_air_start']).sum())} 个（含 BoL 至 2300 h）。"),
        ("5", "周期性固定工况", "有：名义极化曲线，适合固定 j* 的稀疏参考测量。"),
        ("6", "V_ref", "可构建，条件是仅使用冻结 j*、非 Air-Air-start、同构名义 polarization。"),
        ("7", "初始参考", f"{fmt(decision['initial_reference_voltage_V'],9)} V，source-labelled BoL，需保留非零 OpHrs 的限定。"),
        ("8", "末端参考", f"{fmt(decision['final_reference_voltage_V'],9)} V，最后正常 checkpoint。"),
        ("9", "最大确认退化", f"{fmt(decision['maximum_confirmed_degradation_percent'])}%（以冻结 reference 定义）。"),
        ("10", "90%", f"{decision['EOL90_observed']}。"),
        ("11", "95%", f"{decision['EOL95_observed']}。"),
        ("12", "真实 RUL", "不可计算；无已观测 90% EOL point label。"),
        ("13", "可验证性", "可验证 reference voltage/SOH observation 与短时动态电压；不可验证真实 EOL RUL 误差。"),
        ("14", "12h TimeXer 缺口", "其输出不是 reference voltage；缺少慢 health-state、EOL observation 与未来场景定义。"),
        ("15", "唯一研究方向", "reference-condition SOH / long-horizon health-voltage forecasting，而非 supervised true-RUL。"),
    ]
    qmd = "\n".join(f"{n}. **{title}**：{answer}" for n, title, answer in questions)
    write_text(f"""# FINAL RUL90 数据可行性报告

## 最终判断

RealHyFC 2300 h 数据可以支撑以固定名义工况极化曲线构造的稀疏 SOH/健康电压研究，**不能直接支撑“以 90% 初始参考电压为已观测真实 EOL”的监督 RUL 论文**。原因不是 90% 定义本身不合法，而是完整观测没有提供/确认该阈值的 EOL event；在本冻结定义下，它是右删失而非 run-to-failure 标签数据。

冻结方法先在 Development 完成，随后完整 2300 h 数据只作为 `FULL_DATA_EXPLORATORY_DESCRIPTIVE_ANALYSIS` 使用。故 `full_data_target_accessed=true`、`test_seal_for_future_model_selection=compromised`，但 `test_prediction_metrics_computed=false`、`model_training_performed=false`。

## 核心数值

| 项目 | 数值 |
| --- | --- |
| 冻结 j* | {decision['reference_voltage_method']} |
| 初始 V_ref | {fmt(decision['initial_reference_voltage_V'],9)} V |
| 末端 V_ref | {fmt(decision['final_reference_voltage_V'],9)} V |
| 最小有效 SOH | {fmt(decision['minimum_valid_SOH'],8)} |
| 最大确认退化 | {fmt(decision['maximum_confirmed_degradation_percent'])}% |
| 90% 阈值 | {fmt(decision['EOL_threshold_voltage_V'],9)} V |
| 90% 是否观测到 | {decision['EOL90_observed']} |
| 95% 是否观测到 | {decision['EOL95_observed']} |
| 真实 RUL 标签 | {decision['real_RUL_labels_available']} |

## 必须回答的 15 个研究问题

{qmd}

## 关键图表

{chr(10).join('- `' + x + '`' for x in figures)}
""", "FINAL_RUL90_FEASIBILITY_REPORT.md")


def build_decision(context: dict[str, Any], ref: pd.DataFrame, method: dict[str, Any]) -> dict[str, Any]:
    normal = ref[(~ref["after_air_air_start"]) & ref["threshold_eligible"]].sort_values("ageing_hour").copy()
    v0 = float(method["reference_voltage_V"]); vfinal = float(normal["reference_voltage_V"].iloc[-1]); threshold90 = .9 * v0; threshold95 = .95 * v0
    below90 = normal["reference_voltage_V"] <= threshold90; below95 = normal["reference_voltage_V"] <= threshold95
    def first_cross(mask: pd.Series) -> Any:
        return float(normal.loc[mask, "ageing_hour"].iloc[0]) if mask.any() else "NOT_OBSERVED"
    sustained = bool(any(np.convolve(below90.to_numpy(int), np.ones(2, dtype=int), mode="valid") == 2)) if len(below90) >= 2 else False
    return {"dataset_name": "RealHyFC approximately 2300 h PEMFC durability dataset", "dataset_files_found": "see outputs/data_inventory.csv", "data_start_operating_hour": float(context["meta"]["min_OpHrs2_h"].min()), "data_end_operating_hour": float(context["meta"]["max_OpHrs2_h"].max()), "actual_coverage_hours": float(context["meta"]["max_OpHrs2_h"].max() - context["meta"]["min_OpHrs2_h"].min()), "official_eol_definition_found": False, "official_eol_source": "NOT_FOUND; lower-voltage-limit evidence recorded in outputs/eol_evidence.csv", "official_eol_time_found": "NOT_FOUND", "reference_voltage_available": True, "reference_voltage_method": f"nominal polarization, active 10-cell median, frozen j*={method['primary_jstar_A_cm2']:g} A/cm2", "reference_voltage_quality": "CONDITIONAL: source-labelled BoL, same reported 10-cell/280 cm2 polarization configuration; BoL OpHrs is nonzero", "true_BOL_reference_available": "CONDITIONAL_SOURCE_LABELLED_BOL", "first_observed_reference_age": 0.0, "initial_reference_voltage_V": v0, "final_reference_voltage_V": vfinal, "minimum_valid_SOH": float(normal["SOH_U"].min()), "maximum_confirmed_degradation_percent": float((1 - normal["SOH_U"].min()) * 100), "EOL_threshold_fraction": 0.90, "EOL_threshold_voltage_V": threshold90, "EOL95_threshold_voltage_V": threshold95, "EOL90_observed": bool(below90.any()), "EOL90_first_crossing_hour": first_cross(below90), "EOL90_crossing_time_interval": "NOT_OBSERVED" if not below90.any() else "observed at sparse checkpoint; interval requires adjacent checkpoint review", "EOL90_sustained_or_confirmed": sustained, "EOL95_observed": bool(below95.any()), "EOL95_first_crossing_hour": first_cross(below95), "real_RUL_labels_available": False, "right_censored": True, "RUL_lower_bound_available": True, "extrapolation_required": True, "current_12h_Task_directly_supports_RUL": False, "recommended_research_task": "reference-condition SOH estimation and long-horizon health-voltage forecasting", "recommended_timexer_role": "short-horizon dynamic voltage auxiliary branch, aligned to a slow reference-condition health state", "data_limitations": "No confirmed official EOL/90% event; sparse diagnostic anchors; full Test target inspected descriptively after method freeze; future operating conditions are not automatically known.", "full_data_target_accessed": True, "test_seal_for_future_model_selection": "compromised", "test_prediction_metrics_computed": False, "model_training_performed": False}


def development_freeze(context: dict[str, Any], run: dict[str, Any]) -> None:
    raw = context["raw"]; dev_end = context["dev_end"]
    # Only paths at or before 1750 h (and the 1500-h event repeat) enter this stage.
    all_paths = sorted(raw.glob("Polcurve_Nominal_RHFC_*.csv"), key=lambda p: (stage_from_path(p)[1], stage_from_path(p)[2]))
    paths = [p for p in all_paths if stage_from_path(p)[1] < dev_end]
    points, configs, curves = analyze_polarization(paths, J_CANDIDATES)
    candidates = selection_summary(points)
    if candidates.empty or not bool(candidates.iloc[0]["hard_gate_pass"]): raise RuntimeError("Development data 无法通过参考工况硬门限，不能继续 full-data audit。")
    primary = float(candidates.iloc[0]["target_jstar_A_cm2"])
    bol = points[(points["stage"] == "BoL") & (~points["after_air_air_start"]) & np.isclose(points["target_jstar_A_cm2"], primary)]
    if len(bol) != 1 or not math.isfinite(float(bol["active_voltage_median_V"].iloc[0])): raise RuntimeError("冻结失败：未找到唯一的合法 BoL reference point。")
    v0 = float(bol["active_voltage_median_V"].iloc[0])
    anchors = points[(~points["after_air_air_start"]) & np.isclose(points["target_jstar_A_cm2"], primary)].sort_values("ageing_hour").copy(); anchors["reference_voltage_V"] = anchors["active_voltage_median_V"]; anchors["SOH_U"] = anchors["reference_voltage_V"] / v0
    method = {"method_status": "FROZEN_BEFORE_TEST_TARGET_ACCESS", "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "project_git_head_expected": "34c4c447c900ef2ca6f24315ca229e0c3533536a", "development_end_ophrs2_h": dev_end, "selection_scope": "normal nominal polarization checkpoints with age < Development/Test boundary; Test voltage/current/targets not read", "candidate_jstar_A_cm2": J_CANDIDATES, "j_tolerance_A_cm2": J_TOL, "primary_jstar_A_cm2": primary, "reference_voltage_V": v0, "reference_definition": "active 10-cell median voltage at frozen j* on source-labelled nominal BoL polarization curve", "excluded_channel": "U6 [V]", "after_air_air_start_policy": "excluded from reference selection and threshold event confirmation", "SOH_definition": "SOH_U(t)=V_ref(t)/V_ref,0", "EOL_threshold_fraction": 0.90, "EOL_threshold_voltage_V": .9 * v0, "auxiliary_threshold_fraction": 0.95, "auxiliary_threshold_voltage_V": .95 * v0, "centred_smoothing_or_backfill": False, "model_training_performed": False}
    write_csv(points, "polarization_points_development.csv"); write_csv(configs, "polarization_configuration_development.csv"); write_csv(curves, "polarization_curves_development.csv"); write_csv(candidates, "reference_jstar_candidates_development.csv"); write_csv(anchors, "reference_voltage_development.csv"); write_csv(anchors, "soh_trajectory_development.csv")
    timexer = timexer_dev_scale(dev_end); write_csv(timexer, "timexer_12h_signal_scale_dev.csv")
    write_yaml(method, METHOD_PATH); dev_reports(context, candidates, points, configs, anchors, timexer, method)
    run.update({"stage": "development-freeze", "status": "SUCCESS", "development_target_information_accessed": True, "test_target_information_accessed": False, "test_seal_for_future_model_selection": "preserved_until_full_descriptive_stage", "frozen_method": str(METHOD_PATH), "primary_jstar_A_cm2": primary, "reference_voltage_V": v0, "outputs": sorted(str(p.relative_to(PHASE)) for p in PHASE.rglob("*") if p.is_file())})
    RUN_PATH.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")


def full_descriptive(context: dict[str, Any], run: dict[str, Any]) -> None:
    if not METHOD_PATH.is_file(): raise RuntimeError("METHOD_FREEZE.yaml 不存在；必须先执行 development-freeze。")
    method = yaml.safe_load(METHOD_PATH.read_text(encoding="utf-8"))
    if method.get("method_status") != "FROZEN_BEFORE_TEST_TARGET_ACCESS" or not math.isclose(float(method.get("development_end_ophrs2_h", math.nan)), context["dev_end"], abs_tol=1e-9): raise RuntimeError("方法冻结文件不完整或 Development 边界不匹配。")
    raw = context["raw"]; paths = sorted((raw / "Loadcycle_data").rglob("*.csv")); meta, _ = loadcycle_metadata(paths)
    context["meta"] = meta
    stat, half, events, within_gaps, segments = audit_loadcycle(paths)
    boundary_gaps = []
    for i in range(1, len(meta)):
        gap = pd.to_numeric(meta.loc[i, "gap_from_previous_file_h"], errors="coerce")
        if math.isfinite(gap) and abs(gap) > 1 / 3600: boundary_gaps.append({"gap_scope": "between_files", "from_file": meta.loc[i - 1, "filename"], "to_file": meta.loc[i, "filename"], "gap_seconds": gap * 3600, "previous_timestamp": meta.loc[i - 1, "last_timestamp"], "next_timestamp": meta.loc[i, "first_timestamp"], "classification": "calendar-time gap/overlap; physical cause unresolved"})
    gaps = pd.concat([within_gaps, pd.DataFrame(boundary_gaps)], ignore_index=True)
    inv = inventory(raw); evidence = eol_evidence(raw)
    pol_paths = sorted(raw.glob("Polcurve_Nominal_RHFC_*.csv"), key=lambda p: (stage_from_path(p)[1], stage_from_path(p)[2]))
    points, configs, curves = analyze_polarization(pol_paths, [float(method["primary_jstar_A_cm2"])])
    ref = points.copy(); ref["reference_voltage_V"] = ref["active_voltage_median_V"]; ref["reference_voltage_initial_V"] = float(method["reference_voltage_V"]); ref["SOH_U"] = ref["reference_voltage_V"] / float(method["reference_voltage_V"]); ref["threshold_eligible"] = ~ref["after_air_air_start"]; ref["below_90"] = ref["threshold_eligible"] & (ref["reference_voltage_V"] <= float(method["EOL_threshold_voltage_V"])); ref["below_95"] = ref["threshold_eligible"] & (ref["reference_voltage_V"] <= float(method["auxiliary_threshold_voltage_V"])); ref["analysis_scope"] = np.where(ref["ageing_hour"] < context["dev_end"], "development", "FULL_DATA_EXPLORATORY_DESCRIPTIVE_ANALYSIS")
    decision = build_decision(context, ref, method)
    write_csv(inv, "data_inventory.csv"); write_csv(meta, "time_segments.csv"); write_csv(gaps, "time_gaps.csv"); write_csv(stat, "signal_statistics.csv"); write_csv(stat[["original_field","raw_count","missing_count","missing_percent","abnormal_percent"]], "missingness.csv"); write_csv(half, "half_hourly_loadcycle_summary.csv"); write_csv(events, "voltage_jump_events.csv"); write_csv(segments, "read_segments.csv"); write_csv(evidence, "eol_evidence.csv"); write_csv(points, "polarization_points_full_descriptive.csv"); write_csv(configs, "polarization_configuration.csv"); write_csv(curves, "polarization_curves_full.csv"); write_csv(ref, "reference_voltage.csv"); write_csv(ref, "soh_trajectory.csv"); write_csv(ref[["stage","ageing_hour","after_air_air_start","reference_voltage_V","SOH_U","threshold_eligible","below_90","below_95"]], "eol_threshold_events.csv")
    timexer = pd.read_csv(OUTPUTS / "timexer_12h_signal_scale_dev.csv") if (OUTPUTS / "timexer_12h_signal_scale_dev.csv").is_file() else pd.DataFrame()
    figures = make_figures(meta, half, ref, curves, context["dev_end"], timexer)
    final_reports(context, inv, meta, stat, half, events, gaps, segments, evidence, ref, pd.read_csv(OUTPUTS / "reference_jstar_candidates_development.csv"), configs, curves, timexer, decision, figures)
    write_yaml(decision, PHASE / "FINAL_DECISION.yaml")
    run.update({"stage": "full-descriptive", "status": "SUCCESS", "development_target_information_accessed": True, "test_target_information_accessed": True, "full_data_target_accessed": True, "test_seal_for_future_model_selection": "compromised", "test_prediction_metrics_computed": False, "model_training_performed": False, "source_files_scanned": int(len(inv)), "loadcycle_rows": int(meta["rows"].sum()), "warnings": ["Full Test voltage/current were accessed only after METHOD_FREEZE.yaml was written; all final Test-related findings are descriptive and cannot support future model selection."], "outputs": sorted(str(p.relative_to(PHASE)) for p in PHASE.rglob("*") if p.is_file())})
    RUN_PATH.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_existing_full_outputs(context: dict[str, Any], run: dict[str, Any]) -> None:
    """Recover only the final presentation after a completed raw-data pass.

    This mode is deliberately unable to choose a method or modify source data.
    It reads summaries written by the preceding full-data pass and records that
    Test targets had already been accessed in that pass.
    """
    required = ["data_inventory.csv", "time_segments.csv", "signal_statistics.csv", "half_hourly_loadcycle_summary.csv", "eol_evidence.csv", "reference_voltage.csv", "polarization_configuration.csv", "polarization_curves_full.csv", "reference_jstar_candidates_development.csv", "timexer_12h_signal_scale_dev.csv"]
    missing = [name for name in required if not (OUTPUTS / name).is_file()]
    if missing:
        raise RuntimeError(f"无法恢复最终报告；缺少完整全源读取所产生的摘要：{missing}")
    if not METHOD_PATH.is_file():
        raise RuntimeError("METHOD_FREEZE.yaml 不存在；拒绝恢复或创建全数据结论。")
    method = yaml.safe_load(METHOD_PATH.read_text(encoding="utf-8"))
    inv = pd.read_csv(OUTPUTS / "data_inventory.csv"); meta = pd.read_csv(OUTPUTS / "time_segments.csv"); context["meta"] = meta
    stat = pd.read_csv(OUTPUTS / "signal_statistics.csv"); half = pd.read_csv(OUTPUTS / "half_hourly_loadcycle_summary.csv"); evidence = pd.read_csv(OUTPUTS / "eol_evidence.csv"); ref = pd.read_csv(OUTPUTS / "reference_voltage.csv"); configs = pd.read_csv(OUTPUTS / "polarization_configuration.csv"); curves = pd.read_csv(OUTPUTS / "polarization_curves_full.csv"); candidates = pd.read_csv(OUTPUTS / "reference_jstar_candidates_development.csv"); timexer = pd.read_csv(OUTPUTS / "timexer_12h_signal_scale_dev.csv")
    events = pd.read_csv(OUTPUTS / "voltage_jump_events.csv") if (OUTPUTS / "voltage_jump_events.csv").is_file() else pd.DataFrame()
    gaps = pd.read_csv(OUTPUTS / "time_gaps.csv") if (OUTPUTS / "time_gaps.csv").is_file() else pd.DataFrame()
    segments = pd.read_csv(OUTPUTS / "read_segments.csv") if (OUTPUTS / "read_segments.csv").is_file() else pd.DataFrame()
    decision = build_decision(context, ref, method)
    figures = make_figures(meta, half, ref, curves, context["dev_end"], timexer)
    final_reports(context, inv, meta, stat, half, events, gaps, segments, evidence, ref, candidates, configs, curves, timexer, decision, figures)
    write_yaml(decision, PHASE / "FINAL_DECISION.yaml")
    run.update({"stage": "full-descriptive-finalization-recovery", "status": "SUCCESS", "recovery_reason": "Raw full-data audit completed and wrote summaries; previous execution stopped while rendering final reports because of a corrected output-column reference.", "development_target_information_accessed": True, "test_target_information_accessed": True, "full_data_target_accessed": True, "test_seal_for_future_model_selection": "compromised", "test_prediction_metrics_computed": False, "model_training_performed": False, "source_files_scanned": int(len(inv)), "loadcycle_rows": int(meta["rows"].sum()), "warnings": ["No source data was reread in this recovery. Full Test voltage/current had already been accessed after METHOD_FREEZE.yaml in the preceding full-data descriptive pass."], "outputs": sorted(str(p.relative_to(PHASE)) for p in PHASE.rglob("*") if p.is_file())})
    RUN_PATH.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["development-freeze", "full-descriptive", "finalize-existing"], required=True); args = parser.parse_args()
    prepare_dirs(); context = load_context(); run = {"script": str(SCRIPT), "stage_requested": args.stage, "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__, "project": str(PROJECT), "raw_data_root": str(context["raw"]), "raw_data_modified": False, "model_training_performed": False, "git_head_before": os.popen(f'git -C "{PROJECT}" rev-parse HEAD').read().strip()}
    if args.stage == "development-freeze": development_freeze(context, run)
    elif args.stage == "full-descriptive": full_descriptive(context, run)
    else: finalize_existing_full_outputs(context, run)


if __name__ == "__main__":
    main()

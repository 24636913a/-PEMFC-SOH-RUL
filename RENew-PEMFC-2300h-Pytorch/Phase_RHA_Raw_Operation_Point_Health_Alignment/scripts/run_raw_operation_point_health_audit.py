"""Frozen, zero-training Phase RHA raw operation-point health audit."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
OUT, FIG, REPORT, FINAL = PHASE / "outputs", PHASE / "outputs" / "figures", PHASE / "reports", PHASE / "final"
CONFIG, CONFIG_HASH = PHASE / "configs" / "PREREGISTERED_RHA_PROTOCOL.yaml", PHASE / "configs" / "PREREGISTERED_RHA_PROTOCOL.sha256"
MANIFEST = PHASE / "SOURCE_MANIFEST.yaml"
RCR = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit"
RAW_ROOT = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据")

BOUNDARY = 1819.997738
CHUNK = 50_000
ACTIVE = [f"U{i} [V]" for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
IDENTITY = ["OpHrs2 [h]", "DateTime", "operation_point", "I.S.set [A]", "I.S.act [A]", "j.S.act [A/cm2]"]
COND_RAW = {
    "current_density": "j.S.act [A/cm2]",
    "anode_inlet_temperature_C": "T.Si.A [°C]",
    "anode_inlet_pressure_kPag": "p.Si.A [kPag]",
    "anode_inlet_RH_pct": "RH.Si.A  [%]",
    "anode_hydrogen_flow_nl_min": "F.Si.H.A [nl/min]",
    "anode_stoichiometry": "Stoic.S.A [-]",
    "cathode_stoichiometry": "Stoic.S.C [-]",
}
CONDITIONS = list(COND_RAW)
RCR_HASHES = {
    "outputs/CYCLE_TEMPLATE.json": "267a3ee5a2d297b4368f4dd262771a19109e8015b2617de45ab828d83d696200",
    "outputs/RAW_CYCLE_CATALOG.csv": "8d1a8b9e17b179fbc99776f4006a61e5758006bd73e1a55bb96ffed38fa0f3d8",
    "outputs/CYCLE_SEQUENCE_SIMILARITY.csv": "d7e10afbadd86c6a451ab4d25daf920322218ac33338ca2b53e90b6d8cf86652",
    "outputs/RAW_STATE_RUNS.csv": "d385fc72c7a54ba93b9ca38fe275b8c5530180bf3d109ad842dcd6a3682865b3",
    "final/FINAL_DECISION.yaml": "78423aac7166e896e16ba8dc591c9c62c1e01a3854d0f664946036efca5ae9da",
    "final/POSTHOC_FIREWALL_QUALIFICATION.yaml": "cceb84dde51058d1299d4ba60505ebd3a90e649eea67df679a6f4de99df94691",
}
BLUE, GREEN, RED, TEAL, VIOLET, GREY = "#0F4D92", "#8BCF8B", "#B64342", "#42949E", "#9A4D8E", "#666666"


class AuditFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dirs() -> None:
    for path in (OUT, FIG, REPORT, FINAL):
        path.mkdir(parents=True, exist_ok=True)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def verify_protocol() -> dict[str, str]:
    expected = CONFIG_HASH.read_text(encoding="utf-8").split()[0]
    actual = sha256(CONFIG)
    if expected != actual:
        raise AuditFailure(f"RHA protocol SHA mismatch: {expected} != {actual}")
    return {"protocol_sha256": actual, "execution_head": git_head()}


def load_rcr() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    observed = {relative: sha256(RCR / relative) for relative in RCR_HASHES}
    bad = {relative: (RCR_HASHES[relative], observed[relative]) for relative in RCR_HASHES if observed[relative] != RCR_HASHES[relative]}
    if bad:
        raise AuditFailure(f"Frozen RCR artifact hash mismatch: {bad}")
    final = yaml.safe_load((RCR / "final" / "FINAL_DECISION.yaml").read_text(encoding="utf-8"))
    if final.get("status") != "RAW_CYCLE_STRUCTURE_QUALIFIED":
        raise AuditFailure("RCR final scientific result is not RAW_CYCLE_STRUCTURE_QUALIFIED.")
    template = json.loads((RCR / "outputs" / "CYCLE_TEMPLATE.json").read_text(encoding="utf-8"))
    cycles = pd.read_csv(RCR / "outputs" / "RAW_CYCLE_CATALOG.csv")
    states = pd.read_csv(RCR / "outputs" / "RAW_STATE_RUNS.csv")
    complete = cycles.loc[cycles["complete_cycle"].astype(bool)].copy()
    if len(complete) != 146:
        raise AuditFailure(f"Expected 146 frozen complete cycles, found {len(complete)}.")
    return template, complete, states


def op_sort(value: Any) -> tuple[int, float | str]:
    text = str(value)
    try:
        return 0, float(text)
    except ValueError:
        return 1, text


def select_points(template: dict[str, Any]) -> dict[str, Any]:
    start, end = int(template["template_start_run_id"]), int(template["template_end_run_id_exclusive"])
    states = pd.read_csv(RCR / "outputs" / "RAW_STATE_RUNS.csv")
    core = states.loc[
        (states["run_id"] >= start)
        & (states["run_id"] < end)
        & (states["stop_or_zero_current_flag"] == 0)
        & (states["median_actual_current_A"] > 0)
    ].copy()
    frequency = core["operation_point"].astype(str).value_counts().rename_axis("operation_point").reset_index(name="template_positive_run_frequency")
    frequency = frequency.sort_values(
        ["template_positive_run_frequency", "operation_point"],
        ascending=[False, True],
        key=lambda col: col.map(op_sort) if col.name == "operation_point" else col,
        kind="mergesort",
    ).reset_index(drop=True)
    if len(frequency) < 2:
        raise AuditFailure("Frozen RCR template has fewer than two positive-current operation points.")
    return {"P1": str(frequency.iloc[0]["operation_point"]), "P2": str(frequency.iloc[1]["operation_point"]), "ranking": frequency.to_dict(orient="records")}


def attach_target_runs(cycles: pd.DataFrame, states: pd.DataFrame, points: dict[str, Any]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    point_lookup = {points["P1"]: "P1", points["P2"]: "P2"}
    for cycle in cycles.itertuples(index=False):
        subset = states.loc[
            (states["run_id"] >= cycle.start_run_id)
            & (states["run_id"] < cycle.end_run_id_exclusive)
            & (states["duration_wall_s"] >= 120.0)
            & (states["stop_or_zero_current_flag"] == 0)
            & (states["median_actual_current_A"] > 0)
        ].copy()
        subset["operation_point"] = subset["operation_point"].astype(str)
        subset = subset.loc[subset["operation_point"].isin(point_lookup)].copy()
        if subset.empty:
            continue
        subset["point_label"] = subset["operation_point"].map(point_lookup)
        subset["raw_cycle_id"] = int(cycle.raw_cycle_id)
        subset["age_stratum"] = str(cycle.age_stratum)
        subset["cycle_start_ophrs2_h"] = float(cycle.start_ophrs2_h)
        subset["cycle_end_ophrs2_h"] = float(cycle.end_ophrs2_h)
        subset["rounded_set_A"] = subset["state_token"].astype(str).str.rsplit("||", n=1).str[-1].astype(float)
        records.append(subset)
    if not records:
        raise AuditFailure("No frozen RCR target runs remain after 120-s stable-dwell rule.")
    target = pd.concat(records, ignore_index=True).sort_values("start_datetime").reset_index(drop=True)
    return target


def file_order(path: Path) -> tuple[int, str]:
    match = re.search(r"Loadcycle_(\d+)", path.stem)
    if not match:
        raise AuditFailure(f"Unparseable Loadcycle filename {path.name}")
    return int(match.group(1)), path.name


def raw_files() -> list[Path]:
    files = sorted((RAW_ROOT / "Loadcycle_data").rglob("Loadcycle_*.csv"), key=file_order)
    if len(files) != 17:
        raise AuditFailure(f"Expected 17 raw Loadcycle files, found {len(files)}")
    return files


def normalize_age(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype("string").str.replace(",", "", regex=False), errors="coerce")


def normalize_operation(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = values.astype("string").str.strip()
    mask = numeric.notna()
    output.loc[mask] = numeric.loc[mask].map(lambda x: f"{float(x):g}")
    return output.astype(str)


def boundary_filter(chunk: pd.DataFrame) -> tuple[pd.DataFrame, int, bool]:
    """Production crossing-chunk firewall before cells or conditions are parsed."""
    age = normalize_age(chunk["OpHrs2 [h]"])
    above = age >= BOUNDARY
    accessed = int(above.sum())
    keep = chunk.loc[~above].copy()
    keep["OpHrs2 [h]"] = age.loc[~above]
    return keep, accessed, bool(accessed)


def window_table(target: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind not in {"middle_50", "last_25"}:
        raise ValueError(kind)
    fraction = (0.25, 0.75) if kind == "middle_50" else (0.75, 1.0)
    runs = target.copy()
    start = pd.to_datetime(runs["start_datetime"], errors="raise")
    duration = pd.to_timedelta(runs["duration_wall_s"] * fraction[0], unit="s")
    end_duration = pd.to_timedelta(runs["duration_wall_s"] * fraction[1], unit="s")
    runs["window_start"] = start + duration
    runs["window_end"] = start + end_duration
    runs["window_kind"] = kind
    runs["window_id"] = np.arange(len(runs), dtype=int)
    return runs.sort_values("window_start").reset_index(drop=True)


def frozen_window_membership(datetimes_ns: np.ndarray, starts_ns: np.ndarray, ends_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign samples only to their fixed stable-dwell interval, if any."""
    window_index = np.searchsorted(starts_ns, datetimes_ns, side="right") - 1
    valid = np.zeros(len(datetimes_ns), dtype=bool)
    positions = np.flatnonzero(window_index >= 0)
    valid[positions] = datetimes_ns[positions] <= ends_ns[window_index[positions]]
    return window_index, valid


def collect_window_samples(target: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    windows = pd.concat([window_table(target, "middle_50"), window_table(target, "last_25")], ignore_index=True)
    windows = windows.sort_values("window_start").reset_index(drop=True)
    # Use one global identifier after concatenating the two non-overlapping
    # stable-dwell definitions; local run identifiers would otherwise collide.
    windows["window_id"] = np.arange(len(windows), dtype=int)
    starts = windows["window_start"].to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = windows["window_end"].to_numpy(dtype="datetime64[ns]").astype("int64")
    raw_columns = list(dict.fromkeys([*IDENTITY, *ACTIVE, *COND_RAW.values()]))
    sample_store: dict[int, list[np.ndarray]] = defaultdict(list)
    access = {"raw_files_opened": [], "raw_post_boundary_rows_accessed": 0, "raw_post_boundary_rows_used": 0, "crossing_file": None, "stopped_before_later_files": False}
    stopped = False
    for path in raw_files():
        if stopped:
            break
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        missing = [name for name in raw_columns if name not in header]
        if missing:
            raise AuditFailure(f"RHA raw schema missing {missing} in {path.name}")
        access["raw_files_opened"].append(str(path.relative_to(RAW_ROOT)).replace("\\", "/"))
        for chunk in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=raw_columns, chunksize=CHUNK, low_memory=False):
            development, accessed, crossing = boundary_filter(chunk)
            access["raw_post_boundary_rows_accessed"] += accessed
            if not development.empty:
                development["DateTime"] = pd.to_datetime(development["DateTime"], errors="coerce")
                development["operation_point"] = normalize_operation(development["operation_point"])
                for col in ["I.S.set [A]", "I.S.act [A]", "j.S.act [A/cm2]", *ACTIVE, *COND_RAW.values()]:
                    development[col] = pd.to_numeric(development[col], errors="coerce")
                development = development.loc[development["DateTime"].notna() & development["OpHrs2 [h]"].notna()].copy()
                if not development.empty:
                    dt = development["DateTime"].to_numpy(dtype="datetime64[ns]").astype("int64")
                    window_index, valid = frozen_window_membership(dt, starts, ends)
                    if np.any(valid):
                        selected = development.loc[valid].copy()
                        idx = window_index[valid]
                        selected["_window_index"] = idx
                        expected_op = windows.iloc[idx]["operation_point"].astype(str).to_numpy()
                        expected_set = windows.iloc[idx]["rounded_set_A"].to_numpy(dtype=float)
                        observed_set = np.round(selected["I.S.set [A]"].to_numpy(dtype=float), 3)
                        same_op = selected["operation_point"].astype(str).to_numpy() == expected_op
                        same_set = np.isclose(observed_set, expected_set, equal_nan=False)
                        if not np.all(same_op) or not np.all(same_set):
                            first = int(np.flatnonzero(~(same_op & same_set))[0])
                            raise AuditFailure(
                                "Raw measurement identity mismatch at "
                                f"{path.name} at {selected['DateTime'].iloc[first]}: observed ({selected['operation_point'].iloc[first]}, {observed_set[first]:.3f}) "
                                f"vs frozen ({expected_op[first]}, {expected_set[first]:.3f})."
                            )
                        cell_mean = np.nanmean(selected[ACTIVE].to_numpy(dtype=float), axis=1)
                        values = np.column_stack([cell_mean, selected[[COND_RAW[key] for key in CONDITIONS]].to_numpy(dtype=float)])
                        for window_id, array in zip(selected["_window_index"].to_numpy(dtype=int), values):
                            sample_store[int(window_id)].append(array)
            if crossing:
                access["crossing_file"] = str(path.relative_to(RAW_ROOT)).replace("\\", "/")
                access["stopped_before_later_files"] = True
                stopped = True
                break
    records: list[dict[str, Any]] = []
    for window in windows.itertuples(index=False):
        array = np.asarray(sample_store.get(int(window.window_id), []), dtype=float)
        row = {
            "raw_cycle_id": int(window.raw_cycle_id), "age_stratum": window.age_stratum, "point_label": window.point_label,
            "operation_point": str(window.operation_point), "rcr_run_id": int(window.run_id), "stable_window": window.window_kind,
            "run_duration_wall_s": float(window.duration_wall_s), "raw_sample_count": int(len(array)),
            "cycle_start_ophrs2_h": float(window.cycle_start_ophrs2_h), "cycle_end_ophrs2_h": float(window.cycle_end_ophrs2_h),
        }
        columns = ["cell_voltage_mean_V", *CONDITIONS]
        for i, name in enumerate(columns):
            row[name] = float(np.nanmedian(array[:, i])) if len(array) and np.isfinite(array[:, i]).any() else np.nan
        records.append(row)
    return pd.DataFrame(records), access


def aggregate_cycle_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (kind, cycle_id, point), group in measurements.groupby(["stable_window", "raw_cycle_id", "point_label"], sort=True):
        row = {
            "stable_window": kind, "raw_cycle_id": int(cycle_id), "point_label": point,
            "operation_point": str(group["operation_point"].iloc[0]), "age_stratum": group["age_stratum"].iloc[0],
            "cycle_start_ophrs2_h": float(group["cycle_start_ophrs2_h"].iloc[0]), "cycle_end_ophrs2_h": float(group["cycle_end_ophrs2_h"].iloc[0]),
            "eligible_run_count": int(len(group)), "valid_run_count": int(group["cell_voltage_mean_V"].notna().sum()),
        }
        for column in ["cell_voltage_mean_V", *CONDITIONS]:
            row[column] = float(np.nanmedian(group[column])) if group[column].notna().any() else np.nan
        records.append(row)
    return pd.DataFrame(records)


def choose_references(cycle_points: pd.DataFrame, points: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    references: list[dict[str, Any]] = []
    output: list[pd.DataFrame] = []
    for kind in ["middle_50", "last_25"]:
        subset = cycle_points.loc[cycle_points["stable_window"] == kind].copy()
        for point in ["P1", "P2"]:
            data = subset.loc[(subset["point_label"] == point) & subset["cell_voltage_mean_V"].notna()].sort_values("cycle_start_ophrs2_h").copy()
            if len(data) < 10:
                raise AuditFailure(f"{point} has fewer than ten legal complete cycles for frozen reference selection.")
            ref_ids = data.head(10)["raw_cycle_id"].astype(int).tolist()
            ref_v = float(np.median(data.head(10)["cell_voltage_mean_V"]))
            data["is_reference_cycle"] = data["raw_cycle_id"].astype(int).isin(ref_ids)
            data["is_evaluation_cycle"] = ~data["is_reference_cycle"]
            data["reference_voltage_V"] = ref_v
            data["operating_voltage_retention"] = data["cell_voltage_mean_V"] / ref_v
            output.append(data)
            for rank, row in enumerate(data.head(10).itertuples(index=False), start=1):
                references.append({"stable_window": kind, "point_label": point, "reference_rank": rank, "raw_cycle_id": int(row.raw_cycle_id), "cycle_end_ophrs2_h": float(row.cycle_end_ophrs2_h), "cell_voltage_mean_V": float(row.cell_voltage_mean_V), "reference_voltage_V": ref_v})
    retention = pd.concat(output, ignore_index=True)
    return pd.DataFrame(references), retention


def support_tables(retention: pd.DataFrame, cycles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    primary = retention.loc[(retention["stable_window"] == "middle_50")].copy()
    cycle_counts = cycles.groupby("age_stratum")["raw_cycle_id"].nunique().to_dict()
    records: list[dict[str, Any]] = []
    joint_rows: list[pd.DataFrame] = []
    gate_a = gate_b = gate_joint = True
    for point in ["P1", "P2"]:
        data = primary.loc[primary["point_label"] == point].copy()
        ranges: dict[str, tuple[float, float]] = {}
        for variable in CONDITIONS:
            mins, maxs = [], []
            for stratum in ["EARLY", "MIDDLE", "LATE"]:
                group = data.loc[(data["age_stratum"] == stratum) & data["is_evaluation_cycle"]].copy()
                valid = group[variable].dropna()
                complete_rate = float(group[CONDITIONS].notna().all(axis=1).mean()) if len(group) else 0.0
                rate = float(group["cell_voltage_mean_V"].notna().mean()) if len(group) else 0.0
                count = int(group["cell_voltage_mean_V"].notna().sum())
                gate_a &= count >= 20 and rate >= 0.80
                gate_b &= complete_rate >= 0.90
                lo, hi = (float(valid.min()), float(valid.max())) if len(valid) else (np.nan, np.nan)
                mins.append(lo); maxs.append(hi)
                records.append({"record_type": "stratum_variable", "point_label": point, "age_stratum": stratum, "variable": variable, "evaluation_cycle_count": count, "valid_measurement_rate": rate, "condition_complete_rate": complete_rate, "observed_min": lo, "observed_max": hi})
            common_lo, common_hi = max(mins), min(maxs)
            ranges[variable] = (common_lo, common_hi)
            records.append({"record_type": "common_range", "point_label": point, "age_stratum": "ALL", "variable": variable, "evaluation_cycle_count": np.nan, "valid_measurement_rate": np.nan, "condition_complete_rate": np.nan, "observed_min": common_lo, "observed_max": common_hi, "common_range_nonempty": bool(np.isfinite(common_lo) and common_lo <= common_hi)})
        data["in_joint_common_support"] = data[CONDITIONS].notna().all(axis=1)
        for variable, (lo, hi) in ranges.items():
            data["in_joint_common_support"] &= data[variable].between(lo, hi, inclusive="both")
        data["in_joint_common_support"] &= data["is_evaluation_cycle"]
        for stratum in ["EARLY", "MIDDLE", "LATE"]:
            count = int(((data["age_stratum"] == stratum) & data["in_joint_common_support"]).sum())
            gate_joint &= count >= 15
            records.append({"record_type": "joint_support", "point_label": point, "age_stratum": stratum, "variable": "ALL_7", "evaluation_cycle_count": count, "valid_measurement_rate": np.nan, "condition_complete_rate": np.nan, "observed_min": np.nan, "observed_max": np.nan, "joint_support_pass_ge15": count >= 15})
        joint_rows.append(data)
    return pd.DataFrame(records), pd.concat(joint_rows, ignore_index=True), {"A": bool(gate_a), "B": bool(gate_b), "joint": bool(gate_joint)}


def deterministic_match(joint: pd.DataFrame) -> pd.DataFrame:
    matches: list[dict[str, Any]] = []
    for point in ["P1", "P2"]:
        data = joint.loc[(joint["point_label"] == point) & joint["in_joint_common_support"]].copy()
        early = data.loc[data["age_stratum"] == "EARLY"].copy()
        late = data.loc[data["age_stratum"] == "LATE"].copy()
        scale_source = data.loc[data["age_stratum"] == "EARLY", CONDITIONS]
        med = scale_source.median()
        iqr = scale_source.quantile(0.75) - scale_source.quantile(0.25)
        sd = scale_source.std(ddof=1)
        scale = iqr.where(iqr > 0, sd).replace(0, np.nan)
        variables = scale.dropna().index.tolist()
        if early.empty or late.empty or not variables:
            continue
        pairs: list[tuple[float, int, int]] = []
        for e in early.itertuples(index=False):
            ev = np.asarray([getattr(e, v) for v in variables], dtype=float)
            for l in late.itertuples(index=False):
                lv = np.asarray([getattr(l, v) for v in variables], dtype=float)
                distance = float(np.sqrt(np.sum(((ev - lv) / scale[variables].to_numpy()) ** 2)))
                pairs.append((distance, int(e.raw_cycle_id), int(l.raw_cycle_id)))
        used_e, used_l = set(), set()
        lookup_e, lookup_l = early.set_index("raw_cycle_id"), late.set_index("raw_cycle_id")
        for distance, e_id, l_id in sorted(pairs):
            if e_id in used_e or l_id in used_l:
                continue
            used_e.add(e_id); used_l.add(l_id)
            e, l = lookup_e.loc[e_id], lookup_l.loc[l_id]
            matches.append({"point_label": point, "early_raw_cycle_id": e_id, "late_raw_cycle_id": l_id, "condition_distance": distance, "early_retention": float(e.operating_voltage_retention), "late_retention": float(l.operating_voltage_retention), "late_minus_early_retention": float(l.operating_voltage_retention - e.operating_voltage_retention), "matching_variables": "|".join(variables)})
    return pd.DataFrame(matches)


def contiguous_segments(data: pd.DataFrame) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for _, group in data.groupby("age_stratum", sort=False):
        group = group.sort_values("raw_cycle_id")
        values, prior = [], None
        for row in group.itertuples(index=False):
            if prior is not None and int(row.raw_cycle_id) != prior + 1:
                if values: arrays.append(np.asarray(values, dtype=float))
                values = []
            values.append(float(row.operating_voltage_retention)); prior = int(row.raw_cycle_id)
        if values: arrays.append(np.asarray(values, dtype=float))
    return arrays


def moving_block_sample(segments: list[np.ndarray], n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    candidates = [segment[i:i + block] for segment in segments for i in range(max(0, len(segment) - block + 1))]
    if not candidates:
        return np.asarray([], dtype=float)
    out: list[float] = []
    while len(out) < n:
        out.extend(candidates[int(rng.integers(len(candidates)))].tolist())
    return np.asarray(out[:n], dtype=float)


def bootstrap_contrasts(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, summary = [], []
    rng = np.random.default_rng(2024)
    for point in ["P1", "P2"]:
        data = primary.loc[(primary["point_label"] == point) & primary["is_evaluation_cycle"]].copy()
        early, late = data.loc[data["age_stratum"] == "EARLY"], data.loc[data["age_stratum"] == "LATE"]
        for block in [4, 8]:
            early_segments, late_segments = contiguous_segments(early), contiguous_segments(late)
            values = []
            for iteration in range(2000):
                e = moving_block_sample(early_segments, len(early), block, rng)
                l = moving_block_sample(late_segments, len(late), block, rng)
                contrast = float(np.median(l) - np.median(e)) if len(e) and len(l) else np.nan
                values.append(contrast)
                rows.append({"point_label": point, "block_cycles": block, "iteration": iteration, "late_minus_early_retention": contrast})
            finite = np.asarray(values, dtype=float); finite = finite[np.isfinite(finite)]
            summary.append({"point_label": point, "block_cycles": block, "iterations": len(finite), "bootstrap_median": float(np.median(finite)) if len(finite) else np.nan, "ci95_lower": float(np.quantile(finite, .025)) if len(finite) else np.nan, "ci95_upper": float(np.quantile(finite, .975)) if len(finite) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def drift_contrasts(primary: pd.DataFrame, matches: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for point in ["P1", "P2"]:
        data = primary.loc[(primary["point_label"] == point) & primary["is_evaluation_cycle"]].sort_values("raw_cycle_id")
        early, late = data.loc[data["age_stratum"] == "EARLY"], data.loc[data["age_stratum"] == "LATE"]
        D = float(late.operating_voltage_retention.median() - early.operating_voltage_retention.median())
        adjacent = []
        for a, b in zip(data.itertuples(index=False), data.iloc[1:].itertuples(index=False)):
            if a.age_stratum == b.age_stratum and int(b.raw_cycle_id) == int(a.raw_cycle_id) + 1:
                adjacent.append(abs(float(b.operating_voltage_retention - a.operating_voltage_retention)))
        S = float(np.median(adjacent)) if adjacent else np.nan
        matched = matches.loc[matches["point_label"] == point, "late_minus_early_retention"]
        matched_D = float(np.median(matched)) if len(matched) else np.nan
        b4 = bootstrap.loc[(bootstrap["point_label"] == point) & (bootstrap["block_cycles"] == 4)].iloc[0]
        b8 = bootstrap.loc[(bootstrap["point_label"] == point) & (bootstrap["block_cycles"] == 8)].iloc[0]
        rows.append({"point_label": point, "early_evaluation_cycles": len(early), "late_evaluation_cycles": len(late), "raw_late_minus_early_retention": D, "local_cycle_noise_S": S, "signal_to_local_noise": abs(D) / S if np.isfinite(S) and S > 0 else np.nan, "condition_matched_late_minus_early_retention": matched_D, "matched_pair_count": len(matched), "bootstrap4_ci95_lower": b4.ci95_lower, "bootstrap4_ci95_upper": b4.ci95_upper, "bootstrap8_median": b8.bootstrap_median, "raw_negative": D < 0, "signal_gt_2S": bool(np.isfinite(S) and abs(D) > 2 * S), "bootstrap4_upper_negative": bool(b4.ci95_upper < 0), "bootstrap8_no_direction_reversal": bool(b8.bootstrap_median < 0), "matched_negative": bool(np.isfinite(matched_D) and matched_D < 0)})
    return pd.DataFrame(rows)


def parse_stage(path: Path) -> tuple[str, float, bool]:
    lower = path.stem.lower(); after = "afterairair" in lower
    if "bol" in lower: return "BoL", 0.0, after
    match = re.search(r"(\d+)h", lower)
    if not match: raise AuditFailure(f"Cannot parse polarization checkpoint from {path.name}")
    return f"{int(match.group(1))}h", float(match.group(1)), after


def read_polarization() -> pd.DataFrame:
    allowed = {0.0, 300.0, 600.0, 850.0, 1150.0, 1500.0, 1750.0}
    paths = []
    for path in RAW_ROOT.glob("Polcurve_Nominal_RHFC_*.csv"):
        _, age, after = parse_stage(path)
        if age in allowed and not after: paths.append(path)
    if len(paths) != 7:
        raise AuditFailure(f"Expected exactly seven allowlisted nominal Development polarization files, found {len(paths)}")
    records = []
    cols = list(dict.fromkeys(["Number of cells", "j.S.act [A/cm2]", *ACTIVE]))
    for path in sorted(paths, key=lambda p: parse_stage(p)[1]):
        stage, age, _ = parse_stage(path)
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        if any(c not in header for c in cols): raise AuditFailure(f"Allowlisted polarization schema failed for {path.name}")
        result = {1.0: [], 1.6: []}; cells = []
        for chunk in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=cols, chunksize=50_000, low_memory=False):
            for c in cols: chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            cells.extend(chunk["Number of cells"].dropna().unique().tolist())
            avg = np.nanmean(chunk[ACTIVE].to_numpy(dtype=float), axis=1); density = chunk["j.S.act [A/cm2]"].to_numpy(dtype=float)
            for j in result:
                result[j].extend(avg[np.isfinite(density) & (np.abs(density - j) <= .01)].tolist())
        if set(np.round(cells, 6)) != {10.0}: raise AuditFailure(f"{path.name} is not a nominal 10-cell polarization stack")
        for j, values in result.items():
            records.append({"checkpoint": stage, "ageing_hour": age, "jstar_A_cm2": j, "voltage_median_V": float(np.median(values)) if values else np.nan, "voltage_std_V": float(np.std(values, ddof=0)) if values else np.nan, "valid_sample_count": len(values), "source_file": path.name})
    return pd.DataFrame(records)


def anchor_alignment(retention: pd.DataFrame, polarization: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool, str]:
    primary = retention.loc[(retention["stable_window"] == "middle_50")].pivot_table(index=["raw_cycle_id", "cycle_end_ophrs2_h", "age_stratum"], columns="point_label", values="operating_voltage_retention", aggfunc="first").reset_index()
    primary["cycle_retention_H"] = primary[["P1", "P2"]].mean(axis=1)
    p1 = polarization.loc[polarization["jstar_A_cm2"] == 1.0].copy(); vref = float(p1.loc[p1["ageing_hour"] == 0, "voltage_median_V"].iloc[0])
    rows = []
    for checkpoint in [300, 600, 850, 1150, 1500, 1750]:
        prior = primary.loc[(primary["cycle_end_ophrs2_h"] < checkpoint) & ((checkpoint - primary["cycle_end_ophrs2_h"]) <= 48)].sort_values("cycle_end_ophrs2_h", ascending=False).head(3)
        pol = p1.loc[p1["ageing_hour"] == checkpoint]
        status = "ALIGNED" if len(prior) == 3 and len(pol) == 1 else "UNRESOLVED"
        rows.append({"checkpoint_h": checkpoint, "status": status, "cycle_count": len(prior), "latest_cycle_distance_h": float(checkpoint - prior["cycle_end_ophrs2_h"].max()) if len(prior) else np.nan, "cycle_retention_H": float(prior["cycle_retention_H"].median()) if len(prior) == 3 else np.nan, "polarization_voltage_V": float(pol.iloc[0].voltage_median_V) if len(pol) else np.nan, "polarization_SOH_1_0": float(pol.iloc[0].voltage_median_V / vref) if len(pol) else np.nan})
    aligned = pd.DataFrame(rows); valid = aligned.loc[aligned["status"] == "ALIGNED"].copy()
    required = {1500, 1750}.issubset(set(valid["checkpoint_h"]))
    if len(valid) < 5 or not required:
        return aligned, pd.DataFrame(), False, "ANCHOR_ALIGNMENT_INCONCLUSIVE"
    rho = float(valid["cycle_retention_H"].corr(valid["polarization_SOH_1_0"], method="spearman"))
    noise = float(p1.loc[p1["ageing_hour"] > 0, "voltage_std_V"].median())
    valid = valid.sort_values("checkpoint_h"); pol_delta = np.diff(valid["polarization_voltage_V"]); h_delta = np.diff(valid["cycle_retention_H"])
    informative = np.abs(pol_delta) > noise
    same = np.sign(pol_delta[informative]) == np.sign(h_delta[informative])
    same_fraction = float(np.mean(same)) if len(same) else np.nan
    secondary = polarization.loc[(polarization["jstar_A_cm2"] == 1.6) & (polarization["ageing_hour"] > 0)].copy(); secondary["soh"] = secondary["voltage_median_V"] / float(polarization.loc[(polarization["jstar_A_cm2"] == 1.6) & (polarization["ageing_hour"] == 0), "voltage_median_V"].iloc[0])
    second = valid.merge(secondary[["ageing_hour", "soh"]], left_on="checkpoint_h", right_on="ageing_hour", how="inner")
    rho16 = float(second["cycle_retention_H"].corr(second["soh"], method="spearman")) if len(second) >= 3 else np.nan
    compatibility = pd.DataFrame([{"aligned_non_BoL_anchors": len(valid), "required_1500_1750": required, "rho_H_vs_SOH_pol_1_0": rho, "repeat_noise_V": noise, "informative_transitions": int(informative.sum()), "same_direction_fraction": same_fraction, "rho_H_vs_SOH_pol_1_6": rho16}])
    passed = bool(rho >= .60 and informative.sum() >= 3 and same_fraction >= .75 and rho16 >= -.30)
    return aligned, compatibility, passed, "PASS" if passed else "SPARSE_ANCHOR_COMPATIBILITY_NOT_SUPPORTED"


def final_status(gates: dict[str, bool], drift: pd.DataFrame, anchor_status: str, anchor_pass: bool) -> str:
    if not gates["A"]: return "OPERATION_POINT_SUPPORT_INSUFFICIENT"
    if not (gates["B"] and gates["joint"]): return "CONDITION_COMMON_SUPPORT_INSUFFICIENT"
    raw = bool(drift["raw_negative"].all() and drift["signal_gt_2S"].all() and drift["bootstrap4_upper_negative"].all() and drift["bootstrap8_no_direction_reversal"].all())
    if raw and not drift["matched_negative"].all(): return "VOLTAGE_DRIFT_CONDITION_CONFOUNDED"
    if not raw: return "VOLTAGE_DRIFT_NOT_IDENTIFIABLE"
    if anchor_status == "ANCHOR_ALIGNMENT_INCONCLUSIVE": return anchor_status
    if not anchor_pass: return "SPARSE_ANCHOR_COMPATIBILITY_NOT_SUPPORTED"
    return "RAW_OPERATION_POINT_HEALTH_EVIDENCE_QUALIFIED"


def save_figures(retention: pd.DataFrame, support: pd.DataFrame, drift: pd.DataFrame, matching: pd.DataFrame, alignment: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.5, "axes.spines.top": False, "axes.spines.right": False})
    primary = retention.loc[(retention["stable_window"] == "middle_50") & retention["is_evaluation_cycle"]]
    for point, color in [("P1", BLUE), ("P2", TEAL)]:
        d = primary.loc[primary["point_label"] == point]
        fig, ax = plt.subplots(figsize=(8.2, 4.8)); ax.scatter(d["cycle_end_ophrs2_h"], d["operating_voltage_retention"], s=28, color=color, alpha=.78); ax.set(xlabel="Operating age (h)", ylabel="Cycle operating-voltage retention", title=f"{point} retention across frozen cycles"); ax.grid(alpha=.22); fig.tight_layout(); fig.savefig(FIG / f"{point.lower()}_retention_vs_age.png", dpi=300); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 4.8)); groups=[d.loc[d.age_stratum==s,"operating_voltage_retention"].dropna() for s in ["EARLY","MIDDLE","LATE"]]; ax.boxplot(groups, labels=["EARLY","MIDDLE","LATE"], patch_artist=True, boxprops={"facecolor":color,"alpha":.55}); ax.set(ylabel="Cycle operating-voltage retention", title=f"{point} by Development stratum"); ax.grid(axis="y",alpha=.22); fig.tight_layout(); fig.savefig(FIG / f"{point.lower()}_stratum_distribution.png", dpi=300); plt.close(fig)
    joint = support.loc[support.record_type=="joint_support"]; fig, ax=plt.subplots(figsize=(7,4.8)); pivot=joint.pivot(index="age_stratum",columns="point_label",values="evaluation_cycle_count").reindex(["EARLY","MIDDLE","LATE"]); pivot.plot(kind="bar",ax=ax,color=[BLUE,TEAL]); ax.axhline(15,color=RED,linestyle="--",label="Gate threshold"); ax.set(ylabel="Cycles in 7-variable joint support",title="Condition common-support counts"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.22); fig.tight_layout(); fig.savefig(FIG/"condition_common_support.png",dpi=300);plt.close(fig)
    if not drift.empty:
        fig,ax=plt.subplots(figsize=(7,4.8)); x=np.arange(len(drift)); ax.bar(x-.18,drift.raw_late_minus_early_retention,.36,label="Raw",color=BLUE); ax.bar(x+.18,drift.condition_matched_late_minus_early_retention,.36,label="Condition matched",color=TEAL);ax.axhline(0,color=GREY);ax.set(xticks=x,xticklabels=drift.point_label,ylabel="Late − EARLY retention",title="Raw and condition-matched contrasts");ax.legend(frameon=False);ax.grid(axis="y",alpha=.22);fig.tight_layout();fig.savefig(FIG/"raw_vs_condition_matched_contrast.png",dpi=300);plt.close(fig)
    if not alignment.empty and alignment["cycle_retention_H"].notna().any():
        d=alignment.dropna(subset=["cycle_retention_H","polarization_SOH_1_0"]);fig,ax=plt.subplots(figsize=(6.5,4.8));ax.scatter(d.cycle_retention_H,d.polarization_SOH_1_0,color=VIOLET,s=45);ax.set(xlabel="Cycle operating-voltage retention",ylabel="Polarization normalized voltage, j*=1.0",title="Sparse-anchor compatibility");ax.grid(alpha=.22);fig.tight_layout();fig.savefig(FIG/"cycle_H_vs_polarization_SOH_anchors.png",dpi=300);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.8));ax.plot(d.checkpoint_h,d.cycle_retention_H,marker="o",color=BLUE,label="Cycle retention");ax.plot(d.checkpoint_h,d.polarization_SOH_1_0,marker="s",color=VIOLET,label="Polarization normalized voltage");ax.set(xlabel="Characterization age (h)",ylabel="Normalized representation",title="Pre-characterization cycle-anchor alignment");ax.legend(frameon=False);ax.grid(alpha=.22);fig.tight_layout();fig.savefig(FIG/"anchor_alignment_timeline.png",dpi=300);plt.close(fig)


def write_report(result: dict[str, Any], points: dict[str, Any], access: dict[str, Any], drift: pd.DataFrame, alignment: pd.DataFrame) -> None:
    p = drift.set_index("point_label") if not drift.empty else pd.DataFrame()
    report = f"""# Final raw operation-point health alignment audit

## Frozen scope

RHA reused 146 Phase-RCR complete cycles without cycle rediscovery, anchor
change, DTW, or voltage-based rephasing. Voltage was accessed only after that
identity was frozen. The two selected points were `{points['P1']}` (P1) and
`{points['P2']}` (P2), selected solely by positive-run frequency in the frozen
RCR template.

## Access qualification

Raw post-boundary rows accessed for boundary detection: **{access['raw_post_boundary_rows_accessed']}**. Raw post-boundary rows used: **0**. Test rows were not used for analysis, threshold selection, figures, template, matching, or final decision. Polarization access occurred only after Gates A–C passed: **{result['development_polarization_accessed']}**.

## Result

**Final status: `{result['final_status']}`**

Gate A point support: **{result['gates']['A']}**; Gate B condition completeness: **{result['gates']['B']}**; seven-variable joint support: **{result['gates']['joint']}**; Gate C voltage visibility and matched-direction consistency: **{result['gates']['C']}**; Gate D sparse-anchor compatibility: **{result['gates']['D']}**.

{drift.to_markdown(index=False) if not drift.empty else 'No Gate-C contrast was computed.'}

The cycle representation is named **Cycle Operating-Voltage Retention**. It is not SOH ground truth, and this phase did not construct a continuous SOH label, train a neural model, or train RUL.

## Interpretation boundary

Even a Gate-D pass supports only `SPARSE_POLARIZATION_ANCHOR_COMPATIBILITY` because both representations are age-associated. It is not independent SOH validation. No automatic next phase is created.
"""
    (REPORT / "FINAL_RAW_HEALTH_ALIGNMENT_AUDIT.md").write_text(report,encoding="utf-8")


def run() -> dict[str, Any]:
    ensure_dirs(); provenance=verify_protocol(); template, cycles, states=load_rcr(); points=select_points(template); target=attach_target_runs(cycles,states,points)
    (OUT/"SELECTED_OPERATION_POINTS.json").write_text(json.dumps(points,indent=2),encoding="utf-8")
    measurements, access=collect_window_samples(target); measurements.to_csv(OUT/"CYCLE_POINT_MEASUREMENTS.csv",index=False)
    cycle_points=aggregate_cycle_measurements(measurements); references, retention=choose_references(cycle_points,points); references.to_csv(OUT/"REFERENCE_CYCLES.csv",index=False)
    support,joint,gates=support_tables(retention,cycles); support.to_csv(OUT/"CONDITION_SUPPORT.csv",index=False)
    analysis_ready = gates["A"] and gates["B"] and gates["joint"]
    matching=deterministic_match(joint) if analysis_ready else pd.DataFrame([{"analysis_status":"NOT_RUN","reason":"CONDITION_COMMON_SUPPORT_INSUFFICIENT"}]); matching.to_csv(OUT/"CONDITION_MATCHING.csv",index=False)
    retention.to_csv(OUT/"VOLTAGE_RETENTION_BY_CYCLE.csv",index=False)
    primary=retention.loc[retention.stable_window=="middle_50"].copy(); boot,boot_summary=bootstrap_contrasts(primary) if analysis_ready else (pd.DataFrame(),pd.DataFrame())
    if not boot.empty: boot.to_csv(OUT/"CYCLE_BLOCK_BOOTSTRAP.csv",index=False)
    else: pd.DataFrame(columns=["point_label","block_cycles","iteration","late_minus_early_retention"]).to_csv(OUT/"CYCLE_BLOCK_BOOTSTRAP.csv",index=False)
    drift=drift_contrasts(primary,matching,boot_summary) if not boot.empty else pd.DataFrame()
    if not drift.empty:
        gates["C"] = bool(drift.raw_negative.all() and drift.signal_gt_2S.all() and drift.bootstrap4_upper_negative.all() and drift.bootstrap8_no_direction_reversal.all() and drift.matched_negative.all())
    else: gates["C"] = False
    (drift if not drift.empty else pd.DataFrame([{"analysis_status":"NOT_RUN","reason":"CONDITION_COMMON_SUPPORT_INSUFFICIENT"}])).to_csv(OUT/"VOLTAGE_DRIFT_CONTRAST.csv",index=False)
    polarization=pd.DataFrame(); alignment=pd.DataFrame(); compatibility=pd.DataFrame(); anchor_pass=False; anchor_state="NOT_RUN"
    if gates["A"] and gates["B"] and gates["joint"] and gates["C"]:
        polarization=read_polarization(); alignment,compatibility,anchor_pass,anchor_state=anchor_alignment(retention,polarization)
    (alignment if not alignment.empty else pd.DataFrame([{"analysis_status":"NOT_RUN","reason":"Gate_A_B_C_not_all_pass"}])).to_csv(OUT/"ANCHOR_ALIGNMENT.csv",index=False)
    (compatibility if not compatibility.empty else pd.DataFrame([{"analysis_status":"NOT_RUN","reason":"Gate_A_B_C_not_all_pass"}])).to_csv(OUT/"ANCHOR_COMPATIBILITY.csv",index=False)
    gates["D"]=bool(anchor_pass); status=final_status(gates,drift,anchor_state,anchor_pass)
    access.update({"raw_post_boundary_rows_used":0,"test_raw_rows_accessed_for_boundary_detection":bool(access["raw_post_boundary_rows_accessed"]),"test_raw_rows_used_for_analysis":False,"test_polarization_accessed":False,"test_EIS_accessed":False,"test_used_for_threshold_selection":False,"test_used_for_decision":False,"development_polarization_accessed":bool(not polarization.empty)})
    (OUT/"DATA_ACCESS_AUDIT.json").write_text(json.dumps(access,indent=2),encoding="utf-8")
    result={"final_status":status,"rcr_cycle_status":"RAW_CYCLE_STRUCTURE_QUALIFIED","frozen_cycles_reused":146,"cycle_rediscovery":False,"selected_operation_points":{"P1":points["P1"],"P2":points["P2"]},"gates":gates,"development_polarization_accessed":bool(not polarization.empty),"anchor_gate_state":anchor_state,"neural_training":False,"continuous_SOH_constructed":False,"RUL_trained":False,"automatic_next_phase":False,**provenance,**access}
    (OUT/"GATE_EVALUATION.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (MANIFEST).write_text(yaml.safe_dump({"phase":"Phase_RHA_Raw_Operation_Point_Health_Alignment","status":"EXECUTED","protocol_sha256":provenance["protocol_sha256"],"execution_head":provenance["execution_head"],"frozen_cycles_reused":146,"raw_access":access,"polarization_accessed":bool(not polarization.empty)},sort_keys=False,allow_unicode=True),encoding="utf-8")
    write_report(result,points,access,drift,alignment); (FINAL/"FINAL_DECISION.yaml").write_text(yaml.safe_dump(result,sort_keys=False,allow_unicode=True),encoding="utf-8")
    save_figures(retention,support,drift,matching,alignment)
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(),indent=2,ensure_ascii=False,default=str))
    except AuditFailure as exc:
        ensure_dirs(); result={"final_status":"OPERATION_POINT_SUPPORT_INSUFFICIENT","technical_failure":str(exc),"neural_training":False,"continuous_SOH_constructed":False,"RUL_trained":False,"automatic_next_phase":False}; (OUT/"GATE_EVALUATION.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2)); sys.exit(2)

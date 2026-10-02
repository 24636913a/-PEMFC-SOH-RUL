"""Frozen Phase PAA P1S* pre-anchor audit; no training and no Test access."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
OUT, FIG, REPORT, FINAL = PHASE / "outputs", PHASE / "outputs" / "figures", PHASE / "reports", PHASE / "final"
CONFIG = PHASE / "configs" / "PREREGISTERED_PAA_PROTOCOL.yaml"
CONFIG_HASH = PHASE / "configs" / "PREREGISTERED_PAA_PROTOCOL.sha256"
MANIFEST = PHASE / "SOURCE_MANIFEST.yaml"
RCR = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit"
RHA = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_RHA_Raw_Operation_Point_Health_Alignment"
P1 = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_RHA_P1_Posthoc_Single_Point_Identifiability"
RAW_ROOT = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据")

P1_TOKEN = "221||221.200"
P1_OPERATION_POINT = "221"
P1_SET_CURRENT = 221.200
ANCHOR_PATTERN = ("15||14.000", "15||0.000")
BOUNDARY = 1819.997738
CHUNK = 50_000
STRATA = ["EARLY", "MIDDLE", "LATE"]
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
PALETTE = {"blue": "#0F4D92", "green": "#8BCF8B", "red": "#B64342", "teal": "#42949E", "violet": "#9A4D8E", "neutral": "#CFCECE"}


class AuditFailure(RuntimeError):
    """Raised for fail-closed frozen protocol or firewall violations."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def ensure_dirs() -> None:
    for path in (OUT, FIG, REPORT, FINAL):
        path.mkdir(parents=True, exist_ok=True)


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, str]]:
    expected = CONFIG_HASH.read_text(encoding="utf-8").split()[0]
    observed = sha256(CONFIG)
    if observed != expected:
        raise AuditFailure(f"PAA protocol SHA mismatch: {expected} != {observed}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    bad = {}
    for relative, frozen in config["frozen_input_hashes"].items():
        source = REPO / relative
        actual = sha256(source) if source.is_file() else "MISSING"
        if actual != frozen:
            bad[relative] = {"expected": frozen, "observed": actual}
    if bad:
        raise AuditFailure(f"Frozen input hash mismatch: {bad}")
    if config["analysis_designation"] != "POSTHOC_EXPLORATORY":
        raise AuditFailure("PAA designation is not POSTHOC_EXPLORATORY.")
    if config["P1_state_token_selection"]["primary_token"] != P1_TOKEN:
        raise AuditFailure("Frozen P1S token is not 221||221.200.")
    return config, {"protocol_sha256": observed, "execution_head": git_head()}


def normalize_age(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype("string").str.replace(",", "", regex=False), errors="coerce")


def normalize_operation(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = values.astype("string").str.strip()
    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].map(lambda value: f"{float(value):g}")
    return result.astype(str)


def enforce_raw_firewall(ages: pd.Series, cutoff_h: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Production firewall: Test observations fail closed; cutoff is strict."""
    test = ages >= BOUNDARY
    if bool(test.any()):
        raise AuditFailure("DATA_FIREWALL_DEVIATION: Test raw row was accessed.")
    return ages < cutoff_h, ages > cutoff_h, ages == cutoff_h


def file_order(path: Path) -> tuple[int, str]:
    match = re.search(r"Loadcycle_(\d+)", path.stem)
    if not match:
        raise AuditFailure(f"Unparseable raw filename: {path.name}")
    return int(match.group(1)), path.name


def raw_files_through_cutoff(config: dict[str, Any]) -> list[Path]:
    all_paths = sorted((RAW_ROOT / "Loadcycle_data").rglob("Loadcycle_*.csv"), key=file_order)
    timeline = pd.read_csv(RCR / "outputs" / "RAW_TIMELINE_AUDIT.csv")
    expected = timeline["source_file"].tolist()
    observed = [str(path.relative_to(RAW_ROOT)).replace("\\", "/") for path in all_paths]
    if expected != observed:
        raise AuditFailure("Raw source order differs from frozen RCR timeline audit.")
    maximum = config["raw_firewall"]["maximum_source_file"]
    if maximum not in observed:
        raise AuditFailure("Frozen latest-required source file is unavailable.")
    return all_paths[: observed.index(maximum) + 1]


def select_p1s_token(states: pd.DataFrame, template: dict[str, Any]) -> dict[str, Any]:
    core = states.loc[
        (states["run_id"] >= int(template["template_start_run_id"]))
        & (states["run_id"] < int(template["template_end_run_id_exclusive"]))
        & (states["operation_point"].astype(str) == P1_OPERATION_POINT)
        & (states["stop_or_zero_current_flag"].astype(int) == 0)
        & (states["median_actual_current_A"] > 0)
    ].copy()
    counts = core["state_token"].astype(str).value_counts().to_dict()
    selected = max(counts, key=lambda token: (counts[token], token)) if counts else None
    expected = {P1_TOKEN: 93, "221||156.800": 48, "221||313.600": 25}
    observed = {token: int(counts.get(token, 0)) for token in expected}
    if observed != expected or selected != P1_TOKEN:
        raise AuditFailure(f"Frozen P1S frequency selection mismatch: {observed}, selected={selected}")
    return {"state_token_frequencies": observed, "selected": selected, "selection_used_voltage": False, "selection_used_polarization": False, "selection_used_age_trend": False}


def load_frozen_structure(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rcr_final = yaml.safe_load((RCR / "final" / "FINAL_DECISION.yaml").read_text(encoding="utf-8"))
    rha_final = yaml.safe_load((RHA / "final" / "FINAL_DECISION.yaml").read_text(encoding="utf-8"))
    p1_final = yaml.safe_load((P1 / "final" / "FINAL_DECISION.yaml").read_text(encoding="utf-8"))
    if rcr_final.get("status") != "RAW_CYCLE_STRUCTURE_QUALIFIED":
        raise AuditFailure("RCR no longer has RAW_CYCLE_STRUCTURE_QUALIFIED status.")
    if rha_final.get("final_status") != "CONDITION_COMMON_SUPPORT_INSUFFICIENT":
        raise AuditFailure("Original dual-point RHA result changed.")
    if p1_final.get("final_status") != "P1_ANCHOR_ALIGNMENT_INCONCLUSIVE":
        raise AuditFailure("Original P1 result changed.")
    template = json.loads((RCR / "outputs" / "CYCLE_TEMPLATE.json").read_text(encoding="utf-8"))
    cycles = pd.read_csv(RCR / "outputs" / "RAW_CYCLE_CATALOG.csv")
    states = pd.read_csv(RCR / "outputs" / "RAW_STATE_RUNS.csv")
    complete = cycles.loc[cycles["complete_cycle"].astype(bool)].copy()
    if len(complete) != 146:
        raise AuditFailure(f"Expected 146 frozen complete cycles, found {len(complete)}")
    selection = select_p1s_token(states, template)
    return complete, cycles, states, selection


def target_intervals(complete: pd.DataFrame, cycles: pd.DataFrame, states: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    maximum = float(config["raw_firewall"]["maximum_required_precharacterization_cutoff_h"])
    records: list[dict[str, Any]] = []
    for cycle in complete.loc[complete["end_ophrs2_h"] < maximum].itertuples(index=False):
        subset = states.loc[
            (states["run_id"] >= int(cycle.start_run_id))
            & (states["run_id"] < int(cycle.end_run_id_exclusive))
            & (states["state_token"].astype(str) == P1_TOKEN)
            & (states["duration_wall_s"] >= 120.0)
            & (states["stop_or_zero_current_flag"].astype(int) == 0)
        ]
        for run in subset.itertuples(index=False):
            records.append({"cycle_interval_id": f"COMPLETE_{int(cycle.raw_cycle_id):03d}", "interval_kind": "COMPLETE_CYCLE_INTERVAL", "raw_cycle_id": int(cycle.raw_cycle_id), "age_stratum": str(cycle.age_stratum), "nominal_checkpoint_h": np.nan, "interval_start_ophrs2_h": float(cycle.start_ophrs2_h), "interval_end_ophrs2_h": float(cycle.end_ophrs2_h), **run._asdict()})
    anchors = config["partial_interval_characterization_metadata"]["anchors"]
    for meta in anchors:
        cycle = cycles.loc[cycles["raw_cycle_id"] == int(meta["incomplete_raw_cycle_id"])]
        if len(cycle) != 1 or int(cycle.iloc[0].start_run_id) != int(meta["cycle_start_run_id"]):
            raise AuditFailure(f"Partial interval {meta['checkpoint_h']} lacks its frozen cycle start.")
        first = states.loc[states["run_id"] == int(meta["cycle_start_run_id"])].iloc[0]
        second = states.loc[states["run_id"] == int(meta["cycle_start_run_id"]) + 1].iloc[0]
        if (str(first.state_token), str(second.state_token)) != ANCHOR_PATTERN:
            raise AuditFailure(f"Partial interval {meta['checkpoint_h']} does not start with frozen anchor pattern.")
        cutoff_time = pd.Timestamp(meta["pre_characterization_cutoff_datetime"])
        subset = states.loc[
            (states["run_id"] >= int(meta["cycle_start_run_id"]))
            & (states["run_id"] <= int(meta["pre_characterization_last_run_id"]))
            & (states["state_token"].astype(str) == P1_TOKEN)
            & (states["duration_wall_s"] >= 120.0)
            & (states["stop_or_zero_current_flag"].astype(int) == 0)
        ].copy()
        subset["end_timestamp"] = pd.to_datetime(subset["end_datetime"])
        subset = subset.loc[subset["end_timestamp"] < cutoff_time]
        for run in subset.itertuples(index=False):
            records.append({"cycle_interval_id": f"PARTIAL_PRE_{int(meta['checkpoint_h'])}", "interval_kind": "PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL", "raw_cycle_id": np.nan, "age_stratum": "PREANCHOR_PARTIAL", "nominal_checkpoint_h": int(meta["checkpoint_h"]), "interval_start_ophrs2_h": float(meta["pre_characterization_cutoff_ophrs2_h"]), "interval_end_ophrs2_h": float(meta["pre_characterization_cutoff_ophrs2_h"]), "precharacterization_cutoff_datetime": str(meta["pre_characterization_cutoff_datetime"]), "precharacterization_cutoff_ophrs2_h": float(meta["pre_characterization_cutoff_ophrs2_h"]), **run._asdict()})
    output = pd.DataFrame(records)
    if output.empty:
        raise AuditFailure("No P1S stable dwell runs found in frozen intervals.")
    output["operation_point"] = output["operation_point"].astype(str)
    output["rounded_set_A"] = pd.to_numeric(output["median_set_current_A"], errors="coerce").round(3)
    if not ((output["operation_point"] == P1_OPERATION_POINT) & np.isclose(output["rounded_set_A"], P1_SET_CURRENT)).all():
        raise AuditFailure("P1S target runs violate frozen operation/set-current identity.")
    return output.sort_values("start_datetime").reset_index(drop=True)


def make_windows(target: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind not in {"middle_50", "last_25"}:
        raise ValueError(kind)
    lo, hi = (0.25, 0.75) if kind == "middle_50" else (0.75, 1.0)
    result = target.copy()
    start = pd.to_datetime(result["start_datetime"])
    result["window_start"] = start + pd.to_timedelta(result["duration_wall_s"] * lo, unit="s")
    result["window_end"] = start + pd.to_timedelta(result["duration_wall_s"] * hi, unit="s")
    result["stable_window"] = kind
    return result


def frozen_window_membership(datetimes_ns: np.ndarray, starts_ns: np.ndarray, ends_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.searchsorted(starts_ns, datetimes_ns, side="right") - 1
    valid = np.zeros(len(datetimes_ns), dtype=bool)
    positions = np.flatnonzero(index >= 0)
    valid[positions] = datetimes_ns[positions] <= ends_ns[index[positions]]
    return index, valid


def collect_p1s_raw_measurements(target: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    windows = pd.concat([make_windows(target, "middle_50"), make_windows(target, "last_25")], ignore_index=True).sort_values("window_start").reset_index(drop=True)
    windows["window_id"] = np.arange(len(windows), dtype=int)
    starts = windows["window_start"].to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = windows["window_end"].to_numpy(dtype="datetime64[ns]").astype("int64")
    raw_columns = list(dict.fromkeys([*IDENTITY, *ACTIVE, *COND_RAW.values()]))
    store: dict[int, list[np.ndarray]] = defaultdict(list)
    cutoff = float(config["raw_firewall"]["maximum_required_precharacterization_cutoff_h"])
    files = raw_files_through_cutoff(config)
    access = {"raw_files_opened": [], "maximum_required_precharacterization_cutoff_h": cutoff, "rows_accessed_after_latest_required_anchor": 0, "rows_at_latest_required_anchor": 0, "test_raw_rows_accessed": False, "test_raw_rows_used": False, "stopped_before_later_files": True, "latest_opened_source_file": None}
    stopped = False
    for path in files:
        if stopped:
            break
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        missing = [column for column in raw_columns if column not in header]
        if missing:
            raise AuditFailure(f"Raw schema missing {missing} in {path.name}")
        relative = str(path.relative_to(RAW_ROOT)).replace("\\", "/")
        access["raw_files_opened"].append(relative)
        access["latest_opened_source_file"] = relative
        for chunk in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=raw_columns, chunksize=CHUNK, low_memory=False):
            age = normalize_age(chunk["OpHrs2 [h]"])
            try:
                keep, after, at_cutoff = enforce_raw_firewall(age, cutoff)
            except AuditFailure:
                access["test_raw_rows_accessed"] = True
                raise
            access["rows_accessed_after_latest_required_anchor"] += int(after.sum())
            access["rows_at_latest_required_anchor"] += int(at_cutoff.sum())
            development = chunk.loc[keep].copy()
            development["OpHrs2 [h]"] = age.loc[keep]
            if not development.empty:
                development["DateTime"] = pd.to_datetime(development["DateTime"], errors="coerce")
                development["operation_point"] = normalize_operation(development["operation_point"])
                for column in ["I.S.set [A]", "I.S.act [A]", "j.S.act [A/cm2]", *ACTIVE, *COND_RAW.values()]:
                    development[column] = pd.to_numeric(development[column], errors="coerce")
                development = development.loc[development["DateTime"].notna() & development["OpHrs2 [h]"].notna()].copy()
                if not development.empty:
                    datetimes = development["DateTime"].to_numpy(dtype="datetime64[ns]").astype("int64")
                    index, valid = frozen_window_membership(datetimes, starts, ends)
                    if valid.any():
                        selected = development.loc[valid].copy()
                        selected_index = index[valid]
                        expected_op = windows.iloc[selected_index]["operation_point"].astype(str).to_numpy()
                        expected_set = windows.iloc[selected_index]["rounded_set_A"].to_numpy(dtype=float)
                        actual_set = np.round(selected["I.S.set [A]"].to_numpy(dtype=float), 3)
                        matches = (selected["operation_point"].astype(str).to_numpy() == expected_op) & np.isclose(actual_set, expected_set, equal_nan=False)
                        if not matches.all():
                            raise AuditFailure("Raw row does not match frozen P1S operation/set-current identity.")
                        mean_voltage = np.nanmean(selected[ACTIVE].to_numpy(dtype=float), axis=1)
                        values = np.column_stack([mean_voltage, selected[[COND_RAW[name] for name in COND_RAW]].to_numpy(dtype=float)])
                        for window_id, value in zip(selected_index, values):
                            store[int(window_id)].append(value)
            if bool(after.any() or at_cutoff.any()):
                stopped = True
                break
    records: list[dict[str, Any]] = []
    names = ["P1S_voltage_V", *COND_RAW]
    for window in windows.itertuples(index=False):
        values = np.asarray(store.get(int(window.window_id), []), dtype=float)
        row = {key: getattr(window, key) for key in ["cycle_interval_id", "interval_kind", "raw_cycle_id", "age_stratum", "nominal_checkpoint_h", "run_id", "stable_window", "duration_wall_s", "start_datetime", "end_datetime", "start_ophrs2_h", "end_ophrs2_h"] if hasattr(window, key)}
        row["raw_sample_count"] = int(len(values))
        for position, name in enumerate(names):
            row[name] = float(np.nanmedian(values[:, position])) if len(values) and np.isfinite(values[:, position]).any() else np.nan
        records.append(row)
    return pd.DataFrame(records), access


def interval_measurements(dwell: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    names = ["P1S_voltage_V", *COND_RAW]
    for keys, group in dwell.groupby(["stable_window", "cycle_interval_id", "interval_kind", "raw_cycle_id", "age_stratum", "nominal_checkpoint_h"], dropna=False, sort=True):
        window, interval_id, kind, raw_id, stratum, checkpoint = keys
        row = {"stable_window": window, "cycle_interval_id": interval_id, "interval_kind": kind, "raw_cycle_id": raw_id, "age_stratum": stratum, "nominal_checkpoint_h": checkpoint, "eligible_dwell_count": int(len(group)), "valid_dwell_count": int(group["P1S_voltage_V"].notna().sum()), "measurement_end_ophrs2_h": float(group["end_ophrs2_h"].max()), "measurement_end_datetime": str(pd.to_datetime(group["end_datetime"]).max())}
        for name in names:
            row[name] = float(group[name].median()) if group[name].notna().any() else np.nan
        records.append(row)
    return pd.DataFrame(records)


def add_reference_retention(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = intervals.loc[(intervals["stable_window"] == "middle_50") & (intervals["interval_kind"] == "COMPLETE_CYCLE_INTERVAL") & intervals["P1S_voltage_V"].notna()].sort_values("measurement_end_ophrs2_h").copy()
    if len(primary) < 10:
        raise AuditFailure("Fewer than ten legal complete P1S intervals for reference.")
    ids = set(primary.head(10)["cycle_interval_id"])
    ref_voltage = float(primary.head(10)["P1S_voltage_V"].median())
    result = intervals.copy()
    result["is_reference_complete_interval"] = (result["interval_kind"] == "COMPLETE_CYCLE_INTERVAL") & result["cycle_interval_id"].isin(ids)
    result["is_evaluation_complete_interval"] = (result["interval_kind"] == "COMPLETE_CYCLE_INTERVAL") & ~result["is_reference_complete_interval"] & result["P1S_voltage_V"].notna()
    result["reference_voltage_V"] = ref_voltage
    result["P1S_operating_voltage_retention"] = result["P1S_voltage_V"] / ref_voltage
    reference = result.loc[(result["stable_window"] == "middle_50") & result["is_reference_complete_interval"]].sort_values("measurement_end_ophrs2_h").copy()
    reference["reference_rank"] = np.arange(1, len(reference) + 1)
    return reference, result


def support_audit(retention: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]], bool, bool]:
    conditions = list(config["conditions"])
    cfg = config["support_and_drift"]
    primary = retention.loc[(retention["stable_window"] == "middle_50") & (retention["interval_kind"] == "COMPLETE_CYCLE_INTERVAL")].copy()
    records: list[dict[str, Any]] = []
    ranges: dict[str, tuple[float, float]] = {}
    support_gate = True
    joint_gate = True
    for variable in conditions:
        mins, maxs = [], []
        for stratum in STRATA:
            group = primary.loc[(primary["age_stratum"] == stratum) & primary["is_evaluation_complete_interval"]].copy()
            count = int(group["P1S_voltage_V"].notna().sum())
            complete_rate = float(group[conditions].notna().all(axis=1).mean()) if len(group) else 0.0
            values = group[variable].dropna()
            lo, hi = (float(values.min()), float(values.max())) if len(values) else (np.nan, np.nan)
            mins.append(lo); maxs.append(hi)
            support_gate &= bool(count >= cfg["complete_cycle_measurements_min_per_stratum"] and complete_rate >= cfg["seven_condition_complete_rate_min"])
            records.append({"record_type": "complete_support", "age_stratum": stratum, "variable": variable, "complete_cycle_measurements": count, "condition_complete_rate": complete_rate, "observed_min": lo, "observed_max": hi})
        ranges[variable] = (max(mins), min(maxs))
        records.append({"record_type": "common_range", "age_stratum": "ALL", "variable": variable, "common_min": ranges[variable][0], "common_max": ranges[variable][1], "common_range_nonempty": bool(np.isfinite(ranges[variable][0]) and ranges[variable][0] <= ranges[variable][1])})
    joint = primary.copy()
    joint["in_joint_common_support"] = joint[conditions].notna().all(axis=1) & joint["is_evaluation_complete_interval"]
    for variable, (low, high) in ranges.items():
        joint["in_joint_common_support"] &= joint[variable].between(low, high, inclusive="both")
    for stratum in STRATA:
        count = int(((joint["age_stratum"] == stratum) & joint["in_joint_common_support"]).sum())
        passed = count >= cfg["joint_common_support_min_per_stratum"]
        joint_gate &= passed
        records.append({"record_type": "joint_support", "age_stratum": stratum, "variable": "ALL_7", "joint_complete_cycle_measurements": count, "joint_support_pass_ge15": passed})
    return pd.DataFrame(records), joint, ranges, bool(support_gate), bool(joint_gate)


def contiguous_segments(data: pd.DataFrame) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for _, group in data.groupby("age_stratum", sort=False):
        values: list[float] = []
        prior: int | None = None
        for row in group.sort_values("raw_cycle_id").itertuples(index=False):
            current = int(row.raw_cycle_id)
            if prior is not None and current != prior + 1:
                if values:
                    segments.append(np.asarray(values, dtype=float))
                values = []
            values.append(float(row.P1S_operating_voltage_retention))
            prior = current
        if values:
            segments.append(np.asarray(values, dtype=float))
    return segments


def moving_block_sample(segments: list[np.ndarray], n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    candidates = [segment[index : index + block] for segment in segments for index in range(max(0, len(segment) - block + 1))]
    if not candidates:
        return np.asarray([], dtype=float)
    values: list[float] = []
    while len(values) < n:
        values.extend(candidates[int(rng.integers(len(candidates)))].tolist())
    return np.asarray(values[:n], dtype=float)


def long_term_drift(primary: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    data = primary.loc[primary["is_evaluation_complete_interval"]].sort_values("raw_cycle_id").copy()
    early = data.loc[data["age_stratum"] == "EARLY"].copy()
    late = data.loc[data["age_stratum"] == "LATE"].copy()
    D = float(late["P1S_operating_voltage_retention"].median() - early["P1S_operating_voltage_retention"].median())
    adjacent = []
    for left, right in zip(data.itertuples(index=False), data.iloc[1:].itertuples(index=False)):
        if left.age_stratum == right.age_stratum and int(right.raw_cycle_id) == int(left.raw_cycle_id) + 1:
            adjacent.append(abs(float(right.P1S_operating_voltage_retention - left.P1S_operating_voltage_retention)))
    S = float(np.median(adjacent)) if adjacent else np.nan
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config["support_and_drift"]["bootstrap"]["seed"]))
    for block in [4, 8]:
        values = []
        for iteration in range(int(config["support_and_drift"]["bootstrap"]["iterations"])):
            e = moving_block_sample(contiguous_segments(early), len(early), block, rng)
            l = moving_block_sample(contiguous_segments(late), len(late), block, rng)
            value = float(np.median(l) - np.median(e)) if len(e) and len(l) else np.nan
            values.append(value)
            rows.append({"block_cycles": block, "iteration": iteration, "late_minus_early_retention": value})
    bootstrap = pd.DataFrame(rows)
    summary = bootstrap.groupby("block_cycles")["late_minus_early_retention"].agg(["median", lambda value: value.quantile(.025), lambda value: value.quantile(.975)]).reset_index()
    summary.columns = ["block_cycles", "bootstrap_median", "ci95_lower", "ci95_upper"]
    b4 = summary.loc[summary["block_cycles"] == 4].iloc[0]
    b8 = summary.loc[summary["block_cycles"] == 8].iloc[0]
    result = {"raw_early_late_retention_contrast_D": D, "local_cycle_variation_S": S, "signal_to_local_variation": abs(D) / S if np.isfinite(S) and S > 0 else np.nan, "early_complete_evaluation_intervals": int(len(early)), "late_complete_evaluation_intervals": int(len(late)), "adjacent_within_stratum_pairs": int(len(adjacent)), "bootstrap4_ci95_lower": float(b4.ci95_lower), "bootstrap4_ci95_upper": float(b4.ci95_upper), "bootstrap8_median": float(b8.bootstrap_median), "raw_negative": bool(D < 0), "absolute_D_gt_2S": bool(np.isfinite(S) and abs(D) > 2 * S), "bootstrap4_upper_negative": bool(b4.ci95_upper < 0), "bootstrap8_negative": bool(b8.bootstrap_median < 0)}
    return result, bootstrap


def deterministic_match(primary: pd.DataFrame, joint: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    conditions = list(config["conditions"])
    early = joint.loc[(joint["age_stratum"] == "EARLY") & joint["in_joint_common_support"]].copy()
    late = joint.loc[(joint["age_stratum"] == "LATE") & joint["in_joint_common_support"]].copy()
    scale_source = primary.loc[primary["is_reference_complete_interval"] | (primary["age_stratum"] == "EARLY"), conditions].copy()
    med = scale_source.median(); iqr = scale_source.quantile(.75) - scale_source.quantile(.25); std = scale_source.std(ddof=1)
    scale = iqr.where(iqr > 0, std).replace(0, np.nan)
    variables = scale.dropna().index.tolist()
    if early.empty or late.empty or not variables:
        return pd.DataFrame(), pd.DataFrame(), {"matched_pair_count": 0, "matched_contrast": np.nan, "reason": "NO_MATCHABLE_INTERVALS_OR_VARIABLES"}
    candidates = []
    for early_row in early.itertuples(index=False):
        e = np.asarray([getattr(early_row, variable) for variable in variables], dtype=float)
        for late_row in late.itertuples(index=False):
            l = np.asarray([getattr(late_row, variable) for variable in variables], dtype=float)
            distance = float(np.sqrt(np.sum(((e - l) / scale[variables].to_numpy(dtype=float)) ** 2)))
            candidates.append((distance, str(early_row.cycle_interval_id), str(late_row.cycle_interval_id)))
    e_lookup, l_lookup = early.set_index("cycle_interval_id"), late.set_index("cycle_interval_id")
    used_e, used_l, records = set(), set(), []
    for distance, e_id, l_id in sorted(candidates):
        if e_id in used_e or l_id in used_l:
            continue
        used_e.add(e_id); used_l.add(l_id)
        e, l = e_lookup.loc[e_id], l_lookup.loc[l_id]
        row = {"early_cycle_interval_id": e_id, "late_cycle_interval_id": l_id, "condition_distance": distance, "early_retention": float(e.P1S_operating_voltage_retention), "late_retention": float(l.P1S_operating_voltage_retention), "late_minus_early_retention": float(l.P1S_operating_voltage_retention - e.P1S_operating_voltage_retention), "matching_variables": "|".join(variables)}
        for variable in variables:
            row[f"early_{variable}"] = float(e[variable]); row[f"late_{variable}"] = float(l[variable])
        records.append(row)
    matches = pd.DataFrame(records)
    balance = []
    for variable in variables:
        e = matches[f"early_{variable}"]; l = matches[f"late_{variable}"]
        pooled = math.sqrt((float(e.var(ddof=1)) + float(l.var(ddof=1))) / 2) if len(e) > 1 else np.nan
        balance.append({"condition_variable": variable, "early_matched_median": float(e.median()), "late_matched_median": float(l.median()), "standardized_median_difference": float((l.median() - e.median()) / scale[variable]), "standardized_mean_difference": float((l.mean() - e.mean()) / pooled) if np.isfinite(pooled) and pooled > 0 else np.nan, "matching_scale_IQR_or_std": float(scale[variable])})
    distance = matches["condition_distance"] if not matches.empty else pd.Series(dtype=float)
    summary = {"matched_pair_count": int(len(matches)), "matched_contrast": float(matches["late_minus_early_retention"].median()) if len(matches) else np.nan, "distance_median": float(distance.median()) if len(distance) else np.nan, "distance_p75": float(distance.quantile(.75)) if len(distance) else np.nan, "distance_p90": float(distance.quantile(.90)) if len(distance) else np.nan, "distance_max": float(distance.max()) if len(distance) else np.nan, "matching_variables": "|".join(variables), "constant_variables_excluded": "|".join([item for item in conditions if item not in variables])}
    return matches, pd.DataFrame(balance), summary


def anchor_alignment(retention: pd.DataFrame, ranges: dict[str, tuple[float, float]], config: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    conditions = list(config["conditions"])
    data = retention.loc[retention["stable_window"] == "middle_50"].copy()
    data["in_development_joint_common_support"] = data[conditions].notna().all(axis=1)
    for variable, (low, high) in ranges.items():
        data["in_development_joint_common_support"] &= data[variable].between(low, high, inclusive="both")
    records: list[dict[str, Any]] = []
    for meta in config["partial_interval_characterization_metadata"]["anchors"]:
        checkpoint = int(meta["checkpoint_h"]); cutoff = float(meta["pre_characterization_cutoff_ophrs2_h"]); cutoff_time = pd.Timestamp(meta["pre_characterization_cutoff_datetime"])
        candidate = data.loc[(data["measurement_end_ophrs2_h"] >= cutoff - 48.0) & (data["measurement_end_ophrs2_h"] < cutoff) & (pd.to_datetime(data["measurement_end_datetime"]) < cutoff_time) & data[conditions].notna().all(axis=1)].sort_values("measurement_end_ophrs2_h", ascending=False).drop_duplicates("cycle_interval_id").head(3).copy()
        support_count = int(candidate["in_development_joint_common_support"].sum())
        aligned = bool(len(candidate) == 3 and support_count >= 2)
        records.append({"record_type": "anchor_summary", "checkpoint_h": checkpoint, "precharacterization_cutoff_h": cutoff, "eligible_interval_count_within_48h": int(len(candidate)), "selected_cycle_interval_ids": "|".join(candidate["cycle_interval_id"].astype(str)), "selected_interval_kinds": "|".join(candidate["interval_kind"].astype(str)), "joint_support_interval_count": support_count, "status": "ALIGNED" if aligned else "UNRESOLVED", "P1S_anchor_retention": float(candidate["P1S_operating_voltage_retention"].median()) if aligned else np.nan})
        for order, row in enumerate(candidate.itertuples(index=False), start=1):
            output = {"record_type": "anchor_interval", "checkpoint_h": checkpoint, "selection_rank": order, "cycle_interval_id": row.cycle_interval_id, "interval_kind": row.interval_kind, "measurement_end_ophrs2_h": row.measurement_end_ophrs2_h, "P1S_operating_voltage_retention": row.P1S_operating_voltage_retention, "in_development_joint_common_support": bool(row.in_development_joint_common_support)}
            for variable in conditions:
                output[variable] = getattr(row, variable)
            records.append(output)
    alignment = pd.DataFrame(records)
    summary = alignment.loc[alignment["record_type"] == "anchor_summary"].copy()
    required = {1500, 1750}.issubset(set(summary.loc[summary["status"] == "ALIGNED", "checkpoint_h"]))
    coverage = bool(int((summary["status"] == "ALIGNED").sum()) >= 5 and required)
    return alignment, coverage


def parse_checkpoint(path: Path) -> tuple[str, float, bool]:
    lowered = path.stem.lower()
    after = "afterairair" in lowered
    if "bol" in lowered:
        return "BoL", 0.0, after
    match = re.search(r"(\d+)h", lowered)
    if not match:
        raise AuditFailure(f"Cannot parse polarization checkpoint: {path.name}")
    return f"{int(match.group(1))}h", float(match.group(1)), after


def validate_polarization_path(path: Path) -> tuple[str, float]:
    """Reject every Test or post-Air-Air polarization path before reading it."""
    stage, age, after = parse_checkpoint(path)
    if after or age not in {0.0, 300.0, 600.0, 850.0, 1150.0, 1500.0, 1750.0}:
        raise AuditFailure(f"Forbidden polarization path: {path.name}")
    return stage, age


def allowed_polarization_paths() -> list[Path]:
    allowed = {0.0, 300.0, 600.0, 850.0, 1150.0, 1500.0, 1750.0}
    paths = []
    for path in RAW_ROOT.glob("Polcurve_Nominal_RHFC_*.csv"):
        _, age, after = parse_checkpoint(path)
        if age in allowed and not after:
            validate_polarization_path(path)
            paths.append(path)
    if len(paths) != 7:
        raise AuditFailure(f"Expected seven allowlisted Development polarization files, found {len(paths)}")
    return sorted(paths, key=lambda item: parse_checkpoint(item)[1])


def read_allowlisted_polarization() -> pd.DataFrame:
    records = []
    cols = list(dict.fromkeys(["Number of cells", "j.S.act [A/cm2]", *ACTIVE]))
    for path in allowed_polarization_paths():
        stage, age, _ = parse_checkpoint(path)
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        if any(column not in header for column in cols):
            raise AuditFailure(f"Polarization schema mismatch in {path.name}")
        by_density = {1.0: [], 1.6: []}; cells = []
        for chunk in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=cols, chunksize=CHUNK, low_memory=False):
            for column in cols:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            cells.extend(chunk["Number of cells"].dropna().unique().tolist())
            voltage = np.nanmean(chunk[ACTIVE].to_numpy(dtype=float), axis=1)
            density = chunk["j.S.act [A/cm2]"].to_numpy(dtype=float)
            for jstar in by_density:
                by_density[jstar].extend(voltage[np.isfinite(density) & (np.abs(density - jstar) <= .01)].tolist())
        if set(np.round(cells, 6)) != {10.0}:
            raise AuditFailure(f"{path.name} is not a 10-cell nominal durability stack")
        for jstar, values in by_density.items():
            records.append({"checkpoint": stage, "checkpoint_h": age, "jstar_A_cm2": jstar, "voltage_median_V": float(np.median(values)) if values else np.nan, "voltage_std_V": float(np.std(values, ddof=0)) if values else np.nan, "valid_sample_count": int(len(values)), "source_file": path.name})
    return pd.DataFrame(records)


def anchor_compatibility(alignment: pd.DataFrame, polarization: pd.DataFrame) -> pd.DataFrame:
    summary = alignment.loc[(alignment["record_type"] == "anchor_summary") & (alignment["status"] == "ALIGNED")].copy()
    primary = polarization.loc[polarization["jstar_A_cm2"] == 1.0].copy()
    reference = float(primary.loc[primary["checkpoint_h"] == 0, "voltage_median_V"].iloc[0])
    summary = summary.merge(primary[["checkpoint_h", "voltage_median_V", "voltage_std_V"]], on="checkpoint_h", how="inner")
    summary["SOH_pol_1_0"] = summary["voltage_median_V"] / reference
    rho = float(summary["P1S_anchor_retention"].corr(summary["SOH_pol_1_0"], method="spearman"))
    noise = float(primary.loc[primary["checkpoint_h"] > 0, "voltage_std_V"].median())
    ordered = summary.sort_values("checkpoint_h")
    pol_delta = np.diff(ordered["voltage_median_V"]); p1s_delta = np.diff(ordered["P1S_anchor_retention"]); informative = np.abs(pol_delta) > noise
    same = float(np.mean(np.sign(pol_delta[informative]) == np.sign(p1s_delta[informative]))) if informative.any() else np.nan
    secondary = polarization.loc[polarization["jstar_A_cm2"] == 1.6].copy(); secondary_ref = float(secondary.loc[secondary["checkpoint_h"] == 0, "voltage_median_V"].iloc[0]); secondary["SOH_pol_1_6"] = secondary["voltage_median_V"] / secondary_ref
    merged = ordered.merge(secondary[["checkpoint_h", "SOH_pol_1_6"]], on="checkpoint_h", how="inner")
    rho16 = float(merged["P1S_anchor_retention"].corr(merged["SOH_pol_1_6"], method="spearman"))
    passed = bool(rho >= .60 and rho > 0 and informative.sum() >= 3 and same >= .75 and rho16 >= -.30)
    return pd.DataFrame([{"rho_P1S_vs_SOH_pol_1_0": rho, "repeat_noise_V": noise, "informative_transitions": int(informative.sum()), "same_direction_fraction": same, "rho_P1S_vs_SOH_pol_1_6": rho16, "compatibility_pass": passed}])


def placeholder(path: Path, title: str, message: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5)); axis.axis("off"); axis.set_title(title); axis.text(.5, .5, message, ha="center", va="center", wrap=True, fontsize=14); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def save_figures(retention: pd.DataFrame, drift: dict[str, Any], matches: pd.DataFrame, balance: pd.DataFrame, alignment: pd.DataFrame, compatibility: pd.DataFrame, polarization: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.8, "axes.spines.top": False, "axes.spines.right": False})
    primary = retention.loc[(retention["stable_window"] == "middle_50") & (retention["interval_kind"] == "COMPLETE_CYCLE_INTERVAL") & retention["is_evaluation_complete_interval"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.8)); ax.scatter(primary["measurement_end_ophrs2_h"], primary["P1S_operating_voltage_retention"], color=PALETTE["blue"], s=26, alpha=.8); ax.set(xlabel="Operating age (h)", ylabel="P1S operating-voltage retention", title="P1S retention across complete Development cycles"); ax.grid(alpha=.22); fig.tight_layout(); fig.savefig(FIG / "p1s_retention_vs_age.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.8)); groups=[primary.loc[primary.age_stratum==stratum,"P1S_operating_voltage_retention"].dropna() for stratum in STRATA]; ax.boxplot(groups, labels=STRATA, patch_artist=True, boxprops={"facecolor":PALETTE["blue"],"alpha":.55}); ax.set(ylabel="P1S operating-voltage retention", title="P1S retention by age stratum"); ax.grid(axis="y",alpha=.22); fig.tight_layout(); fig.savefig(FIG / "p1s_stratum_distributions.png", dpi=300); plt.close(fig)
    if matches.empty:
        placeholder(FIG / "raw_vs_matched_early_late_contrast.png", "P1S raw vs matched contrast", "Condition matching was not run.")
    else:
        raw=float(drift["raw_early_late_retention_contrast_D"]); matched=float(matches["late_minus_early_retention"].median()); fig, ax=plt.subplots(figsize=(6.6,4.8)); ax.bar(["Raw","Condition matched"],[raw,matched],color=[PALETTE["blue"],PALETTE["teal"]]);ax.axhline(0,color="black",linewidth=1);ax.set(ylabel="LATE − EARLY retention",title="P1S early–late contrast");ax.grid(axis="y",alpha=.22);fig.tight_layout();fig.savefig(FIG/"raw_vs_matched_early_late_contrast.png",dpi=300);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4.8));ax.hist(matches["condition_distance"],bins=min(12,max(4,len(matches)//2)),color=PALETTE["teal"],edgecolor="white");ax.set(xlabel="Matching distance",ylabel="Matched pairs",title="P1S condition-matching distance distribution");ax.grid(axis="y",alpha=.22);fig.tight_layout();fig.savefig(FIG/"matching_distance_distribution.png",dpi=300);plt.close(fig)
    if matches.empty:
        placeholder(FIG / "matching_distance_distribution.png", "P1S matching distance", "Condition matching was not run.")
    if balance.empty:
        placeholder(FIG / "matching_balance_by_condition.png", "P1S matching balance", "Condition matching was not run.")
    else:
        fig,ax=plt.subplots(figsize=(8,5.2)); y=np.arange(len(balance)); ax.barh(y,balance["standardized_mean_difference"],color=PALETTE["violet"]);ax.axvline(0,color="black",linewidth=1);ax.set(yticks=y,yticklabels=balance["condition_variable"],xlabel="Standardized mean difference",title="P1S matched-condition balance audit");ax.grid(axis="x",alpha=.22);fig.tight_layout();fig.savefig(FIG/"matching_balance_by_condition.png",dpi=300);plt.close(fig)
    summary=alignment.loc[alignment["record_type"]=="anchor_summary"] if not alignment.empty else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(8.2,4.8));
    if not summary.empty:
        colors=[PALETTE["green"] if status=="ALIGNED" else PALETTE["red"] for status in summary["status"]];ax.scatter(summary["checkpoint_h"],summary["P1S_anchor_retention"],c=colors,s=70);[ax.axvline(value,color=PALETTE["neutral"],linewidth=.8,zorder=0) for value in summary["checkpoint_h"]];ax.set(xlabel="Nominal polarization checkpoint (h)",ylabel="P1S anchor retention",title="P1S pre-anchor timeline (green: aligned)");ax.grid(alpha=.22)
    else: ax.axis("off");ax.set_title("P1S pre-anchor timeline");ax.text(.5,.5,"Anchor alignment was not run.",ha="center")
    fig.tight_layout();fig.savefig(FIG/"preanchor_checkpoint_timeline.png",dpi=300);plt.close(fig)
    intervals=retention.loc[retention["stable_window"]=="middle_50"];fig,ax=plt.subplots(figsize=(8.2,4.8));
    for kind,color in [("COMPLETE_CYCLE_INTERVAL",PALETTE["blue"]),("PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL",PALETTE["teal"])]:
        d=intervals.loc[intervals.interval_kind==kind];ax.scatter(d["measurement_end_ophrs2_h"],d["P1S_operating_voltage_retention"],s=27,label=kind.replace("_"," "),color=color)
    ax.set(xlabel="Measurement end operating age (h)",ylabel="P1S retention",title="Complete and pre-characterization partial intervals");ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.22);fig.tight_layout();fig.savefig(FIG/"complete_vs_partial_preanchor_intervals.png",dpi=300);plt.close(fig)
    if compatibility.empty:
        placeholder(FIG / "p1s_anchor_retention_vs_polarization_soh.png", "P1S anchor compatibility", "Coverage gate did not permit polarization compatibility computation.")
    else:
        anchor=summary.loc[summary["status"]=="ALIGNED"].copy(); pol=polarization.loc[polarization["jstar_A_cm2"]==1.0].copy(); ref=float(pol.loc[pol["checkpoint_h"]==0,"voltage_median_V"].iloc[0]); anchor=anchor.merge(pol[["checkpoint_h","voltage_median_V"]],on="checkpoint_h",how="inner");anchor["SOH_pol_1_0"]=anchor["voltage_median_V"]/ref;fig,ax=plt.subplots(figsize=(6.6,4.8));ax.scatter(anchor["P1S_anchor_retention"],anchor["SOH_pol_1_0"],s=55,color=PALETTE["violet"]);ax.set(xlabel="P1S anchor retention",ylabel="Polarization normalized voltage, j*=1.0",title="P1S sparse-anchor compatibility");ax.grid(alpha=.22);fig.tight_layout();fig.savefig(FIG/"p1s_anchor_retention_vs_polarization_soh.png",dpi=300);plt.close(fig)


def final_status(gate_support: bool, gate_drift: bool, gate_joint: bool, gate_match: bool, coverage: bool, compatibility: pd.DataFrame) -> str:
    if not gate_support:
        return "P1S_SUPPORT_INSUFFICIENT"
    if not gate_drift:
        return "P1S_LONG_TERM_DRIFT_NOT_IDENTIFIABLE"
    if not gate_joint:
        return "P1S_CONDITION_SUPPORT_INSUFFICIENT"
    if not gate_match:
        return "P1S_DRIFT_CONDITION_CONFOUNDED"
    if not coverage:
        return "P1S_ANCHOR_ALIGNMENT_INCONCLUSIVE"
    if compatibility.empty or not bool(compatibility.iloc[0]["compatibility_pass"]):
        return "P1S_SPARSE_ANCHOR_COMPATIBILITY_NOT_SUPPORTED"
    return "P1S_PREANCHOR_COMPATIBILITY_EXPLORATORY_PASS"


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    """Render a small report table without optional tabulate dependency."""
    if data.empty:
        return "Anchor alignment was not reached."
    table = data[columns].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in table.itertuples(index=False, name=None):
        rendered = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(result: dict[str, Any], drift: dict[str, Any], match: dict[str, Any], alignment: pd.DataFrame) -> None:
    summary = alignment.loc[alignment["record_type"] == "anchor_summary"] if not alignment.empty else pd.DataFrame()
    text = f"""# Final P1S pre-anchor alignment audit

## Scope

This is a **POSTHOC_EXPLORATORY** audit of the fixed state token
`221||221.200`. It does not change Phase RCR, dual-point RHA, or P1 results.
The representation is **P1S Operating-Voltage Retention**, not SOH.

## Firewall

Test raw rows accessed: **{result['test_raw_rows_accessed']}**. Rows accessed
after the latest required pre-characterization anchor cutoff:
**{result['rows_accessed_after_latest_required_anchor']}**. Test polarization
and EIS accessed: **False**. No source file after the frozen
`Loadcycle_134-150.csv` cutoff was opened.

## Final result

**`{result['final_status']}`**

P1S support: **{result['gates']['support']}**; long-term drift:
**{result['gates']['long_term_drift']}**; joint condition support:
**{result['gates']['condition_support']}**; matched direction:
**{result['gates']['matched_direction']}**; anchor coverage:
**{result['gates']['anchor_coverage']}**.

| Quantity | Value |
| --- | ---: |
| Raw EARLY–LATE contrast | {drift.get('raw_early_late_retention_contrast_D', np.nan):.8f} |
| Local variation | {drift.get('local_cycle_variation_S', np.nan):.8f} |
| 4-cycle CI upper | {drift.get('bootstrap4_ci95_upper', np.nan):.8f} |
| Matched contrast | {match.get('matched_contrast', np.nan):.8f} |

{markdown_table(summary, ['checkpoint_h', 'selected_cycle_interval_ids', 'joint_support_interval_count', 'status', 'P1S_anchor_retention'])}

No neural model, continuous SOH, RUL model, or automatic next phase was created.
"""
    (REPORT / "FINAL_P1S_PREANCHOR_AUDIT.md").write_text(text, encoding="utf-8")


def run() -> dict[str, Any]:
    ensure_dirs(); config, provenance = verify_frozen_inputs(); complete, cycles, states, selection = load_frozen_structure(config)
    (OUT / "P1S_SELECTION_AUDIT.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    target = target_intervals(complete, cycles, states, config)
    dwell, access = collect_p1s_raw_measurements(target, config)
    intervals = interval_measurements(dwell)
    reference, retention = add_reference_retention(intervals)
    support, joint, ranges, gate_support, gate_joint = support_audit(retention, config)
    primary = retention.loc[(retention["stable_window"] == "middle_50") & (retention["interval_kind"] == "COMPLETE_CYCLE_INTERVAL")].copy()
    drift, bootstrap = long_term_drift(primary, config)
    gate_drift = bool(gate_support and drift["raw_negative"] and drift["absolute_D_gt_2S"] and drift["bootstrap4_upper_negative"] and drift["bootstrap8_negative"])
    matches, balance, match_summary = deterministic_match(primary, joint, config) if gate_drift and gate_joint else (pd.DataFrame(), pd.DataFrame(), {"matched_pair_count": 0, "matched_contrast": np.nan, "reason": "UPSTREAM_GATE_NOT_PASS"})
    gate_match = bool(gate_drift and gate_joint and np.isfinite(match_summary["matched_contrast"]) and match_summary["matched_contrast"] < 0)
    partial = retention.loc[(retention["stable_window"] == "middle_50") & (retention["interval_kind"] == "PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL")].copy()
    alignment, coverage = anchor_alignment(retention, ranges, config) if gate_match else (pd.DataFrame(), False)
    polarization = pd.DataFrame(); compatibility = pd.DataFrame()
    if gate_match and coverage:
        polarization = read_allowlisted_polarization()
        compatibility = anchor_compatibility(alignment, polarization)
    status = final_status(gate_support, gate_drift, gate_joint, gate_match, coverage, compatibility)
    access.update({"test_raw_rows_used_for_analysis": False, "test_raw_rows_used_for_threshold_selection": False, "test_raw_rows_used_for_decision": False, "test_polarization_accessed": False, "test_EIS_accessed": False, "development_polarization_accessed": bool(not polarization.empty)})
    reference.to_csv(OUT / "P1S_REFERENCE_COMPLETE_CYCLES.csv", index=False)
    retention.loc[retention["interval_kind"] == "COMPLETE_CYCLE_INTERVAL"].to_csv(OUT / "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv", index=False)
    support.to_csv(OUT / "P1S_SUPPORT_AUDIT.csv", index=False)
    pd.DataFrame([drift]).to_csv(OUT / "P1S_LONG_TERM_DRIFT.csv", index=False)
    bootstrap.to_csv(OUT / "P1S_BOOTSTRAP.csv", index=False)
    support.to_csv(OUT / "P1S_CONDITION_SUPPORT.csv", index=False)
    (matches if not matches.empty else pd.DataFrame([{"analysis_status": "NOT_RUN", "reason": match_summary.get("reason", "NO_MATCHES")}])).to_csv(OUT / "P1S_CONDITION_MATCHING.csv", index=False)
    balance.to_csv(OUT / "P1S_MATCHING_BALANCE.csv", index=False)
    partial.to_csv(OUT / "PREANCHOR_PARTIAL_CYCLE_INTERVALS.csv", index=False)
    alignment.to_csv(OUT / "P1S_ANCHOR_ALIGNMENT.csv", index=False)
    if coverage:
        compatibility.to_csv(OUT / "P1S_ANCHOR_COMPATIBILITY.csv", index=False)
    (OUT / "DATA_ACCESS_AUDIT.json").write_text(json.dumps(access, indent=2), encoding="utf-8")
    result = {"final_status": status, "analysis_designation": "POSTHOC_EXPLORATORY", "base_head": config["base_head"], "original_P1_result_changed": False, "primary_operation_point": P1_OPERATION_POINT, "frozen_P1_state_token": P1_TOKEN, "frozen_template_frequency": 93, "selection_based_on_voltage": False, "selection_based_on_polarization": False, "reference_complete_cycles": 10, "gates": {"support": gate_support, "long_term_drift": gate_drift, "condition_support": gate_joint, "matched_direction": gate_match, "anchor_coverage": coverage}, "neural_training": False, "continuous_SOH_constructed": False, "RUL_trained": False, "automatic_next_phase": False, **provenance, **access}
    (OUT / "GATE_EVALUATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    MANIFEST.write_text(yaml.safe_dump({"phase": PHASE.name, "status": "EXECUTED", "analysis_designation": "POSTHOC_EXPLORATORY", "protocol_sha256": provenance["protocol_sha256"], "execution_head": provenance["execution_head"], "selected_state_token": P1_TOKEN, "raw_access": access, "polarization_accessed": bool(not polarization.empty)}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_report(result, drift, match_summary, alignment)
    (FINAL / "FINAL_DECISION.yaml").write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8")
    save_figures(retention, drift, matches, balance, alignment, compatibility, polarization)
    return result


def finalize_existing_outputs() -> dict[str, Any]:
    """Repair an interrupted post-processing step without touching source data."""
    ensure_dirs()
    result = json.loads((OUT / "GATE_EVALUATION.json").read_text(encoding="utf-8"))
    if result.get("final_status") == "DATA_FIREWALL_DEVIATION":
        raise AuditFailure("Cannot finalize an audit that failed the Test firewall.")
    drift = pd.read_csv(OUT / "P1S_LONG_TERM_DRIFT.csv").iloc[0].to_dict()
    match = pd.read_csv(OUT / "P1S_CONDITION_MATCHING.csv")
    match_summary = {"matched_contrast": float(match["late_minus_early_retention"].median()) if "late_minus_early_retention" in match else np.nan}
    alignment = pd.read_csv(OUT / "P1S_ANCHOR_ALIGNMENT.csv")
    write_report(result, drift, match_summary, alignment)
    (FINAL / "FINAL_DECISION.yaml").write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    try:
        result = finalize_existing_outputs() if "--finalize-existing" in sys.argv[1:] else run()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    except AuditFailure as error:
        ensure_dirs(); failure={"final_status": "DATA_FIREWALL_DEVIATION" if "DATA_FIREWALL_DEVIATION" in str(error) else "P1S_SUPPORT_INSUFFICIENT", "technical_failure": str(error), "neural_training": False, "continuous_SOH_constructed": False, "RUL_trained": False, "automatic_next_phase": False}; (OUT / "GATE_EVALUATION.json").write_text(json.dumps(failure, indent=2), encoding="utf-8"); print(json.dumps(failure, indent=2, ensure_ascii=False)); sys.exit(2)

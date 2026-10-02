"""Frozen, zero-training P1-only posthoc exploratory identifiability audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
OUT, REPORT, FINAL = PHASE / "outputs", PHASE / "reports", PHASE / "final"
CONFIG = PHASE / "configs" / "PREREGISTERED_P1_EXPLORATORY_PROTOCOL.yaml"
CONFIG_HASH = PHASE / "configs" / "PREREGISTERED_P1_EXPLORATORY_PROTOCOL.sha256"
MANIFEST = PHASE / "SOURCE_MANIFEST.yaml"
RHA = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_RHA_Raw_Operation_Point_Health_Alignment"
RCR = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit"
RHA_SCRIPT = RHA / "scripts" / "run_raw_operation_point_health_audit.py"
P1, P2 = "221", "30"
STRATA = ["EARLY", "MIDDLE", "LATE"]


class AuditFailure(RuntimeError):
    """Raised for a frozen-input or protocol violation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def ensure_dirs() -> None:
    for path in (OUT, REPORT, FINAL):
        path.mkdir(parents=True, exist_ok=True)


def verify_protocol_and_inputs() -> tuple[dict[str, Any], dict[str, str]]:
    expected = CONFIG_HASH.read_text(encoding="utf-8").split()[0]
    observed = sha256(CONFIG)
    if expected != observed:
        raise AuditFailure(f"P1 protocol SHA mismatch: {expected} != {observed}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["analysis_designation"] != "POSTHOC_EXPLORATORY":
        raise AuditFailure("P1 audit designation is not POSTHOC_EXPLORATORY.")
    if str(config["primary_operation_point"]["operation_point"]) != P1:
        raise AuditFailure("Frozen P1 operation point is not 221.")
    if bool(config["secondary_operation_point"]["replaced"]):
        raise AuditFailure("P2 replacement is prohibited.")
    bad: dict[str, tuple[str, str]] = {}
    for relative, frozen in config["frozen_input_hashes"].items():
        source = REPO / relative
        actual = sha256(source) if source.is_file() else "MISSING"
        if actual != frozen:
            bad[relative] = (frozen, actual)
    if bad:
        raise AuditFailure(f"Frozen input hash mismatch: {bad}")
    return config, {"protocol_sha256": observed, "execution_head": git_head()}


def load_rha_module() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_rha", RHA_SCRIPT)
    if spec is None or spec.loader is None:
        raise AuditFailure("Cannot import the frozen RHA implementation.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_cycle_inputs(rha: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    template, cycles, states = rha.load_rcr()
    old_final = yaml.safe_load((RHA / "final" / "FINAL_DECISION.yaml").read_text(encoding="utf-8"))
    if old_final.get("final_status") != "CONDITION_COMMON_SUPPORT_INSUFFICIENT":
        raise AuditFailure("Original dual-point RHA status has changed.")
    selected = json.loads((RHA / "outputs" / "SELECTED_OPERATION_POINTS.json").read_text(encoding="utf-8"))
    reselected = rha.select_points(template)
    if selected.get("P1") != P1 or selected.get("P2") != P2:
        raise AuditFailure("Original RHA selected operation points are not P1=221/P2=30.")
    if reselected["P1"] != P1 or reselected["P2"] != P2:
        raise AuditFailure("Frozen RCR positive-run ranking no longer reproduces P1/P2.")
    target = rha.attach_target_runs(cycles, states, {"P1": P1, "P2": P2})
    target = target.loc[target["point_label"] == "P1"].copy()
    if target.empty or set(target["operation_point"].astype(str)) != {P1}:
        raise AuditFailure("P1 target-run identity cannot be reconstructed from frozen RCR inputs.")
    return cycles, states, target


def choose_p1_references(cycle_points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    references: list[dict[str, Any]] = []
    retained: list[pd.DataFrame] = []
    for window in ["middle_50", "last_25"]:
        data = cycle_points.loc[
            (cycle_points["stable_window"] == window) & cycle_points["cell_voltage_mean_V"].notna()
        ].sort_values("cycle_start_ophrs2_h").copy()
        if len(data) < 10:
            raise AuditFailure(f"P1 has fewer than ten legal cycles in {window} reference selection.")
        ref = data.head(10)
        ref_ids = set(ref["raw_cycle_id"].astype(int))
        ref_voltage = float(np.median(ref["cell_voltage_mean_V"]))
        data["is_reference_cycle"] = data["raw_cycle_id"].astype(int).isin(ref_ids)
        data["is_evaluation_cycle"] = ~data["is_reference_cycle"]
        data["reference_voltage_V"] = ref_voltage
        data["P1_cycle_operating_voltage_retention"] = data["cell_voltage_mean_V"] / ref_voltage
        retained.append(data)
        for rank, row in enumerate(ref.itertuples(index=False), start=1):
            references.append(
                {
                    "stable_window": window,
                    "reference_rank": rank,
                    "raw_cycle_id": int(row.raw_cycle_id),
                    "cycle_end_ophrs2_h": float(row.cycle_end_ophrs2_h),
                    "reference_voltage_V": ref_voltage,
                }
            )
    return pd.DataFrame(references), pd.concat(retained, ignore_index=True)


def historical_reproduction_ok(retention: pd.DataFrame) -> tuple[bool, float, float]:
    old = pd.read_csv(RHA / "outputs" / "VOLTAGE_RETENTION_BY_CYCLE.csv")
    old = old.loc[(old["point_label"] == "P1")].copy()
    new = retention.copy()
    keys = ["stable_window", "raw_cycle_id"]
    merged = new.merge(old[keys + ["cell_voltage_mean_V", "operating_voltage_retention"]], on=keys, how="outer", suffixes=("_new", "_old"), indicator=True)
    complete = bool((merged["_merge"] == "both").all() and len(merged) == len(new) == len(old))
    if not complete:
        return False, np.inf, np.inf
    voltage_delta = np.abs(merged["cell_voltage_mean_V_new"] - merged["cell_voltage_mean_V_old"])
    retention_delta = np.abs(merged["P1_cycle_operating_voltage_retention"] - merged["operating_voltage_retention"])
    max_v = float(voltage_delta.max())
    max_h = float(retention_delta.max())
    return bool(max_v <= 1e-12 and max_h <= 1e-12), max_v, max_h


def p1_support_reproduction(retention: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    primary = retention.loc[retention["stable_window"] == "middle_50"].copy()
    conditions = list(config["condition_variables"])
    thresholds = config["gate_P1_A_support_reproduction"]
    records: list[dict[str, Any]] = []
    ranges: dict[str, tuple[float, float]] = {}
    gate = True
    for variable in conditions:
        low, high = [], []
        for stratum in STRATA:
            group = primary.loc[(primary["age_stratum"] == stratum) & primary["is_evaluation_cycle"]].copy()
            valid = group[variable].dropna()
            count = int(group["cell_voltage_mean_V"].notna().sum())
            valid_rate = float(group["cell_voltage_mean_V"].notna().mean()) if len(group) else 0.0
            condition_rate = float(group[conditions].notna().all(axis=1).mean()) if len(group) else 0.0
            lo, hi = (float(valid.min()), float(valid.max())) if len(valid) else (np.nan, np.nan)
            low.append(lo)
            high.append(hi)
            gate &= bool(
                count >= thresholds["evaluation_cycles_min_per_stratum"]
                and valid_rate >= thresholds["middle_50_valid_measurement_rate_min"]
                and condition_rate >= thresholds["seven_condition_complete_rate_min"]
            )
            records.append(
                {
                    "record_type": "stratum_variable",
                    "age_stratum": stratum,
                    "variable": variable,
                    "evaluation_cycle_count": count,
                    "valid_measurement_rate": valid_rate,
                    "condition_complete_rate": condition_rate,
                    "observed_min": lo,
                    "observed_max": hi,
                }
            )
        ranges[variable] = (max(low), min(high))
        records.append(
            {
                "record_type": "common_range",
                "age_stratum": "ALL",
                "variable": variable,
                "common_min": ranges[variable][0],
                "common_max": ranges[variable][1],
                "common_range_nonempty": bool(np.isfinite(ranges[variable][0]) and ranges[variable][0] <= ranges[variable][1]),
            }
        )
    joint = primary.copy()
    joint["in_joint_common_support"] = joint[conditions].notna().all(axis=1) & joint["is_evaluation_cycle"]
    for variable, (lo, hi) in ranges.items():
        joint["in_joint_common_support"] &= joint[variable].between(lo, hi, inclusive="both")
    expected = thresholds["required_original_RHA_joint_counts"]
    for stratum in STRATA:
        count = int(((joint["age_stratum"] == stratum) & joint["in_joint_common_support"]).sum())
        matches_rha = count == int(expected[stratum])
        passes = count >= thresholds["joint_common_support_cycles_min_per_stratum"]
        gate &= bool(matches_rha and passes)
        records.append(
            {
                "record_type": "joint_support",
                "age_stratum": stratum,
                "variable": "ALL_7",
                "joint_support_cycle_count": count,
                "historical_RHA_joint_support_cycle_count": int(expected[stratum]),
                "matches_historical_RHA": matches_rha,
                "joint_support_pass_ge15": passes,
            }
        )
    historical_ok, max_v, max_h = historical_reproduction_ok(retention)
    gate &= historical_ok
    records.append(
        {
            "record_type": "historical_RHA_measurement_reproduction",
            "age_stratum": "ALL",
            "variable": "P1",
            "historical_reproduction_pass_atol_1e-12": historical_ok,
            "maximum_voltage_difference_V": max_v,
            "maximum_retention_difference": max_h,
        }
    )
    return pd.DataFrame(records), joint, bool(gate)


def contiguous_segments(data: pd.DataFrame) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for _, group in data.groupby("age_stratum", sort=False):
        values: list[float] = []
        prior: int | None = None
        for row in group.sort_values("raw_cycle_id").itertuples(index=False):
            if prior is not None and int(row.raw_cycle_id) != prior + 1:
                if values:
                    arrays.append(np.asarray(values, dtype=float))
                values = []
            values.append(float(row.P1_cycle_operating_voltage_retention))
            prior = int(row.raw_cycle_id)
        if values:
            arrays.append(np.asarray(values, dtype=float))
    return arrays


def moving_block_sample(segments: list[np.ndarray], n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    candidates = [segment[i : i + block] for segment in segments for i in range(max(0, len(segment) - block + 1))]
    if not candidates:
        return np.asarray([], dtype=float)
    result: list[float] = []
    while len(result) < n:
        result.extend(candidates[int(rng.integers(len(candidates)))].tolist())
    return np.asarray(result[:n], dtype=float)


def raw_drift(primary: pd.DataFrame) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    data = primary.loc[primary["is_evaluation_cycle"]].sort_values("raw_cycle_id").copy()
    early = data.loc[data["age_stratum"] == "EARLY"].copy()
    late = data.loc[data["age_stratum"] == "LATE"].copy()
    D = float(late["P1_cycle_operating_voltage_retention"].median() - early["P1_cycle_operating_voltage_retention"].median())
    adjacent: list[float] = []
    for left, right in zip(data.itertuples(index=False), data.iloc[1:].itertuples(index=False)):
        if left.age_stratum == right.age_stratum and int(right.raw_cycle_id) == int(left.raw_cycle_id) + 1:
            adjacent.append(abs(float(right.P1_cycle_operating_voltage_retention - left.P1_cycle_operating_voltage_retention)))
    S = float(np.median(adjacent)) if adjacent else np.nan
    return {
        "raw_early_late_retention_contrast_D": D,
        "local_cycle_variation_S": S,
        "signal_to_local_variation": abs(D) / S if np.isfinite(S) and S > 0 else np.nan,
        "raw_negative": bool(D < 0),
        "absolute_D_gt_2S": bool(np.isfinite(S) and abs(D) > 2 * S),
        "early_evaluation_cycles": int(len(early)),
        "late_evaluation_cycles": int(len(late)),
        "adjacent_within_stratum_pairs": int(len(adjacent)),
    }, contiguous_segments(early), contiguous_segments(late)


def bootstrap_drift(early_segments: list[np.ndarray], late_segments: list[np.ndarray], n_early: int, n_late: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    rng = np.random.default_rng(2024)
    for block in [4, 8]:
        values: list[float] = []
        for iteration in range(2000):
            early = moving_block_sample(early_segments, n_early, block, rng)
            late = moving_block_sample(late_segments, n_late, block, rng)
            contrast = float(np.median(late) - np.median(early)) if len(early) and len(late) else np.nan
            values.append(contrast)
            rows.append({"block_cycles": block, "iteration": iteration, "late_minus_early_retention": contrast})
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        summary.append(
            {
                "block_cycles": block,
                "iterations": int(len(finite)),
                "bootstrap_median": float(np.median(finite)) if len(finite) else np.nan,
                "ci95_lower": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                "ci95_upper": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def deterministic_match(primary: pd.DataFrame, joint: pd.DataFrame, conditions: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluation = primary.loc[primary["is_evaluation_cycle"]].copy()
    early = joint.loc[(joint["age_stratum"] == "EARLY") & joint["in_joint_common_support"]].copy()
    late = joint.loc[(joint["age_stratum"] == "LATE") & joint["in_joint_common_support"]].copy()
    scale_source = primary.loc[primary["is_reference_cycle"] | (primary["age_stratum"] == "EARLY"), conditions].copy()
    med = scale_source.median()
    iqr = scale_source.quantile(0.75) - scale_source.quantile(0.25)
    std = scale_source.std(ddof=1)
    scale = iqr.where(iqr > 0, std).replace(0, np.nan)
    variables = scale.dropna().index.tolist()
    if early.empty or late.empty or not variables:
        return pd.DataFrame(), {"matched_pair_count": 0, "matched_late_minus_early_retention": np.nan, "matching_variables": "|".join(variables), "reason": "NO_MATCHABLE_VARIABLES_OR_CYCLES"}
    candidates: list[tuple[float, int, int]] = []
    for early_row in early.itertuples(index=False):
        early_v = np.asarray([getattr(early_row, item) for item in variables], dtype=float)
        for late_row in late.itertuples(index=False):
            late_v = np.asarray([getattr(late_row, item) for item in variables], dtype=float)
            distance = float(np.sqrt(np.sum(((early_v - late_v) / scale[variables].to_numpy(dtype=float)) ** 2)))
            candidates.append((distance, int(early_row.raw_cycle_id), int(late_row.raw_cycle_id)))
    early_lookup = early.set_index("raw_cycle_id")
    late_lookup = late.set_index("raw_cycle_id")
    used_early, used_late = set(), set()
    rows: list[dict[str, Any]] = []
    for distance, early_id, late_id in sorted(candidates):
        if early_id in used_early or late_id in used_late:
            continue
        used_early.add(early_id)
        used_late.add(late_id)
        one_early, one_late = early_lookup.loc[early_id], late_lookup.loc[late_id]
        rows.append(
            {
                "early_raw_cycle_id": early_id,
                "late_raw_cycle_id": late_id,
                "condition_distance": distance,
                "early_retention": float(one_early.P1_cycle_operating_voltage_retention),
                "late_retention": float(one_late.P1_cycle_operating_voltage_retention),
                "late_minus_early_retention": float(one_late.P1_cycle_operating_voltage_retention - one_early.P1_cycle_operating_voltage_retention),
                "matching_variables": "|".join(variables),
            }
        )
    matches = pd.DataFrame(rows)
    matched = matches["late_minus_early_retention"] if not matches.empty else pd.Series(dtype=float)
    return matches, {
        "matched_pair_count": int(len(matches)),
        "matched_late_minus_early_retention": float(matched.median()) if len(matched) else np.nan,
        "matching_variables": "|".join(variables),
        "constant_variables_excluded": "|".join([name for name in conditions if name not in variables]),
        "scale_center_reference_plus_early": json.dumps({name: float(med[name]) for name in variables}),
        "scale_IQR_or_std": json.dumps({name: float(scale[name]) for name in variables}),
    }


def stable_window_sensitivity(retention: pd.DataFrame) -> dict[str, Any]:
    data = retention.loc[(retention["stable_window"] == "last_25") & retention["is_evaluation_cycle"]].copy()
    early = data.loc[data["age_stratum"] == "EARLY", "P1_cycle_operating_voltage_retention"]
    late = data.loc[data["age_stratum"] == "LATE", "P1_cycle_operating_voltage_retention"]
    contrast = float(late.median() - early.median()) if len(early) and len(late) else np.nan
    return {
        "stable_window": "last_25",
        "early_evaluation_cycles": int(len(early)),
        "late_evaluation_cycles": int(len(late)),
        "raw_early_late_retention_contrast": contrast,
        "negative_direction": bool(np.isfinite(contrast) and contrast < 0),
    }


def p1_anchor_alignment(primary: pd.DataFrame, polarization: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool, str]:
    data = primary.loc[primary["is_evaluation_cycle"]].copy()
    primary_pol = polarization.loc[polarization["jstar_A_cm2"] == 1.0].copy()
    reference = float(primary_pol.loc[primary_pol["ageing_hour"] == 0, "voltage_median_V"].iloc[0])
    rows: list[dict[str, Any]] = []
    for checkpoint in [300, 600, 850, 1150, 1500, 1750]:
        prior = data.loc[
            (data["cycle_end_ophrs2_h"] < checkpoint) & ((checkpoint - data["cycle_end_ophrs2_h"]) <= 48)
        ].sort_values("cycle_end_ophrs2_h", ascending=False).head(3)
        pol = primary_pol.loc[primary_pol["ageing_hour"] == checkpoint]
        aligned = bool(len(prior) == 3 and len(pol) == 1)
        rows.append(
            {
                "checkpoint_h": checkpoint,
                "status": "ALIGNED" if aligned else "UNRESOLVED",
                "pre_characterization_cycle_count": int(len(prior)),
                "latest_cycle_distance_h": float(checkpoint - prior["cycle_end_ophrs2_h"].max()) if len(prior) else np.nan,
                "P1_cycle_retention": float(prior["P1_cycle_operating_voltage_retention"].median()) if aligned else np.nan,
                "polarization_voltage_V": float(pol.iloc[0].voltage_median_V) if len(pol) else np.nan,
                "SOH_pol_1_0": float(pol.iloc[0].voltage_median_V / reference) if len(pol) else np.nan,
            }
        )
    alignment = pd.DataFrame(rows)
    valid = alignment.loc[alignment["status"] == "ALIGNED"].copy()
    required = {1500, 1750}.issubset(set(valid["checkpoint_h"]))
    if len(valid) < 5 or not required:
        return alignment, pd.DataFrame(), False, "P1_ANCHOR_ALIGNMENT_INCONCLUSIVE"
    rho = float(valid["P1_cycle_retention"].corr(valid["SOH_pol_1_0"], method="spearman"))
    repeat_noise = float(primary_pol.loc[primary_pol["ageing_hour"] > 0, "voltage_std_V"].median())
    valid = valid.sort_values("checkpoint_h")
    pol_delta = np.diff(valid["polarization_voltage_V"])
    p1_delta = np.diff(valid["P1_cycle_retention"])
    informative = np.abs(pol_delta) > repeat_noise
    same_fraction = float(np.mean(np.sign(pol_delta[informative]) == np.sign(p1_delta[informative]))) if informative.any() else np.nan
    secondary = polarization.loc[polarization["jstar_A_cm2"] == 1.6].copy()
    secondary_ref = float(secondary.loc[secondary["ageing_hour"] == 0, "voltage_median_V"].iloc[0])
    secondary["SOH_pol_1_6"] = secondary["voltage_median_V"] / secondary_ref
    merged = valid.merge(secondary[["ageing_hour", "SOH_pol_1_6"]], left_on="checkpoint_h", right_on="ageing_hour", how="inner")
    rho16 = float(merged["P1_cycle_retention"].corr(merged["SOH_pol_1_6"], method="spearman")) if len(merged) >= 3 else np.nan
    compatibility = pd.DataFrame(
        [{
            "aligned_non_BoL_anchors": int(len(valid)),
            "required_1500_1750_present": required,
            "rho_P1_retention_vs_SOH_pol_1_0": rho,
            "repeat_noise_V": repeat_noise,
            "informative_transitions": int(informative.sum()),
            "same_direction_fraction": same_fraction,
            "rho_P1_retention_vs_SOH_pol_1_6": rho16,
        }]
    )
    passed = bool(rho >= 0.60 and rho > 0 and informative.sum() >= 3 and same_fraction >= 0.75 and rho16 >= -0.30)
    return alignment, compatibility, passed, "PASS" if passed else "P1_SPARSE_ANCHOR_COMPATIBILITY_NOT_SUPPORTED"


def final_status(gate_a: bool, gate_b: bool, gate_c: bool, sensitivity_pass: bool, anchor_state: str, anchor_pass: bool) -> str:
    if not gate_a:
        return "P1_SUPPORT_REPRODUCTION_FAIL"
    if not gate_b:
        return "P1_VOLTAGE_DRIFT_NOT_IDENTIFIABLE"
    if not gate_c:
        return "P1_VOLTAGE_DRIFT_CONDITION_CONFOUNDED"
    if not sensitivity_pass:
        return "P1_STABLE_DWELL_SENSITIVITY_UNRESOLVED"
    if anchor_state == "P1_ANCHOR_ALIGNMENT_INCONCLUSIVE":
        return anchor_state
    if not anchor_pass:
        return "P1_SPARSE_ANCHOR_COMPATIBILITY_NOT_SUPPORTED"
    return "P1_SINGLE_POINT_HEALTH_EVIDENCE_EXPLORATORY_PASS"


def write_report(result: dict[str, Any], drift: dict[str, Any], matched: dict[str, Any], sensitivity: dict[str, Any], anchor: pd.DataFrame) -> None:
    report = f"""# Final P1 single-point identifiability audit

## Designation and preserved RHA result

This is a **POSTHOC_EXPLORATORY** audit of the RHA-frozen primary operation
point P1=`221`. The original dual-point RHA remains
`CONDITION_COMMON_SUPPORT_INSUFFICIENT`; it is not changed by this result.
P2 was not replaced or reselected. The representation is **P1 Cycle
Operating-Voltage Retention**, not SOH ground truth.

## Access qualification

Raw post-boundary rows accessed in the boundary-crossing chunk:
**{result['raw_post_boundary_rows_accessed']}**; used: **0**. These rows were
filtered before stable-window assignment, measurements, support, matching,
figures, thresholds, and decisions. Test polarization and EIS were not read.
Development polarization was accessed only after P1-A/B/C and stable-window
sensitivity passed: **{result['development_polarization_accessed']}**.

## Final result

**`{result['final_status']}`**

P1-A support reproduction: **{result['gates']['P1_A']}**. P1-B raw long-term
visibility: **{result['gates']['P1_B']}**. P1-C condition-matched robustness:
**{result['gates']['P1_C']}**. Last-25% direction sensitivity:
**{result['gates']['stable_window_sensitivity']}**. P1-D sparse-anchor
compatibility: **{result['gates']['P1_D']}**.

| Quantity | Value |
| --- | ---: |
| Raw EARLY–LATE retention contrast | {drift.get('raw_early_late_retention_contrast_D', np.nan):.8f} |
| Local cycle variation | {drift.get('local_cycle_variation_S', np.nan):.8f} |
| Condition-matched contrast | {matched.get('matched_late_minus_early_retention', np.nan):.8f} |
| Last-25% contrast | {sensitivity.get('raw_early_late_retention_contrast', np.nan):.8f} |

Sparse-anchor results, if polarization access was allowed, are written below.
They can support only `SPARSE_POLARIZATION_ANCHOR_COMPATIBILITY`, because both
representations are age-associated; they are not independent SOH validation.

{anchor.to_markdown(index=False) if not anchor.empty else 'Polarization was not accessed because the upstream P1 gates did not all pass.'}

No neural model, continuous SOH label, RUL model, or automatic next phase was
created.
"""
    (REPORT / "FINAL_P1_IDENTIFIABILITY_AUDIT.md").write_text(report, encoding="utf-8")


def run() -> dict[str, Any]:
    ensure_dirs()
    config, provenance = verify_protocol_and_inputs()
    rha = load_rha_module()
    cycles, _, target = load_frozen_cycle_inputs(rha)
    measurements, access = rha.collect_window_samples(target)
    cycle_points = rha.aggregate_cycle_measurements(measurements)
    references, retention = choose_p1_references(cycle_points)
    support, joint, gate_a = p1_support_reproduction(retention, config)
    primary = retention.loc[retention["stable_window"] == "middle_50"].copy()
    drift, early_segments, late_segments = raw_drift(primary)
    boot = pd.DataFrame()
    boot_summary = pd.DataFrame()
    if gate_a:
        boot, boot_summary = bootstrap_drift(early_segments, late_segments, drift["early_evaluation_cycles"], drift["late_evaluation_cycles"])
        b4 = boot_summary.loc[boot_summary["block_cycles"] == 4].iloc[0]
        b8 = boot_summary.loc[boot_summary["block_cycles"] == 8].iloc[0]
        drift.update({"bootstrap4_ci95_lower": float(b4.ci95_lower), "bootstrap4_ci95_upper": float(b4.ci95_upper), "bootstrap8_median": float(b8.bootstrap_median)})
    else:
        drift.update({"bootstrap4_ci95_lower": np.nan, "bootstrap4_ci95_upper": np.nan, "bootstrap8_median": np.nan})
    gate_b = bool(gate_a and drift["raw_negative"] and drift["absolute_D_gt_2S"] and drift["bootstrap4_ci95_upper"] < 0 and drift["bootstrap8_median"] < 0)
    matches, matched = deterministic_match(primary, joint, list(config["condition_variables"])) if gate_b else (pd.DataFrame(), {"matched_pair_count": 0, "matched_late_minus_early_retention": np.nan, "reason": "P1_B_NOT_PASS"})
    gate_c = bool(gate_b and np.isfinite(matched["matched_late_minus_early_retention"]) and matched["matched_late_minus_early_retention"] < 0)
    sensitivity = stable_window_sensitivity(retention)
    sensitivity_pass = bool(not drift["raw_negative"] or sensitivity["negative_direction"])
    allow_polarization = bool(gate_a and gate_b and gate_c and sensitivity_pass)
    polarization = pd.DataFrame()
    alignment = pd.DataFrame()
    compatibility = pd.DataFrame()
    anchor_pass = False
    anchor_state = "NOT_RUN"
    if allow_polarization:
        polarization = rha.read_polarization()
        alignment, compatibility, anchor_pass, anchor_state = p1_anchor_alignment(primary, polarization)
    status = final_status(gate_a, gate_b, gate_c, sensitivity_pass, anchor_state, anchor_pass)
    access.update({
        "raw_post_boundary_rows_used": 0,
        "test_raw_rows_accessed_in_boundary_crossing_chunk": bool(access["raw_post_boundary_rows_accessed"]),
        "test_raw_rows_used_for_analysis": False,
        "test_raw_rows_used_for_threshold_selection": False,
        "test_raw_rows_used_for_decision": False,
        "test_polarization_accessed": False,
        "test_EIS_accessed": False,
        "development_polarization_accessed": bool(not polarization.empty),
    })
    references.to_csv(OUT / "P1_REFERENCE_CYCLES.csv", index=False)
    retention.to_csv(OUT / "P1_RETENTION_BY_CYCLE.csv", index=False)
    support.to_csv(OUT / "P1_SUPPORT_REPRODUCTION.csv", index=False)
    pd.DataFrame([drift]).to_csv(OUT / "P1_RAW_DRIFT_CONTRAST.csv", index=False)
    (matches if not matches.empty else pd.DataFrame([{"analysis_status": "NOT_RUN", "reason": matched.get("reason", "NO_MATCHES")}])).to_csv(OUT / "P1_CONDITION_MATCHING.csv", index=False)
    pd.DataFrame([matched]).to_csv(OUT / "P1_MATCHED_DRIFT_CONTRAST.csv", index=False)
    (boot if not boot.empty else pd.DataFrame(columns=["block_cycles", "iteration", "late_minus_early_retention"])).to_csv(OUT / "P1_BLOCK_BOOTSTRAP.csv", index=False)
    pd.DataFrame([sensitivity]).to_csv(OUT / "P1_STABLE_WINDOW_SENSITIVITY.csv", index=False)
    if allow_polarization:
        alignment.to_csv(OUT / "P1_ANCHOR_ALIGNMENT.csv", index=False)
        compatibility.to_csv(OUT / "P1_ANCHOR_COMPATIBILITY.csv", index=False)
    (OUT / "DATA_ACCESS_AUDIT.json").write_text(json.dumps(access, indent=2), encoding="utf-8")
    result = {
        "final_status": status,
        "analysis_designation": "POSTHOC_EXPLORATORY",
        "original_dual_point_RHA": "CONDITION_COMMON_SUPPORT_INSUFFICIENT",
        "original_RHA_changed": False,
        "P1_operation_point": P1,
        "P2_replaced": False,
        "frozen_cycles_reused": int(len(cycles)),
        "cycle_rediscovery": False,
        "gates": {"P1_A": gate_a, "P1_B": gate_b, "P1_C": gate_c, "stable_window_sensitivity": sensitivity_pass, "P1_D": bool(anchor_pass), "P1_D_state": anchor_state},
        "development_polarization_accessed": bool(not polarization.empty),
        "neural_training": False,
        "continuous_SOH_constructed": False,
        "RUL_trained": False,
        "automatic_next_phase": False,
        **provenance,
        **access,
    }
    (OUT / "GATE_EVALUATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    MANIFEST.write_text(yaml.safe_dump({"phase": PHASE.name, "status": "EXECUTED", "analysis_designation": "POSTHOC_EXPLORATORY", "protocol_sha256": provenance["protocol_sha256"], "execution_head": provenance["execution_head"], "original_dual_point_RHA_status": "CONDITION_COMMON_SUPPORT_INSUFFICIENT", "P1_operation_point": P1, "raw_access": access, "development_polarization_accessed": bool(not polarization.empty)}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_report(result, drift, matched, sensitivity, compatibility)
    (FINAL / "FINAL_DECISION.yaml").write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
    except AuditFailure as error:
        ensure_dirs()
        failure = {"final_status": "P1_SUPPORT_REPRODUCTION_FAIL", "technical_failure": str(error), "neural_training": False, "continuous_SOH_constructed": False, "RUL_trained": False, "automatic_next_phase": False}
        (OUT / "GATE_EVALUATION.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        sys.exit(2)

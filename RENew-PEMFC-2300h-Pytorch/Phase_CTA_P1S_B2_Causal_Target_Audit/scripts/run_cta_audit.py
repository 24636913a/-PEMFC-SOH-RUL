#!/usr/bin/env python3
"""Development-only causal target audit for the frozen P1S-B2 indicator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression


PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
OUT = PHASE / "outputs"
REPORT = PHASE / "reports"
PHI = PROJECT / "Phase_PHI_P1S_B2_Health_Index_Consolidation"
PAA = PROJECT / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs"
CSTL = PROJECT / "Phase_CSTL_Chronology_Safe_Pretrained_TimesNet_Health_Transfer" / "outputs"
ELTE = PROJECT / "Phase_ELTE_Exploratory_Late_Life_Target_Evaluation" / "outputs"
CSHSE = PROJECT / "Phase_CSHSE_Causal_SameCycle_Health_State_Estimation" / "outputs"

TRAJECTORY = PHI / "outputs" / "P1S_B2_FULL_HEALTH_TRAJECTORY.csv"
PHI_VALIDATION = PHI / "outputs" / "P1S_B2_POLARIZATION_VALIDATION.csv"
P1S_COMPLETE = PAA / "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv"
P1S_SUPPORT = PAA / "P1S_CONDITION_SUPPORT.csv"
CSTL_B2 = CSTL / "FOLD_SPECIFIC_CHRONOLOGY_SAFE_B2.csv"
ELTE_RESULTS = ELTE / "ELTE_TARGET_RESULTS.csv"
CSHSE_SUMMARY = CSHSE / "CSHSE_SUMMARY.json"

GLOBAL_A = 1.1988114938324186
GLOBAL_B = -0.20903624631182127
BOUNDARY = 1819.997738
TOL = 1e-12
FOLDS = {
    "fold_1": (946.393214, 1164.794345),
    "fold_2": (1164.794345, 1383.195476),
    "fold_3": (1383.195476, 1819.997738),
}
ANCHORS = [300, 600, 850, 1150, 1500, 1750]
SUPPORT_VARIABLES = [
    "current_density",
    "anode_inlet_temperature_C",
    "anode_inlet_pressure_kPag",
    "anode_inlet_RH_pct",
    "anode_hydrogen_flow_nl_min",
    "anode_stoichiometry",
    "cathode_stoichiometry",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_target_reopening() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_polarization_token = "Pol" + "curve"
    forbidden_raw_source_token = "raw" + "-source"
    if forbidden_polarization_token in source or forbidden_raw_source_token in source:
        raise RuntimeError("RAW_TARGET_ACCESS_CODE_DETECTED")


def fold_parameters() -> pd.DataFrame:
    table = pd.read_csv(CSTL_B2)
    table = table.loc[table["analysis_role"].eq("TARGET_AUTHORITY")].copy()
    if set(table["fold"]) != set(FOLDS) or table["future_anchor_used"].astype(str).str.lower().eq("true").any():
        raise RuntimeError("FOLD_SPECIFIC_B2_AUTHORITY_INVALID")
    return table.set_index("fold")


def support_ranges() -> dict[str, tuple[float, float]]:
    frame = pd.read_csv(P1S_SUPPORT)
    rows = frame.loc[frame["record_type"].eq("common_range")]
    ranges = {str(r.variable): (float(r.common_min), float(r.common_max)) for r in rows.itertuples(index=False)}
    if set(ranges) != set(SUPPORT_VARIABLES):
        raise RuntimeError("FROZEN_SUPPORT_RANGE_SCHEMA_MISMATCH")
    return ranges


def load_trajectory() -> pd.DataFrame:
    data = pd.read_csv(TRAJECTORY).sort_values("ophrs_h", kind="stable").reset_index(drop=True)
    if len(data) != 141 or data["cycle_interval_id"].nunique() != 141:
        raise RuntimeError("PHI_EVENT_COUNT_REPRODUCTION_FAILED")
    if not (data["ophrs_h"] < BOUNDARY).all():
        raise RuntimeError("NON_DEVELOPMENT_EVENT_IN_PHI_TRAJECTORY")
    return data


def validation_fold(age: float) -> str | None:
    for name, (lo, hi) in FOLDS.items():
        if lo <= age < hi:
            return name
    return None


def calibration_audit(data: pd.DataFrame, params: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timeline = data[["cycle_interval_id", "ophrs_h"]].copy()
    timeline["availability_basis"] = "NOMINAL_CHECKPOINT_AVAILABILITY_ONLY"
    timeline["nominal_available_anchor_checkpoints"] = timeline["ophrs_h"].map(lambda x: " / ".join(map(str, [a for a in ANCHORS if a <= x])) or "NONE")
    timeline["nominal_available_anchor_count"] = timeline["ophrs_h"].map(lambda x: sum(a <= x for a in ANCHORS))
    timeline["global_future_anchor_checkpoints"] = timeline["ophrs_h"].map(lambda x: " / ".join(map(str, [a for a in ANCHORS if a > x])) or "NONE")
    timeline["global_future_anchor_count"] = timeline["ophrs_h"].map(lambda x: sum(a > x for a in ANCHORS))
    timeline["global_B2_causally_available"] = False
    timeline["global_B2_role"] = "RETROSPECTIVE_REFERENCE_ONLY"

    joined = data.copy()
    joined["fold"] = joined["ophrs_h"].map(validation_fold)
    joined = joined.loc[joined["fold"].notna()].copy()
    joined["R_P1S"] = joined["R_P1S"].astype(float)
    joined["HI_GLOBAL_B2"] = GLOBAL_A * joined["R_P1S"] + GLOBAL_B
    joined["HI_CAUSAL_FOLD_B2"] = [
        float(params.loc[fold, "a_fold"]) * r + float(params.loc[fold, "b_fold"])
        for fold, r in zip(joined["fold"], joined["R_P1S"])
    ]
    joined["delta_HI"] = joined["HI_GLOBAL_B2"] - joined["HI_CAUSAL_FOLD_B2"]
    joined["abs_delta_HI"] = joined["delta_HI"].abs()
    joined["abs_delta_over_sigma_total"] = joined["abs_delta_HI"] / joined["sigma_total_descriptive"]
    keep = [
        "fold", "cycle_interval_id", "ophrs_h", "R_P1S", "HI_GLOBAL_B2", "HI_CAUSAL_FOLD_B2", "delta_HI", "abs_delta_HI",
        "sigma_measurement", "sigma_calibration", "sigma_total_descriptive", "abs_delta_over_sigma_total",
        "condition_support_status", "condition_support_violation_count",
    ]
    joined = joined[keep]

    rows = []
    for fold, group in joined.groupby("fold", sort=False):
        rows.append({
            "fold": fold, "N": len(group),
            "mean_abs_delta": group.abs_delta_HI.mean(), "median_abs_delta": group.abs_delta_HI.median(),
            "P95_abs_delta": group.abs_delta_HI.quantile(0.95), "max_abs_delta": group.abs_delta_HI.max(),
            "Spearman_global_causal": group[["HI_GLOBAL_B2", "HI_CAUSAL_FOLD_B2"]].corr(method="spearman").iloc[0, 1],
            "Pearson_global_causal": group[["HI_GLOBAL_B2", "HI_CAUSAL_FOLD_B2"]].corr(method="pearson").iloc[0, 1],
            "median_abs_delta_over_sigma_total": group.abs_delta_over_sigma_total.median(),
            "P95_abs_delta_over_sigma_total": group.abs_delta_over_sigma_total.quantile(0.95),
        })
    summary = pd.DataFrame(rows)
    return timeline, joined, summary


def replay_trends(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    n = len(x)
    result: dict[str, np.ndarray] = {}
    result["ISO"] = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(x, y)
    if n >= 11:
        result["SG"] = savgol_filter(y, window_length=11, polyorder=2, mode="interp")
    else:
        result["SG"] = np.full(n, np.nan)
    if n >= 2:
        edges = np.linspace(float(x.min()), float(x.max()), 16)
        bins = np.clip(np.digitize(x, edges[1:-1], right=False), 0, 14)
        knots_x, knots_y = [], []
        for bin_id in range(15):
            mask = bins == bin_id
            if mask.any():
                knots_x.append(float(np.median(x[mask])))
                knots_y.append(float(np.median(y[mask])))
        knots_x = np.asarray(knots_x)
        knots_y = np.asarray(knots_y)
        knots_y = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(knots_x, knots_y)
        result["PCHIP"] = PchipInterpolator(knots_x, knots_y, extrapolate=True)(x)
    else:
        result["PCHIP"] = np.full(n, np.nan)
    return result


def prefix_trend_audit(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    x = data["ophrs_h"].to_numpy(float)
    y = data["HI_RAW"].to_numpy(float)
    full = replay_trends(x, y)
    stored = {"SG": data["HI_TREND_SG"].to_numpy(float), "ISO": data["HI_TREND_ISOTONIC"].to_numpy(float), "PCHIP": data["HI_TREND_CONSTRAINED"].to_numpy(float)}
    for method in full:
        valid = ~np.isnan(full[method])
        if valid.any() and np.max(np.abs(full[method][valid] - stored[method][valid])) > TOL:
            raise RuntimeError(f"PHI_FULL_TREND_REPLAY_MISMATCH:{method}")

    endpoints, histories = [], []
    for n in range(1, len(data) + 1):
        pref = replay_trends(x[:n], y[:n])
        endpoint = {"prefix_event_count": n, "prefix_end_ophrs_h": x[n - 1], "HI_RAW_endpoint": y[n - 1]}
        history = {"prefix_event_count": n, "prefix_end_ophrs_h": x[n - 1]}
        for label, out_col in [("SG", "SG"), ("ISO", "ISO"), ("PCHIP", "PCHIP")]:
            current = pref[label]
            full_current = full[label][:n]
            endpoint[f"{out_col}_prefix_endpoint"] = current[-1]
            endpoint[f"{out_col}_full_sequence_endpoint"] = full_current[-1]
            endpoint[f"{out_col}_endpoint_revision"] = full_current[-1] - current[-1]
            diff = np.abs(full_current - current)
            diff = diff[np.isfinite(diff)]
            history[f"{out_col}_method_available"] = bool(len(diff))
            history[f"{out_col}_historical_revision_MAE"] = np.nan if not len(diff) else float(diff.mean())
            history[f"{out_col}_historical_revision_P95"] = np.nan if not len(diff) else float(np.quantile(diff, 0.95))
            history[f"{out_col}_historical_revision_MAX"] = np.nan if not len(diff) else float(diff.max())
        endpoints.append(endpoint)
        histories.append(history)
    endpoint_df, history_df = pd.DataFrame(endpoints), pd.DataFrame(histories)
    audit_summary = {}
    for label in ("SG", "ISO", "PCHIP"):
        abs_rev = endpoint_df[f"{label}_endpoint_revision"].abs().dropna()
        hist = history_df[f"{label}_historical_revision_MAX"].dropna()
        audit_summary[label] = {
            "endpoint_mean_abs_revision": float(abs_rev.mean()), "endpoint_p95_abs_revision": float(abs_rev.quantile(0.95)),
            "endpoint_max_abs_revision": float(abs_rev.max()), "history_max_revision": float(hist.max()),
        }
    return endpoint_df, history_df, audit_summary


def _past_slope(t: np.ndarray, y: np.ndarray, window_h: float) -> float:
    now = t[-1]
    use = (t <= now) & (t >= now - window_h)
    if use.sum() < 3 or np.ptp(t[use]) == 0:
        return np.nan
    return float(np.polyfit(t[use], y[use], 1)[0])


def causal_slope_replay(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    t, y = data.ophrs_h.to_numpy(float), data.HI_RAW.to_numpy(float)
    rows = []
    verdicts: dict[str, bool] = {}
    mapping = {24: "causal_degradation_rate_24h", 48: "causal_degradation_rate_48h", 96: "causal_degradation_rate"}
    for idx in range(len(data)):
        row = {"prefix_event_count": idx + 1, "prefix_end_ophrs_h": t[idx]}
        for hours, stored_col in mapping.items():
            replay = _past_slope(t[:idx + 1], y[:idx + 1], float(hours))
            stored = float(data.iloc[idx][stored_col])
            row[f"slope_{hours}h_prefix"] = replay
            row[f"slope_{hours}h_phi"] = stored
            row[f"slope_{hours}h_abs_difference"] = np.nan if (np.isnan(replay) and np.isnan(stored)) else abs(replay - stored)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for hours in mapping:
        diffs = frame[f"slope_{hours}h_abs_difference"].dropna()
        verdicts[f"{hours}h"] = bool((diffs <= TOL).all())
    return frame, verdicts


def support_audit(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    complete = pd.read_csv(P1S_COMPLETE)
    complete = complete.loc[complete["stable_window"].eq("middle_50")].copy()
    merged = data.merge(complete[["cycle_interval_id", *SUPPORT_VARIABLES]], on="cycle_interval_id", validate="one_to_one")
    ranges = support_ranges()
    rename = {
        "current_density": "current_density_violation", "anode_inlet_temperature_C": "temperature_violation",
        "anode_inlet_pressure_kPag": "pressure_violation", "anode_inlet_RH_pct": "RH_violation",
        "anode_hydrogen_flow_nl_min": "H2_flow_violation", "anode_stoichiometry": "anode_stoich_violation",
        "cathode_stoichiometry": "cathode_stoich_violation",
    }
    for variable, (lo, hi) in ranges.items():
        merged[rename[variable]] = ((merged[variable] < lo) | (merged[variable] > hi))
    flags = list(rename.values())
    merged["violation_count"] = merged[flags].sum(axis=1).astype(int)
    merged["support_status"] = np.where(merged.violation_count.eq(0), "IN_SUPPORT", "OUT_OF_SUPPORT")
    if int(merged.support_status.eq("IN_SUPPORT").sum()) != 82 or int(merged.support_status.eq("OUT_OF_SUPPORT").sum()) != 59:
        raise RuntimeError("SUPPORT_COUNT_REPRODUCTION_FAILED")
    if not (merged["support_status"].eq(merged["condition_support_status"]).all() and merged["violation_count"].eq(merged["condition_support_violation_count"]).all()):
        raise RuntimeError("PHI_SUPPORT_EVENT_ALIGNMENT_MISMATCH")
    merged["time_region"] = pd.cut(
        merged.ophrs_h, bins=[72.788690, 946.393214, 1164.794345, 1383.195476, 1819.997738],
        labels=["Region_A", "Region_B", "Region_C", "Region_D"], right=False, include_lowest=True,
    ).astype(str)
    event = merged[["cycle_interval_id", "ophrs_h", "HI_RAW", "support_status", "violation_count", *flags]].copy()
    event = event.rename(columns={"HI_RAW": "HI_GLOBAL_B2", "violation_count": "condition_support_violation_count"})

    rows = []
    for scope, groupby in [("ALL", ["support_status"]), ("TIME_REGION", ["time_region", "support_status"])]:
        for keys, group in merged.groupby(groupby, dropna=False, observed=False):
            keys = (keys,) if not isinstance(keys, tuple) else keys
            row = {"scope": scope, "time_region": "ALL" if scope == "ALL" else keys[0], "support_status": keys[-1], "N": len(group)}
            for field, name in [("HI_RAW", "HI_RAW"), ("sigma_measurement", "sigma_measurement"), ("sigma_total_descriptive", "sigma_total")]:
                row[f"{name}_median"] = group[field].median()
                row[f"{name}_IQR"] = group[field].quantile(0.75) - group[field].quantile(0.25)
            for field, name in [("causal_degradation_rate_24h", "causal_24h_slope"), ("causal_degradation_rate_48h", "causal_48h_slope"), ("causal_degradation_rate", "causal_96h_slope")]:
                values = group[field].dropna()
                row[f"{name}_median"] = values.median() if len(values) else np.nan
                row[f"{name}_IQR"] = values.quantile(0.75) - values.quantile(0.25) if len(values) else np.nan
            rows.append(row)
    dominant = event[flags].sum().sort_values(ascending=False).index.tolist()
    return event, pd.DataFrame(rows), dominant


def polarization_evidence() -> pd.DataFrame:
    validation = pd.read_csv(PHI_VALIDATION)
    validation["checkpoint"] = validation["checkpoint_h"].astype(int)
    validation["evidence_type"] = np.where(
        validation["evidence_region"].eq("DEVELOPMENT_ANCHOR"),
        "CALIBRATION_ANCHOR_NOT_INDEPENDENT", "ALREADY_UNBLINDED_EXPLORATORY_REFERENCE",
    )
    validation["AE_Y"] = validation["error_Y"].abs()
    validation["AE_mV"] = validation["error_mV"]
    return validation[["checkpoint", "evidence_type", "support_status", "support_count", "P1S_B2_prediction", "polarization_Y", "AE_Y", "AE_mV"]]


def source_manifest() -> dict[str, object]:
    files = [TRAJECTORY, PHI_VALIDATION, P1S_COMPLETE, P1S_SUPPORT, CSTL_B2, ELTE_RESULTS, CSHSE_SUMMARY]
    return {
        "base_commit": "5ffd0895e44b28ff95eb2348d08fccd199fb491c",
        "sources": [{"path": str(file.relative_to(PROJECT)), "sha256": sha256(file)} for file in files],
        "test_targets_reopened": False,
        "model_training_performed": False,
        "RUL_training_performed": False,
        "raw_polarization_files_read": False,
    }


def write_report(calibration_events: pd.DataFrame, trend: dict[str, dict[str, float]], slopes: dict[str, bool], support: pd.DataFrame, dominant: list[str]) -> None:
    overall = {
        "mean_abs_delta": float(calibration_events["abs_delta_HI"].mean()),
        "P95_abs_delta": float(calibration_events["abs_delta_HI"].quantile(0.95)),
        "max_abs_delta": float(calibration_events["abs_delta_HI"].max()),
    }
    report = f"""# P1S-B2 Causal Target Audit

## Calibration time availability

The all-six-anchor mapping used by PHI is explicitly classified as
`RETROSPECTIVE_REFERENCE_ONLY`.  Every one of the 141 PHI events occurs before
the final 1750 h anchor, so the global B2 function was not causally available
at any event.  Availability is recorded only from nominal checkpoint timing
and is labelled `NOMINAL_CHECKPOINT_AVAILABILITY_ONLY`; no unsupported claim is
made about the exact real-time availability timestamp of a characterization
result.

For expanding-fold validation, the frozen CSTL functions are chronology-safe:
fold 1 uses 300/600/850, and folds 2–3 use 300/600/850/1150.  No frozen fold
uses a future calibration anchor.  Across the validation events, mean absolute
global-versus-causal B2 revision is {overall['mean_abs_delta']:.6g}, P95 is
{overall['P95_abs_delta']:.6g}, and the maximum is {overall['max_abs_delta']:.6g}.
This measures label rewriting from future calibration information, not accuracy.

## Prefix trend replay

All PHI full-sequence trend methods revise at least some endpoint or history
when later P1S cycles are supplied: SG endpoint max revision =
{trend['SG']['endpoint_max_abs_revision']:.6g}, isotonic =
{trend['ISO']['endpoint_max_abs_revision']:.6g}, and monotonic PCHIP =
{trend['PCHIP']['endpoint_max_abs_revision']:.6g}.  Thus full-sequence SG,
isotonic and PCHIP values are `RETROSPECTIVE_TREND_ONLY`.

The past-only raw-HI slope replay matches PHI exactly within {TOL:g}: 24 h =
{slopes['24h']}, 48 h = {slopes['48h']}, 96 h = {slopes['96h']}.  This establishes
calculation causality only; it does not establish predictive value.

The 35 upward raw-HI transitions remain empirical observations.  Zero upward
changes in the isotonic trend are `CONSTRAINT-INDUCED_MONOTONICITY`, not proof
of irreversible PEMFC degradation.

## Support-domain stratification

Support reproduction yields 82 `IN_SUPPORT` and 59 `OUT_OF_SUPPORT` events.
The most frequent frozen support violations are {', '.join(dominant[:3])}.
The summary is descriptive only: without per-cycle polarization ground truth it
cannot show that either group is more or less accurate.  Since out-of-support
events constitute a substantial portion of the series and occur over formal
time regions, future work must retain support as a stratum / quality variable
rather than delete those events.

## Cross-phase provenance clarification

CSHSE was planned as current same-cycle health estimation, but its final data
availability audit found `same_cycle_history_event_count = 0` and
`FAIL__FROZEN_P1S_MEASUREMENT_PRECEDES_TARGET_CYCLE_DYNAMIC_HISTORY`.
Its only defensible interpretation is
`DESCRIPTIVE_GUARD_CONSTRAINED_PRE_CYCLE_REPLAY_ONLY`; it is not evidence for
same-cycle current health estimation.

## FINAL TARGET DECISION

**T2**

**TARGET:** `RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET`

**CALIBRATION:** `R_P1S = V_P1S / V_ref` is observable at the P1S cycle.  The
global six-anchor affine B2 mapping remains retrospective interpretation;
fold-specific B2 may only be used in chronology-safe expanding-fold evaluation
after its frozen nominal anchor set is available.

**TREND:** `FULL_SEQUENCE_TRENDS_RETROSPECTIVE_ONLY`

**ONLINE DEGRADATION RATE:** allowed as a causal raw-HI slope candidate;
predictive utility remains untested.

**SUPPORT HANDLING:** `STRATIFIED_MODELING_REQUIRED`

**NEXT PHASE:** `CAUSAL_DEGRADATION_FORECASTING_PROTOCOL`
"""
    (REPORT / "P1S_B2_CAUSAL_TARGET_AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    assert_no_target_reopening()
    data = load_trajectory()
    params = fold_parameters()
    timeline, global_vs_causal, calibration_summary = calibration_audit(data, params)
    endpoint, history, trend_summary = prefix_trend_audit(data)
    slopes, slope_pass = causal_slope_replay(data)
    support_event, support_summary, dominant = support_audit(data)
    evidence = polarization_evidence()

    timeline.to_csv(OUT / "CALIBRATION_TIME_AVAILABILITY.csv", index=False)
    global_vs_causal.to_csv(OUT / "GLOBAL_VS_CHRONOLOGY_SAFE_B2.csv", index=False)
    calibration_summary.to_csv(OUT / "CALIBRATION_REVISION_SUMMARY.csv", index=False)
    endpoint.to_csv(OUT / "TREND_PREFIX_ENDPOINT_REVISION.csv", index=False)
    history.to_csv(OUT / "TREND_PREFIX_HISTORY_REVISION.csv", index=False)
    slopes.to_csv(OUT / "CAUSAL_SLOPE_PREFIX_REPLAY.csv", index=False)
    support_event.to_csv(OUT / "SUPPORT_DOMAIN_EVENT_AUDIT.csv", index=False)
    support_summary.to_csv(OUT / "SUPPORT_STRATIFIED_SUMMARY.csv", index=False)
    evidence.to_csv(OUT / "POLARIZATION_EVIDENCE_BY_SUPPORT.csv", index=False)
    (OUT / "CTA_SOURCE_MANIFEST.json").write_text(json.dumps(source_manifest(), indent=2), encoding="utf-8")

    cal_all = global_vs_causal["abs_delta_HI"]
    summary = {
        "phase": "Phase_CTA_P1S_B2_Causal_Target_Audit", "base_commit": "5ffd0895e44b28ff95eb2348d08fccd199fb491c",
        "n_p1s_events": int(len(data)), "in_support_count": int((data.condition_support_status == "IN_SUPPORT").sum()), "out_of_support_count": int((data.condition_support_status == "OUT_OF_SUPPORT").sum()),
        "global_B2_role": "RETROSPECTIVE_REFERENCE_ONLY",
        "fold1_B2_a": float(params.loc["fold_1", "a_fold"]), "fold1_B2_b": float(params.loc["fold_1", "b_fold"]),
        "fold2_B2_a": float(params.loc["fold_2", "a_fold"]), "fold2_B2_b": float(params.loc["fold_2", "b_fold"]),
        "fold3_B2_a": float(params.loc["fold_3", "a_fold"]), "fold3_B2_b": float(params.loc["fold_3", "b_fold"]),
        "global_vs_causal_mean_abs_delta": float(cal_all.mean()), "global_vs_causal_p95_abs_delta": float(cal_all.quantile(0.95)), "global_vs_causal_max_abs_delta": float(cal_all.max()),
        "global_vs_causal_median_abs_delta_over_sigma_total": float(global_vs_causal.abs_delta_over_sigma_total.median()),
        "global_vs_causal_p95_abs_delta_over_sigma_total": float(global_vs_causal.abs_delta_over_sigma_total.quantile(0.95)),
        "SG_prefix_revision": trend_summary["SG"], "isotonic_prefix_revision": trend_summary["ISO"], "PCHIP_prefix_revision": trend_summary["PCHIP"],
        "causal_24h_prefix_invariance": "PASS" if slope_pass["24h"] else "FAIL", "causal_48h_prefix_invariance": "PASS" if slope_pass["48h"] else "FAIL", "causal_96h_prefix_invariance": "PASS" if slope_pass["96h"] else "FAIL",
        "dominant_support_violation_variables": dominant,
        "causal_target_decision": "T2", "target": "RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET",
        "trend_usage_decision": "FULL_SEQUENCE_TRENDS_RETROSPECTIVE_ONLY", "causal_raw_slope_online_candidate": "YES" if all(slope_pass.values()) else "NO",
        "support_handling_decision": "STRATIFIED_MODELING_REQUIRED",
        "test_targets_reopened": False, "model_training_performed": False, "RUL_training_performed": False,
    }
    (OUT / "CTA_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(global_vs_causal, trend_summary, slope_pass, support_summary, dominant)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

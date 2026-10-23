#!/usr/bin/env python3
"""Consolidate the frozen P1S-B2 mapping without reopening late-life targets.

Inputs are frozen, already-derived artifacts only.  In particular, this code
does not contain a raw polarization path and does not parse a Polcurve file.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
OUT = PHASE / "outputs"
FIG = PHASE / "figures"
REPORT = PHASE / "reports"
CONFIG = PHASE / "configs" / "FROZEN_PHI_PROTOCOL.yaml"

PAA = PROJECT / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs"
AHC_CONFIG = PROJECT / "Phase_AHC_TV_One_Time_Late_Life_Test_Protocol" / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml"
AHC_B2 = PROJECT / "Phase_AHC_Sparse_Anchor_Affine_Health_Calibration" / "outputs"
CHM_TARGETS = PROJECT / "Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model" / "inputs" / "CHM_SPARSE_ANCHOR_TARGETS.csv"
RCR_CATALOG = PROJECT / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit" / "outputs" / "RAW_CYCLE_CATALOG.csv"
ELTE = PROJECT / "Phase_ELTE_Exploratory_Late_Life_Target_Evaluation"

P1S_COMPLETE = PAA / "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv"
P1S_ALIGNMENT = PAA / "P1S_ANCHOR_ALIGNMENT.csv"
P1S_SUPPORT = PAA / "P1S_CONDITION_SUPPORT.csv"
B2_LOAO = AHC_B2 / "B2_LOAO_PARAMETER_AUDIT.csv"
ELTE_RESULTS = ELTE / "outputs" / "ELTE_TARGET_RESULTS.csv"
ELTE_SUMMARY = ELTE / "outputs" / "ELTE_SUMMARY.json"

DEVELOPMENT_BOUNDARY = 1819.997738
V_REF = 0.71383023635
V_BOL = 0.6989533066
FROZEN_A = 1.1988114938324186
FROZEN_B = -0.20903624631182127


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_raw_target_access() -> None:
    """Fail closed if a forbidden raw-target filename has been configured."""
    forbidden = ("Polcurve_Nominal_RHFC_2000h.csv", "Polcurve_Nominal_RHFC_2300h.csv")
    configured = [str(v) for v in globals().values() if isinstance(v, Path)]
    if any(any(name in value for name in forbidden) for value in configured):
        raise RuntimeError("RAW_LATE_LIFE_TARGET_PATH_CONFIGURED")


def parse_frozen_models() -> tuple[float, float]:
    text = AHC_CONFIG.read_text(encoding="utf-8")
    slope = float(re.search(r"^\s*slope:\s*([0-9eE+\-.]+)", text, flags=re.M).group(1))
    intercept = float(re.search(r"^\s*intercept:\s*([0-9eE+\-.]+)", text, flags=re.M).group(1))
    if not (math.isclose(slope, FROZEN_A, rel_tol=0, abs_tol=1e-15) and math.isclose(intercept, FROZEN_B, rel_tol=0, abs_tol=1e-15)):
        raise RuntimeError("FROZEN_B2_PARAMETER_MISMATCH")
    return slope, intercept


def common_support_ranges() -> dict[str, tuple[float, float]]:
    support = pd.read_csv(P1S_SUPPORT)
    rows = support.loc[support["record_type"].eq("common_range")]
    return {
        str(row.variable): (float(row.common_min), float(row.common_max))
        for row in rows.itertuples(index=False)
    }


def load_cycle_trajectory(a: float, b: float) -> pd.DataFrame:
    full = pd.read_csv(P1S_COMPLETE)
    required = {"middle_50", "last_25"}
    if not required.issubset(set(full["stable_window"])):
        raise RuntimeError("FROZEN_STABLE_WINDOWS_NOT_AVAILABLE")
    middle = full.loc[full["stable_window"].eq("middle_50")].copy()
    last = full.loc[full["stable_window"].eq("last_25"), ["cycle_interval_id", "P1S_voltage_V"]].rename(columns={"P1S_voltage_V": "P1S_voltage_last25_V"})
    data = middle.merge(last, on="cycle_interval_id", how="inner", validate="one_to_one")
    if data["cycle_interval_id"].nunique() != len(data):
        raise RuntimeError("NONUNIQUE_COMPLETE_CYCLE_TRAJECTORY")

    catalog = pd.read_csv(RCR_CATALOG)
    catalog["raw_cycle_id"] = catalog["raw_cycle_id"].astype(float)
    data["raw_cycle_id"] = data["raw_cycle_id"].astype(float)
    data = data.merge(
        catalog[["raw_cycle_id", "start_ophrs2_h", "end_ophrs2_h", "complete_cycle"]],
        on="raw_cycle_id", how="left", validate="many_to_one",
    )
    if data["start_ophrs2_h"].isna().any() or not data["complete_cycle"].all():
        raise RuntimeError("COMPLETE_P1S_CYCLE_CATALOG_ALIGNMENT_FAILED")

    data = data.sort_values("measurement_end_ophrs2_h", kind="stable").reset_index(drop=True)
    data["cycle_index"] = np.arange(1, len(data) + 1)
    data["ophrs_h"] = data["measurement_end_ophrs2_h"]
    data["cycle_start_ophrs2_h"] = data["start_ophrs2_h"]
    data["cycle_end_ophrs2_h"] = data["end_ophrs2_h"]
    data["p1s_measurement_time_h"] = data["measurement_end_ophrs2_h"]
    data["R_P1S"] = data["P1S_voltage_V"] / V_REF
    data["HI_RAW"] = a * data["R_P1S"] + b
    data["HI_last25"] = a * (data["P1S_voltage_last25_V"] / V_REF) + b
    data["sigma_measurement"] = np.abs(data["HI_RAW"] - data["HI_last25"]) / math.sqrt(2.0)
    data["region"] = np.where(data["ophrs_h"] < DEVELOPMENT_BOUNDARY, "DEVELOPMENT", "LATE_LIFE_EXPLORATORY_REGION")

    ranges = common_support_ranges()
    violations = np.zeros(len(data), dtype=int)
    for column, (lo, hi) in ranges.items():
        if column not in data:
            raise RuntimeError(f"MISSING_FROZEN_SUPPORT_VARIABLE:{column}")
        violations += ((data[column] < lo) | (data[column] > hi)).astype(int).to_numpy()
    data["condition_support_violation_count"] = violations
    data["condition_support_status"] = np.where(violations == 0, "IN_SUPPORT", "OUT_OF_SUPPORT")
    return data


def add_calibration_uncertainty(data: pd.DataFrame) -> pd.DataFrame:
    loao = pd.read_csv(B2_LOAO)
    if len(loao) != 6 or not {"slope_a", "intercept_b"}.issubset(loao.columns):
        raise RuntimeError("SIX_ANCHOR_LOAO_CALIBRATION_UNAVAILABLE")
    functions = np.vstack([
        loao["slope_a"].to_numpy(float)[:, None] * data["R_P1S"].to_numpy(float)[None, :]
        + loao["intercept_b"].to_numpy(float)[:, None]
    ])
    data["sigma_calibration"] = functions.std(axis=0, ddof=1)
    data["sigma_total_descriptive"] = np.sqrt(data["sigma_measurement"] ** 2 + data["sigma_calibration"] ** 2)
    data["HI_lower"] = data["HI_RAW"] - 1.96 * data["sigma_total_descriptive"]
    data["HI_upper"] = data["HI_RAW"] + 1.96 * data["sigma_total_descriptive"]
    return data


def _causal_slope(t: np.ndarray, y: np.ndarray, window_h: float) -> np.ndarray:
    result = np.full(len(t), np.nan)
    for i, current in enumerate(t):
        use = (t <= current) & (t >= current - window_h)
        if use.sum() >= 3 and np.ptp(t[use]) > 0:
            result[i] = np.polyfit(t[use], y[use], deg=1)[0]
    return result


def add_trends_and_rates(data: pd.DataFrame) -> pd.DataFrame:
    x = data["ophrs_h"].to_numpy(float)
    y = data["HI_RAW"].to_numpy(float)
    data["HI_TREND_SG"] = savgol_filter(y, window_length=11, polyorder=2, mode="interp")
    isotonic = IsotonicRegression(increasing=False, out_of_bounds="clip")
    data["HI_TREND_ISOTONIC"] = isotonic.fit_transform(x, y)

    edges = np.linspace(float(x.min()), float(x.max()), 16)
    bins = np.clip(np.digitize(x, edges[1:-1], right=False), 0, 14)
    knot_x, knot_y = [], []
    for bin_id in range(15):
        take = bins == bin_id
        if take.any():
            knot_x.append(float(np.median(x[take])))
            knot_y.append(float(np.median(y[take])))
    knot_x_arr, knot_y_arr = np.asarray(knot_x), np.asarray(knot_y)
    knot_y_iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(knot_x_arr, knot_y_arr)
    spline = PchipInterpolator(knot_x_arr, knot_y_iso, extrapolate=True)
    data["HI_TREND_CONSTRAINED"] = spline(x)

    data["causal_degradation_rate_24h"] = _causal_slope(x, y, 24.0)
    data["causal_degradation_rate_48h"] = _causal_slope(x, y, 48.0)
    data["causal_degradation_rate"] = _causal_slope(x, y, 96.0)
    data["local_degradation_rate_per_h"] = np.gradient(data["HI_TREND_ISOTONIC"].to_numpy(float), x)
    return data


def trend_comparison(data: pd.DataFrame) -> pd.DataFrame:
    result = []
    for name in ["HI_RAW", "HI_TREND_SG", "HI_TREND_ISOTONIC", "HI_TREND_CONSTRAINED"]:
        values = data[name].to_numpy(float)
        result.append({
            "method": name,
            "monotonicity_violations": int((np.diff(values) > 1e-12).sum()),
            "mean_absolute_distortion_from_raw": float(np.mean(np.abs(values - data["HI_RAW"].to_numpy(float)))),
            "roughness_mean_absolute_first_difference": float(np.mean(np.abs(np.diff(values)))),
            "uses_late_life_polarization_target": False,
            "selected_primary": name == "HI_TREND_ISOTONIC",
        })
    return pd.DataFrame(result)


def build_validation(a: float, b: float) -> pd.DataFrame:
    anchors = pd.read_csv(P1S_ALIGNMENT)
    anchor_summary = anchors.loc[anchors["record_type"].eq("anchor_summary"), ["checkpoint_h", "P1S_anchor_retention", "joint_support_interval_count", "status"]].copy()
    targets = pd.read_csv(CHM_TARGETS)[["checkpoint", "Y"]].rename(columns={"checkpoint": "checkpoint_h", "Y": "polarization_Y"})
    dev = anchor_summary.merge(targets, on="checkpoint_h", validate="one_to_one")
    dev["evidence_region"] = "DEVELOPMENT_ANCHOR"
    dev["P1S_B2_prediction"] = a * dev["P1S_anchor_retention"] + b
    dev["error_Y"] = dev["P1S_B2_prediction"] - dev["polarization_Y"]
    dev["error_mV"] = dev["error_Y"].abs() * V_BOL * 1000.0
    dev["support_status"] = np.where(dev["joint_support_interval_count"].eq(3), "FROZEN_ALIGNMENT_COMPATIBLE", "PARTIAL_OR_OUT_OF_SUPPORT")
    dev["support_count"] = dev["joint_support_interval_count"].astype(int)

    elte = pd.read_csv(ELTE_RESULTS)
    late = pd.DataFrame({
        "checkpoint_h": elte["checkpoint"].astype(int),
        "evidence_region": "EXPLORATORY_LATE_LIFE_ALREADY_UNBLINDED",
        "P1S_B2_prediction": elte["prediction_Raw_P1S_B2"],
        "polarization_Y": elte["target_Y"],
        "error_Y": elte["AE_Raw_P1S_B2"],
        "error_mV": elte["AE_Raw_P1S_B2_mV"],
        "support_status": elte["support_status"],
        "support_count": elte["support_count"].astype(int),
        "P1S_anchor_retention": elte["R_median"],
    })
    return pd.concat([dev, late], ignore_index=True).sort_values("checkpoint_h", kind="stable").reset_index(drop=True)


def source_manifest() -> dict[str, object]:
    files = [CONFIG, P1S_COMPLETE, P1S_ALIGNMENT, P1S_SUPPORT, B2_LOAO, AHC_CONFIG, CHM_TARGETS, RCR_CATALOG, ELTE_RESULTS, ELTE_SUMMARY]
    return {
        "base_commit": "d367ae76958a4289a5be6615ff6e81d75b664d54",
        "sources": [{"path": str(path.relative_to(PROJECT)), "sha256": sha256_file(path)} for path in files],
        "raw_polarization_target_files_read": False,
        "test_targets_reopened": False,
        "post_ELTE_retuning": False,
    }


def write_figures(data: pd.DataFrame, validation: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    dev = validation.loc[validation["evidence_region"].eq("DEVELOPMENT_ANCHOR")]
    late = validation.loc[validation["evidence_region"].eq("EXPLORATORY_LATE_LIFE_ALREADY_UNBLINDED")]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    ax.fill_between(data["ophrs_h"], data["HI_lower"], data["HI_upper"], color="#4C78A8", alpha=0.14, label="Descriptive uncertainty interval")
    ax.scatter(data["ophrs_h"], data["HI_RAW"], s=14, color="#4C78A8", alpha=0.68, label="Raw P1S-B2 cycle HI")
    ax.plot(data["ophrs_h"], data["HI_TREND_ISOTONIC"], color="#E45756", lw=2.2, label="Primary trend: decreasing isotonic")
    ax.scatter(dev["checkpoint_h"], dev["polarization_Y"], marker="s", s=44, color="#54A24B", label="Development polarization anchors", zorder=5)
    ax.scatter(late["checkpoint_h"], late["polarization_Y"], marker="D", s=50, color="#F58518", label="Already-unblinded exploratory late-life targets", zorder=6)
    ax.scatter(late["checkpoint_h"], late["P1S_B2_prediction"], marker="x", s=58, linewidths=2, color="#E45756", label="Frozen P1S-B2 late-life predictions", zorder=7)
    ax.axvline(DEVELOPMENT_BOUNDARY, color="0.35", lw=1, ls="--", label="Development boundary")
    ax.set(xlabel="Operating hours, OpHrs2 (h)", ylabel="Health indicator / polarization health ratio", title="P1S-B2 health trajectory and frozen validation evidence")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.savefig(FIG / "Fig1_P1S_B2_Full_Health_Trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.7), constrained_layout=True)
    ax.plot(data["ophrs_h"], data["causal_degradation_rate"], color="#4C78A8", lw=1.5, label="Causal 96 h OLS slope of raw HI")
    ax.plot(data["ophrs_h"], data["local_degradation_rate_per_h"], color="#E45756", lw=1.8, label="Local slope of isotonic trend")
    ax.axhline(0, color="0.35", lw=0.8)
    ax.axvline(DEVELOPMENT_BOUNDARY, color="0.35", lw=1, ls="--", label="Development boundary")
    ax.set(xlabel="Operating hours, OpHrs2 (h)", ylabel="Degradation rate (HI per h)", title="Causal and trend-layer P1S-B2 degradation rates")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.savefig(FIG / "Fig2_P1S_B2_Degradation_Rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    ax.scatter(dev["polarization_Y"], dev["P1S_B2_prediction"], s=46, color="#54A24B", marker="s", label="Development anchors")
    ax.scatter(late["polarization_Y"], late["P1S_B2_prediction"], s=56, color="#F58518", marker="D", label="Already-unblinded late life")
    lo = min(validation["polarization_Y"].min(), validation["P1S_B2_prediction"].min()) - 0.001
    hi = max(validation["polarization_Y"].max(), validation["P1S_B2_prediction"].max()) + 0.001
    ax.plot([lo, hi], [lo, hi], color="0.35", ls="--", lw=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Polarization-derived Y", ylabel="Frozen P1S-B2 prediction", title="P1S-B2 agreement with polarization health")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.savefig(FIG / "Fig3_P1S_B2_vs_Polarization_Validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(data: pd.DataFrame, validation: pd.DataFrame, comparison: pd.DataFrame, summary: dict[str, object]) -> None:
    development = validation.loc[validation["evidence_region"].eq("DEVELOPMENT_ANCHOR")]
    late = validation.loc[validation["evidence_region"].eq("EXPLORATORY_LATE_LIFE_ALREADY_UNBLINDED")]
    report = f"""# P1S-B2 Health Indicator Consolidation

## Health-indicator definition

`HI_P1S = {FROZEN_A:.16f} × R_P1S {FROZEN_B:+.16f}`, where
`R_P1S = V_P1S / {V_REF:.11f}`.  This is the frozen Raw P1S-B2 calibrated
health indicator: a densely observable, cycle-aligned degradation-related
health proxy, not independently validated true SOH or a validated RUL state.
The `middle_50` P1S stable window was used because it exactly reproduces the
complete-cycle values used in the frozen Development anchor alignment.

## Coverage and trajectory

The source has {len(data)} unique legal complete P1S cycles from
{data.ophrs_h.min():.2f} h to {data.ophrs_h.max():.2f} h.  All lie below the
{DEVELOPMENT_BOUNDARY:.6f} h Development boundary; therefore the cycle-level
late-life event count is zero.  The 2000 h and 2300 h points in figures and
validation are immutable, already-unblinded ELTE checkpoint evidence, not new
P1S observations, not a reopened target, and not an independent Test.

## Trend construction

Raw cycle observations are preserved as `HI_RAW`.  Three predeclared trend
representations were generated: light Savitzky–Golay smoothing, decreasing
isotonic regression, and a fixed-bin monotonic PCHIP spline.  The primary
trend is decreasing isotonic regression because it is the simplest method with
zero upward violations, uses no late-life target criterion, and leaves the raw
indicator unchanged.  It should not be interpreted as a replacement for raw
cycle measurements.

## Uncertainty description

`sigma_measurement` is the sample standard deviation across the two frozen
per-cycle stable-window estimates (`middle_50` and `last_25`), i.e. a
window-sensitivity / measurement-variability proxy.  `sigma_calibration` is
the standard deviation of the six pre-existing leave-one-Development-anchor-out
B2 functions at each retention.  The displayed interval is raw HI ± 1.96 times
their quadrature total.  It is a descriptive calibration-plus-measurement
interval, not a probability-coverage guarantee.

## Polarization consistency

The six Development anchor MAE is {development.error_mV.mean():.4f} mV.
The two already-unblinded exploratory late-life checkpoints have MAE
{late.error_mV.mean():.4f} mV.  The late-life result remains exploratory and
same-stack only: 2000 h is out of frozen joint support and 2300 h has 2/3
support-compatible intervals.  Neither was used to fit B2, choose a trend, or
change uncertainty settings.

## Answer and limitation

**YES.** Within its available data coverage, the frozen P1S-B2 mapping is
sufficiently stable and dense to serve as the primary degradation health
indicator for subsequent degradation-trajectory forecasting and
threshold-conditional RUL research.  This decision does not create true SOH
ground truth, does not define an EOL threshold, and does not solve the absence
of post-1819.997738 h cycle-level P1S observations.  Any later RUL work must
remain explicit about right censoring and threshold conditionality.

## No post-target retuning

No 2000 h/2300 h raw target file was read by PHI.  Existing ELTE output was
used only for descriptive validation.  B2, the reference voltage, support
ranges, trend choice, and uncertainty method were not retuned after ELTE.
"""
    (REPORT / "P1S_B2_HEALTH_INDICATOR_CONSOLIDATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    assert_no_raw_target_access()
    a, b = parse_frozen_models()
    data = add_trends_and_rates(add_calibration_uncertainty(load_cycle_trajectory(a, b)))
    comparison = trend_comparison(data)
    validation = build_validation(a, b)

    export_columns = [
        "cycle_interval_id", "cycle_index", "raw_cycle_id", "cycle_start_ophrs2_h", "cycle_end_ophrs2_h", "p1s_measurement_time_h", "ophrs_h", "region",
        "P1S_voltage_V", "R_P1S", "HI_RAW", "HI_TREND_SG", "HI_TREND_ISOTONIC", "HI_TREND_CONSTRAINED",
        "condition_support_status", "condition_support_violation_count", "local_degradation_rate_per_h", "causal_degradation_rate_24h", "causal_degradation_rate_48h", "causal_degradation_rate",
        "sigma_measurement", "sigma_calibration", "sigma_total_descriptive", "HI_lower", "HI_upper",
    ]
    data[export_columns].to_csv(OUT / "P1S_B2_FULL_HEALTH_TRAJECTORY.csv", index=False)
    comparison.to_csv(OUT / "TREND_METHOD_COMPARISON.csv", index=False)
    validation.to_csv(OUT / "P1S_B2_POLARIZATION_VALIDATION.csv", index=False)
    manifest = source_manifest()
    (OUT / "PHI_SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    primary = comparison.loc[comparison.selected_primary].iloc[0]
    dev = validation.loc[validation.evidence_region.eq("DEVELOPMENT_ANCHOR")]
    late = validation.loc[validation.evidence_region.eq("EXPLORATORY_LATE_LIFE_ALREADY_UNBLINDED")]
    raw_violations = int((np.diff(data.HI_RAW.to_numpy()) > 0).sum())
    trend_violations = int(primary.monotonicity_violations)
    latest_rate = data.loc[data.ophrs_h >= data.ophrs_h.max() - 96, "causal_degradation_rate"].dropna()
    summary = {
        "phase": "Phase_PHI_P1S_B2_Health_Index_Consolidation",
        "base_commit": "d367ae76958a4289a5be6615ff6e81d75b664d54",
        "n_total_legal_p1s_events": int(len(data)),
        "n_development_events": int(data.region.eq("DEVELOPMENT").sum()),
        "n_late_life_events": int(data.region.eq("LATE_LIFE_EXPLORATORY_REGION").sum()),
        "late_life_cycle_level_observations_available": False,
        "health_indicator": "P1S-B2 calibrated health indicator (HI_P1S)",
        "frozen_B2_a": a,
        "frozen_B2_b": b,
        "P1S_reference_voltage": V_REF,
        "raw_HI_min": float(data.HI_RAW.min()),
        "raw_HI_max": float(data.HI_RAW.max()),
        "raw_HI_range": float(data.HI_RAW.max() - data.HI_RAW.min()),
        "development_anchor_MAE": float(dev.error_Y.abs().mean()),
        "development_anchor_MAE_mV": float(dev.error_mV.mean()),
        "late_life_exploratory_MAE": float(late.error_Y.abs().mean()),
        "late_life_exploratory_MAE_mV": float(late.error_mV.mean()),
        "selected_primary_trend": "HI_TREND_ISOTONIC",
        "monotonicity_violation_raw": raw_violations,
        "monotonicity_violation_trend": trend_violations,
        "mean_degradation_rate": float(data.local_degradation_rate_per_h.mean()),
        "late_life_degradation_rate": None,
        "last_observed_96h_causal_degradation_rate": None if latest_rate.empty else float(latest_rate.mean()),
        "uncertainty_method": "two-frozen-window measurement proxy + six-anchor LOAO calibration dispersion",
        "median_uncertainty_width": float(np.median(data.HI_upper - data.HI_lower)),
        "p1s_as_primary_health_indicator": "YES",
        "recommended_next_phase": "DEGRADATION_TRAJECTORY_FORECASTING_WITH_UNCERTAINTY",
        "test_targets_reopened": False,
        "post_ELTE_retuning": False,
        "raw_polarization_target_files_read": False,
    }
    (OUT / "PHI_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_figures(data, validation)
    write_report(data, validation, comparison, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

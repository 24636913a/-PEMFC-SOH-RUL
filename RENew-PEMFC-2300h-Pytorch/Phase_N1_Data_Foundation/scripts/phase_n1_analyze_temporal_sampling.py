#!/usr/bin/env python
"""Sampling-resolution, duty-cycle, ACF, and temporal feasibility analysis.

Sampling selection uses Train+Validation only. ACF and spectral evidence use
Train only. Test is shown only in explicitly labelled durability figures and
is otherwise limited to structural window counts.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram
from scipy.stats import spearmanr

from phase_n1_common import PHASE_ROOT, load_config, markdown_table, write_csv, write_text


def load_tables() -> tuple[dict[float, pd.DataFrame], pd.DataFrame]:
    table_dir = PHASE_ROOT / "outputs" / "causal_feature_tables"
    frames = {
        0.5: pd.read_csv(table_dir / "causal_features_30min.csv"),
        1.0: pd.read_csv(table_dir / "causal_features_1h.csv"),
    }
    windows = pd.read_csv(PHASE_ROOT / "outputs" / "temporal_scale_analysis" / "window_feasibility_summary.csv")
    return frames, windows


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return math.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def matched_subset(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[
        frame["split"].isin(["train", "validation"])
        & (frame["low_coverage_flag"] == 0)
        & (frame["internal_datetime_gap_count"] == 0)
        & (frame["zero_current_fraction"] < 0.20)
        & frame["actual_current_density_A_cm2_mean"].notna()
        & frame["actual_coolant_inlet_temperature_C_mean"].notna()
        & frame["actual_cathode_outlet_pressure_kPag_mean"].notna()
    ].copy()
    work["condition_j_bin"] = np.round(work["actual_current_density_A_cm2_mean"] / 0.05) * 0.05
    work["condition_temperature_bin_C"] = np.round(work["actual_coolant_inlet_temperature_C_mean"] / 5.0) * 5.0
    work["condition_pressure_bin_kPag"] = np.round(work["actual_cathode_outlet_pressure_kPag_mean"] / 10.0) * 10.0
    keys = ["condition_j_bin", "condition_temperature_bin_C", "condition_pressure_bin_kPag"]
    reference = (
        work.loc[work["split"] == "train"]
        .groupby(keys, dropna=False)["cell_voltage_mean"]
        .agg(["median", "count"])
        .reset_index()
    )
    reference = reference.loc[reference["count"] >= 8].rename(columns={"median": "train_condition_voltage_reference"})
    work = work.merge(reference[keys + ["train_condition_voltage_reference"]], on=keys, how="left")
    work["matched_condition_voltage_residual"] = work["cell_voltage_mean"] - work["train_condition_voltage_reference"]
    return work.loc[work["matched_condition_voltage_residual"].notna()].copy()


def estimate_matched_noise(data: pd.DataFrame, width_h: float) -> float:
    differences_parts: list[np.ndarray] = []
    keys = ["condition_j_bin", "condition_temperature_bin_C", "condition_pressure_bin_kPag"]
    for _, group in data.groupby(keys, dropna=False):
        ordered = group.sort_values("bin_start_ophrs2_h")
        value = ordered["matched_condition_voltage_residual"].to_numpy(dtype=float)
        age = ordered["bin_start_ophrs2_h"].to_numpy(dtype=float)
        segment = ordered["temporal_segment_id"].to_numpy(dtype=int)
        if len(value) < 2:
            continue
        pair = (
            np.isfinite(value[:-1])
            & np.isfinite(value[1:])
            & ((age[1:] - age[:-1]) <= 24.0)
            & ((age[1:] - age[:-1]) >= width_h - 1e-7)
            & (segment[:-1] == segment[1:])
        )
        if np.any(pair):
            differences_parts.append(np.diff(value)[pair])
    differences = np.concatenate(differences_parts) if differences_parts else np.asarray([], dtype=float)
    if not differences.size:
        return math.nan
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    return 1.4826 * mad / math.sqrt(2.0)


def sampling_metrics(width: float, frame: pd.DataFrame, windows: pd.DataFrame, period_h: float | None = None) -> dict[str, Any]:
    decision = frame.loc[frame["split"].isin(["train", "validation"])].copy()
    valid = decision.loc[decision["bin_valid_for_windows"] == 1]
    matched = matched_subset(frame)
    age = matched["bin_center_ophrs2_h"].to_numpy(dtype=float)
    residual = matched["matched_condition_voltage_residual"].to_numpy(dtype=float)
    rho = safe_spearman(age, residual)
    slope = float(np.polyfit(age[np.isfinite(age) & np.isfinite(residual)], residual[np.isfinite(age) & np.isfinite(residual)], 1)[0]) if np.count_nonzero(np.isfinite(age) & np.isfinite(residual)) >= 3 else math.nan
    if len(matched):
        q25, q75 = np.quantile(age[np.isfinite(age)], [0.25, 0.75])
        early = residual[age <= q25]
        late = residual[age >= q75]
        amplitude = float(np.nanmedian(early) - np.nanmedian(late))
    else:
        amplitude = math.nan
    noise = estimate_matched_noise(matched, width)
    window_rows = windows.loc[
        np.isclose(windows["sampling_resolution_hours"], width)
        & (windows["lookback_hours"] == 24)
        & (windows["horizon_hours"] == 24)
        & windows["split"].isin(["train", "validation"])
    ]
    overall_current_std = float(valid["actual_current_A_mean"].std(ddof=0)) if len(valid) else math.nan
    within_current_std = float(np.sqrt(np.nanmean(np.square(valid["actual_current_A_std"])))) if len(valid) else math.nan
    profile_preservation = 1.0 / (1.0 + within_current_std / overall_current_std) if overall_current_std > 0 else math.nan
    return {
        "sampling_resolution_hours": width,
        "selection_scope": "train+validation_only",
        "total_bins": int(len(decision)),
        "valid_bins": int(len(valid)),
        "valid_bin_fraction": len(valid) / len(decision) if len(decision) else math.nan,
        "low_coverage_bin_fraction": float((decision["low_coverage_flag"] > 0).mean()) if len(decision) else math.nan,
        "gap_affected_bin_fraction": float(((decision["gap_before_flag"] > 0) | (decision["gap_after_flag"] > 0) | (decision["internal_datetime_gap_count"] > 0)).mean()) if len(decision) else math.nan,
        "startup_affected_bin_fraction": float((decision["startup_flag"] > 0).mean()) if len(decision) else math.nan,
        "shutdown_affected_bin_fraction": float((decision["shutdown_flag"] > 0).mean()) if len(decision) else math.nan,
        "legal_windows_24h_to_24h": int(window_rows["legal_windows"].sum()),
        "median_within_bin_voltage_temporal_std_V": float(valid["cell_voltage_temporal_std"].median()),
        "median_within_bin_current_std_A": float(valid["actual_current_A_std"].median()),
        "matched_condition_residual_std_V": float(matched["matched_condition_voltage_residual"].std(ddof=0)) if len(matched) else math.nan,
        "matched_condition_local_noise_V": noise,
        "matched_voltage_age_spearman_rho": rho,
        "matched_voltage_long_term_slope_V_per_1000h": slope * 1000.0 if math.isfinite(slope) else math.nan,
        "trendability_abs_spearman": abs(rho) if math.isfinite(rho) else math.nan,
        "matched_degradation_amplitude_V": amplitude,
        "noise_to_degradation_ratio": noise / abs(amplitude) if math.isfinite(noise) and math.isfinite(amplitude) and abs(amplitude) > 0 else math.nan,
        "load_transition_count": int(decision["load_transition_count"].sum()),
        "operation_point_transition_count": int(decision["operation_point_transitions"].sum()),
        "median_operation_points_per_bin": float(decision["number_of_operation_points"].median()),
        "mean_operation_points_per_bin": float(decision["number_of_operation_points"].mean()),
        "current_profile_preservation_score": profile_preservation,
        "average_time_points_per_duty_cycle": period_h / width if period_h and period_h > 0 else math.nan,
    }


def gap_aware_acf(frame: pd.DataFrame, signal: str, width_h: float, max_lag_h: int) -> pd.DataFrame:
    train = frame.loc[
        (frame["split"] == "train")
        & (frame["low_coverage_flag"] == 0)
        & (frame["internal_datetime_gap_count"] == 0)
    ].sort_values("bin_start_ophrs2_h")
    values = pd.to_numeric(train[signal], errors="coerce").to_numpy(dtype=float)
    age = train["bin_start_ophrs2_h"].to_numpy(dtype=float)
    segment = train["temporal_segment_id"].to_numpy(dtype=int)
    max_steps = int(max_lag_h / width_h)
    rows = []
    for lag in range(max_steps + 1):
        if lag == 0:
            valid = np.isfinite(values)
            corr = 1.0 if np.count_nonzero(valid) >= 2 else math.nan
            pairs = int(np.count_nonzero(valid))
        else:
            left = values[:-lag]
            right = values[lag:]
            exact = np.isclose(age[lag:] - age[:-lag], lag * width_h, atol=1e-7)
            same_segment = segment[:-lag] == segment[lag:]
            valid = np.isfinite(left) & np.isfinite(right) & exact & same_segment
            pairs = int(np.count_nonzero(valid))
            if pairs >= 3 and np.std(left[valid]) > 0 and np.std(right[valid]) > 0:
                corr = float(np.corrcoef(left[valid], right[valid])[0, 1])
            else:
                corr = math.nan
        rows.append(
            {
                "signal": signal,
                "sampling_resolution_hours": width_h,
                "split": "train",
                "lag_steps": lag,
                "lag_hours": lag * width_h,
                "acf": corr,
                "valid_pair_count": pairs,
                "gap_policy": "pairs require exact grid spacing and same temporal segment",
            }
        )
    return pd.DataFrame(rows)


def categorical_match_periodicity(frame: pd.DataFrame, width_h: float, max_lag_h: int) -> pd.DataFrame:
    train = frame.loc[
        (frame["split"] == "train")
        & (frame["low_coverage_flag"] == 0)
        & (frame["internal_datetime_gap_count"] == 0)
    ].sort_values("bin_start_ophrs2_h")
    values = train["operation_point_mode"].to_numpy(dtype=float)
    age = train["bin_start_ophrs2_h"].to_numpy(dtype=float)
    segment = train["temporal_segment_id"].to_numpy(dtype=int)
    rows = []
    for lag in range(int(max_lag_h / width_h) + 1):
        if lag == 0:
            valid = np.isfinite(values)
            match = 1.0
        else:
            valid = (
                np.isfinite(values[:-lag])
                & np.isfinite(values[lag:])
                & np.isclose(age[lag:] - age[:-lag], lag * width_h, atol=1e-7)
                & (segment[:-lag] == segment[lag:])
            )
            match = float(np.mean(values[:-lag][valid] == values[lag:][valid])) if np.any(valid) else math.nan
        rows.append(
            {
                "signal": "operation_point_categorical_match",
                "sampling_resolution_hours": width_h,
                "split": "train",
                "lag_steps": lag,
                "lag_hours": lag * width_h,
                "acf": match,
                "valid_pair_count": int(np.count_nonzero(valid)),
                "gap_policy": "categorical equality rate; avoids treating operation-point codes as metric values",
            }
        )
    return pd.DataFrame(rows)


def empirical_periodicity(frame: pd.DataFrame, width_h: float, config: dict[str, Any]) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    train = frame.loc[
        (frame["split"] == "train")
        & (frame["low_coverage_flag"] == 0)
        & (frame["internal_datetime_gap_count"] == 0)
    ].copy()
    segment_sizes = train.groupby("temporal_segment_id").size().sort_values(ascending=False)
    if segment_sizes.empty:
        raise RuntimeError("No valid Train segment for periodicity analysis")
    longest_id = int(segment_sizes.index[0])
    longest = train.loc[train["temporal_segment_id"] == longest_id].sort_values("bin_start_ophrs2_h")
    values = longest["actual_current_density_A_cm2_mean"].to_numpy(dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    if len(values) < 48:
        raise RuntimeError("Longest Train segment is too short for duty-cycle spectrum")
    values = values - np.mean(values)
    frequency, power = periodogram(values, fs=1.0 / width_h, detrend="linear", scaling="spectrum")
    period = np.divide(1.0, frequency, out=np.full_like(frequency, np.inf), where=frequency > 0)
    min_period = float(config["temporal"]["spectral_period_min_hours"])
    max_period = float(config["temporal"]["spectral_period_max_hours"])
    spectral = pd.DataFrame({"frequency_cycles_per_h": frequency, "period_hours": period, "spectral_power": power})
    spectral = spectral.loc[(spectral["period_hours"] >= min_period) & (spectral["period_hours"] <= max_period)].sort_values("spectral_power", ascending=False)
    acf_current = gap_aware_acf(frame, "actual_current_density_A_cm2_mean", width_h, int(config["temporal"]["acf_max_lag_hours"]))
    acf_values = acf_current["acf"].to_numpy(dtype=float)
    peak_idx, _ = find_peaks(np.nan_to_num(acf_values, nan=-1.0), distance=max(1, int(2.0 / width_h)))
    acf_peaks = acf_current.iloc[peak_idx].copy()
    acf_peaks = acf_peaks.loc[(acf_peaks["lag_hours"] >= 6.0) & (acf_peaks["lag_hours"] <= 24.0)].sort_values("acf", ascending=False)
    spectral_candidates = spectral.head(12).copy()
    if acf_peaks.empty or spectral_candidates.empty:
        raise RuntimeError("Duty-cycle period unresolved: no ACF/spectral candidates")
    pair_rows = []
    for acf_row in acf_peaks.itertuples(index=False):
        nearest = spectral_candidates.iloc[(spectral_candidates["period_hours"] - acf_row.lag_hours).abs().argmin()]
        agreement = abs(float(nearest["period_hours"]) - float(acf_row.lag_hours))
        pair_rows.append(
            {
                "acf_period_h": float(acf_row.lag_hours),
                "acf_value": float(acf_row.acf),
                "spectral_period_h": float(nearest["period_hours"]),
                "spectral_power": float(nearest["spectral_power"]),
                "method_agreement_h": agreement,
                "combined_score": float(acf_row.acf) * float(nearest["spectral_power"]) / (1.0 + agreement),
            }
        )
    pair_frame = pd.DataFrame(pair_rows).sort_values("combined_score", ascending=False)
    chosen = pair_frame.iloc[0]
    confirmed = float(np.mean([chosen["acf_period_h"], chosen["spectral_period_h"]]))
    rows = []
    for rank, row in enumerate(spectral.head(10).itertuples(index=False), 1):
        rows.append({"method": "periodogram_longest_train_segment", "rank": rank, "period_hours": row.period_hours, "score": row.spectral_power, "segment_id": longest_id, "segment_points": len(longest)})
    for rank, row in enumerate(acf_peaks.head(10).itertuples(index=False), 1):
        rows.append({"method": "gap_aware_acf_peak", "rank": rank, "period_hours": row.lag_hours, "score": row.acf, "segment_id": "all_train_segments", "segment_points": row.valid_pair_count})
    rows.append({"method": "confirmed_empirical_period", "rank": 1, "period_hours": confirmed, "score": chosen["combined_score"], "segment_id": longest_id, "segment_points": len(longest), "notes": "ACF and spectrum agreement; no 2300/200 arithmetic assumption"})
    rows.append({"method": "paper_wall_clock_context", "rank": 1, "period_hours": 12.0, "score": math.nan, "segment_id": "protocol", "segment_points": math.nan, "notes": "Paper says approximately 12 h including three short and one long stop; not used to force the OpHrs2 result"})
    return pd.DataFrame(rows), confirmed, spectral.sort_values("period_hours")


def choose_sampling(comparison: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    a = comparison.loc[np.isclose(comparison["sampling_resolution_hours"], 0.5)].iloc[0]
    b = comparison.loc[np.isclose(comparison["sampling_resolution_hours"], 1.0)].iloc[0]
    criteria = {
        "legal_windows_significantly_more": bool(a["legal_windows_24h_to_24h"] >= 1.5 * b["legal_windows_24h_to_24h"]),
        "coverage_not_materially_worse": bool(a["valid_bin_fraction"] >= b["valid_bin_fraction"] - 0.02),
        "trendability_not_materially_worse": bool(a["trendability_abs_spearman"] >= 0.90 * b["trendability_abs_spearman"]),
        "noise_ratio_not_materially_worse": bool(a["noise_to_degradation_ratio"] <= 1.20 * b["noise_to_degradation_ratio"]),
        "dynamic_resolution_better": bool(a["average_time_points_per_duty_cycle"] >= 1.8 * b["average_time_points_per_duty_cycle"] and a["current_profile_preservation_score"] >= b["current_profile_preservation_score"]),
    }
    if all(criteria.values()):
        return "0.5 h", criteria
    if (not criteria["noise_ratio_not_materially_worse"]) and b["trendability_abs_spearman"] >= a["trendability_abs_spearman"]:
        return "1 h", criteria
    return "UNRESOLVED", criteria


def add_period_to_comparison(frames: dict[float, pd.DataFrame], windows: pd.DataFrame, period_h: float) -> pd.DataFrame:
    return pd.DataFrame([sampling_metrics(width, frame, windows, period_h) for width, frame in frames.items()])


def figure_test_label(ax: plt.Axes, cut80: float, xmax: float) -> None:
    ax.axvspan(cut80, xmax, color="#FEE2E2", alpha=0.38)
    ymin, ymax = ax.get_ylim()
    ax.text(cut80 + max(2, (xmax - cut80) * 0.02), ymin + 0.03 * (ymax - ymin), "TEST — visualization only, not used for selection", color="#B91C1C", fontsize=8, rotation=90, va="bottom")


def make_figures(
    frames: dict[float, pd.DataFrame],
    comparison: pd.DataFrame,
    temporal: pd.DataFrame,
    periodicity: pd.DataFrame,
    spectral: pd.DataFrame,
    period_h: float,
    windows: pd.DataFrame,
    cut80: float,
) -> None:
    figures = PHASE_ROOT / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.alpha": 0.22})
    colors = {0.5: "#2563EB", 1.0: "#F59E0B"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for width, frame in frames.items():
        train = frame.loc[frame["split"] == "train"].head(int(72 / width))
        axes[0].plot(train["bin_center_ophrs2_h"], train["cell_voltage_mean"], color=colors[width], label=f"{width:g} h", linewidth=1.2)
        axes[1].plot(train["bin_center_ophrs2_h"], train["actual_current_density_A_cm2_mean"], color=colors[width], label=f"{width:g} h", linewidth=1.2)
    axes[0].set(ylabel="10-cell mean voltage (V)", title="30 min and 1 h causal signals — Train example")
    axes[1].set(xlabel="Operating ageing (h)", ylabel="Current density (A/cm²)")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "sampling_30min_vs_1h_signal_comparison.png", dpi=180)
    plt.close(fig)

    noise_metrics = ["median_within_bin_voltage_temporal_std_V", "matched_condition_local_noise_V"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(noise_metrics))
    width_bar = 0.34
    for offset, row in enumerate(comparison.itertuples(index=False)):
        values = [getattr(row, metric) * 1000 for metric in noise_metrics]
        ax.bar(x + (offset - 0.5) * width_bar, values, width_bar, color=colors[row.sampling_resolution_hours], label=f"{row.sampling_resolution_hours:g} h")
    ax.set_xticks(x, ["Within-bin voltage std", "Matched local noise"])
    ax.set(ylabel="Voltage noise (mV)", title="Sampling-resolution noise comparison — Train + Validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "sampling_30min_vs_1h_noise_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(comparison))
    ax.bar(x, comparison["valid_bin_fraction"] * 100, color=[colors[x] for x in comparison["sampling_resolution_hours"]])
    ax.set_xticks(x, [f"{x:g} h" for x in comparison["sampling_resolution_hours"]])
    ax.set_ylim(0, 105)
    ax.set(xlabel="Sampling resolution", ylabel="Valid bins (%)", title="Coverage after the frozen 80% threshold — Train + Validation")
    for i, value in enumerate(comparison["valid_bin_fraction"] * 100):
        ax.text(i, min(103, value + 1), f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(figures / "sampling_30min_vs_1h_coverage.png", dpi=180)
    plt.close(fig)

    primary = frames[0.5]
    max_age = float(primary["bin_center_ophrs2_h"].max())
    for column, title, ylabel, filename in [
        ("cell_voltage_mean", "Average cell voltage over operating ageing", "10-cell mean voltage (V)", "average_cell_voltage_vs_operating_hours.png"),
        ("matched_condition_voltage", "Operation-point matched voltage over operating ageing", "Matched voltage residual (V)", "matched_condition_voltage_vs_operating_hours.png"),
        ("cell_voltage_std", "Cell-voltage dispersion over operating ageing", "Cross-cell std (V)", "cell_voltage_dispersion_vs_operating_hours.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.4))
        train_val = primary.loc[primary["split"].isin(["train", "validation"])]
        test = primary.loc[primary["split"] == "test"]
        ax.scatter(train_val["bin_center_ophrs2_h"], train_val[column], s=5, alpha=0.45, color="#2563EB", label="Train + Validation")
        ax.scatter(test["bin_center_ophrs2_h"], test[column], s=5, alpha=0.35, color="#DC2626", label="TEST — visualization only")
        ax.set(xlabel="Operating ageing, OpHrs2 (h)", ylabel=ylabel, title=title)
        figure_test_label(ax, cut80, max_age)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    train = primary.loc[(primary["split"] == "train") & (primary["bin_valid_for_windows"] == 1)].copy()
    segment_id = train.groupby("temporal_segment_id").size().idxmax()
    segment = train.loc[train["temporal_segment_id"] == segment_id].sort_values("bin_start_ophrs2_h")
    start_age = float(segment["bin_start_ophrs2_h"].iloc[0])
    example = segment.loc[segment["bin_start_ophrs2_h"] < start_age + period_h]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.step(example["bin_start_ophrs2_h"] - start_age, example["actual_current_density_A_cm2_mean"], where="post", color="#7C3AED")
    ax.set(xlabel="Time within empirical duty cycle (h)", ylabel="Current density (A/cm²)", title=f"Train load-profile example — empirical period {period_h:.2f} h")
    fig.tight_layout()
    fig.savefig(figures / "load_profile_example_cycle.png", dpi=180)
    plt.close(fig)

    current_acf = temporal.loc[temporal["signal"] == "actual_current_density_A_cm2_mean"]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0))
    axes[0].plot(current_acf["lag_hours"], current_acf["acf"], color="#7C3AED")
    axes[0].axvline(period_h, color="#DC2626", linestyle="--", label=f"empirical {period_h:.2f} h")
    axes[0].set(xlabel="Lag (h)", ylabel="ACF", title="Gap-aware current-density ACF — Train")
    axes[0].legend()
    axes[1].plot(spectral["period_hours"], spectral["spectral_power"], color="#059669")
    axes[1].axvline(period_h, color="#DC2626", linestyle="--")
    axes[1].set_xlim(2, 48)
    axes[1].set(xlabel="Period (h)", ylabel="Spectral power", title="Longest continuous Train segment periodogram")
    fig.tight_layout()
    fig.savefig(figures / "load_cycle_periodicity.png", dpi=180)
    plt.close(fig)

    for signal, title, filename in [
        ("cell_voltage_mean", "Average-cell voltage ACF — Train", "voltage_acf.png"),
        ("actual_current_density_A_cm2_mean", "Current-density ACF — Train", "current_density_acf.png"),
    ]:
        data = temporal.loc[temporal["signal"] == signal]
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        ax.plot(data["lag_hours"], data["acf"], color="#2563EB")
        ax.axvline(period_h, color="#DC2626", linestyle="--", label=f"empirical duty period {period_h:.2f} h")
        ax.set(xlabel="Lag (h)", ylabel="ACF", title=title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    plot_windows = windows.loc[
        windows["horizon_hours"].isin([4, 8, 12, 24])
        & windows["split"].isin(["train", "validation"])
    ].groupby(["sampling_resolution_hours", "lookback_hours", "horizon_hours"], as_index=False)["legal_windows"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    for ax, width in zip(axes, [0.5, 1.0]):
        data = plot_windows.loc[np.isclose(plot_windows["sampling_resolution_hours"], width)]
        pivot = data.pivot(index="lookback_hours", columns="horizon_hours", values="legal_windows")
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues")
        ax.set_xticks(np.arange(len(pivot.columns)), [str(x) for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [str(x) for x in pivot.index])
        ax.set(xlabel="Forecast horizon (h)", ylabel="Lookback (h)", title=f"{width:g} h resolution")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{int(pivot.iloc[i, j])}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Train + Validation legal windows", shrink=0.8)
    fig.suptitle("Legal window counts by physical lookback and horizon")
    fig.savefig(figures / "window_counts_by_lookback_horizon.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = load_config()
    frames, windows = load_tables()
    metadata = json.loads((PHASE_ROOT / "outputs" / "audits" / "foundation_run_metadata.json").read_text(encoding="utf-8"))
    cut80 = float(metadata["cut80_h"])
    periodicity, period_h, spectral = empirical_periodicity(frames[0.5], 0.5, config)
    signals = [
        "cell_voltage_mean",
        "matched_condition_voltage",
        "actual_current_density_A_cm2_mean",
        "actual_current_A_mean",
        "cell_voltage_std",
        "operation_point_mode",
    ]
    temporal_parts = [
        gap_aware_acf(frames[0.5], signal, 0.5, int(config["temporal"]["acf_max_lag_hours"]))
        for signal in signals
    ]
    temporal_parts.append(categorical_match_periodicity(frames[0.5], 0.5, int(config["temporal"]["acf_max_lag_hours"])))
    temporal = pd.concat(temporal_parts, ignore_index=True)
    comparison = add_period_to_comparison(frames, windows, period_h)
    recommendation, criteria = choose_sampling(comparison)

    sampling_dir = PHASE_ROOT / "outputs" / "sampling_comparison"
    temporal_dir = PHASE_ROOT / "outputs" / "temporal_scale_analysis"
    write_csv(comparison, sampling_dir / "sampling_resolution_comparison.csv")
    write_csv(periodicity, temporal_dir / "load_cycle_periodicity_analysis.csv")
    write_csv(temporal, temporal_dir / "temporal_dependency_analysis.csv")
    write_csv(spectral, temporal_dir / "load_cycle_periodogram.csv")

    decision_md = f"""# Sampling Resolution Decision

Recommended primary resolution: **{recommendation}**

Robustness/sensitivity resolution: **{'1 h' if recommendation == '0.5 h' else '0.5 h' if recommendation == '1 h' else 'retain both'}**

Decision scope: Train + Validation only. Test values, trends and forecasting errors were not used.

The comparison uses the same physical task at both resolutions. The reference legal-window comparison is past 24 h to future 24 h: 30 min uses L=48/H=48 and 1 h uses L=24/H=24.

Empirical duty-cycle period from Train-only ACF/spectrum agreement: **{period_h:.3f} h**.

This is the period on the `OpHrs2` operating-age axis. The paper reports about 12 h in wall-clock time and explicitly includes three short and one longer stop; removing stop time explains why the operating-age period is shorter.

## Frozen-rule checks

{markdown_table(pd.DataFrame([{'criterion': key, 'passed_for_30min': value} for key, value in criteria.items()]))}

## Evidence

{markdown_table(comparison.round(6))}

No deep or classical forecasting model was trained. If the recommendation is unresolved, both resolutions must continue only to a small baseline diagnostic in Phase N2.
"""
    write_text(decision_md, sampling_dir / "SAMPLING_RESOLUTION_DECISION.md")

    temporal_summary = {
        "empirical_duty_cycle_period_h": period_h,
        "sampling_recommendation": recommendation,
        "lookback_primary_h": 24,
        "lookback_sensitivity_h": [12, 48],
        "horizon_primary_h": 12,
        "horizon_sensitivity_h": [4, 8, 24],
        "basis": f"24 h covers about {24.0 / period_h:.2f} empirical OpHrs2 duty cycles; 12 h horizon covers about {12.0 / period_h:.2f} cycles; all choices remain first-round candidates",
        "acf_scope": "train_only",
        "sampling_scope": "train+validation_only",
        "test_used_for_selection": False,
    }
    (temporal_dir / "temporal_scale_recommendation.json").write_text(json.dumps(temporal_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(frames, comparison, temporal, periodicity, spectral, period_h, windows, cut80)
    print(json.dumps(temporal_summary, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PHASE N1 TEMPORAL/SAMPLING FAILED: {exc}", file=sys.stderr, flush=True)
        raise

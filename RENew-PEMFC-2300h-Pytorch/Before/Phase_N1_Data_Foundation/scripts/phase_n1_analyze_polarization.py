#!/usr/bin/env python
"""Analyze nominal polarization checkpoints and construct sparse SOH anchors.

j* selection is restricted to Train+Validation checkpoints. Test checkpoints
are retained only as explicitly labelled future-validation anchors.
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
from scipy.stats import spearmanr

from phase_n1_common import (
    PHASE_ROOT,
    assign_checkpoint_split,
    load_config,
    markdown_table,
    normalize_numeric,
    polarization_stage,
    write_csv,
    write_text,
)


CELL_COLUMNS = ["U1 [V]", "U2 [V]", "U3 [V]", "U4 [V]", "U5 [V]", "U7 [V]", "U8 [V]", "U9 [V]", "U10 [V]", "U11 [V]"]
CONFIG_COLUMNS = [
    "Number of cells",
    "Membrane active area [cm2]",
    "I.S.act [A]",
    "j.S.act [A/cm2]",
    "U.S.AveCell [V]",
    "Stoic.S.A [-]",
    "Stoic.S.A.set [-]",
    "Stoic.S.C [-]",
    "Stoic.S.C.set [-]",
    "p.So.A [kPag]",
    "p.So.C [kPag]",
    "T.Si.A [°C]",
    "T.Si.C [°C]",
    "T.Si.CL [°C]",
    "T.So.CL [°C]",
]


def log(message: str) -> None:
    print(message, flush=True)


def read_foundation_metadata() -> dict[str, Any]:
    path = PHASE_ROOT / "outputs" / "audits" / "foundation_run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError("Run phase_n1_build_foundation.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def unique_numeric_summary(values: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        return {"min": math.nan, "median": math.nan, "max": math.nan, "unique_values": ""}
    arr = np.concatenate(values)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"min": math.nan, "median": math.nan, "max": math.nan, "unique_values": ""}
    unique = np.unique(np.round(arr, 6))
    values_text = "|".join(f"{x:g}" for x in unique[:20]) + ("|..." if len(unique) > 20 else "")
    return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max()), "unique_values": values_text}


def analyze_polcurve(
    path: Path,
    candidates: list[float],
    tolerance: float,
    cut60: float,
    cut80: float,
    chunk_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    stage, age_h, after = polarization_stage(path)
    split = assign_checkpoint_split(age_h, cut60, cut80)
    header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
    requested = list(dict.fromkeys([*CONFIG_COLUMNS, *CELL_COLUMNS]))
    missing = [col for col in requested if col not in header]
    permitted_test_metadata_missing = {"Membrane active area [cm2]"} if split == "test" else set()
    forbidden_missing = [col for col in missing if col not in permitted_test_metadata_missing]
    if forbidden_missing:
        raise KeyError(f"{path.name} missing required polarization columns: {forbidden_missing}")
    required = [col for col in requested if col in header]
    candidate_voltage: dict[float, list[np.ndarray]] = {j: [] for j in candidates}
    candidate_exported: dict[float, list[np.ndarray]] = {j: [] for j in candidates}
    candidate_actual_j: dict[float, list[np.ndarray]] = {j: [] for j in candidates}
    config_values: dict[str, list[np.ndarray]] = {col: [] for col in CONFIG_COLUMNS}
    curve_parts: list[pd.DataFrame] = []
    active_exported_abs_error: list[np.ndarray] = []
    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        usecols=required,
        chunksize=chunk_rows,
        low_memory=False,
    ):
        for col in required:
            chunk[col] = normalize_numeric(chunk[col])
        cells = chunk[CELL_COLUMNS].to_numpy(dtype=float)
        active_mean = np.nanmean(cells, axis=1)
        exported = chunk["U.S.AveCell [V]"].to_numpy(dtype=float)
        active_exported_abs_error.append(np.abs(active_mean - exported)[::10].astype(np.float32, copy=True))
        actual_j = chunk["j.S.act [A/cm2]"].to_numpy(dtype=float)
        for col in CONFIG_COLUMNS:
            if col in chunk.columns:
                config_values[col].append(chunk[col].dropna().to_numpy(dtype=np.float32, copy=True)[::20])
        for target in candidates:
            mask = np.isfinite(actual_j) & (np.abs(actual_j - target) <= tolerance)
            if np.any(mask):
                candidate_voltage[target].append(active_mean[mask].astype(np.float32, copy=True))
                candidate_exported[target].append(exported[mask].astype(np.float32, copy=True))
                candidate_actual_j[target].append(actual_j[mask].astype(np.float32, copy=True))
        curve_mask = np.isfinite(actual_j) & np.isfinite(active_mean) & (actual_j >= 0) & (actual_j <= 2.0)
        if np.any(curve_mask):
            curve = pd.DataFrame({"j": actual_j[curve_mask], "voltage": active_mean[curve_mask]})
            curve["j_bin"] = np.round(curve["j"] / 0.02) * 0.02
            curve_parts.append(curve.groupby("j_bin", as_index=False)["voltage"].agg(["median", "count"]).reset_index())
    rows: list[dict[str, Any]] = []
    for target in candidates:
        voltage = np.concatenate(candidate_voltage[target]) if candidate_voltage[target] else np.asarray([], dtype=float)
        exported = np.concatenate(candidate_exported[target]) if candidate_exported[target] else np.asarray([], dtype=float)
        actual_j = np.concatenate(candidate_actual_j[target]) if candidate_actual_j[target] else np.asarray([], dtype=float)
        rows.append(
            {
                "filename": path.name,
                "stage": stage,
                "ageing_hour": age_h,
                "after_air_air_start": after,
                "split": split,
                "target_jstar_A_cm2": target,
                "actual_j_mean_A_cm2": float(np.mean(actual_j)) if actual_j.size else math.nan,
                "actual_j_median_A_cm2": float(np.median(actual_j)) if actual_j.size else math.nan,
                "actual_j_min_A_cm2": float(np.min(actual_j)) if actual_j.size else math.nan,
                "actual_j_max_A_cm2": float(np.max(actual_j)) if actual_j.size else math.nan,
                "voltage_mean_V": float(np.mean(voltage)) if voltage.size else math.nan,
                "voltage_median_V": float(np.median(voltage)) if voltage.size else math.nan,
                "voltage_std_V": float(np.std(voltage, ddof=0)) if voltage.size else math.nan,
                "valid_sample_count": int(voltage.size),
                "exported_voltage_median_V": float(np.median(exported)) if exported.size else math.nan,
                "active_mean_vs_exported_mae_V": float(np.mean(np.abs(voltage - exported))) if voltage.size and exported.size else math.nan,
                "selection_eligible": split in {"train", "validation"} and not after,
                "test_policy": "TEST — visualization only, not used for selection" if split == "test" else "selection permitted by split policy",
            }
        )
    config_summary: dict[str, Any] = {
        "filename": path.name,
        "stage": stage,
        "ageing_hour": age_h,
        "after_air_air_start": after,
        "split": split,
        "missing_configuration_fields": "|".join(missing),
    }
    for col, values in config_values.items():
        summary = unique_numeric_summary(values)
        safe = (
            col.replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
            .replace(".", "_")
            .replace("°", "deg")
        )
        config_summary[f"{safe}_min"] = summary["min"]
        config_summary[f"{safe}_median"] = summary["median"]
        config_summary[f"{safe}_max"] = summary["max"]
        if col in {"Number of cells", "Membrane active area [cm2]"}:
            config_summary[f"{safe}_unique"] = summary["unique_values"]
    sampled_error = np.concatenate(active_exported_abs_error) if active_exported_abs_error else np.asarray([], dtype=float)
    config_summary["active_10cell_mean_vs_exported_mae_V"] = float(np.mean(sampled_error)) if sampled_error.size else math.nan
    if curve_parts:
        combined_curve = pd.concat(curve_parts, ignore_index=True)
        weighted = (
            combined_curve.assign(weighted=lambda x: x["median"] * x["count"])
            .groupby("j_bin", as_index=False)
            .agg(weighted_sum=("weighted", "sum"), count=("count", "sum"))
        )
        weighted["voltage_median_V"] = weighted["weighted_sum"] / weighted["count"]
        curve_result = weighted[["j_bin", "voltage_median_V", "count"]].copy()
        curve_result["filename"] = path.name
        curve_result["stage"] = stage
        curve_result["ageing_hour"] = age_h
        curve_result["after_air_air_start"] = after
        curve_result["split"] = split
    else:
        curve_result = pd.DataFrame()
    return rows, config_summary, curve_result


def inspect_sensitivity_stack_counts(raw_root: Path, chunk_rows: int) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_root.glob("PartialSensitivityTest*.csv")):
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
        if "Number of cells" not in header:
            rows.append({"filename": path.name, "number_of_cells_unique": "missing", "rows_scanned": 0})
            continue
        values: set[float] = set()
        row_count = 0
        for chunk in pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            usecols=["Number of cells"],
            chunksize=chunk_rows,
            low_memory=False,
        ):
            numeric = normalize_numeric(chunk["Number of cells"]).dropna()
            values.update(float(x) for x in numeric.unique())
            row_count += len(chunk)
        rows.append(
            {
                "filename": path.name,
                "stage": "BoL" if "BoL" in path.stem else "2300h",
                "number_of_cells_unique": "|".join(f"{x:g}" for x in sorted(values)),
                "rows_scanned": row_count,
            }
        )
    return pd.DataFrame(rows)


def normalized_metric(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty or float(finite.max() - finite.min()) <= 1e-15:
        result = pd.Series(1.0, index=values.index, dtype=float)
    else:
        result = (numeric - finite.min()) / (finite.max() - finite.min())
    return result if higher_is_better else 1.0 - result


def score_candidates(points: pd.DataFrame, candidates: list[float], lower_limit: float, minimum_margin: float) -> pd.DataFrame:
    eligible = points.loc[points["selection_eligible"]].copy()
    eligible_ages = sorted(eligible["ageing_hour"].dropna().unique())
    rows = []
    for target in candidates:
        data = eligible.loc[eligible["target_jstar_A_cm2"] == target].sort_values("ageing_hour")
        data = data.loc[data["valid_sample_count"] > 0]
        ages = data["ageing_hour"].to_numpy(dtype=float)
        voltage = data["voltage_median_V"].to_numpy(dtype=float)
        difference = np.diff(voltage)
        abs_movement = float(np.abs(difference).sum())
        positive_movement = float(np.clip(difference, 0, None).sum())
        monotonicity = 1.0 - positive_movement / abs_movement if abs_movement > 0 else math.nan
        non_increasing = float(np.mean(difference <= 0)) if difference.size else math.nan
        spearman = float(spearmanr(ages, voltage).statistic) if len(data) >= 3 else math.nan
        pearson = float(np.corrcoef(ages, voltage)[0, 1]) if len(data) >= 3 else math.nan
        first_v = float(voltage[0]) if voltage.size else math.nan
        last_v = float(voltage[-1]) if voltage.size else math.nan
        relative_loss = (first_v - last_v) / first_v if first_v else math.nan
        median_std = float(data["voltage_std_V"].median()) if len(data) else math.nan
        air_before = points.loc[
            (points["target_jstar_A_cm2"] == target)
            & (points["ageing_hour"] == 1500)
            & (~points["after_air_air_start"])
        ]
        air_after = points.loc[
            (points["target_jstar_A_cm2"] == target)
            & (points["ageing_hour"] == 1500)
            & points["after_air_air_start"]
        ]
        air_shift = (
            float(air_after["voltage_median_V"].iloc[0] - air_before["voltage_median_V"].iloc[0])
            if len(air_before) and len(air_after)
            else math.nan
        )
        train_points = int((data["split"] == "train").sum())
        validation_points = int((data["split"] == "validation").sum())
        margin = float(voltage.min() - lower_limit) if voltage.size else math.nan
        coverage = len(data) / len(eligible_ages) if eligible_ages else 0.0
        hard_gate = bool(
            coverage >= 0.90
            and train_points >= 3
            and validation_points >= 2
            and margin >= minimum_margin
            and math.isfinite(median_std)
        )
        rows.append(
            {
                "target_jstar_A_cm2": target,
                "eligible_checkpoint_count": int(len(data)),
                "eligible_checkpoint_total": int(len(eligible_ages)),
                "coverage": coverage,
                "train_checkpoint_count": train_points,
                "validation_checkpoint_count": validation_points,
                "voltage_mean_across_checkpoints_V": float(np.mean(voltage)) if voltage.size else math.nan,
                "median_repeated_measurement_std_V": median_std,
                "ageing_pearson_r": pearson,
                "ageing_spearman_rho": spearman,
                "monotonicity_score": monotonicity,
                "non_increasing_fraction": non_increasing,
                "relative_voltage_loss": relative_loss,
                "absolute_voltage_loss_V": first_v - last_v if voltage.size else math.nan,
                "air_air_start_shift_1500h_V": air_shift,
                "air_air_start_sensitivity_abs_V": abs(air_shift) if math.isfinite(air_shift) else math.nan,
                "minimum_distance_to_500mV_limit_V": margin,
                "degradation_to_repeat_noise_ratio": (first_v - last_v) / median_std if median_std > 0 else math.nan,
                "hard_gate_pass": hard_gate,
                "selection_scope": "train+validation_only",
            }
        )
    result = pd.DataFrame(rows)
    passing = result.loc[result["hard_gate_pass"]].copy()
    result["selection_score"] = math.nan
    if not passing.empty:
        score = pd.concat(
            [
                normalized_metric(passing["coverage"], True),
                normalized_metric(passing["median_repeated_measurement_std_V"], False),
                normalized_metric(passing["monotonicity_score"], True),
                normalized_metric(passing["ageing_spearman_rho"].abs(), True),
                normalized_metric(passing["air_air_start_sensitivity_abs_V"], False),
                normalized_metric(passing["degradation_to_repeat_noise_ratio"], True),
                normalized_metric(passing["minimum_distance_to_500mV_limit_V"], True),
            ],
            axis=1,
        ).mean(axis=1)
        result.loc[passing.index, "selection_score"] = score
    result["selection_rank"] = result["selection_score"].rank(ascending=False, method="min")
    return result.sort_values(["hard_gate_pass", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def select_anchors(candidate_summary: pd.DataFrame) -> tuple[float, float]:
    passing = candidate_summary.loc[candidate_summary["hard_gate_pass"] & candidate_summary["selection_score"].notna()]
    if len(passing) < 2:
        raise RuntimeError("SOH anchor unresolved: fewer than two j* candidates pass hard gates")
    primary = float(passing.iloc[0]["target_jstar_A_cm2"])
    secondary = float(passing.iloc[1]["target_jstar_A_cm2"])
    top_gap = float(passing.iloc[0]["selection_score"] - passing.iloc[1]["selection_score"])
    if not math.isfinite(top_gap):
        raise RuntimeError("SOH anchor unresolved: non-finite selection score")
    return primary, secondary


def build_soh_anchors(points: pd.DataFrame, primary: float, reference_voltage: float) -> pd.DataFrame:
    anchors = points.loc[points["target_jstar_A_cm2"] == primary].copy()
    anchors = anchors.sort_values(["ageing_hour", "after_air_air_start"]).reset_index(drop=True)
    anchors["checkpoint"] = anchors["stage"]
    anchors["jstar_A_cm2"] = primary
    anchors["voltage_anchor_V"] = anchors["voltage_median_V"]
    anchors["reference_voltage_V"] = reference_voltage
    anchors["normalized_health_value"] = anchors["voltage_anchor_V"] / reference_voltage
    anchors["soh_candidate_A"] = anchors["normalized_health_value"]
    anchors["hi_candidate_B"] = 1.0 - (reference_voltage - anchors["voltage_anchor_V"]) / reference_voltage
    anchors["configuration"] = "10-cell durability stack; 280 cm2; nominal RealHyFC polarization"
    anchors["air_air_start_status"] = np.where(anchors["after_air_air_start"], "after Air-Air start", "nominal before Air-Air start")
    anchors["uncertainty_std_V"] = anchors["voltage_std_V"]
    anchors["test_use_restriction"] = np.where(
        anchors["split"] == "test",
        "TEST — future validation anchor only; not used for j* or formula selection",
        "permitted for Phase N1 selection/reference",
    )
    columns = [
        "ageing_hour", "checkpoint", "jstar_A_cm2", "voltage_anchor_V", "reference_voltage_V",
        "normalized_health_value", "soh_candidate_A", "hi_candidate_B", "configuration",
        "air_air_start_status", "split", "uncertainty_std_V", "valid_sample_count",
        "test_use_restriction", "filename",
    ]
    return anchors[columns]


def make_figures(
    curves: pd.DataFrame,
    points: pd.DataFrame,
    candidates: pd.DataFrame,
    anchors: pd.DataFrame,
    primary: float,
    secondary: float,
    cut80: float,
) -> None:
    figure_dir = PHASE_ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.alpha": 0.22})

    selection_curves = curves.loc[(curves["split"].isin(["train", "validation"])) & (~curves["after_air_air_start"])].copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for stage, group in selection_curves.groupby("stage", sort=False):
        ax.plot(group["j_bin"], group["voltage_median_V"], linewidth=1.6, label=stage)
    ax.set(xlabel="Current density (A/cm²)", ylabel="10-cell mean voltage (V)", title="Nominal polarization curves — Train + Validation checkpoints")
    ax.set_xlim(0, 1.85)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "polarization_curves_development_checkpoints.png", dpi=180)
    plt.close(fig)

    eligible = points.loc[points["selection_eligible"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for target, group in eligible.groupby("target_jstar_A_cm2"):
        emphasis = target in {primary, secondary}
        label = f"j*={target:g}" + (" (primary)" if target == primary else " (secondary)" if target == secondary else "")
        ax.errorbar(
            group["ageing_hour"], group["voltage_median_V"], yerr=group["voltage_std_V"],
            marker="o", linewidth=2.3 if emphasis else 1.1, alpha=1.0 if emphasis else 0.65, label=label,
        )
    ax.set(xlabel="Operating ageing (h)", ylabel="Voltage (V)", title="Candidate j* voltage — selection data only")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "candidate_jstar_voltage_vs_ageing.png", dpi=180)
    plt.close(fig)

    ordered = candidates.sort_values("target_jstar_A_cm2")
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bars = ax.bar(x, ordered["monotonicity_score"], color=["#9CA3AF" if j not in {primary, secondary} else "#2563EB" if j == primary else "#F59E0B" for j in ordered["target_jstar_A_cm2"]])
    ax.set_xticks(x, [f"{j:g}" for j in ordered["target_jstar_A_cm2"]])
    ax.set_ylim(0, 1.05)
    ax.set(xlabel="Candidate j* (A/cm²)", ylabel="Monotonicity score", title="Candidate j* monotonicity — Train + Validation")
    for bar, value in zip(bars, ordered["monotonicity_score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, min(1.02, value + 0.02), f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "candidate_jstar_monotonicity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    normal = anchors.loc[~anchors["air_air_start_status"].str.startswith("after")]
    train_val = normal.loc[normal["split"].isin(["train", "validation"])]
    test = normal.loc[normal["split"] == "test"]
    ax.plot(train_val["ageing_hour"], train_val["normalized_health_value"], marker="o", linewidth=2.0, color="#2563EB", label="Train + Validation anchors")
    if len(test):
        ax.plot(test["ageing_hour"], test["normalized_health_value"], marker="s", linestyle="--", color="#DC2626", label="TEST — visualization only")
        ax.axvspan(cut80, max(test["ageing_hour"].max(), cut80), color="#FEE2E2", alpha=0.45)
        ax.text(cut80 + 8, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.8, "TEST — visualization only, not used for selection", color="#B91C1C", fontsize=8, rotation=90, va="bottom")
    after = anchors.loc[anchors["air_air_start_status"].str.startswith("after")]
    if len(after):
        ax.scatter(after["ageing_hour"], after["normalized_health_value"], marker="^", color="#059669", label="After Air-Air start")
    ax.set(xlabel="Operating ageing (h)", ylabel="Sparse normalized health", title=f"Sparse SOH reference anchors at j*={primary:g} A/cm²")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "soh_reference_anchors.png", dpi=180)
    plt.close(fig)


def main() -> None:
    config = load_config()
    raw_root = Path(config["raw_data_root"])
    metadata = read_foundation_metadata()
    cut60 = float(metadata["cut60_h"])
    cut80 = float(metadata["cut80_h"])
    settings = config["polarization"]
    candidates = [float(x) for x in settings["candidate_jstar_A_cm2"]]
    tolerance = float(settings["current_density_tolerance_A_cm2"])
    pol_paths = sorted(raw_root.glob("Polcurve_Nominal_RHFC_*.csv"), key=lambda p: (polarization_stage(p)[1], polarization_stage(p)[2]))
    if len(pol_paths) != 11:
        raise RuntimeError(f"Expected 11 nominal polarization CSV files including after-Air-Air repeats; found {len(pol_paths)}")
    point_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    for index, path in enumerate(pol_paths, 1):
        log(f"[polarization {index:02d}/{len(pol_paths)}] {path.name}")
        rows, config_row, curve = analyze_polcurve(
            path,
            candidates,
            tolerance,
            cut60,
            cut80,
            int(config["chunk_rows"]),
        )
        point_rows.extend(rows)
        config_rows.append(config_row)
        if not curve.empty:
            curve_frames.append(curve)
    points = pd.DataFrame(point_rows)
    configs = pd.DataFrame(config_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    sensitivity_counts = inspect_sensitivity_stack_counts(raw_root, int(config["chunk_rows"]))
    candidate_summary = score_candidates(
        points,
        candidates,
        float(settings["lower_voltage_limit_V"]),
        float(settings["minimum_lower_limit_margin_V"]),
    )
    primary, secondary = select_anchors(candidate_summary)
    reference_row = points.loc[
        (points["target_jstar_A_cm2"] == primary)
        & (points["ageing_hour"] == 0)
        & (~points["after_air_air_start"])
    ]
    if len(reference_row) != 1:
        raise RuntimeError("SOH reference unresolved: expected exactly one BoL nominal polarization anchor")
    reference_voltage = float(reference_row["voltage_median_V"].iloc[0])
    anchors = build_soh_anchors(points, primary, reference_voltage)

    output = PHASE_ROOT / "outputs" / "soh_anchor_analysis"
    write_csv(points, output / "polarization_checkpoint_jstar_measurements.csv")
    write_csv(configs, output / "polarization_configuration_summary.csv")
    write_csv(sensitivity_counts, output / "sensitivity_stack_configuration_summary.csv")
    write_csv(curves, output / "polarization_curve_points_for_plot.csv")
    write_csv(candidate_summary, output / "polarization_soh_anchor_candidates.csv")
    write_csv(anchors, output / "soh_reference_anchors.csv")

    pol_cells = set(configs["Number_of_cells_median"].dropna().round(6).tolist())
    pol_area = set(configs["Membrane_active_area_cm2_median"].dropna().round(6).tolist())
    selection_configs = configs.loc[configs["split"].isin(["train", "validation"])]
    selection_cells = set(selection_configs["Number_of_cells_median"].dropna().round(6).tolist())
    selection_area = set(selection_configs["Membrane_active_area_cm2_median"].dropna().round(6).tolist())
    test_area_missing_files = configs.loc[
        (configs["split"] == "test")
        & configs["missing_configuration_fields"].str.contains("Membrane active area", regex=False),
        "filename",
    ].tolist()
    pol_compatible = selection_cells == {10.0} and selection_area == {280.0} and pol_cells == {10.0}
    bol_sensitivity = set(sensitivity_counts.loc[sensitivity_counts["stage"] == "BoL", "number_of_cells_unique"])
    aged_sensitivity = set(sensitivity_counts.loc[sensitivity_counts["stage"] == "2300h", "number_of_cells_unique"])
    compatibility_status = "CONDITIONAL" if pol_compatible else "UNRESOLVED"
    compatibility_md = f"""# Polarization Configuration Compatibility

Overall BoL-versus-durability answer: **{compatibility_status}**

## Nominal polarization SOH series

**YES — directly comparable on a per-cell basis.** All 11 nominal polarization files, including BoL, report `Number of cells = 10` and `Membrane active area = 280 cm²`. The recomputed mean of the 10 active voltage channels agrees with `U.S.AveCell [V]`. The source paper states that the durability test uses a 10-cell generic HyFaB stack with 280 cm² active area and a project-specific MEA, and that the nominal RealHyFC conditions apply to polarization curves and the durability test.

Nominal protocol from the source paper: anode/cathode inlet temperature 75 °C, coolant 70 °C, humidity 50%/35%, outlet pressure 160/150 kPaG, stoichiometry 1.5/1.7, 16 load points from 0 to 1.8 A/cm² in descending and ascending order, 2 A/s ramps, 5 min dwell at j≥0.3 A/cm², 2 min below 0.3 A/cm², and 30 min conditioning at 1 A/cm².

Therefore the BoL nominal polarization voltage at the selected j* is a legal reference for the sparse SOH series.

## BoL sensitivity/EIS series

**NO for unqualified absolute stack-level comparison; configuration-aware interpretation is required.** The paper explicitly states that the BoL full-sensitivity/EIS study used a separate 8-cell stack while the durability stack used 10 cells. The released BoL sensitivity CSVs nevertheless export `Number of cells = 10`, creating a metadata-versus-paper conflict. That field is therefore not accepted as proof that the physical BoL sensitivity stack was the 10-cell durability stack. The paper also states that the two stacks used identical MEAs from the same production batch and the same 280 cm² hardware family, but they remain separate physical stacks. Per-cell sensitivity trends may be compared only with an explicit configuration caveat; absolute stack voltage and unqualified BoL-to-2300 h health normalization must not mix these records.

## Raw configuration checks

Polarization cell-count values: `{sorted(pol_cells)}`  
Polarization active-area values: `{sorted(pol_area)}`  
Test files omitting the active-area field: `{test_area_missing_files}`; this omission is recorded and not imputed from Test. The 280 cm² protocol value comes from Train+Validation CSV metadata and the source paper.  
BoL sensitivity exported cell-count values: `{sorted(bol_sensitivity)}` (conflicts with the paper's physical 8-cell description)  
2300 h sensitivity cell-count values: `{sorted(aged_sensitivity)}`

Evidence source: original nominal polarization CSV metadata and source paper pages 2–4. No Test performance was used in this configuration decision.
"""
    write_text(compatibility_md, output / "polarization_configuration_compatibility.md")

    top = candidate_summary.iloc[0]
    selection_md = f"""# SOH Anchor Selection

Status: **RESOLVED**

Primary j*: **{primary:g} A/cm²**  
Secondary j* for sensitivity analysis: **{secondary:g} A/cm²**

The decision uses only non-after-Air-Air Train + Validation checkpoints. Test checkpoints at 2000 h and 2300 h do not contribute to coverage, ranking, monotonicity, correlation, noise, Air-Air sensitivity or the normalization formula.

Candidates first pass hard gates for coverage, at least three Train and two Validation checkpoints, and at least {float(settings['minimum_lower_limit_margin_V']):.3f} V distance above the 500 mV lower limit. Passing candidates are then ranked with equal fixed contributions from coverage, repeated-measurement std, monotonicity, |Spearman rho|, 1500 h Air-Air sensitivity, degradation-to-repeat-noise ratio and lower-limit margin. These weights were fixed before viewing Test anchors.

Primary metrics: coverage `{top['coverage']:.3f}`, Spearman rho `{top['ageing_spearman_rho']:.3f}`, monotonicity `{top['monotonicity_score']:.3f}`, non-increasing fraction `{top['non_increasing_fraction']:.3f}`, repeated std `{top['median_repeated_measurement_std_V']:.6f} V`, margin to 500 mV `{top['minimum_distance_to_500mV_limit_V']:.4f} V`.

## Candidate comparison

{markdown_table(candidate_summary.round(6))}

## SOH definition

`SOH_V(t_i) = V(j*, t_i) / V(j*, t_ref)` with `t_ref = BoL` from the same 10-cell nominal polarization configuration.

The requested Candidate B, `1 - [V_ref - V(t_i)] / V_ref`, simplifies exactly to Candidate A. It is retained as an algebraic cross-check, not treated as an independent label.

Only sparse characterization anchors are created. No hourly or 30 min SOH interpolation is generated. No EOL-normalized SOH and no `RUL = 2300 - t` label are used.
"""
    write_text(selection_md, output / "SOH_ANCHOR_SELECTION.md")
    make_figures(curves, points, candidate_summary, anchors, primary, secondary, cut80)
    summary = {
        "primary_jstar_A_cm2": primary,
        "secondary_jstar_A_cm2": secondary,
        "reference_voltage_V": reference_voltage,
        "configuration_status": compatibility_status,
        "polarization_per_cell_comparable": pol_compatible,
        "test_used_for_selection": False,
    }
    (output / "soh_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PHASE N1 POLARIZATION FAILED: {exc}", file=sys.stderr, flush=True)
        raise

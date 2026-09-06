"""Render factual Phase 3D representation diagnostics and paired bootstrap results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase3d_experiment import _representation_summary


BOOTSTRAP_REPLICATES = 2000
BLOCK_LENGTH_HOURS = 24
BOOTSTRAP_SEED = 42


def _markdown(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[column]) else "---" for column in columns) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f"{float(value):.6f}" if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, bool) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _plot_skill(per_horizon: pd.DataFrame, fc: str, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    for representation, group in per_horizon.loc[per_horizon["fc"] == fc].groupby("representation", sort=True):
        group = group.sort_values("horizon")
        axis.plot(group["horizon"], group["Skill_Best_mean"], marker="o", linewidth=1.65, label=representation)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    axis.set_xlabel("forecast horizon h (hours)")
    axis.set_ylabel("mean Skill_Best")
    axis.set_title(f"Phase-3D H=24 representation validation — {fc}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"skill_best_absolute_vs_relative_{fc}.png", dpi=180)
    plt.close(figure)


def _plot_shift(shifts: pd.DataFrame, fc: str, output_dir: Path) -> None:
    summary = shifts.loc[shifts["fc"] == fc].groupby(["representation", "lag_hours"], sort=True)["wasserstein_distance_V"].mean().reset_index()
    figure, axis = plt.subplots(figsize=(8.6, 4.6))
    for representation, group in summary.groupby("representation", sort=True):
        group = group.sort_values("lag_hours")
        axis.plot(group["lag_hours"], group["wasserstein_distance_V"], marker="o", linewidth=1.65, label=representation)
    axis.set_xlabel("history lag k (hours; 0 = prediction origin)")
    axis.set_ylabel("train-validation Wasserstein distance (V)")
    axis.set_title(f"Phase-3D representation shift by lag — {fc}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"representation_shift_by_lag_{fc}.png", dpi=180)
    plt.close(figure)


def _paired_error_matrices(predictions: pd.DataFrame, fc: str, horizons: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    key = ["fold", "origin_row", "target_time_h", "horizon"]
    subset = predictions.loc[(predictions["fc"] == fc) & (predictions["horizon"].isin(horizons))]
    arrays = []
    for fold in sorted(subset["fold"].unique()):
        absolute = subset.loc[(subset["fold"] == fold) & (subset["representation"] == "absolute"), key + ["actual_Utot", "pred_residual_mimo_lstm"]]
        relative = subset.loc[(subset["fold"] == fold) & (subset["representation"] == "relative"), key + ["actual_Utot", "pred_residual_mimo_lstm"]]
        paired = relative.merge(absolute, on=key, how="inner", validate="one_to_one", suffixes=("_relative", "_absolute"))
        if len(paired) != len(relative) or len(paired) != len(absolute):
            raise RuntimeError(f"Unmatched Phase-3D predictions for {fc}, {fold}")
        if not np.allclose(paired["actual_Utot_relative"], paired["actual_Utot_absolute"], atol=0.0, rtol=0.0):
            raise RuntimeError("Relative/absolute forecast targets differ")
        paired = paired.sort_values(["origin_row", "horizon"])
        origins = paired["origin_row"].nunique()
        if len(paired) != origins * len(horizons):
            raise RuntimeError("Each paired origin must contain all requested horizons")
        rel_sq = (paired["pred_residual_mimo_lstm_relative"].to_numpy() - paired["actual_Utot_relative"].to_numpy()) ** 2
        abs_sq = (paired["pred_residual_mimo_lstm_absolute"].to_numpy() - paired["actual_Utot_absolute"].to_numpy()) ** 2
        arrays.append((rel_sq.reshape(origins, len(horizons)), abs_sq.reshape(origins, len(horizons))))
    if len(arrays) != 3:
        raise RuntimeError("Phase-3D paired bootstrap requires all three frozen folds")
    return arrays


def _moving_block_bootstrap(errors: list[tuple[np.ndarray, np.ndarray]], seed: int) -> dict[str, float | bool | int]:
    """Delta is RMSE(relative) minus RMSE(absolute); negative favors relative."""
    denominator = sum(relative.size for relative, _ in errors)
    relative_mse = sum(float(relative.sum()) for relative, _ in errors) / denominator
    absolute_mse = sum(float(absolute.sum()) for _, absolute in errors) / denominator
    point_delta = float(np.sqrt(relative_mse) - np.sqrt(absolute_mse))
    rng = np.random.default_rng(seed)
    relative_sum = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    absolute_sum = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for relative, absolute in errors:
        origin_count = relative.shape[0]
        blocks = int(np.ceil(origin_count / BLOCK_LENGTH_HOURS))
        starts = rng.integers(0, origin_count, size=(BOOTSTRAP_REPLICATES, blocks))
        offsets = np.arange(BLOCK_LENGTH_HOURS, dtype=np.int64)
        indices = ((starts[:, :, None] + offsets[None, None, :]) % origin_count).reshape(BOOTSTRAP_REPLICATES, -1)[:, :origin_count]
        relative_sum += relative[indices].sum(axis=(1, 2))
        absolute_sum += absolute[indices].sum(axis=(1, 2))
    deltas = np.sqrt(relative_sum / denominator) - np.sqrt(absolute_sum / denominator)
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return {
        "Delta_RMSE_relative_minus_absolute": point_delta,
        "CI95_low": float(lower), "CI95_high": float(upper),
        "relative_better_probability": float(np.mean(deltas < 0.0)),
        "CI_excludes_zero": bool(upper < 0.0 or lower > 0.0),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "block_length_hours": BLOCK_LENGTH_HOURS,
    }


def paired_bootstrap_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fc_index, fc in enumerate(("FC1", "FC2"), start=1):
        horizon_sets = [(f"h{h}", (h,)) for h in range(1, 25)] + [("late_h13_h24_pooled", tuple(range(13, 25)))]
        for set_index, (name, horizons) in enumerate(horizon_sets, start=1):
            output = _moving_block_bootstrap(_paired_error_matrices(predictions, fc, horizons), BOOTSTRAP_SEED * 100000 + fc_index * 1000 + set_index)
            rows.append({"fc": fc, "horizon_set": name, "interpretation": "negative delta favors relative", **output})
    return pd.DataFrame(rows)


def _shift_summaries(shifts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_fold = shifts.groupby(["fc", "fold", "representation"], sort=False).agg(
        mean_Wasserstein_V=("wasserstein_distance_V", "mean"),
        min_Wasserstein_V=("wasserstein_distance_V", "min"),
        max_Wasserstein_V=("wasserstein_distance_V", "max"),
    ).reset_index()
    pivot = per_fold.pivot(index=["fc", "fold"], columns="representation", values="mean_Wasserstein_V").reset_index()
    pivot["Delta_W_relative_minus_absolute_V"] = pivot["relative"] - pivot["absolute"]
    overall = per_fold.groupby(["fc", "representation"], sort=False)["mean_Wasserstein_V"].agg(["mean", "std", "min", "max"]).reset_index()
    overall = overall.rename(columns={"mean": "mean_Wasserstein_V_across_folds", "std": "std_Wasserstein_V_across_folds", "min": "min_Wasserstein_V_across_folds", "max": "max_Wasserstein_V_across_folds"})
    return pivot, overall


def _load_fold_level_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild aggregate files from immutable fold outputs, retaining fold identity."""
    metrics_rows, prediction_rows, shift_rows = [], [], []
    for prediction_path in sorted(output_dir.glob("FC*/*/fold_*/validation_predictions.csv")):
        fold_dir = prediction_path.parent
        representation, fc, fold = fold_dir.parent.name, fold_dir.parent.parent.name, fold_dir.name
        metric_path = fold_dir / "validation_metrics_per_horizon.csv"
        shift_path = fold_dir / "representation_shift_by_lag.csv"
        metrics = pd.read_csv(metric_path)
        predictions = pd.read_csv(prediction_path)
        shifts = pd.read_csv(shift_path)
        for frame in (metrics, predictions):
            frame.insert(0, "fold", fold)
            frame.insert(0, "representation", representation)
            frame.insert(0, "fc", fc)
        metrics_rows.append(metrics)
        prediction_rows.append(predictions)
        shift_rows.append(shifts)
    if len(metrics_rows) != 12:
        raise RuntimeError(f"Expected 12 Phase-3D fold outputs; found {len(metrics_rows)}")
    return pd.concat(metrics_rows, ignore_index=True), pd.concat(prediction_rows, ignore_index=True), pd.concat(shift_rows, ignore_index=True)


def _decision_table(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fc in ("FC1", "FC2"):
        absolute = summary.loc[(summary["fc"] == fc) & (summary["representation"] == "absolute")].iloc[0]
        relative = summary.loc[(summary["fc"] == fc) & (summary["representation"] == "relative")].iloc[0]
        late = bootstrap.loc[(bootstrap["fc"] == fc) & (bootstrap["horizon_set"] == "late_h13_h24_pooled")].iloc[0]
        def _cross(value: object) -> float:
            return float("inf") if str(value) == "none" else float(value)
        rows.append({
            "fc": fc,
            "Delta_MeanSkill_relative_minus_absolute": float(relative["MeanSkill_Best_all"] - absolute["MeanSkill_Best_all"]),
            "Delta_LateSkill_relative_minus_absolute": float(relative["LateSkill_Best_h13_h24"] - absolute["LateSkill_Best_h13_h24"]),
            "Delta_stable_positive_horizons": int(relative["stable_positive_horizon_count"] - absolute["stable_positive_horizon_count"]),
            "relative_crossover_not_later": bool(_cross(relative["crossover_h_c"]) <= _cross(absolute["crossover_h_c"])),
            "criteria_1_mean_skill_improved": bool(relative["MeanSkill_Best_all"] > absolute["MeanSkill_Best_all"]),
            "criteria_2_late_skill_improved": bool(relative["LateSkill_Best_h13_h24"] > absolute["LateSkill_Best_h13_h24"]),
            "criteria_3_stability_or_crossover_improved": bool(relative["stable_positive_horizon_count"] > absolute["stable_positive_horizon_count"] or _cross(relative["crossover_h_c"]) < _cross(absolute["crossover_h_c"])),
            "late_Delta_RMSE_mV": float(late["Delta_RMSE_relative_minus_absolute"] * 1000.0),
            "late_CI95_low_mV": float(late["CI95_low"] * 1000.0),
            "late_CI95_high_mV": float(late["CI95_high"] * 1000.0),
            "late_CI_excludes_zero": bool(late["CI_excludes_zero"]),
        })
    return pd.DataFrame(rows)


def _observed_outcome_lines(decisions: pd.DataFrame, shift_per_fold: pd.DataFrame) -> list[str]:
    lines = []
    for _, row in decisions.iterrows():
        fc = str(row["fc"])
        criteria_count = sum(bool(row[column]) for column in (
            "criteria_1_mean_skill_improved",
            "criteria_2_late_skill_improved",
            "criteria_3_stability_or_crossover_improved",
        ))
        shift_reduced_all_folds = bool((shift_per_fold.loc[shift_per_fold["fc"] == fc, "Delta_W_relative_minus_absolute_V"] < 0.0).all())
        if criteria_count >= 2 and bool(row["late_CI_excludes_zero"]) and float(row["late_Delta_RMSE_mV"]) < 0.0:
            statement = "meets the predeclared directional criteria and has a late paired interval favoring relative history"
        else:
            statement = "does not meet the predeclared evidence threshold for retaining relative history"
        lines.append(f"- **{fc}:** relative history {'reduced' if shift_reduced_all_folds else 'did not reduce'} the mean lag-wise Wasserstein shift in every fold and {statement}.")
    return lines


def write_report(clean_dir: Path) -> Path:
    output_dir = clean_dir / "outputs_phase3d" / "seed_42"
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    metrics, predictions, shifts = _load_fold_level_tables(output_dir)
    metrics.to_csv(output_dir / "all_validation_metrics.csv", index=False)
    predictions.to_csv(output_dir / "all_validation_predictions.csv", index=False)
    shifts.to_csv(output_dir / "all_representation_shift_by_lag.csv", index=False)
    summary, per_horizon = _representation_summary(metrics)
    summary.to_csv(output_dir / "representation_summary.csv", index=False)
    per_horizon.to_csv(output_dir / "representation_skill_by_horizon.csv", index=False)
    direct = (
        metrics.loc[(metrics["model"] == "residual_mimo_lstm") & (metrics["horizon"] != "mean_h1_h24")]
        .groupby(["fc", "representation"], sort=False)[["MAE", "RMSE", "R2", "Skill_Persistence", "Skill_Linear", "Skill_Best"]]
        .agg(["mean", "std"]).reset_index()
    )
    direct.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in direct.columns]
    direct.to_csv(output_dir / "representation_direct_metrics_mean_std.csv", index=False)
    shift_per_fold, shift_overall = _shift_summaries(shifts)
    shift_per_fold.to_csv(output_dir / "representation_shift_per_fold.csv", index=False)
    shift_overall.to_csv(output_dir / "representation_shift_summary.csv", index=False)
    bootstrap = paired_bootstrap_table(predictions)
    bootstrap.to_csv(output_dir / "paired_moving_block_bootstrap.csv", index=False)
    (output_dir / "paired_moving_block_bootstrap_metadata.json").write_text(json.dumps({
        "resampling_unit": "paired chronological prediction origins, resampled in 24-hour moving blocks independently within each fold",
        "delta": "RMSE(relative) - RMSE(absolute); negative favors relative",
        "block_length_hours": BLOCK_LENGTH_HOURS,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "scope": "fixed development predictions only; no retraining and no final-test rows",
    }, indent=2), encoding="utf-8")
    decisions = _decision_table(summary, bootstrap)
    decisions.to_csv(output_dir / "relative_representation_decision_checks.csv", index=False)
    observed_outcomes = _observed_outcome_lines(decisions, shift_per_fold)
    [_plot_skill(per_horizon, fc, output_dir) for fc in ("FC1", "FC2")]
    [_plot_shift(shifts, fc, output_dir) for fc in ("FC1", "FC2")]
    late = bootstrap.loc[bootstrap["horizon_set"] == "late_h13_h24_pooled"].copy()
    late["Delta_RMSE_mV"] = late["Delta_RMSE_relative_minus_absolute"] * 1000.0
    late["CI95_low_mV"] = late["CI95_low"] * 1000.0
    late["CI95_high_mV"] = late["CI95_high"] * 1000.0
    late = late[["fc", "Delta_RMSE_mV", "CI95_low_mV", "CI95_high_mV", "relative_better_probability", "CI_excludes_zero"]]
    report = clean_dir / "PHASE3D_RESULTS.md"
    lines = [
        "# Phase 3D results — prediction-origin relative history diagnostic",
        "",
        "All results are from the frozen front 60% development prefix and the three chronological folds only. The final 40% partition was not accessed.",
        "",
        "## Frozen protocol",
        "",
        "FC1/FC2; N=15; H=24; Utot-only residual MIMO-LSTM; one-layer hidden-64 encoder; seed 42; unchanged optimization and early stopping. Phase-3C-R's Nmax=48 legal origin schedule is retained, while both models see only the final 15 observations. The sole model input difference is absolute history versus prediction-origin-relative history.",
        "",
        "## Residual-MIMO validation metrics",
        "",
        "Values are means and standard deviations across all fold–horizon cells; the per-horizon table below supplies the direct three-fold mean/std at each h.",
        "",
        *_markdown(direct),
        "",
        "## Forecast-skill summary",
        "",
        "`MeanSkill_Best_all` averages h=1–24. `LateSkill_Best_h13_h24` averages h=13–24. Stable-positive means Skill_Best > 0 in every fold; crossover is the first of three consecutive stable-positive horizons.",
        "",
        *_markdown(summary),
        "",
        "## Representation-level distribution-shift audit",
        "",
        "Each value is the mean of 15 raw-voltage Wasserstein distances between train and validation input distributions at identical lags. `Delta_W_relative_minus_absolute_V < 0` indicates that prediction-origin normalization reduces this measured train-validation shift.",
        "",
        "### Per-fold mean distance",
        "",
        *_markdown(shift_per_fold),
        "",
        "### Across-fold summary",
        "",
        *_markdown(shift_overall),
        "",
        "## Paired 24-hour moving-block bootstrap — late h=13–24",
        "",
        "`Delta_RMSE = RMSE(relative) − RMSE(absolute)`; a negative result favors relative history. Intervals are 95% quantiles from 2,000 paired moving-block resamples of fixed origins (24-hour blocks within each fold).",
        "",
        *_markdown(late),
        "",
        "## Predeclared decision checks",
        "",
        "A representation can be considered for a later robustness study only if at least two directional criteria improve without a material reversal of the third; paired intervals remain the evidence for whether the late difference is robust.",
        "",
        *_markdown(decisions),
        "",
        "## Observed outcome",
        "",
        *observed_outcomes,
        "",
        "The result is stack-dependent under this one-seed development protocol: reduced marginal representation shift alone is not sufficient to guarantee a forecasting gain. No multi-seed confirmation, final evaluation, or backbone freeze is implied by this phase.",
        "",
        "## Per-horizon three-fold mean/std",
        "",
        *_markdown(per_horizon[["fc", "representation", "horizon", "RMSE_mean", "RMSE_std", "Skill_Best_mean", "Skill_Best_std", "positive_folds"]]),
        "",
        "## Stop boundary",
        "",
        "All per-run and paired representation audits must pass before interpretation. No final test, additional representation, new encoder, multi-seed robustness, probability model, or RUL experiment was run.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print(f"Wrote {write_report(here)}")

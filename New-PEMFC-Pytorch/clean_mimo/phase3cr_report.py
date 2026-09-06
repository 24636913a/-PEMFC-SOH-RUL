"""Summarize Phase 3C-R without selecting any model on a final test partition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOOTSTRAP_REPLICATES = 2000
BLOCK_LENGTH_HOURS = 24
BOOTSTRAP_SEED = 42
PAIR_COMPARISONS = (("N24_minus_N15", 24, 15), ("N48_minus_N15", 48, 15), ("N48_minus_N24", 48, 24))


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


def _plot(per_horizon: pd.DataFrame, fc: str, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    for history, group in per_horizon.loc[per_horizon["fc"] == fc].groupby("N", sort=True):
        group = group.sort_values("horizon")
        axis.plot(group["horizon"], group["Skill_Best_mean"], marker="o", linewidth=1.65, label=f"N={history}")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    axis.set_xlabel("forecast horizon h (hours)")
    axis.set_ylabel("mean Skill_Best")
    axis.set_title(f"Phase-3C-R matched-origin H=24 validation — {fc}")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / f"skill_best_matched_origins_{fc}.png", dpi=180)
    plt.close(figure)


def _paired_error_matrices(
    predictions: pd.DataFrame,
    fc: str,
    first_n: int,
    second_n: int,
    horizons: Iterable[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return per-fold squared-error matrices aligned on exact origin/target pairs."""
    selected = tuple(horizons)
    key = ["fold", "origin_row", "target_time_h", "horizon"]
    result: list[tuple[np.ndarray, np.ndarray]] = []
    subset = predictions.loc[(predictions["fc"] == fc) & (predictions["horizon"].isin(selected))]
    for fold in sorted(subset["fold"].unique()):
        first = subset.loc[(subset["fold"] == fold) & (subset["N"] == first_n), key + ["actual_Utot", "pred_residual_mimo_lstm"]].copy()
        second = subset.loc[(subset["fold"] == fold) & (subset["N"] == second_n), key + ["actual_Utot", "pred_residual_mimo_lstm"]].copy()
        merged = first.merge(second, on=key, how="inner", validate="one_to_one", suffixes=("_first", "_second"))
        if len(merged) != len(first) or len(merged) != len(second):
            raise RuntimeError(f"Unmatched Phase-3C-R paired predictions for {fc}, {fold}, N={first_n} vs N={second_n}")
        if not np.allclose(merged["actual_Utot_first"], merged["actual_Utot_second"], atol=0.0, rtol=0.0):
            raise RuntimeError("Paired variants do not have identical target values")
        merged = merged.sort_values(["origin_row", "horizon"])
        origin_count = merged["origin_row"].nunique()
        if len(merged) != origin_count * len(selected):
            raise RuntimeError("Every paired origin must contain every requested horizon")
        first_sq = (merged["pred_residual_mimo_lstm_first"].to_numpy() - merged["actual_Utot_first"].to_numpy()) ** 2
        second_sq = (merged["pred_residual_mimo_lstm_second"].to_numpy() - merged["actual_Utot_second"].to_numpy()) ** 2
        result.append((first_sq.reshape(origin_count, len(selected)), second_sq.reshape(origin_count, len(selected))))
    if len(result) != 3:
        raise RuntimeError(f"Expected three chronological folds; got {len(result)} for {fc}")
    return result


def _paired_moving_block_bootstrap(
    errors_by_fold: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict[str, float | int | bool]:
    """Resample matched sequential origins in 24-hour moving blocks, not i.i.d."""
    denominator = sum(first.size for first, _ in errors_by_fold)
    first_mse = sum(float(first.sum()) for first, _ in errors_by_fold) / denominator
    second_mse = sum(float(second.sum()) for _, second in errors_by_fold) / denominator
    point_delta = float(np.sqrt(first_mse) - np.sqrt(second_mse))
    rng = np.random.default_rng(seed)
    first_sum = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    second_sum = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for first, second in errors_by_fold:
        origin_count, horizon_count = first.shape
        block_count = int(np.ceil(origin_count / BLOCK_LENGTH_HOURS))
        starts = rng.integers(0, origin_count, size=(BOOTSTRAP_REPLICATES, block_count))
        offsets = np.arange(BLOCK_LENGTH_HOURS, dtype=np.int64)
        indices = ((starts[:, :, None] + offsets[None, None, :]) % origin_count).reshape(BOOTSTRAP_REPLICATES, -1)[:, :origin_count]
        first_sum += first[indices].sum(axis=(1, 2))
        second_sum += second[indices].sum(axis=(1, 2))
        if horizon_count < 1:
            raise RuntimeError("Paired bootstrap received an empty horizon selection")
    delta = np.sqrt(first_sum / denominator) - np.sqrt(second_sum / denominator)
    low, high = np.quantile(delta, [0.025, 0.975])
    return {
        "Delta_RMSE_first_minus_second": point_delta,
        "CI95_low": float(low),
        "CI95_high": float(high),
        "first_better_probability": float(np.mean(delta < 0.0)),
        "CI_excludes_zero": bool(high < 0.0 or low > 0.0),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "block_length_hours": BLOCK_LENGTH_HOURS,
    }


def paired_bootstrap_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fc_index, fc in enumerate(("FC1", "FC2"), start=1):
        for pair_index, (comparison, first_n, second_n) in enumerate(PAIR_COMPARISONS, start=1):
            horizon_sets = [(f"h{h}", (h,)) for h in range(1, 25)] + [("late_h13_h24_pooled", tuple(range(13, 25)))]
            for set_index, (label, horizons) in enumerate(horizon_sets, start=1):
                matrices = _paired_error_matrices(predictions, fc, first_n, second_n, horizons)
                bootstrap = _paired_moving_block_bootstrap(matrices, BOOTSTRAP_SEED * 100000 + fc_index * 10000 + pair_index * 100 + set_index)
                rows.append({
                    "fc": fc,
                    "comparison": comparison,
                    "first_N": first_n,
                    "second_N": second_n,
                    "horizon_set": label,
                    "interpretation": "negative delta favors first_N",
                    **bootstrap,
                })
    return pd.DataFrame(rows)


def _matched_count_summary(counts: pd.DataFrame) -> pd.DataFrame:
    grouped = counts.groupby(["fc", "fold"], sort=False).agg(
        train_windows_min=("train_windows", "min"), train_windows_max=("train_windows", "max"),
        validation_windows_min=("validation_windows", "min"), validation_windows_max=("validation_windows", "max"),
        N_variants=("N", "nunique"),
    ).reset_index()
    grouped["counts_identical_across_N"] = (
        (grouped["train_windows_min"] == grouped["train_windows_max"])
        & (grouped["validation_windows_min"] == grouped["validation_windows_max"])
        & (grouped["N_variants"] == 3)
    )
    return grouped


def write_report(clean_dir: Path) -> Path:
    output_dir = clean_dir / "outputs_phase3cr" / "seed_42"
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(output_dir / "all_validation_metrics.csv")
    predictions = pd.read_csv(output_dir / "all_validation_predictions.csv")
    summary = pd.read_csv(output_dir / "history_length_summary.csv")
    per_horizon = pd.read_csv(output_dir / "history_length_skill_by_horizon.csv")
    counts = pd.read_csv(output_dir / "matched_origin_window_counts.csv")
    count_summary = _matched_count_summary(counts)
    count_summary.to_csv(output_dir / "matched_origin_count_summary.csv", index=False)
    if not bool(count_summary["counts_identical_across_N"].all()):
        raise RuntimeError("Phase-3C-R comparison lost matched train/validation origin counts")
    direct = (
        metrics.loc[(metrics["model"] == "residual_mimo_lstm") & (metrics["horizon"] != "mean_h1_h24")]
        .groupby(["fc", "N"], sort=False)[["MAE", "RMSE", "R2", "Skill_Persistence", "Skill_Linear", "Skill_Best"]]
        .mean().reset_index()
        .rename(columns={"MAE": "mean_MAE", "RMSE": "mean_RMSE", "R2": "mean_R2", "Skill_Persistence": "mean_Skill_Persistence", "Skill_Linear": "mean_Skill_Linear", "Skill_Best": "mean_Skill_Best"})
    )
    direct.to_csv(output_dir / "history_length_direct_metrics.csv", index=False)
    bootstrap = paired_bootstrap_table(predictions)
    bootstrap.to_csv(output_dir / "paired_moving_block_bootstrap.csv", index=False)
    (output_dir / "paired_moving_block_bootstrap_metadata.json").write_text(json.dumps({
        "resampling_unit": "matched prediction origins, sampled in chronological moving blocks within each fold",
        "block_length_hours": BLOCK_LENGTH_HOURS,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "metric": "Delta_RMSE = RMSE(first_N) - RMSE(second_N); negative favors first_N",
        "scope": "fixed validation predictions only; no retraining and no final-test rows",
    }, indent=2), encoding="utf-8")
    [_plot(per_horizon, fc, output_dir) for fc in ("FC1", "FC2")]
    bootstrap_late = bootstrap.loc[bootstrap["horizon_set"] == "late_h13_h24_pooled"].copy()
    bootstrap_late["Delta_RMSE_mV"] = bootstrap_late["Delta_RMSE_first_minus_second"] * 1000.0
    bootstrap_late["CI95_low_mV"] = bootstrap_late["CI95_low"] * 1000.0
    bootstrap_late["CI95_high_mV"] = bootstrap_late["CI95_high"] * 1000.0
    bootstrap_late = bootstrap_late[["fc", "comparison", "first_N", "second_N", "Delta_RMSE_mV", "CI95_low_mV", "CI95_high_mV", "first_better_probability", "CI_excludes_zero"]]
    report = clean_dir / "PHASE3CR_RESULTS.md"
    lines = [
        "# Phase 3C-R results — matched-origin H=24 history diagnostic",
        "",
        "All values are from the frozen front 60% development prefix and three chronological folds only. The final 40% partition was not accessed.",
        "",
        "## Protocol",
        "",
        "FC1/FC2; H=24; Utot-only residual MIMO-LSTM; legal persistence anchor; seed 42; unchanged optimizer and early stopping. Every train and validation origin was first generated as a legal Nmax=48 window. N=15 and N=24 use exactly the final 15 or 24 hours of those same windows. Only N=15/24/48 are compared; N=72 is excluded.",
        "",
        "## Matched origin counts",
        "",
        "Each fold must have identical train/validation origin counts across N. This is verified before metrics are summarized.",
        "",
        *_markdown(count_summary),
        "",
        "## Mean residual-MIMO validation metrics",
        "",
        *_markdown(direct),
        "",
        "## History-length summary",
        "",
        "`MeanSkill_all` averages h=1–24. `MeanSkill_late_h13_h24` averages h=13–24. A stable-positive horizon is positive versus the best analytical baseline in all three folds; `crossover_h_c` is the first of three consecutive stable-positive horizons.",
        "",
        *_markdown(summary),
        "",
        "## Paired moving-block bootstrap — late h=13–24",
        "",
        "For each comparison, `Delta_RMSE = RMSE(first_N) − RMSE(second_N)`: a negative value favors `first_N`. The 95% interval is from 2,000 paired moving-block resamples of fixed validation origins (24-hour blocks within each fold). It quantifies variation under the matched origins; it does not replace independent seed robustness.",
        "",
        *_markdown(bootstrap_late),
        "",
        "The full h=1–24 paired table is saved as `outputs_phase3cr/seed_42/paired_moving_block_bootstrap.csv`.",
        "",
        "## Frozen interpretation",
        "",
        "Under the exact matched-origin comparison, this one-seed development study does not establish a robust incremental benefit from extending the history beyond N=15. FC1 has no late h=13–24 pairwise interval excluding zero for N=24 or N=48 against N=15. FC2 has the highest mean and late Skill_Best at N=15; N=48 is negative in the late summary. These results do not justify selecting a longer-memory architecture or a multi-scale history mechanism.",
        "",
        "The next diagnostic remains unrun: if explicitly approved, compare absolute and prediction-origin-relative voltage history at the frozen H=24 protocol, without reopening the final partition. No such comparison is implied by this report or performed here.",
        "",
        "## Per-horizon Skill_Best",
        "",
        *_markdown(per_horizon[["fc", "N", "horizon", "RMSE_mean", "Skill_Best_mean", "Skill_Best_std", "positive_folds"]]),
        "",
        "## Files and stop boundary",
        "",
        "- `matched_origin_audits/` and each run's `leakage_audit.*` must all pass before results are interpreted.",
        "- `skill_best_matched_origins_FC1.png` and `skill_best_matched_origins_FC2.png` visualize the matched comparison.",
        "- No relative-history input, matrix-memory model, multi-scale model, probability model, final test, multi-seed run, or RUL experiment was performed.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print(f"Wrote {write_report(here)}")

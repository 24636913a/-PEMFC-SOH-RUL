"""Render development-only Phase-3B horizon-skill figures and factual report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _markdown(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[column]) else "---" for column in columns) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f"{float(value):.6f}" if isinstance(value, (float, int)) and not isinstance(value, bool) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_skill_plot(summary: pd.DataFrame, fc: str, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    view = summary.loc[summary["fc"] == fc]
    for horizon, group in view.groupby("H", sort=True):
        group = group.sort_values("horizon")
        axis.plot(group["horizon"], group["Skill_Best_mean"], marker="o", linewidth=1.7, label=f"direct MIMO H={horizon}")
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.65)
    axis.set_xlabel("forecast horizon h (hours)")
    axis.set_ylabel("mean Skill_Best vs min(Persistence, Linear)")
    axis.set_title(f"Phase-3B development validation — {fc}")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path = output_dir / f"skill_best_by_horizon_{fc}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_report(clean_dir: Path) -> Path:
    output_dir = clean_dir / "outputs_phase3b" / "seed_42"
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    all_metrics = pd.read_csv(output_dir / "all_validation_metrics.csv")
    summary = pd.read_csv(output_dir / "residual_skill_by_horizon.csv")
    aggregate = (
        all_metrics.loc[(all_metrics["model"] == "residual_mimo_lstm") & (all_metrics["horizon"] != "mean_h1_hH")]
        .groupby(["fc", "H"], sort=False)[["MAE", "RMSE", "R2", "Skill_Persistence", "Skill_Linear", "Skill_Best"]]
        .mean()
        .reset_index()
        .rename(columns={
            "MAE": "mean_MAE", "RMSE": "mean_RMSE", "R2": "mean_R2",
            "Skill_Persistence": "mean_Skill_Persistence", "Skill_Linear": "mean_Skill_Linear", "Skill_Best": "mean_Skill_Best",
        })
    )
    aggregate.to_csv(output_dir / "residual_skill_by_direct_horizon.csv", index=False)
    crossover = []
    raw_residual = all_metrics.loc[(all_metrics["model"] == "residual_mimo_lstm") & (all_metrics["horizon"] != "mean_h1_hH")].copy()
    stability = (
        raw_residual.assign(positive=raw_residual["Skill_Best"] > 0)
        .groupby(["fc", "H", "horizon"], sort=False)
        .agg(Skill_Best_mean=("Skill_Best", "mean"), Skill_Best_min=("Skill_Best", "min"), Skill_Best_max=("Skill_Best", "max"), positive_folds=("positive", "sum"))
        .reset_index()
    )
    stability.to_csv(output_dir / "skill_best_fold_stability.csv", index=False)
    for (fc, direct_h), group in summary.groupby(["fc", "H"], sort=False):
        stable = stability.loc[(stability["fc"] == fc) & (stability["H"] == direct_h)]
        positive = group.loc[group["Skill_Best_mean"] > 0, "horizon"].astype(int).tolist()
        at_least_two = stable.loc[stable["positive_folds"] >= 2, "horizon"].astype(int).tolist()
        all_three = stable.loc[stable["positive_folds"] == 3, "horizon"].astype(int).tolist()
        crossover.append({
            "fc": fc,
            "direct_H": int(direct_h),
            "mean_positive_horizons": ", ".join(map(str, positive)) if positive else "none",
            "positive_in_at_least_2_folds": ", ".join(map(str, at_least_two)) if at_least_two else "none",
            "positive_in_all_3_folds": ", ".join(map(str, all_three)) if all_three else "none",
        })
    crossover_frame = pd.DataFrame(crossover)
    crossover_frame.to_csv(output_dir / "skill_best_crossover_summary.csv", index=False)
    plot_paths = [_write_skill_plot(summary, fc, output_dir) for fc in ("FC1", "FC2")]
    lines = [
        "# Phase 3B results — development-only horizon-scale diagnostic",
        "",
        "This report uses the first 60% development region only. The final 40% test partition was not constructed, fit, predicted, or scored.",
        "",
        "## Frozen protocol",
        "",
        "FC1/FC2; Utot-only absolute history; N=15; residual MIMO-LSTM with current-value anchor; one layer, hidden 64, MSE/Adam, seed 42; H=4/8/12/24; folds 0–30/30–40%, 0–40/40–50%, 0–50/50–60%.",
        "",
        "## Residual MIMO mean validation performance",
        "",
        "`Skill_Best = 1 - RMSE_model / min(RMSE_persistence, RMSE_linear)` at each horizon.",
        "",
        *_markdown(aggregate),
        "",
        "## Horizon-wise crossover record",
        "",
        *_markdown(crossover_frame),
        "",
        "## Per-horizon residual Skill_Best",
        "",
        *_markdown(summary[["fc", "H", "horizon", "RMSE_mean", "R2_mean", "Skill_Persistence_mean", "Skill_Linear_mean", "Skill_Best_mean", "Skill_Best_std"]]),
        "",
        "## Figures",
        "",
        "- `skill_best_by_horizon_FC1.png` and `skill_best_by_horizon_FC2.png` plot direct-MIMO H curves against zero Skill_Best.",
        "",
        "## Stop boundary",
        "",
        "No final test, N sweep, relative-history input, Full-26/R3, OCM, new architecture, multi-seed, uncertainty, or RUL experiment was run.",
    ]
    report = clean_dir / "PHASE3B_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print(f"Wrote {write_report(here)}")

"""Render factual Phase-3C history-length summaries and plots."""
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


def _plot(per_horizon: pd.DataFrame, fc: str, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    for history, group in per_horizon.loc[per_horizon["fc"] == fc].groupby("N", sort=True):
        group = group.sort_values("horizon")
        axis.plot(group["horizon"], group["Skill_Best_mean"], marker="o", linewidth=1.65, label=f"N={history}")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    axis.set_xlabel("forecast horizon h (hours)")
    axis.set_ylabel("mean Skill_Best")
    axis.set_title(f"Phase-3C H=24 history-length validation — {fc}")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path = output_dir / f"skill_best_by_history_length_{fc}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_report(clean_dir: Path) -> Path:
    output_dir = clean_dir / "outputs_phase3c" / "seed_42"
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output_dir / "history_length_summary.csv")
    per_horizon = pd.read_csv(output_dir / "history_length_skill_by_horizon.csv")
    all_metrics = pd.read_csv(output_dir / "all_validation_metrics.csv")
    window_rows = []
    for prediction_path in sorted(output_dir.glob("FC*/N_*/fold_*/validation_predictions.csv")):
        frame = pd.read_csv(prediction_path, usecols=["sample"])
        window_rows.append({
            "fc": prediction_path.parents[2].name,
            "N": int(prediction_path.parents[1].name.split("_")[1]),
            "fold": prediction_path.parents[0].name,
            "validation_windows": int(frame["sample"].nunique()),
        })
    windows = pd.DataFrame(window_rows)
    windows.to_csv(output_dir / "validation_window_counts.csv", index=False)
    window_summary = windows.groupby(["fc", "N"], sort=False)["validation_windows"].agg(["min", "max"]).reset_index().rename(columns={"min": "validation_windows_min", "max": "validation_windows_max"})
    summary = summary.merge(window_summary, on=["fc", "N"], how="left")
    summary.to_csv(output_dir / "history_length_summary.csv", index=False)
    direct = (
        all_metrics.loc[(all_metrics["model"] == "residual_mimo_lstm") & (all_metrics["horizon"] != "mean_h1_h24")]
        .groupby(["fc", "N"], sort=False)[["MAE", "RMSE", "R2", "Skill_Persistence", "Skill_Linear", "Skill_Best"]]
        .mean().reset_index()
        .rename(columns={"MAE": "mean_MAE", "RMSE": "mean_RMSE", "R2": "mean_R2", "Skill_Persistence": "mean_Skill_Persistence", "Skill_Linear": "mean_Skill_Linear", "Skill_Best": "mean_Skill_Best"})
    )
    direct.to_csv(output_dir / "history_length_direct_metrics.csv", index=False)
    [_plot(per_horizon, fc, output_dir) for fc in ("FC1", "FC2")]
    lines = [
        "# Phase 3C results — H=24 history-length diagnostic",
        "",
        "All values come from the development prefix and three chronological folds only. No final-test row, prediction, or score was used.",
        "",
        "## Frozen protocol",
        "",
        "FC1/FC2; H=24; Utot-only residual MIMO-LSTM with legal persistence anchor; seed 42; N=15/24/48/72; unchanged optimizer/early stopping; 0–30/30–40%, 0–40/40–50%, 0–50/50–60% folds.",
        "",
        "## Mean residual-MIMO validation metrics",
        "",
        *_markdown(direct),
        "",
        "## History-length summary",
        "",
        "`MeanSkill_all` averages h=1–24; `MeanSkill_late_h13_h24` averages h=13–24. A stable-positive horizon is positive versus the best analytical baseline in all three folds. `crossover_h_c` is the first of three consecutive stable-positive horizons.",
        "",
        *_markdown(summary),
        "",
        "Strict split containment means longer N leaves fewer validation windows. `validation_windows_min/max` reports that effective sample-size change; comparisons involving very small counts must be treated as diagnostic rather than stable evidence.",
        "",
        "## Per-horizon Skill_Best",
        "",
        *_markdown(per_horizon[["fc", "N", "horizon", "RMSE_mean", "Skill_Best_mean", "Skill_Best_std", "positive_folds"]]),
        "",
        "## Figures",
        "",
        "- `skill_best_by_history_length_FC1.png` and `skill_best_by_history_length_FC2.png` compare N curves at H=24.",
        "",
        "## Stop boundary",
        "",
        "No M-LSTM, multi-scale model, relative history, uncertainty, final test, multi-seed study, or RUL experiment was run.",
    ]
    report = clean_dir / "PHASE3C_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print(f"Wrote {write_report(here)}")

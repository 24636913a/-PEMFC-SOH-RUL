"""Render factual results for the development-only Phase-3A decomposition."""
from __future__ import annotations

import json
from pathlib import Path

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


def _model_summary(metrics: pd.DataFrame, model: str, group_columns: list[str]) -> pd.DataFrame:
    value = metrics.loc[(metrics["model"] == model) & (metrics["horizon"] != "mean_h1_h4")]
    numeric = [column for column in ("MAE", "RMSE", "R2", "NRMSE", "Skill_slow_persistence", "Skill_slow_linear_trend", "Skill_zero_transient", "Skill_transient_persistence") if column in value.columns]
    return value.groupby(group_columns, sort=False)[numeric].mean().reset_index()


def write_report(clean_dir: Path) -> Path:
    output_dir = clean_dir / "outputs_phase3a" / "FC2" / "seed_42"
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    t1 = pd.read_csv(output_dir / "T1_all_validation_metrics.csv")
    t0 = pd.read_csv(output_dir / "T0_saved_raw_residual_reference.csv")
    tau_table = pd.read_csv(output_dir / "T1_tau_selection_validation_only.csv")
    t2 = pd.read_csv(output_dir / "T2_all_validation_metrics.csv")
    t1_summary = _model_summary(t1, "T1_slow_residual_mimo_lstm", ["tau_hours"])
    t0_summary = t0.groupby("source", sort=False)[["RMSE", "NRMSE"]].mean().reset_index().rename(columns={"RMSE": "mean_RMSE", "NRMSE": "mean_NRMSE"})
    t2_summary = _model_summary(t2, "T2_R_history_only_mimo_lstm", ["variant", "tau_hours"])
    t2_load = _model_summary(t2, "T2_R_history_plus_load_mimo_lstm", ["variant", "tau_hours"])
    t2_summary = pd.concat([t2_summary, t2_load], ignore_index=True)
    selected_tau = int(result["validation_selected_tau_hours"])
    selected_t1 = t1_summary.loc[t1_summary["tau_hours"] == selected_tau].iloc[0]
    selected_t1_by_fold = _model_summary(
        t1.loc[t1["tau_hours"] == selected_tau], "T1_slow_residual_mimo_lstm", ["fold"]
    )
    selected_t2_by_fold = pd.concat([
        _model_summary(t2, "T2_R_history_only_mimo_lstm", ["variant", "fold"]),
        _model_summary(t2, "T2_R_history_plus_load_mimo_lstm", ["variant", "fold"]),
    ], ignore_index=True)
    selected_t1_by_fold.to_csv(output_dir / "T1_selected_tau_by_fold.csv", index=False)
    selected_t2_by_fold.to_csv(output_dir / "T2_selected_tau_by_fold.csv", index=False)
    t1_beats_linear = float(selected_t1["Skill_slow_linear_trend"]) > 0
    t2_best_nrmse = float(t2_summary["NRMSE"].min())
    t1_nrmse = float(selected_t1["NRMSE"])
    t0_nrmse = float(t0_summary.iloc[0]["mean_NRMSE"])
    t1_positive_slow_linear_folds = int((selected_t1_by_fold["Skill_slow_linear_trend"] > 0).sum())
    t2a = t2_summary.loc[t2_summary["variant"] == "T2A_R_history_only"].iloc[0]
    t2b = t2_summary.loc[t2_summary["variant"] == "T2B_R_history_plus_load"].iloc[0]
    load_rmse_reduction = 1.0 - float(t2b["RMSE"]) / float(t2a["RMSE"])
    if t1_nrmse < t0_nrmse and t1_beats_linear and t2_best_nrmse > t1_nrmse:
        conclusion = "Phase 3A supports the decomposition hypothesis: the slow target is better normalized than saved T0 and exceeds Slow Linear, while the transient task is harder."
    else:
        reasons = []
        if t1_nrmse >= t0_nrmse:
            reasons.append("T1 NRMSE does not improve on saved T0")
        if not t1_beats_linear:
            reasons.append("T1 does not exceed Slow Linear on average")
        if t2_best_nrmse <= t1_nrmse:
            reasons.append("T2 is not harder than T1 in NRMSE")
        conclusion = "Phase 3A does not meet the predeclared decomposition-success criteria: " + "; ".join(reasons) + ". EWMA smoothing alone has not established a more forecastable degradation target."
    lines = [
        "# Phase 3A results — FC2 causal decomposition",
        "",
        "All quantities in this report come from the first 60% development region and the three expanding validation folds. No final-test row, prediction, or score was used.",
        "",
        "## T1 tau validation and selection",
        "",
        "T1 input is causal slow-state D history only; target is Delta D(t+h). NRMSE is RMSE divided by the standard deviation of the corresponding target horizon.",
        "",
        *_markdown(tau_table),
        "",
        f"Validation-selected tau: **{selected_tau} h**.",
        "",
        "## T1 neural summary versus saved T0 reference",
        "",
        *_markdown(t0_summary),
        "",
        *_markdown(t1_summary),
        "",
        "### Selected-tau T1 stability by fold",
        "",
        *_markdown(selected_t1_by_fold),
        "",
        "## T2 transient forecasting at selected tau",
        "",
        "T2-A uses R history only. T2-B adds legal historical J_mean/I_mean/I_std/I_p2p. The relevant skills are versus zero transient and transient persistence.",
        "",
        *_markdown(t2_summary),
        "",
        "### T2 stability by fold",
        "",
        *_markdown(selected_t2_by_fold),
        "",
        "## Development-only reading",
        "",
        f"Selected T1 NRMSE is `{t1_nrmse:.6f}` versus saved T0 NRMSE `{t0_nrmse:.6f}`. T1 has positive Slow-Linear skill in `{t1_positive_slow_linear_folds}/3` folds. T2-B lowers mean RMSE by `{load_rmse_reduction:+.2%}` versus T2-A, which is a weak development-only load increment rather than a stable final conclusion.",
        "",
        conclusion,
        "",
        "## Stop boundary",
        "",
        "No final test, OCM, additional decomposition variant, horizon extension, multi-seed run, uncertainty method, RUL experiment, or external validation was performed.",
    ]
    report = clean_dir / "PHASE3A_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print(f"Wrote {write_report(here)}")

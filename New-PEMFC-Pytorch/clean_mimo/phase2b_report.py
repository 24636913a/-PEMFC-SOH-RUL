"""Render a factual Phase-2b validation-only report from saved artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _markdown(frame: pd.DataFrame, decimals: int = 6) -> list[str]:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[column]) else "---" for column in columns) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{float(value):.{decimals}f}" if isinstance(value, (float, int)) and not isinstance(value, bool) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(output_root: Path) -> Path:
    output_dir = output_root / "FC2" / "seed_42"
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(output_dir / "feature_selection_validation_only.csv")
    horizon = pd.read_csv(output_dir / "residual_validation_by_horizon.csv")
    all_metrics = pd.read_csv(output_dir / "all_validation_metrics.csv")
    residual = all_metrics.loc[
        (all_metrics["model"] == "residual_mimo_lstm") & (all_metrics["horizon"] != "mean_h1_h4")
    ]
    by_fold = (
        residual.groupby(["fold", "feature_set"], sort=False)[["MAE", "RMSE", "R2", "Skill_P", "Skill_L"]]
        .mean()
        .reset_index()
        .rename(columns={
            "MAE": "mean_h1_h4_MAE",
            "RMSE": "mean_h1_h4_RMSE",
            "R2": "mean_h1_h4_R2",
            "Skill_P": "mean_h1_h4_Skill_P",
            "Skill_L": "mean_h1_h4_Skill_L",
        })
    )
    by_fold.to_csv(output_dir / "residual_validation_by_fold.csv", index=False)
    selected_fold = by_fold.loc[by_fold["feature_set"] == result["validation_selected_feature_set"]]
    positive_skill_l = int((selected_fold["mean_h1_h4_Skill_L"] > 0).sum())
    lines = [
        "# Phase-2b results — FC2 expanding validation feature ladder",
        "",
        "This report contains development-only validation results. No final-test window was constructed, evaluated, or used for selection.",
        "",
        "## Frozen protocol",
        "",
        "- FC2, seed 42, causal hourly aggregation, N=15, H=4, residual MIMO-LSTM (one-layer LSTM, hidden 64) with train-only fold scalers.",
        "- Folds: 0–35/35–40%, 0–45/45–50%, and 0–55/55–60% train/validation boundaries; the final 40% remains inaccessible to this runner.",
        "- R1=`Utot_mean`; R2=R1 plus U1–U5; R3=R2 plus J_mean/I_mean/I_std/I_p2p. Linear Trend is the primary analytical comparison.",
        "",
        "## Validation-only feature selection",
        "",
        "Selection rule: lowest unweighted residual-MIMO RMSE over 3 folds × 4 horizons; exact ties use higher Skill_L then fixed R1/R2/R3 order.",
        "",
        *_markdown(selected),
        "",
        f"Validation-selected feature set: **{result['validation_selected_feature_set']}**.",
        "",
        "## Residual MIMO average across h=1–4 by fold",
        "",
        *_markdown(by_fold),
        "",
        f"For the validation-selected set, mean Skill_L is positive in `{positive_skill_l}/{len(selected_fold)}` folds. The rank is a validation-only decision rule, not evidence by itself that condition information has a stable increment over the linear baseline.",
        "",
        "## Residual MIMO by horizon across folds",
        "",
        "`Skill_P = 1 - RMSE_model/RMSE_persistence`; `Skill_L = 1 - RMSE_model/RMSE_linear`. Values below are fold mean/std/min/max; they are not final-test estimates.",
        "",
        *_markdown(horizon),
        "",
        "## Stop boundary",
        "",
        "No final-test evaluation, Full-26 run, relative-input run, OCM, causal decomposition, horizon expansion, multiple seeds, uncertainty experiment, or RUL work was performed.",
    ]
    report = output_root.parent / "PHASE2B_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "outputs_phase2b"
    print(f"Wrote {write_report(root)}")

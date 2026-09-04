"""Write a factual report for the completed one-time frozen R3 final test."""
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


def write_report(clean_dir: Path) -> Path:
    output_dir = clean_dir / "outputs_phase2b_final_test" / "FC2" / "seed_42"
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(output_dir / "metrics_per_horizon.csv")
    residual = metrics.loc[metrics["model"] == "R3_residual_mimo_lstm"].copy()
    linear = metrics.loc[metrics["model"] == "linear_trend"].copy()
    residual_mean = residual.loc[residual["horizon"] == "mean_h1_h4"].iloc[0]
    linear_mean = linear.loc[linear["horizon"] == "mean_h1_h4"].iloc[0]
    rmse_relative_to_linear = 1.0 - float(residual_mean["RMSE"]) / float(linear_mean["RMSE"])
    horizon_rows = residual.loc[residual["horizon"] != "mean_h1_h4"]
    positive_skill_l = int((horizon_rows["Skill_L"] > 0).sum())
    if rmse_relative_to_linear > 0 and positive_skill_l >= 3:
        outcome = "R3 is lower-RMSE than Linear Trend overall and has positive Skill_L at a majority of horizons. This remains a one-seed result and requires robustness verification before any architecture decision."
    elif abs(rmse_relative_to_linear) <= 0.03:
        outcome = "R3 and Linear Trend are within the predefined 3% near-tie band overall; any historical load increment is weak and not stable enough to justify OCM."
    else:
        outcome = "R3 is materially weaker than Linear Trend overall; the next scientific branch is causal slow-degradation versus reversible-transient decomposition, not a more complex sequence model."
    display = metrics[["model", "horizon", "MAE", "RMSE", "R2", "Skill_P", "Skill_L"]]
    lines = [
        "# Frozen R3 final-test results — FC2",
        "",
        "This report comes from the single authorized final 40% test evaluation after R3 was selected by Phase-2b validation only. No other feature set or model was evaluated in this final-test stage.",
        "",
        "## Locked configuration",
        "",
        "R3 State+Load (`Utot`, U1–U5, J_mean, I_mean, I_std, I_p2p); 1 h; N=15; H=4; residual target; one-layer hidden-64 LSTM; MSE/Adam; seed 42; original train/validation/test and scaler rules.",
        "",
        "## Final-test metrics",
        "",
        "`Skill_P = 1 - RMSE_model/RMSE_persistence`; `Skill_L = 1 - RMSE_model/RMSE_linear`.",
        "",
        *_markdown(display),
        "",
        "## Frozen-test interpretation",
        "",
        f"R3 mean RMSE relative to Linear Trend: `{rmse_relative_to_linear:+.2%}`. R3 has positive Skill_L at `{positive_skill_l}/4` horizons.",
        "",
        outcome,
        "",
        "## Integrity record",
        "",
        f"The result artifact records `test_evaluation_count = {result['test_evaluation_count']}`. The completion marker blocks a repeat run; Phase-1, Phase-2, and Phase-2b validation outputs were not modified.",
    ]
    report = clean_dir / "PHASE2B_FINAL_TEST_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print(f"Wrote {write_report(here)}")

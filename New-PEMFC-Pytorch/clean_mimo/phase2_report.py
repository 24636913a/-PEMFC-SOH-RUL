"""Write a factual Phase-2 seed-42 report after both runs exist."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from phase2_experiment import build_phase2_data


def _mean_row(metrics: pd.DataFrame, model: str) -> pd.Series:
    rows = metrics.loc[(metrics["model"] == model) & (metrics["horizon"].astype(str) == "mean_h1_h4")]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one mean_h1_h4 metric row for {model}")
    return rows.iloc[0]


def _decision(fc1: pd.DataFrame, fc2: pd.DataFrame) -> tuple[str, str]:
    f1_abs, f1_res = _mean_row(fc1, "absolute_mimo_lstm"), _mean_row(fc1, "residual_mimo_lstm")
    f2_abs, f2_res = _mean_row(fc2, "absolute_mimo_lstm"), _mean_row(fc2, "residual_mimo_lstm")
    f1_best = min(_mean_row(fc1, "persistence")["RMSE"], _mean_row(fc1, "linear_trend")["RMSE"])
    f2_best = min(_mean_row(fc2, "persistence")["RMSE"], _mean_row(fc2, "linear_trend")["RMSE"])
    if f1_res.RMSE < f1_abs.RMSE and f2_res.RMSE < f2_abs.RMSE and f1_res.RMSE <= f1_best and f2_res.RMSE <= f2_best:
        return "A", "Residual improves both neural models and reaches/exceeds the stronger analytical baseline on both stacks."
    if f1_res.RMSE < f1_abs.RMSE and f2_res.RMSE > _mean_row(fc2, "linear_trend")["RMSE"]:
        return "B", "FC1 residual recovers relative to absolute MIMO, while FC2 residual remains weaker than Linear trend."
    if f1_res.RMSE >= 0.02:
        return "C", "FC1 residual remains in the severe-error regime (>=20 mV mean RMSE), indicating input-level lifecycle shift remains material."
    return "C", "Residual did not establish a stable neural advantage; diagnose relative voltage inputs before any condition-modulated model."


def _comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", "horizon", "MAE", "RMSE", "R2", "persistence_RMSE", "RMSE_improvement_vs_persistence", "MSE_skill_vs_persistence"]
    return metrics[columns]


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    def cell(value):
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if column not in {"model", "horizon"} else "---" for column in headers) + "|"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(clean_dir: Path | None = None) -> Path:
    clean_dir = clean_dir or Path(__file__).resolve().parent
    root = clean_dir / "outputs_phase2"
    fc1_dir, fc2_dir = root / "FC1" / "seed_42", root / "FC2" / "seed_42"
    for directory in (fc1_dir, fc2_dir):
        if not (directory / "metrics_per_horizon.csv").exists():
            raise FileNotFoundError(f"Phase-2 seed-42 metrics missing: {directory}")
    fc1_test = pd.read_csv(fc1_dir / "metrics_per_horizon.csv")
    fc2_test = pd.read_csv(fc2_dir / "metrics_per_horizon.csv")
    fc1_val = pd.read_csv(fc1_dir / "validation_metrics_per_horizon.csv")
    fc2_val = pd.read_csv(fc2_dir / "validation_metrics_per_horizon.csv")
    branch, rationale = _decision(fc1_test, fc2_test)
    d1, d2 = build_phase2_data("FC1"), build_phase2_data("FC2")

    def distribution(data):
        return {
            name: (float(split.y_raw.mean()), float(split.y_raw.min()), float(split.y_raw.max()))
            for name, split in data.base.splits.items()
        }

    lines = [
        "# Phase-2 seed-42 results — Persistence-Anchored Residual MIMO",
        "",
        "This report is generated from completed Phase-2 artifacts.  No model was selected by ranking test results, no Phase-1 output was modified, and no multi-seed, feature-ladder, relative-input, OCM, or RUL experiment was run.",
        "",
        "## Configuration",
        "",
        "- FC1 and FC2: original causal 1 h aggregation, N=15, H=4, 50/10/40 split-contained windows, seed 42.",
        "- Inputs: R4 `full_26`, `absolute_inputs`; no elapsed operating time.",
        "- Neural models: same one-layer 64-hidden LSTM, Linear four-output head, MSE, Adam, unchanged Phase-1 learning rate/batch size/weight decay/early stopping.",
        "- Residual label: `Utot(t+h)-Utot(t)`; its independent scaler is fitted only on flattened training residuals. Prediction is restored with the observed origin Utot anchor.",
        "",
        "## Test metrics: FC1",
        "",
        _markdown_table(_comparison_table(fc1_test)),
        "",
        "## Test metrics: FC2",
        "",
        _markdown_table(_comparison_table(fc2_test)),
        "",
        "## Validation diagnostic metrics",
        "",
        "Validation files are recorded for future validation-only formulation decisions; no feature set was chosen from the test table.",
        "",
        "### FC1 validation",
        "",
        _markdown_table(_comparison_table(fc1_val)),
        "",
        "### FC2 validation",
        "",
        _markdown_table(_comparison_table(fc2_val)),
        "",
        "## Absolute-level distribution diagnostic",
        "",
        "| Stack | Split | target mean (V) | target min (V) | target max (V) |",
        "|---|---|---:|---:|---:|",
    ]
    for stack, data in (("FC1", d1), ("FC2", d2)):
        for split, values in distribution(data).items():
            lines.append(f"| {stack} | {split} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
    lines += [
        "",
        "## Decision path",
        "",
        f"**Current path: {branch}.** {rationale}",
        "",
        "The decision is a seed-42 diagnostic, not a claim of statistical stability.  The prescribed next experiment must be confirmed before any Phase-2b expanding validation, R1–R4 feature-ladder run, relative-input run, or OCM work.",
    ]
    event_doc = fc2_dir / "FC2_EVENT_DIAGNOSTICS.md"
    if event_doc.exists():
        lines += ["", "## FC2 event diagnostics", "", f"Completed: `{event_doc.relative_to(clean_dir)}`. Events were retained and analyzed descriptively; they were not deleted or used as model inputs."]
    report_path = clean_dir / "PHASE2_RESULTS.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    path = write_report()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

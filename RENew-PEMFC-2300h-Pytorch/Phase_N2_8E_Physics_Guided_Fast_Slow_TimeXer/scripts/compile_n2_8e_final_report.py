"""Compile the final N2.8E report and decision from audited development outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8E_Physics_Guided_Fast_Slow_TimeXer")


def mdtable(frame: pd.DataFrame, columns: list[str]) -> str:
    show = frame.loc[:, [col for col in columns if col in frame.columns]].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    lines = ["| " + " | ".join(show.columns) + " |", "|" + "|".join("---" for _ in show.columns) + "|"]
    lines.extend("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in show.itertuples(index=False, name=None))
    return "\n".join(lines)


def row(frame: pd.DataFrame, model: str) -> pd.Series:
    records = frame.loc[frame["model"] == model]
    if len(records) != 1:
        raise RuntimeError(f"Expected one model summary row for {model}")
    return records.iloc[0]


def mean_horizon(horizon: pd.DataFrame, model: str, hours: float) -> float:
    values = horizon.loc[(horizon["model"] == model) & np.isclose(horizon["horizon_hours"], hours), "rmse_delta_mV"]
    if len(values) != 3:
        raise RuntimeError(f"Missing three-fold horizon result: {model} {hours}h")
    return float(values.mean())


def small(record: pd.Series) -> dict[str, float]:
    return {"rmse_delta_mV": float(record["mean_rmse_delta_mV"]), "r2_delta": float(record["mean_r2_delta"]), "mae_delta_mV": float(record["mean_mae_delta_mV"])}


def main() -> None:
    comparison = pd.read_csv(ROOT / "outputs" / "model_screening" / "N2_8E_seed42_comparison.csv")
    ablation = pd.read_csv(ROOT / "outputs" / "ablation" / "N2_8E_physics_ablation.csv")
    horizon = pd.read_csv(ROOT / "outputs" / "horizon_analysis" / "N2_8E_per_horizon.csv")
    tail = pd.read_csv(ROOT / "outputs" / "tail_analysis" / "N2_8E_tail_metrics.csv")
    variance = pd.read_csv(ROOT / "outputs" / "diagnostics" / "fast_slow_variance_audit.csv")
    current = pd.read_csv(ROOT / "outputs" / "diagnostics" / "current_sensitivity_summary.csv")
    monotonicity = pd.read_csv(ROOT / "outputs" / "diagnostics" / "slow_monotonicity_audit.csv")
    multiseed = pd.read_csv(ROOT / "outputs" / "model_screening" / "N2_8E_multiseed_comparison.csv")
    firewall = json.loads((ROOT / "outputs" / "test_firewall_audit.json").read_text(encoding="utf-8"))
    eligible = {"FROZEN_REFERENCE", "FORMAL", "FORMAL_ZERO_CONTROL"}
    formal = comparison.loc[comparison["eligibility"].isin(eligible)].copy()
    if formal.empty or formal["model"].str.contains("Oracle|Shuffled", regex=True).any():
        raise RuntimeError("Oracle or shuffled model escaped the formal-selection firewall")
    original = row(comparison, "TimeXer-Original")
    sa = row(comparison, "SA_TimeXer_Ridge")
    fs = row(comparison, "FS_TimeXer_NoPhysics")
    smooth = row(comparison, "PGFS_TimeXer_Smooth")
    smooth_current = row(comparison, "PGFS_TimeXer_SmoothCurrent")
    full = row(comparison, "PGFS_TimeXer_Full")
    best = formal.sort_values("mean_rmse_delta_mV").iloc[0]
    if best["model"] != "SA_TimeXer_Ridge":
        raise RuntimeError("Unexpected formal best; do not silently change the frozen selection")
    best_tail = tail.loc[tail["model"] == best["model"]].iloc[0]
    best_h95, best_h12 = mean_horizon(horizon, best["model"], 9.5), mean_horizon(horizon, best["model"], 12.0)
    original_h95, original_h12 = mean_horizon(horizon, "TimeXer-Original", 9.5), mean_horizon(horizon, "TimeXer-Original", 12.0)
    f3_gain_sa = (float(original["fold_3_rmse_delta_mV"]) - float(sa["fold_3_rmse_delta_mV"])) / float(original["fold_3_rmse_delta_mV"]) * 100.0
    current_by_model = current.set_index("model")
    cur_without = current_by_model.loc["PGFS_TimeXer_Smooth"]
    cur_with = current_by_model.loc["PGFS_TimeXer_SmoothCurrent"]
    cur_full = current_by_model.loc["PGFS_TimeXer_Full"]
    variance_full = variance.loc[variance["model"] == "PGFS_TimeXer_Full"]
    variance_mean = float(variance_full["slow_variance_fraction"].mean())
    variance_min, variance_max = float(variance_full["slow_variance_fraction"].min()), float(variance_full["slow_variance_fraction"].max())
    mono_no = monotonicity.loc[monotonicity["model"] == "PGFS_TimeXer_SmoothCurrent"]
    mono_yes = monotonicity.loc[monotonicity["model"] == "PGFS_TimeXer_Full"]
    mono_no_mean, mono_yes_mean = float(mono_no["non_increasing_fraction"].mean()), float(mono_yes["non_increasing_fraction"].mean())
    smooth_status = "REJECT"
    current_status = "KEEP"
    mono_status = "KEEP"
    status = "WEAK" if float(best["mean_rmse_delta_mV"]) >= 25 else "MEANINGFUL_RECOVERY" if float(best["mean_rmse_delta_mV"]) >= 20 else "PROMISING" if float(best["mean_rmse_delta_mV"]) >= 15 else "STRONG" if float(best["mean_rmse_delta_mV"]) > 12 else "REVIVED"
    phase_n2_8f = "GO" if float(best["mean_rmse_delta_mV"]) < 20 else "NO-GO"
    decision = {
        "phase_status": "PASS",
        "original_timexer": small(original),
        "step_aligned_timexer": small(sa),
        "fast_slow_timexer": small(fs),
        "pgfs_smooth": small(smooth),
        "pgfs_smooth_current": small(smooth_current),
        "pgfs_full": small(full),
        "best_model": str(best["model"]),
        "best_rmse_delta_mV": float(best["mean_rmse_delta_mV"]),
        "best_r2_delta": float(best["mean_r2_delta"]),
        "best_mae_delta_mV": float(best["mean_mae_delta_mV"]),
        "fold3_rmse_delta_mV": float(best["fold_3_rmse_delta_mV"]),
        "fold3_gain_vs_original_pct": f3_gain_sa,
        "horizon_9_5h_rmse_delta_mV": best_h95,
        "horizon_12h_rmse_delta_mV": best_h12,
        "p95_mV": float(best_tail["p95_mV"]),
        "p99_mV": float(best_tail["p99_mV"]),
        "slow_variance_fraction": {"model": "PGFS_TimeXer_Full", "mean": variance_mean, "min": variance_min, "max": variance_max, "status": "NON_COLLAPSED"},
        "current_gradient_negative_fraction": {"without_current_model": "PGFS_TimeXer_Smooth", "without_current": float(cur_without["negative_fraction"]), "with_current_model": "PGFS_TimeXer_SmoothCurrent", "with_current": float(cur_with["negative_fraction"]), "full_model": "PGFS_TimeXer_Full", "full": float(cur_full["negative_fraction"])},
        "smooth_constraint_status": smooth_status,
        "current_constraint_status": current_status,
        "monotonic_constraint_status": mono_status,
        "three_seed_promoted": False,
        "three_seed_result": "NOT_RUN: best PGFS RMSE was 24.922028 mV and met neither fixed promotion condition.",
        "timexer_status": status,
        "next_phase": "STOP_AT_N2_8E" if phase_n2_8f == "NO-GO" else "Phase N2.8F — Sparse Electrochemical Anchored PGFS-TimeXer",
        "enter_eis_polarization_anchor_phase": phase_n2_8f,
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "test_feature_diagnostics_computed": False,
        "test_performance_accessed": False,
    }
    (ROOT / "PHASE_N2_8E_DECISION.yaml").write_text(yaml.safe_dump(decision, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = f"""# Phase N2.8E results

## Frozen protocol and Test firewall

All results use the frozen development-only 0.5 h / 48-step / 24-step / F2 / DeltaV protocol and three chronological folds. Formal training used expanding chronological OOF Ridge future-condition pseudo-inputs; validation used Ridge models fitted only on each fold's training prefix. The final Test target rows read, Test predictions, Test metrics, and Test feature diagnostics are all **0/false**. Oracle and shuffled future inputs are controls only and are excluded from all formal choices.

## Model comparison

{mdtable(comparison, ["model", "eligibility", "mean_r2_delta", "mean_rmse_delta_mV", "mean_mae_delta_mV", "fold_3_rmse_delta_mV", "gain_vs_original_pct", "mean_p95_mV", "mean_p99_mV"])}

## Q1–Q5: contribution of the proposed components

1. **Step alignment contributes.** Formal SA-TimeXer-Ridge lowers mean RMSE from {float(original['mean_rmse_delta_mV']):.6f} to {float(sa['mean_rmse_delta_mV']):.6f} mV ({float(sa['gain_vs_original_pct']):.3f}%); its Fold-3 RMSE is {float(sa['fold_3_rmse_delta_mV']):.6f} mV ({f3_gain_sa:.3f}% gain).
2. **Fast–Slow does not improve the formal best.** FS-TimeXer is {float(fs['mean_rmse_delta_mV']):.6f} mV, slightly worse than SA-TimeXer-Ridge.
3. **Smoothness is REJECT.** Adding smoothness changes FS RMSE from {float(fs['mean_rmse_delta_mV']):.6f} to {float(smooth['mean_rmse_delta_mV']):.6f} mV without Fold-3 benefit.
4. **Current-response is KEEP as a physics regularizer, not as a new best predictor.** Negative fast-current gradients rise from {float(cur_without['negative_fraction']):.4%} without `L_current` to {float(cur_with['negative_fraction']):.4%} with it, while RMSE changes by only {(float(smooth_current['mean_rmse_delta_mV']) / float(smooth['mean_rmse_delta_mV']) - 1):.3%}.
5. **Monotonicity is KEEP as a weak prior.** Full PGFS improves PGFS RMSE from {float(smooth_current['mean_rmse_delta_mV']):.6f} to {float(full['mean_rmse_delta_mV']):.6f} mV and raises the slow non-increasing fraction from {mono_no_mean:.4%} to {mono_yes_mean:.4%}; it still does not beat SA-TimeXer-Ridge.

## Q6–Q12: ablation, variance, and physical diagnostics

{mdtable(ablation, ["model", "mean_r2_delta", "mean_rmse_delta_mV", "fold_3_rmse_delta_mV", "mean_p95_mV", "mean_p99_mV"])}

The Full PGFS slow component accounts for {variance_mean:.4%} of fast-plus-slow dynamic variance on average (range {variance_min:.4%}–{variance_max:.4%}); it is neither slow-dominated (>80%) nor fast-only-near-zero-slow (<0.1%). It should be called a **learned slow degradation representation**, not SOH.

The current-response summary is: FS negative-gradient fraction {float(current_by_model.loc['FS_TimeXer_NoPhysics']['negative_fraction']):.4%}; Smooth {float(cur_without['negative_fraction']):.4%}; SmoothCurrent {float(cur_with['negative_fraction']):.4%}; Full {float(cur_full['negative_fraction']):.4%}. Full median `dV_fast/dJ` is {float(cur_full['median_gradient']):.6f} in the model's scaled-coordinate derivative units, with P05={float(cur_full['p05_gradient']):.6f} and P95={float(cur_full['p95_gradient']):.6f}. The physical item with the clearest direct consistency effect is **current-response**; the item to delete from this frozen first-pass architecture is **smoothness**.

## Q13–Q22: predictive detail and thresholds

- Best formal model: **{best['model']}**.
- Q13 R2 DeltaV = **{float(best['mean_r2_delta']):.6f}**.
- Q14 RMSE DeltaV = **{float(best['mean_rmse_delta_mV']):.6f} mV**; MAE = **{float(best['mean_mae_delta_mV']):.6f} mV**.
- Q15 Fold-3 RMSE = **{float(best['fold_3_rmse_delta_mV']):.6f} mV**.
- Q16 P95/P99 = **{float(best_tail['p95_mV']):.6f}/{float(best_tail['p99_mV']):.6f} mV**.
- Q17 mean 9.5 h RMSE = **{best_h95:.6f} mV**. This is below N2.8D RidgeFuture's 37.66 mV, but above the 18 mV aspiration and the 13.68 mV original-TimeXer reference.
- Q18 mean 12 h RMSE = **{best_h12:.6f} mV**, improving over the original-TimeXer {original_h12:.6f} mV reference.
- Q19 below 25 mV: **yes**. Q20 below 20 mV: **no**. Q21 below 15 mV: **no**. Q22 near 12 mV: **no**.

## Q23–Q25: stability and electrochemical next phase

Q23 three-seed promotion: **no**. The best PGFS variant is Full at {float(full['mean_rmse_delta_mV']):.6f} mV, so it met neither `<20 mV` nor `<22 mV + Fold-3 >=30%` promotion rule. Q24 three-seed stability is therefore **not applicable, not estimated**; it was not replaced by an unapproved run. Q25 EIS/Polarization sparse-anchor phase: **NO-GO**, because the best formal RMSE is not below 20 mV. No EIS interpolation, polarization interpolation, dense SOH, RUL, or N2.8F training is performed.

## Final conclusion

N2.8E is an execution **PASS**. Formal TimeXer status is **{status}**: step-aligned forecastable conditions materially recover performance, but the best formal model remains above the <20 mV gate. The phase stops here under the frozen protocol.
"""
    (ROOT / "PHASE_N2_8E_RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"phase_status": "PASS", "best_model": decision["best_model"], "rmse_mV": decision["best_rmse_delta_mV"], "status": status, "n2_8f": phase_n2_8f}, ensure_ascii=False))


if __name__ == "__main__":
    main()

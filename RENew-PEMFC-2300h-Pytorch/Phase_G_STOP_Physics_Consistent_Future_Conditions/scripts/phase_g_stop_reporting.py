#!/usr/bin/env python
"""Generate final reports, decision record, paper scaffolding, and figures."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE = "#0F4D92"
ORANGE = "#E67E22"
GREEN = "#238B45"
RED = "#B2182B"
GRAY = "#6B7280"


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def write(relative: str, content: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def f(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def model_row(frame: pd.DataFrame, model: str, fold: str | None = None) -> pd.Series:
    selected = frame[frame.model.eq(model)]
    if fold is not None:
        selected = selected[selected.fold.eq(fold)]
    return selected.iloc[0]


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def figures() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    })
    regimes = read_csv("outputs/zero_current/operating_regime_counts.csv")
    states = regimes[(regimes.scope == "all_development") & (regimes.record_type == "state")]
    plt.figure(figsize=(4.6, 3.2))
    bars = plt.bar(states.regime, states.sample_count, color=[BLUE, ORANGE, GRAY])
    for bar, value in zip(bars, states.sample_count):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 25, str(int(value)), ha="center", va="bottom")
    plt.ylabel("Development bins")
    plt.title("Observed current-regime support")
    savefig("g1_operating_regime_support")

    future = read_csv("outputs/future_forecast/future_condition_forecast_metrics.csv")
    future = future[future.fold == "mean_3fold"]
    violation = read_csv("outputs/future_forecast/physical_violation_rates.csv")
    violation = violation[violation.fold == "mean_3fold"]
    labels = ["P0\nRidge", "P1\nProjection", "P2\nq-aware"]
    order = ["P0_Original_Ridge", "P1_Physics_Projected_Ridge", "P2_Zero_Current_Aware"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1))
    colors = [GRAY, ORANGE, GREEN]
    axes[0].bar(labels, [model_row(future, x).q_rmse for x in order], color=colors)
    axes[0].set_ylabel("q RMSE")
    axes[0].set_title("Zero-current fraction")
    axes[1].bar(labels, [model_row(future, x).current_mean_mae_A for x in order], color=colors)
    axes[1].set_ylabel("MAE (A)")
    axes[1].set_title("Current mean")
    axes[2].bar(labels, [model_row(violation, x).physical_violation_trajectory_rate * 100 for x in order], color=colors)
    axes[2].set_ylabel("Trajectories with violation (%)")
    axes[2].set_title("Physical consistency")
    savefig("future_condition_forecast_quality")

    frozen = read_csv("outputs/frozen_timexer/frozen_timexer_summary.csv")
    x = np.arange(3)
    folds = ["fold_1_rmse_delta_mV", "fold_2_rmse_delta_mV", "fold_3_rmse_delta_mV"]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    width = 0.24
    for offset, (model, label, color) in enumerate(zip(order, ["P0 Ridge", "P1 projected", "P2 q-aware"], colors)):
        row = model_row(frozen, model)
        ax.bar(x + (offset - 1) * width, [row[key] for key in folds], width, label=label, color=color)
    ax.set_xticks(x, ["Fold 1", "Fold 2", "Fold 3"])
    ax.set_ylabel("RMSE$_{ΔV}$ (mV)")
    ax.set_title("Frozen SA-TimeXer: matched-origin comparison")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    savefig("frozen_timexer_fold_comparison")

    horizon = read_csv("outputs/horizon/frozen_timexer_per_horizon.csv")
    horizon = horizon[horizon.fold == "mean_3fold"]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for model, label, color in zip(order, ["P0 Ridge", "P1 projected", "P2 q-aware"], colors):
        part = horizon[horizon.model == model]
        ax.plot(part.horizon_hours, part.rmse_delta_mV, label=label, color=color, linewidth=1.8)
    ax.axvline(9.5, color=RED, linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("RMSE$_{ΔV}$ (mV)")
    ax.set_title("Frozen SA-TimeXer horizon profile")
    ax.legend(frameon=False, ncol=3)
    savefig("frozen_timexer_horizon_profile")

    oracle = read_csv("outputs/ablation/oracle_replacement_audit.csv")
    oracle_mean = oracle.groupby("variant", sort=False).rmse_delta_mV.mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.barh(range(len(oracle_mean)), oracle_mean.values, color=[GREEN, BLUE, ORANGE, GRAY])
    ax.set_yticks(range(len(oracle_mean)), [x.replace("_DIAGNOSTIC", "").replace("_", " ") for x in oracle_mean.index])
    ax.set_xlabel("Mean RMSE$_{ΔV}$ (mV)")
    ax.set_title("Diagnostic true-future substitutions (Development only)")
    for bar, value in zip(bars, oracle_mean.values):
        ax.text(value + .25, bar.get_y() + bar.get_height()/2, f"{value:.2f}", va="center")
    savefig("oracle_replacement_consistency")

    lstm = read_csv("outputs/lstm_transfer/lstm_transfer_summary.csv")
    lstm_order = ["L0_Original_LSTM", "L1_Original_Ridge_Future", "L2_Physics_Consistent_Future"]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    width = 0.24
    for offset, (model, label, color) in enumerate(zip(lstm_order, ["L0", "L1", "L2"], [BLUE, GRAY, GREEN])):
        row = model_row(lstm, model)
        ax.bar(x + (offset - 1) * width, [row[key] for key in folds], width, label=label, color=color)
    ax.set_xticks(x, ["Fold 1", "Fold 2", "Fold 3"])
    ax.set_ylabel("RMSE$_{ΔV}$ (mV)")
    ax.set_title("Strong LSTM transfer: residual heads fail")
    ax.legend(frameon=False, ncol=3)
    savefig("lstm_transfer_fold_comparison")


def reports() -> None:
    g1 = json.loads((OUT / "physical_audit/g1_raw_audit_summary.json").read_text(encoding="utf-8"))
    future = read_csv("outputs/future_forecast/future_condition_forecast_metrics.csv")
    future_mean = future[future.fold == "mean_3fold"].set_index("model")
    violation = read_csv("outputs/future_forecast/physical_violation_rates.csv")
    violation_mean = violation[violation.fold == "mean_3fold"].set_index("model")
    frozen = read_csv("outputs/frozen_timexer/frozen_timexer_summary.csv").set_index("model")
    tail = read_csv("outputs/tail/frozen_timexer_tail_metrics.csv").set_index("model")
    horizon = read_csv("outputs/horizon/frozen_timexer_per_horizon.csv")
    horizon = horizon[horizon.fold == "mean_3fold"]
    gate = read_csv("outputs/frozen_timexer/gate_b_audit.csv").set_index("candidate")
    g4 = json.loads((OUT / "frozen_timexer/g4_result.json").read_text(encoding="utf-8"))
    g4sum = read_csv("outputs/frozen_timexer/g4_retraining_summary.csv").set_index("model")
    g5 = json.loads((OUT / "lstm_transfer/g5_result.json").read_text(encoding="utf-8"))
    lstm = read_csv("outputs/lstm_transfer/lstm_transfer_summary.csv").set_index("model")
    lstm_tail = read_csv("outputs/lstm_transfer/lstm_transfer_tail_metrics.csv").set_index("model")
    oracle = read_csv("outputs/ablation/oracle_replacement_audit.csv").groupby("variant").mean(numeric_only=True)
    transitions = read_csv("outputs/zero_current/transition_event_metrics.csv")
    regimes = read_csv("outputs/zero_current/operating_regime_counts.csv")

    area = g1["active_area_estimation"]["active_area_cm2"]
    rel = g1["area_relation_by_statistic"]
    write("PHYSICAL_RELATION_AUDIT.md", f"""# Physical Relation Audit

Status: **PASS** (Development only; Test unopened).

The raw definition is `q = count(|I.S.act| <= 0.10 A) / raw sample count` in each causal 0.5 h bin. Population standard deviation (`ddof=0`) is used, and the audited Development raw rows contain no missing current values. All current, current-density, and q summaries were recomputed from the same bins.

Fold-1 Train estimates a fixed active area of **{area:.6f} cm²** from the median current/current-density ratio. The zero-intercept slopes for mean, standard deviation, and peak-to-peak are {rel['mean']['zero_intercept_slope_cm2']:.6f}, {rel['std']['zero_intercept_slope_cm2']:.6f}, and {rel['peak_to_peak']['zero_intercept_slope_cm2']:.6f} cm². Their Pearson correlations are {rel['mean']['pearson_r']:.9f}, {rel['std']['pearson_r']:.9f}, and {rel['peak_to_peak']['pearson_r']:.9f}; median relative identity errors are {rel['mean']['median_relative_identity_error']:.3e}, {rel['std']['median_relative_identity_error']:.3e}, and {rel['peak_to_peak']['median_relative_identity_error']:.3e}.

The verified relations are `J=I/A`, nonnegative current/statistical ranges, `0<=q<=1`, `mean_I=(1-q)mu_on`, and the population-mixture variance identity. The mixture mean and standard-deviation median absolute reconstruction errors are both numerically zero. Peak-to-peak is **not** derivable from mean and variance and remains independently predicted. No rule linking q to zero voltage is valid. There are no q=1 bins, so q=1 classification is **NOT_ESTIMABLE**.

The originally drafted absolute identity threshold was below source export quantization. Before any downstream inference, it was replaced by the preregistered slope/correlation/relative-error criterion documented in the plan. Raw Test rows parsed: **0**.
""")

    all_dev = regimes[(regimes.scope == "all_development") & (regimes.record_type == "state")].set_index("regime")
    write("ZERO_CURRENT_EVENT_AUDIT.md", f"""# Zero-Current Event Audit

Status: **SUFFICIENT FOR MIXED-CURRENT ANALYSIS; q=1 NOT ESTIMABLE**.

Across Development there are **{int(all_dev.loc['0<q<1','sample_count'])}** mixed-current bins forming **{g1['event_sufficiency']['mixed_segments_all_development']}** independent segments, plus **{g1['event_sufficiency']['nonzero_to_mixed_events']}** nonzero-to-mixed and **{g1['event_sufficiency']['mixed_to_nonzero_events']}** mixed-to-nonzero transitions. Validation support is 77, 67, and 156 mixed segments in Folds 1–3. The mixed-bin proportion is {100*all_dev.loc['0<q<1','proportion']:.2f}%.

No q=1 bin was observed. Therefore this work uses the terms *zero-current fraction*, *mixed-current operating regime*, and *zero-current transition*; it does not claim measured shutdown/startup state. Event counts are based on unique bins or state runs rather than treating overlapping 24-step targets as independent events.

P2 reduces frozen-TimeXer mixed-current RMSE in all folds (31.87→17.59, 35.26→23.72, and 39.98→35.93 mV for P0→P2), but Fold3 continuous-nonzero RMSE worsens (26.89→32.81 mV). This localization explains why the overall Fold3 metric does not improve.
""")

    p0, p1, p2 = [future_mean.loc[x] for x in ("P0_Original_Ridge", "P1_Physics_Projected_Ridge", "P2_Zero_Current_Aware")]
    write("FUTURE_CONDITION_FORECAST_AUDIT.md", f"""# Future-Condition Forecast Audit

All forecasts are causal expanding-window predictions with target-completion purge and Train-only scaling. P0 and P2 use the same legal OOF origins. P1 is a deterministic minimal projection of P0.

| Scheme | q MAE | q RMSE | mixed F1 | current-mean MAE (A) | current-std MAE (A) | current-p2p MAE (A) | violating trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | {p0.q_mae:.6f} | {p0.q_rmse:.6f} | {p0.mixed_event_f1:.6f} | {p0.current_mean_mae_A:.3f} | {p0.current_std_mae_A:.3f} | {p0.current_peak_to_peak_mae_A:.3f} | {100*violation_mean.loc['P0_Original_Ridge','physical_violation_trajectory_rate']:.1f}% |
| P1 | {p1.q_mae:.6f} | {p1.q_rmse:.6f} | {p1.mixed_event_f1:.6f} | {p1.current_mean_mae_A:.3f} | {p1.current_std_mae_A:.3f} | {p1.current_peak_to_peak_mae_A:.3f} | {100*violation_mean.loc['P1_Physics_Projected_Ridge','physical_violation_trajectory_rate']:.1f}% |
| P2 | {p2.q_mae:.6f} | {p2.q_rmse:.6f} | {p2.mixed_event_f1:.6f} | {p2.current_mean_mae_A:.3f} | {p2.current_std_mae_A:.3f} | {p2.current_peak_to_peak_mae_A:.3f} | {100*violation_mean.loc['P2_Zero_Current_Aware','physical_violation_trajectory_rate']:.1f}% |

q=1 Precision/Recall/F1 are **NOT_ESTIMABLE**, not zero. P2 sharply improves q and current mean prediction, while peak-to-peak is unchanged because it remains the independent Ridge forecast. P1/P2 eliminate measured range and current-density identity violations. P0 flags every trajectory because unconstrained q leaves `[0,1]` and/or current-density identities beyond tolerance.
""")

    def hr(model: str, h: float) -> float:
        return float(horizon[(horizon.model == model) & np.isclose(horizon.horizon_hours, h)].rmse_delta_mV.iloc[0])
    write("FROZEN_TIMEXER_COMPARISON.md", f"""# Frozen SA-TimeXer Comparison

The same N2.8F M0 weights, history tensors, validation origins, targets, and scoring code are used. Only future-condition tensors change.

| Scheme | R²_delta | RMSE (mV) | MAE (mV) | Fold1 | Fold2 | Fold3 | pooled P95 | pooled P99 | 9.5 h | 12 h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | {frozen.loc['P0_Original_Ridge','mean_r2_delta']:.6f} | {frozen.loc['P0_Original_Ridge','mean_rmse_delta_mV']:.3f} | {frozen.loc['P0_Original_Ridge','mean_mae_delta_mV']:.3f} | {frozen.loc['P0_Original_Ridge','fold_1_rmse_delta_mV']:.3f} | {frozen.loc['P0_Original_Ridge','fold_2_rmse_delta_mV']:.3f} | {frozen.loc['P0_Original_Ridge','fold_3_rmse_delta_mV']:.3f} | {tail.loc['P0_Original_Ridge','p95_mV']:.3f} | {tail.loc['P0_Original_Ridge','p99_mV']:.3f} | {hr('P0_Original_Ridge',9.5):.3f} | {hr('P0_Original_Ridge',12.0):.3f} |
| P1 | {frozen.loc['P1_Physics_Projected_Ridge','mean_r2_delta']:.6f} | {frozen.loc['P1_Physics_Projected_Ridge','mean_rmse_delta_mV']:.3f} | {frozen.loc['P1_Physics_Projected_Ridge','mean_mae_delta_mV']:.3f} | {frozen.loc['P1_Physics_Projected_Ridge','fold_1_rmse_delta_mV']:.3f} | {frozen.loc['P1_Physics_Projected_Ridge','fold_2_rmse_delta_mV']:.3f} | {frozen.loc['P1_Physics_Projected_Ridge','fold_3_rmse_delta_mV']:.3f} | {tail.loc['P1_Physics_Projected_Ridge','p95_mV']:.3f} | {tail.loc['P1_Physics_Projected_Ridge','p99_mV']:.3f} | {hr('P1_Physics_Projected_Ridge',9.5):.3f} | {hr('P1_Physics_Projected_Ridge',12.0):.3f} |
| P2 | {frozen.loc['P2_Zero_Current_Aware','mean_r2_delta']:.6f} | {frozen.loc['P2_Zero_Current_Aware','mean_rmse_delta_mV']:.3f} | {frozen.loc['P2_Zero_Current_Aware','mean_mae_delta_mV']:.3f} | {frozen.loc['P2_Zero_Current_Aware','fold_1_rmse_delta_mV']:.3f} | {frozen.loc['P2_Zero_Current_Aware','fold_2_rmse_delta_mV']:.3f} | {frozen.loc['P2_Zero_Current_Aware','fold_3_rmse_delta_mV']:.3f} | {tail.loc['P2_Zero_Current_Aware','p95_mV']:.3f} | {tail.loc['P2_Zero_Current_Aware','p99_mV']:.3f} | {hr('P2_Zero_Current_Aware',9.5):.3f} | {hr('P2_Zero_Current_Aware',12.0):.3f} |

P0 reproduces 24.538245 mV within 0.001 mV. P1 improves only 0.39% and fails Gate B. P2 improves mean RMSE by {gate.loc['P2_Zero_Current_Aware','mean_rmse_improvement_pct']:.2f}%, improves two folds, reduces P99 by {-gate.loc['P2_Zero_Current_Aware','p99_change_pct']:.2f}%, reduces 9.5 h RMSE by {-gate.loc['P2_Zero_Current_Aware','horizon_9p5h_change_pct']:.2f}%, and removes measured violations, so it passes Gate B. Fold3 worsens from 27.885 to 30.382 mV and 12 h worsens from 33.528 to 35.475 mV; the gain is not uniform.

The paired moving-block bootstrap for P2 versus P0 has mean RMSE difference −1.070 mV with 95% interval [−4.879, 2.530] mV and probability of improvement 0.703. It describes uncertainty and does not override the preregistered point-estimate gate.
""")

    p0_oracle = oracle.loc["P0_Ridge_formal", "rmse_delta_mV"]
    q_oracle = oracle.loc["true_q_only_DIAGNOSTIC", "rmse_delta_mV"]
    group_oracle = oracle.loc["true_physical_current_group_DIAGNOSTIC", "rmse_delta_mV"]
    all_oracle = oracle.loc["all_true_future_conditions_DIAGNOSTIC", "rmse_delta_mV"]
    l0, l1, l2 = [lstm.loc[x] for x in ("L0_Original_LSTM", "L1_Original_Ridge_Future", "L2_Physics_Consistent_Future")]
    result_md = f"""# Phase G-STOP Results

## Decision

**Phase status: COMPLETE — STOP.** G1 passed, P2 passed frozen-TimeXer Gate B, seed42 retraining improved the fair P0 control, but the preregistered G4 three-seed trigger failed because Fold3 worsened. The mechanism then failed on the strong causal-SG LSTM. No new unique final paper model is selected, Test remains sealed, and no further model search is authorized.

## Required questions

1. **Are zero-current events sufficient?** Yes for mixed-current analysis: 624 independent mixed segments, 624 nonzero→mixed events, and 621 mixed→nonzero events. q=1 is absent and its metrics are NOT_ESTIMABLE.
2. **What is `zero_current_fraction`?** Within-bin raw-sample fraction satisfying `abs(I.S.act)<=0.10 A`; it is not a verified shutdown label.
3. **Which physical relations are verified?** Same-bin aggregation; population std (`ddof=0`); fixed-area `J=I/A` with A={area:.6f} cm²; nonnegative/range constraints; mixture mean and population variance identities. Peak-to-peak has no derived formula.
4. **Was the old 15.416839 mV Oracle physically inconsistent?** Yes: true q was combined with independently forecast current statistics. It remains a diagnostic marker, not a deployable or isolated causal result. Reproduction is {q_oracle:.3f} mV; the consistent current-group replacement is {group_oracle:.3f} mV and all-true is {all_oracle:.3f} mV versus P0 {p0_oracle:.3f} mV.
5. **P0 forecast quality:** q MAE/RMSE {p0.q_mae:.6f}/{p0.q_rmse:.6f}; mixed F1 {p0.mixed_event_f1:.4f}; current mean/std/p2p MAE {p0.current_mean_mae_A:.3f}/{p0.current_std_mae_A:.3f}/{p0.current_peak_to_peak_mae_A:.3f} A.
6. **P1 forecast quality:** q MAE/RMSE {p1.q_mae:.6f}/{p1.q_rmse:.6f}; mixed F1 {p1.mixed_event_f1:.4f}; current mean/std/p2p MAE {p1.current_mean_mae_A:.3f}/{p1.current_std_mae_A:.3f}/{p1.current_peak_to_peak_mae_A:.3f} A.
7. **P2 forecast quality:** q MAE/RMSE {p2.q_mae:.6f}/{p2.q_rmse:.6f}; mixed F1 {p2.mixed_event_f1:.4f}; current mean/std/p2p MAE {p2.current_mean_mae_A:.3f}/{p2.current_std_mae_A:.3f}/{p2.current_peak_to_peak_mae_A:.3f} A.
8. **Physical violation rate:** P0/P1/P2 trajectory rates are 100%/0%/0% under the audited range and identity rules.
9. **Frozen-TimeXer RMSE:** P0/P1/P2 = {frozen.loc['P0_Original_Ridge','mean_rmse_delta_mV']:.3f}/{frozen.loc['P1_Physics_Projected_Ridge','mean_rmse_delta_mV']:.3f}/{frozen.loc['P2_Zero_Current_Aware','mean_rmse_delta_mV']:.3f} mV.
10. **Which folds improve?** P2 improves Fold1 and Fold2; Fold3 worsens.
11. **Do zero/mixed-current regions improve?** Mixed-current RMSE improves in all folds. Continuous-nonzero improves in Folds 1–2 but worsens in Fold3.
12. **Does Fold3 improve?** No: 27.885→30.382 mV frozen; after fair retraining 27.885→28.524 mV.
13. **Does P99 improve?** Yes for frozen P2: 129.652→68.630 mV. Its maximum error nevertheless rises to 593.412 mV, so tail risk is not eliminated.
14. **Does 9.5 h worsen?** No: 26.074→19.864 mV. At 12 h, however, RMSE worsens to 35.475 mV.
15. **Gate B?** PASS for P2; FAIL for P1.
16. **Was TimeXer retraining triggered?** Yes, seed42 fair P0 and P2 were retrained with unchanged architecture/training controls.
17. **Did retraining improve?** Yes on mean seed42 RMSE: {g4sum.loc['P0_Retrained','mean_rmse_delta_mV']:.3f}→{g4sum.loc['P2_Retrained','mean_rmse_delta_mV']:.3f} mV ({g4['seed42_rmse_improvement_pct']:.2f}%). Fold3 worsened {g4['seed42_fold3_worsening_pct']:.2f}%, so the frozen multiseed trigger did not pass.
18. **Was LSTM transfer attempted?** Yes, using chronological OOF LSTM residuals, target-completion purge, Train-only preprocessing, and no Validation residual fitting.
19. **Independent LSTM gain?** No. L0/L1/L2 mean seed42 RMSE = {l0.mean_rmse_delta_mV:.3f}/{l1.mean_rmse_delta_mV:.3f}/{l2.mean_rmse_delta_mV:.3f} mV. The OOF residual correction does not transfer to the full-data incumbent LSTM.
20. **Three seeds?** Not in G4 or G5: each preregistered seed42 trigger failed. Historical LSTM three-seed evidence remains read-only context, not a new G5 result.
21. **Unique final paper model candidate?** No new candidate. The original causal-SG LSTM remains the incumbent benchmark; P2+SA-TimeXer is promising but not multiseed-frozen.
22. **Supported paper contributions:** verified causal physical-consistency construction; P2 future-condition accuracy/violation reduction; matched-weight frozen-TimeXer Gate-B gain; explicit evidence that gains are fold- and backbone-dependent.
23. **Rejected assumptions:** q is not a shutdown label; q-only Oracle is not physically self-consistent; projection alone is insufficient; future-condition residual correction does not automatically improve a strong LSTM; performance is not uniformly improved at Fold3/12 h.
24. **Test firewall:** intact. Test target rows read=0; predictions=false; metrics=false; feature diagnostics=false; performance accessed=false.

## Stop rationale

The study answers its core question conditionally: causal physics-consistent future conditions can materially improve the frozen SA-TimeXer path, but the evidence does not establish a universal or final model improvement. The negative LSTM transfer is retained, not optimized away. The next permitted action is documentation/review or a separately authorized final protocol decision—not Test execution and not another architecture search.
"""
    write("PHASE_G_STOP_RESULTS.md", result_md)

    decision = {
        "phase_status": "COMPLETE_STOP_NO_NEW_FINAL_MODEL",
        "physical_relation_audit": {
            "status": "PASS",
            "verified_relations": ["same-bin causal aggregation", "J=I/A with Fold-1 Train area", "population mixture mean and variance", "nonnegative statistics", "0<=q<=1"],
            "invalid_relations": ["peak-to-peak derived from mean/variance", "q implies zero voltage", "q is a measured shutdown label"],
        },
        "zero_current_audit": {"event_count": 624, "q1_bins": 0, "sufficient_for_claim": True, "q1_metrics": "NOT_ESTIMABLE"},
        "forecast_models": {
            name: {
                "forecast_metrics": {
                    "q_mae": float(future_mean.loc[key, "q_mae"]),
                    "q_rmse": float(future_mean.loc[key, "q_rmse"]),
                    "mixed_event_f1": float(future_mean.loc[key, "mixed_event_f1"]),
                    "current_mean_mae_A": float(future_mean.loc[key, "current_mean_mae_A"]),
                },
                "physical_violation_rate": float(violation_mean.loc[key, "physical_violation_trajectory_rate"]),
                "downstream_rmse_mV": float(frozen.loc[key, "mean_rmse_delta_mV"]),
            }
            for name, key in {"P0": "P0_Original_Ridge", "P1": "P1_Physics_Projected_Ridge", "P2": "P2_Zero_Current_Aware"}.items()
        },
        "gate_b_passed": True,
        "gate_b_winner": "P2_Zero_Current_Aware",
        "timexer_retraining": {
            "status": "SEED42_COMPLETE_MULTISEED_NOT_TRIGGERED",
            "results": {"P0_rmse_mV": float(g4sum.loc["P0_Retrained", "mean_rmse_delta_mV"]), "P2_rmse_mV": float(g4sum.loc["P2_Retrained", "mean_rmse_delta_mV"]), "improvement_pct": g4["seed42_rmse_improvement_pct"], "fold3_worsening_pct": g4["seed42_fold3_worsening_pct"]},
        },
        "lstm_transfer": {
            "status": "FAILED_PROMOTION_STOP",
            "results": {name: float(lstm.loc[key, "mean_rmse_delta_mV"]) for name, key in {"L0": "L0_Original_LSTM", "L1": "L1_Original_Ridge_Future", "L2": "L2_Physics_Consistent_Future"}.items()},
        },
        "multiseed": {"status": "NOT_RUN_PREREGISTERED_TRIGGERS_FAILED", "results": None},
        "final_candidate": None,
        "incumbent_reference": "Original causal-SG LSTM",
        "paper_claims_supported": ["raw-verified physical consistency", "P2 improves frozen SA-TimeXer Development point-estimate gate", "mechanism is fold- and backbone-dependent"],
        "paper_claims_unsupported": ["universal backbone improvement", "stable multiseed new model", "shutdown classification", "SOH prediction", "RUL prediction", "Test performance", "cross-stack generalization"],
        "test_performance_accessed": False,
        "next_action": "STOP; preserve artifacts; Test remains sealed pending separate authorization",
    }
    write("PHASE_G_STOP_DECISION.yaml", yaml.safe_dump(decision, sort_keys=False, allow_unicode=True))

    write("paper/PAPER_OUTLINE.md", """# Paper Outline

## Working title

Physics-Consistent Future Operating-Condition Forecasting for Multi-Horizon Voltage Prediction of Proton Exchange Membrane Fuel Cells

## Research question

Can future operating conditions inferred strictly from information available at the forecast origin, and constrained only by raw-verified physical identities, improve 12 h dynamic PEMFC voltage forecasts?

## Structure

1. Introduction: unavailable future conditions as a deployment gap; dynamic voltage rather than SOH/RUL.
2. Data and causal protocol: 0.5 h bins, 24 h history, 12 h horizon, 14 historical and 9 future-condition variables, three expanding Development folds, sealed Test.
3. Physical audit: operational definition of q, active-area identity, mixture moments, unsupported relations.
4. Methods: P0 Ridge, P1 minimal projection, P2 two-part q-aware forecast; frozen SA-TimeXer; conditional retraining; OOF LSTM transfer.
5. Results: future-condition accuracy, physical violations, frozen-weight causal comparison, retraining, regime/horizon/tail diagnostics, negative LSTM transfer.
6. Discussion: why projection alone is weak; why P2 gains are nonuniform; backbone dependence; limitations from absent q=1 and unopened Test.
7. Conclusion: conditional evidence, no universal final model claim.

No literature citations are inserted here because a verified literature search was outside this execution scope. Citation placeholders must be populated only from checked primary sources before submission.
""")

    write("paper/PAPER_METHOD_DRAFT.md", f"""# Methods Draft

## Task and data protocol

The study formulates dynamic PEMFC voltage prediction as a 24-step multi-horizon task at 0.5 h resolution. Each sample uses 48 historical steps (24 h) and predicts `DeltaV(t+h)=V(t+h)-V(t)` for the next 24 steps (12 h). Historical inputs are the frozen 14-variable F2 set. Candidate future-condition inputs comprise nine current, current-density, transition, and zero-current-fraction variables. Evaluation uses three chronological expanding Development folds with identical origins and target timestamps. The final Test interval is not read for features, targets, prediction, diagnostics, or selection.

## Physical audit

For each causal 0.5 h bin, q is the fraction of raw samples satisfying `abs(I.S.act)<=0.10 A`. It is an operating-regime statistic, not a measured shutdown label. Raw recomputation established population standard deviation (`ddof=0`) and a Fold-1-Train active area of {area:.6f} cm². Current-density moments obey `J=I/A` to source-resolution tolerance. For mixed-current bins, the nonzero-current conditional mean and variance satisfy the mixture identities. Peak-to-peak is independently predicted because it cannot be uniquely recovered from those moments. No constraint forces voltage to zero when q is high.

## Future-condition forecasts

P0 is the repaired multi-output Ridge forecast inherited without hyperparameter changes. P1 applies minimal deterministic projection: nonnegative current/statistical values, q clipped to `[0,1]`, and current density reconstructed from Fold-Train area. P2 uses per-horizon balanced logistic regression for `q>0`, Ridge regression for conditional q, and Ridge forecasts of nonzero-current conditional mean and population standard deviation. Audited mixture identities reconstruct current mean and standard deviation; peak-to-peak remains independent. All OOF predictions use expanding chronological blocks, Train-only imputation/scaling, and the purge condition `max(training target completion)<=prediction origin`.

## Downstream evaluation

First, frozen N2.8F SA-TimeXer M0 checkpoints receive P0, P1, or P2 futures while all weights and historical inputs remain fixed. Gate B requires at least 5% mean RMSE gain, improvement in at least two folds, no more than 5% P99 worsening, no more than 10% worsening at 9.5 h, and lower physical violations. Overlap-aware uncertainty is summarized by a 19-origin moving-block bootstrap with 2,000 replications.

After P2 passed Gate B, P0 and P2 SA-TimeXer variants were retrained with the unchanged architecture and optimization protocol. A seed42 screen controlled whether additional seeds were permitted. Finally, the mechanism was transferred to the strongest causal-SG LSTM using one standardized Ridge residual head per horizon. Head targets came only from chronological OOF LSTM residuals; Validation residuals were never used. Seed42 had to pass a preregistered improvement and tail-safety screen before additional seeds.

## Metrics and interpretation

Primary metrics are `R²_delta`, RMSE and MAE in mV, fold-level RMSE, pooled P95/P99, and RMSE at 9.5 h and 12 h. Regime analyses are diagnostic. True future conditions are Oracle diagnostics only and cannot select a deployable model. This work does not construct dense SOH, RUL, or EOL labels.
""")

    write("paper/PAPER_EXPERIMENT_MATRIX.md", """# Experiment Matrix

| Stage | Variant | Learned component | Formal status | Seeds | Outcome |
|---|---|---|---|---|---|
| G1 | raw physical audit | none | formal audit | n/a | PASS |
| G1 | true q only | none; true future diagnostic | diagnostic only | 42 | physically inconsistent input combination |
| G1 | true physical current group | none; true future diagnostic | diagnostic only | 42 | confirms future-current information gap |
| G2/G3 | P0 | repaired Ridge futures; frozen SA-TimeXer | formal | 42 | baseline reproduced |
| G2/G3 | P1 | minimal physical projection; frozen SA-TimeXer | formal | 42 | physical pass; Gate B fail |
| G2/G3 | P2 | q-aware light forecast; frozen SA-TimeXer | formal | 42 | Gate B pass |
| G4 | P0 retrained | unchanged SA-TimeXer | formal control | 42 | 24.538 mV |
| G4 | P2 retrained | unchanged SA-TimeXer | conditional formal | 42 | 21.553 mV; multiseed trigger fail |
| G5 | L0 | original causal-SG LSTM | formal control | 42 | 7.926 mV |
| G5 | L1 | P0 residual head | formal | 42 | 21.043 mV; fail |
| G5 | L2 | P2 residual head | formal | 42 | 21.467 mV; fail |
| Test | all | none | sealed | none | NOT_RUN |
""")

    write("paper/PAPER_CLAIM_EVIDENCE_TABLE.md", """# Claim–Evidence Table

| Claim | Status | Direct evidence | Boundary |
|---|---|---|---|
| Current/current density obey one fixed-area relation | Supported | raw audit slopes/correlations and identity errors | Development data only |
| P2 predicts mixed-current occurrence better than P0 | Supported | mixed-event F1 0.990 vs 0.510 | q=1 absent |
| P2 removes audited future-condition violations | Supported | trajectory violation 0% vs P0 100% | only audited constraints |
| P2 improves frozen SA-TimeXer mean RMSE | Supported by point-estimate Gate B | 24.538→21.347 mV; 2 folds | bootstrap CI crosses zero; Fold3 worsens |
| P2 retraining improves seed42 SA-TimeXer | Supported for seed42 | 24.538→21.553 mV | multiseed trigger failed |
| Physics-consistent futures improve the strong LSTM | Refuted for tested head | 7.926→21.467 mV | one preregistered transfer design |
| q is a shutdown label | Unsupported | no shutdown log and no q=1 bins | use operating-regime terminology |
| New method is a stable final model | Unsupported | no new three-seed result | incumbent LSTM remains reference |
| Model predicts SOH or RUL | Unsupported | task is dynamic voltage prediction | no dense health/RUL ground truth |
| Test or cross-stack generalization | Unsupported | Test sealed; one experimental system | separate authorization/study needed |
""")


def main() -> None:
    reports()
    figures()
    config = yaml.safe_load((ROOT / "configs/phase_g_stop_frozen_config.yaml").read_text(encoding="utf-8"))
    frozen_sources = [
        ROOT / "configs/phase_g_stop_frozen_config.yaml",
        Path(config["n2_8f_root"]) / "configs/uncertainty_robust_sa_timexer_frozen_config.yaml",
        Path(config["n2_7_root"]) / "configs/phase_n2_7_config.yaml",
    ]
    manifest = {
        "research_start_commit": config["research_start_commit"],
        "files": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in frozen_sources
        ],
        "test_performance_accessed": False,
    }
    write("outputs/frozen_config/FROZEN_PROTOCOL_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "reports_and_figures_complete", "figure_png_count": len(list(FIG.glob("*.png"))), "test_performance_accessed": False}, indent=2))


if __name__ == "__main__":
    main()

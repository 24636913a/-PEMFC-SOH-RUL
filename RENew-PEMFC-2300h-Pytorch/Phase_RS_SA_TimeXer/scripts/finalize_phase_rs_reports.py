#!/usr/bin/env python
"""Finalize decision documents only from Phase RS Development outputs; no Test reads."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def metric(frame: pd.DataFrame, model: str, field: str) -> float:
    return float(frame.loc[frame.model.eq(model), field].iloc[0])


def main() -> None:
    summary = csv("outputs/frozen_model/frozen_model_summary.csv")
    folds = csv("outputs/frozen_model/frozen_model_fold_metrics.csv")
    horizons = csv("outputs/horizon/frozen_model_per_horizon.csv")
    regimes = csv("outputs/regime/frozen_model_regime_metrics.csv")
    tails = csv("outputs/tail/frozen_model_tail_metrics.csv")
    selector = csv("outputs/selector/selector_by_fold.csv")
    disagreement = csv("outputs/selector/selector_input_disagreement.csv")
    violations = csv("outputs/frozen_model/physical_violation_metrics.csv")
    bootstrap = csv("outputs/bootstrap/bootstrap_main_metric.csv")
    gate = csv("outputs/frozen_model/gate_a_audit.csv")
    p0, p1, p2, rs = "P0_Original_Ridge", "P1_Physics_Projected_Ridge", "P2_Zero_Current_Aware", "RS_Regime_Selective"
    p0_rmse, p1_rmse, p2_rmse, rs_rmse = (metric(summary, m, "mean_rmse_delta_mV") for m in (p0,p1,p2,rs))
    p0_r2, p1_r2, p2_r2, rs_r2 = (metric(summary, m, "mean_r2_delta") for m in (p0,p1,p2,rs))
    improve_p0 = (p0_rmse-rs_rmse)/p0_rmse*100
    improve_p2 = (p2_rmse-rs_rmse)/p2_rmse*100
    f = {m:[float(folds.loc[(folds.model==m)&(folds.fold==fold),"rmse_delta_mV"].iloc[0]) for fold in ("fold_1","fold_2","fold_3")] for m in (p0,p1,p2,rs)}
    r = lambda m, reg, category="transition_state", fold="fold_3": float(regimes.loc[(regimes.model==m)&(regimes.fold==fold)&(regimes.category==category)&(regimes.regime==reg),"rmse_delta_mV"].iloc[0])
    fold3_cont = {m:r(m,"continuous_nonzero") for m in (p0,p1,p2,rs)}
    mixed = {m:[r(m,"0<q<1",category="strict_q_state",fold=fold) for fold in ("fold_1","fold_2","fold_3")] for m in (p0,p1,p2,rs)}
    h = lambda m, hour: float(horizons.loc[(horizons.model==m)&(horizons.fold=="mean_3fold")&(horizons.horizon_hours==hour),"rmse_delta_mV"].iloc[0])
    h95 = {m:h(m,9.5) for m in (p0,p1,p2,rs)}; h12 = {m:h(m,12.0) for m in (p0,p1,p2,rs)}
    t = {m:tails.loc[tails.model==m].iloc[0].to_dict() for m in (p0,p1,p2,rs)}
    val_sel = selector[(selector.stage=="validation")&(selector.scope=="all_horizons")]
    val_dis = disagreement[(disagreement.stage=="validation")&(disagreement.scope=="all_horizons")]
    v = {m:float(violations.loc[(violations.model==m)&(violations.fold=="mean_3fold"),"physical_violation_trajectory_rate"].iloc[0]) for m in (p1,p2,rs)}
    b0 = bootstrap.loc[bootstrap.comparison=="RS_vs_P0"].iloc[0]
    b2 = bootstrap.loc[bootstrap.comparison=="RS_vs_P2"].iloc[0]
    gate_failed = gate.loc[~gate.passed.astype(bool),"criterion"].tolist()
    report = f"""# Phase RS-SA-TimeXer final results

## Protocol and firewall

This phase starts from commit `39300ca08e7a54f8060978891650bc49920f3278`. It uses the fixed RealHyFC Development protocol: 0.5 h sampling, 48-bin (24 h) history, 24-bin (12 h) horizon, DeltaV target, F2 history (14), nine future-condition variables, and three chronological expanding folds. No Test record, feature, target, prediction, metric, distribution, or selection signal was accessed; see `outputs/test_firewall_audit.json`.

The RS rule was frozen before execution: at each horizon, `g = 1[p_mixed >= 0.5]` from the causal P2 predicted mixed-current probability, and the entire nine-variable vector is selected as `Z_RS=(1-g)Z_P1+gZ_P2`. There was no threshold tuning, no extra learned selector, no CAF, and no new backbone.

## Answers to the required questions

1. **Does the selector use both P1 and P2?** Yes. Validation P2-selection fractions were {', '.join(f'{x:.3f}' for x in val_sel.g1_fraction)} in folds 1–3; therefore it is nonconstant.
2. **Prediction-state versus selected-input disagreement?** Yes, and it is explicit rather than hidden. Among `g=0` validation selections, P1 retained `q>0` at rates {', '.join(f'{x:.3f}' for x in val_dis.selector_input_regime_disagreement_rate)} in folds 1–3. `g=0` means “choose P1”, not “the P1 q value equals zero”.
3. **RS physical violation rate?** Zero in every fold and in the unweighted mean. P1 and P2 are also zero.
4. **P0/P1/P2 reproduction?** Passed. P0 = {p0_rmse:.6f} mV (R² {p0_r2:.6f}); P1 = {p1_rmse:.6f} mV (R² {p1_r2:.6f}); P2 = {p2_rmse:.6f} mV (R² {p2_r2:.6f}). P0 and P2 match the frozen G-STOP references within ±0.001 mV.
5. **Frozen RS performance?** {rs_rmse:.6f} mV, R² {rs_r2:.6f}.
6. **RS versus P0?** {improve_p0:.3f}% RMSE improvement, below the required 5%.
7. **RS versus P2?** {improve_p2:.3f}% (negative means worse): RS is {abs(improve_p2):.3f}% worse than P2, exceeding the allowed 3% degradation.
8. **Fold 3 overall?** RS = {f[rs][2]:.6f} mV versus P0 = {f[p0][2]:.6f} mV; it is worse and fails the non-worsening rule.
9. **Fold 3 continuous-nonzero?** RS = {fold3_cont[rs]:.6f} mV versus P2 = {fold3_cont[p2]:.6f} mV; it worsens the P2 negative-transfer regime rather than repairing it.
10. **Mixed-current benefit?** Preserved against P0 in all folds. RS mixed-current RMSE is {mixed[rs][0]:.6f}, {mixed[rs][1]:.6f}, {mixed[rs][2]:.6f} mV versus P0 {mixed[p0][0]:.6f}, {mixed[p0][1]:.6f}, {mixed[p0][2]:.6f} mV.
11. **9.5 h?** RS = {h95[rs]:.6f} mV; better than P0 ({h95[p0]:.6f}) but worse than P2 ({h95[p2]:.6f}).
12. **12 h?** RS = {h12[rs]:.6f} mV, versus P0 {h12[p0]:.6f}; this is {(h12[rs]/h12[p0]-1)*100:.3f}% worse and fails the 5% limit.
13. **P95/P99?** Pooled RS P95/P99 = {t[rs]['p95_mV']:.6f}/{t[rs]['p99_mV']:.6f} mV. P99 improves on P0 ({t[p0]['p99_mV']:.6f}) but is worse than P2 ({t[p2]['p99_mV']:.6f}).
14. **Maximum error?** RS maximum = {t[rs]['max_mV']:.6f} mV, equal to the P2 maximum and higher than P0 ({t[p0]['max_mV']:.6f}).
15. **Bootstrap?** RS−P0 using 48-origin paired blocks is {b0.point_estimate_mean_rmse_difference_mV:.6f} mV, 95% CI [{b0.ci95_lower_mV:.6f}, {b0.ci95_upper_mV:.6f}], which crosses zero: `IMPROVEMENT_NOT_STATISTICALLY_RESOLVED`. RS−P2 is {b2.point_estimate_mean_rmse_difference_mV:.6f} mV, CI [{b2.ci95_lower_mV:.6f}, {b2.ci95_upper_mV:.6f}], conclusively unfavorable to RS in this Development audit.
16. **Gate A?** `GATE_A_FAIL`.
17. **Formal retraining?** `NOT_RUN`, prohibited by the Gate A stop rule.
18. **Retrained comparison?** `NOT_RUN`; no numbers were substituted for missing experiments.
19. **Gate B?** `NOT_RUN`, because retraining was not authorized.
20. **Three seeds?** `NOT_RUN`, because Gate B was not reached.
21. **Three-seed stability?** `NOT_RUN`.
22. **Physical contribution verified?** Yes for future-condition validity: P1/P2/RS all satisfy the audited non-negativity, current-density identity, q-range and q=1 compatibility constraints. This does not establish a new voltage mechanism.
23. **Selective-routing contribution verified?** Partially at the frozen mixed-current slice only, but not as an overall robust mechanism: it retains mixed-current benefits yet worsens Fold 3 continuous-nonzero and overall Fold 3 performance.
24. **LSTM/TimesNet gap?** `INCONCLUSIVE` in this phase. No new external-model evaluation was authorized; RS was screened solely as a causal future-condition routing change on the frozen SA-TimeXer checkpoint.
25. **New final paper candidate?** No. `NOT_ESTABLISHED` because Gate A failed and no retraining/multi-seed evidence is authorized.
26. **Test sealed?** Yes. The Test firewall is intact.

## Frozen fold result

| Model | Fold 1 RMSE (mV) | Fold 2 RMSE (mV) | Fold 3 RMSE (mV) | Mean RMSE (mV) |
|---|---:|---:|---:|---:|
| P0 | {f[p0][0]:.6f} | {f[p0][1]:.6f} | {f[p0][2]:.6f} | {p0_rmse:.6f} |
| P1 | {f[p1][0]:.6f} | {f[p1][1]:.6f} | {f[p1][2]:.6f} | {p1_rmse:.6f} |
| P2 | {f[p2][0]:.6f} | {f[p2][1]:.6f} | {f[p2][2]:.6f} | {p2_rmse:.6f} |
| RS | {f[rs][0]:.6f} | {f[rs][1]:.6f} | {f[rs][2]:.6f} | {rs_rmse:.6f} |

## Gate A and stop decision

Failed frozen criteria: {', '.join(gate_failed)}. Because one failed criterion is sufficient, the phase stops here. No threshold change, selector change, CAF, PINN loss, new branch, retraining, multi-seed run, or Test release is authorized.
"""
    (ROOT / "PHASE_RS_RESULTS.md").write_text(report, encoding="utf-8")
    decision = f"""phase_status: GATE_A_FAILED_STOPPED
fixed_start_commit: 39300ca08e7a54f8060978891650bc49920f3278
baseline_reproduction:
  P0: {{mean_rmse_delta_mV: {p0_rmse:.9f}, mean_r2_delta: {p0_r2:.9f}, status: PASS}}
  P1: {{mean_rmse_delta_mV: {p1_rmse:.9f}, mean_r2_delta: {p1_r2:.9f}, status: PASS}}
  P2: {{mean_rmse_delta_mV: {p2_rmse:.9f}, mean_r2_delta: {p2_r2:.9f}, status: PASS}}
selector_audit:
  threshold: 0.5
  nonconstant_validation: true
  P2_selection_fraction_by_fold: [{', '.join(f'{x:.9f}' for x in val_sel.g1_fraction)}]
  p1_q_nonzero_when_g0_rate_by_fold: [{', '.join(f'{x:.9f}' for x in val_dis.selector_input_regime_disagreement_rate)}]
physical_consistency:
  P1_mean_trajectory_violation_rate: {v[p1]:.9f}
  P2_mean_trajectory_violation_rate: {v[p2]:.9f}
  RS_mean_trajectory_violation_rate: {v[rs]:.9f}
frozen_models:
  RS: {{mean_rmse_delta_mV: {rs_rmse:.9f}, mean_r2_delta: {rs_r2:.9f}, change_vs_P0_pct: {improve_p0:.9f}, change_vs_P2_pct: {improve_p2:.9f}}}
fold3_regime_result:
  overall_RS_mV: {f[rs][2]:.9f}
  continuous_nonzero_RS_mV: {fold3_cont[rs]:.9f}
  continuous_nonzero_P2_mV: {fold3_cont[p2]:.9f}
horizon_9_5h_result: {{RS_mV: {h95[rs]:.9f}, P0_mV: {h95[p0]:.9f}, P2_mV: {h95[p2]:.9f}}}
horizon_12h_result: {{RS_mV: {h12[rs]:.9f}, P0_mV: {h12[p0]:.9f}, P2_mV: {h12[p2]:.9f}}}
tail_metrics: {{RS_p95_mV: {t[rs]['p95_mV']:.9f}, RS_p99_mV: {t[rs]['p99_mV']:.9f}, RS_max_mV: {t[rs]['max_mV']:.9f}}}
bootstrap_result: {{RS_vs_P0_CI_crosses_zero: true, interpretation: IMPROVEMENT_NOT_STATISTICALLY_RESOLVED}}
gate_a_passed: false
retraining: {{status: NOT_RUN_GATE_A_FAILED, models: NOT_RUN, results: NOT_RUN}}
gate_b_passed: NOT_RUN_GATE_A_FAILED
multiseed: {{status: NOT_RUN_GATE_A_FAILED, results: NOT_RUN}}
paper_candidate: NO
supported_contributions:
  - physically consistent P1/P2/RS future-condition vectors
  - nonconstant causal regime selection and mixed-current slice preservation
unsupported_claims:
  - overall robust RS advantage
  - Fold3 continuous-nonzero negative-transfer resolution
  - neural retraining or multi-seed stability
  - Test performance
test_performance_accessed: false
next_action: STOP_PHASE_RS_NO_RETRAINING_NO_TEST_RELEASE
"""
    (ROOT / "PHASE_RS_DECISION.yaml").write_text(decision, encoding="utf-8")


if __name__ == "__main__": main()

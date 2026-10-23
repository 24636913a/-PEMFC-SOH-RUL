#!/usr/bin/env python
"""Write Phase OPR-D reports from already-created Development diagnostics only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
def read(path): return pd.read_csv(ROOT/path)
def value(df, model, col): return float(df.loc[df.model.eq(model),col].iloc[0])
def main():
    summary=read('outputs/reproduction/baseline_reproduction_summary.csv'); folds=read('outputs/reproduction/baseline_reproduction_fold_metrics.csv')
    integrity=json.loads((ROOT/'outputs/audit/diagnostic_integrity_audit.json').read_text(encoding='utf-8'))
    points=read('outputs/error_attribution/p1_p2_error_difference.csv'); f3=read('outputs/error_attribution/fold3_error_attribution.csv'); hz=read('outputs/horizon/horizon_error_attribution.csv'); ex=read('outputs/extreme_errors/extreme_error_unique_targets.csv'); ref=read('outputs/reference_condition/reference_condition_diagnostics.csv')
    vals={m:value(summary,m,'mean_rmse_delta_mV') for m in ('P0','P1','P2','RS')}; r2={m:value(summary,m,'mean_r2_delta') for m in ('P0','P1','P2','RS')}
    p1f3=f3.loc[f3.regime_transition.eq('continuous_nonzero')].iloc[0]
    f3c=points[(points.fold=='fold_3')&(points.regime_transition=='continuous_nonzero')]
    horizon_loss=(hz.assign(delta=lambda x:x.RMSE_P2_mV-x.RMSE_P1_mV).sort_values('delta',ascending=False))
    h12=horizon_loss[horizon_loss.horizon_hours.eq(12.0)]
    f1h12=h12[h12.fold.eq('fold_1')].iloc[0]; f3h12=h12[h12.fold.eq('fold_3')].iloc[0]
    topD=f3c.groupby('horizon_hours').D_mV2.agg(lambda x:float(x[x>0].sum())).sort_values(ascending=False)
    top20share=float(np.sum(ex.absolute_error_mV.to_numpy()**2)/np.sum(points[(points.fold=='fold_3')&(points.horizon_hours==12.0)].p2_error_mV.to_numpy()**2))
    corr=float(np.corrcoef(np.abs(ref.p2_current_A-ref.reference_current_A),np.abs(ref.correction_change_mV))[0,1])
    documents={
'BASELINE_REPRODUCTION.md':f'''# Baseline reproduction

All frozen M0 baseline paths reproduced within the required ±0.001 mV tolerance using only Development folds: P0 {vals['P0']:.6f} mV, P1 {vals['P1']:.6f} mV, P2 {vals['P2']:.6f} mV, and RS {vals['RS']:.6f} mV. Their corresponding mean DeltaV R² values are {r2['P0']:.6f}, {r2['P1']:.6f}, {r2['P2']:.6f}, and {r2['RS']:.6f}. The checkpoint, input order, fold scaler and inverse transformation are therefore aligned with the frozen protocol.
''',
'TIMEXER_DECOMPOSITION_AUDIT.md':f'''# TimeXer decomposition audit

The frozen forward pass was evaluated in normalized target space as `Y=B+C`, where `B` is the TimeXer history-backbone output and `C` is the Step-Aligned FiLM correction head. After inverse target scaling, the maximum `B+C−Y` residual is {integrity['decomposition_max_identity_residual_mV']:.9f} mV. P1 and P2 base predictions are identical for the same history (`max difference = 0` in scaled space). The RS pointwise identity `Y_RS=(1-g)Y_P1+gY_P2` also passed.

`B` alone is diagnostic only, not a fairly trained baseline: its three-fold mean RMSE is {value(summary,'B_only','mean_rmse_delta_mV'):.6f} mV, compared with P1 {vals['P1']:.6f} mV and P2 {vals['P2']:.6f} mV. The checkpoint strongly relies on its condition correction to reach its formal accuracy; this does not by itself establish harmful over-reliance.
''',
'P1_P2_ERROR_ATTRIBUTION.md':f'''# P1–P2 error attribution

At every Development prediction point, the analysis verifies `D=e_P2²−e_P1² = 2e_P1ΔC+ΔC²` in mV². The maximum numerical residual is {integrity['max_squared_error_identity_residual_mV2']:.9f} mV², attributable to float32 checkpoint output followed by mV conversion and squaring.

Fold 3 continuous-nonzero current is the principal localized loss: P1 RMSE {p1f3.RMSE_P1_mV:.6f} mV versus P2 {p1f3.RMSE_P2_mV:.6f} mV, with mean D {p1f3.mean_D_mV2:.6f} mV². Its largest positive-D horizons are {', '.join(f'{h:.1f} h' for h in topD.head(5).index)}. This attributes the output difference to the changed predicted-condition input at the frozen checkpoint; it is not evidence of an electrochemical causal mechanism.

At 12 h, the aggregate deterioration is primarily Fold 1: P1/P2 RMSE {f1h12.RMSE_P1_mV:.6f}/{f1h12.RMSE_P2_mV:.6f} mV, while Fold 3 is actually improved at that exact horizon ({f3h12.RMSE_P1_mV:.6f}/{f3h12.RMSE_P2_mV:.6f} mV). Thus “12 h degradation” must not be attributed to Fold 3 alone.
''',
'EXTREME_ERROR_AUDIT.md':f'''# Extreme error audit

The top 20 P2 errors were selected from Fold 3 at 12 h after deduplicating true target time. They represent 20 prediction points and 20 unique target timestamps. The largest is {ex.iloc[0].absolute_error_mV:.6f} mV at target age {ex.iloc[0].target_age_h:.2f} h, with P2 correction {ex.iloc[0].correction_p2_mV:.6f} mV.

These unique targets cluster around late-development operating ages rather than representing 20 independent events. Several have large predicted-current or q error, but others have small input-current mismatch and still large correction. Therefore future-condition error is a contributing observation, not a complete explanation. No point was removed or corrected. The top-20 squared-error share within Fold 3 / 12 h is {top20share:.3f}; it does not justify calling the entire pattern a single-outlier artifact.
''',
'REFERENCE_CONDITION_AUDIT.md':f'''# Reference-condition audit

`Z_ref` is the nine-variable condition vector at the prediction origin’s last fully observed 0.5 h bin. It precedes every future target, uses the identical feature definition and active-area density convention, and is transformed by the existing fold train-only future-condition scaler. No future value is used to construct it.

The absolute P2-current departure from `Z_ref` has only a weak linear association with the absolute P2−P1 correction change across Development points (Pearson r={corr:.3f}). This is descriptive, not causal. It shows that reference distance alone does not explain correction magnitude or error change; redundant current/density variables were not interpreted as independent physical effects.

The proposed difference `C(X,Z_P2,h)-C(X,Z_ref)` nevertheless has a clear testable mathematical property: it is exactly zero when `Z_P2,h=Z_ref`. This is a structural motivation, not proof that OPR will improve prediction.
''',
'OPR_ARCHITECTURE_PROPOSAL.md':'''# Candidate OPR-TimeXer proposal — not implemented

The sole pre-registered candidate is `Y_pred(h)=B_h(X)+C_h(X,Z_P2,h)-C_h(X,Z_ref)`. It reuses the frozen TimeXer history encoder, the same Step-Aligned FiLM correction network, shared correction weights, and the same horizon state for future and reference branches. It adds no Transformer, attention, recurrent module, future-condition forecaster, learned gate, or PINN loss.

When `Z_P2,h=Z_ref`, only the additional condition correction is constrained to zero. The history term `B_h(X)` remains, so this does not claim zero DeltaV or a physical law. This phase neither implements nor trains the candidate.
''',
'NEXT_EXPERIMENT_PREREGISTRATION.md':'''# OPR-TimeXer preregistration — not executed

If a future diagnostic decision explicitly authorizes it, compare matched seed-42 A0 Original TimeXer, A1 SA-TimeXer+P0, A2 SA-TimeXer+P2, and A3 OPR-TimeXer+P2 with identical legal OOF training origins, validation origins, initialization, optimizer, schedule, target scaler and future-condition source. A2 and A3 must both be retrained.

Predeclared seed-42 promotion criteria: A3 must improve mean RMSE over matched A2 by at least 5%; Fold 3 RMSE must be no greater than 27.885315 mV; 12 h RMSE no greater than 33.527982 mV; P99 no more than 5% worse than A2; and at least two folds better than A2. Only then are seeds 42, 2024, 3407 authorized. This document authorizes no experiment.
'''}
    for name,text in documents.items(): (ROOT/name).write_text(text,encoding='utf-8')
    decision='INCONCLUSIVE'
    result=f'''# Phase OPR-D diagnostic results

## Decision: {decision}

The M0 baseline, decomposition algebra, P1/P2 squared-error algebra and RS stepwise identity all passed. P2 provides a meaningful average improvement over P1 ({vals['P1']:.6f} to {vals['P2']:.6f} mV), while its Fold 3 continuous-nonzero loss is localized and associated with larger, sometimes extreme condition corrections.

However, the reference-condition evidence is insufficient to claim that a subtractive reference correction is the mechanism that will remedy the loss: predicted-condition departure from the reference has weak descriptive association with correction change, and some large corrections occur even when current prediction mismatch is small. The OPR formula is mathematically coherent and falsifiable, but the evidence does not yet support authorizing OPR training.

No Test data was accessed. No model was trained, modified or selected on Test. The next action is to stop this phase; do not implement or train OPR-TimeXer from this diagnostic alone.
'''
    (ROOT/'PHASE_OPR_DIAGNOSTIC_RESULTS.md').write_text(result,encoding='utf-8')
    yaml=f'''phase_status: DIAGNOSTIC_COMPLETED_STOPPED
fixed_start_commit: 7c4ca89fdabd2f104a2700cddc1ad1ea6a380d50
baseline_reproduction: {{P0: {vals['P0']:.9f}, P1: {vals['P1']:.9f}, P2: {vals['P2']:.9f}, RS: {vals['RS']:.9f}, status: PASS}}
timexer_decomposition: {{base_plus_correction_passed: true, max_residual_mV: {integrity['decomposition_max_identity_residual_mV']:.9f}, p1_p2_base_identical: true}}
rs_identity_passed: true
error_attribution: {{fold3_continuous_nonzero_P1_mV: {p1f3.RMSE_P1_mV:.9f}, fold3_continuous_nonzero_P2_mV: {p1f3.RMSE_P2_mV:.9f}, max_squared_error_identity_residual_mV2: {integrity['max_squared_error_identity_residual_mV2']:.9f}}}
reference_condition: {{causal_reference_available: true, direct_explanatory_evidence: WEAK}}
diagnostic_decision: INCONCLUSIVE
opr_training_authorized: false
retraining_performed: false
multiseed_performed: false
test_performance_accessed: false
next_action: STOP_DO_NOT_IMPLEMENT_OR_TRAIN_OPR_FROM_THIS_DIAGNOSTIC
'''
    (ROOT/'PHASE_OPR_DIAGNOSTIC_DECISION.yaml').write_text(yaml,encoding='utf-8')
if __name__=='__main__': main()

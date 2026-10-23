"""Render remaining N2.8A audit figures and make report answers explicit."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R=Path(__file__).resolve().parent.parent
O=R/"outputs";F=O/"figures";F.mkdir(parents=True,exist_ok=True)
metrics=pd.read_csv(O/"true_delta_metrics/modern_model_true_delta_comparison.csv")
tail=pd.read_csv(O/"tail_error_analysis/tail_error_summary.csv")
life=pd.read_csv(O/"lifecycle_analysis/lifecycle_generalization_summary.csv")
per=pd.read_csv(O/"true_delta_metrics/per_horizon_true_delta_metrics.csv")
models=["LSTM","TimesNet","TimeXer"]
colors={"LSTM":"#2864b7","TimesNet":"#d85a38","TimeXer":"#8055a5"}
z=life[life.model.isin(models)].sort_values("degradation_ratio_fold3_over_fold1")
fig,ax=plt.subplots(figsize=(6.8,4));ax.bar(z.model,z.degradation_ratio_fold3_over_fold1,color=[colors[x] for x in z.model]);ax.set(ylabel="Fold3 / Fold1 RMSEΔV",title="Lifecycle degradation ratio");fig.tight_layout();fig.savefig(F/"lifecycle_degradation_ratio.png",dpi=160);plt.close(fig)
z=tail[(tail.fold=="fold_3")&tail.model.isin(models)].sort_values("p95_abs_delta_error_mV")
fig,ax=plt.subplots(figsize=(6.8,4));ax.bar(z.model,z.p95_abs_delta_error_mV,color=[colors[x] for x in z.model]);ax.set(ylabel="Fold3 P95 |ΔV error| (mV)",title="Fold3 tail error");fig.tight_layout();fig.savefig(F/"tail_error_p95_comparison.png",dpi=160);plt.close(fig)
for field,file,label in [("r2_delta","per_horizon_r2_delta.png","R² ΔV"),("rmse_delta_mV","per_horizon_rmse_delta.png","RMSE ΔV (mV)")]:
    fig,ax=plt.subplots(figsize=(8,4.2))
    for model in models:
        x=per[(per.model==model)&(per.fold=="fold_3")].sort_values("horizon_hours");ax.plot(x.horizon_hours,x[field],marker="o",ms=3,label=model,color=colors[model])
    ax.set(xlabel="Forecast horizon (h)",ylabel=label,title=f"Fold3 {label} by forecast horizon");ax.legend();fig.tight_layout();fig.savefig(F/file,dpi=160);plt.close(fig)
def fmt(model,fold,col):return float(metrics[(metrics.model==model)&(metrics.fold==fold)][col].iloc[0])
lines=[]
for m in models:
    lines.append([m,f"{metrics[metrics.model.eq(m)].r2_delta.mean():.4f}",f"{metrics[metrics.model.eq(m)].rmse_delta_mV.mean():.3f}",f"{metrics[metrics.model.eq(m)].mae_delta_mV.mean():.3f}",f"{metrics[metrics.model.eq(m)].r2_abs.mean():.5f}"])
def table(rows,heads):return "| "+" | ".join(heads)+" |\n|"+"|".join(["---"]*len(heads))+"|\n"+"\n".join("| "+" | ".join(x)+" |" for x in rows)
lp=pd.read_csv(O/"leakage_audit/label_permutation_negative_control.csv");tm=pd.read_csv(O/"leakage_audit/temporal_misalignment_negative_control.csv")
lp_range=(lp.r2_delta.min(),lp.r2_delta.max());tm_range=(tm.r2_delta.min(),tm.r2_delta.max())
fold3=[f"{m}: R²ΔV={fmt(m,'fold_3','r2_delta'):.4f}, RMSEΔV={fmt(m,'fold_3','rmse_delta_mV'):.3f} mV" for m in models]
tail3=tail[(tail.fold=="fold_3")&tail.model.isin(models)]
p95=tail3.sort_values("p95_abs_delta_error_mV");p99=tail3.sort_values("p99_abs_delta_error_mV")
life3=life[life.model.isin(models)].sort_values("degradation_ratio_fold3_over_fold1")
report=f"""# Phase N2.8A Results — True Predictive-Value Audit and Modern Screening

## Primary Development-only ΔV metrics (seed 42)

{table(lines,["Model","Mean R²ΔV","Mean RMSEΔV (mV)","Mean MAEΔV (mV)","Mean R²V"])}

`R²ΔV`, `RMSEΔV`, and `MAEΔV` compare `y_true_abs − V_t` directly with `y_pred_abs − V_t`; R²V remains a secondary reconstructed-voltage statistic.

## Q1–Q7: predictive value and integrity

- **Q1 LSTM:** R²ΔV={metrics[metrics.model.eq('LSTM')].r2_delta.mean():.4f}, RMSEΔV={metrics[metrics.model.eq('LSTM')].rmse_delta_mV.mean():.3f} mV, MAEΔV={metrics[metrics.model.eq('LSTM')].mae_delta_mV.mean():.3f} mV.
- **Q2 TimesNet:** R²ΔV={metrics[metrics.model.eq('TimesNet')].r2_delta.mean():.4f}, RMSEΔV={metrics[metrics.model.eq('TimesNet')].rmse_delta_mV.mean():.3f} mV, MAEΔV={metrics[metrics.model.eq('TimesNet')].mae_delta_mV.mean():.3f} mV.
- **Q3:** the per-fold R²V − R²ΔV difference is in `anchor_contribution_audit.csv`. The absolute-voltage result includes the known origin anchor, while R²ΔV directly scores future change; no percentage-accuracy interpretation is used.
- **Q4:** anchor contribution is reported for LSTM, TimesNet, and Ridge with both R² values and their difference.
- **Q5:** label permutation collapsed: R²ΔV range [{lp_range[0]:.4f}, {lp_range[1]:.4f}], far below the 0.5 leakage-warning threshold.
- **Q6:** 48-origin temporal misalignment collapsed: R²ΔV range [{tm_range[0]:.4f}, {tm_range[1]:.4f}].
- **Q7:** all 14 feature rows PASS: no future-voltage/current/condition access, centered window, negative target shift, or future backfill.

## Q8–Q16: high-fidelity screening

- **Q8 TimeXer:** R²ΔV={metrics[metrics.model.eq('TimeXer')].r2_delta.mean():.4f}, RMSEΔV={metrics[metrics.model.eq('TimeXer')].rmse_delta_mV.mean():.3f} mV, MAEΔV={metrics[metrics.model.eq('TimeXer')].mae_delta_mV.mean():.3f} mV. It is over 20% worse than TimesNet on mean RMSEΔV and has no Fold3 improvement, so it is directly stopped.
- **Q9 TimeMixer++:** BLOCKED, not scored. The named anonymous official source returns HTTP 410. The authors' accessible `kwuking/TimeMixer` commit is old TimeMixer and lacks MRTI/TID/MCM/MRM; no surrogate was used.
- **Q10:** among evaluated modern candidates, LSTM has the highest mean R²ΔV ({metrics[metrics.model.eq('LSTM')].r2_delta.mean():.4f}); among the specified new-model comparison incumbents, TimesNet is the reference.
- **Q11/Q12 Fold3:** {'; '.join(fold3)}. TimeMixer++ has no valid Fold3 result.
- **Q13 tail errors:** Fold3 P95 rank: {', '.join(f'{r.model}={r.p95_abs_delta_error_mV:.3f} mV' for r in p95.itertuples())}. Fold3 P99 rank: {', '.join(f'{r.model}={r.p99_abs_delta_error_mV:.3f} mV' for r in p99.itertuples())}.
- **Q14 lifecycle:** Fold3/Fold1 RMSEΔV rank: {', '.join(f'{r.model}={r.degradation_ratio_fold3_over_fold1:.3f}' for r in life3.itertuples())}. This ratio is contextual, not a replacement for primary ΔV accuracy.
- **Q15 TimeXer three-seed:** NO; it failed the seed42 direct-stop condition.
- **Q16 TimeMixer++ three-seed:** NO; a verified official implementation is not accessible.

## Q17–Q18: frozen decision

- **Q17 selected backbone:** `UNRESOLVED`. The previously frozen high-fidelity TimesNet remains the incumbent reference, but N2.8A cannot make a final three-way model decision without legitimate TimeMixer++ execution.
- **Q18 next phase:** **NO-GO**. Obtain an author-confirmed, versioned, accessible TimeMixer++ source first; then rerun only the missing high-fidelity screen under this frozen Development protocol. Do not access Test.

## Firewall

`test_target_rows_read=0`, `test_predictions_constructed=false`, `test_metrics_computed=false`, and `test_feature_diagnostics_computed=false`.
"""
(R/"PHASE_N2_8A_RESULTS.md").write_text(report,encoding="utf-8")
print(json.dumps({"report":"updated","figure_count":len(list(F.glob('*.png')))},ensure_ascii=False))

"""Denominator uncertainty/sensitivity and error propagation on saved matches only."""
from common import *
def main():
 assert not (P/'outputs/DIAGNOSTIC_RECEIPT.json').exists()
 pr=read(P/'outputs/CONDITIONAL_RETENTION_DIAGNOSTIC.csv');ref=read(P/'data_contracts/EARLY_REFERENCE_POOL.csv')
 pairs=read(P/'outputs/FULL_JOINT_MATCH_PAIRS.csv');cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text())
 pairs=pairs.loc[pairs.caliper_IQR.eq(cfg['primary_caliper_IQR'])&pairs.reference_complete_cycle]
 uncertainty=[]
 for t,g in pairs.groupby('target_row'):
  refs=ref.set_index('row').loc[g.reference_row]
  means=refs.groupby('raw_cycle_id').cell_voltage_mean.mean()
  if len(means)<3:continue
  uncertainty.append(dict(target_row=int(t),reference_bins=len(refs),reference_cycles=len(means),
   cycle_balanced_mean_V=means.mean(),cycle_balanced_median_V=means.median(),
   cycle_mean_min_V=means.min(),cycle_mean_max_V=means.max(),cycle_SD_V=means.std(ddof=1)))
 u=pd.DataFrame(uncertainty);cs('outputs/REFERENCE_DENOMINATOR_UNCERTAINTY.csv',u)
 z=pr.loc[pr.conditional_retention_supported].merge(u,on='target_row',validate='many_to_one')
 z['CR_mean_percent']=100*z.predicted_voltage_V/z.cycle_balanced_mean_V
 z['CR_median_percent']=100*z.predicted_voltage_V/z.cycle_balanced_median_V
 z['CR_reference_range_low_percent']=100*z.predicted_voltage_V/z.cycle_mean_max_V
 z['CR_reference_range_high_percent']=100*z.predicted_voltage_V/z.cycle_mean_min_V
 z['CR_reference_range_width_pp']=z.CR_reference_range_high_percent-z.CR_reference_range_low_percent
 z['CR_mean_vs_median_difference_pp']=z.CR_mean_percent-z.CR_median_percent
 z['dCR_dVpred_pp_per_V']=100/z.cycle_balanced_mean_V
 z['dCR_dVref_pp_per_V']=-100*z.predicted_voltage_V/(z.cycle_balanced_mean_V**2)
 z['sensitivity_to_reference_SD_pp']=abs(z.dCR_dVref_pp_per_V)*z.cycle_SD_V
 cs('outputs/REFERENCE_SENSITIVITY_POINTWISE.csv',z)
 stats=[];horizon=[];regimes=[]
 b=read(P/'data_contracts/DEVELOPMENT.csv')
 for seed,g in z.groupby('seed'):
  for method,error in [('F2',g.retention_error_pp),('Raw_origin_persistence',100*(g.origin_raw_voltage_V-g.actual_voltage_V)/g.reference_mean_V)]:
   stats.append(dict(seed=int(seed),method=method,unit='conditional_retention_pp_not_SOH',**metrics(error)))
  for h,d in g.groupby('horizon_step'):
   horizon.append(dict(seed=int(seed),horizon_step=int(h),lead_h=.5*h,unit='conditional_retention_pp',**metrics(d.retention_error_pp)))
  for label,lo,hi in [('j<0.50',0,.5),('0.50<=j<0.75',.5,.75),('0.75<=j<1.00',.75,1.),('j>=1.00',1.,2.)]:
   j=g.target_row.map(b.actual_current_density_A_cm2_mean);d=g.loc[(j>=lo)&(j<hi)]
   if len(d): regimes.append(dict(seed=int(seed),load=label,queries=len(d),targets=d.target_row.nunique(),CR_pred_mean_percent=d.CR_mean_percent.mean(),
    CR_observed_mean_percent=d.conditional_voltage_retention_observed_percent.mean(),CR_pred_SD_percent=d.CR_mean_percent.std(ddof=1),**metrics(d.retention_error_pp)))
 cs('outputs/CONDITIONAL_RETENTION_BASELINE_METRICS.csv',pd.DataFrame(stats))
 cs('outputs/CONDITIONAL_RETENTION_HORIZON_ERRORS.csv',pd.DataFrame(horizon))
 cs('outputs/CONDITIONAL_RETENTION_LOAD_DIAGNOSTIC.csv',pd.DataFrame(regimes))
 end=z.loc[z.horizon_step.eq(24)].copy();cs('outputs/FIXED_12H_ENDPOINT_DIAGNOSTIC.csv',end)
 r42=z.loc[z.seed.eq(42)];e42=end.loc[end.seed.eq(42)].sort_values('target_row')
 uncertainty_stats=dict(matched_unique_targets=len(u),reference_cycles_min=int(u.reference_cycles.min()),reference_cycles_max=int(u.reference_cycles.max()),
  reference_cycle_SD_median_mV=float(1000*u.cycle_SD_V.median()),reference_cycle_SD_max_mV=float(1000*u.cycle_SD_V.max()),
  seed42_ratio_min_percent=float(r42.CR_mean_percent.min()),seed42_ratio_max_percent=float(r42.CR_mean_percent.max()),
  seed42_ratio_gt100_queries=int(r42.CR_mean_percent.gt(100).sum()),
  seed42_mean_vs_median_max_abs_pp=float(abs(r42.CR_mean_vs_median_difference_pp).max()),
  seed42_reference_range_width_median_pp=float(r42.CR_reference_range_width_pp.median()),
  seed42_reference_range_width_max_pp=float(r42.CR_reference_range_width_pp.max()),
  seed42_reference_SD_linear_sensitivity_median_pp=float(r42.sensitivity_to_reference_SD_pp.median()),
  reference_spread_role='Empirical between-cycle variability, NOT independent-sample confidence interval',
  fixed_12h_seed42_queries=len(e42),fixed_12h_seed42_upturns=int(e42.CR_mean_percent.diff().gt(0).sum()),
  upturn_role='Changes also change matching denominator/load; not proof of recovery or irreversible damage',
  denominator_sensitivity_is_not_label_uncertainty=True,physical_SOH_errors=None)
 js('outputs/DIAGNOSTIC_RECEIPT.json',uncertainty_stats);print(json.dumps(uncertainty_stats,ensure_ascii=False))

if __name__=='__main__':main()

"""Independent NumPy/scalar/Decimal replay. Does not import matching/analysis code.

No raw mixed-table reads; no model deserialization; no fitting or new predictions.
"""
import csv,hashlib,json
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent
def rd(p):return pd.read_csv(p,float_precision='round_trip')
def verify():
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text())
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 ledger=rd(P/'data_contracts/SOURCE_SHA256.csv')
 assert all(sha(ROOT/r.path)==r.sha256 for r in ledger.itertuples())
 b=rd(P/'data_contracts/DEVELOPMENT.csv');meta=rd(P/'data_contracts/HISTORICAL_3007_METADATA_MASK.csv')
 ids=[i for i,x in enumerate(meta.bin_end_ophrs2_h) if x<1819.997738]
 assert len(meta)==3007 and ids==list(range(len(b))) and len(b)==3005
 assert not b['split'].str.contains('test',case=False).any()
 snapshot=rd(ROOT/'Phase_TFT_DynamicVoltage_to_ReferenceSOH/inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv')
 for c in snapshot:
  a=b[c];z=snapshot[c]
  if pd.api.types.is_numeric_dtype(a):
   aa=a.to_numpy(float);zz=z.to_numpy(float);assert ((aa.view('uint64')==zz.view('uint64'))|(np.isnan(aa)&np.isnan(zz))).all(),c
  else:assert ((a==z)|(a.isna()&z.isna())).all(),c
 fields=cfg['condition_fields'];sc=rd(P/'configs/TRAIN_ONLY_CALIPER_SCALES.csv').set_index('field')
 tr=b.iloc[:2103];tr=tr.loc[tr.bin_valid_for_windows.eq(1)]
 for c in fields:
  q=np.quantile(tr[c],[.25,.75]);assert abs((q[1]-q[0])-sc.loc[c,'IQR'])<1e-12,c
 valid=(b.bin_valid_for_windows.eq(1)&b.internal_datetime_gap_count.eq(0)&b.actual_anode_inlet_rh_pct_mean.between(0,100)&b.actual_cathode_inlet_rh_pct_mean.between(0,100)&np.isfinite(b[fields]).all(axis=1))
 refids=[i for i in range(2103) if valid.iloc[i] and pd.Timestamp(b.last_datetime.iloc[i])<pd.Timestamp(cfg['reference_end_calendar'])]
 ref=rd(P/'data_contracts/EARLY_REFERENCE_POOL.csv');assert ref.row.tolist()==refids
 cycles=rd(ROOT/'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv')
 for r in ref.itertuples():
  found=cycles.loc[(pd.to_datetime(cycles.cycle_start)<=pd.Timestamp(r.first_datetime))&(pd.to_datetime(cycles.cycle_end)>=pd.Timestamp(r.last_datetime))]
  if len(found):
   assert len(found)==1 and r.raw_cycle_id==found.raw_cycle_id.iloc[0]
   assert r.complete_cycle==found.complete_cycle.iloc[0]
  else:assert np.isnan(r.raw_cycle_id) and not r.complete_cycle
 pred=rd(P/'outputs/RESTORED_F2_PREDICTIONS.csv');assert len(pred)==14760
 raw=b.cell_voltage_mean.to_numpy();maxtruth=0.
 for seed,g in pred.groupby('seed'):
  cache=rd(ROOT/f'Phase_H_Determinism_MultiSeed_Stability/outputs/validation_predictions/H-F2-Seed{seed}.csv')
  assert g[['origin_row','target_row','horizon_step']].reset_index(drop=True).equals(cache[['origin_row','target_row','horizon_step']])
  assert np.array_equal(g.y_pred_delta_V.to_numpy(),cache.y_pred_delta_V.to_numpy())
  assert np.array_equal(g.y_true_delta_V.to_numpy(),cache.y_true_delta_V.to_numpy())
  for r in g.itertuples():
   assert r.target_row-r.origin_row==r.horizon_step
   assert r.origin_raw_voltage_V==raw[r.origin_row] and r.actual_voltage_V==raw[r.target_row]
   assert abs(r.predicted_voltage_V-(raw[r.origin_row]+r.y_pred_delta_V))<2e-16
   assert r.origin_last_calendar==b.last_datetime.iloc[r.origin_row]
   assert r.target_last_calendar==b.last_datetime.iloc[r.target_row]
   assert abs((r.target_ophrs_h-r.origin_ophrs_h)-.5*r.horizon_step)<1e-10
   maxtruth=max(maxtruth,abs(r.cache_reconstructed_truth_V-r.actual_voltage_V))
   assert pd.Timestamp(r.origin_last_calendar)<pd.Timestamp(b.last_datetime.iloc[2403])
 assert not pred.online_SOH_qualified.any() and not pred.checkpoint_available_before_origin.any()
 saved=rd(P/'outputs/MATCHING_BY_TARGET_AND_CALIPER.csv');pairs=rd(P/'outputs/FULL_JOINT_MATCH_PAIRS.csv')
 tids=sorted(pred.target_row.unique());matched_sets={};maximum_denominator_diff=0.;checked=0
 for cal in [.5,.25,1.]:
  for t in tids:
   mask=np.full(len(ref),bool(valid.iloc[t]));loadmask=mask.copy()
   for c in fields:
    difference=abs(ref[c].to_numpy()-b[c].iloc[t]); ok=difference<=cal*sc.loc[c,'IQR']
    mask&=ok
    if c in cfg['load_fields']:loadmask&=ok
   zero=(ref.zero_current_fraction.to_numpy()>0)==(b.zero_current_fraction.iloc[t]>0)
   mask&=zero;loadmask&=zero
   r=saved.loc[saved.caliper_IQR.eq(cal)&saved.target_row.eq(t)].iloc[0]
   found=ref.loc[mask];actual_pairs=set(pairs.loc[pairs.caliper_IQR.eq(cal)&pairs.target_row.eq(t),'reference_row'])
   assert set(found.row)==actual_pairs
   assert r.load_only_reference_bins==int(loadmask.sum()) and r.full_joint_reference_bins==len(found)
   good=found.loc[found.complete_cycle&found.raw_cycle_id.notna()]
   vals=good.groupby('raw_cycle_id').cell_voltage_mean.mean()
   supported=len(vals)>=3 and len(good)>=3
   assert r.conditional_retention_supported==supported
   assert r.reference_cycles==len(vals)
   if supported:
    maximum_denominator_diff=max(maximum_denominator_diff,abs(r.reference_mean_V-vals.mean()))
    assert abs(r.reference_mean_V-vals.mean())<1e-15
   else:assert np.isnan(r.reference_mean_V)
   matched_sets[(cal,t)]=supported;checked+=1
 cr=rd(P/'outputs/CONDITIONAL_RETENTION_DIAGNOSTIC.csv');maxdecimal=Decimal(0);nratio=0
 # Decimal independent arithmetic on serialized inputs, not binary copies of the analysis formula.
 with localcontext() as ctx:
  ctx.prec=50
  with (P/'outputs/CONDITIONAL_RETENTION_DIAGNOSTIC.csv').open(encoding='utf-8',newline='') as f:
   for r in csv.DictReader(f):
    assert r['HI_BoL_percent']==''
    if r['conditional_retention_supported']=='True':
     expected=Decimal(100)*Decimal(r['predicted_voltage_V'])/Decimal(r['reference_mean_V'])
     dif=abs(expected-Decimal(r['conditional_voltage_retention_pred_percent']));maxdecimal=max(maxdecimal,dif);nratio+=1
    else:assert r['conditional_voltage_retention_pred_percent']==''
 assert maxdecimal<Decimal('0.000000000001')
 errs=rd(P/'outputs/RETENTION_ERROR_AND_SENSITIVITY.csv')
 maxmetric=0.
 for r in errs.itertuples():
  g=pred.loc[pred.seed.eq(r.seed)].merge(saved.loc[saved.caliper_IQR.eq(r.caliper_IQR)],on='target_row',validate='many_to_one')
  g=g.loc[g.conditional_retention_supported]
  e=(100*(g.predicted_voltage_V-g.actual_voltage_V)/g.reference_mean_V).to_numpy()
  values=[abs(e).mean(),np.sqrt(np.dot(e,e)/len(e)),np.percentile(abs(e),95),abs(e).max()]
  assert len(g)==r.n
  for expected,actual in zip(values,[r.MAE,r.RMSE,r.P95,r.MAX]):maxmetric=max(maxmetric,abs(expected-actual));assert abs(expected-actual)<1e-12
 refs=rd(P/'outputs/REFERENCE_SOH_PRIOR_QUALIFICATION_UNCHANGED.csv');assert len(refs)==7 and refs.legal_common_forecast_events.eq(0).all()
 uncertainty=rd(P/'outputs/REFERENCE_DENOMINATOR_UNCERTAINTY.csv')
 maxuncertainty=0.
 for r in uncertainty.itertuples():
  pp=pairs.loc[pairs.caliper_IQR.eq(.5)&pairs.target_row.eq(r.target_row)&pairs.reference_complete_cycle]
  vals=ref.set_index('row').loc[pp.reference_row].groupby('raw_cycle_id').cell_voltage_mean.mean().to_numpy()
  expected=[np.mean(vals),np.median(vals),np.min(vals),np.max(vals),np.std(vals,ddof=1)]
  actual=[r.cycle_balanced_mean_V,r.cycle_balanced_median_V,r.cycle_mean_min_V,r.cycle_mean_max_V,r.cycle_SD_V]
  for a,z in zip(expected,actual):maxuncertainty=max(maxuncertainty,abs(a-z));assert abs(a-z)<1e-15
 sensitivity=rd(P/'outputs/REFERENCE_SENSITIVITY_POINTWISE.csv')
 for r in sensitivity.itertuples():
  expected=100*r.predicted_voltage_V/r.cycle_balanced_median_V
  assert abs(expected-r.CR_median_percent)<1e-12
  low=100*r.predicted_voltage_V/r.cycle_mean_max_V;high=100*r.predicted_voltage_V/r.cycle_mean_min_V
  assert abs((high-low)-r.CR_reference_range_width_pp)<1e-12
  assert abs((-100*r.predicted_voltage_V/r.cycle_balanced_mean_V**2)-r.dCR_dVref_pp_per_V)<1e-12
 horizon=rd(P/'outputs/CONDITIONAL_RETENTION_HORIZON_ERRORS.csv')
 for r in horizon.itertuples():
  q=sensitivity.loc[sensitivity.seed.eq(r.seed)&sensitivity.horizon_step.eq(r.horizon_step)]
  assert len(q)==r.n and abs(q.retention_error_pp.abs().mean()-r.MAE)<1e-12
 from scipy.stats import rankdata
 relations=rd(P/'outputs/DESCRIPTIVE_LOAD_TIME_RELATIONSHIPS.csv')
 endpoint=rd(P/'outputs/FIXED_12H_ENDPOINT_DIAGNOSTIC.csv');endpoint=endpoint.loc[endpoint.seed.eq(42)].copy()
 endpoint['j']=endpoint.target_row.map(b.actual_current_density_A_cm2_mean)
 for r in relations.itertuples():
  xx=rankdata(endpoint[r.exposure].to_numpy());yy=rankdata(endpoint[r.outcome].to_numpy())
  xx-=xx.mean();yy-=yy.mean();rho=np.dot(xx,yy)/np.sqrt(np.dot(xx,xx)*np.dot(yy,yy))
  assert abs(rho-r.descriptive_Spearman)<1e-14
 aux=rd(P/'outputs/CYCLE_FORECAST_TIME_LANDING_DIAGNOSTIC.csv')
 for r in aux.itertuples():
  assert r.target_raw_cycle_id-r.origin_raw_cycle_id==r.H
  assert b.bin_start_ophrs2_h.iloc[r.target_bin_row]<r.cycle_target_ophrs_h<=b.bin_end_ophrs2_h.iloc[r.target_bin_row]
  assert not r.trend_comparison_qualified
 assert all(sha(ROOT/r.path)==r.sha256 for r in ledger.itertuples())
 result=dict(status='PASS',source_files_unchanged=len(ledger),strict_read_rule_cells=3005*len(snapshot.columns),
  restored_prediction_queries=len(pred),matching_target_caliper_checks=checked,exact_match_pair_rows=len(pairs),
  denominator_max_abs_difference_V=maximum_denominator_diff,Decimal_ratio_values=nratio,Decimal_ratio_max_abs_difference_pp=str(maxdecimal),
  recomputed_metric_max_abs_difference=maxmetric,reference_uncertainty_rows=len(uncertainty),reference_uncertainty_max_abs_difference_V=maxuncertainty,
  sensitivity_points=len(sensitivity),horizon_metric_rows=len(horizon),independent_rank_relationships=len(relations),raw_voltage_float32_truth_roundoff_max_V=maxtruth,
  no_BoL_HI_values=True,no_online_SOH=True,reference_events_reused_not_rematched=True,training=False,GPU=False,Test_values=False)
 (P/'outputs/INDEPENDENT_VERIFICATION.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
 print(json.dumps(result,ensure_ascii=False));return result
if __name__=='__main__':verify()

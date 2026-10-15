"""One-shot fixed retrospective matching and frozen-cache arithmetic, no training."""
from common import *

def validity(d,fields):
 return (d.bin_valid_for_windows.eq(1)&d.internal_datetime_gap_count.eq(0)
  &d.actual_anode_inlet_rh_pct_mean.between(0,100)&d.actual_cathode_inlet_rh_pct_mean.between(0,100)
  &np.isfinite(d[fields].to_numpy()).all(axis=1))

def pair_mask(x,y,scales,caliper,zero_x,zero_y):
 delta=abs(x[:,None,:]-y[None,:,:]);limit=scales*caliper
 good=(delta<=limit).all(axis=2)&(zero_x[:,None]==zero_y[None,:])
 return good,delta

def run():
 assert not (P/'outputs/DECISION.json').exists(),'Completed: independent verify only'
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 ledger=read(P/'data_contracts/SOURCE_SHA256.csv')
 assert all(sha(ROOT/r.path)==r.sha256 for r in ledger.itertuples())
 b=read(P/'data_contracts/DEVELOPMENT.csv');b['row']=np.arange(len(b))
 fields=cfg['condition_fields'];load=cfg['load_fields'];scales=read(P/'configs/TRAIN_ONLY_CALIPER_SCALES.csv').set_index('field').loc[fields,'IQR'].to_numpy()
 cy=read(ROOT/'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv')
 b['raw_cycle_id']=np.nan;b['complete_cycle']=False
 for r in cy.itertuples():
  hit=(pd.to_datetime(b.first_datetime)>=pd.Timestamp(r.cycle_start))&(pd.to_datetime(b.last_datetime)<=pd.Timestamp(r.cycle_end))
  assert b.loc[hit,'raw_cycle_id'].isna().all()
  b.loc[hit,'raw_cycle_id']=r.raw_cycle_id;b.loc[hit,'complete_cycle']=bool(r.complete_cycle)
 eligible=validity(b,fields)
 ref=b.loc[(b.row<2103)&(pd.to_datetime(b.last_datetime)<pd.Timestamp(cfg['reference_end_calendar']))&eligible].copy()
 ref['reference_is_BoL']=False;cs('data_contracts/EARLY_REFERENCE_POOL.csv',ref)
 allpred=[];restoration=[];dynamic=[]
 for seed in cfg['frozen_F2_seeds']:
  file=H/f'outputs/validation_predictions/H-F2-Seed{seed}.csv';d=read(file)
  assert len(d)==4920 and not d.duplicated(['origin_row','target_row','horizon_step']).any()
  assert (d.target_row-d.origin_row==d.horizon_step).all()
  o=d.origin_row.to_numpy(int);t=d.target_row.to_numpy(int)
  assert max(t)<2404 and min(o)>=2103
  raw=b.cell_voltage_mean.to_numpy(); expected=raw[t].astype('float32')-raw[o].astype('float32')
  identity_difference=float(abs(expected-d.y_true_delta_V).max());assert identity_difference<1e-7
  for col,idx in [('origin',o),('target',t)]:
   d[col+'_ophrs_h']=b.bin_end_ophrs2_h.to_numpy()[idx]
   d[col+'_first_calendar']=b.first_datetime.to_numpy()[idx];d[col+'_last_calendar']=b.last_datetime.to_numpy()[idx]
   d[col+'_raw_cycle_id']=b.raw_cycle_id.to_numpy()[idx]
  assert np.allclose(d.target_ophrs_h-d.origin_ophrs_h,d.horizon_step*.5,rtol=0,atol=1e-10)
  d['origin_raw_voltage_V']=raw[o];d['actual_voltage_V']=raw[t]
  d['predicted_voltage_V']=raw[o]+d.y_pred_delta_V
  d['cache_reconstructed_truth_V']=raw[o]+d.y_true_delta_V
  d['voltage_error_mV']=1000*(d.predicted_voltage_V-d.actual_voltage_V)
  d['persistence_voltage_error_mV']=1000*(raw[o]-raw[t])
  d['checkpoint_available_before_origin']=pd.to_datetime(d.origin_last_calendar)>pd.Timestamp(b.iloc[2403].last_datetime)
  d['future_condition_plan_certified']=False;d['online_SOH_qualified']=False
  d['cache_source']=file.relative_to(ROOT).as_posix()
  restoration.append(dict(seed=seed,queries=len(d),origins=d.origin_row.nunique(),targets=d.target_row.nunique(),
   cache_truth_vs_float32_delta_max_V=identity_difference,raw_truth_reconstruction_max_V=float(abs(d.cache_reconstructed_truth_V-d.actual_voltage_V).max()),
   origin_min_h=d.origin_ophrs_h.min(),origin_max_h=d.origin_ophrs_h.max(),target_min_h=d.target_ophrs_h.min(),target_max_h=d.target_ophrs_h.max(),
   post_selection_origins=int(d.loc[d.checkpoint_available_before_origin,'origin_row'].nunique()),cached_delta_units='V_already_inverse_scaled'))
  for method,c in [('F2','voltage_error_mV'),('Raw_origin_persistence','persistence_voltage_error_mV')]:
   dynamic.append(dict(seed=seed,method=method,unit='dynamic_voltage_mV',stage='Development_Validation',**metrics(d[c])))
  allpred.append(d)
 pred=pd.concat(allpred,ignore_index=True);cs('outputs/RESTORED_F2_PREDICTIONS.csv',pred);cs('outputs/VOLTAGE_RESTORATION_AUDIT.csv',pd.DataFrame(restoration));cs('outputs/DYNAMIC_VOLTAGE_METRICS.csv',pd.DataFrame(dynamic))
 target=b.loc[sorted(pred.target_row.unique())].copy();target['matching_input_role']='actual_future_conditions_retrospective_audit_only'
 cs('data_contracts/TARGET_CONDITIONS.csv',target)
 assert pd.to_datetime(ref.last_datetime).max()<pd.to_datetime(pred.origin_first_calendar).min()
 x=target[fields].to_numpy();y=ref[fields].to_numpy();li=[fields.index(c) for c in load]
 summaries=[];pairs=[];marginals=[]
 for cal in [cfg['primary_caliper_IQR']]+cfg['sensitivity_calipers_IQR']:
  joint,delta=pair_mask(x,y,scales,cal,target.zero_current_fraction.to_numpy()>0,ref.zero_current_fraction.to_numpy()>0)
  loadgood,_=pair_mask(x[:,li],y[:,li],scales[li],cal,target.zero_current_fraction.to_numpy()>0,ref.zero_current_fraction.to_numpy()>0)
  joint&=eligible.loc[target.index].to_numpy()[:,None];loadgood&=eligible.loc[target.index].to_numpy()[:,None]
  for f,k in zip(fields,range(len(fields))):
   marginal=((delta[:,:,k]<=cal*scales[k])&loadgood)
   marginals.append(dict(caliper_IQR=cal,field=f,load_matched_targets=int(loadgood.any(axis=1).sum()),load_plus_field_targets=int(marginal.any(axis=1).sum()),absolute_caliper=float(cal*scales[k])))
  for i,(index,tr) in enumerate(target.iterrows()):
   rr=ref.loc[joint[i]]; validcycles=rr.loc[rr.complete_cycle&rr.raw_cycle_id.notna()]
   cyclemeans=validcycles.groupby('raw_cycle_id').cell_voltage_mean.mean()
   supported=len(validcycles)>=3 and len(cyclemeans)>=3
   v=float(cyclemeans.mean()) if supported else np.nan
   sd=float(cyclemeans.std(ddof=1)) if supported else np.nan
   zero=scales==0; normalized=np.zeros(delta[i].shape)
   np.divide(delta[i],scales,out=normalized,where=~zero)
   normalized[:,zero]=np.where(delta[i][:,zero]==0,0,np.inf)
   closest=int(np.argmin(normalized.max(axis=1))); block=int(np.argmax(normalized[closest]))
   summaries.append(dict(caliper_IQR=cal,target_row=int(index),target_ophrs_h=tr.bin_end_ophrs2_h,target_calendar=tr.last_datetime,
    target_valid=bool(eligible.loc[index]),load_only_reference_bins=int(loadgood[i].sum()),full_joint_reference_bins=len(rr),
    full_joint_complete_reference_bins=len(validcycles),reference_cycles=len(cyclemeans),conditional_retention_supported=supported,
    reference_mean_V=v,reference_cycle_SD_V=sd,reference_cycle_min_V=float(cyclemeans.min()) if supported else np.nan,
    reference_cycle_max_V=float(cyclemeans.max()) if supported else np.nan,
    reference_BoL_certified=False,HI_BoL_percent=np.nan,online_SOH_qualified=False,
    nearest_joint_max_scaled_difference=float(normalized.max(axis=1).min()),nearest_reference_row=int(ref.iloc[closest].row),nearest_blocking_field=fields[block],
    rejection='' if supported else ('Invalid target conditions' if not eligible.loc[index] else 'Fewer than3 complete physical reference cycles in joint calipers')))
   for j in np.flatnonzero(joint[i]):
    rrj=ref.iloc[j];record=dict(caliper_IQR=cal,target_row=int(index),reference_row=int(rrj.row),reference_raw_cycle_id=rrj.raw_cycle_id,reference_complete_cycle=bool(rrj.complete_cycle))
    record.update({'absdiff_'+f:float(delta[i,j,k]) for k,f in enumerate(fields)})
    pairs.append(record)
 match=pd.DataFrame(summaries);cs('outputs/MATCHING_BY_TARGET_AND_CALIPER.csv',match)
 cs('outputs/FULL_JOINT_MATCH_PAIRS.csv',pd.DataFrame(pairs,columns=['caliper_IQR','target_row','reference_row','reference_raw_cycle_id','reference_complete_cycle']+['absdiff_'+f for f in fields]))
 cs('outputs/MARGINAL_CONDITION_SUPPORT.csv',pd.DataFrame(marginals))
 coverage=[]
 edges=[0,.5,.75,1.,np.inf];labels=['j<0.50','0.50<=j<0.75','0.75<=j<1.00','j>=1.00']
 for cal,g in match.groupby('caliper_IQR',sort=False):
  for label,lo,hi in [('ALL',-np.inf,np.inf)]+list(zip(labels,edges[:-1],edges[1:])):
   j=target.set_index('row').actual_current_density_A_cm2_mean;z=g.loc[(g.target_row.map(j)>=lo)&(g.target_row.map(j)<hi)]
   coverage.append(dict(caliper_IQR=cal,load=label,targets=len(z),load_only_any=int(z.load_only_reference_bins.gt(0).sum()),
    joint_any=int(z.full_joint_reference_bins.gt(0).sum()),joint_3cycle_supported=int(z.conditional_retention_supported.sum()),
    BoL_HI_qualified=0,strict_online_SOH=0))
 cs('outputs/COVERAGE_BY_LOAD.csv',pd.DataFrame(coverage))
 primary=match.loc[match.caliper_IQR.eq(cfg['primary_caliper_IQR'])]
 for c in ['caliper_IQR']: primary=primary.drop(columns=c)
 pr=pred.merge(primary,on='target_row',validate='many_to_one',suffixes=('','_match'))
 pr['conditional_voltage_retention_pred_percent']=100*pr.predicted_voltage_V/pr.reference_mean_V
 pr['conditional_voltage_retention_observed_percent']=100*pr.actual_voltage_V/pr.reference_mean_V
 pr['retention_error_pp']=100*(pr.predicted_voltage_V-pr.actual_voltage_V)/pr.reference_mean_V
 pr['HI_BoL_percent']=np.nan;cs('outputs/CONDITIONAL_RETENTION_DIAGNOSTIC.csv',pr)
 sensitivity=[]
 for cal,g in match.groupby('caliper_IQR',sort=False):
  z=pred.merge(g,on='target_row',validate='many_to_one',suffixes=('','_match'));z=z.loc[z.conditional_retention_supported]
  for seed,d in z.groupby('seed'):
   e=100*(d.predicted_voltage_V-d.actual_voltage_V)/d.reference_mean_V
   sensitivity.append(dict(caliper_IQR=cal,seed=seed,unit='conditional_retention_percentage_points',**metrics(e),
    reference_denominator_min_V=d.reference_mean_V.min(),reference_denominator_max_V=d.reference_mean_V.max(),
    reference_SD_max_mV=1000*d.reference_cycle_SD_V.max()))
 cs('outputs/RETENTION_ERROR_AND_SENSITIVITY.csv',pd.DataFrame(sensitivity,columns=['caliper_IQR','seed','unit','n','MAE','RMSE','P95','MAX','reference_denominator_min_V','reference_denominator_max_V','reference_SD_max_mV']))
 # Reuse seven frozen qualifications, not a rerun of the already rejected event-matching experiment.
 refs=read(BRIDGE/'outputs/ALL_REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv');assert refs.nominal_checkpoint_h.tolist()==[0,300,600,850,1150,1500,1750]
 cs('outputs/REFERENCE_SOH_PRIOR_QUALIFICATION_UNCHANGED.csv',refs)
 pol=read(ROOT/'Phase_Polarization_Physics_to_Health_Qualification/outputs/POLARIZATION_QUALIFICATION.csv');cs('outputs/SEVEN_POLARIZATION_TIME_CONDITIONS.csv',pol)
 auxiliary(pr,b)
 n=int(primary.conditional_retention_supported.sum())
 decision=dict(conclusion='B_PARTIALLY_FEASIBLE' if n else 'C_INSUFFICIENT_EVIDENCE',
  development_rows=len(b),early_reference_bins=len(ref),early_reference_complete_cycles=int(ref.loc[ref.complete_cycle,'raw_cycle_id'].nunique()),
  early_reference_first_h=float(ref.bin_end_ophrs2_h.min()),early_reference_last_h=float(ref.bin_end_ophrs2_h.max()),
  prediction_seeds=3,queries_per_seed=4920,distinct_targets=len(target),origins_per_seed=205,
  target_first_h=float(target.bin_end_ophrs2_h.min()),target_last_h=float(target.bin_end_ophrs2_h.max()),
  load_only_any=int(primary.load_only_reference_bins.gt(0).sum()),joint_any=int(primary.full_joint_reference_bins.gt(0).sum()),
  conditional_retention_targets=n,conditional_retention_queries_per_seed=int(pred.loc[pred.seed.eq(42),'target_row'].isin(primary.loc[primary.conditional_retention_supported,'target_row']).sum()),
  authenticated_BoL_targets=0,physically_certified_same_condition_targets=0,online_SOH_queries=0,
  reference_SOH_comparison='REFERENCE_SOH_COMPARISON_NOT_QUALIFIED',
  reference_event_matching_repeated=False,physical_equivalence_not_proven=True,
  blockers=['No authenticated dynamic same-condition BoL reference','Empirical half-hour aggregates cannot certify full state or load sequence equivalence','No certified advance future condition plan','Frozen cache model selection labels extend beyond all cached origins','No existing qualified reference polarization bridge'],
  protocol_sha256=sha(P/'configs/FROZEN_PROTOCOL.json'),original_sources_unchanged=all(sha(ROOT/r.path)==r.sha256 for r in ledger.itertuples()),
  trained=False,GPU=False,Test_numeric_values=False)
 js('outputs/DECISION.json',decision);print(json.dumps(decision,ensure_ascii=False))

def auxiliary(pr,b):
 p=read(ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_PREDICTIONS.csv')
 p=p.loc[p.method.str.contains('RIDGE',case=False)].copy();assert len(p)==141
 measurements=read(ROOT/'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv')
 measurements=measurements.loc[measurements.stable_window.eq('middle_50')].set_index('cycle_interval_id')
 out=[];unique=pr.loc[pr.seed.eq(42)].drop_duplicates('target_row')
 for r in p.itertuples():
  m=measurements.loc[r.target_cycle_interval_id]; h=r.target_actual_end_h
  binhit=unique.loc[(unique.target_row.map(b.bin_start_ophrs2_h)<h)&(unique.target_row.map(b.bin_end_ophrs2_h)>=h)]
  for tr in binhit.itertuples():
   out.append(dict(H=r.H,fold=r.fold,origin_raw_cycle_id=r.origin_raw_cycle_id,target_raw_cycle_id=r.target_raw_cycle_id,
    cycle_origin_ophrs_h=r.origin_actual_end_h,cycle_target_ophrs_h=h,cycle_actual_lead_h=h-r.origin_actual_end_h,
    cycle_origin_available_calendar=r.origin_calendar,cycle_target_available_calendar=r.target_calendar,
    target_bin_row=tr.target_row,bin_first_calendar=tr.target_first_calendar,bin_last_calendar=tr.target_last_calendar,
    cycle_target_R=r.target_R,cycle_prediction_R=r.predicted_R,cycle_origin_R=r.origin_R,
    fixed_state_j_A_cm2=m.current_density,dynamic_bin_j_A_cm2=b.iloc[tr.target_row].actual_current_density_A_cm2_mean,
    dynamic_bin_current_std=b.iloc[tr.target_row].actual_current_density_A_cm2_std,
    same_complete_raw_cycle=bool(tr.target_raw_cycle_id==r.target_raw_cycle_id),
    conditional_retention_supported=tr.conditional_retention_supported,
    exact_observation_operator_equal=False,common_origin=False,trend_comparison_qualified=False,
    reason='Operating-hour bin landing only; stable-dwell measurement differs from half-hour dynamic mean; no common registered forecast origin'))
 cols=['H','fold','origin_raw_cycle_id','target_raw_cycle_id','cycle_origin_ophrs_h','cycle_target_ophrs_h','cycle_actual_lead_h','cycle_origin_available_calendar','cycle_target_available_calendar','target_bin_row','bin_first_calendar','bin_last_calendar','cycle_target_R','cycle_prediction_R','cycle_origin_R','fixed_state_j_A_cm2','dynamic_bin_j_A_cm2','dynamic_bin_current_std','same_complete_raw_cycle','conditional_retention_supported','exact_observation_operator_equal','common_origin','trend_comparison_qualified','reason']
 cs('outputs/CYCLE_FORECAST_TIME_LANDING_DIAGNOSTIC.csv',pd.DataFrame(out,columns=cols))
 cs('outputs/FROZEN_CYCLE_PERFORMANCE_PREDICTIONS.csv',p)
 cs('outputs/FROZEN_CYCLE_EQUAL_FOLD_METRICS.csv',read(ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_EQUAL_FOLD_METRICS.csv'))
 js('outputs/AUXILIARY_QUALIFICATION.json',dict(frozen_cycle_forecasts=141,potential_time_landing_rows=len(out),
  unique_physical_target_cycles=len({x['target_raw_cycle_id'] for x in out}),qualified_common_origin_and_operator_pairs=0,
  correlation_or_direction_scores=None,reason='No equal measurement operator and registered common forecast origin; no forced association'))

if __name__=='__main__':run()

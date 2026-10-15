"""Candidate target construction and chronological shape audit. Never fits a model."""
from common import *
import yaml

CONDITION_MAP={'current_density':'j_median','anode_inlet_temperature_C':'T_anode_C_median','anode_inlet_pressure_kPag':'P_anode_kPag_median',
 'anode_inlet_RH_pct':'RH_anode_pct_median','anode_hydrogen_flow_nl_min':'flow_H2_nl_min_median','anode_stoichiometry':'stoic_anode_median','cathode_stoichiometry':'stoic_cathode_median'}

def legal_origins(b,features,lo,hi):
 origins=[]
 for o in range(lo,hi-24):
  start=o-47;end=o+24
  if start<0 or end>=hi:continue
  z=b.iloc[start:end+1]
  if z.bin_valid_for_windows.eq(1).all() and z.temporal_segment_id.eq(b.iloc[o].temporal_segment_id).all() and np.allclose(np.diff(z.bin_end_ophrs2_h),.5,atol=1e-6,rtol=0) and np.isfinite(z[features]).all(axis=None):origins.append(o)
 return origins

def main():
 assert not (P/'outputs/FINAL_DECISION.json').exists(),'Audit already complete; verify saved outputs'
 cfg=json.loads((P/'configs/FROZEN_AUDIT.json').read_text());assert sha(P/'configs/FROZEN_AUDIT.json')==(P/'configs/FROZEN_AUDIT.sha256').read_text().strip()
 raw=rd(P/'outputs/RAW_FIXED_STATE_DWELL_MEASUREMENTS.csv');cycles=rd(RCR/'outputs/RAW_CYCLE_CATALOG.csv')
 meta,ids,b=authority_development();cs('data_contracts/N1_3007_METADATA_MASK.csv',meta.assign(source_row=np.arange(len(meta)),retained=np.isin(np.arange(len(meta)),ids)))
 cs('data_contracts/N1_DEVELOPMENT.csv',b)
 previous=rd(ROOT/'Phase_TFT_DynamicVoltage_to_ReferenceSOH/inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv')
 dif=[]
 for c in previous:
  if pd.api.types.is_numeric_dtype(b[c]):
   x=b[c].to_numpy(float);y=previous[c].to_numpy(float);eq=(x.view('uint64')==y.view('uint64'))|(np.isnan(x)&np.isnan(y));maximum=float(np.nanmax(abs(x-y)))
  else:eq=((b[c]==previous[c])|(b[c].isna()&previous[c].isna())).to_numpy();maximum=0.
  dif.append(dict(field=c,unequal_cells=int((~eq).sum()),max_abs_difference=maximum))
 assert all(r['unequal_cells']==0 for r in dif);cs('outputs/N1_IDENTITY_STRICT_COMPARISON.csv',pd.DataFrame(dif))
 raw['QC_rejection']=raw.exclusion.fillna('');cs('outputs/STATE_EXCLUSION_RECORDS.csv',raw.loc[~raw.eligible_electrical_stability,['run_id','state_token','raw_cycle_id','complete_cycle','QC_rejection']])
 inv=[]
 for token,g in raw.groupby('state_token'):
  c=g.loc[g.complete_cycle];q=c.loc[c.eligible_electrical_stability]
  inv.append(dict(state_token=token,set_current_A=float(g.median_set_current_A.median()),actual_j_A_cm2=float(g.j_median.median()),
   Development_runs=len(g),Development_raw_run_points=int(g.raw_row_count.sum()),nonoverlap_middle50_points=int(g.selected_sample_count.sum()),
   complete_cycle_dwells=len(c),complete_cycles=int(c.raw_cycle_id.nunique()),QC_dwells=len(q),QC_cycles=int(q.raw_cycle_id.nunique()),
   distinct_measurement_end_times=g.measurement_last_calendar.nunique(),independent_life_trajectories=1,
   first_h=g.measurement_first_h.min(),last_h=g.measurement_last_h.max(),median_run_duration_s=g.duration_wall_s.median(),
   median_sampling_interval_s=g.sample_interval_median_s.median(),median_j_std=g.j_std.median(),max_j_std=g.j_std.max(),
   voltage_cache_or_raw='New bounded Development raw scan'))
 cs('outputs/FIXED_STATE_INVENTORY.csv',pd.DataFrame(inv))
 # Every state retains its own label: never mix voltages from different currents.
 med=[];conds=list(CONDITION_MAP)
 for (token,cy),g in raw.loc[raw.complete_cycle].groupby(['state_token','raw_cycle_id']):
  q=g.loc[g.eligible_electrical_stability];c=cycles.set_index('raw_cycle_id').loc[cy]
  record=dict(state_token=token,raw_cycle_id=int(cy),age_stratum=c.age_stratum,dwells=len(g),QC_dwells=len(q),
   raw_window_points=int(g.selected_sample_count.sum()),V_cycle_median=float(g.V_median.median()),V_QC_cycle_median=float(q.V_median.median()) if len(q) else np.nan,
   first_measurement_h=g.measurement_first_h.min(),last_measurement_h=g.measurement_last_h.max(),
   last_measurement_calendar=g.measurement_last_calendar.max(),available_h=g.available_h.max(),available_calendar=g.available_calendar.max(),
   cycle_start_h=c.start_ophrs2_h,cycle_end_h=c.end_ophrs2_h,cycle_start_calendar=c.cycle_start,cycle_end_calendar=c.cycle_end,
   complete_cycle_identity_available_calendar=c.cycle_end,
   within_cycle_V_IQR_mV=1000*(g.V_median.quantile(.75)-g.V_median.quantile(.25)),
   within_cycle_V_span_mV=1000*(g.V_median.max()-g.V_median.min()))
  for label,col in CONDITION_MAP.items():record[label]=float(g[col].median())
  med.append(record)
 cm=pd.DataFrame(med);cs('outputs/STATE_CYCLE_OBSERVATIONS.csv',cm)
 fixed=cm.loc[cm.state_token.eq(cfg['primary_state'])].sort_values('raw_cycle_id').copy()
 fixed['retention_percent_frozen_reference']=100*fixed.V_cycle_median/cfg['frozen_reference_V']
 fixed['reference_is_authenticated_BoL']=False
 legacy=rd(PAA/'outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv');legacy=legacy.loc[legacy.stable_window.eq('middle_50')]
 comparison=legacy.merge(fixed,on='raw_cycle_id',validate='one_to_one',suffixes=('_frozen','_new'))
 comparison['V_difference_V']=comparison.V_cycle_median-comparison.P1S_voltage_V
 assert len(comparison)==141 and comparison.V_difference_V.abs().max()<1e-12
 cs('outputs/FROZEN_141_RECONSTRUCTION_COMPARISON.csv',comparison[['raw_cycle_id','P1S_voltage_V','V_cycle_median','V_difference_V','eligible_dwell_count','dwells']])
 reference=legacy.loc[legacy.is_reference_complete_interval]
 assert len(reference)==10 and abs(reference.P1S_voltage_V.median()-cfg['frozen_reference_V'])<1e-12
 ref_available=reference.measurement_end_ophrs2_h.max();fixed['frozen_reference_available_before_observation']=fixed.available_h>ref_available
 # Initial ten values remain reference-building measurements, not causal retention evaluations.
 cs('outputs/FIXED_PRIMARY_CYCLE_OBSERVATIONS.csv',fixed)
 drift=[];condition=[]
 for token,g in cm.groupby('state_token'):
  if g.raw_cycle_id.nunique()<3:continue
  e=g.loc[g.age_stratum.eq('EARLY')&~g.raw_cycle_id.isin(reference.raw_cycle_id)]
  l=g.loc[g.age_stratum.eq('LATE')];adj=g.sort_values('raw_cycle_id');delta=adj.V_cycle_median.diff();same=adj.raw_cycle_id.diff().eq(1)
  drift.append(dict(state_token=token,cycles=len(g),EARLY_cycles=len(e),LATE_cycles=len(l),
   late_minus_early_median_mV=1000*(l.V_cycle_median.median()-e.V_cycle_median.median()),
   median_adjacent_change_mV=1000*abs(delta.loc[same]).median(),upturns=int(delta.loc[same].gt(0).sum()),
   adjacent_pairs=int(same.sum()),within_cycle_IQR_median_mV=g.within_cycle_V_IQR_mV.median(),
   retrospective_only=True,irreversible_degradation_proven=False))
  for c in conds:
   condition.append(dict(state_token=token,field=c,min=g[c].min(),max=g[c].max(),
    early_median=e[c].median(),late_median=l[c].median(),early_late_difference=l[c].median()-e[c].median(),
    descriptive_time_Spearman=g[c].rank().corr(g.last_measurement_h.rank())))
 cs('outputs/LONG_TERM_AND_WITHIN_CYCLE_VARIATION.csv',pd.DataFrame(drift));cs('outputs/STATE_CONDITION_COMPARABILITY.csv',pd.DataFrame(condition))
 # Legacy condition matching is provenance evidence, no refitting/reoptimization.
 matching=rd(PAA/'outputs/P1S_CONDITION_MATCHING.csv');cs('outputs/FROZEN_PAA_CONDITION_MATCHING.csv',matching)
 rows=[]
 for (token,cy),g in raw.loc[raw.complete_cycle].groupby(['state_token','raw_cycle_id']):
  for pos,r in enumerate(g.sort_values('start_datetime').itertuples(),1):
   rows.append(dict(state_token=token,raw_cycle_id=int(cy),within_cycle_slot=pos,run_id=r.run_id,
    measurement_end_h=r.measurement_last_h,available_h=r.available_h,V=r.V_median,
    eligible_electrical_stability=r.eligible_electrical_stability))
 cs('outputs/NONOVERLAP_DWELL_TARGETS.csv',pd.DataFrame(rows))
 hcfg=yaml.safe_load((ROOT/'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
 features=hcfg['task']['raw_f2_feature_order'];native={role:legal_origins(b,features,lo,hi) for role,lo,hi in [('Train',0,2103),('Validation',2103,2404)]}
 assert len(native['Train'])==1380 and len(native['Validation'])==205
 labels=bin_labels(raw,b,cycles,cfg['primary_state']);cs('outputs/PRIMARY_FIXED_BIN_LABELS.csv',labels)
 windows,coverage=window_audit(native,labels,b,cycles);cs('outputs/T0_T1_WINDOW_QUALIFICATION.csv',windows);cs('outputs/T0_T1_SPLIT_COUNTS.csv',coverage)
 frozen_cycle_inputs(fixed,legacy,b,features)
 seq=cycle_sequence_audit(fixed,cycles);js('outputs/CYCLE_SEQUENCE_48_24_QUALIFICATION.json',seq)
 refsoh=rd(ROOT/'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv');assert len(refsoh)==7
 cs('outputs/T2_SEVEN_TRUE_REFERENCE_EVENTS.csv',refsoh)
 previous_refs=rd(ROOT/'Phase_TFT_DynamicVoltage_to_ReferenceSOH/outputs/ALL_REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv')
 cs('outputs/REFERENCE_TIMING_QUALIFICATION_UNCHANGED.csv',previous_refs)
 qualification=[]
 for key,label in [('T0','Dynamic half-hour DeltaV'),('T1_cycle','Fixed0.78923 cycle voltage'),('T1_bin','Fixed0.78923 half-hour sparse observation'),('T2_fixed','Same0.78923 state with frozen reference'),('T2_pol','Polarization j1.0')]:
  qualification.append(dict(candidate=key,target=label,
   independent_life_trajectories=1,real_primary_cycle_observations=(len(fixed) if key in ['T1_cycle','T2_fixed'] else np.nan),
   true_reference_events=7 if key=='T2_pol' else np.nan,
   authenticated_BoL=(key=='T2_pol'),new_performance_MAE=np.nan,new_training=False,
   original_TFT_score_transfer_allowed=(key=='T0'),
   original_dense48_to24_contract=(key=='T0'),
   status='FROZEN_DYNAMIC_VALIDATION_EVIDENCE' if key=='T0' else 'DATA_AVAILABLE_NEW_TASK_NOT_YET_TRAINING_QUALIFIED' if key!='T2_pol' else 'SEVEN_EVENTS_INSUFFICIENT_FOR_ORIGINAL_SEQUENCE'))
 cs('outputs/CANDIDATE_TARGET_EVIDENCE.csv',pd.DataFrame(qualification))
 decision=dict(recommendation='C_RETAIN_DYNAMIC_TFT_AND_RIDGE96_AUXILIARY',
  reason='Fixed-state observations are real and more interpretable, but denser dwell measurements are not additional life trajectories; native24-step dense target and causal physical comparability are not established for a replacement health task',
  original_frozen_cycle_observations=141,new_audit_reconstructed_cycles=len(fixed),audit_only_extra_cycles=sorted(set(fixed.raw_cycle_id)-set(legacy.raw_cycle_id)),
  complete_primary_dwells=int(((raw.state_token==cfg['primary_state'])&raw.complete_cycle).sum()),
  complete_primary_QC_dwells=int(((raw.state_token==cfg['primary_state'])&raw.complete_cycle&raw.eligible_electrical_stability).sum()),
  independent_life_trajectories=1,reference_events=7,original_sequence_48cycle_24cycle_legal=seq['consecutive_raw_cycle_48_to24_windows'],
  fixed_state_fully_physical_SOH_qualified=False,new_GPU_training_recommended_now=False,
  train_new_model=False,Test_numeric_values=False,protocol_sha256=sha(P/'configs/FROZEN_AUDIT.json'))
 js('outputs/FINAL_DECISION.json',decision);print(json.dumps(decision,ensure_ascii=False));print(coverage.to_string(index=False))

def bin_labels(raw,b,cycles,token):
 q=raw.loc[raw.state_token.eq(token)&raw.complete_cycle].copy();records=[]
 ends=b.bin_end_ophrs2_h.to_numpy();starts=b.bin_start_ophrs2_h.to_numpy()
 for r in q.itertuples():
  # Require all selected measurements plus conservative availability within the same bin.
  k=int(np.searchsorted(ends,r.available_h,side='left'))
  hit=k<len(b) and starts[k]<=r.measurement_first_h and r.measurement_last_h<=ends[k]
  cyc=cycles.set_index('raw_cycle_id').loc[r.raw_cycle_id]
  role='Train' if cyc.start_ophrs2_h>=b.bin_start_ophrs2_h.iloc[0] and cyc.end_ophrs2_h<=b.bin_end_ophrs2_h.iloc[2102] else 'Validation' if cyc.start_ophrs2_h>=b.bin_start_ophrs2_h.iloc[2103] and cyc.end_ophrs2_h<=b.bin_end_ophrs2_h.iloc[2403] else 'Other_or_boundary_cycle'
  records.append(dict(run_id=r.run_id,raw_cycle_id=int(r.raw_cycle_id),target_row=k if hit else np.nan,
   measurement_first_h=r.measurement_first_h,measurement_last_h=r.measurement_last_h,available_h=r.available_h,
   V=r.V_median,QC=r.eligible_electrical_stability,whole_cycle_role=role,bin_observation_qualified=hit,
   rejection='' if hit else 'Stable window and conservative availability not wholly within one N1bin'))
 runs=pd.DataFrame(records);cs('outputs/DWELL_TO_BIN_IDENTITY.csv',runs)
 labels=[]
 for k,g in runs.loc[runs.bin_observation_qualified].groupby('target_row'):
  good=g.loc[g.QC];roles=g.whole_cycle_role.unique();role=roles[0] if len(roles)==1 else 'Other_or_boundary_cycle'
  labels.append(dict(target_row=int(k),actual_h=b.bin_end_ophrs2_h.iloc[int(k)],legacy_dwells=len(g),QC_dwells=len(good),
   raw_cycle_ids='|'.join(str(x) for x in sorted(g.raw_cycle_id.unique())),whole_cycle_role=role,
   actual_V=float(g.V.median()),QC_V=float(good.V.median()) if len(good) else np.nan,
   first_measurement_h=g.measurement_first_h.min(),last_available_h=g.available_h.max()))
 return pd.DataFrame(labels)

def window_audit(native,labels,b,cycles):
 idx=labels.set_index('target_row');records=[]
 for role,origins in native.items():
  for o in origins:
   full=[];qc=[];cy=set();times=[]
   for t in range(o+1,o+25):
    if t in idx.index:
     r=idx.loc[t];ok=r.whole_cycle_role==role and r.first_measurement_h>b.bin_end_ophrs2_h.iloc[o]
     if ok:
      full.append(t)
      if np.isfinite(r.QC_V):qc.append(t);cy.update(map(int,str(r.raw_cycle_ids).split('|')));times.append(float(r.last_available_h))
   records.append(dict(role=role,origin_row=o,origin_h=b.bin_end_ophrs2_h.iloc[o],future_targets=24,
    actual_fixed_bin_labels=len(full),QC_fixed_bin_labels=len(qc),dense24_legacy=len(full)==24,dense24_QC=len(qc)==24,
    target_rows='|'.join(map(str,qc)),target_raw_cycle_ids='|'.join(map(str,sorted(cy))),
    last_label_available_h=max(times) if times else np.nan,
    first_target_future=True,future_conditions_as_input=False,original_input_shape='48x14',original_output_shape='24'))
 w=pd.DataFrame(records);stats=[]
 for role,g in w.groupby('role',sort=False):
  targets=set(int(i) for z in g.target_rows for i in str(z).split('|') if i!='');cy=set(int(i) for z in g.target_raw_cycle_ids for i in str(z).split('|') if i!='')
  stats.append(dict(role=role,T0_legal_origins=len(g),T1_any_QC_origins=int(g.QC_fixed_bin_labels.gt(0).sum()),T1_dense24_origins=int(g.dense24_QC.sum()),
   T1_dense24_before_QC=int(g.dense24_legacy.sum()),T1_unique_QC_bin_labels=len(targets),T1_unique_target_cycles=len(cy),
   T1_repeated_query_label_pairs=int(g.QC_fixed_bin_labels.sum()),median_observed_steps=float(g.QC_fixed_bin_labels.median()),
   min_observed_steps=int(g.QC_fixed_bin_labels.min()),max_observed_steps=int(g.QC_fixed_bin_labels.max())))
 traincycles=set(int(i) for z in w.loc[w.role.eq('Train'),'target_raw_cycle_ids'] for i in str(z).split('|') if i!='')
 valcycles=set(int(i) for z in w.loc[w.role.eq('Validation'),'target_raw_cycle_ids'] for i in str(z).split('|') if i!='')
 assert not traincycles&valcycles
 cutoff=b.bin_end_ophrs2_h.iloc[min(native['Validation'])]
 assert w.loc[w.role.eq('Train'),'last_label_available_h'].max()<cutoff
 js('outputs/CHRONOLOGICAL_LABEL_FIREWALL.json',dict(train_validation_target_cycle_overlap=0,Train_last_label_available_h=float(w.loc[w.role.eq('Train'),'last_label_available_h'].max()),
  first_Validation_origin_h=float(cutoff),label_purge_PASS=True,shared_past_history_allowed=True,
  reference_first10_available_h=float(rd(PAA/'outputs/P1S_REFERENCE_COMPLETE_CYCLES.csv').measurement_end_ophrs2_h.max()),
  future_conditions_in_model_input=False,new_scaler_fit=False,model_fit=False))
 return w,pd.DataFrame(stats)

def frozen_cycle_inputs(fixed,legacy,b,features):
 legal=rd(ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_LEGAL_SAMPLES.csv');rows=[]
 for r in legal.itertuples():
  origin=float(r.origin_available_h);available=np.flatnonzero(b.bin_end_ophrs2_h.to_numpy()<=origin)
  o=int(available[-1]) if len(available) else -1;valid=False
  if o>=47:
   z=b.iloc[o-47:o+1];valid=bool(z.bin_valid_for_windows.eq(1).all() and z.temporal_segment_id.nunique()==1 and np.isfinite(z[features]).all(axis=None) and np.allclose(np.diff(z.bin_end_ophrs2_h),.5,atol=1e-6,rtol=0))
   valid=valid and pd.Timestamp(b.last_datetime.iloc[o])<=pd.Timestamp(r.origin_calendar)
  rows.append(dict(H=r.H,fold=r.fold,role=r.role,origin_raw_cycle_id=r.origin_raw_cycle_id,target_raw_cycle_id=r.target_raw_cycle_id,
   origin_available_h=origin,target_actual_end_h=r.target_actual_end_h,target_available_h=r.target_available_h,
   actual_lead_h=r.target_actual_end_h-r.origin_actual_end_h,last_N1_history_row=o,valid_48step_dynamic_history=valid,
   future_target_is_after_origin=r.target_actual_end_h>origin,
   target_voltage_V=r.target_R*.71383023635,reference_voltage_V=.71383023635))
 z=pd.DataFrame(rows);cs('outputs/CYCLE_TASK_LEGAL_SAMPLES_AND_DYNAMIC_HISTORY.csv',z)
 count=z.groupby(['H','fold','role']).agg(frozen_cycle_samples=('H','size'),with_legal_48step_dynamic_history=('valid_48step_dynamic_history','sum'),lead_min_h=('actual_lead_h','min'),lead_max_h=('actual_lead_h','max')).reset_index()
 cs('outputs/CYCLE_TASK_COUNTS_BY_FOLD.csv',count)

def cycle_sequence_audit(fixed,cycles):
 ids=sorted(fixed.raw_cycle_id);windows=[]
 for i in range(len(ids)-71):
  z=ids[i:i+72]
  if np.all(np.diff(z)==1):windows.append(z)
 # Irregular enumeration would be a different task; retain explicit physical times.
 irregular=[]
 for i in range(len(ids)-71):
  z=fixed.iloc[i:i+72];irregular.append(dict(first_cycle=int(z.raw_cycle_id.iloc[0]),origin_cycle=int(z.raw_cycle_id.iloc[47]),last_target_cycle=int(z.raw_cycle_id.iloc[-1]),
   history_measurement_span_h=float(z.last_measurement_h.iloc[47]-z.last_measurement_h.iloc[0]),future_measurement_span_h=float(z.last_measurement_h.iloc[-1]-z.last_measurement_h.iloc[47]),
   skips_missing_or_partial_cycles=not bool(np.all(np.diff(z.raw_cycle_id)==1))))
 cs('outputs/IRREGULAR_48OBS_TO_24OBS_NOT_NATIVE_TASK.csv',pd.DataFrame(irregular))
 return dict(cycle_points=len(ids),consecutive_raw_cycle_48_to24_windows=len(windows),irregular_enumeration_windows=len(irregular),
  no_halfhour_relabeling=True,irregular_windows_are_new_task=True,independent_life_trajectories=1)

if __name__=='__main__':main()

"""Frozen event qualification; no interpolation, model fitting or late-life reads."""
import csv, hashlib, io, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]; ROOT=P.parent
N1='Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
SOURCES={
 'refs':'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv',
 'timing':'Phase_PAV_PAA_Anchor_Validity_Qualification/outputs/POLARIZATION_TIME_PROVENANCE.csv',
 'support':'Phase_PAV_PAA_Anchor_Validity_Qualification/outputs/ANCHOR_MEDIAN_SUPPORT_PROVENANCE.csv',
 'p1s':'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv',
 'measure':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',
 'ahc':'Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/outputs/B2_CHRONOLOGICAL_PREFIX_EVALUATION.csv',
 'health_decision':'Phase_TFT_P1S96_HealthAware_Fusion/outputs/RESEARCH_DECISION.json',
}
REVIEWS=[
 'Phase_F_TFT_DualPath_RegimeFusion/README.md',
 'Phase_H_Determinism_MultiSeed_Stability/README.md',
 'Phase_P2_Final_Gate_Ablation_Training/outputs/P2_FINAL_RESEARCH_DECISION.json',
 'Phase_P3_Full450_Epoch_Sensitivity/outputs/P3_FINAL_SENSITIVITY_DECISION.json',
 'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_SOH_GROUND_TRUTH_FEASIBILITY.md',
 'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_DEGRADATION_REPRESENTATION_EVIDENCE.md',
 'Phase_PAA_P1_StateToken_PreAnchor_Alignment/reports/FINAL_P1S_PREANCHOR_AUDIT.md',
 'Phase_PAV_PAA_Anchor_Validity_Qualification/reports/FINAL_PAA_VALIDITY_QUALIFICATION.md',
 'Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/reports/FINAL_AFFINE_CALIBRATION_CHARACTERIZATION.md',
 'Phase_PHI_P1S_B2_Health_Index_Consolidation/outputs/PHI_SUMMARY.json',
 'Phase_SOH_LongHorizon_Future_Forecasting/01_数据与预测任务定义.md',
 'Phase_P1S_96h_ThreeFold_Comparison/00_实验总结.md',
 'Phase_TFT_P1S96_HealthAware_Fusion/00_最终实验总结.md',
]
PRED_PATHS=[f'Phase_H_Determinism_MultiSeed_Stability/outputs/validation_predictions/H-F2-Seed{s}.csv' for s in (42,52,62)]+[
 f'Phase_TFT_P1S96_HealthAware_Fusion/outputs/predictions/F2_MATCHED_seed{s}.csv' for s in (42,52,62)]
FIELDS=['bin_start_ophrs2_h','bin_end_ophrs2_h','first_datetime','last_datetime','split','bin_valid_for_windows',
 'cell_voltage_mean','cell_voltage_temporal_std','actual_current_density_A_cm2_mean','actual_current_density_A_cm2_std',
 'actual_current_density_A_cm2_min','actual_current_density_A_cm2_max','load_transition_count','zero_current_fraction',
 'internal_datetime_gap_count','maximum_internal_datetime_gap_s','coverage_ratio']+[
 'actual_'+f+'_'+s for f in ['anode_inlet_temperature_C','cathode_inlet_temperature_C','coolant_inlet_temperature_C',
 'anode_inlet_pressure_kPag','cathode_inlet_pressure_kPag','anode_inlet_rh_pct','cathode_inlet_rh_pct'] for s in ['mean','std']]+[
 'actual_anode_stoichiometry_mean','actual_cathode_stoichiometry_mean','actual_anode_hydrogen_flow_nl_min_mean','actual_cathode_air_flow_nl_min_mean']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(rel,obj):
 path=P/rel;path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def cs(rel,data):
 path=P/rel;path.parent.mkdir(parents=True,exist_ok=True)
 data.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')
def md(rel,s): (P/rel).write_bytes((s.strip()+'\n').encode('utf-8'))
def bounded_bytes():
 # Exactly 3005 Development records, never a full mixed-table hash/read.
 with (ROOT/N1).open('rb') as f: return b''.join(f.readline() for _ in range(3006))
def overlaps(a,b,c,d): return (a<=d)&(b>=c)
def invariant_delta(frame,bins):
 o=frame.origin_row.to_numpy(int);t=frame.target_row.to_numpy(int)
 expected=bins.cell_voltage_mean.to_numpy()[t].astype(np.float32)-bins.cell_voltage_mean.to_numpy()[o].astype(np.float32)
 return float(np.max(np.abs(expected-frame.y_true_delta_V.to_numpy())))

def freeze():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists(),'Already frozen'
 for folder in ['configs','inputs','outputs','figures','reports','tests']: (P/folder).mkdir(exist_ok=True)
 raw=bounded_bytes();b=pd.read_csv(io.BytesIO(raw),usecols=FIELDS)
 assert len(b)==3005 and b.bin_end_ophrs2_h.max()<1819.997738
 cs('inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',b)
 config=dict(base_commit='43b52ea1bc5f6ced3b3c70965ebdfe7a88e1997a',
  purpose='Qualify reference events before any SOH mapping fit',n1_records=3005,n1_prefix_sha256=hashlib.sha256(raw).hexdigest(),
  source_sha256={f:sha(ROOT/f) for f in sorted(set(list(SOURCES.values())+REVIEWS+PRED_PATHS))},
  code_sha256=sha(__file__),bin_snapshot_sha256=sha(P/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv'),
  match_rule='Intersection of observed target-bin calendar interval with observed polarization interval; calendar overlap is only a necessary temporal diagnostic, not measurement equivalence',
  timing_gate='Exact reference OpHrs2 and actual prior-reference release time required; unknown stays unknown',
  condition_gate='Same reference-state measurement definition and independently justified joint physical condition equivalence; current proximity alone never passes',
  measurement_gate='Half-hour dynamic average is not a P1S dwell or polarization point; no ratio bridge allowed',
  future_actual_conditions='audit_only_not_prediction_inputs',j_near_band_A_cm2=[0.95,1.05],
  near_band_role='Fixed descriptive diagnostic only, not physical certification or tunable matching caliper',
  evaluation='Distinct qualified reference events, SOH percentage points; no metrics if no common legal events',
  conditional_minimal_mapping='Only if gates pass: chronological past-only affine calibration with persistence and past-only P1S/M2 on exact common events; no equation search',
  fitting_permitted_until_qualification=False,test_access=False)
 js('configs/FROZEN_PROTOCOL.json',config);md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'))
 md('.gitignore','__pycache__/\n*.pyc\noutputs/GIT_RECEIPT.local.json')
 md('.gitattributes','* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.png binary\n*.pdf binary')
 print('FROZEN',sha(P/'configs/FROZEN_PROTOCOL.json'))

def run():
 assert not (P/'outputs/DECISION.json').exists(),'Completed audit: use verify, do not repeat'
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
 assert sha(P/'configs/FROZEN_PROTOCOL.json')==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 assert cfg['code_sha256']==sha(__file__)
 for f,h in cfg['source_sha256'].items(): assert sha(ROOT/f)==h,f
 assert sha(P/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv')==cfg['bin_snapshot_sha256']
 allowed={(ROOT/f).resolve() for f in cfg['source_sha256']}
 def guard(event,args):
  if event!='open' or isinstance(args[0],int):return
  q=Path(os.fsdecode(args[0])).resolve();m=args[1];flags=args[2]
  writing=(isinstance(m,str) and any(x in m for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if q.is_relative_to(ROOT.parent) and not q.is_relative_to(P):
   if writing or q not in allowed:raise PermissionError('Historical write / unregistered read blocked: '+str(q))
 sys.addaudithook(guard)
 b=pd.read_csv(P/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',float_precision='round_trip')
 b['first_datetime']=pd.to_datetime(b.first_datetime);b['last_datetime']=pd.to_datetime(b.last_datetime)
 data={k:pd.read_csv(ROOT/f) for k,f in SOURCES.items() if f.endswith('.csv')}
 for k,d in data.items():cs(f'inputs/{k.upper()}_FROZEN.csv',d)
 refs=data['refs'];timing=data['timing'];assert refs.nominal_h.tolist()==[0,300,600,850,1150,1500,1750]
 assert len(timing)==6 and not timing.ophrs2_column_present.any()
 summary=[];allp=[]
 for file in PRED_PATHS:
  d=pd.read_csv(ROOT/file,float_precision='round_trip');d['cache_source']=file
  assert not d.duplicated(['origin_row','target_row','horizon_step']).any()
  assert (d.target_row-d.origin_row==d.horizon_step).all() and d.target_row.max()<len(b)
  diff=invariant_delta(d,b);assert diff<1e-7,(file,diff)
  o=d.origin_row.to_numpy(int);t=d.target_row.to_numpy(int)
  d['origin_actual_h']=b.bin_end_ophrs2_h.to_numpy()[o];d['target_actual_h']=b.bin_end_ophrs2_h.to_numpy()[t]
  d['origin_available_calendar']=b.last_datetime.to_numpy()[o]
  d['target_first_calendar']=b.first_datetime.to_numpy()[t];d['target_last_calendar']=b.last_datetime.to_numpy()[t]
  d['origin_voltage_V']=b.cell_voltage_mean.to_numpy()[o]
  d['predicted_dynamic_voltage_V']=d.origin_voltage_V+d.y_pred_delta_V
  d['actual_dynamic_voltage_V']=b.cell_voltage_mean.to_numpy()[t]
  # Explicit lineage: target actual conditions are diagnostic observations only.
  for f in FIELDS:
   if f.startswith('actual_') or f in ['load_transition_count','zero_current_fraction','cell_voltage_temporal_std']:
    d['origin_known_'+f]=b[f].to_numpy()[o];d['target_audit_only_'+f]=b[f].to_numpy()[t]
  err=(d.y_pred_delta_V-d.y_true_delta_V).to_numpy()*1000
  summary.append(dict(cache_source=file,seed=int(d.seed.iloc[0]),records=len(d),origins=d.origin_row.nunique(),distinct_targets=d.target_row.nunique(),
   RMSE_dynamic_mV=float(np.sqrt(np.mean(err**2))),MAE_dynamic_mV=float(np.mean(abs(err))),max_delta_reconstruction_difference_V=diff))
  allp.append(d)
 pred=pd.concat(allp,ignore_index=True);cs('outputs/USED_DYNAMIC_PREDICTIONS_AUDIT_ONLY.csv',pred);cs('outputs/DYNAMIC_CACHE_METRICS.csv',pd.DataFrame(summary))
 # Future true conditions are used only here to assess measurement support.
 target_ids=np.sort(pred.target_row.unique());targets=b.iloc[target_ids].copy();targets.insert(0,'target_row',target_ids)
 j=targets.actual_current_density_A_cm2_mean
 targets['j_near_reference_diagnostic']=j.between(.95,1.05)
 targets['independent_reference_condition_certification']=False
 targets['measurement_equivalent_to_polarization']=False
 cs('outputs/DISTINCT_TARGET_CONDITION_AUDIT.csv',targets)
 hist=[]
 for cache,g in pred.groupby('cache_source'):
  for origin in sorted(g.origin_row.unique()):
   h=b.iloc[int(origin)-47:int(origin)+1];assert len(h)==48
   hist.append(dict(cache_source=cache,origin_row=int(origin),origin_actual_h=float(b.iloc[origin].bin_end_ophrs2_h),
    history_first_h=float(h.bin_start_ophrs2_h.iloc[0]),history_last_h=float(h.bin_end_ophrs2_h.iloc[-1]),
    history_last_calendar=str(h.last_datetime.iloc[-1]),j_mean=float(h.actual_current_density_A_cm2_mean.mean()),
    j_min=float(h.actual_current_density_A_cm2_min.min()),j_max=float(h.actual_current_density_A_cm2_max.max()),
    history_rows=48,future_actual_input=False))
 cs('outputs/ORIGIN_HISTORY_SUPPORT.csv',pd.DataFrame(hist))
 events=[]
 for _,r in refs.iterrows():
  n=int(r.nominal_h);row=dict(nominal_checkpoint_h=n,reference_SOH=float(r.Y),reference_voltage_V=float(r.Y*.6989533305168152),
   reference_denominator_V=.6989533305168152,precise_reference_ophrs_h=np.nan,reference_release_calendar='UNKNOWN',
   legal_common_forecast_events=0,mapping_prediction_SOH=np.nan,persistence_prediction_SOH=np.nan,M2_prediction_SOH=np.nan,
   mapping_error_percentage_points=np.nan,available_reference_count=np.nan,calibration_a=np.nan,calibration_b=np.nan,
   actually_used_origin_row=np.nan,actually_used_target_row=np.nan,condition_certified=False)
  if n==0:
   row.update(reference_first_calendar='UNKNOWN',reference_last_calendar='UNKNOWN',overlapping_target_bins=0,overlapping_prediction_records=0,
    nearest_target_row=np.nan,nearest_gap_wallclock_h=np.nan,rejection='BoL normalization only; no precise forecast event / release identity')
  else:
   tm=timing.loc[timing.checkpoint_h.eq(n)].iloc[0];start=pd.Timestamp(tm.polarization_first_datetime);end=pd.Timestamp(tm.polarization_last_datetime)
   match=overlaps(targets.first_datetime,targets.last_datetime,start,end)
   gaps=np.maximum((start-targets.last_datetime).dt.total_seconds().to_numpy(),(targets.first_datetime-end).dt.total_seconds().to_numpy())/3600
   pos=int(np.argmin(np.maximum(gaps,0)));nr=int(targets.iloc[pos].target_row)
   row.update(reference_first_calendar=str(start),reference_last_calendar=str(end),overlapping_target_bins=int(match.sum()),
    overlapping_prediction_records=int(pred.target_row.isin(targets.loc[match,'target_row']).sum()),nearest_target_row=nr,
    nearest_target_first_calendar=str(targets.iloc[pos].first_datetime),nearest_target_last_calendar=str(targets.iloc[pos].last_datetime),
    nearest_gap_wallclock_h=float(max(gaps[pos],0)),pretest_ophrs_record_h=float(tm.PAA_gap_cutoff_ophrs2_h),
    pretest_ophrs_role='Pre-polarization durability record, not characterization age',
    rejection='Reference exact OpHrs2 absent; release time unknown; dynamic bin/polarization not certified equivalent'+('; no observed calendar overlap' if not match.any() else ''))
  events.append(row)
 event=pd.DataFrame(events);cs('outputs/REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv',event)
 # P1S event landing is retained as an observational comparison, not a target label conversion.
 meas=data['measure'].loc[data['measure'].stable_window.eq('middle_50')].copy()
 p1=data['p1s'].merge(meas[['cycle_interval_id','measurement_end_datetime','measurement_end_ophrs2_h','current_density',
  'anode_inlet_temperature_C','anode_inlet_pressure_kPag','anode_inlet_RH_pct','anode_hydrogen_flow_nl_min']],on='cycle_interval_id',validate='one_to_one')
 assert len(p1)==131
 p1['reference_V']=.71383023635;p1['matched_target_bin_row']=np.nan;p1['exact_measurement_equivalence']=False
 p1['reason']='P1S middle-dwell voltage is not whole-bin dynamic mean; no registered observation operator'
 for i,r in p1.iterrows():
  hits=targets.loc[(targets.bin_start_ophrs2_h<r.measurement_end_h)&(targets.bin_end_ophrs2_h>=r.measurement_end_h)]
  if len(hits)==1:p1.at[i,'matched_target_bin_row']=int(hits.iloc[0].target_row)
 p1['actual_measurement_end_calendar']='NOT_DIRECTLY_SAVED';p1['conservative_available_calendar']=p1.measurement_end_datetime
 cs('outputs/P1S_EVENT_BIN_LANDING_AUDIT.csv',p1)
 method=pd.DataFrame([dict(method=x,legal_common_events=0,new_predictions=0,MAE_SOH_percentage_points=np.nan,
  improvement_vs_persistence_percent=np.nan,status='NOT_EVALUABLE',reason=y) for x,y in [
 ('REFERENCE_PERSISTENCE','Past reference release identity unknown at cache origins'),
 ('PAST_ONLY_P1S_M2','No common qualified future reference; historical prefix records estimate current proxy and assume chronological reference availability'),
 ('TFT_DYNAMIC_BRIDGE','No qualified temporal and condition-equivalent reference events; no map fitted')]])
 cs('outputs/SOH_METHOD_ELIGIBILITY.csv',method)
 cond=dict(distinct_target_bins=len(targets),j_near_reference_bins=int(targets.j_near_reference_diagnostic.sum()),
  independently_condition_certified_bins=0,j_min=float(j.min()),j_max=float(j.max()),
  target_anode_RH_gt100_bins=int((targets.actual_anode_inlet_rh_pct_mean>100).sum()),
  P1S_j_min=float(p1.current_density.min()),P1S_j_max=float(p1.current_density.max()),
  P1S_temporal_bin_landings=int(p1.matched_target_bin_row.notna().sum()),P1S_equivalent_proxy_targets=0,
  hidden_state_missing=['membrane hydration','catalyst-interface reactant activities','independent quasi-steady certification'],
  halfhour_operator='Dynamic whole-bin average, raw P1S uses selected stable dwell subset')
 js('outputs/CONDITION_SUPPORT_SUMMARY.json',cond)
 decision=dict(status='TFT_TO_SOH_INSUFFICIENT_SUPPORT',prediction_records_audited=len(pred),cache_files=6,
  dynamic_origins_not_independent_health_samples=True,reference_rows=7,post_BoL_reference_events=6,
  calendar_overlap_bins_total=int(event.overlapping_target_bins.sum()),legal_common_reference_events=0,legal_proxy_equivalent_events=0,
  SOH_mapping_fits=0,new_SOHPredictions=0,SOH_MAE_percentage_points=None,improvement_percent=None,
  primary_blockers=['No precisely linked characterization OpHrs2 / release identity','No qualified dynamic-to-reference condition/measurement mapping'],
  reference_labels_exist_but_not_matched=True,voltage_error_is_separately_audited=True,
  test_accessed=False,neural_training=False,protocol_sha256=sha(P/'configs/FROZEN_PROTOCOL.json'))
 js('outputs/DECISION.json',decision)
 js('outputs/SOURCE_INTEGRITY.json',dict(all_sources_unchanged=all(sha(ROOT/f)==h for f,h in cfg['source_sha256'].items()),
  mixed_table_read='3005 records during freeze only; raw prefix digest only',raw_records_reparsed=0,Test=False))
 cs('outputs/REVIEWED_SOURCE_LEDGER.csv',pd.DataFrame([dict(path=f,sha256=cfg['source_sha256'][f]) for f in REVIEWS]))
 print(json.dumps(decision,ensure_ascii=False));print(json.dumps(cond,ensure_ascii=False))

if __name__=='__main__':
 {'freeze':freeze,'run':run}[sys.argv[1]]()

"""Add already-exposed Phase-F Development caches under the identical event rules."""
import json,sys
import numpy as np
import pandas as pd
from run_bridge_audit import P,ROOT,sha,js,cs,overlaps,invariant_delta
FILES=[f'Phase_F_TFT_DualPath_RegimeFusion/outputs/predictions/F2_{s}_seed42.csv' for s in ['validation','assessment']]
if sys.argv[1]=='freeze':
 assert not (P/'configs/F_CACHE_EXTENSION.json').exists()
 js('configs/F_CACHE_EXTENSION.json',dict(code_sha256=sha(__file__),source_sha256={f:sha(ROOT/f) for f in FILES},
  parent_protocol_sha256=sha(P/'configs/FROZEN_PROTOCOL.json'),rules_changed=False,
  reason='Cover original F2 saved Development Assessment as well as Validation; no new Assessment inference or selection',
  no_new_fit=True,no_Test=True))
else:
 assert not (P/'outputs/FINAL_DECISION.json').exists()
 cfg=json.loads((P/'configs/F_CACHE_EXTENSION.json').read_text());assert cfg['code_sha256']==sha(__file__)
 b=pd.read_csv(P/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',float_precision='round_trip')
 b['first_datetime']=pd.to_datetime(b.first_datetime);b['last_datetime']=pd.to_datetime(b.last_datetime)
 frames=[];metrics=[]
 for f,h in cfg['source_sha256'].items():
  assert sha(ROOT/f)==h;d=pd.read_csv(ROOT/f,float_precision='round_trip');assert invariant_delta(d,b)<1e-7
  o=d.origin_row.to_numpy(int);t=d.target_row.to_numpy(int)
  assert t.max()==(2403 if d.split.iloc[0]=='validation' else 3004)
  d['cache_source']=f;d['origin_actual_h']=b.bin_end_ophrs2_h.to_numpy()[o];d['target_actual_h']=b.bin_end_ophrs2_h.to_numpy()[t]
  d['origin_available_calendar']=b.last_datetime.to_numpy()[o];d['target_first_calendar']=b.first_datetime.to_numpy()[t];d['target_last_calendar']=b.last_datetime.to_numpy()[t]
  d['origin_voltage_V']=b.cell_voltage_mean.to_numpy()[o];d['predicted_dynamic_voltage_V']=d.origin_voltage_V+d.y_pred_delta_V
  for col in b:
   if col.startswith('actual_') or col in ['cell_voltage_temporal_std','load_transition_count','zero_current_fraction']:
    d['origin_known_'+col]=b[col].to_numpy()[o];d['target_audit_only_'+col]=b[col].to_numpy()[t]
  e=(d.y_pred_delta_V-d.y_true_delta_V)*1000
  metrics.append(dict(cache_source=f,seed=42,records=len(d),origins=d.origin_row.nunique(),distinct_targets=d.target_row.nunique(),RMSE_dynamic_mV=float(np.sqrt(np.mean(e**2))),MAE_dynamic_mV=float(abs(e).mean()),max_delta_reconstruction_difference_V=invariant_delta(d,b)))
  frames.append(d)
 d=pd.concat(frames,ignore_index=True);cs('outputs/F_ORIGINAL_USED_PREDICTIONS_AUDIT_ONLY.csv',d)
 cs('outputs/ALL_DYNAMIC_CACHE_METRICS.csv',pd.concat([pd.read_csv(P/'outputs/DYNAMIC_CACHE_METRICS.csv'),pd.DataFrame(metrics)],ignore_index=True))
 original=pd.read_csv(P/'outputs/DISTINCT_TARGET_CONDITION_AUDIT.csv');ids=sorted(set(original.target_row)|set(d.target_row))
 targets=b.iloc[ids].copy();targets.insert(0,'target_row',ids);targets['j_near_reference_diagnostic']=targets.actual_current_density_A_cm2_mean.between(.95,1.05)
 targets['independent_reference_condition_certification']=False;targets['measurement_equivalent_to_polarization']=False
 cs('outputs/ALL_DISTINCT_TARGET_CONDITION_AUDIT.csv',targets)
 event=pd.read_csv(P/'outputs/REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv');initial=pd.read_csv(P/'outputs/USED_DYNAMIC_PREDICTIONS_AUDIT_ONLY.csv',usecols=['target_row'])
 for i,r in event.iterrows():
  if r.nominal_checkpoint_h==0:continue
  start=pd.Timestamp(r.reference_first_calendar);end=pd.Timestamp(r.reference_last_calendar)
  mask=overlaps(targets.first_datetime,targets.last_datetime,start,end);matched=targets.loc[mask,'target_row']
  gap=np.maximum((start-targets.last_datetime).dt.total_seconds().to_numpy(),(targets.first_datetime-end).dt.total_seconds().to_numpy())/3600
  near=int(np.argmin(np.maximum(gap,0)));tr=targets.iloc[near]
  event.at[i,'overlapping_target_bins']=int(mask.sum());event.at[i,'overlapping_prediction_records']=int(initial.target_row.isin(matched).sum()+d.target_row.isin(matched).sum())
  event.at[i,'nearest_target_row']=int(tr.target_row);event.at[i,'nearest_gap_wallclock_h']=float(max(gap[near],0))
  event.at[i,'nearest_target_first_calendar']=str(tr.first_datetime);event.at[i,'nearest_target_last_calendar']=str(tr.last_datetime)
 cs('outputs/ALL_REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv',event)
 p1=pd.read_csv(P/'outputs/P1S_EVENT_BIN_LANDING_AUDIT.csv');p1['matched_target_bin_row']=np.nan
 for i,r in p1.iterrows():
  hits=targets.loc[(targets.bin_start_ophrs2_h<r.measurement_end_h)&(targets.bin_end_ophrs2_h>=r.measurement_end_h)]
  if len(hits)==1:p1.at[i,'matched_target_bin_row']=int(hits.iloc[0].target_row)
 cs('outputs/ALL_P1S_EVENT_BIN_LANDING_AUDIT.csv',p1)
 result=json.loads((P/'outputs/DECISION.json').read_text());result.update(prediction_records_audited=result['prediction_records_audited']+len(d),cache_files=8,
  distinct_target_bins=len(targets),j_near_reference_bins=int(targets.j_near_reference_diagnostic.sum()),
  P1S_temporal_bin_landings=int(p1.matched_target_bin_row.notna().sum()),calendar_overlap_bins_total=int(event.overlapping_target_bins.sum()),
  F_existing_Development_assessment_read=True,new_assessment_inference=False,qualification_rules_changed=False)
 assert result['calendar_overlap_bins_total']==0
 js('outputs/FINAL_DECISION.json',result)
 print(json.dumps(result,ensure_ascii=False))

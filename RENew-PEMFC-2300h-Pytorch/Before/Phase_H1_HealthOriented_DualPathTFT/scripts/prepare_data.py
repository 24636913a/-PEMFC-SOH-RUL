"""Recheck cycle completion and event availability before model fitting."""
from common import *
import math,yaml

def main():
 assert not (P/'outputs/DATA_PREPARED.json').exists(),'Prepared already'
 for folder in ['configs','data_contracts','outputs','models','figures','tests']:(P/folder).mkdir(exist_ok=True)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
 n1=ROOT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
 meta=pd.read_csv(n1,nrows=3007,usecols=['bin_end_ophrs2_h','split','first_datetime','last_datetime'])
 ids=np.flatnonzero(meta.bin_end_ophrs2_h.to_numpy()<BOUNDARY);assert len(meta)==3007 and np.array_equal(ids,np.arange(3005))
 assert not meta.iloc[ids]['split'].str.contains('test',case=False).any()
 b=pd.read_csv(n1,nrows=len(ids));cs('data_contracts/N1_3007_METADATA_MASK.csv',meta.assign(source_row=np.arange(len(meta)),retained=np.isin(np.arange(len(meta)),ids)))
 cs('data_contracts/N1_DEVELOPMENT.csv',b)
 hcfg=yaml.safe_load((ROOT/'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
 features=hcfg['task']['raw_f2_feature_order']
 source=rd(AUDIT/'outputs/CYCLE_TASK_LEGAL_SAMPLES_AND_DYNAMIC_HISTORY.csv');source=source.loc[source.H.eq(1)&source.valid_48step_dynamic_history].copy()
 original=rd(ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_LEGAL_SAMPLES.csv')
 keys=['H','fold','role','origin_raw_cycle_id','target_raw_cycle_id'];source=source.merge(original[keys+['sample_key']],on=keys,validate='one_to_one')
 cy=rd(RCR/'outputs/RAW_CYCLE_CATALOG.csv').set_index('raw_cycle_id');states=rd(RCR/'outputs/RAW_STATE_RUNS.csv').set_index('run_id')
 measurement=rd(PAA/'outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv');measurement=measurement.loc[measurement.stable_window.eq('middle_50')].set_index('raw_cycle_id');assert len(measurement)==141
 exact=rd(AUDIT/'outputs/FIXED_PRIMARY_CYCLE_OBSERVATIONS.csv').set_index('raw_cycle_id')
 causal=rd(ROOT/'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv')
 def confirmation(k):
  c=cy.loc[k];i=int(c.end_run_id_exclusive);a=states.loc[i];z=states.loc[i+1]
  if (a.state_token,z.state_token)!=('15||14.000','15||0.000'):raise ValueError('Next anchor not certified')
  return float(z.end_ophrs2_h),pd.Timestamp(z.end_datetime),i+1
 rows=[];arrays=[];exclusions=[]
 for r in source.itertuples():
  k=int(r.origin_raw_cycle_id);t=int(r.target_raw_cycle_id);reason=[]
  if k not in measurement.index or t not in measurement.index:reason.append('Not frozen141cycle observations')
  assert t==k+1 and cy.loc[k].complete_cycle and cy.loc[t].complete_cycle
  try:oh,oc,orun=confirmation(k);th,tc,trun=confirmation(t)
  except (ValueError,KeyError) as e:exclusions.append(dict(sample_key=r.sample_key,fold=r.fold,role=r.role,reason='Completion confirmation missing:'+str(e)));continue
  target=exact.loc[t];origin=measurement.loc[k]
  history_ids=np.flatnonzero((b.bin_end_ophrs2_h.to_numpy()<=oh)&(pd.to_datetime(b.last_datetime).to_numpy(dtype='datetime64[ns]')<=oc.to_datetime64()))
  o=int(history_ids[-1]) if len(history_ids) else -1
  x=None
  if o<47:reason.append('Too few dynamic history bins')
  else:
   hist=b.iloc[o-47:o+1]
   if not (hist.bin_valid_for_windows.eq(1).all() and hist.temporal_segment_id.nunique()==1 and np.allclose(np.diff(hist.bin_end_ophrs2_h),.5,rtol=0,atol=1e-6) and np.isfinite(hist[features]).all(axis=None)):reason.append('48bin dynamic history no longer legal at confirmed origin')
   x=hist[features].to_numpy(dtype='float32')
  if target.first_measurement_h<=oh or pd.Timestamp(target.last_measurement_calendar)<=oc:reason.append('Target fixed-state measurement not wholly future')
  # Whole frozen target must lie within its original Train/Validation range.
  lo,mid,hi={'fold_1':(72.78869,946.393214,1164.794345),'fold_2':(72.78869,1164.794345,1383.195476),'fold_3':(72.78869,1383.195476,BOUNDARY)}[r.fold]
  if r.role=='train' and (target.first_measurement_h<lo or th>mid):reason.append('Confirmed Train target outside original Train interval')
  if r.role=='validation' and (target.first_measurement_h<mid or th>=hi):reason.append('Confirmed Validation target outside original Validation interval')
  # Recompute exactly the same five Ridge96 features at the NEW confirmed origin.
  history=[]
  for z in causal.itertuples():
   if z.measurement_end_h<oh-96 or z.measurement_end_h>oh:continue
   try:zh,zc,_=confirmation(int(z.raw_cycle_id))
   except (KeyError,ValueError):continue
   if zh<=oh and zc<=oc:history.append((z.measurement_end_h,z.R))
  history.sort();span=history[-1][0]-history[0][0] if len(history)>1 else 0.
  if len(history)<5 or span<67.2:reason.append('Ridge96 history qualification not met at confirmed origin')
  if reason:exclusions.append(dict(sample_key=r.sample_key,fold=r.fold,role=r.role,reason=';'.join(reason)));continue
  hours=np.array([z[0] for z in history]);values=np.array([z[1] for z in history]);hours-=hours.mean()
  originV=float(origin.P1S_voltage_V);targetV=float(measurement.loc[t].P1S_voltage_V)
  assert abs(targetV-target.V_cycle_median)<1e-12
  row=dict(sample_key=r.sample_key,fold=r.fold,role=r.role,origin_raw_cycle_id=k,target_raw_cycle_id=t,
   origin_confirmed_h=oh,origin_confirmed_calendar=str(oc),confirmation_last_run_id=orun,
   origin_cycle_end_calendar=cy.loc[k].cycle_end,target_cycle_start_calendar=cy.loc[t].cycle_start,
   original_measurement_origin_h=r.origin_available_h,origin_completion_delay_h=oh-r.origin_available_h,
   target_first_measurement_h=target.first_measurement_h,target_last_measurement_h=target.last_measurement_h,
   target_measurement_last_calendar=target.last_measurement_calendar,target_available_confirmed_h=th,target_available_calendar=str(tc),
   forecast_actual_measurement_end_lead_h=target.last_measurement_h-oh,future_actual_lead_input=False,
   target_cycle_has_initial_anchor_observed=pd.Timestamp(cy.loc[t].cycle_start)<oc,
   origin_voltage_V=originV,true_voltage_V=targetV,last_N1_history_row=o,history_last_calendar=b.last_datetime.iloc[o],
   history_first_h=b.bin_start_ophrs2_h.iloc[o-47],history_last_h=b.bin_end_ophrs2_h.iloc[o],
   Ridge_origin_R=originV/VREF,Ridge_mean_R=float(values.mean()),Ridge_std_R=float(values.std()),
   Ridge_slope_R_per_h=float(np.dot(hours,values-values.mean())/np.dot(hours,hours)),
   Ridge_origin_minus_median_R=originV/VREF-float(np.median(values)),Ridge_observations=len(history),Ridge_span_h=span,
   reference_available=True,reference_V=VREF)
  row['array_index']=len(arrays);arrays.append(x);rows.append(row)
 allsamples=pd.DataFrame(rows).sort_values(['fold','role','origin_confirmed_calendar']).reset_index(drop=True)
 # A fixed1hour pre-origin readiness reserve, later checked against measured fit times.
 retained=[];inner=[];counts=[]
 for fold,g in allsamples.groupby('fold',sort=True):
  tr=g.loc[g.role.eq('train')].sort_values('target_available_calendar');va=g.loc[g.role.eq('validation')].sort_values('origin_confirmed_calendar')
  assert len(va)>0
  first=pd.to_datetime(va.origin_confirmed_calendar).min();safe=first-pd.Timedelta(hours=1)
  purge=pd.to_datetime(tr.target_available_calendar)>=safe
  for r in tr.loc[purge].itertuples():exclusions.append(dict(sample_key=r.sample_key,fold=fold,role='train',reason='Train confirmation label too late for1h readiness reserve'))
  tr=tr.loc[~purge];n=math.ceil(.15*len(tr));selection=tr.iloc[-n:];fit=tr.iloc[:-n]
  cutoff=pd.to_datetime(selection.origin_confirmed_calendar).min()-pd.Timedelta(hours=1)
  bad=pd.to_datetime(fit.target_available_calendar)>=cutoff
  for r in fit.itertuples():inner.append(dict(fold=fold,sample_key=r.sample_key,inner_role='purged' if r.sample_key in set(fit.loc[bad,'sample_key']) else 'fit'))
  for r in selection.itertuples():inner.append(dict(fold=fold,sample_key=r.sample_key,inner_role='selection'))
  fit=fit.loc[~bad]
  if len(fit)<2 or len(selection)<2:raise ValueError('Insufficient chronological inner split')
  assert not set(tr.target_raw_cycle_id)&set(va.target_raw_cycle_id)
  retained.extend([tr,va]);counts.append(dict(fold=fold,Train=len(tr),Validation=len(va),inner_fit=len(fit),inner_selection=len(selection),inner_purged=int(bad.sum()),
   outer_min_calendar_reserve_s=float((first-pd.to_datetime(tr.target_available_calendar).max()).total_seconds()),
   inner_min_calendar_reserve_s=float((pd.to_datetime(selection.origin_confirmed_calendar).min()-pd.to_datetime(fit.target_available_calendar).max()).total_seconds())))
 used=pd.concat(retained,ignore_index=True);cs('data_contracts/ALL_CANDIDATE_SAMPLES.csv',allsamples);cs('data_contracts/USED_SAMPLES.csv',used)
 cs('data_contracts/INNER_SPLITS.csv',pd.DataFrame(inner));cs('outputs/EXCLUDED_SAMPLES.csv',pd.DataFrame(exclusions,columns=['sample_key','fold','role','reason']))
 cs('outputs/SAMPLE_COUNTS.csv',pd.DataFrame(counts));np.savez_compressed(P/'data_contracts/RAW_DYNAMIC_HISTORIES.npz',x=np.stack(arrays))
 records={};xx=np.stack(arrays)
 for fold,g in used.groupby('fold'):
  tr=g.loc[g.role.eq('train')];ins=pd.DataFrame(inner);fit=tr.loc[tr.sample_key.isin(ins.loc[ins.fold.eq(fold)&ins.inner_role.eq('fit'),'sample_key'])]
  records[fold]={}
  for role,frame in [('inner',fit),('outer',tr)]:
   records[fold][role]=fit_preprocessing(xx[frame.array_index],frame.true_voltage_V.to_numpy(),features)
   records[fold][role]['sample_keys']=frame.sample_key.tolist()
 js('data_contracts/TRAIN_ONLY_PREPROCESSING.json',records)
 sources=[AUDIT/'outputs/CYCLE_TASK_LEGAL_SAMPLES_AND_DYNAMIC_HISTORY.csv',AUDIT/'outputs/FIXED_PRIMARY_CYCLE_OBSERVATIONS.csv',PAA/'outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',PAA/'outputs/P1S_REFERENCE_COMPLETE_CYCLES.csv',
  RCR/'outputs/RAW_CYCLE_CATALOG.csv',RCR/'outputs/RAW_STATE_RUNS.csv',ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_LEGAL_SAMPLES.csv',
  ROOT/'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv',
  ROOT/'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml',ROOT/'Phase_TFT_Original_Baseline/models/canonical_tft.py',
  ROOT/'Phase_TFT_FullFeature_CausalFilter_Comparison/configs/PREREGISTERED_E_COMPARISON.yaml']
 sources+=list((ROOT/'Phase_F_TFT_DualPath_RegimeFusion/models').glob('*.py'))
 for phase in ['Phase_F_TFT_DualPath_RegimeFusion','Phase_H_Determinism_MultiSeed_Stability']:
  sources+=list((ROOT/phase/'outputs/checkpoints').glob('*F2*.pt'))
 for phase in ['Phase_P1S_96h_ThreeFold_Comparison','Phase_Integrated_SOH_Physics_RUL_Research']:
  sources+=[q for q in (ROOT/phase/'models').glob('*') if q.is_file()]
 sources+=[ROOT/'Phase_TFT_DynamicVoltage_to_ReferenceSOH/outputs/ALL_REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv']
 ledger=[]
 for f in sorted(set(sources)):
  rel=f.relative_to(ROOT).as_posix();ledger.append(dict(path=rel,sha256=sha(f),git_blob=subprocess.check_output(['git','rev-parse','HEAD:'+ROOT.name+'/'+rel],cwd=REPO,text=True).strip()))
 cs('data_contracts/SOURCE_SHA256.csv',pd.DataFrame(ledger))
 js('outputs/DATA_PREPARED.json',dict(status='COMPLETE',base=BASE,Development_rows=3005,retained_history_arrays=len(arrays),frozen_cycle_labels=141,new5cycles_used=False,
  feature_names=features,Top10_rule='Train-only origin-feature abs Spearman to scalar H1 V; mandatory mean_voltage, stable index ties',
  confirmation='Successor two-run15||14,15||0 anchor completed; target has only initial anchor known, all fixed-state voltage observations remain future',
  reference_available_h=171.82091,Test_features=False,new_raw_reads=False))
 print(pd.DataFrame(counts).to_string(index=False));print('excluded',len(exclusions),'sources',len(ledger))
if __name__=='__main__':main()

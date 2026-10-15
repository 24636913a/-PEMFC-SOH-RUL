"""Independent count/label replay; historical all-dwell and nine raw spot checks.

No extraction/analysis imports. Raw spot checks use csv + datetime + statistics
to independently avoid pandas time-unit and vectorized matching assumptions.
"""
import csv,datetime,hashlib,json,statistics
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
def rd(p):return pd.read_csv(p,float_precision='round_trip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw_checks(d):
 manifest=rd(P/'outputs/RAW_SOURCE_PREFIX_PROGRESS.csv');chosen=[]
 for token in ['221||221.200','156||156.800','313||313.600']:
  g=d.loc[d.state_token.eq(token)&d.complete_cycle&d.eligible_electrical_stability].sort_values('run_id')
  chosen.extend(g.iloc[[0,len(g)//2,len(g)-1]].to_dict('records'))
 files={};store={int(r['run_id']):[] for r in chosen}
 for r in chosen:
  found=manifest.loc[(manifest.min_h<=r['measurement_first_h'])&(manifest.max_h>=r['measurement_last_h'])]
  assert len(found)==1
  source=found.iloc[0];files.setdefault(source.source_file,dict(count=int(source.certified_prefix_records),rows=[]))['rows'].append(r)
 active=[f'U{i} [V]' for i in [1,2,3,4,5,7,8,9,10,11]]
 base=REPO/'数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据'
 outputs=[]
 for name,info in files.items():
  print('Independent raw spot check:',name,flush=True)
  windows=[(datetime.datetime.fromisoformat(r['window_start']),datetime.datetime.fromisoformat(r['window_end']),int(r['run_id'])) for r in info['rows']]
  digest=hashlib.sha256()
  # Bound by the certified raw-file row count, never N1's3007 rule.
  with (base/name).open('rb') as f:
   for _ in range(info['count']+1):line=f.readline();assert line;digest.update(line)
  certificate=manifest.loc[manifest.source_file.eq(name)].iloc[0];assert digest.hexdigest()==certificate.certified_source_prefix_SHA256
  with (base/name).open(encoding='utf-8-sig',newline='') as stream:
   reader=csv.DictReader(stream,delimiter=';')
   for index,row in zip(range(info['count']),reader):
    # Left iterator stops before fetching another CSV record at the boundary.
    dt=datetime.datetime.fromisoformat(row['DateTime'])
    for start,end,run in windows:
     if start<=dt<end:
      numeric=lambda col:float(row[col].replace(',',''))
      age=numeric('OpHrs2 [h]');assert age<1819.997738
      v=statistics.fmean(numeric(c) for c in active);j=numeric('j.S.act [A/cm2]')
      store[run].append((dt,age,v,j))
 for r in chosen:
  a=store[int(r['run_id'])];assert len(a)==r['selected_sample_count']
  assert a[0][0]==datetime.datetime.fromisoformat(r['measurement_first_calendar'])
  assert a[-1][0]==datetime.datetime.fromisoformat(r['measurement_last_calendar'])
  vv=[z[2] for z in a];jj=[z[3] for z in a];tt=[(z[0]-a[0][0]).total_seconds() for z in a]
  vm=statistics.median(vv);std=statistics.pstdev(jj)
  tmean=statistics.fmean(tt);vmean=statistics.fmean(vv)
  slope=sum((x-tmean)*(v-vmean) for x,v in zip(tt,vv))/sum((x-tmean)**2 for x in tt)*60000
  assert abs(vm-r['V_median'])<1e-12 and abs(std-r['j_std'])<1e-12 and abs(slope-r['V_drift_mV_per_min'])<1e-9
  outputs.append(dict(run_id=int(r['run_id']),state_token=r['state_token'],samples=len(a),V_median_difference_V=vm-r['V_median'],
   current_std_difference=std-r['j_std'],voltage_slope_difference_mV_min=slope-r['V_drift_mV_per_min'],calendar_identity_exact=True))
 return outputs

def verify():
 cfg=json.loads((P/'configs/FROZEN_AUDIT.json').read_text());assert sha(P/'configs/FROZEN_AUDIT.json')==(P/'configs/FROZEN_AUDIT.sha256').read_text().strip()
 sources=rd(P/'data_contracts/SOURCE_SHA256.csv');assert all(sha(ROOT/r.path)==r.sha256 for r in sources.itertuples())
 raw=rd(P/'outputs/RAW_FIXED_STATE_DWELL_MEASUREMENTS.csv');states=rd(ROOT/'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_STATE_RUNS.csv')
 expected=states.loc[states.duration_wall_s.ge(120)&states.median_set_current_A.gt(0)&states.stop_or_zero_current_flag.eq(0)]
 assert raw.run_id.tolist()==expected.sort_values('start_datetime').run_id.tolist()
 assert raw.measurement_last_h.max()<1819.997738
 q=(raw.selected_sample_count.ge(30)&raw.j_std.le(.005)&raw.V_drift_mV_per_min.abs().le(2.)&raw[['T_anode_C_span','T_cathode_C_span','T_coolant_C_span']].max(axis=1).le(2.))
 assert np.array_equal(q.to_numpy(),raw.eligible_electrical_stability.to_numpy())
 hist=rd(ROOT/'Phase_RHA_Raw_Operation_Point_Health_Alignment/outputs/CYCLE_POINT_MEASUREMENTS.csv');hist=hist.loc[hist.stable_window.eq('middle_50')]
 h=hist.merge(raw,left_on='rcr_run_id',right_on='run_id',validate='one_to_one')
 assert len(h)==3796
 assert h.raw_sample_count.eq(h.selected_sample_count).all()
 maxd=float(abs(h.V_median-h.cell_voltage_mean_V).max());assert maxd<1e-12
 old=rd(ROOT/'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv');old=old.loc[old.stable_window.eq('middle_50')]
 cyc=raw.loc[raw.state_token.eq('221||221.200')&raw.complete_cycle].groupby('raw_cycle_id').V_median.median()
 diff=cyc.loc[old.raw_cycle_id].to_numpy()-old.P1S_voltage_V.to_numpy();assert abs(diff).max()<1e-12
 assert len(cyc)==146 and len(old)==141
 b=rd(P/'data_contracts/N1_DEVELOPMENT.csv');meta=rd(P/'data_contracts/N1_3007_METADATA_MASK.csv')
 assert np.flatnonzero(meta.bin_end_ophrs2_h.to_numpy()<1819.997738).tolist()==list(range(len(b))) and len(b)==3005
 assert not b['split'].str.contains('test',case=False).any()
 land=rd(P/'outputs/DWELL_TO_BIN_IDENTITY.csv')
 for r in land.itertuples():
  candidates=np.flatnonzero((b.bin_start_ophrs2_h.to_numpy()<=r.measurement_first_h)&(b.bin_end_ophrs2_h.to_numpy()>=r.available_h))
  assert bool(r.bin_observation_qualified)==bool(len(candidates)==1)
  if len(candidates):assert int(candidates[0])==r.target_row
 labels=rd(P/'outputs/PRIMARY_FIXED_BIN_LABELS.csv');windows=rd(P/'outputs/T0_T1_WINDOW_QUALIFICATION.csv')
 labelindex=labels.set_index('target_row');targetcycles={'Train':set(),'Validation':set()};pairs=0
 for r in windows.itertuples():
  selected=[]
  for t in range(r.origin_row+1,r.origin_row+25):
   if t not in labelindex.index:continue
   z=labelindex.loc[t]
   if z.whole_cycle_role==r.role and z.first_measurement_h>r.origin_h and np.isfinite(z.QC_V):selected.append(t);targetcycles[r.role].update(map(int,str(z.raw_cycle_ids).split('|')))
  assert len(selected)==r.QC_fixed_bin_labels and bool(len(selected)==24)==r.dense24_QC
  pairs+=len(selected)
 assert not targetcycles['Train']&targetcycles['Validation']
 assert not windows.dense24_QC.any() and not windows.dense24_legacy.any()
 seq=json.loads((P/'outputs/CYCLE_SEQUENCE_48_24_QUALIFICATION.json').read_text());ids=cyc.index.to_numpy()
 dense=sum(np.all(np.diff(ids[i:i+72])==1) for i in range(len(ids)-71));assert dense==seq['consecutive_raw_cycle_48_to24_windows']==0
 cycle=rd(P/'outputs/CYCLE_TASK_LEGAL_SAMPLES_AND_DYNAMIC_HISTORY.csv')
 assert (cycle.target_raw_cycle_id-cycle.origin_raw_cycle_id==cycle.H).all()
 assert (cycle.target_actual_end_h>cycle.origin_available_h).all()
 import yaml
 features=yaml.safe_load((ROOT/'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))['task']['raw_f2_feature_order']
 authority=rd(ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_LEGAL_SAMPLES.csv')
 keys=['H','fold','role','origin_raw_cycle_id','target_raw_cycle_id']
 joined=cycle.merge(authority[keys+['origin_calendar','target_calendar']],on=keys,validate='one_to_one')
 for r in joined.itertuples():
  older=[i for i,v in enumerate(b.bin_end_ophrs2_h) if v<=r.origin_available_h]
  o=older[-1] if older else -1;expected=False
  if o>=47:
   hist=b.iloc[o-47:o+1]
   expected=bool(hist.bin_valid_for_windows.eq(1).all() and hist.temporal_segment_id.nunique()==1 and np.isfinite(hist[features]).all(axis=None)
    and np.allclose(np.diff(hist.bin_end_ophrs2_h),.5,rtol=0,atol=1e-6) and pd.Timestamp(b.last_datetime.iloc[o])<=pd.Timestamp(r.origin_calendar))
  assert o==r.last_N1_history_row and expected==r.valid_48step_dynamic_history
 purge=[]
 for (hh,ff),g in joined.groupby(['H','fold']):
  g=g.loc[g.valid_48step_dynamic_history];first=pd.to_datetime(g.loc[g.role.eq('validation'),'origin_calendar']).min()
  tr=g.loc[g.role.eq('train')];bad=tr.loc[pd.to_datetime(tr.target_calendar)>=first]
  purge.extend((r.H,r.fold,r.origin_raw_cycle_id,r.target_raw_cycle_id) for r in bad.itertuples())
 saved=rd(P/'outputs/FUTURE_STRICT_TIME_PURGE_RECORDS.csv')
 assert set(purge)==set((r.H,r.fold,r.origin_raw_cycle_id,r.target_raw_cycle_id) for r in saved.itertuples())
 spots=raw_checks(raw)
 pd.DataFrame(spots).to_csv(P/'outputs/INDEPENDENT_RAW_SPOT_CHECKS.csv',index=False,float_format='%.17g',lineterminator='\n')
 assert all(sha(ROOT/r.path)==r.sha256 for r in sources.itertuples())
 result=dict(status='PASS',historical_sources_unchanged=len(sources),all_candidate_dwell_count=len(raw),all_QC_gates_recomputed=len(raw),
  historical_dwell_numeric_replays=len(h),historical_dwell_max_abs_difference_V=maxd,frozen_141cycle_max_abs_difference_V=float(abs(diff).max()),
  exact_dwell_to_bin_assignments=len(land),window_label_count_checks=len(windows),overlapping_query_label_pairs_checked=pairs,
  raw_independent_spot_checks=len(spots),raw_spot_check_scope='First/middle/last QC complete dwell of each of3current states; csv+datetime+statistics; other states counts and gates replayed from summaries',
  complete_cycle_points=len(cyc),dynamic_history_qualification_replays=len(joined),strict_calendar_purges_recomputed=len(purge),
  independent_life_trajectories=1,no_dense_native_fixed_target=True,
  no_new_training=True,no_GPU=True,no_Test_numeric_values=True)
 (P/'outputs/INDEPENDENT_VERIFICATION.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':verify()

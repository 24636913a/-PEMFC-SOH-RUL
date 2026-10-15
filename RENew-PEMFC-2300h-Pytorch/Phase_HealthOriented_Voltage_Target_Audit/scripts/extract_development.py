"""One bounded scan of certified Development raw prefixes, no Test feature parsing."""
from common import *
import sys,time

def stable_window(start,duration):return pd.Timestamp(start)+pd.Timedelta(seconds=.25*duration),pd.Timestamp(start)+pd.Timedelta(seconds=.75*duration)
def require_Development_age(age):
 if age.isna().any() or (age>=BOUNDARY).any():raise PermissionError('Raw Development prefix certificate mismatch: no Test feature read allowed')

def main():
 assert not (P/'outputs/RAW_EXTRACTION_COMPLETE.json').exists(),'One-shot raw extraction already completed'
 cfg=json.loads((P/'configs/FROZEN_AUDIT.json').read_text());assert sha(P/'configs/FROZEN_AUDIT.json')==(P/'configs/FROZEN_AUDIT.sha256').read_text().strip()
 ledger=rd(P/'data_contracts/SOURCE_SHA256.csv');assert all(sha(ROOT/r.path)==r.sha256 for r in ledger.itertuples())
 s=rd(P/'data_contracts/ALL_LONG_FIXED_STATE_RUNS.csv').sort_values('start_datetime').reset_index(drop=True)
 starts=pd.to_datetime(s.start_datetime)+pd.to_timedelta(s.duration_wall_s*.25,unit='s')
 ends=pd.to_datetime(s.start_datetime)+pd.to_timedelta(s.duration_wall_s*.75,unit='s')
 ss=starts.to_numpy(dtype='datetime64[ns]').astype('int64');ee=ends.to_numpy(dtype='datetime64[ns]').astype('int64');assert (ss[1:]>ee[:-1]).all()
 store={};files=rd(RCR/'outputs/RAW_TIMELINE_AUDIT.csv');files=files.loc[files.development_rows_retained.gt(0)]
 names=list(CHANNELS);rawcols=list(dict.fromkeys(['DateTime','OpHrs2 [h]','operation_point','I.S.set [A]']+ACTIVE+[v for v in CHANNELS.values() if v!='ACTIVE10_MEAN']))
 receipt=[];begin=time.time()
 for f in files.itertuples():
  path=RAW/f.source_file;n=int(f.development_rows_retained)
  print(f'File {f.file_order}/14 identity-only prefix verification: {n} Development records',flush=True)
  # Derive trust from the frozen file-specific count, then verify before numeric feature parsing.
  age=number(pd.read_csv(path,sep=';',encoding='utf-8-sig',nrows=n,usecols=['OpHrs2 [h]'],dtype='string',low_memory=False)['OpHrs2 [h]']);require_Development_age(age)
  assert len(age)==n and abs(age.iloc[-1]-f.development_age_end_h)<1e-8
  count=0;selected_count=0
  for chunk in pd.read_csv(path,sep=';',encoding='utf-8-sig',nrows=n,usecols=rawcols,chunksize=50000,low_memory=False):
   age=number(chunk['OpHrs2 [h]']);require_Development_age(age);dt=pd.to_datetime(chunk.DateTime,errors='raise');ns=dt.to_numpy(dtype='datetime64[ns]').astype('int64')
   # Original PAA creates middle50 and last25 jointly. Its side='right' membership
   # assigns the shared75% boundary to last25; primary is therefore [25%,75%).
   index=np.searchsorted(ss,ns,side='right')-1;valid=index>=0;pos=np.flatnonzero(valid);valid[pos]&=ns[pos]<ee[index[pos]]
   count+=len(chunk)
   if not valid.any():continue
   q=chunk.loc[valid].copy();which=index[valid];times=ns[valid];ages=age.loc[valid].to_numpy()
   observed_op=number(q.operation_point).to_numpy();expected_op=s.operation_point.to_numpy()[which]
   observed_set=number(q['I.S.set [A]']).to_numpy();expected_set=s.median_set_current_A.to_numpy()[which]
   assert np.array_equal(observed_op,expected_op) and np.allclose(observed_set,expected_set,rtol=0,atol=.0005)
   assert (times%1000==0).all(),'Raw timestamps must preserve microsecond precision'
   a=np.column_stack([times//1000,ages]+[np.mean(np.column_stack([number(q[c]).to_numpy() for c in ACTIVE]),axis=1) if key=='V' else number(q[CHANNELS[key]]).to_numpy() for key in names])
   for k in np.unique(which):store.setdefault(int(k),[]).append(a[which==k])
   selected_count+=len(q)
  assert count==n
  # Bounded bytes, never hash the full mixed final file. Raw CSV has one record per line.
  digest=hashlib.sha256()
  with path.open('rb') as stream:
   for _ in range(n+1):
    line=stream.readline();assert line;digest.update(line)
  receipt.append(dict(source_file=f.source_file,certified_prefix_records=n,metadata_records_verified=n,numeric_Development_records=count,
   selected_nonoverlap_middle50_records=selected_count,min_h=float(f.development_age_start_h),max_h=float(f.development_age_end_h),
   certified_source_prefix_SHA256=digest.hexdigest(),full_file_hash_allowed=n==int(f.raw_rows_seen),Test_features_parsed=False))
  cs('outputs/RAW_SOURCE_PREFIX_PROGRESS.csv',pd.DataFrame(receipt));print(f'Completed file {f.file_order}, selected {selected_count}, elapsed {time.time()-begin:.1f}s',flush=True)
 records=[];pointdig=hashlib.sha256()
 for k,r in enumerate(s.itertuples()):
  row=r._asdict();row.update(window_start=str(starts.iloc[k]),window_end=str(ends.iloc[k]),sampling_window='middle_50')
  arrays=store.get(k,[])
  if not arrays:row.update(selected_sample_count=0,eligible_electrical_stability=False,exclusion='No raw samples');records.append(row);continue
  a=np.concatenate(arrays);order=np.argsort(a[:,0]);a=a[order];pointdig.update(a.tobytes());times=(a[:,0]-a[0,0])/1e6
  row.update(selected_sample_count=len(a),measurement_first_calendar=str(pd.Timestamp(int(a[0,0])*1000)),measurement_last_calendar=str(pd.Timestamp(int(a[-1,0])*1000)),
   measurement_first_h=float(a[0,1]),measurement_last_h=float(a[-1,1]),available_calendar=r.end_datetime,available_h=r.end_ophrs2_h,
   sample_interval_median_s=float(np.median(np.diff(times))) if len(times)>1 else np.nan,
   sample_interval_max_s=float(np.max(np.diff(times))) if len(times)>1 else np.nan)
  for ci,key in enumerate(names):
   z=a[:,ci+2];row[key+'_median']=float(np.nanmedian(z));row[key+'_mean']=float(np.nanmean(z));row[key+'_std']=float(np.nanstd(z));row[key+'_span']=float(np.nanmax(z)-np.nanmin(z))
  v=a[:,2+names.index('V')];dt=times-times.mean();slope=np.dot(dt,v-v.mean())/np.dot(dt,dt) if np.dot(dt,dt)>0 else np.nan
  row['V_drift_mV_per_min']=float(60000*slope)
  finite=np.isfinite(a).all();qc=cfg['electrical_stability_diagnostic'];reasons=[]
  if not finite:reasons.append('Missing raw channels')
  if len(a)<qc['minimum_samples']:reasons.append('Too few stable-window points')
  if row['j_std']>qc['j_std_max_A_cm2']:reasons.append('Current variability')
  if abs(row['V_drift_mV_per_min'])>qc['absolute_V_OLS_drift_max_mV_per_min']:reasons.append('Voltage drift')
  if max(row[t+'_span'] for t in ['T_anode_C','T_cathode_C','T_coolant_C'])>qc['max_T_span_C']:reasons.append('Temperature drift')
  row['eligible_electrical_stability']=not reasons;row['exclusion']=';'.join(reasons);records.append(row)
 out=pd.DataFrame(records);cs('outputs/RAW_FIXED_STATE_DWELL_MEASUREMENTS.csv',out)
 js('outputs/RAW_EXTRACTION_COMPLETE.json',dict(status='COMPLETE',candidate_dwells=len(out),source_files=len(receipt),
  Development_raw_rows=sum(r['numeric_Development_records'] for r in receipt),nonoverlap_middle50_points=sum(r['selected_nonoverlap_middle50_records'] for r in receipt),
  normalized_selected_point_array_SHA256=pointdig.hexdigest(),Test_numeric_feature_rows=0,raw_files_after_Development_opened=False,
  bounded_mixed_file_prefix=True,elapsed_seconds=time.time()-begin,training=False,GPU=False))
 print('RAW COMPLETE',len(out),'dwells',flush=True)
if __name__=='__main__':main()

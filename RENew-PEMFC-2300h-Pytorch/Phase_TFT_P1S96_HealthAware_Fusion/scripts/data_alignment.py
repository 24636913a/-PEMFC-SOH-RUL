"""CPU-only bounded Development alignment, conservative P1S availability."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import yaml
from frozen_data_functions import Windows,build_windows,CommonScaler,ewma
P=Path(__file__).resolve().parents[1];PROJECT=P.parent
N1=PROJECT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
IDENTITY=PROJECT/'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv'
H_PROTOCOL=PROJECT/'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml'
CATALOG=PROJECT/'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv'
VREF=.71383023635;REFERENCE_AVAILABLE=171.82091
HEALTH_NAMES=['latest_R','mean_R','std_R','slope_R_per_actual_h','latest_minus_median_R']
def ah(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes());return h.hexdigest()
def json_write(rel,data):
 path=(P/rel).resolve();assert path.is_relative_to(P);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.writing');tmp.write_bytes((json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode());tmp.replace(path)
def csv_write(rel,data):data.to_csv(P/rel,index=False,float_format='%.17g',lineterminator='\n')
def health_at(p1s,t):
 if not np.isfinite(t):return None,dict(reason='TIME_IDENTITY_AMBIGUOUS',eligible=False,count=0,span_h=0.,raw_cycle_ids=[],latest_age_h=None)
 if t<REFERENCE_AVAILABLE:return None,dict(reason='REFERENCE_NOT_YET_AVAILABLE',eligible=False,count=0,span_h=0.,raw_cycle_ids=[],latest_age_h=None)
 q=p1s[(p1s.measurement_end_h>=t-96-1e-9)&(p1s.measurement_end_h<=t+1e-9)&(p1s.conservative_available_end_h<=t+1e-9)].sort_values('measurement_end_h')
 assert not q.raw_cycle_id.duplicated().any()
 x=q.measurement_end_h.to_numpy(float);y=q.R.to_numpy(float);n=len(q);span=float(np.ptp(x)) if n else 0.
 reason='COUNT_BELOW_5' if n<5 else 'SPAN_BELOW_67_2_H' if span<67.2-1e-9 else 'OK'
 info=dict(reason=reason,eligible=reason=='OK',count=n,span_h=span,raw_cycle_ids=q.raw_cycle_id.astype(int).tolist(),
  cycle_interval_ids=q.cycle_interval_id.tolist(),measurement_end_hours=x.tolist(),conservative_available_hours=q.conservative_available_end_h.tolist(),
  latest_age_h=float(t-x[-1]) if n else None)
 if reason!='OK':return None,info
 xc=x-x.mean();features=np.array([y[-1],y.mean(),y.std(ddof=0),xc@(y-y.mean())/(xc@xc),y[-1]-np.median(y)],dtype=np.float64)
 assert np.isfinite(features).all();return features,info
def prep(x,scaler,mask):
 z=scaler.transform(ewma(x));z[:,:,[i for i in range(14) if i not in mask]]=0.;return z
def qualify():
 cfg=yaml.safe_load(H_PROTOCOL.read_text(encoding='utf-8'))
 prefix=pd.read_csv(N1,nrows=3007);assert len(prefix)==3007
 dev=prefix[prefix.bin_end_ophrs2_h<1819.997738].reset_index(drop=True);assert len(dev)==3005
 train=build_windows(dev,cfg,0,2103);val=build_windows(dev,cfg,2103,2404)
 assert (len(train.origins),len(val.origins))==(1380,205)
 assert not set(train.targets.ravel())&set(val.targets.ravel())
 p1s=pd.read_csv(IDENTITY,float_precision='round_trip');cat=pd.read_csv(CATALOG).set_index('raw_cycle_id')
 assert len(p1s)==131 and not p1s.raw_cycle_id.duplicated().any()
 assert p1s.raw_cycle_id.min()>10 and p1s.causal_target_eligible.all()
 assert p1s.conservative_available_end_h.max()<1819.997738
 assert (p1s.conservative_available_end_h>=p1s.measurement_end_h).all()
 assert np.max(np.abs(p1s.R*VREF-p1s.P1S_voltage_reconstructed_V))<1e-12
 assert cat.loc[p1s.raw_cycle_id,'complete_cycle'].all()
 scaler=CommonScaler().fit(train.x)
 authority=json.loads((PROJECT/'Phase_TFT_FullFeature_CausalFilter_Comparison/outputs/E_COMMON_SCALER.json').read_text())
 assert np.array_equal(scaler.mean,np.array(authority['mean'])) and np.array_equal(scaler.scale,np.array(authority['scale']))
 raw_authority=json.loads((PROJECT/'Phase_F_TFT_DualPath_RegimeFusion/outputs/audits/F_DATA_ACCESS_AUDIT.json').read_text())
 tx0=prep(train.x,scaler,cfg['preprocessing']['Top10_indices']);vx0=prep(val.x,scaler,cfg['preprocessing']['Top10_indices'])
 hashes={'train_raw_x_sha256':ah(train.x),'train_preprocessed_x_sha256':ah(tx0),'validation_x_sha256':ah(vx0),'train_y_sha256':ah(train.y),'scaler_sha256':ah(np.stack([scaler.mean,scaler.scale]))}
 for k,v in hashes.items():assert raw_authority[k]==v,k
 rows=[];details=[];selected={};contexts={};summary={}
 for role,w in [('train',train),('validation',val)]:
  indices=[];health=[]
  for i,(origin,t) in enumerate(zip(w.origins,w.ages)):
   assert t==float(dev.iloc[origin].bin_end_ophrs2_h)
   values,info=health_at(p1s,float(t));details.append(dict(role=role,origin_row=int(origin),origin_time_h=float(t),**info))
   row=dict(role=role,original_index=i,origin_row=int(origin),origin_time_h=float(t),eligible=info['eligible'],reason=info['reason'],
    observations=info['count'],span_h=info['span_h'],latest_P1S_age_h=info['latest_age_h'],first_target_row=int(w.targets[i,0]),last_target_row=int(w.targets[i,-1]),
    first_target_time_h=float(dev.iloc[w.targets[i,0]].bin_end_ophrs2_h),last_target_time_h=float(dev.iloc[w.targets[i,-1]].bin_end_ophrs2_h),
    raw_cycle_ids='|'.join(map(str,info['raw_cycle_ids'])))
   rows.append(row)
   if info['eligible']:indices.append(i);health.append(values)
  index=np.array(indices,dtype=int);selected[role]=index;contexts[role]=np.array(health,dtype=float).reshape(-1,5)
  summary[role]=dict(original_n=len(w.origins),eligible_n=len(index),excluded_n=len(w.origins)-len(index),
   reason_counts={str(k):int(v) for k,v in pd.Series([a['reason'] for a in rows if a['role']==role]).value_counts().items()},
   origin_min_h=float(w.ages[index].min()) if len(index) else None,origin_max_h=float(w.ages[index].max()) if len(index) else None,
   unique_target_rows=len(np.unique(w.targets[index])),target_min_h=float(dev.iloc[w.targets[index].min()].bin_end_ophrs2_h) if len(index) else None,
   target_max_h=float(dev.iloc[w.targets[index].max()].bin_end_ophrs2_h) if len(index) else None)
 tr,va=selected['train'],selected['validation']
 time_ok=bool(len(tr) and len(va) and dev.iloc[train.targets[tr].max()].bin_end_ophrs2_h<=val.ages[va].min()+1e-9)
 evaluable=len(tr)>=100 and len(va)>=30 and summary['validation']['unique_target_rows']>=24 and time_ok
 summary.update(status='CPU_QUALIFIED' if evaluable else 'HEALTH_CONTEXT_NOT_EVALUABLE',minimum_conditions=dict(train=100,validation=30,validation_unique_targets=24),
  time_isolation_pass=time_ok,development_rows=3005,read_nrows=3007,Assessment_windows_constructed=False,Test_accessed=False,full_N1_hash=False,
  raw_input_fingerprints_reproduce_history=hashes,raw_scaler_scope='Original full 1380 unfiltered legal Train windows, unchanged',reference_available_h=REFERENCE_AVAILABLE,
  prefix_fingerprint_sha256=hashlib.sha256(dev.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()).hexdigest())
 csv_write('inputs/P1S_TFT_ALIGNMENT.csv',pd.DataFrame(rows));json_write('inputs/P1S_HISTORY_TIME_IDENTITIES.json',details)
 json_write('outputs/CPU_QUALIFICATION.json',summary)
 csv_write('inputs/LEGAL_ORIGINS.csv',pd.DataFrame(rows).query('eligible'));csv_write('inputs/EXCLUDED_ORIGINS.csv',pd.DataFrame(rows).query('not eligible'))
 csv_write('inputs/P1S_DEVELOPMENT_IDENTITY.csv',p1s)
 if evaluable:
  htrain=contexts['train'];mu=htrain.mean(axis=0);sd=htrain.std(axis=0);sd=np.where(sd>0,sd,1.)
  json_write('inputs/HEALTH_TRAIN_SCALER.json',dict(mean=mu.tolist(),scale=sd.tolist(),feature_names=HEALTH_NAMES,fit_scope='matched Train only',train_n=len(tr),ddof=0))
  z=dict(train_x=tx0[tr],validation_x=vx0[va],train_y=train.y[tr],validation_y=val.y[va],
   train_health=((htrain-mu)/sd).astype(np.float32),validation_health=((contexts['validation']-mu)/sd).astype(np.float32),
   train_health_raw=htrain,validation_health_raw=contexts['validation'],train_origins=train.origins[tr],validation_origins=val.origins[va],
   train_targets=train.targets[tr],validation_targets=val.targets[va],train_ages=train.ages[tr],validation_ages=val.ages[va],
   validation_target_times=dev.bin_end_ophrs2_h.to_numpy()[val.targets[va]],target_scale_V=np.array(scaler.target_scale))
  np.savez_compressed(P/'inputs/MATCHED_INPUTS.npz',**z)
  json_write('inputs/INPUT_FINGERPRINTS.json',{k:ah(a) for k,a in z.items()})
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':qualify()

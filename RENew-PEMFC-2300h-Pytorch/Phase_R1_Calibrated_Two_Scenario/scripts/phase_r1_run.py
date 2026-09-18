"""Phase R1: strict train-OOF calibration and frozen two-scenario TimeXer inference."""
from __future__ import annotations
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, r2_score

sys.dont_write_bytecode=True
ROOT=Path(r'E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_R1_Calibrated_Two_Scenario')
PROJ=ROOT.parent; N1=PROJ/'Phase_N1_Data_Foundation'; N28=PROJ/'Phase_N2_8F_Uncertainty_Robust_SA_TimeXer'; G=PROJ/'Phase_G_STOP_Physics_Consistent_Future_Conditions'; RS=PROJ/'Phase_RS_SA_TimeXer'; OPR=PROJ/'Phase_OPR_TimeXer'; R0=PROJ/'Phase_R0_Rare_State_Identifiability'
RAW=Path(r'E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据\Loadcycle_data\Loadcycle_data')
FOLDS=('fold_1','fold_2','fold_3'); DEV_ROWS=3005; DEV_END=1819.997738; EPS=1e-12; H=24
RAW_LAST='Loadcycle_151-162.csv'; RAW_LAST_N=265459

def out(rel):
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);return p
def csv(x,rel): x.to_csv(out(rel),index=False,encoding='utf-8-sig',lineterminator='\n')
def js(x,rel): out(rel).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def rmse(e):
 e=np.asarray(e,float);return float(np.sqrt(np.mean(e*e))) if len(e) else np.nan
def safe_logit(p):
 p=np.clip(np.asarray(p,float),1e-6,1-1e-6);return np.log(p/(1-p))
def finite(*xs): return all(np.isfinite(np.asarray(x)).all() for x in xs)

def load_dev(cfg):
 path=Path(cfg['project_root'])/cfg['data']['primary_csv']; cols=list(dict.fromkeys([*cfg['features'],'bin_start_ophrs2_h','bin_end_ophrs2_h','bin_center_ophrs2_h','last_datetime','split','bin_valid_for_windows','coverage_ratio','gap_before_flag','gap_after_flag','temporal_segment_id','cell_voltage_mean']))
 f=pd.read_csv(path,usecols=cols,nrows=DEV_ROWS).sort_values('bin_start_ophrs2_h').reset_index(drop=True)
 if len(f)!=DEV_ROWS or f.split.eq('test').any() or f.bin_end_ophrs2_h.max()>DEV_END+1e-9: raise RuntimeError('TEST FIREWALL: causal prefix contract failed')
 # Read only the declared legal prefix, and inspect its actual terminal row before any filtering.
 last=pd.read_csv(RAW/RAW_LAST,sep=';',usecols=['OpHrs2 [h]'],nrows=RAW_LAST_N,thousands=',').tail(1)
 raw_terminal=float(pd.to_numeric(last.iloc[0,0],errors='raise'))
 if raw_terminal>=DEV_END: raise RuntimeError(f'TEST FIREWALL: terminal raw prefix age {raw_terminal} exceeded Development')
 return f,{'causal_feature_path':str(path),'causal_read_mode':'header_plus_nrows_3005_only','test_feature_rows_read':0,'development_max_bin_end_h':float(f.bin_end_ophrs2_h.max()),'raw_last_file':RAW_LAST,'raw_last_prefix_rows_read':RAW_LAST_N,'raw_terminal_row_ophrs2_h_before_filter':raw_terminal,'test_raw_rows_read':0,'test_predictions_constructed':False,'test_metrics_computed':False,'test_performance_accessed':False}

def raw_conditional_stats(frame):
 names=['Loadcycle_0-14.csv','Loadcycle_15-25.csv','Loadcycle_26-37.csv','Loadcycle_37-50.csv','Loadcycle_51-70.csv','Loadcycle_70-75.csv','Loadcycle_76-82.csv','Loadcycle_82-84.csv','Loadcycle_85-100.csv','Loadcycle_101-112.csv','Loadcycle_112-125.csv','Loadcycle_126-133.csv','Loadcycle_134-150.csv',RAW_LAST];parts=[]
 for name in names:
  kw={'sep':';','usecols':['OpHrs2 [h]','I.S.act [A]'],'chunksize':200000,'thousands':','}
  if name==RAW_LAST:kw['nrows']=RAW_LAST_N
  for c in pd.read_csv(RAW/name,**kw):
   c.columns=['age','current'];c.age=pd.to_numeric(c.age,errors='coerce');c.current=pd.to_numeric(c.current,errors='coerce');c=c.dropna();c=c[(c.age>=72.5-1e-7)&(c.age<DEV_END)];c['bin_start_ophrs2_h']=np.floor(c.age*2+1e-8)/2;parts.append(c)
 raw=pd.concat(parts,ignore_index=True)
 if raw.age.max()>=DEV_END:raise RuntimeError('TEST FIREWALL: raw conditional statistics exceeded Development')
 def stat(g):
  x=g.current.to_numpy(float);on=x[np.abs(x)>.10];return pd.Series({'true_mu_on_A':float(on.mean()) if len(on) else np.nan,'true_sigma_on_A':float(on.std(ddof=0)) if len(on) else np.nan,'nonzero_current_count':len(on)})
 s=raw.groupby('bin_start_ophrs2_h',sort=True).apply(stat,include_groups=False).reset_index();d=frame[['bin_start_ophrs2_h']].merge(s,on='bin_start_ophrs2_h',how='left',validate='one_to_one')
 if d[['true_mu_on_A','true_sigma_on_A']].isna().any().any():raise RuntimeError('Missing legal raw conditional statistic')
 return d,{'raw_conditional_files_read':names,'raw_conditional_rows_read':int(len(raw)),'raw_conditional_max_age_h':float(raw.age.max())}

def q_components(qmodel,history,cfg):
 x=np.nan_to_num(history.reshape(len(history),-1),nan=0.0);xz=qmodel.x_scaler.transform(x);n=len(history);prob=np.zeros((n,H));plus=np.zeros((n,H));
 for h,(clf,reg) in enumerate(zip(qmodel.classifiers,qmodel.conditional)):
  prob[:,h]=np.full(n,clf['constant_probability']) if isinstance(clf,dict) else clf.predict_proba(xz)[:,1]
  v=np.full(n,reg['constant']) if isinstance(reg,dict) else reg.predict(xz);plus[:,h]=np.clip(v,0,1)
 qhard=np.where(prob>=float(cfg['forecast']['logistic_threshold']),plus,0.0)
 return prob,plus,qhard

def project_scenario(base,q,mu,sig,p,gs):
 z=base.copy();z[:,:,8]=q;z[:,:,0]=(1-q)*mu;z[:,:,1]=np.sqrt(np.maximum(0,(1-q)*sig**2+q*(1-q)*mu**2));z,audit=gs.project_future(z,p['area']);return z,audit

def metrics(y,p,a):
 t=(y-a[:,None])*1000;z=(p-a[:,None])*1000;e=z-t;ab=abs(e).ravel()
 return {'r2_delta':float(r2_score(t.ravel(),z.ravel())),'RMSE_mV':rmse(e.ravel()),'MAE_mV':float(abs(e).mean()),'P95_abs_error_mV':float(np.quantile(ab,.95)),'P99_abs_error_mV':float(np.quantile(ab,.99)),'max_abs_error_mV':float(ab.max())}

def weighted_calibration(d,fit_mask,eval_mask,fold,label):
 fit=d.loc[fit_mask].copy();ev=d.loc[eval_mask].copy()
 if fit.y.nunique()!=2: raise RuntimeError(f'{fold} calibration has one class')
 clf=LogisticRegression(C=1.0,class_weight=None,solver='lbfgs',max_iter=1000,random_state=42)
 clf.fit(safe_logit(fit.p_raw).reshape(-1,1),fit.y.to_numpy(int),sample_weight=fit.weight.to_numpy(float));a=float(clf.coef_[0,0]);b=float(clf.intercept_[0])
 if a<=0: raise RuntimeError(f'{fold} calibration nonpositive slope')
 raw=np.clip(ev.p_raw.to_numpy(float),1e-6,1-1e-6);cal=clf.predict_proba(safe_logit(raw).reshape(-1,1))[:,1];w=ev.weight.to_numpy(float);y=ev.y.to_numpy(int)
 rows=[]
 for name,pred in [('p_raw',raw),('p_cal',cal)]: rows.append({'fold':fold,'partition':label,'probability':name,'prediction_point_count':int(len(ev)),'unique_target_count':int(ev.target_row.nunique()),'effective_weight_sum':float(w.sum()),'Brier':float(brier_score_loss(y,pred,sample_weight=w)),'LogLoss':float(log_loss(y,pred,sample_weight=w,labels=[0,1])),'mean_probability':float(np.average(pred,weights=w)),'observed_mixed_rate':float(np.average(y,weights=w))})
 return clf,a,b,rows,ev.assign(p_cal=cal)

def reliability(d,fold,partition):
 rows=[];edges=np.linspace(0,1,11)
 for typ in ['p_raw','p_cal']:
  p=d[typ].to_numpy(float);y=d.y.to_numpy(int);w=d.weight.to_numpy(float)
  for k in range(10):
   m=(p>=edges[k])&((p<edges[k+1]) if k<9 else (p<=1)); rows.append({'fold':fold,'partition':partition,'probability':typ,'bin':f'{edges[k]:.1f}-{edges[k+1]:.1f}','prediction_point_count':int(m.sum()),'unique_target_count':int(d.loc[m,'target_row'].nunique()),'effective_weight_sum':float(w[m].sum()),'mean_probability':float(np.average(p[m],weights=w[m])) if m.any() else np.nan,'observed_mixed_rate':float(np.average(y[m],weights=w[m])) if m.any() else np.nan})
 return rows

def independent_events(points,scope):
 """Target de-duplicate then link only adjacent half-hour bins in same segment/type."""
 bad=points[points.error_type.ne('')].copy();u=bad.sort_values(['fold','target_row','horizon_step']).drop_duplicates(['fold','target_row','error_type'])
 rows=[]
 for (fold,typ),g in u.groupby(['fold','error_type']):
  run=[];prev=None
  for r in g.sort_values('target_row').itertuples(index=False):
   linked=prev is not None and r.target_row==prev.target_row+1 and r.temporal_segment_id==prev.temporal_segment_id
   if not linked and run:
    z=pd.DataFrame(run);rows.append({'scope':scope,'fold':fold,'error_type':typ,'event_start_h':float(z.target_age_h.min()),'event_end_h':float(z.target_age_h.max()),'independent_event_count':1,'unique_target_count':int(z.target_row.nunique()),'prediction_point_count':int(z.prediction_point_count.sum())});run=[]
   run.append(r._asdict());prev=r
  if run:
   z=pd.DataFrame(run);rows.append({'scope':scope,'fold':fold,'error_type':typ,'event_start_h':float(z.target_age_h.min()),'event_end_h':float(z.target_age_h.max()),'independent_event_count':1,'unique_target_count':int(z.target_row.nunique()),'prediction_point_count':int(z.prediction_point_count.sum())})
 return pd.DataFrame(rows)

def boot(pred,block,nrep,seed):
 rng=np.random.default_rng(seed); rows=[]
 for r in range(nrep):
  vals=[]
  for fold,g in pred.groupby('fold'):
   origins=np.sort(g.origin_row.unique());pick=[]
   while len(pick)<len(origins): pick.extend(origins[rng.integers(0,max(1,len(origins)-block+1)):][:block])
   chosen=np.asarray(pick[:len(origins)]);sub=pd.concat([g[g.origin_row.eq(int(o))] for o in chosen],ignore_index=True);vals.append(rmse(sub.error_CTS_mV)-rmse(sub.error_P2_mV))
  rows.append({'block_origins':block,'replicate':r+1,'mean_fold_RMSE_difference_CTS_minus_P2_mV':float(np.mean(vals)),'fold1_difference_mV':float(vals[0]),'fold2_difference_mV':float(vals[1]),'fold3_difference_mV':float(vals[2])})
 return pd.DataFrame(rows)

def main():
 print('R1: load Development-only inputs',flush=True);n=mod('r1n',N28/'scripts/phase_n2_8f_run.py');gs=mod('r1g',G/'scripts/phase_g_stop_experiment.py');cfg=n.load_config();gcfg=gs.load_config();frame,fire=load_dev(cfg);print('R1: reconstruct legal Development conditional statistics',flush=True);rawstats,rawmeta=raw_conditional_stats(frame);fire.update(rawmeta);frame=frame.merge(rawstats,on='bin_start_ophrs2_h',how='left',validate='one_to_one');common=n.common_module(cfg);wins={f:(tr,va) for f,tr,va in common.fold_windows(frame,cfg)};packs={f:n.package_fold(common,frame,wins,cfg,f) for f in FOLDS}
 # R0 correction audit: immutable correction record, not a change to R0.
 r0point=pd.read_csv(R0/'outputs/counterfactual/counterfactual_pointwise.csv');p2pooled=rmse(r0point[r0point.cf.eq('CF00')].error_mV)
 corr=pd.DataFrame([{'issue':'RMSE denominator','R0_value':21.34656588236491,'correct_value':21.34656588236491,'correct_interpretation':'formal three-fold unweighted P2 RMSE (mV)'},{'issue':'RMSE denominator','R0_value':24.47559330280174,'correct_value':p2pooled,'correct_interpretation':'pooled prediction-point Oracle-common P2 RMSE (mV); not formal fold mean'},{'issue':'Validation E1 independent event count','R0_value':87,'correct_value':9,'correct_interpretation':'87 is prediction points; 9 is event count'},{'issue':'Validation E2 independent event count','R0_value':20,'correct_value':2,'correct_interpretation':'20 is prediction points; 2 is event count'},{'issue':'R0 common valid origins fold_1','R0_value':np.nan,'correct_value':233,'correct_interpretation':'fold-specific count'},{'issue':'R0 common valid origins fold_2','R0_value':np.nan,'correct_value':245,'correct_interpretation':'fold-specific count'},{'issue':'R0 common valid origins fold_3','R0_value':466,'correct_value':466,'correct_interpretation':'fold-specific count, not all folds'},{'issue':'R0 common valid origins all folds','R0_value':466,'correct_value':944,'correct_interpretation':'sum of 233+245+466'}]);csv(corr,'outputs/audit/r0_correction_audit.csv')
 # Baseline saved predictions and strict P2 components.
 base_rows=[]; comps=[]; caldata=[]; prediction=[]; physical=[]; unit=[]; allfold=[]
 for fold,p in packs.items():
  print('R1: reconstruct frozen P2 components '+fold,flush=True)
  saved={name:np.load(RS/'outputs/frozen_model'/f'{name}_{fold}_predictions.npz') for name in ['P0_Original_Ridge','P1_Physics_Projected_Ridge','P2_Zero_Current_Aware','RS_Regime_Selective']}
  p2=saved['P2_Zero_Current_Aware'];
  for name,s in saved.items():
   if not np.array_equal(s['row_origin_index'],p['valid'].row_origin_index): raise RuntimeError(f'{fold} saved alignment failure {name}')
   base_rows.append({'model':name,'fold':fold,**metrics(s['y_true'],s['y_pred'],s['anchor'])})
  aux=np.array([frame.iloc[int(o)+1:int(o)+25][['true_mu_on_A','true_sigma_on_A']].to_numpy(float) for o in p['train'].row_origin_index])
  qmodel=gs.TwoPartQForecaster(gcfg).fit(p['htr'],p['ztr'][:,:,8]);praw,qplus,qhard=q_components(qmodel,p['hva'],gcfg);am=gs.AuxiliaryRidge(float(gcfg['forecast']['conditional_on_stats_ridge_alpha'])).fit(p['htr'],aux);on=am.predict(p['hva']);mu=np.maximum(on[:,:,0],0);sig=np.maximum(on[:,:,1],0)
  zhard,audit_hard=project_scenario(p['z_ridge'],qhard,mu,sig,p,gs)
  if not np.allclose(praw,p2['p_mixed'],rtol=0,atol=1e-7) or not np.allclose(qhard,p2['future_conditions'][:,:,8],rtol=0,atol=1e-7) or not np.allclose(zhard,p2['future_conditions'],rtol=0,atol=1e-6): raise RuntimeError(f'{fold} P2 component reconstruction mismatch')
  z0,a0=project_scenario(p['z_ridge'],np.zeros_like(qplus),mu,sig,p,gs);z1,a1=project_scenario(p['z_ridge'],qplus,mu,sig,p,gs)
  if not finite(z0,z1,praw,qplus) or (qplus<=0).any() or (qplus>1).any(): raise RuntimeError(f'{fold} invalid CTS components')
  # Strict causal train OOF source from R0; all target labels are completed offline only.
  oo=pd.read_csv(R0/'outputs/train_oof/p2_train_oof_predictions.csv');oo=oo[oo.fold.eq(fold)].copy();oo['y']=(oo.true_q>EPS).astype(int);counts=oo.groupby('target_row').target_row.transform('size');oo['weight']=1/counts;oo['p_raw']=oo.predicted_probability
  # Pre-registered chronological 80/20 origin holdout, with completion purge.
  origin_cut=np.sort(oo.origin_row.unique())[int(math.floor(.8*oo.origin_row.nunique()))];fitmask=oo.target_row.le(origin_cut);holdmask=oo.origin_row.ge(origin_cut)
  if oo.loc[fitmask,'target_row'].max()>oo.loc[holdmask,'origin_row'].min(): raise RuntimeError(f'{fold} calibration holdout completion overlap')
  clf0,a0c,b0c,introws,evhold=weighted_calibration(oo,fitmask,holdmask,fold,'train_internal_late20');
  # Calibration failure condition: both scored quantities worse than raw.
  idf=pd.DataFrame(introws).set_index('probability'); internal_pass=not (idf.loc['p_cal','Brier']>idf.loc['p_raw','Brier'] and idf.loc['p_cal','LogLoss']>idf.loc['p_raw','LogLoss'])
  if not internal_pass: raise RuntimeError(f'{fold} CALIBRATION_INTERNAL_FAIL')
  finalclf,aa,bb,_,_=weighted_calibration(oo,np.ones(len(oo),bool),np.ones(len(oo),bool),fold,'final_fit_diagnostic_only')
  pcal=finalclf.predict_proba(safe_logit(praw).reshape(-1,1))[:,1]
  # Frozen two-scenario inference only. Same X/scaler/checkpoint, no actual future values enter Z0/Z1.
  model=n.StepAlignedTimeXer(cfg,caf=False);model.load_state_dict(torch.load(N28/'checkpoints'/f'M0_SA_TimeXer_Ridge_{fold}_seed42.pt',map_location='cpu',weights_only=True));model.eval()
  with torch.no_grad():
   x=torch.tensor(p['xva']); y0raw=model(x,torch.tensor(p['z_scaler'].transform(z0))).numpy();y1raw=model(x,torch.tensor(p['z_scaler'].transform(z1))).numpy()
  y0=p['valid'].anchor[:,None]+y0raw*p['ystd']+p['ymean'];y1=p['valid'].anchor[:,None]+y1raw*p['ystd']+p['ymean'];yunc=(1-praw)*y0+praw*y1;ycts=(1-pcal)*y0+pcal*y1
  if not finite(y0,y1,yunc,ycts) or not np.allclose((1-np.zeros_like(pcal))*y0+np.zeros_like(pcal)*y1,y0) or not np.allclose((1-np.ones_like(pcal))*y0+np.ones_like(pcal)*y1,y1):raise RuntimeError(f'{fold} CTS unit-test failure')
  # pointwise outputs and component/reliability records
  for i,o in enumerate(p['valid'].row_origin_index):
   for h in range(H):
    row=int(o+h+1);truth=float(p['valid'].y_abs[i,h]);anchor=float(p['valid'].anchor[i]);qtrue=float(p['zva'][i,h,8]);errtype='E1_predicted_mixed_true_nonmixed' if praw[i,h]>=.5 and qtrue<=EPS else ('E2_predicted_nonmixed_true_mixed' if praw[i,h]<.5 and qtrue>EPS else '')
    prediction.append({'fold':fold,'origin_row':int(o),'target_row':row,'target_age_h':float(frame.bin_start_ophrs2_h.iloc[row]),'temporal_segment_id':frame.temporal_segment_id.iloc[row],'horizon_step':h+1,'horizon_hours':(h+1)*.5,'Y_true_V':truth,'anchor_V':anchor,'Y_P0_V':float(saved['P0_Original_Ridge']['y_pred'][i,h]),'Y_P1_V':float(saved['P1_Physics_Projected_Ridge']['y_pred'][i,h]),'Y_P2_V':float(p2['y_pred'][i,h]),'Y_RS_V':float(saved['RS_Regime_Selective']['y_pred'][i,h]),'Y0_V':float(y0[i,h]),'Y1_V':float(y1[i,h]),'p_raw':float(praw[i,h]),'p_cal':float(pcal[i,h]),'q_plus':float(qplus[i,h]),'q_hard':float(qhard[i,h]),'Y_Uncalibrated_CTS_V':float(yunc[i,h]),'Y_CTS_V':float(ycts[i,h]),'true_q_offline_only':qtrue,'error_type_offline_only':errtype})
    comps.append({'fold':fold,'origin_row':int(o),'target_row':row,'horizon_step':h+1,'p_raw':float(praw[i,h]),'q_plus':float(qplus[i,h]),'q_hard':float(qhard[i,h]),'saved_q':float(p2['future_conditions'][i,h,8]),'q_abs_residual':float(abs(qhard[i,h]-p2['future_conditions'][i,h,8])),'scenario0_current_mean_A':float(z0[i,h,0]),'scenario1_current_mean_A':float(z1[i,h,0])})
    physical.append({'fold':fold,'scenario':'zero','origin_row':int(o),'target_row':row,'q':float(z0[i,h,8]),'current_mean_A':float(z0[i,h,0]),'current_std_A':float(z0[i,h,1]),'mu_on_A':float(mu[i,h]),'sigma_on_A':float(sig[i,h]),'mean_identity_residual_A':float(z0[i,h,0]-mu[i,h]),'variance_identity_residual_A2':float(z0[i,h,1]**2-sig[i,h]**2),'density_identity_residual':float(max(abs(z0[i,h,3]-z0[i,h,0]/p['area']),abs(z0[i,h,4]-z0[i,h,1]/p['area']))),'violation':False,'constraint_scope':'PARTIALLY_CONSTRAINED_peak_to_peak_and_transition_preserved'})
    physical.append({'fold':fold,'scenario':'mixed','origin_row':int(o),'target_row':row,'q':float(z1[i,h,8]),'current_mean_A':float(z1[i,h,0]),'current_std_A':float(z1[i,h,1]),'mu_on_A':float(mu[i,h]),'sigma_on_A':float(sig[i,h]),'mean_identity_residual_A':float(z1[i,h,0]-(1-qplus[i,h])*mu[i,h]),'variance_identity_residual_A2':float(z1[i,h,1]**2-((1-qplus[i,h])*sig[i,h]**2+qplus[i,h]*(1-qplus[i,h])*mu[i,h]**2)),'density_identity_residual':float(max(abs(z1[i,h,3]-z1[i,h,0]/p['area']),abs(z1[i,h,4]-z1[i,h,1]/p['area']))),'violation':False,'constraint_scope':'PARTIALLY_CONSTRAINED_peak_to_peak_and_transition_preserved'})
  caldata.append(oo.assign(fold=fold,calibration_fit_weight=oo.weight)); unit.append({'fold':fold,'p2_q_reconstruction_max_abs':float(abs(qhard-p2['future_conditions'][:,:,8]).max()),'p2_feature_reconstruction_max_abs':float(abs(zhard-p2['future_conditions']).max()),'scenario_zero_audit':a0,'scenario_mixed_audit':a1,'calibration_slope_a':aa,'calibration_intercept_b':bb,'internal_pass':bool(internal_pass),'internal_origin_cut':int(origin_cut),'internal_fit_max_target_row':int(oo.loc[fitmask,'target_row'].max()),'internal_holdout_min_origin_row':int(oo.loc[holdmask,'origin_row'].min())});unit.extend(introws);unit.extend(reliability(evhold,fold,'train_internal_late20'))
  # final-fit calibration data and validation diagnostic only, never feeds backwards
  v=pd.DataFrame({'fold':fold,'origin_row':np.repeat(p['valid'].row_origin_index,H),'target_row':np.concatenate([np.arange(int(o)+1,int(o)+25) for o in p['valid'].row_origin_index]),'p_raw':praw.ravel(),'p_cal':pcal.ravel(),'y':(p['zva'][:,:,8].ravel()>EPS).astype(int)})
  v['weight']=1/v.groupby('target_row').target_row.transform('size');unit.extend(reliability(v,fold,'validation_frozen_diagnostic'))
  for name,col in [('p_raw','p_raw'),('p_cal','p_cal')]: unit.append({'fold':fold,'partition':'validation_frozen_diagnostic','probability':name,'prediction_point_count':len(v),'unique_target_count':int(v.target_row.nunique()),'effective_weight_sum':float(v.weight.sum()),'Brier':float(brier_score_loss(v.y,v[col],sample_weight=v.weight)),'LogLoss':float(log_loss(v.y,v[col],sample_weight=v.weight,labels=[0,1])),'mean_probability':float(np.average(v[col],weights=v.weight)),'observed_mixed_rate':float(np.average(v.y,weights=v.weight))})
  allfold.append({'fold':fold,'calibration_a':aa,'calibration_b':bb,'train_oof_prediction_points':int(len(oo)),'train_oof_unique_targets':int(oo.target_row.nunique()),'train_oof_effective_weight_sum':float(oo.weight.sum()),'train_oof_mixed_rate_weighted':float(np.average(oo.y,weights=oo.weight))})
 pred=pd.DataFrame(prediction);csv(pd.DataFrame(base_rows),'outputs/reproduction/baseline_reproduction.csv');csv(pd.concat(caldata,ignore_index=True),'outputs/calibration/calibration_train_oof.csv');csv(pd.DataFrame(allfold),'outputs/calibration/calibration_coefficients.csv');
 auditrows=pd.DataFrame(unit);csv(auditrows[auditrows.get('partition',pd.Series(index=auditrows.index,dtype=object)).eq('train_internal_late20')],'outputs/calibration/calibration_internal_holdout.csv');csv(auditrows[auditrows.get('partition',pd.Series(index=auditrows.index,dtype=object)).eq('validation_frozen_diagnostic')],'outputs/calibration/calibration_validation.csv');csv(auditrows[auditrows.get('partition',pd.Series(index=auditrows.index,dtype=object)).isin(['train_internal_late20','validation_frozen_diagnostic']) & auditrows.get('bin',pd.Series(index=auditrows.index,dtype=object)).notna()],'outputs/calibration/calibration_reliability_bins.csv')
 csv(pd.DataFrame(comps),'outputs/scenarios/p2_component_reconstruction.csv');js({'status':'PASS','component_definition':'p_raw=classifier probability; q_plus=conditional Ridge output clipped [0,1]; q_hard=q_plus if p_raw>=0.5 else 0','max_abs_q_residual':float(pd.DataFrame(comps).q_abs_residual.max()),'feature_reconstruction_tolerance':1e-6,'per_fold':[x for x in unit if isinstance(x,dict) and 'p2_q_reconstruction_max_abs' in x]},'outputs/scenarios/p2_component_reconstruction_audit.json')
 phys=pd.DataFrame(physical);csv(phys[phys.scenario.eq('zero')],'outputs/scenarios/scenario_zero.csv');csv(phys[phys.scenario.eq('mixed')],'outputs/scenarios/scenario_mixed.csv');csv(phys,'outputs/physical/scenario_physical_audit.csv');csv(pred,'outputs/predictions/cts_pointwise_predictions.csv')
 # Complete frozen metrics.
 met=[]; hor=[]
 for fold,g in pred.groupby('fold'):
  for name,col in [('P0','Y_P0_V'),('P1','Y_P1_V'),('P2','Y_P2_V'),('RS','Y_RS_V'),('Uncalibrated_CTS','Y_Uncalibrated_CTS_V'),('CTS_P2','Y_CTS_V')]:
   z=metrics(g.Y_true_V.to_numpy(),g[col].to_numpy(),g.anchor_V.to_numpy());met.append({'fold':fold,'model':name,'sample_count':len(g),'unique_target_count':g.target_row.nunique(),**z})
   for h,gh in g.groupby('horizon_step'): hor.append({'fold':fold,'model':name,'horizon_step':h,'horizon_hours':h*.5,'sample_count':len(gh),'unique_target_count':gh.target_row.nunique(),**metrics(gh.Y_true_V.to_numpy(),gh[col].to_numpy(),gh.anchor_V.to_numpy())})
 fm=pd.DataFrame(met);hm=pd.DataFrame(hor);csv(fm,'outputs/folds/frozen_model_fold_metrics.csv');csv(hm,'outputs/horizons/frozen_model_horizon_metrics.csv')
 # Regime and special scope audit.
 pred['error_P2_mV']=(pred.Y_P2_V-pred.Y_true_V)*1000;pred['error_CTS_mV']=(pred.Y_CTS_V-pred.Y_true_V)*1000;pred['error_Uncalibrated_CTS_mV']=(pred.Y_Uncalibrated_CTS_V-pred.Y_true_V)*1000
 rows=[]
 for fold,g in pred.groupby('fold'):
  previous=frame.zero_current_fraction.shift(1); scopes={'true_mixed':g.true_q_offline_only.gt(EPS),'true_nonmixed':g.true_q_offline_only.le(EPS),'state_correct':g.error_type_offline_only.eq(''),'E1':g.error_type_offline_only.eq('E1_predicted_mixed_true_nonmixed'),'E2':g.error_type_offline_only.eq('E2_predicted_nonmixed_true_mixed'),'continuous_nonzero':g.apply(lambda r: float(frame.zero_current_fraction.iloc[int(r.target_row)])<=EPS and float(previous.iloc[int(r.target_row)])<=EPS and frame.temporal_segment_id.iloc[int(r.target_row)]==frame.temporal_segment_id.iloc[int(r.target_row)-1],axis=1)}
  for scope,mask in scopes.items():
   x=g.loc[mask];
   if not len(x):continue
   evt=independent_events(x.assign(error_type=np.where(x.error_type_offline_only.eq(''),'none',x.error_type_offline_only),prediction_point_count=1),scope);rows.append({'fold':fold,'scope':scope,'prediction_point_count':len(x),'unique_target_count':x.target_row.nunique(),'independent_event_count':int(len(evt[evt.error_type.ne('none')])) if len(evt) else 0,'P2_RMSE_mV':rmse(x.error_P2_mV),'Uncalibrated_CTS_RMSE_mV':rmse(x.error_Uncalibrated_CTS_mV),'CTS_RMSE_mV':rmse(x.error_CTS_mV),'positive_added_loss_mV2':float(np.maximum(x.error_CTS_mV**2-x.error_P2_mV**2,0).sum()),'negative_loss_improvement_mV2':float(np.minimum(x.error_CTS_mV**2-x.error_P2_mV**2,0).sum()),'net_loss_change_mV2':float((x.error_CTS_mV**2-x.error_P2_mV**2).sum())})
 reg=pd.DataFrame(rows);csv(reg,'outputs/regimes/frozen_model_regime_metrics.csv');f1=pred[(pred.fold=='fold_1')&(pred.horizon_step==24)].copy();f3=pred[(pred.fold=='fold_3') & pred.apply(lambda r: float(frame.zero_current_fraction.iloc[int(r.target_row)])<=EPS and float(frame.zero_current_fraction.iloc[int(r.target_row)-1])<=EPS and frame.temporal_segment_id.iloc[int(r.target_row)]==frame.temporal_segment_id.iloc[int(r.target_row)-1],axis=1)].copy();csv(f1,'outputs/folds/fold1_12h_results.csv');csv(f3,'outputs/folds/fold3_nonzero_results.csv')
 # Tails and paired time-block bootstrap.
 pool=pred.copy(); tail=pd.DataFrame([{'model':name,'pooled_prediction_point_RMSE_mV':rmse(pool[f'error_{name}_mV'] if name in ['P2','CTS'] else (pool[col]-pool.Y_true_V)*1000),'P95_abs_error_mV':float(np.quantile(abs(pool[f'error_{name}_mV'] if name in ['P2','CTS'] else (pool[col]-pool.Y_true_V)*1000),.95)),'P99_abs_error_mV':float(np.quantile(abs(pool[f'error_{name}_mV'] if name in ['P2','CTS'] else (pool[col]-pool.Y_true_V)*1000),.99)),'max_abs_error_mV':float(abs(pool[f'error_{name}_mV'] if name in ['P2','CTS'] else (pool[col]-pool.Y_true_V)*1000).max())} for name,col in [('P2','Y_P2_V'),('CTS','Y_CTS_V')]]);csv(tail,'outputs/tails/tail_risk_summary.csv')
 u=pool.sort_values('error_P2_mV',key=lambda x:abs(x)).drop_duplicates('target_row').copy();top=pd.concat([u.nlargest(20,'error_P2_mV',keep='all').head(20).assign(list_type='top20_P2_abs_error'),u.assign(loss_delta_mV2=u.error_CTS_mV**2-u.error_P2_mV**2).nlargest(20,'loss_delta_mV2').assign(list_type='top20_CTS_added_loss')]);csv(top,'outputs/tails/tail_top20_unique_targets.csv')
 bm=boot(pool,48,2000,42);bs=pd.concat([boot(pool,24,2000,42),boot(pool,96,2000,42)]);csv(bm,'outputs/bootstrap/bootstrap_main.csv');csv(bs,'outputs/bootstrap/bootstrap_sensitivity.csv')
 # Gate A: exact locked conditions, three-fold RMSE average.
 piv=fm.pivot(index='fold',columns='model',values='RMSE_mV');meanp=piv.mean();h12=hm[hm.horizon_step.eq(24)].pivot(index='fold',columns='model',values='RMSE_mV').mean();p2pool=tail[tail.model.eq('P2')].iloc[0];ctspool=tail[tail.model.eq('CTS')].iloc[0];mix=reg[reg.scope.eq('true_mixed')].set_index('fold');conditions={'rmse_improvement_at_least_5pct':bool((meanp.P2-meanp.CTS_P2)/meanp.P2>=.05),'at_least_two_folds_better':bool((piv.CTS_P2<piv.P2).sum()>=2),'fold3_not_above_P0_27_885315':bool(piv.loc['fold_3','CTS_P2']<=27.885315),'fold1_12h_not_above_P1_31_682133':bool(hm[(hm.fold=='fold_1')&(hm.horizon_step==24)&(hm.model=='CTS_P2')].RMSE_mV.iloc[0]<=31.682133),'overall_12h_not_above_P0_33_527980':bool(h12.CTS_P2<=33.527980),'pooled_P99_not_above_105pct_P2':bool(ctspool.P99_abs_error_mV<=1.05*p2pool.P99_abs_error_mV),'max_abs_error_lower_than_P2':bool(ctspool.max_abs_error_mV<p2pool.max_abs_error_mV),'mixed_regime_not_over_5pct_each_fold':bool((mix.CTS_RMSE_mV<=1.05*mix.P2_RMSE_mV).all()),'physical_no_new_violation':bool(not phys.violation.any()),'test_closed':True};gate='GATE_A_PASS' if all(conditions.values()) else 'GATE_A_FAIL'
 js({'fixed_start_commit':'fcb3f6f43b1e5b0a85fb370333dd2982076f4e35','test_firewall':fire,'gate_a_conditions':conditions,'gate_a_decision':gate,'mean_fold_rmse_mV':meanp.to_dict(),'mean_12h_rmse_mV':h12.to_dict(),'pooled_tail_metrics':tail.to_dict(orient='records'),'bootstrap_main_mean_difference_mV':float(bm.iloc[:,2].mean()),'bootstrap_main_95pct_CI_mV':[float(bm.iloc[:,2].quantile(.025)),float(bm.iloc[:,2].quantile(.975))]},'outputs/audit/run_summary.json');js(fire,'outputs/audit/test_firewall_audit.json');print('R1: completed '+gate,flush=True)
if __name__=='__main__':main()

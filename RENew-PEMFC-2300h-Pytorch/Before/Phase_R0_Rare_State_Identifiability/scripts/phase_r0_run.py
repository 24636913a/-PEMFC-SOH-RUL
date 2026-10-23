"""Phase R0. Strict development-only OOF and frozen-M0 counterfactual audit."""
from __future__ import annotations
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import brier_score_loss

sys.dont_write_bytecode=True
ROOT=Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_R0_Rare_State_Identifiability")
PROJ=ROOT.parent; N1=PROJ/'Phase_N1_Data_Foundation'; N28=PROJ/'Phase_N2_8F_Uncertainty_Robust_SA_TimeXer'; G=PROJ/'Phase_G_STOP_Physics_Consistent_Future_Conditions'; RS=PROJ/'Phase_RS_SA_TimeXer'; OPR=PROJ/'Phase_OPR_TimeXer'; TTD=PROJ/'Phase_TTD_TimeXer_Failure_Diagnosis'
DEV_ROWS=3005; DEV_END=1819.997738; FOLDS=('fold_1','fold_2','fold_3'); EPS=1e-12
RAW=Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据\Loadcycle_data\Loadcycle_data")

def out(rel):
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);return p
def csv(d,rel): d.to_csv(out(rel),index=False,encoding='utf-8-sig',lineterminator='\n')
def js(d,rel): out(rel).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def rmse(x): return float(np.sqrt(np.mean(np.square(np.asarray(x,float))))) if len(x) else np.nan

def load_dev(cfg):
 path=Path(cfg['project_root'])/cfg['data']['primary_csv']; cols=list(dict.fromkeys([*cfg['features'],'bin_start_ophrs2_h','bin_end_ophrs2_h','bin_center_ophrs2_h','last_datetime','split','bin_valid_for_windows','coverage_ratio','gap_before_flag','gap_after_flag','temporal_segment_id','cell_voltage_mean']))
 f=pd.read_csv(path,usecols=cols,nrows=DEV_ROWS)
 if len(f)!=DEV_ROWS or f['bin_end_ophrs2_h'].max()>DEV_END+1e-9 or f.split.eq('test').any(): raise RuntimeError('TEST FIREWALL: development prefix contract failed')
 return f.sort_values('bin_start_ophrs2_h').reset_index(drop=True),{'source_file':str(path),'read_mode':'header_plus_nrows_3005_only','test_rows_read':0,'development_max_bin_end_h':float(f.bin_end_ophrs2_h.max()),'test_accessed':False}

def raw_true(frame):
 # File order is the Phase N1 numeric chronology.  The last prefix length comes from development-only causal-bin sample accounting.
 names=['Loadcycle_0-14.csv','Loadcycle_15-25.csv','Loadcycle_26-37.csv','Loadcycle_37-50.csv','Loadcycle_51-70.csv','Loadcycle_70-75.csv','Loadcycle_76-82.csv','Loadcycle_82-84.csv','Loadcycle_85-100.csv','Loadcycle_101-112.csv','Loadcycle_112-125.csv','Loadcycle_126-133.csv','Loadcycle_134-150.csv','Loadcycle_151-162.csv']
 parts=[]; use=['OpHrs2 [h]','I.S.act [A]']; last='Loadcycle_151-162.csv'
 for name in names:
  kw={'usecols':use,'sep':';','chunksize':200000,'thousands':','}
  if name==last: kw['nrows']=265459
  for c in pd.read_csv(RAW/name,**kw):
   c.columns=['age','current']
   for col in ['age','current']: c[col]=pd.to_numeric(c[col],errors='coerce')
   c=c.dropna()
   c=c[(c.age>=72.5-1e-7)&(c.age<1820.0-1e-7)]
   c['bin_start_ophrs2_h']=np.floor(c.age*2+1e-8)/2
   parts.append(c)
 raw=pd.concat(parts,ignore_index=True)
 if raw.age.max()>=DEV_END or raw.age.min()<72.5-1e-6: raise RuntimeError('TEST FIREWALL: raw prefix exceeded development')
 def stat(g):
  x=g.current.to_numpy(float); on=x[np.abs(x)>.10]; q=float(np.mean(np.abs(x)<=.10)); mu=float(on.mean()) if len(on) else np.nan; sig=float(on.std(ddof=0)) if len(on) else np.nan
  return pd.Series({'true_q':q,'true_mu_on_A':mu,'true_sigma_on_A':sig,'nonzero_current_count':len(on),'raw_count':len(x),'reconstructed_current_mean_A':(1-q)*mu if len(on) else np.nan,'reconstructed_current_std_A':math.sqrt(max(0,(1-q)*sig**2+q*(1-q)*mu**2)) if len(on) else np.nan})
 s=raw.groupby('bin_start_ophrs2_h',sort=True).apply(stat,include_groups=False).reset_index()
 a=frame[['bin_start_ophrs2_h','actual_current_A_mean','actual_current_A_std','zero_current_fraction']].merge(s,on='bin_start_ophrs2_h',how='left',validate='one_to_one')
 a['mean_reconstruction_residual_A']=a.reconstructed_current_mean_A-a.actual_current_A_mean;a['std_reconstruction_residual_A']=a.reconstructed_current_std_A-a.actual_current_A_std;a['q_residual']=a.true_q-a.zero_current_fraction;a['oracle_estimable']=a[['true_mu_on_A','true_sigma_on_A']].notna().all(axis=1)
 missing=a.loc[a.true_q.isna(),'bin_start_ophrs2_h'].head(30).to_list()
 if a.bin_start_ophrs2_h.isna().any() or a.true_q.isna().any(): raise RuntimeError(f'Missing development raw conditional bin count={int(a.true_q.isna().sum())} starts={missing}')
 return a,{'raw_files_read':names,'last_file_max_rows':265459,'raw_max_age_h':float(raw.age.max()),'raw_rows':int(len(raw))}

def p2_oof(p,aux,gcfg,n,gs,fold):
 count=len(p['htr']); start=n.first_index_with_completed(p['train'].target_end_age_h,p['train'].origin_age_h,int(gcfg['forecast']['oof_min_completed_windows'])); q=np.full((count,24),np.nan);prob=q.copy();a=np.full((count,24,2),np.nan);audit=[]
 for b,(l,r) in enumerate(n.block_edges(start,count,int(gcfg['forecast']['oof_blocks'])),1):
  origin=float(p['train'].origin_age_h[l]);fit=n.completed_indices(p['train'].target_end_age_h,origin)
  if p['train'].target_end_age_h[fit].max()>origin+1e-7:raise RuntimeError('P2 OOF target leakage')
  qm=gs.TwoPartQForecaster(gcfg).fit(p['htr'][fit],p['ztr'][fit,:,8]);q[l:r],prob[l:r]=qm.predict(p['htr'][l:r]);am=gs.AuxiliaryRidge(float(gcfg['forecast']['conditional_on_stats_ridge_alpha'])).fit(p['htr'][fit],aux[fit]);a[l:r]=am.predict(p['htr'][l:r])
  audit.append({'fold':fold,'block':b,'fit_count':int(len(fit)),'max_fit_target_end_h':float(p['train'].target_end_age_h[fit].max()),'min_prediction_origin_h':origin,'causal_pass':True})
 cand=p['z_oof'].copy();cand[:,:,8]=q;mu=np.maximum(a[:,:,0],0);sg=np.maximum(a[:,:,1],0);cand[:,:,0]=(1-q)*mu;cand[:,:,1]=np.sqrt(np.maximum(0,(1-q)*sg**2+q*(1-q)*mu**2));z,_=gs.project_future(cand,p['area']);return z,q,prob,a,np.isfinite(z).all((1,2)),audit

def event_tables(points,label):
 e=[]
 for (fold,typ),g in points.groupby(['fold','error_type']):
  uq=g.sort_values(['target_row','horizon_step']).drop_duplicates(['target_row','error_type']).copy();uq['scope']=label;e.append(uq)
 unique=pd.concat(e,ignore_index=True) if e else points.iloc[:0].copy()
 def events(g,scope):
  res=[]
  for typ,x in g.groupby('error_type'):
   x=x.sort_values(['target_row','target_age_h']).drop_duplicates(['target_row','error_type']);run=[];prev=None
   for r in x.itertuples(index=False):
    linked=prev is not None and r.temporal_segment_id==prev.temporal_segment_id and r.target_row==prev.target_row+1
    if not linked and run:
     z=pd.DataFrame(run);res.append({'scope':scope,'error_type':typ,'event_start_h':z.target_age_h.min(),'event_end_h':z.target_age_h.max(),'duration_h':z.target_age_h.max()-z.target_age_h.min()+.5,'unique_target_timestamp_count':z.target_row.nunique(),'prediction_point_count':int(z.prediction_point_count.sum()),'origin_count':int(z.origin_count.sum()),'temporal_segment_id':z.temporal_segment_id.iloc[0]});run=[]
    run.append(r._asdict());prev=r
   if run:
    z=pd.DataFrame(run);res.append({'scope':scope,'error_type':typ,'event_start_h':z.target_age_h.min(),'event_end_h':z.target_age_h.max(),'duration_h':z.target_age_h.max()-z.target_age_h.min()+.5,'unique_target_timestamp_count':z.target_row.nunique(),'prediction_point_count':int(z.prediction_point_count.sum()),'origin_count':int(z.origin_count.sum()),'temporal_segment_id':z.temporal_segment_id.iloc[0]})
  return pd.DataFrame(res)
 uniq=unique.groupby(['fold','target_row','error_type','target_age_h','temporal_segment_id'],as_index=False).agg(prediction_point_count=('origin_row','size'),origin_count=('origin_row','nunique'),predicted_probability=('predicted_probability','mean'),true_q=('true_q','mean'),predicted_q=('predicted_q','mean'),current_abs_error_A=('current_abs_error_A','mean'))
 return unique,uniq,events(uniq,label),events(uniq.drop(columns='fold').drop_duplicates(['target_row','error_type']),label+'_global_dedup')

def main():
 print('R0: loading development prefix only',flush=True);n=mod('r0n',N28/'scripts/phase_n2_8f_run.py');gs=mod('r0g',G/'scripts/phase_g_stop_experiment.py');cfg=n.load_config();gcfg=gs.load_config();frame,fire=load_dev(cfg); common=n.common_module(cfg); windows={f:(tr,va) for f,tr,va in common.fold_windows(frame,cfg)}
 print('R0: deriving raw development-only condition statistics',flush=True);true,rawmeta=raw_true(frame);frame=frame.merge(true[['bin_start_ophrs2_h','true_q','true_mu_on_A','true_sigma_on_A','nonzero_current_count','oracle_estimable']],on='bin_start_ophrs2_h',how='left',validate='one_to_one');csv(true,'outputs/physical/true_conditional_current_audit.csv')
 packs={f:n.package_fold(common,frame,windows,cfg,f) for f in FOLDS};oof=[];trainpoints=[];valpoints=[];valz={};
 for fold,p in packs.items():
  print('R0: causal P2 OOF '+fold,flush=True); aux=np.array([frame.iloc[int(o)+1:int(o)+25][['true_mu_on_A','true_sigma_on_A']].to_numpy(float) for o in p['train'].row_origin_index]);z,q,pr,ast,keep,aud=p2_oof(p,aux,gcfg,n,gs,fold);oof+=aud
  for i in np.flatnonzero(keep):
   for h in range(24):
    row=int(p['train'].row_origin_index[i]+h+1);tq=float(p['ztr'][i,h,8]);typ='E1_predicted_mixed_true_nonmixed' if pr[i,h]>=.5 and tq<=EPS else ('E2_predicted_nonmixed_true_mixed' if pr[i,h]<.5 and tq>EPS else '')
    dim={}
    for j,feature in enumerate(cfg['future_exogenous_features']):
     dim[f'true_{feature}']=float(p['ztr'][i,h,j]);dim[f'predicted_{feature}']=float(z[i,h,j])
    trainpoints.append({'fold':fold,'origin_row':int(p['train'].row_origin_index[i]),'target_row':row,'target_age_h':float(frame.bin_start_ophrs2_h.iloc[row]),'temporal_segment_id':frame.temporal_segment_id.iloc[row],'horizon_step':h+1,'horizon_hours':(h+1)*.5,'true_q':tq,'predicted_q':float(q[i,h]),'predicted_probability':float(pr[i,h]),'true_current_mean_A':float(p['ztr'][i,h,0]),'predicted_current_mean_A':float(z[i,h,0]),'current_abs_error_A':float(abs(z[i,h,0]-p['ztr'][i,h,0])),'error_type':typ,**dim})
  saved=np.load(RS/'outputs/frozen_model'/f'P2_Zero_Current_Aware_{fold}_predictions.npz');
  if not np.array_equal(saved['row_origin_index'],p['valid'].row_origin_index):raise RuntimeError('RS validation alignment mismatch')
  valz[fold]=saved['future_conditions'].astype(float)
  for i,o in enumerate(p['valid'].row_origin_index):
   for h in range(24):
    row=int(o+h+1);tq=float(p['zva'][i,h,8]);prob=float(saved['p_mixed'][i,h]);typ='E1_predicted_mixed_true_nonmixed' if prob>=.5 and tq<=EPS else ('E2_predicted_nonmixed_true_mixed' if prob<.5 and tq>EPS else '')
    valpoints.append({'fold':fold,'origin_row':int(o),'target_row':row,'target_age_h':float(frame.bin_start_ophrs2_h.iloc[row]),'temporal_segment_id':frame.temporal_segment_id.iloc[row],'horizon_step':h+1,'horizon_hours':(h+1)*.5,'true_q':tq,'predicted_q':float(valz[fold][i,h,8]),'predicted_probability':prob,'true_current_mean_A':float(p['zva'][i,h,0]),'predicted_current_mean_A':float(valz[fold][i,h,0]),'current_abs_error_A':float(abs(valz[fold][i,h,0]-p['zva'][i,h,0])),'error_type':typ})
 train=pd.DataFrame(trainpoints);valid=pd.DataFrame(valpoints);csv(train[train.error_type.ne('')],'outputs/train_oof/train_oof_error_points.csv');csv(valid[valid.error_type.ne('')],'outputs/events/validation_error_points.csv');tu,tut,te,tglob=event_tables(train[train.error_type.ne('')],'train_oof');vu,vut,ve,vglob=event_tables(valid[valid.error_type.ne('')],'validation');csv(tut,'outputs/train_oof/train_oof_unique_targets.csv');csv(te,'outputs/train_oof/train_oof_independent_events.csv');csv(vut,'outputs/events/validation_unique_targets.csv');csv(ve,'outputs/events/validation_independent_events.csv');csv(pd.concat([tglob.assign(kind='train_global'),vglob.assign(kind='validation_global')]),'outputs/events/event_overlap_audit.csv');csv(pd.DataFrame(oof),'outputs/train_oof/p2_train_oof_audit.csv')
 # training OOF complete predictions and calibration
 csv(train[train.error_type.ne('')|train.error_type.eq('')],'outputs/train_oof/p2_train_oof_predictions.csv')
 js({'protocol':'strict causal completed-target-block OOF','folds':list(FOLDS),'horizon_steps':24,'forecast_horizon_h':12.0,'prediction_rows':int(len(train)),'oof_blocks':int(gcfg['forecast']['oof_blocks']),'minimum_completed_windows':int(gcfg['forecast']['oof_min_completed_windows']),'target_leakage_guard':'max fit target_end_age_h <= prediction origin_age_h'},'outputs/train_oof/p2_train_oof_reproduction.json')
 bins=np.linspace(0,1,11);cal=[]
 for fold,g in train.groupby('fold'):
  y=(g.true_q>EPS).astype(int);cal.append({'fold':fold,'bin':'overall','count':len(g),'mean_probability':g.predicted_probability.mean(),'observed_mixed_rate':y.mean(),'brier_score':brier_score_loss(y,g.predicted_probability)})
  for b in range(10):
   x=g[(g.predicted_probability>=bins[b])&((g.predicted_probability<bins[b+1]) if b<9 else (g.predicted_probability<=1))];cal.append({'fold':fold,'bin':f'{bins[b]:.1f}-{bins[b+1]:.1f}','count':len(x),'mean_probability':x.predicted_probability.mean() if len(x) else np.nan,'observed_mixed_rate':y.loc[x.index].mean() if len(x) else np.nan,'brier_score':np.nan})
 csv(pd.DataFrame(cal),'outputs/calibration/probability_calibration.csv')
 # frozen counterfactuals
 print('R0: frozen-M0 counterfactual inference',flush=True); records=[];inputaudit=[]
 for fold,p in packs.items():
  saved=np.load(RS/'outputs/frozen_model'/f'P2_Zero_Current_Aware_{fold}_predictions.npz');z00=saved['future_conditions'].astype(float);qpred=z00[:,:,8];den=1-qpred;mu=np.divide(z00[:,:,0],den,out=np.full_like(qpred,np.nan),where=den>1e-7);var=z00[:,:,1]**2;sig=np.sqrt(np.maximum(0,np.divide(var-qpred*den*mu**2,den,out=np.full_like(qpred,np.nan),where=den>1e-7)))
  tq=np.array([frame.iloc[int(o)+1:int(o)+25].true_q.to_numpy(float) for o in p['valid'].row_origin_index]);tm=np.array([frame.iloc[int(o)+1:int(o)+25].true_mu_on_A.to_numpy(float) for o in p['valid'].row_origin_index]);ts=np.array([frame.iloc[int(o)+1:int(o)+25].true_sigma_on_A.to_numpy(float) for o in p['valid'].row_origin_index]);est=np.array([frame.iloc[int(o)+1:int(o)+25].oracle_estimable.to_numpy(bool) for o in p['valid'].row_origin_index]);commonvalid=est.all(1)&np.isfinite(mu).all(1)&np.isfinite(sig).all(1)
  def build(q,m,s):
   z=z00.copy();z[:,:,8]=q;z[:,:,0]=(1-q)*m;z[:,:,1]=np.sqrt(np.maximum(0,(1-q)*s**2+q*(1-q)*m**2));z[:,:,3]=z[:,:,0]/p['area'];z[:,:,4]=z[:,:,1]/p['area'];return z
  cfs={'CF00':z00,'CF10':build(tq,mu,sig),'CF01':build(qpred,tm,ts),'CF11':build(tq,tm,ts)}
  model=n.StepAlignedTimeXer(cfg,caf=False);model.load_state_dict(torch.load(N28/'checkpoints'/f'M0_SA_TimeXer_Ridge_{fold}_seed42.pt',map_location='cpu',weights_only=True));model.eval()
  with torch.no_grad():
   x=torch.tensor(p['xva']);hh,b=model._historical_state_and_base(x);base=b.numpy();
   for name,z in cfs.items():
    zz=torch.tensor(p['z_scaler'].transform(z));gamma,beta=model.film(zz).chunk(2,dim=-1);corr=model.correction((1+gamma)*(hh.unsqueeze(1)+model.horizon_state)+beta).squeeze(-1).numpy(); pred=p['valid'].anchor[:,None]+(base+corr)*p['ystd']+p['ymean'];
    for i,o in enumerate(p['valid'].row_origin_index):
     for h in range(24): records.append({'fold':fold,'cf':name,'origin_row':int(o),'target_row':int(o+h+1),'target_age_h':float(frame.bin_start_ophrs2_h.iloc[int(o+h+1)]),'temporal_segment_id':frame.temporal_segment_id.iloc[int(o+h+1)],'horizon_step':h+1,'true_deltaV_mV':float((p['valid'].y_abs[i,h]-p['valid'].anchor[i])*1000),'prediction_deltaV_mV':float((pred[i,h]-p['valid'].anchor[i])*1000),'error_mV':float((pred[i,h]-p['valid'].y_abs[i,h])*1000),'B_mV':float(base[i,h]*p['ystd']*1000),'C_mV':float(corr[i,h]*p['ystd']*1000),'oracle_estimable':bool(commonvalid[i])})
    inputaudit.append({'fold':fold,'cf':name,'common_valid_origins':int(commonvalid.sum()),'CF00_equals_official_P2':bool(name!='CF00' or np.array_equal(z,z00)),'q_range_pass':bool(np.all((z[:,:,8]>=-1e-7)&(z[:,:,8]<=1+1e-7))),'current_nonnegative_pass':bool((z[:,:,0]>=-1e-7).all()),'std_nonnegative_pass':bool((z[:,:,1]>=-1e-7).all()),'density_identity_max_abs':float(np.max(abs(z[:,:,3]-z[:,:,0]/p['area']))),'untouched_peak_to_peak_max_abs_change':float(np.max(abs(z[:,:,2]-z00[:,:,2]))),'untouched_qp2p_max_abs_change':float(np.max(abs(z[:,:,5]-z00[:,:,5])))})
 point=pd.DataFrame(records);csv(point,'outputs/counterfactual/counterfactual_pointwise.csv');csv(pd.DataFrame(inputaudit),'outputs/counterfactual/counterfactual_input_audit.csv')
 base=point[point.cf.eq('CF00')].rename(columns={'error_mV':'e00','prediction_deltaV_mV':'y00','C_mV':'C00'});wide=base[['fold','origin_row','target_row','horizon_step','e00','y00','C00','oracle_estimable','temporal_segment_id','target_age_h']].copy()
 for cf in ['CF10','CF01','CF11']:
  q=point[point.cf.eq(cf)][['fold','origin_row','target_row','horizon_step','error_mV','prediction_deltaV_mV','C_mV']].rename(columns={'error_mV':f'e{cf[-2:]}','prediction_deltaV_mV':f'y{cf[-2:]}','C_mV':f'C{cf[-2:]}'})
  wide=wide.merge(q,on=['fold','origin_row','target_row','horizon_step'],validate='one_to_one')
 wide=wide[wide.oracle_estimable].copy();wide['L00']=wide.e00**2;wide['L10']=wide.e10**2;wide['L01']=wide.e01**2;wide['L11']=wide.e11**2;wide['Gain_q']=wide.L00-wide.L10;wide['Gain_cond']=wide.L00-wide.L01;wide['Gain_both']=wide.L00-wide.L11;wide['Interaction']=wide.L00-wide.L10-wide.L01+wide.L11;wide['Shapley_q']=.5*((wide.L00-wide.L10)+(wide.L01-wide.L11));wide['Shapley_cond']=.5*((wide.L00-wide.L01)+(wide.L10-wide.L11));csv(wide,'outputs/counterfactual/counterfactual_attribution.csv');csv(wide[['fold','origin_row','target_row','horizon_step','Interaction','Shapley_q','Shapley_cond']],'outputs/counterfactual/counterfactual_interaction.csv')
 def mets(d,scope):
  r=[]
  for cf,g in d.groupby('cf'):
   x=g[g.oracle_estimable];r.append({'scope':scope,'cf':cf,'prediction_point_count':len(x),'unique_target_count':x.target_row.nunique(),'RMSE_mV':rmse(x.error_mV),'P95_abs_error_mV':float(np.quantile(abs(x.error_mV),.95)),'P99_abs_error_mV':float(np.quantile(abs(x.error_mV),.99)),'max_abs_error_mV':float(abs(x.error_mV).max())})
  return r
 foldrows=[]
 for f in FOLDS: foldrows.extend(mets(point[point.fold.eq(f)],f))
 foldrows.extend(mets(point,'pooled_all_folds'))
 foldm=pd.DataFrame(foldrows);csv(foldm,'outputs/counterfactual/counterfactual_fold_metrics.csv')
 # scopes for attribution
 scopes={'fold1_h12':wide[(wide.fold=='fold_1')&(wide.horizon_step==24)],'fold3_continuous_nonzero':wide[(wide.fold=='fold_3')&(wide.target_row.map(frame.zero_current_fraction).shift(0).fillna(0)<=EPS)]}
 # refine F3 exact transition label from preceding raw q
 f3=wide[wide.fold.eq('fold_3')].copy();prev=frame.zero_current_fraction.shift(1);f3=f3[(f3.target_row.map(prev)<=EPS)&(f3.target_row.map(frame.zero_current_fraction)<=EPS)&(f3.target_row.map(frame.temporal_segment_id)==(f3.target_row-1).map(frame.temporal_segment_id))];scopes['fold3_continuous_nonzero']=f3
 at=[]
 for name,d in scopes.items():
  if d.empty: continue
  at.append({'scope':name,'prediction_point_count':len(d),'unique_target_count':d.target_row.nunique(),'CF00_RMSE_mV':rmse(d.e00),'CF10_RMSE_mV':rmse(d.e10),'CF01_RMSE_mV':rmse(d.e01),'CF11_RMSE_mV':rmse(d.e11),'mean_Gain_q_mV2':d.Gain_q.mean(),'mean_Gain_cond_mV2':d.Gain_cond.mean(),'mean_Gain_both_mV2':d.Gain_both.mean(),'mean_Shapley_q_mV2':d.Shapley_q.mean(),'mean_Shapley_cond_mV2':d.Shapley_cond.mean(),'mean_Interaction_mV2':d.Interaction.mean()})
 attr=pd.DataFrame(at);csv(attr,'outputs/counterfactual/counterfactual_regime_metrics.csv');csv(attr[attr.scope.eq('fold1_h12')],'outputs/fold1_12h/fold1_12h_counterfactual.csv');csv(attr[attr.scope.eq('fold3_continuous_nonzero')],'outputs/fold3_nonzero/fold3_nonzero_counterfactual.csv')
 evrows=[]
 for typ,eg in vu.groupby('error_type'):
  d=point.merge(eg[['fold','origin_row','target_row','horizon_step']].drop_duplicates(),on=['fold','origin_row','target_row','horizon_step'],how='inner')
  for r in mets(d,'validation_'+typ): r['error_type']=typ;evrows.append(r)
 csv(pd.DataFrame(evrows),'outputs/counterfactual/counterfactual_event_metrics.csv')
 response=point.pivot_table(index=['fold','origin_row','target_row','horizon_step'],columns='cf',values=['B_mV','C_mV','prediction_deltaV_mV'],aggfunc='first').reset_index();response.columns=['_'.join(str(x) for x in c if x!='') for c in response.columns];csv(response,'outputs/response/condition_response_components.csv')
 remaining=[]
 for fold,p in packs.items():
  for j,name in enumerate(cfg['future_exogenous_features']):
   if j in (0,1,3,4,8):continue
   remaining.append({'fold':fold,'feature':name,'raw_MAE':float(abs(valz[fold][:,:,j]-p['zva'][:,:,j]).mean()),'raw_RMSE':float(np.sqrt(np.mean((valz[fold][:,:,j]-p['zva'][:,:,j])**2))),'note':'current-density features are redundant with current under fixed train-only area' if j in (3,4,5) else 'unintervened future feature'})
 csv(pd.DataFrame(remaining),'outputs/counterfactual/remaining_condition_error_audit.csv')
 baseold=pd.read_csv(OPR/'outputs/reproduction/baseline_reproduction_summary.csv');csv(baseold,'outputs/reproduction/baseline_reproduction.csv');support=pd.DataFrame([{'scope':'train_OOF_global','E1_independent_events':int((tglob.error_type=='E1_predicted_mixed_true_nonmixed').sum()),'E2_independent_events':int((tglob.error_type=='E2_predicted_nonmixed_true_mixed').sum()),'temporal_segments':int(tglob.temporal_segment_id.nunique()),'support_status':'SUFFICIENT' if (tglob.error_type=='E1_predicted_mixed_true_nonmixed').sum()+(tglob.error_type=='E2_predicted_nonmixed_true_mixed').sum()>=20 and tglob.temporal_segment_id.nunique()>=3 else 'INSUFFICIENT'}]);csv(support,'outputs/events/event_support_summary.csv')
 js({**fire,'raw_prefix':rawmeta,'forbidden_auxiliary_files_read':[],'test_predictions_constructed':False,'test_metrics_computed':False,'test_performance_accessed':False},'outputs/audit/test_firewall_audit.json');print('R0: completed',flush=True)
if __name__=='__main__':main()

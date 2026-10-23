"""One frozen CPU run using cached L96 only; historical predictions are read-only."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
from pathlib import Path
import sys, json, hashlib, subprocess, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from models import FEATURES, ALPHAS, train_validation, inner_split, ridge_fit, error_metrics

STAGE=Path(__file__).resolve().parents[1]
PROJECT=STAGE.parent
VREF=.71383023635
KEY=['fold','origin_raw_cycle_id','target_raw_cycle_id']
OLD='Phase_P1S_LongHistory_ShortHorizon_Forecasting'
SOURCES={
 'H1_features':f'{OLD}/inputs/H1_FEATURES_AND_ELIGIBILITY.csv',
 'H3_features':f'{OLD}/inputs/H3_FEATURES_AND_ELIGIBILITY.csv',
 'history':f'{OLD}/outputs/HISTORY_WINDOW_IDENTITIES.json',
 'identity':f'{OLD}/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv',
 'prior_protocol':f'{OLD}/configs/FROZEN_LONG_HISTORY_PROTOCOL.json',
 'prior_models':f'{OLD}/scripts/history_models.py',
 'catalog':'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv',
 'H1_authority':'Phase_CPRF_H1_Causal_P1S_Retention_Forecasting/inputs/H1_FROZEN_SAMPLE_AUTHORITY.csv',
 'H3_authority':'Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/inputs/H3_FROZEN_SAMPLE_AUTHORITY.csv',
 'H1_baseline':'Phase_CPRF_H1_Causal_P1S_Retention_Forecasting/outputs/H1_BASELINE_PREDICTIONS.csv',
 'H1_timesnet':'Phase_CPRF_H1_Causal_P1S_Retention_Forecasting/outputs/H1_TIMESNET_PREDICTIONS.csv',
 'H3_history':'Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/outputs/H3_PREDICTIONS.csv',
 'H1_lead_erratum':'Phase_H1E_Lead_Availability_And_Information_Set_Erratum/reports/H1_LEAD_AND_INFORMATION_SET_ERRATUM.md',
}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(rel,data):
 p=STAGE/rel;assert p.resolve().is_relative_to(STAGE);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(rel,data):
 p=STAGE/rel;assert p.resolve().is_relative_to(STAGE);p.parent.mkdir(parents=True,exist_ok=True)
 data.to_csv(p,index=False,float_format='%.17g',lineterminator='\n')
def read_source(name):return pd.read_csv(PROJECT/SOURCES[name],float_precision='round_trip')
def stamp():return datetime.now(timezone.utc).isoformat()

def freeze():
 p=STAGE/'configs/FROZEN_PROTOCOL.json'
 if p.exists():raise RuntimeError('Protocol already frozen; refusing overwrite')
 old=json.loads((PROJECT/SOURCES['prior_protocol']).read_text(encoding='utf-8'))
 cfg=dict(phase=STAGE.name,UTC=stamp(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
  Vref_V=VREF,folds=old['folds'],history_h=96,minimum_observations=5,minimum_span_h=67.2,
  feature_names=['L96_'+x for x in FEATURES],target='DeltaR=R_target-R_origin; R=V_P1S/Vref',
  horizons={'H1':1,'H3':3},alpha_grid=ALPHAS,inner_last_fraction=.15,inner_count_rule='ceil',
  outer_purge='train label end <= earliest used validation origin',inner_purge='inner fit label end <= earliest inner selection origin',
  scaler='inner fit only for selection; outer train only for final',target_scaled=False,solver='cholesky float64 with intercept',
  primary_metric='three-fold equal MAE_mV_equivalent; missing fold => no formal mean',
  decision={'H1_minimum_MAE_improvement':.05,'H1_minimum_fold_wins':2,'all_three_folds_required':True,
    'tail_screen':'Conservative sufficient condition: every fold Ridge RMSE, P95 and maximum <= Persistence (tolerance 1e-12 mV); failure with other gains => MIXED. This operational screen is stricter than qualitative obvious worsening; not an additional tuning target.'},
  historical_comparison='exact horizon/fold/origin raw ID/target raw ID intersection; separate all seed errors, no ensemble; only descriptive',
  historical_H1_Ridge_actual_future_lead=True,CPU_only=True,no_test_or_raw_reparse=True,no_192_experiment=True,
  sources=SOURCES,source_sha256={k:digest(PROJECT/v) for k,v in SOURCES.items()},
  code_sha256={str(x.relative_to(STAGE)).replace('\\','/'):digest(x) for root in ['scripts','tests'] for x in sorted((STAGE/root).glob('*.py'))})
 write_json('configs/FROZEN_PROTOCOL.json',cfg)
 (STAGE/'configs/FROZEN_PROTOCOL.sha256').write_text(digest(p)+'\n',encoding='utf-8')
 print('FROZEN',cfg['base_commit'],flush=True)

def verify_sources(cfg):
 assert digest(STAGE/'configs/FROZEN_PROTOCOL.json')==(STAGE/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 for k,v in cfg['source_sha256'].items():assert digest(PROJECT/SOURCES[k])==v,k
 for k,v in cfg['code_sha256'].items():assert digest(STAGE/k)==v,k

def load_qualified(h,cfg,identity,details,catalog):
 # Deliberately do not select lead, future condition, target time or L192 as features.
 keep=['fold','role','sample_index','origin_cycle_interval_id','target_cycle_interval_id','origin_raw_cycle_id','target_raw_cycle_id',
  'origin_p1s_measurement_start_ophrs2_h','origin_p1s_measurement_end_ophrs2_h',
  'target_p1s_measurement_start_ophrs2_h','target_p1s_measurement_end_ophrs2_h','origin_R_P1S_causal','target_R_P1S_causal',
  *cfg['feature_names'],'L96_eligible','L96_observations','L96_span_h','L96_reason']
 d=read_source(h+'_features')[keep].copy();d.insert(0,'horizon',h)
 authority=read_source(h+'_authority')
 assert len(d)==len(authority)
 for col in keep[:13]:
  if col in authority:
   if pd.api.types.is_numeric_dtype(d[col]):assert np.allclose(d[col],authority[col],atol=1e-12,rtol=0),col
   else:assert np.array_equal(d[col].to_numpy(),authority[col].to_numpy()),col
 assert not d.duplicated(['fold','role','sample_index']).any()
 assert np.array_equal(d.target_raw_cycle_id-d.origin_raw_cycle_id,np.full(len(d),cfg['horizons'][h]))
 cat=catalog.set_index('raw_cycle_id')
 evidence=[]
 for r in d.itertuples():
  lo,mid,hi=cfg['folds'][r.fold];left,right=(lo,mid) if r.role=='train' else (mid,hi)
  assert left<=r.target_p1s_measurement_start_ophrs2_h<=r.target_p1s_measurement_end_ophrs2_h<right
  for raw in range(r.origin_raw_cycle_id,r.target_raw_cycle_id+1):assert bool(cat.loc[raw,'complete_cycle'])
  a=details[(h,r.fold,r.role,r.sample_index)]
  hist=identity.loc[a['cycle_ids']].sort_values('measurement_end_h')
  x=hist.measurement_end_h.to_numpy();y=hist.R.to_numpy();origin=r.origin_p1s_measurement_end_ophrs2_h
  assert len(hist)==r.L96_observations==a['observations']
  assert np.all(x<=origin+1e-9) and np.all(x>=origin-96-1e-9)
  # The frozen raw reconstruction locates the actual P1S measurement end.
  # Old conservative end is the later dwell-state end, not a required publication delay.
  assert abs(np.ptp(x)-r.L96_span_h)<1e-9
  legal=len(hist)>=5 and np.ptp(x)>=67.2-1e-9
  assert legal==r.L96_eligible==a['eligible']
  assert abs(x[-1]-origin)<1e-9 and abs(y[-1]-r.origin_R_P1S_causal)<1e-12
  assert bool(hist.causal_target_eligible.all())
  assert abs(identity.loc[r.target_cycle_interval_id,'measurement_end_h']-r.target_p1s_measurement_end_ophrs2_h)<1e-9
  assert abs(identity.loc[r.target_cycle_interval_id,'R']-r.target_R_P1S_causal)<1e-12
  evidence.append({'horizon':h,'fold':r.fold,'role':r.role,'sample_index':r.sample_index,**a,
    'max_actual_measurement_end_h':float(x.max()),'old_conservative_dwell_end_h':float(hist.conservative_available_end_h.max()),'reused_cached_L96':True})
 return d,evidence

def compare_history(h,current):
 caches=[read_source('H1_baseline'),read_source('H1_timesnet')] if h=='H1' else [read_source('H3_history')]
 old=pd.concat(caches,ignore_index=True)
 names=['PERSISTENCE','RIDGE_CAUSAL_STATE','TIMESNET'] if h=='H1' else ['PERSISTENCE','RIDGE_MATCHED_CONTEXT','TIMESNET_ORIGIN_CONDITIONED']
 old=old[old.model.isin(names)].copy();old['seed_label']=old.seed.fillna(-1).astype(int).astype(str)
 ridge=current[current.model=='RIDGE_96H'].copy()
 records=[];matches=[];metrics=[];summary=[]
 for (model,seed),group in old.groupby(['model','seed_label']):
  assert not group.duplicated(KEY).any()
  m=ridge.merge(group[KEY+['true_R_P1S_causal','predicted_R_P1S_causal','target_p1s_measurement_end_ophrs2_h']],on=KEY,how='outer',indicator=True,suffixes=('_current','_old'),validate='one_to_one')
  for r in m.itertuples(index=False):
   records.append(dict(horizon=h,historical_model=model,seed=seed,fold=r.fold,origin_raw_cycle_id=int(r.origin_raw_cycle_id),target_raw_cycle_id=int(r.target_raw_cycle_id),match_status=str(r._asdict().get('_merge',''))))
  # pandas renames underscore tuple fields: read merge status directly.
  for idx,status in enumerate(m['_merge'].astype(str)):records[-len(m)+idx]['match_status']=status
  matched=m[m['_merge']=='both'].copy()
  assert np.max(np.abs(matched.true_R_P1S_causal_current-matched.true_R_P1S_causal_old))<1e-12
  assert np.max(np.abs(matched.target_p1s_measurement_end_ophrs2_h_current-matched.target_p1s_measurement_end_ophrs2_h_old))<1e-9
  if model=='PERSISTENCE':assert np.max(np.abs(matched.origin_R_P1S_causal-matched.predicted_R_P1S_causal_old))<1e-12
  matched['historical_model']=model;matched['historical_seed']=seed;matched['horizon']=h
  matches.append(matched.drop(columns=['_merge']))
  for fold,g in matched.groupby('fold'):
   for label,column in [('CURRENT_RIDGE_96H','predicted_R_P1S_causal_current'),('HISTORICAL_'+model,'predicted_R_P1S_causal_old')]:
    metrics.append(dict(horizon=h,historical_model=model,seed=seed,fold=fold,method=label,**error_metrics(g.true_R_P1S_causal_current,g[column],VREF)))
  mm=pd.DataFrame(metrics);sub=mm[(mm.horizon==h)&(mm.historical_model==model)&(mm.seed==seed)]
  av=sub.groupby('method').MAE_mV_equivalent.mean();both=set(sub.fold)=={'fold_1','fold_2','fold_3'}
  new=float(av['CURRENT_RIDGE_96H']);past=float(av['HISTORICAL_'+model])
  summary.append(dict(horizon=h,historical_model=model,seed=seed,intersection_n=len(matched),
    fold_n={f:int((matched.fold==f).sum()) for f in ['fold_1','fold_2','fold_3']},three_fold_evaluable=both,
    current_equal_fold_MAE_mV=new if both else None,historical_equal_fold_MAE_mV=past if both else None,
    current_improvement_fraction=1-new/past if both else None,
    comparison_class='historical actual future lead reference only' if h=='H1' and model=='RIDGE_CAUSAL_STATE' else 'exact-target descriptive; information and training conditions differ' if 'PERSISTENCE' not in model else 'same deterministic formula and targets, verified'))
 return pd.DataFrame(records),pd.concat(matches),pd.DataFrame(metrics),summary

def decide(metrics,h):
 d=metrics[metrics.horizon==h];p=d[d.model=='PERSISTENCE'].set_index('fold');r=d[d.model=='RIDGE_96H'].set_index('fold')
 if len(p)!=3 or len(r)!=3 or min(p.n.min(),r.n.min())==0:return {'status':'THREE_FOLD_NOT_EVALUABLE'}
 a=float(p.MAE_mV_equivalent.mean());b=float(r.MAE_mV_equivalent.mean());gain=1-b/a
 wins=int((r.MAE_mV_equivalent<p.MAE_mV_equivalent).sum())
 tail=all((r[c]<=p[c]+1e-12).all() for c in ['RMSE_mV_equivalent','P95_abs_error_mV_equivalent','max_abs_error_mV_equivalent'])
 status='96H_SHORT_HORIZON_SIGNAL_SUPPORTED' if gain>=.05 and wins>=2 and tail else '96H_SHORT_HORIZON_MIXED' if gain>0 or wins>0 else '96H_NO_CLEAR_GAIN'
 return dict(status=status,Persistence_equal_fold_MAE_mV=a,Ridge_equal_fold_MAE_mV=b,improvement_fraction=gain,fold_wins=wins,
  at_least_5_percent=gain>=.05,at_least_two_fold_wins=wins>=2,conservative_tail_screen_pass=bool(tail),three_fold_evaluable=True)

def run():
 cfg=json.loads((STAGE/'configs/FROZEN_PROTOCOL.json').read_text(encoding='utf-8'));verify_sources(cfg)
 if (STAGE/'outputs/RUN_LEDGER.json').exists():raise RuntimeError('Run exists; refusing repeat fitting')
 write_json('outputs/RUN_LEDGER.json',dict(status='RUNNING',UTC=stamp(),CPU=True,base_commit=cfg['base_commit']))
 identity=read_source('identity').set_index('cycle_interval_id');catalog=read_source('catalog')
 assert identity.measurement_end_h.max()<1819.997738
 assert np.max(np.abs(identity.R*VREF-identity.P1S_voltage_reconstructed_V))<1e-12
 all_details=json.loads((PROJECT/SOURCES['history']).read_text(encoding='utf-8'))
 details={(a['horizon'],a['fold'],a['role'],a['sample_index']):a for a in all_details if a['length_h']==96}
 frames=[];evidence=[];legal=[];excluded=[];counts=[];predictions=[];metrics=[];parameters=[];inner_rows=[]
 for h in ['H1','H3']:
  d,ev=load_qualified(h,cfg,identity,details,catalog);frames.append(d);evidence+=ev
  for fold in cfg['folds']:
   train,val,rejected,cutoff=train_validation(d,fold)
   for role,g in [('train',train),('validation',val)]:
    legal.append(g.assign(final_role=role,model_available_cutoff_h=cutoff))
    counts.append(dict(horizon=h,fold=fold,role=role,authority_n=int(((d.fold==fold)&(d.role==role)).sum()),
      L96_eligible_n=int(((d.fold==fold)&(d.role==role)&d.L96_eligible).sum()),used_n=len(g),outer_purged_n=len(rejected) if role=='train' else 0))
   bad=d[(d.fold==fold)&~d.L96_eligible].copy();bad['exclusion_reason']=bad.L96_reason;bad['exclusion_stage']='history_qualification';excluded.append(bad)
   rejected=rejected.assign(exclusion_reason='TRAIN_LABEL_NOT_AVAILABLE_AT_FIRST_VALIDATION_ORIGIN',exclusion_stage='outer_parameter_time');excluded.append(rejected)
   fit,select,purge,ic=inner_split(train)
   for role,g in [('inner_fit',fit),('inner_selection',select),('inner_purged',purge)]:inner_rows.append(g.assign(inner_role=role,inner_cutoff_h=ic))
   delta,param=ridge_fit(train,val,96);param.update(horizon=h,fold=fold,outer_cutoff_h=cutoff,outer_purged_samples=rejected.sample_index.astype(int).tolist());parameters.append(param)
   for model,pred in [('PERSISTENCE',val.origin_R_P1S_causal.to_numpy()),('RIDGE_96H',val.origin_R_P1S_causal.to_numpy()+delta)]:
    z=val.copy();z['model']=model;z['true_R_P1S_causal']=val.target_R_P1S_causal;z['predicted_R_P1S_causal']=pred
    z['error_R']=pred-val.target_R_P1S_causal;z['absolute_error_mV_equivalent']=np.abs(z.error_R)*VREF*1000
    predictions.append(z);metrics.append(dict(horizon=h,fold=fold,model=model,**error_metrics(val.target_R_P1S_causal,pred,VREF)))
   print(h,fold,'train',len(train),'validation',len(val),'alpha',param['alpha'],flush=True)
 preds=pd.concat(predictions,ignore_index=True);met=pd.DataFrame(metrics)
 write_csv('inputs/L96_REUSED_FEATURES.csv',pd.concat(frames));write_json('inputs/L96_TIME_IDENTITIES.json',evidence)
 write_csv('outputs/LEGAL_SAMPLES.csv',pd.concat(legal));write_csv('outputs/EXCLUDED_SAMPLES.csv',pd.concat(excluded));write_csv('outputs/INNER_SPLITS.csv',pd.concat(inner_rows))
 write_csv('outputs/SAMPLE_COUNTS.csv',pd.DataFrame(counts));write_csv('outputs/PREDICTIONS.csv',preds);write_csv('outputs/FOLD_METRICS.csv',met);write_json('outputs/TRAINED_PARAMETERS.json',parameters)
 pairs=[]
 for h in ['H1','H3']:
  for f in cfg['folds']:
   a=met[(met.horizon==h)&(met.fold==f)].set_index('model');row=dict(horizon=h,fold=f,n=int(a.loc['PERSISTENCE','n']))
   for c in ['MAE_mV_equivalent','RMSE_mV_equivalent','P95_abs_error_mV_equivalent','max_abs_error_mV_equivalent']:
    row['Persistence_'+c]=float(a.loc['PERSISTENCE',c]);row['Ridge_'+c]=float(a.loc['RIDGE_96H',c]);row['improvement_'+c]=1-row['Ridge_'+c]/row['Persistence_'+c]
   pairs.append(row)
 write_csv('outputs/PAIRED_COMPARISON.csv',pd.DataFrame(pairs))
 logs=[];matches=[];oldmet=[];summ=[]
 for h in ['H1','H3']:
  a,b,c,s=compare_history(h,preds[preds.horizon==h]);logs.append(a);matches.append(b);oldmet.append(c);summ+=s
 write_csv('outputs/HISTORICAL_KEY_MATCH_LOG.csv',pd.concat(logs));write_csv('outputs/HISTORICAL_MATCHED_PREDICTIONS.csv',pd.concat(matches));write_csv('outputs/HISTORICAL_INTERSECTION_METRICS.csv',pd.concat(oldmet));write_json('outputs/HISTORICAL_COMPARISON_SUMMARY.json',summ)
 decision={h:decide(met,h) for h in ['H1','H3']};write_json('outputs/DECISIONS.json',decision)
 verify_sources(cfg)
 write_json('outputs/RUN_LEDGER.json',dict(status='COMPLETED',UTC=stamp(),CPU=True,ridge_final_fits=6,ridge_inner_fits=24,total_fits=30,
   validation_prediction_rows=len(preds),historical_retrained=False,raw_reparsed=False,Test_accessed=False,environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)))
 print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
 if sys.argv[1]=='freeze':freeze()
 elif sys.argv[1]=='run':run()
 else:raise ValueError('Use freeze or run')

"""One run: trajectory eligibility, frozen-native replay, new long-horizon CPU forecasts."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys
from common import *
from methods import features,trend,replay,train_new
def freeze():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists()
 old=jl('native_protocol')
 c={'base_commit':'91b5263c3f11683a450b702d77f8ee551044012c','source_sha256':{v:sha(ROOT/v) for v in SOURCES.values()},
  'calculation_code_sha256':{q.relative_to(P).as_posix():sha(q) for q in [P/'scripts/common.py',P/'scripts/methods.py',P/'scripts/run_experiment.py']},
  'folds':old['folds'],'horizons':[1,3,6,12],'Vref_V':VREF,'history_h':96,'minimum_observations':5,'minimum_span_h':67.2,
  'forecast_target':'Raw physical cycle ID origin+H; require all intermediate raw cycles complete and measured origin/target; no skipping',
  'trajectory':'131 causal eligible complete-cycle middle50 measurements;10 reference warmup separate;6 partials descriptive only; no inferred values for uncached complete cycles',
  'history_clock':'Actual stable measurement end hours; each observation conservative available end <= origin conservative end; no nominal cycle-time conversion',
  'trend':'Anchor currentR plus fixed96h OLS slope vs actual measurement hours times H*median adjacent raw-cycle past actual-hour gaps within same96h history',
  'features':FEATURES,'alpha_grid':old['alpha_grid'],'inner_last_fraction':.15,
  'native_H1_H3':'First replay original188 predictions; use frozen coefficients unchanged on new P1S-only cohort with their actual Train label availability cutoff; zero H1/H3 refits',
  'new_H6_H12':'New CPU experiments with inherited5 features, grid and Train-only scaler/inner last15% chronological purge; exactly6 final+24 inner fits if all folds usable',
  'cohort':'New P1S-only cohort does not require TFT input window; all target start/end and availability within original role bounds; H1/H3 expanded-cohort results distinct from native results',
  'outer_purge':'NewH6/H12 Train labels conservatively available by earliest eligible Validation origin; frozenH1/H3 model excludes origins earlier than its Train label availability',
  'inner_purge':'Inner fit labels available by first inner-selection origin; scaler fit only corresponding fit set',
  'primary':'Threefold equal mean MAE_mVeq per horizon on shared legal A/B/C samples; any empty fold -> no formal threefold mean',
  'cross_horizon':'Also exact same fold+origin intersection across all4 horizons; report empty intersections explicitly',
  'secondary':'Per-fold RMSE,P95,max; observed futureR increase and known-origin increase strata; future target Train-condition-range OOD diagnostics only, never a selection/model input',
  'condition_audit':'Retain all131, no outcome-dependent filtering; min/max for target conditions derived only each Train. Original PAA common support descriptive only.',
  'D_model':'Not proposed; fixed A/B/C suffice to test predictability; no rescue models or feature/hyperparameter revision after scores',
  'physics':'Fixedj ohmic relation qualification and existing EIS/identifiability gates; no regularizer relabeling, no new electrochemical fit; invalid loss/gradient staysNA',
  'TFT':'Only post-selection parameters and causal event features eligible; assess available Train/Validation samples and cached origins without inference/retraining',
  'SOH_RUL':'No optimization of three reference events, no new SOH calibration/labels/EOL thresholds/RUL training',
  'no_Test':True,'CPU_only':True,'stop_rule':'Run one fixed grid; if fit/validation missing report not evaluable; do not retune after results'}
 js('configs/FROZEN_PROTOCOL.json',c);md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'))
 for key,value in SOURCES.items():out('inputs/'+key+Path(value).suffix).write_bytes((ROOT/value).read_bytes())
 cs('inputs/SOURCE_SHA256.csv',pd.DataFrame([{'key':k,'path':v,'sha256':c['source_sha256'][v]} for k,v in SOURCES.items()]))
 md('.gitignore','__pycache__/\n*.pyc\noutputs/GIT_RECEIPT.local.json')
 md('.gitattributes','* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.png binary\n*.pdf binary')
 print('FROZEN',sha(P/'configs/FROZEN_PROTOCOL.json'))
def run():
 cfg=config();assert not (P/'outputs/RUN_LEDGER.json').exists(),'Never repeat model fitting'
 js('outputs/RUN_LEDGER.json',{'status':'RUNNING','new_fits':0,'CPU':True})
 ident=source('identity').sort_values('measurement_end_h');meas=source('measurement');meas=meas[meas.stable_window.eq('middle_50')].copy();partial=source('partial');cat=source('catalog').set_index('raw_cycle_id')
 assert len(ident)==131 and len(meas)==141 and len(partial)==6 and ident.causal_target_eligible.all()
 assert ident.measurement_end_h.max()<1819.997738 and partial.measurement_end_ophrs2_h.max()<1819.997738
 assert ident.cycle_interval_id.is_unique and meas.cycle_interval_id.is_unique
 m=meas.set_index('cycle_interval_id');ids=ident.set_index('raw_cycle_id');warm=meas[meas.is_reference_complete_interval]
 assert len(warm)==10 and abs(warm.P1S_voltage_V.median()-VREF)<1e-12
 assert abs(warm.measurement_end_ophrs2_h.max()-171.82091)<1e-9
 trajectory=[]
 for _,row in meas.iterrows():
  raw=int(row.raw_cycle_id);assert bool(cat.loc[raw,'complete_cycle']);has=raw in ids.index
  q=ids.loc[raw] if has else None
  if has:assert row.cycle_interval_id==q.cycle_interval_id and abs(row.P1S_voltage_V/VREF-q.R)<1e-12
  trajectory.append({**row.to_dict(),'actual_stable_start_h':float(q.measurement_start_h) if has else np.nan,
   'actual_stable_end_h':float(q.measurement_end_h) if has else np.nan,'conservative_available_end_h':float(row.measurement_end_ophrs2_h),
   'available_calendar':row.measurement_end_datetime,'exact_stable_end_calendar':None,'causal_target_eligible':has,
   'data_role':'forecast_trajectory' if has else 'reference_warmup_not_causal_target','Development':True,'complete_cycle':True})
 for _,row in partial.iterrows():trajectory.append({**row.to_dict(),'actual_stable_start_h':np.nan,'actual_stable_end_h':np.nan,'conservative_available_end_h':float(row.measurement_end_ophrs2_h),
  'available_calendar':row.measurement_end_datetime,'exact_stable_end_calendar':None,'causal_target_eligible':False,'data_role':'partial_descriptive_only','Development':True,'complete_cycle':False})
 traj=pd.DataFrame(trajectory);cs('outputs/REAL_P1S_OBSERVATIONS.csv',traj)
 gaps=[]
 for raw,q in cat.iterrows():
  gaps.append({'raw_cycle_id':int(raw),'complete_cycle':bool(q.complete_cycle),'cycle_start':q.cycle_start,'cycle_end':q.cycle_end,
   'end_ophrs2_h':float(q.end_ophrs2_h),'cached_complete_P1S':int(raw) in set(meas.raw_cycle_id.astype(int)),
   'status':'CACHED_P1S' if int(raw) in set(meas.raw_cycle_id.astype(int)) else 'NO_P1S_VALUE_IN_EXISTING_CACHE' if q.complete_cycle else 'INCOMPLETE_PHYSICAL_CYCLE'})
 cs('outputs/CYCLE_MEASUREMENT_COVERAGE.csv',pd.DataFrame(gaps))
 cond=[];eligible_meas=meas[meas.cycle_interval_id.isin(ident.cycle_interval_id)]
 for col in CONDITIONS:
  for name,g in [('causal131',eligible_meas),('reference_warmup10',warm),('partial6',partial)]:
   cond.append({'scope':name,'variable':col,'n':len(g),'missing':int(g[col].isna().sum()),'min':float(g[col].min()),'max':float(g[col].max()),'unique':int(g[col].nunique())})
 cs('outputs/MEASURED_CONDITION_RANGES.csv',pd.DataFrame(cond))
 # Rebuild all96h features; cross-check the inherited cache where a matching origin exists.
 hist={};evidence=[];feature_replay=[];fc=source('native_features');names=['L96_origin_R','L96_history_mean_R','L96_history_std_R','L96_history_slope_R_per_h','L96_origin_minus_median_R']
 for row in ident.itertuples(index=False):
  values,ev=features(ident,row);hist[int(row.raw_cycle_id)]=(values,ev)
  evidence.append({'origin_raw_cycle_id':int(row.raw_cycle_id),'origin_actual_end_h':float(row.measurement_end_h),'origin_available_h':float(row.conservative_available_end_h),**values,**ev})
  old=fc[fc.origin_raw_cycle_id.eq(row.raw_cycle_id)]
  if len(old):
   maxdiff=float(np.max(abs(old[names].to_numpy()-np.array([values[k] for k in FEATURES]))));assert maxdiff<1e-12
   feature_replay.append({'raw_cycle_id':int(row.raw_cycle_id),'cache_rows':len(old),'max_feature_difference':maxdiff})
 js('outputs/HISTORY_FEATURE_IDENTITIES.json',evidence);cs('outputs/HISTORY_FEATURE_REPLAY.csv',pd.DataFrame(feature_replay))
 # Native188 predictions are reconstructed from coefficients, no native fitting.
 oldpred=source('native_predictions');pars=jl('native_parameters');nt=source('native_time');nativereplay=[]
 for (h,f,model),g in oldpred.groupby(['horizon','fold','model']):
  frame=pd.DataFrame([hist[int(raw)][0] for raw in g.origin_raw_cycle_id]);par=next(p for p in pars if p['horizon']==h and p['fold']==f)
  pred=frame.origin_R.to_numpy() if model=='PERSISTENCE' else replay(frame,par)
  assert np.max(abs(pred-g.predicted_R_P1S_causal.to_numpy()))<1e-12
  nativereplay.append({'horizon':h,'fold':f,'model':model,'max_prediction_difference':float(np.max(abs(pred-g.predicted_R_P1S_causal))),**metric(g.true_R_P1S_causal,pred)})
 cs('outputs/NATIVE_H1_H3_REPLAY.csv',pd.DataFrame(nativereplay))
 samples=[];excluded=[]
 for H in cfg['horizons']:
  for fold,(lo,mid,hi) in cfg['folds'].items():
   for o in ident.itertuples(index=False):
    raw=int(o.raw_cycle_id);targetraw=raw+H;key=f'H{H}_{fold}_{raw}_{targetraw}';reason=[]
    if targetraw not in ids.index:reason.append('NO_MEASURED_TARGET')
    if any(r not in cat.index or not bool(cat.loc[r,'complete_cycle']) for r in range(raw,targetraw+1)):reason.append('INCOMPLETE_INTERMEDIATE_PHYSICAL_CYCLE')
    values,ev=hist[raw]
    if not ev['eligible']:reason.append('96H_HISTORY_NOT_QUALIFIED')
    if targetraw not in ids.index:
     excluded.append({'sample_id':key,'H':H,'fold':fold,'origin_raw_cycle_id':raw,'target_raw_cycle_id':targetraw,'reason':'|'.join(reason)});continue
    t=ids.loc[targetraw];role='train' if lo<=t.measurement_start_h and t.conservative_available_end_h<mid else 'validation' if mid<=t.measurement_start_h and t.conservative_available_end_h<hi else None
    if role is None:reason.append('TARGET_NOT_ENTIRELY_IN_ROLE_RANGE')
    if reason:
     excluded.append({'sample_id':key,'H':H,'fold':fold,'origin_raw_cycle_id':raw,'target_raw_cycle_id':targetraw,'reason':'|'.join(reason)});continue
    origin_meas=m.loc[o.cycle_interval_id];target_meas=m.loc[t.cycle_interval_id];lead=float(t.measurement_end_h-o.measurement_end_h)
    previous=ident[(ident.raw_cycle_id<raw)&(ident.conservative_available_end_h<=o.conservative_available_end_h)].sort_values('measurement_end_h')
    samples.append({'sample_id':key,'H':H,'fold':fold,'role':role,'origin_raw_cycle_id':raw,'target_raw_cycle_id':targetraw,
     'origin_cycle_interval_id':o.cycle_interval_id,'target_cycle_interval_id':t.cycle_interval_id,'origin_actual_end_h':float(o.measurement_end_h),
     'origin_available_h':float(o.conservative_available_end_h),'origin_calendar':origin_meas.measurement_end_datetime,'target_actual_start_h':float(t.measurement_start_h),
     'target_actual_end_h':float(t.measurement_end_h),'target_available_h':float(t.conservative_available_end_h),'target_calendar':target_meas.measurement_end_datetime,
     'actual_lead_h_AUDIT_ONLY':lead,'actual_calendar_lead_h_AUDIT_ONLY':(pd.Timestamp(target_meas.measurement_end_datetime)-pd.Timestamp(origin_meas.measurement_end_datetime)).total_seconds()/3600,
     'past_cadence_h':ev['cadence_h_per_cycle'],'estimated_lead_h':H*ev['cadence_h_per_cycle'],'history_observations':ev['observations'],'history_span_h':ev['span_h'],
     'origin_observed_increase':bool(len(previous) and o.R>float(previous.iloc[-1].R)),
     'future_observed_increase_AUDIT_ONLY':bool(t.R>o.R),'target_R':float(t.R),**values,
     **{'origin_condition_'+col:float(origin_meas[col]) for col in CONDITIONS},**{'target_condition_AUDIT_ONLY_'+col:float(target_meas[col]) for col in CONDITIONS}})
 frame=pd.DataFrame(samples);cs('outputs/INITIAL_QUALIFIED_SAMPLES.csv',frame)
 models=[];predictions=[];legal=[];time=[];condition_support=[];c_unavailable=[]
 for H in cfg['horizons']:
  for fold in cfg['folds']:
   sub=frame[frame.H.eq(H)&frame.fold.eq(fold)];train=sub[sub.role.eq('train')].sort_values('target_actual_end_h');val=sub[sub.role.eq('validation')].sort_values('origin_available_h')
   if H in [1,3]:
    par=next(p for p in pars if p['horizon']==f'H{H}' and p['fold']==fold);native=source('native_legal');native=native[native.horizon.eq(f'H{H}')&native.fold.eq(fold)&native.role.eq('train')]
    nativeids={f'H{H}_{fold}_{int(r.origin_raw_cycle_id)}_{int(r.target_raw_cycle_id)}' for r in native.itertuples()}
    # Reconstruct exactly inherited Train rows; they all satisfy the new clock qualification.
    train=frame[frame.sample_id.isin(nativeids)].sort_values('target_actual_end_h');assert len(train)==len(nativeids)
    cutoff=float(train.target_available_h.max());bad=val[val.origin_available_h<cutoff];val=val[val.origin_available_h>=cutoff]
    for r in bad.itertuples():c_unavailable.append({'sample_id':r.sample_id,'reason':'FROZEN_NATIVE_TRAIN_LABEL_NOT_YET_AVAILABLE'})
    kind='FROZEN_NATIVE_COEFFICIENTS_NEW_P1S_ONLY_COHORT'
   else:
    cutoff=float(val.origin_available_h.min()) if len(val) else float(cfg['folds'][fold][1]);bad=train[train.target_available_h>cutoff];train=train[train.target_available_h<=cutoff]
    for r in bad.itertuples():excluded.append({'sample_id':r.sample_id,'H':H,'fold':fold,'reason':'OUTER_TRAIN_LABEL_AFTER_FIRST_VALIDATION_ORIGIN'})
    par=train_new(train,H,fold,cfg['alpha_grid']);models.append(par);kind='NEW_HORIZON_EXPERIMENT'
   assert not len(val) or train.target_available_h.max()<=val.origin_available_h.min()
   cs(f'inputs/H{H}_{fold}_USED_TRAIN.csv',train);cs(f'inputs/H{H}_{fold}_USED_VALIDATION.csv',val)
   legal+=[train,val];ridge=replay(val,par);localtrend=trend(val.origin_R.to_numpy(),val.slope_R_per_h.to_numpy(),H,val.past_cadence_h.to_numpy())
   # Target conditions are assessed post hoc against Train ranges, never used in fitting or forecast inputs.
   flags=np.zeros(len(val),bool)
   for col in CONDITIONS:
    low=float(train['target_condition_AUDIT_ONLY_'+col].min());high=float(train['target_condition_AUDIT_ONLY_'+col].max());v=val['target_condition_AUDIT_ONLY_'+col].to_numpy();flags|=(v<low)|(v>high)
    condition_support.append({'H':H,'fold':fold,'condition':col,'Train_min':low,'Train_max':high,'Validation_outside_count':int(((v<low)|(v>high)).sum()),'Validation_n':len(val)})
   for method,pred in [('PERSISTENCE',val.origin_R.to_numpy()),('LOCAL_TREND96',localtrend),('RIDGE96',ridge)]:
    z=val.copy();z['method']=method;z['prediction_registration']=kind if method=='RIDGE96' else 'NEW_FIXED_BASELINE';z['predicted_R']=pred;z['error_mVeq']=(pred-z.target_R)*VREF*1000;z['absolute_error_mVeq']=abs(z.error_mVeq);z['target_condition_OOD_AUDIT_ONLY']=flags
    predictions.append(z)
   time.append({'H':H,'fold':fold,'Train_n':len(train),'Validation_n':len(val),'Train_max_label_available_h':float(train.target_available_h.max()),'first_Validation_origin_available_h':float(val.origin_available_h.min()) if len(val) else None,
    'inner_label_cutoff_h':par.get('inner_cutoff_h',par.get('inner_label_availability_cutoff_h')),'new_fits':5 if H in [6,12] else 0,'model_registration':kind})
   print('H',H,fold,'Train',len(train),'Validation',len(val),'new_fits',5 if H in [6,12] else 0,flush=True)
 pred=pd.concat(predictions,ignore_index=True);cs('outputs/PREDICTIONS.csv',pred);cs('outputs/USED_SAMPLES.csv',pd.concat(legal).drop_duplicates('sample_id'));cs('outputs/EXCLUDED_SAMPLES.csv',pd.DataFrame(excluded));cs('outputs/FROZEN_MODEL_UNAVAILABLE_ROWS.csv',pd.DataFrame(c_unavailable,columns=['sample_id','reason']))
 js('models/NEW_H6_H12_RIDGE_PARAMETERS.json',models);cs('outputs/PARAMETER_TIME_AUDIT.csv',pd.DataFrame(time));cs('outputs/TRAIN_ONLY_CONDITION_SUPPORT.csv',pd.DataFrame(condition_support))
 met=[];strata=[]
 for (H,fold,method),g in pred.groupby(['H','fold','method']):
  met.append({'scope':'own_horizon_cohort','H':H,'fold':fold,'method':method,**metric(g.target_R,g.predicted_R)})
  for col in ['future_observed_increase_AUDIT_ONLY','origin_observed_increase','target_condition_OOD_AUDIT_ONLY']:
   for state in [False,True]:
    z=g[g[col].eq(state)];strata.append({'H':H,'fold':fold,'method':method,'stratum':col,'state':state,**metric(z.target_R,z.predicted_R)})
 cs('outputs/STRATIFIED_METRICS.csv',pd.DataFrame(strata))
 common=[];counts=[]
 for fold in cfg['folds']:
  sets=[set(pred[pred.H.eq(h)&pred.fold.eq(fold)].origin_raw_cycle_id) for h in cfg['horizons']];joint=set.intersection(*sets);counts.append({'fold':fold,'common_origin_n_all4':len(joint)})
  for H in cfg['horizons']:
   for method in ['PERSISTENCE','LOCAL_TREND96','RIDGE96']:
    z=pred[pred.H.eq(H)&pred.fold.eq(fold)&pred.method.eq(method)&pred.origin_raw_cycle_id.isin(joint)];common.append(z)
    met.append({'scope':'common_origins_all4_horizons','H':H,'fold':fold,'method':method,**metric(z.target_R,z.predicted_R)})
 cs('outputs/COMMON_ORIGIN_COUNTS.csv',pd.DataFrame(counts));cs('outputs/COMMON_ORIGIN_PREDICTIONS.csv',pd.concat(common));metrics=pd.DataFrame(met);cs('outputs/FOLD_METRICS.csv',metrics)
 equal=[]
 for (scope,H,method),g in metrics.groupby(['scope','H','method']):
  full=len(g)==3 and (g.n>0).all();equal.append({'scope':scope,'H':H,'method':method,'threefold_evaluable':bool(full),'fold_counts':'|'.join(str(int(v)) for v in g.sort_values('fold').n),
   **{k:float(g[k].mean()) if full else None for k in ['MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq']}})
 cs('outputs/THREE_FOLD_EQUAL_MEANS.csv',pd.DataFrame(equal))
 lead=[]
 for (H,fold),g in pred[pred.method.eq('PERSISTENCE')].groupby(['H','fold']):
  lead.append({'H':H,'fold':fold,'n':len(g),'actual_operating_h_min':float(g.actual_lead_h_AUDIT_ONLY.min()),'actual_operating_h_median':float(g.actual_lead_h_AUDIT_ONLY.median()),'actual_operating_h_max':float(g.actual_lead_h_AUDIT_ONLY.max()),
   'actual_calendar_h_min':float(g.actual_calendar_lead_h_AUDIT_ONLY.min()),'actual_calendar_h_max':float(g.actual_calendar_lead_h_AUDIT_ONLY.max()),'causal_estimated_lead_MAE_h':float(abs(g.estimated_lead_h-g.actual_lead_h_AUDIT_ONLY).mean())})
 cs('outputs/ACTUAL_LEAD_AUDIT.csv',pd.DataFrame(lead))
 js('outputs/TRAJECTORY_COUNTS.json',{'real_distinct_middle50_measurements':len(traj),'complete_cycle_measurements':len(meas),'reference_warmup':len(warm),'causal_forecast_observations':len(ident),'partial_descriptive_measurements':len(partial),
  'physical_catalog_cycles':len(cat),'complete_catalog_cycles':int(cat.complete_cycle.sum()),'complete_cycles_without_cached_P1S':int(sum(r['status']=='NO_P1S_VALUE_IN_EXISTING_CACHE' for r in gaps)),
  'causal_observed_positive_steps':int((ident.R.diff()>0).sum()),'first_causal_actual_end_h':float(ident.measurement_end_h.min()),'last_causal_actual_end_h':float(ident.measurement_end_h.max()),'interpolated_health_labels':0})
 js('outputs/READ_LEDGER.json',sorted(set(ACCESS)));config()
 js('outputs/RUN_LEDGER.json',{'status':'COMPLETED','CPU':True,'new_H6_H12_final_fits':len(models),'new_H6_H12_inner_fits':sum(len(x['inner_records']) for x in models),'native_H1_H3_refits':0,'new_D_models':0,
  'prediction_rows':len(pred),'Test_read':False,'GPU_runs':0,'new_SOH_or_RUL_fits':0,'stop_after_delivery':True})
 print(pd.DataFrame(equal).to_string(index=False));print('RUN_COMPLETED')
if __name__=='__main__':
 firewall();{'freeze':freeze,'run':run}[sys.argv[1]]()

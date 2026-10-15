"""Preserve interrupted run; recover H6 via normal equations; skip unqualified H12 folds.

No historical fit and no second sklearn H6 fit. Original frozen code/protocol remain intact.
The original15 H6 sklearn fits were not serialized before H12 failed; this numerical
recovery uses the already saved exact Train sets and fixed alpha grid, without a new
experimental choice. Its15 linear-system evaluations are separately disclosed.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import math
from common import *
from methods import replay,trend,train_new
METHODS=['PERSISTENCE','LOCAL_TREND96','RIDGE96']
def closed_form_record(train,alpha):
 x=train[FEATURES].to_numpy();mean=x.mean(axis=0);scale=x.std(axis=0,ddof=0);scale=np.where(scale==0,1.,scale);z=(x-mean)/scale
 y=(train.target_R-train.origin_R).to_numpy();xm=z.mean(axis=0);ym=float(y.mean());xc=z-xm;yc=y-ym
 lhs=xc.T@xc+alpha*np.eye(len(FEATURES));coef=np.linalg.solve(lhs,xc.T@yc);intercept=ym-float(xm@coef)
 return dict(alpha=alpha,scaler_mean=mean.tolist(),scaler_scale=scale.tolist(),coef=coef.tolist(),intercept=intercept,
  sample_ids=train.sample_id.tolist(),max_label_available_h=float(train.target_available_h.max()),target_standardized=False,
  solver='NUMPY_NORMAL_EQUATION_RECOVERY_OF_UNSERIALIZED_H6',normal_equation_residual_max=float(np.max(abs(lhs@coef-xc.T@yc))))
def recover_h6(train,fold,grid):
 assert len(train)>=2;train=train.sort_values('target_actual_end_h');n=math.ceil(.15*len(train));select=train.iloc[-n:];before=train.iloc[:-n]
 cutoff=float(select.origin_available_h.min());fit=before[before.target_available_h<=cutoff];purged=before[before.target_available_h>cutoff];assert len(fit)>=2
 records=[]
 for a in grid:
  p=closed_form_record(fit,a);p['inner_MAE_R']=float(np.mean(abs(replay(select,p)-select.target_R.to_numpy())));records.append(p)
 best=min(records,key=lambda r:(r['inner_MAE_R'],r['alpha']));final=closed_form_record(train,best['alpha'])
 final.update(horizon=6,fold=fold,registration='NEW_H6_NUMERICAL_RECOVERY_SAME_FROZEN_EXPERIMENT',inner_records=records,
  inner_selection_ids=select.sample_id.tolist(),inner_purged_ids=purged.sample_id.tolist(),inner_cutoff_h=cutoff,
  final_fits=1,inner_fits=4,new_sklearn_fits_in_resume=0,original_unsaved_sklearn_fits=5,linear_system_recovery_evaluations=5)
 return final
def run():
 cfg=config();assert not (P/'outputs/RESUME_COMPLETE.json').exists(),'Completed; verify only'
 assert local('outputs/RUN_LEDGER.json')['status']=='RUNNING'
 out('outputs/ORIGINAL_RUN_LEDGER_AT_INTERRUPTION.json').write_bytes((P/'outputs/RUN_LEDGER.json').read_bytes())
 js('configs/RESUME_CODE_IDENTITY.json',{'source_code_sha256':sha(__file__),'original_protocol_sha256':sha(P/'configs/FROZEN_PROTOCOL.json'),
  'reason':'Original executor asserted on H12 fold1 insufficient inner-fit labels instead of marking NA; preserve protocol and original code',
  'H6_recovery':'Exact saved Train tables, normal equations for same grid; no sklearn refit, separately count15 linear-system evaluations',
  'H12_policy':'Insufficient inner Train or empty Validation -> no fit and NA; other qualified fold may execute once','no_rule_relaxation':True})
 d=pd.read_csv(P/'outputs/INITIAL_QUALIFIED_SAMPLES.csv',float_precision='round_trip');pars=jl('native_parameters')
 predictions=[];used=[];models=[];time=[];unavailable=[];support=[]
 for H in cfg['horizons']:
  for fold in cfg['folds']:
   if H in [1,3,6]:
    train=pd.read_csv(P/f'inputs/H{H}_{fold}_USED_TRAIN.csv',float_precision='round_trip');val=pd.read_csv(P/f'inputs/H{H}_{fold}_USED_VALIDATION.csv',float_precision='round_trip')
   else:
    sub=d[d.H.eq(H)&d.fold.eq(fold)];train=sub[sub.role.eq('train')].sort_values('target_actual_end_h');val=sub[sub.role.eq('validation')].sort_values('origin_available_h')
    cutoff=float(val.origin_available_h.min()) if len(val) else cfg['folds'][fold][1]
    purged=train[train.target_available_h>cutoff];train=train[train.target_available_h<=cutoff]
    for row in purged.itertuples():unavailable.append({'H':H,'fold':fold,'sample_id':row.sample_id,'reason':'OUTER_TRAIN_LABEL_NOT_AVAILABLE'})
    cs(f'inputs/H{H}_{fold}_USED_TRAIN.csv',train);cs(f'inputs/H{H}_{fold}_USED_VALIDATION.csv',val)
   eligible_ridge=True;status=''
   n=math.ceil(.15*len(train));selection=train.iloc[-n:];inner=train.iloc[:-n];innercut=float(selection.origin_available_h.min()) if len(selection) else None
   inner=inner[inner.target_available_h<=innercut] if innercut is not None else inner.iloc[:0]
   if H in [1,3]:
    par=next(p for p in pars if p['horizon']==f'H{H}' and p['fold']==fold);status='FROZEN_NATIVE_COEFFICIENT_REPLAY';fits_now=0
   elif H==6:par=recover_h6(train,fold,cfg['alpha_grid']);models.append(par);status='H6_NORMAL_EQUATION_RECOVERY';fits_now=0
   elif not len(val) or len(inner)<2 or not len(selection):
    eligible_ridge=False;par=None;fits_now=0;status='NO_VALIDATION' if not len(val) else 'INSUFFICIENT_INNER_TRAIN_LABEL_ISOLATION'
   else:
    par=train_new(train,H,fold,cfg['alpha_grid']);models.append(par);status='NEW_H12_SINGLE_QUALIFIED_FOLD';fits_now=5
   if not eligible_ridge:unavailable.append({'H':H,'fold':fold,'sample_id':'ALL_RIDGE_VALIDATION','reason':status})
   assert not len(val) or train.target_available_h.max()<=val.origin_available_h.min()
   used+=[train,val];flags=np.zeros(len(val),bool)
   for col in CONDITIONS:
    low=float(train['target_condition_AUDIT_ONLY_'+col].min());high=float(train['target_condition_AUDIT_ONLY_'+col].max());v=val['target_condition_AUDIT_ONLY_'+col].to_numpy();flag=(v<low)|(v>high);flags|=flag
    support.append({'H':H,'fold':fold,'condition':col,'Train_min':low,'Train_max':high,'Validation_outside_count':int(flag.sum()),'Validation_n':len(val)})
   outputs=[('PERSISTENCE',val.origin_R.to_numpy()),('LOCAL_TREND96',trend(val.origin_R.to_numpy(),val.slope_R_per_h.to_numpy(),H,val.past_cadence_h.to_numpy()))]
   if eligible_ridge:outputs.append(('RIDGE96',replay(val,par)))
   for method,pred in outputs:
    z=val.copy();z['method']=method;z['prediction_registration']=status if method=='RIDGE96' else 'NEW_FIXED_BASELINE';z['predicted_R']=pred;z['error_mVeq']=(pred-z.target_R)*VREF*1000;z['absolute_error_mVeq']=abs(z.error_mVeq);z['target_condition_OOD_AUDIT_ONLY']=flags;predictions.append(z)
   time.append({'H':H,'fold':fold,'Train_n':len(train),'Validation_n':len(val),'Ridge_qualified':eligible_ridge,'model_status':status,
    'Train_max_label_available_h':float(train.target_available_h.max()),'first_Validation_origin_available_h':float(val.origin_available_h.min()) if len(val) else None,
    'inner_fit_n_diagnostic':len(inner),'inner_selection_n_diagnostic':len(selection),'inner_cutoff_h_diagnostic':innercut,'new_sklearn_fits_in_resume':fits_now})
   print(H,fold,'Train',len(train),'Val',len(val),'Ridge',status,'new sklearn fits',fits_now,flush=True)
 pred=pd.concat(predictions,ignore_index=True);cs('outputs/PREDICTIONS.csv',pred);cs('outputs/USED_SAMPLES.csv',pd.concat(used).drop_duplicates('sample_id'));cs('outputs/RIDGE_UNAVAILABLE_AND_PURGES.csv',pd.DataFrame(unavailable));cs('outputs/PARAMETER_TIME_AUDIT.csv',pd.DataFrame(time));cs('outputs/TRAIN_ONLY_CONDITION_SUPPORT.csv',pd.DataFrame(support));js('models/NEW_H6_H12_RIDGE_PARAMETERS.json',models)
 met=[];strata=[]
 for H in cfg['horizons']:
  for fold in cfg['folds']:
   for method in METHODS:
    g=pred[pred.H.eq(H)&pred.fold.eq(fold)&pred.method.eq(method)];met.append({'scope':'own_horizon_cohort','H':H,'fold':fold,'method':method,**metric(g.target_R,g.predicted_R)})
    for col in ['future_observed_increase_AUDIT_ONLY','origin_observed_increase','target_condition_OOD_AUDIT_ONLY']:
     for state in [False,True]:
      z=g[g[col].eq(state)];strata.append({'H':H,'fold':fold,'method':method,'stratum':col,'state':state,**metric(z.target_R,z.predicted_R)})
 cs('outputs/STRATIFIED_METRICS.csv',pd.DataFrame(strata));common=[];counts=[]
 for fold in cfg['folds']:
  sets=[set(pred[pred.H.eq(h)&pred.fold.eq(fold)&pred.method.eq('PERSISTENCE')].origin_raw_cycle_id) for h in cfg['horizons']];joint=set.intersection(*sets);counts.append({'fold':fold,'common_origin_n_all4':len(joint)})
  for H in cfg['horizons']:
   for method in METHODS:
    z=pred[pred.H.eq(H)&pred.fold.eq(fold)&pred.method.eq(method)&pred.origin_raw_cycle_id.isin(joint)];common.append(z);met.append({'scope':'common_origins_all4_horizons','H':H,'fold':fold,'method':method,**metric(z.target_R,z.predicted_R)})
 cs('outputs/COMMON_ORIGIN_COUNTS.csv',pd.DataFrame(counts));cs('outputs/COMMON_ORIGIN_PREDICTIONS.csv',pd.concat(common));metrics=pd.DataFrame(met);cs('outputs/FOLD_METRICS.csv',metrics);equal=[]
 for (scope,H,method),g in metrics.groupby(['scope','H','method']):
  full=len(g)==3 and (g.n>0).all();equal.append({'scope':scope,'H':H,'method':method,'threefold_evaluable':bool(full),'fold_counts':'|'.join(str(int(v)) for v in g.sort_values('fold').n),
   **{k:float(g[k].mean()) if full else None for k in ['MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq']}})
 cs('outputs/THREE_FOLD_EQUAL_MEANS.csv',pd.DataFrame(equal));lead=[]
 for (H,fold),g in pred[pred.method.eq('PERSISTENCE')].groupby(['H','fold']):
  lead.append({'H':H,'fold':fold,'n':len(g),'actual_operating_h_min':float(g.actual_lead_h_AUDIT_ONLY.min()),'actual_operating_h_median':float(g.actual_lead_h_AUDIT_ONLY.median()),'actual_operating_h_max':float(g.actual_lead_h_AUDIT_ONLY.max()),
   'actual_calendar_h_min':float(g.actual_calendar_lead_h_AUDIT_ONLY.min()),'actual_calendar_h_max':float(g.actual_calendar_lead_h_AUDIT_ONLY.max()),'causal_estimated_lead_MAE_h':float(abs(g.estimated_lead_h-g.actual_lead_h_AUDIT_ONLY).mean())})
 cs('outputs/ACTUAL_LEAD_AUDIT.csv',pd.DataFrame(lead))
 ident=source('identity').sort_values('measurement_end_h');meas=source('measurement');meas=meas[meas.stable_window.eq('middle_50')];partial=source('partial');cat=source('catalog');warm=meas[meas.is_reference_complete_interval]
 js('outputs/TRAJECTORY_COUNTS.json',{'real_distinct_middle50_measurements':len(meas)+len(partial),'complete_cycle_measurements':len(meas),'reference_warmup':len(warm),'causal_forecast_observations':len(ident),'partial_descriptive_measurements':len(partial),
  'physical_catalog_cycles':len(cat),'complete_catalog_cycles':int(cat.complete_cycle.sum()),'complete_cycles_without_cached_P1S':int(sum(cat.complete_cycle & ~cat.raw_cycle_id.isin(meas.raw_cycle_id))),
  'causal_observed_positive_steps':int((ident.R.diff()>0).sum()),'first_causal_actual_end_h':float(ident.measurement_end_h.min()),'last_causal_actual_end_h':float(ident.measurement_end_h.max()),'interpolated_health_labels':0})
 cs('outputs/MISSING_RIDGE_SCORES_ARE_NA.csv',pd.DataFrame(time)[['H','fold','Ridge_qualified','model_status','Train_n','Validation_n','inner_fit_n_diagnostic']])
 config();js('outputs/READ_LEDGER_RESUME.json',sorted(set(ACCESS)))
 ledger={'status':'COMPLETED_WITH_EXPLICIT_NOT_EVALUABLE_FOLDS','CPU':True,'original_H6_sklearn_fits_before_interruption':15,
  'H6_numerical_recovery_linear_system_evaluations':15,'new_H12_sklearn_fits_in_resume':5,'final_H6_models':3,'final_H12_models':1,
  'native_H1_H3_refits':0,'new_D_models':0,'prediction_rows':len(pred),'Test_read':False,'GPU_runs':0,'new_SOH_or_RUL_fits':0,'H12_fold1_Ridge':'INNER_LABEL_ISOLATION_FAIL','H12_fold2':'EMPTY_VALIDATION'}
 js('outputs/RUN_LEDGER.json',ledger);js('outputs/RESUME_COMPLETE.json',ledger);print(pd.DataFrame(equal).to_string(index=False));print('RESUME_COMPLETE')
if __name__=='__main__':firewall();run()

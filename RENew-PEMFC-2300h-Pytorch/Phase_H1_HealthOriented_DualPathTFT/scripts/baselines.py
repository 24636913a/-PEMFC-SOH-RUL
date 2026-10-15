"""Same-event Persistence and new Train-only Ridge96; old coefficients untouched."""
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
FEATURES=['Ridge_origin_R','Ridge_mean_R','Ridge_std_R','Ridge_slope_R_per_h','Ridge_origin_minus_median_R']
def main():
 assert not (P/'outputs/BASELINES_COMPLETE.json').exists(),'Baselines already fitted'
 cfg=load_cfg();assert protected_pass();samples=rd(P/'data_contracts/USED_SAMPLES.csv');inner=rd(P/'data_contracts/INNER_SPLITS.csv')
 allpred=[];records={}
 for fold,g in samples.groupby('fold'):
  started=time.perf_counter();tr=g.loc[g.role.eq('train')];va=g.loc[g.role.eq('validation')];keys=inner.loc[inner.fold.eq(fold)]
  fit=tr.loc[tr.sample_key.isin(keys.loc[keys.inner_role.eq('fit'),'sample_key'])];sel=tr.loc[tr.sample_key.isin(keys.loc[keys.inner_role.eq('selection'),'sample_key'])]
  ix=fit[FEATURES].to_numpy();iy=(fit.true_voltage_V-fit.origin_voltage_V).to_numpy()/VREF
  sx=sel[FEATURES].to_numpy();sy=(sel.true_voltage_V-sel.origin_voltage_V).to_numpy()/VREF
  scaler=StandardScaler().fit(ix);scores=[]
  for alpha in [.0001,.01,1.,100.]:
   model=Ridge(alpha=alpha,solver='cholesky').fit(scaler.transform(ix),iy)
   scores.append(dict(alpha=alpha,inner_MAE_R=float(np.mean(abs(model.predict(scaler.transform(sx))-sy)))))
  alpha=min(scores,key=lambda r:(r['inner_MAE_R'],r['alpha']))['alpha']
  outer=StandardScaler().fit(tr[FEATURES]);y=(tr.true_voltage_V-tr.origin_voltage_V).to_numpy()/VREF
  ridge=Ridge(alpha=alpha,solver='cholesky').fit(outer.transform(tr[FEATURES]),y)
  pred=va.origin_voltage_V.to_numpy()+VREF*ridge.predict(outer.transform(va[FEATURES]))
  elapsed=time.perf_counter()-started;available=pd.to_datetime(tr.target_available_calendar).max()+pd.Timedelta(seconds=elapsed)
  assert available<pd.to_datetime(va.origin_confirmed_calendar).min()
  records[fold]=dict(alpha=alpha,grid=scores,features=FEATURES,train_sample_keys=tr.sample_key.tolist(),inner_fit_keys=fit.sample_key.tolist(),inner_selection_keys=sel.sample_key.tolist(),
   inner_mean=scaler.mean_.tolist(),inner_scale=scaler.scale_.tolist(),outer_mean=outer.mean_.tolist(),outer_scale=outer.scale_.tolist(),
   coef_deltaR=ridge.coef_.tolist(),intercept_deltaR=float(ridge.intercept_),fitting_seconds=elapsed,
   replay_model_available_calendar=str(available),ridge_fit_passes=5,selection='Train_inner_MAE_only')
  for name,p in [('Persistence',va.origin_voltage_V.to_numpy()),('Ridge96_control',pred)]:
   d=va.copy();d['model']=name;d['seed']=-1;d['run_id']=fold+'_'+name;d['predicted_voltage_V']=p
   d['predicted_SOH_V_percent']=100*d.predicted_voltage_V/VREF;d['true_SOH_V_percent']=100*d.true_voltage_V/VREF
   d['replay_model_available_calendar']=str(available) if name=='Ridge96_control' else d.origin_confirmed_calendar
   d['historical_online_deployment_claimed']=False;allpred.append(d)
 cs('outputs/BASELINE_PREDICTIONS.csv',pd.concat(allpred,ignore_index=True));js('models/RIDGE96_CONTROL_PARAMETERS.json',records)
 js('outputs/BASELINES_COMPLETE.json',dict(status='COMPLETE',CPU_Ridge_fit_passes=15,old_Ridge_untouched=True,TFT_history_h=24,Ridge_history_h=96))
 print('BASELINES COMPLETE:15 CPU fits; no history mutations')
if __name__=='__main__':main()

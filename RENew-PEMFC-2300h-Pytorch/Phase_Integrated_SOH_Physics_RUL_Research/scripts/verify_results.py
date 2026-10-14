"""Independent metric and parameter recomputation; no training/import of experiment functions."""
import sys
from common import *
def verify_a():
 cfg=config();pairs=pd.read_csv(P/'data_contracts/NEAREST_COMPLETE_CALIBRATION_PAIRS.csv');models=json.loads((P/'models/PAST_ONLY_AFFINE_PARAMETERS.json').read_text())
 out=pd.read_csv(P/'outputs/SOH_EVENT_PREDICTIONS.csv');met=pd.read_csv(P/'outputs/SOH_EVENT_METRICS.csv');links=pd.read_csv(P/'outputs/P1S_TO_SOH_EXACT_FORECAST_LINKS.csv')
 worst=0.
 for m in models:
  past=pairs[pairs.nominal_h<m['nominal_h']];assert past.nominal_h.tolist()==m['training_events']
  a,b=np.linalg.lstsq(np.column_stack([past.R_nearest_complete,np.ones(len(past))]),past.Y,rcond=None)[0]
  y=a*m['R_input_forecast']+b;z=out[out.nominal_h.eq(m['nominal_h'])&out.horizon.eq(m['horizon'])&out.model.eq('M1_RIDGE96_M2_NC')].iloc[0]
  worst=max(worst,abs(y-z.predicted_SOH));assert abs(y-z.predicted_SOH)<1e-12
 assert len(links)==6 and (links.max_Train_label_end_h<=links.origin_conservative_available_h).all()
 assert (links.inner_train_max_label_h<=links.inner_selection_first_origin_h).all()
 assert (pd.to_datetime(links.prior_reference_end_max_calendar)<pd.to_datetime(links.origin_calendar)).all()
 assert (links.target_raw_cycle_id-links.origin_raw_cycle_id==links.horizon.map({'H1':1,'H3':3})).all()
 for _,r in met.iterrows():
  z=out[out.horizon.eq(r.horizon)&out.model.eq(r.model)];e=(z.predicted_SOH-z.true_SOH).to_numpy()*100
  assert len(e)==r.n;assert abs(np.abs(e).mean()-r.MAE_SOH_pp)<1e-12;assert abs(np.sqrt(np.mean(e*e))-r.RMSE_SOH_pp)<1e-12
 original=[];commonkeys=None;commontruth=None
 for model in ['F0','F1','F2']:
  for seed in [42,52,62]:
   d=source(f'{model}_{seed}');v=(d.y_pred_delta_V-d.y_true_delta_V).to_numpy()*1000
   keys=d[['origin_row','target_row','horizon_step']].to_numpy();truth=d.y_true_delta_V.to_numpy()
   if commonkeys is None:commonkeys=keys;commontruth=truth
   else:assert np.array_equal(keys,commonkeys) and np.array_equal(truth,commontruth)
   rmse=float(np.sqrt(np.mean(v*v)));mae=float(np.abs(v).mean());ref=source('o1metrics');row=ref[ref.model.eq(model)&ref.seed.eq(seed)].iloc[0]
   target_equal=float(np.sqrt(pd.DataFrame({'target':d.target_row,'SE':v*v}).groupby('target').SE.mean().mean()))
   end=v[d.horizon_step.to_numpy()==24];endpoint=float(np.sqrt(np.mean(end*end)));tail=float(np.max(np.abs(v)))
   assert abs(target_equal-row.target_time_equal_RMSE_mV)<1e-9 and abs(endpoint-row.endpoint_RMSE_mV)<1e-9 and abs(tail-row.max_abs_error_mV)<1e-9
   assert abs(rmse-row.RMSE_mV)<1e-9 and abs(mae-row.MAE_mV)<1e-9
   original.append({'model':model,'seed':seed,'RMSE_mV':rmse,'MAE_mV':mae,'target_time_equal_RMSE_mV':target_equal,'endpoint_RMSE_mV':endpoint,'max_abs_error_mV':tail,'source':'H Validation cache; independently recomputed'})
 cs('outputs/RECOMPUTED_DYNAMIC_METRICS.csv',pd.DataFrame(original))
 p=source('preds');fm=source('p1folds');check=[]
 for (h,f,m),z in p.groupby(['horizon','fold','model']):
  v=(z.predicted_R_P1S_causal-z.true_R_P1S_causal).to_numpy();mae=float(np.abs(v).mean()*.71383023635*1000)
  row=fm[fm.horizon.eq(h)&fm.fold.eq(f)&fm.model.eq(m)].iloc[0];assert abs(mae-row.MAE_mV_equivalent)<1e-10
  check.append({'horizon':h,'fold':f,'model':m,'n':len(z),'MAE_mV_equivalent':mae,'RMSE_mV_equivalent':float(np.sqrt(np.mean(v*v))*.71383023635*1000)})
 cs('outputs/RECOMPUTED_P1S_FOLD_METRICS.csv',pd.DataFrame(check))
 js('outputs/STAGE_A_INDEPENDENT_VERIFICATION.json',{'PASS':True,'affine_matrix_vs_centered_prediction_max_diff':worst,'SOH_predictions':len(out),'SOH_exact_links':len(links),'dynamic_runs_recomputed':9,'P1S_cached_predictions_recomputed':len(p),'no_refits':True,'source_SHA_pass':True})
 print('A independently verified')
def verify_final():
 verify_a()
 b=json.loads((P/'outputs/PHYSICS_CPU_DIAGNOSTICS.json').read_text());assert b['state_nullity']==b['Train_n'];assert b['observed_recovery_steps']>0
 assert b['slow_prior_vs_smoothing_loss_diff']==0 and b['slow_prior_vs_smoothing_gradient_max_diff']==0
 assert b['real_certified_physics_loss'] is None and b['real_certified_model_parameter_gradient'] is None
 scenarios=pd.read_csv(P/'outputs/CONDITIONAL_CALENDAR_LIFETIME_SCENARIOS.csv');slopes=pd.read_csv(P/'outputs/CONDITIONAL_TRAJECTORY_SLOPES.csv');y=float(source('refs').iloc[-1].Y)
 for _,r in scenarios.iterrows():
  slope=float(slopes.loc[slopes.model.eq(r.model),'slope_SOH_per_calendar_h'].iloc[0]);expected=(float(r.threshold)-y)/slope if slope<0 else np.nan
  if slope<0:assert abs(expected-r.conditional_crossing_calendar_h)<1e-6
  else:assert pd.isna(r.conditional_crossing_calendar_h)
 decision=json.loads((P/'outputs/FINAL_RESEARCH_DECISION.json').read_text());assert decision['GPU_training_runs']==0 and decision['real_RUL_accuracy'] is None
 band=pd.read_csv(P/'outputs/CONDITIONAL_TRAJECTORY_SENSITIVITY_ENVELOPE.csv');loo=pd.read_csv(P/'outputs/CONDITIONAL_SLOPE_SENSITIVITY.csv')
 all_slopes=np.r_[slopes.slope_SOH_per_calendar_h.to_numpy(),loo.slope_SOH_per_calendar_h.to_numpy()]
 np.testing.assert_allclose(band.low_SOH,y+all_slopes.min()*band.elapsed_calendar_h,atol=1e-14,rtol=0)
 np.testing.assert_allclose(band.high_SOH,y+all_slopes.max()*band.elapsed_calendar_h,atol=1e-14,rtol=0)
 js('outputs/INDEPENDENT_VERIFICATION.json',{'PASS':True,'A_verified':True,'B_identifiability_and_matched_smoothing_verified':True,'C_scenario_algebra_verified':True,'all_source_hashes_unchanged':True,'no_training_repeated':True,'no_Test':True})
 print('Final independent recomputation PASS')
if __name__=='__main__':
 firewall();{'a':verify_a,'final':verify_final}[sys.argv[1]]()

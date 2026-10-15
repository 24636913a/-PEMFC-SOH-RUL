"""Three fixed methods; no refit, no hyperparameters, no inferential significance test."""
import sys
from common import *
def freeze():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists(),'Already frozen'
 c={'base_commit':'5141b79d7116d3e9721fe1bd626ab97412b3d485',
  'source_sha256':{s:sha(ROOT/s) for s in SOURCES.values()},
  'audit_code_sha256':{str(q.relative_to(P)).replace('\\','/'):sha(q) for q in [P/'scripts/common.py',P/'scripts/run_audit.py']},
  'events':[1150,1500,1750],'horizons':['H1','H3'],'event_selection':'Reuse six exact links; retrospective scheduled-event replay',
  'calibration':'Exactly frozen event/horizon M2_NC a,b; no refit or re-selection',
  'methods':METHODS,'origin_value':'Latest legal complete-cycle middle_50 measured R, available by linked origin; not earlier reference, not future target',
  'error_units':'SOH percentage points = (prediction - reference)*100; P1S mV-equivalent uses0.71383023635V*1000',
  'strict_identity':['horizon','fold','origin_raw_cycle_id','target_raw_cycle_id','origin_cycle_interval_id','target_cycle_interval_id'],
  'numerical_comparison':'Report float/serialization differences; absolute tolerance1e-12 retention or SOHpp for independently recomputed formula results',
  'tie_tolerance_SOH_pp':1e-12,'decomposition':'e_SOH = 100*(a*R_target+b-Y) + 100*a*(R_prediction-R_target)',
  'actual_target_R_role':'Diagnostic residual decomposition only; cannot compete as a deployable model',
  'contribution':'Sequential MAE difference M0 -> measured-origin+calibration -> Ridge+same-calibration; descriptive, not unique causal attribution',
  'statistical_independence':'Same3 events reused acrossH1/H3, not6 independent outcomes; no p-values/bootstrap/significance test',
  'paper_policy':'Export only after independent recomputation, tests, explicit claim boundaries and visual review; new unique subdirectory, never overwrite',
  'reference_release_known':False,'strict_online_SOH':False,'no_Test':True,'no_training':True,'no_GPU':True}
 js('configs/FROZEN_PROTOCOL.json',c);md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'))
 for key,s in SOURCES.items():
  output('inputs/'+key+Path(s).suffix).write_bytes((ROOT/s).read_bytes())
 cs('inputs/SOURCE_SHA256.csv',pd.DataFrame([{'key':k,'source':s,'sha256':c['source_sha256'][s]} for k,s in SOURCES.items()]))
 md('.gitignore','__pycache__/\n*.pyc\noutputs/GIT_RECEIPT.local.json')
 md('.gitattributes','* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.png binary\n*.pdf binary')
 print('FROZEN',sha(P/'configs/FROZEN_PROTOCOL.json'))
def summarize(frame,true,pred,unit):
 err=(frame[pred]-frame[true]).to_numpy(float)*unit
 return {'n':len(err),'MAE':float(np.mean(abs(err))),'RMSE':float(np.sqrt(np.mean(err**2))),'max_abs':float(np.max(abs(err)))}
def run():
 cfg=config();assert not (P/'outputs/COMPLETE.json').exists(),'Completed, verify without repeating'
 cache=source('cache');links=source('links');cal=load('calibration');ident=source('identity');measure=source('measurement');measure=measure[measure.stable_window.eq('middle_50')]
 refs=source('refs');timing=source('timing');old=source('old_predictions');oldmet=source('old_metrics');ta=source('ridge_time');pars=load('ridge_parameters')
 assert len(cache)==188 and len(links)==6 and len(cal)==6 and ident.cycle_interval_id.is_unique and len(ident)==131 and len(measure)==141
 assert set(links.nominal_h)==set(cfg['events']) and set(links.horizon)==set(cfg['horizons'])
 assert not links.duplicated(['nominal_h','horizon']).any()
 full_checks=[]
 for (h,fold),g in cache[cache.model.eq('RIDGE_96H')].groupby(['horizon','fold']):
  par=next(x for x in pars if x['horizon']==h and x['fold']==fold)
  rp=g.origin_R_P1S_causal.to_numpy()+((g[par['features']].to_numpy()-np.array(par['train_scaler_mean']))/np.array(par['train_scaler_scale']))@np.array(par['coef_standardized_X'])+par['intercept_deltaR']
  diff=float(np.max(np.abs(rp-g.predicted_R_P1S_causal.to_numpy())));assert diff<1e-12
  full_checks.append({'horizon':h,'fold':fold,'n':len(g),'max_Ridge_coefficient_replay_difference_R':diff,'fitted':False})
 points=[];decomp=[];availability=[];matched=[];replay=[]
 for _,link in links.iterrows():
  n=int(link.nominal_h);h=link.horizon;m=next(x for x in cal if x['nominal_h']==n and x['horizon']==h)
  q=cache[cache.horizon.eq(h)&cache.fold.eq(link.fold)&cache.origin_raw_cycle_id.eq(link.origin_raw_cycle_id)&cache.target_raw_cycle_id.eq(link.target_raw_cycle_id)]
  assert len(q)==2 and set(q.model)=={'PERSISTENCE','RIDGE_96H'}
  ridge=q[q.model.eq('RIDGE_96H')].iloc[0];persist=q[q.model.eq('PERSISTENCE')].iloc[0]
  for key in cfg['strict_identity']:
   if key in ridge.index:assert ridge[key]==link[key],key
  assert ridge.target_raw_cycle_id-ridge.origin_raw_cycle_id=={'H1':1,'H3':3}[h]
  row=ident[ident.cycle_interval_id.eq(link.origin_cycle_interval_id)].iloc[0];tr=ident[ident.cycle_interval_id.eq(link.target_cycle_interval_id)].iloc[0]
  ms=measure[measure.cycle_interval_id.eq(link.origin_cycle_interval_id)].iloc[0]
  assert int(row.raw_cycle_id)==int(link.origin_raw_cycle_id) and int(tr.raw_cycle_id)==int(link.target_raw_cycle_id)
  latest=ident[ident.conservative_available_end_h<=link.origin_conservative_available_h].sort_values('conservative_available_end_h').iloc[-1]
  assert latest.cycle_interval_id==row.cycle_interval_id
  assert row.measurement_end_h<=row.conservative_available_end_h<=link.origin_conservative_available_h
  assert pd.Timestamp(ms.measurement_end_datetime)==pd.Timestamp(link.origin_calendar)
  numeric_diffs={'R_origin_cache_diff':abs(row.R-ridge.origin_R_P1S_causal),'origin_P1S_persistence_cache_diff':abs(row.R-persist.predicted_R_P1S_causal),
   'origin_voltage_normalization_diff':abs(row.R-ms.P1S_voltage_V/.71383023635),'target_R_cache_diff':abs(tr.R-ridge.true_R_P1S_causal),
   'Ridge_exact_link_diff':abs(ridge.predicted_R_P1S_causal-link.Ridge_recomputed_prediction),'Ridge_calibration_input_record_diff':abs(ridge.predicted_R_P1S_causal-m['R_input_forecast'])}
  assert max(numeric_diffs.values())<1e-12
  audit=ta[ta.horizon.eq(h)&ta.fold.eq(link.fold)].iloc[0]
  assert audit.train_max_label_end_h<=link.origin_conservative_available_h and audit.inner_max_label_end_h<=audit.first_inner_selection_origin_h
  assert all(x<n for x in m['training_events']) and m['past_only'] and len(m['training_events'])==int(link.calibration_prior_count)
  assert pd.Timestamp(link.prior_reference_end_max_calendar)<pd.Timestamp(link.origin_calendar)
  tref=timing[timing.checkpoint_h.eq(n)].iloc[0];assert pd.Timestamp(link.origin_calendar)<pd.Timestamp(tref.polarization_first_datetime)
  true=float(refs.loc[refs.nominal_h.eq(n),'Y'].iloc[0]);prev=float(refs[refs.nominal_h<n].iloc[-1].Y)
  a=float(m['a']);b=float(m['b']);ro=float(row.R);rp=float(ridge.predicted_R_P1S_causal);rt=float(tr.R)
  pred=methods(prev,ro,rp,a,b)
  base={'nominal_h':n,'horizon':h,'fold':link.fold,'origin_raw_cycle_id':int(link.origin_raw_cycle_id),'target_raw_cycle_id':int(link.target_raw_cycle_id),
   'origin_cycle_interval_id':row.cycle_interval_id,'target_cycle_interval_id':tr.cycle_interval_id,'origin_calendar':link.origin_calendar,
   'origin_actual_P1S_measurement_start_h':float(row.measurement_start_h),'origin_actual_P1S_measurement_end_h':float(row.measurement_end_h),
   'origin_conservative_available_h':float(row.conservative_available_end_h),'origin_recorded_available_calendar':ms.measurement_end_datetime,
   'origin_exact_stable_measurement_end_calendar':None,'calendar_precision_note':'Recorded conservative dwell end; exact stable subwindow calendar end is not separately saved',
   'reference_start_calendar':tref.polarization_first_datetime,'reference_end_calendar':tref.polarization_last_datetime,'reference_release_calendar':None,
   'origin_measured_R':ro,'ridge_predicted_R':rp,'calibration_a':a,'calibration_b':b,'calibration_training_events':'|'.join(str(x) for x in m['training_events']),
   'true_SOH':true,'strict_online_qualified':False,'evaluation':'Exploratory retrospective scheduled-event replay'}
  for method,value in pred.items():points.append({**base,'method':method,'predicted_SOH':value,'error_SOH_pp':(value-true)*100,'absolute_error_SOH_pp':abs(value-true)*100})
  for oldname,newname in [('M0_REFERENCE_PERSISTENCE',METHODS[0]),('M1_RIDGE96_M2_NC',METHODS[2])]:
   o=old[old.nominal_h.eq(n)&old.horizon.eq(h)&old.model.eq(oldname)].iloc[0];delta=abs(float(o.predicted_SOH)-pred[newname]);assert delta<1e-12
   replay.append({'nominal_h':n,'horizon':h,'model':oldname,'prediction_difference_R':delta,'error_difference_SOH_pp':abs(float(o.error_SOH_pp)-(pred[newname]-true)*100)})
  d=decomposition(true,ro,rp,rt,a,b)
  assert abs(d['ridge_final_error_SOH_pp']-(pred[METHODS[2]]-true)*100)<1e-12
  assert abs(d['origin_final_error_SOH_pp']-(pred[METHODS[1]]-true)*100)<1e-12
  decomp.append({'nominal_h':n,'horizon':h,'a':a,'b':b,'origin_R':ro,'ridge_R':rp,'actual_future_target_R_AUDIT_ONLY':rt,
   'true_SOH':true,'ridge_proxy_error_R':rp-rt,'origin_proxy_error_R':ro-rt,
   'ridge_proxy_abs_error_mV_equivalent':abs(rp-rt)*.71383023635*1000,'origin_proxy_abs_error_mV_equivalent':abs(ro-rt)*.71383023635*1000,**d})
  availability.append({**{k:base[k] for k in ['nominal_h','horizon','origin_cycle_interval_id','origin_calendar','origin_actual_P1S_measurement_end_h','origin_conservative_available_h','origin_recorded_available_calendar']},
   'latest_eligible_observation_is_origin':True,'Train_max_label_h':float(audit.train_max_label_end_h),'inner_Train_max_label_h':float(audit.inner_max_label_end_h),
   'inner_selection_first_origin_h':float(audit.first_inner_selection_origin_h),'prior_reference_end_max_calendar':link.prior_reference_end_max_calendar,
   'measurement_already_available':True,'no_future_actual_prediction_inputs':True,**numeric_diffs})
  for _,r in q.iterrows():matched.append(r.to_dict()|{'nominal_h':n})
 out=pd.DataFrame(points);dec=pd.DataFrame(decomp);cs('outputs/POINT_PREDICTIONS.csv',out);cs('outputs/ERROR_DECOMPOSITION.csv',dec)
 cs('outputs/ORIGIN_MEASUREMENT_TIME_AND_IDENTITY_AUDIT.csv',pd.DataFrame(availability));cs('outputs/EXACT_MATCHED_P1S_CACHE_ROWS.csv',pd.DataFrame(matched))
 cs('outputs/HISTORICAL_RESULT_REPLAY.csv',pd.DataFrame(replay));cs('outputs/FULL_CACHE_COEFFICIENT_REPLAY.csv',pd.DataFrame(full_checks))
 metrics=[]
 for (h,method),g in out.groupby(['horizon','method']):
  z=summarize(g,'true_SOH','predicted_SOH',100);metrics.append({'horizon':h,'method':method,'n_events':z['n'],'MAE_SOH_pp':z['MAE'],'RMSE_SOH_pp':z['RMSE'],'max_abs_error_SOH_pp':z['max_abs']})
 met=pd.DataFrame(metrics);cs('outputs/SOH_METRICS.csv',met)
 pair=[];pointwins=[];contrib=[]
 for h in ['H1','H3']:
  e=out[out.horizon.eq(h)].pivot(index='nominal_h',columns='method',values='absolute_error_SOH_pp')
  for candidate,baseline in [(METHODS[1],METHODS[0]),(METHODS[2],METHODS[0]),(METHODS[2],METHODS[1])]:
   delta=e[baseline]-e[candidate];pair.append({'horizon':h,'candidate':candidate,'baseline':baseline,'n_paired_events':len(e),
    'candidate_wins':int((delta>cfg['tie_tolerance_SOH_pp']).sum()),'ties':int((abs(delta)<=cfg['tie_tolerance_SOH_pp']).sum()),'candidate_losses':int((delta<-cfg['tie_tolerance_SOH_pp']).sum()),
    'MAE_reduction_SOH_pp':float(delta.mean()),'relative_MAE_reduction_percent':float(100*delta.mean()/e[baseline].mean())})
   for n,v in delta.items():pointwins.append({'nominal_h':n,'horizon':h,'candidate':candidate,'baseline':baseline,'absolute_error_reduction_SOH_pp':float(v),'winner':'candidate' if v>cfg['tie_tolerance_SOH_pp'] else 'baseline' if v<-cfg['tie_tolerance_SOH_pp'] else 'tie'})
  m0=float(e[METHODS[0]].mean());origin=float(e[METHODS[1]].mean());ridge=float(e[METHODS[2]].mean());total=m0-ridge
  contrib.append({'horizon':h,'n_unique_events':3,'M0_MAE_SOH_pp':m0,'Origin_MAE_SOH_pp':origin,'Ridge_MAE_SOH_pp':ridge,
   'recent_P1S_and_calibration_MAE_reduction_SOH_pp':m0-origin,'Ridge_forecast_step_MAE_reduction_SOH_pp':origin-ridge,
   'Ridge_step_relative_to_origin_percent':100*(origin-ridge)/origin,'total_MAE_reduction_vs_M0_SOH_pp':total,
   'Ridge_step_share_of_total_MAE_reduction_percent':100*(origin-ridge)/total,
   'attribution_limit':'Order-dependent descriptive MAE differences, not causal variance/feature attribution'})
 cs('outputs/PAIRWISE_COMPARISONS.csv',pd.DataFrame(pair));cs('outputs/POINTWISE_WINS.csv',pd.DataFrame(pointwins));cs('outputs/SEQUENTIAL_CONTRIBUTION.csv',pd.DataFrame(contrib))
 proxy=[]
 for scope,frame in [('full_historical_validation_cache',cache),('three_matched_reference_events',pd.DataFrame(matched))]:
  for (h,m),g in frame.groupby(['horizon','model']):
   z=summarize(g,'true_R_P1S_causal','predicted_R_P1S_causal',.71383023635*1000);proxy.append({'scope':scope,'horizon':h,'model':m,'n':z['n'],'MAE_mV_equivalent':z['MAE'],'RMSE_mV_equivalent':z['RMSE']})
 cs('outputs/P1S_PROXY_METRICS.csv',pd.DataFrame(proxy))
 contrast=[]
 for n in cfg['events']:
  one=dec[dec.nominal_h.eq(n)].set_index('horizon');assert abs(one.loc['H1','calibration_at_actual_target_SOH_pp']-one.loc['H3','calibration_at_actual_target_SOH_pp'])<1e-12
  contrast.append({'nominal_h':n,'same_target_R':bool(one.loc['H1','actual_future_target_R_AUDIT_ONLY']==one.loc['H3','actual_future_target_R_AUDIT_ONLY']),
   'common_calibration_residual_SOH_pp':float(one.loc['H1','calibration_at_actual_target_SOH_pp']),
   'H1_proxy_AE_mV_equivalent':float(one.loc['H1','ridge_proxy_abs_error_mV_equivalent']),'H3_proxy_AE_mV_equivalent':float(one.loc['H3','ridge_proxy_abs_error_mV_equivalent']),
   'H1_SOH_AE_pp':abs(float(one.loc['H1','ridge_final_error_SOH_pp'])),'H3_SOH_AE_pp':abs(float(one.loc['H3','ridge_final_error_SOH_pp'])),
   'H1_signed_proxy_term_SOH_pp':float(one.loc['H1','ridge_proxy_error_translated_SOH_pp']),'H3_signed_proxy_term_SOH_pp':float(one.loc['H3','ridge_proxy_error_translated_SOH_pp']),
   'H3_error_cancellation':bool(one.loc['H3','ridge_error_cancels_calibration'])})
 cs('outputs/H3_VS_H1_ERROR_CANCELLATION.csv',pd.DataFrame(contrast))
 # Verify the previously saved per-horizon results from rederived values, not user preliminary numbers.
 historical_metrics=[]
 for h in ['H1','H3']:
  for om,nm in [('M0_REFERENCE_PERSISTENCE',METHODS[0]),('M1_RIDGE96_M2_NC',METHODS[2])]:
   prior=oldmet[oldmet.horizon.eq(h)&oldmet.model.eq(om)].iloc[0];new=met[met.horizon.eq(h)&met.method.eq(nm)].iloc[0]
   diff=abs(prior.MAE_SOH_pp-new.MAE_SOH_pp);assert diff<1e-12
   historical_metrics.append({'horizon':h,'historical_method':om,'MAE_difference_SOH_pp':diff,'RMSE_difference_SOH_pp':abs(prior.RMSE_SOH_pp-new.RMSE_SOH_pp)})
 cs('outputs/HISTORICAL_METRIC_REPLAY.csv',pd.DataFrame(historical_metrics));js('outputs/READ_LEDGER.json',sorted(set(ACCESS)))
 js('outputs/COMPLETE.json',{'computed':True,'new_fits':0,'unique_reference_events':3,'linked_origins':6,'method_event_predictions':18,'full_P1S_cache_rows':len(cache),
  'coefficient_replayed_Ridge_predictions':int(cache.model.eq('RIDGE_96H').sum()),'no_Test':True,'no_GPU':True,'strict_online':False,'reference_release_known':False})
 print(met.to_string(index=False));print(pd.DataFrame(contrib).to_string(index=False));print(pd.DataFrame(contrast).to_string(index=False));print('AUDIT_COMPLETE')
if __name__=='__main__':
 firewall();{'freeze':freeze,'run':run}[sys.argv[1]]()

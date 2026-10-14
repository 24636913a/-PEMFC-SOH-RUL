"""Past-only affine calibration of actual cached H1/H3 predictions; no Ridge refit."""
import sys
from common import *
def freeze():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists(),'Already frozen'
 # No mixed raw/N1 input; existing independent read-rule audit verified this snapshot.
 c={'base_commit':'abaf4ccb15137946bb7e2535a83604bf9f2d4060','dataset':'RealHyFC only;Development only',
  'source_sha256':{v:sha(ROOT/v) for v in sorted(set(list(SOURCES.values())+REVIEWS))},
  'stage_a_code_sha256':{s:sha(P/'scripts'/s) for s in ['common.py','run_stage_a.py']},
  'reference_voltage_V':.6989533305168152,'P1S_reference_voltage_V':.71383023635,
  'event_selection':'Each nominal reference: nearest complete middle_50 P1S measurement strictly before polarization calendar start; max48 calendar h. Event scheduling is retrospective, not certified online.',
  'M0':'Last chronologically measured reference; availability assumes completion/release before next origin, release times unknown',
  'M1':'Existing time-purged 96h Ridge H1/H3 exact physical target; affine calibration trained on >=3 earlier post-BoL reference / nearest-complete P1S pairs; no BoL/P1S pseudo-pair',
  'calibration_operator':'Nearest complete single measurement, not original AHC median-of-three. New past-only affine M2_NC, no reuse of all-six coefficients.',
  'min_prior_calibration_events':3,'max_proxy_calendar_gap_h':48,'evaluation':'Exploratory scheduled-event chronological replay; strict-online metrics unavailable',
  'TFT':'Frozen original F2; Train+Validation selection label availability checked, not just input timestamps. No long-horizon extrapolation.',
  'P1':'Additive state identifiability, recovery counterexamples, slow-prior gradient and exactly matched ordinary-smoothing control on Train only. No hyperparameter scan; stop if not mechanistically identifiable.',
  'diagnostic_smoothing_lambda':.1,'P2':'Reuse qualified EIS gates; no R0-as-HFR or partial ohmic-as-full-voltage teacher',
  'RUL':'No true EOL label or accuracy; descriptive conditional calendar-time scenarios only, not operating-hour RUL',
  'scenario_thresholds':[.97,.95,.90],'scenario_horizon_calendar_h':8760,
  'trajectory_models':['post_BoL_all_past_OLS','last3_past_OLS','zero_slope_recovery_sensitivity'],
  'trajectory_uncertainty':'Deterministic slope envelope including leave-one-past-point-out; no probabilistic CI',
  'no_Test':True,'no_GPU_before_qualification':True,'no_F2_or_96h_Ridge_retraining':True}
 js('configs/FROZEN_PROTOCOL.json',c);md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'))
 cs('data_contracts/SOURCE_IDENTITY.csv',pd.DataFrame([{'path':v,'sha256':h,'role':'Read-only Development derived artifact / historical review'} for v,h in c['source_sha256'].items()]))
 md('.gitignore','__pycache__/\n*.pyc\noutputs/GIT_RECEIPT.local.json')
 md('.gitattributes','* -text\n*.png binary\n*.pdf binary')
 print('FROZEN',sha(P/'configs/FROZEN_PROTOCOL.json'))
def affine(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);assert len(x)>=3 and np.ptp(x)>1e-10
 a=float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2));return a,float(y.mean()-a*x.mean())
def run():
 assert not (P/'outputs/STAGE_A_COMPLETE.json').exists(),'Do not repeat completed computations'
 cfg=config()
 for s,h in cfg['stage_a_code_sha256'].items():assert sha(P/'scripts'/s)==h
 refs=source('refs');assert refs.nominal_h.tolist()==[0,300,600,850,1150,1500,1750]
 timing=source('timing');c=source('measure');c=c[c.stable_window.eq('middle_50')].copy();c['calendar']=pd.to_datetime(c.measurement_end_datetime)
 ident=source('identity');ip=ident.set_index('cycle_interval_id');pred=source('preds');params=load_json('params');time=source('parameter_time')
 assert len(c)==141 and c.cycle_interval_id.is_unique and len(ident)==131
 bins=source('bins');assert len(bins)==3005 and not bins.split.eq('test').any() and bins.bin_end_ophrs2_h.max()<1819.997738
 events=[{'nominal_h':0,'Y':1.,'polarization_start_calendar':'2024-12-09','polarization_end_calendar':None,'reference_actual_OpHrs2':None,'release_calendar':None,'prior_anchor_count':0,'eligible_formal':False,'reason':'BoL denominator, not forecast event; no prior health calibration'}]
 nearest={};anchorrows=[]
 for _,t in timing.iterrows():
  stamp=pd.Timestamp(t.polarization_first_datetime);prior=c[c['calendar']<stamp].sort_values('calendar');last=prior.iloc[-1];gap=(stamp-last['calendar']).total_seconds()/3600
  assert gap<=cfg['max_proxy_calendar_gap_h'];assert float(last.current_density)!=1.
  n=int(t.checkpoint_h);y=float(refs.loc[refs.nominal_h.eq(n),'Y'].iloc[0]);nearest[n]=last
  anchorrows.append({'nominal_h':n,'R_nearest_complete':float(last.P1S_operating_voltage_retention),'Y':y,'cycle_interval_id':last.cycle_interval_id,'proxy_calendar':str(last['calendar']),'reference_end_calendar':t.polarization_last_datetime})
  previous=c[c.raw_cycle_id.eq(float(last.raw_cycle_id)-1)].iloc[0]
  events.append({'nominal_h':n,'Y':y,'polarization_start_calendar':t.polarization_first_datetime,'polarization_end_calendar':t.polarization_last_datetime,
   'reference_actual_OpHrs2':None,'reference_preceding_durability_cutoff_h':float(t.PAA_gap_cutoff_ophrs2_h),'reference_cutoff_calendar':t.PAA_gap_cutoff_datetime,
   'release_calendar':None,'target_proxy_cycle_id':last.cycle_interval_id,'target_proxy_raw_cycle_id':int(last.raw_cycle_id),
   'target_proxy_available_h':float(last.measurement_end_ophrs2_h),'target_proxy_calendar':str(last['calendar']),'proxy_to_reference_calendar_h':gap,
   'baseline_origin_cycle_id':previous.cycle_interval_id,'baseline_origin_calendar':str(previous['calendar']),
   'prior_anchor_count':sum(refs.nominal_h<n)-1,'P1S_current_density_A_cm2':float(last.current_density),
   'P1S_temperature_C':float(last.anode_inlet_temperature_C),'P1S_anode_RH_pct':float(last.anode_inlet_RH_pct),
   'reference_j_A_cm2':1.,'measurement_operator_equal':False,'condition_comparability':'Nominal reference protocol conditional; P1S-to-reference affine is an empirical hypothesis',
   'eligible_formal':False,'eligible_calendar_replay':True,'reason':'Reference release and exact OpHrs2 unknown; event-target choice retrospective; complete-cycle online identity not certified'})
 e=pd.DataFrame(events);cs('outputs/SOH_EVENT_LEVEL_ELIGIBILITY.csv',e);cs('data_contracts/NEAREST_COMPLETE_CALIBRATION_PAIRS.csv',pd.DataFrame(anchorrows))
 rows=[];reject=[];models=[];links=[]
 for idx,ar in enumerate(anchorrows):
  n=ar['nominal_h'];last=nearest[n];past=[x for x in anchorrowcopy(anchorrows) if x['nominal_h']<n]
  previousY=float(refs.loc[refs.nominal_h.eq([0,300,600,850,1150,1500][idx]),'Y'].iloc[0])
  ev=e.loc[e.nominal_h.eq(n)].iloc[0]
  rows.append({'nominal_h':n,'horizon':'M0_ALL_EVENTS','model':'M0_REFERENCE_PERSISTENCE','true_SOH':ar['Y'],'predicted_SOH':previousY,
   'origin_calendar':ev.baseline_origin_calendar,'origin_cycle_interval_id':ev.baseline_origin_cycle_id,'evaluation_class':'RETROSPECTIVE_CALENDAR_REPLAY','formal_online_qualified':False})
  for horizon in ['H1','H3']:
   q=pred[pred.model.eq('RIDGE_96H')&pred.horizon.eq(horizon)&pred.target_cycle_interval_id.eq(last.cycle_interval_id)]
   reasons=[]
   if len(q)!=1:reasons.append('NO_EXACT_FROZEN_VALIDATION_FORECAST')
   if len(past)<cfg['min_prior_calibration_events']:reasons.append('LESS_THAN_3_PRIOR_POST_BOL_CALIBRATION_EVENTS')
   if reasons:reject.append({'nominal_h':n,'horizon':horizon,'reason':'|'.join(reasons),'prior_count':len(past),'forecast_matches':len(q)});continue
   r=q.iloc[0];par=next(x for x in params if x['fold']==r.fold and x['horizon']==horizon);ta=time[(time.fold==r.fold)&time.horizon.eq(horizon)].iloc[0]
   origin=ip.loc[r.origin_cycle_interval_id];target=ip.loc[r.target_cycle_interval_id];origincal=pd.Timestamp(c.set_index('cycle_interval_id').loc[r.origin_cycle_interval_id,'measurement_end_datetime'])
   assert ta.train_max_label_end_h<=float(origin.conservative_available_end_h)
   assert ta.inner_max_label_end_h<=ta.first_inner_selection_origin_h
   assert all(pd.Timestamp(x['reference_end_calendar'])<origincal for x in past)
   # Reproduce cached Ridge from saved Train-only scaler/coefficients, no fitting.
   x=r[par['features']].to_numpy(float);rp=float(r.origin_R_P1S_causal+np.dot((x-np.array(par['train_scaler_mean']))/np.array(par['train_scaler_scale']),par['coef_standardized_X'])+par['intercept_deltaR'])
   assert abs(rp-r.predicted_R_P1S_causal)<1e-12
   a,b=affine([x['R_nearest_complete'] for x in past],[x['Y'] for x in past]);predY=a*rp+b
   models.append({'nominal_h':n,'horizon':horizon,'a':a,'b':b,'training_events':[x['nominal_h'] for x in past],'R_input_forecast':rp,'past_only':True,'min_prior_events':3,'proxy_operator':'nearest_complete; not AHC median3','model':'M2_NC_AFFINE'})
   origin_h=float(origin.conservative_available_end_h);originactual=float(origin.measurement_end_h)
   link={'nominal_h':n,'horizon':horizon,'fold':r.fold,'origin_cycle_interval_id':r.origin_cycle_interval_id,'target_cycle_interval_id':r.target_cycle_interval_id,
    'origin_raw_cycle_id':int(r.origin_raw_cycle_id),'target_raw_cycle_id':int(r.target_raw_cycle_id),'origin_actual_measurement_end_h':originactual,
    'origin_conservative_available_h':origin_h,'origin_calendar':str(origincal),'target_actual_measurement_end_h':float(target.measurement_end_h),
    'target_calendar':ar['proxy_calendar'],'actual_proxy_lead_h_audit_only':float(target.measurement_end_h)-originactual,
    'calendar_lead_to_reference_h_audit_only':(pd.Timestamp(ev.polarization_start_calendar)-origincal).total_seconds()/3600,
    'prior_reference_end_max_calendar':past[-1]['reference_end_calendar'],'reference_release_known':False,'max_Train_label_end_h':float(ta.train_max_label_end_h),
    'inner_train_max_label_h':float(ta.inner_max_label_end_h),'inner_selection_first_origin_h':float(ta.first_inner_selection_origin_h),
    'Ridge_recomputed_prediction':rp,'calibration_prior_count':len(past),'target_identification':'retrospective scheduled-event selector, never model feature'}
   links.append(link)
   for method,value in [('M0_REFERENCE_PERSISTENCE',previousY),('M1_RIDGE96_M2_NC',predY)]:
    rows.append({'nominal_h':n,'horizon':horizon,'model':method,'true_SOH':ar['Y'],'predicted_SOH':value,'origin_calendar':str(origincal),
     'origin_cycle_interval_id':r.origin_cycle_interval_id,'evaluation_class':'RETROSPECTIVE_CALENDAR_REPLAY','formal_online_qualified':False})
 out=pd.DataFrame(rows);out['error_SOH_pp']=(out.predicted_SOH-out.true_SOH)*100;out['abs_error_SOH_pp']=out.error_SOH_pp.abs()
 cs('outputs/SOH_EVENT_PREDICTIONS.csv',out);cs('outputs/SOH_EVENT_EXCLUSIONS.csv',pd.DataFrame(reject));cs('outputs/P1S_TO_SOH_EXACT_FORECAST_LINKS.csv',pd.DataFrame(links));js('models/PAST_ONLY_AFFINE_PARAMETERS.json',models)
 met=[]
 for (h,m),z in out.groupby(['horizon','model']):met.append({'horizon':h,'model':m,'evaluation_class':'Exploratory calendar replay',**metrics(z.true_SOH,z.predicted_SOH)})
 cs('outputs/SOH_EVENT_METRICS.csv',pd.DataFrame(met))
 # TFT audit changes from direct overlap to past information + parameter availability.
 d=source('F2_42');selection_end_h=float(bins.iloc[int(d.target_row.max())].bin_end_ophrs2_h);selection_end_calendar=pd.Timestamp(bins.iloc[int(d.target_row.max())].last_datetime)
 assert d.origin_row.min()>=2103
 tft=[];history=[]
 for ar in anchorrows:
  n=ar['nominal_h'];proxy=nearest[n];origin=c[c.raw_cycle_id.eq(float(proxy.raw_cycle_id)-1)].iloc[0];oc=origin['calendar']
  hist=bins[pd.to_datetime(bins.last_datetime)<=oc].tail(48)
  history.append({'nominal_h':n,'origin_calendar':str(oc),'n_bins_up_to_48':len(hist),'history_last_calendar':hist.last_datetime.iloc[-1],
   'history_mean_dynamic_voltage_V':float(hist.cell_voltage_mean.mean()),'latest_dynamic_voltage_V':float(hist.cell_voltage_mean.iloc[-1]),
   'history_mean_j_A_cm2':float(hist.actual_current_density_A_cm2_mean.mean()),'history_T_anode_C':float(hist.actual_anode_inlet_temperature_C_mean.mean()),
   'P1S_origin_R':float(origin.P1S_operating_voltage_retention),'input_history_end_before_origin':bool(pd.Timestamp(hist.last_datetime.iloc[-1])<=oc)})
  available=pd.to_datetime(bins.iloc[d.origin_row.to_numpy(int)].last_datetime).to_numpy()
  ids=(available<=oc.to_datetime64()) & (available>=selection_end_calendar.to_datetime64())
  tft.append({'nominal_h':n,'origin_calendar':str(oc),'frozen_F2_selection_label_end_h':selection_end_h,'frozen_F2_selection_end_calendar':str(selection_end_calendar),
   'model_selection_labels_available_before_origin':bool(selection_end_calendar<oc),'causal_cached_queries_available':int(ids.sum()),
   'earlier_reference_events_with_causal_cached_TFT_predictors':0,
   'future_reference_actual_OpHrs2_known':False,'TFT_horizon_h':12.,'status':'TFT_HEALTH_ALIGNMENT_INSUFFICIENT',
   'reason':'Existing selected checkpoint uses entire Validation labels; zero post-selection cached origins. Later 1750 could use new inference but no past causal event-level TFT calibration/holdout pairs; no identifiable incremental comparison.'})
 cs('outputs/TFT_PAST_INFORMATION_AND_PARAMETER_AVAILABILITY.csv',pd.DataFrame(tft));cs('outputs/PAST_DYNAMIC_AND_P1S_INFORMATION.csv',pd.DataFrame(history))
 js('outputs/STAGE_A_COMPLETE.json',{'SOH_replay_events_per_horizon':3,'SOH_formal_online_events':0,'affine_fits':len(models),'Ridge_refits':0,'TFT_refits':0,'TFT_health_status':'TFT_HEALTH_ALIGNMENT_INSUFFICIENT','model_selection_end_h':selection_end_h,'source_unchanged':True,'no_Test':True,'no_GPU':True})
 js('outputs/STAGE_A_READ_LEDGER.json',ACCESS)
 print(pd.DataFrame(met).to_string(index=False));print('STAGE_A_COMPLETE')
def anchorrowcopy(rows):return [dict(x) for x in rows]
if __name__=='__main__':
 firewall()
 {'freeze':freeze,'run':run}[sys.argv[1]]()

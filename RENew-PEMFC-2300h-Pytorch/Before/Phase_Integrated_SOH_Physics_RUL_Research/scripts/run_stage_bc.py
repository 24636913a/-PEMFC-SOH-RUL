"""CPU qualification and conditional calendar scenarios; zero optimization training."""
import sys
from common import *
def smooth_loss_grad(z,scale):
 z=np.asarray(z,float);v=np.diff(z);factor=1/(len(v)*scale*scale)
 grad=np.zeros_like(z);grad[:-1]-=2*factor*v;grad[1:]+=2*factor*v
 return float(np.dot(v,v)*factor),grad
def freeze():
 config();assert not (P/'configs/BC_CODE_IDENTITY.json').exists()
 js('configs/BC_CODE_IDENTITY.json',{'protocol_sha256':sha(P/'configs/FROZEN_PROTOCOL.json'),'code_sha256':{s:sha(P/'scripts'/s) for s in ['common.py','run_stage_bc.py']},'run_mode':'CPU only, qualification before any training'})
 print('B/C code identity frozen')
def run():
 cfg=config();assert not (P/'outputs/STAGE_BC_COMPLETE.json').exists(),'Completed, do not repeat'
 code=json.loads((P/'configs/BC_CODE_IDENTITY.json').read_text());assert code['protocol_sha256']==sha(P/'configs/FROZEN_PROTOCOL.json')
 for s,h in code['code_sha256'].items():assert sha(P/'scripts'/s)==h
 samples=source('samples');train=samples[samples.horizon.eq('H1')&samples.fold.eq('fold_1')&samples.role.eq('train')].sort_values('origin_p1s_measurement_end_ophrs2_h')
 assert len(train)==45;assert train.target_p1s_measurement_end_ophrs2_h.max()<=source('parameter_time').iloc[0].first_validation_origin_h
 z=train.origin_R_P1S_causal.to_numpy(float);n=len(z);scale=float(z.std(ddof=0));x=train.origin_p1s_measurement_end_ophrs2_h.to_numpy(float)
 rank=int(np.linalg.matrix_rank(np.concatenate([np.eye(n),np.eye(n)],axis=1)))
 D1=np.full(n,z[0]);D2=z[0]-.01*(x-x.min())/(x.max()-x.min());Q1=z-D1;Q2=z-D2
 diff=float(max(np.max(np.abs(D1+Q1-z)),np.max(np.abs(D2+Q2-z))))
 assert diff<1e-14 and np.all(np.diff(D2)<=0);assert np.count_nonzero(np.diff(z)>0)>0
 loss,gradient=smooth_loss_grad(z,scale)
 # Independent matrix form of an exactly matched ordinary statistical regularizer.
 T=np.diff(np.eye(n),axis=0);regloss=float(np.dot(T@z,T@z)/((n-1)*scale*scale));reggrad=2*T.T@T@z/((n-1)*scale*scale)
 # Floating point multiplication order may differ; exact equality of the objectives is also algebraically recorded.
 numericdiff=float(np.max(np.abs(reggrad-gradient)));assert abs(regloss-loss)<1e-12 and numericdiff<1e-8
 eps=1e-7;fd=np.array([(smooth_loss_grad(z+eps*np.eye(n)[i],scale)[0]-smooth_loss_grad(z-eps*np.eye(n)[i],scale)[0])/(2*eps) for i in range(n)])
 fd_error=float(np.max(np.abs(fd-gradient)));assert fd_error<1e-5
 data_loss=float(np.mean((train.target_R_P1S_causal.to_numpy()-z)**2)/(scale*scale))
 # Synthetic trainable parameter vector, separate from the real-Train node diagnostic.
 xx=np.linspace(0,1,12);X=np.column_stack([np.ones(12),xx]);theta=np.array([1.,-.02]);sl,sg=smooth_loss_grad(X@theta,.01);pg=X.T@sg
 sfd=np.array([(smooth_loss_grad(X@(theta+eps*np.eye(2)[i]),.01)[0]-smooth_loss_grad(X@(theta-eps*np.eye(2)[i]),.01)[0])/(2*eps) for i in range(2)])
 assert np.max(np.abs(pg-sfd))<1e-6
 cs('outputs/PHYSICS_STATE_DECOMPOSITION.csv',pd.DataFrame({'sample_index':train.sample_index.to_numpy(),'available_end_h':x,'observed_R':z,'D_flat':D1,'Q_flat':Q1,'D_declining':D2,'Q_declining':Q2}))
 cs('outputs/PHYSICS_GRADIENT_DIAGNOSTIC.csv',pd.DataFrame({'node':np.arange(n),'slow_prior_gradient':gradient,'ordinary_smoothing_gradient':gradient.copy(),'finite_difference_gradient':fd}))
 cs('data_contracts/PHYSICS_USED_TRAIN_SAMPLES.csv',train)
 diag={'Train_n':n,'Train_horizon':'H1','Train_fold':'fold_1','state_rank':rank,'state_nullity':2*n-rank,'reconstruction_max_error':diff,
  'observed_recovery_steps':int(np.count_nonzero(np.diff(z)>0)),'potential_decomposition_shift_max':float(np.max(np.abs(D1-D2))),
  'real_Train_scale_R':scale,'actual_Train_statistical_smoothing_loss':loss,'actual_Train_statistical_smoothing_gradient_L2':float(np.linalg.norm(gradient)),
  'slow_prior_vs_smoothing_loss_diff':0.,'slow_prior_vs_smoothing_gradient_max_diff':0.,'independent_matrix_gradient_floating_diff':numericdiff,
  'finite_difference_gradient_max_diff':fd_error,'synthetic_param_gradient':pg.tolist(),'synthetic_param_gradient_FD_max_diff':float(np.max(np.abs(pg-sfd))),
  'synthetic_loss':sl,'real_Train_P1S_persistence_residual_MSE_normalized':data_loss,'fixed_lambda':cfg['diagnostic_smoothing_lambda'],
  'regularizer_to_persistence_MSE_ratio_DIAGNOSTIC_ONLY':cfg['diagnostic_smoothing_lambda']*loss/data_loss,
  'real_certified_physics_loss':None,'real_certified_model_parameter_gradient':None,'real_physics_to_prediction_loss_ratio':None,
  'regularizer_gradient_kind':'Latent/output node derivative on real Train values; not trained model parameter gradient',
  'P1_status':'NOT_IDENTIFIABLE;EQUIVALENT_TO_ORDINARY_SMOOTHING','P2_status':'NOT_QUALIFIED_NO_CERTIFIED_HFR_ASR','GPU_runs':0,'optimization_runs':0}
 js('outputs/PHYSICS_CPU_DIAGNOSTICS.json',diag)
 gates=[]
 for candidate in ['P1','P2']:
  states={'formula_units':('PASS_STATISTICAL_ONLY' if candidate=='P1' else 'CONDITIONAL_OHMIC_FORMULA_ONLY'),
   'state_identifiability':('FAIL_NULLSPACE' if candidate=='P1' else 'FAIL_EIS_IDENTITY_AND_R0_CERTIFICATION'),
   'real_Train_support':('45_P1S_TRAIN_NODES;NO_SEPARATE_RECOVERY_MEASURE' if candidate=='P1' else 'NO_CERTIFIED_ASR'),
   'synthetic_gradient':('PASS_REGULARIZER_NOT_MECHANISM' if candidate=='P1' else 'NOT_REPEATED;HISTORICAL_INTERFACE_TEST_PASS'),
   'real_Train_physics_activation':'NA_NO_CERTIFIED_PHYSICAL_LOSS','physics_to_data_scale':'NA;REGULARIZER_RATIO_SEPARATE' if candidate=='P1' else 'NA',
   'origin_information_isolation':'PASS_TRAIN_ONLY;REFERENCE_RELEASE_UNKNOWN','matched_nonphysical_control':'EXACT_SMOOTHING_EQUIVALENCE' if candidate=='P1' else 'NA_NO_TRAINING',
   'controlled_training':'NOT_QUALIFIED'}
  for name,value in states.items():gates.append({'candidate':candidate,'gate':name,'result':value})
 cs('outputs/PHYSICS_CPU_QUALIFICATION.csv',pd.DataFrame(gates))
 eis=load_json('eis');assert eis['certified_ASR_values']==0 and not eis['same_stack_verified']
 js('outputs/P2_EIS_REUSE_QUALIFICATION.json',{'source_decision_sha256':sha(ROOT/SOURCES['eis']),'core_spectra':eis['core_spectra'],'real_points':eis['real_points'],
  'certified_ASR_values':0,'measured_HFR_values':0,'not_reusing_R0_as_ageing_resistance':True,'physics_loss':None,'training_allowed':False})
 # Reference event calendar, not nominal life hours; no Test or late-target trajectory validation.
 pairs=pd.read_csv(P/'data_contracts/NEAREST_COMPLETE_CALIBRATION_PAIRS.csv');t=source('timing');calendar=pd.to_datetime(t.polarization_first_datetime)
 hours=(calendar-calendar.iloc[0]).dt.total_seconds().to_numpy()/3600;y=pairs.Y.to_numpy(float);last=float(y[-1])
 def slope(x,v):return float(np.dot(x-x.mean(),v-v.mean())/np.dot(x-x.mean(),x-x.mean()))
 slopes=[{'model':'post_BoL_all_past_OLS','slope_SOH_per_calendar_h':slope(hours,y),'training_events':','.join(pairs.nominal_h.astype(str))},
  {'model':'last3_past_OLS','slope_SOH_per_calendar_h':slope(hours[-3:],y[-3:]),'training_events':','.join(pairs.nominal_h.iloc[-3:].astype(str))},
  {'model':'zero_slope_recovery_sensitivity','slope_SOH_per_calendar_h':0.,'training_events':'explicit flat/recovery sensitivity assumption'}]
 loo=[]
 for i in range(len(y)):
  keep=np.arange(len(y))!=i;loo.append({'left_out_reference':int(pairs.nominal_h.iloc[i]),'slope_SOH_per_calendar_h':slope(hours[keep],y[keep])})
 cs('outputs/CONDITIONAL_TRAJECTORY_SLOPES.csv',pd.DataFrame(slopes));cs('outputs/CONDITIONAL_SLOPE_SENSITIVITY.csv',pd.DataFrame(loo))
 curves=[];scenarios=[]
 for s in slopes:
  k=s['slope_SOH_per_calendar_h']
  for elapsed in np.linspace(0,cfg['scenario_horizon_calendar_h'],366):curves.append({'model':s['model'],'elapsed_calendar_h':elapsed,'projected_SOH':last+k*elapsed,'source':'Fixed past-only slope assumption; not measured future label'})
  for threshold in cfg['scenario_thresholds']:
   crossing=(threshold-last)/k if k<0 else np.nan
   scenarios.append({'model':s['model'],'threshold':threshold,'threshold_is_assumption':True,'conditional_crossing_calendar_h':crossing,
    'conditional_crossing_calendar_days':crossing/24,'reaches_within365_calendar_days':bool(np.isfinite(crossing) and 0<=crossing<=cfg['scenario_horizon_calendar_h']),
    'status':'NO_CROSSING_UNDER_FLAT_ASSUMPTION' if k>=0 else 'CONDITIONAL_EXTRAPOLATION_NOT_RUL_VALIDATION',
    'actual_failure_time':None,'actual_RUL_accuracy':None,'operating_hour_conversion_available':False})
 cs('outputs/CONDITIONAL_CALENDAR_TRAJECTORIES.csv',pd.DataFrame(curves));cs('outputs/CONDITIONAL_CALENDAR_LIFETIME_SCENARIOS.csv',pd.DataFrame(scenarios))
 # Evaluate a fixed calendar slope against past future references before interpreting long extrapolation.
 back=[]
 for i in range(3,len(y)):
  k=slope(hours[:i],y[:i]);predicted=float(y[i-1]+k*(hours[i]-hours[i-1]))
  for method,value in [('M0_LAST_REFERENCE',float(y[i-1])),('PAST_CALENDAR_SLOPE',predicted)]:back.append({'target_nominal_h':int(pairs.nominal_h.iloc[i]),'past_n':i,'model':method,'true_SOH':float(y[i]),'predicted_SOH':value,'abs_error_SOH_pp':abs(value-y[i])*100})
 cs('outputs/LONG_TRAJECTORY_CALENDAR_BACKTEST.csv',pd.DataFrame(back))
 cs('outputs/RUL_EOL_AND_CENSORING_ELIGIBILITY.csv',pd.DataFrame([
  {'requirement':'same_reference_failure_threshold','status':'NOT_ESTABLISHED','evidence':'Historical methods include condition-specific500mV limit; not j=1 retention EOL'},
  {'requirement':'observed_failure_timestamp','status':'NOT_ESTABLISHED','evidence':'Test not opened; research end is not verified failure'},
  {'requirement':'right_censoring_certificate','status':'NOT_ESTABLISHED','evidence':'Requires prespecified failure definition and proof not reached'},
  {'requirement':'operating_age_of_reference','status':'UNKNOWN','evidence':'Polarization actual OpHrs2 absent, calendar intervals available'},
  {'requirement':'conditional_calendar_scenarios','status':'COMPUTED_NOT_VALIDATED','evidence':'97/95/90% frozen illustrative thresholds, flat sensitivity; no RUL accuracy'}]))
 evidence=[
  ('TFT_complete_F2_Validation_RMSE','SUPPORTED','7.245400mV vs F0 9.067563mV; O1 metric dependence and Seed62 tail retained'),
  ('gate_independent_gain','NOT_SUPPORTED','P2/P3 paired controls fail stable independent gain'),
  ('96h_future_P1S','EXPLORATORY','H1/H3 threefold MAE gain10.64/10.42%; Development selected after prior results'),
  ('future_reference_SOH_replay','EXPLORATORY','3 events/horizon; H1 MAE0.079730,H3 0.032385 vs M0 .281172SOHpp; release and scheduling assumptions'),
  ('strict_online_future_SOH','NOT_EVALUABLE','Unknown reference release and exact operating age; scheduled-event retrospective selector'),
  ('TFT_extra_health_information','NOT_EVALUABLE','No causal cached post-selection predictors and insufficient future reference train/holdout events'),
  ('real_physics_optimization','NOT_EVALUABLE','P1 nullspace and smoothing equivalence; P2 no certified HFR/ASR; no qualified A/B/C training'),
  ('physics_beyond_nonphysical_smoothing','NOT_SUPPORTED','P1 objective and gradient algebraically identical to matching ordinary smoothing'),
  ('true_RUL','NOT_EVALUABLE','No verified reference EOL/failure/censor label; no accuracy'),
  ('conditional_lifetime_scenarios','EXPLORATORY','Calendar-time assumed slopes and thresholds only, leave-one-out sensitivity and flat scenario'),
  ('single_dataset_stability','EXPLORATORY','3 seeds for historical dynamic branch; 3 Development folds P1S; only3 SOH events, no physical/RUL validation')]
 cs('outputs/EVIDENCE_MATRIX.csv',pd.DataFrame(evidence,columns=['claim','status','evidence_and_limit']))
 js('outputs/FINAL_RESEARCH_DECISION.json',{'status':'PARTIALLY_SUPPORTED_NEEDS_EVIDENCE','strict_online_SOH_qualified_events':0,'SOH_calendar_replay_events_per_horizon':3,
  'TFT_health_status':'TFT_HEALTH_ALIGNMENT_INSUFFICIENT','physics_training_qualified':False,'GPU_training_runs':0,'new_TFT_training_runs':0,'new_Ridge_training_runs':0,
  'real_physics_training_loss':None,'real_physics_model_parameter_gradient':None,'controlled_physics_ablation_accuracy':None,'real_RUL_accuracy':None,
  'conditional_lifetime_kind':'Calendar-time illustrative extrapolation, not operating-hour RUL','no_Test':True,'all_historical_sources_unchanged':True,'stop_after_delivery':True})
 js('outputs/STAGE_BC_READ_LEDGER.json',ACCESS);js('outputs/STAGE_BC_COMPLETE.json',{'completed':True,'CPU_qualification':True,'conditional_scenarios':len(scenarios),'new_prediction_training':False})
 print('PHYSICS',diag['P1_status'],diag['P2_status']);print('SCENARIOS',len(scenarios));print('STAGE_BC_COMPLETE')
if __name__=='__main__':
 firewall();{'freeze':freeze,'run':run}[sys.argv[1]]()

"""Real condition/parameter-time diagnostics, not another predictive or physics fit."""
from common import *
def run():
 config();assert (P/'outputs/RESUME_COMPLETE.json').exists()
 pred=pd.read_csv(P/'outputs/PREDICTIONS.csv',float_precision='round_trip');fm=pd.read_csv(P/'outputs/FOLD_METRICS.csv');eq=pd.read_csv(P/'outputs/THREE_FOLD_EQUAL_MEANS.csv');used=pd.read_csv(P/'outputs/USED_SAMPLES.csv',float_precision='round_trip')
 paired=[]
 for scope in fm.scope.unique():
  for H in [1,3,6,12]:
   for fold in ['fold_1','fold_2','fold_3']:
    g=fm[fm.scope.eq(scope)&fm.H.eq(H)&fm.fold.eq(fold)].set_index('method');base=g.loc['PERSISTENCE']
    for method in ['LOCAL_TREND96','RIDGE96']:
     r=g.loc[method];valid=base.n>0 and r.n==base.n
     paired.append({'scope':scope,'H':H,'fold':fold,'method':method,'paired_n':int(base.n) if valid else 0,'same_samples_evaluable':bool(valid),
      'MAE_improvement_percent':float((1-r.MAE_mVeq/base.MAE_mVeq)*100) if valid else None,
      'MAE_win':bool(r.MAE_mVeq<base.MAE_mVeq) if valid else None,'RMSE_win':bool(r.RMSE_mVeq<base.RMSE_mVeq) if valid else None,
      'P95_win':bool(r.P95_mVeq<base.P95_mVeq) if valid else None,'max_win':bool(r.max_mVeq<base.max_mVeq) if valid else None})
 cs('outputs/PAIRED_FOLD_COMPARISONS.csv',pd.DataFrame(paired))
 overall=[]
 for H in [1,3,6,12]:
  g=eq[eq.scope.eq('own_horizon_cohort')&eq.H.eq(H)].set_index('method')
  for method in ['LOCAL_TREND96','RIDGE96']:
   base=g.loc['PERSISTENCE'];r=g.loc[method];full=bool(base.threefold_evaluable and r.threefold_evaluable)
   comp=pd.DataFrame(paired);comp=comp[comp.scope.eq('own_horizon_cohort')&comp.H.eq(H)&comp.method.eq(method)]
   overall.append({'H':H,'method':method,'threefold_evaluable':full,'Persistence_MAE_mVeq':float(base.MAE_mVeq) if full else None,
    'method_MAE_mVeq':float(r.MAE_mVeq) if full else None,'equal_fold_MAE_improvement_percent':float((1-r.MAE_mVeq/base.MAE_mVeq)*100) if full else None,
    'MAE_fold_wins':int(comp.MAE_win.fillna(False).sum()),'evaluable_fold_count':int(comp.same_samples_evaluable.sum()),'fold_counts':r.fold_counts})
 cs('outputs/PRIMARY_METHOD_COMPARISON.csv',pd.DataFrame(overall))
 diagnostics=[]
 for (H,method),g in pred.groupby(['H','method']):
  for col in ['future_observed_increase_AUDIT_ONLY','origin_observed_increase','target_condition_OOD_AUDIT_ONLY']:
   for state in [False,True]:
    z=g[g[col].eq(state)];diagnostics.append({'scope':'POOLED_DESCRIPTIVE_NOT_PRIMARY','H':H,'method':method,'stratum':col,'state':state,**metric(z.target_R,z.predicted_R)})
 cs('outputs/POOLED_STRATUM_DIAGNOSTICS.csv',pd.DataFrame(diagnostics))
 # Physical reference state: fixed current is well measured, but does not identify separate losses.
 physical=[];ranges=[]
 for (H,fold),g in used[used.role.eq('train')].groupby(['H','fold']):
  j=g.target_condition_AUDIT_ONLY_current_density.to_numpy();X=np.column_stack([np.ones(len(j)),-j]);rank=int(np.linalg.matrix_rank(X));assert rank==1
  # Gauge transform Eeff'=Eeff+j*c, ASR'=ASR+c at measured fixedj.
  e=.9;r=.05;delta=.02;gauge=float(np.max(abs((e+j*delta)-j*(r+delta)-(e-j*r))))
  physical.append({'H':H,'fold':fold,'Train_n':len(g),'current_density_unique':int(len(np.unique(j))),'fixed_j_A_cm2':float(j[0]),'ohmic_design_rank':rank,
   'ohmic_parameter_count':2,'gauge_voltage_difference_V':gauge,'gradient_dV_dASR_A_cm2':float(-j[0]),
   'real_certified_physics_loss':None,'real_certified_physics_model_parameter_gradient':None,'control_training_score':None,'qualification':'NOT_IDENTIFIABLE_AND_NO_CERTIFIED_ASR'})
  for col in CONDITIONS:ranges.append({'H':H,'fold':fold,'Train_target_variable':col,'min':float(g['target_condition_AUDIT_ONLY_'+col].min()),'max':float(g['target_condition_AUDIT_ONLY_'+col].max())})
 cs('outputs/REAL_TRAIN_PHYSICS_IDENTIFIABILITY.csv',pd.DataFrame(physical));cs('outputs/USED_TRAIN_TARGET_CONDITION_RANGES.csv',pd.DataFrame(ranges))
 eis=jl('eis');oldphysics=jl('physics');assert eis['certified_ASR_values']==0 and eis['measured_HFR_values']==0
 qualification=[
  {'candidate':'P1S_reference_ohmic_consistency','requirement':'law_units','evidence':'V=Eeff-j*ASR; V, A/cm2 and Ohm*cm2; activation/transport remain in Eeff, cannot set them zero'},
  {'candidate':'P1S_reference_ohmic_consistency','requirement':'observable_variables','evidence':'Real P1S voltage,j,anode T/RH/gauge pressure,flows,stoichiometry; no certified ASR, all reactive partial pressures or cellwise loss decomposition'},
  {'candidate':'P1S_reference_ohmic_consistency','requirement':'applicable_domain','evidence':'Recorded fixedj P1S middle50 conditional domain;131 causal observations contain4 colder/high-RH early values; forecast Train target conditions saved separately, no silent removal'},
  {'candidate':'P1S_reference_ohmic_consistency','requirement':'independent_of_statistics','evidence':'Ohmic law is physical, but substituting free Eeff makes only ordinary voltage regression; absent ASR cannot define an independent physical residual'},
  {'candidate':'P1S_reference_ohmic_consistency','requirement':'identifiability','evidence':'Every real Train design[1,-j]rank1 for2 parameters; Eeff+j*c,ASR+c leaves measured voltage unchanged'},
  {'candidate':'P1S_reference_ohmic_consistency','requirement':'real_Train_loss_and_gradient','evidence':'NA; algebraic derivative-j is not a measured valid physics-loss gradient; no made-up zero'},
  {'candidate':'P1S_reference_ohmic_consistency','requirement':'matched_nonphysical_control','evidence':'Same2-parameter ordinary regression shares rank1 gauge. No A/B/C physical training allowed, so actual training scores remainNA'},
  {'candidate':'slow_degradation_state','requirement':'previous_qualification','evidence':f"Inherited additive state nullity{oldphysics['state_nullity']}, ordinary smoothing equality; not reopened or renamed PINN"},
  {'candidate':'EIS_ohmic_teacher','requirement':'previous_qualification','evidence':f"Reuse{eis['core_spectra']} spectra/{eis['real_points']} points,33 previous circuit fits; certifiedASR0, same-stack verifiedFalse. No refit"}]
 cs('outputs/PHYSICS_QUALIFICATION_MATRIX.csv',pd.DataFrame(qualification))
 # Frozen F2 parameter-time availability comes before any increment experiment.
 t=source('tft_time').iloc[0];cut=pd.Timestamp(t.frozen_F2_selection_end_calendar)
 bins=source('bins',usecols=['last_datetime']);cache=source('f2_cache',usecols=['origin_row','target_row','horizon_step'])
 cached_cal=pd.to_datetime(bins.iloc[cache.origin_row.to_numpy(int)].last_datetime).to_numpy();assert not (cached_cal>=cut.to_datetime64()).any()
 tft=[]
 for (H,fold,role),g in used.groupby(['H','fold','role']):
  times=pd.to_datetime(g.origin_calendar);n=int((times>=cut).sum())
  tft.append({'H':H,'fold':fold,'role':role,'n':len(g),'parameters_available_by_origin':n,'checkpoint_selection_end_calendar':str(cut),
   'cached_post_selection_queries':0,'P1S_vs_dynamic_measurement_equivalence_certified':False,'TFT_health_increment_status':'NOT_EVALUABLE'})
 tf=pd.DataFrame(tft);assert tf[tf.role.eq('train')].parameters_available_by_origin.sum()==0
 cs('outputs/TFT_PARAMETER_TIME_AND_HEALTH_SUPPORT.csv',tf)
 # No new SOH calibration or EOL labels are constructed.
 refs=source('soh');assert len(refs)==7 and jl('forecast_contribution')['strict_online_SOH'] is False
 bridge={'P1S_to_reference_SOH':'No new mapping or event optimization; seven frozen sparse reference values remain distinct from131 P1S proxy observations',
  'reference_SOH_Vref_V':.6989533305168152,'P1S_Vref_V':VREF,'future_reference_release_known':False,
  'three_event_forecast_contribution':'Prior H1 negligible; H3 gain with proxy/calibration error cancellation unchanged',
  'EOL':'NOT_ESTABLISHED; previous condition-specific500mV limit is not a verifiedj1 SOH EOL timestamp',
  'real_RUL_accuracy':None,'new_lifetime_scenarios':0,'new_SOH_fits':0,'new_physics_training':0,'TFT_health_increment':None}
 js('outputs/SOH_RUL_AND_TFT_BOUNDARIES.json',bridge)
 decision={'status':'SHORT_HORIZON_P1S_SIGNAL_SUPPORTED_LONG_TERM_INTEGRATION_NOT_ESTABLISHED','study_class':'Exploratory single-RealHyFC Development analysis',
  'H1':'Frozen signal reproduced; Ridge improves10.64% vs Persistence','H3':'Frozen signal reproduced; Ridge improves10.42% vs Persistence',
  'H6':'New experiment worse equal-fold MAE than Persistence, fold2n1; no reliable long-term gain','H12':'THREE_FOLD_NOT_EVALUABLE; inner Train failsfold1, Validation emptyfold2, qualifiedfold3 Ridge worse',
  'new_physics_qualified':False,'TFT_health_increment_supported':False,'new_reference_SOH_or_RUL_supported':False,
  'ordinary_trend_is_not_physics':True,'new_D_model':False,'stop_after_delivery':True}
 js('outputs/RESEARCH_DECISION.json',decision);print(pd.DataFrame(overall).to_string(index=False));print('QUALIFICATION_COMPLETE')
if __name__=='__main__':firewall();run()

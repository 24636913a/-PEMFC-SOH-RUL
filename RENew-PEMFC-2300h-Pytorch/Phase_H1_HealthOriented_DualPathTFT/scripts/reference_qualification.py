"""Event linkage and measurement qualifications, no new SOH calibration fitting."""
from common import *
def main():
 src=ROOT/'Phase_Integrated_SOH_Physics_RUL_Research/outputs/P1S_TO_SOH_EXACT_FORECAST_LINKS.csv'
 params=ROOT/'Phase_Integrated_SOH_Physics_RUL_Research/models/PAST_ONLY_AFFINE_PARAMETERS.json'
 refs=ROOT/'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv'
 sources=[]
 for q in [src,params,refs]:
  rel=q.relative_to(ROOT).as_posix();sources.append(dict(path=rel,sha256=sha(q),git_blob=subprocess.check_output(['git','rev-parse','HEAD:'+ROOT.name+'/'+rel],cwd=REPO,text=True).strip()))
 cs('data_contracts/REFERENCE_SOURCE_SHA256.csv',pd.DataFrame(sources))
 links=rd(src);links=links.loc[links.horizon.eq('H1')];cal=json.loads(params.read_text());s=rd(P/'data_contracts/USED_SAMPLES.csv');va=s.loc[s.role.eq('validation')]
 records=[]
 for r in links.itertuples():
  m=va.loc[va.fold.eq(r.fold)&va.origin_raw_cycle_id.eq(r.origin_raw_cycle_id)&va.target_raw_cycle_id.eq(r.target_raw_cycle_id)]
  c=next(z for z in cal if z['horizon']=='H1' and z['nominal_h']==r.nominal_h)
  record=dict(nominal_h=r.nominal_h,origin_raw_cycle_id=r.origin_raw_cycle_id,target_raw_cycle_id=r.target_raw_cycle_id,
   exact_same_cycle_key_exists=len(m)==1,new_confirmed_origin_calendar=str(m.origin_confirmed_calendar.iloc[0]) if len(m) else '',
   target_measurement_calendar=str(m.target_measurement_last_calendar.iloc[0]) if len(m) else '',
   frozen_calibration_a=c['a'],frozen_calibration_b=c['b'],past_calibration_events='|'.join(map(str,c['training_events'])),
   prior_reference_release_certified=bool(r.reference_release_known),
   selection='Existing retrospective scheduled-event selector; not new forecast input',
   fixed_j_A_cm2=.789229989,polarization_j_A_cm2=1.,identical_voltage_operator=False,
   strict_standard_SOH_comparison_qualified=False,new_calibration_fitted=False,
   new_predicted_standard_SOH=np.nan,new_standard_SOH_error_pp=np.nan,
   status='REFERENCE_SOH_COMPARISON_NOT_QUALIFIED',
   reason='Sparse past-only M2 is an existing conditional calibration; unknown polarization result release time and nonidentical observation protocol prevent certified online standardSOH validation')
  records.append(record)
 cs('outputs/REFERENCE_EVENT_LINK_QUALIFICATION.csv',pd.DataFrame(records));cs('outputs/SEVEN_FROZEN_REFERENCE_SOH.csv',rd(refs))
 condition=rd(PAA/'outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv');condition=condition.loc[condition.stable_window.eq('middle_50')]
 cs('outputs/VALIDATION_TARGET_MEASUREMENT_CONDITIONS.csv',va.merge(condition[['raw_cycle_id','current_density','anode_inlet_temperature_C','anode_inlet_pressure_kPag','anode_inlet_RH_pct']],left_on='target_raw_cycle_id',right_on='raw_cycle_id',validate='one_to_one'))
 cy=rd(RCR/'outputs/RAW_CYCLE_CATALOG.csv').set_index('raw_cycle_id');st=rd(RCR/'outputs/RAW_STATE_RUNS.csv').set_index('run_id');end=st.loc[int(cy.loc[10].end_run_id_exclusive)+1]
 assert s.origin_confirmed_h.gt(end.end_ophrs2_h).all()
 js('outputs/REFERENCE_QUALIFICATION.json',dict(standard_SOH_qualified_pairs=0,exact_existing_retrospective_H1_links=int(sum(r['exact_same_cycle_key_exists'] for r in records)),
  reference_first10_measurements_available_h=171.82091,reference_first10_complete_cycle_confirmation_h=float(end.end_ophrs2_h),
  reference_first10_complete_cycle_confirmation_calendar=end.end_datetime,all_forecast_origins_after_reference_confirmation=True,
  fixed_state_voltage_proxy_qualified=True,proxy_is_certified_standard_SOH=False,calibration_fits=0,physical_loss_added=False,
  new_standard_SOH_error=None,new_PINN_or_RUL=False))
 print('REFERENCE QUALIFICATION: proxy available, standardSOH pairs0')
if __name__=='__main__':main()

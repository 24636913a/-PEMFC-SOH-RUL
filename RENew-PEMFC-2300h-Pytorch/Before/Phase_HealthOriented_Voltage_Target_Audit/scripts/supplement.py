"""Sample-unit, condition and task interface tables derived from the one scan."""
from common import *
def main():
 raw=rd(P/'outputs/RAW_FIXED_STATE_DWELL_MEASUREMENTS.csv');fixed=rd(P/'outputs/FIXED_PRIMARY_CYCLE_OBSERVATIONS.csv')
 old=rd(PAA/'outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv');old=old.loc[old.stable_window.eq('middle_50')]
 unit=[]
 for pop,g in [('All Development fixed0.78923 runs',raw.loc[raw.state_token.eq('221||221.200')]),
  ('Complete146 cycles',raw.loc[raw.state_token.eq('221||221.200')&raw.complete_cycle]),
  ('Original141 frozen cycles',raw.loc[raw.state_token.eq('221||221.200')&raw.raw_cycle_id.isin(old.raw_cycle_id)]),
  ('Electrical_QC_complete_cycles',raw.loc[raw.state_token.eq('221||221.200')&raw.complete_cycle&raw.eligible_electrical_stability])]:
  unit.append(dict(population=pop,raw_run_points=int(g.raw_row_count.sum()),disjoint_middle50_points=int(g.selected_sample_count.sum()),
   disjoint_dwell_observations=len(g),physical_cycles=int(g.raw_cycle_id.nunique()),complete_cycles=int(g.loc[g.complete_cycle,'raw_cycle_id'].nunique()),
   different_measured_times=g.measurement_last_calendar.nunique(),independent_life_trajectories=1))
 cs('outputs/SAMPLE_UNIT_LEDGER.csv',pd.DataFrame(unit))
 primary=raw.loc[raw.state_token.eq('221||221.200')&raw.complete_cycle];stats=[]
 for c in ['duration_wall_s','selected_sample_count','sample_interval_median_s','sample_interval_max_s','j_median','j_std','j_span',
  'V_std','V_drift_mV_per_min','T_anode_C_median','T_anode_C_span','RH_anode_pct_median','RH_anode_pct_span','P_anode_kPag_median','P_anode_kPag_span']:
  x=primary[c];stats.append(dict(field=c,n=len(x),minimum=x.min(),median=x.median(),maximum=x.max()))
 cs('outputs/PRIMARY_STATE_MEASUREMENT_PROFILE.csv',pd.DataFrame(stats))
 runs=rd(P/'outputs/NONOVERLAP_DWELL_TARGETS.csv');slot=[]
 for (token,pos),g in runs.groupby(['state_token','within_cycle_slot']):
  slot.append(dict(state_token=token,slot=int(pos),cycles=g.raw_cycle_id.nunique(),median_V=g.V.median(),min_V=g.V.min(),max_V=g.V.max(),
   warning='Position-dependent load history; not extra independent life trajectories'))
 cs('outputs/WITHIN_CYCLE_POSITION_PROFILE.csv',pd.DataFrame(slot))
 # Same raw cycle IDs/targets, prospective model must use only pre-origin model selection.
 candidate=rd(P/'outputs/CYCLE_TASK_LEGAL_SAMPLES_AND_DYNAMIC_HISTORY.csv')
 authority=rd(ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_LEGAL_SAMPLES.csv')
 keys=['H','fold','role','origin_raw_cycle_id','target_raw_cycle_id']
 candidate=candidate.merge(authority[keys+['origin_calendar','target_calendar']],on=keys,validate='one_to_one');future=[];purged=[]
 for (Hn,fold),g in candidate.groupby(['H','fold']):
  tr=g.loc[g.role.eq('train')&g.valid_48step_dynamic_history];va=g.loc[g.role.eq('validation')&g.valid_48step_dynamic_history]
  first=float(va.origin_available_h.min()) if len(va) else np.nan
  shared=set(tr.target_raw_cycle_id)&set(va.target_raw_cycle_id)
  assert not shared
  last=float(tr.target_available_h.max()) if len(tr) else np.nan
  first_calendar=pd.to_datetime(va.origin_calendar).min()
  strict=pd.to_datetime(tr.target_calendar)<first_calendar
  for r in tr.loc[~strict].itertuples():purged.append(dict(H=Hn,fold=fold,origin_raw_cycle_id=r.origin_raw_cycle_id,target_raw_cycle_id=r.target_raw_cycle_id,
   training_label_available_calendar=r.target_calendar,first_Validation_origin_calendar=str(first_calendar),reason='Future-model protocol needs labels strictly before first origin, with actual fit time; equal timestamp is not earlier'))
  future.append(dict(H=Hn,fold=fold,Train=len(tr),Validation=len(va),shared_target_cycles=len(shared),
   Train_label_last_available_h=last,first_Validation_origin_h=first,all_Train_labels_available=last<=first,
   Train_strict_before_first_Validation=int(strict.sum()),purged_equal_or_later=len(tr)-int(strict.sum()),
   same_48_step_features=True,same_existing_Ridge_training_population=False,
   Ridge_role='Frozen historical reference until a separately authorized common-Train refit',model_selection_performed=False))
 cs('outputs/FUTURE_CYCLE_TASK_TIME_GATES.csv',pd.DataFrame(future))
 cs('outputs/FUTURE_STRICT_TIME_PURGE_RECORDS.csv',pd.DataFrame(purged))
 # Stable-window screening is not synonymous with cross-age environment control.
 extra=fixed.loc[~fixed.raw_cycle_id.isin(old.raw_cycle_id)]
 cs('outputs/ADDITIONAL_FIVE_CYCLES_AUDIT_ONLY.csv',extra)
 stages=[]
 for target in ['221||221.200','156||156.800','313||313.600','30||14.000']:
  for age,cy in [('EARLY',range(1,51)),('MIDDLE',range(51,101)),('LATE',range(101,157))]:
   g=raw.loc[raw.state_token.eq(target)&raw.raw_cycle_id.isin(cy)&raw.complete_cycle]
   for col in ['T_anode_C_median','RH_anode_pct_median','P_anode_kPag_median','j_median']:
    stages.append(dict(state_token=target,stage=age,field=col,n=len(g),minimum=g[col].min(),median=g[col].median(),maximum=g[col].max()))
 cs('outputs/CONDITIONS_BY_REAL_CYCLE_STAGE.csv',pd.DataFrame(stages))
 print('Sample units and time gates complete')
if __name__=='__main__':main()

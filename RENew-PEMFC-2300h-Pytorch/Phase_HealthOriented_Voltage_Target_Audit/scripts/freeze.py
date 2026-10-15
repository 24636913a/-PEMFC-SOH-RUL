"""One fixed audit before extracting new raw stable-state voltage values."""
from common import *
import yaml
def main():
 assert not (P/'configs/FROZEN_AUDIT.json').exists(),'No refreeze'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='bd5b79392b1a38f4fbf7c3b43ba666ecb9a2877c'
 for d in ['configs','data_contracts','outputs','figures','tests']:(P/d).mkdir(exist_ok=True)
 sources=[RCR/'outputs/RAW_STATE_RUNS.csv',RCR/'outputs/RAW_CYCLE_CATALOG.csv',RCR/'outputs/RAW_TIMELINE_AUDIT.csv',
 RCR/'outputs/CYCLE_TEMPLATE.json',PAA/'configs/PREREGISTERED_PAA_PROTOCOL.yaml',PAA/'scripts/run_p1_state_token_preanchor_audit.py',
 PAA/'outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',PAA/'outputs/P1S_REFERENCE_COMPLETE_CYCLES.csv',PAA/'outputs/P1S_CONDITION_SUPPORT.csv',
 PAA/'outputs/P1S_LONG_TERM_DRIFT.csv',PAA/'outputs/P1S_CONDITION_MATCHING.csv',PAA/'reports/FINAL_P1S_PREANCHOR_AUDIT.md',
 RHA/'outputs/CYCLE_POINT_MEASUREMENTS.csv',RHA/'scripts/run_raw_operation_point_health_audit.py',RHA/'outputs/DATA_ACCESS_AUDIT.json',
 ROOT/'Phase_PHI_P1S_B2_Health_Index_Consolidation/README.md',ROOT/'Phase_PHI_P1S_B2_Health_Index_Consolidation/configs/FROZEN_PHI_PROTOCOL.yaml',
 ROOT/'Phase_F_TFT_DualPath_RegimeFusion/README.md',ROOT/'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml',
 ROOT/'Phase_H_Determinism_MultiSeed_Stability/README.md',ROOT/'Phase_TFT_DynamicVoltage_to_ReferenceSOH/00_通俗研究结论.md',
 ROOT/'Phase_TFT_DynamicVoltage_to_ReferenceSOH/outputs/ALL_REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv',
 PREV/'outputs/DECISION.json',PREV/'outputs/DESCRIPTIVE_LOAD_TIME_RELATIONSHIPS.csv',PREV/'06_下一步决策.md',
 ROOT/'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv',
 ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_LEGAL_SAMPLES.csv',
 ROOT/'REAL_Paper_File/Integrated_Research_Route_Closure/H123_PREDICTIONS.csv']
 # Resolve PHI config by actual frozen filename, without reading late-life outputs.
 sources=[p if p.exists() else next((ROOT/'Phase_PHI_P1S_B2_Health_Index_Consolidation/configs').glob('*.yaml')) if 'FROZEN_PHI' in p.name else p for p in sources]
 for phase in ['Phase_H_Determinism_MultiSeed_Stability','Phase_F_TFT_DualPath_RegimeFusion']:
  sources += [p for p in (ROOT/phase/'outputs/checkpoints').glob('*F2*.pt')]
 for phase in ['Phase_P1S_96h_ThreeFold_Comparison','Phase_Integrated_SOH_Physics_RUL_Research']:
  sources += [p for p in (ROOT/phase/'models').glob('*') if p.is_file()]
 records=[]
 for f in sorted(set(sources)):
  assert f.exists(),f
  rel=f.relative_to(ROOT).as_posix();blob=subprocess.check_output(['git','rev-parse','HEAD:'+ROOT.name+'/'+rel],cwd=REPO,text=True).strip()
  records.append(dict(path=rel,sha256=sha(f),git_blob=blob))
 cs('data_contracts/SOURCE_SHA256.csv',pd.DataFrame(records))
 states=rd(RCR/'outputs/RAW_STATE_RUNS.csv');cycles=rd(RCR/'outputs/RAW_CYCLE_CATALOG.csv')
 targets=states.loc[states.duration_wall_s.ge(120)&states.median_set_current_A.gt(0)&states.stop_or_zero_current_flag.eq(0)].copy()
 targets['raw_cycle_id']=np.nan;targets['complete_cycle']=False
 for c in cycles.itertuples():
  hit=targets.run_id.ge(c.start_run_id)&targets.run_id.lt(c.end_run_id_exclusive)
  targets.loc[hit,'raw_cycle_id']=c.raw_cycle_id;targets.loc[hit,'complete_cycle']=c.complete_cycle
 cs('data_contracts/ALL_LONG_FIXED_STATE_RUNS.csv',targets)
 config=dict(base_commit='bd5b79392b1a38f4fbf7c3b43ba666ecb9a2877c',mode='ONE_PASS_DATA_AND_METHOD_AUDIT_NO_TRAINING',
  selection='All positive non-stop state tokens with frozen duration>=120s, no voltage/trend selection',
  window='Original PAA middle50% of dwell; last25% retained only as window sensitivity, never extra independent labels',
  identity='operation_point+set_current token, complete raw cycle IDs preserved, no cycle rediscovery',
  raw_firewall='Only first certified development_rows_retained records per source from existing RCR timeline; pre-verify age-only prefix, then read numeric columns with nrows bound; no later source files',
  raw_prefix_max_h=BOUNDARY,raw_Tail_Test_numeric_reads=False,
  primary_state='221||221.200',frozen_reference_V=.71383023635,reference_BoL_certified=False,
  electrical_stability_diagnostic=dict(j_std_max_A_cm2=.005,absolute_V_OLS_drift_max_mV_per_min=2.,max_T_span_C=2.,minimum_samples=30),
  stability_basis='Pre-existing polarization qualification current/std and voltage-drift/temperature-span criteria, now explicit diagnostic on fixed-load windows; not new certified sensor accuracy',
  environment='Record RH/pressure/range/within-window variation, no post-hoc calipers and no claim of hydration/irreversible-aging equivalence',
  new_raw_cycle_medians='Audit-only all146 complete-cycle reconstruction from raw dwells, compare existing141; never overwrite or refit Ridge on extra5',
  sample_units=['raw points','nonoverlapping dwells','complete cycles','distinct observation times','one durability trajectory'],
  task_T0='Frozen48 half-hour raw14 histories ->24 dynamic half-hour DeltaV outputs',
  task_T1a='Same48 half-hour inputs and24 future bins; actual fixed-state dwell medians in each bin, missing labels stay missing',
  task_T1b='Cycle-level original fixed-state voltages, H1/H2/H3; inspect48half-hour-input availability without fitting',
  task_T2='Same real0.78923 fixed state reference, not independent labels fromT1; j1.0 polarization has7 real events only',
  split='Original N1 Train rows[0,2103), Validation[2103,2404); exclude full cycles crossing split boundaries from supervised labels; Train labels before first Validation origin; past feature reuse is explicit not label reuse',
  candidate_counts='Report dense24/24 and masked coverage separately; no interpolation, no arbitrary neural minimum sample threshold',
  chronological_selection='Any future fit must finish before validation origin, Train-only scaler and inner model selection; current frozen F2 cache not a prospective health model',
  no_global_retrospective_B2_or_isotonic_targets=True,
  decisions='A only if comparable new target fits current task; B if usable target requires a genuinely new task with justified independent validation; C if neural qualification not established. No GPU this round under any decision',
  stop='One fixed audit, no threshold tuning, no model fitting, no test values, no new paper-mirror export')
 js('configs/FROZEN_AUDIT.json',config);md('configs/FROZEN_AUDIT.sha256',sha(P/'configs/FROZEN_AUDIT.json'))
 print('FROZEN',len(targets),'candidate long dwells;',len(records),'historical sources')
if __name__=='__main__':main()

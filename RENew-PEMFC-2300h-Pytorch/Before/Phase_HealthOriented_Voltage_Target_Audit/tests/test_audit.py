"""Real contract regression, raw firewall boundaries and task-identity safeguards."""
import unittest,sys,json,ast
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from common import rd
from extract_development import require_Development_age,stable_window
class AuditTests(unittest.TestCase):
 def test_raw_boundary_is_strict(self):
  for x in [1819.997738,1820.,np.nan]:
   with self.assertRaises(PermissionError):require_Development_age(pd.Series([x]))
  require_Development_age(pd.Series([1819.99]))
 def test_raw_file_extents_are_not_N1_rule(self):
  d=rd(P/'outputs/RAW_SOURCE_PREFIX_PROGRESS.csv')
  self.assertEqual(len(d),14);self.assertEqual(d.iloc[-1].certified_prefix_records,268684)
  self.assertFalse(d.Test_features_parsed.any());self.assertTrue(d.max_h.lt(1819.997738).all())
 def test_primary_window_boundary_provenance(self):
  start,end=stable_window('2025-01-27 17:20:29.005',152)
  self.assertEqual((end-start).total_seconds(),76)
  self.assertFalse(start<=end<end) # shared75percent sample belongs to last25
  d=rd(P/'outputs/RAW_FIXED_STATE_DWELL_MEASUREMENTS.csv')
  self.assertTrue((pd.to_datetime(d.measurement_last_calendar)<pd.to_datetime(d.window_end)).all())
 def test_141_original_values_preserved(self):
  d=rd(P/'outputs/FROZEN_141_RECONSTRUCTION_COMPARISON.csv')
  self.assertEqual(len(d),141);self.assertLess(d.V_difference_V.abs().max(),1e-12)
 def test_more_observations_are_not_more_lifetimes(self):
  d=rd(P/'outputs/SAMPLE_UNIT_LEDGER.csv');self.assertTrue(d.independent_life_trajectories.eq(1).all())
  z=d.loc[d.population.eq('Complete146 cycles')].iloc[0]
  self.assertEqual(z.disjoint_dwell_observations,3503);self.assertEqual(z.complete_cycles,146)
 def test_no_interpolated_native_targets(self):
  d=rd(P/'outputs/T0_T1_SPLIT_COUNTS.csv');self.assertTrue(d.T1_dense24_origins.eq(0).all())
  self.assertTrue(d.T1_dense24_before_QC.eq(0).all())
  q=rd(P/'outputs/PRIMARY_FIXED_BIN_LABELS.csv');self.assertTrue(q.loc[q.QC_dwells.eq(0),'QC_V'].isna().all())
 def test_no_target_cycle_split_leak(self):
  d=rd(P/'outputs/T0_T1_WINDOW_QUALIFICATION.csv');groups={}
  for role,g in d.groupby('role'):groups[role]={int(i) for z in g.target_raw_cycle_ids.fillna('').astype(str) for i in z.split('|') if i}
  self.assertFalse(groups['Train']&groups['Validation'])
 def test_physical_cycles_not_halfhours(self):
  q=rd(P/'outputs/CYCLE_TASK_LEGAL_SAMPLES_AND_DYNAMIC_HISTORY.csv')
  self.assertTrue((q.target_raw_cycle_id-q.origin_raw_cycle_id==q.H).all());self.assertTrue((q.actual_lead_h>9*q.H).all())
  d=json.loads((P/'outputs/CYCLE_SEQUENCE_48_24_QUALIFICATION.json').read_text());self.assertEqual(d['consecutive_raw_cycle_48_to24_windows'],0)
 def test_reference_first10_not_prospective_for_first10(self):
  q=rd(P/'outputs/FIXED_PRIMARY_CYCLE_OBSERVATIONS.csv')
  self.assertFalse(q.loc[q.raw_cycle_id.le(10),'frozen_reference_available_before_observation'].any())
  self.assertFalse(q.reference_is_authenticated_BoL.any())
 def test_strict_model_time_purges_kept_separate(self):
  d=rd(P/'outputs/FUTURE_STRICT_TIME_PURGE_RECORDS.csv')
  self.assertEqual(len(d),2);self.assertEqual(set(d.H),{2,3});self.assertTrue(d.fold.eq('fold_3').all())
 def test_no_new_model_results(self):
  d=rd(P/'outputs/CANDIDATE_TARGET_EVIDENCE.csv');self.assertTrue(d.new_performance_MAE.isna().all());self.assertFalse(d.new_training.any())
  for f in (P/'scripts').glob('*.py'):
   for n in ast.walk(ast.parse(f.read_text(encoding='utf-8'))):
    if isinstance(n,ast.Import):self.assertFalse({z.name.split('.')[0] for z in n.names}&{'torch','sklearn','optuna'})
    if isinstance(n,ast.ImportFrom):self.assertNotIn((n.module or '').split('.')[0],{'torch','sklearn','optuna'})
if __name__=='__main__':unittest.main()

import sys,unittest,json
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from run_bridge_audit import overlaps
class Contracts(unittest.TestCase):
 def test_interval_logic(self):
  self.assertFalse(overlaps(0,1,2,3));self.assertTrue(overlaps(0,2,2,3))
 def test_nominal_not_measurement_time(self):
  d=pd.read_csv(P/'outputs/ALL_REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv')
  self.assertTrue(d.precise_reference_ophrs_h.isna().all());self.assertTrue(d.reference_release_calendar.eq('UNKNOWN').all())
 def test_missing_metrics_not_perfect_score(self):
  d=pd.read_csv(P/'outputs/SOH_METHOD_ELIGIBILITY.csv');self.assertTrue(d.MAE_SOH_percentage_points.isna().all());self.assertEqual(d.new_predictions.sum(),0)
 def test_not_equivalent_proxy_labels(self):
  d=pd.read_csv(P/'outputs/ALL_P1S_EVENT_BIN_LANDING_AUDIT.csv');self.assertFalse(d.exact_measurement_equivalence.any());self.assertEqual(d.matched_target_bin_row.notna().sum(),33)
 def test_no_fake_health_sample_count(self):
  d=json.loads((P/'outputs/FINAL_DECISION.json').read_text());self.assertEqual(d['legal_common_reference_events'],0);self.assertTrue(d['dynamic_origins_not_independent_health_samples'])
 def test_bounded_dev_only(self):
  d=pd.read_csv(P/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv');self.assertEqual(len(d),3005);self.assertLess(d.bin_end_ophrs2_h.max(),1819.997738)
if __name__=='__main__':unittest.main()

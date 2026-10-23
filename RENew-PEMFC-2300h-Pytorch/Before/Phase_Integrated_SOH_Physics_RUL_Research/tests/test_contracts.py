import sys,unittest,json
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from common import config,metrics
class Contracts(unittest.TestCase):
 def test_protocol_source_identities(self):self.assertTrue(config()['no_Test'])
 def test_separate_normalizations(self):
  c=config();self.assertNotEqual(c['reference_voltage_V'],c['P1S_reference_voltage_V'])
 def test_early_refs_not_manufactured(self):
  d=pd.read_csv(P/'outputs/SOH_EVENT_EXCLUSIONS.csv');self.assertEqual(set(d.nominal_h),{300,600,850});self.assertTrue(d.reason.str.contains('NO_EXACT').all())
 def test_exact_forecast_keys_and_labels(self):
  d=pd.read_csv(P/'outputs/P1S_TO_SOH_EXACT_FORECAST_LINKS.csv');self.assertTrue((d.target_raw_cycle_id-d.origin_raw_cycle_id==d.horizon.map({'H1':1,'H3':3})).all());self.assertTrue((d.max_Train_label_end_h<=d.origin_conservative_available_h).all())
 def test_calibration_excludes_future_and_bol(self):
  m=json.loads((P/'models/PAST_ONLY_AFFINE_PARAMETERS.json').read_text());self.assertEqual(len(m),6)
  for r in m:self.assertTrue(all(0<i<r['nominal_h'] for i in r['training_events']));self.assertGreaterEqual(len(r['training_events']),3)
 def test_no_online_claim(self):
  d=pd.read_csv(P/'outputs/SOH_EVENT_LEVEL_ELIGIBILITY.csv');self.assertFalse(d.eligible_formal.any());self.assertTrue(d.release_calendar.isna().all())
 def test_tft_parameter_availability(self):
  d=pd.read_csv(P/'outputs/TFT_PAST_INFORMATION_AND_PARAMETER_AVAILABILITY.csv');self.assertEqual(d.causal_cached_queries_available.sum(),0);self.assertEqual(d.model_selection_labels_available_before_origin.sum(),1)
 def test_metric_units(self):
  d=metrics([1.,1.],[.99,1.01]);self.assertAlmostEqual(d['MAE_SOH_pp'],1.);self.assertAlmostEqual(d['RMSE_SOH_pp'],1.)
 def test_no_continuous_soh_labels(self):
  d=pd.read_csv(P/'outputs/SOH_EVENT_LEVEL_ELIGIBILITY.csv');self.assertEqual(len(d),7)
 def test_no_retraining(self):
  d=json.loads((P/'outputs/STAGE_A_COMPLETE.json').read_text());self.assertEqual(d['Ridge_refits'],0);self.assertEqual(d['TFT_refits'],0)
if __name__=='__main__':unittest.main(verbosity=2)

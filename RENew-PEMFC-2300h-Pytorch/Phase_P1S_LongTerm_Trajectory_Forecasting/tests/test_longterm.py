import inspect,json,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from common import config,firewall,ROOT,metric
from methods import trend,train_new
class LongTerm(unittest.TestCase):
 def test_frozen_identities(self):self.assertTrue(config()['no_Test'])
 def test_real_counts_no_interpolated_labels(self):
  d=json.loads((P/'outputs/TRAJECTORY_COUNTS.json').read_text());self.assertEqual(d['real_distinct_middle50_measurements'],147);self.assertEqual(d['causal_forecast_observations'],131);self.assertEqual(d['interpolated_health_labels'],0)
 def test_warmup_and_partial_not_labels(self):
  d=pd.read_csv(P/'outputs/USED_SAMPLES.csv');self.assertFalse(d.target_cycle_interval_id.str.contains('PARTIAL').any());self.assertTrue((d.target_raw_cycle_id>10).all())
 def test_new_physical_horizons_exact(self):
  d=pd.read_csv(P/'outputs/PREDICTIONS.csv');np.testing.assert_array_equal(d.target_raw_cycle_id-d.origin_raw_cycle_id,d.H)
 def test_no_future_time_in_trend_interface(self):
  self.assertEqual(list(inspect.signature(trend).parameters),['origin_R','slope_R_per_h','known_horizon_cycles','past_cadence_h']);self.assertAlmostEqual(trend(1.,-.001,3,9.),.973)
 def test_native_fit_forbidden(self):
  with self.assertRaises(AssertionError):train_new(None,1,'fold_1',[])
 def test_true_lead_is_not_fixed_twelve_h(self):
  d=pd.read_csv(P/'outputs/ACTUAL_LEAD_AUDIT.csv');self.assertTrue((d[d.H.eq(1)].actual_operating_h_max<12).all());self.assertTrue((d[d.H.eq(1)].actual_calendar_h_max>300).any())
 def test_h12_missing_fold_not_zero_error(self):
  d=pd.read_csv(P/'outputs/FOLD_METRICS.csv');z=d[d.H.eq(12)&d.fold.eq('fold_2')];self.assertTrue(z.n.eq(0).all());self.assertTrue(z.MAE_mVeq.isna().all())
 def test_h12_unavailable_ridge_not_fake_scores(self):
  d=pd.read_csv(P/'outputs/PREDICTIONS.csv');self.assertFalse(((d.H==12)&(d.fold=='fold_1')&(d.method=='RIDGE96')).any())
 def test_no_partial_mean_as_threefold(self):
  d=pd.read_csv(P/'outputs/THREE_FOLD_EQUAL_MEANS.csv');self.assertTrue(d[d.H.eq(12)].MAE_mVeq.isna().all());self.assertFalse(d[d.H.eq(12)].threefold_evaluable.any())
 def test_ohmic_gauge(self):
  j=.789229989;e=.9;r=.05;c=.02;self.assertAlmostEqual(e-j*r,e+j*c-j*(r+c))
 def test_physics_na_not_zero(self):
  d=pd.read_csv(P/'outputs/REAL_TRAIN_PHYSICS_IDENTIFIABILITY.csv');self.assertTrue(d.real_certified_physics_loss.isna().all());self.assertTrue(d.ohmic_design_rank.eq(1).all())
 def test_independent_verifier(self):self.assertTrue(json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text())['PASS'])
 def test_no_native_refit_or_new_soh(self):
  d=json.loads((P/'outputs/RUN_LEDGER.json').read_text());self.assertEqual(d['native_H1_H3_refits'],0);self.assertEqual(d['new_SOH_or_RUL_fits'],0);self.assertEqual(d['GPU_runs'],0)
 def test_metric_units(self):self.assertAlmostEqual(metric([1.],[.999])['MAE_mVeq'],.71383023635)
 def test_no_causal_tft_train_samples(self):
  d=pd.read_csv(P/'outputs/TFT_PARAMETER_TIME_AND_HEALTH_SUPPORT.csv');self.assertEqual(d[d.role.eq('train')].parameters_available_by_origin.sum(),0)
 def test_recovery_plot_keeps_both_boolean_strata(self):
  from render_figures import recovery_panel_data
  d=pd.DataFrame({'H':[1,1],'method':['PERSISTENCE','PERSISTENCE'],'state':[False,True],'n':[9,3],'MAE_mVeq':[.2,.1]})
  r=recovery_panel_data(d,1,'PERSISTENCE');self.assertEqual(r['n'].tolist(),[9,3]);self.assertEqual(r.MAE_mVeq.tolist(),[.2,.1])
 def test_protected_paths(self):
  firewall()
  with self.assertRaises(PermissionError):(ROOT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv').open('rb')
  with self.assertRaises(PermissionError):(ROOT/'Phase_P1S_96h_ThreeFold_Comparison/outputs/PREDICTIONS.csv').open('wb')
if __name__=='__main__':unittest.main(verbosity=2)

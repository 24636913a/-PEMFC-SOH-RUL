import ast,inspect,json,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from common import methods,decomposition,config,firewall,ROOT,METHODS
class ContributionTests(unittest.TestCase):
 def test_source_and_frozen_calculation_identity(self):self.assertTrue(config()['no_training'])
 def test_three_methods_same_calibration(self):
  p=methods(.98,.99,.991,1.2,-.2);self.assertAlmostEqual(p[METHODS[1]],.988);self.assertAlmostEqual(p[METHODS[2]],.9892)
 def test_prediction_signature_has_no_target(self):
  self.assertEqual(list(inspect.signature(methods).parameters),['prev_ref','origin_R','predicted_R','a','b'])
 def test_exact_forecast_is_only_calibration_residual(self):
  d=decomposition(.97,.98,.99,.99,1.,0.);self.assertAlmostEqual(d['ridge_proxy_error_translated_SOH_pp'],0.);self.assertAlmostEqual(d['ridge_final_error_SOH_pp'],2.)
 def test_worse_proxy_can_improve_SOH_by_cancellation(self):
  d=decomposition(.98,.97,.979,.97,1.,0.);self.assertTrue(d['ridge_error_cancels_calibration']);self.assertLess(abs(d['ridge_final_error_SOH_pp']),abs(d['calibration_at_actual_target_SOH_pp']))
 def test_error_terms_same_direction_amplify(self):
  d=decomposition(.97,.98,1.,.99,1.,0.);self.assertFalse(d['ridge_error_cancels_calibration']);self.assertAlmostEqual(d['ridge_cancellation_magnitude_SOH_pp'],0.)
 def test_cancelled_sum_does_not_prove_proxy_accuracy(self):
  d=decomposition(.98,.97,.98,.97,1.,0.);self.assertAlmostEqual(d['ridge_final_error_SOH_pp'],0.);self.assertGreater(abs(d['ridge_proxy_error_translated_SOH_pp']),0.)
 def test_18_predictions_but_only_three_events(self):
  d=pd.read_csv(P/'outputs/POINT_PREDICTIONS.csv');self.assertEqual(len(d),18);self.assertEqual(d.nominal_h.nunique(),3);self.assertFalse(d.duplicated(['nominal_h','horizon','method']).any())
 def test_measured_origin_available(self):
  d=pd.read_csv(P/'outputs/ORIGIN_MEASUREMENT_TIME_AND_IDENTITY_AUDIT.csv');self.assertTrue(d.measurement_already_available.all());self.assertTrue((d.origin_actual_P1S_measurement_end_h<=d.origin_conservative_available_h).all())
 def test_decomposition_identity_real_data(self):
  d=pd.read_csv(P/'outputs/ERROR_DECOMPOSITION.csv');np.testing.assert_allclose(d.calibration_at_actual_target_SOH_pp+d.ridge_proxy_error_translated_SOH_pp,d.ridge_final_error_SOH_pp,atol=1e-12,rtol=0)
 def test_no_online_promotion(self):
  d=pd.read_csv(P/'outputs/POINT_PREDICTIONS.csv');self.assertFalse(d.strict_online_qualified.any());self.assertTrue(d.reference_release_calendar.isna().all())
 def test_pairing_counts_not_six_independent(self):
  d=pd.read_csv(P/'outputs/PAIRWISE_COMPARISONS.csv');self.assertTrue((d.n_paired_events==3).all());self.assertTrue((d.candidate_wins+d.ties+d.candidate_losses==3).all())
 def test_increment_is_actual_difference_not_total_gain(self):
  d=pd.read_csv(P/'outputs/SEQUENTIAL_CONTRIBUTION.csv');np.testing.assert_allclose(d.recent_P1S_and_calibration_MAE_reduction_SOH_pp+d.Ridge_forecast_step_MAE_reduction_SOH_pp,d.total_MAE_reduction_vs_M0_SOH_pp,atol=1e-12,rtol=0)
 def test_independent_high_precision_verifier_pass(self):
  d=json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text());self.assertTrue(d['PASS']);self.assertEqual(d['method_predictions_verified'],18)
 def test_no_training_or_gpu_imports_in_code(self):
  for f in (P/'scripts').glob('*.py'):
   tree=ast.parse(f.read_text(encoding='utf-8'))
   for node in ast.walk(tree):
    if isinstance(node,(ast.Import,ast.ImportFrom)):
     names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
     self.assertFalse(any(s.split('.')[0] in ['torch','sklearn','tensorflow'] for s in names))
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):self.assertNotEqual(node.func.attr,'fit')
 def test_firewall_blocks_raw_mixed_N1_and_history_write(self):
  firewall()
  with self.assertRaises(PermissionError):(ROOT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv').open('rb')
  with self.assertRaises(PermissionError):(ROOT/'Phase_Integrated_SOH_Physics_RUL_Research/outputs/SOH_EVENT_PREDICTIONS.csv').open('wb')
if __name__=='__main__':unittest.main(verbosity=2)

import sys,json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from run_stage_bc import smooth_loss_grad
class Qualification(unittest.TestCase):
 def test_state_gauge_nullspace(self):
  z=np.array([1.,.98,.99]);d=np.array([1.,.99,.98]);q=z-d;shift=np.array([.2,-.1,.05]);np.testing.assert_allclose((d+shift)+(q-shift),z)
 def test_real_recovery_present(self):
  q=pd.read_csv(P/'outputs/PHYSICS_STATE_DECOMPOSITION.csv');self.assertTrue((q.observed_R.diff()>0).any())
 def test_smoothing_finite_difference(self):
  z=np.array([1.,.99,1.01,.98]);loss,g=smooth_loss_grad(z,.01);eps=1e-7
  fd=[(smooth_loss_grad(z+eps*np.eye(4)[i],.01)[0]-smooth_loss_grad(z-eps*np.eye(4)[i],.01)[0])/(2*eps) for i in range(4)]
  np.testing.assert_allclose(g,fd,atol=1e-5)
 def test_flat_trivial_solution(self):
  loss,g=smooth_loss_grad(np.ones(5),.01);self.assertEqual(loss,0);self.assertEqual(np.linalg.norm(g),0)
 def test_physics_unavailable_is_na(self):
  d=json.loads((P/'outputs/PHYSICS_CPU_DIAGNOSTICS.json').read_text());self.assertIsNone(d['real_certified_physics_loss']);self.assertIsNone(d['real_certified_model_parameter_gradient'])
 def test_equivalent_nonphysical_control(self):
  d=pd.read_csv(P/'outputs/PHYSICS_GRADIENT_DIAGNOSTIC.csv');np.testing.assert_array_equal(d.slow_prior_gradient,d.ordinary_smoothing_gradient)
 def test_thresholds_not_eol(self):
  d=pd.read_csv(P/'outputs/CONDITIONAL_CALENDAR_LIFETIME_SCENARIOS.csv');self.assertEqual(set(d.threshold),{.97,.95,.9});self.assertTrue(d.threshold_is_assumption.all());self.assertTrue(d.actual_failure_time.isna().all());self.assertTrue(d.actual_RUL_accuracy.isna().all())
 def test_flat_no_crossing(self):
  d=pd.read_csv(P/'outputs/CONDITIONAL_CALENDAR_LIFETIME_SCENARIOS.csv');q=d[d.model.eq('zero_slope_recovery_sensitivity')];self.assertTrue(q.conditional_crossing_calendar_h.isna().all())
 def test_no_training_permission_from_formula_only(self):
  d=json.loads((P/'outputs/FINAL_RESEARCH_DECISION.json').read_text());self.assertFalse(d['physics_training_qualified']);self.assertEqual(d['GPU_training_runs'],0)
if __name__=='__main__':unittest.main(verbosity=2)

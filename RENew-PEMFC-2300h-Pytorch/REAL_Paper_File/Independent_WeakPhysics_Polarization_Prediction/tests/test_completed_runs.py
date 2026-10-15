import sys,unittest,json,hashlib
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import P,sha,jl
class CompletedExperimentTests(unittest.TestCase):
 def test_all_runids_once_and_fixed_steps(self):
  records=list((P/'outputs/runs').glob('*_COMPLETE.json'));self.assertEqual(len(records),24)
  for q in records:self.assertEqual(json.loads(q.read_text())['optimizer_steps'],2500)
 def test_same_initial_state_each_AB(self):
  ids=[q for q in (P/'outputs/runs').glob('*_A_DATA_ONLY_STARTED.json')]
  for q in ids:
   a=json.loads(q.read_text());b=json.loads(q.with_name(q.name.replace('_A_DATA_ONLY_','_B_PHYSICS_')).read_text());self.assertEqual(a['initial_state_SHA256'],b['initial_state_SHA256']);self.assertEqual(a['Train_ids'],b['Train_ids']);self.assertEqual(a['Validation_ids'],b['Validation_ids'])
 def test_actual_physics_activation_not_falsely_inactive(self):
  d=pd.read_csv(P/'outputs/REAL_PHYSICS_ACTIVITY.csv');b=d[d.arm.eq('B_PHYSICS')];self.assertGreater(b.active_epochs.sum(),0);self.assertLessEqual(b.positive_loss_last_epoch.max(),8);self.assertTrue((b.physics_loss_last_preupdate==0).all())
 def test_no_claim_of_increment_when_both_final_monotone(self):
  d=jl('outputs/RESEARCH_DECISION.json');self.assertEqual(d['integration_decision'],'RETAIN_DESCRIPTIVE_PHYSICS_ONLY');self.assertFalse(d['TFT_integration_qualified']);self.assertFalse(d['SOH_integration_qualified'])
  for r in d['task_decisions']:self.assertEqual(r['status'],'PHYSICS_DESCRIPTIVE_ONLY_NO_INCREMENT');self.assertTrue(r['ordinary_final_all_monotone'])
 def test_independent_models_and_gradients_verified(self):
  d=jl('outputs/INDEPENDENT_VERIFICATION.json');self.assertEqual(d['status'],'PASS');self.assertEqual(d['runs_checked'],24);self.assertEqual(d['predictions_checked'],492);self.assertLess(d['NumPy_prediction_max_abs_difference_V'],1e-12)
 def test_completed_run_guard_does_not_touch_weights(self):
  import train
  q=P/'models/A_within_curve_42_A_DATA_ONLY.pt';before=sha(q);train.run_one({},dict(task='A',fold='within_curve'),None,None,42,'A_DATA_ONLY');self.assertEqual(sha(q),before)

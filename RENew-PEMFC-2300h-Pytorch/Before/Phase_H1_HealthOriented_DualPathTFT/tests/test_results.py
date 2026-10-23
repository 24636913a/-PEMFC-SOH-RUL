"""Post-training result invariants; these tests never fit a model."""
import sys,json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from common import rd,sha,load_cfg,VREF,ROOT
class ResultTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.pred=rd(P/'outputs/ALL_POINT_PREDICTIONS.csv');cls.met=rd(P/'outputs/FOLD_SEED_METRICS.csv')
  cls.cfg=load_cfg();cls.primary=rd(P/'outputs/PRIMARY_EQUAL_FOLD_METRICS.csv').set_index('model')
 def test_all_registered_combinations_and_identical_targets(self):
  self.assertEqual(len(self.pred),451);self.assertEqual(len(self.met),33)
  for ff,n in [('fold_1',13),('fold_2',6),('fold_3',22)]:
   group=self.pred.loc[self.pred.fold.eq(ff)];self.assertEqual(group.sample_key.nunique(),n)
   self.assertEqual(len(group),11*n)
   self.assertTrue(group.groupby('sample_key').true_voltage_V.nunique().eq(1).all())
   self.assertTrue(group.groupby('sample_key').target_raw_cycle_id.nunique().eq(1).all())
   for name in ['H-F0','H-F1','H-F2']:
    for seed in [42,52,62]:self.assertEqual(len(group.loc[group.model.eq(name)&group.seed.eq(seed)]),n)
 def test_train_only_epoch_choice_and_all_54_weights(self):
  folders=list((P/'outputs/training_logs').glob('fold_*'));self.assertEqual(len(folders),27)
  for folder in folders:
   done=json.loads((folder/'COMPLETION.json').read_text());self.assertEqual(done['status'],'COMPLETE')
   self.assertFalse(done['outer_used_for_selection']);self.assertTrue(done['checkpoint_reload_exact'])
   rows=[json.loads(z) for z in (folder/'inner_epochs.jsonl').read_text().splitlines()]
   best=float('inf');epoch=None
   for r in rows:
    if r['inner_selection_standardized_RMSE']<best-1e-12:best=r['inner_selection_standardized_RMSE'];epoch=r['epoch']
   self.assertEqual(epoch,done['final']['selected_epoch']);self.assertEqual(epoch,done['final']['optimizer_epochs'])
   self.assertEqual(len((folder/'final_epochs.jsonl').read_text().splitlines()),epoch)
   self.assertEqual(done['inner']['initialization_sha256'],done['final']['initialization_sha256'])
   self.assertLess(pd.Timestamp(done['replay_model_available_calendar']),pd.Timestamp(done['earliest_Validation_origin_calendar']))
   for level in ['inner','final']:self.assertEqual(sha(P/done[level]['checkpoint']),done[level]['checkpoint_sha256'])
  self.assertEqual(len(list((P/'outputs/checkpoints').glob('*.pt'))),54)
 def test_equal_fold_primary_independently_recalculated(self):
  for name,g in self.pred.groupby('model'):
   maes=[];rmses=[]
   for ff,fold in g.groupby('fold'):
    m=[];r=[]
    for _,seed in fold.groupby('seed'):
     e=1000*(seed.predicted_voltage_V-seed.true_voltage_V);m.append(abs(e).mean());r.append(np.sqrt(np.mean(e**2)))
    maes.append(np.mean(m));rmses.append(np.mean(r))
   self.assertAlmostEqual(np.mean(maes),self.primary.loc[name,'MAE_mV'],12)
   self.assertAlmostEqual(np.mean(rmses),self.primary.loc[name,'RMSE_mV'],12)
 def test_frozen_decision_and_all_negatives_retained(self):
  d=json.loads((P/'outputs/FINAL_DECISION.json').read_text());q=self.primary
  gain=(q.loc['Persistence','MAE_mV']-q.loc['H-F2','MAE_mV'])/q.loc['Persistence','MAE_mV']
  self.assertAlmostEqual(100*gain,d['F2_improvement_vs_Persistence_percent'],10)
  self.assertTrue(d['rule_frozen_before_results']);self.assertTrue(d['threefold_complete'])
  if gain<=0:self.assertEqual(d['status'],'H1_HEALTH_TFT_NO_GAIN')
  self.assertFalse(d['reference_standard_SOH_qualified']);self.assertFalse(d['historical_online_deployment_claimed'])
 def test_proxy_units_no_clipping_or_fake_standard_SOH(self):
  for col,source in [('predicted_SOH_V_percent','predicted_voltage_V'),('true_SOH_V_percent','true_voltage_V')]:
   self.assertLess(np.max(abs(self.pred[col]-100*self.pred[source]/VREF)),5e-12)
  self.assertTrue((self.met.MAE_SOH_pp-self.met.MAE_mV/(10*VREF)).abs().lt(1e-12).all())
  r=json.loads((P/'outputs/REFERENCE_QUALIFICATION.json').read_text())
  self.assertEqual(r['standard_SOH_qualified_pairs'],0);self.assertIsNone(r['new_standard_SOH_error'])
  self.assertTrue(r['all_forecast_origins_after_reference_confirmation']);self.assertFalse(r['physical_loss_added'])
  samples=rd(P/'data_contracts/USED_SAMPLES.csv')
  self.assertTrue(samples.origin_confirmed_h.gt(r['reference_first10_complete_cycle_confirmation_h']).all())
 def test_global_tails_are_actual_maxima(self):
  table=rd(P/'outputs/POOLED_GLOBAL_TAIL_AND_BIAS.csv').set_index('model')
  for name,g in self.pred.groupby('model'):
   e=1000*(g.predicted_voltage_V-g.true_voltage_V)
   self.assertAlmostEqual(abs(e).max(),table.loc[name,'MAX_mV'],12)
 def test_independent_gpu_replay_and_preserved_cross_device_failure(self):
  r=json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text());self.assertEqual(r['status'],'PASS')
  self.assertEqual(r['same_GPU_checkpoint_replays_exact'],27);self.assertEqual(r['optimizer_steps_in_verification'],0)
  rows=rd(P/'outputs/INDEPENDENT_CHECKPOINT_REPLAYS.csv');self.assertTrue(rows.same_GPU_standardized_replay_exact.all())
  self.assertEqual(int((~rows.original_CPU_precision_probe_PASS).sum()),r['crossdevice_probe_failures'])
  self.assertTrue((P/'outputs/failures/FIRST_INDEPENDENT_CPU_GPU_PROBE_FAILURE.json').exists())
 def test_frozen_code_data_and_all_source_bytes_unchanged(self):
  for path,h in {**self.cfg['training_code_sha256'],**self.cfg['data_sha256']}.items():self.assertEqual(sha(P/path),h,path)
  for ledger in ['SOURCE_SHA256.csv','REFERENCE_SOURCE_SHA256.csv']:
   for row in rd(P/'data_contracts'/ledger).itertuples():self.assertEqual(sha(ROOT/row.path),row.sha256,row.path)
if __name__=='__main__':unittest.main()

import sys,json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from common import rd,prep,protected_pass,VREF
class ProtocolTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.s=rd(P/'data_contracts/USED_SAMPLES.csv');cls.inner=rd(P/'data_contracts/INNER_SPLITS.csv');cls.x=np.load(P/'data_contracts/RAW_DYNAMIC_HISTORIES.npz')['x'];cls.rec=json.loads((P/'data_contracts/TRAIN_ONLY_PREPROCESSING.json').read_text())
 def test_exact_next_raw_cycle_and_no_new_five(self):
  self.assertTrue((self.s.target_raw_cycle_id-self.s.origin_raw_cycle_id==1).all());self.assertLessEqual(self.s.target_raw_cycle_id.max(),150)
 def test_confirmed_cycle_and_wholly_future_measurements(self):
  self.assertTrue((pd.to_datetime(self.s.origin_confirmed_calendar)>pd.to_datetime(self.s.origin_cycle_end_calendar)).all())
  self.assertTrue((self.s.target_first_measurement_h>self.s.origin_confirmed_h).all())
 def test_history_has_no_future(self):
  self.assertTrue((self.s.history_last_h<=self.s.origin_confirmed_h).all())
  self.assertTrue((pd.to_datetime(self.s.history_last_calendar)<=pd.to_datetime(self.s.origin_confirmed_calendar)).all())
  self.assertEqual(self.x.shape[1:],(48,14));self.assertTrue(np.isfinite(self.x).all())
 def test_target_is_frozen_absolute_voltage(self):
  old=rd(P.parent/'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv');old=old.loc[old.stable_window.eq('middle_50')].set_index('raw_cycle_id')
  self.assertTrue(np.array_equal(self.s.true_voltage_V.to_numpy(),old.loc[self.s.target_raw_cycle_id,'P1S_voltage_V'].to_numpy()))
 def test_outer_and_inner_time_isolation(self):
  for fold,g in self.s.groupby('fold'):
   tr=g.loc[g.role.eq('train')];va=g.loc[g.role.eq('validation')];self.assertFalse(set(tr.target_raw_cycle_id)&set(va.target_raw_cycle_id))
   self.assertLess(pd.to_datetime(tr.target_available_calendar).max()+pd.Timedelta(hours=1),pd.to_datetime(va.origin_confirmed_calendar).min())
   inn=self.inner.loc[self.inner.fold.eq(fold)];fit=tr.loc[tr.sample_key.isin(inn.loc[inn.inner_role.eq('fit'),'sample_key'])];sel=tr.loc[tr.sample_key.isin(inn.loc[inn.inner_role.eq('selection'),'sample_key'])]
   self.assertLess(pd.to_datetime(fit.target_available_calendar).max()+pd.Timedelta(hours=1),pd.to_datetime(sel.origin_confirmed_calendar).min())
 def test_train_only_scaler_mask_and_target_scale(self):
  for fold,g in self.s.groupby('fold'):
   for level in ['inner','outer']:
    r=self.rec[fold][level];q=g.loc[g.sample_key.isin(r['sample_keys'])];self.assertTrue(q.role.eq('train').all());a=self.x[q.array_index].astype('float64').reshape(-1,14)
    self.assertTrue(np.array_equal(a.mean(axis=0),np.array(r['x_mean'])));self.assertAlmostEqual(q.true_voltage_V.mean(),r['y_mean_V'],14)
    self.assertEqual(len(r['top10']),10);self.assertIn(0,r['top10'])
    if level=='inner':self.assertEqual(set(q.sample_key),set(self.inner.loc[self.inner.fold.eq(fold)&self.inner.inner_role.eq('fit'),'sample_key']))
 def test_EWMA_prefix_causality(self):
  r=self.rec['fold_1']['outer'];a=self.x[:2].copy();b=a.copy();b[:,25:,0]+=10
  self.assertTrue(np.array_equal(prep(a,r)[:,:25],prep(b,r)[:,:25]))
 def test_reference_formed_and_proxy_units(self):
  self.assertTrue(self.s.reference_available.all());self.assertTrue(self.s.origin_confirmed_h.gt(171.82091).all());self.assertAlmostEqual((.001/VREF)*100,1/(10*VREF))
 def test_native_engineering_evidence(self):
  e=json.loads((P/'outputs/ENGINEERING_EVIDENCE.json').read_text());self.assertEqual(e['status'],'PASS')
  self.assertEqual(e['counts'],{'F0':323483,'F1':332379,'F2':332436})
  self.assertTrue(all(r['checkpoint_reload_exact'] and r['no_future_decoder_variables'] for r in e['groups']))
 def test_legacy_sources_unchanged(self):self.assertTrue(protected_pass())
if __name__=='__main__':unittest.main()

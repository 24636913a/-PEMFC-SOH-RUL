"""Production guards and independent metrics; no real Assessment forward in tests."""
import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import i2_common as q
from audit_i2_predictions import compare_metrics, audit_cache
import numpy as np
import pandas as pd
import torch

class I2Guards(unittest.TestCase):
 def setUp(self):
  origins=np.repeat(np.arange(2404,2782),24);steps=np.tile(np.arange(1,25),378)
  truth=np.sin((origins+steps)/10)*.01+steps*.0001
  self.truth=pd.DataFrame({'origin_row':origins,'target_row':origins+steps,'horizon_step':steps,
   'origin_age_h':1400+(origins-2404)*.5,'target_age_h':1400+(origins+steps-2404)*.5,'y_true_delta_V':truth})
  self.spec=q.RUNS[0]
  self.frame=self.truth.copy();self.frame['y_pred_delta_V']=truth+np.cos(steps)*.001
  for k,v in {**self.spec,'protocol_sha256':'protocol','checkpoint_sha256':'checkpoint','assessment_input_fingerprint':'input'}.items():self.frame[k]=v
  self.cfg={'checkpoints_by_run':{self.spec['run_id']:{'sha256':'checkpoint'}},'input_fingerprints':{'assessment_input_sha256':'input'}}
 def check(self,frame):
  with patch.object(q,'digest',return_value='protocol'):return q.validate_frame(frame,self.truth,self.spec,self.cfg)
 def test_valid_full_frame_and_shuffled_alignment(self):
  result=self.check(self.frame.sample(frac=1,random_state=42))
  np.testing.assert_array_equal(result.y_pred_delta_V.to_numpy(),self.frame.y_pred_delta_V.to_numpy())
 def test_missing_key(self):
  with self.assertRaises(RuntimeError):self.check(self.frame.iloc[:-1])
 def test_duplicate_key(self):
  f=self.frame.copy();f.iloc[1]=f.iloc[0]
  with self.assertRaises(RuntimeError):self.check(f)
 def test_truth_mutation(self):
  f=self.frame.copy();f.loc[0,'y_true_delta_V']+=1e-6
  with self.assertRaises(RuntimeError):self.check(f)
 def test_target_age_boundary(self):
  f=self.frame.copy();f.loc[0,'target_age_h']=1819.997738
  t=self.truth.copy();t.loc[0,'target_age_h']=1819.997738
  with patch.object(q,'digest',return_value='protocol'), self.assertRaises(RuntimeError):q.validate_frame(f,t,self.spec,self.cfg)
 def test_wrong_target_row(self):
  f=self.frame.copy();f.loc[0,'target_row']+=1
  with self.assertRaises(RuntimeError):self.check(f)
 def test_wrong_seed(self):
  f=self.frame.copy();f['seed']=52
  with self.assertRaises(RuntimeError):self.check(f)
 def test_nonfinite_prediction(self):
  f=self.frame.copy();f.loc[0,'y_pred_delta_V']=np.nan
  with self.assertRaises(RuntimeError):self.check(f)
 def test_project_independent_metrics_and_quantiles(self):
  metrics,checks=compare_metrics(self.frame)
  self.assertEqual(len(checks),8);self.assertTrue(all(x['pass'] for x in checks));self.assertGreater(metrics['RMSE_mV'],0)
 def test_target_equal_not_prediction_ensemble(self):
  f=pd.DataFrame({'target_row':[10,10,11],'horizon_step':[24,1,24], 'y_true_delta_V':[0.,0.,1.],'y_pred_delta_V':[1.,-1.,1.]})
  m=q.independent_metrics(f)
  self.assertAlmostEqual(m['target_time_equal_RMSE_mV'],np.sqrt(.5)*1000)
  self.assertNotEqual(m['target_time_equal_RMSE_mV'],0)
 def test_target_sse_and_horizon_decomposition(self):
  _,targets,horizons=q.diagnostics(self.frame)
  self.assertEqual(len(horizons),24)
  expected=np.square((self.frame.y_pred_delta_V-self.frame.y_true_delta_V)*1000).sum()
  self.assertAlmostEqual(targets.SSE_mV2.sum(),expected,places=8)
 def test_full_n1_digest_denied(self):
  with self.assertRaisesRegex(RuntimeError,'FULL_N1'):q.digest(q.c.N1)
 def test_test_and_raw_paths_denied(self):
  for p in [q.PROJECT/'Test_data.csv',q.REPO/'数据/Loadcycle_2000.csv']:
   with self.assertRaises(RuntimeError):q.allowed(p)
 def test_old_phase_write_denied(self):
  with self.assertRaises(RuntimeError):q.write_target(q.i.LEDGER)
 def test_one_shot_guard(self):
  with tempfile.TemporaryDirectory(dir=q.PHASE) as d:
   path=Path(d)/'ledger.json';path.write_text('{}',encoding='utf-8')
   with self.assertRaises(RuntimeError):q.assert_fresh(path)
 def test_hash_mutation_in_complete_path_set(self):
  with tempfile.TemporaryDirectory(dir=q.PHASE) as d:
   paths=[Path(d)/f'{j}.json' for j in range(3)]
   for j,p in enumerate(paths):p.write_text(str(j),encoding='utf-8')
   records={str(p):q.digest(p) for p in paths};q.verify_hashes(records)
   records[str(paths[1])]='0'*64
   with self.assertRaises(RuntimeError):q.verify_hashes(records)
 def test_state_mutation_and_flags_production_guard(self):
  m=torch.nn.Linear(2,1);before=q.ir.snapshot(m);flags={k:p.requires_grad for k,p in m.named_parameters()}
  q.verify_state(q.ir.state_audit(m,before,flags))
  with torch.no_grad():m.weight.add_(1)
  with self.assertRaises(RuntimeError):q.verify_state(q.ir.state_audit(m,before,flags))
 def test_missing_cache_fails_closed(self):
  with tempfile.TemporaryDirectory(dir=q.PHASE) as d,patch('audit_i2_predictions.OUT',Path(d)):
   with self.assertRaisesRegex(RuntimeError,'REQUIRED_CACHE'):audit_cache(self.spec,self.cfg,self.truth)
 def test_scientific_rule_and_negative_difference_semantics(self):
  p=pd.DataFrame({'D20_F2_minus_F0_mV':[-1,-2,-3],'D21_F2_minus_F1_mV':[-1,-2,-3]})
  self.assertEqual(q.scientific_decision(p)[0],'F2_REUSED_ASSESSMENT_MULTISEED_GAIN_OBSERVED')
  p.loc[0,'D20_F2_minus_F0_mV']=4
  self.assertEqual(q.scientific_decision(p)[0],'F2_GATE_GAIN_WITHOUT_NATIVE_SUPERIORITY')
 def test_production_static_no_training_or_semantics_change(self):
  self.assertTrue(q.static_audit()['pass'])
  source=(q.PHASE/'scripts/run_i2_assessment.py').read_text(encoding='utf-8')
  self.assertIn('actual=e.predict(model,x,32,device)',source)
  self.assertNotIn('torch.inference_mode(',source);self.assertNotIn('requires_grad_(False)',source)
 def test_prefix_suffix_invariance_and_development_mutation(self):
  # Real prescribed reader, synthetic CSVs only; prefix has 3005 legal +2 excluded rows.
  ages=np.r_[np.arange(3005)*.5,1820.,1820.5]
  a=pd.DataFrame({'bin_end_ophrs2_h':ages,'cell_voltage_mean':np.ones(3007)})
  with tempfile.TemporaryDirectory(dir=q.PHASE) as d:
   p=Path(d)/'synthetic.csv'
   def read(source):
    source.to_csv(p,index=False)
    with patch.object(q.c,'N1',p),patch.object(Path,'read_bytes',side_effect=AssertionError('FULL_FILE_BYTES_FORBIDDEN')):
     dev,access=q.i.read_dev()
    self.assertEqual(len(dev),3005);self.assertEqual(access[0]['nrows'],3007)
    return dev,hashlib.sha256(dev.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()).hexdigest()
   aa,ha=read(pd.concat([a,pd.DataFrame({'bin_end_ophrs2_h':[1900.],'cell_voltage_mean':[99.]})],ignore_index=True))
   bb,hb=read(pd.concat([a,pd.DataFrame({'bin_end_ophrs2_h':[2200.,2300.],'cell_voltage_mean':[-999.,123.]})],ignore_index=True))
   pd.testing.assert_frame_equal(aa,bb);self.assertEqual(ha,hb)
   changed=a.copy();changed.loc[0,'cell_voltage_mean']=.99
   _,hc=read(changed);self.assertNotEqual(ha,hc)
 def test_unbounded_inherited_reader_rejected(self):
  with patch.object(q.c,'load_development',side_effect=lambda cfg:pd.read_csv(q.c.N1)):
   with self.assertRaisesRegex(RuntimeError,'UNBOUNDED_N1_READ'):q.i.read_dev()
 def test_code_identity_mutation_rejected(self):
  if not q.PROTOCOL.exists():self.skipTest('Pre-freeze; mandatory rerun after protocol freeze')
  identity=q.own_code();identity['sha256']='0'*64
  with patch.object(q,'own_code',return_value=identity),self.assertRaisesRegex(RuntimeError,'CODE_IDENTITY_CHANGED'):q.load_protocol()

if __name__=='__main__':unittest.main()

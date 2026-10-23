"""Synthetic tests exercise the production guards, real models and metric APIs."""
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import i_common as a
from i_common import np,pd,torch
from i_metrics import project_metrics,independent_metrics,scientific_decision
from prepare_i_assessment import static_inference_audit
from audit_i_predictions import audit_all

def example():
 origins=np.repeat([2404,2405],24);steps=np.tile(np.arange(1,25),2)
 truth=pd.DataFrame({'origin_row':origins,'target_row':origins+steps,'horizon_step':steps,'origin_age_h':1500+origins*.01,'target_age_h':1500+(origins+steps)*.01,'y_true_delta_V':np.arange(48,dtype=float)/1000})
 frame=truth.copy();frame['y_pred_delta_V']=frame.y_true_delta_V+np.linspace(-.008,.01,48)
 spec=a.RUNS[0]
 for k,v in {**spec,'protocol_sha256':'p','checkpoint_sha256':'c','assessment_input_fingerprint':'x'}.items():frame[k]=v
 return frame,truth,spec

class Guards(unittest.TestCase):
 def validate(self,frame,truth,spec):return a.validate_predictions(frame,truth,spec,'p','c','x',expected_records=48)
 def test_roundtrip_shuffled_alignment(self):
  frame,truth,spec=example()
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'synthetic.csv';frame.sample(frac=1,random_state=2026).to_csv(path,index=False,float_format='%.17g');re=pd.read_csv(path,float_precision='round_trip')
   ordered=self.validate(re,truth,spec);self.assertTrue(np.array_equal(ordered.y_pred_delta_V,frame.y_pred_delta_V))
   for k,v in project_metrics(frame).items():self.assertAlmostEqual(v,independent_metrics(re)[k],places=10)
 def test_missing_key_fails(self):
  f,t,s=example()
  with self.assertRaisesRegex(RuntimeError,'COUNT'):self.validate(f.iloc[:-1],t,s)
 def test_duplicate_key_fails(self):
  f,t,s=example();f.iloc[1]=f.iloc[0]
  with self.assertRaisesRegex(RuntimeError,'DUPLICATE'):self.validate(f,t,s)
 def test_truth_misalignment_fails(self):
  f,t,s=example();f.loc[0,'y_true_delta_V']+=1e-8
  with self.assertRaisesRegex(RuntimeError,'TRUTH'):self.validate(f,t,s)
 def test_wrong_seed_fails(self):
  f,t,s=example();f['seed']=52
  with self.assertRaisesRegex(RuntimeError,'IDENTITY'):self.validate(f,t,s)
 def test_test_boundary_fails(self):
  f,t,s=example();f['target_age_h']=1820.;t['target_age_h']=1820.
  with self.assertRaisesRegex(RuntimeError,'BOUNDARY'):self.validate(f,t,s)
 def test_wrong_target_relation_fails(self):
  f,t,s=example();f.loc[0,'target_row']=3005;t.loc[0,'target_row']=3005
  with self.assertRaisesRegex(RuntimeError,'BOUNDARY'):self.validate(f,t,s)
 def test_one_shot_refuses_existing_ledger(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'ledger.json';path.write_text('{}')
   with self.assertRaisesRegex(RuntimeError,'NO_RERUN'):a.assert_fresh(path)
 def test_full_N1_hash_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'FULL_N1'):a.digest(a.c.N1)
 def test_Test_raw_source_not_allowlisted(self):
  for path in [a.PROJECT/'Test/raw.csv',a.REPO/'数据/Loadcycle.csv',a.PROJECT/'RUL90/full.csv']:
   with self.assertRaisesRegex(RuntimeError,'NOT_ALLOWLISTED'):a.check_read(path)
 def test_missing_cache_audit_fails(self):
  f,t,s=example()
  with tempfile.TemporaryDirectory() as folder:
   out=Path(folder);(out/'preparation_audit').mkdir();(out/'execution_ledger').mkdir()
   t.to_csv(out/'preparation_audit/I_ASSESSMENT_KEY_TRUTH_AUTHORITY.csv',index=False)
   (out/'execution_ledger'/f"{s['run_id']}_result.json").write_text('{"prediction_file_sha256":"none"}')
   with patch('audit_i_predictions.OUT',out),patch('audit_i_predictions.check_read',side_effect=lambda x:Path(x)),patch('audit_i_predictions.read_json',side_effect=lambda x:__import__('json').loads(Path(x).read_text())):
    with self.assertRaisesRegex(RuntimeError,'MISSING_OR_CHANGED'):audit_all({'checkpoints':[{}]})
 def test_no_scaler_refit(self):
  with patch.object(a.e.CommonScaler,'fit',side_effect=RuntimeError('FIT_FORBIDDEN')):
   scaler=a.frozen_scaler();self.assertEqual(scaler.mean.shape,(14,));self.assertGreater(scaler.target_scale,0)
 def test_production_has_no_training_calls(self):self.assertTrue(static_inference_audit()['pass'])
 def test_frozen_code_change_invalidates_authorization(self):
  if not a.PROTOCOL.exists():self.skipTest('Protocol not yet frozen; required post-freeze run repeats this test')
  with patch.object(a,'code_identity',return_value={'files':{},'sha256':'tampered'}):
   with self.assertRaisesRegex(RuntimeError,'CODE_IDENTITY_CHANGED'):a.load_protocol()

class PrefixFirewall(unittest.TestCase):
 def test_suffix_invariance_and_prefix_sensitivity(self):
  prefix=pd.DataFrame({'bin_end_ophrs2_h':np.r_[np.arange(3005)*.5+73,1820.,1820.5],'cell_voltage_mean':np.full(3007,.7)})
  with tempfile.TemporaryDirectory() as folder:
   paths=[Path(folder)/f'synthetic_{letter}.csv' for letter in 'ABC'];variants=[prefix.copy(),prefix.copy(),prefix.copy()];variants[2].loc[50,'cell_voltage_mean']=.8
   fingerprints=[];frames=[]
   for index,(path,frame) in enumerate(zip(paths,variants)):
    suffix=pd.DataFrame({'bin_end_ophrs2_h':[2000.,2300.],'cell_voltage_mean':[index*2.,index*7.]});pd.concat([frame,suffix]).to_csv(path,index=False)
    with patch.object(a.c,'N1',path):dev,access=a.read_dev()
    self.assertEqual(len(dev),3005);self.assertEqual(access[0]['nrows'],3007)
    frames.append(dev);fingerprints.append(hashlib.sha256(dev.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()).hexdigest())
   pd.testing.assert_frame_equal(frames[0],frames[1]);self.assertEqual(fingerprints[0],fingerprints[1]);self.assertNotEqual(fingerprints[0],fingerprints[2])
 def test_unbounded_read_is_rejected_before_io(self):
  def wrong_reader(_):return pd.read_csv(a.c.N1)
  with patch.object(a.c,'load_development',side_effect=wrong_reader):
   with self.assertRaisesRegex(RuntimeError,'UNBOUNDED'):a.read_dev()
 def test_wrong_nrows_3005_rejected(self):
  def wrong_reader(_):return pd.read_csv(a.c.N1,nrows=3005)
  with patch.object(a.c,'load_development',side_effect=wrong_reader):
   with self.assertRaisesRegex(RuntimeError,'UNBOUNDED'):a.read_dev()

class MetricsAndRealForward(unittest.TestCase):
 def test_target_equal_not_prediction_ensemble(self):
  f=pd.DataFrame({'target_row':[1,1,2],'horizon_step':[1,24,24],'y_true_delta_V':[0.,0.,1.],'y_pred_delta_V':[1.,-1.,1.]})
  self.assertAlmostEqual(independent_metrics(f)['target_time_equal_RMSE_mV'],math.sqrt(.5)*1000)
  self.assertNotEqual(independent_metrics(f)['target_time_equal_RMSE_mV'],0.)
 def test_project_all_metrics_equal_independent(self):
  f,_,_=example()
  for key,value in project_metrics(f).items():self.assertLess(abs(value-independent_metrics(f)[key]),1e-10)
 def test_seed_mean_is_not_pooled(self):
  values=np.array([1.,2.,9.]);self.assertNotEqual(values.mean(),np.sqrt(np.mean(values**2)));self.assertAlmostEqual(values.std(ddof=1),math.sqrt(19.))
 def test_all_scientific_branches(self):
  for d20,d21,expected in [([-1]*3,[-1]*3,'F2_DEVELOPMENT_ASSESSMENT_MULTISEED_GAIN_OBSERVED'),([1]*3,[-1]*3,'F2_GATE_GAIN_WITHOUT_NATIVE_SUPERIORITY'),([-1,1,1],[1]*3,'F2_ASSESSMENT_GAIN_SEED_DEPENDENT'),([1]*3,[1]*3,'F2_VALIDATION_GAIN_NOT_TRANSFERRED_TO_ASSESSMENT')]:
   status,_=scientific_decision(pd.DataFrame({'D20_F2_minus_F0_mV':d20,'D21_F2_minus_F1_mV':d21}));self.assertEqual(status,expected)
 def test_real_F0_F1_F2_forward_no_updates(self):
  torch.set_num_threads(1);cfg=a.h.load_protocol();x=np.random.default_rng(2026).normal(size=(2,48,14)).astype(np.float32)
  for group in ['F0','F1','F2']:
   model=a.h.model(group,cfg['hyperparameters'],torch.device('cpu'));before=a.e.state_digest(model);pred,gate=a.predict_frozen(model,x,gate=group=='F2');self.assertEqual(pred.shape,(2,24));self.assertEqual(a.e.state_digest(model),before);self.assertTrue(np.isfinite(pred).all())
   if group=='F2':self.assertEqual(len(gate),2)

if __name__=='__main__':unittest.main()

"""Production firewall/state/statistic regressions, synthetic arrays only."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ir_common as r
from audit_ir_differences import independent_statistics

class ReadOnlyAudit(unittest.TestCase):
 def test_difference_statistics_independent(self):
  reference=r.np.linspace(-1,1,768,dtype=r.np.float32).reshape(32,24);actual=reference.copy();actual[7,10]+=1e-5
  stats=r.difference_statistics(actual,reference,.1368);ind=independent_statistics(actual,reference,.1368)
  for key,value in ind.items():self.assertLessEqual(abs(value-stats[key]),1e-12)
  self.assertEqual(stats['unequal_elements'],1);self.assertEqual(stats['max_difference_sample_index_zero_based'],7);self.assertEqual(stats['max_difference_horizon_step'],11);self.assertFalse(stats['array_equal'])
 def test_tiny_difference_does_not_pass_exact_gate(self):
  reference=r.np.ones((32,24),dtype=r.np.float32);actual=reference.copy();actual[0,0]=r.np.nextafter(r.np.float32(1),r.np.float32(2));self.assertFalse(r.difference_statistics(actual,reference,.1)['array_equal']);self.assertTrue(r.np.allclose(actual,reference))
 def test_identical_arrays_exact_pass(self):
  reference=r.np.ones((32,24),dtype=r.np.float32);stats=r.difference_statistics(reference,reference,.1);self.assertTrue(stats['array_equal']);self.assertEqual(stats['max_abs_difference_mV'],0)
 def test_invalid_shape_fail_closed(self):
  with self.assertRaisesRegex(RuntimeError,'SHAPE'):r.difference_statistics(r.np.zeros((31,24)),r.np.zeros((32,24)),.1)
 def test_nonfinite_is_recorded_not_fabricated(self):
  reference=r.np.ones((32,24));actual=reference.copy();actual[0,0]=r.np.nan;stats=r.difference_statistics(actual,reference,.1);self.assertFalse(stats['finite']);self.assertEqual(stats['actual_nan'],1);self.assertIsNone(stats['max_abs_difference_mV'])
 def test_parameter_state_preserves_original_gradient_flags(self):
  model=r.torch.nn.Linear(2,1);before=r.snapshot(model);flags={k:p.requires_grad for k,p in model.named_parameters()}
  with r.torch.no_grad():model(r.torch.ones(1,2))
  audit=r.state_audit(model,before,flags);self.assertTrue(audit['all_state_entries_equal']);self.assertTrue(audit['requires_grad_unchanged']);self.assertTrue(audit['all_parameter_grad_None']);self.assertTrue(all(flags.values()))
 def test_parameter_mutation_detected(self):
  model=r.torch.nn.Linear(2,1);before=r.snapshot(model);flags={k:p.requires_grad for k,p in model.named_parameters()}
  with r.torch.no_grad():model.weight.add_(1)
  self.assertFalse(r.state_audit(model,before,flags)['all_state_entries_equal'])
 def test_buffer_mutation_detected(self):
  model=r.torch.nn.BatchNorm1d(2);before=r.snapshot(model);flags={k:p.requires_grad for k,p in model.named_parameters()}
  model.running_mean.add_(1);self.assertFalse(r.state_audit(model,before,flags)['all_state_entries_equal'])
 def test_full_N1_hash_forbidden(self):
  with self.assertRaisesRegex(RuntimeError,'FULL_N1'):r.digest(r.c.N1)
 def test_Assessment_cache_read_forbidden(self):
  for path in r.i.F_PRED+[r.I/'outputs/preparation_audit/I_ASSESSMENT_KEY_TRUTH_AUTHORITY.csv']:
   with self.assertRaisesRegex(RuntimeError,'NOT_ALLOWLISTED'):r.allowed(path)
 def test_Test_and_raw_paths_forbidden(self):
  for path in [r.REPO/'数据/Loadcycle.csv',r.PROJECT/'Test/predictions.csv',r.PROJECT/'Polcurve_2000h.csv']:
   with self.assertRaisesRegex(RuntimeError,'NOT_ALLOWLISTED'):r.allowed(path)
 def test_historical_writes_forbidden(self):
  with self.assertRaisesRegex(RuntimeError,'HISTORICAL_WRITE'):r.json_write(r.i.LEDGER,{'state':'changed'})
 def test_unbounded_N1_reader_forbidden(self):
  with patch.object(r.c,'load_development',side_effect=lambda _:r.pd.read_csv(r.c.N1)):
   with self.assertRaisesRegex(RuntimeError,'UNBOUNDED'):r.i.read_dev()
 def test_prefix_suffix_invariance(self):
  frame=r.pd.DataFrame({'bin_end_ophrs2_h':r.np.r_[73+r.np.arange(3005)*.5,1820.,1820.5],'value':r.np.ones(3007)})
  with tempfile.TemporaryDirectory() as folder:
   results=[]
   for number in [1,9]:
    path=Path(folder)/str(number);r.pd.concat([frame,r.pd.DataFrame({'bin_end_ophrs2_h':[2000.],'value':[number]})]).to_csv(path,index=False)
    with patch.object(r.c,'N1',path):dev,access=r.i.read_dev()
    results.append(dev);self.assertEqual(len(dev),3005);self.assertEqual(access[0]['nrows'],3007)
   r.pd.testing.assert_frame_equal(results[0],results[1])
 def test_scaler_no_fit(self):
  with patch.object(r.e.CommonScaler,'fit',side_effect=RuntimeError('SCALER_FIT_FORBIDDEN')):self.assertEqual(r.i.frozen_scaler().mean.shape,(14,))
 def test_no_production_training_path(self):self.assertTrue(r.static_audit()['pass'])
 def test_modified_code_identity_rejected(self):
  if not r.PROTOCOL.exists():self.skipTest('Repeated after freeze in formal finite audit')
  with patch.object(r,'own_code',return_value={'sha256':'bad','files':{}}):
   with self.assertRaisesRegex(RuntimeError,'CODE_IDENTITY'):r.load_protocol()

if __name__=='__main__':unittest.main()

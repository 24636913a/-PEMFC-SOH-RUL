import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,unittest,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch,numpy as np,pandas as pd
from torch import nn
from network import VoltageMLP,slope_loss
P=Path(__file__).resolve().parents[1]
torch.set_default_dtype(torch.float64)
class DerivativeTests(unittest.TestCase):
 def linear(self,w):
  m=nn.Linear(3,1).double()
  with torch.no_grad():m.weight[:]=torch.tensor([[w,.3,-.2]]);m.bias[:]=.5
  return m
 def test_units_chain_rule(self):
  m=self.linear(.7);x=torch.tensor([[0.,1.,2.],[1.,2.,3.]]);loss,d,raw=slope_loss(m,x,.12,.4)
  self.assertAlmostEqual(float(loss),.49);np.testing.assert_allclose(d.detach().numpy(),.21);self.assertAlmostEqual(float(raw),.21**2)
 def test_nontrivial_parameter_gradient(self):
  m=self.linear(.7);loss,_,_=slope_loss(m,torch.zeros(3,3),.12,.4);g=torch.autograd.grad(loss,m.weight)[0];self.assertAlmostEqual(float(g[0,0]),1.4);self.assertEqual(float(g[0,1]),0)
 def test_inactive_loss_true_zero(self):
  m=self.linear(-.4);loss,d,raw=slope_loss(m,torch.zeros(3,3),.12,.4);self.assertEqual(float(loss),0.);self.assertEqual(float(raw),0.);self.assertTrue((d<0).all())
 def test_MLP_slope_finite_difference(self):
  torch.manual_seed(42);m=VoltageMLP().double();x=torch.tensor([[.2,-.7,.4],[-.6,.1,-.2]]);_,d,_=slope_loss(m,x,.03,.4);h=1e-5;lo=x.clone();hi=x.clone();lo[:,0]-=h;hi[:,0]+=h
  numeric=(m(hi)-m(lo))/(2*h)*.03/.4;np.testing.assert_allclose(d.detach().numpy(),numeric.detach().numpy(),rtol=1e-7,atol=1e-10)
 def test_same_capacity_same_initial_state(self):
  torch.manual_seed(52);a=VoltageMLP().double();torch.manual_seed(52);b=VoltageMLP().double();self.assertEqual(sum(p.numel() for p in a.parameters()),1217)
  for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
 def test_platform_no_label_leak(self):
  d=pd.read_csv(P/'data_contracts/A_SAMPLES.csv');tr=d[d.role.eq('train')];v=d[d.role.eq('validation')];self.assertFalse(set(tr.sample_id)&set(v.sample_id));self.assertTrue(set(d.context_source_id)<=set(tr.sample_id));self.assertTrue((v.j_design_A_cm2>v.domain_low).all());self.assertTrue((v.j_design_A_cm2<v.domain_high).all())
 def test_B_origins_past_labels(self):
  d=pd.read_csv(P/'data_contracts/B_SAMPLES.csv')
  if not len(d):return
  for _,g in d.groupby('fold'):
   tr=g[g.role.eq('train')];v=g[g.role.eq('validation')];self.assertLess(pd.to_datetime(tr.end_calendar).max(),pd.to_datetime(v.origin_calendar).min());self.assertTrue((pd.to_datetime(g.origin_calendar)<pd.to_datetime(g.start_calendar)).all())
 def test_train_only_scalers(self):
  d=pd.read_csv(P/'data_contracts/ALL_LEGAL_SAMPLES.csv',float_precision='round_trip');s=json.loads((P/'data_contracts/TRAIN_ONLY_SCALERS.json').read_text())
  for q in s:
   tr=d[d.task.eq(q['task'])&d.fold.eq(q['fold'])&d.role.eq('train')];np.testing.assert_allclose(tr[['j_design_A_cm2','direction_ascending','context']].mean(),q['x_mean'],rtol=0,atol=1e-12);self.assertAlmostEqual(tr.true_V.mean(),q['y_mean']);self.assertEqual(tr.sample_id.tolist(),q['training_sample_ids'])
 def test_physics_probes_unlabeled_train_context(self):
  d=pd.read_csv(P/'data_contracts/ALL_LEGAL_SAMPLES.csv');ph=pd.read_csv(P/'data_contracts/UNLABELED_TRAIN_PHYSICS_PROBES.csv');self.assertTrue(ph.no_voltage_label.all());self.assertNotIn('true_V',ph)
  for (task,fold),g in ph.groupby(['task','fold']):
   tr=d[d.task.eq(task)&d.fold.eq(fold)&d.role.eq('train')];self.assertTrue(set(g.context_source_id)<=set(tr.context_source_id))
 def test_fixed_lambda_budget_no_validation_selection(self):
  c=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());self.assertEqual(c['lambda'],1.);self.assertEqual(c['epochs'],2500);self.assertEqual(c['seeds'],[42,52,62]);self.assertTrue(c['early_stopping'].startswith('NONE'))
if __name__=='__main__':
 import io
 out=io.StringIO();result=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DerivativeTests));(P/'outputs/SYNTHETIC_AND_CONTRACT_TEST_LOG.txt').write_text(out.getvalue(),encoding='utf-8');record={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'test_code_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(P/'outputs/SYNTHETIC_TEST_RESULTS.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(out.getvalue());sys.exit(0 if result.wasSuccessful() else 1)

import unittest,sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'scripts'),str(P/'models')]
from data_alignment import health_at,IDENTITY,H_PROTOCOL,REFERENCE_AVAILABLE,ah
from health_fusion import cohort,MODELS,forward_model,gate_model,state_hash,HealthContextF2

class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.cfg=yaml.safe_load(H_PROTOCOL.read_text(encoding='utf-8'));cls.models=cohort(42,cls.cfg)
  cls.data=dict(np.load(P/'inputs/MATCHED_INPUTS.npz'));cls.p1s=pd.read_csv(IDENTITY,float_precision='round_trip')
  for model in cls.models.values():model.eval()
 def test_01_future_P1S_features_and_predictions(self):
  t=float(self.data['validation_ages'][0]);a,_=health_at(self.p1s,t);changed=self.p1s.copy();changed.loc[changed.conservative_available_end_h>t,'R']=np.random.default_rng(42).normal(size=int((changed.conservative_available_end_h>t).sum()));b,_=health_at(changed,t)
  self.assertTrue(np.array_equal(a,b));sc=json.loads((P/'inputs/HEALTH_TRAIN_SCALER.json').read_text());ha=torch.tensor(((a-sc['mean'])/sc['scale']).astype(np.float32)).view(1,5);hb=torch.tensor(((b-sc['mean'])/sc['scale']).astype(np.float32)).view(1,5)
  model=cohort(62,self.cfg)[MODELS[2]];model.eval()
  with torch.no_grad():
   model.regime_gate.output.weight.fill_(.07);x=torch.from_numpy(self.data['validation_x'][:1]);self.assertTrue(torch.equal(forward_model(model,x,ha),forward_model(model,x,hb)))
 def test_02_future_targets_not_in_input_or_forward(self):
  before={k:ah(self.data[k]) for k in ['validation_x','validation_health']};altered=self.data['validation_y'].copy();altered[:]=999.
  x=torch.from_numpy(self.data['validation_x'][:1]);h=torch.from_numpy(self.data['validation_health'][:1]);model=self.models[MODELS[2]]
  with torch.no_grad():a=forward_model(model,x,h);b=forward_model(model,x,h)
  self.assertTrue(torch.equal(a,b));self.assertEqual(before,{k:ah(self.data[k]) for k in before})
  packet=model.build_input(x);self.assertEqual(tuple(packet['decoder_cont'].shape),(1,24,15));self.assertTrue(torch.equal(packet['decoder_cont'][...,:14],torch.zeros(1,24,14)))
 def test_03_count_insufficient_excluded(self):
  q=self.p1s.iloc[:4];v,a=health_at(q,float(q.conservative_available_end_h.iloc[-1])+1);self.assertIsNone(v);self.assertEqual(a['reason'],'COUNT_BELOW_5')
 def test_04_span_insufficient_excluded(self):
  q=self.p1s.iloc[:5];v,a=health_at(q,float(q.conservative_available_end_h.iloc[-1])+1);self.assertIsNone(v);self.assertEqual(a['reason'],'SPAN_BELOW_67_2_H')
 def test_05_reference_unavailable_excluded(self):
  v,a=health_at(self.p1s,REFERENCE_AVAILABLE-1e-6);self.assertIsNone(v);self.assertEqual(a['reason'],'REFERENCE_NOT_YET_AVAILABLE')
 def test_06_available_end_not_start_or_measured_end(self):
  # A finished measurement whose conservative availability is later is excluded.
  q=self.p1s.copy();r=q.iloc[15];t=float(r.measurement_end_h);_,a=health_at(q,t);self.assertNotIn(int(r.raw_cycle_id),a['raw_cycle_ids'])
 def test_07_no_duplicate_cycle(self):
  q=pd.concat([self.p1s,self.p1s.iloc[[15]]]);r=self.p1s.iloc[20]
  with self.assertRaises(AssertionError):health_at(q,float(r.conservative_available_end_h)+1)
 def test_08_shared_core_and_gate_initialization_all_seeds(self):
  for seed in [42,52,62]:
   c=cohort(seed,self.cfg);a,b,h=[c[m].state_dict() for m in MODELS]
   self.assertEqual(state_hash(c[MODELS[1]]),state_hash(c[MODELS[2]]))
   for k,v in a.items():self.assertTrue(torch.equal(v,b[k][:,:5] if k=='regime_gate.hidden.weight' else b[k]),k)
   counts=[sum(t.numel() for t in c[m].parameters() if t.requires_grad) for m in MODELS];self.assertEqual(counts,[332436,332476,332476])
 def test_09_initial_gate_and_predictions_same(self):
  x=torch.from_numpy(self.data['train_x'][:2]);h=torch.from_numpy(self.data['train_health'][:2]);out=[]
  with torch.no_grad():
   for m in self.models.values():
    self.assertTrue(torch.equal(gate_model(m,x,h),torch.full((2,1),.5)));out.append(forward_model(m,x,h))
  self.assertTrue(torch.equal(out[0],out[1]) and torch.equal(out[1],out[2]))
 def test_10_capacity_ignores_real_health(self):
  model=cohort(52,self.cfg)[MODELS[1]];model.eval()
  x=torch.from_numpy(self.data['train_x'][:2]);h=torch.from_numpy(self.data['train_health'][:2])
  with torch.no_grad():
   model.regime_gate.output.weight.fill_(.1);self.assertTrue(torch.equal(forward_model(model,x,h),forward_model(model,x,h+100)))
 def test_11_health_scaler_train_only(self):
  cfg=json.loads((P/'inputs/HEALTH_TRAIN_SCALER.json').read_text());raw=self.data['train_health_raw'];self.assertEqual(cfg['train_n'],870)
  self.assertTrue(np.array_equal(raw.mean(axis=0),cfg['mean']));self.assertTrue(np.array_equal(raw.std(axis=0),cfg['scale']))
  validation=self.data['validation_health_raw'].copy();validation[:]=1e10
  self.assertTrue(np.array_equal(((raw-cfg['mean'])/cfg['scale']).astype(np.float32),self.data['train_health']))
 def test_12_role_and_label_time_isolation(self):
  a=self.data;self.assertEqual((len(a['train_origins']),len(a['validation_origins'])),(870,178));self.assertFalse(set(a['train_targets'].ravel())&set(a['validation_targets'].ravel()))
  self.assertEqual(len(np.unique(a['validation_targets'])),224);self.assertLess(float(a['train_ages'].max()+12),float(a['validation_ages'].min()))
 def test_13_no_historical_core_override(self):
  self.assertNotIn('fuse_encoder',HealthContextF2.__dict__);self.assertNotIn('encode_history_branch',HealthContextF2.__dict__)
  self.assertNotIn('build_input',HealthContextF2.__dict__)
 def test_14_capacity_columns_have_no_data_gradient(self):
  c=cohort(62,self.cfg);x=torch.from_numpy(self.data['train_x'][:2]);h=torch.from_numpy(self.data['train_health'][:2])
  for group in MODELS[1:]:
   m=c[group];m.eval()
   with torch.no_grad():m.regime_gate.output.weight.fill_(.1)
   forward_model(m,x,h).square().mean().backward();grad=m.regime_gate.hidden.weight.grad[:,5:]
   if group==MODELS[1]:self.assertTrue(torch.equal(grad,torch.zeros_like(grad)))
   else:self.assertGreater(float(grad.abs().sum()),0.)
if __name__=='__main__':unittest.main()

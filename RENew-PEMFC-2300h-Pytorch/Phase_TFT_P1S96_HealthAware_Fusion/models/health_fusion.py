"""Read-only inheritance of F2; only expand historical fusion-gate context."""
import importlib.util,sys
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
OLD_MODELS=ROOT/'Phase_F_TFT_DualPath_RegimeFusion/models'
NAME='_health96_frozen_f_models'
if NAME not in sys.modules:
 spec=importlib.util.spec_from_file_location(NAME,OLD_MODELS/'__init__.py',submodule_search_locations=[str(OLD_MODELS)])
 module=importlib.util.module_from_spec(spec);sys.modules[NAME]=module;spec.loader.exec_module(module)
else:module=sys.modules[NAME]
OriginalF2=module.RegimeAwareDualPathTFT
MODELS=['F2_MATCHED','F2_CAPACITY_CONTROL','F2_P1S96_HEALTH_AWARE']

class ExpandedGate(nn.Module):
 def __init__(self):
  super().__init__();self.hidden=nn.Linear(10,8);self.activation=nn.ELU();self.output=nn.Linear(8,1)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,context):
  if context.ndim!=2 or context.shape[1]!=10:raise RuntimeError('GATE_CONTEXT_NOT_B_BY_10')
  return torch.sigmoid(self.output(self.activation(self.hidden(context))))

class HealthContextF2(OriginalF2):
 def __init__(self,*,use_health,**kwargs):
  super().__init__(**kwargs);self.use_health=bool(use_health);self.regime_gate=ExpandedGate();self._current_health=None
 def regime_weight(self,encoder_cont):
  regime=self.regime_context(encoder_cont)
  if self.use_health:
   health=self._current_health
   if health is None or health.shape!=(len(regime),5):raise RuntimeError('HEALTH_CONTEXT_NOT_B_BY_5')
   if not torch.isfinite(health).all():raise RuntimeError('NONFINITE_HEALTH')
  else:health=torch.zeros_like(regime)
  return self.regime_gate(torch.cat([regime,health],dim=-1))
 def forward(self,x):
  if self._current_health is not None:raise RuntimeError('CONCURRENT_HEALTH_FORWARD_NOT_SUPPORTED')
  self._current_health=x.get('health_context')
  try:return super().forward(x)
  finally:self._current_health=None
 def predict_delta(self,x,health):
  packet=self.build_input(x);packet['health_context']=health
  return self.forward(packet)['prediction'].squeeze(-1)
 def gate_weight(self,x,health):
  encoder=self.build_input(x)['encoder_cont'];regime=self.regime_context(encoder)
  return self.regime_gate(torch.cat([regime,health if self.use_health else torch.zeros_like(regime)],dim=-1))

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def kwargs(cfg):
 p=cfg['hyperparameters'];return dict(hidden_size=p['hidden_size'],hidden_continuous_size=p['hidden_continuous_size'],attention_head_size=4,lstm_layers=1,dropout=p['dropout'])
def cohort(seed,cfg,device='cpu'):
 # Every group gets the exact native/conv/regime parameters of fresh original F2.
 # The new health columns are isolated RNG draws; B/C share them exactly.
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);original=OriginalF2(**kwargs(cfg))
  common={k:v.detach().clone() for k,v in original.state_dict().items()}
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed+100000);prototype=HealthContextF2(use_health=True,**kwargs(cfg));expanded=prototype.state_dict()
  for k,v in common.items():
   if k=='regime_gate.hidden.weight':expanded[k][:,:5].copy_(v)
   else:expanded[k].copy_(v)
  output={MODELS[0]:original.to(device)}
  for group in MODELS[1:]:
   model=HealthContextF2(use_health=group==MODELS[2],**kwargs(cfg));model.load_state_dict(expanded,strict=True);output[group]=model.to(device)
 return output
def forward_model(model,x,health):
 return model.predict_delta(x,health) if isinstance(model,HealthContextF2) else model.predict_delta(x)
def gate_model(model,x,health):
 return model.gate_weight(x,health) if isinstance(model,HealthContextF2) else model.regime_weight(model.build_input(x)['encoder_cont'])
def state_hash(model):
 import hashlib
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

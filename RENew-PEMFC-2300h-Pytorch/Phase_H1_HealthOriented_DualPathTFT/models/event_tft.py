"""History-only event query through the unmodified native/F1/F2 TFT forward body."""
from pathlib import Path
import sys,importlib.util
import torch
P=Path(__file__).resolve().parents[1];ROOT=P.parent
# Load old extensions under a unique package name; never import old training scripts.
package=ROOT/'Phase_F_TFT_DualPath_RegimeFusion/models'
spec=importlib.util.spec_from_file_location('frozen_dualpath_extensions',package/'__init__.py',submodule_search_locations=[str(package)])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from canonical_tft import CanonicalStrictHistoryTFT

class EventReadout:
 @staticmethod
 def event_input(history):
  if history.ndim!=3 or history.shape[1:]!=(48,14):raise ValueError('Expected history[B,48,14]')
  batch=history.shape[0];device=history.device;dtype=history.dtype
  relative=torch.arange(-47,1,device=device,dtype=dtype).view(1,48,1).expand(batch,-1,-1)/24.
  return dict(encoder_cont=torch.cat([history,relative],dim=-1),
   decoder_cont=torch.zeros((batch,1,15),device=device,dtype=dtype),
   encoder_cat=torch.empty((batch,48,0),device=device,dtype=torch.long),decoder_cat=torch.empty((batch,1,0),device=device,dtype=torch.long),
   encoder_lengths=torch.full((batch,),48,device=device,dtype=torch.long),decoder_lengths=torch.ones(batch,device=device,dtype=torch.long),
   target_scale=torch.zeros((batch,2),device=device,dtype=dtype))
 def event_forward(self,history):
  # One constant event query, NOT a half-hour lead or actual future timestamp.
  # Native forward retains VSN, LSTM, enrichment, attention, FF/GRN, scalar head.
  out=self.forward(self.event_input(history));assert out['prediction'].shape==(len(history),1,1);return out
 def predict_voltage(self,history):return self.event_forward(history)['prediction'].squeeze(-1)
 def predict_delta(self,*args,**kwargs):raise RuntimeError('This event model predicts standardized absoluteV, not dynamic DeltaV')

class EventF0(EventReadout,CanonicalStrictHistoryTFT):pass
class EventF1(EventReadout,module.DualPathStrictHistoryTFT):pass
class EventF2(EventReadout,module.RegimeAwareDualPathTFT):pass

def create_model(group,params,device):
 cls={'F0':EventF0,'F1':EventF1,'F2':EventF2}[group]
 return cls(hidden_size=params['hidden_size'],hidden_continuous_size=params['hidden_continuous_size'],attention_head_size=4,lstm_layers=1,dropout=params['dropout']).to(device)

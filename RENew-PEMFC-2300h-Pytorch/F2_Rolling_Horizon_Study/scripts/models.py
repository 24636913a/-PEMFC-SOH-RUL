"""Only native F2, independent plain LSTM and the predeclared C1 output head."""
import math
import random
import numpy as np
import torch
from torch import nn
import native_models as native

class OutputResidualF2(native.RegimeAwareDualPathTFT):
    """Native output + bounded historical fused-state shared residual head.

    The encoder fusion formula is NOT changed. Stored fused states remain in
    the autograd graph. No detached/cached future observation enters the head.
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.residual_head=nn.Sequential(nn.Linear(65,16),nn.ELU(),nn.Linear(16,1))
            nn.init.zeros_(self.residual_head[-1].weight);nn.init.zeros_(self.residual_head[-1].bias)
        self.residual_logit=nn.Parameter(torch.tensor(math.log(.1/.9),dtype=torch.float32))
        self.disable_residual=False
        self._fused_history=None
        self.last_native=None;self.last_correction=None
    @property
    def residual_alpha(self):return .25*torch.sigmoid(self.residual_logit)
    def fuse_encoder(self,encoder_lstm,encoder_conv,encoder_cont):
        fused=super().fuse_encoder(encoder_lstm,encoder_conv,encoder_cont)
        self._fused_history=fused
        return fused
    def predict_delta(self,x):
        self._fused_history=None
        base=super().predict_delta(x)
        assert self._fused_history is not None and self._fused_history.shape[1:]==(48,64)
        context=self._fused_history.mean(dim=1)
        horizon=base.shape[1]
        h=torch.arange(1,horizon+1,device=x.device,dtype=x.dtype).view(1,horizon,1)/horizon
        pooled=context[:,None,:].expand(-1,horizon,-1)
        r=torch.tanh(self.residual_head(torch.cat([pooled,h.expand(len(x),-1,-1)],dim=-1))).squeeze(-1)
        correction=self.residual_alpha*r
        self.last_native=base.detach();self.last_correction=correction.detach()
        return base if self.disable_residual else base+correction

def seed_all(seed=202):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def new_model(kind,horizon,seed=202):
    assert seed==202 and horizon in [6,12,24]
    seed_all(seed);native.HORIZON=horizon
    if kind=='LSTM':model=native.PlainLSTM(horizon)
    else:
        cls=OutputResidualF2 if kind=='C1' else native.RegimeAwareDualPathTFT
        model=cls(hidden_size=64,hidden_continuous_size=32,attention_head_size=4,lstm_layers=1,dropout=.1)
    expected=333510 if kind=='C1' else (20480+65*horizon if kind=='LSTM' else 332436)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==expected
    return model

def configure():
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(False)
    return dict(torch=str(torch.__version__),CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),strict_deterministic_algorithms=False,bitwise_repeatability_promised=False)

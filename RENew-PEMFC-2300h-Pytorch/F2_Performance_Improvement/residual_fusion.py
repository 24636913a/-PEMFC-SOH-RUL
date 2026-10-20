"""One Stage-B hypothesis, registered after Stage A and before any B update.

No extra parameters, no changed backbone or future input. The norm is over
the already available 48-step historical latent sequence of each example.
"""
import math
import torch
import frozen_f2 as f

CAP=.25
EPS=1e-6


def latent_rms(value):
    # vector_norm has a defined zero gradient at a zero vector. No detach:
    # normalization and both paths participate in the actual training graph.
    return torch.linalg.vector_norm(value,ord=2,dim=(1,2),keepdim=True)/math.sqrt(value.shape[1]*value.shape[2])


class ResidualFusionF2(f.RegimeAwareDualPathTFT):
    """Only the historical encoder fusion is overridden; all TFT stays native."""
    fusion_kind='residual'

    def fuse_encoder(self,encoder_lstm,encoder_conv,encoder_cont):
        gate=self.regime_weight(encoder_cont).view(-1,1,1)
        scale=latent_rms(encoder_lstm)/latent_rms(encoder_conv).clamp_min(EPS)
        return encoder_lstm+CAP*(1-gate)*scale*encoder_conv


def build_residual(config):
    f.require(config['fusion']=='residual' and config['residual_cap']==CAP and config['normalization_epsilon']==EPS,
              'REGISTERED_RESIDUAL_FORMULA_ONLY')
    model=ResidualFusionF2(hidden_size=64,hidden_continuous_size=32,
        attention_head_size=4,lstm_layers=1,dropout=config['dropout'])
    f.require(sum(p.numel() for p in model.parameters())==332436,'RESIDUAL_PARAMETER_COUNT')
    return model

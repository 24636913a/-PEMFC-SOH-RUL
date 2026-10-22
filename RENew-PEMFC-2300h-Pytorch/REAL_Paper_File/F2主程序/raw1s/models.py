"""New initialization only. All native TFT/CNN/gate definitions are local."""
import torch
from torch import nn
from .native_f2 import RegimeAwareDualPathTFT

MODEL_PARAMETERS = dict(hidden_size=64, hidden_continuous_size=32,
                        attention_head_size=4, lstm_layers=1, dropout=.1)


class F2(nn.Module):
    def __init__(self, lookback=300, horizon=120):
        super().__init__()
        self.lookback, self.horizon = lookback, horizon
        self.native = RegimeAwareDualPathTFT(**MODEL_PARAMETERS)
        self.native.hparams.max_encoder_length = lookback

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Lightning children cache .device/.dtype outside tensor state.
        self.native.to(*args, **kwargs)
        return self

    def forward(self, x):
        b, l, f = x.shape
        if (l, f) != (self.lookback, 24):
            raise ValueError('Expected 300 history rows and 24 masked slots')
        h = self.horizon
        rh = torch.arange(1-l, 1, device=x.device, dtype=x.dtype)[None, :, None].expand(b, -1, -1) / h
        rf = torch.arange(1, h+1, device=x.device, dtype=x.dtype)[None, :, None].expand(b, -1, -1) / h
        inputs = dict(encoder_cont=torch.cat((x, rh), -1),
                      decoder_cont=torch.cat((x.new_zeros((b, h, f)), rf), -1),
                      encoder_cat=torch.empty((b, l, 0), device=x.device, dtype=torch.long),
                      decoder_cat=torch.empty((b, h, 0), device=x.device, dtype=torch.long),
                      encoder_lengths=torch.full((b,), l, device=x.device, dtype=torch.long),
                      decoder_lengths=torch.full((b,), h, device=x.device, dtype=torch.long),
                      target_scale=x.new_zeros((b, 2)))
        result = self.native(inputs)['prediction'].squeeze(-1)
        if result.shape != (b, h):
            raise RuntimeError('F2 output shape mismatch')
        return result


class LSTM(nn.Module):
    def __init__(self, lookback=300, horizon=120):
        super().__init__()
        self.lstm = nn.LSTM(24, 64, num_layers=1, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, horizon))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


def create_model(cfg):
    return {'F2': F2, 'LSTM': LSTM}[cfg['model']](cfg['lookback'], cfg['horizon'])

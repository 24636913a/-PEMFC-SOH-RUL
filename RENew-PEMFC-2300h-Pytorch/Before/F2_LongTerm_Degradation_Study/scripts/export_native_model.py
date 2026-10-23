"""Reproduce the model export to a NEW file; never overwrite source/models."""
import ast
import argparse
from pathlib import Path

def export(output):
    here=Path(__file__).resolve().parents[1]
    source=here.parent/'F2_Performance_Improvement/frozen_f2.py'
    text=source.read_text(encoding='utf-8');tree=ast.parse(text)
    names=['CanonicalStrictHistoryTFT','_LeftPaddedDepthwiseConv1d','CausalMultiScaleHistoryBranch',
        'DualPathStrictHistoryTFT','RegimeFusionGate','RegimeAwareDualPathTFT']
    parts=['''"""Native F2; forecast length interface only; no old entry import."""
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
HORIZON=48
''']
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id in ['FEATURES','RAW_NAMES','RELATIVE_STEP','X_REALS','RAW_F2_FEATURE_NAMES','REGIME_FEATURE_NAMES'] for n in node.targets):
            parts.append(ast.get_source_segment(text,node))
        if isinstance(node,ast.ClassDef) and node.name in names:
            s=ast.get_source_segment(text,node)
            if node.name=='CanonicalStrictHistoryTFT':
                for a,b in [('/ 24.0','/ float(HORIZON)'),('torch.arange(1, 25,','torch.arange(1, HORIZON+1,'),
                    ('view(1, 24, 1)','view(1, HORIZON, 1)'),('(batch, 24, 14)','(batch, HORIZON, 14)'),
                    ('(batch, 24, 0)','(batch, HORIZON, 0)'),('torch.full((batch,), 24,','torch.full((batch,), HORIZON,'),
                    ('(x_history_scaled.shape[0], 24, 1)','(x_history_scaled.shape[0], HORIZON, 1)')]:s=s.replace(a,b)
            parts.append(s)
    # This independent ordinary LSTM adds only the direct horizon-sized head.
    parts.append('''class PlainLSTM(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.lstm=nn.LSTM(14,64,num_layers=1,batch_first=True)
        self.head=nn.Linear(64,horizon)
    def predict_delta(self,x):
        _,(hidden,_)=self.lstm(x)
        return self.head(hidden[-1])
''')
    output=Path(output)
    if output.exists():raise FileExistsError('Export only to a new path')
    output.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);export(p.parse_args().out)

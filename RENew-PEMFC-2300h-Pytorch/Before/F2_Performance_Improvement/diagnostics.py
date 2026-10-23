"""Read-only observers of the native fusion; never substitutes a training graph."""
import math
import types
import numpy as np
import torch
import frozen_f2 as f


def new_model(seed, config):
    f.seed_everything(seed)
    if config.get('fusion','native')=='native':
        model=f.RegimeAwareDualPathTFT(hidden_size=64,hidden_continuous_size=32,
            attention_head_size=4,lstm_layers=1,dropout=config['dropout'])
        f.require(sum(p.numel() for p in model.parameters())==332436,'NATIVE_PARAMETER_COUNT')
    else:
        # This module may only be registered after Stage A acceptance.
        from residual_fusion import build_residual
        model=build_residual(config)
    return attach(model.cuda())


def rms(value):
    return value.square().mean(dim=(1,2)).sqrt()


def attach(model):
    """Observer delegates NORMAL forward exactly to the native bound method."""
    model._fusion_mode='normal';model._probe={}
    native=model.fuse_encoder
    def gate_hook(module, args, output):
        model._last_gate=output
    model.regime_gate.register_forward_hook(gate_hook)
    def observed(self, hl, hc, history):
        normal=native(hl,hc,history)
        gate=self._last_gate
        if self._fusion_mode!='normal':
            f.require(not self.training,'NO_COUNTERFACTUAL_TRAINING')
            off=hl if getattr(self,'fusion_kind','native')=='residual' else gate.view(-1,1,1)*hl
            output={'lstm_only':hl,'cnn_off':off}[self._fusion_mode]
        else:output=normal
        self._probe={'lstm':hl,'cnn':hc,'gate':gate,'fused':output}
        if torch.is_grad_enabled():
            for tensor in self._probe.values():
                if tensor.requires_grad:tensor.retain_grad()
        return output
    model.fuse_encoder=types.MethodType(observed,model)
    return model


def path_rows(model):
    p=model._probe;hl,hc,g,hs=p['lstm'],p['cnn'],p['gate'],p['fused']
    a,b,s=rms(hl),rms(hc),rms(hs)
    cosine=torch.nn.functional.cosine_similarity(hl.flatten(1),hc.flatten(1),dim=1)
    def array(x):return x.detach().cpu().numpy().astype(float)
    residual=getattr(model,'fusion_kind','native')=='residual'
    return dict(g=array(g.flatten()),lstm_rms=array(a),cnn_rms=array(b),fused_rms=array(s),
        path_cosine=array(cosine),weighted_lstm_rms=array(a if residual else a*g.flatten()),
        weighted_cnn_rms=array(rms(hs-hl) if residual else b*(1-g.flatten())),
        fusion_to_lstm_rms=array(s/(a+1e-12)),change_from_lstm_rms=array(rms(hs-hl)),
        cnn_to_lstm_contribution_ratio=array((rms(hs-hl) if residual else b*(1-g.flatten()))/(a+1e-12)))


def gradient_norm(parameters):
    values=[p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else None


def gradient_row(model):
    result={}
    for group,prefix in [('lstm_encoder','lstm_encoder.'),('cnn','history_branch.'),('gate','regime_gate.')]:
        parameters=[p for name,p in model.named_parameters() if name.startswith(prefix)]
        result[group+'_parameter_grad_norm']=gradient_norm(parameters)
        result[group+'_parameters_with_grad']=sum(p.grad is not None for p in parameters)
    for key,tensor in model._probe.items():
        result[key+'_output_grad_norm']=None if tensor.grad is None else float(tensor.grad.detach().norm().cpu())
    return result


def clear(model):
    model._probe={};model._last_gate=None


def predict(model,x, observe=False):
    model.eval();values=[];rows=[]
    with torch.no_grad():
        for start in range(0,len(x),32):
            values.append(model.predict_delta(torch.from_numpy(x[start:start+32]).cuda()).cpu().numpy())
            if observe:
                p=path_rows(model)
                rows.extend({key:float(v[index]) for key,v in p.items()} for index in range(len(p['g'])))
            clear(model)
    prediction=np.concatenate(values)
    f.require(prediction.shape==(len(x),24) and np.isfinite(prediction).all(),'PREDICTIONS')
    return prediction,rows


def summarize_paths(rows):
    values=np.asarray([row['g'] for row in rows],float)
    result=dict(g_mean=float(values.mean()),g_median=float(np.median(values)),g_std=float(values.std()),
        g_min=float(values.min()),g_max=float(values.max()),g_P05=float(np.quantile(values,.05)),
        g_P95=float(np.quantile(values,.95)),near_half_fraction=float(np.mean(abs(values-.5)<.01)),
        saturated_fraction=float(np.mean((values<.01)|(values>.99))))
    for key in rows[0]:
        if key!='g':result[key+'_mean']=math.fsum(row[key] for row in rows)/len(rows)
    return result

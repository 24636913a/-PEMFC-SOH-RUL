"""Synthetic probes in independent processes; no real dataset access."""
from __future__ import annotations
import argparse
import os
import sys
# Strict worker only; setup before imports of torch or helpers.
if '--strict' in sys.argv:os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import traceback
import numpy as np
import torch
from h_common import *
from crash_safe_recorder import CrashSafeRecorder

def strict_probe(device_name,kind):
    cfg=load_protocol();folder=OUT/'compatibility'/f'{device_name}_strict_{kind}'
    result={'synthetic_only':True,'device':device_name,'kind':kind,'forward_pass':False,'backward_pass':False,
        'optimizer_step_performed':False,'exception':None,'traceback':None}
    recorder=CrashSafeRecorder(folder,'strict_'+device_name+'_'+kind)
    try:
        torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=False);e.set_seed(42)
        device=torch.device(device_name)
        if kind=='interpolation':
            x=torch.randn(2,1,32,device=device,requires_grad=True);y=torch.randn(2,1,64,device=device);m=None
        else:
            m=model('F0',cfg['hyperparameters'],device);x=torch.randn(2,48,14,device=device);y=torch.randn(2,24,device=device)
        recorder.emit('BATCH_STARTED',epoch=1,batch=0,input_sha256=array_hash(x.detach().cpu().numpy()),label_sha256=array_hash(y.cpu().numpy()),
            model_state_sha256=tree_hash(m.state_dict()) if m is not None else 'OPERATOR_ONLY',rng=rng_summary())
        pred=torch.nn.functional.interpolate(x,size=64,mode='linear',align_corners=True) if m is None else m.predict_delta(x)
        recorder.emit('FORWARD_COMPLETED',output_sha256=array_hash(pred.detach().cpu().numpy()),output_shape=list(pred.shape))
        loss=torch.nn.functional.mse_loss(pred,y);recorder.emit('LOSS_COMPUTED',loss=float(loss.detach().cpu()))
        result.update(forward_pass=True,input_shape=list(x.shape),output_shape=list(pred.shape),loss=float(loss.detach().cpu()),
            input_sha256=array_hash(x.detach().cpu().numpy()),output_sha256=array_hash(pred.detach().cpu().numpy()))
        recorder.emit('BACKWARD_STARTED')
        try:loss.backward()
        except Exception as exc:recorder.emit('BACKWARD_FAILED',exception=repr(exc),traceback=traceback.format_exc());raise
        recorder.emit('BACKWARD_COMPLETED');result['backward_pass']=True;result['status']='PASS'
    except Exception as exc:
        result.update(exception=repr(exc),traceback=traceback.format_exc(),status=
            'CUDA_STRICT_DETERMINISM_UNSUPPORTED' if device_name=='cuda' and 'upsample_linear1d_backward_out_cuda' in str(exc) else 'PROBE_FAILED')
    result['environment']=environment();atomic_json(folder/'result.json',result)
    print(result['status'],flush=True)

def smoke():
    cfg=load_protocol();configure_normal(cfg);device=torch.device('cuda');results={}
    for group in ['F0','F1','F2']:
        e.set_seed(42);m=model(group,cfg['hyperparameters'],device);x=torch.randn(2,48,14,device=device);y=torch.randn(2,24,device=device)
        before=tree_hash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=cfg['hyperparameters']['learning_rate'],weight_decay=cfg['hyperparameters']['weight_decay'])
        opt.zero_grad(set_to_none=True);p=m.predict_delta(x);loss=torch.nn.functional.mse_loss(p,y)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm) or any(not torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None):raise RuntimeError('NORMAL_SMOKE_NONFINITE')
        opt.step()
        if before==tree_hash(m.state_dict()) or p.shape!=(2,24):raise RuntimeError('NORMAL_SMOKE_NO_UPDATE_OR_SHAPE')
        path=OUT/'compatibility'/f'{group}_SYNTHETIC_SMOKE_CHECKPOINT.pt';path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':m.state_dict(),'scope':'SYNTHETIC_ONLY_NOT_FORMAL_MODEL'},path)
        m.eval()
        with torch.no_grad():expected=m.predict_delta(x).cpu().numpy()
        reloaded=model(group,cfg['hyperparameters'],device);reloaded.load_state_dict(torch.load(path,map_location=device,weights_only=True)['state_dict']);reloaded.eval()
        with torch.no_grad():actual=reloaded.predict_delta(x).cpu().numpy()
        if not np.array_equal(expected,actual):raise RuntimeError('SMOKE_RELOAD_PARITY_FAILURE')
        results[group]={'pass':True,'loss':float(loss.detach().cpu()),'gradient_norm':float(norm.detach().cpu()),
            'parameter_update':True,'checkpoint_reload_max_difference':float(np.max(abs(expected-actual))),
            'checkpoint_sha256':file_hash(path)}
    atomic_json(OUT/'compatibility/NORMAL_CUDA_SMOKE.json',{'pass':True,'groups':results,'environment':environment(),'synthetic_only':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--strict',action='store_true');p.add_argument('--device',choices=['cpu','cuda']);p.add_argument('--kind',choices=['interpolation','tft']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.strict:strict_probe(a.device,a.kind)
    elif a.smoke:smoke()
    else:p.error('Choose strict or smoke')

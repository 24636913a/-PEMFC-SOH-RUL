"""Reusable T1-R fit primitives. No function here is called by static audit."""
from __future__ import annotations
import math, time
from typing import Any
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

class TrainingContractError(RuntimeError): pass
def make_loader(x:np.ndarray,y:np.ndarray,batch_size:int,seed:int)->DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0,drop_last=False)
def fit(model:torch.nn.Module,x:np.ndarray,y:np.ndarray,*,params:dict[str,Any],training:dict[str,Any],device:torch.device,epochs:int|None=None,stop_x:np.ndarray|None=None,stop_y:np.ndarray|None=None,predict=None)->dict[str,Any]:
    required={'learning_rate','weight_decay','batch_size'}
    if required-set(params): raise TrainingContractError('TRIAL_PARAMETER_MISSING')
    loader=make_loader(x,y,int(params['batch_size']),int(training['seed']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(params['learning_rate']),weight_decay=float(params['weight_decay']))
    budget=int(training['max_epochs'] if epochs is None else epochs); best=math.inf; best_epoch=0; stale=0; state=None; logs=[]; started=time.perf_counter()
    for epoch in range(1,budget+1):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(model.predict_delta(xb),yb)
            if not torch.isfinite(loss): raise TrainingContractError('NONFINITE_LOSS')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(training['gradient_clip_norm'])); optimizer.step(); losses.append(float(loss.detach().cpu()))
        value=float(np.sqrt(np.mean(losses))) if stop_x is None else float(np.sqrt(np.mean((predict(model,stop_x,int(params['batch_size']),device)-stop_y)**2)))
        logs.append({'epoch':epoch,'train_standardized_MSE':float(np.mean(losses)),'selection_standardized_RMSE':value})
        if value<best-1e-12: best,best_epoch,stale=value,epoch,0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if stop_x is not None else None
        else: stale+=1
        if stop_x is not None and stale>=int(training['patience']): break
    if state is not None: model.load_state_dict(state)
    return {'best_epoch':best_epoch if stop_x is not None else budget,'best_value':best,'logs':logs,'training_time_s':time.perf_counter()-started,'optimizer_lr':optimizer.param_groups[0]['lr'],'optimizer_weight_decay':optimizer.param_groups[0]['weight_decay']}

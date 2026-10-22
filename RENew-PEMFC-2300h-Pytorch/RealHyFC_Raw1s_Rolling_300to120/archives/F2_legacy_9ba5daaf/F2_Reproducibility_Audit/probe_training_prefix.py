"""Diagnostic prefixes only: fixed 2 x 6 epochs, no holdout scoring or model selection.

Loads definitions from the frozen program, not its executable training cells.
Each invocation has a fresh process/model/optimizer; no historical weights loaded.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import types

import numpy as np
import pandas as pd
import torch

AUDIT = Path(__file__).resolve().parent
FOLDER = AUDIT.parent
SOURCE = FOLDER / 'F2_Train_Interactive.py'
SOURCE_SHA = '3dbdc602eb2229a54ee7d5eaa841a6dd94798e8227dbfc6f60db1e78f66419de'


def digest(data): return hashlib.sha256(data).hexdigest()
def tensor_sha(value): return digest(value.detach().cpu().contiguous().numpy().tobytes())
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def definitions():
    assert digest(SOURCE.read_bytes()) == SOURCE_SHA
    module = types.ModuleType('frozen_f2_diagnostic'); module.__file__ = str(SOURCE)
    sys.modules[module.__name__] = module
    text = SOURCE.read_text(encoding='utf-8').split('# %% 4. 每次Run All')[0]
    exec(compile(text, str(SOURCE), 'exec'), module.__dict__)
    return module


def rng():
    return dict(cpu=tensor_sha(torch.get_rng_state()), cuda=tensor_sha(torch.cuda.get_rng_state()),
                python=digest(repr(random.getstate()).encode()), numpy=digest(repr(np.random.get_state()).encode()))


def gradients(model):
    result = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.grad is not None:
            result.update(name.encode()); result.update(p.grad.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', choices=['original_a', 'original_b', 'strict_check'], required=True)
    args = parser.parse_args()
    output = AUDIT / 'outputs' / args.probe
    output.mkdir(parents=True, exist_ok=False)
    m = definitions(); env = m.configure_environment()
    env.update(cudnn=torch.backends.cudnn.version(), platform=platform.platform(),
        executable=sys.executable, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        variables={k: os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','CUDA_LAUNCH_BLOCKING','PYTHONHASHSEED']},
        packages={p:importlib.metadata.version(p) for p in ['torch','pytorch-forecasting','numpy','pandas','scikit-learn','lightning']},
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip())
    write(output / 'ENVIRONMENT.json', env)
    # Only Train and Validation records are parsed; no holdout values or predictions.
    frame = pd.read_csv(m.SOURCE, nrows=3074)
    train, _ = m.build_windows(frame, 0, 2690)
    validation, _ = m.build_windows(frame, 2690, 3074)
    scaler = m.TrainScaler().fit(train.x_raw)
    x = m.masked_inputs(train.x_raw, scaler, 'top10')
    xv = m.masked_inputs(validation.x_raw, scaler, 'top10')
    y = (train.y / scaler.target_scale).astype('float32')
    yv = (validation.y / scaler.target_scale).astype('float32')
    model = m.initialize_model(202)
    optimizer = torch.optim.AdamW(model.parameters(), lr=m.HYPERPARAMETERS['learning_rate'],
                                weight_decay=m.HYPERPARAMETERS['weight_decay'])
    generator = torch.Generator().manual_seed(202)
    loader = m.DataLoader(m.TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.arange(len(x))),
                        batch_size=32, shuffle=True, generator=generator, num_workers=0, drop_last=False)
    write(output / 'INITIALIZATION.json', dict(seed=202, feature_mode='top10',
        initialization_sha256=m.state_sha(model), optimizer_state_empty=not optimizer.state,
        original_hyperparameters=m.HYPERPARAMETERS, original_max_epochs=450, original_patience=40,
        diagnostic_prefix_epochs=6 if args.probe!='strict_check' else 0,
        train_origins_sha256=m.array_sha(train.origins), validation_origins_sha256=m.array_sha(validation.origins),
        prepared_train_sha256=m.array_sha(x), prepared_validation_sha256=m.array_sha(xv),
        source_program_sha256=SOURCE_SHA, no_historical_weights=True, no_holdout_scoring=True, rng=rng()))
    if args.probe == 'strict_check':
        torch.use_deterministic_algorithms(True)
        phase = 'forward'
        try:
            xb, yb, _ = next(iter(loader)); model.train(); optimizer.zero_grad(set_to_none=True)
            loss = m.F.mse_loss(model.predict_delta(xb.cuda()), yb.cuda())
            phase = 'backward'; loss.backward()
            phase = 'gradient_clip'; torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            phase = 'optimizer_step'; optimizer.step()
            result=dict(status='STRICT_SINGLE_STEP_PASSED', optimizer_updates=1)
        except RuntimeError as error:
            result=dict(status='STRICT_DIAGNOSTIC_REJECTED_OPERATION', phase=phase, error=str(error),
                        optimizer_updates=0, traceback=traceback.format_exc())
        write(output / 'STRICT_RESULT.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    dropout_calls = []
    def dropout_hook(name):
        def hook(layer, inputs, outputs):
            if not layer.training: return
            eligible = inputs[0].detach().ne(0)
            # Natural exact-zero inputs are excluded; record their eligibility separately.
            kept = outputs.detach().ne(0) & eligible
            dropout_calls.append(dict(module=name, p=layer.p, eligible_sha=tensor_sha(eligible),
                                      kept_sha=tensor_sha(kept), output_sha=tensor_sha(outputs)))
        return hook
    handles = [layer.register_forward_hook(dropout_hook(name)) for name, layer in model.named_modules()
               if isinstance(layer, torch.nn.Dropout)]
    history=[]; best=float('inf'); best_epoch=0; stale=0; step=0
    started=time.monotonic()
    with (output / 'BATCH_TRACE.jsonl').open('w',encoding='utf-8',newline='\n') as trace:
        for epoch in range(1,7):
            model.train(); losses=[]; norms=[]; order=[]
            for batch, (xb,yb,indices) in enumerate(loader,1):
                step+=1; xb,yb=xb.cuda(),yb.cuda(); dropout_calls.clear()
                before=rng(); weights_before=m.state_sha(model)
                optimizer.zero_grad(set_to_none=True)
                pred=model.predict_delta(xb)
                after_forward=rng()
                loss=m.F.mse_loss(pred,yb); loss.backward()
                grad_before=gradients(model)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                grad_after=gradients(model)
                optimizer.step()
                record=dict(epoch=epoch,batch=batch,step=step,indices=indices.tolist(),
                    input_sha=tensor_sha(xb),target_sha=tensor_sha(yb),weights_before=weights_before,
                    rng_before=before,rng_after_forward=after_forward,rng_after_step=rng(),
                    output_sha=tensor_sha(pred),loss=float(loss.detach()),gradient_norm=float(norm),
                    gradients_before_clip=grad_before,gradients_after_clip=grad_after,
                    weights_after=m.state_sha(model),dropouts=list(dropout_calls))
                trace.write(json.dumps(record,separators=(',',':'))+'\n');trace.flush()
                losses.append(float(loss.detach()));norms.append(float(norm));order.extend(indices.tolist())
            vp=m.predict_standardized(model,xv,32)
            val_mse=float(np.mean((vp.astype('float64')-yv.astype('float64'))**2))
            if val_mse**.5<best-1e-12:best,best_epoch,stale=val_mse**.5,epoch,0
            else:stale+=1
            row=dict(epoch=epoch,train_standardized_MSE=float(np.mean(losses)),validation_standardized_MSE=val_mse,
                     best_epoch=best_epoch,stale_epochs=stale,gradient_norm_min=min(norms),gradient_norm_max=max(norms),
                     sampler_order_sha256=m.array_sha(np.asarray(order,np.int64)),weights_sha256=m.state_sha(model),rng=rng())
            history.append(row)
            write(output/'EPOCH_TRACE.json',history)
            print(args.probe,'epoch',epoch,'train',row['train_standardized_MSE'],'val',val_mse,flush=True)
    for handle in handles:handle.remove()
    # Diagnostic final snapshot is separate and never used as a formal best model.
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    diagnostic_only=True,epoch=6,seed=202,feature_mode='top10'),output/'diagnostic_epoch6.pt')
    write(output/'COMPLETE.json',dict(status='DIAGNOSTIC_PREFIX_COMPLETED',epochs=6,optimizer_updates=step,
        formal_performance_result=False,holdout_scored=False,seconds=time.monotonic()-started))


if __name__=='__main__':main()

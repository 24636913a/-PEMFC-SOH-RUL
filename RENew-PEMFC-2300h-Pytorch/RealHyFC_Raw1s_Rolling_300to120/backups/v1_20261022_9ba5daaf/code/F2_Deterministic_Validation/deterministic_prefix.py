"""Fresh-process GPU determinism diagnostic; never executes formal training cells.

Only runtime determinism flags change. Model, masking, data, optimizer and loss
are loaded from the byte-verified formal source. No historical weights loaded.
"""
import argparse
import os
from pathlib import Path
import sys

if 'torch' in sys.modules:
    raise RuntimeError('Run this entry in a fresh Python process, not an initialized interactive kernel.')
parser=argparse.ArgumentParser()
parser.add_argument('--task',choices=['strict_a','strict_b','benchmark_baseline','benchmark_strict'],required=True)
parser.add_argument('--run-dir',required=True)
args=parser.parse_args()
STRICT=args.task!='benchmark_baseline'
INHERITED_WORKSPACE=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
# BEFORE importing PyTorch or initializing CUDA; this affects this child only.
if STRICT:os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
else:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')

from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
import traceback
import types
import warnings

import numpy as np
import pandas as pd
import torch

CUDA_UNINITIALIZED_AFTER_TORCH_IMPORT=not torch.cuda.is_initialized()
torch.set_num_threads(1)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
torch.use_deterministic_algorithms(STRICT,warn_only=False)
warnings.filterwarnings('ignore',message="Attribute '(loss|logging_metrics)' is an instance of")

HERE=Path(__file__).resolve().parent
FOLDER=HERE.parent
SOURCE=FOLDER/'F2_Train_Interactive.py'
SOURCE_SHA='3dbdc602eb2229a54ee7d5eaa841a6dd94798e8227dbfc6f60db1e78f66419de'
RUN=Path(args.run_dir).resolve()
assert RUN.parent==HERE/'outputs'
OUT=RUN/args.task
OUT.mkdir(exist_ok=False)
PHASE='initialization'
EPOCH=BATCH=UPDATES=0


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def rng():
    return dict(cpu=tensor_sha(torch.get_rng_state()),cuda=tensor_sha(torch.cuda.get_rng_state()),
        python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest())


def parameter_data_sha(model,gradient=False):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        value=p.grad if gradient else p
        if value is None:continue
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_sha(model,optimizer):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        for key,value in sorted(optimizer.state.get(p,{}).items()):
            h.update((name+'/'+key).encode())
            if torch.is_tensor(value):
                h.update(str((str(value.dtype),tuple(value.shape))).encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:h.update(repr(value).encode())
    for group in optimizer.param_groups:
        h.update(repr(sorted((k,v) for k,v in group.items() if k!='params')).encode())
    return h.hexdigest()


def setup():
    assert CUDA_UNINITIALIZED_AFTER_TORCH_IMPORT
    assert sha(SOURCE)==SOURCE_SHA
    m=types.ModuleType('frozen_f2_strict_prefix');m.__file__=str(SOURCE);sys.modules[m.__name__]=m
    text=SOURCE.read_text(encoding='utf-8').split('# %% 4. 每次Run All')[0]
    exec(compile(text,str(SOURCE),'exec'),m.__dict__)
    assert torch.are_deterministic_algorithms_enabled()==STRICT
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.cuda.is_available()
    network=m.verify_network_definitions()
    env=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),GPU=torch.cuda.get_device_name(0),platform=platform.platform(),
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        inherited_workspace=INHERITED_WORKSPACE,actual_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        workspace_configured_before_torch_import=STRICT,cuda_uninitialized_after_torch_import=CUDA_UNINITIALIZED_AFTER_TORCH_IMPORT,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),warn_only=False,
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
        packages={p:importlib.metadata.version(p) for p in ['torch','pytorch-forecasting','numpy','pandas','lightning']},
        launch_command=sys.argv,process_id=os.getpid(),network=network)
    save(OUT/'ENVIRONMENT.json',env)
    # No holdout values parsed; keep original 7:1:2 boundaries fixed.
    frame=pd.read_csv(m.SOURCE,nrows=3074 if args.task.startswith('strict_') else 2690)
    train,_=m.build_windows(frame,0,2690)
    assert len(train.origins)==1750
    scaler=m.TrainScaler().fit(train.x_raw)
    x=m.masked_inputs(train.x_raw,scaler,'top10');y=(train.y/scaler.target_scale).astype('float32')
    xv=yv=None
    if args.task.startswith('strict_'):
        val,_=m.build_windows(frame,2690,3074);assert len(val.origins)==218
        xv=m.masked_inputs(val.x_raw,scaler,'top10');yv=(val.y/scaler.target_scale).astype('float32')
    model=m.initialize_model(202)
    optimizer=torch.optim.AdamW(model.parameters(),lr=m.HYPERPARAMETERS['learning_rate'],weight_decay=m.HYPERPARAMETERS['weight_decay'])
    assert not optimizer.state
    loader=m.DataLoader(m.TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.arange(len(x))),batch_size=32,
        shuffle=True,generator=torch.Generator().manual_seed(202),num_workers=0,drop_last=False)
    save(OUT/'INITIALIZATION.json',dict(initial_state_sha256=m.state_sha(model),initial_parameter_sha256=parameter_data_sha(model),
        optimizer_initial_state_empty=True,optimizer_initial_sha256=optimizer_sha(model,optimizer),rng=rng(),
        prepared_train_sha256=m.array_sha(x),prepared_validation_sha256=m.array_sha(xv) if xv is not None else None,
        train_origins_sha256=m.array_sha(train.origins),data_sha256=sha(m.SOURCE),code_sha256=SOURCE_SHA,
        hyperparameters=m.HYPERPARAMETERS,top10=m.MASKS['top10'],top11=m.MASKS['top11'],
        partitions={'TRAIN':[0,2690],'VALIDATION':[2690,3074],'TEST':[3074,3843]},
        target='raw DeltaV',history=48,horizon=24,target_scale_V=scaler.target_scale,
        seed=202,feature_mode='top10',formal_max_epochs=450,formal_patience=40,
        diagnostic_epoch_budget=6 if args.task.startswith('strict_') else None,no_historical_weights=True,no_holdout_scoring=True))
    return m,model,optimizer,loader,xv,yv


def update(m,model,optimizer,xb,yb):
    global PHASE,UPDATES
    PHASE='forward';optimizer.zero_grad(set_to_none=True)
    prediction=model.predict_delta(xb);loss=m.F.mse_loss(prediction,yb)
    assert bool(torch.isfinite(loss))
    PHASE='backward';loss.backward()
    PHASE='gradient_clip';norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert bool(torch.isfinite(norm)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    PHASE='optimizer_step';optimizer.step();UPDATES+=1
    return loss


def prefix(m,model,optimizer,loader,xv,yv):
    global PHASE,EPOCH,BATCH,UPDATES
    dropout_calls=[]
    def hook(name):
        def capture(layer,inputs,outputs):
            if layer.training:
                eligible=inputs[0].detach().ne(0);kept=outputs.detach().ne(0)&eligible
                dropout_calls.append(dict(module=name,p=layer.p,eligible_sha=tensor_sha(eligible),kept_sha=tensor_sha(kept)))
        return capture
    handles=[layer.register_forward_hook(hook(name)) for name,layer in model.named_modules() if isinstance(layer,torch.nn.Dropout)]
    initial=m.state_sha(model);started=time.perf_counter();history=[];best=math.inf;best_epoch=stale=0
    with (OUT/'BATCH_TRACE.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for EPOCH in range(1,7):
            assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
            model.train();losses=[];norms=[];order=[];epoch_start=time.perf_counter()
            for BATCH,(xb,yb,indices) in enumerate(loader,1):
                xb,yb=xb.cuda(),yb.cuda();dropout_calls.clear()
                record=dict(epoch=EPOCH,batch=BATCH,step=UPDATES+1,indices=indices.tolist(),input_sha=tensor_sha(xb),
                    target_sha=tensor_sha(yb),rng_before=rng(),weights_before=m.state_sha(model),optimizer_before=optimizer_sha(model,optimizer))
                PHASE='forward';optimizer.zero_grad(set_to_none=True)
                pred=model.predict_delta(xb);record['rng_after_forward']=rng()
                loss=m.F.mse_loss(pred,yb);assert bool(torch.isfinite(loss))
                PHASE='backward';loss.backward();record['gradients_before_clip']=parameter_data_sha(model,gradient=True)
                if UPDATES==0:
                    torch.save({name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None},OUT/'first_batch_gradients.pt')
                PHASE='gradient_clip';norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert bool(torch.isfinite(norm)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                record['gradients_after_clip']=parameter_data_sha(model,gradient=True)
                PHASE='optimizer_step';optimizer.step();UPDATES+=1
                record.update(loss=float(loss.detach()),gradient_norm=float(norm),output_sha=tensor_sha(pred),rng_after_step=rng(),
                    weights_after=m.state_sha(model),optimizer_after=optimizer_sha(model,optimizer),dropouts=list(dropout_calls))
                stream.write(json.dumps(record,separators=(',',':'))+'\n');stream.flush()
                losses.append(float(loss.detach()));norms.append(float(norm));order.extend(indices.tolist())
            PHASE='validation';vp=m.predict_standardized(model,xv,32)
            val_mse=float(np.mean((vp.astype('float64')-yv.astype('float64'))**2))
            if val_mse**.5<best-1e-12:best,best_epoch,stale=val_mse**.5,EPOCH,0
            else:stale+=1
            row=dict(epoch=EPOCH,train_standardized_MSE=float(np.mean(losses)),validation_standardized_MSE=val_mse,
                best_epoch=best_epoch,stale_epochs=stale,gradient_norm_min=min(norms),gradient_norm_max=max(norms),
                sampler_order_sha256=m.array_sha(np.asarray(order,np.int64)),weights_sha256=m.state_sha(model),
                optimizer_sha256=optimizer_sha(model,optimizer),rng=rng(),epoch_seconds=time.perf_counter()-epoch_start)
            history.append(row);save(OUT/'EPOCH_TRACE.json',history)
            print(args.task,'Epoch',EPOCH,'Train',row['train_standardized_MSE'],'Validation',val_mse,flush=True)
    for handle in handles:handle.remove()
    assert UPDATES==330 and m.state_sha(model)!=initial
    PHASE='snapshot'
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer_state_dict=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
        diagnostic_only=True,seed=202,feature_mode='top10',epoch=6),OUT/'diagnostic_epoch6.pt')
    save(OUT/'COMPLETE.json',dict(status='STRICT_PREFIX_COMPLETE',epochs=6,optimizer_updates=UPDATES,
        training_seconds_including_hashes=time.perf_counter()-started,final_state_sha256=m.state_sha(model),
        final_optimizer_sha256=optimizer_sha(model,optimizer),no_holdout_scored=True,formal_full_training=False))


def benchmark(m,model,optimizer,loader):
    global PHASE,BATCH
    xb,yb,indices=next(iter(loader));xb,yb=xb.cuda(),yb.cuda();model.train()
    PHASE='benchmark_warmup'
    for BATCH in range(1,6):update(m,model,optimizer,xb,yb)
    torch.cuda.synchronize();records=[]
    for BATCH in range(1,21):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter();start.record();update(m,model,optimizer,xb,yb);end.record();end.synchronize()
        records.append(dict(step=BATCH,GPU_elapsed_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-wall)*1000))
    save(OUT/'BENCHMARK.json',dict(status='SPEED_ONLY_COMPLETE',strict=STRICT,warmup_updates=5,timed_updates=20,total_updates=UPDATES,
        rows=records,input_sha=tensor_sha(xb),target_sha=tensor_sha(yb),indices=indices.tolist(),
        scope='same fixed Train batch; real forward/backward/AdamW; no hashes or dropout hooks in timed region',
        no_holdout_scored=True,no_accuracy_score=True))
    print(args.task,'speed-only updates',UPDATES,flush=True)


if __name__=='__main__':
    try:
        m,model,optimizer,loader,xv,yv=setup()
        if args.task.startswith('strict_'):prefix(m,model,optimizer,loader,xv,yv)
        else:benchmark(m,model,optimizer,loader)
    except Exception as error:
        save(OUT/'FAILURE.json',dict(status='FAILURE_PRESERVED',error_type=type(error).__name__,error=str(error),
            phase=PHASE,epoch=EPOCH,batch=BATCH,optimizer_updates=UPDATES,traceback=traceback.format_exc(),
            source_sha256=sha(SOURCE),workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            strict_enabled=torch.are_deterministic_algorithms_enabled(),network_replacement_attempted=False))
        raise

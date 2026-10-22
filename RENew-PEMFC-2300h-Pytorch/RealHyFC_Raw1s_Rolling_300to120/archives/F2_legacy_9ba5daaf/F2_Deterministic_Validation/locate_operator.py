"""One zero-update error-location probe after the strict prefix is rejected.

Autograd anomaly tracing records the original forward callsite; no optimizer is
constructed, no operation is replaced, and no holdout record is parsed.
"""
import os
import sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if 'torch' in sys.modules:raise RuntimeError('Fresh process required')

from pathlib import Path
import hashlib
import inspect
import json
import traceback
import types
import warnings

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
FOLDER=HERE.parent
RUN=HERE/json.loads((HERE/'LATEST_RUN.json').read_text(encoding='utf-8'))['run']
OUT=RUN/'operator_location'
OUT.mkdir(exist_ok=False)


def save(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    save('PLAN.json',dict(reason='Locate the exact forward source behind the already observed unsupported backward operator.',
        maximum_forward_backward_attempts=1,optimizer_constructed=False,optimizer_updates=0,
        no_model_replacement=True,no_holdout_read=True,anomaly_tracing_only=True))
    uninitialized=not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(True,warn_only=False)
    source=FOLDER/'F2_Train_Interactive.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='3dbdc602eb2229a54ee7d5eaa841a6dd94798e8227dbfc6f60db1e78f66419de'
    m=types.ModuleType('f2_strict_operator_location');m.__file__=str(source);sys.modules[m.__name__]=m
    exec(compile(source.read_text(encoding='utf-8').split('# %% 4. 每次Run All')[0],str(source),'exec'),m.__dict__)
    warnings.filterwarnings('ignore',message="Attribute '(loss|logging_metrics)' is an instance of")
    frame=pd.read_csv(m.SOURCE,nrows=2690);train,_=m.build_windows(frame,0,2690)
    scaler=m.TrainScaler().fit(train.x_raw)
    x=m.masked_inputs(train.x_raw,scaler,'top10');y=(train.y/scaler.target_scale).astype('float32')
    model=m.initialize_model(202);initial=m.state_sha(model)
    loader=m.DataLoader(m.TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.arange(len(x))),
        batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(202),num_workers=0,drop_last=False)
    xb,yb,indices=next(iter(loader));model.train();model.zero_grad(set_to_none=True)
    modules=[]
    for name,layer in model.named_modules():
        if type(layer).__name__=='TimeDistributedInterpolation':
            function=type(layer).interpolate
            lines,first_line=inspect.getsourcelines(function);file=Path(inspect.getsourcefile(function))
            calls=[first_line+i for i,line in enumerate(lines) if 'F.interpolate(' in line]
            modules.append(dict(module=name,class_name=type(layer).__name__,output_size=layer.output_size,
                source_file=str(file),function_first_line=first_line,interpolate_call_lines=calls,
                source_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    result=dict(workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],cuda_uninitialized_after_import=uninitialized,
        strict=True,warn_only=False,optimizer_constructed=False,optimizer_updates=0,
        matched_initialization=initial==json.loads((RUN/'strict_a/INITIALIZATION.json').read_text(encoding='utf-8'))['initial_state_sha256'],
        interpolation_modules=modules,indices=indices.tolist())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                with torch.autograd.detect_anomaly(check_nan=True):
                    prediction=model.predict_delta(xb.cuda())
                    loss=m.F.mse_loss(prediction,yb.cuda())
                    loss.backward()
            except RuntimeError as error:
                result.update(status='EXPECTED_UNSUPPORTED_BACKWARD_LOCATED',error=str(error),traceback=traceback.format_exc())
            result['anomaly_warnings']=[str(item.message) for item in caught]
    finally:
        result['weights_unchanged']=m.state_sha(model)==initial
        save('OPERATOR_LOCATION.json',result)
    print(result.get('status','NO_ERROR_REPRODUCED'),result.get('error','').split('.')[0],flush=True)

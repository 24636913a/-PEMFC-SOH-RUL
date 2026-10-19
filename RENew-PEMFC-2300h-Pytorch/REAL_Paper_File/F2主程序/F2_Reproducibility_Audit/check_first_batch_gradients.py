"""Two first-batch backward checks, ZERO optimizer updates and no holdout access.

Additional fixed evidence check: distinguish signed-zero hash differences from
nonzero numerical gradient differences. Does not extend formal training.
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from probe_training_prefix import definitions, rng, tensor_sha, write

HERE=Path(__file__).resolve().parent


def child(label):
    out=HERE/'outputs'/('backward_'+label);out.mkdir(exist_ok=False)
    m=definitions();m.configure_environment()
    frame=pd.read_csv(m.SOURCE,nrows=2690)
    train,_=m.build_windows(frame,0,2690)
    scaler=m.TrainScaler().fit(train.x_raw)
    x=m.masked_inputs(train.x_raw,scaler,'top10');y=(train.y/scaler.target_scale).astype('float32')
    model=m.initialize_model(202)
    loader=m.DataLoader(m.TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.arange(len(x))),
        batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(202),num_workers=0,drop_last=False)
    xb,yb,indices=next(iter(loader));xb,yb=xb.cuda(),yb.cuda()
    model.train();model.zero_grad(set_to_none=True)
    before=m.state_sha(model);rb=rng();pred=model.predict_delta(xb)
    loss=m.F.mse_loss(pred,yb);loss.backward()
    grads={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
    torch.save(grads,out/'gradients_before_clip.pt')
    write(out/'RECEIPT.json',dict(initialization=before,weights_after=m.state_sha(model),rng_before=rb,rng_after=rng(),
        input_sha=tensor_sha(xb),target_sha=tensor_sha(yb),indices=indices.tolist(),output_sha=tensor_sha(pred),
        loss=float(loss.detach()),optimizer_updates=0,no_holdout=True))


if __name__=='__main__':
    if len(sys.argv)>1:child(sys.argv[1])
    else:
        plan=HERE/'BACKWARD_CHECK_PLAN.json';assert not plan.exists(),'Never overwrite diagnostic evidence'
        write(plan,dict(reason='Per-batch gradient hashes differ; quantify genuine numeric differences versus signed zero.',
            fixed_budget='two fresh first-batch forward/backward passes; ZERO optimizer steps',
            source_program_unchanged=True,no_historical_weights=True,no_holdout=True))
        for label in ['a','b']:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),label],check=True,cwd=HERE.parent)
        print('TWO_NO_UPDATE_BACKWARD_CHECKS_COMPLETED',flush=True)

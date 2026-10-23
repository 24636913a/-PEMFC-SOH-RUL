"""Four isolated diagnostic processes; strict determinism, max 48 epochs."""
from __future__ import annotations
import os
# Set BEFORE PyTorch or pytorch-forecasting is imported (including helpers).
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from g_common import *

TRACES = ('G-E-A','G-E-B','G-F-A','G-F-B')

def freeze():
    if PROTOCOL.exists() or (OUT/'execution_audit/RUN_LEDGER.json').exists():
        raise RuntimeError('EXISTING_PROTOCOL_OR_RUN_NO_REFREEZE')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=BASE:
        raise RuntimeError('BASE_COMMIT_CHANGED')
    if not (OUT/'historical_audit/HISTORICAL_SUMMARY.json').exists():
        raise RuntimeError('HISTORICAL_AUDIT_REQUIRED')
    tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=PHASE,text=True,capture_output=True)
    write_json(OUT/'execution_audit/REGRESSION_TEST_RESULTS.json',{'returncode':tests.returncode,
        'stdout':tests.stdout,'stderr':tests.stderr,'synthetic_only':True,'test_data_accessed':False})
    if tests.returncode!=0:raise RuntimeError('REGRESSION_TEST_FAILURE')
    f=yaml.safe_load((F/'configs/F0_F2_EXECUTION_PROTOCOL.yaml').read_text(encoding='utf-8'))
    cfg={'phase':PHASE.name,'base_commit':BASE,'purpose':'DIAGNOSTIC_ONLY_NOT_FORMAL_MODEL',
         'trajectories':list(TRACES),'seed':42,'epochs_per_trajectory':12,'maximum_total_epochs':48,
         'instrumented_epochs':5,'data':{'read_nrows':3007,'filter':'bin_end_ophrs2_h < 1819.997738',
             'development_rows':3005,'train_rows':[0,2103],'validation_rows':[2103,2404],
             'expected_windows':{'train':1380,'validation':205},'assessment_windows':'PROHIBITED',
             'test_access':'PROHIBITED','full_N1_hash':'PROHIBITED'},
         'task':f['task'],'hyperparameters':f['hyperparameters'],
         'preprocessing':f['preprocessing'],
         'controlled_runtime':{'cpu_threads':1,'device':'cuda_if_available_else_cpu',
             'cudnn_deterministic':True,'cudnn_benchmark':False,'torch_deterministic_algorithms':True,
             'deterministic_warn_only':False,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
             'TF32_matmul':False,'TF32_cudnn':False,'float32_matmul_precision':'highest',
             'separate_fresh_python_process_for_each_trajectory':True},
         'execution_paths':{'E':'independent faithful E loop; 2-field dataset; CPU clone best state',
             'F':'independent faithful F loop; 3-field dataset; finite gradient check; improvement checkpoint'},
         'instrumentation':'symmetric before/after hashes; CUDA-to-CPU copies synchronize and perturb timing; no tensor replicas saved',
         'comparison':{'exact_equality':True,'no_posthoc_tolerance_for_bitwise_equality':True,
             'compare_all_12_epoch_metrics_and_first_5_epoch_batch_records':True},
         'early_stopping':'record historical criterion and best state, but fixed 12 epochs; patience40 cannot trigger',
         'fail_closed':['unsafe/missing historical checkpoint','truth mismatch','data assertions',
             'strict determinism error','nonfinite loss/gradient','history/source modification','Test access'],
         'source_and_code_hashes':identity(),
         'interpretation':'controlled environment may differ from history; no unique historical cause attribution without intervention evidence'}
    PROTOCOL.parent.mkdir(parents=True,exist_ok=True)
    PROTOCOL.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    digest=file_hash(PROTOCOL)
    SHA.write_text(digest+'  G_REPLAY_PROTOCOL.yaml\n',encoding='utf-8')
    write_json(OUT/'execution_audit/PROTOCOL_FREEZE.json',{'protocol_sha256':digest,
        'frozen_before_replay':True,'code_identity_sha256':hashlib.sha256(json.dumps(identity(),sort_keys=True).encode()).hexdigest()})
    print('PROTOCOL_FROZEN '+digest,flush=True)

def verify():
    if file_hash(PROTOCOL)!=SHA.read_text(encoding='utf-8').split()[0]:raise RuntimeError('PROTOCOL_HASH_MISMATCH')
    cfg=yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    if identity()!=cfg['source_and_code_hashes']:raise RuntimeError('HISTORICAL_OR_EXECUTION_SOURCE_CHANGED')
    if cfg['maximum_total_epochs']!=48 or cfg['epochs_per_trajectory']!=12:raise RuntimeError('BUDGET_CHANGED')
    return cfg

def environment():
    import pytorch_forecasting as pf
    return {'python':platform.python_version(),'torch':str(torch.__version__),'pytorch_forecasting':pf.__version__,
        'CUDA':torch.version.cuda,'cuDNN':torch.backends.cudnn.version(),
        'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cpu_threads':torch.get_num_threads(),'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']}

def configure():
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True,warn_only=False)

def prepare(cfg):
    e=import_e();ecfg,_=e.load_protocol()
    dev=e.c.load_development(ecfg)
    train=e.c.build_windows(dev,ecfg,0,2103); val=e.c.build_windows(dev,ecfg,2103,2404)
    if (len(dev),len(train.origins),len(val.origins))!=(3005,1380,205):raise RuntimeError('DATA_PROTOCOL_ASSERTION')
    if set(train.targets.ravel())&set(val.targets.ravel()):raise RuntimeError('TARGET_OVERLAP')
    _,top10=e.top10_scores(train,cfg['task']['raw_f2_feature_order'])
    if top10!=cfg['preprocessing']['Top10_indices']:raise RuntimeError('TOP10_CHANGED')
    scaler=e.CommonScaler().fit(train.x)
    for path in [E/'outputs/E_COMMON_SCALER.json',F/'outputs/audits/F_COMMON_SCALER.json']:
        old=read_json(path)
        if not np.array_equal(scaler.mean,np.array(old['mean'])) or not np.array_equal(scaler.scale,np.array(old['scale'])):
            raise RuntimeError('HISTORICAL_SCALER_MISMATCH')
    tx,vx=e.prep(train.x,scaler,top10,'EWMA'),e.prep(val.x,scaler,top10,'EWMA')
    fp={'train_raw_x_sha256':array_hash(train.x),'train_y_sha256':array_hash(train.y),
        'train_preprocessed_x_sha256':array_hash(tx),'validation_x_sha256':array_hash(vx),
        'scaler_sha256':array_hash(np.stack([scaler.mean,scaler.scale]))}
    hist=read_json(F/'outputs/audits/F_DATA_ACCESS_AUDIT.json')
    if any(fp[k]!=hist[k] for k in fp):raise RuntimeError('F_HISTORICAL_INPUT_HASH_MISMATCH')
    fp.update({'development_prefix_fingerprint_sha256':hashlib.sha256(dev.to_csv(index=False,header=True,
        lineterminator='\n',float_format='%.17g').encode('utf-8')).hexdigest(),
        'train_origins_sha256':array_hash(train.origins),'validation_origins_sha256':array_hash(val.origins),
        'train_target_rows_sha256':array_hash(train.targets),'validation_y_sha256':array_hash(val.y),
        'Top10_indices':top10,'dtype':str(tx.dtype),'read_nrows':3007,'development_rows':3005,
        'assessment_windows_constructed':False,'new_assessment_inference':False,'test_accessed':False,
        'whole_N1_read_or_hash':False,'EWMA_initialization':'per-window x[:,0,0]; frozen alpha .3'})
    return e,train,val,scaler,tx,vx,fp

class TracedDataset(TensorDataset):
    """Observe sampler indices without changing item arity or drawing RNG."""
    def __init__(self,*tensors):super().__init__(*tensors);self.indices=[]
    def __getitem__(self,index):self.indices.append(int(index));return super().__getitem__(index)

class Recorder:
    def __init__(self,name,model,opt,generator,origins,folder):
        self.name,self.model,self.opt,self.generator,self.origins,self.folder=name,model,opt,generator,origins,folder
    def rng(self):
        return {'cpu':tensor_tree_hash(torch.get_rng_state()),
            'cuda':tensor_tree_hash(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
            'loader':tensor_tree_hash(self.generator.get_state()),
            'numpy':tensor_tree_hash(np.random.get_state()),'python':tensor_tree_hash(random.getstate())}
    def gradients(self):return tensor_tree_hash({k:p.grad for k,p in self.model.named_parameters()})
    def start(self,epoch,batch,xb,yb,ids):
        self.row=None
        if epoch>5:return
        self.row={'epoch':epoch,'batch':batch,'origins':self.origins[ids].tolist(),
            'input_sha256':array_hash(xb.detach().cpu().numpy()),'label_sha256':array_hash(yb.detach().cpu().numpy()),
            'parameters_before':tensor_tree_hash(self.model.state_dict()),'rng_before':self.rng(),
            'optimizer_before':tensor_tree_hash(self.opt.state_dict())}
    def forward(self,pred,loss):
        if self.row is None:return
        a=pred.detach().cpu().numpy()
        self.row.update(forward_sha256=array_hash(a),forward_mean=float(a.mean()),forward_std=float(a.std()),
            forward_min=float(a.min()),forward_max=float(a.max()),loss=float(loss.detach().cpu()),rng_after_forward=self.rng())
    def backward(self):
        if self.row is not None:self.row.update(gradient_before_clip=self.gradients(),rng_after_backward=self.rng())
    def clipped(self,norm):
        if self.row is not None:self.row.update(gradient_norm=float(norm.detach().cpu()),gradient_after_clip=self.gradients())
    def stepped(self):
        if self.row is None:return
        self.row.update(parameters_after=tensor_tree_hash(self.model.state_dict()),optimizer_after=tensor_tree_hash(self.opt.state_dict()),rng_after_step=self.rng())
        with (self.folder/'batches.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(self.row,allow_nan=False)+'\n')

def validation_record(epoch,model,opt,rec,vx,vy,scale,e,device):
    before=rec.rng();pred=e.predict(model,vx,32,device);after=rec.rng()
    rmse=float(np.sqrt(np.mean((pred-vy)**2)))
    if not math.isfinite(rmse):raise RuntimeError('NONFINITE_VALIDATION')
    return rmse,{'epoch':epoch,'rng_before_validation':before,'rng_after_validation':after,
        'validation_prediction_sha256':array_hash(pred),'parameters_after_validation':tensor_tree_hash(model.state_dict()),
        'optimizer_after_validation':tensor_tree_hash(opt.state_dict()),'validation_standardized_RMSE':rmse,
        'validation_RMSE_mV':rmse*scale*1000}

def loop_e(name,cfg,e,train,val,scaler,tx,vx,device,folder):
    # E2 order: seed → preprocessing → model → init digest → optimizer → 2-field loader.
    e.set_seed(42)
    tx=e.prep(train.x,scaler,cfg['preprocessing']['Top10_indices'],'EWMA')
    vx=e.prep(val.x,scaler,cfg['preprocessing']['Top10_indices'],'EWMA')
    ty,vy=train.y/scaler.target_scale,val.y/scaler.target_scale
    p=cfg['hyperparameters'];model=e.make_model(p,device);init=e.state_digest(model)
    if init!=read_json(F/'outputs/audits/F_FORMAL_INITIALIZATION_AUDIT.json')['shared_native_sha256']:
        raise RuntimeError('E_REPLAY_INITIALIZATION_MISMATCH')
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(p['learning_rate']),weight_decay=float(p['weight_decay']))
    ds=TracedDataset(torch.from_numpy(tx),torch.from_numpy(ty.astype(np.float32)))
    gen=torch.Generator().manual_seed(42)
    loader=DataLoader(ds,batch_size=32,shuffle=True,generator=gen,num_workers=0)
    rec=Recorder(name,model,optimizer,gen,train.origins,folder)
    write_json(folder/'initial.json',{'state_sha256':init,'parameters':sum(p.numel() for p in model.parameters()),'rng':rec.rng()})
    best,best_epoch,stale,best_state=math.inf,0,0,None
    history=[]
    for epoch in range(1,13):
        model.train();losses=[];ds.indices=[]
        for batch,(xb,yb) in enumerate(loader):
            ids=np.array(ds.indices[-len(xb):]);xb,yb=xb.to(device),yb.to(device)
            optimizer.zero_grad(set_to_none=True);rec.start(epoch,batch,xb,yb,ids)
            pred=model.predict_delta(xb);loss=torch.nn.functional.mse_loss(pred,yb);rec.forward(pred,loss)
            if not torch.isfinite(loss):raise RuntimeError('E_NONFINITE_LOSS')
            loss.backward();rec.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(p['gradient_clip_norm']));rec.clipped(norm)
            # Symmetric instrumentation guard; no arithmetic change.
            if not torch.isfinite(norm):raise RuntimeError('E_DIAGNOSTIC_NONFINITE_GRADIENT')
            optimizer.step();rec.stepped();losses.append(float(loss.detach().cpu()))
        rmse,row=validation_record(epoch,model,optimizer,rec,vx,vy,scaler.target_scale,e,device)
        row['train_standardized_MSE']=float(np.mean(losses))
        if rmse<best-1e-12:
            best,best_epoch,stale=rmse,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        row.update(best_epoch=best_epoch,stale_epochs=stale,rng_after_best_handling=rec.rng())
        history.append(row);write_json(folder/'epochs.json',history)
        print(f'{name} epoch {epoch}/12 train={row["train_standardized_MSE"]:.12g} val={rmse:.12g}',flush=True)
    # Diagnostic summary only, no formal predictions/checkpoint.
    write_json(folder/'final.json',{'status':'COMPLETED','epochs':12,'initialization_sha256':init,
        'final_state_sha256':e.state_digest(model),'best_epoch':best_epoch,'best_validation_standardized_RMSE':best,
        'best_state_sha256':tensor_tree_hash(best_state),'diagnostic_only':True})

def loop_f(name,cfg,e,train,val,scaler,tx,vx,device,folder):
    # F0 order: seed → direct canonical construction/init hash → optimizer → target scaling → 3-field loader.
    e.set_seed(42)
    sys.path.insert(0,str(T0.parent))
    from canonical_tft import CanonicalStrictHistoryTFT
    p=cfg['hyperparameters']
    model=CanonicalStrictHistoryTFT(hidden_size=p['hidden_size'],hidden_continuous_size=p['hidden_continuous_size'],
        attention_head_size=4,lstm_layers=1,dropout=p['dropout']).to(device)
    init=e.state_digest(model)
    if init!=read_json(F/'outputs/audits/F_FORMAL_INITIALIZATION_AUDIT.json')['shared_native_sha256']:
        raise RuntimeError('F_REPLAY_INITIALIZATION_MISMATCH')
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    ty,vy=train.y/scaler.target_scale,val.y/scaler.target_scale
    ds=TracedDataset(torch.from_numpy(tx),torch.from_numpy(ty.astype(np.float32)),torch.arange(len(tx)))
    gen=torch.Generator().manual_seed(42)
    loader=DataLoader(ds,batch_size=32,shuffle=True,generator=gen,num_workers=0)
    rec=Recorder(name,model,optimizer,gen,train.origins,folder)
    write_json(folder/'initial.json',{'state_sha256':init,'parameters':sum(p.numel() for p in model.parameters()),'rng':rec.rng()})
    history=[];best,best_epoch,stale=math.inf,0,0
    ck=folder/'DIAGNOSTIC_ONLY_NOT_FORMAL_MODEL_best.pt'
    for epoch in range(1,13):
        model.train();losses=[];ds.indices=[]
        for batch,(xb,yb,ids) in enumerate(loader):
            xb,yb=xb.to(device),yb.to(device)
            optimizer.zero_grad(set_to_none=True);rec.start(epoch,batch,xb,yb,ids.numpy())
            pred=model.predict_delta(xb);loss=torch.nn.functional.mse_loss(pred,yb);rec.forward(pred,loss)
            if not torch.isfinite(loss):raise RuntimeError('F_NONFINITE_LOSS')
            loss.backward();rec.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip_norm']);rec.clipped(norm)
            if not torch.isfinite(norm):raise RuntimeError('F_NONFINITE_GRADIENT')
            optimizer.step();rec.stepped();losses.append(float(loss.detach().cpu()))
        rmse,row=validation_record(epoch,model,optimizer,rec,vx,vy,scaler.target_scale,e,device)
        if rmse<best-1e-12:
            best,best_epoch,stale=rmse,epoch,0
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'status':'DIAGNOSTIC_ONLY_NOT_FORMAL_MODEL','best_epoch':best_epoch,
                'validation_fixed_prediction_sha256':row['validation_prediction_sha256']},ck)
        else:stale+=1
        row.update(train_standardized_MSE=float(np.mean(losses)),best_epoch=best_epoch,stale_epochs=stale,rng_after_best_handling=rec.rng())
        history.append(row);write_json(folder/'epochs.json',history)
        print(f'{name} epoch {epoch}/12 train={row["train_standardized_MSE"]:.12g} val={rmse:.12g}',flush=True)
    write_json(folder/'final.json',{'status':'COMPLETED','epochs':12,'initialization_sha256':init,
        'final_state_sha256':e.state_digest(model),'best_epoch':best_epoch,'best_validation_standardized_RMSE':best,
        'best_state_sha256':tensor_tree_hash(torch.load(ck,map_location='cpu',weights_only=True)['state_dict']),
        'diagnostic_only':True})

def worker(name):
    folder=OUT/'replay_logs'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'epochs.json').exists() or (folder/'final.json').exists():raise RuntimeError('TRAJECTORY_REPEAT_PROHIBITED')
    try:
        cfg=verify();configure();write_json(folder/'environment.json',environment())
        e,train,val,scaler,tx,vx,fp=prepare(cfg);write_json(folder/'input_audit.json',fp)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loop=loop_e if name.startswith('G-E') else loop_f
        loop(name,cfg,e,train,val,scaler,tx,vx,device,folder)
    except Exception as exc:
        write_json(folder/'TECHNICAL_FAILURE.json',{'status':'PHASE_G_EXECUTION_BLOCKED','exception':repr(exc),
            'traceback':traceback.format_exc(),'strict_determinism_not_disabled':True})
        raise

def run():
    verify();path=OUT/'execution_audit/RUN_LEDGER.json'
    if path.exists():raise RuntimeError('RUN_ALREADY_ENTERED_NO_AUTOMATIC_REPLAY')
    ledger={'status':'RUNNING','trajectories':{},'max_epochs':48,'test_accessed':False,'new_assessment_inference':False}
    write_json(path,ledger)
    for name in TRACES:
        ledger['trajectories'][name]={'status':'STARTED'};write_json(path,ledger)
        folder=OUT/'replay_logs'/name;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'console.log').open('w',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,'-B',__file__,'--worker',name],cwd=PHASE,stdout=log,stderr=subprocess.STDOUT)
        ledger['trajectories'][name]={'status':'COMPLETED' if proc.returncode==0 else 'BLOCKED','returncode':proc.returncode}
        if proc.returncode:
            ledger['status']='PHASE_G_EXECUTION_BLOCKED';write_json(path,ledger)
            raise RuntimeError('CONTROLLED_TRAJECTORY_FAILED:'+name)
        write_json(path,ledger);print(name+' completed 12 epochs',flush=True)
    verify();ledger['status']='COMPLETED';ledger['total_epochs']=48;write_json(path,ledger)

if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--worker',choices=TRACES)
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.run:run()
    else:worker(args.worker)

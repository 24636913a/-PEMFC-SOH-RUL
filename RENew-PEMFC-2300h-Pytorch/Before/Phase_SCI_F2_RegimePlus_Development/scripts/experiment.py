"""One predeclared same-information gate experiment; historical code is read-only.

prepare -> run (six serial fits) -> analyze. No Assessment/late-life loader.
"""
from __future__ import annotations
import os
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse, ctypes, hashlib, importlib.util, json, math, random, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

P = Path(__file__).resolve().parents[1]
R = P.parent
BASE = '36572a6678a9e64df689c8c6ec282c7233b6ac36'
spec = importlib.util.spec_from_file_location('historical_f2_readonly', R/'RealHyFC_F2_Main.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
D = old.FROZEN_DATA
FEATURES = ['j_recent6_minus48', 'zero_recent12_minus_previous12', 'last_majority_stop_distance_over48']
SEEDS = [42, 52, 62]

def sha(p): return old.sha256(Path(p))
def now(): return datetime.now(timezone.utc).isoformat()
def req(c, m):
    if not c: raise RuntimeError(m)
def path(s):
    p=(P/s).resolve(); req(p.is_relative_to(P.resolve()), 'OUTSIDE_NEW_STAGE'); p.parent.mkdir(parents=True, exist_ok=True); return p
def text(s, value):
    p=path(s); t=p.with_suffix(p.suffix+'.writing'); t.write_text(value, encoding='utf-8', newline='\n')
    for delay in [.05,.1,.2,.5,1,2,None]:
        try: os.replace(t,p); return
        except PermissionError:
            if delay is None: raise
            time.sleep(delay)
def js(s,v): text(s,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv(s,v): text(s,pd.DataFrame(v).to_csv(index=False,float_format='%.17g',lineterminator='\n'))
def readj(s): return json.loads(path(s).read_text(encoding='utf-8'))
def save_torch(s,v):
    p=path(s); t=p.with_suffix(p.suffix+'.writing'); torch.save(v,t)
    for delay in [.05,.1,.2,.5,1,2,None]:
        try: os.replace(t,p); return
        except PermissionError:
            if delay is None: raise
            time.sleep(delay)

def load_data():
    req(sha(D/'FROZEN_H_DATA.npz')=='cbb212134eb664f1a43d4851f20686fe575c06eb6b2b0b893aed0f166fcef83e','CACHE_SHA')
    z=dict(np.load(D/'FROZEN_H_DATA.npz',allow_pickle=False))
    req(z['tx'].shape==(1380,48,14) and z['vx'].shape==(205,48,14),'SHAPE')
    req(z['train_targets'].max()<2103 and z['val_targets'].min()>=2103 and z['val_targets'].max()<2404,'TARGET_PARTITIONS')
    for split in ['train','val']:
        req(np.array_equal(z[split+'_targets'],z[split+'_origins'][:,None]+np.arange(1,25)), 'TARGET_IDENTITY')
    sc=json.loads((D/'SCALER.json').read_text(encoding='utf-8'))
    # Reproduce causal EWMA and frozen masked inputs from bounded history only.
    for split,key in [('train','tx'),('val','vx')]:
        x=z[split+'_x_raw'].copy()
        for t in range(1,48): x[:,t,0]=.3*x[:,t,0]+.7*x[:,t-1,0]
        x=((x-np.array(sc['mean']))/np.array(sc['scale'])).astype(np.float32)
        x[:,:,[1,2,5,12]]=0
        req(np.array_equal(x,z[key]),'PREPROCESS_IDENTITY:'+split)
    return z,sc

def derive(x, zero_mean, zero_scale):
    """Only retained scaled history channels 8 and 13; no target or time input.

    Recency is a *within-window bin distance*, never calendar downtime.
    Last observed majority-zero bucket is censored when absent in 48 bins.
    """
    j=x[:,:,8]; zero=x[:,:,13]*zero_scale+zero_mean
    stop=zero>.5
    index=torch.arange(48,device=x.device).expand(x.shape[0],-1)
    last=torch.where(stop,index,-torch.ones_like(index)).amax(1)
    distance=torch.where(last>=0,47-last,48).to(x.dtype)/48.
    return torch.stack((j[:,-6:].mean(1)-j.mean(1),zero[:,-12:].mean(1)-zero[:,-24:-12].mean(1),distance),1)

class PlusGate(nn.Module):
    def __init__(self, original):
        super().__init__(); self.hidden=nn.Linear(8,8); self.activation=nn.ELU(); self.output=nn.Linear(8,1)
        with torch.no_grad():
            self.hidden.weight[:,:5].copy_(original.hidden.weight)
            self.hidden.bias.copy_(original.hidden.bias)
            self.output.load_state_dict(original.output.state_dict())
    def forward(self,x): return torch.sigmoid(self.output(self.activation(self.hidden(x))))

class RegimePlus(old.RegimeAwareDualPathTFT):
    def regime_context(self,x):
        base=super().regime_context(x)
        extra=(derive(x,self.zero_mean,self.zero_scale)-self.extra_mean)/self.extra_scale
        if self.ablate: extra=extra*0
        if getattr(self,'zero_features_inference',False): extra=extra*0
        return torch.cat([base,extra],1)

def initialize(seed,variant,protocol,device):
    # Original F2 initialization verified first. Shared parameters are untouched.
    model=old.initialize_f2(seed,protocol['hyperparameters'],torch.device('cpu'))
    rng=torch.get_rng_state().clone()
    model.__class__=RegimePlus
    model.regime_gate=PlusGate(model.regime_gate)
    model.register_buffer('zero_mean',torch.tensor(protocol['raw_zero_mean'],dtype=torch.float32))
    model.register_buffer('zero_scale',torch.tensor(protocol['raw_zero_scale'],dtype=torch.float32))
    model.register_buffer('extra_mean',torch.tensor(protocol['extra_mean'],dtype=torch.float32))
    model.register_buffer('extra_scale',torch.tensor(protocol['extra_scale'],dtype=torch.float32))
    model.ablate=variant=='A1'
    torch.set_rng_state(rng) # extra initialization cannot shift paired dropout stream
    req(sum(p.numel() for p in model.parameters())==332460,'PARAMETER_COUNT')
    return model.to(device)

def predict(model,x,device): return old.predict_standardized(model,x,32,device)

def prepare():
    req(not path('configs/FROZEN_PROTOCOL.json').exists(),'PROTOCOL_ALREADY_FROZEN')
    req(subprocess.check_output(['git','rev-parse','HEAD'],cwd=R.parent,text=True).strip()==BASE,'BASE_CHANGED')
    runtime=old.configure_environment('cuda'); device=torch.device('cuda')
    z,sc=load_data(); hp=old.load_historical_hyperparameters()
    train=torch.from_numpy(z['tx'])
    ef=derive(train,sc['mean'][13],sc['scale'][13]).numpy()
    req(np.isfinite(ef).all() and (ef.std(0)>1e-8).all(),'UNCOMPUTABLE_FEATURE')
    original=train[:,:,[11,13,9,6,8]].mean(1).numpy()
    candidates=pd.DataFrame(np.concatenate([original,ef],1),columns=['old_'+str(i) for i in range(5)]+FEATURES)
    csv('outputs/TRAIN_FEATURE_CORRELATIONS.csv',candidates.corr(method='spearman').reset_index(names='feature'))
    csv('outputs/TRAIN_FEATURE_DISTRIBUTIONS.csv',pd.DataFrame(ef,columns=FEATURES).describe(percentiles=[.05,.25,.5,.75,.95]).reset_index(names='statistic'))
    # State grouping uses past observations only; target strata below are separately labelled retrospective.
    load=z['train_x_raw'][:,-6:,11].mean(1)
    transition_threshold=float(np.quantile(load,.75))
    inventory=[]; replay=[]
    for seed in SEEDS:
        cp=old.HISTORY/f'outputs/checkpoints/H-F2-Seed{seed}_best.pt'
        oldpred=old.HISTORY/f'outputs/validation_predictions/H-F2-Seed{seed}.csv'
        ck=torch.load(cp,map_location='cpu',weights_only=True)
        model=old.initialize_f2(seed,hp['parameters'],device); model.load_state_dict(ck['state_dict'],strict=True)
        pred=predict(model,z['vx'],device)*sc['target_scale_V']
        ref=pd.read_csv(oldpred,float_precision='round_trip')
        req(np.array_equal(ref.origin_row.to_numpy(),np.repeat(z['val_origins'],24)),'BASELINE_ORIGIN')
        req(np.array_equal(ref.target_row.to_numpy(),z['val_targets'].ravel()),'BASELINE_TARGET')
        # Historical saved float32 physical predictions must match bit for bit.
        col='y_pred_delta_V'
        if col not in ref: col=[c for c in ref if 'pred' in c and 'delta' in c][0]
        req(np.array_equal(pred.ravel().astype(float),ref[col].to_numpy()),'BASELINE_PREDICTION:'+str(seed))
        csv(f'outputs/predictions/B0_seed{seed}.csv',ref)
        inventory.extend([dict(path=str(cp.relative_to(R)),sha256=sha(cp)),dict(path=str(oldpred.relative_to(R)),sha256=sha(oldpred))])
        replay.append(dict(seed=seed,queries=pred.size,exact=True,max_difference_V=0.))
        del model; torch.cuda.empty_cache()
    protocol=dict(phase=P.name,base_commit=BASE,frozen_utc=now(),seeds=SEEDS,
      run_order=[dict(seed=s,variant=v) for s in SEEDS for v in ['A1','B1']],
      B0='historical frozen F2, replayed exactly',B1='same-information three-feature gate',A1='identical parameters/init, added standardized slots fixed zero',
      train_rows=[0,2103],validation_rows=[2103,2404],train_windows=1380,validation_windows=205,validation_queries=4920,
      information='only retained channels of original 48x14 history; no extra clock or longer history',
      features=FEATURES,stop_definition='zero_current_fraction > 0.5 (majority of bin raw samples, not shutdown duration)',
      recency='(47-last_majority_zero_index)/48; absent -> 1 and an explicit audit-only censor flag (constant in Train, excluded from gate); operating-bin distance, not wall clock',
      raw_zero_mean=sc['mean'][13],raw_zero_scale=sc['scale'][13],extra_mean=ef.astype(float).mean(0).tolist(),extra_scale=ef.astype(float).std(0).tolist(),
      grouping=dict(stop_related='any zero_current_fraction > 0.5 in last 12 history bins',frequent_switch_threshold_train_p75=transition_threshold,
                    frequent_switch='not stop_related and mean last6 load_transition_count >= Train P75',normal='remaining origins'),
      hyperparameters=hp['parameters'],hyperparameter_provenance=hp,runtime=runtime,parameters_B0=332436,parameters_B1_A1=332460,
      success=dict(mean_RMSE_improvement_vs_B0_at_least=.03,mean_RMSE_improvement_vs_A1_at_least=.03,
       seeds_improved_vs_each_at_least=2,no_seed_RMSE_relative_worsening_above=.05,no_stratum_mean_RMSE_worsening_above=.05,
       no_mean_P95_or_endpoint_RMSE_worsening_above=.05,
       concentration='must improve pooled SSE after removing the highest 5% positive-gain unique target times, separately vs B0 and A1; diagnostic only, official metrics unchanged',
       promising='mean RMSE improves vs both but any confirmation condition fails; otherwise NO_CONFIRMED_GAIN'),
      stopping='exactly one candidate, six fits; max450/patience40; no retries of successful fits; technical failure preserved and stops controller',
      limitations=['reused Development Validation','historical B0 training, paired B1/A1 fresh','overlapping targets not independent','no Assessment or late-life evaluation'],
      sources=inventory+[dict(path=str((D/f).relative_to(R)),sha256=sha(D/f)) for f in ['FROZEN_H_DATA.npz','SCALER.json']],
      model_source_sha256=sha(R/'RealHyFC_F2_Main.py'),script_sha256=sha(Path(__file__)))
    csv('outputs/BASELINE_EXACT_REPLAY.csv',replay)
    csv('outputs/TRAIN_WINDOW_FEATURES.csv',pd.DataFrame(ef,columns=FEATURES).assign(origin_row=z['train_origins']))
    # Meaningful leakage tests: future mutations do not affect features; reversing history changes new state while old means agree.
    reversed_ef=derive(train.flip(1),sc['mean'][13],sc['scale'][13]).numpy()
    req(np.max(abs(reversed_ef-ef))>0,'ORDER_SENSITIVITY')
    req(np.allclose(train.flip(1)[:,:,[11,13,9,6,8]].mean(1),torch.from_numpy(original),atol=1e-6),'OLD_MEAN_PERMUTATION')
    req(np.array_equal(z['train_targets'].max(1)<2103,np.ones(1380,dtype=bool)),'TRAIN_LABEL_LEAK')
    # Freeze once; every training process checks immutable digest.
    js('configs/FROZEN_PROTOCOL.json',protocol)
    text('configs/FROZEN_PROTOCOL.sha256',sha(path('configs/FROZEN_PROTOCOL.json'))+'\n')
    pair=[]
    for seed in SEEDS:
        a=initialize(seed,'A1',protocol,device); b=initialize(seed,'B1',protocol,device)
        req(old.state_sha256(a)==old.state_sha256(b),'PAIRED_INITIALIZATION')
        req(np.array_equal(predict(a,z['vx'][:32],device),predict(b,z['vx'][:32],device)),'INITIAL_FUNCTION')
        pair.append(dict(seed=seed,initialization_sha256=old.state_sha256(a),parameters=332460,initial_predictions_exact=True))
        del a,b
    csv('outputs/PAIRED_INITIALIZATION.csv',pair)
    js('outputs/PRETRAIN_QA.json',dict(passed=True,finished_utc=now(),baseline_exact=True,order_sensitive=True,train_only_scaling=True,only_history_features=True,paired_init_exact=True))
    print('PREPARED',sha(path('configs/FROZEN_PROTOCOL.json')),flush=True)

def get_protocol():
    req(sha(path('configs/FROZEN_PROTOCOL.json'))==path('configs/FROZEN_PROTOCOL.sha256').read_text().strip(),'PROTOCOL_MODIFIED')
    p=readj('configs/FROZEN_PROTOCOL.json')
    req(sha(Path(__file__))==p['script_sha256'],'TRAINING_SOURCE_MODIFIED_AFTER_FREEZE')
    req(readj('outputs/PRETRAIN_QA.json')['passed'],'PRETRAIN_QA')
    return p

def fit(seed,variant):
    p=get_protocol(); old.configure_environment('cuda'); device=torch.device('cuda'); z,sc=load_data()
    run=f'{variant}_seed{seed}'; state=f'outputs/ledger/{run}.json'
    req(not path(state).exists(),'RUN_ALREADY_ATTEMPTED:'+run)
    js(state,dict(status='RUNNING',seed=seed,variant=variant,started_utc=now(),protocol_sha256=sha(path('configs/FROZEN_PROTOCOL.json'))))
    # Thread-local sleep inhibition is restored at process exit; no persistent power-setting change.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    started=time.perf_counter()
    try:
        model=initialize(seed,variant,p,device); initial=old.state_sha256(model)
        hp=p['hyperparameters']; optimizer=torch.optim.AdamW(model.parameters(),lr=hp['learning_rate'],weight_decay=hp['weight_decay'])
        ty=(z['train_y']/sc['target_scale_V']).astype(np.float32); vy=z['val_y']/sc['target_scale_V']
        loader=DataLoader(TensorDataset(torch.from_numpy(z['tx']),torch.from_numpy(ty)),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
        best=math.inf; best_epoch=0; stale=0; history=[]
        cp=f'checkpoints/{run}_best.pt'
        for epoch in range(1,hp['max_epochs']+1):
            model.train(); losses=[]
            for xb,yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.mse_loss(model.predict_delta(xb.to(device)),yb.to(device))
                req(bool(torch.isfinite(loss)),'NONFINITE_LOSS'); loss.backward()
                norm=nn.utils.clip_grad_norm_(model.parameters(),hp['gradient_clip_norm']); req(bool(torch.isfinite(norm)),'NONFINITE_GRADIENT')
                optimizer.step(); losses.append(float(loss.detach().cpu()))
            pred=predict(model,z['vx'],device); rmse=float(np.sqrt(np.mean((pred-vy)**2)))
            req(math.isfinite(rmse),'NONFINITE_VAL')
            if rmse<best-1e-12:
                best=rmse; best_epoch=epoch; stale=0
                save_torch(cp,dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},seed=seed,variant=variant,
                  best_epoch=epoch,validation_standardized_prediction=torch.from_numpy(pred.copy()),initialization_sha256=initial,
                  protocol_sha256=sha(path('configs/FROZEN_PROTOCOL.json')),target_scale_V=sc['target_scale_V']))
            else: stale+=1
            history.append(dict(epoch=epoch,train_standardized_MSE=float(np.mean(losses)),validation_standardized_RMSE=rmse,validation_RMSE_mV=rmse*sc['target_scale_V']*1000,best_epoch=best_epoch,stale_epochs=stale,elapsed_s=time.perf_counter()-started))
            csv(f'outputs/training_logs/{run}.csv',history)
            js(state,dict(status='RUNNING',seed=seed,variant=variant,epoch=epoch,best_epoch=best_epoch,stale=stale,elapsed_s=time.perf_counter()-started))
            if epoch==1 or epoch%10==0: print(run,'epoch',epoch,'best_mV',best*sc['target_scale_V']*1000,'elapsed_s',round(time.perf_counter()-started),flush=True)
            if stale>=hp['patience']: break
        ck=torch.load(path(cp),map_location='cpu',weights_only=True); model.load_state_dict(ck['state_dict'])
        pred=predict(model,z['vx'],device); req(np.array_equal(pred,ck['validation_standardized_prediction'].numpy()),'BEST_RELOAD_PREDICTION')
        physical=pred*sc['target_scale_V']
        frame=pd.DataFrame(dict(origin_row=np.repeat(z['val_origins'],24),target_row=z['val_targets'].ravel(),horizon_step=np.tile(np.arange(1,25),205),
            y_true_delta_V=z['val_y'].ravel(),y_pred_delta_V=physical.ravel(),origin_age_h=np.repeat(z['val_origin_age_h'],24),target_age_h=z['val_target_age_h'].ravel(),
            origin_raw_voltage_V=np.repeat(z['val_origin_raw_voltage_V'],24)))
        csv(f'outputs/predictions/{run}.csv',frame)
        js(state,dict(status='COMPLETED',seed=seed,variant=variant,best_epoch=best_epoch,stop_epoch=epoch,elapsed_s=time.perf_counter()-started,
            best_RMSE_mV=best*sc['target_scale_V']*1000,checkpoint=cp,checkpoint_sha256=sha(path(cp)),initialization_sha256=initial,best_reload_exact=True,finished_utc=now()))
        print('COMPLETED',run,'best',best*sc['target_scale_V']*1000,flush=True)
    except BaseException:
        text(f'outputs/failures/{run}.txt',traceback.format_exc()); js(state,dict(status='FAILED',seed=seed,variant=variant,error=traceback.format_exc(),finished_utc=now())); raise
    finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def run():
    p=get_protocol()
    for item in p['run_order']:
        run=f"{item['variant']}_seed{item['seed']}"; ledger=path(f'outputs/ledger/{run}.json')
        if ledger.exists():
            req(readj(f'outputs/ledger/{run}.json')['status']=='COMPLETED','UNFINISHED_RUN_REQUIRES_AUDIT:'+run)
            print('PRESERVED_COMPLETED',run,flush=True); continue
        subprocess.run([sys.executable,'-B',str(Path(__file__)),'fit','--seed',str(item['seed']),'--variant',item['variant']],check=True,cwd=P)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run','fit']); parser.add_argument('--seed',type=int,choices=SEEDS); parser.add_argument('--variant',choices=['A1','B1'])
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='run': run()
    else: fit(args.seed,args.variant)

"""Small Train-internal Optuna refinement of the ORIGINAL F2, not RegimePlus.

prepare -> search (6 trials) -> final (3 fixed seeds) -> report -> verify.
Final Validation never selects hyperparameters, epochs or seeds.
"""
from __future__ import annotations
import os
os.environ.setdefault('PYTHONUTF8','1'); os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse, ctypes, hashlib, importlib.util, json, math, random, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import optuna

P=Path(__file__).resolve().parents[1]; R=P.parent; REPO=R.parent
BASE='9aa7ac0c04e1edacdba8a452cb881d5a74aad042'
spec=importlib.util.spec_from_file_location('readonly_original_f2',R/'RealHyFC_F2_Main.py')
old=importlib.util.module_from_spec(spec); sys.modules[spec.name]=old; spec.loader.exec_module(old)
D=old.FROZEN_DATA; SEEDS=[42,52,62]
def req(c,m):
    if not c: raise RuntimeError(m)
def sha(p): return old.sha256(Path(p))
def path(s):
    f=(P/s).resolve(); req(f.is_relative_to(P.resolve()),'WRITE_OUTSIDE_PHASE');f.parent.mkdir(parents=True,exist_ok=True);return f
def text(s,v):
    f=path(s); t=f.with_suffix(f.suffix+'.writing');t.write_text(v,encoding='utf-8',newline='\n')
    for delay in [.05,.1,.2,.5,1,2,None]:
        try: os.replace(t,f);return
        except PermissionError:
            if delay is None: raise
            time.sleep(delay)
def js(s,v):text(s,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv(s,v):text(s,pd.DataFrame(v).to_csv(index=False,float_format='%.17g',lineterminator='\n'))
def read(s):return json.loads(path(s).read_text(encoding='utf-8'))
def rd(s):return pd.read_csv(path(s),float_precision='round_trip')
def now():return datetime.now(timezone.utc).isoformat()
def save(s,v):
    f=path(s);t=f.with_suffix(f.suffix+'.writing');torch.save(v,t)
    for delay in [.05,.1,.2,.5,1,2,None]:
        try:os.replace(t,f);return
        except PermissionError:
            if delay is None:raise
            time.sleep(delay)

def train_cache():
    req(sha(D/'FROZEN_H_DATA.npz')=='cbb212134eb664f1a43d4851f20686fe575c06eb6b2b0b893aed0f166fcef83e','FROZEN_CACHE_CHANGED')
    with np.load(D/'FROZEN_H_DATA.npz',allow_pickle=False) as z:
        return {k:z[k] for k in ['train_x_raw','train_y','train_origins','train_targets','tx']}
def validation_cache():
    req(path('configs/SELECTED_CONFIGURATION.json').exists(),'VALIDATION_FORBIDDEN_BEFORE_SELECTION_FREEZE')
    with np.load(D/'FROZEN_H_DATA.npz',allow_pickle=False) as z:
        return {k:z[k] for k in ['vx','val_y','val_origins','val_targets','val_origin_raw_voltage_V','val_target_raw_voltage_V']}
def preprocess(raw,mean,scale):
    x=raw.copy()
    for i in range(1,48):x[:,i,0]=.3*x[:,i,0]+.7*x[:,i-1,0]
    x=((x-mean)/scale).astype(np.float32);x[:,:,[1,2,5,12]]=0;return x
def inner(z):
    t=z['train_targets'];o=z['train_origins']
    return dict(FIT=np.flatnonzero(t.max(1)<1472),EARLY_STOP=np.flatnonzero((o>=1472)&(t.max(1)<1787)),TUNE=np.flatnonzero(o>=1787))
def protocol():
    p=read('configs/PROTOCOL.json')
    req(sha(path('configs/PROTOCOL.json'))==path('configs/PROTOCOL.sha256').read_text().strip(),'PROTOCOL_CHANGED')
    req(sha(Path(__file__))==p['script_sha256'],'TRAINING_CODE_CHANGED')
    req(sha(R/'RealHyFC_F2_Main.py')==p['model_source_sha256'],'HISTORICAL_SOURCE_CHANGED')
    return p
def initialize(seed,hp,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    m=old.RegimeAwareDualPathTFT(hidden_size=64,hidden_continuous_size=32,attention_head_size=4,lstm_layers=1,dropout=hp['dropout']).to(device)
    req(sum(p.numel() for p in m.parameters())==332436,'ORIGINAL_F2_STRUCTURE_CHANGED')
    return m
def prediction(m,x,batch=32):return old.predict_standardized(m,x,batch,torch.device('cuda'))
def metrics(y,pred):
    e=pred.astype(float)-y.astype(float);y=y.astype(float)
    return dict(RMSE_mV=float(np.sqrt(np.mean(e**2))*1000),MAE_mV=float(np.mean(abs(e))*1000),R2_deltaV=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)))

def prepare():
    req(not path('configs/PROTOCOL.json').exists(),'ALREADY_FROZEN')
    req(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE,'HEAD_CHANGED')
    runtime=old.configure_environment('cuda');z=train_cache();parts=inner(z)
    req({k:len(v) for k,v in parts.items()}==dict(FIT=1000,EARLY_STOP=136,TUNE=220),'INNER_SPLIT_IDENTITY')
    fit=z['train_x_raw'][parts['FIT']].astype(float).reshape(-1,14)
    mean=fit.mean(0);scale=fit.std(0);scale=np.where(scale>1e-8,scale,1.)
    rows=[]
    for name,ix in parts.items():
        for i in ix:rows.append(dict(role=name,index_in_frozen_train=int(i),origin_row=int(z['train_origins'][i]),first_target_row=int(z['train_targets'][i,0]),last_target_row=int(z['train_targets'][i,-1])))
    for a,b in [('FIT','EARLY_STOP'),('EARLY_STOP','TUNE')]:
        req(z['train_targets'][parts[a]].max()<z['train_origins'][parts[b]].min(),'INNER_LABEL_TIME_LEAK')
        req(not np.intersect1d(z['train_targets'][parts[a]],z['train_targets'][parts[b]]).size,'INNER_TARGET_OVERLAP')
    base=old.load_historical_hyperparameters()['parameters']
    p=dict(base_commit=BASE,frozen_utc=now(),model='original F2, 332436 parameters, no RegimePlus',history=48,horizon=24,raw_slots=14,target='raw future delta V',Top10=[0,3,13,11,4,9,6,7,10,8],EWMA_alpha=.3,
      inner_rows=dict(FIT=[0,1472],EARLY_STOP=[1472,1787],TUNE=[1787,2103]),inner_counts={k:len(v) for k,v in parts.items()},
      inner_scaler='fitted only unfiltered FIT histories before per-window causal EWMA',
      budget=dict(trials=6,search_seed=42,sampler_seed=20261018,max_epochs=450,patience=40,pruning=False,parallel_jobs=1),
      search_space=dict(learning_rate=dict(low=.0003,high=.003,log=True),dropout=[.05,.1,.2],weight_decay=dict(low=1e-7,high=.001,log=True),batch_size=[16,32,64]),
      original_hyperparameters=base,objective='TUNE physical delta-V RMSE after FIT training and EARLY_STOP checkpoint selection; MAE exact-RMSE tie-break only',
      selection_rule='smallest TUNE RMSE, then TUNE MAE, then trial number; include original configuration as trial 0',
      final=dict(seeds=SEEDS,training_windows=1380,validation_windows=205,validation_queries=4920,
       epochs='selected trial best EARLY_STOP epoch, frozen before outer Validation access',scaler='original full-Train-only frozen scaler',
       all_full_train_samples=True,outer_validation_early_stopping=False,outer_validation_selects_nothing=True),
      training=dict(loss='MSE',optimizer='AdamW',gradient_clip_norm=1.,hidden_size=64,hidden_continuous_size=32,attention_heads=4,lstm_layers=1),
      plot_rule='middle of 205 legal Validation origins; show all three seeds, actual raw voltage, frozen baseline and refined F2, each a single 24-step forecast',
      runtime=runtime,model_source_sha256=sha(R/'RealHyFC_F2_Main.py'),script_sha256=sha(Path(__file__)),
      data_cache_sha256=sha(D/'FROZEN_H_DATA.npz'),frozen_scaler_sha256=sha(D/'SCALER.json'),
      limitations=['reused Development Validation, not independent Test','historical baseline used Validation early stopping; refined uses Train-internal frozen epochs','three seeds are fixed, not a new seed screen'],
      stop='exactly six trials and three final fits; preserve technical failures, no automatic restart or expanded budget',scientific_success='mean RMSE and MAE both improve vs original; R2 reported on same deltaV labels')
    csv('outputs/INNER_WINDOW_SPLIT.csv',rows);js('configs/INNER_SCALER.json',dict(mean=mean.tolist(),scale=scale.tolist(),target_scale_V=float(scale[0])))
    js('configs/PROTOCOL.json',p);text('configs/PROTOCOL.sha256',sha(path('configs/PROTOCOL.json'))+'\n')
    js('outputs/PRETRAIN_CHECK.json',dict(status='PASS',Train_internal_three_way=True,disjoint_target_rows=True,labels_before_next_origins=True,outer_Validation_not_loaded=True,model_parameters=332436))
    print('FROZEN',p['inner_counts'],flush=True)

def fit(run,seed,hp,tx,ty,scale,epochs,ex=None,ey=None):
    """Fixed epochs when ex=None; otherwise ES only on the specified inner block."""
    state=f'outputs/ledger/{run}.json';req(not path(state).exists(),'RUN_ALREADY_STARTED:'+run)
    js(state,dict(status='RUNNING',seed=seed,hyperparameters=hp,started_utc=now()))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    started=time.perf_counter();m=initialize(seed,hp,torch.device('cuda'));initial=old.state_sha256(m)
    optimizer=torch.optim.AdamW(m.parameters(),lr=hp['learning_rate'],weight_decay=hp['weight_decay'])
    loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy((ty/scale).astype(np.float32))),batch_size=hp['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    best=math.inf;best_epoch=0;stale=0;history=[];checkpoint=f'checkpoints/{run}.pt'
    try:
        for epoch in range(1,epochs+1):
            m.train();losses=[]
            for x,y in loader:
                optimizer.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(m.predict_delta(x.cuda()),y.cuda())
                req(bool(torch.isfinite(loss)),'NONFINITE_LOSS');loss.backward()
                norm=nn.utils.clip_grad_norm_(m.parameters(),1.);req(bool(torch.isfinite(norm)),'NONFINITE_GRADIENT')
                optimizer.step();losses.append(float(loss.detach().cpu()))
            improved=False;score=None
            if ex is not None:
                pred=prediction(m,ex);score=float(np.sqrt(np.mean((pred-ey/scale)**2)))
                if score<best-1e-12:best=score;best_epoch=epoch;stale=0;improved=True
                else:stale+=1
            elif epoch==epochs:best_epoch=epoch;improved=True
            if improved:
                save(checkpoint,dict(state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},seed=seed,model='F2',hyperparameters=hp,
                    epoch=epoch,target_scale_V=scale,initialization_sha256=initial,protocol_sha256=sha(path('configs/PROTOCOL.json')),
                    early_stop_prediction=torch.from_numpy(pred.copy()) if ex is not None else torch.empty(0)))
            history.append(dict(epoch=epoch,train_standardized_MSE=float(np.mean(losses)),early_stop_standardized_RMSE=score,best_epoch=best_epoch,stale_epochs=stale,elapsed_s=time.perf_counter()-started))
            csv(f'outputs/logs/{run}.csv',history)
            js(state,dict(status='RUNNING',seed=seed,epoch=epoch,best_epoch=best_epoch,stale=stale,elapsed_s=time.perf_counter()-started))
            if epoch==1 or epoch%25==0:print(run,'epoch',epoch,'best_inner_mV',None if ex is None else best*scale*1000,flush=True)
            if ex is not None and stale>=40:break
        ck=torch.load(path(checkpoint),map_location='cpu',weights_only=True);m.load_state_dict(ck['state_dict'])
        if ex is not None:req(np.array_equal(prediction(m,ex),ck['early_stop_prediction'].numpy()),'INNER_BEST_RELOAD')
        js(state,dict(status='COMPLETED',seed=seed,best_epoch=best_epoch,stop_epoch=epoch,elapsed_s=time.perf_counter()-started,
            checkpoint=checkpoint,checkpoint_sha256=sha(path(checkpoint)),hyperparameters=hp,protocol_sha256=sha(path('configs/PROTOCOL.json'))))
        return m,best_epoch
    except BaseException:
        text(f'outputs/failures/{run}.txt',traceback.format_exc());js(state,dict(status='FAILED',error=traceback.format_exc()));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def search():
    p=protocol();old.configure_environment('cuda');z=train_cache();parts=inner(z);sc=read('configs/INNER_SCALER.json')
    tx=preprocess(z['train_x_raw'],np.array(sc['mean']),np.array(sc['scale']));ys=z['train_y'];scale=sc['target_scale_V']
    db=path('outputs/study.sqlite3');req(not db.exists(),'STUDY_ALREADY_EXISTS_NO_AUTOMATIC_RESTART')
    study=optuna.create_study(study_name='original_F2_train_internal',storage='sqlite:///'+db.as_posix(),direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=p['budget']['sampler_seed'],n_startup_trials=3),pruner=optuna.pruners.NopPruner())
    original=p['original_hyperparameters'];study.enqueue_trial({k:original[k] for k in ['learning_rate','dropout','weight_decay','batch_size']})
    def objective(trial):
        hp=dict(learning_rate=trial.suggest_float('learning_rate',.0003,.003,log=True),dropout=trial.suggest_categorical('dropout',[.05,.1,.2]),
            weight_decay=trial.suggest_float('weight_decay',1e-7,.001,log=True),batch_size=trial.suggest_categorical('batch_size',[16,32,64]))
        m,epoch=fit(f'trial{trial.number:02d}',42,hp,tx[parts['FIT']],ys[parts['FIT']],scale,450,tx[parts['EARLY_STOP']],ys[parts['EARLY_STOP']])
        pred=prediction(m,tx[parts['TUNE']])*scale;values=metrics(ys[parts['TUNE']],pred)
        trial.set_user_attr('best_epoch',epoch);trial.set_user_attr('MAE_mV',values['MAE_mV']);trial.set_user_attr('R2_deltaV',values['R2_deltaV'])
        np.savez_compressed(path(f'outputs/trials/trial{trial.number:02d}_tune.npz'),prediction=pred,truth=ys[parts['TUNE']],origins=z['train_origins'][parts['TUNE']],targets=z['train_targets'][parts['TUNE']])
        js(f'outputs/trials/trial{trial.number:02d}.json',dict(number=trial.number,parameters=hp,epochs=epoch,**values))
        del m;torch.cuda.empty_cache();return values['RMSE_mV']
    study.optimize(objective,n_trials=6,n_jobs=1,catch=())
    req(len(study.trials)==6 and all(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials),'INCOMPLETE_SEARCH')
    best=min(study.trials,key=lambda t:(t.value,t.user_attrs['MAE_mV'],t.number))
    selected=dict(frozen_utc=now(),trial_number=best.number,hyperparameters=best.params,final_epochs=best.user_attrs['best_epoch'],inner_TUNE_RMSE_mV=best.value,
        selection_used_only_Train=True,outer_Validation_access_before_selection=False,protocol_sha256=sha(path('configs/PROTOCOL.json')))
    js('configs/SELECTED_CONFIGURATION.json',selected);text('configs/SELECTED_CONFIGURATION.sha256',sha(path('configs/SELECTED_CONFIGURATION.json'))+'\n')
    csv('outputs/OPTUNA_TRIALS.csv',[dict(number=t.number,**t.params,RMSE_mV=t.value,MAE_mV=t.user_attrs['MAE_mV'],R2_deltaV=t.user_attrs['R2_deltaV'],best_epoch=t.user_attrs['best_epoch']) for t in study.trials])
    print('SELECTION_FROZEN',selected,flush=True)

def selected():
    req(sha(path('configs/SELECTED_CONFIGURATION.json'))==path('configs/SELECTED_CONFIGURATION.sha256').read_text().strip(),'SELECTED_CONFIGURATION_CHANGED')
    return read('configs/SELECTED_CONFIGURATION.json')
def frame(z,pred):
    return pd.DataFrame(dict(origin_row=np.repeat(z['val_origins'],24),target_row=z['val_targets'].ravel(),horizon_step=np.tile(np.arange(1,25),205),
        y_true_delta_V=z['val_y'].ravel(),y_pred_delta_V=pred.ravel(),origin_raw_voltage_V=np.repeat(z['val_origin_raw_voltage_V'],24),target_raw_voltage_V=z['val_target_raw_voltage_V'].ravel()))
def final():
    protocol();best=selected();old.configure_environment('cuda');z=train_cache();v=validation_cache();sc=json.loads((D/'SCALER.json').read_text())
    req(np.array_equal(preprocess(z['train_x_raw'],np.array(sc['mean']),np.array(sc['scale'])),z['tx']),'FULL_TRAIN_PREPROCESS_CHANGED')
    for seed in SEEDS:
        run=f'refined_seed{seed}'
        if path(f'outputs/ledger/{run}.json').exists():
            req(read(f'outputs/ledger/{run}.json')['status']=='COMPLETED' and path(f'outputs/predictions/{run}.csv').exists(),'UNFINISHED_FINAL_FIT_REQUIRES_REVIEW')
            continue
        m,epoch=fit(run,seed,best['hyperparameters'],z['tx'],z['train_y'],sc['target_scale_V'],best['final_epochs'])
        pred=prediction(m,v['vx'])*sc['target_scale_V'];csv(f'outputs/predictions/{run}.csv',frame(v,pred));del m;torch.cuda.empty_cache()
    for seed in SEEDS:
        hp=old.load_historical_hyperparameters()['parameters'];m=old.initialize_f2(seed,hp,torch.device('cuda'))
        ckfile=old.HISTORY/f'outputs/checkpoints/H-F2-Seed{seed}_best.pt';ck=torch.load(ckfile,map_location='cpu',weights_only=True);m.load_state_dict(ck['state_dict'])
        f=frame(v,prediction(m,v['vx'])*sc['target_scale_V'])
        source=pd.read_csv(old.HISTORY/f'outputs/validation_predictions/H-F2-Seed{seed}.csv',float_precision='round_trip')
        for key in ['origin_row','target_row','horizon_step','y_true_delta_V','y_pred_delta_V']:req(np.array_equal(f[key],source[key]),'BASELINE_EXACT_REPLAY:'+key)
        csv(f'outputs/predictions/original_seed{seed}.csv',f);del m
    print('THREE_SEED_FINAL_COMPLETED',flush=True)

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=protocol();best=selected();v=validation_cache();rows=[]
    for name in ['original','refined']:
        for seed in SEEDS:
            f=rd(f'outputs/predictions/{name}_seed{seed}.csv')
            for key,value in [('origin_row',np.repeat(v['val_origins'],24)),('target_row',v['val_targets'].ravel()),('horizon_step',np.tile(np.arange(1,25),205)),('y_true_delta_V',v['val_y'].ravel())]:req(np.array_equal(f[key],value),'FINAL_KEY_OR_TRUTH_CHANGED')
            rows.append(dict(model=name,seed=seed,**metrics(f.y_true_delta_V.to_numpy(),f.y_pred_delta_V.to_numpy())))
    details=pd.DataFrame(rows);csv('outputs/PER_SEED_METRICS.csv',details)
    summary=details.groupby('model')[['RMSE_mV','MAE_mV','R2_deltaV']].mean().reset_index();csv('SUMMARY.csv',summary)
    a=summary.set_index('model');gain=(1-a.loc['refined','RMSE_mV']/a.loc['original','RMSE_mV'])*100;magain=(1-a.loc['refined','MAE_mV']/a.loc['original','MAE_mV'])*100
    js('outputs/RESULT.json',dict(status='IMPROVED' if gain>0 and magain>0 else 'NO_JOINT_RMSE_MAE_IMPROVEMENT',RMSE_improvement_pct=float(gain),MAE_improvement_pct=float(magain),seeds=SEEDS,queries_per_seed=4920,epochs=best['final_epochs']))
    origin=int(v['val_origins'][len(v['val_origins'])//2]);plotdata=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    for ax,seed in zip(axes,SEEDS):
        for name,color in [('original','#6B7280'),('refined','#2563EB')]:
            f=rd(f'outputs/predictions/{name}_seed{seed}.csv');f=f.loc[f.origin_row==origin].copy();f['predicted_voltage_V']=f.origin_raw_voltage_V+f.y_pred_delta_V;f['model']=name;f['seed']=seed;plotdata.append(f)
            ax.plot(f.horizon_step,f.predicted_voltage_V,color=color,label='Frozen F2' if name=='original' else 'Refined F2')
        ax.plot(f.horizon_step,f.target_raw_voltage_V,'k--',label='Observed');ax.set(title=f'Seed {seed} | origin {origin}',xlabel='Step (0.5 operating h)');ax.set_xticks([1,6,12,18,24])
    axes[0].set_ylabel('Raw absolute voltage (V)');axes[-1].legend(fontsize=8);fig.suptitle('Development Validation | one preselected origin, 24 steps per model');fig.tight_layout()
    fig.savefig(path('figures/VOLTAGE_COMPARISON.png'),dpi=300,bbox_inches='tight');fig.savefig(path('figures/VOLTAGE_COMPARISON.pdf'),bbox_inches='tight');plt.close(fig);csv('outputs/PLOT_DATA.csv',pd.concat(plotdata))
    text('README.md',f"# F2专属小规模超参数优化\n\n基准`{BASE}`。原始F2结构332436参数、48×14→24 ΔV、Top10与因果EWMA不变。仅搜索学习率、Dropout、weight_decay、batch_size；6个Optuna试验（含原配置），不扩大预算。\n\nTrain内部按原始行[0,1472)、[1472,1787)、[1787,2103)分别用于拟合、早停与选参，合法窗口1000/136/220。scaler仅拟合第一段。原冻结Top10来自历史Train研究，不宣称内部调参集是全新独立数据。\n\n选择trial {best['trial_number']}，参数：`{best['hyperparameters']}`；最终训练轮数{best['final_epochs']}由该trial内部早停最佳轮数冻结。三个最终模型从头使用全部1380个原Train窗口及原Train-only scaler，固定轮数后只评价205个原Validation起点、4920条目标。Validation不选参数、不选最终epoch、不筛选种子。原基准曾使用Validation早停，二者选择历史不同，不能声称这是两套完全相同的epoch选择机制。\n\n三seed等权平均：\n\n{summary.to_csv(index=False)}\n\nRMSE改善{gain:.4f}%，MAE改善{magain:.4f}%（正值为改善）。R²按原冻结ΔV标签计算；绝对电压图用未EWMA原始起点电压恢复，未拼接重叠窗口。状态见outputs/RESULT.json。\n\n仅Development探索性评价，不读取Assessment、Late-life或独立Test来选参/评价。历史checkpoint、预测、代码和论文不替换。图为预定中间起点，非挑选误差最低样本。没有新增门控特征、启停诊断或分工况报告。\n\n复现：\n```powershell\npython -B scripts/refine.py prepare\npython -B scripts/refine.py search\npython -B scripts/refine.py final\npython -B scripts/refine.py report\npython -B scripts/verify.py\n```\nprepare/search拒绝覆盖既有冻结方案或重跑已存在study；final保留已完成实例。环境沿用历史CUDA运行协议。配置、每次训练日志、权重、预测、Optuna完整记录均在本目录。\n")
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','search','final','report']);a=ap.parse_args()
    {'prepare':prepare,'search':search,'final':final,'report':report}[a.action]()

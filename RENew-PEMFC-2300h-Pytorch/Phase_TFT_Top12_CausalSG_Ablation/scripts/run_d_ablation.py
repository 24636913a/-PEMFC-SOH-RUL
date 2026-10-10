"""Phase D: fixed Top10/Top12 mask and causal SG(7,2) controlled ablation.

Only the first 3007 rows of the mixed causal table are ever opened.  The four
conditions use the same scaler fitted to *unfiltered* legal Train histories and
the same target scale.  This deliberately removes the C-series filter/scaler
coupling while preserving the Tensor shape and Canonical TFT body.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, random, sys, time
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, yaml
from torch.utils.data import DataLoader, TensorDataset

PHASE=Path(__file__).resolve().parents[1]; PROJECT=PHASE.parent
C_PHASE=PROJECT/'Phase_TFT_SingleSplit_Feature_Filter_Ablation'; T1R=PROJECT/'Phase_TFT_Hyperparameter_Optimization_Restart'/'outputs'/'T1R_SELECTED_CONFIGURATIONS.json'
PROTOCOL=PHASE/'configs'/'PREREGISTERED_D_ABLATION.yaml'; PROTOCOL_SHA=PHASE/'configs'/'PREREGISTERED_D_ABLATION.sha256'
OUT=PHASE/'outputs'; FIG=PHASE/'figures'; REPORT=PHASE/'reports'
sys.path.insert(0,str(C_PHASE/'scripts')); import run_c_ablation as c

def write_json(path:Path,obj:Any): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load_protocol():
    sha=digest(PROTOCOL)
    if sha!=PROTOCOL_SHA.read_text(encoding='utf-8').split()[0]: raise RuntimeError('D_PROTOCOL_SHA_MISMATCH')
    return yaml.safe_load(PROTOCOL.read_text(encoding='utf-8')),sha
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def one_sided_sg(x:np.ndarray)->np.ndarray: return c.one_sided_sg(x)
def causal_sg_audit()->dict[str,Any]:
    base=np.linspace(-1,1,48,dtype=np.float32).reshape(1,48,1); base=np.pad(base,((0,0),(0,0),(0,13)))
    mutated=base.copy(); mutated[:,19:,0]+=3.0
    a,b=one_sided_sg(base),one_sided_sg(mutated)
    return {'pass':bool(np.allclose(a[:,:19,0],b[:,:19,0],atol=1e-7) and np.isfinite(a).all()),'future_mutation_start_index':19,'short_prefix_degree_rule':'min(2,n-1)','test_accessed':False}

def masks(train,features):
    score=c.spearman_scores(train,features).copy(); score['rank']=np.arange(1,len(score)+1)
    def choose(k):
        selected=score.head(k).raw_feature_index.astype(int).tolist(); v=features.index('cell_voltage_mean')
        if v not in selected: selected[-1]=v
        return sorted(set(selected))
    top10,top12=choose(10),choose(12)
    if not set(top10).issubset(top12): raise RuntimeError('TOP10_NOT_TOP12_SUBSET')
    score['selected_top10']=score.raw_feature_index.isin(top10); score['selected_top12']=score.raw_feature_index.isin(top12)
    return score,top10,top12

class CommonScaler:
    def fit(self,x):
        flat=x.astype(np.float64).reshape(-1,x.shape[-1]); self.mean=flat.mean(0); self.scale=flat.std(0); self.scale=np.where(self.scale>1e-8,self.scale,1.0); self.samples=int(flat.shape[0]); return self
    def transform(self,x): return ((x-self.mean)/self.scale).astype(np.float32)
    @property
    def target_scale(self): return float(self.scale[0])

def prep(x,scaler,mask,sg):
    raw=one_sided_sg(x) if sg else x.copy(); z=scaler.transform(raw); excluded=[i for i in range(14) if i not in mask]
    z[:,:,excluded]=0.; return z

def make_model(params,device):
    mod=c.load_model_module(); return c.make_model(mod,params,device)
def predict(model,x,batch,device): return c.predict(model,x,batch,device)
def fit(params,train,val,scaler,mask,sg,checkpoint,device):
    set_seed(); tx,vx=prep(train.x,scaler,mask,sg),prep(val.x,scaler,mask,sg); ty=train.y/scaler.target_scale; vy=val.y/scaler.target_scale
    model=make_model(params,device); opt=torch.optim.AdamW(model.parameters(),lr=float(params['learning_rate']),weight_decay=float(params['weight_decay']))
    loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty.astype(np.float32))),batch_size=int(params['batch_size']),shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0)
    best,best_epoch,stale,state=math.inf,0,0,None; history=[]; started=time.perf_counter()
    for epoch in range(1,451):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(model.predict_delta(xb),yb)
            if not torch.isfinite(loss): raise RuntimeError('NONFINITE_LOSS')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
        score=float(np.sqrt(np.mean((predict(model,vx,int(params['batch_size']),device)-vy)**2))); history.append({'epoch':epoch,'train_standardized_mse':float(np.mean(losses)),'validation_standardized_rmse':score})
        if score<best-1e-12: best,best_epoch,stale,state=score,epoch,0,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=40: break
    model.load_state_dict(state); checkpoint.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'best_epoch':best_epoch,'common_scaler_target_scale':scaler.target_scale},checkpoint)
    return model,{'best_epoch':best_epoch,'validation_standardized_RMSE':best,'validation_RMSE_mV':best*scaler.target_scale*1000,'training_time_s':time.perf_counter()-started,'parameter_count':int(sum(p.numel() for p in model.parameters() if p.requires_grad)),'history':history}

def frames(label,win,pred,sha):
    rows=[]
    for q,o in enumerate(win.origins):
        for h in range(24): rows.append({'experiment_group':label,'seed':42,'origin_row':int(o),'target_row':int(win.targets[q,h]),'horizon_step':h+1,'origin_age_h':float(win.ages[q]),'y_true_delta_V':float(win.y[q,h]),'y_pred_delta_V':float(pred[q,h]),'protocol_sha256':sha})
    return pd.DataFrame(rows)
def metric_cache(path):
    f=pd.read_csv(path); o=sorted(f.origin_row.unique()); y=[];p=[];t=[]
    for k in o:
        q=f[f.origin_row.eq(k)].sort_values('horizon_step'); y.append(q.y_true_delta_V.to_numpy());p.append(q.y_pred_delta_V.to_numpy());t.append(q.target_row.to_numpy())
    return c.metric(np.asarray(y),np.asarray(p),np.asarray(t))

def pretrain_audit(write=False):
    cfg,sha=load_protocol(); dev=c.load_development(cfg); ranges={'train':(0,2103),'validation':(2103,2404),'assessment':(2404,3005)}
    w={n:c.build_windows(dev,cfg,*r) for n,r in ranges.items()}; c.assert_isolation(w['train'],w['validation'],w['assessment'])
    if {n:len(x.origins) for n,x in w.items()}!=cfg['split']['expected_legal_windows']: raise RuntimeError('D_WINDOW_COMPATIBILITY_MISMATCH')
    scores,t10,t12=masks(w['train'],cfg['task']['raw_f2_feature_order']); scaler=CommonScaler().fit(w['train'].x); params=json.loads(T1R.read_text(encoding='utf-8'))['folds']['fold_2']['best_params']; device=torch.device('cpu'); counts=[int(sum(p.numel() for p in make_model(params,device).parameters() if p.requires_grad)) for _ in range(4)]
    audit={'pass':True,'protocol_sha256':sha,'development_rows':len(dev),'read_nrows':3007,'full_causal_file_hash_computed':False,'test_accessed':False,'legal_windows':{n:len(x.origins) for n,x in w.items()},'target_isolation':True,'top10_subset_top12':True,'causal_sg':causal_sg_audit(),'common_scaler_train_windows':len(w['train'].origins),'common_scaler_flat_samples':scaler.samples,'common_target_scale_V':scaler.target_scale,'parameter_counts':counts,'parameter_counts_equal':len(set(counts))==1}
    if not audit['causal_sg']['pass'] or not audit['parameter_counts_equal']: raise RuntimeError('D_PRETRAIN_AUDIT_FAILURE')
    if write: write_json(OUT/'D_PRETRAIN_AUDIT.json',audit)
    return cfg,sha,dev,w,scores,t10,t12,scaler,audit

def independent_metric_audit(write=False):
    cfg,sha=load_protocol(); allrows=[]
    for label in ('D0','D1','D2','D3'):
        path=OUT/f'{label}_predictions_seed42.csv'; f=pd.read_csv(path)
        if len(f)!=378*24 or not f.target_row.between(2404,3004).all(): raise RuntimeError('D_PREDICTION_CACHE_CONTRACT_FAILURE')
        m=metric_cache(path); allrows.append({'experiment_group':label,**m,'rows':len(f),'protocol_sha256_match':bool((f.protocol_sha256==sha).all())})
    result={'pass':True,'assessment_target_rows':[2404,3004],'test_accessed':False,'results':allrows}
    if write: write_json(OUT/'D_INDEPENDENT_METRIC_AUDIT.json',result)
    return result

def figures(preds,metrics):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False}); keys=['D0','D1','D2','D3']; colors={'D0':'#2166ac','D1':'#1b7837','D2':'#b2182b','D3':'#762a83'}
    fig,axs=plt.subplots(1,4,figsize=(16,4)); cols=['RMSE_mV','MAE_mV','endpoint_RMSE_mV','P99_mV']; titles=['RMSE','MAE','12 h endpoint RMSE','P99 absolute error']
    for ax,col,title in zip(axs,cols,titles): ax.bar(keys,metrics.set_index('experiment_group').loc[keys,col],color=[colors[k] for k in keys],edgecolor='black');ax.set_title(title);ax.set_ylabel('mV')
    fig.tight_layout();fig.savefig(FIG/'Fig1_Overall_Metrics.png',dpi=220);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,9),sharex=True,sharey=True); lim=max(max(abs(f.y_true_delta_V).max(),abs(f.y_pred_delta_V).max()) for f in preds.values())*1000
    for ax,k in zip(axs.ravel(),keys):
        f=preds[k];ax.scatter(f.y_true_delta_V*1000,f.y_pred_delta_V*1000,s=3,alpha=.25,color=colors[k]);ax.plot([-lim,lim],[-lim,lim],'k--');ax.set_title(k);ax.set_xlabel('True ΔV (mV)');ax.set_ylabel('Predicted ΔV (mV)')
    fig.tight_layout();fig.savefig(FIG/'Fig2_Predicted_vs_True.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));
    for k in keys:
        f=preds[k].assign(a=lambda x:(x.y_pred_delta_V-x.y_true_delta_V).abs()*1000);v=f.groupby('target_row').a.mean();ax.plot(v.index,v.values,label=k,color=colors[k])
    ax.legend(ncol=4);ax.set(title='Chronological assessment error',xlabel='Target row',ylabel='Mean absolute error (mV)');fig.tight_layout();fig.savefig(FIG/'Fig3_Chronological_Error.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));
    for k in keys:
        f=preds[k].assign(se=lambda x:(x.y_pred_delta_V-x.y_true_delta_V)**2);v=np.sqrt(f.groupby('horizon_step').se.mean())*1000;ax.plot(v.index,v.values,marker='o',ms=3,label=k,color=colors[k])
    ax.legend(ncol=4);ax.set(title='Horizon-wise RMSE',xlabel='Forecast step (0.5 h)',ylabel='RMSE (mV)');fig.tight_layout();fig.savefig(FIG/'Fig4_Horizon_Error.png',dpi=220);plt.close(fig)
    base=metrics.set_index('experiment_group').loc['D0']; fig,ax=plt.subplots(figsize=(8,4)); effects=[]
    for k,txt in [('D1','Top12−Top10, raw'),('D2','SG−raw, Top10'),('D3','Top12+SG−Top10 raw')]: effects.append((txt,float(metrics.set_index('experiment_group').loc[k,'RMSE_mV']-base.RMSE_mV)))
    ax.bar([x[0] for x in effects],[x[1] for x in effects],color=['#1b7837','#b2182b','#762a83'],edgecolor='black');ax.axhline(0,color='black',lw=1);ax.set_ylabel('Δ RMSE vs D0 (mV)');ax.tick_params(axis='x',rotation=15);fig.tight_layout();fig.savefig(FIG/'Fig5_Feature_Filter_Effects.png',dpi=220);plt.close(fig)

def write_report(sha,scores,t10,t12,metrics,audit):
    x=metrics.set_index('experiment_group'); c1=pd.read_csv(C_PHASE/'outputs'/'C_SEED42_METRICS.csv').query("experiment_group=='C1'").iloc[0]
    status=[]
    if x.loc['D1','RMSE_mV']<x.loc['D0','RMSE_mV'] and x.loc['D3','RMSE_mV']>=x.loc['D1','RMSE_mV']: status=['TOP12_FEATURE_EXPANSION_EXPLORATORY_PROMISE','CAUSAL_SG_ADDITIONAL_GAIN_NOT_ESTABLISHED']
    elif x.loc['D3','RMSE_mV']<x.loc['D1','RMSE_mV'] and x.loc['D3','RMSE_mV']<x.loc['D2','RMSE_mV']: status=['TOP12_CAUSAL_SG_EXPLORATORY_PROMISE']
    elif x.loc['D3','RMSE_mV']>=min(x.loc['D1','RMSE_mV'],x.loc['D2','RMSE_mV']): status=['TOP12_SG_COMPLEMENTARITY_NOT_ESTABLISHED']
    else: status=['FEATURE_FILTER_EXPANSION_NO_CLEAR_GAIN']
    table='\n'.join(f"|{r.experiment_group}|{r.RMSE_mV:.4f}|{r.MAE_mV:.4f}|{r.R2:.4f}|{r.endpoint_RMSE_mV:.4f}|{r.P95_mV:.4f}|{r.P99_mV:.4f}|{r.target_time_equal_RMSE_mV:.4f}|" for _,r in metrics.iterrows())
    text=f'''# Phase D — Top12 特征扩展与因果 SG 消融实验

## 1. 实验目的

本轮在相同 RealHyFC Development 单次时间顺序 7:1:2 划分上，分离考察 Top10→Top12 特征掩码与单侧因果 SG(7,2) 的影响。它是已暴露 Assessment 上的 Seed42 探索性控制变量实验，不是独立盲测或新的 backbone 排名。

## 2. 数据与安全协议

仅通过 `pd.read_csv(nrows=3007)` 读取混合表，过滤后保留 3005 个 Development rows。Train/Validation/Assessment 分别为 [0,2103)、[2103,2404)、[2404,3005)，其合法窗口数为 {audit['legal_windows']}。所有目标集合隔离；未访问 Test、未来协变量或 Test 标签。

## 3. 模型、公共 scaler 与筛选

CanonicalStrictHistoryTFT 未改变，参数来自历史 T1-R Fold2。四组参数量均为 {audit['parameter_counts'][0]}。公共 scaler 仅拟合未滤波 Train histories，公共 ΔV scale 为 {audit['common_target_scale_V']:.8f} V；SG 只变换历史平均电压通道，之后仍使用该 scaler。

Top10：{', '.join(scores.loc[scores.selected_top10,'feature'])}

Top12：{', '.join(scores.loc[scores.selected_top12,'feature'])}

新增：{', '.join([f for f in scores.loc[scores.selected_top12,'feature'] if f not in set(scores.loc[scores.selected_top10,'feature'])])}。这些 Spearman 仅描述定义为 ΔV 的训练关联，不能解释为退化机理重要性。

## 4. 结果

|组别|RMSE mV|MAE mV|R²|12h endpoint RMSE mV|P95 mV|P99 mV|Target-equal RMSE mV|
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

## 5. 对照解释

- **D1 vs D0**：Top12 的 RMSE 变化为 {x.loc['D1','RMSE_mV']-x.loc['D0','RMSE_mV']:+.4f} mV；应同时结合 MAE、endpoint 与 P99 判断，而非仅凭 RMSE。
- **D2 vs D0**：在 Top10 下，SG 的 RMSE 变化为 {x.loc['D2','RMSE_mV']-x.loc['D0','RMSE_mV']:+.4f} mV。
- **D3 vs D1**：在 Top12 下，SG 的 RMSE 变化为 {x.loc['D3','RMSE_mV']-x.loc['D1','RMSE_mV']:+.4f} mV。
- **D3 vs D2**：在 SG 下，Top12 的 RMSE 变化为 {x.loc['D3','RMSE_mV']-x.loc['D2','RMSE_mV']:+.4f} mV。

## 6. 历史 C1 复现性

D0 RMSE 为 {x.loc['D0','RMSE_mV']:.6f} mV，历史 C1 为 {float(c1.RMSE_mV):.6f} mV；差值为 {x.loc['D0','RMSE_mV']-float(c1.RMSE_mV):+.6f} mV。D0 使用同一排序、窗口、原始 Train scaler、初始化与训练算法；任何差异仅作运行环境/数值确定性审计，不用于再训练。

## 7. 失败案例与局限

逐 target-row、horizon 与 Top10 极端误差均已保存。重叠窗口不被当作独立样本，未做显著性声称，也未删除 row 2952 或其他高误差事件。本轮只含一次时间切分和 Seed42；Assessment 已被先前查看，故不能视为独立泛化证明。短期 ΔV 精度也不等价于 SOH 或 RUL 精度。

## 8. 最终证据状态

`{'; '.join(status)}`。D0–D3 均已完成真实训练、独立缓存复算与因果 SG 审计。到此停止，不启动 Top13/Top14、SG 参数搜索、Optuna、Test 或其他模型。
'''
    (REPORT/'D_SERIES_EXPERIMENT_REPORT.md').write_text(text,encoding='utf-8')
    return status

def run():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    cfg,sha,dev,w,scores,t10,t12,scaler,audit=pretrain_audit(write=True); scores.to_csv(OUT/'D_FEATURE_SELECTION.csv',index=False)
    manifest={'protocol_sha256':sha,'data_access':{'read_nrows':3007,'development_rows':3005,'full_causal_file_hash_computed':False,'test_accessed':False},'segments':{n:{'rows':list(r),'start_h':float(dev.bin_start_ophrs2_h.iloc[r[0]]),'end_h':float(dev.bin_end_ophrs2_h.iloc[r[1]-1]),'legal_windows':len(w[n].origins)} for n,r in {'train':(0,2103),'validation':(2103,2404),'assessment':(2404,3005)}.items()},'target_sets_disjoint':True}; write_json(OUT/'D_SPLIT_MANIFEST.json',manifest)
    write_json(OUT/'D_COMMON_SCALER.json',{'mean':scaler.mean.tolist(),'scale':scaler.scale.tolist(),'fit_split':'unfiltered_train_legal_windows_only','flat_samples':scaler.samples,'common_target_scale_V':scaler.target_scale,'all_conditions_share_scaler':True,'test_accessed':False})
    params=json.loads(T1R.read_text(encoding='utf-8'))['folds']['fold_2']['best_params']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    specs={'D0':(t10,False),'D1':(t12,False),'D2':(t10,True),'D3':(t12,True)}; metrics=[];preds={}
    for label,(mask,sg) in specs.items():
        model,info=fit(params,w['train'],w['validation'],scaler,mask,sg,OUT/'checkpoints'/f"{label}_{'Top12' if len(mask)==12 else 'Top10'}_{'SG' if sg else 'Raw'}_seed42.pt",device)
        ax=prep(w['assessment'].x,scaler,mask,sg); pred=predict(model,ax,int(params['batch_size']),device)*scaler.target_scale; f=frames(label,w['assessment'],pred,sha); path=OUT/f'{label}_predictions_seed42.csv';f.to_csv(path,index=False);preds[label]=f
        m=metric_cache(path);m.update({'experiment_group':label,'seed':42,'best_epoch':info['best_epoch'],'training_time_s':info['training_time_s'],'validation_RMSE_mV':info['validation_RMSE_mV'],'parameter_count':info['parameter_count'],'features':'Top12' if len(mask)==12 else 'Top10','filter':'OneSidedSG_window_7_order_2' if sg else 'none'});metrics.append(m)
    met=pd.DataFrame(metrics)[['experiment_group','seed','RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV','target_time_equal_RMSE_mV','training_time_s','validation_RMSE_mV','best_epoch','parameter_count','features','filter']];met.to_csv(OUT/'D_SEED42_METRICS.csv',index=False)
    auditm=independent_metric_audit(write=True); write_json(OUT/'D_SEED42_COMPARISON.json',{'protocol_sha256':sha,'real_training_completed':True,'test_accessed':False,'independent_metric_audit_pass':auditm['pass'],'results':met.to_dict(orient='records'),'best_RMSE_group':str(met.loc[met.RMSE_mV.idxmin(),'experiment_group'])})
    figures(preds,met); status=write_report(sha,scores,t10,t12,met,audit)
    print(json.dumps({'status':status,'best':str(met.loc[met.RMSE_mV.idxmin(),'experiment_group']),'test_accessed':False},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--audit',action='store_true');args=p.parse_args()
    if args.audit: pretrain_audit(write=True)
    elif args.run: run()
    else: p.error('choose --audit or --run')
if __name__=='__main__':main()

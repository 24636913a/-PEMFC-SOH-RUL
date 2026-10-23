"""Development-only Phase N2.8D Future-Conditioned TimeXer experiment."""
from __future__ import annotations
import importlib.util, json, math, random, shutil, sys, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8D_Future_Conditioned_TimeXer")
CFG=ROOT/"configs/fc_timexer_frozen_config.yaml"; FOLDS=("fold_1","fold_2","fold_3")

def out(rel:str)->Path:
    p=(ROOT/rel).resolve(); p.parent.mkdir(parents=True,exist_ok=True)
    if p.parent!=ROOT.resolve() and ROOT.resolve() not in p.parents: raise RuntimeError("output escaped phase root")
    return p
def csv(x:pd.DataFrame,rel:str): x.to_csv(out(rel),index=False,encoding="utf-8-sig",lineterminator="\n")
def text(x:str,rel:str): out(rel).write_text(x.rstrip()+"\n",encoding="utf-8")
def jdump(x:Any,rel:str): out(rel).write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def ydump(x:dict[str,Any],rel:str): out(rel).write_text(yaml.safe_dump(x,allow_unicode=True,sort_keys=False),encoding="utf-8")
def mdtable(df:pd.DataFrame,cols:list[str])->str:
    if df.empty:return "| status |\n|---|\n| no data |"
    d=df[[c for c in cols if c in df]].copy()
    for c in d:
        if pd.api.types.is_float_dtype(d[c]): d[c]=d[c].map(lambda v:"" if pd.isna(v) else f"{v:.6f}")
    rows=["| "+" | ".join(d.columns)+" |","|"+"|".join("---" for _ in d.columns)+"|"]
    rows += ["| "+" | ".join(str(v).replace("|","/") for v in r)+" |" for r in d.itertuples(index=False,name=None)]
    return "\n".join(rows)
def cfg()->dict[str,Any]:
    c=yaml.safe_load(CFG.read_text(encoding="utf-8"))
    if Path(c["phase_root"]).resolve()!=ROOT.resolve():raise RuntimeError("wrong phase root")
    if (c["task"]["resolution_hours"],c["task"]["lookback_steps"],c["task"]["horizon_steps"])!=(0.5,48,24):raise RuntimeError("task changed")
    if len(c["features"])!=14 or len(c["future_exogenous_features"])!=9:raise RuntimeError("feature contract changed")
    if any(x.startswith("cell_voltage") for x in c["future_exogenous_features"]):raise RuntimeError("future voltage forbidden")
    return c
def seed(s:int):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def dev():return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def common_module(c):
    spec=importlib.util.spec_from_file_location("n28d_common",Path(c["common_script"]));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def load(c):
    cm=common_module(c);f,a=cm.load_development_data(c,c["features"])
    if a.get("target_rows_read_from_test")!=0 or f.split.eq("test").any():raise RuntimeError("TEST FIREWALL")
    w={n:(tr,va) for n,tr,va in cm.fold_windows(f,c)}
    for n,(_,v) in w.items():
        if v.target_end_age_h.max()>=c["split"]["development_end_h"]:raise RuntimeError(f"TEST FIREWALL {n}")
    a.update(target_rows_read_from_test=0,test_predictions_constructed=False,test_metrics_computed=False,test_feature_diagnostics_computed=False,test_performance_accessed=False,enforcement="fail_closed_development_prefix_only")
    return cm,f,w,a
def future(f,w,c):
    z=[]; h=c["task"]["horizon_steps"]
    for o in w.row_origin_index.astype(int):
        x=f.iloc[o+1:o+1+h].loc[:,c["future_exogenous_features"]].to_numpy(float)
        if x.shape!=(h,9):raise RuntimeError("incomplete future exogenous")
        z.append(x)
    return np.asarray(z,float)

class ZScale:
    def __init__(self):self.med=None;self.s=StandardScaler()
    def fit(self,h):
        q=h.reshape(-1,h.shape[-1]);self.med=np.nanmedian(q,0);self.s.fit(np.where(np.isfinite(q),q,self.med));return self
    def fix(self,x):return np.where(np.isfinite(x),x,self.med.reshape((1,)*(x.ndim-1)+(-1,)))
    def trans(self,x):
        q=self.fix(x);return self.s.transform(q.reshape(-1,q.shape[-1])).reshape(q.shape).astype("float32")

class RidgeMH:
    def __init__(self,a:float):self.a=a;self.xs=StandardScaler();self.ys=StandardScaler();self.m=Ridge(alpha=a)
    def fit(self,h,z,sc):
        x=sc.fix(h).reshape(len(h),-1);y=sc.fix(z).reshape(len(z),-1);self.xs.fit(x);self.ys.fit(y);self.m.fit(self.xs.transform(x),self.ys.transform(y));return self
    def pred(self,h,sc):
        x=sc.fix(h).reshape(len(h),-1);return self.ys.inverse_transform(self.m.predict(self.xs.transform(x))).reshape(len(h),24,9)

def oof_ridge(h,z,c,sc):
    n=len(h);warm=int(c["future_condition"]["oof_warmup_windows"]);blocks=int(c["future_condition"]["oof_blocks"])
    if n<=warm+1:raise RuntimeError("not enough windows for causal OOF Ridge")
    p=np.full_like(z,np.nan); audit=[];start=warm;size=math.ceil((n-warm)/blocks)
    while start<n:
        end=min(n,start+size); model=RidgeMH(c["future_condition"]["ridge_alpha"]).fit(h[:start],z[:start],sc);p[start:end]=model.pred(h[start:end],sc)
        audit.append({"fit_start_index":0,"fit_end_exclusive":start,"predict_start_index":start,"predict_end_exclusive":end,"uses_prediction_row_future_label":False})
        start=end
    keep=np.isfinite(p).all(axis=(1,2));return p,keep,audit

def clear_time():
    for n in list(sys.modules):
        if n=="models" or n.startswith("models.") or n=="layers" or n.startswith("layers."):del sys.modules[n]
def module(v:Path):
    clear_time();sys.path.insert(0,str(v))
    try:return __import__("models.TimeXer",fromlist=["Model"])
    finally:sys.path.remove(str(v))

class FCTimeXer(nn.Module):
    def __init__(self,c):
        super().__init__();tc=c["timexer"];M=module(Path(c["timexer_vendor"]))
        q=SimpleNamespace(task_name="long_term_forecast",features="MS",seq_len=48,pred_len=24,use_norm=False,patch_len=tc["patch_len"],enc_in=14,c_out=1,d_model=tc["d_model"],n_heads=tc["n_heads"],e_layers=tc["e_layers"],d_ff=tc["d_ff"],dropout=tc["dropout"],factor=5,embed="fixed",freq="h",activation="gelu")
        self.base=M.Model(q);d=tc["d_model"];self.future=nn.Sequential(nn.Linear(9,d),nn.GELU(),nn.LayerNorm(d));self.future_pos=nn.Parameter(torch.zeros(1,24,d));nn.init.normal_(self.future_pos,std=.02)
        self.horizon=nn.Parameter(torch.zeros(1,24,d));nn.init.normal_(self.horizon,std=.02);self.cross=nn.MultiheadAttention(d,tc["n_heads"],batch_first=True);self.norm=nn.LayerNorm(d);self.correct=nn.Linear(d,1);self.alpha=nn.Parameter(torch.tensor(float(c["future_condition"]["alpha_init"])))
    def forward(self,x,z,need_attention=False):
        if x.shape[1:]!=(48,14) or z.shape[1:]!=(24,9):raise RuntimeError("frozen shape mismatch")
        e=torch.cat([x[:,:,1:],x[:,:,:1]],2);en,nv=self.base.en_embedding(e[:,:,-1:].permute(0,2,1));ex=self.base.ex_embedding(e[:,:,:-1],None);enc=self.base.encoder(en,ex)
        q=self.horizon+enc[:,-1,:].unsqueeze(1);fh=self.future(z)+self.future_pos;att,wt=self.cross(q,fh,fh,need_weights=True,average_attn_weights=False);cond=self.norm(q+att);core=enc.reshape(-1,nv,enc.shape[-2],enc.shape[-1]).permute(0,1,3,2);base=self.base.head(core).permute(0,2,1).squeeze(-1);pred=base+self.alpha*self.correct(cond).squeeze(-1)
        return (pred,wt) if need_attention else pred
def count(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)

def metric(y,p,a):
    yd=y-a[:,None];pdv=p-a[:,None];e=pdv-yd;ae=np.abs(e).ravel()*1000
    return {"r2_delta":float(r2_score(yd.ravel(),pdv.ravel())),"rmse_delta_mV":float(np.sqrt(np.mean(e*e))*1000),"mae_delta_mV":float(np.mean(np.abs(e))*1000),"p50_mV":float(np.quantile(ae,.5)),"p90_mV":float(np.quantile(ae,.9)),"p95_mV":float(np.quantile(ae,.95)),"p99_mV":float(np.quantile(ae,.99)),"max_mV":float(ae.max())}
def scale(cm,tr,va,idx):
    p=cm.FoldPreprocessor().fit(tr.x);xt=p.transform(tr.x);xv=p.transform(va.x);ym=float(tr.y_delta.mean());ys=float(tr.y_delta.std());
    if ys<=1e-10:raise RuntimeError("degenerate y")
    return xt,xv,((tr.y_delta-ym)/ys).astype("float32"),((va.y_delta-ym)/ys).astype("float32"),ym,ys,ZScale().fit(tr.x[:,:,idx])

def train(model,xt,yt,zt,xv,yv,zv,c,checkpoint):
    d=dev();model.to(d);loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(yt),torch.from_numpy(zt)),batch_size=c["training"]["batch_size"],shuffle=True);opt=torch.optim.AdamW(model.parameters(),lr=c["training"]["learning_rate"],weight_decay=c["training"]["weight_decay"]);loss=nn.MSELoss();best=1e99;state=None;be=0;stale=0;times=[]
    xv=torch.from_numpy(xv).to(d);yv=torch.from_numpy(yv).to(d);zv=torch.from_numpy(zv).to(d)
    if d.type=="cuda":torch.cuda.reset_peak_memory_stats(d)
    for ep in range(1,c["training"]["max_epochs"]+1):
        t=time.perf_counter();model.train()
        for x,y,z in loader:
            opt.zero_grad(set_to_none=True);pr=model(x.to(d),z.to(d));L=loss(pr,y.to(d));L.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c["training"]["gradient_clip_norm"]);opt.step()
        if d.type=="cuda":torch.cuda.synchronize(d)
        times.append(time.perf_counter()-t);model.eval()
        with torch.no_grad():v=float(torch.sqrt(loss(model(xv,zv),yv)).item())
        if v<best-1e-7:best=v;be=ep;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=c["training"]["patience"]:break
    model.load_state_dict(state);torch.save(state,out(checkpoint));model.eval();t=time.perf_counter()
    with torch.no_grad():pred,att=model(xv,zv,True)
    if d.type=="cuda":torch.cuda.synchronize(d)
    info={"parameter_count":count(model),"best_epoch":be,"epochs_run":len(times),"validation_scaled_delta_rmse":best,"training_time_per_epoch_s":float(np.mean(times)),"peak_gpu_memory_mb":float(torch.cuda.max_memory_allocated(d)/1024**2) if d.type=="cuda" else 0.0,"inference_time_batch_s":time.perf_counter()-t,"device":str(d)}
    return info,pred.detach().cpu().numpy(),att.detach().cpu().numpy()

def original(c,w):
    root=Path(c["phase_n2_8a_root"])/"outputs/model_screening";rows=[];arr={}
    for f in FOLDS:
        a=dict(np.load(root/f"timexer_official_{f}_seed42_predictions.npz",allow_pickle=False));v=w[f][1]
        if not np.array_equal(a["row_origin_index"],v.row_origin_index) or a["target_end_age_h"].max()>=c["split"]["development_end_h"]:raise RuntimeError("bad original reference")
        arr[f]=a;rows.append({"model":"TimeXer-Original","fold":f,"seed":42,"eligibility":"reference_only","parameter_count":np.nan,**metric(a["y_true"],a["y_pred"],a["anchor"])})
    return pd.DataFrame(rows),arr
def run(label,kind,packages,cm,w,c,seedid):
    idx=[c["features"].index(x) for x in c["future_exogenous_features"]];rows=[];arrays={};atts=[]
    for fi,f in enumerate(FOLDS):
        tr,va=w[f];xt,xv,yt,yv,ym,ys,sc=scale(cm,tr,va,idx);P=packages[f];keep=P["keep"]
        if kind=="oracle": ztr,zva=P["true_train"],P["true_val"];mask=np.ones(len(tr.x),bool)
        elif kind=="zero": ztr,zva=np.zeros_like(P["oof"]),np.zeros_like(P["ridge_val"]);mask=keep
        else:
            ztr,zva=P["oof"],P["ridge_val"];mask=keep
            if kind=="shuffle": zva=zva[np.random.default_rng(seedid+fi).permutation(len(zva))]
        seed(seedid);m=FCTimeXer(c);info,pred,att=train(m,xt[mask],yt[mask],sc.trans(ztr[mask]),xv,yv,sc.trans(zva),c,f"checkpoints/{label}_{f}_seed{seedid}.pt")
        pa=va.anchor[:,None]+pred*ys+ym;a={"y_true":va.y_abs.astype(float),"y_pred":pa.astype(float),"anchor":va.anchor.astype(float),"target_end_age_h":va.target_end_age_h.astype(float),"row_origin_index":va.row_origin_index.astype(int),"future_exogenous_input":zva.astype(float),"future_exogenous_true":P["true_val"].astype(float)}
        if a["target_end_age_h"].max()>=c["split"]["development_end_h"]:raise RuntimeError("TEST FIREWALL prediction")
        np.savez_compressed(out(f"outputs/model_runs/{label}_{f}_seed{seedid}_predictions.npz"),**a);arrays[f]=a;atts.append(att.mean(axis=(0,1)));rows.append({"model":label,"fold":f,"seed":seedid,"eligibility":"DIAGNOSTIC_ORACLE_ONLY" if kind=="oracle" else "FORMAL" if kind in ("ridge","zero") else "NEGATIVE_CONTROL",**metric(a["y_true"],a["y_pred"],a["anchor"]),**info,"oof_train_windows":int(mask.sum())})
    return pd.DataFrame(rows),arrays,np.mean(atts,axis=0)
def summary(rows):
    r=[]
    for name,g in rows.groupby("model"):
        x={"model":name,"parameter_count":float(g.parameter_count.dropna().iloc[0]) if g.parameter_count.notna().any() else np.nan}
        for f in FOLDS:
            q=g[g.fold==f].iloc[0];x[f"{f}_r2_delta"]=q.r2_delta;x[f"{f}_rmse_delta_mV"]=q.rmse_delta_mV
        for k in ["r2_delta","rmse_delta_mV","mae_delta_mV","p95_mV","p99_mV"]:x[f"mean_{k}"]=float(g[k].mean())
        x["lifecycle_ratio"]=x["fold_3_rmse_delta_mV"]/x["fold_1_rmse_delta_mV"];x["eligibility"]=g.eligibility.iloc[0];r.append(x)
    return pd.DataFrame(r)
def horizons(name,arr,c):
    q=[]
    for f,a in arr.items():
        y=a["y_true"]-a["anchor"][:,None];p=a["y_pred"]-a["anchor"][:,None]
        for i in range(24):q.append({"model":name,"fold":f,"horizon_step":i+1,"horizon_hours":(i+1)*.5,"r2_delta":float(r2_score(y[:,i],p[:,i])),"rmse_delta_mV":float(np.sqrt(np.mean((p[:,i]-y[:,i])**2))*1000),"mae_delta_mV":float(np.mean(np.abs(p[:,i]-y[:,i]))*1000)})
    return q
def tail(name,arr):
    e=[]
    for a in arr.values():e.extend(((a["y_pred"]-a["anchor"][:,None])-(a["y_true"]-a["anchor"][:,None])).ravel()*1000)
    e=np.abs(e);return {"model":name,"p50_mV":float(np.quantile(e,.5)),"p90_mV":float(np.quantile(e,.9)),"p95_mV":float(np.quantile(e,.95)),"p99_mV":float(np.quantile(e,.99)),"max_mV":float(e.max())}

def plots(comp,hor,taildf,att):
    fig=out("outputs/figures/.keep").parent
    v=comp.copy();ax=v.set_index("model").mean_rmse_delta_mV.plot(kind="bar",figsize=(8,4));ax.set_ylabel("Mean RMSEΔV (mV)");plt.tight_layout();plt.savefig(fig/"fc_timexer_rmse_comparison.png");plt.close()
    g=hor.groupby(["model","horizon_hours"],as_index=False).rmse_delta_mV.mean();F,A=plt.subplots(figsize=(8,4))
    for n,x in g.groupby("model"):A.plot(x.horizon_hours,x.rmse_delta_mV,label=n)
    A.axvline(9.5,ls="--",c="gray");A.axvline(12,ls=":",c="gray");A.legend(fontsize=7);A.set_ylabel("RMSEΔV (mV)");plt.tight_layout();plt.savefig(fig/"fc_timexer_per_horizon_rmse.png");plt.close()
    ax=taildf.set_index("model")[["p95_mV","p99_mV"]].plot(kind="bar",figsize=(8,4));ax.set_ylabel("Absolute ΔV error (mV)");plt.tight_layout();plt.savefig(fig/"fc_timexer_tail_error.png");plt.close()
    F,A=plt.subplots(figsize=(7,6));im=A.imshow(att,cmap="viridis",aspect="auto");A.set_xlabel("Future condition step j");A.set_ylabel("Forecast horizon h");A.set_title("FC-TimeXer-Ridge future-condition attention");F.colorbar(im,ax=A);plt.tight_layout();plt.savefig(fig/"future_attention_heatmap.png");plt.close()

def main():
    c=cfg();shutil.copy2(CFG,out("outputs/frozen_config/fc_timexer_frozen_config.yaml"));cm,frame,w,fire=load(c);jdump(fire,"outputs/test_firewall_audit.json")
    idx=[c["features"].index(x) for x in c["future_exogenous_features"]];packages={};audit=[]
    for f in FOLDS:
        tr,va=w[f];htr=tr.x[:,:,idx];hva=va.x[:,:,idx];ztr=future(frame,tr,c);zva=future(frame,va,c);sc=ZScale().fit(htr);oof,keep,blocks=oof_ridge(htr,ztr,c,sc);ridge=RidgeMH(c["future_condition"]["ridge_alpha"]).fit(htr,ztr,sc);pred=ridge.pred(hva,sc)
        packages[f]={"true_train":ztr,"true_val":zva,"oof":oof,"ridge_val":pred,"keep":keep};audit += [{"fold":f,**x,"warmup_windows_dropped":int((~keep).sum())} for x in blocks]
        np.savez_compressed(out(f"outputs/future_exogenous/future_exogenous_predictions_{f.replace('_','')}.npz"),origin_time=va.origin_datetime,predicted_future_exogenous=pred,true_future_exogenous=zva,row_origin_index=va.row_origin_index)
    text("# Future exogenous OOF audit\n\nFormal FC-TimeXer training inputs use expanding chronological OOF Ridge predictions. Each predicted training block is generated by a Ridge model fitted only on earlier training-window rows. The warm-up rows without a prior-only fit are excluded from formal FC training rather than filled with true future exogenous values.\n\n"+mdtable(pd.DataFrame(audit),["fold","fit_start_index","fit_end_exclusive","predict_start_index","predict_end_exclusive","warmup_windows_dropped","uses_prediction_row_future_label"]),"FUTURE_EXO_OOF_AUDIT.md")
    base=FCTimeXer(c);total=count(base);original_count=count(base.base);branch=total-original_count
    if branch/original_count>=.30:raise RuntimeError("future branch exceeds 30% parameter cap")
    text(f"# Future-condition architecture\n\nOfficial TimeXer commit `{c['timexer']['commit']}` remains the historical backbone. FC-TimeXer adds a 9→d_model MLP with positions, 24 horizon queries, one MultiheadAttention cross-condition layer, residual LayerNorm/correction head, and learnable alpha initialized at {c['future_condition']['alpha_init']}.\n\n- parameter_count_original: {original_count}\n- parameter_count_future_branch: {branch}\n- parameter_count_total: {total}\n- future branch / original: {branch/original_count:.4%}\n","FUTURE_CONDITION_ARCHITECTURE.md")
    o_rows,o_arr=original(c,w);z_rows,z_arr,_=run("FC_TimeXer_ZeroFuture","zero",packages,cm,w,c,42);r_rows,r_arr,r_att=run("FC_TimeXer_Ridge","ridge",packages,cm,w,c,42);q_rows,q_arr,_=run("FC_TimeXer_Oracle","oracle",packages,cm,w,c,42);s_rows,s_arr,_=run("FC_TimeXer_ShuffledFuture","shuffle",packages,cm,w,c,42)
    allrows=pd.concat([o_rows,z_rows,r_rows,q_rows,s_rows],ignore_index=True);comp=summary(allrows);orig=comp[comp.model=="TimeXer-Original"].iloc[0];zero=comp[comp.model=="FC_TimeXer_ZeroFuture"].iloc[0];ridge=comp[comp.model=="FC_TimeXer_Ridge"].iloc[0];oracle=comp[comp.model=="FC_TimeXer_Oracle"].iloc[0];shuffle=comp[comp.model=="FC_TimeXer_ShuffledFuture"].iloc[0]
    comp["future_condition_gain_pct"]=comp.mean_rmse_delta_mV.map(lambda x:(orig.mean_rmse_delta_mV-x)/orig.mean_rmse_delta_mV*100);comp.loc[comp.model=="TimeXer-Original","future_condition_gain_pct"]=0.0
    h=pd.DataFrame(horizons("TimeXer-Original",o_arr,c)+horizons("FC_TimeXer_ZeroFuture",z_arr,c)+horizons("FC_TimeXer_Ridge",r_arr,c)+horizons("FC_TimeXer_Oracle",q_arr,c)+horizons("FC_TimeXer_ShuffledFuture",s_arr,c));taildf=pd.DataFrame([tail("TimeXer-Original",o_arr),tail("FC_TimeXer_ZeroFuture",z_arr),tail("FC_TimeXer_Ridge",r_arr),tail("FC_TimeXer_Oracle",q_arr),tail("FC_TimeXer_ShuffledFuture",s_arr)])
    gains=pd.DataFrame([{"comparison":"FC_Ridge_vs_Original","gain_pct":(orig.mean_rmse_delta_mV-ridge.mean_rmse_delta_mV)/orig.mean_rmse_delta_mV*100},{"comparison":"FC_Ridge_vs_ZeroFuture","gain_pct":(zero.mean_rmse_delta_mV-ridge.mean_rmse_delta_mV)/zero.mean_rmse_delta_mV*100}]);np.save(out("outputs/attention_analysis/future_condition_attention.npy"),r_att)
    d19=h.groupby(["model","horizon_hours"],as_index=False).rmse_delta_mV.mean();h19=lambda m,x:float(d19[(d19.model==m)&np.isclose(d19.horizon_hours,x)].rmse_delta_mV.iloc[0]);foldgain=(orig.fold_3_rmse_delta_mV-ridge.fold_3_rmse_delta_mV)/orig.fold_3_rmse_delta_mV*100
    promote=(ridge.mean_rmse_delta_mV<=12) or (ridge.mean_rmse_delta_mV<15 and foldgain>=25);mr=[]
    if promote:
        for sd in (2024,3407):rr,_,_=run("FC_TimeXer_Ridge","ridge",packages,cm,w,c,sd);allrows=pd.concat([allrows,rr],ignore_index=True)
        v=allrows[allrows.model=="FC_TimeXer_Ridge"].groupby("seed",as_index=False)[["r2_delta","rmse_delta_mV"]].mean();mr=[{"model":"FC_TimeXer_Ridge","status":"promoted_three_seed","n_seeds":len(v),"seeds":";".join(v.seed.astype(str)),"mean_r2_delta":float(v.r2_delta.mean()),"mean_rmse_delta_mV":float(v.rmse_delta_mV.mean()),"rmse_std_mV":float(v.rmse_delta_mV.std(ddof=0))}]
    else:mr=[{"model":"FC_TimeXer_Ridge","status":"not_promoted_seed42_only","n_seeds":1,"seeds":"42","mean_r2_delta":ridge.mean_r2_delta,"mean_rmse_delta_mV":ridge.mean_rmse_delta_mV,"rmse_std_mV":0.0}]
    multi=pd.DataFrame(mr);status="FAIL" if ridge.mean_rmse_delta_mV>=25 else "STRUCTURAL_RECOVERY" if ridge.mean_rmse_delta_mV>=20 else "PROMISING" if ridge.mean_rmse_delta_mV>12 else "REVIVED";diag=float(np.trace(r_att)/24);arg=np.argmax(r_att,1);offset=float(np.mean(np.abs(arg-np.arange(24))))
    csv(comp,"outputs/model_screening/N2_8D_seed42_comparison.csv");csv(comp[["model","eligibility","mean_r2_delta","mean_rmse_delta_mV","mean_mae_delta_mV","future_condition_gain_pct"]],"outputs/ablation/N2_8D_ablation.csv");csv(h,"outputs/horizon_analysis/N2_8D_per_horizon.csv");csv(taildf,"outputs/tail_analysis/N2_8D_tail_error.csv");csv(gains,"outputs/ablation/future_condition_gain.csv");csv(multi,"outputs/model_screening/N2_8D_multiseed_comparison.csv");csv(allrows,"outputs/model_screening/foldwise_fc_timexer_metrics.csv");plots(comp,h,taildf,r_att)
    dec={"phase_status":"PASS","original_timexer":{"r2_delta":float(orig.mean_r2_delta),"rmse_delta_mV":float(orig.mean_rmse_delta_mV)},"zero_future":{"r2_delta":float(zero.mean_r2_delta),"rmse_delta_mV":float(zero.mean_rmse_delta_mV)},"ridge_future":{"r2_delta":float(ridge.mean_r2_delta),"rmse_delta_mV":float(ridge.mean_rmse_delta_mV),"gain_vs_original_pct":float(gains.gain_pct.iloc[0]),"gain_vs_zero_future_pct":float(gains.gain_pct.iloc[1])},"oracle_future":{"r2_delta":float(oracle.mean_r2_delta),"rmse_delta_mV":float(oracle.mean_rmse_delta_mV)},"shuffled_future":{"r2_delta":float(shuffle.mean_r2_delta),"rmse_delta_mV":float(shuffle.mean_rmse_delta_mV)},"fold3":{"original_rmse_delta_mV":float(orig.fold_3_rmse_delta_mV),"ridge_future_rmse_delta_mV":float(ridge.fold_3_rmse_delta_mV),"gain_pct":float(foldgain)},"horizon_9_5h":{"original_rmse_delta_mV":h19("TimeXer-Original",9.5),"ridge_future_rmse_delta_mV":h19("FC_TimeXer_Ridge",9.5)},"horizon_12h":{"original_rmse_delta_mV":h19("TimeXer-Original",12),"ridge_future_rmse_delta_mV":h19("FC_TimeXer_Ridge",12)},"p95_mV":float(ridge.mean_p95_mV),"p99_mV":float(ridge.mean_p99_mV),"three_seed_promoted":bool(promote),"multiseed_r2_delta":float(multi.mean_r2_delta.iloc[0]),"multiseed_rmse_delta_mV":float(multi.mean_rmse_delta_mV.iloc[0]),"future_condition_status":status,"attention":{"diagonal_mean":diag,"mean_abs_argmax_offset_steps":offset},"recommended_next_phase":"Phase N2.8E Physics-Informed Future-Conditioned TimeXer" if ridge.mean_rmse_delta_mV<20 else "Stop TimeXer and retain Lifecycle-Aware Robust TimesNet","test_performance_accessed":False};ydump(dec,"PHASE_N2_8D_DECISION.yaml")
    rep=f"""# Phase N2.8D results

## Protocol and firewall
Development-only frozen 0.5 h / 48 / 24 / F2 / DeltaV task. Test targets read: 0. No Test prediction, metric, or diagnostic was built.

## Q1–Q6: FC-TimeXer and controls
{mdtable(comp,["model","eligibility","mean_r2_delta","mean_rmse_delta_mV","mean_mae_delta_mV","fold_3_rmse_delta_mV","mean_p95_mV","mean_p99_mV","lifecycle_ratio","future_condition_gain_pct"])}

Correct RidgeFuture vs ZeroFuture gain: {gains.gain_pct.iloc[1]:.6f}%. Correct RidgeFuture vs original gain: {gains.gain_pct.iloc[0]:.6f}%.

## Q7–Q12: Fold3, tails, and horizons
Fold3 RidgeFuture gain = {foldgain:.6f}%. At 9.5 h: Original {h19("TimeXer-Original",9.5):.6f} mV; RidgeFuture {h19("FC_TimeXer_Ridge",9.5):.6f} mV. At 12 h: Original {h19("TimeXer-Original",12):.6f} mV; RidgeFuture {h19("FC_TimeXer_Ridge",12):.6f} mV.

{mdtable(taildf,["model","p50_mV","p90_mV","p95_mV","p99_mV","max_mV"])}

## Q13: Attention
RidgeFuture mean diagonal attention = {diag:.6f}; mean absolute argmax offset = {offset:.6f} future steps. The full 24×24 mean attention map is saved as `future_condition_attention.npy` and visualized in `future_attention_heatmap.png`.

## Q14–Q22: Decision
Oracle–Ridge RMSE gap = {oracle.mean_rmse_delta_mV-ridge.mean_rmse_delta_mV:.6f} mV. Status: **{status}**. Three-seed promoted: **{promote}**. Recommended next phase: **{dec['recommended_next_phase']}**. PINN is recommended only if RidgeFuture RMSE is below 20 mV: **{ridge.mean_rmse_delta_mV<20}**.

{mdtable(multi,["model","status","n_seeds","seeds","mean_r2_delta","mean_rmse_delta_mV","rmse_std_mV"])}
""";text(rep,"PHASE_N2_8D_RESULTS.md");print(json.dumps({"phase_status":"PASS","ridge_rmse":float(ridge.mean_rmse_delta_mV),"status":status,"promote":bool(promote)},ensure_ascii=False))
if __name__=="__main__":main()

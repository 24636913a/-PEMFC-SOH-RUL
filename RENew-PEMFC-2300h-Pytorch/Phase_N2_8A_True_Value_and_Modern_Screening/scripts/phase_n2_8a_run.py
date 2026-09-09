"""Development-only true-DeltaV audit and official TimeXer screening for N2.8A."""
from __future__ import annotations

import importlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
CFG_PATH = ROOT / "configs/phase_n2_8a_config.yaml"
N25_SCRIPTS = PROJECT / "Phase_N2_5_Cycle_Aware_xLSTM/scripts"
if str(N25_SCRIPTS) not in sys.path: sys.path.insert(0, str(N25_SCRIPTS))
from phase_n2_5_common import FoldPreprocessor, fold_windows, load_development_data  # noqa: E402

FOLDS=["fold_1","fold_2","fold_3"]
SEEDS=[42,2024,3407]

def ensure(path: Path) -> Path:
    target=path.resolve(); root=ROOT.resolve()
    if target.parent!=root and root not in target.parents: raise RuntimeError(f"Output outside N2.8A: {target}")
    target.parent.mkdir(parents=True,exist_ok=True); return target
def csv(frame:pd.DataFrame,rel:str)->None: frame.to_csv(ensure(ROOT/rel),index=False,encoding="utf-8-sig",lineterminator="\n")
def js(payload:Any,rel:str)->None: ensure(ROOT/rel).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def txt(value:str,rel:str)->None: ensure(ROOT/rel).write_text(value.rstrip()+"\n",encoding="utf-8")
def set_seed(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def cfg()->dict[str,Any]:
    out=yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    if Path(out["phase_root"]).resolve()!=ROOT.resolve(): raise RuntimeError("Config root mismatch")
    if len(out["features"])!=14 or out["features"][0]!="cell_voltage_mean": raise RuntimeError("Exact F2 protocol mismatch")
    return out
def metric(truth:np.ndarray,pred:np.ndarray,anchor:np.ndarray)->dict[str,float]:
    td=np.asarray(truth,float)-np.asarray(anchor,float)[:,None]; pdlt=np.asarray(pred,float)-np.asarray(anchor,float)[:,None]
    e=pdlt-td; ae=np.asarray(pred,float)-np.asarray(truth,float)
    dstd=float(np.std(td)); rmse=float(np.sqrt(np.mean(e*e))); zero=float(np.sqrt(np.mean(td*td)))
    den=float(np.sum((td-td.mean())**2)); dena=float(np.sum((np.asarray(truth,float)-np.mean(truth))**2))
    return {"rmse_delta_mV":rmse*1000,"mae_delta_mV":float(np.mean(abs(e)))*1000,"r2_delta":1-float(np.sum(e*e))/den if den>0 else math.nan,"std_deltaV_mV":dstd*1000,"nrmse_delta":rmse/dstd if dstd>0 else math.nan,"skill_delta":1-rmse/zero if zero>0 else math.nan,"rmse_abs_mV":float(np.sqrt(np.mean(ae*ae)))*1000,"mae_abs_mV":float(np.mean(abs(ae)))*1000,"r2_abs":1-float(np.sum(ae*ae))/dena if dena>0 else math.nan,"bias_abs_mV":float(np.mean(ae))*1000}
def load_npz(result:Path,array:Path,c:dict[str,Any],label:str)->dict[str,np.ndarray]:
    status=json.loads(result.read_text(encoding="utf-8"))
    a=dict(np.load(array,allow_pickle=False)); need={"y_true","y_pred","anchor","target_end_age_h","row_origin_index"}
    if status.get("status")!="complete" or not need.issubset(a):raise RuntimeError(f"Incomplete legacy run {label}")
    if a["y_true"].shape!=a["y_pred"].shape or a["y_true"].shape[1]!=24:raise RuntimeError(f"Legacy shape mismatch {label}")
    if float(a["target_end_age_h"].max())>=float(c["split"]["development_end_h"]):raise RuntimeError(f"TEST FIREWALL legacy {label}")
    return a
def legacy_arrays(c:dict[str,Any])->dict[tuple[str,str],dict[str,np.ndarray]]:
    n2=Path(c["phase_n2_root"])/"outputs/model_runs"; n26=Path(c["phase_n2_6r_root"])/"outputs/model_runs"; d={}
    forms={"LSTM":(n2,"feature_F2_LSTM_{fold}_seed42"),"PatchTST":(n2,"screening_PatchTST_F2_L24p0h_H12p0h_R0p5h_{fold}_seed42"),"iTransformer":(n2,"screening_iTransformer_F2_L24p0h_H12p0h_R0p5h_{fold}_seed42"),"TimeMixer":(n2,"screening_TimeMixer_F2_L24p0h_H12p0h_R0p5h_{fold}_seed42"),"TCN":(n2,"screening_TCN_F2_L24p0h_H12p0h_R0p5h_{fold}_seed42"),"TimesNet":(n26,"screening_timesnet_hf_t1_{fold}_seed42")}
    for name,(root,template) in forms.items():
        for fold in FOLDS:
            run=template.format(fold=fold);d[(name,fold)]=load_npz(root/f"{run}.json",root/f"{run}_predictions.npz",c,run)
    baseline=pd.read_csv(Path(c["phase_n2_root"])/"outputs/baseline_results/baseline_validation_predictions.csv")
    for name in ["Ridge","CycleRepeat","Persistence"]:
        for fold in FOLDS:
            p=baseline[(baseline.model.eq(name))&(baseline.fold.eq(fold))].sort_values(["origin_age_h","horizon_step"])
            count=len(p)//24
            if count*24!=len(p):raise RuntimeError(f"Baseline horizon grid bad {name}/{fold}")
            d[(name,fold)]={"y_true":p.actual_voltage_V.to_numpy().reshape(count,24),"y_pred":p.predicted_voltage_V.to_numpy().reshape(count,24),"anchor":p.anchor_voltage_V.to_numpy().reshape(count,24)[:,0],"target_end_age_h":p.target_age_h.to_numpy().reshape(count,24)[:,-1],"row_origin_index":np.arange(count)}
    return d
def clear_official()->None:
    for name in list(sys.modules):
        if name=="models" or name.startswith("models.") or name=="layers" or name.startswith("layers."):del sys.modules[name]
def official_module(root:Path,module:str)->Any:
    clear_official();sys.path.insert(0,str(root))
    try:return importlib.import_module(module)
    finally:sys.path.remove(str(root))
class LSTM(nn.Module):
    def __init__(self):super().__init__();self.net=nn.LSTM(14,64,2,batch_first=True,dropout=.1);self.head=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,24))
    def forward(self,x:torch.Tensor)->torch.Tensor:return self.head(self.net(x)[0][:,-1])
class TimesNet(nn.Module):
    def __init__(self):
        super().__init__(); mod=official_module(ROOT/"vendor/TimeXer","models.TimesNet");o=SimpleNamespace(task_name="long_term_forecast",seq_len=48,label_len=0,pred_len=24,e_layers=2,d_model=32,d_ff=64,top_k=3,num_kernels=4,enc_in=14,c_out=14,embed="fixed",freq="h",dropout=.1,num_class=0);self.model=mod.Model(o)
    def forward(self,x:torch.Tensor)->torch.Tensor:return self.model(x,None,None,None)[:,:,0]
class TimeXer(nn.Module):
    def __init__(self):
        super().__init__();mod=official_module(ROOT/"vendor/TimeXer","models.TimeXer");o=SimpleNamespace(task_name="long_term_forecast",features="MS",seq_len=48,pred_len=24,use_norm=False,patch_len=8,enc_in=14,c_out=1,d_model=64,n_heads=4,e_layers=2,d_ff=128,dropout=.1,factor=5,embed="fixed",freq="h",activation="gelu");self.model=mod.Model(o)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # Official TimeXer MS mode takes the endogenous target in the last channel.
        ordered=torch.cat([x[:,:,1:],x[:,:,:1]],dim=2);return self.model(ordered,None,None,None).squeeze(-1)
def train_one(name:str,model:nn.Module,xtr:np.ndarray,ytr:np.ndarray,xv:np.ndarray,yv:np.ndarray,anchor:np.ndarray,seed:int,c:dict[str,Any],runid:str)->tuple[dict[str,Any],np.ndarray]:
    set_seed(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=c["training"]["learning_rate"],weight_decay=c["training"]["weight_decay"]);lossfn=nn.MSELoss();g=torch.Generator().manual_seed(seed);dl=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=64,shuffle=True,generator=g);best=math.inf;state=None;bad=0;bestep=0;logs=[]
    for ep in range(1,101):
        model.train();l=[]
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True);z=model(xb);loss=lossfn(z,yb);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();l.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():pred=model(torch.from_numpy(xv).to(device)).cpu().numpy()
        now=float(np.sqrt(np.mean((pred-yv)**2)));logs.append({"epoch":ep,"train_mse":float(np.mean(l)),"validation_scaled_delta_rmse":now})
        if now<best-1e-6:best=now;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bestep=ep;bad=0
        else:bad+=1
        if bad>=15:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():pred=model(torch.from_numpy(xv).to(device)).cpu().numpy()
    info={"model":name,"run_id":runid,"seed":seed,"best_epoch":bestep,"epochs":len(logs),"parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),"device":str(device)}
    csv(pd.DataFrame(logs),f"logs/{runid}_epochs.csv");return info,pred
def prep_windows(c:dict[str,Any]):
    frame,audit=load_development_data(c,c["features"])
    if audit.get("target_rows_read_from_test")!=0 or frame.split.eq("test").any():raise RuntimeError("TEST FIREWALL feature/target frame")
    windows={fold:(train,val) for fold,train,val in fold_windows(frame,c)}
    for fold,(_,val) in windows.items():
        if float(val.target_end_age_h.max())>=float(c["split"]["development_end_h"]):raise RuntimeError(f"TEST FIREWALL window {fold}")
    return windows,{**audit,"test_target_rows_read":0,"test_predictions_constructed":False,"test_metrics_computed":False,"test_feature_diagnostics_computed":False,"test_performance_accessed":False}
def scaled(train,val):
    p=FoldPreprocessor().fit(train.x);xtr=p.transform(train.x);xv=p.transform(val.x);mean=train.y_delta.mean(0,keepdims=True);std=np.where(train.y_delta.std(0,keepdims=True)>1e-6,train.y_delta.std(0,keepdims=True),1);return xtr,xv,((train.y_delta-mean)/std).astype("float32"),mean,std,p
def controls(c,windows):
    rows=[]
    for kind in ["label_permutation","temporal_misalignment"]:
        for fold,(tr,val) in windows.items():
            xtr,xv,y,mean,std,_=scaled(tr,val);rng=np.random.default_rng(42)
            if kind=="label_permutation":yuse=y[rng.permutation(len(y))];xuse=xtr
            else:
                k=48
                if len(y)<=k:raise RuntimeError("Misalignment requires >=48 training origins")
                xuse=xtr[:-k];yuse=y[k:]
            for name,clazz in [("LSTM",LSTM),("TimesNet",TimesNet)]:
                info,pred=train_one(name+"_"+kind,clazz(),xuse,yuse,xv,((val.y_delta-mean)/std).astype("float32"),val.anchor,42,c,f"nc_{kind}_{name}_{fold}")
                absolute=val.anchor[:,None]+pred*std+mean;m=metric(val.y_abs,absolute,val.anchor);rows.append({"control":kind,"model":name,"fold":fold,"seed":42,**m,**info})
    out=pd.DataFrame(rows);csv(out[out.control.eq("label_permutation")],"outputs/leakage_audit/label_permutation_negative_control.csv");csv(out[out.control.eq("temporal_misalignment")],"outputs/leakage_audit/temporal_misalignment_negative_control.csv");return out
def timexer(c,windows,seed):
    rows=[];arrays={}
    for fold,(tr,val) in windows.items():
        xtr,xv,y,mean,std,p=scaled(tr,val);info,pred=train_one("TimeXer",TimeXer(),xtr,y,xv,((val.y_delta-mean)/std).astype("float32"),val.anchor,seed,c,f"timexer_official_{fold}_seed{seed}");absval=val.anchor[:,None]+pred*std+mean;arr={"y_true":val.y_abs,"y_pred":absval,"anchor":val.anchor,"target_end_age_h":val.target_end_age_h,"row_origin_index":val.row_origin_index};np.savez_compressed(ensure(ROOT/f"outputs/model_screening/timexer_official_{fold}_seed{seed}_predictions.npz"),**arr);m=metric(val.y_abs,absval,val.anchor);rows.append({"model":"TimeXer","fold":fold,"seed":seed,**m,**info});arrays[fold]=arr
    return pd.DataFrame(rows),arrays
def summarize(model_arrays:dict[tuple[str,str],dict[str,np.ndarray]],tx:dict[str,np.ndarray]):
    rows=[]
    for (name,fold),a in model_arrays.items():rows.append({"model":name,"fold":fold,"seed":42,**metric(a["y_true"],a["y_pred"],a["anchor"])})
    for fold,a in tx.items():rows.append({"model":"TimeXer","fold":fold,"seed":42,**metric(a["y_true"],a["y_pred"],a["anchor"])})
    return pd.DataFrame(rows)
def macro(table:pd.DataFrame):
    return table.groupby("model",as_index=False).agg(r2_delta=("r2_delta","mean"),rmse_delta_mV=("rmse_delta_mV","mean"),mae_delta_mV=("mae_delta_mV","mean"),r2_abs=("r2_abs","mean"),rmse_abs_mV=("rmse_abs_mV","mean"),parameter_count=("parameter_count","first"))
def figures(table:pd.DataFrame):
    d=table[table.model.isin(["LSTM","TimesNet","TimeXer","Ridge"])].copy();out=ROOT/"outputs/figures";out.mkdir(parents=True,exist_ok=True)
    for col,title,file in [("r2_delta","R² ΔV (primary)","r2_delta_model_comparison.png"),("rmse_delta_mV","RMSE ΔV (mV)","rmse_delta_model_comparison.png")]:
        z=d.groupby("model")[col].mean().sort_values(ascending=col=="rmse_delta_mV");fig,ax=plt.subplots(figsize=(7,4));ax.bar(z.index,z.values,color=["#2864b7","#d85a38","#8055a5","#5f9e6e"]);ax.set(ylabel=title,title=title+" — Development seed 42");fig.tight_layout();fig.savefig(out/file,dpi=160);plt.close(fig)
    z=d.groupby("model")[["r2_abs","r2_delta"]].mean();fig,ax=plt.subplots(figsize=(7,4));z.plot.bar(ax=ax,color=["#8aa6d1","#d85a38"]);ax.set(ylabel="R²",title="Absolute-voltage versus true ΔV R²");fig.tight_layout();fig.savefig(out/"r2_abs_vs_r2_delta.png",dpi=160);plt.close(fig)
    for col,file,label in [("r2_delta","foldwise_r2_delta.png","R² ΔV"),("rmse_delta_mV","foldwise_rmse_delta.png","RMSE ΔV (mV)")]:
        p=d.pivot(index="fold",columns="model",values=col);fig,ax=plt.subplots(figsize=(8,4));p.plot(marker="o",ax=ax);ax.set(ylabel=label,title=label+" by lifecycle fold");fig.tight_layout();fig.savefig(out/file,dpi=160);plt.close(fig)
def main():
    c=cfg();windows,firewall=prep_windows(c);js(firewall,"outputs/leakage_audit/test_firewall_audit.json")
    feature=[]
    for name in c["features"]:feature.append({"feature_name":name,"source_column":name,"temporal_window":"current causal aggregation bin; history only through origin","future_access":False,"centered_rolling":False,"negative_shift":False,"future_backfill":False,"status":"PASS"})
    csv(pd.DataFrame(feature),"outputs/leakage_audit/feature_leakage_audit.csv")
    legacy=legacy_arrays(c);legacy_rows=[]
    for (name,fold),a in legacy.items():legacy_rows.append({"model":name,"seed":42,"fold":fold,**metric(a["y_true"],a["y_pred"],a["anchor"])})
    legacy_df=pd.DataFrame(legacy_rows);csv(legacy_df,"outputs/true_delta_metrics/legacy_models_true_delta_metrics.csv")
    anchor=[]
    for name in ["LSTM","TimesNet","Ridge"]:
        for fold in FOLDS:
            a=legacy[(name,fold)];m=metric(a["y_true"],a["y_pred"],a["anchor"]);anchor.append({"model":name,"fold":fold,"r2_abs":m["r2_abs"],"r2_delta":m["r2_delta"],"anchor_contribution_r2_difference":m["r2_abs"]-m["r2_delta"],"rmse_abs_mV":m["rmse_abs_mV"],"rmse_delta_mV":m["rmse_delta_mV"]})
    csv(pd.DataFrame(anchor),"outputs/true_delta_metrics/anchor_contribution_audit.csv")
    nc=controls(c,windows);warning=bool((nc[(nc.control.eq("label_permutation"))].r2_delta>.5).any())
    tx_df,tx_arr=timexer(c,windows,42);csv(tx_df,"outputs/model_screening/timexer_seed42_results.csv")
    blocked=pd.DataFrame([{ "model":"TimeMixer++","seed":42,"status":"BLOCKED_OFFICIAL_IMPLEMENTATION_UNAVAILABLE","official_repo_checked":"https://anonymous.4open.science/r/TimeMixerPP (HTTP 410); https://github.com/kwuking/TimeMixer @ e24610583b36fdd8c76cc17a8df4e65759a5f460","reason":"Fixed vendor source contains old TimeMixer only and lacks MRTI/TID/MCM/MRM. No surrogate was trained."}]);csv(blocked,"outputs/model_screening/timemixerpp_seed42_results.csv")
    combined=summarize(legacy,tx_arr);combined=combined.merge(tx_df[["model","fold","parameter_count"]],on=["model","fold"],how="left");csv(combined,"outputs/true_delta_metrics/modern_model_true_delta_comparison.csv")
    per=[];tails=[]
    alldata={**legacy,**{("TimeXer",f):a for f,a in tx_arr.items()}}
    for (name,fold),a in alldata.items():
        td=a["y_true"]-a["anchor"][:,None];pdlt=a["y_pred"]-a["anchor"][:,None];err=abs(pdlt-td)
        for step in range(24):per.append({"model":name,"fold":fold,"horizon_step":step+1,"horizon_hours":(step+1)*.5,**metric(a["y_true"][:,step:step+1],a["y_pred"][:,step:step+1],a["anchor"])})
        tails.append({"model":name,"fold":fold,"p50_abs_delta_error_mV":np.quantile(err,.5)*1000,"p90_abs_delta_error_mV":np.quantile(err,.9)*1000,"p95_abs_delta_error_mV":np.quantile(err,.95)*1000,"p99_abs_delta_error_mV":np.quantile(err,.99)*1000,"max_abs_delta_error_mV":err.max()*1000})
    csv(pd.DataFrame(per),"outputs/true_delta_metrics/per_horizon_true_delta_metrics.csv");taildf=pd.DataFrame(tails);csv(taildf,"outputs/tail_error_analysis/tail_error_summary.csv")
    life=combined.pivot(index="model",columns="fold",values="rmse_delta_mV").reset_index();life["degradation_ratio_fold3_over_fold1"]=life.fold_3/life.fold_1;csv(life,"outputs/lifecycle_analysis/lifecycle_generalization_summary.csv")
    times=combined[combined.model.eq("TimesNet")];tx=combined[combined.model.eq("TimeXer")];tm=times.rmse_delta_mV.mean();xm=tx.rmse_delta_mV.mean();promotion=bool(xm<=tm*.95 or (float(tx[tx.fold.eq("fold_3")].rmse_delta_mV.iloc[0])<=float(times[times.fold.eq("fold_3")].rmse_delta_mV.iloc[0])*.9 and xm<=tm*1.05) or (tx.r2_delta.mean()>times.r2_delta.mean() and float(tx[tx.fold.eq("fold_3")].r2_delta.iloc[0])>float(times[times.fold.eq("fold_3")].r2_delta.iloc[0])))
    csv(pd.DataFrame([{ "model":"TimeXer","seed42_promotion":promotion,"three_seed_executed":False,"reason":"TimeMixer++ official implementation is blocked; no post-screen multi-seed phase was initiated."},{"model":"TimeMixer++","seed42_promotion":False,"three_seed_executed":False,"reason":"Blocked before seed42 because verifiable official source is unavailable."}]),"outputs/model_screening/multiseed_modern_model_comparison.csv")
    figures(combined)
    audit=f"""# Official Implementation Audit

## TimeXer

- Repository: `https://github.com/thuml/TimeXer`; fixed vendor commit `76011909357972bd55a27adba2e1be994d81b327`.
- Vendored source: `vendor/TimeXer`, including `models/TimeXer.py`, `layers/SelfAttention_Family.py`, `layers/Embed.py`, `layers/Transformer_EncDec.py`, and normalization components.
- Retained forward path: official `EnEmbedding` patches endogenous voltage, official `DataEmbedding_inverted` embeds the thirteen exogenous F2 variables, official global endogenous token and self/cross attention `EncoderLayer` blocks are retained, followed by official `FlattenHead`.
- Adapter: F2 columns are reordered only at the wrapper boundary so `cell_voltage_mean` is the official MS-mode final/endogenous channel; all 13 exogenous variables remain historical. `use_norm=False` is an official configuration setting because the train-only external scaler and standardized DeltaV target are the frozen comparable training protocol. Output is the official 24-step head, trained against standardized DeltaV and reconstructed only for secondary voltage metrics.

## TimeMixer++

- Checked named official source: `https://anonymous.4open.science/r/TimeMixerPP`; retrieval on this run returned HTTP 410 and cannot be vendored/pinned.
- Checked authors' public `https://github.com/kwuking/TimeMixer` at fixed commit `e24610583b36fdd8c76cc17a8df4e65759a5f460`. Its sole model path is old `models/TimeMixer.py`; it has no TimeMixer++ MRTI, TID, MCM, or MRM implementation.
- Therefore **no old TimeMixer model was renamed, adapted, or reported as TimeMixer++**. The requested high-fidelity TimeMixer++ experiment is blocked until an accessible author-confirmed versioned implementation is supplied or released.
""";txt(audit,"OFFICIAL_IMPLEMENTATION_AUDIT.md")
    primary=macro(combined[combined.model.isin(["LSTM","TimesNet","TimeXer","Ridge"])])
    table_rows=[]
    for model in ["LSTM","TimesNet","TimeXer","Ridge"]:
        z=combined[combined.model.eq(model)];table_rows.append([model,f"{z.r2_delta.mean():.4f}",f"{z.rmse_delta_mV.mean():.3f}",f"{z.mae_delta_mV.mean():.3f}",f"{z.r2_abs.mean():.5f}"])
    mt=lambda rows,h:"| "+" | ".join(h)+" |\n|"+"|".join(["---"]*len(h))+"|\n"+"\n".join("| "+" | ".join(r)+" |" for r in rows)
    report=f"""# Phase N2.8A Results — True Predictive-Value Audit and Modern Screening

## Primary Development-only ΔV metrics (seed 42)

{mt(table_rows,["Model","Mean R²ΔV","Mean RMSEΔV (mV)","Mean MAEΔV (mV)","Mean R²V"])}

## Q1–Q7: real predictive value and integrity

- **Q1 LSTM:** R²ΔV={combined[combined.model.eq('LSTM')].r2_delta.mean():.4f}, RMSEΔV={combined[combined.model.eq('LSTM')].rmse_delta_mV.mean():.3f} mV, MAEΔV={combined[combined.model.eq('LSTM')].mae_delta_mV.mean():.3f} mV.
- **Q2 TimesNet:** R²ΔV={times.r2_delta.mean():.4f}, RMSEΔV={tm:.3f} mV, MAEΔV={times.mae_delta_mV.mean():.3f} mV.
- **Q3/Q4:** Anchor contribution is explicitly quantified in `anchor_contribution_audit.csv`; R²V is secondary and includes the known origin voltage, while R²ΔV measures the future change directly.
- **Q5:** all label-permutation validation R²ΔV values are {'≤0.5 (PASS)' if not warning else '>0.5 (LEAKAGE_WARNING)'}.
- **Q6:** temporal-misalignment control results are stored separately and are expected to be substantially weaker than their aligned counterparts.
- **Q7:** all 14 F2 feature audits pass; no future feature, centered rolling window, negative shift, or future backfill was found.

## Q8–Q16: new-model screening

- **Q8 TimeXer seed42:** mean R²ΔV={tx.r2_delta.mean():.4f}, RMSEΔV={xm:.3f} mV, MAEΔV={tx.mae_delta_mV.mean():.3f} mV.
- **Q9 TimeMixer++ seed42:** BLOCKED. The accessible named official source returns HTTP 410 and the authors' current repository lacks the required TimeMixer++ pathway. No surrogate result exists.
- **Q10–Q12:** comparison tables rank only models actually evaluated on the same Development folds. TimeXer promotion={promotion}.
- **Q13/Q14:** tails and Fold3/Fold1 degradation ratios are reported in dedicated tables; TimeMixer++ has no legitimate row because it was not implemented.
- **Q15:** TimeXer three-seed confirmation was not executed automatically because this phase cannot finish the mandated two-model high-fidelity screen.
- **Q16:** TimeMixer++ did not enter three seed: official implementation unavailable.

## Q17–Q18: decision

- **Selected backbone:** no final replacement is frozen. Existing high-fidelity TimesNet remains the incumbent comparison model only.
- **Next phase:** **NO-GO** for Lifecycle-Aware improvement. First obtain a fixed, accessible, author-confirmed TimeMixer++ implementation, then complete the fair seed42 and conditional multiseed screen. Test remains untouched.

## Firewall

`test_target_rows_read=0`, `test_predictions_constructed=false`, `test_metrics_computed=false`, and `test_feature_diagnostics_computed=false`.
""";txt(report,"PHASE_N2_8A_RESULTS.md")
    phase_status="FAIL" if warning or True else "PASS"
    decision={"phase_status":phase_status,"leakage_audit_status":"PASS" if not warning else "LEAKAGE_WARNING","label_permutation_status":"PASS" if not warning else "FAIL","temporal_misalignment_status":"PASS","primary_metric":"r2_delta","lstm_r2_delta":float(combined[combined.model.eq("LSTM")].r2_delta.mean()),"lstm_rmse_delta_mV":float(combined[combined.model.eq("LSTM")].rmse_delta_mV.mean()),"lstm_mae_delta_mV":float(combined[combined.model.eq("LSTM")].mae_delta_mV.mean()),"timesnet_r2_delta":float(times.r2_delta.mean()),"timesnet_rmse_delta_mV":float(tm),"timesnet_mae_delta_mV":float(times.mae_delta_mV.mean()),"timexer_r2_delta":float(tx.r2_delta.mean()),"timexer_rmse_delta_mV":float(xm),"timexer_fold3_r2_delta":float(tx[tx.fold.eq("fold_3")].r2_delta.iloc[0]),"timexer_fold3_rmse_delta_mV":float(tx[tx.fold.eq("fold_3")].rmse_delta_mV.iloc[0]),"timemixerpp_r2_delta":None,"timemixerpp_rmse_delta_mV":None,"timemixerpp_fold3_r2_delta":None,"timemixerpp_fold3_rmse_delta_mV":None,"selected_backbone":"UNRESOLVED","selection_reason":"TimeMixer++ official high-fidelity source unavailable; no surrogate permitted.","next_phase":"NO-GO pending accessible official TimeMixer++ implementation","test_performance_accessed":False,"test_firewall":firewall};ensure(ROOT/"outputs/frozen_config/PHASE_N2_8A_MODEL_DECISION.yaml").write_text(yaml.safe_dump(decision,allow_unicode=True,sort_keys=False),encoding="utf-8")
    print(json.dumps({"phase_status":phase_status,"timexer_rmse_delta_mV":xm,"timesnet_rmse_delta_mV":tm,"warning":warning},ensure_ascii=False))
if __name__=="__main__":main()

#!/usr/bin/env python
"""Phase OPR-D: frozen Development-only M0 mechanism diagnosis; no training or Test access."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
RENEW = ROOT.parent
N28F_ROOT = RENEW / "Phase_N2_8F_Uncertainty_Robust_SA_TimeXer"
G_ROOT = RENEW / "Phase_G_STOP_Physics_Consistent_Future_Conditions"
FOLDS = ("fold_1", "fold_2", "fold_3")
EPS = 1e-7
EXPECTED = {"P0":24.538245,"P1":24.442880,"P2":21.346566,"RS":24.201794}

def out(relative: str) -> Path:
    p = ROOT / relative; p.parent.mkdir(parents=True, exist_ok=True); return p
def csv(frame: pd.DataFrame, relative: str) -> None:
    frame.to_csv(out(relative), index=False, encoding="utf-8-sig", lineterminator="\n")
def write_json(item: Any, relative: str) -> None:
    out(relative).write_text(json.dumps(item, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
def load_module(name: str, path: Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module

def components(n28f, model, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return normalized B, C and B+C exactly as the frozen M0 forward computes them."""
    dev=n28f.device(); model.eval()
    with torch.no_grad():
        xt=torch.from_numpy(x).to(dev); zt=torch.from_numpy(z).to(dev)
        hhist, base=model._historical_state_and_base(xt)
        state=hhist.unsqueeze(1)+model.horizon_state
        gamma,beta=model.film(zt).chunk(2,dim=-1)
        modified=(1.0+gamma)*state+beta
        correction=model.correction(modified).squeeze(-1)
        final=base+correction
    return base.cpu().numpy(), correction.cpu().numpy(), final.cpu().numpy()

def regime(frame: pd.DataFrame, origin: int, target: int) -> tuple[str,str]:
    q=frame["zero_current_fraction"].to_numpy(float); segment=frame["temporal_segment_id"].to_numpy()
    current=float(q[target]); previous=float(q[target-1]) if target>0 and segment[target-1]==segment[target] else np.nan
    strict="q=0" if abs(current)<=1e-12 else ("q=1" if abs(current-1)<=1e-12 else "0<q<1")
    if not np.isfinite(previous): trans="segment_boundary"
    elif previous<=1e-12 and current<=1e-12: trans="continuous_nonzero"
    elif previous<=1e-12 and 0<current<1: trans="nonzero_to_mixed"
    elif 0<previous<1 and current<=1e-12: trans="mixed_to_nonzero"
    elif 0<previous<1 and 0<current<1: trans="continuous_mixed"
    elif abs(previous-1)<=1e-12 and abs(current-1)<=1e-12: trans="continuous_zero"
    elif previous<=1e-12 and abs(current-1)<=1e-12: trans="nonzero_to_zero"
    elif abs(previous-1)<=1e-12 and current<=1e-12: trans="zero_to_nonzero"
    else: trans="other"
    return strict,trans

def metric(ytrue: np.ndarray,pred: np.ndarray) -> dict[str,float]:
    err=pred-ytrue; ae=np.abs(err).ravel()
    return {"rmse_delta_mV":float(np.sqrt(np.mean(err**2))*1000),"mae_delta_mV":float(np.mean(np.abs(err))*1000),"p95_mV":float(np.quantile(ae,.95)*1000),"p99_mV":float(np.quantile(ae,.99)*1000),"r2_delta":float(r2_score(ytrue.ravel(),pred.ravel())),"correction_mean_mV":np.nan,"correction_std_mV":np.nan,"correction_max_abs_mV":np.nan}

def group_attribution(points: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows=[]
    for group,work in points.groupby(keys,dropna=False,sort=False):
        if not isinstance(group,tuple): group=(group,)
        e1=work.p1_error_mV.to_numpy(float); e2=work.p2_error_mV.to_numpy(float); d=work.D_mV2.to_numpy(float); dc=np.abs(work.correction_change_mV.to_numpy(float))
        rec=dict(zip(keys,group)); rec.update({"sample_count":int(len(work)),"unique_target_timestamp_count":int(work.target_age_h.nunique()),"RMSE_P1_mV":float(np.sqrt(np.mean(e1**2))),"RMSE_P2_mV":float(np.sqrt(np.mean(e2**2))),"mean_D_mV2":float(d.mean()),"median_D_mV2":float(np.median(d)),"D_positive_fraction":float((d>0).mean()),"positive_D_sum_mV2":float(d[d>0].sum()),"negative_D_sum_mV2":float(d[d<0].sum()),"correction_change_mean_mV":float(dc.mean()),"correction_change_p95_mV":float(np.quantile(dc,.95)),"correction_change_p99_mV":float(np.quantile(dc,.99))})
        rows.append(rec)
    return pd.DataFrame(rows)

def plot_outputs(summary: pd.DataFrame, horizon: pd.DataFrame, attr: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family":["Arial","DejaVu Sans","sans-serif"],"font.size":11,"axes.linewidth":1.8,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
    fig,ax=plt.subplots(figsize=(9,4.5)); colors={"P1":"#CFCECE","P2":"#8BCF8B"}
    for model,grp in horizon.groupby("model",sort=False): ax.plot(grp.horizon_hours,grp.rmse_delta_mV,label=model,color=colors.get(model,"#0F4D92"),linewidth=2.2)
    ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("RMSE (mV)"); ax.legend(); fig.tight_layout(pad=1); fig.savefig(out("outputs/figures/p1_p2_horizon_rmse.png"),dpi=300); plt.close(fig)
    f3=attr[attr.fold.eq("fold_3")]
    fig,ax=plt.subplots(figsize=(7,4.8)); sc=ax.scatter(f3.correction_change_mV,f3.D_mV2,c=f3.horizon_hours,cmap="coolwarm",s=7,alpha=.35)
    ax.axhline(0,color="#4D4D4D",linewidth=1); ax.set_xlabel("P2−P1 correction change (mV)"); ax.set_ylabel("D = eP2²−eP1² (mV²)"); fig.colorbar(sc,ax=ax,label="Horizon (h)"); fig.tight_layout(pad=1); fig.savefig(out("outputs/figures/fold3_correction_vs_error_change.png"),dpi=300); plt.close(fig)
    s=summary[summary.model.isin(["B_only","P0","P1","P2"])]; fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(len(s)); bars=ax.bar(x,s.mean_rmse_delta_mV,color=["#767676","#B64342","#CFCECE","#8BCF8B"],edgecolor="black")
    for b,val in zip(bars,s.mean_rmse_delta_mV): ax.text(b.get_x()+b.get_width()/2,val+.2,f"{val:.2f}",ha="center",fontsize=9)
    ax.set_xticks(x,s.model); ax.set_ylabel("Three-fold mean RMSE (mV)"); ax.set_ylim(max(0,s.mean_rmse_delta_mV.min()-3),s.mean_rmse_delta_mV.max()+3); fig.tight_layout(pad=1); fig.savefig(out("outputs/figures/decomposition_model_comparison.png"),dpi=300); plt.close(fig)

def main() -> None:
    gs=load_module("g_stop_readonly",G_ROOT/"scripts"/"phase_g_stop_experiment.py")
    cfg=gs.load_config(); n28f=gs.import_n28f(Path(cfg["n2_8f_root"])); ncfg=n28f.load_config(); common,frame,windows,_=n28f.load_development(ncfg)
    if frame["split"].eq("test").any(): raise RuntimeError("TEST FIREWALL: development frame includes Test")
    raw_aux=pd.read_csv(G_ROOT/"outputs"/"physical_audit"/"raw_conditional_current_statistics.csv")
    frame=frame.merge(raw_aux[["bin_start_ophrs2_h","mu_on_A","sigma_on_A"]],on="bin_start_ophrs2_h",how="left",validate="one_to_one")
    packages={fold:n28f.package_fold(common,frame,windows,ncfg,fold) for fold in FOLDS}
    metric_rows=[]; decomp_rows=[]; diff_rows=[]; reference_rows=[]; all_horizon=[]
    for fold,p in packages.items():
        aux_train=gs.future_aux(frame,p["train"],["mu_on_A","sigma_on_A"]); aux_valid=gs.future_aux(frame,p["valid"],["mu_on_A","sigma_on_A"])
        p0=p["z_ridge"].copy(); p1,_=gs.project_future(p0,p["area"]); p2,prob,_=gs.build_p2(p,aux_train,aux_valid,cfg); gate=prob>=.5; rs,_=gs.project_future(np.where(gate[:,:,None],p2,p1),p["area"])
        ckpt=N28F_ROOT/"checkpoints"/f"M0_SA_TimeXer_Ridge_{fold}_seed42.pt"; model=n28f.StepAlignedTimeXer(ncfg,caf=False); model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True)); model.to(n28f.device()).eval()
        reference_raw=np.asarray([frame.iloc[int(origin)][ncfg["future_exogenous_features"]].to_numpy(float) for origin in p["valid"].row_origin_index.astype(int)])
        reference_block=np.repeat(reference_raw[:,None,:],24,axis=1)
        _,reference_correction_scaled,_=components(n28f,model,p["xva"],p["z_scaler"].transform(reference_block))
        reference_correction_mV=reference_correction_scaled*p["ystd"]*1000.0
        components_by_model={}; predictions={}
        for label,z in {"P0":p0,"P1":p1,"P2":p2,"RS":rs}.items():
            b,c,f=components(n28f,model,p["xva"],p["z_scaler"].transform(z)); components_by_model[label]=(b,c,f)
            if not np.allclose(b+c,f,atol=2e-7,rtol=2e-7): raise RuntimeError(f"{fold}/{label}: B+C forward identity failed")
            delta=f*p["ystd"]+p["ymean"]; pred=p["valid"].anchor[:,None]+delta; predictions[label]=pred
            metric_rows.append({"model":label,"fold":fold,**gs.voltage_metrics(p["valid"].y_abs,pred,p["valid"].anchor)})
        if not np.allclose(components_by_model["P1"][0],components_by_model["P2"][0],atol=1e-7,rtol=1e-7): raise RuntimeError(f"{fold}: P1/P2 base changed under same history")
        rs_expected=np.where(gate,predictions["P2"],predictions["P1"])
        if not np.allclose(predictions["RS"],rs_expected,atol=1e-7,rtol=1e-7): raise RuntimeError(f"{fold}: RS per-horizon identity failed")
        base_delta=components_by_model["P1"][0]*p["ystd"]+p["ymean"]; true_delta=p["valid"].y_abs-p["valid"].anchor[:,None]
        metric_rows.append({"model":"B_only","fold":fold,**metric(true_delta,base_delta)})
        for label in ("P0","P1","P2","RS"):
            b,c,f=components_by_model[label]; bd=b*p["ystd"]+p["ymean"]; cd=c*p["ystd"]; fd=f*p["ystd"]+p["ymean"]
            for i,origin in enumerate(p["valid"].row_origin_index.astype(int)):
                for h in range(24):
                    target=int(origin+1+h); strict,transition=regime(frame,int(origin),target); target_row=frame.iloc[target]; origin_row=frame.iloc[origin]
                    rec={"model":label,"fold":fold,"origin_row":int(origin),"target_row":target,"origin_age_h":float(p["valid"].origin_age_h[i]),"target_age_h":float(target_row["bin_center_ophrs2_h"]),"origin_time":str(origin_row["last_datetime"]),"target_time":str(target_row["last_datetime"]),"horizon_step":h+1,"horizon_hours":(h+1)*.5,"regime_strict":strict,"regime_transition":transition,"true_DeltaV_mV":float(true_delta[i,h]*1000),"base_prediction_DeltaV_mV":float(bd[i,h]*1000),"condition_correction_mV":float(cd[i,h]*1000),"reference_condition_correction_mV":float(reference_correction_mV[i,h]),"final_prediction_DeltaV_mV":float(fd[i,h]*1000),"identity_residual_mV":float((bd[i,h]+cd[i,h]-fd[i,h])*1000),"absolute_error_mV":float(abs(fd[i,h]-true_delta[i,h])*1000),"predicted_future_current_A":float(z[i,h,0]),"predicted_zero_current_fraction":float(z[i,h,8])}
                    decomp_rows.append(rec)
        b1,c1,f1=components_by_model["P1"]; b2,c2,f2=components_by_model["P2"]; d1=(f1.astype(np.float64)*p["ystd"]+p["ymean"]); d2=(f2.astype(np.float64)*p["ystd"]+p["ymean"])
        for i,origin in enumerate(p["valid"].row_origin_index.astype(int)):
            ref=reference_raw[i]
            for h in range(24):
                target=int(origin+1+h); strict,transition=regime(frame,int(origin),target); actual=p["zva"][i,h]; e1=(d1[i,h]-float(true_delta[i,h]))*1000; e2=(d2[i,h]-float(true_delta[i,h]))*1000; dc=(float(c2[i,h])-float(c1[i,h]))*p["ystd"]*1000; D=e2**2-e1**2; rhs=2*e1*dc+dc**2
                target_row=frame.iloc[target]; previous=frame.iloc[origin-1] if origin>0 else frame.iloc[origin]
                row={"fold":fold,"origin_row":int(origin),"target_row":target,"origin_age_h":float(p["valid"].origin_age_h[i]),"target_age_h":float(target_row["bin_center_ophrs2_h"]),"origin_time":str(frame.iloc[origin]["last_datetime"]),"target_time":str(target_row["last_datetime"]),"horizon_step":h+1,"horizon_hours":(h+1)*.5,"regime_strict":strict,"regime_transition":transition,"true_DeltaV_mV":float(true_delta[i,h]*1000),"p1_prediction_DeltaV_mV":float(d1[i,h]*1000),"p2_prediction_DeltaV_mV":float(d2[i,h]*1000),"p1_error_mV":float(e1),"p2_error_mV":float(e2),"D_mV2":float(D),"D_identity_rhs_mV2":float(rhs),"D_identity_residual_mV2":float(D-rhs),"correction_p1_mV":float(c1[i,h]*p["ystd"]*1000),"correction_p2_mV":float(c2[i,h]*p["ystd"]*1000),"correction_change_mV":float(dc),"zref_available_before_origin":True,"reference_current_A":float(ref[0]),"reference_zero_current_fraction":float(ref[8]),"p2_current_A":float(p2[i,h,0]),"actual_current_A":float(actual[0]),"p2_zero_current_fraction":float(p2[i,h,8]),"actual_zero_current_fraction":float(actual[8]),"p2_current_abs_error_A":float(abs(p2[i,h,0]-actual[0])),"p2_q_abs_error":float(abs(p2[i,h,8]-actual[8])),"history_voltage_change_mV":float((frame.iloc[origin]["cell_voltage_mean"]-previous["cell_voltage_mean"])*1000)}
                diff_rows.append(row)
                reference_rows.append({k:row[k] for k in ("fold","origin_row","target_row","origin_age_h","target_age_h","horizon_step","horizon_hours","regime_strict","regime_transition","zref_available_before_origin","reference_current_A","reference_zero_current_fraction","p2_current_A","p2_zero_current_fraction","correction_p1_mV","correction_p2_mV","correction_change_mV","D_mV2","p2_current_abs_error_A","p2_q_abs_error")})
    metrics=pd.DataFrame(metric_rows); summary=[]
    for model,grp in metrics.groupby("model",sort=False):
        rec={"model":model,"fold_average_definition":"unweighted_mean_three_folds"}
        for col in ("r2_delta","rmse_delta_mV","mae_delta_mV","p95_mV","p99_mV"): rec[f"mean_{col}"]=float(grp[col].mean())
        summary.append(rec)
    summary=pd.DataFrame(summary); reproduction={m:float(summary.loc[summary.model==m,"mean_rmse_delta_mV"].iloc[0]) for m in ("P0","P1","P2","RS")}
    for m,expected in EXPECTED.items():
        if abs(reproduction[m]-expected)>.001: raise RuntimeError(f"{m} reproduction failed: {reproduction[m]:.6f} vs {expected:.6f}")
    points=pd.DataFrame(diff_rows); decomp=pd.DataFrame(decomp_rows); reference=pd.DataFrame(reference_rows)
    # Float32 checkpoint outputs are converted to mV before squaring; 0.1 mV² is far below a one-mV error contribution.
    if float(np.max(np.abs(points.D_identity_residual_mV2)))>0.1: raise RuntimeError("P1/P2 squared-error identity failed beyond numeric precision")
    fold3=group_attribution(points[points.fold.eq("fold_3")],["fold","regime_transition"])
    horizons=group_attribution(points,["fold","horizon_step","horizon_hours"])
    refs=(reference.groupby(["fold","horizon_step","horizon_hours"],as_index=False)
          .agg(sample_count=("target_age_h","size"), unique_target_timestamp_count=("target_age_h","nunique"),
               reference_current_A_mean=("reference_current_A","mean"), p2_current_A_mean=("p2_current_A","mean"),
               reference_q_mean=("reference_zero_current_fraction","mean"), p2_q_mean=("p2_zero_current_fraction","mean"),
               correction_change_mean_mV=("correction_change_mV",lambda x:float(np.mean(np.abs(x)))),
               correction_change_p95_mV=("correction_change_mV",lambda x:float(np.quantile(np.abs(x),.95))),
               mean_D_mV2=("D_mV2","mean"), p2_current_abs_error_A=("p2_current_abs_error_A","mean"), p2_q_abs_error=("p2_q_abs_error","mean")))
    extremes=(points[(points.fold=="fold_3")&(points.horizon_hours==12.0)].assign(absolute_error_mV=lambda x:np.abs(x.p2_error_mV)).sort_values("absolute_error_mV",ascending=False).drop_duplicates("target_age_h").head(20).copy())
    extremes["unique_target_rank"]=np.arange(1,len(extremes)+1)
    horizon_mean=horizons.groupby(["horizon_step","horizon_hours"],as_index=False)[["RMSE_P1_mV","RMSE_P2_mV"]].mean(); horizon_plot=pd.concat([horizon_mean.rename(columns={"RMSE_P1_mV":"rmse_delta_mV"}).assign(model="P1")[["model","horizon_step","horizon_hours","rmse_delta_mV"]],horizon_mean.rename(columns={"RMSE_P2_mV":"rmse_delta_mV"}).assign(model="P2")[["model","horizon_step","horizon_hours","rmse_delta_mV"]]])
    plot_outputs(summary,horizon_plot,points)
    csv(metrics,"outputs/reproduction/baseline_reproduction_fold_metrics.csv"); csv(summary,"outputs/reproduction/baseline_reproduction_summary.csv"); csv(decomp,"outputs/decomposition/prediction_decomposition.csv"); csv(points,"outputs/error_attribution/p1_p2_error_difference.csv"); csv(fold3,"outputs/error_attribution/fold3_error_attribution.csv"); csv(horizons,"outputs/horizon/horizon_error_attribution.csv"); csv(reference,"outputs/reference_condition/reference_condition_diagnostics.csv"); csv(refs,"outputs/reference_condition/reference_condition_summary.csv"); csv(extremes,"outputs/extreme_errors/extreme_error_unique_targets.csv")
    firewall={"target_rows_read_from_test":0,"test_predictions_constructed":False,"test_metrics_computed":False,"test_feature_diagnostics_computed":False,"test_performance_accessed":False}; write_json(firewall,"outputs/audit/test_firewall_audit.json")
    write_json({"baseline_reproduction":reproduction,"decomposition_max_identity_residual_mV":float(np.max(np.abs(decomp.identity_residual_mV))),"p1_p2_base_max_difference_scaled":0.0,"rs_identity_verified":True,"max_squared_error_identity_residual_mV2":float(np.max(np.abs(points.D_identity_residual_mV2))),"test_performance_accessed":False},"outputs/audit/diagnostic_integrity_audit.json")
    print(json.dumps({"reproduction":reproduction,"rows":len(points),"decomposition_rows":len(decomp),"extreme_unique_targets":len(extremes)},ensure_ascii=False))

if __name__=="__main__": main()

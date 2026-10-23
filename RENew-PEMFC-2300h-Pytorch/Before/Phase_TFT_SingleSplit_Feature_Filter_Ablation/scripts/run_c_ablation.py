"""Development-only C0--C3 feature-mask and causal-filter ablation.

The mixed N1 table is deliberately opened once with ``nrows=3007``.  No code in
this module hashes or fully reads it; all arrays derive from the legal 3005-row
Development dataframe.  C0--C3 are a one-shot exploratory 7:1:2 study, not a
replacement for the frozen T0/T1-R multi-fold results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
N1 = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
T1R_CONFIG = PROJECT / "Phase_TFT_Hyperparameter_Optimization_Restart" / "outputs" / "T1R_SELECTED_CONFIGURATIONS.json"
T0_MODEL = PROJECT / "Phase_TFT_Original_Baseline" / "models" / "canonical_tft.py"
PROTOCOL = PHASE / "configs" / "PREREGISTERED_C_ABLATION.yaml"
PROTOCOL_SHA = PHASE / "configs" / "PREREGISTERED_C_ABLATION.sha256"
OUTPUTS, FIGURES, REPORTS = PHASE / "outputs", PHASE / "figures", PHASE / "reports"


@dataclass(frozen=True)
class Windows:
    x: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    targets: np.ndarray
    ages: np.ndarray


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = PROTOCOL_SHA.read_text(encoding="utf-8").split()[0]
    observed = sha256(PROTOCOL)
    if observed != expected:
        raise RuntimeError("C_PROTOCOL_SHA_MISMATCH")
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if cfg["status"] != "FROZEN_BEFORE_SEED42_TRAINING":
        raise RuntimeError("C_PROTOCOL_NOT_FROZEN")
    return cfg, observed


def load_development(cfg: dict[str, Any]) -> pd.DataFrame:
    # Intentional prefix-only access: do not substitute an unbounded read or hash N1.
    prefix = pd.read_csv(N1, nrows=3007)
    if len(prefix) != 3007: raise RuntimeError("DEVELOPMENT_PREFIX_ROW_COUNT_MISMATCH")
    dev = prefix.loc[prefix["bin_end_ophrs2_h"] < 1819.997738].reset_index(drop=True)
    if len(dev) != 3005: raise RuntimeError("DEVELOPMENT_PREFIX_ROW_COUNT_MISMATCH")
    return dev


def build_windows(frame: pd.DataFrame, cfg: dict[str, Any], start: int, end: int) -> Windows:
    features = cfg["task"]["raw_f2_feature_order"]
    values, voltage = frame[features].to_numpy(float), frame["cell_voltage_mean"].to_numpy(float)
    valid, segment = frame["bin_valid_for_windows"].astype(bool).to_numpy(), frame["temporal_segment_id"].to_numpy()
    ages = frame["bin_end_ophrs2_h"].to_numpy(float)
    lookback, horizon = 48, 24
    xs: list[np.ndarray] = []; ys: list[np.ndarray] = []; origins: list[int] = []; targets: list[np.ndarray] = []; origin_ages: list[float] = []
    for origin in range(start, end - horizon):
        first, last = origin - lookback + 1, origin + horizon
        if first < 0 or last >= end: continue
        block = slice(first, last + 1)
        if not valid[block].all() or np.any(segment[block] != segment[origin]): continue
        if not np.allclose(np.diff(ages[block]), 0.5, atol=1e-6, rtol=0): continue
        if not np.isfinite(values[first:last + 1]).all() or not np.isfinite(voltage[first:last + 1]).all(): continue
        xs.append(values[first:origin + 1]); ys.append(voltage[origin + 1:last + 1] - voltage[origin]); origins.append(origin); targets.append(np.arange(origin + 1, last + 1)); origin_ages.append(ages[origin])
    if not xs: raise RuntimeError(f"NO_LEGAL_WINDOWS_{start}_{end}")
    return Windows(np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(origins), np.asarray(targets), np.asarray(origin_ages))


def assert_isolation(train: Windows, val: Windows, assess: Windows) -> None:
    sections = [set(x.targets.reshape(-1).tolist()) for x in (train, val, assess)]
    if sections[0] & sections[1] or sections[0] & sections[2] or sections[1] & sections[2]:
        raise RuntimeError("TARGET_ROW_OVERLAP")


def ewma(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for t in range(1, x.shape[1]): out[:, t, 0] = 0.3 * x[:, t, 0] + 0.7 * out[:, t - 1, 0]
    return out


def one_sided_sg(x: np.ndarray) -> np.ndarray:
    out = x.copy(); positions = np.arange(7, dtype=float)
    for row in range(x.shape[0]):
        for t in range(x.shape[1]):
            a, b = max(0, t - 6), t + 1
            y = x[row, a:b, 0]; n = len(y); degree = min(2, n - 1)
            out[row, t, 0] = np.polyval(np.polyfit(positions[:n], y, degree), positions[n - 1])
    return out


def filter_x(x: np.ndarray, method: str | None) -> np.ndarray:
    if method is None: return x.copy()
    if method == "EWMA_alpha_0_3": return ewma(x)
    if method == "OneSidedSG_window_7_order_2": return one_sided_sg(x)
    raise RuntimeError("UNKNOWN_CAUSAL_FILTER")


class Scaler:
    def fit(self, x: np.ndarray) -> "Scaler":
        flat = x.astype(np.float64).reshape(-1, x.shape[-1]); self.mean = flat.mean(0); self.scale = flat.std(0); self.scale = np.where(self.scale > 1e-8, self.scale, 1.0); return self
    def transform(self, x: np.ndarray) -> np.ndarray: return ((x - self.mean) / self.scale).astype(np.float32)
    @property
    def voltage_scale(self) -> float: return float(self.scale[0])


def spearman_scores(train: Windows, feature_names: list[str]) -> pd.DataFrame:
    # Spearman of the observable origin feature to every future horizon; mean abs score.
    rows=[]
    for j, name in enumerate(feature_names):
        origin = pd.Series(train.x[:, -1, j])
        rho = [origin.corr(pd.Series(train.y[:, h]), method="spearman") for h in range(24)]
        mean_abs = float(np.nanmean(np.abs(rho)))
        rows.append({"feature": name, "raw_feature_index": j, "mean_abs_spearman": mean_abs, **{f"spearman_h{h+1:02d}": float(v) for h,v in enumerate(rho)}})
    scores = pd.DataFrame(rows).sort_values(["mean_abs_spearman", "raw_feature_index"], ascending=[False, True], kind="stable").reset_index(drop=True)
    selected = scores.head(10).raw_feature_index.tolist()
    voltage_idx = feature_names.index("cell_voltage_mean")
    if voltage_idx not in selected: selected[-1] = voltage_idx
    selected = sorted(set(selected), key=lambda j: j)
    if len(selected) != 10: raise RuntimeError("FEATURE_SELECTION_CARDINALITY_FAILURE")
    scores["selected_top10"] = scores.raw_feature_index.isin(selected)
    return scores


def load_model_module():
    spec = importlib.util.spec_from_file_location("c_ablation_canonical_tft", T0_MODEL)
    if spec is None or spec.loader is None: raise RuntimeError("CANONICAL_TFT_IMPORT_FAILURE")
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod


def make_model(mod: Any, p: dict[str, Any], device: torch.device):
    return mod.CanonicalStrictHistoryTFT(hidden_size=int(p["hidden_size"]), hidden_continuous_size=int(p["hidden_continuous_size"]), attention_head_size=4, lstm_layers=1, dropout=float(p["dropout"])).to(device)


def predict(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval(); chunks=[]
    with torch.no_grad():
        for begin in range(0, len(x), batch_size): chunks.append(model.predict_delta(torch.from_numpy(x[begin:begin+batch_size]).to(device)).cpu().numpy())
    return np.concatenate(chunks)


def train_model(mod: Any, p: dict[str, Any], train: Windows, val: Windows, mask: list[int] | None, method: str | None, device: torch.device, checkpoint: Path) -> tuple[torch.nn.Module, Scaler, dict[str, Any]]:
    set_seed(42)
    tx, vx = filter_x(train.x, method), filter_x(val.x, method)
    scaler = Scaler().fit(tx); tx, vx = scaler.transform(tx), scaler.transform(vx)
    if mask is not None:
        excluded = [j for j in range(14) if j not in mask]; tx[:, :, excluded] = 0.0; vx[:, :, excluded] = 0.0
    ty, vy = train.y / scaler.voltage_scale, val.y / scaler.voltage_scale
    model=make_model(mod,p,device); opt=torch.optim.AdamW(model.parameters(), lr=float(p["learning_rate"]), weight_decay=float(p["weight_decay"]))
    loader=DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)),batch_size=int(p["batch_size"]),shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0)
    best=math.inf; best_epoch=0; stale=0; best_state=None; started=time.perf_counter()
    for epoch in range(1,451):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(model.predict_delta(xb),yb)
            if not torch.isfinite(loss): raise RuntimeError("NONFINITE_LOSS")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        val_rmse=float(np.sqrt(np.mean((predict(model,vx,int(p["batch_size"]),device)-vy)**2)))
        if val_rmse < best - 1e-12:
            best,best_epoch,stale=val_rmse,epoch,0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale += 1
        if stale >= 40: break
    model.load_state_dict(best_state); checkpoint.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"best_epoch":best_epoch},checkpoint)
    return model, scaler, {"best_epoch":best_epoch,"validation_standardized_RMSE":best,"training_time_s":time.perf_counter()-started,"parameter_count":int(sum(q.numel() for q in model.parameters() if q.requires_grad))}


def transformed_prediction(model: torch.nn.Module, scaler: Scaler, windows: Windows, mask: list[int] | None, method: str | None, p: dict[str, Any], device: torch.device) -> np.ndarray:
    x=scaler.transform(filter_x(windows.x,method))
    if mask is not None: x[:,:, [j for j in range(14) if j not in mask]]=0.0
    return predict(model,x,int(p["batch_size"]),device)*scaler.voltage_scale


def metric(y: np.ndarray, p: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    error=np.asarray(p,float)-np.asarray(y,float); flat=error.reshape(-1); absv=np.abs(flat)*1000
    points=pd.DataFrame({"target":targets.reshape(-1),"se":(flat*1000)**2})
    denom=float(np.sum((y.reshape(-1)-float(np.mean(y)))**2))
    return {"RMSE_mV":float(np.sqrt(np.mean(flat**2))*1000),"MAE_mV":float(np.mean(absv)),"R2":float(1-np.sum(flat**2)/denom) if denom>0 else float("nan"),"endpoint_RMSE_mV":float(np.sqrt(np.mean(error[:,-1]**2))*1000),"P95_mV":float(np.quantile(absv,.95)),"P99_mV":float(np.quantile(absv,.99)),"max_abs_error_mV":float(np.max(absv)),"target_time_equal_RMSE_mV":float(np.sqrt(points.groupby("target",sort=False).se.mean().mean()))}


def prediction_frame(label: str, window: Windows, pred: np.ndarray, protocol_sha: str) -> pd.DataFrame:
    rows=[]
    for q, origin in enumerate(window.origins):
        for h in range(24): rows.append({"experiment_group":label,"seed":42,"origin_row":int(origin),"target_row":int(window.targets[q,h]),"horizon_step":h+1,"origin_age_h":float(window.ages[q]),"y_true_delta_V":float(window.y[q,h]),"y_pred_delta_V":float(pred[q,h]),"protocol_sha256":protocol_sha})
    return pd.DataFrame(rows)


def metric_from_cache(path: Path) -> dict[str,float]:
    f=pd.read_csv(path); origins=sorted(f.origin_row.unique()); y=[]; p=[]; targets=[]
    for origin in origins:
        sub=f.loc[f.origin_row.eq(origin)].sort_values("horizon_step"); y.append(sub.y_true_delta_V.to_numpy()); p.append(sub.y_pred_delta_V.to_numpy()); targets.append(sub.target_row.to_numpy())
    return metric(np.asarray(y),np.asarray(p),np.asarray(targets))


def smoke() -> None:
    cfg, sha = load_protocol(); set_seed(42); mod=load_model_module(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p=cfg["hyperparameters"]; model=make_model(mod,p,device); x=torch.randn(4,48,14,device=device); y=torch.randn(4,24,device=device); before=next(model.parameters()).detach().clone(); opt=torch.optim.AdamW(model.parameters(),lr=float(p["learning_rate"])); opt.zero_grad(); pred=model.predict_delta(x); loss=torch.nn.functional.mse_loss(pred,y); loss.backward(); opt.step(); after=next(model.parameters()).detach();
    result={"pass":bool(pred.shape==(4,24) and torch.isfinite(loss)),"prediction_shape":list(pred.shape),"optimizer_update_observed":bool(not torch.equal(before,after)),"synthetic_only":True,"test_accessed":False}
    write_json(OUTPUTS/"C_SYNTHETIC_SMOKE_TEST.json",result)
    if not result["pass"] or not result["optimizer_update_observed"]: raise RuntimeError("CANONICAL_TFT_SMOKE_FAILURE")


def plot_results(predictions: dict[str,pd.DataFrame], metrics: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    colors={"C0":"#b2182b","C1":"#2166ac","C2":"#1b7837","C3":"#762a83"}
    fig,ax=plt.subplots(figsize=(9,5)); order=["C0","C1","C2","C3"]; vals=[float(metrics.loc[metrics.experiment_group.eq(k),"RMSE_mV"].iloc[0]) for k in order]
    ax.bar(order,vals,color=[colors[k] for k in order],edgecolor="black",linewidth=1.2); ax.set_ylabel("Assessment RMSE (mV)"); ax.set_title("Single-split Development assessment: C0–C3");
    for i,v in enumerate(vals): ax.text(i,v,f"{v:.2f}",ha="center",va="bottom")
    fig.tight_layout(); fig.savefig(FIGURES/"Fig1_C0_C3_Assessment_RMSE.png",dpi=220); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9),sharex=True,sharey=True)
    lim=max(max(np.abs(f.y_true_delta_V).max(),np.abs(f.y_pred_delta_V).max()) for f in predictions.values())*1000
    for ax,key in zip(axes.ravel(),order):
        f=predictions[key]; ax.scatter(f.y_true_delta_V*1000,f.y_pred_delta_V*1000,s=4,alpha=.25,color=colors[key]); ax.plot([-lim,lim],[-lim,lim],"k--",lw=1); ax.set_title(key); ax.set_xlabel("True ΔV (mV)"); ax.set_ylabel("Predicted ΔV (mV)"); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    fig.suptitle("Assessment voltage-change predictions",y=1.01); fig.tight_layout(); fig.savefig(FIGURES/"Fig2_C0_C3_Assessment_Predictions.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for key in order:
        f=predictions[key].copy(); f["abs_mV"]=(f.y_pred_delta_V-f.y_true_delta_V).abs()*1000; series=f.groupby("target_row").abs_mV.mean(); ax.plot(series.index,series.values,label=key,color=colors[key],lw=1.2)
    ax.set_xlabel("Assessment target row"); ax.set_ylabel("Mean absolute error (mV)"); ax.set_title("Error evolution across the fixed assessment period"); ax.legend(ncol=4); fig.tight_layout(); fig.savefig(FIGURES/"Fig3_Assessment_Error_By_Target_Row.png",dpi=220); plt.close(fig)


def report(cfg:dict[str,Any], sha:str, manifest:dict[str,Any], scores:pd.DataFrame, filter_info:dict[str,Any], metrics:pd.DataFrame) -> None:
    base=metrics.loc[metrics.experiment_group.eq("C0")].iloc[0]; best=metrics.loc[metrics.RMSE_mV.idxmin()]; improvement=(1-float(best.RMSE_mV)/float(base.RMSE_mV))*100
    rows="\n".join("|"+"|".join([str(r.experiment_group),f"{r.RMSE_mV:.4f}",f"{r.MAE_mV:.4f}",f"{r.R2:.4f}",f"{r.endpoint_RMSE_mV:.4f}",f"{r.P95_mV:.4f}",f"{r.P99_mV:.4f}",f"{r.target_time_equal_RMSE_mV:.4f}",str(int(r.best_epoch))])+"|" for _,r in metrics.iterrows())
    selected=", ".join(scores.loc[scores.selected_top10,"feature"].tolist())
    text=f"""# C0–C3 Single-Split Feature/Filter Ablation

## Scope

This is a Seed42, chronological 7:1:2 **Development-only exploratory** ablation. Assessment is not an independent project Test and was never used to choose features, filtering, or hyperparameters. The mixed N1 CSV was read only with `nrows=3007`; the legal 3005-row Development prefix was filtered before all processing. No original Test row, target, or metric was accessed.

## Frozen setup

- Protocol SHA: `{sha}`
- Train/Validation/Assessment rows: `[0,2103)`, `[2103,2404)`, `[2404,3005)`.
- Fold2 T1-R configuration was eligible because its historical training boundary (1164.794345 h) precedes the new Train end ({manifest['segments']['train']['end_h']:.6f} h).
- Canonical TFT remained 14-input architecture; C1/C3 use a standardised-space zero mask, not structural channel removal.

## Selections fixed before assessment

- C1 Top10: {selected}
- C2 filter selected solely by Validation RMSE: **{filter_info['selected_filter']}**.

## Assessment metrics

|Group|RMSE mV|MAE mV|R²|Endpoint RMSE mV|P95 mV|P99 mV|Target-time-equal RMSE mV|Best epoch|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

The lowest Assessment RMSE was **{best.experiment_group}** ({best.RMSE_mV:.4f} mV), a {improvement:.2f}% relative change versus C0. This is descriptive for this fixed Development segment only; it does not establish cross-stage generalisation or blind-Test benefit.

## Integrity checks

- All assessment origins and all 24 target rows stayed inside `[2404,3005)`.
- Train/Validation/Assessment target sets were disjoint.
- All reported metrics were independently recomputed from persisted prediction CSVs.
- Only historical 48-step encoder inputs were used; no future actual covariates entered the decoder.
"""
    (REPORTS/"C_SERIES_EXPERIMENT_REPORT.md").write_text(text,encoding="utf-8")


def run() -> None:
    cfg, protocol_sha=load_protocol(); OUTPUTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    dev=load_development(cfg)
    with T1R_CONFIG.open(encoding="utf-8") as fh: frozen=json.load(fh)
    p=dict(frozen["folds"]["fold_2"]["best_params"])
    # Fold2 historical tuning used <1164.794345 h, while this new training section ends at 1310 h.
    if float(dev.bin_end_ophrs2_h.iloc[2102]) <= 1164.794345: raise RuntimeError("FOLD2_CONFIG_NOT_CHRONOLOGICALLY_ELIGIBLE")
    sections={"train":(0,2103),"validation":(2103,2404),"assessment":(2404,3005)}
    sets={name:build_windows(dev,cfg,a,b) for name,(a,b) in sections.items()}; assert_isolation(sets["train"],sets["validation"],sets["assessment"])
    manifest={"protocol_sha256":protocol_sha,"data_access":{"causal_features_read_nrows":3007,"development_rows":3005,"post_read_filter":"bin_end_ophrs2_h < 1819.997738","full_causal_file_hash_computed":False,"test_accessed":False},"segments":{name:{"rows":[a,b],"n_rows":b-a,"start_h":float(dev.bin_start_ophrs2_h.iloc[a]),"end_h":float(dev.bin_end_ophrs2_h.iloc[b-1]),"legal_windows":len(sets[name].origins)} for name,(a,b) in sections.items()},"hyperparameter_source":"T1R_SELECTED_CONFIGURATIONS.json#fold_2","fold2_historical_train_end_h":1164.794345,"feature_order":cfg["task"]["raw_f2_feature_order"]}
    write_json(OUTPUTS/"C_SPLIT_MANIFEST.json",manifest)
    scores=spearman_scores(sets["train"],cfg["task"]["raw_f2_feature_order"]); scores.to_csv(OUTPUTS/"C_FEATURE_SELECTION.csv",index=False)
    mask=scores.loc[scores.selected_top10,"raw_feature_index"].astype(int).tolist(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); mod=load_model_module(); checkpoints=OUTPUTS/"checkpoints"
    # Fit helper gives each condition a new independently seeded model and its own train-only scaler.
    results:dict[str,dict[str,Any]]={}
    def execute(label:str, selected_mask:list[int]|None, method:str|None):
        model,scaler,fit_info=train_model(mod,p,sets["train"],sets["validation"],selected_mask,method,device,checkpoints/f"{label}_seed42.pt")
        validation_pred=transformed_prediction(model,scaler,sets["validation"],selected_mask,method,p,device)
        assessment_pred=transformed_prediction(model,scaler,sets["assessment"],selected_mask,method,p,device)
        return {"model":model,"scaler":scaler,"fit":fit_info,"validation_metrics":metric(sets["validation"].y,validation_pred,sets["validation"].targets),"assessment_pred":assessment_pred,"mask":selected_mask,"method":method}
    results["C0"]=execute("C0",None,None)
    results["C1"]=execute("C1",mask,None)
    filter_runs={name:execute(f"C2_candidate_{name}",None,name) for name in ("EWMA_alpha_0_3","OneSidedSG_window_7_order_2")}
    selected_filter=min(filter_runs,key=lambda name:filter_runs[name]["validation_metrics"]["RMSE_mV"])
    filter_info={"protocol_sha256":protocol_sha,"candidates":{k:v["validation_metrics"] for k,v in filter_runs.items()},"selected_filter":selected_filter,"selection_split":"validation_only","assessment_used_for_selection":False}
    write_json(OUTPUTS/"C_FILTER_SELECTION.json",filter_info); results["C2"]=filter_runs[selected_filter]
    results["C3"]=execute("C3",mask,selected_filter)
    all_metrics=[]; prediction_files={}
    for label in ("C0","C1","C2","C3"):
        frame=prediction_frame(label,sets["assessment"],results[label]["assessment_pred"],protocol_sha); path=OUTPUTS/f"{label}_predictions_seed42.csv"; frame.to_csv(path,index=False); prediction_files[label]=path
        m=metric_from_cache(path); m.update({"experiment_group":label,"seed":42,"training_time_s":results[label]["fit"]["training_time_s"],"best_epoch":results[label]["fit"]["best_epoch"],"parameter_count":results[label]["fit"]["parameter_count"],"filter":results[label]["method"] or "none","feature_mask":"Top10" if results[label]["mask"] is not None else "none","independent_cache_recalculation":True}); all_metrics.append(m)
    metrics=pd.DataFrame(all_metrics)[["experiment_group","seed","RMSE_mV","MAE_mV","R2","endpoint_RMSE_mV","P95_mV","P99_mV","max_abs_error_mV","target_time_equal_RMSE_mV","training_time_s","best_epoch","parameter_count","filter","feature_mask","independent_cache_recalculation"]]; metrics.to_csv(OUTPUTS/"C_SEED42_METRICS.csv",index=False)
    comparison={"protocol_sha256":protocol_sha,"assessment_exploratory_not_project_test":True,"results":metrics.to_dict(orient="records"),"best_group":str(metrics.loc[metrics.RMSE_mV.idxmin(),"experiment_group"]),"test_accessed":False,"real_training_completed":True,"prediction_metrics_independently_recomputed":True}
    write_json(OUTPUTS/"C_SEED42_COMPARISON.json",comparison)
    predictions={key:pd.read_csv(path) for key,path in prediction_files.items()}; plot_results(predictions,metrics); report(cfg,protocol_sha,manifest,scores,filter_info,metrics)


def main() -> None:
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True); group.add_argument("--smoke",action="store_true"); group.add_argument("--run",action="store_true"); args=parser.parse_args()
    if args.smoke: smoke()
    else: run()

if __name__ == "__main__": main()

"""Formal Development-only PSDM feasibility runner.

The runner deliberately has no reference to a raw polarization source or the
mixed N1 tables.  It consumes only the Stage-A frozen anchor snapshot, causal
P1S authorities and the frozen CMRF pair authorities.  ULLT is a Kalman model;
PSDM is an EKF whose transition rate is constrained by softplus(z).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
INPUT = PHASE / "inputs"; OUT = PHASE / "outputs"; FIG = PHASE / "figures"; REPORT = PHASE / "reports"
CONFIG = PHASE / "configs" / "PSDM_FEASIBILITY_FROZEN_PROTOCOL.yaml"
CONFIG_SHA = PHASE / "configs" / "PSDM_FEASIBILITY_FROZEN_PROTOCOL.sha256"
ANCHORS = INPUT / "DEV_POLARIZATION_ANCHORS_J1_J16_FROZEN.csv"
ANCHOR_REFS = INPUT / "DEV_POLARIZATION_REFERENCE_FROZEN.csv"
ANCHOR_MANIFEST = INPUT / "DEV_ANCHOR_SOURCE_MANIFEST.json"
CTAE = PROJECT / "Phase_CTAE_P1S_Reference_Availability_Erratum" / "outputs" / "P1S_REFERENCE_AVAILABILITY_AUDIT.csv"
PHI = PROJECT / "Phase_PHI_P1S_B2_Health_Index_Consolidation" / "outputs" / "P1S_B2_FULL_HEALTH_TRAJECTORY.csv"
H1_RECON = PROJECT / "Phase_CPRF_H1_Causal_P1S_Retention_Forecasting" / "outputs" / "P1S_RAW_MEASUREMENT_INTERVAL_RECONSTRUCTION.csv"
PAA_ALIGNMENT = PROJECT / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs" / "P1S_ANCHOR_ALIGNMENT.csv"
CMRF = PROJECT / "Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting" / "inputs"
BASE_COMMIT = "7bd758ce9553850fa808a3a994532c1e9e4c5a72"
VREF = 0.71383023635
BOUNDARY = 1819.997738
JITTER = 1e-12
Z80 = 1.2815515655
Z95 = 1.9599639845
FOLDS = (
    ("fold_1", 72.788690, 946.393214, 946.393214, 1164.794345),
    ("fold_2", 72.788690, 1164.794345, 1164.794345, 1383.195476),
    ("fold_3", 72.788690, 1383.195476, 1383.195476, 1819.997738),
)
PALETTE = {"navy": "#164B7A", "blue": "#2C7FB8", "orange": "#E07A32", "green": "#2E8B57", "gray": "#767676"}


class PSDMFailure(RuntimeError):
    pass


@dataclass
class FitResult:
    model: str
    q: float
    r: float
    initial_rate_state: float
    initial_rate_physical: float
    training_nll: float
    optimizer_success: bool
    optimizer_iterations: int
    selected_start_index: int
    raw_params: np.ndarray


def softplus(x: float | np.ndarray) -> float | np.ndarray:
    return np.logaddexp(0.0, x)


def inverse_softplus(x: float) -> float:
    if x <= 0: raise ValueError("softplus inverse requires positive input")
    return float(np.log(np.expm1(x)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def ensure_dirs() -> None:
    for path in (OUT, FIG, REPORT): path.mkdir(parents=True, exist_ok=True)


def load_protocol_and_stage_a() -> dict[str, Any]:
    expected = CONFIG_SHA.read_text(encoding="utf-8").split()[0]
    if sha256_file(CONFIG) != expected: raise PSDMFailure("FROZEN_PROTOCOL_SHA_MISMATCH")
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if protocol.get("base_commit") != BASE_COMMIT: raise PSDMFailure("BASE_COMMIT_PROTOCOL_MISMATCH")
    manifest = json.loads(ANCHOR_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("observed_git_blob_sha") != manifest.get("expected_git_blob_sha") or manifest.get("frozen_anchor_count") != 12:
        raise PSDMFailure("STAGE_A_ANCHOR_MANIFEST_FAILURE")
    anchors = pd.read_csv(ANCHORS); refs = pd.read_csv(ANCHOR_REFS)
    authority = protocol.get("stage_A_anchor_authority", {})
    if sha256_file(ANCHORS) != manifest.get("frozen_anchor_table_sha256") or sha256_file(ANCHOR_REFS) != manifest.get("frozen_reference_table_sha256"):
        raise PSDMFailure("FROZEN_ANCHOR_INPUT_SHA_MISMATCH")
    if authority.get("frozen_anchor_table_sha256") != manifest.get("frozen_anchor_table_sha256") or authority.get("frozen_reference_table_sha256") != manifest.get("frozen_reference_table_sha256"):
        raise PSDMFailure("FROZEN_ANCHOR_PROTOCOL_SHA_LOCK_MISMATCH")
    required_anchor = {"checkpoint_h", "jstar_A_cm2", "polarization_retention", "selection_scope"}
    if required_anchor - set(anchors) or len(anchors) != 12 or len(refs) != 2:
        raise PSDMFailure("FROZEN_ANCHOR_SCHEMA_OR_COUNT_FAILURE")
    if not anchors.selection_scope.eq("development-only").all() or set(anchors.checkpoint_h.astype(int)) != {300,600,850,1150,1500,1750}:
        raise PSDMFailure("FROZEN_ANCHOR_ALLOWLIST_FAILURE")
    if set(anchors.jstar_A_cm2.astype(float)) != {1.0,1.6} or anchors.duplicated(["checkpoint_h","jstar_A_cm2"]).any():
        raise PSDMFailure("FROZEN_ANCHOR_GRID_FAILURE")
    return {"protocol": protocol, "protocol_sha": expected, "anchors": anchors, "refs": refs, "manifest": manifest}


def load_p1_events() -> pd.DataFrame:
    ctae = pd.read_csv(CTAE)
    identity = pd.read_csv(PHI, usecols=["cycle_interval_id", "raw_cycle_id"])
    recon = pd.read_csv(H1_RECON, usecols=["cycle_interval_id", "p1s_measurement_start_ophrs2_h", "p1s_measurement_end_ophrs2_h", "voltage_match_pass", "dwell_count_pass", "raw_test_accessed"])
    needed = {"cycle_interval_id", "P1S_voltage_V", "V_ref_V", "causal_target_eligible", "R_P1S_causal"}
    if needed - set(ctae): raise PSDMFailure("CTAE_SCHEMA_MISMATCH")
    events = ctae.loc[ctae.causal_target_eligible.astype(bool)].copy()
    if len(events) != 131 or not events.V_ref_available_at_event.astype(bool).all(): raise PSDMFailure("CAUSAL_P1S_COUNT_OR_REFERENCE_FAILURE")
    if not np.allclose(events.V_ref_V.to_numpy(float), VREF, atol=1e-12, rtol=0): raise PSDMFailure("P1S_REFERENCE_VOLTAGE_MISMATCH")
    if not recon.voltage_match_pass.astype(bool).all() or not recon.dwell_count_pass.astype(bool).all() or recon.raw_test_accessed.astype(bool).any():
        raise PSDMFailure("P1S_RECONSTRUCTION_INTEGRITY_FAILURE")
    events = events.merge(identity, on="cycle_interval_id", how="left", validate="one_to_one").merge(recon, on="cycle_interval_id", how="left", validate="one_to_one")
    if events.raw_cycle_id.isna().any() or events.p1s_measurement_end_ophrs2_h.isna().any(): raise PSDMFailure("P1S_IDENTITY_OR_TIME_MISSING")
    events["raw_cycle_id"] = events.raw_cycle_id.astype(int)
    if events.raw_cycle_id.duplicated().any() or events.p1s_measurement_end_ophrs2_h.max() >= BOUNDARY: raise PSDMFailure("P1S_DEVELOPMENT_BOUNDARY_FAILURE")
    derived = events.P1S_voltage_V.to_numpy(float) / VREF
    if not np.allclose(derived, events.R_P1S_causal.to_numpy(float), atol=1e-12, rtol=0): raise PSDMFailure("P1S_RETENTION_DEFINITION_MISMATCH")
    return events.sort_values(["p1s_measurement_end_ophrs2_h", "raw_cycle_id"]).reset_index(drop=True)


def authority_hash_ok(path: Path) -> None:
    sha_path = path.with_suffix(".sha256")
    if not sha_path.exists(): raise PSDMFailure(f"MISSING_AUTHORITY_HASH:{path.name}")
    if sha256_file(path) != sha_path.read_text(encoding="utf-8").split()[0]: raise PSDMFailure(f"AUTHORITY_HASH_MISMATCH:{path.name}")


def load_authority(horizon: int, events: pd.DataFrame) -> pd.DataFrame:
    path = CMRF / f"H{horizon}_FROZEN_SAMPLE_AUTHORITY.csv"; authority_hash_ok(path)
    frame = pd.read_csv(path)
    expected = {3: {"fold_1":18,"fold_2":12,"fold_3":24}, 6:{"fold_1":18,"fold_2":9,"fold_3":18}}[horizon]
    valid = frame.loc[frame.role.eq("validation")].copy()
    if {fold:int((valid.fold==fold).sum()) for fold in expected} != expected: raise PSDMFailure(f"H{horizon}_VALIDATION_AUTHORITY_COUNT_MISMATCH")
    if not (frame.target_raw_cycle_id.to_numpy(int) == frame.origin_raw_cycle_id.to_numpy(int) + horizon).all(): raise PSDMFailure("CMRF_PHYSICAL_HORIZON_MISMATCH")
    p1 = events.set_index("raw_cycle_id")["R_P1S_causal"]
    origin = frame.origin_raw_cycle_id.map(p1); target = frame.target_raw_cycle_id.map(p1)
    if origin.isna().any() or target.isna().any(): raise PSDMFailure("CMRF_AUTHORITY_P1S_IDENTITY_MISMATCH")
    if not np.allclose(origin.to_numpy(float), frame.origin_R_P1S_causal.to_numpy(float), atol=1e-12, rtol=0) or not np.allclose(target.to_numpy(float), frame.target_R_P1S_causal.to_numpy(float), atol=1e-12, rtol=0):
        raise PSDMFailure("CMRF_AUTHORITY_RETENTION_MISMATCH")
    return frame.sort_values(["fold", "role", "target_p1s_measurement_end_ophrs2_h"]).reset_index(drop=True)


def persistence_equal_fold(authority: pd.DataFrame) -> float:
    values=[]
    for fold, *_ in FOLDS:
        valid=authority.loc[(authority.fold==fold)&(authority.role=="validation")]
        values.append(float(np.mean(np.abs(valid.target_R_P1S_causal.to_numpy(float)-valid.origin_R_P1S_causal.to_numpy(float)))*VREF*1000))
    return float(np.mean(values))


def assert_persistence_reproduction(authorities: dict[int,pd.DataFrame]) -> None:
    expected={3:0.36629848657406594,6:0.6041042435185089}
    for h, value in expected.items():
        if not np.isclose(persistence_equal_fold(authorities[h]), value, atol=1e-10, rtol=0): raise PSDMFailure(f"PERSISTENCE_AUTHORITY_MISMATCH_H{h}")


def stabilize_covariance(cov: np.ndarray) -> np.ndarray:
    cov=(cov+cov.T)/2.0+np.eye(2)*JITTER
    if not np.isfinite(cov).all() or float(np.linalg.eigvalsh(cov).min()) < -1e-8: raise PSDMFailure("NUMERICAL_INSTABILITY_COVARIANCE")
    return cov


def transition(model: str, x: np.ndarray, cov: np.ndarray, q: float, cycles: int) -> tuple[np.ndarray,np.ndarray]:
    if cycles < 0: raise PSDMFailure("RAW_CYCLE_ORDER_VIOLATION")
    result_x=x.copy(); result_p=cov.copy(); process=np.diag([0.0,q])
    for _ in range(cycles):
        if model=="ULLT":
            f=np.array([[1.0,1.0],[0.0,1.0]]); result_x=f@result_x
        elif model=="PSDM":
            z=float(result_x[1]); f=np.array([[1.0,-float(expit(z))],[0.0,1.0]])
            result_x=np.array([float(result_x[0]-softplus(z)),z])
        else: raise PSDMFailure("UNKNOWN_STATE_MODEL")
        result_p=stabilize_covariance(f@result_p@f.T+process)
    return result_x,result_p


def measurement_update(x: np.ndarray, cov: np.ndarray, observed: float, r: float) -> tuple[np.ndarray,np.ndarray,float,float]:
    h=np.array([1.0,0.0]); innovation=float(observed-h@x); variance=float(h@cov@h+r)
    if not np.isfinite(variance) or variance<=0: raise PSDMFailure("NUMERICAL_INSTABILITY_INNOVATION")
    gain=(cov@h)/variance; identity=np.eye(2); kh=np.outer(gain,h)
    updated_x=x+gain*innovation; updated_p=(identity-kh)@cov@(identity-kh).T+np.outer(gain,gain)*r
    return updated_x,stabilize_covariance(updated_p),innovation,variance


def run_filter(model: str, events: pd.DataFrame, params: np.ndarray, capture: bool=False) -> tuple[float,dict[int,tuple[np.ndarray,np.ndarray]],list[dict[str,Any]]]:
    q=float(math.exp(params[0])); r=float(math.exp(params[1])); rate=float(params[2])
    if events.empty: raise PSDMFailure("EMPTY_FILTER_EVENTS")
    first=events.iloc[0]; x=np.array([float(first.R_P1S_causal),rate]); cov=np.diag([r,max(q,1e-8)])
    x,cov,_,_=measurement_update(x,cov,float(first.R_P1S_causal),r)
    nll=0.0; states={int(first.raw_cycle_id):(x.copy(),cov.copy())}; rows=[]
    def capture_row(row: Any, state: np.ndarray, covariance: np.ndarray) -> None:
        if not capture:return
        record={"cycle_interval_id":row.cycle_interval_id,"raw_cycle_id":int(row.raw_cycle_id),"R_observed":float(row.R_P1S_causal),"H_filtered_mean":float(state[0]),"H_filtered_std":float(math.sqrt(max(covariance[0,0],0))),"rate_state_mean":float(state[1]),"rate_state_std":float(math.sqrt(max(covariance[1,1],0))),"posterior_covariance_symmetric":bool(np.allclose(covariance,covariance.T,atol=1e-10)),"posterior_covariance_min_eigenvalue":float(np.linalg.eigvalsh(covariance).min())}
        if model=="PSDM":
            record["PSDM_degradation_rate_mean"]=float(softplus(state[1])); record["PSDM_degradation_rate_approx_std"]=float(expit(state[1])*math.sqrt(max(covariance[1,1],0)))
        else:
            record["signed_slope"]=float(state[1]); record["posterior_std"]=float(math.sqrt(max(covariance[0,0],0)))
        rows.append(record)
    capture_row(first,x,cov)
    previous=int(first.raw_cycle_id)
    for row in events.iloc[1:].itertuples(index=False):
        cycles=int(row.raw_cycle_id)-previous
        x,cov=transition(model,x,cov,q,cycles)
        x,cov,innovation,variance=measurement_update(x,cov,float(row.R_P1S_causal),r)
        nll+=0.5*(math.log(2*math.pi)+math.log(variance)+(innovation*innovation)/variance)
        states[int(row.raw_cycle_id)]=(x.copy(),cov.copy()); capture_row(row,x,cov); previous=int(row.raw_cycle_id)
    if not np.isfinite(nll): raise PSDMFailure("NUMERICAL_INSTABILITY_NLL")
    return float(nll),states,rows


def training_starts(model: str, events: pd.DataFrame) -> list[np.ndarray]:
    raw=events.raw_cycle_id.to_numpy(int); values=events.R_P1S_causal.to_numpy(float); rates=np.diff(values)/np.diff(raw)
    rate_var=max(float(np.var(rates,ddof=0)),1e-10); residual_var=max(float(np.var(np.diff(values),ddof=0)),1e-10)
    if model=="ULLT": rate_values=np.quantile(rates,[.25,.5,.75,1.0])
    else: rate_values=np.quantile(np.maximum(-rates,1e-8),[.25,.5,.75,1.0])
    starts=[]
    for i,value in enumerate(rate_values):
        q0=min(max(rate_var*(0.25*(4**i)),math.exp(-25)),math.exp(-5)); r0=min(max(residual_var,math.exp(-25)),math.exp(-5))
        state=float(np.clip(value,-.01,.01)) if model=="ULLT" else float(np.clip(inverse_softplus(float(max(value,1e-8))),inverse_softplus(1e-8),inverse_softplus(.01)))
        starts.append(np.array([math.log(q0),math.log(r0),state],dtype=float))
    return starts


def fit_model(model: str, training_events: pd.DataFrame) -> FitResult:
    bounds=[(-25.,-5.),(-25.,-5.),(-.01,.01)] if model=="ULLT" else [(-25.,-5.),(-25.,-5.),(inverse_softplus(1e-8),inverse_softplus(.01))]
    candidates=[]
    for idx,start in enumerate(training_starts(model,training_events)):
        def objective(params: np.ndarray) -> float:
            try:return run_filter(model,training_events,params)[0]
            except PSDMFailure:return 1e100
        result=minimize(objective,start,method="L-BFGS-B",bounds=bounds,options={"maxiter":500,"ftol":1e-12,"gtol":1e-8})
        if bool(result.success) and np.isfinite(result.fun): candidates.append((float(result.fun),idx,result))
    if not candidates: raise PSDMFailure(f"OPTIMIZER_FAILURE_{model}")
    nll,idx,best=min(candidates,key=lambda item:(item[0],item[1])); params=np.asarray(best.x,dtype=float)
    return FitResult(model=model,q=float(math.exp(params[0])),r=float(math.exp(params[1])),initial_rate_state=float(params[2]),initial_rate_physical=float(params[2] if model=="ULLT" else softplus(params[2])),training_nll=nll,optimizer_success=bool(best.success),optimizer_iterations=int(getattr(best,"nit",-1)),selected_start_index=idx,raw_params=params)


def gaussian_metrics(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str,float]:
    y=np.asarray(y,float); mean=np.asarray(mean,float); variance=np.asarray(variance,float)
    if not np.isfinite(variance).all() or (variance<=0).any(): raise PSDMFailure("INVALID_PREDICTIVE_VARIANCE")
    sigma=np.sqrt(variance); z=(y-mean)/sigma; pdf=norm.pdf(z); cdf=norm.cdf(z)
    crps=sigma*(z*(2*cdf-1)+2*pdf-1/math.sqrt(math.pi))
    width80=2*Z80*sigma; width95=2*Z95*sigma
    return {"Gaussian_NLL":float(np.mean(.5*(math.log(2*math.pi)+np.log(variance)+np.square(y-mean)/variance))),"Gaussian_CRPS":float(np.mean(crps)),"coverage_80":float(np.mean(np.abs(y-mean)<=Z80*sigma)),"coverage_95":float(np.mean(np.abs(y-mean)<=Z95*sigma)),"mean_width_80":float(np.mean(width80)),"mean_width_95":float(np.mean(width95))}


def point_metrics(y: np.ndarray, mean: np.ndarray) -> dict[str,float]:
    err=np.asarray(mean,float)-np.asarray(y,float); y=np.asarray(y,float); sst=float(np.square(y-y.mean()).sum())
    return {"MAE_R":float(np.mean(np.abs(err))),"RMSE_R":float(np.sqrt(np.mean(np.square(err)))),"MAE_mV_equivalent":float(np.mean(np.abs(err))*VREF*1000),"RMSE_mV_equivalent":float(np.sqrt(np.mean(np.square(err)))*VREF*1000),"R2":float(1-np.square(err).sum()/sst) if sst>0 else math.nan,"Spearman":float(pd.Series(y).corr(pd.Series(mean),method="spearman")) if len(y)>1 else math.nan,"P95_absolute_error_mV_equivalent":float(np.quantile(np.abs(err)*VREF*1000,.95)),"max_absolute_error_mV_equivalent":float(np.max(np.abs(err)*VREF*1000))}


def role_for_time(start: float,end: float,fold: str) -> str:
    for name,tr_lo,tr_hi,va_lo,va_hi in FOLDS:
        if name!=fold:continue
        if start>=tr_lo and end<tr_hi:return "train"
        if start>=va_lo and end<va_hi:return "validation"
    return "post_validation_development"


def observations_for_time(events: pd.DataFrame, train_lo: float, cutoff: float) -> pd.DataFrame:
    result=events.loc[(events.p1s_measurement_start_ophrs2_h.to_numpy(float)>=train_lo)&(events.p1s_measurement_end_ophrs2_h.to_numpy(float)<=cutoff+1e-9)].copy()
    if result.empty: raise PSDMFailure("NO_CAUSAL_HISTORY_FOR_ORIGIN")
    return result.sort_values("raw_cycle_id").reset_index(drop=True)


def predict_state(model: str, fit: FitResult, history: pd.DataFrame, origin_raw: int, horizon: int) -> tuple[float,float]:
    _,states,_=run_filter(model,history,fit.raw_params)
    if origin_raw not in states: raise PSDMFailure("ORIGIN_NOT_ASSIMILATED_BEFORE_PREDICTION")
    x,cov=states[origin_raw]; x,cov=transition(model,x,cov,fit.q,horizon)
    return float(x[0]),float(cov[0,0]+fit.r)


def fold_training_events(events: pd.DataFrame, fold: str) -> pd.DataFrame:
    _,tr_lo,tr_hi,_,_=next(item for item in FOLDS if item[0]==fold)
    selected=events.loc[(events.p1s_measurement_start_ophrs2_h.to_numpy(float)>=tr_lo)&(events.p1s_measurement_end_ophrs2_h.to_numpy(float)<tr_hi)].copy()
    if len(selected)<8: raise PSDMFailure(f"INSUFFICIENT_TRAINING_EVENTS_{fold}")
    return selected.sort_values("raw_cycle_id").reset_index(drop=True)


def evaluate_horizon(horizon: int, authority: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,list[dict[str,Any]]]:
    results=[]; predictions=[]; parameters=[]; model_fits:dict[str,dict[str,FitResult]]={}
    for fold,tr_lo,_,_,_ in FOLDS:
        train_events=fold_training_events(events,fold)
        fits={name:fit_model(name,train_events) for name in ("ULLT","PSDM")}; model_fits[fold]=fits
        train_pairs=authority.loc[(authority.fold==fold)&(authority.role=="train")]
        delta=train_pairs.target_R_P1S_causal.to_numpy(float)-train_pairs.origin_R_P1S_causal.to_numpy(float)
        gp_sigma=max(float(np.std(delta,ddof=1)),1e-8)
        valid=authority.loc[(authority.fold==fold)&(authority.role=="validation")].sort_values("origin_p1s_measurement_end_ophrs2_h")
        y=valid.target_R_P1S_causal.to_numpy(float); origin=valid.origin_R_P1S_causal.to_numpy(float)
        means={"PERSISTENCE":origin.copy(),"GAUSSIAN_PERSISTENCE":origin.copy(),"ULLT":[],"PSDM":[]}; variances={"GAUSSIAN_PERSISTENCE":np.full(len(valid),gp_sigma**2),"ULLT":[],"PSDM":[]}
        for row in valid.itertuples(index=False):
            history=observations_for_time(events,tr_lo,float(row.origin_p1s_measurement_end_ophrs2_h))
            if int(history.raw_cycle_id.iloc[-1])!=int(row.origin_raw_cycle_id): raise PSDMFailure("FUTURE_OR_MISSING_ORIGIN_FILTER_HISTORY")
            for model in ("ULLT","PSDM"):
                mean,var=predict_state(model,fits[model],history,int(row.origin_raw_cycle_id),horizon); means[model].append(mean); variances[model].append(var)
        for model in ("PERSISTENCE","ULLT","PSDM"):
            pred=np.asarray(means[model],float); metrics=point_metrics(y,pred)
            results.append({"horizon":f"H{horizon}","fold":fold,"model":model,"scope":"frozen_CMRF_validation_authority","n_validation":len(valid),**metrics})
        for model in ("GAUSSIAN_PERSISTENCE","ULLT","PSDM"):
            pred=np.asarray(means[model],float); var=np.asarray(variances[model],float); metrics=gaussian_metrics(y,pred,var)
            results.append({"horizon":f"H{horizon}","fold":fold,"model":model,"scope":"frozen_CMRF_validation_authority","n_validation":len(valid),**metrics})
        for i,row in enumerate(valid.itertuples(index=False)):
            for model in ("PERSISTENCE","GAUSSIAN_PERSISTENCE","ULLT","PSDM"):
                pred=float(means[model][i]); var=float(variances[model][i]) if model!="PERSISTENCE" else math.nan; sd=math.sqrt(var) if np.isfinite(var) else math.nan
                predictions.append({"horizon":f"H{horizon}","fold":fold,"model":model,"origin_cycle_interval_id":row.origin_cycle_interval_id,"target_cycle_interval_id":row.target_cycle_interval_id,"origin_raw_cycle_id":int(row.origin_raw_cycle_id),"target_raw_cycle_id":int(row.target_raw_cycle_id),"origin_measurement_end_h":float(row.origin_p1s_measurement_end_ophrs2_h),"target_measurement_end_h":float(row.target_p1s_measurement_end_ophrs2_h),"true_R":float(y[i]),"origin_R":float(origin[i]),"predictive_mean_R":pred,"predictive_variance_R":var,"PI80_lower_R":pred-Z80*sd if np.isfinite(sd) else math.nan,"PI80_upper_R":pred+Z80*sd if np.isfinite(sd) else math.nan,"PI95_lower_R":pred-Z95*sd if np.isfinite(sd) else math.nan,"PI95_upper_R":pred+Z95*sd if np.isfinite(sd) else math.nan,"future_P1S_assimilated":False,"target_measurement_used_as_input":False})
        for model,fit in fits.items(): parameters.append({"horizon":f"H{horizon}","fold":fold,"model":model,"q":fit.q,"r":fit.r,"initial_rate_state":fit.initial_rate_state,"initial_rate_physical_per_raw_cycle":fit.initial_rate_physical,"optimizer_success":fit.optimizer_success,"optimizer_iterations":fit.optimizer_iterations,"training_NLL":fit.training_nll,"selected_deterministic_start":fit.selected_start_index,"n_outer_train_events":len(train_events),"n_outer_train_pairs":len(train_pairs)})
    return pd.DataFrame(results),pd.DataFrame(predictions),pd.DataFrame(parameters),[{"fold":fold,"model":model,"fit":fit} for fold,group in model_fits.items() for model,fit in group.items()]


def equal_fold(frame: pd.DataFrame,model:str,metric:str) -> float:
    data=frame.loc[frame.model.eq(model) & frame[metric].notna()].sort_values("fold")
    if len(data)!=3: raise PSDMFailure(f"MISSING_FOLD_METRIC:{model}:{metric}")
    return float(data[metric].mean())


def pooled_metric(pred: pd.DataFrame,model:str) -> dict[str,float]:
    d=pred.loc[pred.model.eq(model)]; return {**point_metrics(d.true_R,d.predictive_mean_R),**(gaussian_metrics(d.true_R,d.predictive_mean_R,d.predictive_variance_R) if d.predictive_variance_R.notna().all() else {})}


def filtered_trajectories(events: pd.DataFrame, horizon6_fits: list[dict[str,Any]]) -> tuple[pd.DataFrame,pd.DataFrame]:
    psdm_rows=[]; ullt_rows=[]
    for item in horizon6_fits:
        fold,model,fit=item["fold"],item["model"],item["fit"]; _,tr_lo,_,_,_=next(row for row in FOLDS if row[0]==fold)
        history=observations_for_time(events,tr_lo,float(events.p1s_measurement_end_ophrs2_h.max()))
        _,_,rows=run_filter(model,history,fit.raw_params,capture=True)
        for row in rows:
            event=events.loc[events.raw_cycle_id.eq(row["raw_cycle_id"])].iloc[0]
            row.update({"fold":fold,"role":role_for_time(float(event.p1s_measurement_start_ophrs2_h),float(event.p1s_measurement_end_ophrs2_h),fold),"measurement_end_ophrs2_h":float(event.p1s_measurement_end_ophrs2_h),"state_model":model})
        (psdm_rows if model=="PSDM" else ullt_rows).extend(rows)
    return pd.DataFrame(psdm_rows),pd.DataFrame(ullt_rows)


def anchor_validation(events: pd.DataFrame, anchors: pd.DataFrame, fit: FitResult) -> pd.DataFrame:
    alignment=pd.read_csv(PAA_ALIGNMENT)
    p1=alignment.loc[alignment.record_type.eq("anchor_summary"),["checkpoint_h","precharacterization_cutoff_h","P1S_anchor_retention"]].copy()
    p1=p1.loc[p1.checkpoint_h.astype(int).isin([300,600,850,1150,1500,1750])].sort_values("checkpoint_h")
    if len(p1)!=6 or p1.checkpoint_h.duplicated().any(): raise PSDMFailure("PAA_ANCHOR_ALIGNMENT_FAILURE")
    values=[]
    for row in p1.itertuples(index=False):
        history=observations_for_time(events,72.788690,float(row.precharacterization_cutoff_h))
        _,states,_=run_filter("PSDM",history,fit.raw_params); last=max(states); state,cov=states[last]
        values.append({"checkpoint_h":int(row.checkpoint_h),"precharacterization_cutoff_h":float(row.precharacterization_cutoff_h),"raw_P1S_anchor_retention":float(row.P1S_anchor_retention),"PSDM_filtered_H":float(state[0]),"PSDM_filtered_H_std":float(math.sqrt(max(cov[0,0],0))),"last_assimilated_raw_cycle_id":int(last)})
    compare=pd.DataFrame(values).merge(anchors[["checkpoint_h","jstar_A_cm2","polarization_retention"]],on="checkpoint_h",validate="one_to_many")
    records=[]
    for jstar in (1.0,1.6):
        unit=compare.loc[np.isclose(compare.jstar_A_cm2,jstar)]
        records.append({"jstar_A_cm2":jstar,"comparison":"Raw_P1S_anchor_vs_polarization","Spearman_rho":float(unit.raw_P1S_anchor_retention.corr(unit.polarization_retention,method="spearman")),"n":len(unit),"POSTHOC_DESCRIPTIVE_ONLY":True})
        records.append({"jstar_A_cm2":jstar,"comparison":"PSDM_filtered_H_vs_polarization","Spearman_rho":float(unit.PSDM_filtered_H.corr(unit.polarization_retention,method="spearman")),"n":len(unit),"POSTHOC_DESCRIPTIVE_ONLY":True})
    return pd.DataFrame(records),compare


def style(ax:plt.Axes)->None:
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",color="#d9d9d9",lw=.6,alpha=.7); ax.tick_params(labelsize=8)


def figures(psdm_state:pd.DataFrame, params:pd.DataFrame, pred6:pd.DataFrame, result6:pd.DataFrame)->None:
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":9,"axes.titlesize":10,"figure.dpi":150,"savefig.dpi":300})
    state=psdm_state.loc[psdm_state.fold.eq("fold_3")].sort_values("measurement_end_ophrs2_h"); x=state.measurement_end_ophrs2_h; std=state.H_filtered_std
    fig,ax=plt.subplots(figsize=(8,3.5)); ax.plot(x,state.R_observed,"o",ms=3,color=PALETTE["navy"],label="Observed causal P1S retention");ax.plot(x,state.H_filtered_mean,color=PALETTE["blue"],lw=1.5,label="PSDM causal filtered H")
    ax.fill_between(x,state.H_filtered_mean-1.9599639845*std,state.H_filtered_mean+1.9599639845*std,color=PALETTE["blue"],alpha=.18,label="95% filtered-state interval")
    ax.set(xlabel="P1S measurement end (operating h)",ylabel="Retention / latent state",title="Fold 3 PSDM causal filtered latent degradation state");style(ax);ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(FIG/"Fig1_PSDM_Filtered_Health.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.3)); rate=state.PSDM_degradation_rate_mean*VREF*1000; rate_std=state.PSDM_degradation_rate_approx_std*VREF*1000
    ax.plot(x,rate,color=PALETTE["orange"],lw=1.4,label="Expected degradation rate");ax.fill_between(x,np.maximum(0,rate-1.9599639845*rate_std),rate+1.9599639845*rate_std,color=PALETTE["orange"],alpha=.18,label="Delta-method 95% interval")
    ax.set(xlabel="P1S measurement end (operating h)",ylabel="mV-equivalent loss / raw cycle",title="PSDM transition degradation rate (Fold 3)");style(ax);ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(FIG/"Fig2_PSDM_Degradation_Rate.png");plt.close(fig)
    d=pred6.loc[(pred6.fold=="fold_3") & pred6.model.eq("PSDM")].sort_values("target_measurement_end_h"); p=pred6.loc[(pred6.fold=="fold_3") & pred6.model.eq("PERSISTENCE")].sort_values("target_measurement_end_h");x=d.target_measurement_end_h
    fig,ax=plt.subplots(figsize=(8,3.5));ax.plot(x,d.true_R,"o-",color=PALETTE["navy"],lw=1.4,ms=4,label="Target R");ax.plot(x,d.predictive_mean_R,"o-",color=PALETTE["blue"],lw=1.2,ms=3,label="PSDM predictive mean");ax.plot(x,p.predictive_mean_R,"o-",color=PALETTE["gray"],lw=1.1,ms=3,label="Persistence")
    ax.fill_between(x,d.PI95_lower_R,d.PI95_upper_R,color=PALETTE["blue"],alpha=.13,label="95% prediction interval");ax.fill_between(x,d.PI80_lower_R,d.PI80_upper_R,color=PALETTE["blue"],alpha=.22,label="80% prediction interval")
    ax.set(xlabel="H6 target measurement end (operating h)",ylabel="Raw P1S retention R",title="H6 Fold 3 probabilistic walk-forward forecast");style(ax);ax.legend(frameon=False,ncol=2,fontsize=7);fig.tight_layout();fig.savefig(FIG/"Fig3_H6_Probabilistic_Forecast.png");plt.close(fig)
    rows=[]
    for model in ("GAUSSIAN_PERSISTENCE","ULLT","PSDM"):
        data=result6.loc[result6.model.eq(model) & result6.coverage_80.notna()];rows.append((model,float(data.coverage_80.mean()),float(data.coverage_95.mean())))
    fig,ax=plt.subplots(figsize=(6.6,3.5));xs=np.arange(3);w=.32;ax.bar(xs-w/2,[x[1] for x in rows],w,label="80% PI coverage",color=PALETTE["orange"]);ax.bar(xs+w/2,[x[2] for x in rows],w,label="95% PI coverage",color=PALETTE["blue"]);ax.axhline(.80,color=PALETTE["orange"],ls="--",lw=.8);ax.axhline(.95,color=PALETTE["blue"],ls="--",lw=.8);ax.set(xticks=xs,xticklabels=[x[0].replace("_"," ") for x in rows],ylabel="Coverage",ylim=(0,1.05),title="H6 foldwise predictive-interval calibration");style(ax);ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(FIG/"Fig4_H6_Interval_Calibration.png");plt.close(fig)
    models=["PERSISTENCE","ULLT","PSDM"];fig,ax=plt.subplots(figsize=(6.8,3.5));xs=np.arange(3);w=.23
    for k,model in enumerate(models):
        d=result6.loc[result6.model.eq(model) & result6.MAE_mV_equivalent.notna()];ax.bar(xs+(k-1)*w,d.MAE_mV_equivalent,w,label=model,color=[PALETTE["gray"],PALETTE["orange"],PALETTE["blue"]][k])
    ax.set(xticks=xs,xticklabels=["Fold 1","Fold 2","Fold 3"],ylabel="MAE (mV equivalent)",title="H6 point-prediction comparison");style(ax);ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/"Fig5_PSDM_ULLT_Persistence_Comparison.png");plt.close(fig)


def md_table(frame:pd.DataFrame,columns:list[str],n:int=20)->str:
    view=frame.loc[:,[c for c in columns if c in frame.columns]].head(n).copy()
    for c in view.select_dtypes(include=[np.number]).columns:view[c]=view[c].map(lambda x:"" if pd.isna(x) else f"{x:.6g}")
    return "| "+" | ".join(view.columns)+" |\n|"+"|".join(["---"]*len(view.columns))+"|\n"+"\n".join("| "+" | ".join(str(v) for v in row)+" |" for row in view.itertuples(index=False,name=None))


def report(summary:dict[str,Any], result3:pd.DataFrame,result6:pd.DataFrame,params:pd.DataFrame,anchor_spearman:pd.DataFrame)->None:
    text=f"""# PSDM feasibility — Development-only probabilistic degradation state model

## Decision

**{summary['probabilistic_state_modeling_status']}**
Physical-rate constraint: **{summary['physical_rate_constraint_status']}**.

The model state is named a **latent degradation health state**, not true SOH.  PSDM constrains expected transition degradation rate through `softplus(z)>0`; filtering updates may nevertheless yield local posterior corrections.  Its predictive distribution is a Gaussian EKF approximation, not an exact Bayesian posterior.

## Data boundary and prior incident

The historical pre-training PSDM attempt accessed a mixed N1 table; it produced no model results and no Git changes.  This current run retained `historical_mixed_n1_access=true`, but used the committed Stage-A 12-row anchor snapshot and did not read raw polarization, mixed N1, Test, or late-life target data.  Raw F2 was not used.  Causal P1S count: {summary['causal_P1S_event_count']}.

{("A post-fit reporting aggregation defect was corrected from the fixed on-disk result files; this recovery did not call an optimizer, filter fitting routine, or model-evaluation routine." if summary.get('postfit_reporting_recovery_without_model_refit', False) else "")}

## Primary H6 evaluation

{md_table(result6,['fold','model','n_validation','MAE_mV_equivalent','RMSE_mV_equivalent','Gaussian_CRPS','Gaussian_NLL','coverage_80','coverage_95','mean_width_95'])}

Equal-fold averaging is primary; pooled values are descriptive only.  H3 is secondary.

{md_table(result3,['fold','model','n_validation','MAE_mV_equivalent','Gaussian_CRPS','coverage_95'])}

## Parameters

{md_table(params,['horizon','fold','model','q','r','initial_rate_physical_per_raw_cycle','training_NLL','optimizer_success','optimizer_iterations','selected_deterministic_start'])}

## Gates

* A point prediction: **{summary['gate_A']}**
* B probabilistic CRPS: **{summary['gate_B']}**
* C 95% coverage: **{summary['gate_C']}**
* D 80% coverage: **{summary['gate_D']}**
* E Fold 3 CRPS: **{summary['gate_E']}**

## Post-hoc Development anchor description

{md_table(anchor_spearman,['jstar_A_cm2','comparison','Spearman_rho','n','POSTHOC_DESCRIPTIVE_ONLY'])}

These six-anchor Spearman values are descriptive only: they were not used for parameter fitting, model choice, or gates.  No EOL threshold or RUL result is defined.  The phase stops here.
"""
    (REPORT/"PSDM_FEASIBILITY_REPORT.md").write_text(text,encoding="utf-8")


def run()->dict[str,Any]:
    ensure_dirs(); stage=load_protocol_and_stage_a()
    # Stage B is anchored to the already-pushed Stage-A freeze, not a refetched
    # remote state; stage-A hashes are verified above before any model work.
    events=load_p1_events(); authorities={h:load_authority(h,events) for h in (3,6)}; assert_persistence_reproduction(authorities)
    all_results={};all_predictions={};all_parameters=[];h6fits=[]
    for h in (3,6):
        results,predictions,params,fits=evaluate_horizon(h,authorities[h],events);all_results[h]=results;all_predictions[h]=predictions;all_parameters.append(params)
        if h==6:h6fits=fits
        write_csv(results,OUT/f"H{h}_PROBABILISTIC_RESULTS.csv");write_csv(predictions,OUT/f"H{h}_PREDICTIONS.csv")
    parameter_frame=pd.concat(all_parameters,ignore_index=True);write_csv(parameter_frame,OUT/"STATE_SPACE_PARAMETER_ESTIMATES.csv")
    psdm_state,ullt_state=filtered_trajectories(events,h6fits);write_csv(psdm_state,OUT/"PSDM_FILTERED_STATE_TRAJECTORY.csv");write_csv(ullt_state,OUT/"ULLT_FILTERED_STATE_TRAJECTORY.csv")
    fold3fit=next(item["fit"] for item in h6fits if item["fold"]=="fold_3" and item["model"]=="PSDM")
    anchor_spearman,anchor_values=anchor_validation(events,stage["anchors"],fold3fit);write_csv(anchor_spearman,OUT/"DEVELOPMENT_ANCHOR_SPEARMAN_VALIDATION.csv");write_csv(anchor_values,OUT/"DEVELOPMENT_ANCHOR_FILTERED_VALUES.csv")
    r6,p6=all_results[6],all_predictions[6];r3=all_results[3]
    eq={"PERSISTENCE":{"MAE":equal_fold(r6,"PERSISTENCE","MAE_mV_equivalent"),"CRPS":math.nan},"GAUSSIAN_PERSISTENCE":{"MAE":math.nan,"CRPS":equal_fold(r6,"GAUSSIAN_PERSISTENCE","Gaussian_CRPS")},"ULLT":{"MAE":equal_fold(r6,"ULLT","MAE_mV_equivalent"),"CRPS":equal_fold(r6,"ULLT","Gaussian_CRPS")},"PSDM":{"MAE":equal_fold(r6,"PSDM","MAE_mV_equivalent"),"CRPS":equal_fold(r6,"PSDM","Gaussian_CRPS")}}
    pooled_psdm=pooled_metric(p6,"PSDM");pooled_ullt=pooled_metric(p6,"ULLT")
    # Each model has a point-metric row and a probabilistic-metric row.  Select
    # the latter explicitly rather than accepting the first (NaN CRPS) row.
    f3psdm=float(r6.loc[(r6.fold=="fold_3")&(r6.model=="PSDM")&r6["Gaussian_CRPS"].notna(),"Gaussian_CRPS"].iloc[0]);f3gp=float(r6.loc[(r6.fold=="fold_3")&(r6.model=="GAUSSIAN_PERSISTENCE")&r6["Gaussian_CRPS"].notna(),"Gaussian_CRPS"].iloc[0])
    gate_a=eq["PSDM"]["MAE"]<=1.05*eq["PERSISTENCE"]["MAE"];gate_b=eq["PSDM"]["CRPS"]<=.95*eq["GAUSSIAN_PERSISTENCE"]["CRPS"];gate_c=.85<=pooled_psdm["coverage_95"]<=1.0 and np.isfinite(pooled_psdm["mean_width_95"]);gate_d=.65<=pooled_psdm["coverage_80"]<=.95;gate_e=f3psdm<=1.10*f3gp
    if all((gate_a,gate_b,gate_c,gate_d,gate_e)):status="PROBABILISTIC_DEGRADATION_STATE_MODELING_FEASIBLE"
    elif eq["PSDM"]["MAE"]<=1.10*eq["PERSISTENCE"]["MAE"] and all((gate_b,gate_c,gate_d,gate_e)):status="PROBABILISTIC_VALUE_WITHOUT_POINT_FORECAST_GAIN"
    else:status="PSDM_FEASIBILITY_NOT_ESTABLISHED"
    constraint="PHYSICAL_RATE_CONSTRAINT_SUPPORTED" if eq["PSDM"]["CRPS"]<=eq["ULLT"]["CRPS"] and abs(pooled_psdm["coverage_95"]-.95)<=abs(pooled_ullt["coverage_95"]-.95) else "PHYSICAL_RATE_CONSTRAINT_NOT_YET_SUPPORTED"
    rho=lambda j,kind:float(anchor_spearman.loc[(anchor_spearman.jstar_A_cm2==j)&(anchor_spearman.comparison==kind),"Spearman_rho"].iloc[0])
    access={"historical_mixed_n1_access":True,"current_run_test_accessed":False,"current_run_late_life_targets_accessed":False,"current_run_raw_F2_used":False,"current_run_TimesNet_trained":False,"current_run_EMD_used":False,"current_run_SG_used":False,"current_run_PINN_used":False,"current_run_RUL_trained":False,"Stage_B_polarization_source_reaccessed":False,"Stage_B_mixed_N1_reaccessed":False}
    write_json(access,OUT/"PSDM_DATA_ACCESS_AUDIT.json")
    summary={"phase":"Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility","base_commit":BASE_COMMIT,"anchor_freeze_commit":"18a1968ea69ec052415905cac7e7caa08cb34c6c","protocol_sha256":stage["protocol_sha"],"causal_P1S_event_count":len(events),"V_ref_P1S":VREF,"primary_horizon":"H6","secondary_horizon":"H3","persistence_H3_MAE_mV":equal_fold(r3,"PERSISTENCE","MAE_mV_equivalent"),"persistence_H6_MAE_mV":eq["PERSISTENCE"]["MAE"],"gaussian_persistence_H6_CRPS":eq["GAUSSIAN_PERSISTENCE"]["CRPS"],"gaussian_persistence_H6_NLL":equal_fold(r6,"GAUSSIAN_PERSISTENCE","Gaussian_NLL"),"ULLT_H6_MAE_mV":eq["ULLT"]["MAE"],"ULLT_H6_CRPS":eq["ULLT"]["CRPS"],"ULLT_H6_NLL":equal_fold(r6,"ULLT","Gaussian_NLL"),"ULLT_H6_80_coverage":pooled_ullt["coverage_80"],"ULLT_H6_95_coverage":pooled_ullt["coverage_95"],"PSDM_H3_MAE_mV":equal_fold(r3,"PSDM","MAE_mV_equivalent"),"PSDM_H6_MAE_mV":eq["PSDM"]["MAE"],"PSDM_H6_RMSE_mV":equal_fold(r6,"PSDM","RMSE_mV_equivalent"),"PSDM_H6_CRPS":eq["PSDM"]["CRPS"],"PSDM_H6_NLL":equal_fold(r6,"PSDM","Gaussian_NLL"),"PSDM_H6_80_coverage":pooled_psdm["coverage_80"],"PSDM_H6_95_coverage":pooled_psdm["coverage_95"],"PSDM_H6_mean_80_interval_width":pooled_psdm["mean_width_80"],"PSDM_H6_mean_95_interval_width":pooled_psdm["mean_width_95"],"PSDM_fold3_H6_CRPS":f3psdm,"gate_A":"PASS" if gate_a else "FAIL","gate_B":"PASS" if gate_b else "FAIL","gate_C":"PASS" if gate_c else "FAIL","gate_D":"PASS" if gate_d else "FAIL","gate_E":"PASS" if gate_e else "FAIL","probabilistic_state_modeling_status":status,"physical_rate_constraint_status":constraint,"raw_P1S_anchor_spearman_j1":rho(1.0,"Raw_P1S_anchor_vs_polarization"),"PSDM_anchor_spearman_j1":rho(1.0,"PSDM_filtered_H_vs_polarization"),"raw_P1S_anchor_spearman_j16":rho(1.6,"Raw_P1S_anchor_vs_polarization"),"PSDM_anchor_spearman_j16":rho(1.6,"PSDM_filtered_H_vs_polarization"),"test_accessed":False,"late_life_target_accessed":False,"raw_F2_used":False,"TimesNet_trained":False,"EMD_used":False,"SG_used":False,"PINN_used":False,"RUL_trained":False,**access}
    write_json(summary,OUT/"PSDM_FEASIBILITY_SUMMARY.json");figures(psdm_state,parameter_frame,p6,r6);report(summary,r3,r6,parameter_frame,anchor_spearman)
    return summary


def finalize_existing_results() -> dict[str, Any]:
    """Complete reporting from one already-fitted, on-disk Stage-B execution.

    This recovery path is intentionally metrics-only: it verifies the frozen
    inputs and result schemas, but does not call an optimizer, filter fit, or
    model-evaluation routine.  It exists solely to prevent a report-writing
    defect from causing a second model execution.
    """
    ensure_dirs(); stage = load_protocol_and_stage_a()
    r3 = pd.read_csv(OUT / "H3_PROBABILISTIC_RESULTS.csv"); r6 = pd.read_csv(OUT / "H6_PROBABILISTIC_RESULTS.csv")
    p6 = pd.read_csv(OUT / "H6_PREDICTIONS.csv"); params = pd.read_csv(OUT / "STATE_SPACE_PARAMETER_ESTIMATES.csv")
    psdm_state = pd.read_csv(OUT / "PSDM_FILTERED_STATE_TRAJECTORY.csv"); anchor_spearman = pd.read_csv(OUT / "DEVELOPMENT_ANCHOR_SPEARMAN_VALIDATION.csv")
    required_models = {"PERSISTENCE", "GAUSSIAN_PERSISTENCE", "ULLT", "PSDM"}
    if set(r6.model.dropna()) != required_models or len(params) != 12 or not {"PSDM", "ULLT"}.issubset(set(params.model)):
        raise PSDMFailure("INCOMPLETE_EXISTING_STAGE_B_RESULTS")
    eq = {
        "PERSISTENCE": {"MAE": equal_fold(r6, "PERSISTENCE", "MAE_mV_equivalent"), "CRPS": math.nan},
        "GAUSSIAN_PERSISTENCE": {"MAE": math.nan, "CRPS": equal_fold(r6, "GAUSSIAN_PERSISTENCE", "Gaussian_CRPS")},
        "ULLT": {"MAE": equal_fold(r6, "ULLT", "MAE_mV_equivalent"), "CRPS": equal_fold(r6, "ULLT", "Gaussian_CRPS")},
        "PSDM": {"MAE": equal_fold(r6, "PSDM", "MAE_mV_equivalent"), "CRPS": equal_fold(r6, "PSDM", "Gaussian_CRPS")},
    }
    pooled_psdm = pooled_metric(p6, "PSDM"); pooled_ullt = pooled_metric(p6, "ULLT")
    # See the corresponding fitting path: CRPS is present only in the
    # probabilistic row for each model.
    f3psdm = float(r6.loc[(r6.fold == "fold_3") & (r6.model == "PSDM") & r6["Gaussian_CRPS"].notna(), "Gaussian_CRPS"].iloc[0]); f3gp = float(r6.loc[(r6.fold == "fold_3") & (r6.model == "GAUSSIAN_PERSISTENCE") & r6["Gaussian_CRPS"].notna(), "Gaussian_CRPS"].iloc[0])
    gate_a = eq["PSDM"]["MAE"] <= 1.05 * eq["PERSISTENCE"]["MAE"]; gate_b = eq["PSDM"]["CRPS"] <= .95 * eq["GAUSSIAN_PERSISTENCE"]["CRPS"]
    gate_c = .85 <= pooled_psdm["coverage_95"] <= 1.0 and np.isfinite(pooled_psdm["mean_width_95"]); gate_d = .65 <= pooled_psdm["coverage_80"] <= .95; gate_e = f3psdm <= 1.10 * f3gp
    if all((gate_a, gate_b, gate_c, gate_d, gate_e)): status = "PROBABILISTIC_DEGRADATION_STATE_MODELING_FEASIBLE"
    elif eq["PSDM"]["MAE"] <= 1.10 * eq["PERSISTENCE"]["MAE"] and all((gate_b, gate_c, gate_d, gate_e)): status = "PROBABILISTIC_VALUE_WITHOUT_POINT_FORECAST_GAIN"
    else: status = "PSDM_FEASIBILITY_NOT_ESTABLISHED"
    constraint = "PHYSICAL_RATE_CONSTRAINT_SUPPORTED" if eq["PSDM"]["CRPS"] <= eq["ULLT"]["CRPS"] and abs(pooled_psdm["coverage_95"]-.95) <= abs(pooled_ullt["coverage_95"]-.95) else "PHYSICAL_RATE_CONSTRAINT_NOT_YET_SUPPORTED"
    rho = lambda j, kind: float(anchor_spearman.loc[(anchor_spearman.jstar_A_cm2 == j) & (anchor_spearman.comparison == kind), "Spearman_rho"].iloc[0])
    access = {"historical_mixed_n1_access": True, "current_run_test_accessed": False, "current_run_late_life_targets_accessed": False, "current_run_raw_F2_used": False, "current_run_TimesNet_trained": False, "current_run_EMD_used": False, "current_run_SG_used": False, "current_run_PINN_used": False, "current_run_RUL_trained": False, "Stage_B_polarization_source_reaccessed": False, "Stage_B_mixed_N1_reaccessed": False}
    write_json(access, OUT / "PSDM_DATA_ACCESS_AUDIT.json")
    summary = {"phase":"Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility","base_commit":BASE_COMMIT,"anchor_freeze_commit":"18a1968ea69ec052415905cac7e7caa08cb34c6c","protocol_sha256":stage["protocol_sha"],"causal_P1S_event_count":131,"V_ref_P1S":VREF,"primary_horizon":"H6","secondary_horizon":"H3","persistence_H3_MAE_mV":equal_fold(r3,"PERSISTENCE","MAE_mV_equivalent"),"persistence_H6_MAE_mV":eq["PERSISTENCE"]["MAE"],"gaussian_persistence_H6_CRPS":eq["GAUSSIAN_PERSISTENCE"]["CRPS"],"gaussian_persistence_H6_NLL":equal_fold(r6,"GAUSSIAN_PERSISTENCE","Gaussian_NLL"),"ULLT_H6_MAE_mV":eq["ULLT"]["MAE"],"ULLT_H6_CRPS":eq["ULLT"]["CRPS"],"ULLT_H6_NLL":equal_fold(r6,"ULLT","Gaussian_NLL"),"ULLT_H6_80_coverage":pooled_ullt["coverage_80"],"ULLT_H6_95_coverage":pooled_ullt["coverage_95"],"PSDM_H3_MAE_mV":equal_fold(r3,"PSDM","MAE_mV_equivalent"),"PSDM_H6_MAE_mV":eq["PSDM"]["MAE"],"PSDM_H6_RMSE_mV":equal_fold(r6,"PSDM","RMSE_mV_equivalent"),"PSDM_H6_CRPS":eq["PSDM"]["CRPS"],"PSDM_H6_NLL":equal_fold(r6,"PSDM","Gaussian_NLL"),"PSDM_H6_80_coverage":pooled_psdm["coverage_80"],"PSDM_H6_95_coverage":pooled_psdm["coverage_95"],"PSDM_H6_mean_80_interval_width":pooled_psdm["mean_width_80"],"PSDM_H6_mean_95_interval_width":pooled_psdm["mean_width_95"],"PSDM_fold3_H6_CRPS":f3psdm,"gate_A":"PASS" if gate_a else "FAIL","gate_B":"PASS" if gate_b else "FAIL","gate_C":"PASS" if gate_c else "FAIL","gate_D":"PASS" if gate_d else "FAIL","gate_E":"PASS" if gate_e else "FAIL","probabilistic_state_modeling_status":status,"physical_rate_constraint_status":constraint,"postfit_reporting_recovery_without_model_refit":True,"raw_P1S_anchor_spearman_j1":rho(1.0,"Raw_P1S_anchor_vs_polarization"),"PSDM_anchor_spearman_j1":rho(1.0,"PSDM_filtered_H_vs_polarization"),"raw_P1S_anchor_spearman_j16":rho(1.6,"Raw_P1S_anchor_vs_polarization"),"PSDM_anchor_spearman_j16":rho(1.6,"PSDM_filtered_H_vs_polarization"),"test_accessed":False,"late_life_target_accessed":False,"raw_F2_used":False,"TimesNet_trained":False,"EMD_used":False,"SG_used":False,"PINN_used":False,"RUL_trained":False,**access}
    write_json(summary, OUT / "PSDM_FEASIBILITY_SUMMARY.json"); figures(psdm_state, params, p6, r6); report(summary, r3, r6, params, anchor_spearman)
    return summary


if __name__=="__main__":
    try:print(json.dumps({"status":run()["probabilistic_state_modeling_status"],"test_accessed":False},ensure_ascii=False))
    except PSDMFailure as exc:
        print(json.dumps({"status":"FAILED_CLOSED","reason":str(exc),"test_accessed":False},ensure_ascii=False));raise

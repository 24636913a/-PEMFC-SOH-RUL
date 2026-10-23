"""Independent CPU verification of the frozen raw-second F2 optimization.

CLI: --stage native | B1 | final, or --run A1 (etc.). No optimizer or
training is performed. Default operation is CPU-only. Independent GPU reload
requires --allow-serial-gpu-replay AND explicit parent confirmation that all
training has exited; it never runs alongside GPU training. Complete same-SHA
replay evidence can then be reused. Metrics never import core's metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
OUT = P / "outputs"
OLD = P.parent / "RealHyFC_Raw1s_Dynamic_Study"
REPO = P.parent.parent
CELLS = [f"U{i} [V]" for i in range(1,12) if i != 6]
SPECS = {"A1":("LSTM",30),"A2":("F2",30),"A3":("LSTM",120),"A4":("F2",120),"B1":("F2",30),"B2":("F2",120)}
CPU_REPLAY_ATOL_V = 1e-5  # Predeclared CPU/GPU backend tolerance: 0.01 mV.


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(name, value):
    path = (P/name).resolve()
    require(path.is_relative_to(P),"Verifier write outside new study")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for value in state.values():
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def cache():
    schema = read(OLD/"outputs/cache_schema.json")
    n = schema["rows"]
    dtypes = {"x":"float32","time_ms":"int64","age_h":"float64","file_id":"uint16",
              "source_row":"int32","segment":"int32","valid":"uint8","voltage_V":"float64"}
    a = {}
    for key,dtype in dtypes.items():
        shape = (n,24) if key=="x" else (n,)
        path = OLD/"cache"/f"{key}.bin"
        require(path.stat().st_size==int(np.prod(shape))*np.dtype(dtype).itemsize,"Cache byte size differs: "+key)
        a[key] = np.memmap(path,mode="r",dtype=dtype,shape=shape)
    return schema,a


def metrics(y,p):
    y = np.asarray(y,np.float64).reshape(-1)
    p = np.asarray(p,np.float64).reshape(-1)
    require(y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),"Nonfinite or unequal metric input")
    if y.size==0:
        return dict(queries=0,RMSE_V=None,MAE_V=None,R2_delta=None,P95_abs_V=None,P99_abs_V=None)
    error = p-y
    absolute = np.abs(error)
    sse = float(np.sum(np.square(error),dtype=np.float64))
    denom = float(np.sum(np.square(y-np.mean(y,dtype=np.float64)),dtype=np.float64))
    q95,q99 = np.percentile(absolute,[95.,99.],method="linear")
    return dict(queries=int(y.size),RMSE_V=float(np.sqrt(sse/y.size)),MAE_V=float(absolute.sum(dtype=np.float64)/y.size),
                R2_delta=float(1-sse/denom) if denom>0 else None,P95_abs_V=float(q95),P99_abs_V=float(q99))


def compare_metric(expected, stored, label):
    maximum = 0.
    for key,value in expected.items():
        observed = stored[key]
        if value is None:
            require(pd.isna(observed),f"{label}: undefined {key} was replaced")
        elif key=="queries":
            require(int(observed)==value,f"{label}: query count differs")
        else:
            require(np.isfinite(observed),f"{label}: nonfinite {key}")
            difference = abs(float(observed)-value)
            maximum = max(maximum,difference)
            require(difference<=2e-12+abs(value)*1e-10,f"{label}: {key} differs by {difference}")
    return maximum


def latest_indices(origins,targets):
    """Independent lexicographic selection, not the runner's reverse-unique algorithm."""
    target = targets.reshape(-1)
    origin = np.repeat(origins,targets.shape[1])
    index = np.arange(target.size,dtype=np.int64)
    order = np.lexsort((index,origin,target))
    last = np.r_[target[order][1:]!=target[order][:-1],True]
    return order[last]


def verify_frozen(cfg,a,check_all_old=True):
    require(cfg["seed"]==202 and cfg["history_steps"]==300 and cfg["n_features"]==24,"Frozen task identity differs")
    require(cfg["voltage_channels"]==CELLS,"Active ten-cell target mask changed")
    previous = read(OLD/"outputs/frozen_protocol.json")
    for name in ["input_features","scaler","voltage_event","current_event","seams","target"]:
        require(cfg[name]==previous[name],"Inherited authority changed: "+name)
    require(cfg["voltage_event"]["threshold_V"]==0.038477648942,"User-specified event threshold changed")
    require(cfg["training"]["max_epochs"]==50 and cfg["training"]["patience"]==10,"Training budget changed")
    require(cfg["training"]["effective_batch_size"]==32 and cfg["max_trained_instances"]==6,"Batch/instance budget changed")
    for name in ["learning_rate","weight_decay","gradient_clip_norm","target_scale_V"]:
        require(cfg["training"][name]==previous["training"][name],"Optimizer/scale changed: "+name)
    for name,expected in cfg["implementation_sha256"].items():
        require(sha(P/name)==expected,"Frozen implementation changed: "+name)
    for name,expected in cfg["frozen_file_sha256"].items():
        require(sha(P/name)==expected,"Frozen sampling/rules artifact changed: "+name)
    require(sha(P.parent/"RealHyFC_F2_Main.py")==cfg["historical_code_sha256"],"Native historical class source changed")
    if check_all_old:
        for name,expected in cfg["historical_artifact_sha256"].items():
            require(sha(OLD/name)==expected,"Protected historical artifact changed: "+name)
        for name,item in read(OUT/"legacy_integrity.json")["cache_files"].items():
            path = OLD/"cache"/name
            require(path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],"Protected read-only cache changed: "+name)
        # Actual source SHA, not merely a recorded inventory value, is rechecked
        # in every stage. Exact full timestamps/voltage are reparsed at final.
        for item in read(OLD/"outputs/cache_schema.json")["source_files"]:
            path = REPO/item["path"]
            require(path.stat().st_size==item["bytes"]==item["lfs_size"] and sha(path)==item["sha256"]==item["lfs_oid"],"Original LFS payload SHA/size changed: "+item["file"])
    pre = read(OUT/"review_checkpoint1.json")
    require(pre["decision"]=="PASS" and pre["protocol_sha256"]==sha(OUT/"frozen_protocol.json"),"Pretraining independent gate missing/mismatched")
    return True


def verify_windows(a,origins,h,partition):
    require(origins.dtype==np.int64 and origins.ndim==1 and len(origins)>0,"Invalid frozen origin array")
    require(np.all(np.diff(origins)>0),"Duplicate/reversed frozen origins")
    for start in range(0,len(origins),512):
        o = origins[start:start+512]
        raw = o[:,None]+np.arange(-328,h+1)
        model = o[:,None]+np.arange(-299,h+1)
        require(raw.min()>=0 and raw.max()<len(partition),"Raw information support outside data")
        require((partition[raw]==1).all(),"Validation window/warm-up crosses a time partition")
        require((a["segment"][raw]==a["segment"][o,None]).all(),"Window crosses a recorded segment")
        require((a["file_id"][raw]==a["file_id"][o,None]).all(),"Window crosses raw CSV seam")
        require((np.diff(a["source_row"][raw],axis=1)==1).all(),"Raw window rows are not consecutive")
        dt = np.diff(a["time_ms"][raw],axis=1)
        require(((dt>=950)&(dt<=1050)).all(),"Raw adjacency is not approximately one second")
        require((np.diff(a["age_h"][raw],axis=1)>=-1e-7).all(),"Backwards operating clock")
        require(a["valid"][model].all() and np.isfinite(a["x"][model]).all(),"Invalid historical/target model row")
        targets = o[:,None]+np.arange(1,h+1)
        require(np.abs(a["time_ms"][targets]-a["time_ms"][o,None]-np.arange(1,h+1)*1000).max()<=100,"Wrong physical prediction distance")


def check_predictions(run_id,kind,h,run,cfg,a,partition):
    table = pd.read_csv(run/"metrics.csv",float_precision="round_trip")
    require(len(table)==12,"Metric table omits a grid, scope or stratum")
    records = []; summary = {}; primary = None
    name = "Persistence" if run_id=="Persistence" else "F2_EventWeighted" if run_id.startswith("B") else kind
    for grid,suffix in [("60s",""),("600s","_600s")]:
        origins = np.load(OUT/f"origins_H{h}{suffix}.npz",allow_pickle=False)["VALIDATION"]
        filename = run/f"predictions_VALIDATION_{grid}.npz"
        with np.load(filename,allow_pickle=False) as stream:
            z = {k:stream[k] for k in stream.files}
        shape = (len(origins),h)
        require(np.array_equal(z["origins"],origins),f"{run_id}/{grid}: frozen origins changed")
        targets = origins[:,None]+np.arange(1,h+1,dtype=np.int64)
        require(z["targets"].dtype==np.int64 and np.array_equal(z["targets"],targets),"Some future seconds omitted/shifted")
        require(z["origin_time_ms"].dtype==np.int64 and z["target_time_ms"].dtype==np.int64,"Timestamp integer precision lost")
        require(np.array_equal(z["origin_time_ms"],a["time_ms"][origins]),"Original origin timestamps differ")
        require(np.array_equal(z["target_time_ms"],a["time_ms"][targets]),"Original target timestamps differ")
        require(np.array_equal(z["origin_voltage_V"],a["voltage_V"][origins]),"Raw voltage anchor differs")
        require(np.array_equal(z["target_file_id"],a["file_id"][targets]),"Target source file identity differs")
        require(np.array_equal(z["target_source_row"],a["source_row"][targets]),"Target source row identity differs")
        require(z["truth_delta_V"].dtype==np.float64 and z["prediction_delta_V"].dtype==np.float64,"Double-precision output authority lost")
        require(z["truth_delta_V"].shape==z["prediction_delta_V"].shape==shape,"Incomplete direct multi-step shape")
        truth = np.asarray(a["voltage_V"][targets]-a["voltage_V"][origins,None])
        require(np.array_equal(z["truth_delta_V"],truth),"Original float64 ten-cell DeltaV target differs")
        increments = np.diff(np.c_[a["voltage_V"][origins],a["voltage_V"][targets]],axis=1)
        event = np.logical_or.accumulate(np.abs(increments)>cfg["voltage_event"]["threshold_V"],axis=1)
        require(z["event_query"].dtype==np.bool_ and np.array_equal(z["event_query"],event),"Frozen event labels changed")
        require(np.isfinite(z["prediction_delta_V"]).all(),"Nonfinite predictions")
        if run_id=="Persistence":
            require(np.count_nonzero(z["prediction_delta_V"])==0,"Persistence is not exact last-value hold")
        verify_windows(a,origins,h,partition)
        identities = {key:array_sha(z[key]) for key in z if key!="prediction_delta_V"}
        summary[grid] = dict(origins=len(origins),all_queries=int(truth.size),unique_seconds=int(np.unique(targets).size),
                             prediction_file_sha256=sha(filename),identity_sha256=identities)
        for scope,selection in [("ALL_QUERIES",np.arange(truth.size)),("UNIQUE_TARGET_LATEST_ORIGIN",latest_indices(origins,targets))]:
            y = truth.reshape(-1)[selection]
            forecast = z["prediction_delta_V"].reshape(-1)[selection]
            ev = event.reshape(-1)[selection]
            if scope!="ALL_QUERIES":
                require(len(selection)==len(np.unique(targets)),"Target-second deduplication is incomplete")
                require(np.array_equal(targets.reshape(-1)[selection],np.unique(targets)),"Deduplicated targets not chronological/unique")
            for stratum,mask in [("ALL",np.ones(len(y),bool)),("EVENT",ev),("NORMAL",~ev)]:
                result = metrics(y[mask],forecast[mask])
                row = table.loc[(table.grid==grid)&(table.scope==scope)&(table.stratum==stratum)]
                require(len(row)==1 and row.iloc[0].model==name and row.iloc[0].partition=="VALIDATION" and row.iloc[0].horizon_seconds==h,"Metric identity differs")
                require(row.iloc[0].origins==len(origins) and row.iloc[0].unique_target_seconds==len(np.unique(targets)),"Metric origin/unique-second count differs")
                compare_metric(result,row.iloc[0],f"{run_id}/{h}/{grid}/{scope}/{stratum}")
                records.append(dict(id=run_id,model=name,horizon_seconds=h,grid=grid,scope=scope,stratum=stratum,**result))
        if grid=="60s":
            primary = z
    return records,summary,primary


def branch(name):
    for prefix,label in [("lstm.","LSTM_BASELINE"),("head.","BASELINE_HEAD"),
                         ("native.lstm_encoder.","F2_LSTM_ENCODER"),("native.lstm_decoder.","F2_LSTM_DECODER"),
                         ("native.history_branch.","F2_CAUSAL_CNN"),("native.regime_gate.","F2_REGIME_GATE"),
                         ("native.output_layer.","F2_NATIVE_OUTPUT")]:
        if name.startswith(prefix):
            return label
    return "F2_REMAINING_TFT"


def fresh_cpu_model(kind,h):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(202)
    path = OLD/"scripts/models.py"
    key = "independent_optimization_cpu_model_adapter"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key,path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key].create_model(kind,24,300,h).cpu()


def check_training(run_id,kind,h,run,cfg,a,primary,replay,allow_serial_gpu=False):
    import torch
    state = read(run/"training_state.json")
    require(state["status"]=="COMPLETE" and state["instance_count"]==1,"Run incomplete, failed or repeated")
    require((state["id"],state["model"],state["horizon"],state["seed"])==(run_id,kind,h,202),"Training run identity differs")
    require(state["protocol_sha256"]==sha(OUT/"frozen_protocol.json"),"Training used another protocol")
    require(state["effective_batch"]==32 and state["microbatch"] in [1,2,4,8,16],"Effective batch changed")
    require(state["data_order"]=="torch.randperm with separate CPU generator Seed202 per epoch","Frozen training-order definition changed")
    origins = np.load(OUT/f"origins_H{h}.npz")["TRAIN"]
    require(state["training_origins"]==len(origins) and state["validation_origins"]==len(primary["origins"]),"Run sample support changed")
    run_cfg = read(run/"config.json")
    require(run_cfg["protocol"]==cfg and run_cfg["microbatch"]==state["microbatch"],"Saved run configuration mismatch")
    require(run_cfg["run"]==next(r for r in cfg["runs"] if r["id"]==run_id),"Run loss/architecture differs")
    weighted = run_id.startswith("B")
    require(state["loss"]==("EVENT_WEIGHTED_MSE" if weighted else "MSE"),"Actual training loss identity differs")
    history = pd.read_csv(run/"training_history.csv",float_precision="round_trip")
    require(0<len(history)<=50 and np.array_equal(history.epoch,np.arange(1,len(history)+1)),"Epoch budget/log invalid")
    require(state["stop_epoch"]==state["completed_epoch"]==len(history),"Epoch count differs from complete log")
    required_columns = ["train_loss_standardized","validation_loss_standardized","validation_RMSE_V","validation_MAE_V",
                        "best_epoch","optimizer_steps","epoch_wall_seconds","training_cuda_seconds","validation_cuda_interval_seconds",
                        "GPU_peak_allocated_MiB","GPU_peak_reserved_MiB"]
    require(set(required_columns).issubset(history.columns),"Required real epoch log is incomplete")
    require(np.isfinite(history[required_columns].to_numpy()).all(),"Nonfinite training log")
    require((history[required_columns]>=0).all().all(),"Negative loss/time/memory in log")
    steps_per_epoch = math.ceil(len(origins)/32)
    require(np.array_equal(history.optimizer_steps,np.arange(1,len(history)+1)*steps_per_epoch),"Optimizer updates differ from real macro batches")
    require(state["optimizer_steps"]==len(history)*steps_per_epoch and state["optimizer_steps"]>0,"Real optimizer updates missing")
    require(np.allclose(history.validation_loss_standardized,(history.validation_RMSE_V/cfg["training"]["target_scale_V"])**2,rtol=1e-10,atol=1e-12),"Validation loss is not unweighted standardized MSE")
    require((history.validation_MAE_V<=history.validation_RMSE_V+1e-12).all(),"MAE/RMSE query definitions inconsistent")
    best_value = math.inf;best_epoch = 0;stale = 0
    for row in history.itertuples():
        if row.validation_RMSE_V<best_value-1e-12:
            best_value = float(row.validation_RMSE_V);best_epoch = int(row.epoch);stale = 0
        else:
            stale += 1
        require(row.best_epoch==best_epoch,"Epoch best-checkpoint trace differs")
        if row.epoch<len(history):
            require(stale<10,"Training continued beyond frozen early-stop patience")
    require(len(history)==50 or stale>=10,"Training stopped before patience or budget")
    require(state["best_epoch"]==best_epoch,"Selected epoch differs from full Validation trace")
    for field in ["GPU_peak_allocated_MiB","GPU_peak_reserved_MiB"]:
        require(0<max(float(state[field]),float(history[field].max()))<=8192.,"GPU process peak exceeds8GiB")
    require(sha(run/"best.pt")==state["checkpoint_sha256"] and sha(run/"final_epoch.pt")==state["final_epoch_checkpoint_sha256"],"Checkpoint byte SHA differs")
    checkpoint = torch.load(run/"best.pt",map_location="cpu",weights_only=False)
    require((checkpoint["id"],checkpoint["model"],checkpoint["horizon"],checkpoint["seed"],checkpoint["epoch"])==(run_id,kind,h,202,best_epoch),"Selected checkpoint metadata differs")
    require(checkpoint["protocol_sha256"]==state["protocol_sha256"],"Checkpoint protocol mismatch")
    model = fresh_cpu_model(kind,h)
    initial = {name:value.detach().clone() for name,value in model.named_parameters()}
    require(state_sha(model.state_dict())==state["initial_state_sha256"]==checkpoint["initial_state_sha256"],"Run is not fresh Seed202 initialization")
    initial_hashes = read(run/"initial_parameter_sha256.json")
    require(initial_hashes=={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in initial.items()},"Per-parameter initialization identity differs")
    require(sum(value.numel() for value in model.parameters())==state["parameters"],"Architecture parameter count differs")
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    require(all(torch.isfinite(value).all().item() for value in model.state_dict().values()),"Nonfinite checkpoint weight")
    require(state_sha(model.state_dict())!=state["initial_state_sha256"],"Selected checkpoint remained at initialization")
    groups = {}
    for name,value in model.named_parameters():
        before = initial[name]
        delta = (value.detach()-before).double()
        group = groups.setdefault(branch(name),dict(parameters=0,changed_elements=0,squared_L2_change=0.))
        group["parameters"] += value.numel()
        group["changed_elements"] += int(torch.count_nonzero(value.detach()!=before))
        group["squared_L2_change"] += float(torch.sum(delta*delta))
    recorded = read(run/"branch_parameter_updates.json")
    require(set(recorded)==set(groups),"Missing branch update report")
    for name,result in groups.items():
        require(result["changed_elements"]>0 and recorded[name]["changed_elements"]==result["changed_elements"],"Selected branch did not update: "+name)
        require(recorded[name]["parameters"]==result["parameters"],"Branch parameter support differs")
        require(abs(result["squared_L2_change"]-recorded[name]["squared_L2_change"])<=1e-12+abs(result["squared_L2_change"])*1e-10,"Branch weight change magnitude differs")
    grad = pd.read_csv(run/"gradient_log.csv",float_precision="round_trip")
    require(np.isfinite(grad.preclip_gradient_L2).all() and (grad.preclip_gradient_L2>=0).all(),"Invalid branch gradient log")
    for name in groups:
        rows = grad.loc[grad.branch==name]
        require(len(rows)==2*len(history) and (rows.preclip_gradient_L2>0).any(),"No genuine observed branch gradient: "+name)
        require(np.array_equal(rows.epoch,np.repeat(np.arange(1,len(history)+1),2)),"Branch epoch gradient record missing")
        expected_steps = np.column_stack([np.arange(len(history))*steps_per_epoch+1,np.arange(1,len(history)+1)*steps_per_epoch]).reshape(-1)
        require(np.array_equal(rows.step,expected_steps),"Gradient log does not match actual update steps")
    saved = np.load(run/"best_validation_prediction.npy",allow_pickle=False)
    require(saved.dtype==np.float64 and saved.shape==primary["prediction_delta_V"].shape,"Saved-best validation cache shape/dtype differs")
    reload_error = float(np.max(np.abs(saved-primary["prediction_delta_V"])))
    require(reload_error<=1e-7 and state["reload_max_difference_V"]<=1e-7,"Recorded all-Validation GPU reload parity failed")
    score = metrics(primary["truth_delta_V"],primary["prediction_delta_V"])["RMSE_V"]
    require(abs(score-best_value)<1e-10 and abs(score-state["best_validation_RMSE_V"])<1e-10 and abs(score-checkpoint["validation_RMSE_V"])<1e-10,"Selected checkpoint full-query metric mismatch")
    gate = None
    if kind=="F2":
        weights = []
        mean = np.array(cfg["scaler"]["mean"],np.float32);scale = np.array(cfg["scaler"]["scale"],np.float32)
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(primary["origins"]),512):
                oo = primary["origins"][start:start+512]
                raw = np.asarray(a["x"][oo[:,None]+np.arange(-299,1)])
                x = torch.from_numpy(np.ascontiguousarray((raw-mean)/scale))
                context = x[:,:,list(model.native.regime_indices)].mean(dim=1)
                weights.append(model.native.regime_gate(context).numpy().reshape(-1))
        weights = np.concatenate(weights)
        require(np.isfinite(weights).all() and ((weights>=0)&(weights<=1)).all(),"Invalid learned gate weight")
        gate = dict(all_validation_origins=len(weights),mean=float(weights.mean()),std=float(weights.std()),min=float(weights.min()),max=float(weights.max()),
                    fraction_below_0p05=float((weights<.05).mean()),fraction_above_0p95=float((weights>.95).mean()),
                    all_exactly_0p5=bool(np.all(weights==.5)),all_saturated=bool(np.all((weights<.05)|(weights>.95))))
        require(not gate["all_exactly_0p5"],"Learned F2 gate remains exactly0.5 on every Validation history")
    replay_result = dict(performed=False,reason="Single-run partial check: saved complete GPU reload checked; independent replay required by full-stage gate")
    if replay:
        replay_result = independent_gpu_confirmation(run_id,kind,h,run,model,cfg,a,allow_serial_gpu)
    del model
    return dict(id=run_id,epochs=len(history),best_epoch=best_epoch,optimizer_steps=state["optimizer_steps"],
                initial_state_sha256=state["initial_state_sha256"],checkpoint_sha256=state["checkpoint_sha256"],
                training_origin_sha256=array_sha(origins),training_order_definition=state["data_order"],
                GPU_peak_reserved_MiB=state["GPU_peak_reserved_MiB"],GPU_peak_allocated_MiB=state["GPU_peak_allocated_MiB"],
                recorded_complete_validation_reload_max_difference_V=reload_error,branch_updates=groups,
                branches_have_nonzero_recorded_gradients=True,gate=gate,independent_checkpoint_reload=replay_result,
                training_history_sha256=sha(run/"training_history.csv"),gradient_log_sha256=sha(run/"gradient_log.csv"))


def cpu_replay(model,primary,cfg,a,*,fail_on_difference=True,diagnostic_name="CPU_replay"):
    import torch
    model.eval();torch.set_num_threads(1)
    before = state_sha(model.state_dict())
    mean = np.array(cfg["scaler"]["mean"],np.float32);scale = np.array(cfg["scaler"]["scale"],np.float32)
    maximum = 0.;sum_sq = 0.;count = 0;started = time.time()
    with torch.inference_mode():
        for start in range(0,len(primary["origins"]),16):
            oo = primary["origins"][start:start+16]
            raw = np.asarray(a["x"][oo[:,None]+np.arange(-299,1)])
            x = torch.from_numpy(np.ascontiguousarray((raw-mean)/scale))
            pred = model(x).numpy().astype(np.float64)*cfg["training"]["target_scale_V"]
            error = pred-primary["prediction_delta_V"][start:start+len(oo)]
            require(np.isfinite(error).all(),"Nonfinite independent CPU checkpoint prediction")
            maximum = max(maximum,float(np.max(np.abs(error))));sum_sq += float(np.sum(error*error));count += error.size
            if (start+len(oo))%1024==0 or start+len(oo)==len(primary["origins"]):
                print("INDEPENDENT_CPU_REPLAY_PROGRESS",type(model).__name__,"origins",start+len(oo),"/",len(primary["origins"]),"max_difference_V",maximum,flush=True)
    require(before==state_sha(model.state_dict()),"CPU inference mutated checkpoint weights")
    result = dict(performed=True,device="CPU",all_primary_queries=count,max_absolute_difference_V=maximum,
                difference_RMSE_V=math.sqrt(sum_sq/count),predeclared_absolute_tolerance_V=CPU_REPLAY_ATOL_V,
                tolerance_pass=bool(maximum<=CPU_REPLAY_ATOL_V),weights_unchanged=True,seconds=time.time()-started,
                interpretation="Cross-backend difference is not uniquely attributed by this comparison alone; no tolerance relaxation or prediction changes.")
    write(f"outputs/review_checkpoint_{diagnostic_name}_cpu_replay.json",result)
    if fail_on_difference:
        require(result["tolerance_pass"],"CPU vs GPU replay exceeds predeclared1e-5V tolerance; see persisted CPU diagnostic; coordinate serial GPU replay, never retrain")
    return result


def cpu_first_windows_probe(run_id="A2",count=64):
    """Fixed first windows diagnose the failed CPU probe, never select by error."""
    import torch
    cfg = read(OUT/"frozen_protocol.json");_,a = cache();kind,h = SPECS[run_id]
    run = P/f"runs/{run_id}_{kind}_H{h}"
    with np.load(run/"predictions_VALIDATION_60s.npz",allow_pickle=False) as stream:
        origins = stream["origins"][:count];stored = stream["prediction_delta_V"][:count]
    raw = np.asarray(a["x"][origins[:,None]+np.arange(-299,1)])
    mean = np.array(cfg["scaler"]["mean"],np.float32);scale = np.array(cfg["scaler"]["scale"],np.float32)
    independent_x = np.ascontiguousarray((raw-mean)/scale)
    if str(P/"scripts") not in sys.path:
        sys.path.insert(0,str(P/"scripts"))
    spec = importlib.util.spec_from_file_location("independent_probe_production_dataset",P/"scripts/train.py")
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    production = runner.Windows(a,origins,h,cfg).x.numpy()
    require(independent_x.dtype==production.dtype==np.float32 and independent_x.tobytes()==production.tobytes(),"Probe normalization inputs differ bytewise")
    model = fresh_cpu_model(kind,h)
    checkpoint = torch.load(run/"best.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval()
    before = state_sha(model.state_dict());pieces = []
    with torch.inference_mode():
        for first in range(0,len(origins),16):
            pieces.append(model(torch.from_numpy(independent_x[first:first+16])).numpy().astype(np.float64)*cfg["training"]["target_scale_V"])
    error = np.concatenate(pieces)-stored
    result = dict(run=run_id,selection="fixed first64 chronological primary Validation origins, not error-selected",origins=len(origins),queries=int(error.size),
        max_absolute_difference_V=float(np.abs(error).max()),difference_RMSE_V=float(np.sqrt(np.square(error).mean())),
        input_dtype="float32",mean_dtype="float32",scale_dtype="float32",production_inputs_bytewise_identical=True,
        input_sha256=array_sha(independent_x),checkpoint_sha256=sha(run/"best.pt"),batch_size=16,model_eval=True,
        all_inference_no_grad=True,weights_unchanged=bool(before==state_sha(model.state_dict())),
        production_CuDNN_settings_from_frozen_source=dict(deterministic=True,benchmark=False,matmul_allow_tf32=False,cudnn_allow_tf32=True),
        prior_full_probe=dict(completed=True,queries=int(11249*30),exit_code=1,predeclared_tolerance_V=CPU_REPLAY_ATOL_V,
            log="AssertionError: CPU vs GPU replay exceeds predeclared1e-5V tolerance; coordinate serial GPU replay, never retrain",
            full_max_absolute_difference_V=None,full_difference_RMSE_V=None,
            limitation="The first full probe reached the end-of-loop tolerance assertion but did not serialize max/RMSE; those values are unavailable, not zero. Fixed-first64 values above are not substitutes for full-query values."),
        cause="Not uniquely determined. CPU/GPU backend behavior remains a candidate; matching inputs alone does not prove a unique cause.",
        GPU_calls=0,optimizer_steps=0,requires_serial_GPU_confirmation=True)
    write("outputs/review_checkpoint_cpu_probe.json",result)
    return result


def independent_gpu_confirmation(run_id,kind,h,run,model,cfg,a,allow_serial_gpu):
    """Reuse only identical SHA evidence, or perform explicitly allowed replay."""
    identity = dict(run_id=run_id,model=kind,horizon=h,checkpoint_sha256=sha(run/"best.pt"),
        protocol_sha256=sha(OUT/"frozen_protocol.json"),model_adapter_sha256=sha(OLD/"scripts/models.py"),
        native_code_sha256=sha(P.parent/"RealHyFC_F2_Main.py"),
        primary_prediction_sha256=sha(run/"predictions_VALIDATION_60s.npz"),
        auxiliary_prediction_sha256=sha(run/"predictions_VALIDATION_600s.npz"),
        replay_function_sha256=hashlib.sha256(inspect.getsource(serial_gpu_replay).encode()).hexdigest())
    archive_path = OUT/"review_checkpoint_gpu_reloads.json"
    archive = read(archive_path) if archive_path.exists() else {"runs":{}}
    saved = archive["runs"].get(run_id)
    if saved is not None:
        require(saved["identity"]==identity and saved["pass"],"Cached independent GPU evidence identity differs")
        require(all(item["all_queries_compared"] and item["max_absolute_difference_V"]<=1e-7 for item in saved["grids"].values()),"Cached GPU proof incomplete/outside tolerance")
        return dict(performed=False,reused_same_SHA_evidence=True,proof=saved,executed_GPU_forward_batches_this_invocation=0)
    require(allow_serial_gpu,"Independent GPU confirmation not yet available: wait for root GPU-idle authorization, then explicitly use --allow-serial-gpu-replay")
    result = serial_gpu_replay(run_id,run,model,cfg,a)
    proof = dict(identity=identity,**result,producer_verifier_sha256=sha(Path(__file__)))
    archive["runs"][run_id] = proof
    write("outputs/review_checkpoint_gpu_reloads.json",archive)
    return dict(performed=True,reused_same_SHA_evidence=False,proof=proof,
                executed_GPU_forward_batches_this_invocation=result["GPU_forward_batches"])


def serial_gpu_replay(run_id,run,model,cfg,a):
    """Full 60s AND auxiliary600s inference, no optimizer and no backward."""
    import torch
    # WDDM reports desktop graphics processes here too. Only another Python
    # process could be this study's serial training runner; never kill any PID.
    listed = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader,nounits"],text=True)
    other_python = []
    for row in csv.reader(listed.splitlines()):
        if len(row)>=2 and row[0].strip().isdigit():
            pid = int(row[0].strip());name = Path(row[1].strip()).name.lower()
            if "python" in name and pid!=os.getpid():
                other_python.append(pid)
    require(not other_python,"Other Python GPU process still active; wait, do not run parallel GPU inference: "+str(other_python))
    require(torch.cuda.is_available(),"Explicit independent GPU replay requested but CUDA unavailable")
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    if str(P/"scripts") not in sys.path:
        sys.path.insert(0,str(P/"scripts"))
    spec = importlib.util.spec_from_file_location("independent_gpu_production_inputs",P/"scripts/train.py")
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    before = state_sha(model.state_dict())
    model.cuda().eval();torch.cuda.reset_peak_memory_stats();started = time.time();grids = {};batches = 0
    require(all(parameter.grad is None for parameter in model.parameters()),"Unexpected preexisting gradient before independent inference")
    mean = np.array(cfg["scaler"]["mean"],np.float32);scale = np.array(cfg["scaler"]["scale"],np.float32)
    for grid in ["60s","600s"]:
        with np.load(run/f"predictions_VALIDATION_{grid}.npz",allow_pickle=False) as stream:
            z = {key:stream[key] for key in stream.files}
        h = z["targets"].shape[1];origins = z["origins"]
        production = runner.Windows(a,origins,h,cfg)
        maximum = 0.;squared = 0.;count = 0;input_hash = hashlib.sha256()
        with torch.inference_mode():
            for first in range(0,len(origins),16):
                oo = origins[first:first+16]
                raw = np.asarray(a["x"][oo[:,None]+np.arange(-299,1)])
                independent = np.ascontiguousarray((raw-mean)/scale)
                expected = production.x[first:first+len(oo)].numpy()
                require(independent.dtype==expected.dtype==np.float32 and independent.tobytes()==expected.tobytes(),"Full independent inputs differ bytewise from production Windows")
                input_hash.update(independent.tobytes())
                forecast = model(torch.from_numpy(independent).cuda()).cpu().numpy().astype(np.float64)*cfg["training"]["target_scale_V"]
                error = forecast-z["prediction_delta_V"][first:first+len(oo)]
                require(np.isfinite(error).all(),"Nonfinite independent GPU reload output")
                maximum=max(maximum,float(np.abs(error).max()));squared+=float(np.square(error).sum(dtype=np.float64));count+=error.size;batches+=1
                if (first+len(oo))%2048==0 or first+len(oo)==len(origins):
                    print("INDEPENDENT_SERIAL_GPU_REPLAY",run_id,grid,"origins",first+len(oo),"/",len(origins),"max_difference_V",maximum,flush=True)
        grids[grid] = dict(all_queries_compared=bool(count==z["prediction_delta_V"].size),origins=len(origins),queries=count,
            max_absolute_difference_V=maximum,difference_RMSE_V=math.sqrt(squared/count),predeclared_absolute_tolerance_V=1e-7,
            production_inputs_bytewise_identical=True,normalized_input_sha256=input_hash.hexdigest(),batch_size=16)
        # Preserve the actual numeric diagnostic before any tolerance assertion.
        write(f"outputs/review_checkpoint_{run_id}_gpu_replay.json",dict(pass_all_so_far=bool(all(item["max_absolute_difference_V"]<=1e-7 for item in grids.values())),grids=grids,
              optimizer_steps=0,backward_calls=0))
        require(maximum<=1e-7,"Independent serial GPU reload exceeds fixed1e-7V tolerance; stop without retraining")
        del production
    torch.cuda.synchronize()
    require(before==state_sha(model.state_dict()) and all(parameter.grad is None for parameter in model.parameters()),"Independent GPU inference changed weights or created gradients")
    peak_allocated = torch.cuda.max_memory_allocated()/1024**2;peak_reserved=torch.cuda.max_memory_reserved()/1024**2
    require(peak_allocated<=8192 and peak_reserved<=8192,"Independent reload exceeds8GiB budget")
    result = {"pass":True,"device":"CUDA0","device_name":torch.cuda.get_device_name(0),"torch_version":str(torch.__version__),
        "cudnn_version":torch.backends.cudnn.version(),"grids":grids,"GPU_forward_batches":batches,
        "GPU_peak_allocated_MiB":peak_allocated,"GPU_peak_reserved_MiB":peak_reserved,"batch_size":16,"model_eval":True,
        "inference_mode":True,"weights_unchanged":True,"all_gradients_remain_None":True,"optimizer_steps":0,"backward_calls":0,
        "flags":dict(cudnn_deterministic=True,cudnn_benchmark=False,matmul_allow_tf32=False,cudnn_allow_tf32=True),
        "no_other_Python_GPU_process_observed":True,"explicit_parent_idle_confirmation_required_by_workflow":True,"seconds":time.time()-started}
    model.cpu();torch.cuda.empty_cache()
    return result


def run_check(run_id,cfg,a,partition,replay=False,allow_serial_gpu=False):
    kind,h = SPECS[run_id]
    run = P/f"runs/{run_id}_{kind}_H{h}"
    rows,identities,primary = check_predictions(run_id,kind,h,run,cfg,a,partition)
    training = check_training(run_id,kind,h,run,cfg,a,primary,replay,allow_serial_gpu)
    pd.DataFrame(rows).to_csv(OUT/f"independent_{run_id}_metrics.csv",index=False,float_format="%.17g")
    result = dict(id=run_id,passed=True,predictions=identities,training=training,all_query_metrics=rows)
    print("INDEPENDENT_RUN_PASS",run_id,"queries",sum(x["all_queries"] for x in identities.values()),flush=True)
    return result


def baseline_check(h,cfg,a,partition):
    run = P/f"runs/Persistence_H{h}"
    rows,identities,_ = check_predictions("Persistence","Persistence",h,run,cfg,a,partition)
    pd.DataFrame(rows).to_csv(OUT/f"independent_Persistence_H{h}_metrics.csv",index=False,float_format="%.17g")
    return dict(horizon=h,passed=True,predictions=identities,all_query_metrics=rows,fitted=False)


def historical_matched_check(h,cfg,a):
    kept = np.load(OUT/f"origins_H{h}_600s.npz")["VALIDATION"]
    table = pd.read_csv(OUT/f"historical_same_queries_H{h}.csv",float_precision="round_trip")
    require(len(table)==18,"Historical matched comparison table incomplete")
    records = []
    for kind in ["Persistence","LSTM","F2"]:
        directory = OLD/f"runs/H{h}_{kind}{'' if kind=='Persistence' else '_Seed202'}"
        with np.load(directory/"predictions_VALIDATION.npz",allow_pickle=False) as stream:
            source = {k:stream[k] for k in stream.files}
        take = np.searchsorted(source["origins"],kept)
        require(np.array_equal(source["origins"][take],kept),"Historical comparison did not retain exactly frozen auxiliary origins")
        targets = kept[:,None]+np.arange(1,h+1)
        require(np.array_equal(source["targets"][take],targets),"Historical matched target seconds differ")
        truth = a["voltage_V"][targets]-a["voltage_V"][kept,None]
        require(np.array_equal(source["truth_delta_V"][take],truth),"Historical matched original target authority differs")
        pred = source["prediction_delta_V"][take]
        steps = np.diff(np.c_[a["voltage_V"][kept],a["voltage_V"][targets]],axis=1)
        event = np.logical_or.accumulate(np.abs(steps)>cfg["voltage_event"]["threshold_V"],axis=1)
        require(np.array_equal(source["event_query"][take],event),"Historical matched event definition differs")
        for scope,selection in [("ALL_QUERIES",np.arange(truth.size)),("UNIQUE_TARGET_LATEST_ORIGIN",latest_indices(kept,targets))]:
            y = truth.ravel()[selection];p = pred.ravel()[selection];ev = event.ravel()[selection]
            for name,mask in [("ALL",np.ones(len(y),bool)),("EVENT",ev),("NORMAL",~ev)]:
                result = metrics(y[mask],p[mask])
                row = table.loc[(table.model=="Historical_"+kind)&(table.scope==scope)&(table.stratum==name)&(table.grid=="600s")]
                require(len(row)==1,"Historical matched metric identity missing")
                compare_metric(result,row.iloc[0],f"historical/{h}/{kind}/{scope}/{name}")
                records.append(dict(horizon_seconds=h,model="Historical_"+kind,grid="600s",scope=scope,stratum=name,**result))
    pd.DataFrame(records).to_csv(OUT/f"independent_historical_same_queries_H{h}.csv",index=False,float_format="%.17g")
    return dict(passed=True,horizon=h,retained_origins=len(kept),all_query_metrics=records,
                original_old_csv_sha256=sha(OUT/f"historical_same_queries_H{h}.csv"))


def compare_model_identities(results):
    groups = {30:[],120:[]}
    for run_id,result in results.items():
        groups[SPECS[run_id][1]].append(result)
    for h,group in groups.items():
        for result in group[1:]:
            require(all(result["predictions"][grid]["identity_sha256"]==group[0]["predictions"][grid]["identity_sha256"] for grid in ["60s","600s"]),"Models compared on different raw query identities")


def independent_training_coverage():
    """Count episode x origin coverage without importing the preparation code."""
    triggers = pd.read_csv(OUT/"event_triggers.csv")
    catalogue = pd.read_csv(OUT/"event_catalog.csv")
    stored = pd.read_csv(OUT/"event_origin_coverage.csv")
    rows = triggers.trigger_row.to_numpy(np.int64)
    event_ids = triggers.event_id.to_numpy(np.int64)
    train_episode_ids = catalogue.loc[catalogue.partition=="TRAIN","event_id"].to_numpy(np.int64)
    require(np.all(np.diff(rows)>0),"Event trigger catalogue not chronological")
    records = [];summary = []
    for horizon in [30,120]:
        cases = [("OLD_2048",np.load(OLD/f"outputs/origins_H{horizon}.npz")["TRAIN"]),
                 ("NEW_TRAIN",np.load(OUT/f"origins_H{horizon}.npz")["TRAIN"])]
        for label,origins in cases:
            counts = {};hit_origins = 0
            for origin in origins:
                # A trigger exactly at the forecast origin is already observed;
                # it is not a future event for this query.
                covered = np.unique(event_ids[(rows>origin)&(rows<=origin+horizon)])
                hit_origins += bool(len(covered))
                for event in covered:
                    counts[int(event)] = counts.get(int(event),0)+1
            require(set(counts).issubset(set(train_episode_ids.tolist())),"Training coverage reached non-Train episode")
            expected = stored.loc[(stored.horizon==horizon)&(stored.sampling==label)].sort_values("event_id")
            observed = pd.DataFrame(sorted(counts.items()),columns=["event_id","origins_covering_event"])
            require(np.array_equal(expected[["event_id","origins_covering_event"]].to_numpy(),observed.to_numpy()),"Frozen event coverage/repetition table differs")
            for event,count in sorted(counts.items()):
                records.append(dict(horizon=horizon,sampling=label,event_id=event,origins_covering_event=count))
            repetitions = np.array(list(counts.values()),np.int64)
            summary.append(dict(horizon=horizon,sampling=label,origins=len(origins),event_origins=hit_origins,
                                distinct_episodes=len(counts),total_train_episodes=len(train_episode_ids),
                                episode_origin_pairs=int(repetitions.sum()),
                                repetition_P50=float(np.percentile(repetitions,50)),
                                repetition_P95=float(np.percentile(repetitions,95)),
                                repetition_max=int(repetitions.max())))
    pd.DataFrame(records).to_csv(OUT/"independent_training_event_coverage.csv",index=False)
    return dict(passed=True,coverage_and_repetition_independently_recomputed=True,summary=summary)


def independent_diagnostic_tables(run_ids=None):
    """Final editorial support, run only after the diagnostic tables exist.

    Rebuild latest-trigger bookkeeping directly from the frozen trigger list,
    not from the diagnostic implementation or its derived episode map.
    """
    _,a = cache()
    directory = OUT/"diagnostics"
    names = ["extreme_error_summary.csv","error_by_independent_episode.csv",
             "error_by_current_condition.csv","error_by_event_category.csv",
             "error_by_forecast_lead.csv","top_1pct_prediction_queries.csv","error_contribution_curve.csv"]
    tables = {name:pd.read_csv(directory/name,float_precision="round_trip") for name in names}
    trigger = pd.read_csv(OUT/"event_triggers.csv")
    triggers = trigger.trigger_row.to_numpy(np.int64)
    ids = trigger.event_id.to_numpy(np.int64)
    classified = pd.read_csv(directory/"classified_events.csv")
    category_for = dict(zip(classified.event_id,classified.category))
    if run_ids is None:
        run_ids = list(read(OUT/"final_acceptance.json")["runs"])
    cases = [(name,*SPECS[name],P/f"runs/{name}_{SPECS[name][0]}_H{SPECS[name][1]}") for name in run_ids]
    cases += [(f"Persistence_H{h}","Persistence",h,P/f"runs/Persistence_H{h}") for h in [30,120]]
    records = []
    def row_from(table,name,field=None,value=None):
        rows = tables[table].loc[tables[table].run==name]
        if field is not None:
            rows = rows.loc[rows[field]==value]
        require(len(rows)==1,f"Diagnostic row missing/duplicated: {table}/{name}/{field}/{value}")
        return rows.iloc[0]
    def check(y,p,stored,label):
        result = metrics(y,p);compare_metric(result,stored,label)
        sse = float(np.square(np.asarray(p,np.float64)-np.asarray(y,np.float64)).sum(dtype=np.float64))
        require(abs(sse-stored.SSE_V2)<=2e-12+abs(sse)*1e-10,"Diagnostic SSE differs: "+label)
        return result,sse
    for name,kind,h,run in cases:
        with np.load(run/"predictions_VALIDATION_60s.npz",allow_pickle=False) as stream:
            z = {key:stream[key] for key in stream.files}
        origins = z["origins"];targets = z["targets"]
        y = z["truth_delta_V"].ravel();p = z["prediction_delta_V"].ravel();error = p-y;squared = error*error
        last = np.searchsorted(triggers,targets,side="right")-1
        safe = np.maximum(last,0)
        # A trigger at/before this forecast origin is already known, not a
        # future episode. Search the global list, then impose this condition.
        episode = np.where((last>=0)&(triggers[safe]>origins[:,None]),ids[safe],0).ravel()
        require(np.array_equal(episode>0,z["event_query"].ravel()),"Independent latest-trigger event assignment differs")
        all_sse = float(squared.sum(dtype=np.float64));k = int(np.ceil(len(y)*.01))
        order = np.lexsort((np.arange(len(y),dtype=np.int64),-squared));top = order[:k]
        stored = row_from("extreme_error_summary.csv",name)
        result,_ = check(y,p,stored,name+"/all")
        require(stored.top_1pct_queries==k and abs(stored.top_1pct_SSE_V2-squared[top].sum())<=2e-12+all_sse*1e-10,"Top1% contribution differs")
        require(abs(stored.top_1pct_SSE_fraction-squared[top].sum()/all_sse)<=1e-10,"Top1% share differs")
        require(int(stored.independent_operational_episodes_reached)==len(np.unique(episode[episode>0])),"Reached episode count differs")
        require(abs(stored.normal_SSE_fraction-squared[episode==0].sum()/all_sse)<=1e-10,"Ordinary SSE share differs")
        top_table = tables["top_1pct_prediction_queries.csv"]
        top_table = top_table.loc[top_table.run==name].sort_values("rank")
        require(np.array_equal(top_table["rank"],np.arange(1,k+1)),"Top1% ranks incomplete")
        flat_origin = np.repeat(origins,h);flat_target = targets.ravel()
        require(np.array_equal(top_table.origin_row,flat_origin[top]) and np.array_equal(top_table.target_row,flat_target[top]),"Top1% query identity differs")
        require(np.array_equal(top_table.origin_time_ms,a["time_ms"][flat_origin[top]]) and
                np.array_equal(top_table.target_time_ms,a["time_ms"][flat_target[top]]) and
                np.array_equal(top_table.lead_seconds,top%h+1),"Top1% original time/lead identity differs")
        require(np.array_equal(top_table.event_id,episode[top]) and np.allclose(top_table.error_V,error[top],rtol=1e-12,atol=1e-14),"Top1% errors/episode identity differ")
        cumulative = np.cumsum(squared[order])
        curve = tables["error_contribution_curve.csv"].loc[tables["error_contribution_curve.csv"].run==name]
        for entry in curve.itertuples():
            count = min(len(y),int(np.ceil(len(y)*entry.largest_error_query_fraction)))
            require(entry.queries==count and abs(entry.cumulative_SSE_fraction-cumulative[count-1]/all_sse)<=1e-10,"Cumulative contribution curve differs")
        for event in np.unique(episode):
            mask = episode==event;stored = row_from("error_by_independent_episode.csv",name,"event_id",int(event))
            _,sse = check(y[mask],p[mask],stored,f"{name}/episode/{event}")
            require(abs(stored.all_SSE_fraction-sse/all_sse)<=1e-10,"Episode SSE share differs")
            require(stored.unique_origins==len(np.unique(flat_origin[mask])) and stored.unique_target_seconds==len(np.unique(flat_target[mask])),"Episode overlap counts differ")
        categories = np.array([category_for.get(int(event),"ordinary_no_voltage_trigger") for event in episode],dtype=object)
        for category in np.unique(categories):
            mask = categories==category;stored = row_from("error_by_event_category.csv",name,"category",category)
            check(y[mask],p[mask],stored,f"{name}/category/{category}")
            require(stored.episodes==len(np.unique(episode[mask & (episode>0)])),"Category episode count differs")
        for lead in range(h):
            check(z["truth_delta_V"][:,lead],z["prediction_delta_V"][:,lead],row_from("error_by_forecast_lead.csv",name,"lead_seconds",lead+1),f"{name}/lead/{lead+1}")
        origin_zero = np.abs(a["x"][origins,5])<=.1;target_zero = np.abs(a["x"][targets,5])<=.1
        all_zero = origin_zero[:,None] & np.logical_and.accumulate(target_zero,axis=1)
        all_loaded = (~origin_zero)[:,None] & np.logical_and.accumulate(~target_zero,axis=1)
        event = z["event_query"]
        conditions = {"origin_loaded":np.repeat(~origin_zero,h),"origin_low_or_zero":np.repeat(origin_zero,h),
            "target_loaded":(~target_zero).ravel(),"target_low_or_zero":target_zero.ravel(),
            "ordinary_stable_low_or_zero_interval":((~event)&all_zero).ravel(),
            "ordinary_loaded_interval":((~event)&all_loaded).ravel(),
            "ordinary_mixed_current_interval":((~event)&~all_zero&~all_loaded).ravel(),
            "event_query":event.ravel(),"ordinary_query":(~event).ravel()}
        for condition,mask in conditions.items():
            check(y[mask],p[mask],row_from("error_by_current_condition.csv",name,"condition",condition),f"{name}/{condition}")
        records.append(dict(run=name,all_queries=len(y),episodes_reached=len(np.unique(episode[episode>0])),top1pct_queries=k,
                            top1pct_SSE_fraction=float(squared[top].sum()/all_sse),**{key:value for key,value in result.items() if key!="queries"}))
    pd.DataFrame(records).to_csv(OUT/"independent_diagnostic_summary.csv",index=False,float_format="%.17g")
    return dict(passed=True,source_tables=[dict(path=str((directory/name).relative_to(P)),sha256=sha(directory/name)) for name in names],
                all_query_episode_category_condition_lead_tail_recomputed=True,runs=records)


def independent_precursor_tables():
    """Verify Train-fitted rules and all past-only precursor records on CPU."""
    _,a = cache();cfg = read(OUT/"frozen_protocol.json");rules = read(OUT/"diagnostic_rules.json")
    directory = OUT/"diagnostics"
    features = ["current_ptp_A","set_current_ptp_A","temperature_max_ptp_C","pressure_max_ptp_kPag","operation_point_changed"]
    def values(rows,w):
        result = np.empty((len(rows),5),np.float64)
        for start in range(0,len(rows),512):
            oo = rows[start:start+512]
            x = np.asarray(a["x"][oo[:,None]+np.arange(-w,1)])
            span = x.max(axis=1)-x.min(axis=1)
            result[start:start+len(oo)] = np.column_stack([span[:,5],span[:,14],span[:,16:20].max(axis=1),
                                                        span[:,20:22].max(axis=1),np.any(x[:,:,22]!=x[:,:1,22],axis=1)])
        return result
    summary = pd.read_csv(directory/"precursor_fixed_grid_summary.csv",float_precision="round_trip")
    triggers = pd.read_csv(OUT/"event_triggers.csv").trigger_row.to_numpy(np.int64)
    limits = [0]+[int(seam["after_raw_row"]) for seam in cfg["seams"][:2]]
    records = []
    for part,lo,hi in zip(["TRAIN","VALIDATION"],limits[:-1],limits[1:]):
        rem = (a["time_ms"][lo:hi]-cfg["sampling"]["anchor_ms"])%60000
        origins = np.flatnonzero((rem<=50)|(rem>=59950))+lo
        origins = origins[(origins>=lo+328)&(origins+120<hi)]
        prefix = np.r_[0,np.cumsum(a["valid"][lo:hi]==0)]
        origins = origins[(a["segment"][origins-328]==a["segment"][origins+120]) &
                          ((prefix[origins+121-lo]-prefix[origins-299-lo])==0) &
                          (np.abs(a["time_ms"][origins+120]-a["time_ms"][origins]-120000)<=100)]
        if part=="TRAIN":
            require(np.array_equal(origins,np.load(directory/"precursor_train_threshold_origins.npy")),"Train-only precursor fit origins differ")
            require(hashlib.sha256(origins.astype(np.int64).tobytes()).hexdigest()==rules["train_threshold_origin_sha256"],"Train-only fit-origin SHA differs")
        for w in [30,60,120]:
            table = pd.read_csv(directory/f"precursor_{part}_W{w}.csv",float_precision="round_trip")
            require(np.array_equal(table.origin_row,origins) and np.array_equal(table.origin_time_ms,a["time_ms"][origins]),"Full diagnostic grid/time changed")
            v = values(origins,w)
            require(np.array_equal(v,table[features].to_numpy()),"Past-only feature values differ")
            threshold = np.array([rules["train_only_fitted_thresholds"][str(w)][name] for name in features[:4]])
            if part=="TRAIN":
                require(np.array_equal(np.quantile(v[:,:4],.995,axis=0,method="linear"),threshold),"99.5% Train-only diagnostic quantiles differ")
            flag = np.any(v[:,:4]>threshold,axis=1)|(v[:,4]!=0)
            label = np.searchsorted(triggers,origins+w,side="right")>np.searchsorted(triggers,origins,side="right")
            require(np.array_equal(flag,table.causal_precursor_flag) and np.array_equal(label,table.future_voltage_event),"Future-label or past-only flag differs")
            tp=int(np.sum(flag&label));fp=int(np.sum(flag&~label));fn=int(np.sum(~flag&label));tn=int(np.sum(~flag&~label))
            observed = summary.loc[(summary.partition==part)&(summary.past_window_seconds==w)]
            require(len(observed)==1,"Precursor confusion row missing/duplicated")
            expected = dict(origins=len(origins),positive_origins=int(label.sum()),positive_rate=float(label.mean()),flag_rate=float(flag.mean()),
                TP=tp,FP=fp,FN=fn,TN=tn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                false_positive_rate=fp/(fp+tn) if fp+tn else 0.)
            for key,val in expected.items():
                require(abs(float(observed.iloc[0][key])-val)<=1e-12,"Precursor confusion metric differs: "+key)
            records.append(dict(partition=part,past_window_seconds=w,**expected))
    alignment = pd.read_csv(directory/"precursor_event_alignment.csv",float_precision="round_trip")
    rows = alignment.observable_row.to_numpy(np.int64);v = values(rows,30)
    require(np.array_equal(v,alignment[features].to_numpy()),"Event-aligned past-only features differ")
    threshold = np.array([rules["train_only_fitted_thresholds"]["30"][name] for name in features[:4]])
    flag = np.any(v[:,:4]>threshold,axis=1)|(v[:,4]!=0)
    require(np.array_equal(flag,alignment.causal_past30_precursor_flag),"Event-aligned clue flags differ")
    require(np.array_equal(np.where(alignment.first_trigger_offset_seconds<0,"BEFORE",np.where(alignment.first_trigger_offset_seconds==0,"AT","AFTER")),alignment.timing),"Before/at/after information status mislabelled")
    require(np.array_equal(alignment.observable_time_ms,a["time_ms"][rows]),"Event-alignment timestamp differs")
    rates = alignment.groupby(["partition","first_trigger_offset_seconds","timing"]).agg(events=("event_id","count"),clue_fraction=("causal_past30_precursor_flag","mean")).reset_index()
    stored = pd.read_csv(directory/"precursor_event_alignment_summary.csv",float_precision="round_trip")
    require(rates.equals(stored),"Event-alignment summary differs")
    pd.DataFrame(records).to_csv(OUT/"independent_precursor_confusion.csv",index=False,float_format="%.17g")
    return dict(passed=True,all_Train_Validation_past_features_recomputed=True,Train_only_quantiles_exact=True,
                event_alignment_records=len(alignment),before_at_after_separated=True,confusion=records)


def b1_gate(results,cfg):
    native = results["A2"];weighted = results["B1"]
    require(native["training"]["initial_state_sha256"]==weighted["training"]["initial_state_sha256"],"B1/A2 initial states differ")
    require(native["training"]["training_origin_sha256"]==weighted["training"]["training_origin_sha256"] and
            native["training"]["training_order_definition"]==weighted["training"]["training_order_definition"],"B1/A2 frozen samples/order differ")
    def row(result,stratum):
        return next(x for x in result["all_query_metrics"] if x["grid"]=="60s" and x["scope"]=="ALL_QUERIES" and x["stratum"]==stratum)
    a_all,b_all = row(native,"ALL"),row(weighted,"ALL")
    a_ev,b_ev = row(native,"EVENT"),row(weighted,"EVENT")
    a_normal,b_normal = row(native,"NORMAL"),row(weighted,"NORMAL")
    rmse_gain = (a_all["RMSE_V"]-b_all["RMSE_V"])/a_all["RMSE_V"]
    event_gain = (a_ev["RMSE_V"]-b_ev["RMSE_V"])/a_ev["RMSE_V"]
    normal_increase = b_normal["MAE_V"]/a_normal["MAE_V"]-1
    criteria = dict(all_RMSE_reduction_at_least_3pct=bool(rmse_gain>=.03),event_RMSE_reduction_at_least_5pct=bool(event_gain>=.05),
                    normal_MAE_increase_at_most_10pct=bool(normal_increase<=.10),same_all_query_set=True,integrity_pass=True)
    return {"pass":True,"B2_allowed":bool(all(criteria.values())),"decision":"ALLOW_B2" if all(criteria.values()) else "CANCEL_B2_NO_OTHER_WEIGHT_SEARCH",
            "scope":"unrounded double-precision primary VALIDATION60s ALL_QUERIES; not auxiliary600s or unique-second scope",
            "relative_all_RMSE_reduction":rmse_gain,"relative_event_RMSE_reduction":event_gain,
            "relative_normal_MAE_increase":normal_increase,"criteria":criteria,
            "A2_primary_all":a_all,"B1_primary_all":b_all,"A2_event":a_ev,"B1_event":b_ev,
            "A2_normal":a_normal,"B1_normal":b_normal,"protocol_sha256":sha(OUT/"frozen_protocol.json"),
            "A2_prediction_sha256":native["predictions"]["60s"]["prediction_file_sha256"],
            "B1_prediction_sha256":weighted["predictions"]["60s"]["prediction_file_sha256"],
            "reviewer_GPU_calls":0,"reviewer_optimizer_steps":0}


def raw_full_recheck(schema,a):
    maximum = 0.;count = 0;sources = []
    def numeric(s):
        if pd.api.types.is_numeric_dtype(s):
            return s.to_numpy(np.float64)
        x = s.astype("string").str.strip().str.replace("\ufeff","",regex=False)
        both = x.str.contains(",",regex=False,na=False)&x.str.contains(".",regex=False,na=False)
        x = x.mask(both,x.str.replace(",","",regex=False))
        return pd.to_numeric(x.str.replace(",",".",regex=False),errors="coerce").to_numpy(np.float64,na_value=np.nan)
    for source in schema["source_files"]:
        path = REPO/source["path"]
        pointer = subprocess.check_output(["git","show","HEAD:"+source["path"]],cwd=REPO)
        match = re.search(rb"oid sha256:([0-9a-f]{64})\s+size (\d+)",pointer)
        require(match is not None,"Original source is not the committed LFS identity")
        require(path.stat().st_size==int(match[2])==source["bytes"] and sha(path)==match[1].decode()==source["sha256"],"Original CSV/LFS payload changed")
        offset = 0
        for chunk in pd.read_csv(path,sep=";",encoding="utf-8-sig",usecols=["DateTime"]+CELLS,chunksize=100000,low_memory=False):
            start = int(source["first_row"])+offset;end = start+len(chunk)
            ns = pd.to_datetime(chunk.DateTime,errors="raise").to_numpy(dtype="datetime64[ns]").astype(np.int64)
            require((ns%1000000==0).all(),"Original submillisecond precision was truncated")
            require(np.array_equal(ns//1000000,a["time_ms"][start:end]),"Full original DateTime/cache stream differs")
            cells = np.stack([numeric(chunk[name]) for name in CELLS],axis=1)
            voltage = np.add.reduce(cells,axis=1)/10.
            authority = a["voltage_V"][start:end]
            require(np.array_equal(np.isnan(voltage),np.isnan(authority)),"Raw voltage missingness changed")
            diff = np.abs(voltage-authority);clean = diff[np.isfinite(diff)]
            if clean.size:
                maximum = max(maximum,float(clean.max()))
                require(clean.max()<1e-14,"Full original double-precision target authority differs")
            offset += len(chunk)
        require(offset==source["rows"],"Original source row count changed")
        count += offset;sources.append(dict(file=source["file"],rows=offset,sha256=source["sha256"]))
        print("FINAL_ORIGINAL_SOURCE_PASS",source["file"],offset,flush=True)
    require(count==schema["rows"],"Not every original record rechecked")
    return dict(files=len(sources),rows=count,all_original_timestamps_exact_ms=True,max_original_voltage_difference_V=maximum,sources=sources)


def final_causality(cfg):
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(800,24)).astype(np.float32)
    origin = 500
    changed = raw.copy();changed[origin+1:] *= 100
    v = rng.normal(size=800)
    future_v = v.copy();future_v[origin+1:] += 100
    # Import only the frozen runner's definitions to test its actual dataset
    # transformation. Independent numerical metrics still never import it.
    if str(P/"scripts") not in sys.path:
        sys.path.insert(0,str(P/"scripts"))
    spec = importlib.util.spec_from_file_location("independent_opt_dataset_falsification",P/"scripts/train.py")
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    results = {}
    for h in [30,120]:
        before = runner.Windows({"x":raw,"voltage_V":v},np.array([origin],np.int64),h,cfg)
        after = runner.Windows({"x":changed,"voltage_V":future_v},np.array([origin],np.int64),h,cfg)
        require(torch.equal(before.x,after.x),"Future perturbation changes actual runner input tensor")
        require(not np.array_equal(before.truth,after.truth),"Future target perturbation was ineffective")
        model = fresh_cpu_model("F2",h).eval()
        data = model.build_input(torch.randn(2,300,24))
        with torch.inference_mode():
            output = model.native(data)["prediction"]
            altered = {k:v.clone() for k,v in data.items()}
            altered["decoder_cont"][:,:,:24] = torch.randn_like(altered["decoder_cont"][:,:,:24])*100
            alternate = model.native(altered)["prediction"]
            require(torch.equal(output,alternate),"Native decoder consumed future real placeholders")
        results[str(h)] = float(torch.max(torch.abs(output-alternate)))
    return dict(future_raw_perturbation_history_equal=True,future_real_placeholder_max_prediction_difference_V=results,
                GPU_calls=0,optimizer_steps=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["native","B1","final"])
    parser.add_argument("--run",choices=list(SPECS))
    parser.add_argument("--allow-serial-gpu-replay",action="store_true",help="Explicit opt-in only after root confirms GPU training process exited and GPU is idle")
    args = parser.parse_args()
    require(bool(args.stage)^bool(args.run),"Specify exactly one of --stage or --run")
    stage = args.stage or args.run
    started = time.time();results = {}
    try:
        cfg = read(OUT/"frozen_protocol.json");schema,a = cache();verify_frozen(cfg,a)
        partition = np.searchsorted([s["cut_ms"] for s in cfg["seams"]],a["time_ms"],side="right")
        ids = [args.run] if args.run else ["A1","A2","A3","A4"] if stage=="native" else ["A2","B1"] if stage=="B1" else ["A1","A2","A3","A4","B1"]
        if stage=="final":
            gate = read(OUT/"B1_acceptance.json")
            require(gate["pass"],"B1 integrity gate failed")
            if gate["B2_allowed"]:
                ids.append("B2")
            else:
                require(not (P/"runs/B2_F2_H120/training_state.json").exists(),"B2 trained despite failed frozen gate")
        for run_id in ids:
            results[run_id] = run_check(run_id,cfg,a,partition,replay=not bool(args.run) or args.allow_serial_gpu_replay,
                                       allow_serial_gpu=args.allow_serial_gpu_replay)
        compare_model_identities(results)
        if stage=="final":
            require(gate==b1_gate(results,cfg),"Final independent recomputation differs from the saved B1/B2 gate")
        coverage = independent_training_coverage()
        baselines = {};historical = {}
        for h in sorted(set(SPECS[run_id][1] for run_id in ids)):
            baselines[str(h)] = baseline_check(h,cfg,a,partition)
            historical[str(h)] = historical_matched_check(h,cfg,a)
            for run_id,result in results.items():
                if SPECS[run_id][1]==h:
                    require(all(result["predictions"][grid]["identity_sha256"]==baselines[str(h)]["predictions"][grid]["identity_sha256"] for grid in ["60s","600s"]),"Model and Persistence target/query identities differ")
        if stage=="B1":
            gate = b1_gate(results,cfg);write("outputs/B1_acceptance.json",gate)
        proof = None;causal = None
        if stage=="final":
            proof = raw_full_recheck(schema,a);causal = final_causality(cfg)
            # A final read prevents inference-side writes to historical models/cache.
            verify_frozen(cfg,a)
            if "B2" in results:
                require(results["B2"]["training"]["initial_state_sha256"]==results["A4"]["training"]["initial_state_sha256"],"B2/A4 initialization differs")
            actual = list((P/"runs").glob("*/training_state.json"))
            require(len(actual)==len(ids)<=6 and all(read(path)["status"]=="COMPLETE" for path in actual),"Unreported, failed or extra real training instance")
        # Numerical final acceptance precedes the separate third report/figure review.
        path = OUT/("final_acceptance.json" if stage=="final" else "review_checkpoint2.json")
        previous = read(path) if path.exists() else {}
        all_results = previous.get("runs",{});all_results.update(results)
        report = dict(checkpoint="FINAL_NUMERIC" if stage=="final" else 2,decision="PARTIAL_PASS" if args.run else "PASS",stage=stage,
            independent_metrics_and_all_queries=True,runs=all_results,protocol_sha256=sha(OUT/"frozen_protocol.json"),
            persistence_baselines=baselines,historical_matched_comparison=historical,
            training_event_coverage=coverage,
            verifier_sha256=sha(Path(__file__)),
            GPU_calls=sum(item["training"]["independent_checkpoint_reload"].get("executed_GPU_forward_batches_this_invocation",0) for item in results.values()),
            GPU_permission_explicit=bool(args.allow_serial_gpu_replay),optimizer_steps=0,backward_calls=0,
            original_payload_SHA256_and_size_rechecked=True,protected_historical_cache_bytes_rechecked=True,
            full_original_authority_recheck=proof,final_future_falsification=causal,
            trained_instance_count=len(results) if stage=="final" else None,seconds=time.time()-started,
            fatal_findings=[],limitations=["Single-seed exploratory evidence, no independent replication claim.",
                "Unique latest-origin seconds are auxiliary and alter the H120 lead mixture.",
                "Raw episode clusters and branch representation diagnostics are not hardware truth or causal branch attribution.",
                "Gradient/update/reload checks confirm execution identity, not eventual convergence or predictive superiority.",
                "The initial A2 CPU probe exceeded the fixed1e-5V cross-backend tolerance; this failure is retained, not waived. Its cause is not uniquely determined.",
                "Full native/B1/final gates require independent complete serial GPU reload <=1e-7V with byte-identical float32 production inputs and unchanged CuDNN flags, or reuse exact same-SHA proof. No optimizer or backward call is made."])
        if stage=="final":
            report["pass"] = True
            report["report_figure_editorial_review_still_required"] = True
        write(str(path.relative_to(P)),report)
        print("INDEPENDENT_STAGE_PASS",stage,"seconds",time.time()-started,flush=True)
    except Exception as error:
        failure = dict(decision="FAIL_STOP",stage=stage,error=repr(error),traceback=traceback.format_exc(),
                       completed_run_checks=list(results),GPU_permission_explicit=bool(args.allow_serial_gpu_replay),optimizer_steps=0,backward_calls=0)
        if stage=="B1":
            write("outputs/B1_acceptance.json",{"pass":False,"B2_allowed":False,**failure})
        if stage=="final":
            failure["pass"] = False
        write("outputs/final_acceptance.json" if stage=="final" else "outputs/review_checkpoint2.json",failure)
        print(json.dumps(failure,ensure_ascii=False),flush=True)
        raise


if __name__=="__main__":
    main()

"""Independent exhaustive CPU acceptance for a completed raw-second pilot.

No training runner, model module or shared metric implementation is imported.
Every query and every historical-window adjacency is checked. Original CSV
rows underlying all windows are reread, not sampled. Acceptance depends on
scientific/engineering identity, not on whether F2 beats LSTM/Persistence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parent.parent
OUT = STUDY / "outputs"
CACHE = STUDY / "cache"
PARTS = ("TRAIN", "VALIDATION", "HISTORICAL_ASSESSMENT", "LATE_LIFE")
KINDS = ("Persistence", "LSTM", "F2")
CELLS = [f"U{i} [V]" for i in range(1,12) if i != 6]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    assert path.resolve().is_relative_to(STUDY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def require(condition, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value) -> str:
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def number(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series,errors="coerce").to_numpy(dtype=np.float64,na_value=np.nan)
    text = series.astype("string").str.strip().str.replace("\ufeff","",regex=False)
    grouped = text.str.contains(",",regex=False,na=False)&text.str.contains(".",regex=False,na=False)
    text = text.mask(grouped,text.str.replace(",","",regex=False))
    return pd.to_numeric(text.str.replace(",",".",regex=False),errors="coerce").to_numpy(dtype=np.float64,na_value=np.nan)


def independent_metrics(truth, forecast) -> dict:
    truth = np.asarray(truth,dtype=np.float64).reshape(-1)
    forecast = np.asarray(forecast,dtype=np.float64).reshape(-1)
    require(truth.shape==forecast.shape,"Metric truth/prediction shape differs")
    require(np.isfinite(truth).all() and np.isfinite(forecast).all(),"Nonfinite metric input")
    if truth.size==0:
        return dict(queries=0,RMSE_V=None,MAE_V=None,R2_delta=None,P95_abs_V=None,P99_abs_V=None)
    residual = forecast-truth
    squared = np.dot(residual,residual)
    centered = truth-np.mean(truth,dtype=np.float64)
    denominator = np.dot(centered,centered)
    absolute = np.abs(residual)
    q95,q99 = np.percentile(absolute,[95.,99.],method="linear")
    return dict(queries=int(truth.size),RMSE_V=float(np.sqrt(squared/truth.size)),
                MAE_V=float(np.sum(absolute,dtype=np.float64)/truth.size),
                R2_delta=float(1.-squared/denominator) if denominator>0 else None,
                P95_abs_V=float(q95),P99_abs_V=float(q99))


def compare_metrics(expected: dict, stored, identity: str) -> float:
    maximum = 0.
    for key, value in expected.items():
        observed = stored[key]
        if value is None:
            require(pd.isna(observed),f"{identity}: {key} should be undefined")
        elif key=="queries":
            require(int(observed)==value,f"{identity}: query count differs")
        else:
            require(np.isfinite(observed),f"{identity}: nonfinite stored {key}")
            difference = abs(float(observed)-value)
            maximum = max(maximum,difference)
            require(difference<=2e-12+abs(value)*1e-11,f"{identity}: independent {key} mismatch {difference}")
    return maximum


def open_cache(schema: dict) -> dict:
    n = schema["rows"]
    types = {"x":"float32","time_ms":"int64","age_h":"float64","file_id":"uint16",
             "source_row":"int32","segment":"int32","valid":"uint8","voltage_V":"float64"}
    a = {}
    for name, dtype in types.items():
        shape = (n,24) if name=="x" else (n,)
        expected_bytes = int(np.prod(shape))*np.dtype(dtype).itemsize
        require((CACHE/f"{name}.bin").stat().st_size==expected_bytes,f"Cache byte size differs: {name}")
        a[name] = np.memmap(CACHE/f"{name}.bin",mode="r",dtype=dtype,shape=shape)
    return a


def check_frozen_training_statistics(cfg: dict,a: dict) -> dict:
    n = len(a["time_ms"])
    cuts = np.array([s["cut_ms"] for s in cfg["seams"]],dtype=np.int64)
    partition = np.searchsorted(cuts,a["time_ms"],side="right")
    require(np.all(np.diff(partition)>=0),"Original timestamp partitions interleave")
    sums = np.zeros(24,np.float64); sums2 = np.zeros(24,np.float64); count = 0
    for start in range(0,n,77777):
        last = min(n,start+77777)
        keep = (partition[start:last]==0)&a["valid"][start:last].astype(bool)
        block = np.asarray(a["x"][start:last],np.float64)[keep]
        count += len(block); sums += block.sum(axis=0); sums2 += np.square(block).sum(axis=0)
    mean = sums/count
    scale = np.sqrt(np.maximum(sums2/count-np.square(mean),0.))
    scale = np.where(scale<1e-8,1.,scale)
    require(count==cfg["scaler"]["fit_rows"],"Train scaler row support mismatch")
    require(np.allclose(mean,cfg["scaler"]["mean"],atol=1e-9,rtol=1e-12),"Train-only scaler mean mismatch")
    require(np.allclose(scale,cfg["scaler"]["scale"],atol=1e-7,rtol=1e-8),"Train-only scaler scale mismatch")
    pairs = ((partition[:-1]==0)&(partition[1:]==0)&a["valid"][:-1].astype(bool)&a["valid"][1:].astype(bool)
             &(a["segment"][:-1]==a["segment"][1:]))
    voltage_threshold = float(np.quantile(np.abs(np.diff(a["voltage_V"]))[pairs],.995,method="linear"))
    current_threshold = float(np.quantile(np.abs(np.diff(a["x"][:,5].astype(np.float64)))[pairs],.995,method="linear"))
    require(abs(voltage_threshold-cfg["voltage_event"]["threshold_V"])<1e-14,"Train-only voltage event threshold mismatch")
    require(abs(current_threshold-cfg["current_event"]["threshold_A"])<1e-10,"Train-only current threshold mismatch")
    for i, part in enumerate(PARTS):
        require(int(np.count_nonzero(partition==i))==cfg["partition_raw_rows"][part],"Partition raw row count mismatch: "+part)
    return dict(partition=partition,scaler_rows=count,voltage_threshold_V=voltage_threshold,
                max_scaler_mean_difference=float(np.max(np.abs(mean-np.array(cfg["scaler"]["mean"])))),
                max_scaler_scale_difference=float(np.max(np.abs(scale-np.array(cfg["scaler"]["scale"])))) )


def check_window(cfg: dict,a: dict,origins: np.ndarray,h: int,part_index: int,partition: np.ndarray) -> np.ndarray:
    require(origins.dtype.kind in "iu" and origins.ndim==1 and len(origins)>0,"Invalid origins array")
    require(np.all(np.diff(origins)>0),"Origins not uniquely chronological")
    window = origins[:,None]+np.arange(-299,h+1,dtype=np.int64)[None,:]
    require(window.min()>=0 and window.max()<len(a["time_ms"]),"Window out of bounds")
    require(a["valid"][window].all(),"A historical or future instrument row is invalid")
    require((a["segment"][window]==a["segment"][origins,None]).all(),"Window crosses original continuous segment")
    require((a["file_id"][window]==a["file_id"][origins,None]).all(),"Window crosses a raw file seam")
    require((np.diff(a["source_row"][window],axis=1)==1).all(),"Window source rows are not consecutive")
    intervals = np.diff(a["time_ms"][window],axis=1)
    require(((intervals>=950)&(intervals<=1050)).all(),"Window includes non-1-second adjacency")
    require((np.diff(a["age_h"][window],axis=1)>=-1e-7).all(),"Window includes backwards OpHrs2")
    require(np.isfinite(a["x"][window]).all(),"Nonfinite model historical/future audit field")
    require((partition[origins]==part_index).all(),"Origin partition mismatch")
    targets = origins[:,None]+np.arange(1,h+1)
    require((partition[targets]==part_index).all(),"Future target crosses partition")
    origin_times = a["time_ms"][origins]
    offsets = a["time_ms"][targets]-origin_times[:,None]
    require(np.abs(offsets-np.arange(1,h+1)*1000).max()<=100,"Target original-second time identity mismatch")
    require(np.abs(origin_times-a["time_ms"][origins-299]-299000).max()<=100,"Historical extent mismatch")
    lattice_delta = (origin_times-cfg["sampling"]["anchor_ms"])%600000
    require(np.minimum(lattice_delta,600000-lattice_delta).max()<=50,"Origin deviates from frozen time lattice")
    return window


def original_rows_check(wanted: np.ndarray,schema: dict,a: dict) -> dict:
    """Exhaustively compare raw CSV time/age/voltage for all queried and history rows."""
    checked = 0; source_checks = []; maximum_v = 0.
    for fid, source in enumerate(schema["source_files"]):
        path = REPO/source["path"]
        require(path.stat().st_size==source["bytes"],"Original source size changed")
        require(file_sha(path)==source["sha256"]==source["lfs_oid"],"Original source/LFS SHA mismatch")
        first = int(source["first_row"]); offset = 0; selected = 0
        for chunk in pd.read_csv(path,sep=";",encoding="utf-8-sig",usecols=["DateTime","OpHrs2 [h]"]+CELLS,
                                 chunksize=100000,low_memory=False):
            begin = first+offset; end = begin+len(chunk)
            chunk_times = pd.to_datetime(chunk.DateTime,errors="raise").to_numpy(dtype="datetime64[ns]").astype(np.int64)
            require((chunk_times%1000000==0).all(),"Original DateTime has sub-ms precision not retained in cache")
            require(np.array_equal(chunk_times//1000000,a["time_ms"][begin:end]),"Full original raw timestamp stream differs from cache")
            keep = wanted[begin:end]
            if keep.any():
                raw = chunk.loc[keep]
                ids = begin+np.flatnonzero(keep)
                times = chunk_times[keep]
                require(np.array_equal(times//1000000,a["time_ms"][ids]),"Original raw DateTime differs from saved target/cache")
                cells = np.stack([number(raw[c]) for c in CELLS],axis=1)
                voltage = np.add.reduce(cells,axis=1)/10.
                delta = np.abs(voltage-a["voltage_V"][ids])
                require(np.isfinite(voltage).all(),"Original queried voltage is nonfinite")
                require(delta.max(initial=0.)<1e-14,"Original queried ten-cell voltage differs from float64 authority")
                require(np.array_equal(number(raw["OpHrs2 [h]"]),a["age_h"][ids]),"Original queried OpHrs2 differs")
                require((a["file_id"][ids]==fid).all(),"Original queried file ID differs")
                require(np.array_equal(a["source_row"][ids],offset+np.flatnonzero(keep)),"Original queried file row differs")
                maximum_v = max(maximum_v,float(delta.max(initial=0.)))
                selected += len(ids); checked += len(ids)
            offset += len(chunk)
        require(offset==source["rows"],"Original CSV row count differs")
        source_checks.append(dict(file=source["file"],rows_read=offset,rows_checked=selected,sha256=source["sha256"]))
        print(json.dumps(dict(stage="VERIFY_RAW_ROWS",file=source["file"],checked=selected)),flush=True)
    require(checked==int(wanted.sum()),"Not every historical/target source row was independently checked")
    return dict(unique_original_rows_checked=checked,all_query_and_history_rows=True,max_voltage_difference_V=maximum_v,
                timestamp_precision="original millisecond timestamps exact; no resampling",all_original_timestamps_checked=len(a["time_ms"]),
                submillisecond_truncation_absent=True,sources=source_checks)


def check_training(kind: str,h: int,cfg: dict,run: Path,val_npz: dict) -> dict:
    import torch  # CPU checkpoint inspection only; no model or CUDA initialization.
    state = read_json(run/"training_state.json")
    require(state["status"]=="COMPLETE" and state["instance_count"]==1,"Training incomplete or retried")
    require(state["model"]==kind and state["horizon"]==h and state["seed"]==202,"Training identity mismatch")
    protocol_hash = file_sha(OUT/"frozen_protocol.json")
    require(state["protocol_sha256"]==protocol_hash,"Training protocol SHA differs")
    startup_failures = int(state.get("zero_optimizer_step_startup_failures",0))
    require(startup_failures in (0,1),"Undeclared repeated startup recovery")
    if startup_failures:
        failure = read_json(run/"zero_step_startup_failure.json")
        require(failure["status"]=="FAILED" and failure["optimizer_steps"]==0,"Recovery included an earlier learned run")
        require(failure["initial_state_sha256"]==state["initial_state_sha256"],"Startup recovery changed initialization")
        require(failure["protocol_sha256"]==protocol_hash,"Startup recovery changed task protocol")
    require(state["checkpoint_sha256"]==file_sha(run/"best.pt"),"Checkpoint byte SHA differs")
    history = pd.read_csv(run/"training_history.csv",float_precision="round_trip")
    require(len(history)>0 and len(history)<=cfg["training"]["max_epochs"],"Invalid epoch budget/log")
    require(np.array_equal(history.epoch.to_numpy(),np.arange(1,len(history)+1)),"Epoch log missing rows")
    require(state["completed_epoch"]==state["stop_epoch"]==len(history),"Final epoch differs from complete log")
    require(np.isfinite(history.select_dtypes(include=["number"]).to_numpy()).all(),"Nonfinite training log")
    require((history.train_standardized_MSE>=0).all() and (history.validation_RMSE_V>=0).all(),"Negative training metric")
    train_count = next(x["sampled_origins"] for x in cfg["origin_summaries"] if x["horizon"]==h and x["partition"]=="TRAIN")
    microbatches = math.ceil(train_count/cfg["training"]["batch_size"])
    steps_per_epoch = math.ceil(microbatches/cfg["training"]["gradient_accumulation"])
    require(np.array_equal(history.optimizer_steps,np.arange(1,len(history)+1)*steps_per_epoch),"Optimizer step log inconsistent")
    require(state["optimizer_steps"]==len(history)*steps_per_epoch and state["optimizer_steps"]>0,"No genuine optimizer steps recorded")
    best_index = int(history.validation_RMSE_V.to_numpy().argmin())
    require(state["best_epoch"]==best_index+1,"Checkpoint did not select best validation epoch")
    checkpoint = torch.load(run/"best.pt",map_location="cpu",weights_only=False)
    require(checkpoint["epoch"]==state["best_epoch"] and checkpoint["seed"]==202 and checkpoint["model"]==kind and checkpoint["horizon"]==h,"Checkpoint metadata mismatch")
    require(checkpoint["protocol_sha256"]==protocol_hash,"Checkpoint protocol SHA mismatch")
    tensors = checkpoint["state_dict"]
    require(len(tensors)>0 and all(torch.isfinite(v).all().item() for v in tensors.values()),"Checkpoint weights missing/nonfinite")
    final = hashlib.sha256()
    for tensor in tensors.values():
        final.update(tensor.numpy().tobytes())
    require(final.hexdigest()!=state["initial_state_sha256"],"Final checkpoint equals untrained initialization")
    cached = np.load(run/"best_validation_prediction.npy",allow_pickle=False)
    require(cached.dtype==np.float64 and cached.shape==val_npz["prediction_delta_V"].shape,"Validation reload cache wrong shape/precision")
    difference = float(np.max(np.abs(cached-val_npz["prediction_delta_V"])))
    require(difference<=1e-7 and state["reload_max_difference_V"]<=1e-7,"Checkpoint reload prediction parity failed")
    best_score = independent_metrics(val_npz["truth_delta_V"],val_npz["prediction_delta_V"])["RMSE_V"]
    require(abs(best_score-checkpoint["validation_rmse_V"])<1e-10,"Checkpoint validation metric disagrees with full predictions")
    require(abs(best_score-state["best_validation_RMSE_V"])<1e-10,"Final recorded validation metric differs")
    require(abs(best_score-float(history.validation_RMSE_V.min()))<1e-10,"Epoch best metric differs")
    peaks = []
    for key in ["GPU_peak_allocated_MiB","GPU_peak_reserved_MiB"]:
        maximum = max(float(history[key].max()),float(state[key]))
        require(0<maximum<=8192.,f"{kind}: process GPU {key} exceeds 8 GiB budget")
        peaks.append(maximum)
    return dict(model=kind,epochs=len(history),optimizer_steps=state["optimizer_steps"],best_epoch=state["best_epoch"],
                best_validation_RMSE_V=best_score,initial_state_sha256=state["initial_state_sha256"],checkpoint_sha256=state["checkpoint_sha256"],
                checkpoint_weights_finite=True,checkpoint_changed_from_initial=True,validation_reload_vs_saved_best_max_V=difference,
                recorded_reload_max_difference_V=state["reload_max_difference_V"],GPU_peak_allocated_MiB=peaks[0],GPU_peak_reserved_MiB=peaks[1],
                checkpoint_cpu_inspection=True,independent_GPU_reload_inference=False,
                complete_training_log_sha256=file_sha(run/"training_history.csv"),
                documented_zero_optimizer_step_startup_failures=startup_failures,
                startup_recovery_preserved_initialization_and_protocol=True)


def verify(h: int) -> dict:
    started = time.time()
    cfg = read_json(OUT/"frozen_protocol.json")
    schema = read_json(OUT/"cache_schema.json")
    require(cfg["seed"]==202 and cfg["history_steps"]==300 and h in cfg["horizons_seconds"],"Frozen task identity differs")
    require(cfg["max_trained_instances"]==4 and cfg["n_features"]==24,"Frozen instance/input identity differs")
    require(cfg["voltage_channels"]==CELLS,"Ten active voltage channel identity differs")
    require(file_sha(STUDY.parent/"RealHyFC_F2_Main.py")==cfg["historical_code_sha256"],"Historical source changed")
    require(file_sha(OUT/"raw_file_inventory.csv")==cfg["source_inventory_sha256"],"Frozen source inventory changed")
    require(file_sha(OUT/f"origins_H{h}.npz")==cfg["sampling_hashes"][str(h)],"Frozen sampled origins changed")
    completion = read_json(OUT/f"H{h}_completion.json")
    require(completion["status"]=="COMPLETED" and completion["trained_instances"]==2 and completion["persistence_fitted"] is False,"Horizon completion identity differs")
    a = open_cache(schema)
    stats = check_frozen_training_statistics(cfg,a)
    partitions = stats.pop("partition")
    origins_npz = np.load(OUT/f"origins_H{h}.npz",allow_pickle=False)
    wanted = np.zeros(schema["rows"],dtype=bool)
    reference = {}; verification = []; metric_rows = []; training_records = []; total_queries = 0
    global_metrics = pd.read_csv(OUT/f"metrics_H{h}.csv",float_precision="round_trip")
    require(len(global_metrics)==36,"Global metric table lacks three models/four partitions/three strata")
    for kind in KINDS:
        run = STUDY/"runs"/(f"H{h}_Persistence" if kind=="Persistence" else f"H{h}_{kind}_Seed202")
        model_metrics = pd.read_csv(run/"metrics.csv",float_precision="round_trip")
        require(len(model_metrics)==12,"Model metric table lacks complete strata")
        validation_npz = None
        for pi, part in enumerate(PARTS):
            with np.load(run/f"predictions_{part}.npz",allow_pickle=False) as stream:
                p = {key:stream[key] for key in stream.files}
            origins = origins_npz[part]
            shape = (len(origins),h)
            require(np.array_equal(p["origins"],origins),f"{kind}/{part}: origins not frozen list")
            targets = origins[:,None]+np.arange(1,h+1,dtype=np.int64)
            require(p["targets"].dtype==np.int64 and p["target_time_ms"].dtype==np.int64 and p["origin_time_ms"].dtype==np.int64,"Row/time serialization lost integer identity")
            require(p["targets"].shape==shape and np.array_equal(p["targets"],targets),"Some future target seconds omitted/shifted")
            require(p["truth_delta_V"].dtype==np.float64 and p["prediction_delta_V"].dtype==np.float64,"Targets/predictions not saved in float64")
            require(p["truth_delta_V"].shape==p["prediction_delta_V"].shape==shape,"Direct horizon output shape differs")
            require(np.isfinite(p["prediction_delta_V"]).all(),"Nonfinite direct predictions")
            require(np.array_equal(p["target_time_ms"],a["time_ms"][targets]),"Original target DateTime serialization mismatch")
            require(np.array_equal(p["origin_time_ms"],a["time_ms"][origins]),"Original origin DateTime serialization mismatch")
            require(np.array_equal(p["origin_voltage_V"],a["voltage_V"][origins]),"Raw origin voltage anchor mismatch")
            require(np.array_equal(p["target_file_id"],a["file_id"][targets]),"Target source file metadata differs")
            require(np.array_equal(p["target_source_row"],a["source_row"][targets]),"Target source row metadata differs")
            truth = a["voltage_V"][targets]-a["voltage_V"][origins,None]
            require(np.array_equal(p["truth_delta_V"],truth),"Float64 per-second target delta truth differs")
            one_step = np.diff(np.concatenate([a["voltage_V"][origins,None],a["voltage_V"][targets]],axis=1),axis=1)
            event = np.logical_or.accumulate(np.abs(one_step)>cfg["voltage_event"]["threshold_V"],axis=1)
            require(p["event_query"].dtype==np.bool_ and np.array_equal(p["event_query"],event),"Train-threshold event labels differ")
            if kind=="Persistence":
                require(np.count_nonzero(p["prediction_delta_V"])==0,"Persistence is not exact last-value hold")
                window = check_window(cfg,a,origins,h,pi,partitions)
                wanted[window.ravel()] = True
                reference[part] = {key:array_sha(p[key]) for key in ["origins","targets","origin_time_ms","target_time_ms","origin_voltage_V","truth_delta_V","event_query","target_file_id","target_source_row"]}
            else:
                require(all(array_sha(p[key])==value for key,value in reference[part].items()),"Different models use different target/clock/event identities")
            max_difference = 0.
            for name,mask in [("ALL",np.ones(shape,dtype=bool)),("EVENT",event),("NORMAL",~event)]:
                metric = independent_metrics(truth[mask],p["prediction_delta_V"][mask])
                row = model_metrics.loc[(model_metrics.partition==part)&(model_metrics.stratum==name)]
                require(len(row)==1 and row.iloc[0].model==kind and row.iloc[0].horizon_seconds==h,"Metric table row identity differs")
                require(pd.isna(row.iloc[0].seed) if kind=="Persistence" else row.iloc[0].seed==202,"Metric seed identity differs")
                require(int(row.iloc[0].origins)==len(origins),"Metric origin count differs")
                max_difference = max(max_difference,compare_metrics(metric,row.iloc[0],f"{kind}/{part}/{name}"))
                combined = global_metrics.loc[(global_metrics.model==kind)&(global_metrics.partition==part)&(global_metrics.stratum==name)]
                require(len(combined)==1,"Global metrics duplicate or missing identity")
                compare_metrics(metric,combined.iloc[0],f"global/{kind}/{part}/{name}")
                metric_rows.append(dict(horizon_seconds=h,model=kind,partition=part,stratum=name,origins=len(origins),**metric))
            if part=="VALIDATION":
                validation_npz = p
            total_queries += int(truth.size)
            verification.append(dict(model=kind,partition=part,origins=len(origins),queries=int(truth.size),event_queries=int(event.sum()),
                                     normal_queries=int((~event).sum()),all_queries_checked=True,metric_max_difference=max_difference,
                                     predictions_sha256=file_sha(run/f"predictions_{part}.npz")))
        if kind!="Persistence":
            training_records.append(check_training(kind,h,cfg,run,validation_npz))
    raw_proof = original_rows_check(wanted,schema,a)
    records = list((STUDY/"runs").glob("H*_Seed202/training_state.json"))
    require(len(records)<=4,"More than four training instance directories")
    pd.DataFrame(metric_rows).to_csv(OUT/f"H{h}_independently_recomputed_metrics.csv",index=False,float_format="%.17g")
    return dict(**{"pass":True},horizon_seconds=h,decision="PASS_TASK_RESOURCE_METRIC_IDENTITY_ONLY",
                not_a_performance_gate=True,queries_checked_across_three_models=total_queries,
                independent_source_rows=raw_proof,partition_checks=verification,training_logs=training_records,
                frozen_training_statistics=stats,protocol_sha256=file_sha(OUT/"frozen_protocol.json"),
                verifier_sha256=file_sha(Path(__file__)),completed_training_instances_this_horizon=2,
                completed_training_instances_total=sum(read_json(p).get("status")=="COMPLETE" for p in records),
                seconds=time.time()-started)


def update_review(h: int,result: dict) -> None:
    path = OUT/"review_checkpoint2.json"
    old = read_json(path) if path.exists() else {}
    horizons = old.get("horizons",{})
    horizons[str(h)] = dict(decision="PASS" if result["pass"] else "FAIL",acceptance_file=f"H{h}_acceptance.json",
                           queries_checked=result.get("queries_checked_across_three_models"),error=result.get("error"))
    review = dict(checkpoint=2,reviewer="historical_audit agent, independent of model/runner author",
                  decision="PASS" if all(x["decision"]=="PASS" for x in horizons.values()) else "FAIL",
                  horizons=horizons,training_executed_by_reviewer=False,
                  review_source_sha256={str(p.relative_to(STUDY)):file_sha(p) for p in
                     [STUDY/"scripts/train_pilot.py",STUDY/"scripts/freeze_protocol.py",OUT/"frozen_protocol.json",Path(__file__)]},
                  checks=["Every saved query has every future step, original source time, float64 raw ten-cell target and origin anchor",
                          "All original historical/future window adjacencies, file/segment identities, validity, timestamp jitter and partition memberships checked",
                          "All three models have identical frozen origins, targets, clocks, event labels and source identities within each horizon",
                          "Independent double-precision RMSE/MAE/delta-R2/P95/P99 for all/event/normal; no shared metric function imported",
                          "Every underlying original history/target source row reread from CSV, all 17 LFS payload SHA/size values rechecked",
                          "Train-only scaler, event threshold and raw split counts independently recomputed",
                          "Complete actual epoch/optimizer logs, best validation checkpoint, saved reload predictions and <=8 GiB process GPU peak evidence checked"],
                  findings=[dict(severity="NON_BLOCKING",item="Evaluation is not an untouched independent Test",
                      evidence="Both post-validation regions have historical exposure; this acceptance checks identity, not independent generalization."),
                      dict(severity="NON_BLOCKING",item="Pilot optimization budget is unequal to architecture convergence proof",
                      evidence="Shared <=12 epochs, one Seed202 and no search; F2 and LSTM have different parameter counts and optimization difficulty."),
                      dict(severity="NON_BLOCKING",item="Finite fixed origin lattice",
                      evidence="600-second grid and 2048-origin Train cap can miss/phase-lock brief events. No claim covers every possible raw-second origin."),
                      dict(severity="NON_BLOCKING",item="GPU evidence scope",
                      evidence="PyTorch allocated/reserved process peaks are recorded; other applications' full-device usage is not independently attributed here."),
                      dict(severity="NON_BLOCKING",item="Checkpoint reload evidence scope",
                      evidence="Reviewer checks checkpoint bytes/finite tensors and complete saved-best vs evaluated Validation predictions on CPU; no additional independent GPU inference or training is performed."),
                      dict(severity="NON_BLOCKING",item="Horizon pairing",
                      evidence="H30/H120 eligibility/capped Train ranks may differ; target identity is required across models within a horizon, not across horizons."),
                      dict(severity="RESOLVED_ZERO_STEP_ENGINEERING_FAILURE",item="Initial F2 GPU device tracking",
                      evidence="The first H30 F2 attempt stopped before optimizer step 1 because a wrapped LightningModule retained CPU device metadata. The new adapter explicitly synchronizes native device metadata without changing the historical model class bodies. Acceptance requires the preserved failure record to show zero steps and exactly identical initialization/protocol on its one compatibility recovery.")],
                  fatal_findings=[] if result["pass"] else [result.get("error","Unknown acceptance failure")],
                  performance_does_not_affect_acceptance=True)
    write_json(path,review)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",type=int,choices=[30,120],required=True)
    args = parser.parse_args()
    try:
        result = verify(args.horizon)
    except Exception as error:
        result = {"pass":False,"horizon_seconds":args.horizon,"decision":"FAIL_STOP",
                  "error":repr(error),"traceback":traceback.format_exc(),"training_executed":False}
        write_json(OUT/f"H{args.horizon}_acceptance.json",result)
        update_review(args.horizon,result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        raise
    write_json(OUT/f"H{args.horizon}_acceptance.json",result)
    update_review(args.horizon,result)
    print(json.dumps(dict(stage="ACCEPTANCE_COMPLETE",horizon=args.horizon,passed=True,
                         queries=result["queries_checked_across_three_models"],seconds=result["seconds"])),flush=True)


if __name__=="__main__":
    main()

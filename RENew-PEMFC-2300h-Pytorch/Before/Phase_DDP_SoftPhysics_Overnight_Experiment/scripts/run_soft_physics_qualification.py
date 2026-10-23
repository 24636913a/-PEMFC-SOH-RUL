"""Freeze then Train-only activity test. Never run any historical training runner."""
from __future__ import annotations
import argparse,json,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from ddp_common import *
from soft_physics import *

def freeze():
    if PROTOCOL.exists():raise RuntimeError("NO_PROTOCOL_REFREEZE")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    if head!=BASE:raise RuntimeError("BASE_CHANGED")
    old=read_json(P3/"configs/P3_FROZEN_FULL450_PROTOCOL.json")
    if file_hash(P3/"configs/P3_FROZEN_FULL450_PROTOCOL.json")!=old_digest():
        raise RuntimeError("P3_PROTOCOL_SHA_MISMATCH")
    expected={**old["H"]["code_identity"]["files"],**old["code_identity"]["files"]}
    paths=[x for x in STATIC if x.suffix!=".pt"]
    paths+=sorted((PHASE/"scripts").glob("*.py"))+sorted((PHASE/"tests").glob("*.py"))
    records={}
    for path in paths:
        name=str(path.relative_to(REPO)).replace("\\","/")
        actual=file_hash(path)
        if name in expected and actual!=expected[name]:
            raise RuntimeError("P3_FROZEN_SOURCE_CHANGED:"+name)
        records[str(path.resolve())]=actual
    for name,sha in old["H"]["code_identity"]["files"].items():
        if name.startswith("C:"):
            if file_hash(name)!=sha:raise RuntimeError("UPSTREAM_TFT_CODE_CHANGED")
            records[name]=sha
    for seed in (42,52,62):
        for role in ("best","last450"):
            result=read_json(P3/f"outputs/training_logs/P3-G2-Seed{seed}/{role}_result.json")
            path=P3/result["checkpoint"]
            if file_hash(path)!=result["checkpoint_sha256"]:
                raise RuntimeError("P3_CHECKPOINT_FILE_SHA_MISMATCH")
            records[str(path.resolve())]=result["checkpoint_sha256"]
    cfg={"phase":PHASE.name,"base_commit":BASE,"frozen_UTC":utc(),
         "authorization":"USER_CONDITIONAL_SOFT_PHYSICS_3_RUNS",
         "H":old["H"],"seeds":[42,52,62],"maximum_formal_runs":3,"epochs_each":450,
         "candidate":{"minimum_V":0.0,"maximum_V":None,"upper_bound_certified":False,
                      "domain":"conditional normal fuel-cell operation only; not fault/reversal",
                      "scale":"frozen unfiltered Train voltage std","lambda":0.01,
                      "no_other_bound_or_weight_search":True},
         "train_activity_gate":{"all_Train_points":True,
              "minimum_activation_fraction":0.001,"minimum_loss":1e-12,
              "required_in_fresh_and_at_least_one_saved_role":True,
              "max_weighted_physics_to_data_MSE_ratio":0.1,
              "threshold_type":"PROJECT_SCREENING_DESIGN_NOT_PHYSICAL_BOUNDS"},
         "read_nrows":3007,"filter":"bin_end_ophrs2_h < 1819.997738","expected_rows":3005,
         "qualification_windows":"Train only [0,2103); no Validation/Assessment construction",
         "new_health_fit":False,"RUL_training":False,
         "input_and_code_sha256":records,
         "execution_identity_sha256":hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}
    write_json(PROTOCOL,cfg)
    PROTOCOL.with_suffix(".sha256").write_text(file_hash(PROTOCOL)+"\n",encoding="utf-8")
    write_json(PHASE/"outputs/SOURCE_AND_CODE_MANIFEST.json",cfg["input_and_code_sha256"])
    print("FROZEN "+file_hash(PROTOCOL),flush=True)

def old_digest():
    return (P3/"configs/P3_FROZEN_FULL450_PROTOCOL.sha256").read_text().split()[0]

def prepare_train(cfg,e):
    ecfg,_=e.load_protocol()
    dev,prefix_fingerprint=read_development_prefix()
    # Slice BEFORE window builder: it cannot inspect Assessment/Validation values.
    train=e.c.build_windows(dev.iloc[:2103].copy(),ecfg,0,2103)
    if len(train.origins)!=1380:raise RuntimeError("TRAIN_WINDOW_COUNT_CHANGED")
    scaler=e.CommonScaler().fit(train.x)
    authority=read_json(E/"outputs/E_COMMON_SCALER.json")
    if not np.array_equal(scaler.mean,np.asarray(authority["mean"])) or not np.array_equal(scaler.scale,np.asarray(authority["scale"])):
        raise RuntimeError("P3_SCALER_NOT_REPRODUCED")
    tx=e.prep(train.x,scaler,cfg["H"]["preprocessing"]["Top10_indices"],"EWMA")
    observed={"train_raw_x_sha256":array_hash(train.x),"train_preprocessed_x_sha256":array_hash(tx),
              "train_y_sha256":array_hash(train.y),"train_origins_sha256":array_hash(train.origins),
              "train_target_rows_sha256":array_hash(train.targets),
              "scaler_sha256":array_hash(np.stack([scaler.mean,scaler.scale])),
              "target_scale_V":scaler.target_scale,"development_prefix_fingerprint_sha256":prefix_fingerprint}
    old=read_json(P3/"outputs/audits/P3_DATA_AUTHORITY.json")
    for key,value in observed.items():
        if value!=old[key]:raise RuntimeError("P3_TRAIN_AUTHORITY_MISMATCH:"+key)
    anchor=dev.cell_voltage_mean.to_numpy()[train.origins].astype(np.float32)
    if not np.array_equal(anchor,train.x[:,-1,0]):raise RuntimeError("UNFILTERED_ANCHOR_NOT_EXACT")
    return train,scaler,tx,anchor,observed

def qualify():
    target=PHASE/"outputs/QUALIFICATION_DECISION.json"
    if target.exists():raise RuntimeError("QUALIFICATION_ALREADY_EXECUTED_NO_RERUN")
    started=time.perf_counter()
    try:
        cfg,digest=load_protocol()
        tests=read_json(PHASE/"outputs/SYNTHETIC_TEST_RESULTS.json")
        if tests["returncode"]!=0:raise RuntimeError("SYNTHETIC_TESTS_FAILED")
        a,e=load_legacy();env,asts=a.configure_runtime(cfg["H"])
        device=torch.device("cuda")
        train,scaler,tx,anchor,data=prepare_train(cfg,e)
        write_json(PHASE/"outputs/TRAIN_DATA_AUTHORITY.json",data)
        rows=[];identities=[]
        for seed in cfg["seeds"]:
            a.set_seed(seed);cohort=a.build_cohort(seed,cfg["H"],device)
            model=cohort["G2"]
            # No G0/G1 training, just the original initialization cohort.
            del cohort
            expected=read_json(P3/f"outputs/training_logs/P3-G2-Seed{seed}/initial.json")
            initial=a.state_digest(model.state_dict())
            if initial!=expected["initial_model_sha256"]:raise RuntimeError("P3_INITIAL_WEIGHTS_CHANGED")
            gen=torch.Generator().manual_seed(seed)
            rng=a.rng_digest(a.rng_snapshot(),gen)
            if rng!=expected["initial_training_rng"]:raise RuntimeError("P3_INITIAL_RNG_CHANGED")
            identities.append({"seed":seed,"initial_weight_match":True,"initial_RNG_match":True,
                               "parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad)})
            if identities[-1]["parameter_count"]!=332436:raise RuntimeError("PARAMETERS_CHANGED")
            print("QUALIFY fresh Seed"+str(seed),flush=True)
            # Initial Train mode with dropout and original shuffled first-epoch order; no backward/optimizer.
            loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(train.y/scaler.target_scale),
                                           torch.arange(len(tx))),batch_size=32,shuffle=True,generator=gen,num_workers=0)
            model.train();pred=np.empty_like(train.y);order=[]
            with torch.no_grad():
                for xb,yb,ids in loader:
                    pred[ids.numpy()]=model.predict_delta(xb.to(device)).cpu().numpy()
                    order.extend(train.origins[ids.numpy()].tolist())
            authority=P3/f"outputs/audits/sampler_authority/seed{seed}_epoch001.json"
            # One explicit small Train sampler identity is allowlisted; no other epochs opened.
            old_order=read_json(authority)
            if order!=old_order["origin_order"]:raise RuntimeError("FIRST_EPOCH_SAMPLER_NOT_MATCHED")
            identities[-1]["first_epoch_sampler_match"]=True
            roles=[("fresh_train_mode",pred)]
            for role in ("best","last450"):
                result=read_json(P3/f"outputs/training_logs/P3-G2-Seed{seed}/{role}_result.json")
                checkpoint=torch.load(P3/result["checkpoint"],map_location="cpu",weights_only=True)
                if checkpoint["run_id"]!=f"P3-G2-Seed{seed}" or checkpoint["model"]!="G2" or checkpoint["seed"]!=seed or checkpoint["protocol_sha256"]!=old_digest():
                    raise RuntimeError("P3_CHECKPOINT_METADATA_MISMATCH")
                if checkpoint["initial_model_sha256"]!=initial:raise RuntimeError("CHECKPOINT_INITIAL_IDENTITY_MISMATCH")
                model.load_state_dict(checkpoint["state_dict"],strict=True)
                print("QUALIFY "+role+" Seed"+str(seed),flush=True)
                roles.append((role,a.predict_synthetic(model,tx,device)))
                del checkpoint
            for role,prediction in roles:
                ps=torch.from_numpy(prediction)
                total,mse,phys,voltage=combined_objective(ps,torch.from_numpy(train.y/scaler.target_scale),
                    torch.from_numpy(anchor),scaler.target_scale,cfg["candidate"]["lambda"],0.0,None)
                row={"seed":seed,"role":role,"train_origins":len(train.origins),
                     **physics_activity(voltage,scaler.target_scale),
                     "data_MSE_dimensionless":float(mse),
                     "weighted_physics_to_MSE_ratio":float(0.01*phys/mse),
                     "parameter_updates":0}
                rows.append(row);print(json.dumps(row),flush=True)
                write_json(PHASE/"outputs/TRAIN_PHYSICS_ACTIVITY.json",rows)
            del model;torch.cuda.empty_cache()
        gate=cfg["train_activity_gate"]
        active=lambda x:x["activation_fraction"]>=gate["minimum_activation_fraction"] and x["physics_loss_dimensionless"]>gate["minimum_loss"]
        activity_pass=all(active(next(x for x in rows if x["seed"]==seed and x["role"]=="fresh_train_mode")) and
                          any(active(x) for x in rows if x["seed"]==seed and x["role"]!="fresh_train_mode") for seed in cfg["seeds"])
        reasons=[]
        if not activity_pass:reasons.append("PHYSICS_LOSS_INACTIVE")
        # This weak half-space is not certified for the whole future fault/transient domain.
        reasons.append("NO_CERTIFIED_NONTRIVIAL_ELECTROCHEMICAL_ENVELOPE")
        balance=all(x["weighted_physics_to_MSE_ratio"]<=gate["max_weighted_physics_to_data_MSE_ratio"] for x in rows)
        if not balance:reasons.append("PHYSICS_WEIGHT_SCALE_IMBALANCE")
        decision={"status":"SOFT_PHYSICS_NOT_QUALIFIED","reasons":reasons,
                  "formal_training_admitted":False,"formal_runs_started":0,"epochs_trained":0,
                  "source_identity_pass":True,"physical_domain_pass":False,
                  "train_activity_pass":activity_pass,"weight_balance_pass":balance,"model_parity_pass":True,
                  "protocol_sha256":digest,"execution_identity_sha256":cfg["execution_identity_sha256"],
                  "initialization_evidence":identities,"runtime_environment":env,"AST_identity":asts,
                  "Train_forward_checks":len(rows),"Train_predictions_per_check":1380*24,
                  "all_train_forward_checks_zero":all(x["violations"]==0 for x in rows),
                  "qualification_time_s":time.perf_counter()-started,"Test_accessed":False,
                  "Assessment_accessed":False,"new_validation_forward":False,
                  "neural_optimizer_steps":0,"completed_UTC":utc()}
        write_json(target,decision)
        import pandas as pd
        pd.DataFrame(rows).to_csv(PHASE/"outputs/TRAIN_PHYSICS_ACTIVITY.csv",index=False,float_format="%.17g")
        print(json.dumps(decision,ensure_ascii=False),flush=True)
    except Exception:
        write_json(PHASE/"outputs/QUALIFICATION_TECHNICAL_FAILURE.json",
                   {"exception":traceback.format_exc(),"formal_runs_started":0,"UTC":utc()})
        raise
    finally:
        write_json(PHASE/"outputs/RUNTIME_DATA_ACCESS_AUDIT.json",runtime_access_audit())

def formal_entry():
    cfg,_=load_protocol()
    require_training_admission(read_json(PHASE/"outputs/QUALIFICATION_DECISION.json"))
    # No runnable 450-epoch job is released absent a certified physical domain.
    raise PermissionError("NO_CERTIFIED_PHYSICAL_DOMAIN_NO_FORMAL_WORKER_RELEASED")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--qualify",action="store_true")
    mode.add_argument("--run",action="store_true")
    args=parser.parse_args();sys.dont_write_bytecode=True;install_firewall()
    if args.freeze:freeze()
    elif args.qualify:qualify()
    else:formal_entry()

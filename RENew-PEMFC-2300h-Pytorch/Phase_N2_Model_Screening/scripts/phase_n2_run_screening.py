"""Modern-model screening, multi-seed, horizon and 1 h robustness for Phase N2."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from phase_n2_common import PHASE_ROOT, FoldPreprocessor, build_windows, development_folds, load_config, load_development_data, metric_dict, per_horizon_metrics, write_csv, write_json
from phase_n2_models import train_model_run
from phase_n2_run_diagnostics import moving_block_bootstrap


LEARNED_MODELS = ["LSTM", "TCN", "iTransformer", "TimeMixer", "PatchTST"]


def simple_predictions(train, val, alpha: float, cycle_steps: int) -> dict[str, np.ndarray]:
    h = val.y_abs.shape[1]
    persistence = np.repeat(val.anchor[:, None], h, axis=1)
    history = val.x[:, :, 0].astype(float)
    x = np.arange(history.shape[1], dtype=float); xf = np.arange(history.shape[1], history.shape[1] + h, dtype=float)
    linear = np.empty_like(val.y_abs, dtype=float); cycle = np.empty_like(val.y_abs, dtype=float)
    for i, values in enumerate(history):
        slope, intercept = np.polyfit(x, values, 1); linear[i] = intercept + slope * xf
        period = values[-cycle_steps:]; cycle[i] = period[np.arange(h) % cycle_steps]
    pre = FoldPreprocessor().fit(train.x)
    ridge = Ridge(alpha=alpha).fit(pre.transform(train.x).reshape(len(train.x), -1), train.y_delta)
    ridge_pred = val.anchor[:, None] + ridge.predict(pre.transform(val.x).reshape(len(val.x), -1))
    return {"Persistence": persistence, "LocalLinear": linear, "CycleRepeat": cycle, "Ridge": ridge_pred}


def baseline_denominators(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    rmses = {name: metric_dict(y_true, pred)["rmse_V"] for name, pred in predictions.items()}
    return {**rmses, "best": min(rmses.values()), "best_model": min(rmses, key=rmses.get)}


def add_skill(metrics: dict[str, float], den: dict[str, float]) -> dict[str, float]:
    return {
        **metrics,
        "skill_persistence": 1 - metrics["rmse_V"] / den["Persistence"],
        "skill_cycle": 1 - metrics["rmse_V"] / den["CycleRepeat"],
        "skill_best": 1 - metrics["rmse_V"] / den["best"],
        "strongest_simple_baseline": den["best_model"],
        "strongest_simple_rmse_V": den["best"],
    }


def fold_windows(frame, features, lookback_h: float, horizon_h: float, resolution: float, cfg: dict):
    l = round(lookback_h / resolution); h = round(horizon_h / resolution)
    for spec in development_folds(cfg):
        train = build_windows(frame, features, l, h, float(spec["train_start_h"]), float(spec["train_end_h"]), resolution, cfg)
        val = build_windows(frame, features, l, h, float(spec["validation_start_h"]), float(spec["validation_end_h"]), resolution, cfg)
        yield str(spec["fold"]), train, val


def model_run_id(stage: str, model: str, fold: str, feature_group: str, lookback_h: float, horizon_h: float, resolution: float, seed: int) -> str:
    def f(v: float) -> str: return str(v).replace(".", "p")
    return f"{stage}_{model}_{feature_group}_L{f(lookback_h)}h_H{f(horizon_h)}h_R{f(resolution)}h_{fold}_seed{seed}"


def main() -> None:
    cfg = load_config()
    diagnostic_path = PHASE_ROOT / "outputs" / "frozen_config" / "diagnostic_selection.json"
    if not diagnostic_path.is_file(): raise RuntimeError("Diagnostic gate missing")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not diagnostic.get("matched_origin_audit_pass"): raise RuntimeError("Matched-Origin gate failed")
    group = diagnostic["selected_feature_group"]; features = cfg["feature_groups"][group]
    lookback_h = float(diagnostic["selected_lookback_hours"]); primary_h = float(cfg["task"]["primary_horizon_hours"])
    resolution = .5; seed42 = int(cfg["training"]["screening_seed"])
    alpha = float(json.loads((PHASE_ROOT / "outputs" / "baseline_results" / "ridge_alpha_selection.json").read_text(encoding="utf-8"))["selected_ridge_alpha"])
    frame, firewall = load_development_data(cfg, resolution, features)
    if firewall["target_rows_read_from_test"] != 0: raise RuntimeError("TEST FIREWALL VIOLATION")

    # Shared references and model screening: same features, origins, targets and scalers policy.
    references: dict[str, dict] = {}; learned: dict[str, dict[str, dict]] = {m: {} for m in LEARNED_MODELS}
    baseline_ref_rows: list[dict] = []
    for fold, train, val in fold_windows(frame, features, lookback_h, primary_h, resolution, cfg):
        bp = simple_predictions(train, val, alpha, int(cfg["task"]["cycle_period_steps_30min"])); den = baseline_denominators(val.y_abs, bp)
        references[fold] = {"predictions": bp, "den": den, "y_true": val.y_abs, "origin_age_h": val.origin_age_h, "anchor": val.anchor}
        for name, pred in bp.items(): baseline_ref_rows.append({"fold": fold, "model": name, "feature_group": group, "lookback_hours": lookback_h, "horizon_hours": primary_h, **add_skill(metric_dict(val.y_abs, pred), den)})
        for model in LEARNED_MODELS:
            # The existing F2/24 h/12 h LSTM run is exactly the screening protocol.
            if model == "LSTM" and group == "F2" and lookback_h == 24 and primary_h == 12:
                run_id = f"feature_F2_LSTM_{fold}_seed{seed42}"
            else:
                run_id = model_run_id("screening", model, fold, group, lookback_h, primary_h, resolution, seed42)
            result, arrays = train_model_run(run_id, model, train, val, cfg["architectures"][model], cfg["training"], seed42, {"stage": "modern_screening", "fold": fold, "feature_group": group, "feature_count": len(features), "lookback_hours": lookback_h, "horizon_hours": primary_h, "resolution_hours": resolution})
            if not np.array_equal(arrays["y_true"], val.y_abs): raise RuntimeError("Screening target mismatch")
            learned[model][fold] = {"result": result, "arrays": arrays}
    write_csv(pd.DataFrame(baseline_ref_rows), PHASE_ROOT / "outputs" / "baseline_results" / "screening_reference_baselines.csv")

    summary_rows: list[dict] = []; per_h_rows: list[dict] = []
    for model in LEARNED_MODELS:
        for fold in references:
            result = learned[model][fold]["result"]; arrays = learned[model][fold]["arrays"]; den = references[fold]["den"]
            metrics = add_skill(metric_dict(arrays["y_true"], arrays["y_pred"]), den)
            summary_rows.append({"aggregation": "fold", "fold": fold, "model": model, "seed": seed42, "feature_group": group, "lookback_hours": lookback_h, "horizon_hours": primary_h, "validation_origin_count": len(arrays["y_true"]), "parameter_count": result["parameter_count"], "peak_gpu_memory_mb": result["peak_gpu_memory_mb"], "mean_epoch_time_s": result["mean_epoch_time_s"], "total_training_time_s": result["total_training_time_s"], "best_epoch": result["best_epoch"], **metrics})
            ph = pd.DataFrame(per_horizon_metrics(arrays["y_true"], arrays["y_pred"], resolution))
            ref_ph = {name: pd.DataFrame(per_horizon_metrics(arrays["y_true"], pred, resolution)) for name, pred in references[fold]["predictions"].items()}
            for _, row in ph.iterrows():
                step = int(row.horizon_step); rmap = {n: float(t.loc[t.horizon_step == step, "rmse_V"].iloc[0]) for n, t in ref_ph.items()}
                per_h_rows.append({"fold": fold, "model": model, "seed": seed42, **row.to_dict(), "skill_persistence": 1-row.rmse_V/rmap["Persistence"], "skill_cycle": 1-row.rmse_V/rmap["CycleRepeat"], "skill_best": 1-row.rmse_V/min(rmap.values()), "strongest_simple_baseline": min(rmap, key=rmap.get)})

    fold_summary = pd.DataFrame(summary_rows)
    means = fold_summary.groupby("model", as_index=False).agg(rmse_V=("rmse_V","mean"),rmse_mV=("rmse_mV","mean"),mae_V=("mae_V","mean"),mae_mV=("mae_mV","mean"),bias_V=("bias_V","mean"),bias_mV=("bias_mV","mean"),r2=("r2","mean"),skill_persistence=("skill_persistence","mean"),skill_cycle=("skill_cycle","mean"),skill_best=("skill_best","mean"),worst_fold_skill_best=("skill_best","min"),positive_skill_fold_count=("skill_best",lambda x:int((x>0).sum())),validation_origin_count=("validation_origin_count","sum"),parameter_count=("parameter_count","first"),peak_gpu_memory_mb=("peak_gpu_memory_mb","max"),mean_epoch_time_s=("mean_epoch_time_s","mean"),total_training_time_s=("total_training_time_s","sum"),best_epoch=("best_epoch","mean"))
    means.insert(0,"horizon_hours",primary_h); means.insert(0,"lookback_hours",lookback_h); means.insert(0,"feature_group",group); means.insert(0,"seed",seed42); means.insert(0,"fold","mean_3fold"); means.insert(0,"aggregation","mean")
    screening_summary = pd.concat([fold_summary, means], ignore_index=True, sort=False)
    modern_candidates = means[means.model != "LSTM"].sort_values(["rmse_V","parameter_count"])
    top2 = modern_candidates.head(2).model.tolist()
    if len(top2) != 2: raise RuntimeError("Could not select Top-2")

    # Multi-seed robustness for Top-2 plus LSTM reference.
    multi_rows: list[dict] = []
    stable_models = list(dict.fromkeys(["LSTM", *top2]))
    for model in stable_models:
        for seed in cfg["training"]["robustness_seeds"]:
            for fold, train, val in fold_windows(frame, features, lookback_h, primary_h, resolution, cfg):
                if seed == seed42 and model == "LSTM" and group == "F2" and lookback_h == 24:
                    run_id = f"feature_F2_LSTM_{fold}_seed{seed42}"
                elif seed == seed42:
                    run_id = model_run_id("screening", model, fold, group, lookback_h, primary_h, resolution, seed)
                else:
                    run_id = model_run_id("multiseed", model, fold, group, lookback_h, primary_h, resolution, seed)
                result, arrays = train_model_run(run_id, model, train, val, cfg["architectures"][model], cfg["training"], int(seed), {"stage": "multi_seed", "fold": fold, "feature_group": group, "lookback_hours": lookback_h, "horizon_hours": primary_h, "resolution_hours": resolution})
                den = references[fold]["den"]; metrics = add_skill(metric_dict(arrays["y_true"], arrays["y_pred"]), den)
                multi_rows.append({"row_type":"fold","model":model,"seed":seed,"fold":fold,"parameter_count":result["parameter_count"],"peak_gpu_memory_mb":result["peak_gpu_memory_mb"],"mean_epoch_time_s":result["mean_epoch_time_s"],"total_training_time_s":result["total_training_time_s"],**metrics})
    multi = pd.DataFrame(multi_rows)
    seed_summary = multi.groupby(["model","seed"],as_index=False).agg(rmse_V=("rmse_V","mean"),rmse_mV=("rmse_mV","mean"),mae_V=("mae_V","mean"),mae_mV=("mae_mV","mean"),skill_best=("skill_best","mean"),skill_cycle=("skill_cycle","mean"),worst_fold_skill_best=("skill_best","min"),positive_skill_fold_count=("skill_best",lambda x:int((x>0).sum())),parameter_count=("parameter_count","first"),peak_gpu_memory_mb=("peak_gpu_memory_mb","max"),mean_epoch_time_s=("mean_epoch_time_s","mean"),total_training_time_s=("total_training_time_s","sum"))
    seed_summary.insert(0,"fold","mean_3fold"); seed_summary.insert(0,"row_type","seed_summary")
    model_summary = seed_summary.groupby("model",as_index=False).agg(rmse_V=("rmse_V","mean"),rmse_mV=("rmse_mV","mean"),rmse_mV_std=("rmse_mV","std"),rmse_mV_min=("rmse_mV","min"),rmse_mV_max=("rmse_mV","max"),mae_V=("mae_V","mean"),mae_mV=("mae_mV","mean"),skill_best=("skill_best","mean"),skill_best_std=("skill_best","std"),skill_best_min=("skill_best","min"),skill_best_max=("skill_best","max"),skill_cycle=("skill_cycle","mean"),worst_fold_skill_best=("worst_fold_skill_best","min"),positive_skill_fold_count=("positive_skill_fold_count","sum"),parameter_count=("parameter_count","first"),peak_gpu_memory_mb=("peak_gpu_memory_mb","max"),mean_epoch_time_s=("mean_epoch_time_s","mean"),total_training_time_s=("total_training_time_s","sum"))
    model_summary["rmse_mV_ci95_halfwidth"] = 1.96*model_summary["rmse_mV_std"]/np.sqrt(len(cfg["training"]["robustness_seeds"])); model_summary["seed"]="all_3"; model_summary["fold"]="all_3"; model_summary.insert(0,"row_type","model_summary")
    multi_out = pd.concat([multi,seed_summary,model_summary],ignore_index=True,sort=False)

    # Choose deterministic backbone with the frozen near-tie simplicity rule.
    candidate_summary = model_summary.sort_values(["rmse_V","parameter_count"]).copy()
    best = candidate_summary.iloc[0]; final_model = str(best.model)
    if len(candidate_summary)>1:
        second=candidate_summary.iloc[1]
        if float(second.rmse_V/best.rmse_V-1) < float(cfg["selection"]["near_tie_fraction"]) and int(second.parameter_count)<int(best.parameter_count): final_model=str(second.model)
    screening_mean = means.set_index("model")
    final_multi = model_summary.set_index("model").loc[final_model]
    learned_go = bool(screening_mean.loc[final_model,"skill_best"]>0 and screening_mean.loc[final_model,"positive_skill_fold_count"]>=2 and final_multi.skill_best>0)

    # Horizon sensitivity for Top-2, seed 42 only; references use identical origins per task.
    horizon_rows: list[dict] = []; horizon_predictions: dict[tuple[str,float,str],tuple[np.ndarray,np.ndarray,dict]] = {}
    for horizon_h in [float(v) for v in cfg["task"]["horizon_candidates_hours"]]:
        for fold, train, val in fold_windows(frame, features, lookback_h, horizon_h, resolution, cfg):
            bp=simple_predictions(train,val,alpha,int(cfg["task"]["cycle_period_steps_30min"])); den=baseline_denominators(val.y_abs,bp)
            for model in top2:
                if horizon_h==primary_h:
                    run_id=model_run_id("screening",model,fold,group,lookback_h,primary_h,resolution,seed42)
                else:
                    run_id=model_run_id("horizon",model,fold,group,lookback_h,horizon_h,resolution,seed42)
                result,arrays=train_model_run(run_id,model,train,val,cfg["architectures"][model],cfg["training"],seed42,{"stage":"horizon_sensitivity","fold":fold,"feature_group":group,"lookback_hours":lookback_h,"horizon_hours":horizon_h,"resolution_hours":resolution})
                metrics=add_skill(metric_dict(arrays["y_true"],arrays["y_pred"]),den)
                horizon_rows.append({"aggregation":"fold","model":model,"fold":fold,"seed":seed42,"lookback_hours":lookback_h,"horizon_hours":horizon_h,"origin_count":len(val.x),**metrics})
                horizon_predictions[(model,horizon_h,fold)]=(arrays["y_true"],arrays["y_pred"],den)
    horizon_df=pd.DataFrame(horizon_rows)
    horizon_mean=horizon_df.groupby(["model","horizon_hours"],as_index=False).agg(rmse_V=("rmse_V","mean"),rmse_mV=("rmse_mV","mean"),mae_V=("mae_V","mean"),mae_mV=("mae_mV","mean"),skill_best=("skill_best","mean"),skill_cycle=("skill_cycle","mean"),worst_fold_skill_best=("skill_best","min"),positive_skill_fold_count=("skill_best",lambda x:int((x>0).sum())),origin_count=("origin_count","sum"))
    horizon_mean.insert(0,"fold","mean_3fold"); horizon_mean.insert(0,"aggregation","mean")
    horizon_out=pd.concat([horizon_df,horizon_mean],ignore_index=True,sort=False)

    # 1 h robustness: strongest aggregate simple baseline and final candidate only.
    baseline_means=pd.DataFrame(baseline_ref_rows).groupby("model")["rmse_V"].mean(); strongest_baseline=str(baseline_means.idxmin())
    robustness_rows=[]
    for res in [.5,1.0]:
        if res==.5: rframe=frame
        else:
            rframe, fw1=load_development_data(cfg,resolution_h=res,required_columns=features)
            if fw1["target_rows_read_from_test"]!=0: raise RuntimeError("TEST FIREWALL VIOLATION")
        cycle_steps=round(float(cfg["task"]["cycle_period_hours"])/res)
        for fold,train,val in fold_windows(rframe,features,lookback_h,primary_h,res,cfg):
            bp=simple_predictions(train,val,alpha,cycle_steps); den=baseline_denominators(val.y_abs,bp)
            bpred=bp[strongest_baseline]; robustness_rows.append({"resolution_hours":res,"model":strongest_baseline,"fold":fold,"role":"strongest_simple_baseline",**add_skill(metric_dict(val.y_abs,bpred),den)})
            if res==.5:
                if final_model=="LSTM" and group=="F2": run_id=f"feature_F2_LSTM_{fold}_seed{seed42}"
                else: run_id=model_run_id("screening",final_model,fold,group,lookback_h,primary_h,res,seed42)
            else: run_id=model_run_id("robustness1h",final_model,fold,group,lookback_h,primary_h,res,seed42)
            result,arrays=train_model_run(run_id,final_model,train,val,cfg["architectures"][final_model],cfg["training"],seed42,{"stage":"1h_robustness","fold":fold,"feature_group":group,"lookback_hours":lookback_h,"horizon_hours":primary_h,"resolution_hours":res})
            robustness_rows.append({"resolution_hours":res,"model":final_model,"fold":fold,"role":"final_candidate",**add_skill(metric_dict(arrays["y_true"],arrays["y_pred"]),den)})
    robustness=pd.DataFrame(robustness_rows)
    robmean=robustness.groupby(["resolution_hours","model","role"],as_index=False).agg(rmse_V=("rmse_V","mean"),rmse_mV=("rmse_mV","mean"),mae_V=("mae_V","mean"),mae_mV=("mae_mV","mean"),skill_best=("skill_best","mean"),skill_cycle=("skill_cycle","mean")); robmean["fold"]="mean_3fold"; robustness=pd.concat([robustness,robmean],ignore_index=True,sort=False)

    # Paired moving-block bootstrap: final candidate against strongest simple baseline.
    pairs={}
    for fold in references:
        yt=learned[final_model][fold]["arrays"]["y_true"] if final_model in learned else references[fold]["y_true"]
        pc=learned[final_model][fold]["arrays"]["y_pred"]
        pr=references[fold]["predictions"][strongest_baseline]
        pairs[fold]=(yt,pc,pr)
    modern_boot=moving_block_bootstrap(pairs,cfg,f"{final_model}_vs_{strongest_baseline}","top_modern_model",final_model,strongest_baseline)
    boot_path=PHASE_ROOT/"outputs"/"statistical_tests"/"bootstrap_pairwise_comparison.csv"
    boot=pd.read_csv(boot_path) if boot_path.is_file() else pd.DataFrame()
    boot=pd.concat([boot,pd.DataFrame([modern_boot])],ignore_index=True,sort=False)

    # Efficiency and outputs.
    efficiency=means[["model","parameter_count","peak_gpu_memory_mb","mean_epoch_time_s","total_training_time_s","rmse_mV","skill_best"]].copy(); efficiency["screening_seed"]=seed42; efficiency["fold_count"]=3; efficiency["implementation"]="local_auditable_phase_n2_v1"; efficiency["torch_version"]=next(iter(learned["LSTM"].values()))["result"]["torch_version"]
    out=PHASE_ROOT/"outputs"
    write_csv(screening_summary,out/"model_screening"/"modern_model_screening_summary.csv")
    write_csv(pd.DataFrame(per_h_rows),out/"model_screening"/"modern_model_per_horizon_metrics.csv")
    write_csv(multi_out,out/"robustness_analysis"/"multi_seed_robustness.csv")
    write_csv(horizon_out,out/"robustness_analysis"/"horizon_sensitivity.csv")
    write_csv(robustness,out/"robustness_analysis"/"1h_robustness_comparison.csv")
    write_csv(boot,out/"statistical_tests"/"bootstrap_pairwise_comparison.csv")
    write_csv(efficiency,out/"model_screening"/"model_efficiency_summary.csv")
    baseline_ph=pd.read_csv(out/"baseline_results"/"baseline_per_horizon_metrics.csv"); baseline_ph["source"]="initial_F1_baseline"; modern_ph=pd.DataFrame(per_h_rows); modern_ph["source"]="modern_screening"; write_csv(pd.concat([baseline_ph,modern_ph],ignore_index=True,sort=False),out/"model_screening"/"per_horizon_metrics.csv")

    # Figures: all Validation-only.
    figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    ph=pd.DataFrame(per_h_rows).groupby(["model","horizon_hours"],as_index=False)["skill_best"].mean()
    fig,ax=plt.subplots(figsize=(9,5))
    for model,g in ph.groupby("model"): ax.plot(g.horizon_hours,g.skill_best,label=model)
    ax.axhline(0,color="black",lw=1); ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("Skill_Best"); ax.set_title("Per-horizon learned skill (development Validation only)"); ax.legend(ncol=2); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir/"per_horizon_skill_curves.png",dpi=180); plt.close(fig)
    cycle_compare=pd.concat([means[["model","rmse_mV"]],pd.DataFrame([{"model":"CycleRepeat","rmse_mV":baseline_means["CycleRepeat"]*1000}])]).sort_values("rmse_mV")
    plt.figure(figsize=(9,5)); plt.bar(cycle_compare.model,cycle_compare.rmse_mV,color="#2563EB"); plt.ylabel("Validation RMSE (mV)"); plt.title("Cycle Repeat versus learned models"); plt.xticks(rotation=20); plt.grid(axis="y",alpha=.25); plt.tight_layout(); plt.savefig(figdir/"cycle_repeat_vs_models.png",dpi=180); plt.close()
    p=means.sort_values("rmse_mV"); plt.figure(figsize=(9,5)); plt.bar(p.model,p.rmse_mV,color="#10B981"); plt.ylabel("Validation RMSE (mV)"); plt.title("Modern model screening RMSE (three-fold mean)"); plt.xticks(rotation=20); plt.grid(axis="y",alpha=.25); plt.tight_layout(); plt.savefig(figdir/"modern_model_validation_rmse.png",dpi=180); plt.close()
    p=means.sort_values("skill_best",ascending=False); plt.figure(figsize=(9,5)); plt.bar(p.model,p.skill_best,color="#F59E0B"); plt.axhline(0,color="black",lw=1); plt.ylabel("Skill_Best"); plt.title("Modern model skill over strongest simple baseline"); plt.xticks(rotation=20); plt.grid(axis="y",alpha=.25); plt.tight_layout(); plt.savefig(figdir/"modern_model_skill_best.png",dpi=180); plt.close()
    seedplot=seed_summary.copy(); fig,ax=plt.subplots(figsize=(9,5));
    for model,g in seedplot.groupby("model"): ax.plot(g.seed.astype(str),g.rmse_mV,marker="o",label=model)
    ax.set_xlabel("Random seed"); ax.set_ylabel("Three-fold mean RMSE (mV)"); ax.set_title("Multi-seed variability"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir/"multi_seed_variability.png",dpi=180); plt.close(fig)
    plt.figure(figsize=(8,5)); plt.scatter(efficiency.parameter_count,efficiency.rmse_mV,s=70)
    for _,r in efficiency.iterrows(): plt.annotate(r.model,(r.parameter_count,r.rmse_mV),xytext=(5,4),textcoords="offset points")
    plt.xscale("log"); plt.xlabel("Trainable parameters (log scale)"); plt.ylabel("Validation RMSE (mV)"); plt.title("Accuracy versus parameter count"); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(figdir/"model_accuracy_vs_parameter_count.png",dpi=180); plt.close()
    # Representative median-error validation origin from fold 3, final candidate.
    arr=learned[final_model]["fold_3"]["arrays"]; origin_rmse=np.sqrt(np.mean((arr["y_pred"]-arr["y_true"])**2,axis=1)); i=int(np.argsort(origin_rmse)[len(origin_rmse)//2]); hours=np.arange(1,arr["y_true"].shape[1]+1)*resolution
    plt.figure(figsize=(9,5)); plt.plot(hours,arr["y_true"][i],label="Actual",lw=2); plt.plot(hours,arr["y_pred"][i],label=final_model,lw=2); plt.axhline(arr["anchor"][i],color="gray",ls="--",label="Origin voltage"); plt.xlabel("Forecast horizon (h)"); plt.ylabel("10-cell mean voltage (V)"); plt.title("Representative Fold 3 Validation trajectory"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(figdir/"representative_validation_trajectory.png",dpi=180); plt.close()

    selection={"status":"PASS" if learned_go else "FAIL","test_predictions_constructed":False,"selected_feature_group":group,"selected_lookback_hours":lookback_h,"primary_horizon_hours":primary_h,"top2_modern_models":top2,"multi_seed_models":stable_models,"strongest_simple_baseline":strongest_baseline,"final_candidate_model":final_model,"screening_mean_skill_best":float(screening_mean.loc[final_model,"skill_best"]),"screening_positive_skill_fold_count":int(screening_mean.loc[final_model,"positive_skill_fold_count"]),"multi_seed_mean_skill_best":float(final_multi.skill_best),"learned_model_go":learned_go,"modern_bootstrap":modern_boot}
    write_json(selection,out/"frozen_config"/"model_selection.json")
    print(json.dumps(selection,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

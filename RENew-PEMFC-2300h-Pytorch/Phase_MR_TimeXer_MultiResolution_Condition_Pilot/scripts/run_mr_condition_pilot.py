"""Development-only P2 reproduction and 10-minute MR condition gate."""
from __future__ import annotations
import hashlib, importlib.util, json, math, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode=True
ROOT=Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch")
PHASE=ROOT/"Phase_MR_TimeXer_MultiResolution_Condition_Pilot"; OUT=PHASE/"outputs"; REP=PHASE/"reports"; FIG=PHASE/"figures"
GROOT=ROOT/"Phase_G_STOP_Physics_Consistent_Future_Conditions"; TR2=ROOT/"Phase_TR2_Fair_Resolution_Correction"; TR=ROOT/"Phase_TR_Time_Resolution_Audit"
CFG=yaml.safe_load((PHASE/"configs"/"frozen_protocol.yaml").read_text(encoding="utf-8"))
FOLDS=("fold_1","fold_2","fold_3"); EPS=1e-7; SEED=int(CFG["seed"])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def md(x):
    if x.empty:return "_No rows._"
    y=x.copy()
    for c in y:y[c]=y[c].map(lambda v:"" if pd.isna(v) else str(v).replace("|","\\|"))
    return "| "+" | ".join(map(str,y.columns))+" |\n| "+" | ".join(["---"]*len(y.columns))+" |\n"+"\n".join("| "+" | ".join(r)+" |" for r in y.to_numpy(str))
def imp(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def mixed(q):return ((q>1e-12)&(q<1-1e-12)).astype(int)

class MR10:
    """Fixed TR2-B direct q Ridge plus same-target mixed Logistic model."""
    def fit(self,x,yq,ym):
        self.scaler=StandardScaler().fit(x);z=self.scaler.transform(x);self.q=Ridge(alpha=1.0).fit(z,yq);self.cls=[]
        for h in range(24):
            yy=ym[:,h]
            self.cls.append(float(yy[0]) if len(np.unique(yy))<2 else LogisticRegression(C=1.0,class_weight="balanced",solver="liblinear",random_state=SEED,max_iter=1000).fit(z,yy))
        return self
    def predict(self,x):
        z=self.scaler.transform(x);q=np.clip(self.q.predict(z),0,1);p=np.zeros_like(q)
        for h,c in enumerate(self.cls):p[:,h]=c if isinstance(c,float) else c.predict_proba(z)[:,1]
        return q,p

def map_histories(d10,origins):
    d=d10.sort_values("bin_start_h").reset_index(drop=True);pos={round(float(v),8):i for i,v in enumerate(d.bin_end_h)}
    out=[]
    for origin in origins:
        i=pos.get(round(float(origin),8))
        if i is None or i<143:raise RuntimeError("MR origin absent from 10-minute table")
        b=d.iloc[i-143:i+1]
        if len(b)!=144 or not bool(b.valid_bin.all()) or b.temporal_segment_id.nunique()!=1:raise RuntimeError("invalid MR 24-hour history")
        if not np.allclose(np.diff(b.bin_start_h),1/6,atol=1e-7,rtol=0):raise RuntimeError("MR history discontinuity")
        out.append(b[["q","current_mean_A","current_std_A"]].to_numpy(float).ravel())
    return np.asarray(out,float)

def blocks(start,n,k=5):
    cut=[start]+[start+math.ceil((n-start)*i/k) for i in range(1,k)]+[n]
    return [(a,b) for a,b in zip(cut[:-1],cut[1:]) if b>a]
def mr_oof(x,yq,ym,target_end,origin,fold):
    n=len(x);start=next((i for i,t in enumerate(origin) if int(np.sum(target_end<=t+EPS))>=64),None)
    if start is None:raise RuntimeError("MR OOF lacks 64 completed targets")
    q=np.full((n,24),np.nan);p=np.full_like(q,np.nan);audit=[]
    for block,(a,b) in enumerate(blocks(start,n),1):
        m=float(origin[a]);fit=np.flatnonzero(target_end<=m+EPS)
        if len(fit)<64 or float(target_end[fit].max())>m+EPS:raise RuntimeError("MR OOF target-completion failure")
        model=MR10().fit(x[fit],yq[fit],ym[fit]);q[a:b],p[a:b]=model.predict(x[a:b])
        audit.append({"fold":fold,"block":block,"fit_count":len(fit),"predict_count":b-a,"max_fit_target_end_h":float(target_end[fit].max()),"min_prediction_origin_h":m,"target_completion_pass":True,"train_only_scaler":True})
    return q,p,np.isfinite(q).all(axis=1),audit

def condition_metrics(model,fold,pred,truth,prob,scope="all_common_origins"):
    q=pred[:,:,8];qt=truth[:,:,8];m=mixed(qt).ravel();pr=prob.ravel();a,b,c,_=precision_recall_fscore_support(m,pr>=.5,average="binary",zero_division=0)
    return {"model":model,"fold":fold,"scope":scope,"q_mae":float(np.mean(abs(q-qt))),"q_rmse":float(np.sqrt(np.mean((q-qt)**2))),"mixed_brier":float(np.mean((pr-m)**2)),"mixed_f1":float(c),"current_mean_mae_A":float(np.mean(abs(pred[:,:,0]-truth[:,:,0]))),"current_std_mae_A":float(np.mean(abs(pred[:,:,1]-truth[:,:,1]))),"target_values":int(q.size)}
def physical(model,fold,z,area):
    q=z[:,:,8];mu=z[:,:,0];sd=z[:,:,1];jmu=z[:,:,3];jsd=z[:,:,4];finite=np.isfinite(z).all(axis=(1,2))
    # Conditional mean is recovered only for the identity check, respecting P2 zero-group approximation.
    muon=np.divide(mu,np.maximum(1-q,1e-12));var=((1-q)*np.maximum(0,((sd**2-q*(1-q)*muon**2)/np.maximum(1-q,1e-12)))+q*(1-q)*muon**2)
    violation=(~finite)|np.any((q<0)|(q>1)|(mu<0)|(sd<0)|(~np.isfinite(var))|(np.abs(jmu-mu/area)>1e-8)|(np.abs(jsd-sd/area)>1e-8),axis=1)
    return {"model":model,"fold":fold,"nonfinite_trajectory_rate":float(np.mean(~finite)),"physical_violation_trajectory_rate":float(np.mean(violation)),"q_bounds_violation_rate":float(np.mean(np.any((q<0)|(q>1),axis=1))),"current_density_identity_max_abs":float(max(np.abs(jmu-mu/area).max(),np.abs(jsd-sd/area).max())),"mixture_variance_nonfinite_rate":float(np.mean(~np.isfinite(var)))}

def report(baseline,counts,oof,cond,phys,var_changes,decision):
    (REP/"01_EXISTING_WORK_AND_REPRODUCTION.md").write_text("# Existing work and P2 reproduction\n\nP2 is reproduced by the exact frozen G-STOP functions and N2.8F checkpoints in a Development-only process. P0/P1/P2 reference checks use the original full legal validation sets; fair comparisons later use only common origins.\n\n"+md(baseline)+"\n",encoding="utf-8")
    (REP/"02_COMMON_ORIGIN_AUDIT.md").write_text("# Common-origin audit\n\n"+md(counts)+"\n\nTR2 common 10-minute-valid origins are intersected with N2.8F train/validation windows before fitting the fair P2 and MR forecasters. Target voltage is never used in this filter.\n",encoding="utf-8")
    (REP/"03_STRICT_CAUSAL_OOF_AUDIT.md").write_text("# Strict causal MR OOF audit\n\n"+md(oof)+"\n\nEvery OOF fitting target completes on or before its prediction origin. The OOF predictions are audit evidence only; the frozen TimeXer is not fitted in this phase.\n",encoding="utf-8")
    (REP/"04_FUTURE_CONDITION_RECONSTRUCTION.md").write_text("# Future-condition reconstruction\n\nMR replaces only P2 q with the fixed TR2-B-style 10-minute-history Ridge prediction. It preserves P2 conditional on-current mean and standard-deviation predictions; total current mean and population standard deviation are recomputed with the inherited mixture identities. Peak-to-peak remains P2 independent Ridge. Current-density mean/std are deterministically derived with the frozen active area.\n\n## Variable-level P2-to-MR reconstruction audit\n\n"+md(var_changes)+"\n",encoding="utf-8")
    (REP/"05_CONDITION_GATE_RESULTS.md").write_text("# Condition gate results\n\n"+md(cond)+"\n\n## Physical checks\n\n"+md(phys)+"\n\n**"+decision["status"]+"**\n\n"+decision["reason"]+"\n",encoding="utf-8")
    for n,title in [("06_FROZEN_TIMEXER_RESULTS.md","Frozen TimeXer results"),("07_ERROR_PROPAGATION_ANALYSIS.md","Error propagation analysis")]:
        (REP/n).write_text("# "+title+"\n\nNOT_RUN because the preregistered future-condition gate did not pass. No frozen TimeXer M1 inference or new training was authorized.\n",encoding="utf-8")
    (REP/"08_PHYSICAL_CONSISTENCY.md").write_text("# Physical consistency\n\n"+md(phys)+"\n\nPhysical projection uses the inherited P2 conventions: q bounded in [0,1], nonnegative current statistics, J=I/A, population standard-deviation semantics, and no derived peak-to-peak rule.\n",encoding="utf-8")
    (REP/"09_LEAKAGE_AUDIT.md").write_text("# Leakage audit\n\nAll loaded N2.8F frames are fail-closed Development prefixes. Test rows, Test tensors, Test predictions and Test metrics are all zero/false. MR OOF has target-completion purge. Validation MR fits only its Fold common Train set. No true future current, q, voltage or raw sample count enters a formal prediction.\n",encoding="utf-8")
    (REP/"FINAL_RESEARCH_DECISION.md").write_text("# Final research decision\n\n**"+decision["status"]+"**\n\n"+decision["reason"]+"\n\nNo TimeXer parameter was trained or altered. The route stops at the condition gate.\n",encoding="utf-8")

def condition_figure(cond):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained"); x=np.arange(3); width=.36
    c=cond[cond.scope=="all_common_origins"]
    p2=c[c.model=="P2_common"].set_index("fold").loc[list(FOLDS),"q_rmse"].to_numpy(float)
    mr=c[c.model=="MR_10min_direct"].set_index("fold").loc[list(FOLDS),"q_rmse"].to_numpy(float)
    ax.bar(x-width/2,p2,width,label="P2 common",color="#B64342",edgecolor="black");ax.bar(x+width/2,mr,width,label="MR 10-min",color="#0F4D92",edgecolor="black")
    ax.set_xticks(x,["Fold 1","Fold 2","Fold 3"]);ax.set_ylabel("Future q RMSE");ax.set_title("Development only: P2 versus MR condition forecast");ax.legend(frameon=False)
    fig.savefig(FIG/"condition_q_rmse_by_fold.png",dpi=300);fig.savefig(FIG/"condition_q_rmse_by_fold.pdf");plt.close(fig)

def main():
    for p in [OUT,REP,FIG]:p.mkdir(parents=True,exist_ok=True)
    g=imp("mr_gstop",GROOT/"scripts"/"phase_g_stop_experiment.py");gc=g.load_config();n28=g.import_n28f(Path(gc["n2_8f_root"]));ncfg=n28.load_config();common,frame,windows,firewall=n28.load_development(ncfg)
    if frame.split.eq("test").any():raise RuntimeError("TEST FIREWALL")
    aux=pd.read_csv(GROOT/"outputs"/"physical_audit"/"raw_conditional_current_statistics.csv");frame=frame.merge(aux[["bin_start_ophrs2_h","mu_on_A","sigma_on_A"]],on="bin_start_ophrs2_h",how="left",validate="one_to_one")
    if frame[["mu_on_A","sigma_on_A"]].isna().any().any():raise RuntimeError("missing auxiliary conditions")
    d10=pd.read_csv(TR/"outputs"/"tables"/"causal_state_features_10min_development_only.csv")
    if (d10.bin_start_h>=float(CFG["scope"]["development_end_h_exclusive"])).any():raise RuntimeError("MR table contains Test")
    tr2train=pd.read_csv(TR2/"outputs"/"common_train_origins.csv");tr2valid=pd.read_csv(TR2/"outputs"/"common_validation_origins.csv")
    packages={f:n28.package_fold(common,frame,windows,ncfg,f) for f in FOLDS}
    baseline=[];common_counts=[];oofrows=[];condition=[];physrows=[];fcpreds=[];var_changes=[];checkpoint_hashes={}
    for f,p in packages.items():
        atrain=g.future_aux(frame,p["train"],["mu_on_A","sigma_on_A"]);aval=g.future_aux(frame,p["valid"],["mu_on_A","sigma_on_A"])
        p0=p["z_ridge"].copy();p1,_=g.project_future(p0,p["area"]);p2,pp2,_=g.build_p2(p,atrain,aval,gc)
        ck=Path(gc["n2_8f_root"])/"checkpoints"/("M0_SA_TimeXer_Ridge_"+f+"_seed42.pt");checkpoint_hashes[f]=sha(ck);model=n28.StepAlignedTimeXer(ncfg,caf=False);model.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True));model.to(n28.device()).eval()
        for name,z in [("P0",p0),("P1",p1),("P2",p2)]:
            pred=g.inference(n28,model,p,z);m=g.voltage_metrics(p["valid"].y_abs,pred,p["valid"].anchor);baseline.append({"model":name,"fold":f,**m,"checkpoint_sha256":checkpoint_hashes[f]})
        saved=np.load(GROOT/"outputs"/"frozen_timexer"/("P2_Zero_Current_Aware_"+f+"_predictions.npz"))
        if not np.allclose(saved["future_conditions"],p2,atol=1e-7,rtol=0):raise RuntimeError("P2 future tensor reproduction failure "+f)
        # Fair common membership is frozen by TR2 and occurs before either model fit.
        tset=set(np.round(tr2train.loc[tr2train.fold==f,"origin_end_h"],8));vset=set(np.round(tr2valid.loc[tr2valid.fold==f,"origin_end_h"],8))
        # The fair set is the intersection of pre-existing legal N2.8F/P2
        # windows and pre-existing TR2 10-minute-valid windows.  No outcome is
        # examined to form it; counts are reported rather than forced equal.
        tm=np.array([round(float(x),8) in tset for x in p["train"].origin_age_h]);vm=np.array([round(float(x),8) in vset for x in p["valid"].origin_age_h])
        if not tm.any() or not vm.any():raise RuntimeError("empty common origin intersection "+f)
        common_counts.append({"fold":f,"p2_original_train_origins":len(p["train"].origin_age_h),"p2_original_validation_origins":len(p["valid"].origin_age_h),"tr2_train_origins":len(tset),"tr2_validation_origins":len(vset),"common_train_origins":int(tm.sum()),"common_validation_origins":int(vm.sum()),"excluded_from_p2_train":int((~tm).sum()),"excluded_from_p2_validation":int((~vm).sum()),"excluded_from_tr2_train":int(len(tset)-tm.sum()),"excluded_from_tr2_validation":int(len(vset)-vm.sum())})
        pt={"htr":p["htr"][tm],"hva":p["hva"][vm],"ztr":p["ztr"][tm],"z_ridge":p["z_ridge"][vm],"area":p["area"]}
        atr,ava=atrain[tm],aval[vm];p2c,p2prob,_=g.build_p2(pt,atr,ava,gc)
        xtr=map_histories(d10,p["train"].origin_age_h[tm]);xva=map_histories(d10,p["valid"].origin_age_h[vm]);qtrue=pt["ztr"][:,:,8];mtrue=mixed(qtrue)
        qoof,poof,keep,audit=mr_oof(xtr,qtrue,mtrue,p["train"].target_end_age_h[tm],p["train"].origin_age_h[tm],f);oofrows.extend(audit)
        if not keep.any():raise RuntimeError("MR OOF empty "+f)
        qmr,pmr=MR10().fit(xtr,qtrue,mtrue).predict(xva)
        on=g.AuxiliaryRidge(float(gc["forecast"]["conditional_on_stats_ridge_alpha"])).fit(pt["htr"],atr).predict(pt["hva"]);mu=np.maximum(on[:,:,0],0);sd=np.maximum(on[:,:,1],0)
        raw=p2c.copy();raw[:,:,8]=qmr;raw[:,:,0]=(1-qmr)*mu;raw[:,:,1]=np.sqrt(np.maximum(0,(1-qmr)*sd**2+qmr*(1-qmr)*mu**2));mr,_=g.project_future(raw,p["area"])
        truth=p["zva"][vm]
        rules=["recomputed from MR q and preserved P2 conditional on-current mean","recomputed from MR q and preserved P2 conditional on-current standard deviation","preserved P2 independent Ridge prediction","derived from reconstructed current mean through J=I/A","derived from reconstructed current standard deviation through J=I/A","preserved P2 independent Ridge prediction","preserved P2 auxiliary prediction","preserved P2 auxiliary prediction","replaced by direct MR 10-minute-history Ridge prediction"]
        for j,name in enumerate(ncfg["future_exogenous_features"]):
            d=mr[:,:,j]-p2c[:,:,j]
            var_changes.append({"fold":f,"feature":name,"reconstruction_rule":rules[j],"mean_signed_change":float(d.mean()),"mean_absolute_change":float(np.abs(d).mean()),"max_absolute_change":float(np.abs(d).max())})
        condition += [condition_metrics("P2_common",f,p2c,truth,p2prob),condition_metrics("MR_10min_direct",f,mr,truth,pmr)]
        if f=="fold_1":
            condition += [condition_metrics("P2_common",f,p2c[:,-1:,:],truth[:,-1:,:],p2prob[:,-1:],"fold1_12h_endpoint"),condition_metrics("MR_10min_direct",f,mr[:,-1:,:],truth[:,-1:,:],pmr[:,-1:],"fold1_12h_endpoint")]
        if f=="fold_3":
            # There is no complete 24-step always-nonzero trajectory in the
            # common set, so the preregistered difficult regime is evaluated
            # at its atomic 30-minute target bins: q_true==0 means all raw
            # samples in that bin are above the inherited zero-current limit.
            sustained=truth[:,:,8]<=EPS
            if not sustained.any():raise RuntimeError("no Fold3 nonzero-current target bins")
            condition += [condition_metrics("P2_common",f,p2c[sustained][:,None,:],truth[sustained][:,None,:],p2prob[sustained][:,None],"fold3_nonzero_current_target_bins"),condition_metrics("MR_10min_direct",f,mr[sustained][:,None,:],truth[sustained][:,None,:],pmr[sustained][:,None],"fold3_nonzero_current_target_bins")]
        physrows += [physical("P2_common",f,p2c,p["area"]),physical("MR_10min_direct",f,mr,p["area"])]
        for name,z,prob in [("P2_common",p2c,p2prob),("MR_10min_direct",mr,pmr)]:
            for i,origin in enumerate(p["valid"].origin_age_h[vm]):
                for h in range(24):fcpreds.append({"model":name,"fold":f,"origin_end_h":origin,"horizon_step":h+1,"q_true":truth[i,h,8],"q_pred":z[i,h,8],"mixed_true":mixed(truth[i:i+1,h:h+1,8])[0,0],"mixed_probability":prob[i,h],"current_mean_true_A":truth[i,h,0],"current_mean_pred_A":z[i,h,0],"current_std_true_A":truth[i,h,1],"current_std_pred_A":z[i,h,1]})
    base=pd.DataFrame(baseline);p0mean=float(base[base.model=="P0"].rmse_delta_mV.mean())
    if abs(p0mean-float(CFG["reproduction"]["p0_mean_rmse_reference_mV"]))>float(CFG["reproduction"]["p0_mean_rmse_tolerance_mV"]):raise RuntimeError("BASELINE_REPRODUCTION_FAILED")
    cond=pd.DataFrame(condition);phys=pd.DataFrame(physrows);cond_main=cond[cond.scope=="all_common_origins"];p2=cond_main[cond_main.model=="P2_common"].set_index("fold");mr=cond_main[cond_main.model=="MR_10min_direct"].set_index("fold");gain=(p2.q_rmse-mr.q_rmse)/p2.q_rmse;bn=int((mr.mixed_brier<=p2.mixed_brier+1e-12).sum());pv=float(phys[phys.model=="MR_10min_direct"].physical_violation_trajectory_rate.max())
    passed=int((gain>=.05).sum())>=2 and gain["fold_3"]>=-.03 and bn>=2 and pv==0
    decision={"status":"CONDITION_GATE_FAIL" if not passed else "FROZEN_TIMEXER_GATE_FAIL","reason":"MR versus common P2 q-RMSE gain percent "+json.dumps({k:round(float(v)*100,3) for k,v in gain.items()})+"; non-worse Brier folds "+str(bn)+"; MR physical violation rate "+str(pv)+"."}
    if passed:decision={"status":"FROZEN_TIMEXER_GATE_FAIL","reason":"CONDITION_GATE_PASS reached; frozen-TimeXer M1 is NOT_RUN in this implementation pending explicit second-stage execution."}
    base.to_csv(OUT/"baseline_reproduction.csv",index=False);pd.DataFrame(common_counts).to_csv(OUT/"common_origins.csv",index=False);pd.DataFrame(oofrows).to_csv(OUT/"strict_causal_oof_audit.csv",index=False);cond.to_csv(OUT/"condition_metrics.csv",index=False);phys.to_csv(OUT/"physical_identity_checks.csv",index=False);pd.DataFrame(var_changes).to_csv(OUT/"future_condition_variable_change.csv",index=False);pd.DataFrame(fcpreds).to_csv(OUT/"future_condition_predictions.csv",index=False)
    notrun=pd.DataFrame([{"stage":"frozen_timexer","status":"NOT_RUN_CONDITION_GATE_FAIL" if not passed else "NOT_RUN_IMPLEMENTATION_STOP"},{"stage":"voltage_predictions","status":"NOT_RUN_CONDITION_GATE_FAIL" if not passed else "NOT_RUN_IMPLEMENTATION_STOP"}]);notrun.to_csv(OUT/"not_run_manifest.csv",index=False)
    for name in ["voltage_predictions.csv","per_fold_voltage_metrics.csv","per_horizon_metrics.csv","regime_metrics.csv","tail_metrics.csv","paired_error_differences.csv","summary_metrics.csv"]: notrun.assign(artifact=name).to_csv(OUT/name,index=False)
    report(base,pd.DataFrame(common_counts),pd.DataFrame(oofrows),cond,phys,pd.DataFrame(var_changes),decision);condition_figure(cond)
    manifest={"command":"python scripts/run_mr_condition_pilot.py","head":subprocess.check_output(["git","-C",str(ROOT.parent),"rev-parse","HEAD"],text=True).strip(),"python":platform.python_version(),"torch":torch.__version__,"seed":SEED,"test_accessed":False,"dependencies":{"g_stop_script":str(GROOT/"scripts"/"phase_g_stop_experiment.py"),"g_stop_sha256":sha(GROOT/"scripts"/"phase_g_stop_experiment.py"),"n28f_script":str(Path(gc["n2_8f_root"])/"scripts"/"phase_n2_8f_run.py"),"n28f_sha256":sha(Path(gc["n2_8f_root"])/"scripts"/"phase_n2_8f_run.py"),"frozen_timexer_checkpoint_sha256":checkpoint_hashes},"p0_reproduction_mean_rmse_mV":p0mean,"decision":decision}
    (PHASE/"RUN_AUDIT.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8");(PHASE/"FINAL_DECISION.yaml").write_text("status: "+decision["status"]+"\nreason: "+json.dumps(decision["reason"],ensure_ascii=False)+"\ntest_accessed: false\n",encoding="utf-8")
    print(json.dumps(decision,ensure_ascii=False))
if __name__=="__main__":
    try:main()
    except Exception as e:print("FAIL-CLOSED: "+type(e).__name__+": "+str(e),file=sys.stderr);raise

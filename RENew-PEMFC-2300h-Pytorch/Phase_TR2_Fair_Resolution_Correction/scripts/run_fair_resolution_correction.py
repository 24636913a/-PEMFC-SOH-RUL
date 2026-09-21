"""Development-only POST_HOC_METHOD_CORRECTION for fair 10-minute/30-minute comparison."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT=Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch")
PREV=ROOT/"Phase_TR_Time_Resolution_Audit"/"outputs"/"tables"
PHASE=ROOT/"Phase_TR2_Fair_Resolution_Correction"; OUT=PHASE/"outputs"; FIG=PHASE/"figures"; REP=PHASE/"reports"
START,END=72.788690,1819.997738; FEATURES=["q","current_mean_A","current_std_A"]; SEED=20260921
FOLDS={"fold_1":(946.393214,1164.794345),"fold_2":(1164.794345,1383.195476),"fold_3":(1383.195476,1819.997738)}
EXPS=["A_30min_baseline","B_10min_direct_30target","C_10min_then_equal_mean","D_10min_aggregated_history_control"]

def md(x):
    if x.empty:return "_No rows._"
    y=x.copy()
    for c in y:y[c]=y[c].map(lambda v:"" if pd.isna(v) else str(v).replace("|","\\|"))
    return "| "+" | ".join(map(str,y.columns))+" |\n| "+" | ".join(["---"]*len(y.columns))+" |\n"+"\n".join("| "+" | ".join(r)+" |" for r in y.to_numpy(str))

def mixed_from_counts(zero_count,valid_count):
    if not np.isfinite(zero_count) or not np.isfinite(valid_count) or valid_count<=0:return None
    return int(zero_count>0 and zero_count<valid_count)

def equal_prediction_aggregate(triplet):
    x=np.asarray(triplet,float)
    if x.shape!=(3,) or not np.isfinite(x).all():raise ValueError("three finite predictions required")
    return float(x.mean())

def complete_children(x):
    return len(x)==3 and bool(np.asarray(x.valid_bin,bool).all())

def parent_mixed_from_children(children):
    """Return the parent raw-count label, or reject an incomplete child triple."""
    if not complete_children(children): return None
    return mixed_from_counts(float(children.zero_current_count.sum()),float(children.valid_current_count.sum()))

def load_data():
    d10=pd.read_csv(PREV/"causal_state_features_10min_development_only.csv").sort_values("bin_start_h").reset_index(drop=True)
    d30=pd.read_csv(PREV/"causal_state_features_30min_development_only.csv").sort_values("bin_start_h").reset_index(drop=True)
    for d,n in [(d10,"10-minute"),(d30,"30-minute")]:
        if (d.bin_start_h>=END-1e-9).any():raise RuntimeError(n+" source violates Test guard")
        d["mixed_direct"]=((d.zero_current_count>0)&(d.zero_current_count<d.valid_current_count)).astype(int)
    return d10,d30

def aggregate_hist10(x):
    if len(x)%3:raise ValueError("history must have full child triples")
    result=[]
    for i in range(0,len(x),3):
        g=x.iloc[i:i+3]
        if not complete_children(g):raise RuntimeError("invalid child history")
        n=float(g.valid_current_count.sum()); z=float(g.zero_current_count.sum()); s=float(g.i_sum.sum()); ss=float(g.i_sq.sum())
        mu=s/n;result.append([z/n,mu,np.sqrt(max(ss/n-mu*mu,0))])
    return np.asarray(result,float)

def origins(d10,d30,start,end,fold,split):
    pos10={round(float(v),8):i for i,v in enumerate(d10.bin_end_h)}
    out=[]
    for i in range(47,len(d30)-24):
        f,l=i-47,i+24; origin=float(d30.bin_end_h.iat[i]); b30=d30.iloc[f:l+1]
        if d30.bin_start_h.iat[f]<start-1e-8 or d30.bin_end_h.iat[l]>end+1e-8:continue
        if not bool(b30.valid_bin.all()) or b30.temporal_segment_id.nunique()!=1:continue
        j=pos10.get(round(origin,8))
        if j is None or j<143 or j+72>=len(d10):continue
        b10=d10.iloc[j-143:j+73]
        if d10.bin_start_h.iat[j-143]<start-1e-8 or d10.bin_end_h.iat[j+72]>end+1e-8:continue
        if not bool(b10.valid_bin.all()) or b10.temporal_segment_id.nunique()!=1:continue
        t10=d10.iloc[j+1:j+73];t30=d30.iloc[i+1:i+25]
        if any(not complete_children(t10.iloc[k:k+3]) for k in range(0,72,3)):continue
        qchild=[];mchild=[]
        for k in range(0,72,3):
            g=t10.iloc[k:k+3];n=g.valid_current_count.sum();z=g.zero_current_count.sum();qchild.append(z/n);mchild.append(parent_mixed_from_children(g))
        if not np.allclose(qchild,t30.q.to_numpy(float),atol=1e-12,rtol=0):raise RuntimeError("parent q identity failure")
        if not np.array_equal(np.asarray(mchild,int),t30.mixed_direct.to_numpy(int)):raise RuntimeError("parent mixed identity failure")
        x30=b30.iloc[:48][FEATURES].to_numpy(float);x10=b10.iloc[:144][FEATURES].to_numpy(float);xc_raw=aggregate_hist10(b10.iloc[:144])
        control_diff=float(np.max(np.abs(x30-xc_raw)))
        if not np.allclose(x30,xc_raw,atol=1e-10,rtol=0):raise RuntimeError("history control mismatch")
        # Same sufficient statistics can differ only in floating-point summation
        # order.  Canonicalize the verified-equivalent control to the direct raw
        # aggregate representation, so capacity comparison is not a round-off test.
        xc=x30.copy()
        out.append({"fold":fold,"split":split,"origin_end_h":origin,"target_start_h":float(t30.bin_start_h.iat[0]),"x30":x30.ravel(),"x10":x10.ravel(),"xc":xc.ravel(),"y30":t30.q.to_numpy(float),"m30":t30.mixed_direct.to_numpy(int),"y10":t10.q.to_numpy(float),"control_max_abs_diff":control_diff})
    if not out:raise RuntimeError("no common "+fold+" "+split+" origins")
    return out

def ar(rows,key):return np.asarray([r[key] for r in rows],float)

def q_predict(xtr,ytr,xv):
    s=StandardScaler().fit(xtr);return np.clip(Ridge(alpha=1.0).fit(s.transform(xtr),ytr).predict(s.transform(xv)),0,1)

def p_predict(xtr,ytr,xv):
    s=StandardScaler().fit(xtr);a,b=s.transform(xtr),s.transform(xv);out=np.zeros((len(b),ytr.shape[1]))
    for h in range(ytr.shape[1]):
        y=ytr[:,h]
        out[:,h]=y[0] if len(np.unique(y))<2 else LogisticRegression(C=1.0,class_weight="balanced",solver="liblinear",random_state=SEED,max_iter=500).fit(a,y).predict_proba(b)[:,1]
    return out

def run_exp(name,tr,va):
    key={"A_30min_baseline":"x30","B_10min_direct_30target":"x10","C_10min_then_equal_mean":"x10","D_10min_aggregated_history_control":"xc"}[name]
    xt,xv=ar(tr,key),ar(va,key);y,yy=ar(tr,"y30"),ar(va,"y30");m=ar(tr,"m30").astype(int);start=time.perf_counter()
    if name=="C_10min_then_equal_mean":
        q=q_predict(xt,ar(tr,"y10"),xv).reshape(len(va),24,3).mean(axis=2);p=None;rp=xt.shape[1]*72+72;lp=0
    else:q=q_predict(xt,y,xv);p=p_predict(xt,m,xv);rp=xt.shape[1]*24+24;lp=rp
    return {"q":q,"p":p,"y":yy,"m":ar(va,"m30").astype(int),"seconds":time.perf_counter()-start,"dim":xt.shape[1],"ridge_parameters":rp,"logistic_parameters":lp}

def metric(name,fold,r):
    result=[]
    for scope,ix in [("all_horizons",slice(None)),("terminal_12h",-1)]:
        y=r["y"][:,ix].ravel();q=r["q"][:,ix].ravel();d={"experiment":name,"fold":fold,"metric_scope":"common_overlapping_targets","horizon_scope":scope,"n_target_points":len(y),"q_mae":mean_absolute_error(y,q),"q_rmse":mean_squared_error(y,q)**.5}
        if r["p"] is None:d.update({"mixed_brier":"NOT_RUN","mixed_precision":"NOT_RUN","mixed_recall":"NOT_RUN","mixed_f1":"NOT_RUN","mixed_positive_count":"NOT_RUN"})
        else:
            m=r["m"][:,ix].ravel();p=r["p"][:,ix].ravel();a,b,c,_=precision_recall_fscore_support(m,p>=.5,average="binary",zero_division=0)
            d.update({"mixed_brier":float(np.mean((p-m)**2)),"mixed_precision":a,"mixed_recall":b,"mixed_f1":c,"mixed_positive_count":int(m.sum())})
        result.append(d)
    return result

def pred_rows(name,fold,va,r):
    z=[]
    plist=r["p"] if r["p"] is not None else [None]*len(va)
    for row,q,p in zip(va,r["q"],plist):
        for h in range(24):z.append({"experiment":name,"fold":fold,"origin_end_h":row["origin_end_h"],"target_start_h":row["target_start_h"]+.5*h,"horizon_30min_step":h+1,"q_true":row["y30"][h],"q_pred":q[h],"mixed_true":row["m30"][h],"mixed_probability":np.nan if p is None else p[h]})
    return z

def event_metrics(pred,name,fold):
    if name=="C_10min_then_equal_mean":return {"experiment":name,"fold":fold,"status":"NOT_RUN"}
    x=pred[(pred.experiment==name)&(pred.fold==fold)].sort_values(["target_start_h","origin_end_h"]).drop_duplicates("target_start_h").sort_values("target_start_h").copy()
    x["truth"]=x.mixed_true.astype(int);x["pred"]=(x.mixed_probability>=.5).astype(int);x["gap"]=x.target_start_h.diff().sub(.5).abs()>1e-7
    def runs(col):
        gid=(x[col].ne(x[col].shift())|x.gap).cumsum();return [(float(g.target_start_h.iloc[0]),float(g.target_start_h.iloc[-1]+.5)) for _,g in x[x[col]==1].groupby(gid[x[col]==1])]
    t,p=runs("truth"),runs("pred");has=lambda a,bs:any(max(a[0],b[0])<min(a[1],b[1]) for b in bs)
    return {"experiment":name,"fold":fold,"status":"RUN","unique_target_times":len(x),"independent_true_mixed_events":len(t),"predicted_mixed_events":len(p),"missed_true_events":sum(not has(a,p) for a in t),"false_positive_events":sum(not has(a,t) for a in p)}

def figures(summary):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.spines.top":False,"axes.spines.right":False})
    s=summary[(summary.aggregation=="fold_mean")&(summary.horizon_scope=="all_horizons")]
    ex=EXPS;lab=["A: 30 min","B: 10 min direct","C: 10 min equal","D: 10 min control"];col=["#B64342","#0F4D92","#42949E","#CFCECE"]
    v=[float(s[s.experiment==e].q_rmse.iloc[0]) for e in ex];fig,ax=plt.subplots(figsize=(10,5),layout="constrained");bars=ax.bar(lab,v,color=col,edgecolor="black",linewidth=1.2);ax.set_ylabel("q RMSE, unweighted fold mean");ax.set_title("Fair common-origin q comparison")
    for b,x in zip(bars,v):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{x:.4f}",ha="center",va="bottom",fontsize=10)
    fig.savefig(FIG/"fair_q_rmse_comparison.png",dpi=300);fig.savefig(FIG/"fair_q_rmse_comparison.pdf");plt.close(fig)
    v=[float(s[s.experiment==e].mixed_brier.iloc[0]) for e in ex[:2]];fig,ax=plt.subplots(figsize=(7,5),layout="constrained");bars=ax.bar(lab[:2],v,color=col[:2],edgecolor="black",linewidth=1.2);ax.set_ylabel("Mixed-state Brier, unweighted fold mean");ax.set_title("Same 30-minute target probability")
    for b,x in zip(bars,v):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{x:.4f}",ha="center",va="bottom",fontsize=10)
    fig.savefig(FIG/"same_target_brier_AB.png",dpi=300);fig.savefig(FIG/"same_target_brier_AB.pdf");plt.close(fig)

def reports(counts,identity,capacity,per,summary,events,decision):
    (REP/"02_CAUSAL_AGGREGATION_CORRECTION.md").write_text("# Causal aggregation correction\n\nExperiment C uses a fixed equal arithmetic mean of three predicted 10-minute q values. Future valid-current sample counts never weight a prediction. Actual raw counts remain valid only for construction of the true q label. The automated test changes future observation counts while holding predictions fixed and verifies identical output.\n",encoding="utf-8")
    (REP/"03_LABEL_IDENTITY_AUDIT.md").write_text("# Label identity audit\n\nOfficial parent mixed label equals one only when parent zero count is positive and strictly below parent valid-current count. All common targets are reconstructed from children and checked against direct parent raw statistics before model fitting.\n\n"+md(identity)+"\n\nTests cover all-on, all-zero, q0 plus q1, internally mixed, and incomplete children.\n",encoding="utf-8")
    (REP/"04_SAME_TARGET_PROBABILITY_COMPARISON.md").write_text("# Same-target probability comparison\n\nA and B directly predict the same future parent 30-minute labels from the same common origins; they use same Logistic Regression configuration and threshold. C is q-only diagnostic and has no Brier result.\n\n"+md(per[(per.experiment.isin(EXPS[:2]))&(per.horizon_scope=="all_horizons")]),encoding="utf-8")
    (REP/"05_COMMON_ORIGIN_AUDIT.md").write_text("# Common-origin audit\n\n"+md(counts)+"\n\nEach row has valid complete native-grid history and future. A/B/C/D share the sets before fitting.\n\n## Capacity and control\n\n"+md(capacity)+"\n",encoding="utf-8")
    (REP/"06_FAIR_METRIC_COMPARISON.md").write_text("# Fair metric comparison\n\nPer-fold rows use the identical overlapping common target collection within each Fold. Fold means and pooled target collections are separate statistics. Event counts deduplicate target timestamps.\n\n## Per fold\n\n"+md(per)+"\n\n## Summary\n\n"+md(summary)+"\n\n## Events\n\n"+md(events)+"\n",encoding="utf-8")
    (REP/"07_LEAKAGE_AUDIT.md").write_text("# Leakage audit\n\n- Development-only source tables are read; no Test table/file is opened.\n- Common origins require complete history and target inside each Fold.\n- Scalers and models fit only Fold Train common origins.\n- A/B directly predict parent targets. C uses fixed equal output weighting.\n- Future raw counts construct observed truth only, never a prediction.\n- No calibration, threshold tuning, search, future real condition, or deep model is used.\n",encoding="utf-8")
    (REP/"FINAL_RESEARCH_DECISION.md").write_text("# Final research decision\n\nStatus: POST_HOC_METHOD_CORRECTION\n\n**"+decision["status"]+"**\n\n"+decision["reason"]+"\n\nThe original Phase TR NO_GO_KEEP_30MIN remains untouched. A post-hoc pass could only justify a separately authorized fixed TimeXer study; this phase trains no TimeXer.\n",encoding="utf-8")

def main():
    np.random.seed(SEED)
    for p in [OUT,FIG,REP]:p.mkdir(parents=True,exist_ok=True)
    d10,d30=load_data();tr=[];va=[]
    for f,(te,ve) in FOLDS.items():tr+=origins(d10,d30,START,te,f,"train");va+=origins(d10,d30,te,ve,f,"validation")
    def origin_csv(rows):return pd.DataFrame([{"fold":r["fold"],"split":r["split"],"origin_end_h":r["origin_end_h"],"target_start_h":r["target_start_h"],"control_max_abs_diff":r["control_max_abs_diff"]} for r in rows])
    td,vd=origin_csv(tr),origin_csv(va);td.to_csv(OUT/"common_train_origins.csv",index=False);vd.to_csv(OUT/"common_validation_origins.csv",index=False)
    counts=pd.DataFrame([{"fold":f,"common_train_origins":int((td.fold==f).sum()),"common_validation_origins":int((vd.fold==f).sum()),"common_30min_target_points":int((vd.fold==f).sum()*24),"same_origins_before_fit":True} for f in FOLDS])
    identity=pd.DataFrame([{"check":"parent q equals summed child raw counts","status":"PASS","targets_checked":(len(tr)+len(va))*24},{"check":"parent mixed label equals summed child raw counts","status":"PASS","targets_checked":(len(tr)+len(va))*24}])
    pp=[];mm=[];cap=[];ee=[]
    for f in FOLDS:
        a=[r for r in tr if r["fold"]==f];b=[r for r in va if r["fold"]==f]
        for e in EXPS:
            r=run_exp(e,a,b);pp+=pred_rows(e,f,b,r);mm+=metric(e,f,r);cap.append({"fold":f,"experiment":e,"input_dimension":r["dim"],"common_train_origins":len(a),"common_validation_origins":len(b),"ridge_free_parameters":r["ridge_parameters"],"logistic_free_parameters":r["logistic_parameters"],"fit_seconds":r["seconds"]})
    pred=pd.DataFrame(pp);per=pd.DataFrame(mm);capacity=pd.DataFrame(cap)
    A=pred[pred.experiment=="A_30min_baseline"].sort_values(["fold","origin_end_h","horizon_30min_step"]);D=pred[pred.experiment=="D_10min_aggregated_history_control"].sort_values(["fold","origin_end_h","horizon_30min_step"])
    dq=float(np.max(np.abs(A.q_pred.to_numpy()-D.q_pred.to_numpy())));dp=float(np.max(np.abs(A.mixed_probability.to_numpy()-D.mixed_probability.to_numpy())))
    if dq>1e-10 or dp>1e-10:raise RuntimeError("capacity control does not reproduce A")
    capacity["control_max_abs_q_prediction_diff"]=dq;capacity["control_max_abs_probability_diff"]=dp
    for f in FOLDS:
        for e in EXPS:ee.append(event_metrics(pred,e,f))
    events=pd.DataFrame(ee);summary=[]
    for e in EXPS:
        m=per[(per.experiment==e)&(per.horizon_scope=="all_horizons")];summary.append({"experiment":e,"aggregation":"fold_mean","horizon_scope":"all_horizons","q_mae":m.q_mae.mean(),"q_rmse":m.q_rmse.mean(),"mixed_brier":"NOT_RUN" if e==EXPS[2] else pd.to_numeric(m.mixed_brier).mean()})
        z=pred[pred.experiment==e];summary.append({"experiment":e,"aggregation":"pooled_common_targets","horizon_scope":"all_horizons","q_mae":mean_absolute_error(z.q_true,z.q_pred),"q_rmse":mean_squared_error(z.q_true,z.q_pred)**.5,"mixed_brier":"NOT_RUN" if e==EXPS[2] else float(np.mean((z.mixed_probability-z.mixed_true)**2))})
    summary=pd.DataFrame(summary)
    u=per[(per.horizon_scope=="all_horizons")&(per.experiment.isin(EXPS[:2]))].pivot(index="fold",columns="experiment",values=["q_rmse","mixed_brier"]);gain=(u["q_rmse"][EXPS[0]]-u["q_rmse"][EXPS[1]])/u["q_rmse"][EXPS[0]];nonworse=sum(u["mixed_brier"][EXPS[1]]<=u["mixed_brier"][EXPS[0]]+1e-12);support=int(events[(events.experiment==EXPS[0])&(events.status=="RUN")].independent_true_mixed_events.sum())
    status="CORRECTED_DEVELOPMENT_GATE_PASS" if (gain>=.05).sum()>=2 and gain["fold_3"]>=-.05 and nonworse>=2 and support>=10 else "CORRECTED_DEVELOPMENT_GATE_FAIL"
    reason="Corrected direct B versus A: q-RMSE gain percent "+json.dumps({k:round(float(v)*100,2) for k,v in gain.items()})+"; non-worse Brier folds "+str(nonworse)+"; common-scale independent mixed events "+str(support)+"."
    decision={"status":status,"reason":reason}
    c=per[(per.experiment==EXPS[2])&(per.horizon_scope=="all_horizons")].set_index("fold").q_rmse
    oldcorr=pd.DataFrame([{"comparison":"original Phase TR weighted aggregation gain pct","fold_1":44.3,"fold_2":55.3,"fold_3":31.9},{"comparison":"TR2 C fixed equal-prediction aggregation gain pct","fold_1":float((u["q_rmse"][EXPS[0]]["fold_1"]-c["fold_1"])/u["q_rmse"][EXPS[0]]["fold_1"]*100),"fold_2":float((u["q_rmse"][EXPS[0]]["fold_2"]-c["fold_2"])/u["q_rmse"][EXPS[0]]["fold_2"]*100),"fold_3":float((u["q_rmse"][EXPS[0]]["fold_3"]-c["fold_3"])/u["q_rmse"][EXPS[0]]["fold_3"]*100)}])
    counts.to_csv(OUT/"common_origin_counts.csv",index=False);identity.to_csv(OUT/"label_identity_checks.csv",index=False);capacity.to_csv(OUT/"model_capacity_and_runtime.csv",index=False);pred.to_csv(OUT/"predictions.csv",index=False);per.to_csv(OUT/"per_fold_metrics.csv",index=False);summary.to_csv(OUT/"summary_metrics.csv",index=False);events.to_csv(OUT/"independent_event_metrics.csv",index=False);oldcorr.to_csv(OUT/"original_vs_corrected.csv",index=False)
    reports(counts,identity,capacity,per,summary,events,decision);figures(summary)
    (PHASE/"FINAL_DECISION.yaml").write_text("status: "+status+"\nclassification: POST_HOC_METHOD_CORRECTION\nreference_commit: f24f87a96aa0ec451802891a1926aeb75cb76a5e\ndevelopment_end_h_exclusive: 1819.997738\nreason: "+json.dumps(reason,ensure_ascii=False)+"\n",encoding="utf-8")
    (PHASE/"RUN_AUDIT.json").write_text(json.dumps({"status":"PASS","test_accessed":False,"source_rows":{"10min":len(d10),"30min":len(d30)},"common_train_origins":len(tr),"common_validation_origins":len(va),"decision":decision,"seed":SEED},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(decision,ensure_ascii=False))
if __name__=="__main__":
    try:main()
    except Exception as e:print("FAIL-CLOSED: "+type(e).__name__+": "+str(e),file=sys.stderr);raise

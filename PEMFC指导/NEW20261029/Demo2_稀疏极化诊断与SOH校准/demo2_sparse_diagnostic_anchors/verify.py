"""Scientific safeguards: no future anchors, no truth leakage, calibrated arithmetic."""
from pathlib import Path
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
from estimator import estimate, _update, scan_anchor


def run_checks(config,input_dir,check_fixture=True):
    input_dir=Path(input_dir)
    op=pd.read_csv(input_dir/"operation.csv")
    scans=pd.read_csv(input_dir/"diagnostics.csv")
    c=config["estimator"]
    records=[]
    def record(name,passed,detail):
        records.append({"name":name,"passed":bool(passed),"detail":detail})
    full,up,meta=estimate(op,scans,c,True)
    cutoff=float(op.time_h.iloc[len(op)//2])
    prefix,_,_=estimate(op[op.time_h<=cutoff],scans[scans.time_h<=cutoff],c,True)
    delta=float(np.max(np.abs(full.loc[full.time_h<=cutoff,"estimated_soh"].to_numpy()-prefix.estimated_soh.to_numpy())))
    record("prefix_causality",delta<1e-12,f"Truncate future operation and scans at {cutoff:g} h; max earlier SOH change={delta:.3g}")
    changed_op=op.copy();changed_op.loc[changed_op.time_h>cutoff,"voltage_cell_v"]+=.075
    changed_scans=scans.copy();changed_scans.loc[changed_scans.time_h>cutoff,"voltage_cell_v"]-=.030
    modified,_,_=estimate(changed_op,changed_scans,c,True)
    delta=float(np.max(np.abs(full.loc[full.time_h<=cutoff,"estimated_soh"].to_numpy()-modified.loc[modified.time_h<=cutoff,"estimated_soh"].to_numpy())))
    record("future_mutation_invariance",delta<1e-12,f"Alter all future operating voltages and diagnostics; max earlier change={delta:.3g}")
    # Exercise the exact file-reading production function, while adversarial truth
    # files are missing and then nonsensical. No inference file reader needs truth.
    from run_demo import predict_from_files
    with tempfile.TemporaryDirectory(prefix="demo2_truth_check_") as td:
        td=Path(td)
        for name in ["operation.csv","diagnostics.csv"]:shutil.copyfile(input_dir/name,td/name)
        pred_missing=predict_from_files(td,c)[2]
        (td/"truth_for_evaluation_only.csv").write_text("this,is,not,a,valid,truth,file\n",encoding="utf-8")
        pred_poisoned=predict_from_files(td,c)[2]
        delta=float(np.max(np.abs(pred_missing.select_dtypes("number").to_numpy()-pred_poisoned.select_dtypes("number").to_numpy())))
    record("truth_file_independence",delta==0,f"Production inference with missing vs poisoned truth file; max change={delta:.3g}")
    noanchor,_,no_meta=estimate(op,scans,c,False)
    record("ablation_uses_identical_filter",no_meta["post_initial_anchor_count_used"]==0,
           "Same estimator code and settings; only post-initial diagnostic assimilation disabled")
    counts=scans.groupby("session_id").size().tolist()
    metadata_file=input_dir/"metadata.json"
    fixture=json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    if check_fixture and fixture.get("kind")=="SYNTHETIC_DATA":
        expected_sessions=int(fixture["diagnostic_sessions"])
        expected_points=int(fixture["points_per_diagnostic"])
        count_ok=(scans.session_id.nunique()==expected_sessions and all(n==expected_points for n in counts))
        detail=f"Synthetic fixture metadata expects {expected_sessions} sessions x {expected_points} points; actual counts={counts}"
    else:
        count_ok=scans.session_id.nunique()>=1 and all(n>=3 for n in counts)
        detail=f"General input check: {scans.session_id.nunique()} sessions, each with at least three points; counts={counts}"
    record("diagnostic_session_shape",count_ok,detail)
    arrived=np.sort(scans.loc[(scans.time_h>op.time_h.iloc[0]) & (scans.time_h<=op.time_h.iloc[-1]),"time_h"].unique())
    record("later_update_count",len(up)==len(arrived) and meta["post_initial_anchor_count_used"]==len(arrived),
           f"t=0 initializes only; {len(arrived)} later arrived sessions should update")
    actual_updates=up.time_h.to_numpy() if len(up) else np.array([])
    record("no_future_anchor_assimilation",len(actual_updates)==len(arrived) and np.all(actual_updates==arrived),
           "Every update timestamp exactly matches that scan's availability")
    # A controlled scan with known 20 mV loss and 0.03 slope increment must recover both.
    j=np.linspace(.2,1.4,21);coeff=np.array([-.04,-.15,.72]);dj=j-c["reference_current_density_a_cm2"]
    diagnostic=pd.DataFrame({"current_density_a_cm2":j,"voltage_cell_v":np.polyval(coeff,dj)-.020-.030*dj,
                            "temperature_c":c["reference_temperature_c"]})
    theta,_=scan_anchor(diagnostic,coeff,c)
    record("diagnostic_regression_units",np.max(np.abs(theta-np.array([.02,.03])))<1e-12,
           f"Recovered dc={theta[0]:.6f} V; dr={theta[1]:.6f} ohm cm2")
    x,P=_update(np.zeros(2),np.eye(2),np.array([[1.,1.]]),np.array([1.]),np.array([[.1]]))
    record("covariance_psd",np.linalg.eigvalsh(P).min()>-1e-12,
           "Joseph-form measurement covariance remains symmetric positive semidefinite")
    return {"passed":sum(r["passed"] for r in records),"total":len(records),"checks":records}

if __name__=="__main__":
    root=Path(__file__).resolve().parent
    result=run_checks(json.loads((root/"config.json").read_text(encoding="utf-8")),root/"data")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result["passed"]==result["total"] else 1)

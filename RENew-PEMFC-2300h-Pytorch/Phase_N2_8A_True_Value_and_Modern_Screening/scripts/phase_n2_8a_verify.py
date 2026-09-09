import json
from pathlib import Path
from PIL import Image
import pandas as pd
import yaml
R=Path(__file__).resolve().parent.parent
files=["outputs/true_delta_metrics/legacy_models_true_delta_metrics.csv","outputs/true_delta_metrics/anchor_contribution_audit.csv","outputs/leakage_audit/label_permutation_negative_control.csv","outputs/leakage_audit/temporal_misalignment_negative_control.csv","outputs/leakage_audit/feature_leakage_audit.csv","outputs/model_screening/timexer_seed42_results.csv","outputs/model_screening/timemixerpp_seed42_results.csv","outputs/true_delta_metrics/modern_model_true_delta_comparison.csv","outputs/true_delta_metrics/per_horizon_true_delta_metrics.csv","outputs/lifecycle_analysis/lifecycle_generalization_summary.csv","outputs/tail_error_analysis/tail_error_summary.csv","outputs/model_screening/multiseed_modern_model_comparison.csv"]
figs=["r2_delta_model_comparison.png","rmse_delta_model_comparison.png","r2_abs_vs_r2_delta.png","foldwise_r2_delta.png","foldwise_rmse_delta.png","lifecycle_degradation_ratio.png","tail_error_p95_comparison.png","per_horizon_r2_delta.png","per_horizon_rmse_delta.png"]
rows=[]
for f in files:
 d=pd.read_csv(R/f);rows.append({"path":f,"rows":len(d),"columns":len(d.columns),"pass":bool(len(d)>0 and len(d.columns)>0)})
ims=[]
for f in figs:
 p=R/"outputs/figures"/f
 with Image.open(p) as im:w,h=im.size
 ims.append({"path":f,"width":w,"height":h,"bytes":p.stat().st_size,"pass":bool(w>=600 and h>=400 and p.stat().st_size>5000)})
fw=json.loads((R/"outputs/leakage_audit/test_firewall_audit.json").read_text());dec=yaml.safe_load((R/"outputs/frozen_config/PHASE_N2_8A_MODEL_DECISION.yaml").read_text());art=json.loads((R/"outputs/leakage_audit/artifact_tool_csv_audit.json").read_text())
checks={"csvs":bool(all(x["pass"] for x in rows) and len(rows)==12),"figures":bool(all(x["pass"] for x in ims) and len(ims)==9),"firewall":bool(fw["test_target_rows_read"]==0 and not fw["test_predictions_constructed"] and not fw["test_metrics_computed"] and not fw["test_feature_diagnostics_computed"]),"feature_audit":bool(pd.read_csv(R/"outputs/leakage_audit/feature_leakage_audit.csv").status.eq("PASS").all()),"artifact_csv_audit":bool(art["status"]=="PASS"),"report":bool((R/"PHASE_N2_8A_RESULTS.md").is_file()),"freeze_test_false":bool(dec["test_performance_accessed"] is False)}
payload={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"csv":rows,"figures":ims,"phase_decision_status":dec["phase_status"]};(R/"outputs/leakage_audit/output_verification.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n");print(json.dumps(payload["checks"],ensure_ascii=False))

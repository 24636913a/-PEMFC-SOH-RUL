"""Fail-closed structural verification for Phase N2.7R deliverables."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CSV = [
    "outputs/integrity_audit/paired_expert_prediction_audit.csv", "outputs/stability/sg_lstm_stability_bootstrap.csv",
    "outputs/gate_diagnostics/pointwise_winner_by_horizon.csv", "outputs/gate_diagnostics/expert_disagreement_analysis.csv",
    "outputs/gate_diagnostics/gate_feature_definition.csv", "outputs/gate_diagnostics/gate_logistic_state_horizon.csv",
    "outputs/gate_diagnostics/gate_logistic_disagreement.csv", "outputs/gate_diagnostics/gate_continuous_mlp.csv",
    "outputs/gate_diagnostics/fixed_fusion_meta_oos.csv", "outputs/gate_diagnostics/horizon_static_fusion.csv",
    "outputs/gate_diagnostics/adaptive_gate_meta_oos.csv", "outputs/gate_diagnostics/gate_weight_by_horizon.csv",
    "outputs/gate_diagnostics/gate_weight_by_ageing_stage.csv", "outputs/gate_diagnostics/gate_weight_by_disagreement.csv",
    "outputs/gate_diagnostics/timesnet_tail_error_gate_analysis.csv", "outputs/gate_diagnostics/ophrs2_ablation.csv",
    "outputs/gate_diagnostics/gate_multiseed_summary.csv", "outputs/stability/gate_bootstrap_comparison.csv",
]
FIGURES = ["timesnet_win_rate_by_horizon.png", "expert_disagreement_vs_error.png", "gate_weight_by_horizon.png", "gate_weight_by_ageing_stage.png", "gate_weight_by_disagreement.png", "meta_oos_rmse_comparison.png", "fold2_rmse_comparison.png", "fold3_rmse_comparison.png", "tail_error_gate_response.png", "oracle_fixed_horizonstatic_adaptive_comparison.png"]

def main() -> None:
    csv_rows=[]
    for rel in CSV:
        path=ROOT/rel; frame=pd.read_csv(path)
        csv_rows.append({"path":rel,"exists":path.is_file(),"rows":len(frame),"columns":len(frame.columns),"pass":path.is_file() and len(frame)>0 and len(frame.columns)>0})
    figures=[]
    for name in FIGURES:
        path=ROOT/"outputs/figures"/name
        with Image.open(path) as img: width,height=img.size
        figures.append({"path":str(path.relative_to(ROOT)),"exists":path.is_file(),"bytes":path.stat().st_size,"width":width,"height":height,"pass":path.is_file() and path.stat().st_size>5000 and width>=600 and height>=400})
    paired=pd.read_csv(ROOT/"outputs/integrity_audit/paired_expert_prediction_audit.csv")
    firewall=json.loads((ROOT/"outputs/integrity_audit/test_firewall_audit.json").read_text(encoding="utf-8"))
    firewall_keys=["test_target_rows_read","test_predictions_constructed","test_gate_predictions_constructed","test_gate_weights_constructed","test_metrics_computed","test_performance_accessed"]
    firewall_ok=firewall.get("test_target_rows_read")==0 and all(firewall.get(key) is False for key in firewall_keys[1:])
    artifact=json.loads((ROOT/"outputs/integrity_audit/artifact_tool_csv_audit.json").read_text(encoding="utf-8"))
    freeze=yaml.safe_load((ROOT/"outputs/frozen_config/PHASE_N2_7R_GATE_DECISION.yaml").read_text(encoding="utf-8"))
    checks={"required_csvs":bool(all(x["pass"] for x in csv_rows) and len(csv_rows)==18),"required_figures":bool(all(x["pass"] for x in figures) and len(figures)==10),"paired_audit_9_of_9":bool(len(paired)==9 and paired.status.eq("PASS").all()),"test_firewall":bool(firewall_ok),"artifact_tool_audit":bool(artifact.get("status")=="PASS" and artifact.get("csvCount")==18),"report":bool((ROOT/"PHASE_N2_7R_RESULTS.md").is_file()),"freeze":bool(freeze.get("test_performance_accessed") is False)}
    payload={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"csv":csv_rows,"figures":figures,"test_firewall":{key:firewall.get(key) for key in firewall_keys},"best_gate":freeze.get("best_adaptive_gate"),"phase_n2_8_decision":freeze.get("phase_n2_8_decision")}
    (ROOT/"outputs/integrity_audit/output_verification.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":payload["status"],"checks":checks},ensure_ascii=False))

if __name__=="__main__": main()

"""Fail-closed verification for the completed Phase N2.6R artifact set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from phase_n2_6r_common import PHASE_ROOT, write_csv, write_json


REQUIRED_CSV = [
    "outputs/implementation_audit/implementation_fidelity_audit.csv",
    "outputs/implementation_audit/lstm_reproduction_check.csv",
    "outputs/screening/high_fidelity_screening.csv",
    "outputs/multiseed/multiseed_comparison.csv",
    "outputs/per_horizon/per_horizon_rmse_r2_mae.csv",
    "outputs/per_horizon/cycle_region_rmse_r2.csv",
    "outputs/ageing_stage/ageing_stage_rmse_r2_mae.csv",
    "outputs/statistical_tests/bootstrap_rmse_difference.csv",
    "outputs/screening/model_efficiency.csv",
]
REQUIRED_FIGURES = [
    "mean_rmse_comparison.png", "mean_r2_comparison.png", "fold_rmse_comparison.png",
    "fold_r2_comparison.png", "per_horizon_rmse.png", "per_horizon_r2.png",
    "cycle_region_rmse.png", "late_development_rmse.png", "rmse_vs_parameter_count.png",
    "representative_validation_prediction.png",
]


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    for relative in REQUIRED_CSV:
        path = PHASE_ROOT / relative
        frame = pd.read_csv(path)
        check(f"required_csv:{relative}", path.is_file() and len(frame) > 0, f"rows={len(frame)}, columns={len(frame.columns)}")

    report = (PHASE_ROOT / "PHASE_N2_6R_RESULTS.md").read_text(encoding="utf-8")
    plan = (PHASE_ROOT / "PHASE_N2_6R_PLAN.md").read_text(encoding="utf-8")
    check("report_exists", len(report) > 3000, f"characters={len(report)}")
    check("plan_exists", len(plan) > 1000, f"characters={len(plan)}")
    check("report_q1_q12", all(f"### Q{i}" in report for i in range(1, 13)), "Q1-Q12 present")
    check("report_metric_priority", all(token in report for token in ["RMSE", "R²", "MAE"]), "RMSE/R²/MAE present")
    check("report_general_search_stop", "General backbone search STOP" in report, "STOP recorded")

    firewall = json.loads((PHASE_ROOT / "outputs/implementation_audit/test_firewall_audit.json").read_text(encoding="utf-8"))
    firewall_ok = (
        firewall["test_target_rows_read"] == 0
        and firewall["test_predictions_constructed"] is False
        and firewall["test_rmse_computed"] is False
        and firewall["test_r2_computed"] is False
        and firewall["test_mae_computed"] is False
    )
    check("test_firewall", firewall_ok, json.dumps({key: firewall[key] for key in ["test_target_rows_read", "test_predictions_constructed", "test_rmse_computed", "test_r2_computed", "test_mae_computed"]}))

    prediction_files = sorted((PHASE_ROOT / "outputs/model_runs").glob("*_predictions.npz"))
    max_target_end = -np.inf
    predictions_ok = True
    for path in prediction_files:
        arrays = dict(np.load(path, allow_pickle=False))
        max_target_end = max(max_target_end, float(np.max(arrays["target_end_age_h"])))
        predictions_ok &= (
            arrays["y_true"].shape == arrays["y_pred"].shape
            and arrays["y_true"].shape[1] == 24
            and np.isfinite(arrays["y_true"]).all()
            and np.isfinite(arrays["y_pred"]).all()
            and float(np.max(arrays["target_end_age_h"])) < 1819.997738
        )
    check("all_local_predictions_development_only", predictions_ok, f"files={len(prediction_files)}, max_target_end_h={max_target_end:.6f}")

    fidelity = json.loads((PHASE_ROOT / "outputs/implementation_audit/implementation_fidelity_audit.json").read_text(encoding="utf-8"))
    check("implementation_fidelity", fidelity["status"] == "PASS" and all(fidelity["config_status"].values()), f"configs={len(fidelity['config_status'])}")
    reproduction = pd.read_csv(PHASE_ROOT / "outputs/implementation_audit/lstm_reproduction_check.csv")
    check("lstm_reproduction_within_2pct", bool(reproduction["within_2pct"].all()), f"max_fraction={reproduction['relative_difference_fraction'].max():.8g}")

    screening = pd.read_csv(PHASE_ROOT / "outputs/screening/high_fidelity_screening.csv")
    screening_folds = screening[screening["row_type"] == "fold"]
    check("only_three_model_families", set(screening_folds["family"]) == {"LSTM", "TimesNet", "ModernTCN"}, str(sorted(screening_folds["family"].unique())))
    check("exact_screening_configs", set(screening_folds["config_name"]) == {"LSTM", "timesnet_hf_t1", "timesnet_hf_t2", "moderntcn_hf_m1", "moderntcn_hf_m2"}, f"configs={screening_folds['config_name'].nunique()}")
    check("three_folds_each_config", bool((screening_folds.groupby("config_name")["fold"].nunique() == 3).all()), "all configs have 3 folds")

    multiseed = pd.read_csv(PHASE_ROOT / "outputs/multiseed/multiseed_comparison.csv")
    detail = multiseed[multiseed["row_type"] == "fold_seed"]
    check("multiseed_exact_models", set(detail["model"]) == {"LSTM", "TimesNet"}, str(sorted(detail["model"].unique())))
    check("multiseed_exact_seeds", set(detail["seed"].astype(int)) == {42, 2024, 3407}, str(sorted(detail["seed"].astype(int).unique())))
    check("multiseed_complete_grid", len(detail) == 18, f"rows={len(detail)}")

    horizon = pd.read_csv(PHASE_ROOT / "outputs/per_horizon/per_horizon_rmse_r2_mae.csv")
    horizon_mean = horizon[(horizon["row_type"] == "mean") & (horizon["evaluation_scope"] == "seed42_three_model_comparison")]
    check("per_horizon_three_models", set(horizon_mean["model"]) == {"LSTM", "TimesNet", "ModernTCN"}, f"models={horizon_mean['model'].nunique()}")
    check("per_horizon_24_steps", bool((horizon_mean.groupby("model")["horizon_step"].nunique() == 24).all()), f"rows={len(horizon_mean)}")
    check("per_horizon_metrics_finite", bool(np.isfinite(horizon_mean[["rmse_mV", "mae_mV", "r2", "bias_mV"]].to_numpy()).all()), "all finite")

    cycle = pd.read_csv(PHASE_ROOT / "outputs/per_horizon/cycle_region_rmse_r2.csv")
    check("cycle_region_complete", len(cycle) == 3 and "rmse_mV_h9_5" in cycle and "r2_h9_5" in cycle, f"rows={len(cycle)}")
    ageing = pd.read_csv(PHASE_ROOT / "outputs/ageing_stage/ageing_stage_rmse_r2_mae.csv")
    age_mean = ageing[(ageing["row_type"] == "mean") & (ageing["evaluation_scope"] == "seed42_three_model_comparison")]
    check("ageing_stages_complete", set(age_mean["ageing_stage"]) == {"Early Development", "Middle Development", "Late Development"} and set(age_mean["model"]) == {"LSTM", "TimesNet", "ModernTCN"}, f"rows={len(age_mean)}")

    bootstrap = pd.read_csv(PHASE_ROOT / "outputs/statistical_tests/bootstrap_rmse_difference.csv")
    check("bootstrap_protocol", len(bootstrap) == 3 and bool((bootstrap["bootstrap_repetitions"] == 2000).all()) and bool((bootstrap["block_length_steps"] == 19).all()), f"rows={len(bootstrap)}")

    frozen = yaml.safe_load((PHASE_ROOT / "outputs/frozen_config/PHASE_N2_6R_BACKBONE_SELECTION.yaml").read_text(encoding="utf-8"))
    required_yaml = ["selected_backbone", "mean_rmse_mV", "mean_mae_mV", "mean_r2", "fold1_rmse_mV", "fold2_rmse_mV", "fold3_rmse_mV", "late_development_rmse_mV", "h9_5_rmse_mV", "random_seeds", "parameter_count", "implementation_source", "implementation_commit", "implementation_fidelity_status", "test_performance_accessed"]
    check("frozen_yaml_required_fields", all(key in frozen for key in required_yaml), f"required={len(required_yaml)}")
    check("frozen_backbone_lstm", frozen["selected_backbone"] == "LSTM" and frozen["general_backbone_search"] == "STOP", f"selected={frozen['selected_backbone']}")
    check("frozen_test_false", frozen["test_performance_accessed"] is False and frozen["phase_n3_authorized"] is False, "Test false, Phase N3 false")

    artifact = json.loads((PHASE_ROOT / "outputs/implementation_audit/artifact_tool_csv_audit.json").read_text(encoding="utf-8"))
    check("artifact_tool_csv_audit", artifact["status"] == "PASS" and artifact["audited_file_count"] == 9, f"audited={artifact['audited_file_count']}/9")

    for name in REQUIRED_FIGURES:
        path = PHASE_ROOT / "outputs/figures" / name
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        check(f"figure:{name}", path.stat().st_size > 10000 and width >= 1000 and height >= 800, f"bytes={path.stat().st_size}, size={width}x{height}")

    vendor_checks = [
        PHASE_ROOT / "vendor/TimesNet_official/models/TimesNet.py",
        PHASE_ROOT / "vendor/TimesNet_official/LICENSE",
        PHASE_ROOT / "vendor/ModernTCN_official/ModernTCN-Long-term-forecasting/models/ModernTCN.py",
        PHASE_ROOT / "vendor/ModernTCN_official/LICENSE",
    ]
    check("vendored_sources_and_licenses", all(path.is_file() for path in vendor_checks), ", ".join(path.name for path in vendor_checks))

    manifest_rows = []
    for path in sorted(item for item in PHASE_ROOT.rglob("*") if item.is_file()):
        if path.name in {"phase_n2_6r_file_manifest.csv", "phase_n2_6r_verification_results.json"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_rows.append({"relative_path": path.relative_to(PHASE_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    write_csv(pd.DataFrame(manifest_rows), PHASE_ROOT / "outputs/implementation_audit/phase_n2_6r_file_manifest.csv")

    passed = all(row["status"] == "PASS" for row in checks)
    write_json(
        {"status": "PASS" if passed else "FAIL", "check_count": len(checks), "passed_count": sum(row["status"] == "PASS" for row in checks), "checks": checks},
        PHASE_ROOT / "outputs/implementation_audit/phase_n2_6r_verification_results.json",
    )
    print(json.dumps({"status": "PASS" if passed else "FAIL", "checks": len(checks), "passed": sum(row["status"] == "PASS" for row in checks)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

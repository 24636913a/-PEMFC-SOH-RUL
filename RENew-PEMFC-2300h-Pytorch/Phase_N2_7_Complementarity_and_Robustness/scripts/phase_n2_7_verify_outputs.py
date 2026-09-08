"""Fail-closed structural verification of all frozen Phase N2.7 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from phase_n2_7_common import PHASE_ROOT, write_json


REQUIRED_CSV = [
    "outputs/integrity_audit/paired_prediction_integrity_audit.csv",
    "outputs/error_complementarity/timesnet_win_rate.csv",
    "outputs/error_complementarity/error_correlation.csv",
    "outputs/error_complementarity/large_error_overlap.csv",
    "outputs/fusion_diagnostics/oracle_fusion_upper_bound.csv",
    "outputs/fusion_diagnostics/fixed_fusion_baseline.csv",
    "outputs/shift_diagnostics/feature_shift_smd_psi.csv",
    "outputs/shift_diagnostics/window_shift_score.csv",
    "outputs/shift_diagnostics/shift_conditioned_model_performance.csv",
    "outputs/shift_diagnostics/expert_regime_analysis.csv",
    "outputs/shift_diagnostics/winner_predictability.csv",
    "outputs/timesnet_tuning/timesnet_robustness_tuning.csv",
    "outputs/preprocessing/preprocessing_screening.csv",
    "outputs/multiseed/multiseed_robustness.csv",
    "outputs/error_complementarity/final_complementarity_analysis.csv",
]

REQUIRED_FIGURES = [
    "lstm_timesnet_absolute_error_scatter.png",
    "timesnet_win_rate_by_fold.png",
    "timesnet_win_rate_by_horizon.png",
    "error_correlation_heatmap.png",
    "large_error_overlap.png",
    "shift_score_vs_error.png",
    "rmse_by_shift_level.png",
    "rmse_by_ageing_stage.png",
    "timesnet_tuning_fold_rmse.png",
    "preprocessing_rmse_comparison.png",
    "oracle_vs_lstm_rmse.png",
    "fixed_fusion_rmse.png",
]


def main() -> None:
    checks = []
    for relative in REQUIRED_CSV:
        path = PHASE_ROOT / relative
        frame = pd.read_csv(path)
        status = path.is_file() and path.stat().st_size > 20 and len(frame) > 0 and len(frame.columns) > 1
        status &= not any(str(column).startswith("Unnamed:") for column in frame.columns)
        checks.append({"type": "core_csv", "path": relative, "rows": len(frame), "columns": len(frame.columns), "status": "PASS" if status else "FAIL"})

    for name in REQUIRED_FIGURES:
        path = PHASE_ROOT / "outputs/figures" / name
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        status = path.stat().st_size > 10_000 and width >= 800 and height >= 600
        checks.append({"type": "core_figure", "path": f"outputs/figures/{name}", "width": width, "height": height, "status": "PASS" if status else "FAIL"})

    paired = pd.read_csv(PHASE_ROOT / REQUIRED_CSV[0])
    checks.append({
        "type": "paired_predictions",
        "units": len(paired),
        "status": "PASS" if len(paired) == 9 and paired["status"].eq("PASS").all() and not paired["enters_test"].any() else "FAIL",
    })
    firewall = json.loads((PHASE_ROOT / "outputs/integrity_audit/test_firewall_audit.json").read_text(encoding="utf-8"))
    firewall_ok = (
        firewall.get("test_target_rows_read") == 0
        and firewall.get("test_predictions_constructed") is False
        and firewall.get("test_metrics_computed") is False
        and firewall.get("test_shift_diagnostics_computed") is False
    )
    checks.append({"type": "test_firewall", "status": "PASS" if firewall_ok else "FAIL", **firewall})

    prediction_files = list((PHASE_ROOT / "outputs/model_runs").glob("*_predictions.npz"))
    max_target_end = -np.inf
    prediction_ok = True
    for path in prediction_files:
        arrays = np.load(path, allow_pickle=False)
        maximum = float(np.max(arrays["target_end_age_h"]))
        max_target_end = max(max_target_end, maximum)
        prediction_ok &= maximum < 1819.997738
        prediction_ok &= arrays["y_true"].shape == arrays["y_pred"].shape and arrays["y_true"].shape[1] == 24
    checks.append({
        "type": "local_prediction_arrays",
        "file_count": len(prediction_files),
        "maximum_target_end_h": max_target_end,
        "status": "PASS" if prediction_ok and len(prediction_files) == 39 else "FAIL",
    })
    artifact_audit = json.loads((PHASE_ROOT / "outputs/integrity_audit/artifact_tool_csv_audit.json").read_text(encoding="utf-8"))
    checks.append({"type": "artifact_tool_csv_import_and_inspect", "csv_count": artifact_audit.get("csvCount"), "status": artifact_audit.get("status")})
    report_ok = (PHASE_ROOT / "PHASE_N2_7_RESULTS.md").is_file()
    freeze_ok = (PHASE_ROOT / "outputs/frozen_config/PHASE_N2_7_FINAL_EXPERT_SELECTION.yaml").is_file()
    checks.append({"type": "report_and_freeze", "status": "PASS" if report_ok and freeze_ok else "FAIL"})
    overall = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    payload = {"status": overall, "required_csv_count": len(REQUIRED_CSV), "required_figure_count": len(REQUIRED_FIGURES), "checks": checks}
    write_json(payload, PHASE_ROOT / "outputs/integrity_audit/automated_output_verification.json")
    if overall != "PASS":
        raise RuntimeError("Phase N2.7 output verification failed")
    print(json.dumps({"status": overall, "csv": len(REQUIRED_CSV), "figures": len(REQUIRED_FIGURES), "local_prediction_files": len(prediction_files)}))


if __name__ == "__main__":
    main()

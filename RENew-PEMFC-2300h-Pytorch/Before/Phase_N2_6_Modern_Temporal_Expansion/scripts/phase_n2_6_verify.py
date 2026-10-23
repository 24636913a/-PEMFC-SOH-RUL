"""Fail-closed final verification for Phase N2.6.

This verifier reads only frozen protocol metadata and Development artifacts.  It
does not open Test targets, construct Test predictions, or compute Test metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parent
AUDIT_DIR = PHASE_ROOT / "outputs" / "implementation_audit"
TEST_START_H = 1819.997738

CORE_CSV = {
    "screening": PHASE_ROOT / "outputs/screening_results/modern_architecture_screening.csv",
    "multiseed": PHASE_ROOT / "outputs/multi_seed_robustness/multi_seed_backbone_comparison.csv",
    "per_horizon": PHASE_ROOT / "outputs/per_horizon_analysis/per_horizon_model_comparison.csv",
    "cycle": PHASE_ROOT / "outputs/per_horizon_analysis/cycle_region_model_comparison.csv",
    "ageing": PHASE_ROOT / "outputs/ageing_stage_analysis/ageing_stage_architecture_comparison.csv",
    "bootstrap": PHASE_ROOT / "outputs/statistical_tests/bootstrap_vs_lstm.csv",
    "efficiency": PHASE_ROOT / "outputs/model_efficiency/model_accuracy_efficiency_comparison.csv",
    "sanity": PHASE_ROOT / "outputs/implementation_audit/implementation_sanity_audit.csv",
}

FIGURES = [
    "modern_architecture_rmse_comparison.png",
    "skill_vs_lstm_comparison.png",
    "multi_seed_rmse_distribution.png",
    "per_horizon_rmse_curves.png",
    "per_horizon_skill_lstm.png",
    "cycle_region_8_5_to_10_5h.png",
    "ageing_stage_rmse_comparison.png",
    "accuracy_vs_parameter_count.png",
    "accuracy_vs_training_time.png",
    "representative_validation_prediction.png",
]

EXPECTED_FAMILY_BEST = {
    "TimesNet": "timesnet_d48_k3",
    "ModernTCN": "moderntcn_d64_k13_b3",
    "PatchTSMixer": "patchtsmixer_p6_d64",
    "CompactMamba": "compact_mamba_d64_l2",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    required = [
        PHASE_ROOT / "PHASE_N2_6_PLAN.md",
        PHASE_ROOT / "MODEL_IMPLEMENTATION_NOTES.md",
        PHASE_ROOT / "PHASE_N2_6_RESULTS.md",
        PHASE_ROOT / "configs/phase_n2_6_config.yaml",
        PHASE_ROOT / "outputs/frozen_config/PHASE_N2_6_BACKBONE_SELECTION.yaml",
        PHASE_ROOT / "outputs/frozen_config/final_backbone_selection.json",
        PHASE_ROOT / "outputs/frozen_config/screening_selection.json",
        PHASE_ROOT / "outputs/frozen_config/multiseed_completion.json",
        PHASE_ROOT / "outputs/implementation_audit/test_firewall_audit.json",
        PHASE_ROOT / "outputs/implementation_audit/implementation_sanity_audit.json",
        PHASE_ROOT / "outputs/implementation_audit/lstm_reproduction_check.csv",
        PHASE_ROOT / "outputs/implementation_audit/matched_origin_audit.csv",
        PHASE_ROOT / "outputs/implementation_audit/artifact_tool_csv_audit.json",
        PHASE_ROOT / "outputs/implementation_audit/FIGURE_VISUAL_QA.md",
        *CORE_CSV.values(),
    ]
    missing = [str(path.relative_to(PHASE_ROOT)) for path in required if not path.is_file()]
    check("required_outputs_present", not missing, f"missing={missing}")
    if missing:
        write_results(checks)
        raise SystemExit(1)

    firewall = load_json(PHASE_ROOT / "outputs/implementation_audit/test_firewall_audit.json")
    firewall_ok = (
        firewall.get("target_rows_read_from_test") == 0
        and firewall.get("test_predictions_constructed") is False
        and firewall.get("test_metrics_computed") is False
        and firewall.get("test_performance_accessed") is False
    )
    check("test_firewall_zero_access", firewall_ok, json.dumps(firewall, ensure_ascii=False))

    artifact = load_json(PHASE_ROOT / "outputs/implementation_audit/artifact_tool_csv_audit.json")
    artifact_ok = (
        artifact.get("status") == "PASS"
        and artifact.get("audited_file_count") == 8
        and artifact.get("required_file_count") == 8
        and all(row.get("artifact_tool_import") == "PASS" and not row.get("formula_error_tokens") for row in artifact.get("files", []))
    )
    check("artifact_tool_csv_audit_8_of_8", artifact_ok, f"audited={artifact.get('audited_file_count')}")

    sanity = pd.read_csv(CORE_CSV["sanity"])
    sanity_json = load_json(PHASE_ROOT / "outputs/implementation_audit/implementation_sanity_audit.json")
    sanity_ok = (
        len(sanity) == 7
        and set(sanity["status"]) == {"PASS"}
        and sanity["parameter_count"].between(50_000, 200_000).all()
        and sanity["input_shape"].eq("4x48x14").all()
        and sanity["output_shape"].eq("4x24").all()
        and all(bool_value(value) for value in sanity["cpu_forward_backward"])
        and all(bool_value(value) for value in sanity["cuda_forward_backward"])
        and not any(bool_value(value) for value in sanity["test_target_accessed"])
        and sanity_json.get("status") == "PASS"
        and sanity_json.get("passed_config_count") == 7
    )
    check("implementation_sanity_7_of_7", sanity_ok, f"rows={len(sanity)}")

    reproduction = pd.read_csv(PHASE_ROOT / "outputs/implementation_audit/lstm_reproduction_check.csv")
    reproduction_ok = (
        len(reproduction) == 4
        and reproduction["relative_difference_fraction"].abs().lt(0.02).all()
        and all(bool_value(value) for value in reproduction["within_2pct"])
        and math.isclose(float(reproduction.loc[reproduction["fold"] == "mean_3fold", "reproduced_rmse_mV"].iloc[0]), 9.005857842572587, rel_tol=0, abs_tol=1e-12)
    )
    check("lstm_reproduction_within_2pct", reproduction_ok, f"max_relative_difference={reproduction['relative_difference_fraction'].abs().max():.6g}")

    matched = pd.read_csv(PHASE_ROOT / "outputs/implementation_audit/matched_origin_audit.csv")
    matched_ok = (
        len(matched) == 3
        and all(bool_value(value) for value in matched["phase_n2_origins_exact_match"])
        and all(bool_value(value) for value in matched["phase_n2_targets_exact_match"])
        and not any(bool_value(value) for value in matched["enters_test"])
        and float(matched["maximum_target_end_h"].max()) < TEST_START_H
    )
    check("matched_origin_and_target_exact", matched_ok, f"maximum_target_end_h={matched['maximum_target_end_h'].max()}")

    screening = pd.read_csv(CORE_CSV["screening"])
    fold_rows = screening.loc[screening["row_type"] == "fold"]
    summaries = screening.loc[screening["row_type"] == "config_summary"]
    best_by_family = summaries.loc[summaries.groupby("family")["rmse_mV"].idxmin()].set_index("family")
    family_best_ok = all(
        family in best_by_family.index and best_by_family.loc[family, "config_name"] == config
        for family, config in EXPECTED_FAMILY_BEST.items()
    )
    screening_ok = (
        len(screening) == 32
        and len(fold_rows) == 24
        and len(summaries) == 8
        and set(fold_rows["fold"]) == {"fold_1", "fold_2", "fold_3"}
        and set(fold_rows["seed"].astype(int)) == {42}
        and family_best_ok
    )
    check("screening_complete_and_family_best", screening_ok, f"fold_rows={len(fold_rows)}, summary_rows={len(summaries)}")

    lstm_rmse = float(best_by_family.loc["LSTM", "rmse_mV"])
    new_best = best_by_family.loc[list(EXPECTED_FAMILY_BEST)]
    all_worse_10pct = bool((new_best["rmse_mV"] > 1.10 * lstm_rmse).all())
    selection = load_json(PHASE_ROOT / "outputs/frozen_config/screening_selection.json")
    selection_ok = (
        all_worse_10pct
        and selection.get("eligible_within_10pct") == []
        and selection.get("top_configs_for_multiseed") == []
        and selection.get("all_new_models_worse_than_10pct") is True
    )
    check("multiseed_entry_threshold_fail_closed", selection_ok, f"eligible={selection.get('eligible_within_10pct')}")

    multiseed = pd.read_csv(CORE_CSV["multiseed"])
    multi_folds = multiseed.loc[multiseed["row_type"] == "fold"]
    multi_summary = multiseed.loc[multiseed["row_type"] == "model_summary"]
    multiseed_ok = (
        len(multiseed) == 13
        and len(multi_folds) == 9
        and set(multiseed["family"]) == {"LSTM"}
        and set(multi_folds["seed"].astype(int)) == {42, 2024, 3407}
        and len(multi_summary) == 1
        and int(float(multi_summary.iloc[0]["seed_count"])) == 3
    )
    check("multiseed_only_incumbent_after_no_qualifiers", multiseed_ok, f"rows={len(multiseed)}, families={sorted(multiseed['family'].unique())}")

    run_json = sorted((PHASE_ROOT / "outputs/model_runs").glob("*.json"))
    run_npz = sorted((PHASE_ROOT / "outputs/model_runs").glob("*_predictions.npz"))
    run_ok = len(run_json) == 24 and len(run_npz) == 24
    max_target_end = -np.inf
    for manifest_path in run_json:
        run = load_json(manifest_path)
        run_ok &= (
            run.get("status") == "complete"
            and run.get("test_predictions_constructed") is False
            and run.get("test_metrics_computed") is False
            and run.get("lookback_hours") == 24
            and run.get("horizon_hours") == 12
            and run.get("resolution_hours") == 0.5
        )
    for npz_path in run_npz:
        with np.load(npz_path) as arrays:
            required_keys = {"y_true", "y_pred", "anchor", "origin_age_h", "target_end_age_h", "row_origin_index"}
            run_ok &= required_keys.issubset(arrays.files)
            run_ok &= arrays["y_true"].shape == arrays["y_pred"].shape
            run_ok &= arrays["y_true"].shape[1] == 24
            run_ok &= bool(np.isfinite(arrays["y_true"]).all() and np.isfinite(arrays["y_pred"]).all())
            max_target_end = max(max_target_end, float(arrays["target_end_age_h"].max()))
    run_ok &= max_target_end < TEST_START_H
    check("development_run_artifacts_complete", bool(run_ok), f"json={len(run_json)}, npz={len(run_npz)}, max_target_end_h={max_target_end}")

    per_horizon = pd.read_csv(CORE_CSV["per_horizon"])
    expected_horizons = {step / 2 for step in range(1, 25)}
    per_horizon_ok = (
        len(per_horizon) == 480
        and set(per_horizon["horizon_hours"].astype(float)) == expected_horizons
        and len(per_horizon.loc[per_horizon["row_type"] == "fold"]) == 360
        and len(per_horizon.loc[per_horizon["row_type"] == "mean"]) == 120
        and set(per_horizon["model"]) == {"LSTM", *EXPECTED_FAMILY_BEST.keys()}
    )
    check("per_horizon_24_steps_complete", per_horizon_ok, f"rows={len(per_horizon)}, horizons={per_horizon['horizon_hours'].nunique()}")

    cycle = pd.read_csv(CORE_CSV["cycle"])
    cycle_ok = len(cycle) == 100 and set(cycle["horizon_hours"].astype(float)) == {8.5, 9.0, 9.5, 10.0, 10.5}
    check("cycle_region_complete", cycle_ok, f"rows={len(cycle)}")

    ageing = pd.read_csv(CORE_CSV["ageing"])
    ageing_means = ageing.loc[ageing["row_type"] == "mean"]
    ageing_ok = (
        len(ageing_means) == 15
        and set(ageing_means["ageing_stage"]) == {"Early Development", "Middle Development", "Late Development"}
        and set(ageing_means["model"]) == {"LSTM", *EXPECTED_FAMILY_BEST.keys()}
    )
    check("ageing_stage_complete", ageing_ok, f"mean_rows={len(ageing_means)}")

    bootstrap = pd.read_csv(CORE_CSV["bootstrap"])
    versus_lstm = bootstrap.loc[bootstrap["reference"] == "LSTM"]
    bootstrap_ok = (
        len(bootstrap) == 5
        and len(versus_lstm) == 4
        and set(versus_lstm["candidate"]) == set(EXPECTED_FAMILY_BEST)
        and set(bootstrap["bootstrap_repetitions"].astype(int)) == {2000}
        and set(bootstrap["block_length_steps"].astype(int)) == {19}
        and (versus_lstm["ci95_lower_V"] > 0).all()
        and (versus_lstm["probability_candidate_better"] == 0).all()
    )
    check("paired_moving_block_bootstrap_complete", bootstrap_ok, f"rows={len(bootstrap)}, comparisons_vs_lstm={len(versus_lstm)}")

    efficiency = pd.read_csv(CORE_CSV["efficiency"])
    efficiency_ok = (
        len(efficiency) == 5
        and set(efficiency["model"]) == {"LSTM", *EXPECTED_FAMILY_BEST.keys()}
        and efficiency["parameter_count"].between(50_000, 200_000).all()
        and efficiency.sort_values(["rmse_mV", "parameter_count"]).iloc[0]["model"] == "LSTM"
    )
    check("accuracy_efficiency_complete", efficiency_ok, f"rows={len(efficiency)}")

    frozen_path = PHASE_ROOT / "outputs/frozen_config/PHASE_N2_6_BACKBONE_SELECTION.yaml"
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8-sig"))
    frozen_ok = (
        frozen.get("status") == "PASS"
        and frozen.get("sampling_resolution") == 0.5
        and frozen.get("feature_group") == "F2"
        and len(frozen.get("exact_feature_names", [])) == 14
        and frozen.get("lookback_hours") == 24
        and frozen.get("lookback_steps") == 48
        and frozen.get("horizon_hours") == 12
        and frozen.get("horizon_steps") == 24
        and frozen.get("selected_backbone") == "LSTM"
        and frozen.get("selection_conclusion") == "LSTM remains incumbent"
        and frozen.get("eligible_new_models_for_multiseed") == []
        and frozen.get("phase_n3_authorized") is False
        and frozen.get("test_performance_accessed") is False
        and frozen.get("test_target_rows_read") == 0
        and frozen.get("test_predictions_constructed") is False
        and frozen.get("test_metrics_computed") is False
        and np.allclose(frozen.get("test_boundary"), [1819.997738, 2256.8], rtol=0, atol=1e-9)
    )
    check("frozen_backbone_protocol", bool(frozen_ok), f"selected={frozen.get('selected_backbone')}, phase_n3_authorized={frozen.get('phase_n3_authorized')}")

    report = (PHASE_ROOT / "PHASE_N2_6_RESULTS.md").read_text(encoding="utf-8-sig")
    report_ok = (
        "Phase N2.6：**PASS**" in report
        and "LSTM remains incumbent" in report
        and all(f"### Q{number}" in report for number in range(1, 13))
        and "继续禁止打开 Test" in report
        and "CompactMamba" in report
        and "TimesNet" in report
    )
    check("results_report_q1_to_q12", report_ok, f"characters={len(report)}")

    figure_dir = PHASE_ROOT / "outputs/figures"
    figure_details = []
    figures_ok = True
    for name in FIGURES:
        path = figure_dir / name
        if not path.is_file():
            figures_ok = False
            figure_details.append(f"{name}:missing")
            continue
        try:
            width, height = png_dimensions(path)
            size = path.stat().st_size
            valid = size > 10_000 and width >= 800 and height >= 450
            figures_ok &= valid
            figure_details.append(f"{name}:{width}x{height}:{size}")
        except Exception as exc:  # fail closed
            figures_ok = False
            figure_details.append(f"{name}:{exc}")
    qa_text = (AUDIT_DIR / "FIGURE_VISUAL_QA.md").read_text(encoding="utf-8-sig")
    figures_ok &= "审计状态：**PASS**" in qa_text and "10/10" in qa_text
    check("figures_10_of_10_and_visual_qa", bool(figures_ok), "; ".join(figure_details))

    scripts = sorted((PHASE_ROOT / "scripts").glob("*.py"))
    compile_errors: list[str] = []
    for script in scripts:
        try:
            compile(script.read_text(encoding="utf-8-sig"), str(script), "exec")
        except Exception as exc:
            compile_errors.append(f"{script.name}: {exc}")
    check("python_scripts_compile", not compile_errors, f"scripts={len(scripts)}, errors={compile_errors}")

    root_readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8-sig")
    check("root_readme_registered", "Phase N2.6 — 现代时序架构扩展与骨干模型再筛选" in root_readme, "README contains Phase N2.6 section")

    inherited = [
        PROJECT_ROOT / "Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv",
        PROJECT_ROOT / "Phase_N2_Model_Screening/outputs/frozen_config/PHASE_N2_FROZEN_CONFIG.yaml",
        PROJECT_ROOT / "Phase_N2_5_Cycle_Aware_xLSTM/outputs/frozen_config/PHASE_N2_5_FROZEN_CONFIG.yaml",
    ]
    inherited_ok = all(path.is_file() for path in inherited)
    check("inherited_frozen_inputs_present", inherited_ok, f"files={len(inherited)}")

    write_hash_manifests(inherited)
    status = write_results(checks)
    print(json.dumps({"status": status, "checks_passed": sum(row["status"] == "PASS" for row in checks), "checks_total": len(checks)}, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


def write_hash_manifests(inherited: list[Path]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    excluded_names = {"phase_n2_6_file_manifest.csv", "phase_n2_6_verification_results.json", "inherited_input_hashes.csv"}
    rows: list[dict[str, object]] = []
    for path in sorted(PHASE_ROOT.rglob("*")):
        if not path.is_file() or path.name in excluded_names:
            continue
        relative_parts = path.relative_to(PHASE_ROOT).parts
        if any(part in {"vendor", "node_modules", "__pycache__"} for part in relative_parts):
            continue
        rows.append({"relative_path": path.relative_to(PHASE_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    with (AUDIT_DIR / "phase_n2_6_file_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    inherited_rows = []
    for path in inherited:
        inherited_rows.append(
            {
                "relative_path_from_project_root": path.relative_to(PROJECT_ROOT).as_posix(),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else "",
                "sha256": sha256(path) if path.is_file() else "",
                "usage": "read-only inherited protocol/data input",
            }
        )
    with (AUDIT_DIR / "inherited_input_hashes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path_from_project_root", "exists", "bytes", "sha256", "usage"],
        )
        writer.writeheader()
        writer.writerows(inherited_rows)


def write_results(checks: list[dict[str, object]]) -> str:
    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    payload = {
        "phase": "Phase N2.6",
        "status": status,
        "checks_passed": sum(row["status"] == "PASS" for row in checks),
        "checks_total": len(checks),
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "checks": checks,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "phase_n2_6_verification_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return status


if __name__ == "__main__":
    main()

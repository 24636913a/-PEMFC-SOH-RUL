from __future__ import annotations

import hashlib
import json
import math
import py_compile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from PIL import Image


ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_5_Cycle_Aware_xLSTM")
PROJECT_ROOT = ROOT.parent
N1_ROOT = PROJECT_ROOT / "Phase_N1_Data_Foundation"
N2_ROOT = PROJECT_ROOT / "Phase_N2_Model_Screening"
OUTPUTS = ROOT / "outputs"
AUDITS = OUTPUTS / "audits"
TEST_START_H = 1819.997738
EXPECTED_SEEDS = {42, 2024, 3407}
EXPECTED_FOLDS = {"fold_1", "fold_2", "fold_3"}


@dataclass
class Check:
    category: str
    name: str
    passed: bool
    detail: str


CHECKS: list[Check] = []


def record(category: str, name: str, condition: bool, detail: Any) -> None:
    CHECKS.append(Check(category, name, bool(condition), str(detail)))


def guarded(category: str, name: str, fn: Callable[[], tuple[bool, Any]]) -> None:
    try:
        condition, detail = fn()
        record(category, name, condition, detail)
    except Exception as exc:  # fail-closed audit
        record(category, name, False, f"{type(exc).__name__}: {exc}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_fold_seed_rows(frame: pd.DataFrame, models: set[str]) -> tuple[bool, str]:
    fold_rows = frame.loc[(frame["row_type"] == "fold") & frame["model"].isin(models)].copy()
    observed = {
        (str(row.model), int(row.seed), str(row.fold))
        for row in fold_rows.itertuples(index=False)
    }
    expected = {(model, seed, fold) for model in models for seed in EXPECTED_SEEDS for fold in EXPECTED_FOLDS}
    return observed == expected, f"observed={len(observed)}, expected={len(expected)}"


def verify_attention_npz() -> tuple[bool, str]:
    run_index = read_json(OUTPUTS / "frozen_config" / "architecture_run_index.json")
    checked = 0
    maximum_sum_error = 0.0
    for key, entry in run_index.items():
        model, _, _ = key.split("|")
        if model not in {"CA-LSTM", "CA-xLSTM", "CA-xLSTM-P15"}:
            continue
        prediction_path = OUTPUTS / "model_runs" / f"{entry['run_id']}_predictions.npz"
        with np.load(prediction_path) as archive:
            weights = archive["attention_weights"]
            if weights.ndim != 2 or weights.shape[1] != 3 or not np.isfinite(weights).all():
                return False, f"invalid attention tensor: {prediction_path.name} {weights.shape}"
            maximum_sum_error = max(maximum_sum_error, float(np.max(np.abs(weights.sum(axis=1) - 1.0))))
            if float(weights.min()) < -1e-7 or float(weights.max()) > 1.0 + 1e-7:
                return False, f"attention outside [0,1]: {prediction_path.name}"
            checked += 1
    return checked == 21 and maximum_sum_error < 1e-5, f"runs={checked}, max_row_sum_error={maximum_sum_error:.3e}"


def verify_run_manifests_and_firewall() -> tuple[bool, str]:
    run_index = read_json(OUTPUTS / "frozen_config" / "architecture_run_index.json")
    if len(run_index) != 39:
        return False, f"run index contains {len(run_index)} entries, expected 39"
    local_manifests = list((OUTPUTS / "model_runs").glob("*.json"))
    inspected_manifests = 0
    inspected_predictions = 0
    max_target_end = -math.inf
    for key, entry in run_index.items():
        model, seed_text, fold = key.split("|")
        seed = int(seed_text)
        source_root = OUTPUTS if entry["source"] == "Phase_N2_5_Cycle_Aware_xLSTM" else N2_ROOT / "outputs"
        manifest_path = source_root / "model_runs" / f"{entry['run_id']}.json"
        prediction_path = source_root / "model_runs" / f"{entry['run_id']}_predictions.npz"
        manifest = read_json(manifest_path)
        test_predictions_constructed = manifest.get("test_predictions_constructed", False)
        if manifest.get("status") != "complete" or test_predictions_constructed is not False:
            return False, f"unsafe/incomplete manifest: {manifest_path}"
        if str(manifest.get("fold")) != fold or int(manifest.get("seed")) != seed:
            return False, f"run index mismatch: {key} -> {manifest_path.name}"
        with np.load(prediction_path) as archive:
            target_end = np.asarray(archive["target_end_age_h"], dtype=float)
            if target_end.size == 0 or not np.isfinite(target_end).all():
                return False, f"empty/nonfinite target endpoints: {prediction_path.name}"
            max_target_end = max(max_target_end, float(target_end.max()))
            if np.any(target_end >= TEST_START_H):
                return False, f"Test boundary entered: {prediction_path.name}"
        inspected_manifests += 1
        inspected_predictions += 1
    unsafe_names = [p.name for p in OUTPUTS.rglob("*") if p.is_file() and "test" in p.name.lower() and "predict" in p.name.lower()]
    return (
        inspected_manifests == 39
        and inspected_predictions == 39
        and len(local_manifests) == 39
        and not unsafe_names,
        f"indexed_runs=39, local_manifests={len(local_manifests)}, max_target_end={max_target_end:.6f} h, unsafe_names={unsafe_names}",
    )


def verify_core_csv() -> tuple[bool, str]:
    frame = pd.read_csv(OUTPUTS / "multi_seed_robustness" / "multi_seed_architecture_robustness.csv")
    ok, detail = exact_fold_seed_rows(frame, {"LSTM", "xLSTM", "CA-LSTM", "CA-xLSTM"})
    finite_columns = ["rmse_V", "mae_V", "r2", "skill_best"]
    fold_rows = frame.loc[frame["row_type"] == "fold"]
    finite = np.isfinite(fold_rows[finite_columns].to_numpy(dtype=float)).all()
    return ok and finite, f"{detail}; finite_metrics={finite}"


def verify_wrong_period() -> tuple[bool, str]:
    frame = pd.read_csv(OUTPUTS / "cycle_aware_ablation" / "wrong_period_control.csv")
    fold_rows = frame.loc[frame["row_type"] == "fold"].copy()
    p19 = fold_rows.loc[(fold_rows["model"] == "CA-xLSTM") & (fold_rows["seed"].astype(str) == "42")]
    p15 = fold_rows.loc[(fold_rows["model"] == "CA-xLSTM-P15") & (fold_rows["seed"].astype(str) == "42")]
    return (
        set(p19["fold"]) == EXPECTED_FOLDS
        and set(p15["fold"]) == EXPECTED_FOLDS
        and set(p19["period_steps"].astype(int)) == {19}
        and set(p15["period_steps"].astype(int)) == {15},
        f"P19 rows={len(p19)}, P15 rows={len(p15)}, common seed=42",
    )


def verify_screening() -> tuple[bool, str]:
    frame = pd.read_csv(OUTPUTS / "xlstm_variant_screening" / "xlstm_variant_screening.csv")
    models = {"LSTM", "sLSTM", "mLSTM", "Hybrid_xLSTM"}
    fold_rows = frame.loc[frame["aggregation"] == "fold"]
    observed = {(str(r.model), int(r.seed), str(r.fold)) for r in fold_rows.itertuples(index=False)}
    expected = {(m, 42, f) for m in models for f in EXPECTED_FOLDS}
    means = frame.loc[frame["aggregation"] == "mean"].set_index("model")
    global_minimum = str(means["rmse_mV"].idxmin())
    selected_xlstm = str(means.loc[["sLSTM", "mLSTM", "Hybrid_xLSTM"], "rmse_mV"].idxmin())
    params_in_range = means.loc[["sLSTM", "mLSTM", "Hybrid_xLSTM"], "parameter_count"].between(50_000, 150_000).all()
    return observed == expected and global_minimum == "LSTM" and selected_xlstm == "sLSTM" and params_in_range, (
        f"fold_rows={len(observed)}/12; global minimum={global_minimum}; best xLSTM={selected_xlstm}; "
        f"xLSTM params within 50k-150k={params_in_range}"
    )


def verify_selected_xlstm() -> tuple[bool, str]:
    selection = read_json(OUTPUTS / "frozen_config" / "xlstm_screening_selection.json")
    return (
        selection.get("selected_xlstm_variant") == "sLSTM"
        and selection.get("reproduction_check_pass") is True
        and float(selection.get("reproduction_mean_difference_fraction", 1.0)) <= 0.02
        and selection.get("target_rows_read_from_test") == 0
        and selection.get("test_predictions_constructed") is False,
        json.dumps(selection, ensure_ascii=False),
    )


def verify_bootstrap() -> tuple[bool, str]:
    frame = pd.read_csv(OUTPUTS / "statistical_tests" / "bootstrap_architecture_comparison.csv")
    expected = {
        "xLSTM_vs_LSTM",
        "CA-LSTM_vs_LSTM",
        "CA-xLSTM_vs_xLSTM",
        "CA-xLSTM_vs_LSTM",
        "P19_vs_P15",
    }
    numeric = frame[["mean_delta_rmse_V", "median_delta_rmse_V", "ci95_lower_V", "ci95_upper_V", "probability_candidate_better"]]
    return (
        set(frame["comparison"]) == expected
        and set(frame["bootstrap_repetitions"].astype(int)) == {2000}
        and set(frame["block_length_steps"].astype(int)) == {19}
        and np.isfinite(numeric.to_numpy(dtype=float)).all()
        and frame["probability_candidate_better"].between(0, 1).all(),
        f"comparisons={len(frame)}, repetitions={set(frame['bootstrap_repetitions'])}, block={set(frame['block_length_steps'])}",
    )


def verify_cycle_region() -> tuple[bool, str]:
    frame = pd.read_csv(OUTPUTS / "cycle_analysis" / "cycle_region_per_horizon_metrics.csv")
    means = frame.loc[frame["row_type"] == "mean"]
    horizons = set(means["horizon_hours"].astype(float).round(1))
    required_models = {"LSTM", "xLSTM", "CA-LSTM", "CA-xLSTM", "Ridge", "CycleRepeat"}
    return (
        horizons == {8.5, 9.0, 9.5, 10.0, 10.5}
        and required_models.issubset(set(means["model"])),
        f"horizons={sorted(horizons)}, models={sorted(set(means['model']))}",
    )


def verify_attention_summary() -> tuple[bool, str]:
    frame = pd.read_csv(OUTPUTS / "cycle_analysis" / "cycle_attention_weight_summary.csv")
    grouped = frame.groupby(["model", "ageing_stage"], dropna=False)["mean_weight"].sum()
    valid = (
        set(frame["model"]) == {"CA-LSTM", "CA-xLSTM"}
        and set(frame["lag_label"]) == {"t", "t-P", "t-2P"}
        and np.allclose(grouped.to_numpy(), 1.0, atol=1e-5)
    )
    return valid, f"groups={len(grouped)}, max_sum_error={float(np.max(np.abs(grouped.to_numpy()-1.0))):.3e}"


def verify_ageing_stage() -> tuple[bool, str]:
    frame = pd.read_csv(OUTPUTS / "cycle_analysis" / "ageing_stage_model_comparison.csv")
    means = frame.loc[frame["row_type"] == "mean"]
    return (
        set(means["model"]) == {"LSTM", "CA-LSTM", "CA-xLSTM"}
        and set(means["ageing_stage"]) == {"Early Development", "Middle Development", "Late Development"}
        and np.isfinite(means[["rmse_V", "mae_V", "r2"]].to_numpy(dtype=float)).all(),
        f"mean rows={len(means)}, stages={sorted(set(means['ageing_stage']))}",
    )


def verify_efficiency() -> tuple[bool, str]:
    frame = pd.read_csv(OUTPUTS / "model_comparison" / "model_efficiency_comparison.csv")
    required_models = {"LSTM", "xLSTM", "CA-LSTM", "CA-xLSTM", "CA-xLSTM-P15"}
    numeric = frame[["parameter_count", "peak_gpu_memory_mb", "mean_epoch_time_s", "total_training_time_s", "rmse_mV"]]
    return (
        set(frame["model"]) == required_models
        and frame["parameter_count"].between(50_000, 150_000).all()
        and np.isfinite(numeric.to_numpy(dtype=float)).all()
        and (numeric.to_numpy(dtype=float) > 0).all(),
        f"models={sorted(set(frame['model']))}, parameter_range=[{frame['parameter_count'].min()}, {frame['parameter_count'].max()}]",
    )


def verify_frozen_config() -> tuple[bool, str]:
    config_path = OUTPUTS / "frozen_config" / "PHASE_N2_5_FROZEN_CONFIG.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_features = [
        "cell_voltage_mean", "cell_voltage_std", "cell_voltage_range", "cell_voltage_min", "cell_voltage_max",
        "actual_current_A_mean", "actual_current_A_std", "actual_current_A_peak_to_peak",
        "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std",
        "actual_current_density_A_cm2_peak_to_peak", "load_transition_count",
        "operation_point_transitions", "zero_current_fraction",
    ]
    valid = (
        config.get("status") == "PASS"
        and config.get("architecture_innovation_status") == "NO-GO"
        and config.get("test_performance_accessed") is False
        and float(config.get("sampling_resolution")) == 0.5
        and config.get("feature_group") == "F2"
        and config.get("exact_feature_names") == expected_features
        and int(config.get("lookback_hours")) == 24
        and int(config.get("lookback_steps")) == 48
        and int(config.get("horizon_hours")) == 12
        and int(config.get("horizon_steps")) == 24
        and int(config.get("empirical_period_steps")) == 19
        and config.get("model") == "LSTM"
        and config.get("backbone_type") == "LSTM"
        and int(config.get("parameter_count")) == 55_448
        and set(config.get("random_seeds", [])) == EXPECTED_SEEDS
        and np.allclose(config.get("train_boundary"), [72.788690, 1383.195476])
        and np.allclose(config.get("validation_boundary"), [1383.195476, 1819.997738])
        and np.allclose(config.get("test_boundary"), [1819.997738, 2256.8])
        and config.get("ensemble") is False
    )
    return valid, f"model={config.get('model')}, features={len(config.get('exact_feature_names', []))}, seeds={config.get('random_seeds')}"


def verify_selection() -> tuple[bool, str]:
    selection = read_json(OUTPUTS / "frozen_config" / "final_architecture_selection.json")
    valid = (
        selection.get("phase_status") == "PASS"
        and selection.get("architecture_innovation_status") == "NO-GO"
        and selection.get("selected_xlstm_variant") == "sLSTM"
        and selection.get("final_frozen_model") == "LSTM"
        and selection.get("phase_n3_go") is True
        and selection.get("ca_lstm_go") is False
        and selection.get("ca_xlstm_go") is False
        and selection.get("target_rows_read_from_test") == 0
        and selection.get("test_predictions_constructed") is False
    )
    return valid, json.dumps(selection, ensure_ascii=False)


def verify_firewall_records() -> tuple[bool, str]:
    audit = read_json(OUTPUTS / "frozen_config" / "test_firewall_audit.json")
    matched = pd.read_csv(AUDITS / "matched_origin_and_test_firewall_audit.csv")
    valid = (
        audit.get("target_rows_read_from_test") == 0
        and audit.get("test_predictions_constructed") is False
        and audit.get("test_performance_accessed") is False
        and float(audit.get("test_metadata_start_h")) >= TEST_START_H
        and matched["phase_n2_origins_exact_match"].astype(bool).all()
        and matched["phase_n2_targets_exact_match"].astype(bool).all()
        and (~matched["enters_test"].astype(bool)).all()
        and float(matched["maximum_target_end_h"].max()) < TEST_START_H
    )
    return valid, f"target_rows={audit.get('target_rows_read_from_test')}, max_development_target={matched['maximum_target_end_h'].max()}"


def verify_reproduction() -> tuple[bool, str]:
    frame = pd.read_csv(OUTPUTS / "xlstm_variant_screening" / "lstm_reproduction_check.csv")
    return (
        len(frame) == 4
        and frame["within_2pct"].astype(bool).all()
        and float(frame["relative_difference_fraction"].max()) <= 0.02,
        f"rows={len(frame)}, max_relative_difference={frame['relative_difference_fraction'].max():.8f}",
    )


def verify_report() -> tuple[bool, str]:
    text = (ROOT / "PHASE_N2_5_RESULTS.md").read_text(encoding="utf-8")
    q_present = all(f"### Q{i}" in text for i in range(1, 13))
    terms = ["PASS", "Architecture innovation", "NO-GO", "最终冻结模型：**LSTM**", "Phase N3：**GO**"]
    return q_present and all(term in text for term in terms), f"Q1-Q12={q_present}; required conclusions={all(term in text for term in terms)}"


def verify_figures() -> tuple[bool, str]:
    required = [
        "xlstm_variant_rmse.png", "cycle_aware_ablation_rmse.png", "lstm_vs_ca_xlstm_skill.png",
        "per_horizon_rmse_curves.png", "cycle_region_8_5_to_10_5h.png", "true_period_vs_wrong_period.png",
        "multi_seed_variability.png", "bootstrap_delta_rmse.png", "cycle_attention_weights.png",
        "cycle_attention_weights_vs_ageing.png", "accuracy_vs_parameter_count.png",
        "representative_validation_prediction.png",
    ]
    dimensions: list[str] = []
    for name in required:
        path = OUTPUTS / "figures" / name
        if not path.exists() or path.stat().st_size < 10_000:
            return False, f"missing or undersized figure: {name}"
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.width < 800 or image.height < 450:
                return False, f"insufficient dimensions: {name} {image.size}"
            dimensions.append(f"{name}:{image.width}x{image.height}")
    qa_text = (AUDITS / "FIGURE_VISUAL_QA.md").read_text(encoding="utf-8")
    return "**PASS**" in qa_text and len(required) == 12, f"12 decoded figures; visual QA PASS; {dimensions[0]}"


def verify_required_files() -> tuple[bool, str]:
    required = [
        ROOT / "PHASE_N2_5_PLAN.md", ROOT / "MODEL_ARCHITECTURE.md", ROOT / "PHASE_N2_5_RESULTS.md",
        OUTPUTS / "xlstm_variant_screening" / "xlstm_variant_screening.csv",
        OUTPUTS / "cycle_aware_ablation" / "cycle_aware_ablation.csv",
        OUTPUTS / "cycle_aware_ablation" / "wrong_period_control.csv",
        OUTPUTS / "multi_seed_robustness" / "multi_seed_architecture_robustness.csv",
        OUTPUTS / "cycle_analysis" / "cycle_region_per_horizon_metrics.csv",
        OUTPUTS / "statistical_tests" / "bootstrap_architecture_comparison.csv",
        OUTPUTS / "cycle_analysis" / "cycle_attention_weight_summary.csv",
        OUTPUTS / "cycle_analysis" / "ageing_stage_model_comparison.csv",
        OUTPUTS / "model_comparison" / "model_efficiency_comparison.csv",
        OUTPUTS / "frozen_config" / "PHASE_N2_5_FROZEN_CONFIG.yaml",
        AUDITS / "artifact_tool_csv_audit.json", AUDITS / "FIGURE_VISUAL_QA.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists() or path.stat().st_size == 0]
    return not missing, f"required={len(required)}, missing={missing}"


def verify_artifact_csv_audit() -> tuple[bool, str]:
    audit = read_json(AUDITS / "artifact_tool_csv_audit.json")
    return (
        audit.get("status") == "PASS"
        and audit.get("audited_file_count") == 9
        and all(item.get("artifact_tool_import") == "PASS" and not item.get("formula_error_tokens") for item in audit.get("files", [])),
        f"status={audit.get('status')}, files={audit.get('audited_file_count')}/9",
    )


def verify_scripts_compile() -> tuple[bool, str]:
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    for script in scripts:
        py_compile.compile(str(script), doraise=True)
    return len(scripts) >= 6, f"compiled Python scripts={len(scripts)}"


def write_inventory_and_hash_manifest() -> tuple[bool, str]:
    excluded_names = {"artifact_inventory.csv", "phase_n2_5_verification_results.csv", "hash_manifest.json"}
    files = [
        path for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and "node_modules" not in path.parts
        and path.name not in excluded_names
    ]
    records = []
    for path in sorted(files):
        records.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    pd.DataFrame(records).to_csv(AUDITS / "artifact_inventory.csv", index=False, encoding="utf-8-sig")
    input_paths = {
        "phase_n1_causal_features_30min": N1_ROOT / "outputs" / "causal_feature_tables" / "causal_features_30min.csv",
        "phase_n2_frozen_config": N2_ROOT / "outputs" / "frozen_config" / "PHASE_N2_FROZEN_CONFIG.yaml",
        "phase_n2_model_selection": N2_ROOT / "outputs" / "frozen_config" / "model_selection.json",
        "phase_n2_bootstrap_pairwise": N2_ROOT / "outputs" / "statistical_tests" / "bootstrap_pairwise_comparison.csv",
    }
    manifest = {
        "status": "PASS",
        "hash_algorithm": "SHA-256",
        "phase_n2_5_inventory_file_count": len(records),
        "input_files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
    }
    (AUDITS / "hash_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(records) > 100 and all(len(item["sha256"]) == 64 for item in records), f"inventoried_files={len(records)}, frozen_inputs={len(input_paths)}"


def main() -> int:
    AUDITS.mkdir(parents=True, exist_ok=True)
    guarded("artifacts", "required outputs exist", verify_required_files)
    guarded("spreadsheet", "artifact-tool CSV import and inspection", verify_artifact_csv_audit)
    guarded("protocol", "Phase N2 LSTM reproduction within 2%", verify_reproduction)
    guarded("protocol", "matched origins and Test firewall records", verify_firewall_records)
    guarded("protocol", "all indexed predictions remain before Test", verify_run_manifests_and_firewall)
    guarded("screening", "xLSTM screening protocol", verify_screening)
    guarded("screening", "selected xLSTM and reproduction gate", verify_selected_xlstm)
    guarded("ablation", "core A0-A3 exact 3-seed x 3-fold grid", verify_core_csv)
    guarded("ablation", "pre-registered wrong-period P15 control", verify_wrong_period)
    guarded("statistics", "moving-block bootstrap specification", verify_bootstrap)
    guarded("cycle", "8.5-10.5 h neighborhood metrics", verify_cycle_region)
    guarded("cycle", "attention summary simplex", verify_attention_summary)
    guarded("cycle", "attention arrays simplex", verify_attention_npz)
    guarded("cycle", "ageing-stage comparison", verify_ageing_stage)
    guarded("efficiency", "model efficiency and parameter limits", verify_efficiency)
    guarded("freeze", "final architecture selection", verify_selection)
    guarded("freeze", "frozen Phase N2.5 configuration", verify_frozen_config)
    guarded("report", "Chinese report answers Q1-Q12", verify_report)
    guarded("figures", "figure files and visual QA", verify_figures)
    guarded("reproducibility", "all Python scripts compile", verify_scripts_compile)
    guarded("reproducibility", "artifact inventory and input hashes", write_inventory_and_hash_manifest)

    frame = pd.DataFrame([check.__dict__ for check in CHECKS])
    frame.to_csv(AUDITS / "phase_n2_5_verification_results.csv", index=False, encoding="utf-8-sig")
    passed = bool(frame["passed"].all()) and len(frame) >= 20
    summary = {
        "status": "PASS" if passed else "FAIL",
        "passed_checks": int(frame["passed"].sum()),
        "total_checks": int(len(frame)),
        "failed_checks": frame.loc[~frame["passed"], "name"].tolist(),
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_performance_accessed": False,
    }
    (AUDITS / "phase_n2_5_verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Phase N2.5 verification: {summary['status']} ({summary['passed_checks']}/{summary['total_checks']})")
    if summary["failed_checks"]:
        print("Failed checks:")
        for name in summary["failed_checks"]:
            print(f"- {name}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

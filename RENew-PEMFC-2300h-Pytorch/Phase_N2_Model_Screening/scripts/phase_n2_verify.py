"""Phase N2 final fail-closed verification and immutable artifact inventory."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import py_compile
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
AUDIT_DIR = OUT / "audits"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

CHECKS: list[dict[str, object]] = []


def record(name: str, passed: bool, detail: str) -> None:
    CHECKS.append({"check": name, "passed": bool(passed), "detail": detail})


def require(name: str, condition: bool, detail: str) -> None:
    record(name, condition, detail)


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def finite_column(frame: pd.DataFrame, column: str) -> bool:
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and values.map(math.isfinite).all())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    required_tables = [
        "outputs/baseline_results/baseline_summary.csv",
        "outputs/baseline_results/baseline_per_horizon_metrics.csv",
        "outputs/feature_group_diagnostics/feature_group_comparison.csv",
        "outputs/lookback_diagnostics/lookback_matched_origin_comparison.csv",
        "outputs/lookback_diagnostics/matched_origin_audit.csv",
        "outputs/model_screening/modern_model_screening_summary.csv",
        "outputs/model_screening/modern_model_per_horizon_metrics.csv",
        "outputs/model_screening/per_horizon_metrics.csv",
        "outputs/robustness_analysis/multi_seed_robustness.csv",
        "outputs/robustness_analysis/horizon_sensitivity.csv",
        "outputs/statistical_tests/bootstrap_pairwise_comparison.csv",
        "outputs/robustness_analysis/1h_robustness_comparison.csv",
        "outputs/model_screening/model_efficiency_summary.csv",
        "outputs/frozen_config/development_fold_manifest.csv",
    ]
    required_figures = [
        "baseline_rmse_comparison.png",
        "baseline_skill_comparison.png",
        "cycle_repeat_vs_models.png",
        "feature_group_incremental_value.png",
        "lookback_12_24_48h_comparison.png",
        "modern_model_validation_rmse.png",
        "modern_model_skill_best.png",
        "multi_seed_variability.png",
        "per_horizon_skill_curves.png",
        "model_accuracy_vs_parameter_count.png",
        "representative_validation_trajectory.png",
    ]
    required_docs = [
        "PHASE_N2_PLAN.md",
        "PHASE_N2_FEATURE_GROUPS.md",
        "PHASE_N2_RESULTS.md",
        "configs/phase_n2_config.yaml",
        "outputs/frozen_config/PHASE_N2_FROZEN_CONFIG.yaml",
        "outputs/audits/FIGURE_VISUAL_QA.md",
    ]

    missing = [p for p in required_tables + required_docs if not (ROOT / p).is_file()]
    undersized_figures = [
        name
        for name in required_figures
        if not (OUT / "figures" / name).is_file()
        or (OUT / "figures" / name).stat().st_size < 10_000
    ]
    require("required_files_exist", not missing, f"missing={missing}")
    require("figures_exist_and_nontrivial", not undersized_figures, f"invalid={undersized_figures}")

    firewall = read_json("outputs/frozen_config/test_firewall_audit.json")
    require(
        "test_target_rows_zero",
        firewall.get("target_rows_read_from_test") == 0,
        f"target_rows_read_from_test={firewall.get('target_rows_read_from_test')}",
    )
    require(
        "test_predictions_absent",
        firewall.get("test_predictions_constructed") is False,
        f"test_predictions_constructed={firewall.get('test_predictions_constructed')}",
    )
    require(
        "test_boundary_metadata_only",
        math.isclose(float(firewall.get("test_metadata_start_h", -1)), 1820.0, abs_tol=1e-6)
        and float(firewall.get("test_metadata_rows", 0)) > 0,
        f"start={firewall.get('test_metadata_start_h')}, rows={firewall.get('test_metadata_rows')}",
    )

    gate = read_json("outputs/frozen_config/baseline_gate.json")
    require(
        "baseline_gate",
        gate.get("status") == "PASS"
        and gate.get("baselines_complete") is True
        and set(gate.get("models", [])) == {"Persistence", "LocalLinear", "CycleRepeat", "Ridge"}
        and math.isclose(float(gate.get("selected_ridge_alpha", -1)), 1.0),
        json.dumps(gate, ensure_ascii=False),
    )

    fold_manifest = pd.read_csv(ROOT / "outputs/frozen_config/development_fold_manifest.csv")
    expected_fold_roles = {(f"fold_{i}", role) for i in (1, 2, 3) for role in ("train", "validation")}
    observed_fold_roles = set(zip(fold_manifest["fold"], fold_manifest["role"]))
    require("development_fold_roles", observed_fold_roles == expected_fold_roles, str(observed_fold_roles))
    require(
        "development_never_enters_test",
        float(fold_manifest["ageing_end_h"].max()) <= 1819.997738 + 1e-9,
        f"maximum development end={fold_manifest['ageing_end_h'].max()}",
    )
    require(
        "development_legal_windows_positive",
        (fold_manifest["legal_window_count_24h_to_12h"] > 0).all(),
        f"minimum legal windows={fold_manifest['legal_window_count_24h_to_12h'].min()}",
    )

    diagnostics = read_json("outputs/frozen_config/diagnostic_selection.json")
    require(
        "feature_group_frozen_F2",
        diagnostics.get("status") == "PASS"
        and diagnostics.get("selected_feature_group") == "F2"
        and diagnostics.get("feature_status") == "resolved",
        f"selection={diagnostics.get('selected_feature_group')}, status={diagnostics.get('feature_status')}",
    )
    require(
        "lookback_frozen_24h",
        diagnostics.get("selected_lookback_hours") == 24
        and diagnostics.get("matched_origin_audit_pass") is True,
        f"lookback={diagnostics.get('selected_lookback_hours')}",
    )
    require(
        "F2_increment_rules_pass",
        diagnostics["feature_rules"]["Ridge"]["F2_vs_F1"]["rule_pass"] is True
        and diagnostics["feature_rules"]["LSTM"]["F2_vs_F1"]["rule_pass"] is True,
        "Ridge and LSTM both pass the pre-frozen F2 versus F1 rule",
    )
    require(
        "F3_increment_rules_fail",
        diagnostics["feature_rules"]["Ridge"]["F3_vs_F2"]["rule_pass"] is False
        and diagnostics["feature_rules"]["LSTM"]["F3_vs_F2"]["rule_pass"] is False,
        "F3 is not admitted by the pre-frozen rule",
    )

    matched = pd.read_csv(ROOT / "outputs/lookback_diagnostics/matched_origin_audit.csv")
    audit_columns = [c for c in matched.columns if c.endswith("_48h")]
    bool_pass = all(matched[c].astype(str).str.lower().eq("true").all() for c in audit_columns)
    require(
        "matched_origin_fail_closed_audit",
        len(matched) == 9 and set(matched["lookback_hours"]) == {12, 24, 48} and bool_pass,
        f"rows={len(matched)}, compared_boolean_columns={audit_columns}",
    )

    screen = pd.read_csv(ROOT / "outputs/model_screening/modern_model_screening_summary.csv")
    screen_fold = screen[screen["aggregation"] == "fold"].copy()
    screen_mean = screen[screen["aggregation"] == "mean"].copy()
    expected_models = {"LSTM", "TCN", "iTransformer", "TimeMixer", "PatchTST"}
    require(
        "exact_screening_model_set",
        set(screen_mean["model"]) == expected_models and len(screen_fold) == 15,
        f"mean_models={sorted(screen_mean['model'].unique())}, fold_rows={len(screen_fold)}",
    )
    unified = (
        screen_fold["seed"].eq(42).all()
        and screen_fold["feature_group"].eq("F2").all()
        and screen_fold["lookback_hours"].eq(24.0).all()
        and screen_fold["horizon_hours"].eq(12.0).all()
        and set(screen_fold["fold"]) == {"fold_1", "fold_2", "fold_3"}
    )
    require("unified_screening_protocol", unified, "seed=42, F2, lookback=24 h, horizon=12 h, three folds")
    require(
        "screening_metrics_finite",
        all(finite_column(screen_fold, c) for c in ["rmse_V", "mae_V", "skill_best"]),
        "fold RMSE, MAE and Skill_Best are finite",
    )

    selection = read_json("outputs/frozen_config/model_selection.json")
    require(
        "model_selection_freeze",
        selection.get("status") == "PASS"
        and selection.get("strongest_simple_baseline") == "Ridge"
        and selection.get("final_candidate_model") == "LSTM"
        and selection.get("top2_modern_models") == ["PatchTST", "iTransformer"]
        and selection.get("learned_model_go") is True
        and selection.get("test_predictions_constructed") is False,
        json.dumps(selection, ensure_ascii=False),
    )

    multi = pd.read_csv(ROOT / "outputs/robustness_analysis/multi_seed_robustness.csv")
    seed_rows = multi[multi["row_type"] == "seed_summary"].copy()
    model_rows = multi[multi["row_type"] == "model_summary"].copy()
    require(
        "multi_seed_exact_models_and_seeds",
        set(seed_rows["model"]) == {"LSTM", "PatchTST", "iTransformer"}
        and set(pd.to_numeric(seed_rows["seed"])) == {42, 2024, 3407}
        and len(seed_rows) == 9,
        f"rows={len(seed_rows)}, models={sorted(seed_rows['model'].unique())}",
    )
    require(
        "multi_seed_positive_skill",
        (pd.to_numeric(seed_rows["skill_best"]) > 0).all()
        and (pd.to_numeric(model_rows["positive_skill_fold_count"]) == 9).all(),
        "all nine fold-seed cases per retained model have positive Skill_Best",
    )

    bootstrap = pd.read_csv(ROOT / "outputs/statistical_tests/bootstrap_pairwise_comparison.csv")
    modern_boot = bootstrap[bootstrap["category"] == "top_modern_model"]
    require(
        "bootstrap_protocol",
        (bootstrap["bootstrap_repetitions"] == 2000).all()
        and (bootstrap["block_length_steps"] == 19).all(),
        "2000 moving-block replicates, block length 19 steps",
    )
    require(
        "LSTM_significantly_beats_Ridge",
        len(modern_boot) == 1
        and modern_boot.iloc[0]["comparison"] == "LSTM_vs_Ridge"
        and float(modern_boot.iloc[0]["ci95_upper_V"]) < 0
        and float(modern_boot.iloc[0]["probability_candidate_better"]) >= 0.95,
        modern_boot.to_json(orient="records"),
    )

    horizon = pd.read_csv(ROOT / "outputs/robustness_analysis/horizon_sensitivity.csv")
    horizon_mean = horizon[horizon["aggregation"] == "mean"]
    require(
        "horizon_sensitivity_complete",
        set(horizon_mean["model"]) == {"PatchTST", "iTransformer"}
        and set(horizon_mean["horizon_hours"]) == {4.0, 8.0, 9.5, 12.0, 24.0}
        and len(horizon_mean) == 10,
        f"mean rows={len(horizon_mean)}",
    )

    one_hour = pd.read_csv(ROOT / "outputs/robustness_analysis/1h_robustness_comparison.csv")
    one_hour_mean = one_hour[one_hour["fold"] == "mean_3fold"]
    require(
        "one_hour_robustness_only_after_freeze",
        set(one_hour_mean["resolution_hours"]) == {0.5, 1.0}
        and set(one_hour_mean["model"]) == {"Ridge", "LSTM"}
        and len(one_hour_mean) == 4,
        one_hour_mean[["resolution_hours", "model", "rmse_mV"]].to_json(orient="records"),
    )

    frozen_text = (ROOT / "outputs/frozen_config/PHASE_N2_FROZEN_CONFIG.yaml").read_text(encoding="utf-8")
    frozen_patterns = {
        "status": r"(?m)^status:\s*PASS$",
        "resolution": r"(?m)^sampling_resolution_hours:\s*0\.5$",
        "features": r"(?m)^feature_group:\s*F2$",
        "lookback": r"(?m)^lookback_hours:\s*24$",
        "horizon": r"(?m)^primary_horizon_hours:\s*12$",
        "model": r"(?m)^model:\s*LSTM$",
        "test": r"(?m)^\s*ophrs2_start_h:\s*1819\.997738$",
    }
    require(
        "frozen_config_exact_core",
        all(re.search(pattern, frozen_text) for pattern in frozen_patterns.values()),
        f"patterns={list(frozen_patterns)}",
    )
    feature_block = re.search(r"exact_feature_names:\n(?P<body>(?:\s+- .+\n)+)", frozen_text)
    frozen_features = re.findall(r"^\s+-\s+(.+)$", feature_block.group("body"), flags=re.MULTILINE) if feature_block else []
    require("frozen_feature_count", len(frozen_features) == 14, f"feature_count={len(frozen_features)}")

    run_jsons = sorted((OUT / "model_runs").glob("*.json"))
    invalid_runs: list[str] = []
    for path in run_jsons:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "complete"
            or payload.get("fold") not in {"fold_1", "fold_2", "fold_3"}
            or float(payload.get("resolution_hours", -1)) not in {0.5, 1.0}
            or str(payload.get("stage", "")).lower() == "test"
            or "test" in str(payload.get("run_id", "")).lower()
        ):
            invalid_runs.append(path.name)
    require("model_run_manifests_valid", bool(run_jsons) and not invalid_runs, f"count={len(run_jsons)}, invalid={invalid_runs}")
    forbidden_prediction_files = [
        str(p.relative_to(ROOT))
        for p in OUT.rglob("*")
        if p.is_file() and "test" in p.name.lower() and "predict" in p.name.lower()
    ]
    require("no_test_prediction_artifacts", not forbidden_prediction_files, str(forbidden_prediction_files))

    compile_failures: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_failures.append(f"{path.name}: {exc}")
    require("all_python_scripts_compile", not compile_failures, str(compile_failures))

    report_text = (ROOT / "PHASE_N2_RESULTS.md").read_text(encoding="utf-8")
    visual_text = (ROOT / "outputs/audits/FIGURE_VISUAL_QA.md").read_text(encoding="utf-8")
    require("results_report_declares_PASS_GO", "PASS" in report_text and "GO" in report_text, "report contains PASS and GO")
    require("figure_visual_QA_pass", "**PASS**" in visual_text, "11 figures visually inspected")

    audit_csv = AUDIT_DIR / "phase_n2_verification_audit.csv"
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(CHECKS)

    excluded = {
        (AUDIT_DIR / "phase_n2_output_inventory.csv").resolve(),
        (AUDIT_DIR / "phase_n2_verification_manifest.json").resolve(),
    }
    inventory_rows: list[dict[str, object]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.resolve() in excluded:
            continue
        inventory_rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    inventory_csv = AUDIT_DIR / "phase_n2_output_inventory.csv"
    with inventory_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(inventory_rows)

    source_files = [
        ROOT.parent / "Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv",
        ROOT.parent / "Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_1h.csv",
    ]
    source_hashes = {
        str(path.resolve()): sha256(path) if path.is_file() else None for path in source_files
    }
    failed = [item for item in CHECKS if not item["passed"]]
    manifest = {
        "phase": "Phase N2",
        "status": "PASS" if not failed else "FAIL",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_performance_accessed": False,
        "test_predictions_constructed": False,
        "check_count": len(CHECKS),
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "inventory_file_count": len(inventory_rows),
        "phase_n1_source_sha256": source_hashes,
    }
    (AUDIT_DIR / "phase_n2_verification_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())


"""Final fail-closed integrity audit for Phase N1.

This script is read-only with respect to raw data.  It writes audit artifacts only
inside the Phase N1 output tree and exits non-zero if any required invariant fails.
It does not inspect Test target performance and does not train any model.
"""

from __future__ import annotations

import json
import py_compile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from phase_n1_common import PHASE_ROOT, load_config, sha256_file, write_csv, write_text


OUTPUTS = PHASE_ROOT / "outputs"


def source_role(path: Path) -> str:
    name = path.name.lower()
    full = str(path).lower()
    if "loadcycle" in name:
        return "durability_loadcycle_csv"
    if "polcurve" in name:
        return "polarization_csv"
    if "sensitivity" in full or "airair" in name or "anode" in name or "cathode" in name:
        return "bol_sensitivity_csv"
    if path.suffix.lower() == ".pdf":
        return "publication_pdf"
    return "other_core_csv"


def add_check(rows: list[dict[str, object]], name: str, passed: bool, evidence: str) -> None:
    rows.append({"check": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    config = load_config()
    raw_root = Path(config["raw_data_root"])
    audit_dir = OUTPUTS / "audits"

    # Phase N1 consumed all 36 CSV sources and the associated publication PDF.
    source_files = sorted(raw_root.rglob("*.csv")) + sorted(raw_root.rglob("*.pdf"))
    source_rows: list[dict[str, object]] = []
    for path in source_files:
        stat = path.stat()
        source_rows.append(
            {
                "source_role": source_role(path),
                "relative_path": path.relative_to(raw_root).as_posix(),
                "size_bytes": stat.st_size,
                "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": sha256_file(path),
                "read_only_usage": True,
            }
        )
    source_manifest = pd.DataFrame(source_rows)
    write_csv(source_manifest, audit_dir / "source_file_manifest.csv")

    checks: list[dict[str, object]] = []
    role_counts = source_manifest["source_role"].value_counts().to_dict()
    add_check(checks, "source_csv_count", len(list(raw_root.rglob("*.csv"))) == 36, "expected=36")
    add_check(
        checks,
        "loadcycle_file_count",
        int(role_counts.get("durability_loadcycle_csv", 0)) == 17,
        f"observed={role_counts.get('durability_loadcycle_csv', 0)}",
    )
    add_check(checks, "publication_pdf_count", len(list(raw_root.rglob("*.pdf"))) == 1, "expected=1")
    add_check(
        checks,
        "source_hashes_complete",
        source_manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all(),
        f"hashed_files={len(source_manifest)}",
    )

    required_paths = [
        PHASE_ROOT / "PHASE_N1_PLAN.md",
        PHASE_ROOT / "PHASE_N1_RESULTS.md",
        PHASE_ROOT / "configs" / "phase_n1_config.yaml",
        OUTPUTS / "audits" / "active_cell_channel_audit.csv",
        OUTPUTS / "audits" / "ACTIVE_CELL_MASK.md",
        OUTPUTS / "audits" / "FIGURE_VISUAL_QA.md",
        OUTPUTS / "causal_feature_tables" / "causal_features_30min.csv",
        OUTPUTS / "causal_feature_tables" / "causal_features_1h.csv",
        OUTPUTS / "causal_feature_tables" / "causal_feature_dictionary.csv",
        OUTPUTS / "causal_feature_tables" / "causal_feature_availability.csv",
        OUTPUTS / "split_manifest" / "split_manifest.csv",
        OUTPUTS / "split_manifest" / "FROZEN_DATA_SPLIT.md",
        OUTPUTS / "sampling_comparison" / "sampling_resolution_comparison.csv",
        OUTPUTS / "sampling_comparison" / "SAMPLING_RESOLUTION_DECISION.md",
        OUTPUTS / "soh_anchor_analysis" / "polarization_configuration_compatibility.md",
        OUTPUTS / "soh_anchor_analysis" / "polarization_soh_anchor_candidates.csv",
        OUTPUTS / "soh_anchor_analysis" / "SOH_ANCHOR_SELECTION.md",
        OUTPUTS / "soh_anchor_analysis" / "soh_reference_anchors.csv",
        OUTPUTS / "temporal_scale_analysis" / "load_cycle_periodicity_analysis.csv",
        OUTPUTS / "temporal_scale_analysis" / "temporal_dependency_analysis.csv",
        OUTPUTS / "temporal_scale_analysis" / "window_feasibility_summary.csv",
    ]
    missing = [str(path.relative_to(PHASE_ROOT)) for path in required_paths if not path.is_file()]
    add_check(checks, "required_core_outputs_exist", not missing, "missing=" + "|".join(missing))

    active = pd.read_csv(OUTPUTS / "audits" / "active_cell_channel_audit.csv")
    expected_active = set(config["active_cell_audit"]["expected_active_channels"])
    observed_active = set(active.loc[active["is_active"].astype(bool), "channel"])
    u6 = active.loc[active["channel"] == "U6 [V]"].iloc[0]
    add_check(
        checks,
        "active_cell_mask_exact",
        observed_active == expected_active,
        "observed=" + "|".join(sorted(observed_active)),
    )
    add_check(
        checks,
        "u6_nonphysical_confirmed",
        (not bool(u6["is_active"])) and float(u6["negative_ratio"]) > 0.99 and float(u6["loaded_median_V"]) < 0.0,
        f"negative_ratio={u6['negative_ratio']};loaded_median_V={u6['loaded_median_V']}",
    )

    feature_required = {
        "cell_voltage_mean",
        "cell_voltage_median",
        "cell_voltage_std",
        "cell_voltage_range",
        "cell_voltage_min",
        "cell_voltage_max",
        "actual_current_A_mean",
        "actual_current_A_std",
        "actual_current_density_A_cm2_mean",
        "coverage_ratio",
        "gap_before_flag",
        "gap_after_flag",
        "startup_flag",
        "shutdown_flag",
        "split",
        "temporal_segment_id",
    }
    expected_shapes = {"causal_features_30min.csv": (3843, 214, 0.5), "causal_features_1h.csv": (1927, 214, 1.0)}
    for name, (expected_rows, expected_cols, resolution) in expected_shapes.items():
        frame = pd.read_csv(OUTPUTS / "causal_feature_tables" / name)
        shape_ok = frame.shape == (expected_rows, expected_cols)
        keys_ok = frame["bin_start_ophrs2_h"].is_unique and frame["bin_start_ophrs2_h"].is_monotonic_increasing
        resolution_ok = np.allclose(frame["resolution_hours"].dropna().to_numpy(), resolution)
        columns_ok = feature_required.issubset(frame.columns)
        soh_columns_absent = not any("soh" in column.lower() for column in frame.columns)
        add_check(checks, f"{name}_shape", shape_ok, f"observed={frame.shape};expected={(expected_rows, expected_cols)}")
        add_check(checks, f"{name}_keys", keys_ok, "unique_and_strictly_ordered")
        add_check(checks, f"{name}_resolution", resolution_ok, f"expected={resolution}")
        add_check(checks, f"{name}_required_features", columns_ok, "missing=" + "|".join(sorted(feature_required - set(frame.columns))))
        add_check(checks, f"{name}_no_dense_soh", soh_columns_absent, "no SOH label columns permitted")

    split = pd.read_csv(OUTPUTS / "split_manifest" / "split_manifest.csv")
    observed_splits = split.set_index("split")
    split_ok = (
        np.isclose(float(observed_splits.loc["train", "ophrs2_start_h"]), 72.78869)
        and np.isclose(float(observed_splits.loc["train", "ophrs2_end_h"]), 1383.195476)
        and np.isclose(float(observed_splits.loc["validation", "ophrs2_start_h"]), 1383.195476)
        and np.isclose(float(observed_splits.loc["validation", "ophrs2_end_h"]), 1819.997738)
        and np.isclose(float(observed_splits.loc["test", "ophrs2_start_h"]), 1819.997738)
        and np.isclose(float(observed_splits.loc["test", "ophrs2_end_h"]), 2256.8)
    )
    add_check(checks, "frozen_split_boundaries", split_ok, "60/20/20 OpHrs2 blocked split")

    measurements = pd.read_csv(OUTPUTS / "soh_anchor_analysis" / "polarization_checkpoint_jstar_measurements.csv")
    test_rows = measurements[measurements["split"] == "test"]
    add_check(
        checks,
        "test_polarization_excluded_from_selection",
        (not test_rows.empty) and (~test_rows["selection_eligible"].astype(bool)).all(),
        f"test_rows={len(test_rows)};eligible={int(test_rows['selection_eligible'].astype(bool).sum())}",
    )
    candidates = pd.read_csv(OUTPUTS / "soh_anchor_analysis" / "polarization_soh_anchor_candidates.csv")
    primary = candidates.sort_values("selection_rank").iloc[0]
    add_check(
        checks,
        "primary_jstar_frozen",
        np.isclose(float(primary["target_jstar_A_cm2"]), 1.0) and primary["selection_scope"] == "train+validation_only",
        f"jstar={primary['target_jstar_A_cm2']};scope={primary['selection_scope']}",
    )
    anchors = pd.read_csv(OUTPUTS / "soh_anchor_analysis" / "soh_reference_anchors.csv")
    algebra_ok = np.allclose(anchors["soh_candidate_A"], anchors["hi_candidate_B"], equal_nan=True)
    add_check(checks, "soh_candidate_algebra", algebra_ok, "Candidate A equals Candidate B")
    add_check(
        checks,
        "sparse_soh_only",
        len(anchors) <= 12 and set(anchors["checkpoint"]).issubset({"BoL", "300h", "600h", "850h", "1150h", "1500h", "1500h_afterAirAirStart", "1750h", "2000h", "2300h", "2300h_afterAirAirStart"}),
        f"anchor_rows={len(anchors)}",
    )

    sampling = pd.read_csv(OUTPUTS / "sampling_comparison" / "sampling_resolution_comparison.csv")
    add_check(
        checks,
        "sampling_selection_scope",
        set(sampling["selection_scope"]) == {"train+validation_only"},
        "Test metrics excluded",
    )
    decision_text = (OUTPUTS / "sampling_comparison" / "SAMPLING_RESOLUTION_DECISION.md").read_text(encoding="utf-8")
    add_check(checks, "sampling_decision_primary_30min", "**0.5 h**" in decision_text, "1 h retained for sensitivity")

    window_log = json.loads((OUTPUTS / "audits" / "phase_n1_window_audit_log.json").read_text(encoding="utf-8"))
    windows = pd.read_csv(OUTPUTS / "temporal_scale_analysis" / "window_feasibility_summary.csv")
    add_check(checks, "window_audit_passed", bool(window_log.get("passed")), json.dumps(window_log, ensure_ascii=False))
    add_check(checks, "window_audit_no_causal_failures", int(windows["causal_audit_failures"].sum()) == 0, "sum=0")
    add_check(
        checks,
        "test_windows_count_only",
        windows.loc[windows["split"] == "test", "test_usage_policy"].str.contains("counts/structure only", regex=False).all(),
        "no Test target performance inspected",
    )

    required_figures = [
        "sampling_30min_vs_1h_signal_comparison.png",
        "sampling_30min_vs_1h_noise_comparison.png",
        "sampling_30min_vs_1h_coverage.png",
        "average_cell_voltage_vs_operating_hours.png",
        "matched_condition_voltage_vs_operating_hours.png",
        "cell_voltage_dispersion_vs_operating_hours.png",
        "load_profile_example_cycle.png",
        "polarization_curves_development_checkpoints.png",
        "candidate_jstar_voltage_vs_ageing.png",
        "candidate_jstar_monotonicity.png",
        "soh_reference_anchors.png",
        "load_cycle_periodicity.png",
        "voltage_acf.png",
        "current_density_acf.png",
        "window_counts_by_lookback_horizon.png",
    ]
    missing_or_small_figures = [
        name for name in required_figures if not (OUTPUTS / "figures" / name).is_file() or (OUTPUTS / "figures" / name).stat().st_size < 10_000
    ]
    add_check(checks, "required_figures_exist", not missing_or_small_figures, "missing_or_small=" + "|".join(missing_or_small_figures))

    scripts = sorted((PHASE_ROOT / "scripts").glob("phase_n1_*.py"))
    compile_errors: list[str] = []
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_errors.append(f"{script.name}:{exc}")
    add_check(checks, "phase_scripts_compile", not compile_errors, "|".join(compile_errors))

    # Scan executable pipeline code for actual model-framework imports and
    # explicitly forbidden preprocessing operations.  Prohibition phrases may
    # legitimately appear in generated audit prose and are therefore not tokens.
    forbidden_tokens = ["import torch", "from torch", "optuna", ".bfill(", "center=True"]
    token_hits: list[str] = []
    audited_pipeline_scripts = [script for script in scripts if script.name != "phase_n1_verify.py"]
    for script in audited_pipeline_scripts:
        content = script.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in content:
                token_hits.append(f"{script.name}:{token}")
    add_check(checks, "no_model_or_forbidden_preprocessing_tokens", not token_hits, "|".join(token_hits))
    add_check(
        checks,
        "strict_rul_ground_truth_disabled",
        config["forecasting"].get("strict_rul_ground_truth") is False,
        "No verified EOL/failure threshold",
    )

    audit = pd.DataFrame(checks)
    write_csv(audit, audit_dir / "phase_n1_integrity_audit.csv")
    passed = bool(audit["passed"].all())

    # Inventory all material outputs after the audit exists.  It intentionally
    # excludes itself to avoid a recursive hash and excludes Python bytecode.
    inventory_rows: list[dict[str, object]] = []
    for path in sorted(PHASE_ROOT.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.name in {"phase_n1_run_manifest.json", "core_output_inventory.csv"}
        ):
            continue
        inventory_rows.append(
            {
                "relative_path": path.relative_to(PHASE_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    write_csv(inventory, audit_dir / "core_output_inventory.csv")

    run_manifest = {
        "phase": config["phase_name"],
        "status": "PASS" if passed else "FAIL",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file_count": len(source_manifest),
        "output_inventory_count": len(inventory),
        "integrity_check_count": len(audit),
        "failed_checks": audit.loc[~audit["passed"], "check"].tolist(),
        "test_policy": "counts and structural integrity only; no model/feature/target/hyperparameter selection",
        "models_trained": False,
    }
    write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), audit_dir / "phase_n1_run_manifest.json")

    print(json.dumps(run_manifest, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

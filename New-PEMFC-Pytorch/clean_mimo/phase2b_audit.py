"""Fail-closed audit for Phase-2b development-only expanding validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from phase2b_experiment import FEATURE_LADDER, Phase2bFoldData


def audit_phase2b(data: Phase2bFoldData, output_dir: Path) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_train_end = int(data.total_hourly_rows * data.fold.train_end_fraction)
    expected_validation_end = int(data.total_hourly_rows * data.fold.validation_end_fraction)
    expected_development_end = int(data.total_hourly_rows * 0.60)
    check(
        "frozen-expanding-fold-boundaries",
        data.split_rows == {"train": (0, expected_train_end), "validation": (expected_train_end, expected_validation_end)},
        f"Expected train [0,{expected_train_end}) and validation [{expected_train_end},{expected_validation_end}).",
    )
    check(
        "development-prefix-ends-at-60-percent",
        data.development_end_row == expected_development_end and len(data.hourly_development) == expected_development_end,
        f"Only hourly rows [0,{expected_development_end}) are available to this fold.",
    )
    check(
        "no-final-test-split-constructed",
        set(data.splits) == {"train", "validation"} and expected_validation_end <= data.development_end_row,
        "This object has only train/validation windows; the final 40% is absent from its split dictionary.",
    )
    hourly_causal = bool(
        np.all(data.hourly_development["raw_time_max"].to_numpy(float) <= data.hourly_development["origin_time_h"].to_numpy(float) + 1e-9)
    )
    check("hourly-aggregation-is-causal", hourly_causal, "Every aggregated source sample occurs at or before its origin timestamp.")

    split_ok, timestamp_ok, development_ok = True, True, True
    for split_name, split in data.splits.items():
        start, end = data.split_rows[split_name]
        history_min = split.origins - data.config.history_hours + 1
        target_max = split.origins + data.config.horizon_hours
        split_ok &= bool(np.all(history_min >= start) and np.all(target_max < end))
        timestamp_ok &= bool(np.all(split.feature_times[:, -1] < split.target_times[:, 0]))
        development_ok &= bool(np.all(target_max < data.development_end_row))
    check("windows-do-not-cross-fold-boundaries", split_ok, "Every 15-hour history and four targets remain in its own fold partition.")
    check("input-timestamps-precede-targets", timestamp_ok, "All input timestamps are strictly before h=1 targets.")
    check("no-window-enters-final-test-region", development_ok, "No Phase-2b target index reaches the final 40% test region.")

    utot_index = data.all_feature_names.index(data.config.target_name)
    anchors_ok, residual_ok = True, True
    for split_name, split in data.splits.items():
        expected_anchor = split.X_raw[:, -1, utot_index, None]
        anchors_ok &= bool(np.array_equal(data.anchors[split_name], expected_anchor))
        residual_ok &= bool(np.array_equal(data.residual_labels[split_name], split.y_raw - expected_anchor))
    check("residual-anchor-is-observed-origin-utot", anchors_ok, "Anchor is the final legal historical Utot, never a target-time voltage.")
    check("residual-label-is-future-minus-origin", residual_ok, "Every label is Utot(t+h)-Utot(t), h=1..4.")

    train_inputs = data.inputs_raw["train"].reshape(-1, len(data.feature_names)).astype(np.float64)
    train_residuals = data.residual_labels["train"].reshape(-1, 1).astype(np.float64)
    check(
        "input-scaler-fits-current-fold-training-windows-only",
        bool(np.allclose(data.X_scaler.mean_, train_inputs.mean(axis=0))),
        "Input scaler mean equals only this fold's flattened training-window mean.",
    )
    check(
        "residual-scaler-fits-current-fold-training-labels-only",
        bool(np.allclose(data.residual_scaler.mean_, train_residuals.mean(axis=0))),
        "Residual scaler mean equals only this fold's flattened training residual mean.",
    )
    expected_names = list(FEATURE_LADDER[data.feature_set])
    forbidden_names = [name for name in data.feature_names if any(token in name.lower() for token in ("soh", "delta", "future", "label"))]
    check("feature-set-is-predeclared-and-not-label-derived", data.feature_names == expected_names and not forbidden_names, f"Features: {data.feature_names}")

    clean_dir = Path(__file__).resolve().parent
    expected_root = (clean_dir / "outputs_phase2b").resolve()
    output_ok = output_dir.resolve().is_relative_to(expected_root)
    phase1_root = (clean_dir / "outputs").resolve()
    phase2_root = (clean_dir / "outputs_phase2").resolve()
    output_ok &= not output_dir.resolve().is_relative_to(phase1_root) and not output_dir.resolve().is_relative_to(phase2_root)
    check("phase2b-output-is-isolated", output_ok, f"Output must be under {expected_root}, outside Phase-1/Phase-2 outputs.")

    source_files = ("phase2b_experiment.py", "run_phase2b.py", "phase2b_report.py")
    source = "\n".join((clean_dir / name).read_text(encoding="utf-8") for name in source_files if (clean_dir / name).exists()).lower()
    forbidden_source = [token for token in ("savgol_filter", ".bfill(", "delta_utot", "soh =", "optuna", "transformer", "mclstm", "adaptive grn", "mamba", "attention", "test_predictions") if token in source]
    check("phase2b-source-excludes-forbidden-methods-and-test-evaluation", not forbidden_source, f"Forbidden source-token scan found: {forbidden_source}")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "phase": "2b", "fc": data.fc, "fold": data.fold.name, "feature_set": data.feature_set, "checks": checks}


def write_phase2b_audit(data: Phase2bFoldData, output_dir: Path) -> Dict[str, object]:
    result = audit_phase2b(data, output_dir)
    (output_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# Phase-2b leakage audit — {data.fc}, {data.fold.name}, {data.feature_set}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}" for item in result["checks"])
    (output_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Phase-2b leakage audit failed: {output_dir / 'leakage_audit.md'}")
    return result

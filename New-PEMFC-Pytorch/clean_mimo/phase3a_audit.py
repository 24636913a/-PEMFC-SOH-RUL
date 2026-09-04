"""Fail-closed audit for Phase-3A causal decomposition tasks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from phase3a_experiment import LOAD_FEATURES, TAUS, Phase3FoldData, TaskData, alpha_for_tau, causal_ewma


def audit_phase3a(data: Phase3FoldData, task: TaskData, output_dir: Path) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_train_end = int(data.total_hourly_rows * data.fold.train_end_fraction)
    expected_validation_end = int(data.total_hourly_rows * data.fold.validation_end_fraction)
    expected_development_end = int(data.total_hourly_rows * 0.60)
    check("predeclared-fold-boundaries", data.split_rows == {"train": (0, expected_train_end), "validation": (expected_train_end, expected_validation_end)}, "Fold boundaries match the frozen 35/40, 45/50, or 55/60 protocol.")
    check("development-prefix-only", data.development_end_row == expected_development_end and set(data.splits) == {"train", "validation"}, "Only rows before 60% and no test split are available.")
    causal_hourly = bool(np.all(data.hourly_development["raw_time_max"].to_numpy(float) <= data.hourly_development["origin_time_h"].to_numpy(float) + 1e-9))
    check("hourly-aggregation-is-causal", causal_hourly, "Hourly aggregation is causal before EWMA decomposition.")
    raw_u = data.hourly_development[data.config.target_name].to_numpy(dtype=np.float32)
    expected_slow = causal_ewma(raw_u, alpha_for_tau(data.tau_hours))
    check("ewma-is-one-sided-and-predeclared", data.tau_hours in TAUS and np.array_equal(data.slow_component, expected_slow), "D[t] equals the one-sided recurrence from U[0] through U[t] only.")
    check("transient-is-u-minus-d", bool(np.array_equal(data.transient_component, raw_u - data.slow_component)), "R[t] is exactly observed U[t]-D[t].")

    boundary_ok, timestamp_ok, development_ok = True, True, True
    for split_name, split in data.splits.items():
        start, end = data.split_rows[split_name]
        boundary_ok &= bool(np.all(split.origins - 14 >= start) and np.all(split.origins + 4 < end))
        timestamp_ok &= bool(np.all(split.feature_times[:, -1] < split.target_times[:, 0]))
        development_ok &= bool(np.all(split.origins + 4 < data.development_end_row))
    check("windows-do-not-cross-fold-boundaries", boundary_ok, "Every N=15/H=4 window is split-contained.")
    check("all-input-times-precede-targets", timestamp_ok, "Latest feature time is before h=1 target time.")
    check("no-final-test-row-in-task", development_ok, "No task target reaches the final 40% test partition.")

    train_x = task.inputs_raw["train"].reshape(-1, len(task.feature_names)).astype(np.float64)
    train_y = task.labels_raw["train"].reshape(-1, 1).astype(np.float64)
    check("input-scaler-is-current-fold-train-only", bool(np.allclose(task.X_scaler.mean_, train_x.mean(axis=0))), "Input scaler mean equals current-fold train inputs only.")
    check("target-scaler-is-current-fold-train-only", bool(np.allclose(task.y_scaler.mean_, train_y.mean(axis=0))), "Target scaler mean equals current-fold labels only.")
    t1_ok = task.name == "T1_slow_residual_mimo_lstm" and task.feature_names == ["D_causal_ewma"]
    t2a_ok = task.name == "T2_R_history_only_mimo_lstm" and task.feature_names == ["R_causal_ewma"]
    t2b_ok = task.name == "T2_R_history_plus_load_mimo_lstm" and task.feature_names == ["R_causal_ewma", *LOAD_FEATURES]
    check("task-features-are-predeclared", t1_ok or t2a_ok or t2b_ok, f"Task {task.name} uses {task.feature_names}")

    clean_dir = Path(__file__).resolve().parent
    expected_root = (clean_dir / "outputs_phase3a").resolve()
    output_ok = output_dir.resolve().is_relative_to(expected_root)
    for legacy in (clean_dir / "outputs", clean_dir / "outputs_phase2", clean_dir / "outputs_phase2b", clean_dir / "outputs_phase2b_final_test"):
        output_ok &= not output_dir.resolve().is_relative_to(legacy.resolve())
    check("phase3a-output-is-isolated", output_ok, f"Output must be under {expected_root} only.")
    source_files = ("phase3a_experiment.py", "run_phase3a.py", "phase3a_report.py")
    source = "\n".join((clean_dir / name).read_text(encoding="utf-8") for name in source_files if (clean_dir / name).exists()).lower()
    forbidden = [token for token in ("savgol_filter", ".bfill(", "delta_utot", "soh =", "optuna", "transformer", "mclstm", "adaptive grn", "mamba", "attention", "full_26", "test_predictions") if token in source]
    check("source-excludes-forbidden-methods-and-test-evaluation", not forbidden, f"Forbidden source-token scan found: {forbidden}")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "phase": "3A", "fc": data.fc, "fold": data.fold.name, "tau_hours": data.tau_hours, "task": task.name, "checks": checks}


def write_phase3a_audit(data: Phase3FoldData, task: TaskData, output_dir: Path) -> Dict[str, object]:
    result = audit_phase3a(data, task, output_dir)
    (output_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# Phase-3A leakage audit — {data.fold.name}, tau={data.tau_hours} h, {task.name}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}" for item in result["checks"])
    (output_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Phase-3A audit failed: {output_dir / 'leakage_audit.md'}")
    return result

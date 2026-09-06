"""Fail-closed audit for Phase-3B development-only horizon diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from phase3b_experiment import HORIZONS, HorizonFoldData


def audit_phase3b(data: HorizonFoldData, output_dir: Path) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    train_end = int(data.total_hourly_rows * data.fold.train_end_fraction)
    validation_end = int(data.total_hourly_rows * data.fold.validation_end_fraction)
    development_end = int(data.total_hourly_rows * 0.60)
    check("predeclared-30-40-50-60-fold-boundaries", data.split_rows == {"train": (0, train_end), "validation": (train_end, validation_end)}, "Fold matches 0–30/30–40, 0–40/40–50, or 0–50/50–60%.")
    check("development-prefix-only", data.development_end_row == development_end and set(data.splits) == {"train", "validation"}, "No final-test split is constructed.")
    check("frozen-n15-horizon", data.config.history_hours == 15 and data.config.horizon_hours in HORIZONS, f"N=15 and H in {HORIZONS}.")
    causal_hourly = bool(np.all(data.hourly_development["raw_time_max"].to_numpy(float) <= data.hourly_development["origin_time_h"].to_numpy(float) + 1e-9))
    check("hourly-aggregation-is-causal", causal_hourly, "Every source sample occurs at or before its hourly origin timestamp.")

    boundary_ok, timestamp_ok, development_ok, anchor_ok, residual_ok = True, True, True, True, True
    for name, split in data.splits.items():
        start, end = data.split_rows[name]
        boundary_ok &= bool(np.all(split.origins - 14 >= start) and np.all(split.origins + data.config.horizon_hours < end))
        timestamp_ok &= bool(np.all(split.feature_times[:, -1] < split.target_times[:, 0]))
        development_ok &= bool(np.all(split.origins + data.config.horizon_hours < data.development_end_row))
        expected_anchor = split.X_raw[:, -1, 0, None]
        anchor_ok &= bool(np.array_equal(data.anchors[name], expected_anchor))
        residual_ok &= bool(np.array_equal(data.residual_labels[name], split.y_raw - expected_anchor))
    check("windows-are-fold-contained", boundary_ok, "All N=15/H windows remain in their chronological fold.")
    check("inputs-precede-targets", timestamp_ok, "All final input timestamps precede h=1 target timestamps.")
    check("no-final-test-row-in-windows", development_ok, "No model input or target reaches the final 40% partition.")
    check("residual-anchor-is-legal-origin-utot", anchor_ok, "Anchor is exactly the final observed Utot history value.")
    check("residual-label-is-future-minus-origin", residual_ok, "All labels equal Utot(t+h)-Utot(t).")
    train_x = data.inputs_raw["train"].reshape(-1, 1).astype(np.float64)
    train_y = data.residual_labels["train"].reshape(-1, 1).astype(np.float64)
    check("utot-scaler-fits-current-fold-train-only", bool(np.allclose(data.X_scaler.mean_, train_x.mean(axis=0))), "Input scaler fit scope is current fold train only.")
    check("residual-scaler-fits-current-fold-train-only", bool(np.allclose(data.residual_scaler.mean_, train_y.mean(axis=0))), "Target scaler fit scope is current fold train residual only.")

    clean_dir = Path(__file__).resolve().parent
    expected_root = (clean_dir / "outputs_phase3b").resolve()
    output_ok = output_dir.resolve().is_relative_to(expected_root)
    for previous in (clean_dir / "outputs", clean_dir / "outputs_phase2", clean_dir / "outputs_phase2b", clean_dir / "outputs_phase2b_final_test", clean_dir / "outputs_phase3a"):
        output_ok &= not output_dir.resolve().is_relative_to(previous.resolve())
    check("phase3b-output-is-isolated", output_ok, f"Output must be under {expected_root} only.")
    source_files = ("phase3b_experiment.py", "run_phase3b.py", "phase3b_report.py")
    source = "\n".join((clean_dir / name).read_text(encoding="utf-8") for name in source_files if (clean_dir / name).exists()).lower()
    forbidden = [token for token in ("savgol_filter", ".bfill(", "delta_utot", "soh =", "optuna", "transformer", "mclstm", "adaptive grn", "mamba", "attention", "full_26", "test_predictions") if token in source]
    check("source-excludes-forbidden-methods-and-test-evaluation", not forbidden, f"Forbidden source-token scan found: {forbidden}")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "phase": "3B", "fc": data.fc, "fold": data.fold.name, "H": data.config.horizon_hours, "checks": checks}


def write_phase3b_audit(data: HorizonFoldData, output_dir: Path) -> Dict[str, object]:
    result = audit_phase3b(data, output_dir)
    (output_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# Phase-3B leakage audit — {data.fc}, H={data.config.horizon_hours}, {data.fold.name}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}" for item in result["checks"])
    (output_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Phase-3B audit failed: {output_dir / 'leakage_audit.md'}")
    return result

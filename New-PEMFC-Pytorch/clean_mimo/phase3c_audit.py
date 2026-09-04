"""Fail-closed leakage audit for the development-only Phase-3C N sweep."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from phase3c_experiment import HISTORY_LENGTHS, HORIZON_HOURS, HistoryFoldData


def audit_phase3c(data: HistoryFoldData, output_dir: Path) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    train_end = int(data.total_hourly_rows * data.fold.train_end_fraction)
    validation_end = int(data.total_hourly_rows * data.fold.validation_end_fraction)
    development_end = int(data.total_hourly_rows * 0.60)
    check("predeclared-fold-boundaries", data.split_rows == {"train": (0, train_end), "validation": (train_end, validation_end)}, "Fold matches frozen 30/40, 40/50, or 50/60 boundaries.")
    check("development-prefix-only", data.development_end_row == development_end and set(data.splits) == {"train", "validation"}, "No final-test split is created.")
    check("only-n-varies-at-h24", data.config.history_hours in HISTORY_LENGTHS and data.config.horizon_hours == HORIZON_HOURS, f"N in {HISTORY_LENGTHS}; H=24.")
    hourly_causal = bool(np.all(data.hourly_development["raw_time_max"].to_numpy(float) <= data.hourly_development["origin_time_h"].to_numpy(float) + 1e-9))
    check("hourly-aggregation-is-causal", hourly_causal, "Hourly sources do not exceed their origin timestamps.")

    boundary_ok, timestamp_ok, development_ok, anchor_ok, residual_ok = True, True, True, True, True
    for name, split in data.splits.items():
        start, end = data.split_rows[name]
        boundary_ok &= bool(np.all(split.origins - data.config.history_hours + 1 >= start) and np.all(split.origins + HORIZON_HOURS < end))
        timestamp_ok &= bool(np.all(split.feature_times[:, -1] < split.target_times[:, 0]))
        development_ok &= bool(np.all(split.origins + HORIZON_HOURS < data.development_end_row))
        expected_anchor = split.X_raw[:, -1, 0, None]
        anchor_ok &= bool(np.array_equal(data.anchors[name], expected_anchor))
        residual_ok &= bool(np.array_equal(data.residual_labels[name], split.y_raw - expected_anchor))
    check("windows-are-split-contained", boundary_ok, "Every history/target window remains within its fold.")
    check("input-times-precede-targets", timestamp_ok, "Latest observed input precedes h=1 target.")
    check("no-final-test-row-in-windows", development_ok, "No input or target reaches the final 40% partition.")
    check("origin-anchor-is-observed-utot", anchor_ok, "Anchor is final legal Utot history value.")
    check("residual-label-is-future-minus-origin", residual_ok, "Labels equal Utot(t+h)-Utot(t).")
    train_x = data.inputs_raw["train"].reshape(-1, 1).astype(np.float64)
    train_y = data.residual_labels["train"].reshape(-1, 1).astype(np.float64)
    check("input-scaler-fits-current-fold-train-only", bool(np.allclose(data.X_scaler.mean_, train_x.mean(axis=0))), "Input scaler fit uses current-fold train windows only.")
    check("residual-scaler-fits-current-fold-train-only", bool(np.allclose(data.residual_scaler.mean_, train_y.mean(axis=0))), "Residual scaler fit uses current-fold train labels only.")

    clean_dir = Path(__file__).resolve().parent
    expected_root = (clean_dir / "outputs_phase3c").resolve()
    output_ok = output_dir.resolve().is_relative_to(expected_root)
    for prior in ("outputs", "outputs_phase2", "outputs_phase2b", "outputs_phase2b_final_test", "outputs_phase3a", "outputs_phase3b"):
        output_ok &= not output_dir.resolve().is_relative_to((clean_dir / prior).resolve())
    check("phase3c-output-is-isolated", output_ok, f"Output must be under {expected_root} only.")
    source_files = ("phase3c_experiment.py", "run_phase3c.py", "phase3c_report.py")
    source = "\n".join((clean_dir / name).read_text(encoding="utf-8") for name in source_files if (clean_dir / name).exists()).lower()
    forbidden = [token for token in ("savgol_filter", ".bfill(", "delta_utot", "soh =", "optuna", "transformer", "mclstm", "adaptive grn", "mamba", "attention", "relative_history", "test_predictions") if token in source]
    check("source-excludes-forbidden-methods-and-test-evaluation", not forbidden, f"Forbidden source-token scan found: {forbidden}")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "phase": "3C", "fc": data.fc, "N": data.config.history_hours, "fold": data.fold.name, "checks": checks}


def write_phase3c_audit(data: HistoryFoldData, output_dir: Path) -> Dict[str, object]:
    result = audit_phase3c(data, output_dir)
    (output_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# Phase-3C leakage audit — {data.fc}, N={data.config.history_hours}, {data.fold.name}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}" for item in result["checks"])
    (output_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Phase-3C audit failed: {output_dir / 'leakage_audit.md'}")
    return result

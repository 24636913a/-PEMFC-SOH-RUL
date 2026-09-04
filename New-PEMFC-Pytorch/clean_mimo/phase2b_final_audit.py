"""Fail-closed audit for the one authorized frozen R3 final-test run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from leakage_audit import audit as phase1_audit
from phase2_experiment import Phase2Data


R3_FEATURES = [
    "Utot_mean", "U1_mean", "U2_mean", "U3_mean", "U4_mean", "U5_mean",
    "J_mean", "I_mean", "I_std", "I_p2p",
]


def audit_frozen_r3(data: Phase2Data, output_dir: Path, completion_marker: Path) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    base = phase1_audit(data.base)
    check("phase1-causal-pipeline-remains-pass", base["passed"], "Original hourly aggregation, split isolation, timestamps, and scaler rules must pass.")
    check("feature-set-is-frozen-r3-state-load", data.feature_set == "state_load" and data.feature_names == R3_FEATURES, f"Features: {data.feature_names}")
    check("input-mode-is-frozen-absolute-history", data.input_mode == "absolute_inputs", "No relative-input change is permitted in this final run.")
    check("protocol-is-frozen-n15-h4", data.base.config.history_hours == 15 and data.base.config.horizon_hours == 4, "History and direct MIMO horizon are fixed.")
    check("protocol-is-frozen-50-10-40", data.base.split_rows == {"train": (0, int(len(data.base.hourly) * 0.5)), "validation": (int(len(data.base.hourly) * 0.5), int(len(data.base.hourly) * 0.6)), "test": (int(len(data.base.hourly) * 0.6), len(data.base.hourly))}, "The original final 40% test partition is required.")

    utot_index = data.base.feature_names.index(data.base.config.target_name)
    anchor_ok, residual_ok, boundaries_ok = True, True, True
    for split_name, split in data.base.splits.items():
        expected_anchor = split.X_raw[:, -1, utot_index, None]
        anchor_ok &= bool(np.array_equal(data.anchors[split_name], expected_anchor))
        residual_ok &= bool(np.array_equal(data.residual_labels[split_name], split.y_raw - expected_anchor))
        start, end = data.base.split_rows[split_name]
        boundaries_ok &= bool(np.all(split.origins - 14 >= start) and np.all(split.origins + 4 < end))
    check("residual-anchor-is-observed-origin-utot", anchor_ok, "Anchor equals the last legal historical Utot in every split.")
    check("residual-label-is-future-minus-origin", residual_ok, "Label equals Utot(t+h)-Utot(t), h=1..4.")
    check("final-test-windows-remain-split-contained", boundaries_ok, "No history or target row crosses train/validation/test boundaries.")

    train_inputs = data.inputs_repr["train"].reshape(-1, len(data.feature_names)).astype(np.float64)
    train_residuals = data.residual_labels["train"].reshape(-1, 1).astype(np.float64)
    check("input-scaler-fits-training-windows-only", bool(np.allclose(data.X_scaler.mean_, train_inputs.mean(axis=0))), "Input scaler fits flattened train representations only.")
    check("residual-scaler-fits-training-residuals-only", bool(np.allclose(data.residual_target_scaler.mean_, train_residuals.mean(axis=0))), "Residual scaler fits flattened training residual labels only.")

    clean_dir = Path(__file__).resolve().parent
    expected_root = (clean_dir / "outputs_phase2b_final_test").resolve()
    output_ok = output_dir.resolve().is_relative_to(expected_root)
    output_ok &= not output_dir.resolve().is_relative_to((clean_dir / "outputs").resolve())
    output_ok &= not output_dir.resolve().is_relative_to((clean_dir / "outputs_phase2").resolve())
    output_ok &= not output_dir.resolve().is_relative_to((clean_dir / "outputs_phase2b").resolve())
    check("final-test-output-is-isolated", output_ok, f"Output must be under {expected_root} only.")
    check("single-run-completion-marker-is-absent-before-run", not completion_marker.exists(), f"Existing marker blocks a repeat evaluation: {completion_marker}")

    source_files = ("phase2b_final_test.py", "phase2b_final_report.py")
    source = "\n".join((clean_dir / name).read_text(encoding="utf-8") for name in source_files if (clean_dir / name).exists()).lower()
    forbidden = [token for token in ("savgol_filter", ".bfill(", "delta_utot", "soh =", "optuna", "transformer", "mclstm", "adaptive grn", "mamba", "attention", "top-k", "ensemble") if token in source]
    check("final-test-source-excludes-forbidden-methods", not forbidden, f"Forbidden source-token scan found: {forbidden}")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "phase": "2b_final_test", "fc": data.base.fc, "checks": checks}


def write_frozen_r3_audit(data: Phase2Data, output_dir: Path, completion_marker: Path) -> Dict[str, object]:
    result = audit_frozen_r3(data, output_dir, completion_marker)
    (output_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Frozen R3 final-test leakage audit", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}" for item in result["checks"])
    (output_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Frozen R3 final-test audit failed: {output_dir / 'leakage_audit.md'}")
    return result

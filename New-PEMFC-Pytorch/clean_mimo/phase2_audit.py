"""Fail-closed leakage and output-isolation audit for Phase 2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from leakage_audit import audit as phase1_audit
from phase2_experiment import Phase2Data, VOLTAGE_FEATURES


def audit_phase2(data: Phase2Data, output_dir: Path) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # Preserve every pre-existing Phase-1 causal guarantee.
    base_result = phase1_audit(data.base)
    check("phase1-causal-audit-remains-pass", base_result["passed"], "The original hourly/timestamp/window/scaler audit must pass.")

    utot_idx = data.base.feature_names.index(data.base.config.target_name)
    residual_definition_ok = True
    anchors_ok = True
    for split_name, split in data.base.splits.items():
        expected_anchor = split.X_raw[:, -1, utot_idx, None]
        anchors_ok &= bool(np.array_equal(data.anchors[split_name], expected_anchor))
        expected_residual = split.y_raw - expected_anchor
        residual_definition_ok &= bool(np.array_equal(data.residual_labels[split_name], expected_residual))
    check(
        "residual-anchor-is-observed-origin-utot",
        anchors_ok,
        "Every anchor equals the last legal historical Utot_mean, not a target-time Utot.",
    )
    check(
        "residual-label-is-future-minus-origin",
        residual_definition_ok,
        "Every residual label is exactly Utot(t+h) - observed Utot(t), h=1..4.",
    )

    train_residual = data.residual_labels["train"].reshape(-1, 1).astype(np.float64)
    residual_scaler_ok = bool(np.allclose(data.residual_target_scaler.mean_, train_residual.mean(axis=0)))
    check(
        "residual-scaler-fits-training-residuals-only",
        residual_scaler_ok,
        "Residual StandardScaler mean equals the flattened training residual mean only.",
    )
    train_absolute = data.base.splits["train"].y_raw.reshape(-1, 1).astype(np.float64)
    absolute_scaler_ok = bool(np.allclose(data.absolute_target_scaler.mean_, train_absolute.mean(axis=0)))
    check(
        "absolute-scaler-fits-training-labels-only",
        absolute_scaler_ok,
        "Absolute target StandardScaler mean equals the flattened training-label mean only.",
    )
    train_input = data.inputs_repr["train"].reshape(-1, len(data.feature_names)).astype(np.float64)
    input_scaler_ok = bool(np.allclose(data.X_scaler.mean_, train_input.mean(axis=0)))
    check(
        "input-scaler-fits-training-representations-only",
        input_scaler_ok,
        "Input StandardScaler mean equals the flattened training representation mean only.",
    )

    no_label_features = not any(any(token in name.lower() for token in ("soh", "delta", "future", "label", "target")) for name in data.feature_names)
    check(
        "feature-set-has-no-label-derived-feature",
        no_label_features,
        f"Declared input names are limited to predeclared causal sensors: {data.feature_names}",
    )

    relative_ok = True
    if data.input_mode == "relative_voltage_inputs":
        for split_name, representation in data.inputs_repr.items():
            raw = data.inputs_raw[split_name]
            for feature_idx in data.voltage_feature_indices:
                expected = raw[:, :, feature_idx] - raw[:, -1, feature_idx, None]
                relative_ok &= bool(np.array_equal(representation[:, :, feature_idx], expected))
    check(
        "relative-voltage-reference-is-final-history-row",
        relative_ok,
        "Relative mode uses only the final historical row of each same-split window; absolute mode has no relative transform.",
    )

    clean_dir = Path(__file__).resolve().parent
    required_root = (clean_dir / "outputs_phase2").resolve()
    output_ok = output_dir.resolve().is_relative_to(required_root) and not output_dir.resolve().is_relative_to((clean_dir / "outputs").resolve())
    check(
        "phase2-output-is-isolated-from-phase1",
        output_ok,
        f"Output must be under {required_root}, never Phase-1 outputs/.",
    )

    # Search Phase-2 implementation sources, excluding this audit's own literal search terms.
    source_files = ("phase2_experiment.py", "run_phase2.py", "fc2_event_diagnostics.py", "phase2_report.py")
    source = "\n".join((clean_dir / filename).read_text(encoding="utf-8") for filename in source_files if (clean_dir / filename).exists()).lower()
    forbidden_source = [token for token in ("savgol_filter", ".bfill(", "delta_utot", "soh =", "optuna", "transformer", "mclstm", "adaptive grn", "mamba", "attention") if token in source]
    check(
        "phase2-source-excludes-forbidden-methods-and-noncausal-preprocessing",
        not forbidden_source,
        f"Forbidden source-token scan found: {forbidden_source}",
    )
    passed = all(check_item["passed"] for check_item in checks)
    return {"passed": passed, "fc": data.base.fc, "phase": 2, "checks": checks}


def write_phase2_audit(data: Phase2Data, output_dir: Path) -> Dict[str, object]:
    result = audit_phase2(data, output_dir)
    (output_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# Phase-2 leakage audit — {data.base.fc}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    for item in result["checks"]:
        lines.append(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}")
    (output_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Phase-2 leakage audit failed. See {output_dir / 'leakage_audit.md'}")
    return result

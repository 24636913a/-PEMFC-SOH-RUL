"""Fail-closed causal, representation, and paired-origin audit for Phase 3D."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from phase3d_experiment import HISTORY_HOURS, HORIZON_HOURS, MAX_HISTORY_HOURS, REPRESENTATIONS, RepresentationFoldData


def _record(checks: List[Dict[str, object]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _isolated_output(output_path: Path) -> bool:
    clean_dir = Path(__file__).resolve().parent
    expected = (clean_dir / "outputs_phase3d").resolve()
    resolved = output_path.resolve()
    if not resolved.is_relative_to(expected):
        return False
    previous = (
        "outputs", "outputs_phase2", "outputs_phase2b", "outputs_phase2b_final_test",
        "outputs_phase3a", "outputs_phase3b", "outputs_phase3c", "outputs_phase3cr",
    )
    return all(not resolved.is_relative_to((clean_dir / root).resolve()) for root in previous)


def _source_is_clean() -> tuple[bool, list[str]]:
    clean_dir = Path(__file__).resolve().parent
    files = ("phase3d_experiment.py", "phase3d_report.py", "run_phase3d.py")
    source = "\n".join((clean_dir / name).read_text(encoding="utf-8") for name in files if (clean_dir / name).exists()).lower()
    forbidden = [
        token for token in (
            "savgol_filter", ".bfill(", "delta_utot", "soh =", "optuna", "transformer",
            "mclstm", "adaptive grn", "mamba", "attention", "test_predictions", "final_test_predictions",
        ) if token in source
    ]
    return not forbidden, forbidden


def audit_phase3d(data: RepresentationFoldData, output_dir: Path) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []
    train_end = int(data.total_hourly_rows * data.fold.train_end_fraction)
    validation_end = int(data.total_hourly_rows * data.fold.validation_end_fraction)
    development_end = int(data.total_hourly_rows * 0.60)
    _record(checks, "predeclared-fold-boundaries", data.split_rows == {"train": (0, train_end), "validation": (train_end, validation_end)}, "Fold remains frozen at 30/40, 40/50, or 50/60 development boundaries.")
    _record(checks, "frozen-h24-n15", data.config.history_hours == HISTORY_HOURS and data.config.horizon_hours == HORIZON_HOURS, "N=15 and H=24 are unchanged.")
    _record(checks, "nmax48-origin-schedule-retained", MAX_HISTORY_HOURS == 48, "Only the Phase-3C-R Nmax=48 legal origin schedule is retained; Nmax is not an input feature.")
    _record(checks, "development-prefix-only", data.development_end_row == development_end and set(data.splits) == {"train", "validation"}, "No final 40% partition is constructed as a split.")
    hourly_causal = bool(np.all(data.hourly_development["raw_time_max"].to_numpy(float) <= data.hourly_development["origin_time_h"].to_numpy(float) + 1e-9))
    _record(checks, "hourly-aggregation-is-causal", hourly_causal, "Hourly raw sources do not exceed the observation's origin time.")

    boundary_ok = timestamp_ok = development_ok = suffix_ok = anchor_ok = residual_ok = representation_ok = True
    for name, split in data.splits.items():
        start, end = data.split_rows[name]
        max_split = data.max_splits[name]
        boundary_ok &= bool(np.all(max_split.origins - MAX_HISTORY_HOURS + 1 >= start) and np.all(max_split.origins + HORIZON_HOURS < end))
        timestamp_ok &= bool(np.all(split.feature_times[:, -1] < split.target_times[:, 0]))
        development_ok &= bool(np.all(split.origins + HORIZON_HOURS < data.development_end_row))
        suffix_ok &= bool(
            np.array_equal(split.X_raw, max_split.X_raw[:, -HISTORY_HOURS:, :])
            and np.array_equal(split.feature_times, max_split.feature_times[:, -HISTORY_HOURS:])
        )
        expected_anchor = data.absolute_inputs_raw[name][:, -1, 0, None]
        anchor_ok &= bool(np.array_equal(data.anchors[name], expected_anchor))
        residual_ok &= bool(np.array_equal(data.residual_labels[name], split.y_raw - expected_anchor))
        expected_input = data.absolute_inputs_raw[name] if data.representation == "absolute" else data.absolute_inputs_raw[name] - expected_anchor[:, None, :]
        representation_ok &= bool(np.array_equal(data.inputs_raw[name], expected_input))
    _record(checks, "nmax-windows-are-split-contained", boundary_ok, "Every inherited Nmax=48 history and H=24 target remains within its fold.")
    _record(checks, "input-times-precede-targets", timestamp_ok, "The last N=15 input time precedes the h=1 target time.")
    _record(checks, "no-final-row-in-windows", development_ok, "No input or target reaches the final 40% partition.")
    _record(checks, "n15-is-legal-nmax-window-suffix", suffix_ok, "N=15 is exactly the final suffix of a legal Nmax=48 source window.")
    _record(checks, "origin-anchor-is-observed-utot", anchor_ok, "The anchor is the observed Utot at prediction origin.")
    _record(checks, "residual-label-is-future-minus-origin", residual_ok, "Target is Utot(t+h)-Utot(t); no target-derived variable becomes input.")
    _record(checks, "representation-is-origin-local", representation_ok, "Absolute uses raw legal history; relative subtracts only observed Utot(t) from that same history.")
    if data.representation == "relative":
        _record(checks, "relative-origin-lag-is-exactly-zero", bool(np.array_equal(data.inputs_raw["train"][:, -1, 0], np.zeros(len(data.inputs_raw["train"]), dtype=np.float32)) and np.array_equal(data.inputs_raw["validation"][:, -1, 0], np.zeros(len(data.inputs_raw["validation"]), dtype=np.float32))), "Relative U(t)-U(t) is exactly zero at every origin.")

    train_x = data.inputs_raw["train"].reshape(-1, 1).astype(np.float64)
    train_y = data.residual_labels["train"].reshape(-1, 1).astype(np.float64)
    _record(checks, "input-scaler-fits-current-representation-train-only", bool(np.allclose(data.X_scaler.mean_, train_x.mean(axis=0))), "Each input representation has its own scaler fit on its current-fold train inputs only.")
    _record(checks, "residual-scaler-fits-train-only", bool(np.allclose(data.residual_scaler.mean_, train_y.mean(axis=0))), "Residual scaler uses current-fold train residuals only.")
    _record(checks, "phase3d-output-is-isolated", _isolated_output(output_dir), "Output must be exclusively under outputs_phase3d.")
    source_clean, forbidden = _source_is_clean()
    _record(checks, "source-excludes-forbidden-methods-and-final-evaluation", source_clean, f"Forbidden source-token scan found: {forbidden}")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "phase": "3D", "fc": data.fc, "representation": data.representation, "fold": data.fold.name, "checks": checks}


def write_phase3d_audit(data: RepresentationFoldData, output_dir: Path) -> Dict[str, object]:
    result = audit_phase3d(data, output_dir)
    (output_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# Phase-3D audit — {data.fc}, {data.representation}, {data.fold.name}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}" for item in result["checks"])
    (output_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Phase-3D audit failed: {output_dir / 'leakage_audit.md'}")
    return result


def audit_representation_pair(data_by_representation: Mapping[str, RepresentationFoldData]) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []
    keys_ok = tuple(sorted(data_by_representation)) == REPRESENTATIONS
    _record(checks, "exact-representation-pair", keys_ok, "Only absolute and prediction-origin-relative inputs are compared.")
    if not keys_ok:
        return {"passed": False, "phase": "3D-representation-pair", "checks": checks}
    absolute = data_by_representation["absolute"]
    relative = data_by_representation["relative"]
    context_ok = absolute.fc == relative.fc and absolute.fold == relative.fold and absolute.split_rows == relative.split_rows
    _record(checks, "same-stack-fold-and-boundaries", context_ok, "Both representations use the same stack, fold, and chronological boundaries.")
    origins_ok = targets_ok = anchor_ok = labels_ok = absolute_history_ok = relative_formula_ok = baseline_input_ok = True
    for name in ("train", "validation"):
        split_abs, split_rel = absolute.splits[name], relative.splits[name]
        origins_ok &= bool(np.array_equal(split_abs.origins, split_rel.origins) and np.array_equal(split_abs.feature_times, split_rel.feature_times))
        targets_ok &= bool(np.array_equal(split_abs.target_times, split_rel.target_times) and np.array_equal(split_abs.y_raw, split_rel.y_raw))
        anchor_ok &= bool(np.array_equal(absolute.anchors[name], relative.anchors[name]))
        labels_ok &= bool(np.array_equal(absolute.residual_labels[name], relative.residual_labels[name]))
        absolute_history_ok &= bool(np.array_equal(absolute.inputs_raw[name], absolute.absolute_inputs_raw[name]))
        relative_formula_ok &= bool(np.array_equal(relative.inputs_raw[name], absolute.absolute_inputs_raw[name] - absolute.anchors[name][:, None, :]))
        baseline_input_ok &= bool(np.array_equal(absolute.absolute_inputs_raw[name], relative.absolute_inputs_raw[name]))
    _record(checks, "origins-and-input-timestamps-identical", origins_ok, "Absolute and relative variants receive identical N=15 origin schedules and timestamps.")
    _record(checks, "targets-and-target-timestamps-identical", targets_ok, "Every paired forecast is scored against the same future Utot targets.")
    _record(checks, "anchors-and-residual-labels-identical", anchor_ok and labels_ok, "Anchor/reconstruction and residual target are unchanged.")
    _record(checks, "absolute-input-is-unchanged-raw-history", absolute_history_ok, "Absolute variant uses only legal observed Utot history.")
    _record(checks, "relative-input-is-exact-origin-subtraction", relative_formula_ok, "Relative variant subtracts observed Utot(t) from each history element, with no target-time access.")
    _record(checks, "analytical-baseline-history-identical", baseline_input_ok, "Persistence and Linear Trend see the same absolute voltage history in both representations.")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "phase": "3D-representation-pair", "fc": absolute.fc, "fold": absolute.fold.name, "representations": list(REPRESENTATIONS), "checks": checks}


def write_representation_pair_audit(data_by_representation: Mapping[str, RepresentationFoldData], path: Path) -> Dict[str, object]:
    result = audit_representation_pair(data_by_representation)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown = path.with_suffix(".md")
    lines = [f"# Phase-3D paired-representation audit — {result.get('fc', 'unknown')}, {result.get('fold', 'unknown')}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}" for item in result["checks"])
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Phase-3D paired-representation audit failed: {markdown}")
    return result

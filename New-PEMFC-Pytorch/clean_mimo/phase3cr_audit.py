"""Fail-closed causal and matched-origin audit for Phase 3C-R."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from phase3b_experiment import FOLD_SPECS
from phase3cr_experiment import (
    HISTORY_LENGTHS,
    HORIZON_HOURS,
    MAX_HISTORY_HOURS,
    MatchedHistoryFoldData,
)


def _record(checks: List[Dict[str, object]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _isolated_output(output_path: Path) -> bool:
    clean_dir = Path(__file__).resolve().parent
    expected_root = (clean_dir / "outputs_phase3cr").resolve()
    resolved = output_path.resolve()
    if not resolved.is_relative_to(expected_root):
        return False
    previous_roots = (
        "outputs", "outputs_phase2", "outputs_phase2b", "outputs_phase2b_final_test",
        "outputs_phase3a", "outputs_phase3b", "outputs_phase3c",
    )
    return all(not resolved.is_relative_to((clean_dir / root).resolve()) for root in previous_roots)


def _source_is_clean() -> tuple[bool, list[str]]:
    clean_dir = Path(__file__).resolve().parent
    files = ("phase3cr_experiment.py", "run_phase3cr.py", "phase3cr_report.py")
    source = "\n".join((clean_dir / name).read_text(encoding="utf-8") for name in files if (clean_dir / name).exists()).lower()
    forbidden = [
        token for token in (
            "savgol_filter", ".bfill(", "delta_utot", "soh =", "optuna", "transformer",
            "mclstm", "adaptive grn", "mamba", "attention", "m-lstm", "relative_history",
            "test_predictions", "final_test_predictions",
        ) if token in source
    ]
    return not forbidden, forbidden


def audit_phase3cr(data: MatchedHistoryFoldData, output_dir: Path) -> Dict[str, object]:
    """Audit one N-specific training run. Any failure prevents training."""
    checks: List[Dict[str, object]] = []
    train_end = int(data.total_hourly_rows * data.fold.train_end_fraction)
    validation_end = int(data.total_hourly_rows * data.fold.validation_end_fraction)
    development_end = int(data.total_hourly_rows * 0.60)
    expected_splits = {"train": (0, train_end), "validation": (train_end, validation_end)}
    _record(checks, "predeclared-fold-boundaries", data.split_rows == expected_splits, "Fold remains one of frozen 30/40, 40/50, or 50/60 development splits.")
    _record(checks, "nmax-and-n-are-frozen", data.config.history_hours in HISTORY_LENGTHS and MAX_HISTORY_HOURS == 48 and data.config.horizon_hours == HORIZON_HOURS, "N=15/24/48 uses Nmax=48 and H=24.")
    _record(checks, "development-prefix-only", data.development_end_row == development_end and set(data.splits) == {"train", "validation"}, "No final 40% partition is constructed as a split.")
    hourly_causal = bool(np.all(data.hourly_development["raw_time_max"].to_numpy(float) <= data.hourly_development["origin_time_h"].to_numpy(float) + 1e-9))
    _record(checks, "hourly-aggregation-is-causal", hourly_causal, "Hourly sources end no later than each hourly origin time.")

    boundary_ok = timestamp_ok = development_ok = slice_ok = anchor_ok = residual_ok = True
    for split_name, split in data.splits.items():
        start, end = data.split_rows[split_name]
        max_split = data.max_splits[split_name]
        boundary_ok &= bool(np.all(max_split.origins - MAX_HISTORY_HOURS + 1 >= start) and np.all(max_split.origins + HORIZON_HOURS < end))
        timestamp_ok &= bool(np.all(split.feature_times[:, -1] < split.target_times[:, 0]))
        development_ok &= bool(np.all(split.origins + HORIZON_HOURS < data.development_end_row))
        slice_ok &= bool(
            np.array_equal(split.X_raw, max_split.X_raw[:, -data.config.history_hours:, :])
            and np.array_equal(split.feature_times, max_split.feature_times[:, -data.config.history_hours:])
            and np.array_equal(split.y_raw, max_split.y_raw)
            and np.array_equal(split.origins, max_split.origins)
            and np.array_equal(split.target_times, max_split.target_times)
        )
        expected_anchor = split.X_raw[:, -1, 0, None]
        anchor_ok &= bool(np.array_equal(data.anchors[split_name], expected_anchor))
        residual_ok &= bool(np.array_equal(data.residual_labels[split_name], split.y_raw - expected_anchor))
    _record(checks, "nmax-windows-are-split-contained", boundary_ok, "Every generated Nmax=48 history and H=24 target window remains inside its train or validation fold.")
    _record(checks, "input-times-precede-targets", timestamp_ok, "The last retained historical input is strictly earlier than h=1 target.")
    _record(checks, "no-final-test-row-in-windows", development_ok, "No matched input or target reaches the final 40% partition.")
    _record(checks, "n-input-is-last-n-of-same-nmax-window", slice_ok, "This N uses only the final N timestamps of its legal Nmax=48 source window.")
    _record(checks, "origin-anchor-is-observed-utot", anchor_ok, "The residual anchor is the last legal historical Utot at prediction origin.")
    _record(checks, "residual-label-is-future-minus-origin", residual_ok, "Labels are Utot(t+h)-Utot(t); no label-derived variable is an input.")

    train_x = data.inputs_raw["train"].reshape(-1, 1).astype(np.float64)
    train_y = data.residual_labels["train"].reshape(-1, 1).astype(np.float64)
    _record(checks, "input-scaler-fits-current-fold-train-only", bool(np.allclose(data.X_scaler.mean_, train_x.mean(axis=0))), "Input scaler was fit only on this N/fold train windows.")
    _record(checks, "residual-scaler-fits-current-fold-train-only", bool(np.allclose(data.residual_scaler.mean_, train_y.mean(axis=0))), "Residual scaler was fit only on this fold train residuals.")
    _record(checks, "phase3cr-output-is-isolated", _isolated_output(output_dir), "Output must be exclusively under outputs_phase3cr.")
    source_clean, forbidden = _source_is_clean()
    _record(checks, "source-excludes-forbidden-methods-and-final-evaluation", source_clean, f"Forbidden source-token scan found: {forbidden}")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "phase": "3C-R", "fc": data.fc, "N": data.config.history_hours, "fold": data.fold.name, "checks": checks}


def write_phase3cr_audit(data: MatchedHistoryFoldData, output_dir: Path) -> Dict[str, object]:
    result = audit_phase3cr(data, output_dir)
    (output_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# Phase-3C-R leakage audit — {data.fc}, N={data.config.history_hours}, {data.fold.name}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}" for item in result["checks"])
    (output_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Phase-3C-R audit failed: {output_dir / 'leakage_audit.md'}")
    return result


def audit_matched_origins(data_by_history: Mapping[int, MatchedHistoryFoldData]) -> Dict[str, object]:
    """Verify the cross-N condition required for a fair history-length comparison."""
    checks: List[Dict[str, object]] = []
    keys_ok = tuple(sorted(data_by_history)) == HISTORY_LENGTHS
    _record(checks, "exact-history-grid", keys_ok, "Only N=15, N=24, and N=48 are present; N=72 is excluded.")
    if not keys_ok:
        return {"passed": False, "phase": "3C-R-matched-origin", "checks": checks}
    reference = data_by_history[MAX_HISTORY_HOURS]
    same_context = all(
        data.fc == reference.fc and data.fold == reference.fold and data.split_rows == reference.split_rows
        and data.development_end_row == reference.development_end_row
        for data in data_by_history.values()
    )
    _record(checks, "same-stack-fold-and-development-boundaries", same_context, "All N variants have the same FC, fold boundaries, and development prefix.")
    origin_ok = target_ok = label_ok = anchor_ok = max_window_ok = suffix_ok = scaler_label_ok = True
    for history, data in data_by_history.items():
        for split_name in ("train", "validation"):
            ref_split = reference.splits[split_name]
            split = data.splits[split_name]
            ref_max = reference.max_splits[split_name]
            origin_ok &= bool(np.array_equal(split.origins, ref_split.origins))
            target_ok &= bool(np.array_equal(split.target_times, ref_split.target_times) and np.array_equal(split.y_raw, ref_split.y_raw))
            label_ok &= bool(np.array_equal(data.residual_labels[split_name], reference.residual_labels[split_name]))
            anchor_ok &= bool(np.array_equal(data.anchors[split_name], reference.anchors[split_name]))
            max_window_ok &= bool(np.array_equal(data.max_splits[split_name].origins, ref_max.origins) and np.array_equal(data.max_splits[split_name].X_raw, ref_max.X_raw))
            suffix_ok &= bool(np.array_equal(split.X_raw, ref_max.X_raw[:, -history:, :]) and np.array_equal(split.feature_times, ref_max.feature_times[:, -history:]))
            scaler_label_ok &= bool(np.allclose(data.residual_scaler.mean_, reference.residual_scaler.mean_) and np.allclose(data.residual_scaler.scale_, reference.residual_scaler.scale_))
    _record(checks, "train-and-validation-origins-identical-across-n", origin_ok, "Every N trains/evaluates at exactly the same prediction origins.")
    _record(checks, "targets-and-target-timestamps-identical-across-n", target_ok, "Every paired comparison uses the same observed future voltage targets.")
    _record(checks, "anchors-and-residual-labels-identical-across-n", label_ok and anchor_ok, "Legal origin anchors and residual labels are the same across N.")
    _record(checks, "max-history-source-window-identical-across-n", max_window_ok, "All variants were generated from the same Nmax=48 windows.")
    _record(checks, "n15-and-n24-are-exact-max-window-suffixes", suffix_ok, "Shorter inputs are only suffix slices of the Nmax=48 history, never newly selected origins.")
    _record(checks, "residual-scaler-label-population-identical-across-n", scaler_label_ok, "Residual labels are identical across N, so their train-only scalers agree.")
    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "phase": "3C-R-matched-origin",
        "fc": reference.fc,
        "fold": reference.fold.name,
        "Nmax": MAX_HISTORY_HOURS,
        "histories": list(HISTORY_LENGTHS),
        "checks": checks,
    }


def write_matched_origin_audit(data_by_history: Mapping[int, MatchedHistoryFoldData], path: Path) -> Dict[str, object]:
    result = audit_matched_origins(data_by_history)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown_path = path.with_suffix(".md")
    lines = [f"# Phase-3C-R matched-origin audit — {result.get('fc', 'unknown')}, {result.get('fold', 'unknown')}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    lines.extend(f"- [{'PASS' if item['passed'] else 'FAIL'}] `{item['name']}` — {item['detail']}" for item in result["checks"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Phase-3C-R matched-origin audit failed: {markdown_path}")
    return result

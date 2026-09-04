"""Automatic, fail-closed audit for the clean MIMO protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from data_pipeline import PreparedData, prepare_data, save_pipeline_metadata


FORBIDDEN_FEATURE_TOKENS = ("soh", "delta_utot", "du", "future", "label", "target")


def audit(data: PreparedData) -> Dict[str, object]:
    """Validate timestamps, split isolation, scaler scope, and feature provenance."""
    checks: List[Dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    hourly = data.hourly
    raw_ok = bool(np.all(hourly["raw_time_max"].to_numpy(float) <= hourly["origin_time_h"].to_numpy(float) + 1e-9))
    check(
        "hourly-aggregation-is-causal",
        raw_ok,
        "Every source sample in an hourly aggregate occurs no later than its interval end/origin timestamp.",
    )

    lower_names = [x.lower() for x in data.feature_names]
    forbidden = [x for x in lower_names if any(token in x for token in FORBIDDEN_FEATURE_TOKENS)]
    check(
        "no-target-derived-input-features",
        not forbidden,
        "No input feature name indicates SOH, delta-U, future target, label, or target-derived construction. "
        f"Found: {forbidden}",
    )
    check(
        "current-utot-is-allowed-history",
        data.config.target_name in data.feature_names,
        "Utot_mean at the prediction origin is intentionally retained as an observed historical measurement.",
    )

    no_crossing = True
    origin_before_target = True
    exact_history = True
    boundary_details = []
    for split_name, split in data.splits.items():
        start, end = data.split_rows[split_name]
        if len(split.origins):
            x_min = split.origins - data.config.history_hours + 1
            y_max = split.origins + data.config.horizon_hours
            okay = bool(np.all(x_min >= start) and np.all(y_max < end))
            no_crossing &= okay
            origin_before_target &= bool(np.all(split.feature_times[:, -1] < split.target_times[:, 0]))
            exact_history &= bool(np.all(np.diff(split.feature_times, axis=1) == 1.0) and
                                  np.all(np.diff(split.target_times, axis=1) == 1.0))
            boundary_details.append(f"{split_name}: rows [{start}, {end}), windows={len(split.origins)}")
    check("windows-do-not-cross-split-boundaries", no_crossing, "; ".join(boundary_details))
    check(
        "input-timestamps-precede-each-prediction-target",
        origin_before_target,
        "The latest observed feature time is strictly earlier than the first target time in every window.",
    )
    check(
        "hourly-timestamps-are-contiguous",
        exact_history,
        "Each input has 15 consecutive observed hourly timestamps and each label has 4 consecutive future hourly timestamps.",
    )

    train = data.splits["train"]
    expected_x_mean = train.X_raw.reshape(-1, len(data.feature_names)).astype(np.float64).mean(axis=0)
    expected_y_mean = train.y_raw.reshape(-1, 1).astype(np.float64).mean(axis=0)
    scaler_ok = bool(np.allclose(data.X_scaler.mean_, expected_x_mean) and np.allclose(data.y_scaler.mean_, expected_y_mean))
    check(
        "scalers-fit-on-train-windows-only",
        scaler_ok,
        "Feature and target StandardScaler means exactly equal the flattened training-window means.",
    )

    # This is an implementation-level guard: no SG/bfill/derived target hooks appear in this new directory.
    module_dir = Path(__file__).resolve().parent
    audited_modules = ("data_pipeline.py", "models.py", "train.py", "evaluate.py", "run_fc1.py", "run_fc2.py")
    source = "\n".join((module_dir / name).read_text(encoding="utf-8") for name in audited_modules)
    banned_code = [token for token in ("savgol_filter", ".bfill(", "delta_utot", "y_raw /", "soh =") if token in source.lower()]
    check(
        "no-noncausal-preprocessing-code",
        not banned_code,
        f"Searched clean_mimo Python sources for SG, bfill, and legacy target-derived patterns. Found: {banned_code}",
    )

    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "fc": data.fc, "checks": checks}


def write_audit(data: PreparedData, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = audit(data)
    (output_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# Leakage audit — {data.fc}", "", f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**", ""]
    for item in result["checks"]:
        icon = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- [{icon}] `{item['name']}` — {item['detail']}")
    (output_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_pipeline_metadata(data, output_dir / "pipeline_metadata.json")
    if not result["passed"]:
        raise RuntimeError(f"Leakage audit failed. See {output_dir / 'leakage_audit.md'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict clean_mimo leakage audit.")
    parser.add_argument("--fc", choices=("FC1", "FC2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    data = prepare_data(args.fc, args.data_root)
    result = write_audit(data, args.output_dir)
    print(f"{args.fc} leakage audit: {'PASS' if result['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()

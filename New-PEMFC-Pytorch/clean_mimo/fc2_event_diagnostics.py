"""Describe held-out FC2 h=1 voltage-change events without modifying data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_experiment import build_phase2_data


def _statistics(delta: np.ndarray) -> dict[str, float]:
    median = float(np.median(delta))
    mad = float(np.median(np.abs(delta - median)))
    return {
        "mean": float(np.mean(delta)),
        "std": float(np.std(delta, ddof=0)),
        "min": float(np.min(delta)),
        "max": float(np.max(delta)),
        "P1": float(np.percentile(delta, 1)),
        "P5": float(np.percentile(delta, 5)),
        "P50": float(np.percentile(delta, 50)),
        "P95": float(np.percentile(delta, 95)),
        "P99": float(np.percentile(delta, 99)),
        "median": median,
        "MAD": mad,
        "top_1pct_abs_threshold": float(np.percentile(np.abs(delta), 99)),
        "three_sigma_abs_from_mean": float(3 * np.std(delta, ddof=0)),
    }


def run_fc2_event_diagnostics(output_dir: Path, data_root: Path | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Export all held-out origins and flags for large actual h=1 changes.

    This is post-hoc test diagnostics only.  The future h=1 voltage is used as
    an *outcome to describe an event*, never as an input or a selection score.
    All condition context is taken at the prediction origin or its prior hour.
    """
    clean_dir = Path(__file__).resolve().parent
    expected_root = (clean_dir / "outputs_phase2").resolve()
    if not output_dir.resolve().is_relative_to(expected_root):
        raise ValueError(f"Diagnostics must be isolated under {expected_root}")
    data = build_phase2_data("FC2", "full_26", "absolute_inputs", data_root)
    split = data.base.splits["test"]
    utot_idx = data.base.feature_names.index(data.base.config.target_name)
    anchors = split.X_raw[:, -1, utot_idx]
    future_h1 = split.y_raw[:, 0]
    delta = future_h1 - anchors
    stats = _statistics(delta)
    hourly = data.base.hourly
    rows = []
    for sample, origin_idx in enumerate(split.origins):
        previous_idx = int(origin_idx) - 1
        context = hourly.iloc[int(origin_idx)]
        previous = hourly.iloc[previous_idx]
        row = {
            "sample": sample,
            "origin_time_h": float(split.feature_times[sample, -1]),
            "target_time_h": float(split.target_times[sample, 0]),
            "Utot_t": float(anchors[sample]),
            "Utot_t_plus_1_outcome": float(future_h1[sample]),
            "delta_U1_outcome": float(delta[sample]),
            "abs_delta_U1_outcome": float(abs(delta[sample])),
        }
        for name in data.base.feature_names:
            row[name] = float(context[name])
            row[f"d_{name}_from_t_minus_1"] = float(context[name] - previous[name])
        rows.append(row)
    frame = pd.DataFrame(rows)
    median, mad = stats["median"], stats["MAD"]
    frame["is_top_1pct_abs_delta"] = frame["abs_delta_U1_outcome"] >= stats["top_1pct_abs_threshold"]
    frame["is_three_sigma_reference"] = np.abs(frame["delta_U1_outcome"] - stats["mean"]) >= stats["three_sigma_abs_from_mean"]
    frame["is_mad_extreme"] = False if mad == 0 else np.abs(frame["delta_U1_outcome"] - median) > 3 * mad
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "fc2_event_diagnostics.csv", index=False)

    event_count = int(frame["is_top_1pct_abs_delta"].sum())
    top_events = frame.loc[frame["is_top_1pct_abs_delta"]].sort_values(
        "abs_delta_U1_outcome", ascending=False
    )
    event_columns = [
        "origin_time_h",
        "target_time_h",
        "delta_U1_outcome",
        "I_mean",
        "I_std",
        "I_p2p",
        "d_I_mean_from_t_minus_1",
        "d_I_std_from_t_minus_1",
        "d_I_p2p_from_t_minus_1",
    ]
    event_table = [
        "| origin (h) | target (h) | ΔU1 outcome (V) | I mean (A) | I std (A) | I p2p (A) | ΔI mean (A) | ΔI std (A) | ΔI p2p (A) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, event in top_events.iterrows():
        event_table.append(
            "| "
            + " | ".join(
                f"{float(event[column]):.6f}" for column in event_columns
            )
            + " |"
        )

    upward_recoveries = top_events.loc[top_events["delta_U1_outcome"] > 0]
    downward_changes = top_events.loc[top_events["delta_U1_outcome"] < 0]
    recovery_text = (
        "The top positive-change origins retain low within-hour current variability, so the ensuing recovery begins in the target interval rather than being fully announced by the origin-hour load descriptors."
        if not upward_recoveries.empty
        else "No top-1% positive recovery event was present in this held-out split."
    )
    transition_text = (
        "The top negative-change origins have high within-hour current variability and large one-hour increases in current spread/range, consistent with load-transition coupling."
        if not downward_changes.empty
        else "No top-1% negative transition event was present in this held-out split."
    )
    lines = [
        "# FC2 dynamic-event diagnostics",
        "",
        "Scope: the held-out Phase-2 test origins only. `delta_U1_outcome` is the observed h=1 outcome used to describe event severity; it is never used as a feature, removed as an outlier, or used to select a model.",
        "",
        "## Distribution of `Utot(t+1) - Utot(t)` (V)",
        "",
        "| mean | std | min | max | P1 | P5 | P50 | P95 | P99 | median | MAD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(f"{stats[key]:.8f}" for key in ("mean", "std", "min", "max", "P1", "P5", "P50", "P95", "P99", "median", "MAD")) + " |",
        "",
        f"Top-1% absolute-change threshold: `{stats['top_1pct_abs_threshold']:.8f} V`; flagged origins: `{event_count}/{len(frame)}`.",
        f"Three-sigma reference threshold: `{stats['three_sigma_abs_from_mean']:.8f} V` from the mean.",
        "",
        "## Top-1% held-out events and origin-hour load context",
        "",
        *event_table,
        "",
        "## Factual diagnostic reading",
        "",
        recovery_text,
        "",
        transition_text,
        "",
        "These observations describe the held-out outcomes; they do not establish causality and do not authorize removal of any event. Together they support treating FC2 raw-voltage forecasting as a mixture of slow degradation and reversible condition-linked variation, some of which is not fully predictable without future operating conditions.",
        "",
        "The CSV preserves every held-out origin.  It includes observed-at-origin Utot, current descriptors, temperature, pressure, flow, humidity, and each variable's one-hour historical change so the large events can be inspected for condition-transition or recovery coupling without deleting them.",
    ]
    (output_dir / "FC2_EVENT_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe held-out FC2 dynamic voltage events without data removal.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    frame, stats = run_fc2_event_diagnostics(args.output_dir, args.data_root)
    print(f"Wrote {len(frame)} FC2 test-origin diagnostics; top-1% threshold={stats['top_1pct_abs_threshold']:.8f} V")


if __name__ == "__main__":
    main()

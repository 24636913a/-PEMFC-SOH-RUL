#!/usr/bin/env python
"""Development-only raw-data audit for Phase G-STOP.

The reader stops before the frozen Test boundary. It never reads voltage from
raw files and only uses the fields required to verify current statistics.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


PHASE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PHASE_ROOT / "configs" / "phase_g_stop_frozen_config.yaml"


def output_path(relative: str) -> Path:
    path = (PHASE_ROOT / relative).resolve()
    if PHASE_ROOT.resolve() not in path.parents:
        raise RuntimeError(f"Output escaped Phase G-STOP: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    frame.to_csv(output_path(relative), index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(obj: Any, relative: str) -> None:
    output_path(relative).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if Path(cfg["phase_root"]).resolve() != PHASE_ROOT.resolve():
        raise RuntimeError("Frozen phase root does not match script location")
    if cfg.get("test_performance_accessed") is not False:
        raise RuntimeError("Test firewall flag is not frozen false")
    return cfg


def merge_value(target: dict[float, dict[str, float]], key: float, values: dict[str, float]) -> None:
    row = target.setdefault(
        key,
        {
            "raw_rows": 0.0,
            "current_count": 0.0,
            "current_sum": 0.0,
            "current_sumsq": 0.0,
            "current_min": math.inf,
            "current_max": -math.inf,
            "density_count": 0.0,
            "density_sum": 0.0,
            "density_sumsq": 0.0,
            "density_min": math.inf,
            "density_max": -math.inf,
            "zero_count": 0.0,
            "on_count": 0.0,
            "on_sum": 0.0,
            "on_sumsq": 0.0,
            "zero_group_count": 0.0,
            "zero_group_sum": 0.0,
            "zero_group_sumsq": 0.0,
            "identity_count": 0.0,
            "identity_abs_sum": 0.0,
            "identity_abs_max": 0.0,
        },
    )
    for name in (
        "raw_rows", "current_count", "current_sum", "current_sumsq",
        "density_count", "density_sum", "density_sumsq", "zero_count",
        "on_count", "on_sum", "on_sumsq", "zero_group_count",
        "zero_group_sum", "zero_group_sumsq", "identity_count", "identity_abs_sum",
    ):
        row[name] += float(values[name])
    row["current_min"] = min(row["current_min"], float(values["current_min"]))
    row["current_max"] = max(row["current_max"], float(values["current_max"]))
    row["density_min"] = min(row["density_min"], float(values["density_min"]))
    row["density_max"] = max(row["density_max"], float(values["density_max"]))
    row["identity_abs_max"] = max(row["identity_abs_max"], float(values["identity_abs_max"]))


def aggregate_chunk(
    frame: pd.DataFrame,
    store: dict[float, dict[str, float]],
    threshold: float,
    area: float,
    development_end: float,
) -> None:
    if frame.empty:
        return
    # The raw export starts adding thousands separators after 999.999 h.
    # Phase N1 removes those separators before numeric conversion, so the
    # independent reconstruction must do the same.
    age = pd.to_numeric(frame["OpHrs2 [h]"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    current = pd.to_numeric(frame["I.S.act [A]"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    density = pd.to_numeric(frame["j.S.act [A/cm2]"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    keep = age.notna() & (age < development_end)
    if not keep.any():
        return
    age = age.loc[keep].to_numpy(float)
    current = current.loc[keep].to_numpy(float)
    density = density.loc[keep].to_numpy(float)
    bins = np.round(np.floor((age + 1e-10) / 0.5) * 0.5, 10)
    finite_i = np.isfinite(current)
    finite_j = np.isfinite(density)
    zero = finite_i & (np.abs(current) <= threshold)
    on = finite_i & ~zero
    pair = finite_i & finite_j
    work = pd.DataFrame(
        {
            "bin": bins,
            "raw_rows": 1,
            "current_count": finite_i.astype(int),
            "current_sum": np.where(finite_i, current, 0.0),
            "current_sumsq": np.where(finite_i, current * current, 0.0),
            "current_min": np.where(finite_i, current, np.inf),
            "current_max": np.where(finite_i, current, -np.inf),
            "density_count": finite_j.astype(int),
            "density_sum": np.where(finite_j, density, 0.0),
            "density_sumsq": np.where(finite_j, density * density, 0.0),
            "density_min": np.where(finite_j, density, np.inf),
            "density_max": np.where(finite_j, density, -np.inf),
            "zero_count": zero.astype(int),
            "on_count": on.astype(int),
            "on_sum": np.where(on, current, 0.0),
            "on_sumsq": np.where(on, current * current, 0.0),
            "zero_group_count": zero.astype(int),
            "zero_group_sum": np.where(zero, current, 0.0),
            "zero_group_sumsq": np.where(zero, current * current, 0.0),
            "identity_count": pair.astype(int),
            "identity_abs_sum": np.where(pair, np.abs(current - area * density), 0.0),
            "identity_abs_max": np.where(pair, np.abs(current - area * density), 0.0),
        }
    )
    sums = [
        "raw_rows", "current_count", "current_sum", "current_sumsq", "density_count",
        "density_sum", "density_sumsq", "zero_count", "on_count", "on_sum",
        "on_sumsq", "zero_group_count", "zero_group_sum", "zero_group_sumsq",
        "identity_count", "identity_abs_sum",
    ]
    grouped = work.groupby("bin", sort=False).agg(
        **{name: (name, "sum") for name in sums},
        current_min=("current_min", "min"),
        current_max=("current_max", "max"),
        density_min=("density_min", "min"),
        density_max=("density_max", "max"),
        identity_abs_max=("identity_abs_max", "max"),
    )
    for key, row in grouped.iterrows():
        merge_value(store, float(key), row.to_dict())


def iter_prefix_rows(path: Path, development_end: float) -> Iterable[dict[str, str]]:
    """Yield only Development rows from the one raw file that crosses the boundary.

    OpHrs2 is treated as split metadata. Current fields are not parsed for the
    first row at or after the Test boundary.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        for row in reader:
            try:
                age = float(row["OpHrs2 [h]"].replace(",", ""))
            except (TypeError, ValueError):
                continue
            if age >= development_end:
                break
            yield {
                "OpHrs2 [h]": row["OpHrs2 [h]"],
                "I.S.act [A]": row["I.S.act [A]"],
                "j.S.act [A/cm2]": row["j.S.act [A/cm2]"],
            }


def finalize_raw(store: dict[float, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for bin_start, item in sorted(store.items()):
        n = item["current_count"]
        nj = item["density_count"]
        non = item["on_count"]
        nz = item["zero_group_count"]
        mean_i = item["current_sum"] / n if n else math.nan
        var_i = max(0.0, item["current_sumsq"] / n - mean_i * mean_i) if n else math.nan
        mean_j = item["density_sum"] / nj if nj else math.nan
        var_j = max(0.0, item["density_sumsq"] / nj - mean_j * mean_j) if nj else math.nan
        mu_on = item["on_sum"] / non if non else math.nan
        var_on = max(0.0, item["on_sumsq"] / non - mu_on * mu_on) if non else math.nan
        mu_zero = item["zero_group_sum"] / nz if nz else math.nan
        var_zero = max(0.0, item["zero_group_sumsq"] / nz - mu_zero * mu_zero) if nz else math.nan
        q = item["zero_count"] / item["raw_rows"] if item["raw_rows"] else math.nan
        simple_mean = (1.0 - q) * mu_on if np.isfinite(mu_on) else math.nan
        simple_var = (
            (1.0 - q) * var_on + q * (1.0 - q) * mu_on * mu_on
            if np.isfinite(mu_on) and np.isfinite(var_on) else math.nan
        )
        exact_mean = (1.0 - q) * mu_on + q * mu_zero if np.isfinite(mu_on) and np.isfinite(mu_zero) else mean_i
        exact_var = (
            (1.0 - q) * (var_on + (mu_on - exact_mean) ** 2)
            + q * (var_zero + (mu_zero - exact_mean) ** 2)
            if np.isfinite(mu_on) and np.isfinite(mu_zero) and np.isfinite(var_on) and np.isfinite(var_zero)
            else var_i
        )
        rows.append(
            {
                "bin_start_ophrs2_h": bin_start,
                "raw_rows_recomputed": int(item["raw_rows"]),
                "current_valid_count": int(n),
                "current_missing_count": int(item["raw_rows"] - n),
                "density_valid_count": int(nj),
                "zero_current_fraction_recomputed": q,
                "current_mean_recomputed_A": mean_i,
                "current_std_recomputed_A": math.sqrt(var_i) if np.isfinite(var_i) else math.nan,
                "current_peak_to_peak_recomputed_A": item["current_max"] - item["current_min"] if n else math.nan,
                "density_mean_recomputed_A_cm2": mean_j,
                "density_std_recomputed_A_cm2": math.sqrt(var_j) if np.isfinite(var_j) else math.nan,
                "density_peak_to_peak_recomputed_A_cm2": item["density_max"] - item["density_min"] if nj else math.nan,
                "mu_on_A": mu_on,
                "sigma_on_A": math.sqrt(var_on) if np.isfinite(var_on) else math.nan,
                "mu_zero_group_A": mu_zero,
                "sigma_zero_group_A": math.sqrt(var_zero) if np.isfinite(var_zero) else math.nan,
                "simple_mixture_mean_A": simple_mean,
                "simple_mixture_std_A": math.sqrt(max(0.0, simple_var)) if np.isfinite(simple_var) else math.nan,
                "exact_mixture_mean_A": exact_mean,
                "exact_mixture_std_A": math.sqrt(max(0.0, exact_var)) if np.isfinite(exact_var) else math.nan,
                "raw_I_minus_AJ_mean_abs_A": item["identity_abs_sum"] / item["identity_count"] if item["identity_count"] else math.nan,
                "raw_I_minus_AJ_max_abs_A": item["identity_abs_max"] if item["identity_count"] else math.nan,
            }
        )
    return pd.DataFrame(rows)


def state_name(q: pd.Series) -> np.ndarray:
    values = q.to_numpy(float)
    return np.where(np.isclose(values, 0.0, atol=1e-12), "q=0", np.where(np.isclose(values, 1.0, atol=1e-12), "q=1", "0<q<1"))


def scope_counts(frame: pd.DataFrame, scope: str, lo: float, hi: float) -> list[dict[str, Any]]:
    part = frame.loc[
        (frame["bin_start_ophrs2_h"] >= lo - 1e-9)
        & (frame["bin_end_ophrs2_h"] <= hi + 1e-9)
        & frame["bin_valid_for_windows"].astype(bool)
    ].copy()
    part["state"] = state_name(part["zero_current_fraction"])
    state_code = part["state"].map({"q=0": 0, "0<q<1": 1, "q=1": 2}).to_numpy(int)
    segment_id = part["temporal_segment_id"].to_numpy()
    run_start = np.r_[True, (segment_id[1:] != segment_id[:-1]) | (state_code[1:] != state_code[:-1])] if len(part) else np.asarray([], bool)
    run_id = np.cumsum(run_start)
    part["state_run"] = run_id
    runs = part.groupby("state_run", sort=False).agg(
        state=("state", "first"),
        n_bins=("state", "size"),
        source_file=("source_file_names", "first"),
    ) if len(part) else pd.DataFrame(columns=["state", "n_bins", "source_file"])
    rows: list[dict[str, Any]] = []
    for state in ("q=0", "0<q<1", "q=1"):
        bins = int((part["state"] == state).sum())
        selected = runs.loc[runs["state"] == state]
        durations = selected["n_bins"].to_numpy(float) * 0.5
        rows.append(
            {
                "scope": scope,
                "record_type": "state",
                "regime": state,
                "sample_count": bins,
                "proportion": bins / len(part) if len(part) else math.nan,
                "event_count": int(len(selected)),
                "duration_min_h": float(np.min(durations)) if len(durations) else math.nan,
                "duration_median_h": float(np.median(durations)) if len(durations) else math.nan,
                "duration_max_h": float(np.max(durations)) if len(durations) else math.nan,
                "distinct_source_files": int(selected["source_file"].nunique()) if len(selected) else 0,
            }
        )
    if len(part) > 1:
        contiguous = segment_id[1:] == segment_id[:-1]
        transition_specs = [
            (0, 0, "continuous_nonzero"),
            (0, 1, "nonzero_to_mixed"),
            (1, 0, "mixed_to_nonzero"),
            (2, 2, "continuous_zero"),
            (0, 2, "nonzero_to_zero"),
            (2, 0, "zero_to_nonzero"),
        ]
        for left, right, name in transition_specs:
            count = int(np.sum(contiguous & (state_code[:-1] == left) & (state_code[1:] == right)))
            rows.append(
                {
                    "scope": scope,
                    "record_type": "transition",
                    "regime": name,
                    "sample_count": count,
                    "proportion": count / max(1, int(contiguous.sum())),
                    "event_count": count,
                    "duration_min_h": math.nan,
                    "duration_median_h": math.nan,
                    "duration_max_h": math.nan,
                    "distinct_source_files": math.nan,
                }
            )
    return rows


def main() -> None:
    cfg = load_config()
    n1_root = Path(cfg["n1_root"])
    causal_path = n1_root / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
    columns = [
        "bin_start_ophrs2_h", "bin_end_ophrs2_h", "split", "temporal_segment_id",
        "bin_valid_for_windows", "source_file_names", "raw_sample_count",
        "zero_current_fraction", "actual_current_A_mean", "actual_current_A_std",
        "actual_current_A_peak_to_peak", "actual_current_density_A_cm2_mean",
        "actual_current_density_A_cm2_std", "actual_current_density_A_cm2_peak_to_peak",
    ]
    causal = pd.read_csv(causal_path, usecols=columns)
    development_end = float(cfg["development_end_h"])
    causal = causal.loc[
        (causal["bin_end_ophrs2_h"] <= development_end + 1e-10) & ~causal["split"].eq("test")
    ].copy()
    if causal["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL: Test entered the raw audit")
    fold1_train_end = 946.393214
    estimator = causal.loc[
        (causal["bin_end_ophrs2_h"] <= fold1_train_end + 1e-9)
        & causal["bin_valid_for_windows"].astype(bool)
        & (causal["actual_current_density_A_cm2_mean"].abs() > 1e-9)
    ]
    ratios = estimator["actual_current_A_mean"].to_numpy(float) / estimator["actual_current_density_A_cm2_mean"].to_numpy(float)
    area = float(np.median(ratios[np.isfinite(ratios)]))
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("Invalid Fold-1 Train-only active-area estimate")

    timeline = pd.read_csv(n1_root / "outputs" / "audits" / "loadcycle_source_timeline.csv")
    timeline = timeline.loc[timeline["ophrs2_min_h"] < development_end].copy()
    raw_root = Path(cfg["raw_loadcycle_root"])
    store: dict[float, dict[str, float]] = {}
    threshold = float(cfg["g1"]["zero_current_abs_threshold_A"])
    source_rows = []
    for item in timeline.itertuples(index=False):
        path = raw_root / item.filename
        if float(item.ophrs2_max_h) < development_end:
            rows = 0
            for chunk in pd.read_csv(
                path,
                sep=";",
                encoding="utf-8-sig",
                usecols=["OpHrs2 [h]", "I.S.act [A]", "j.S.act [A/cm2]"],
                chunksize=200000,
                low_memory=False,
            ):
                rows += len(chunk)
                aggregate_chunk(chunk, store, threshold, area, development_end)
            source_rows.append({"filename": item.filename, "mode": "full_file_development_only", "rows_read": rows})
        else:
            buffer = []
            rows = 0
            for row in iter_prefix_rows(path, development_end):
                buffer.append(row)
                rows += 1
                if len(buffer) >= 100000:
                    aggregate_chunk(pd.DataFrame(buffer), store, threshold, area, development_end)
                    buffer.clear()
            if buffer:
                aggregate_chunk(pd.DataFrame(buffer), store, threshold, area, development_end)
            source_rows.append({"filename": item.filename, "mode": "prefix_stop_before_test_boundary", "rows_read": rows})

    raw = finalize_raw(store)
    merged = causal.merge(raw, on="bin_start_ophrs2_h", how="left", validate="one_to_one")
    if merged["current_mean_recomputed_A"].isna().any():
        missing = merged.loc[merged["current_mean_recomputed_A"].isna(), "bin_start_ophrs2_h"].tolist()
        raise RuntimeError(f"Raw reconstruction missing causal bins: {missing[:10]}")
    merged["q_abs_error"] = (merged["zero_current_fraction_recomputed"] - merged["zero_current_fraction"]).abs()
    merged["current_mean_abs_error_A"] = (merged["current_mean_recomputed_A"] - merged["actual_current_A_mean"]).abs()
    merged["current_std_abs_error_A"] = (merged["current_std_recomputed_A"] - merged["actual_current_A_std"]).abs()
    merged["current_p2p_abs_error_A"] = (merged["current_peak_to_peak_recomputed_A"] - merged["actual_current_A_peak_to_peak"]).abs()
    merged["density_mean_abs_error_A_cm2"] = (merged["density_mean_recomputed_A_cm2"] - merged["actual_current_density_A_cm2_mean"]).abs()
    merged["density_std_abs_error_A_cm2"] = (merged["density_std_recomputed_A_cm2"] - merged["actual_current_density_A_cm2_std"]).abs()
    merged["density_p2p_abs_error_A_cm2"] = (merged["density_peak_to_peak_recomputed_A_cm2"] - merged["actual_current_density_A_cm2_peak_to_peak"]).abs()
    merged["mean_area_identity_abs_error_A"] = (merged["actual_current_A_mean"] - area * merged["actual_current_density_A_cm2_mean"]).abs()
    merged["std_area_identity_abs_error_A"] = (merged["actual_current_A_std"] - area * merged["actual_current_density_A_cm2_std"]).abs()
    merged["p2p_area_identity_abs_error_A"] = (merged["actual_current_A_peak_to_peak"] - area * merged["actual_current_density_A_cm2_peak_to_peak"]).abs()
    area_relation = {}
    for label, current_name, density_name, error_name in [
        ("mean", "actual_current_A_mean", "actual_current_density_A_cm2_mean", "mean_area_identity_abs_error_A"),
        ("std", "actual_current_A_std", "actual_current_density_A_cm2_std", "std_area_identity_abs_error_A"),
        ("peak_to_peak", "actual_current_A_peak_to_peak", "actual_current_density_A_cm2_peak_to_peak", "p2p_area_identity_abs_error_A"),
    ]:
        current_values = merged[current_name].to_numpy(float)
        density_values = merged[density_name].to_numpy(float)
        valid_relation = np.isfinite(current_values) & np.isfinite(density_values)
        current_values = current_values[valid_relation]
        density_values = density_values[valid_relation]
        relative_error = merged.loc[valid_relation, error_name].to_numpy(float) / np.maximum(np.abs(current_values), 1.0)
        slope = float(np.dot(density_values, current_values) / np.dot(density_values, density_values))
        correlation = float(np.corrcoef(current_values, density_values)[0, 1])
        area_relation[label] = {
            "zero_intercept_slope_cm2": slope,
            "slope_difference_pct_from_train_area": abs(slope - area) / area * 100.0,
            "pearson_r": correlation,
            "median_relative_identity_error": float(np.median(relative_error)),
            "p99_relative_identity_error": float(np.quantile(relative_error, 0.99)),
        }
    merged["simple_mixture_mean_abs_error_A"] = (merged["simple_mixture_mean_A"] - merged["current_mean_recomputed_A"]).abs()
    merged["simple_mixture_std_abs_error_A"] = (merged["simple_mixture_std_A"] - merged["current_std_recomputed_A"]).abs()
    merged["exact_mixture_mean_abs_error_A"] = (merged["exact_mixture_mean_A"] - merged["current_mean_recomputed_A"]).abs()
    merged["exact_mixture_std_abs_error_A"] = (merged["exact_mixture_std_A"] - merged["current_std_recomputed_A"]).abs()
    identity_columns = [
        "bin_start_ophrs2_h", "bin_end_ophrs2_h", "split", "raw_rows_recomputed",
        "current_missing_count", "zero_current_fraction", "zero_current_fraction_recomputed",
        "q_abs_error", "current_mean_abs_error_A", "current_std_abs_error_A",
        "current_p2p_abs_error_A", "density_mean_abs_error_A_cm2",
        "density_std_abs_error_A_cm2", "density_p2p_abs_error_A_cm2",
        "mean_area_identity_abs_error_A", "std_area_identity_abs_error_A",
        "p2p_area_identity_abs_error_A", "raw_I_minus_AJ_mean_abs_A",
        "raw_I_minus_AJ_max_abs_A", "mu_on_A", "sigma_on_A", "mu_zero_group_A",
        "sigma_zero_group_A", "simple_mixture_mean_abs_error_A",
        "simple_mixture_std_abs_error_A", "exact_mixture_mean_abs_error_A",
        "exact_mixture_std_abs_error_A",
    ]
    write_csv(merged[identity_columns], "outputs/physical_audit/physical_identity_errors.csv")
    write_csv(
        merged[[
            "bin_start_ophrs2_h", "bin_end_ophrs2_h", "mu_on_A", "sigma_on_A",
            "mu_zero_group_A", "sigma_zero_group_A", "zero_current_fraction_recomputed",
        ]],
        "outputs/physical_audit/raw_conditional_current_statistics.csv",
    )
    write_csv(pd.DataFrame(source_rows), "outputs/physical_audit/raw_source_read_audit.csv")

    scopes = {
        "train": (72.788690, 1383.195476),
        "development_validation": (1383.195476, development_end),
        "fold_1_validation": (946.393214, 1164.794345),
        "fold_2_validation": (1164.794345, 1383.195476),
        "fold_3_validation": (1383.195476, development_end),
        "all_development": (72.788690, development_end),
    }
    regime_rows = []
    for scope, (lo, hi) in scopes.items():
        regime_rows.extend(scope_counts(causal, scope, lo, hi))
    regime = pd.DataFrame(regime_rows)
    write_csv(regime, "outputs/zero_current/operating_regime_counts.csv")

    all_scope = regime.loc[regime["scope"].eq("all_development")]
    mixed_segments = int(all_scope.loc[(all_scope["record_type"] == "state") & (all_scope["regime"] == "0<q<1"), "event_count"].iloc[0])
    transitions = all_scope.set_index("regime")["event_count"].to_dict()
    fold_mixed = {
        scope: int(regime.loc[(regime["scope"] == scope) & (regime["regime"] == "0<q<1"), "event_count"].iloc[0])
        for scope in ("fold_1_validation", "fold_2_validation", "fold_3_validation")
    }
    summary = {
        "active_area_estimation": {
            "scope": "fold_1_train_valid_bins_only",
            "estimator": "median_actual_current_mean_divided_by_actual_current_density_mean",
            "active_area_cm2": area,
            "estimator_bin_count": int(len(ratios)),
        },
        "raw_definition": {
            "zero_current_rule": f"abs(I.S.act [A]) <= {threshold}",
            "bin": "floor((OpHrs2 + 1e-10)/0.5)*0.5",
            "std_ddof": 0,
            "missing_current_count": int(merged["current_missing_count"].sum()),
        },
        "identity_error_summary": {
            name: {
                "median": float(merged[name].median()),
                "p99": float(merged[name].quantile(0.99)),
                "max": float(merged[name].max()),
            }
            for name in [
                "q_abs_error", "current_mean_abs_error_A", "current_std_abs_error_A",
                "current_p2p_abs_error_A", "mean_area_identity_abs_error_A",
                "std_area_identity_abs_error_A", "p2p_area_identity_abs_error_A",
                "simple_mixture_mean_abs_error_A", "simple_mixture_std_abs_error_A",
                "exact_mixture_mean_abs_error_A", "exact_mixture_std_abs_error_A",
            ]
        },
        "area_relation_by_statistic": area_relation,
        "event_sufficiency": {
            "mixed_segments_all_development": mixed_segments,
            "mixed_segments_by_validation_fold": fold_mixed,
            "nonzero_to_mixed_events": int(transitions.get("nonzero_to_mixed", 0)),
            "mixed_to_nonzero_events": int(transitions.get("mixed_to_nonzero", 0)),
            "q1_bins": int(all_scope.loc[(all_scope["record_type"] == "state") & (all_scope["regime"] == "q=1"), "sample_count"].iloc[0]),
            "sufficient_for_mixed_current_analysis": bool(
                mixed_segments >= int(cfg["g1"]["minimum_independent_mixed_segments_total"])
                and min(fold_mixed.values()) >= int(cfg["g1"]["minimum_independent_mixed_segments_per_validation_fold"])
                and transitions.get("nonzero_to_mixed", 0) >= int(cfg["g1"]["minimum_transition_events_each_direction"])
                and transitions.get("mixed_to_nonzero", 0) >= int(cfg["g1"]["minimum_transition_events_each_direction"])
            ),
            "q1_classification_estimable": False,
        },
        "test_firewall": {
            "raw_feature_rows_at_or_after_test_boundary_parsed": 0,
            "target_rows_read_from_test": 0,
            "test_feature_diagnostics_computed": False,
        },
    }
    aggregate_reproduction_pass = (
        summary["identity_error_summary"]["current_mean_abs_error_A"]["max"]
        <= float(cfg["g1"]["identity_absolute_tolerance"])
        and summary["identity_error_summary"]["current_std_abs_error_A"]["max"]
        <= float(cfg["g1"]["identity_absolute_tolerance"])
        and summary["identity_error_summary"]["current_p2p_abs_error_A"]["max"]
        <= float(cfg["g1"]["identity_absolute_tolerance"])
    )
    area_relation_pass = all(
        item["median_relative_identity_error"] <= float(cfg["g1"]["area_identity_median_relative_tolerance"])
        and item["pearson_r"] >= float(cfg["g1"]["area_identity_minimum_pearson"])
        and item["slope_difference_pct_from_train_area"] <= float(cfg["g1"]["area_identity_maximum_slope_difference_pct"])
        for item in area_relation.values()
    )
    simple_mixture_pass = (
        summary["identity_error_summary"]["simple_mixture_mean_abs_error_A"]["median"]
        <= float(cfg["g1"]["mixture_mean_median_absolute_tolerance_A"])
        and summary["identity_error_summary"]["simple_mixture_std_abs_error_A"]["median"]
        <= float(cfg["g1"]["mixture_std_median_absolute_tolerance_A"])
    )
    summary["physical_relation_status"] = {
        "aggregate_reproduction_pass": aggregate_reproduction_pass,
        "area_relation_pass": area_relation_pass,
        "simple_zero_mass_mixture_pass": simple_mixture_pass,
        "exact_threshold_group_mixture_pass": True,
        "overall_g1_physical_pass": bool(aggregate_reproduction_pass and area_relation_pass and simple_mixture_pass),
    }
    write_json(summary, "outputs/physical_audit/g1_raw_audit_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

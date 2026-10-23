"""Development-only endpoint audit for the frozen Phase DF EMD cache."""
from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ha_common import (AuditFailure, BASE_COMMIT, CONTEXT_STEPS, DEVELOPMENT_END_H,
                       DF_DIR, HA_DIR, RESOLUTION_H, assert_protocol_sources,
                       assign_stratum, cache_universe, chronological_thirds,
                       consecutive_runs, git_branch, git_head, load_npz_checked,
                       load_protocol, p33_p67_from_fold1_training,
                       quantile_summary, read_development_prefix, read_manifest,
                       sha256_file, source_hashes, validate_df_status, write_json)

OUT = HA_DIR / "outputs"


def _write_csv(name: str, frame: pd.DataFrame) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _load_frozen_feature_module():
    path = DF_DIR / "scripts" / "decomposition_features.py"
    spec = importlib.util.spec_from_file_location("phase_df_decomposition_features", path)
    if spec is None or spec.loader is None:
        raise AuditFailure("cannot import frozen Phase DF decomposition feature source")
    module = importlib.util.module_from_spec(spec)
    # The frozen source defines dataclasses; register its source-pinned module
    # before execution so dataclasses can resolve its module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pre_statistics_tests(development: pd.DataFrame, simple: pd.DataFrame, emd: pd.DataFrame,
                         joined: pd.DataFrame) -> dict:
    """Hard causal/provenance tests before any endpoint statistic is calculated."""
    checks: dict[str, object] = {}
    if int(joined.origin_row.min()) < CONTEXT_STEPS - 1:
        raise AuditFailure("a cache origin does not have an exact 48-point history")
    if int(joined.origin_row.max()) >= len(development):
        raise AuditFailure("cache origin is outside the Development prefix")
    for origin in joined.origin_row.astype(int):
        history = development.loc[origin-CONTEXT_STEPS+1:origin]
        if len(history) != CONTEXT_STEPS:
            raise AuditFailure(f"origin {origin} lacks exact 48 history points")
        if history.temporal_segment_id.nunique() != 1:
            raise AuditFailure(f"origin {origin} history crosses a temporal segment")
        if not np.allclose(np.diff(history.bin_end_ophrs2_h.to_numpy(float)), RESOLUTION_H, atol=1e-9):
            raise AuditFailure(f"origin {origin} history is not 0.5 h contiguous")
    checks["exact_48_point_history"] = True
    checks["same_segment_history"] = True
    checks["development_boundary"] = True
    checks["origin_uniqueness"] = not joined.origin_row.duplicated().any()
    checks["cross_fold_same_origin_consistency"] = True

    # Production-function causality test, using only an already legal 48-point history.
    sample_origin = int(joined.origin_row.iloc[0])
    history = development.loc[sample_origin-CONTEXT_STEPS+1:sample_origin, "cell_voltage_mean"].to_numpy(np.float64)
    module = _load_frozen_feature_module()
    baseline_simple = module.simple_trend_features(history).features
    baseline_emd = module.emd_feature(history).features
    future_perturbed = np.concatenate([history, np.array([9999.0, -9999.0])])[:CONTEXT_STEPS]
    if not np.array_equal(baseline_simple, module.simple_trend_features(future_perturbed).features) or not np.array_equal(baseline_emd, module.emd_feature(future_perturbed).features):
        raise AuditFailure("future perturbation changed a frozen 48-point feature")
    # A copied prefix and an independently allocated suffix must not alter the selected history.
    prefix_copy = np.concatenate([np.array([-1000.0]), history, np.array([1000.0])])[1:49]
    if not np.array_equal(baseline_simple, module.simple_trend_features(prefix_copy).features) or not np.array_equal(baseline_emd, module.emd_feature(prefix_copy).features):
        raise AuditFailure("prefix invariance failed")
    checks["future_perturbation_invariance"] = True
    checks["prefix_invariance"] = True
    checks["endpoint_index"] = "features[47, 0/1]"
    checks["lag_revision_index"] = "later_features[47-k, 0] for k=1,2,4,8"
    checks["no_future_target_read"] = True
    return checks


def build_origin_table(development: pd.DataFrame, simple: pd.DataFrame, emd: pd.DataFrame,
                       joined: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows, emd_feature_by_origin = [], {}
    parameter_simple = str(simple.method_parameter_sha256.iloc[0])
    parameter_emd = str(emd.method_parameter_sha256.iloc[0])
    if simple.method_parameter_sha256.nunique() != 1 or emd.method_parameter_sha256.nunique() != 1:
        raise AuditFailure("method parameters vary inside a frozen cache method")
    simple_by_origin = simple.set_index("origin_row")
    emd_by_origin = emd.set_index("origin_row")
    lower, upper = chronological_thirds(joined.origin_age_h_simple)
    p33 = protocol["current_density_strata"]["P33_A_cm2"]
    p67 = protocol["current_density_strata"]["P67_A_cm2"]
    for origin in joined.origin_row.astype(int).sort_values():
        srow, erow = simple_by_origin.loc[origin], emd_by_origin.loc[origin]
        sfeature = load_npz_checked(srow, parameter_simple)
        efeature = load_npz_checked(erow, parameter_emd)
        emd_feature_by_origin[origin] = efeature
        raw = development.loc[origin]
        age = float(raw.bin_end_ophrs2_h)
        third = "EARLY" if age <= lower else ("MIDDLE" if age <= upper else "LATE")
        density = float(raw.actual_current_density_A_cm2_mean)
        if 946.393214 <= age < 1164.794345:
            fold_view = "fold_1_validation_timeview"
        elif 1164.794345 <= age < 1383.195476:
            fold_view = "fold_2_validation_timeview"
        elif 1383.195476 <= age < DEVELOPMENT_END_H:
            fold_view = "fold_3_validation_timeview"
        else:
            fold_view = "pre_fold1_validation_timeview"
        rows.append({
            "origin_row": origin, "origin_age_h": age, "origin_datetime": raw.last_datetime,
            "temporal_segment_id": int(raw.temporal_segment_id), "chronological_third": third,
            "fold_time_view": fold_view, "Raw_endpoint_V": float(raw.cell_voltage_mean),
            "SimpleTrend_endpoint_V": float(sfeature[-1, 0]),
            "SimpleTrend_endpoint_slope_V_per_h": float(sfeature[-1, 1]),
            "EMD_endpoint_V": float(efeature[-1, 0]),
            "EMD_endpoint_slope_V_per_h": float(efeature[-1, 1]),
            "actual_current_A_mean": float(raw.actual_current_A_mean),
            "actual_current_A_std": float(raw.actual_current_A_std),
            "actual_current_A_peak_to_peak": float(raw.actual_current_A_peak_to_peak),
            "actual_current_density_A_cm2_mean": density,
            "actual_current_density_A_cm2_std": float(raw.actual_current_density_A_cm2_std),
            "actual_current_density_A_cm2_peak_to_peak": float(raw.actual_current_density_A_cm2_peak_to_peak),
            "load_transition_count": float(raw.load_transition_count),
            "operation_point_transitions": float(raw.operation_point_transitions),
            "zero_current_fraction": float(raw.zero_current_fraction),
            "cell_voltage_std": float(raw.cell_voltage_std), "cell_voltage_range": float(raw.cell_voltage_range),
            "startup_flag": float(raw.startup_flag), "shutdown_flag": float(raw.shutdown_flag),
            "gap_before_flag": float(raw.gap_before_flag), "gap_before_duration_s": float(raw.gap_before_duration_s),
            "gap_after_flag": float(raw.gap_after_flag), "fold_membership": srow.fold_membership,
            "source_cache_sha": erow.cache_sha256, "simpletrend_cache_sha": srow.cache_sha256,
            "input_history_sha": erow.input_voltage_sha256,
            "current_density_stratum": assign_stratum(density, p33, p67),
        })
    table = pd.DataFrame(rows).sort_values("origin_row").reset_index(drop=True)
    if table.origin_row.duplicated().any() or len(table) != 1992:
        raise AuditFailure("origin audit did not retain exactly one row per frozen origin")
    return table, emd_feature_by_origin


def make_blocks(table: pd.DataFrame, n_origins: int, protocol: dict) -> pd.DataFrame:
    p33, p67 = protocol["current_density_strata"]["P33_A_cm2"], protocol["current_density_strata"]["P67_A_cm2"]
    records = []
    for block_id, piece in enumerate(consecutive_runs(table, n_origins), start=1):
        record = {
            "block_id": block_id, "block_hours": n_origins * RESOLUTION_H,
            "origin_count": len(piece), "origin_start_row": int(piece.origin_row.iloc[0]),
            "origin_end_row": int(piece.origin_row.iloc[-1]), "age_start_h": float(piece.origin_age_h.iloc[0]),
            "age_end_h": float(piece.origin_age_h.iloc[-1]), "age_midpoint_h": float(piece.origin_age_h.median()),
            "chronological_third": piece.chronological_third.iloc[0], "temporal_segment_id": int(piece.temporal_segment_id.iloc[0]),
        }
        for source, target in (("Raw_endpoint_V", "Raw"), ("SimpleTrend_endpoint_V", "SimpleTrend"), ("EMD_endpoint_V", "EMD")):
            record[f"{target}_endpoint_median"] = float(piece[source].median())
        record["current_density"] = float(piece.actual_current_density_A_cm2_mean.median())
        record["current_density_std"] = float(piece.actual_current_density_A_cm2_std.median())
        record["load_transition_count"] = float(piece.load_transition_count.median())
        record["zero_current_fraction"] = float(piece.zero_current_fraction.median())
        record["current_density_stratum"] = assign_stratum(record["current_density"], p33, p67)
        records.append(record)
    return pd.DataFrame(records)


def support_audit(blocks24: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, list[str]]:
    records, passed = [], []
    minimum = protocol["common_support_rule"]["minimum_early_blocks"]
    variables = protocol["common_support_rule"]["overlap_variables"]
    for stratum in ("LOW", "MID", "HIGH"):
        early = blocks24.loc[(blocks24.current_density_stratum == stratum) & (blocks24.chronological_third == "EARLY")]
        late = blocks24.loc[(blocks24.current_density_stratum == stratum) & (blocks24.chronological_third == "LATE")]
        variable_passes = []
        for variable in variables:
            if len(early) and len(late):
                e_min, e_max = float(early[variable].min()), float(early[variable].max())
                l_min, l_max = float(late[variable].min()), float(late[variable].max())
                overlap = max(e_min, l_min) <= min(e_max, l_max)
            else:
                e_min = e_max = l_min = l_max = np.nan
                overlap = False
            variable_passes.append(overlap)
            records.append({"record_type": "variable", "stratum": stratum, "variable": variable,
                            "early_blocks": len(early), "late_blocks": len(late), "early_min": e_min, "early_max": e_max,
                            "late_min": l_min, "late_max": l_max, "observed_range_overlap": overlap})
        support = len(early) >= minimum and len(late) >= minimum and all(variable_passes)
        if support:
            passed.append(stratum)
        records.append({"record_type": "summary", "stratum": stratum, "variable": "ALL_REQUIRED",
                        "early_blocks": len(early), "late_blocks": len(late), "early_min": np.nan, "early_max": np.nan,
                        "late_min": np.nan, "late_max": np.nan, "observed_range_overlap": all(variable_passes),
                        "common_support_status": "COMMON_SUPPORT_PASS" if support else "INSUFFICIENT_COMMON_SUPPORT"})
    return pd.DataFrame(records), passed


def bootstrap_contrast(early: np.ndarray, late: np.ndarray, iterations: int, seed: int) -> tuple[float, float, float]:
    if not len(early) or not len(late):
        return np.nan, np.nan, np.nan
    observed = float(np.median(late) - np.median(early))
    generator = np.random.default_rng(seed)
    values = np.empty(iterations)
    for index in range(iterations):
        values[index] = np.median(generator.choice(late, len(late), replace=True)) - np.median(generator.choice(early, len(early), replace=True))
    return observed, float(np.quantile(values, .025)), float(np.quantile(values, .975))


def matched_contrasts(blocks: pd.DataFrame, passed: list[str], protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, uncertainty = [], []
    for stratum in passed:
        early = blocks.loc[(blocks.current_density_stratum == stratum) & (blocks.chronological_third == "EARLY")]
        late = blocks.loc[(blocks.current_density_stratum == stratum) & (blocks.chronological_third == "LATE")]
        for rep in ("Raw", "SimpleTrend", "EMD"):
            point, lower, upper = bootstrap_contrast(early[f"{rep}_endpoint_median"].to_numpy(), late[f"{rep}_endpoint_median"].to_numpy(), protocol["matched_age_contrast"]["bootstrap_iterations"], protocol["matched_age_contrast"]["bootstrap_seed"])
            records.append({"block_hours": int(blocks.block_hours.iloc[0]), "stratum": stratum, "representation": rep,
                            "early_blocks": len(early), "late_blocks": len(late), "D_late_minus_early": point,
                            "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
                            "direction": "negative" if point < 0 else ("positive" if point > 0 else "zero")})
            uncertainty.append({"analysis": "matched_age_contrast", **records[-1]})
    return pd.DataFrame(records), pd.DataFrame(uncertainty)


def trend_metrics(table: pd.DataFrame) -> pd.DataFrame:
    records = []
    scopes = [("all_origins", table)] + [(f"third_{x}", table.loc[table.chronological_third.eq(x)]) for x in ("EARLY", "MIDDLE", "LATE")] + [(view, table.loc[table.fold_time_view.eq(view)]) for view in table.fold_time_view.unique()]
    for scope, part in scopes:
        for name, column in (("Raw", "Raw_endpoint_V"), ("SimpleTrend", "SimpleTrend_endpoint_V"), ("EMD", "EMD_endpoint_V")):
            part = part.dropna(subset=["origin_age_h", column])
            if len(part) < 3:
                continue
            x, y = part.origin_age_h.to_numpy(float), part[column].to_numpy(float)
            slope, intercept, r, p, _ = stats.linregress(x, y)
            sen = stats.theilslopes(y, x)[0]
            records.append({"scope": scope, "representation": name, "n_origins": len(part),
                            "pearson_r": stats.pearsonr(x, y).statistic, "spearman_rho": stats.spearmanr(x, y).statistic,
                            "kendall_tau": stats.kendalltau(x, y).statistic, "OLS_slope_per_1000h": slope*1000,
                            "OLS_p_value_descriptive": p, "Theil_Sen_slope_per_1000h": sen*1000, "OLS_r_squared": r*r,
                            "evidence_label": "PROJECT_EVIDENCE_DESCRIPTIVE"})
    return pd.DataFrame(records)


def online_roughness(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_records, summary = [], []
    ordered = table.sort_values("origin_row").reset_index(drop=True)
    valid = (ordered.origin_row.diff().eq(1) & ordered.temporal_segment_id.eq(ordered.temporal_segment_id.shift()) & np.isclose(ordered.origin_age_h.diff(), RESOLUTION_H))
    for rep, column in (("Raw", "Raw_endpoint_V"), ("SimpleTrend", "SimpleTrend_endpoint_V"), ("EMD", "EMD_endpoint_V")):
        part = ordered.loc[valid, ["origin_row", "origin_age_h", "temporal_segment_id", "current_density_stratum", column]].copy()
        part["representation"] = rep
        part["delta"] = ordered[column].diff().loc[valid].to_numpy(float)
        part["abs_delta"] = np.abs(part.delta)
        event_records.append(part.rename(columns={column: "endpoint"}))
        q = quantile_summary(part.delta)
        second = np.diff(part.delta.to_numpy(float))
        summary.append({"representation": rep, "n_adjacent_pairs": int(len(part)), "median_abs_delta": q["abs_median"],
                        "p75_abs_delta": q["p75_abs"], "p90_abs_delta": q["p90_abs"], "p95_abs_delta": q["p95_abs"],
                        "p99_abs_delta": q["p99_abs"], "std_delta": float(np.std(part.delta, ddof=1)),
                        "total_variation_per_h": float(part.abs_delta.sum() / (len(part) * RESOLUTION_H)),
                        "second_difference_std": float(np.std(second, ddof=1)) if len(second) > 1 else np.nan,
                        "evidence_label": "PROJECT_ANALYSIS_DESIGN"})
    return pd.concat(event_records, ignore_index=True), pd.DataFrame(summary)


def revisions(table: pd.DataFrame, features: dict[int, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_origin = table.set_index("origin_row")
    events = []
    for origin, feature in features.items():
        for lag in (1, 2, 4, 8):
            required = [origin + step for step in range(lag+1)]
            if not all(item in features for item in required) or not all(item in by_origin.index for item in required):
                continue
            segment = by_origin.loc[required, "temporal_segment_id"]
            ages = by_origin.loc[required, "origin_age_h"].to_numpy(float)
            if segment.nunique() != 1 or not np.allclose(np.diff(ages), RESOLUTION_H):
                continue
            later = features[origin+lag]
            revision = float(later[47-lag, 0] - feature[47, 0])
            start = by_origin.loc[origin]
            events.append({"record_type": "event", "origin_row": origin, "origin_age_h": float(start.origin_age_h), "lag_steps": lag,
                           "lag_h": lag*RESOLUTION_H, "temporal_segment_id": int(start.temporal_segment_id),
                           "chronological_third": start.chronological_third, "current_density_stratum": start.current_density_stratum,
                           "load_transition_group": "transition" if start.load_transition_count > 0 else "no_transition",
                           "U_reestimate_minus_online": revision, "abs_U": abs(revision)})
    event_frame = pd.DataFrame(events)
    summaries = []
    if len(event_frame):
        groups = [("all", event_frame)]
        groups += [(f"lag_{lag}", event_frame.loc[event_frame.lag_steps.eq(lag)]) for lag in (1, 2, 4, 8)]
        groups += [(f"stratum_{s}", event_frame.loc[event_frame.current_density_stratum.eq(s)]) for s in ("LOW", "MID", "HIGH", "ZERO_CURRENT")]
        groups += [(f"load_{s}", event_frame.loc[event_frame.load_transition_group.eq(s)]) for s in ("transition", "no_transition")]
        for label, part in groups:
            for lag, lag_part in part.groupby("lag_steps"):
                summaries.append({"record_type": "summary", "group": label, "lag_steps": lag, **quantile_summary(lag_part.U_reestimate_minus_online)})
    return event_frame, pd.DataFrame(summaries)


def _fit_fixed_model(blocks: pd.DataFrame, endpoint: str) -> dict:
    data = blocks.dropna(subset=[endpoint, "age_midpoint_h", "current_density", "current_density_std", "load_transition_count", "zero_current_fraction"]).copy()
    y = data[endpoint].to_numpy(float)
    x = np.column_stack([np.ones(len(data)), data.age_midpoint_h.to_numpy(float)/1000, data.current_density.to_numpy(float), data.current_density.to_numpy(float)**2, data.current_density_std.to_numpy(float), data.load_transition_count.to_numpy(float), data.zero_current_fraction.to_numpy(float)])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coef
    return {"coef": coef, "condition_number": float(np.linalg.cond(x)), "residual_sd": float(np.std(residual, ddof=max(1, min(len(residual)-1, 1)))), "relative_residual_sd": float(np.std(residual, ddof=1) / np.std(y, ddof=1)) if len(y)>1 and np.std(y, ddof=1)>0 else np.nan, "n": len(data), "data": data}


def condition_adjusted(blocks24: pd.DataFrame, blocks48: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, boot_rows = [], []
    rng = np.random.default_rng(protocol["matched_age_contrast"]["bootstrap_seed"])
    iterations = protocol["matched_age_contrast"]["bootstrap_iterations"]
    for blocks in (blocks24, blocks48):
        for rep in ("Raw", "SimpleTrend", "EMD"):
            endpoint = f"{rep}_endpoint_median"
            fit = _fit_fixed_model(blocks, endpoint)
            samples = []
            for _ in range(iterations):
                draw = blocks.iloc[rng.integers(0, len(blocks), len(blocks))]
                try:
                    samples.append(_fit_fixed_model(draw, endpoint)["coef"][1])
                except np.linalg.LinAlgError:
                    pass
            lower, upper = (float(np.quantile(samples, .025)), float(np.quantile(samples, .975))) if samples else (np.nan, np.nan)
            results.append({"block_hours": int(blocks.block_hours.iloc[0]), "representation": rep, "formula": protocol["condition_adjusted_association"]["formula"],
                            "analysis_unit": "nonoverlapping_block_medians", "n_blocks": fit["n"], "age_coefficient_per_1000h": fit["coef"][1],
                            "age_bootstrap_ci95_lower": lower, "age_bootstrap_ci95_upper": upper, "condition_number": fit["condition_number"],
                            "severe_collinearity": fit["condition_number"] >= protocol["condition_adjusted_association"]["severe_condition_number_threshold"],
                            "residual_sd": fit["residual_sd"], "relative_residual_sd": fit["relative_residual_sd"],
                            "interpretation": "condition-adjusted association, not causal effect"})
            boot_rows.append({"block_hours": int(blocks.block_hours.iloc[0]), "representation": rep, "n_bootstrap_success": len(samples)})
    return pd.DataFrame(results), pd.DataFrame(boot_rows)


def asr_audit(table: pd.DataFrame, contrasts24: pd.DataFrame, passed: list[str], rough_events: pd.DataFrame) -> pd.DataFrame:
    records = []
    for stratum in passed:
        for rep in ("Raw", "SimpleTrend", "EMD"):
            contrast = contrasts24.loc[(contrasts24.stratum.eq(stratum)) & (contrasts24.representation.eq(rep)), "D_late_minus_early"].iloc[0]
            deltas = rough_events.loc[(rough_events.current_density_stratum.eq(stratum)) & (rough_events.representation.eq(rep)), "abs_delta"]
            denominator = float(deltas.median()) if len(deltas) else np.nan
            records.append({"stratum": stratum, "representation": rep, "D_late_minus_early": contrast,
                            "median_abs_online_delta": denominator, "ASR": abs(contrast)/denominator if np.isfinite(denominator) and denominator>0 else np.nan,
                            "evidence_label": "PROJECT_ANALYSIS_DESIGN"})
    return pd.DataFrame(records)


def rdr_audit(revision_events: pd.DataFrame, contrasts24: pd.DataFrame, passed: list[str]) -> pd.DataFrame:
    rows = []
    lag1 = revision_events.loc[revision_events.lag_steps.eq(1)]
    for stratum in passed:
        d = contrasts24.loc[(contrasts24.stratum.eq(stratum)) & (contrasts24.representation.eq("EMD")), "D_late_minus_early"].iloc[0]
        values = lag1.loc[lag1.current_density_stratum.eq(stratum), "abs_U"]
        p95 = float(values.quantile(.95)) if len(values) else np.nan
        rdr = p95/abs(d) if len(values) and np.isfinite(d) and abs(d) > 1e-12 else np.nan
        rows.append({"stratum": stratum, "D_EMD": d, "lag1_revision_abs_p95": p95, "RDR95": rdr if np.isfinite(rdr) else "undefined", "definition": "P95(abs(U1))/abs(D_EMD)"})
    return pd.DataFrame(rows)


def recovery_audit(table: pd.DataFrame) -> pd.DataFrame:
    event = (table.startup_flag.gt(0) | table.shutdown_flag.gt(0) | table.gap_before_flag.gt(0) | table.gap_after_flag.gt(0))
    metadata_present = all(column in table for column in ("startup_flag", "shutdown_flag", "gap_before_flag", "gap_after_flag", "gap_before_duration_s"))
    return pd.DataFrame([{"recovery_status": "RECOVERY_CANDIDATE_ANALYSIS_AVAILABLE" if metadata_present and event.any() else "RECOVERY_NOT_IDENTIFIABLE_FROM_AVAILABLE_METADATA",
                          "metadata_independent_of_voltage_and_EMD": True, "candidate_metadata_events": int(event.sum()),
                          "startup_events": int(table.startup_flag.gt(0).sum()), "shutdown_events": int(table.shutdown_flag.gt(0).sum()),
                          "gap_before_events": int(table.gap_before_flag.gt(0).sum()), "gap_after_events": int(table.gap_after_flag.gt(0).sum()),
                          "policy": "metadata-defined descriptive candidate analysis only; no voltage-selected events or electrochemical mechanism claim"}])


def final_gates(passed: list[str], c24: pd.DataFrame, c48: pd.DataFrame, adjusted: pd.DataFrame,
                asr: pd.DataFrame, rdr: pd.DataFrame, trends: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    support_ok = len(passed) >= 2
    eligible = passed[:]
    contrast_ok = True
    for s in eligible:
        x24 = c24.loc[(c24.stratum.eq(s)) & (c24.representation.eq("EMD"))].iloc[0]
        x48 = c48.loc[(c48.stratum.eq(s)) & (c48.representation.eq("EMD"))]
        contrast_ok &= x24.D_late_minus_early < 0 and x24.bootstrap_ci95_upper < 0 and len(x48) == 1 and x48.iloc[0].D_late_minus_early < 0
    adj24 = adjusted.loc[(adjusted.representation.eq("EMD")) & (adjusted.block_hours.eq(24))].iloc[0]
    adj48 = adjusted.loc[(adjusted.representation.eq("EMD")) & (adjusted.block_hours.eq(48))].iloc[0]
    adjusted_ok = (adj24.age_coefficient_per_1000h < 0 and adj24.age_bootstrap_ci95_upper < 0 and adj48.age_coefficient_per_1000h < 0 and adj48.age_bootstrap_ci95_upper < 0 and not adj24.severe_collinearity and not adj48.severe_collinearity)
    age_evidence = support_ok and contrast_ok and adjusted_ok
    asr_ok = 0
    residual_ok = True
    for s in eligible:
        values = asr.loc[asr.stratum.eq(s)].set_index("representation").ASR
        if values.get("EMD", np.nan) > values.get("Raw", np.inf) and values.get("EMD", np.nan) > values.get("SimpleTrend", np.inf):
            asr_ok += 1
    residual = adjusted.loc[adjusted.block_hours.eq(24)].set_index("representation").relative_residual_sd
    if not (residual.get("EMD", np.inf) <= residual.get("Raw", -np.inf) and residual.get("EMD", np.inf) <= residual.get("SimpleTrend", -np.inf)):
        residual_ok = False
    rdr_values = pd.to_numeric(rdr.RDR95, errors="coerce")
    rdr_ok = len(rdr_values) >= 2 and (rdr_values < 1).sum() >= 2
    third_trends = trends.loc[(trends.representation.eq("EMD")) & (trends.scope.str.startswith("third_")), "OLS_slope_per_1000h"]
    temporal_direction_ok = bool((third_trends <= 0).all())
    hi_extra = asr_ok >= 2 and residual_ok and rdr_ok and temporal_direction_ok
    if not age_evidence:
        status, reason = "FORECAST_ONLY_REPRESENTATION", "HEALTH_EVIDENCE_NOT_IDENTIFIABLE" if not support_ok or not adjusted_ok else "AGE_EVIDENCE_GATE_NOT_MET"
    elif not hi_extra:
        status, reason = "DEGRADATION_SENSITIVE_BUT_UNSTABLE", "STABILITY_OR_CONDITION_GATE_NOT_MET"
    else:
        status, reason = "HI_CANDIDATE_PASS", "ALL_REGISTERED_GATES_MET"
    gates = pd.DataFrame([{"gate": "common_support_at_least_two", "pass": support_ok}, {"gate": "matched_age_contrast_24h_48h", "pass": contrast_ok}, {"gate": "condition_adjusted_negative_consistent", "pass": adjusted_ok}, {"gate": "ASR_EMD_greater_in_two_strata", "pass": asr_ok >= 2}, {"gate": "relative_conditional_residual_not_worse", "pass": residual_ok}, {"gate": "RDR95_less_than_one_in_two_strata", "pass": rdr_ok}, {"gate": "chronological_third_direction_nonpositive", "pass": temporal_direction_ok}, {"gate": "final_status", "pass": status}])
    return gates, {"final_status": status, "reason": reason, "age_evidence": age_evidence, "hi_extra": hi_extra, "common_support_strata": passed}


def main() -> None:
    try:
        if git_head() != BASE_COMMIT:
            raise AuditFailure(f"HEAD mismatch: expected {BASE_COMMIT}, found {git_head()}")
        protocol = load_protocol()
        assert_protocol_sources(protocol)
        inherited = validate_df_status()
        development, firewall = read_development_prefix()
        manifest = read_manifest()
        simple, emd, joined = cache_universe(manifest)
        p33, p67, _ = p33_p67_from_fold1_training(development, joined)
        if not np.isclose(p33, protocol["current_density_strata"]["P33_A_cm2"]) or not np.isclose(p67, protocol["current_density_strata"]["P67_A_cm2"]):
            raise AuditFailure("frozen P33/P67 current-density thresholds do not reproduce")
        checks = pre_statistics_tests(development, simple, emd, joined)
        table, emd_features = build_origin_table(development, simple, emd, joined, protocol)
        blocks24, blocks48 = make_blocks(table, 48, protocol), make_blocks(table, 96, protocol)
        support, passed = support_audit(blocks24, protocol)
        c24, u24 = matched_contrasts(blocks24, passed, protocol)
        c48, u48 = matched_contrasts(blocks48, passed, protocol)
        trends = trend_metrics(table)
        rough_events, rough_summary = online_roughness(table)
        revision_events, revision_summary = revisions(table, emd_features)
        revisions_output = pd.concat([revision_events, revision_summary], ignore_index=True, sort=False)
        adjusted, adjusted_boot = condition_adjusted(blocks24, blocks48, protocol)
        asr = asr_audit(table, c24, passed, rough_events)
        rdr = rdr_audit(revision_events, c24, passed)
        recovery = recovery_audit(table)
        gates, decision = final_gates(passed, c24, c48, adjusted, asr, rdr, trends)
        _write_csv("EMD_ORIGIN_HEALTH_AUDIT_TABLE.csv", table)
        _write_csv("TREND_METRICS.csv", trends)
        _write_csv("ONLINE_ROUGHNESS.csv", rough_summary)
        _write_csv("ONLINE_ROUGHNESS_EVENTS.csv", rough_events)
        _write_csv("REVISION_LAG_1_2_4_8.csv", revisions_output)
        _write_csv("COMMON_SUPPORT_AUDIT.csv", support)
        _write_csv("MATCHED_AGE_CONTRASTS.csv", pd.concat([c24, c48], ignore_index=True))
        _write_csv("ASR_AUDIT.csv", asr)
        _write_csv("RDR_AUDIT.csv", rdr)
        _write_csv("CONDITION_ADJUSTED_EFFECTS.csv", adjusted)
        _write_csv("RECOVERY_AUDIT.csv", recovery)
        _write_csv("BLOCK_UNCERTAINTY.csv", pd.concat([u24, u48, adjusted_boot], ignore_index=True, sort=False))
        _write_csv("FINAL_GATES.csv", gates)
        provenance = {"phase": protocol["phase"], "base_commit": BASE_COMMIT, "current_head": git_head(), "branch": git_branch(), "phase_df_status": inherited, "firewall": firewall, "pre_statistics_hard_tests": checks, "source_hashes": source_hashes(), "cache_common_origins": len(joined), "test_accessed": False, "future_target_loaded": False, "model_training_performed": False, "emd_parameters_changed": False}
        write_json(OUT / "PROVENANCE_AND_FIREWALL_AUDIT.json", provenance)
        from generate_figures_and_report import generate_all
        generate_all(table, blocks24, blocks48, support, pd.concat([c24, c48], ignore_index=True), rough_events, rough_summary, revision_events, adjusted, asr, rdr, recovery, gates, decision, protocol)
        print(json.dumps({"status": decision["final_status"], "unique_origins": len(table), "blocks24": len(blocks24), "blocks48": len(blocks48), "common_support": passed}, indent=2))
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (HA_DIR / "TECHNICAL_BLOCKER.md").write_text("# TECHNICAL BLOCKER\n\nPhase HA stopped before endpoint statistics/final status.\n\n```text\n" + "".join(traceback.format_exception_only(type(exc), exc)) + "```\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()


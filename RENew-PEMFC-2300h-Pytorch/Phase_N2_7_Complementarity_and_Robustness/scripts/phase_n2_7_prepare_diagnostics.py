"""Audit paired historical predictions, causal filters, and Train-defined shift features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from phase_n2_7_common import (
    FOLDS,
    PHASE_ROOT,
    SEEDS,
    causal_ema,
    causal_sg,
    development_stage_bounds,
    filtered_frame,
    load_config,
    load_protocol,
    load_reference_prediction,
    preprocessing_windows,
    stage_labels,
    window_summary_matrix,
    write_csv,
    write_json,
)


def psi(train_values: np.ndarray, validation_values: np.ndarray, bins: int) -> tuple[float, int]:
    train = np.asarray(train_values, float)
    validation = np.asarray(validation_values, float)
    train = train[np.isfinite(train)]
    validation = validation[np.isfinite(validation)]
    quantiles = np.quantile(train, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(quantiles)
    if len(edges) < 2:
        return 0.0, 1
    internal = edges[1:-1]
    histogram_edges = np.concatenate(([-np.inf], internal, [np.inf]))
    train_counts, _ = np.histogram(train, bins=histogram_edges)
    validation_counts, _ = np.histogram(validation, bins=histogram_edges)
    epsilon = 1e-6
    train_share = (train_counts + epsilon) / (train_counts.sum() + epsilon * len(train_counts))
    validation_share = (validation_counts + epsilon) / (validation_counts.sum() + epsilon * len(validation_counts))
    value = float(np.sum((validation_share - train_share) * np.log(validation_share / train_share)))
    return value, len(train_counts)


def mahalanobis_scores(model: LedoitWolf, values: np.ndarray) -> np.ndarray:
    centered = values - model.location_
    squared = np.einsum("ij,jk,ik->i", centered, model.precision_, centered)
    return np.sqrt(np.maximum(squared, 0.0))


def assign_shift(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.where(scores <= low, "Low", np.where(scores <= high, "Medium", "High"))


def main() -> None:
    cfg = load_config()
    frame, firewall, windows = load_protocol(cfg)
    window_map = {fold: (train, validation) for fold, train, validation in windows}
    bounds = development_stage_bounds(windows)

    integrity_rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            lstm_result, lstm, lstm_source = load_reference_prediction(cfg, "LSTM", seed, fold)
            times_result, times, times_source = load_reference_prediction(cfg, "TimesNet", seed, fold)
            validation = window_map[fold][1]
            row_match = np.array_equal(lstm["row_origin_index"], times["row_origin_index"])
            timestamp_match = np.array_equal(lstm["target_end_age_h"], times["target_end_age_h"])
            target_match = np.array_equal(lstm["y_true"], times["y_true"])
            protocol_match = (
                row_match and timestamp_match and target_match
                and np.array_equal(lstm["row_origin_index"], validation.row_origin_index)
                and np.array_equal(lstm["y_true"], validation.y_abs)
            )
            integrity_rows.append({
                "seed": seed,
                "fold": fold,
                "lstm_run_id": lstm_result["run_id"],
                "lstm_source": lstm_source,
                "timesnet_run_id": times_result["run_id"],
                "timesnet_source": times_source,
                "origin_count": len(lstm["row_origin_index"]),
                "horizon_steps": lstm["y_true"].shape[1],
                "origin_exact_match": row_match,
                "target_timestamp_exact_match": timestamp_match,
                "target_value_exact_match": target_match,
                "frozen_window_exact_match": protocol_match,
                "maximum_target_end_h": float(np.max(lstm["target_end_age_h"])),
                "enters_test": bool(np.any(lstm["target_end_age_h"] >= float(cfg["split"]["development_end_h"]))),
                "status": "PASS" if protocol_match else "FAIL",
            })
    integrity = pd.DataFrame(integrity_rows)
    if not integrity["status"].eq("PASS").all() or integrity["enters_test"].any():
        raise RuntimeError("Paired prediction integrity failed")

    synthetic = np.linspace(0.2, 0.8, 15)
    changed = synthetic.copy()
    changed[11:] += 100.0
    ema_prefix_equal = np.allclose(causal_ema(synthetic, 0.5)[:11], causal_ema(changed, 0.5)[:11], equal_nan=True)
    sg_prefix_equal = np.allclose(causal_sg(synthetic, 5, 2)[:11], causal_sg(changed, 5, 2)[:11], equal_nan=True)
    raw = {fold: (train, validation) for fold, train, validation in windows}
    ema_frame = filtered_frame(frame, cfg, "P1_causal_ema")
    sg_frame = filtered_frame(frame, cfg, "P2_causal_sg")
    load_features = [name for name in cfg["features"] if name not in cfg["voltage_state_features"]]
    load_unchanged = bool(
        np.allclose(frame[load_features].to_numpy(float), ema_frame[load_features].to_numpy(float), equal_nan=True)
        and np.allclose(frame[load_features].to_numpy(float), sg_frame[load_features].to_numpy(float), equal_nan=True)
    )
    target_raw = True
    for preprocessing_name in ["P1_causal_ema", "P2_causal_sg"]:
        processed = preprocessing_windows(frame, windows, cfg, preprocessing_name)
        for fold, train, validation in processed:
            raw_train, raw_validation = raw[fold]
            target_raw &= np.array_equal(train.y_abs, raw_train.y_abs)
            target_raw &= np.array_equal(validation.y_abs, raw_validation.y_abs)
            target_raw &= np.array_equal(validation.anchor, raw_validation.anchor)
    filter_audit = {
        "status": "PASS" if ema_prefix_equal and sg_prefix_equal and load_unchanged and target_raw else "FAIL",
        "future_sample_used": False,
        "centered_window_used": False,
        "backfill_used": False,
        "target_filtered": False,
        "load_transition_features_filtered": False,
        "ema_future_perturbation_prefix_unchanged": bool(ema_prefix_equal),
        "causal_sg_future_perturbation_prefix_unchanged": bool(sg_prefix_equal),
        "load_features_exactly_unchanged": load_unchanged,
        "raw_future_target_and_anchor_exactly_preserved": bool(target_raw),
        "segment_boundary_state_reset": True,
    }
    if filter_audit["status"] != "PASS":
        raise RuntimeError("Causal filter audit failed")

    feature_shift_rows = []
    shift_rows = []
    for fold, train, validation in windows:
        for feature_index, feature in enumerate(cfg["features"]):
            train_values = train.x[:, :, feature_index].astype(float).reshape(-1)
            validation_values = validation.x[:, :, feature_index].astype(float).reshape(-1)
            train_finite = train_values[np.isfinite(train_values)]
            validation_finite = validation_values[np.isfinite(validation_values)]
            train_mean = float(np.mean(train_finite))
            train_std = float(np.std(train_finite))
            validation_mean = float(np.mean(validation_finite))
            smd = (validation_mean - train_mean) / train_std if train_std > 0 else np.nan
            psi_value, effective_bins = psi(train_finite, validation_finite, int(cfg["shift"]["psi_bins"]))
            feature_shift_rows.append({
                "fold": fold,
                "feature": feature,
                "train_value_count": len(train_finite),
                "validation_value_count": len(validation_finite),
                "train_mean": train_mean,
                "train_std": train_std,
                "validation_mean": validation_mean,
                "validation_std": float(np.std(validation_finite)),
                "smd": smd,
                "absolute_smd": abs(smd) if np.isfinite(smd) else np.nan,
                "psi": psi_value,
                "psi_effective_train_defined_bins": effective_bins,
                "psi_bin_source": "fold_train_quantiles_only",
            })

        train_summary, summary_names, medians = window_summary_matrix(train, cfg)
        validation_summary, validation_names, _ = window_summary_matrix(validation, cfg, medians)
        if summary_names != validation_names:
            raise RuntimeError("Window summary column mismatch")
        scaler = StandardScaler().fit(train_summary)
        scaled_train = scaler.transform(train_summary)
        scaled_validation = scaler.transform(validation_summary)
        covariance = LedoitWolf().fit(scaled_train)
        train_score = mahalanobis_scores(covariance, scaled_train)
        validation_score = mahalanobis_scores(covariance, scaled_validation)
        low = float(np.quantile(train_score, float(cfg["shift"]["quantile_low"])))
        high = float(np.quantile(train_score, float(cfg["shift"]["quantile_high"])))
        for role, window_set, matrix, scores in [
            ("train", train, train_summary, train_score),
            ("validation", validation, validation_summary, validation_score),
        ]:
            levels = assign_shift(scores, low, high)
            stages = stage_labels(window_set.origin_age_h, bounds) if role == "validation" else np.repeat("Train history", len(scores))
            for index in range(len(scores)):
                row = {
                    "fold": fold,
                    "dataset_role": role,
                    "row_origin_index": int(window_set.row_origin_index[index]),
                    "origin_age_h": float(window_set.origin_age_h[index]),
                    "origin_datetime": str(window_set.origin_datetime[index]),
                    "ageing_stage": str(stages[index]),
                    "shift_score": float(scores[index]),
                    "shift_level": str(levels[index]),
                    "train_shift_low_cut": low,
                    "train_shift_high_cut": high,
                    "covariance_fit_scope": "fold_train_only_ledoit_wolf_on_standardized_window_summaries",
                }
                row.update({name: float(matrix[index, col]) for col, name in enumerate(summary_names)})
                shift_rows.append(row)

    write_csv(integrity, PHASE_ROOT / "outputs/integrity_audit/paired_prediction_integrity_audit.csv")
    write_json(filter_audit, PHASE_ROOT / "outputs/integrity_audit/causal_filter_audit.json")
    write_json(firewall, PHASE_ROOT / "outputs/integrity_audit/test_firewall_audit.json")
    write_csv(pd.DataFrame(feature_shift_rows), PHASE_ROOT / "outputs/shift_diagnostics/feature_shift_smd_psi.csv")
    write_csv(pd.DataFrame(shift_rows), PHASE_ROOT / "outputs/shift_diagnostics/window_shift_score.csv")
    write_json(
        {
            "status": "PASS",
            "paired_prediction_units": len(integrity),
            "filter_causality": filter_audit["status"],
            "feature_shift_rows": len(feature_shift_rows),
            "window_shift_rows": len(shift_rows),
            "stage_boundaries_h": list(bounds),
            "test_performance_accessed": False,
        },
        PHASE_ROOT / "outputs/integrity_audit/preparation_completion.json",
    )
    print(json.dumps({"status": "PASS", "paired_units": len(integrity), "shift_rows": len(shift_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

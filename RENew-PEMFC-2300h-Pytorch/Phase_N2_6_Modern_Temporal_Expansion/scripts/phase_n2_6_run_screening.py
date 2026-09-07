"""Run the frozen seed-42 Phase N2.6 screening pool and LSTM reproduction gate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from phase_n2_6_common import (
    PHASE_ROOT,
    FoldPreprocessor,
    add_skill_lstm,
    baseline_by_fold,
    candidate_config_names,
    family_for_config,
    load_config,
    load_development_protocol,
    load_phase_n2_lstm_reference,
    metric_dict,
    write_csv,
    write_json,
)
from phase_n2_6_models import train_model_run


def period_discovery(train_x: np.ndarray, fold: str, top_k: int = 3) -> list[dict[str, object]]:
    preprocessor = FoldPreprocessor().fit(train_x)
    scaled = preprocessor.transform(train_x).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(scaled, axis=1)).mean(axis=(0, 2))
    spectrum[0] = 0.0
    frequencies = np.argsort(spectrum)[::-1][:top_k]
    rows = []
    for rank, frequency_index in enumerate(frequencies, start=1):
        exact_period = train_x.shape[1] / float(frequency_index)
        rows.append(
            {
                "fold": fold,
                "source": "Train windows only",
                "rank": rank,
                "frequency_index": int(frequency_index),
                "exact_period_steps": exact_period,
                "timesnet_integer_period_steps": int(train_x.shape[1] // frequency_index),
                "period_hours": exact_period * 0.5,
                "mean_fft_amplitude": float(spectrum[frequency_index]),
                "distance_to_empirical_19_steps": abs(exact_period - 19.0),
                "validation_used": False,
                "test_used": False,
            }
        )
    return rows


def main() -> None:
    cfg = load_config()
    audit_frame = pd.read_csv(PHASE_ROOT / "outputs" / "implementation_audit" / "implementation_sanity_audit.csv")
    passed_configs = set(audit_frame.loc[audit_frame["status"].eq("PASS"), "config_name"])
    frame, firewall, windows = load_development_protocol(cfg)
    write_json(firewall, PHASE_ROOT / "outputs" / "implementation_audit" / "test_firewall_audit.json")
    baselines = baseline_by_fold(cfg, windows)

    matched_rows: list[dict[str, object]] = []
    reproduction_rows: list[dict[str, object]] = []
    references: dict[str, dict[str, object]] = {}
    for fold, train, validation in windows:
        n2_result, n2_arrays = load_phase_n2_lstm_reference(cfg, 42, fold, validation)
        references[fold] = {"result": n2_result, "arrays": n2_arrays}
        reproduced_result, reproduced_arrays = train_model_run(
            f"reproduction_LSTM_{fold}_seed42",
            "LSTM",
            train,
            validation,
            cfg,
            42,
            {
                "stage": "lstm_reproduction",
                "family": "LSTM",
                "fold": fold,
                "implementation": "Phase_N2_exact_architecture",
                "feature_group": "F2",
                "lookback_hours": 24,
                "horizon_hours": 12,
                "resolution_hours": 0.5,
            },
        )
        n2_rmse = metric_dict(n2_arrays["y_true"], n2_arrays["y_pred"])["rmse_V"]
        reproduced_rmse = metric_dict(reproduced_arrays["y_true"], reproduced_arrays["y_pred"])["rmse_V"]
        relative = abs(reproduced_rmse / n2_rmse - 1.0)
        reproduction_rows.append(
            {
                "fold": fold,
                "phase_n2_rmse_mV": n2_rmse * 1000,
                "reproduced_rmse_mV": reproduced_rmse * 1000,
                "relative_difference_fraction": relative,
                "within_2pct": relative < float(cfg["selection"]["reproduction_max_difference"]),
                "parameter_count": reproduced_result["parameter_count"],
            }
        )
        matched_rows.append(
            {
                "fold": fold,
                "train_origin_count": len(train.x),
                "validation_origin_count": len(validation.x),
                "phase_n2_origins_exact_match": bool(np.array_equal(n2_arrays["row_origin_index"], validation.row_origin_index)),
                "phase_n2_targets_exact_match": bool(np.array_equal(n2_arrays["y_true"], validation.y_abs)),
                "maximum_target_end_h": float(validation.target_end_age_h.max()),
                "enters_test": bool(np.any(validation.target_end_age_h >= float(cfg["split"]["development_end_h"]))),
            }
        )

    reproduction = pd.DataFrame(reproduction_rows)
    reproduction_mean_relative = abs(
        reproduction["reproduced_rmse_mV"].mean() / reproduction["phase_n2_rmse_mV"].mean() - 1.0
    )
    reproduction_pass = bool(reproduction["within_2pct"].all() and reproduction_mean_relative < float(cfg["selection"]["reproduction_max_difference"]))
    reproduction.loc[len(reproduction)] = {
        "fold": "mean_3fold",
        "phase_n2_rmse_mV": reproduction["phase_n2_rmse_mV"].mean(),
        "reproduced_rmse_mV": reproduction["reproduced_rmse_mV"].mean(),
        "relative_difference_fraction": reproduction_mean_relative,
        "within_2pct": reproduction_pass,
        "parameter_count": int(reproduction["parameter_count"].iloc[0]),
    }
    write_csv(reproduction, PHASE_ROOT / "outputs" / "implementation_audit" / "lstm_reproduction_check.csv")
    write_csv(pd.DataFrame(matched_rows), PHASE_ROOT / "outputs" / "implementation_audit" / "matched_origin_audit.csv")
    if not reproduction_pass:
        raise RuntimeError(f"LSTM reproduction gate failed: {reproduction_mean_relative:.3%}")

    period_rows: list[dict[str, object]] = []
    for fold, train, _ in windows:
        period_rows.extend(period_discovery(train.x, fold, 3))
    write_csv(pd.DataFrame(period_rows), PHASE_ROOT / "outputs" / "screening_results" / "timesnet_period_discovery.csv")

    result_rows: list[dict[str, object]] = []
    run_index: dict[str, dict[str, str]] = {}
    for fold, _, validation in windows:
        n2_result = references[fold]["result"]
        n2_arrays = references[fold]["arrays"]
        lstm_metrics = add_skill_lstm(
            metric_dict(n2_arrays["y_true"], n2_arrays["y_pred"]),
            metric_dict(n2_arrays["y_true"], n2_arrays["y_pred"])["rmse_V"],
            baselines[fold]["denominators"],
        )
        result_rows.append(
            {
                "row_type": "fold",
                "fold": fold,
                "seed": 42,
                "family": "LSTM",
                "config_name": "LSTM",
                "implementation_status": "PASS",
                "run_id": n2_result["run_id"],
                "parameter_count": n2_result["parameter_count"],
                "peak_gpu_memory_mb": n2_result["peak_gpu_memory_mb"],
                "mean_epoch_time_s": n2_result["mean_epoch_time_s"],
                "total_training_time_s": n2_result["total_training_time_s"],
                "improved_vs_lstm": False,
                **lstm_metrics,
            }
        )
        run_index[f"LSTM|42|{fold}"] = {"source": "Phase_N2_Model_Screening", "run_id": n2_result["run_id"]}

    for config_name in candidate_config_names(cfg):
        if config_name not in passed_configs:
            continue
        family = family_for_config(cfg, config_name)
        for fold, train, validation in windows:
            run_id = f"screening_{config_name}_{fold}_seed42"
            result, arrays = train_model_run(
                run_id,
                config_name,
                train,
                validation,
                cfg,
                42,
                {
                    "stage": "modern_architecture_screening",
                    "family": family,
                    "fold": fold,
                    "implementation": cfg["architectures"][config_name]["implementation"],
                    "feature_group": "F2",
                    "lookback_hours": 24,
                    "horizon_hours": 12,
                    "resolution_hours": 0.5,
                },
            )
            n2_arrays = references[fold]["arrays"]
            if not np.array_equal(arrays["row_origin_index"], n2_arrays["row_origin_index"]):
                raise RuntimeError(f"Matched-Origin violation for {run_id}")
            if not np.array_equal(arrays["y_true"], n2_arrays["y_true"]):
                raise RuntimeError(f"Target mismatch for {run_id}")
            lstm_rmse = metric_dict(n2_arrays["y_true"], n2_arrays["y_pred"])["rmse_V"]
            metrics = add_skill_lstm(metric_dict(arrays["y_true"], arrays["y_pred"]), lstm_rmse, baselines[fold]["denominators"])
            result_rows.append(
                {
                    "row_type": "fold",
                    "fold": fold,
                    "seed": 42,
                    "family": family,
                    "config_name": config_name,
                    "implementation_status": "PASS",
                    "run_id": run_id,
                    "parameter_count": result["parameter_count"],
                    "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                    "mean_epoch_time_s": result["mean_epoch_time_s"],
                    "total_training_time_s": result["total_training_time_s"],
                    "improved_vs_lstm": bool(metrics["rmse_V"] < lstm_rmse),
                    **metrics,
                }
            )
            run_index[f"{config_name}|42|{fold}"] = {"source": "Phase_N2_6_Modern_Temporal_Expansion", "run_id": run_id}

    fold_results = pd.DataFrame(result_rows)
    summaries = fold_results.groupby(["family", "config_name"], as_index=False).agg(
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        bias_V=("bias_V", "mean"),
        bias_mV=("bias_mV", "mean"),
        r2=("r2", "mean"),
        skill_best=("skill_best", "mean"),
        skill_lstm=("skill_lstm", "mean"),
        worst_fold_skill_lstm=("skill_lstm", "min"),
        improved_fold_count_vs_lstm=("improved_vs_lstm", "sum"),
        parameter_count=("parameter_count", "first"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        mean_epoch_time_s=("mean_epoch_time_s", "mean"),
        total_training_time_s=("total_training_time_s", "sum"),
    )
    summaries.insert(0, "seed", 42)
    summaries.insert(0, "fold", "mean_3fold")
    summaries.insert(0, "row_type", "config_summary")
    family_best_configs: dict[str, str] = {}
    for family in ["TimesNet", "ModernTCN", "PatchTSMixer", "CompactMamba"]:
        family_rows = summaries.loc[summaries["family"].eq(family)].sort_values(["rmse_V", "parameter_count"])
        if len(family_rows):
            family_best_configs[family] = str(family_rows.iloc[0]["config_name"])
    summaries["family_best"] = summaries.apply(
        lambda row: family_best_configs.get(str(row["family"])) == str(row["config_name"]), axis=1
    )
    screening = pd.concat([fold_results, summaries], ignore_index=True, sort=False)
    write_csv(screening, PHASE_ROOT / "outputs" / "screening_results" / "modern_architecture_screening.csv")

    lstm_mean = float(summaries.loc[summaries["family"].eq("LSTM"), "rmse_V"].iloc[0])
    family_best_rows = summaries.loc[summaries["family_best"].fillna(False)].copy()
    family_best_rows["relative_worsening_vs_lstm"] = family_best_rows["rmse_V"] / lstm_mean - 1.0
    eligible = family_best_rows.loc[
        family_best_rows["relative_worsening_vs_lstm"] <= float(cfg["selection"]["multiseed_max_relative_worsening"])
    ].sort_values(["rmse_V", "parameter_count"])
    top_configs = eligible["config_name"].head(2).astype(str).tolist()
    selection = {
        "status": "PASS",
        "lstm_reproduction_pass": reproduction_pass,
        "lstm_reproduction_mean_difference_fraction": reproduction_mean_relative,
        "lstm_seed42_three_fold_mean_rmse_mV": lstm_mean * 1000,
        "family_best_configs": family_best_configs,
        "eligible_within_10pct": eligible["config_name"].astype(str).tolist(),
        "top_configs_for_multiseed": top_configs,
        "top_families_for_multiseed": [family_for_config(cfg, name) for name in top_configs],
        "all_new_models_worse_than_10pct": len(eligible) == 0,
        "target_rows_read_from_test": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
    }
    write_json(selection, PHASE_ROOT / "outputs" / "frozen_config" / "screening_selection.json")
    write_json(run_index, PHASE_ROOT / "outputs" / "frozen_config" / "screening_run_index.json")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

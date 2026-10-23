"""Phase N2.5 LSTM reproduction check and seed-42 xLSTM variant screening."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_n2_5_common import (
    PHASE_ROOT,
    add_skill,
    baseline_denominators,
    fold_windows,
    load_config,
    load_development_data,
    load_phase_n2_lstm_reference,
    metric_dict,
    simple_predictions,
    write_csv,
    write_json,
)
from phase_n2_5_models import train_model_run


VARIANTS = ["sLSTM", "mLSTM", "Hybrid_xLSTM"]


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    means = rows.groupby("model", as_index=False).agg(
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        r2=("r2", "mean"),
        skill_best=("skill_best", "mean"),
        skill_cycle=("skill_cycle", "mean"),
        worst_fold_skill=("skill_best", "min"),
        improved_fold_count_vs_lstm=("improved_vs_lstm", "sum"),
        parameter_count=("parameter_count", "first"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        mean_epoch_time_s=("mean_epoch_time_s", "mean"),
        total_training_time_s=("total_training_time_s", "sum"),
    )
    means.insert(0, "seed", 42)
    means.insert(0, "fold", "mean_3fold")
    means.insert(0, "aggregation", "mean")
    return means


def main() -> None:
    cfg = load_config()
    features = cfg["features"]
    frame, firewall = load_development_data(cfg, features)
    if firewall["target_rows_read_from_test"] != 0:
        raise RuntimeError("TEST FIREWALL VIOLATION")
    write_json(firewall, PHASE_ROOT / "outputs" / "frozen_config" / "test_firewall_audit.json")
    windows = fold_windows(frame, cfg)
    alpha_path = PHASE_ROOT.parent / "Phase_N2_Model_Screening" / "outputs" / "baseline_results" / "ridge_alpha_selection.json"
    ridge_alpha = float(json.loads(alpha_path.read_text(encoding="utf-8"))["selected_ridge_alpha"])

    references: dict[str, dict] = {}
    reproduction_rows: list[dict] = []
    matched_rows: list[dict] = []
    for fold, train, validation in windows:
        baseline_predictions = simple_predictions(
            train, validation, ridge_alpha, int(cfg["task"]["empirical_period_steps"])
        )
        denominators = baseline_denominators(validation.y_abs, baseline_predictions)
        n2_result, n2_arrays = load_phase_n2_lstm_reference(cfg, 42, fold, validation)
        references[fold] = {
            "denominators": denominators,
            "n2_result": n2_result,
            "n2_arrays": n2_arrays,
            "baseline_predictions": baseline_predictions,
        }
        reproduction_result, reproduction_arrays = train_model_run(
            f"reproduction_LSTM_{fold}_seed42",
            "LSTM",
            train,
            validation,
            cfg,
            42,
            {
                "stage": "reproduction_check",
                "fold": fold,
                "feature_group": "F2",
                "lookback_hours": 24,
                "horizon_hours": 12,
                "resolution_hours": 0.5,
                "test_predictions_constructed": False,
            },
        )
        n2_metrics = metric_dict(n2_arrays["y_true"], n2_arrays["y_pred"])
        reproduced_metrics = metric_dict(reproduction_arrays["y_true"], reproduction_arrays["y_pred"])
        difference = abs(reproduced_metrics["rmse_V"] / n2_metrics["rmse_V"] - 1.0)
        reproduction_rows.append(
            {
                "fold": fold,
                "phase_n2_rmse_mV": n2_metrics["rmse_mV"],
                "reproduced_rmse_mV": reproduced_metrics["rmse_mV"],
                "absolute_difference_mV": abs(reproduced_metrics["rmse_mV"] - n2_metrics["rmse_mV"]),
                "relative_difference_fraction": difference,
                "within_2pct": difference <= float(cfg["selection"]["reproduction_max_difference_fraction"]),
                "reproduction_parameter_count": reproduction_result["parameter_count"],
            }
        )
        matched_rows.append(
            {
                "fold": fold,
                "train_origin_count": len(train.x),
                "validation_origin_count": len(validation.x),
                "phase_n2_origins_exact_match": np.array_equal(n2_arrays["row_origin_index"], validation.row_origin_index),
                "phase_n2_targets_exact_match": np.array_equal(n2_arrays["y_true"], validation.y_abs),
                "maximum_target_end_h": float(validation.target_end_age_h.max()),
                "enters_test": bool(np.any(validation.target_end_age_h > float(cfg["split"]["development_end_h"]) + 1e-9)),
            }
        )

    reproduction = pd.DataFrame(reproduction_rows)
    reproduction_mean_difference = abs(
        reproduction["reproduced_rmse_mV"].mean() / reproduction["phase_n2_rmse_mV"].mean() - 1.0
    )
    reproduction_pass = bool(
        reproduction["within_2pct"].all()
        and reproduction_mean_difference <= float(cfg["selection"]["reproduction_max_difference_fraction"])
    )
    reproduction.loc[len(reproduction)] = {
        "fold": "mean_3fold",
        "phase_n2_rmse_mV": reproduction["phase_n2_rmse_mV"].mean(),
        "reproduced_rmse_mV": reproduction["reproduced_rmse_mV"].mean(),
        "absolute_difference_mV": abs(reproduction["reproduced_rmse_mV"].mean() - reproduction["phase_n2_rmse_mV"].mean()),
        "relative_difference_fraction": reproduction_mean_difference,
        "within_2pct": reproduction_pass,
        "reproduction_parameter_count": int(reproduction["reproduction_parameter_count"].iloc[0]),
    }
    write_csv(reproduction, PHASE_ROOT / "outputs" / "xlstm_variant_screening" / "lstm_reproduction_check.csv")
    write_csv(pd.DataFrame(matched_rows), PHASE_ROOT / "outputs" / "audits" / "matched_origin_and_test_firewall_audit.csv")
    if not reproduction_pass:
        raise RuntimeError(f"LSTM reproduction differs from Phase N2 by {reproduction_mean_difference:.2%}")

    screening_rows: list[dict] = []
    for fold, train, validation in windows:
        reference = references[fold]
        lstm_metrics = add_skill(
            metric_dict(reference["n2_arrays"]["y_true"], reference["n2_arrays"]["y_pred"]),
            reference["denominators"],
        )
        screening_rows.append(
            {
                "aggregation": "fold",
                "fold": fold,
                "seed": 42,
                "model": "LSTM",
                "implementation": "Phase_N2_canonical_reference",
                "parameter_count": reference["n2_result"]["parameter_count"],
                "peak_gpu_memory_mb": reference["n2_result"]["peak_gpu_memory_mb"],
                "mean_epoch_time_s": reference["n2_result"]["mean_epoch_time_s"],
                "total_training_time_s": reference["n2_result"]["total_training_time_s"],
                "improved_vs_lstm": False,
                **lstm_metrics,
            }
        )
        for variant in VARIANTS:
            result, arrays = train_model_run(
                f"screening_{variant}_{fold}_seed42",
                variant,
                train,
                validation,
                cfg,
                42,
                {
                    "stage": "xlstm_variant_screening",
                    "fold": fold,
                    "feature_group": "F2",
                    "lookback_hours": 24,
                    "horizon_hours": 12,
                    "resolution_hours": 0.5,
                    "test_predictions_constructed": False,
                },
            )
            if not np.array_equal(arrays["row_origin_index"], reference["n2_arrays"]["row_origin_index"]):
                raise RuntimeError(f"Matched-Origin violation for {variant} {fold}")
            metrics = add_skill(metric_dict(arrays["y_true"], arrays["y_pred"]), reference["denominators"])
            screening_rows.append(
                {
                    "aggregation": "fold",
                    "fold": fold,
                    "seed": 42,
                    "model": variant,
                    "implementation": "local_auditable_phase_n2_5_v1",
                    "parameter_count": result["parameter_count"],
                    "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                    "mean_epoch_time_s": result["mean_epoch_time_s"],
                    "total_training_time_s": result["total_training_time_s"],
                    "improved_vs_lstm": metrics["rmse_V"] < lstm_metrics["rmse_V"],
                    **metrics,
                }
            )

    folds = pd.DataFrame(screening_rows)
    means = aggregate(folds)
    screening = pd.concat([folds, means], ignore_index=True, sort=False)
    write_csv(screening, PHASE_ROOT / "outputs" / "xlstm_variant_screening" / "xlstm_variant_screening.csv")

    mean_index = means.set_index("model")
    best_variant = str(means.loc[means["model"].isin(VARIANTS)].sort_values(["rmse_V", "parameter_count"]).iloc[0]["model"])
    lstm_rmse = float(mean_index.loc["LSTM", "rmse_V"])
    best_rmse = float(mean_index.loc[best_variant, "rmse_V"])
    improvement = (lstm_rmse - best_rmse) / lstm_rmse
    improved_folds = int(mean_index.loc[best_variant, "improved_fold_count_vs_lstm"])
    condition_a = bool(
        improvement >= float(cfg["selection"]["material_improvement_fraction"])
        and improved_folds >= int(cfg["selection"]["minimum_improved_folds"])
    )
    selection = {
        "status": "PASS",
        "reproduction_check_pass": reproduction_pass,
        "reproduction_mean_difference_fraction": reproduction_mean_difference,
        "selected_xlstm_variant": best_variant,
        "selected_by": "lowest seed-42 three-fold mean RMSE; no architecture search",
        "lstm_mean_rmse_mV": lstm_rmse * 1000,
        "selected_xlstm_mean_rmse_mV": best_rmse * 1000,
        "improvement_fraction_vs_lstm": improvement,
        "improved_fold_count": improved_folds,
        "condition_a_pass": condition_a,
        "cycle_aware_experiment_authorized_independently": True,
        "target_rows_read_from_test": 0,
        "test_predictions_constructed": False,
    }
    write_json(selection, PHASE_ROOT / "outputs" / "frozen_config" / "xlstm_screening_selection.json")

    plot = means.sort_values("rmse_mV")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(9, 5))
    colors = ["#1D4ED8" if model == "LSTM" else "#0F766E" for model in plot["model"]]
    plt.bar(plot["model"], plot["rmse_mV"], color=colors)
    plt.ylabel("验证集 RMSE（mV）")
    plt.title("xLSTM 变体筛选：三折平均 RMSE")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PHASE_ROOT / "outputs" / "figures" / "xlstm_variant_rmse.png", dpi=180)
    plt.close()
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


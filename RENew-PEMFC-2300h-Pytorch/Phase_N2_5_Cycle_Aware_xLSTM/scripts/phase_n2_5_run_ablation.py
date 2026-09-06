"""Run Phase N2.5 core cycle-aware ablations and multi-seed robustness."""

from __future__ import annotations

import json
from pathlib import Path

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
from phase_n2_5_models import load_run, train_model_run


def get_or_train(
    label: str,
    backbone: str,
    period_steps: int | None,
    seed: int,
    fold: str,
    train,
    validation,
    cfg: dict,
) -> tuple[str, dict, dict[str, np.ndarray]]:
    if label == "xLSTM" and seed == 42:
        run_id = f"screening_{backbone}_{fold}_seed42"
        cached = load_run(run_id)
        if cached is None:
            raise RuntimeError(f"Missing xLSTM screening run {run_id}")
        result, arrays = cached
        return run_id, result, arrays
    if label == "xLSTM":
        run_id = f"multiseed_{backbone}_{fold}_seed{seed}"
    elif label == "CA-LSTM":
        run_id = f"ablation_CA_LSTM_P19_{fold}_seed{seed}"
    elif label == "CA-xLSTM":
        run_id = f"ablation_CA_{backbone}_P19_{fold}_seed{seed}"
    elif label == "CA-xLSTM-P15":
        run_id = f"wrong_period_CA_{backbone}_P15_{fold}_seed{seed}"
    else:
        raise KeyError(label)
    result, arrays = train_model_run(
        run_id,
        backbone,
        train,
        validation,
        cfg,
        seed,
        {
            "stage": "cycle_aware_ablation" if label != "CA-xLSTM-P15" else "wrong_period_control",
            "architecture_label": label,
            "backbone": backbone,
            "fold": fold,
            "feature_group": "F2",
            "lookback_hours": 24,
            "horizon_hours": 12,
            "resolution_hours": 0.5,
            "test_predictions_constructed": False,
        },
        period_steps=period_steps,
    )
    return run_id, result, arrays


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_summary = rows.groupby(["model", "backbone", "period_steps", "seed"], dropna=False, as_index=False).agg(
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        r2=("r2", "mean"),
        skill_best=("skill_best", "mean"),
        skill_cycle=("skill_cycle", "mean"),
        worst_fold_skill=("skill_best", "min"),
        positive_skill_fold_count=("skill_best", lambda values: int((values > 0).sum())),
        improved_fold_count_vs_lstm=("improved_vs_lstm", "sum"),
        parameter_count=("parameter_count", "first"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        mean_epoch_time_s=("mean_epoch_time_s", "mean"),
        total_training_time_s=("total_training_time_s", "sum"),
    )
    seed_summary.insert(0, "fold", "mean_3fold")
    seed_summary.insert(0, "row_type", "seed_summary")
    model_summary = seed_summary.groupby(["model", "backbone", "period_steps"], dropna=False, as_index=False).agg(
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        rmse_mV_seed_std=("rmse_mV", "std"),
        rmse_mV_seed_min=("rmse_mV", "min"),
        rmse_mV_seed_max=("rmse_mV", "max"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        skill_best=("skill_best", "mean"),
        skill_best_seed_std=("skill_best", "std"),
        skill_cycle=("skill_cycle", "mean"),
        worst_fold_skill=("worst_fold_skill", "min"),
        positive_skill_fold_seed_count=("positive_skill_fold_count", "sum"),
        improved_fold_seed_count_vs_lstm=("improved_fold_count_vs_lstm", "sum"),
        parameter_count=("parameter_count", "first"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        mean_epoch_time_s=("mean_epoch_time_s", "mean"),
        total_training_time_s=("total_training_time_s", "sum"),
        seed_count=("seed", "count"),
    )
    model_summary["rmse_mV_seed_std"] = model_summary["rmse_mV_seed_std"].fillna(0.0)
    model_summary["skill_best_seed_std"] = model_summary["skill_best_seed_std"].fillna(0.0)
    model_summary.insert(0, "seed", "all_available")
    model_summary.insert(0, "fold", "all_folds")
    model_summary.insert(0, "row_type", "model_summary")
    return seed_summary, model_summary


def main() -> None:
    cfg = load_config()
    selection_path = PHASE_ROOT / "outputs" / "frozen_config" / "xlstm_screening_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("reproduction_check_pass"):
        raise RuntimeError("Reproduction gate is not open")
    backbone = str(selection["selected_xlstm_variant"])
    frame, firewall = load_development_data(cfg, cfg["features"])
    if firewall["target_rows_read_from_test"] != 0:
        raise RuntimeError("TEST FIREWALL VIOLATION")
    windows = fold_windows(frame, cfg)
    alpha_path = Path(cfg["phase_n2_root"]) / "outputs" / "baseline_results" / "ridge_alpha_selection.json"
    ridge_alpha = float(json.loads(alpha_path.read_text(encoding="utf-8"))["selected_ridge_alpha"])

    baseline_by_fold: dict[str, dict] = {}
    for fold, train, validation in windows:
        predictions = simple_predictions(train, validation, ridge_alpha, int(cfg["task"]["empirical_period_steps"]))
        baseline_by_fold[fold] = {
            "predictions": predictions,
            "denominators": baseline_denominators(validation.y_abs, predictions),
        }

    rows: list[dict] = []
    run_index: dict[str, dict[str, str]] = {}
    robustness_seeds = [int(value) for value in cfg["training"]["robustness_seeds"]]
    experiment_specs = [
        ("xLSTM", backbone, None, robustness_seeds),
        ("CA-LSTM", "LSTM", int(cfg["task"]["empirical_period_steps"]), robustness_seeds),
        ("CA-xLSTM", backbone, int(cfg["task"]["empirical_period_steps"]), robustness_seeds),
        ("CA-xLSTM-P15", backbone, int(cfg["task"]["wrong_period_steps"]), [42]),
    ]

    for seed in robustness_seeds:
        for fold, _, validation in windows:
            n2_result, n2_arrays = load_phase_n2_lstm_reference(cfg, seed, fold, validation)
            metrics = add_skill(metric_dict(n2_arrays["y_true"], n2_arrays["y_pred"]), baseline_by_fold[fold]["denominators"])
            key = f"LSTM|{seed}|{fold}"
            run_index[key] = {
                "source": "Phase_N2_Model_Screening",
                "run_id": n2_result["run_id"],
            }
            rows.append(
                {
                    "row_type": "fold",
                    "fold": fold,
                    "seed": seed,
                    "model": "LSTM",
                    "backbone": "LSTM",
                    "period_steps": np.nan,
                    "run_id": n2_result["run_id"],
                    "parameter_count": n2_result["parameter_count"],
                    "peak_gpu_memory_mb": n2_result["peak_gpu_memory_mb"],
                    "mean_epoch_time_s": n2_result["mean_epoch_time_s"],
                    "total_training_time_s": n2_result["total_training_time_s"],
                    "improved_vs_lstm": False,
                    **metrics,
                }
            )

    for label, model_backbone, period_steps, seeds in experiment_specs:
        for seed in seeds:
            for fold, train, validation in windows:
                run_id, result, arrays = get_or_train(
                    label,
                    model_backbone,
                    period_steps,
                    seed,
                    fold,
                    train,
                    validation,
                    cfg,
                )
                _, lstm_arrays = load_phase_n2_lstm_reference(cfg, seed, fold, validation)
                if not np.array_equal(arrays["row_origin_index"], lstm_arrays["row_origin_index"]):
                    raise RuntimeError(f"Matched-Origin violation in {run_id}")
                if not np.array_equal(arrays["y_true"], lstm_arrays["y_true"]):
                    raise RuntimeError(f"Target mismatch in {run_id}")
                metrics = add_skill(metric_dict(arrays["y_true"], arrays["y_pred"]), baseline_by_fold[fold]["denominators"])
                lstm_rmse = metric_dict(lstm_arrays["y_true"], lstm_arrays["y_pred"])["rmse_V"]
                key = f"{label}|{seed}|{fold}"
                run_index[key] = {"source": "Phase_N2_5_Cycle_Aware_xLSTM", "run_id": run_id}
                rows.append(
                    {
                        "row_type": "fold",
                        "fold": fold,
                        "seed": seed,
                        "model": label,
                        "backbone": model_backbone,
                        "period_steps": period_steps if period_steps is not None else np.nan,
                        "run_id": run_id,
                        "parameter_count": result["parameter_count"],
                        "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                        "mean_epoch_time_s": result["mean_epoch_time_s"],
                        "total_training_time_s": result["total_training_time_s"],
                        "improved_vs_lstm": metrics["rmse_V"] < lstm_rmse,
                        **metrics,
                    }
                )

    fold_rows = pd.DataFrame(rows)
    seed_summary, model_summary = summarize(fold_rows)
    full = pd.concat([fold_rows, seed_summary, model_summary], ignore_index=True, sort=False)
    write_csv(full, PHASE_ROOT / "outputs" / "cycle_aware_ablation" / "cycle_aware_ablation.csv")
    write_csv(
        full[full["model"].isin(["CA-xLSTM", "CA-xLSTM-P15"])].copy(),
        PHASE_ROOT / "outputs" / "cycle_aware_ablation" / "wrong_period_control.csv",
    )
    write_csv(
        full[full["model"].isin(["LSTM", "xLSTM", "CA-LSTM", "CA-xLSTM"])].copy(),
        PHASE_ROOT / "outputs" / "multi_seed_robustness" / "multi_seed_architecture_robustness.csv",
    )
    efficiency = model_summary[
        [
            "model",
            "backbone",
            "period_steps",
            "seed_count",
            "parameter_count",
            "peak_gpu_memory_mb",
            "mean_epoch_time_s",
            "total_training_time_s",
            "rmse_mV",
            "rmse_mV_seed_std",
            "skill_best",
        ]
    ].copy()
    efficiency["implementation"] = np.where(
        efficiency["model"].eq("LSTM"),
        "Phase_N2_canonical_reference",
        "local_auditable_phase_n2_5_v1",
    )
    write_csv(efficiency, PHASE_ROOT / "outputs" / "model_comparison" / "model_efficiency_comparison.csv")
    write_json(run_index, PHASE_ROOT / "outputs" / "frozen_config" / "architecture_run_index.json")
    write_json(
        {
            "status": "PASS",
            "selected_xlstm_backbone": backbone,
            "core_models": ["LSTM", "xLSTM", "CA-LSTM", "CA-xLSTM"],
            "wrong_period_control": "CA-xLSTM-P15",
            "core_seeds": robustness_seeds,
            "wrong_period_seeds": [42],
            "target_rows_read_from_test": 0,
            "test_predictions_constructed": False,
        },
        PHASE_ROOT / "outputs" / "frozen_config" / "ablation_completion.json",
    )
    print(model_summary[["model", "backbone", "seed_count", "rmse_mV", "skill_best"]].to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()

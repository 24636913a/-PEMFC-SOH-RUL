"""Run only screening-qualified Phase N2.6 candidates with the frozen three seeds."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from phase_n2_6_common import (
    PHASE_ROOT,
    add_skill_lstm,
    aggregate_seed_rows,
    baseline_by_fold,
    family_for_config,
    load_config,
    load_development_protocol,
    load_local_run,
    load_phase_n2_lstm_reference,
    metric_dict,
    write_csv,
    write_json,
)
from phase_n2_6_models import train_model_run


def main() -> None:
    cfg = load_config()
    selection = json.loads((PHASE_ROOT / "outputs" / "frozen_config" / "screening_selection.json").read_text(encoding="utf-8"))
    if not selection.get("lstm_reproduction_pass"):
        raise RuntimeError("LSTM reproduction gate is closed")
    top_configs = [str(value) for value in selection.get("top_configs_for_multiseed", [])]
    _, firewall, windows = load_development_protocol(cfg)
    if firewall.get("target_rows_read_from_test") != 0:
        raise RuntimeError("TEST FIREWALL VIOLATION")
    baselines = baseline_by_fold(cfg, windows)
    seeds = [int(value) for value in cfg["training"]["robustness_seeds"]]
    rows: list[dict[str, object]] = []
    run_index: dict[str, dict[str, str]] = {}

    for seed in seeds:
        for fold, _, validation in windows:
            result, arrays = load_phase_n2_lstm_reference(cfg, seed, fold, validation)
            metrics_raw = metric_dict(arrays["y_true"], arrays["y_pred"])
            metrics = add_skill_lstm(metrics_raw, metrics_raw["rmse_V"], baselines[fold]["denominators"])
            rows.append(
                {
                    "row_type": "fold",
                    "fold": fold,
                    "seed": seed,
                    "family": "LSTM",
                    "config_name": "LSTM",
                    "run_id": result["run_id"],
                    "parameter_count": result["parameter_count"],
                    "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                    "mean_epoch_time_s": result["mean_epoch_time_s"],
                    "total_training_time_s": result["total_training_time_s"],
                    "improved_vs_lstm": False,
                    **metrics,
                }
            )
            run_index[f"LSTM|{seed}|{fold}"] = {"source": "Phase_N2_Model_Screening", "run_id": result["run_id"]}

    screening_index = json.loads((PHASE_ROOT / "outputs" / "frozen_config" / "screening_run_index.json").read_text(encoding="utf-8"))
    for config_name in top_configs:
        family = family_for_config(cfg, config_name)
        for seed in seeds:
            for fold, train, validation in windows:
                if seed == 42:
                    entry = screening_index[f"{config_name}|42|{fold}"]
                    cached = load_local_run(entry["run_id"])
                    if cached is None:
                        raise RuntimeError(f"Missing screening run: {entry['run_id']}")
                    result, arrays = cached
                    run_id = entry["run_id"]
                else:
                    run_id = f"multiseed_{config_name}_{fold}_seed{seed}"
                    result, arrays = train_model_run(
                        run_id,
                        config_name,
                        train,
                        validation,
                        cfg,
                        seed,
                        {
                            "stage": "multi_seed_backbone_comparison",
                            "family": family,
                            "fold": fold,
                            "implementation": cfg["architectures"][config_name]["implementation"],
                            "feature_group": "F2",
                            "lookback_hours": 24,
                            "horizon_hours": 12,
                            "resolution_hours": 0.5,
                        },
                    )
                lstm_result, lstm_arrays = load_phase_n2_lstm_reference(cfg, seed, fold, validation)
                if not np.array_equal(arrays["row_origin_index"], lstm_arrays["row_origin_index"]):
                    raise RuntimeError(f"Matched-Origin violation for {run_id}")
                if not np.array_equal(arrays["y_true"], lstm_arrays["y_true"]):
                    raise RuntimeError(f"Target mismatch for {run_id}")
                lstm_rmse = metric_dict(lstm_arrays["y_true"], lstm_arrays["y_pred"])["rmse_V"]
                metrics = add_skill_lstm(metric_dict(arrays["y_true"], arrays["y_pred"]), lstm_rmse, baselines[fold]["denominators"])
                rows.append(
                    {
                        "row_type": "fold",
                        "fold": fold,
                        "seed": seed,
                        "family": family,
                        "config_name": config_name,
                        "run_id": run_id,
                        "parameter_count": result["parameter_count"],
                        "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                        "mean_epoch_time_s": result["mean_epoch_time_s"],
                        "total_training_time_s": result["total_training_time_s"],
                        "improved_vs_lstm": bool(metrics["rmse_V"] < lstm_rmse),
                        **metrics,
                    }
                )
                run_index[f"{config_name}|{seed}|{fold}"] = {"source": "Phase_N2_6_Modern_Temporal_Expansion", "run_id": run_id}

    fold_rows = pd.DataFrame(rows)
    seed_summary, model_summary = aggregate_seed_rows(fold_rows)
    full = pd.concat([fold_rows, seed_summary, model_summary], ignore_index=True, sort=False)
    write_csv(full, PHASE_ROOT / "outputs" / "multi_seed_robustness" / "multi_seed_backbone_comparison.csv")
    write_json(run_index, PHASE_ROOT / "outputs" / "frozen_config" / "multiseed_run_index.json")
    write_json(
        {
            "status": "PASS",
            "top_configs": top_configs,
            "seeds": seeds,
            "trained_new_model_count": len(top_configs),
            "target_rows_read_from_test": 0,
            "test_predictions_constructed": False,
            "test_metrics_computed": False,
        },
        PHASE_ROOT / "outputs" / "frozen_config" / "multiseed_completion.json",
    )
    print(model_summary[["family", "config_name", "seed_count", "rmse_mV", "mean_skill_lstm"]].to_json(orient="records"))


if __name__ == "__main__":
    main()

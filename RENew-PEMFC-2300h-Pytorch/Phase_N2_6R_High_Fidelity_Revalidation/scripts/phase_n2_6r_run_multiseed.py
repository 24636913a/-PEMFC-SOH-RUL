"""Run the preregistered multi-seed gate for eligible high-fidelity models."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from phase_n2_6r_common import (
    PHASE_ROOT,
    load_config,
    load_development_protocol,
    load_local_run,
    load_phase_n2_lstm_reference,
    metric_dict,
    verify_no_test_arrays,
    write_csv,
    write_json,
)
from phase_n2_6r_models import train_model_run


def metric_row(
    model: str,
    config_name: str,
    seed: int,
    fold: str,
    result: dict,
    arrays: dict[str, np.ndarray],
    lstm_rmse_mV: float,
    source: str,
) -> dict:
    metrics = metric_dict(arrays["y_true"], arrays["y_pred"])
    return {
        "row_type": "fold_seed",
        "model": model,
        "family": model,
        "config_name": config_name,
        "seed": seed,
        "fold": fold,
        **metrics,
        "rmse_improvement_pct_vs_paired_lstm": 100.0 * (lstm_rmse_mV - metrics["rmse_mV"]) / lstm_rmse_mV,
        "parameter_count": int(result["parameter_count"]),
        "peak_gpu_memory_mb": float(result.get("peak_gpu_memory_mb", np.nan)),
        "mean_epoch_time_s": float(result.get("mean_epoch_time_s", np.nan)),
        "total_training_time_s": float(result.get("total_training_time_s", np.nan)),
        "best_epoch": int(result.get("best_epoch", -1)),
        "epochs_ran": int(result.get("epochs_ran", -1)),
        "artifact_source": source,
        "implementation_fidelity_status": "REFERENCE" if model == "LSTM" else "PASS",
    }


def add_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = {
        "rmse_V": ("rmse_V", "mean"),
        "rmse_mV": ("rmse_mV", "mean"),
        "mae_V": ("mae_V", "mean"),
        "mae_mV": ("mae_mV", "mean"),
        "bias_V": ("bias_V", "mean"),
        "bias_mV": ("bias_mV", "mean"),
        "r2": ("r2", "mean"),
        "rmse_std_mV": ("rmse_mV", "std"),
        "parameter_count": ("parameter_count", "first"),
        "peak_gpu_memory_mb": ("peak_gpu_memory_mb", "max"),
        "mean_epoch_time_s": ("mean_epoch_time_s", "mean"),
        "total_training_time_s": ("total_training_time_s", "sum"),
    }
    seed_summary = frame.groupby(["model", "family", "config_name", "seed"], as_index=False).agg(**metric_columns)
    seed_summary.insert(0, "row_type", "seed_mean_3fold")
    seed_summary["fold"] = "mean_3fold"
    seed_summary["artifact_source"] = "summary"
    seed_summary["implementation_fidelity_status"] = np.where(seed_summary["model"].eq("LSTM"), "REFERENCE", "PASS")

    fold_summary = frame.groupby(["model", "family", "config_name", "fold"], as_index=False).agg(**metric_columns)
    fold_summary.insert(0, "row_type", "fold_mean_3seed")
    fold_summary["seed"] = -1
    fold_summary["artifact_source"] = "summary"
    fold_summary["implementation_fidelity_status"] = np.where(fold_summary["model"].eq("LSTM"), "REFERENCE", "PASS")

    overall = frame.groupby(["model", "family", "config_name"], as_index=False).agg(**metric_columns)
    overall.insert(0, "row_type", "overall_mean_9runs")
    overall["seed"] = -1
    overall["fold"] = "all_3fold_3seed"
    overall["artifact_source"] = "summary"
    overall["implementation_fidelity_status"] = np.where(overall["model"].eq("LSTM"), "REFERENCE", "PASS")

    overall_map = overall.set_index("model")["rmse_mV"]
    lstm_overall = float(overall_map["LSTM"])
    for table in (seed_summary, fold_summary, overall):
        table["rmse_improvement_pct_vs_paired_lstm"] = 100.0 * (lstm_overall - table["rmse_mV"]) / lstm_overall
    return pd.concat([frame, seed_summary, fold_summary, overall], ignore_index=True, sort=False)


def main() -> None:
    cfg = load_config()
    selection = json.loads((PHASE_ROOT / "outputs/frozen_config/screening_selection.json").read_text(encoding="utf-8"))
    if selection.get("status") != "PASS":
        raise RuntimeError("Screening gate did not pass")
    eligible = list(selection.get("eligible_for_multiseed", []))
    if eligible != ["timesnet_hf_t1"]:
        raise RuntimeError(f"Unexpected preregistered eligible list: {eligible}")

    _, firewall, windows = load_development_protocol(cfg)
    rows: list[dict] = []
    run_index: dict[str, dict[str, str]] = {}
    seeds = [int(value) for value in cfg["training"]["robustness_seeds"]]

    for fold, train, validation in windows:
        for seed in seeds:
            lstm_result, lstm_arrays = load_phase_n2_lstm_reference(cfg, seed, fold, validation)
            verify_no_test_arrays(lstm_arrays, cfg, f"PhaseN2_LSTM_{fold}_seed{seed}")
            lstm_metrics = metric_dict(lstm_arrays["y_true"], lstm_arrays["y_pred"])
            rows.append(
                metric_row(
                    "LSTM", "LSTM", seed, fold, lstm_result, lstm_arrays,
                    lstm_metrics["rmse_mV"], "Phase_N2_frozen_reference",
                )
            )
            run_index[f"LSTM|{seed}|{fold}"] = {
                "run_id": str(lstm_result["run_id"]),
                "source": "Phase_N2_Model_Screening",
            }

            if seed == 42:
                run_id = f"screening_timesnet_hf_t1_{fold}_seed42"
                cached = load_local_run(run_id)
                if cached is None:
                    raise RuntimeError(f"Missing seed-42 screening artifact: {run_id}")
                result, arrays = cached
            else:
                run_id = f"multiseed_timesnet_hf_t1_{fold}_seed{seed}"
                result, arrays = train_model_run(
                    run_id,
                    "timesnet_hf_t1",
                    train,
                    validation,
                    cfg,
                    seed,
                    {
                        "stage": "multiseed_revalidation",
                        "fold": fold,
                        "implementation_source": cfg["official_sources"]["timesnet"]["repository"],
                        "implementation_commit": cfg["official_sources"]["timesnet"]["commit"],
                        "implementation_fidelity_status": "PASS",
                        "lookback_hours": 24,
                        "horizon_hours": 12,
                        "resolution_hours": 0.5,
                    },
                )
            verify_no_test_arrays(arrays, cfg, run_id)
            if not np.array_equal(arrays["row_origin_index"], lstm_arrays["row_origin_index"]):
                raise RuntimeError(f"Matched-Origin violation: {run_id}")
            if not np.array_equal(arrays["y_true"], lstm_arrays["y_true"]):
                raise RuntimeError(f"Matched target violation: {run_id}")
            rows.append(
                metric_row(
                    "TimesNet", "timesnet_hf_t1", seed, fold, result, arrays,
                    lstm_metrics["rmse_mV"], "Phase_N2_6R",
                )
            )
            run_index[f"TimesNet|{seed}|{fold}"] = {"run_id": run_id, "source": "Phase_N2_6R"}

    fold_seed = pd.DataFrame(rows)
    comparison = add_summaries(fold_seed)
    write_csv(comparison, PHASE_ROOT / "outputs/multiseed/multiseed_comparison.csv")
    write_json(run_index, PHASE_ROOT / "outputs/frozen_config/multiseed_run_index.json")
    write_json(firewall, PHASE_ROOT / "outputs/implementation_audit/test_firewall_audit.json")
    overall = comparison.loc[comparison["row_type"] == "overall_mean_9runs", ["model", "rmse_mV", "mae_mV", "r2"]]
    print(json.dumps({"status": "PASS", "overall": overall.to_dict(orient="records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

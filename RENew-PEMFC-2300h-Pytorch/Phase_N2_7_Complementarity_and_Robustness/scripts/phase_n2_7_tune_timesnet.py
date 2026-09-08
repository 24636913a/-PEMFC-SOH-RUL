"""Finite, preregistered TimesNet robustness screen and multi-seed confirmation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from phase_n2_7_common import (
    FOLDS,
    PHASE_ROOT,
    SEEDS,
    development_stage_bounds,
    load_config,
    load_protocol,
    load_reference_prediction,
    metric_dict,
    stage_labels,
    write_csv,
    write_json,
)
from phase_n2_7_models import train_model_run


def get_predictions(
    cfg: dict,
    config_name: str,
    spec: dict,
    seed: int,
    fold: str,
    train,
    validation,
) -> tuple[dict, dict[str, np.ndarray], str]:
    if config_name == "T0_original":
        result, arrays, source = load_reference_prediction(cfg, "TimesNet", seed, fold)
        return result, arrays, source
    run_id = f"n27_timesnet_{config_name}_{fold}_seed{seed}"
    result, arrays = train_model_run(
        run_id=run_id,
        family="TimesNet",
        train=train,
        validation=validation,
        cfg=cfg,
        seed=seed,
        training_spec=spec,
        metadata={
            "stage": "timesnet_finite_robustness_tuning",
            "config_name": config_name,
            "fold": fold,
            "implementation_fidelity": "Phase_N2_6R_official_TimesNet_unchanged",
            "preprocessing": "P0_raw",
            "lookback_hours": 24,
            "horizon_hours": 12,
            "resolution_hours": 0.5,
        },
    )
    return result, arrays, "Phase_N2_7_local"


def summarize_config(
    config_name: str,
    seed: int,
    predictions: dict[str, dict[str, np.ndarray]],
    validation_map: dict,
    bounds: tuple[float, float, float, float],
) -> dict:
    row: dict[str, object] = {"config_name": config_name, "seed": seed}
    metrics_by_fold = {}
    late_true, late_pred = [], []
    for fold in FOLDS:
        arrays = predictions[fold]
        metrics = metric_dict(arrays["y_true"], arrays["y_pred"])
        metrics_by_fold[fold] = metrics
        row[f"{fold}_rmse_mV"] = metrics["rmse_mV"]
        row[f"{fold}_mae_mV"] = metrics["mae_mV"]
        row[f"{fold}_r2"] = metrics["r2"]
        validation = validation_map[fold]
        if not np.array_equal(validation.row_origin_index, arrays["row_origin_index"]):
            raise RuntimeError(f"Origin mismatch while summarizing {config_name}, seed={seed}, {fold}")
        mask = stage_labels(validation.origin_age_h, bounds) == "Late Development"
        if np.any(mask):
            late_true.append(arrays["y_true"][mask])
            late_pred.append(arrays["y_pred"][mask])
    for key in ["rmse_mV", "mae_mV", "r2", "bias_mV"]:
        row[f"mean_{key}"] = float(np.mean([metrics_by_fold[fold][key] for fold in FOLDS]))
    late_metrics = metric_dict(np.concatenate(late_true), np.concatenate(late_pred))
    row["late_rmse_mV"] = late_metrics["rmse_mV"]
    row["late_mae_mV"] = late_metrics["mae_mV"]
    row["late_r2"] = late_metrics["r2"]
    return row


def add_gate_columns(row: dict, reference: dict, cfg: dict) -> None:
    selection = cfg["selection"]
    row["fold3_improvement_fraction_vs_T0"] = (
        float(reference["fold_3_rmse_mV"]) - float(row["fold_3_rmse_mV"])
    ) / float(reference["fold_3_rmse_mV"])
    row["mean_change_fraction_vs_T0"] = (
        float(row["mean_rmse_mV"]) - float(reference["mean_rmse_mV"])
    ) / float(reference["mean_rmse_mV"])
    row["late_improvement_fraction_vs_T0"] = (
        float(reference["late_rmse_mV"]) - float(row["late_rmse_mV"])
    ) / float(reference["late_rmse_mV"])
    row["gate_A_fold3_at_least_10pct_better"] = bool(
        row["fold3_improvement_fraction_vs_T0"] >= float(selection["timesnet_fold3_min_improvement"])
    )
    row["gate_B_mean_not_worse_over_2pct"] = bool(
        row["mean_change_fraction_vs_T0"] <= float(selection["timesnet_mean_max_worsening"])
    )
    row["gate_C_late_rmse_lower"] = bool(row["late_rmse_mV"] < reference["late_rmse_mV"])
    row["passes_all_timesnet_tuning_gates"] = bool(
        row["gate_A_fold3_at_least_10pct_better"]
        and row["gate_B_mean_not_worse_over_2pct"]
        and row["gate_C_late_rmse_lower"]
    )


def mean_across_seeds(rows: list[dict], config_name: str, stage: str) -> dict:
    numeric_columns = [
        key
        for key in rows[0]
        if key.startswith("fold_") and (key.endswith("_rmse_mV") or key.endswith("_mae_mV") or key.endswith("_r2"))
    ] + [
        "mean_rmse_mV", "mean_mae_mV", "mean_r2", "mean_bias_mV",
        "late_rmse_mV", "late_mae_mV", "late_r2",
    ]
    output: dict[str, object] = {"config_name": config_name, "seed": "3-seed mean", "evaluation_stage": stage}
    for key in numeric_columns:
        output[key] = float(np.mean([float(row[key]) for row in rows]))
    return output


def main() -> None:
    cfg = load_config()
    _, firewall, windows = load_protocol(cfg)
    validation_map = {fold: validation for fold, _, validation in windows}
    window_map = {fold: (train, validation) for fold, train, validation in windows}
    bounds = development_stage_bounds(windows)
    all_rows: list[dict] = []
    run_index: dict[str, dict[str, dict[str, str]]] = {}

    screen_summaries: dict[str, dict] = {}
    for config_name, spec in cfg["timesnet_tuning"].items():
        fold_predictions = {}
        run_index.setdefault(config_name, {}).setdefault("42", {})
        for fold in FOLDS:
            train, validation = window_map[fold]
            result, arrays, source = get_predictions(cfg, config_name, spec, 42, fold, train, validation)
            fold_predictions[fold] = arrays
            run_index[config_name]["42"][fold] = result["run_id"]
        summary = summarize_config(config_name, 42, fold_predictions, validation_map, bounds)
        summary.update({
            "evaluation_stage": "seed42_preregistered_screen",
            "loss": spec["loss"],
            "huber_delta": spec.get("huber_delta"),
            "learning_rate": spec["learning_rate"],
            "dropout": spec["dropout"],
            "weight_decay": spec["weight_decay"],
            "prediction_source": "historical_Phase_N2_6R" if config_name == "T0_original" else "Phase_N2_7_new",
        })
        screen_summaries[config_name] = summary
        print(json.dumps({"screen_complete": config_name, "mean_rmse_mV": summary["mean_rmse_mV"], "fold3_rmse_mV": summary["fold_3_rmse_mV"]}), flush=True)

    reference42 = screen_summaries["T0_original"]
    for row in screen_summaries.values():
        add_gate_columns(row, reference42, cfg)
        if row["config_name"] == "T0_original":
            row["passes_all_timesnet_tuning_gates"] = False
        all_rows.append(row)
    passing = [row for name, row in screen_summaries.items() if name != "T0_original" and row["passes_all_timesnet_tuning_gates"]]
    screen_winner = min(passing, key=lambda row: (row["fold_3_rmse_mV"], row["late_rmse_mV"], row["mean_rmse_mV"])) if passing else reference42
    screen_winner_name = str(screen_winner["config_name"])

    # Always materialize the 3-seed T0 reference; only the gate-passing screen winner gets new multi-seed runs.
    multi_candidates = ["T0_original"] + ([screen_winner_name] if screen_winner_name != "T0_original" else [])
    multiseed_summaries: dict[str, dict] = {}
    per_config_seed_rows: dict[str, list[dict]] = {}
    for config_name in multi_candidates:
        spec = cfg["timesnet_tuning"][config_name]
        seed_rows = []
        for seed in SEEDS:
            if seed == 42:
                row = dict(screen_summaries[config_name])
                row["evaluation_stage"] = "multiseed_confirmation_seed"
            else:
                fold_predictions = {}
                run_index.setdefault(config_name, {}).setdefault(str(seed), {})
                for fold in FOLDS:
                    train, validation = window_map[fold]
                    result, arrays, _ = get_predictions(cfg, config_name, spec, seed, fold, train, validation)
                    fold_predictions[fold] = arrays
                    run_index[config_name][str(seed)][fold] = result["run_id"]
                row = summarize_config(config_name, seed, fold_predictions, validation_map, bounds)
                row.update({
                    "evaluation_stage": "multiseed_confirmation_seed",
                    "loss": spec["loss"],
                    "huber_delta": spec.get("huber_delta"),
                    "learning_rate": spec["learning_rate"],
                    "dropout": spec["dropout"],
                    "weight_decay": spec["weight_decay"],
                    "prediction_source": "historical_Phase_N2_6R" if config_name == "T0_original" else "Phase_N2_7_new",
                })
            seed_rows.append(row)
            all_rows.append(row)
        per_config_seed_rows[config_name] = seed_rows
        mean_row = mean_across_seeds(seed_rows, config_name, "multiseed_confirmation_mean")
        mean_row.update({
            "loss": spec["loss"],
            "huber_delta": spec.get("huber_delta"),
            "learning_rate": spec["learning_rate"],
            "dropout": spec["dropout"],
            "weight_decay": spec["weight_decay"],
            "prediction_source": "three_seed_macro_mean",
        })
        multiseed_summaries[config_name] = mean_row

    reference_multi = multiseed_summaries["T0_original"]
    for row in multiseed_summaries.values():
        add_gate_columns(row, reference_multi, cfg)
        if row["config_name"] == "T0_original":
            row["passes_all_timesnet_tuning_gates"] = False
        all_rows.append(row)

    if screen_winner_name != "T0_original" and multiseed_summaries[screen_winner_name]["passes_all_timesnet_tuning_gates"]:
        final_name = screen_winner_name
        final_reason = "seed42 gates passed and all three gates were reconfirmed on the 3-seed macro mean"
    else:
        final_name = "T0_original"
        final_reason = "no new configuration passed both seed42 screening and 3-seed confirmation; original high-fidelity TimesNet retained"

    ordered_columns = [
        "evaluation_stage", "config_name", "seed", "loss", "huber_delta", "learning_rate", "dropout", "weight_decay",
        "fold_1_rmse_mV", "fold_2_rmse_mV", "fold_3_rmse_mV", "mean_rmse_mV", "mean_mae_mV", "mean_r2",
        "late_rmse_mV", "late_mae_mV", "late_r2", "fold3_improvement_fraction_vs_T0", "mean_change_fraction_vs_T0",
        "late_improvement_fraction_vs_T0", "gate_A_fold3_at_least_10pct_better", "gate_B_mean_not_worse_over_2pct",
        "gate_C_late_rmse_lower", "passes_all_timesnet_tuning_gates", "prediction_source",
    ]
    output = pd.DataFrame(all_rows)
    output = output[[column for column in ordered_columns if column in output.columns] + [column for column in output.columns if column not in ordered_columns]]
    write_csv(output, PHASE_ROOT / "outputs/timesnet_tuning/timesnet_robustness_tuning.csv")
    write_json(run_index, PHASE_ROOT / "outputs/timesnet_tuning/timesnet_run_index.json")
    write_json(
        {
            "status": "PASS",
            "screen_winner": screen_winner_name,
            "seed42_passing_configs": [str(row["config_name"]) for row in passing],
            "final_timesnet_tuning": final_name,
            "final_reason": final_reason,
            "final_3seed_metrics": multiseed_summaries[final_name],
            "reference_3seed_metrics": reference_multi,
            "test_firewall": firewall,
        },
        PHASE_ROOT / "outputs/timesnet_tuning/timesnet_tuning_selection.json",
    )
    print(json.dumps({"status": "PASS", "screen_winner": screen_winner_name, "final_timesnet_tuning": final_name}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

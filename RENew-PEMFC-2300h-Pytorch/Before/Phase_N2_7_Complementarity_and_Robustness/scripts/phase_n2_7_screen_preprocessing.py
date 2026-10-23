"""Strictly causal voltage-state preprocessing screen and multi-seed confirmation."""

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
    preprocessing_windows,
    stage_labels,
    write_csv,
    write_json,
)
from phase_n2_7_models import load_local_run, train_model_run


def summarize(
    family: str,
    preprocessing_name: str,
    seed: int,
    predictions: dict[str, dict[str, np.ndarray]],
    validation_map: dict,
    bounds: tuple[float, float, float, float],
) -> dict:
    row: dict[str, object] = {"family": family, "preprocessing": preprocessing_name, "seed": seed}
    fold_metrics = {}
    late_true, late_pred = [], []
    for fold in FOLDS:
        arrays = predictions[fold]
        metrics = metric_dict(arrays["y_true"], arrays["y_pred"])
        fold_metrics[fold] = metrics
        row[f"{fold}_rmse_mV"] = metrics["rmse_mV"]
        row[f"{fold}_mae_mV"] = metrics["mae_mV"]
        row[f"{fold}_r2"] = metrics["r2"]
        validation = validation_map[fold]
        if not np.array_equal(validation.row_origin_index, arrays["row_origin_index"]):
            raise RuntimeError(f"Origin mismatch for {family}, {preprocessing_name}, seed={seed}, {fold}")
        mask = stage_labels(validation.origin_age_h, bounds) == "Late Development"
        if np.any(mask):
            late_true.append(arrays["y_true"][mask])
            late_pred.append(arrays["y_pred"][mask])
    for key in ["rmse_mV", "mae_mV", "r2", "bias_mV"]:
        row[f"mean_{key}"] = float(np.mean([fold_metrics[fold][key] for fold in FOLDS]))
    late = metric_dict(np.concatenate(late_true), np.concatenate(late_pred))
    row["late_rmse_mV"] = late["rmse_mV"]
    row["late_mae_mV"] = late["mae_mV"]
    row["late_r2"] = late["r2"]
    return row


def add_gates(row: dict, raw: dict, cfg: dict) -> None:
    row["mean_improvement_fraction_vs_raw"] = (
        float(raw["mean_rmse_mV"]) - float(row["mean_rmse_mV"])
    ) / float(raw["mean_rmse_mV"])
    row["fold3_change_fraction_vs_raw"] = (
        float(row["fold_3_rmse_mV"]) - float(raw["fold_3_rmse_mV"])
    ) / float(raw["fold_3_rmse_mV"])
    row["late_change_fraction_vs_raw"] = (
        float(row["late_rmse_mV"]) - float(raw["late_rmse_mV"])
    ) / float(raw["late_rmse_mV"])
    row["gate_mean_improvement_at_least_2pct"] = bool(
        row["mean_improvement_fraction_vs_raw"] >= float(cfg["selection"]["preprocessing_mean_min_improvement"])
    )
    row["gate_fold3_not_worse_over_5pct"] = bool(
        row["fold3_change_fraction_vs_raw"] <= float(cfg["selection"]["preprocessing_fold3_max_worsening"])
    )
    row["gate_late_not_worse"] = bool(row["late_change_fraction_vs_raw"] <= 0.0)
    row["passes_all_preprocessing_gates"] = bool(
        row["gate_mean_improvement_at_least_2pct"]
        and row["gate_fold3_not_worse_over_5pct"]
        and row["gate_late_not_worse"]
    )


def metric_mean(rows: list[dict], family: str, preprocessing_name: str) -> dict:
    keys = [
        f"fold_{fold}_{metric}"
        for fold in [1, 2, 3]
        for metric in ["rmse_mV", "mae_mV", "r2"]
    ] + ["mean_rmse_mV", "mean_mae_mV", "mean_r2", "mean_bias_mV", "late_rmse_mV", "late_mae_mV", "late_r2"]
    output: dict[str, object] = {
        "family": family,
        "preprocessing": preprocessing_name,
        "seed": "3-seed mean",
        "evaluation_stage": "multiseed_confirmation_mean",
    }
    for key in keys:
        output[key] = float(np.mean([float(row[key]) for row in rows]))
    return output


def load_raw_prediction(cfg: dict, family: str, final_timesnet_config: str, seed: int, fold: str):
    if family == "LSTM":
        result, arrays, source = load_reference_prediction(cfg, "LSTM", seed, fold)
        return result, arrays, source
    if final_timesnet_config == "T0_original":
        result, arrays, source = load_reference_prediction(cfg, "TimesNet", seed, fold)
        return result, arrays, source
    run_id = f"n27_timesnet_{final_timesnet_config}_{fold}_seed{seed}"
    loaded = load_local_run(run_id, cfg)
    if loaded is None:
        raise RuntimeError(f"Missing tuned TimesNet run: {run_id}")
    result, arrays = loaded
    return result, arrays, "Phase_N2_7_tuned"


def main() -> None:
    cfg = load_config()
    frame, firewall, raw_windows = load_protocol(cfg)
    filter_audit = json.loads((PHASE_ROOT / "outputs/integrity_audit/causal_filter_audit.json").read_text(encoding="utf-8"))
    if filter_audit.get("status") != "PASS":
        raise RuntimeError("Causal preprocessing screen is closed because the filter audit did not pass")
    tuning_selection = json.loads((PHASE_ROOT / "outputs/timesnet_tuning/timesnet_tuning_selection.json").read_text(encoding="utf-8"))
    final_timesnet_config = str(tuning_selection["final_timesnet_tuning"])
    timesnet_spec = cfg["timesnet_tuning"][final_timesnet_config]
    bounds = development_stage_bounds(raw_windows)
    processed = {
        name: preprocessing_windows(frame, raw_windows, cfg, name)
        for name in cfg["preprocessing"]
    }
    maps = {
        name: {fold: (train, validation) for fold, train, validation in windows}
        for name, windows in processed.items()
    }
    raw_validation_map = {fold: validation for fold, _, validation in raw_windows}
    run_index: dict[str, dict] = {}
    all_rows: list[dict] = []
    screen: dict[tuple[str, str], dict] = {}

    for family in ["LSTM", "TimesNet"]:
        for preprocessing_name in cfg["preprocessing"]:
            predictions = {}
            run_index.setdefault(family, {}).setdefault(preprocessing_name, {}).setdefault("42", {})
            for fold in FOLDS:
                if preprocessing_name == "P0_raw":
                    result, arrays, source = load_raw_prediction(cfg, family, final_timesnet_config, 42, fold)
                else:
                    train, validation = maps[preprocessing_name][fold]
                    training_spec = (
                        timesnet_spec
                        if family == "TimesNet"
                        else {
                            "loss": "MSE", "huber_delta": None,
                            "learning_rate": cfg["training"]["default_learning_rate"],
                            "dropout": 0.1,
                            "weight_decay": cfg["training"]["default_weight_decay"],
                        }
                    )
                    run_id = f"n27_preproc_{family.lower()}_{preprocessing_name}_{fold}_seed42"
                    result, arrays = train_model_run(
                        run_id=run_id,
                        family=family,
                        train=train,
                        validation=validation,
                        cfg=cfg,
                        seed=42,
                        training_spec=training_spec,
                        metadata={
                            "stage": "causal_preprocessing_screen",
                            "config_name": final_timesnet_config if family == "TimesNet" else "Frozen_LSTM",
                            "preprocessing": preprocessing_name,
                            "fold": fold,
                            "lookback_hours": 24,
                            "horizon_hours": 12,
                            "resolution_hours": 0.5,
                            "target_is_raw": True,
                            "load_features_are_raw": True,
                        },
                    )
                    source = "Phase_N2_7_new"
                predictions[fold] = arrays
                run_index[family][preprocessing_name]["42"][fold] = result["run_id"]
            summary = summarize(family, preprocessing_name, 42, predictions, raw_validation_map, bounds)
            summary.update({
                "evaluation_stage": "seed42_causal_preprocessing_screen",
                "filter_method": cfg["preprocessing"][preprocessing_name]["method"],
                "target_is_raw": True,
                "load_features_are_raw": True,
                "timesnet_tuning": final_timesnet_config if family == "TimesNet" else "not_applicable",
                "prediction_source": source,
            })
            screen[(family, preprocessing_name)] = summary
            print(json.dumps({"screen_complete": family, "preprocessing": preprocessing_name, "mean_rmse_mV": summary["mean_rmse_mV"], "fold3_rmse_mV": summary["fold_3_rmse_mV"]}), flush=True)

    screen_winners: dict[str, str] = {}
    for family in ["LSTM", "TimesNet"]:
        raw = screen[(family, "P0_raw")]
        family_candidates = []
        for preprocessing_name in cfg["preprocessing"]:
            row = screen[(family, preprocessing_name)]
            add_gates(row, raw, cfg)
            if preprocessing_name == "P0_raw":
                row["passes_all_preprocessing_gates"] = False
            elif row["passes_all_preprocessing_gates"]:
                family_candidates.append(row)
            all_rows.append(row)
        screen_winners[family] = str(min(
            family_candidates,
            key=lambda row: (row["mean_rmse_mV"], row["fold_3_rmse_mV"], row["late_rmse_mV"]),
        )["preprocessing"]) if family_candidates else "P0_raw"

    final_preprocessing: dict[str, str] = {}
    multi_summary: dict[tuple[str, str], dict] = {}
    multi_rows_for_csv: list[dict] = []
    for family in ["LSTM", "TimesNet"]:
        candidates = ["P0_raw"] + ([screen_winners[family]] if screen_winners[family] != "P0_raw" else [])
        for preprocessing_name in candidates:
            seed_rows = []
            for seed in SEEDS:
                if seed == 42:
                    row = dict(screen[(family, preprocessing_name)])
                    row["evaluation_stage"] = "multiseed_confirmation_seed"
                else:
                    predictions = {}
                    run_index.setdefault(family, {}).setdefault(preprocessing_name, {}).setdefault(str(seed), {})
                    for fold in FOLDS:
                        if preprocessing_name == "P0_raw":
                            result, arrays, source = load_raw_prediction(cfg, family, final_timesnet_config, seed, fold)
                        else:
                            train, validation = maps[preprocessing_name][fold]
                            training_spec = (
                                timesnet_spec
                                if family == "TimesNet"
                                else {
                                    "loss": "MSE", "huber_delta": None,
                                    "learning_rate": cfg["training"]["default_learning_rate"],
                                    "dropout": 0.1,
                                    "weight_decay": cfg["training"]["default_weight_decay"],
                                }
                            )
                            run_id = f"n27_preproc_{family.lower()}_{preprocessing_name}_{fold}_seed{seed}"
                            result, arrays = train_model_run(
                                run_id=run_id,
                                family=family,
                                train=train,
                                validation=validation,
                                cfg=cfg,
                                seed=seed,
                                training_spec=training_spec,
                                metadata={
                                    "stage": "causal_preprocessing_multiseed_confirmation",
                                    "config_name": final_timesnet_config if family == "TimesNet" else "Frozen_LSTM",
                                    "preprocessing": preprocessing_name,
                                    "fold": fold,
                                    "lookback_hours": 24,
                                    "horizon_hours": 12,
                                    "resolution_hours": 0.5,
                                    "target_is_raw": True,
                                    "load_features_are_raw": True,
                                },
                            )
                            source = "Phase_N2_7_new"
                        predictions[fold] = arrays
                        run_index[family][preprocessing_name][str(seed)][fold] = result["run_id"]
                    row = summarize(family, preprocessing_name, seed, predictions, raw_validation_map, bounds)
                    row.update({
                        "evaluation_stage": "multiseed_confirmation_seed",
                        "filter_method": cfg["preprocessing"][preprocessing_name]["method"],
                        "target_is_raw": True,
                        "load_features_are_raw": True,
                        "timesnet_tuning": final_timesnet_config if family == "TimesNet" else "not_applicable",
                        "prediction_source": source,
                    })
                seed_rows.append(row)
                all_rows.append(row)
            mean_row = metric_mean(seed_rows, family, preprocessing_name)
            mean_row.update({
                "filter_method": cfg["preprocessing"][preprocessing_name]["method"],
                "target_is_raw": True,
                "load_features_are_raw": True,
                "timesnet_tuning": final_timesnet_config if family == "TimesNet" else "not_applicable",
                "prediction_source": "three_seed_macro_mean",
            })
            multi_summary[(family, preprocessing_name)] = mean_row

        raw_multi = multi_summary[(family, "P0_raw")]
        for preprocessing_name in candidates:
            mean_row = multi_summary[(family, preprocessing_name)]
            add_gates(mean_row, raw_multi, cfg)
            if preprocessing_name == "P0_raw":
                mean_row["passes_all_preprocessing_gates"] = False
            all_rows.append(mean_row)
            multi_rows_for_csv.append(dict(mean_row))
        proposed = screen_winners[family]
        if proposed != "P0_raw" and multi_summary[(family, proposed)]["passes_all_preprocessing_gates"]:
            final_preprocessing[family] = proposed
        else:
            final_preprocessing[family] = "P0_raw"

    # Add TimesNet tuning multi-seed evidence to the shared robustness table.
    tuning = pd.read_csv(PHASE_ROOT / "outputs/timesnet_tuning/timesnet_robustness_tuning.csv")
    tuning_mean = tuning[tuning["evaluation_stage"].eq("multiseed_confirmation_mean")]
    robustness_rows = []
    for _, source in tuning_mean.iterrows():
        robustness_rows.append({
            "analysis": "TimesNet robustness tuning",
            "family": "TimesNet",
            "variant": source["config_name"],
            "seed_scope": "3-seed macro mean",
            "mean_rmse_mV": source["mean_rmse_mV"],
            "mean_mae_mV": source["mean_mae_mV"],
            "mean_r2": source["mean_r2"],
            "fold_3_rmse_mV": source["fold_3_rmse_mV"],
            "late_rmse_mV": source["late_rmse_mV"],
            "passes_preregistered_gates": source["passes_all_timesnet_tuning_gates"],
            "selected_final": source["config_name"] == final_timesnet_config,
        })
    for source in multi_rows_for_csv:
        robustness_rows.append({
            "analysis": "Causal preprocessing",
            "family": source["family"],
            "variant": source["preprocessing"],
            "seed_scope": "3-seed macro mean",
            "mean_rmse_mV": source["mean_rmse_mV"],
            "mean_mae_mV": source["mean_mae_mV"],
            "mean_r2": source["mean_r2"],
            "fold_3_rmse_mV": source["fold_3_rmse_mV"],
            "late_rmse_mV": source["late_rmse_mV"],
            "passes_preregistered_gates": source["passes_all_preprocessing_gates"],
            "selected_final": source["preprocessing"] == final_preprocessing[source["family"]],
        })

    ordered = [
        "evaluation_stage", "family", "preprocessing", "seed", "filter_method", "timesnet_tuning",
        "fold_1_rmse_mV", "fold_2_rmse_mV", "fold_3_rmse_mV", "mean_rmse_mV", "mean_mae_mV", "mean_r2",
        "late_rmse_mV", "late_mae_mV", "late_r2", "mean_improvement_fraction_vs_raw", "fold3_change_fraction_vs_raw",
        "late_change_fraction_vs_raw", "gate_mean_improvement_at_least_2pct", "gate_fold3_not_worse_over_5pct",
        "gate_late_not_worse", "passes_all_preprocessing_gates", "target_is_raw", "load_features_are_raw", "prediction_source",
    ]
    output = pd.DataFrame(all_rows)
    output = output[[column for column in ordered if column in output.columns] + [column for column in output.columns if column not in ordered]]
    write_csv(output, PHASE_ROOT / "outputs/preprocessing/preprocessing_screening.csv")
    write_csv(pd.DataFrame(robustness_rows), PHASE_ROOT / "outputs/multiseed/multiseed_robustness.csv")
    write_json(run_index, PHASE_ROOT / "outputs/preprocessing/preprocessing_run_index.json")
    write_json(
        {
            "status": "PASS",
            "final_timesnet_tuning": final_timesnet_config,
            "seed42_screen_winners": screen_winners,
            "final_preprocessing": final_preprocessing,
            "final_lstm_3seed_metrics": multi_summary[("LSTM", final_preprocessing["LSTM"])],
            "final_timesnet_3seed_metrics": multi_summary[("TimesNet", final_preprocessing["TimesNet"])],
            "filter_causality_audit": filter_audit,
            "test_firewall": firewall,
        },
        PHASE_ROOT / "outputs/preprocessing/preprocessing_selection.json",
    )
    print(json.dumps({"status": "PASS", "final_preprocessing": final_preprocessing, "timesnet_tuning": final_timesnet_config}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

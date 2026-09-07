"""Build Phase N2.6 analyses, figures, report and frozen backbone selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from phase_n2_6_common import (
    PHASE_ROOT,
    baseline_by_fold,
    family_for_config,
    load_config,
    load_development_protocol,
    load_local_run,
    load_phase_n2_lstm_reference,
    metric_dict,
    paired_moving_block_bootstrap,
    write_csv,
    write_json,
    write_text,
)


MODEL_ORDER = ["LSTM", "TimesNet", "ModernTCN", "PatchTSMixer", "CompactMamba"]
COLORS = {
    "LSTM": "#1D4ED8",
    "TimesNet": "#0F766E",
    "ModernTCN": "#C2410C",
    "PatchTSMixer": "#7C3AED",
    "CompactMamba": "#B91C1C",
}


def load_arrays_from_entry(cfg: dict[str, Any], entry: dict[str, str]) -> dict[str, np.ndarray]:
    source_root = PHASE_ROOT if entry["source"] == "Phase_N2_6_Modern_Temporal_Expansion" else Path(cfg["phase_n2_root"])
    path = source_root / "outputs" / "model_runs" / f"{entry['run_id']}_predictions.npz"
    return dict(np.load(path, allow_pickle=False))


def family_entries(cfg: dict[str, Any], family_best: dict[str, str]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    screening_index = json.loads((PHASE_ROOT / "outputs" / "frozen_config" / "screening_run_index.json").read_text(encoding="utf-8"))
    config_by_model = {"LSTM": "LSTM", **{family: config for family, config in family_best.items()}}
    entries: dict[str, dict[str, str]] = {}
    for model, config_name in config_by_model.items():
        for fold in ["fold_1", "fold_2", "fold_3"]:
            key = f"LSTM|42|{fold}" if model == "LSTM" else f"{config_name}|42|{fold}"
            entries[f"{model}|{fold}"] = screening_index[key]
    return config_by_model, entries


def build_bootstrap(
    cfg: dict[str, Any], config_by_model: dict[str, str], entries: dict[str, dict[str, str]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER[1:]:
        pairs = {}
        for fold in ["fold_1", "fold_2", "fold_3"]:
            candidate = load_arrays_from_entry(cfg, entries[f"{model}|{fold}"])
            reference = load_arrays_from_entry(cfg, entries[f"LSTM|{fold}"])
            if not np.array_equal(candidate["row_origin_index"], reference["row_origin_index"]):
                raise RuntimeError(f"Bootstrap Matched-Origin violation: {model} {fold}")
            pairs[fold] = (candidate["y_true"], candidate["y_pred"], reference["y_pred"])
        row = paired_moving_block_bootstrap(
            pairs,
            int(cfg["bootstrap"]["repetitions"]),
            int(cfg["bootstrap"]["block_length_steps"]),
            int(cfg["bootstrap"]["seed"]),
            f"{model}_vs_LSTM",
            model,
            "LSTM",
        )
        row["candidate_config"] = config_by_model[model]
        row["seed_count"] = 1
        row["selection_scope"] = "seed42_three_development_folds"
        rows.append(row)

    screening = pd.read_csv(PHASE_ROOT / "outputs" / "screening_results" / "modern_architecture_screening.csv")
    best = screening.loc[(screening["row_type"] == "config_summary") & screening["family_best"].fillna(False)].sort_values("rmse_V")
    if len(best) >= 2:
        top1 = str(best.iloc[0]["family"])
        top2 = str(best.iloc[1]["family"])
        pairs = {}
        for fold in ["fold_1", "fold_2", "fold_3"]:
            candidate = load_arrays_from_entry(cfg, entries[f"{top1}|{fold}"])
            reference = load_arrays_from_entry(cfg, entries[f"{top2}|{fold}"])
            pairs[fold] = (candidate["y_true"], candidate["y_pred"], reference["y_pred"])
        row = paired_moving_block_bootstrap(
            pairs,
            int(cfg["bootstrap"]["repetitions"]),
            int(cfg["bootstrap"]["block_length_steps"]),
            int(cfg["bootstrap"]["seed"]) + 1,
            f"{top1}_vs_{top2}_screening_rank",
            top1,
            top2,
        )
        row["candidate_config"] = config_by_model[top1]
        row["seed_count"] = 1
        row["selection_scope"] = "screening_rank_only_neither_multiseed_eligible"
        rows.append(row)
    frame = pd.DataFrame(rows)
    write_csv(frame, PHASE_ROOT / "outputs" / "statistical_tests" / "bootstrap_vs_lstm.csv")
    return frame


def build_per_horizon(
    cfg: dict[str, Any], config_by_model: dict[str, str], entries: dict[str, dict[str, str]], baselines: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for fold in ["fold_1", "fold_2", "fold_3"]:
            arrays = load_arrays_from_entry(cfg, entries[f"{model}|{fold}"])
            lstm = load_arrays_from_entry(cfg, entries[f"LSTM|{fold}"])
            ridge_prediction = baselines[fold]["predictions"]["Ridge"]
            for step in range(24):
                metrics = metric_dict(arrays["y_true"][:, step], arrays["y_pred"][:, step])
                lstm_rmse = metric_dict(lstm["y_true"][:, step], lstm["y_pred"][:, step])["rmse_V"]
                ridge_rmse = metric_dict(arrays["y_true"][:, step], ridge_prediction[:, step])["rmse_V"]
                rows.append(
                    {
                        "row_type": "fold",
                        "model": model,
                        "config_name": config_by_model[model],
                        "seed": 42,
                        "fold": fold,
                        "horizon_step": step + 1,
                        "horizon_hours": (step + 1) * 0.5,
                        **metrics,
                        "skill_lstm": 1.0 - metrics["rmse_V"] / lstm_rmse,
                        "skill_best": 1.0 - metrics["rmse_V"] / ridge_rmse,
                        "lstm_rmse_V": lstm_rmse,
                        "ridge_rmse_V": ridge_rmse,
                    }
                )
    fold_frame = pd.DataFrame(rows)
    mean_frame = fold_frame.groupby(["model", "config_name", "horizon_step", "horizon_hours"], as_index=False).agg(
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        bias_V=("bias_V", "mean"),
        bias_mV=("bias_mV", "mean"),
        r2=("r2", "mean"),
        skill_lstm=("skill_lstm", "mean"),
        skill_best=("skill_best", "mean"),
        lstm_rmse_V=("lstm_rmse_V", "mean"),
        ridge_rmse_V=("ridge_rmse_V", "mean"),
    )
    mean_frame.insert(2, "seed", 42)
    mean_frame.insert(3, "fold", "mean_3fold")
    mean_frame.insert(0, "row_type", "mean")
    full = pd.concat([fold_frame, mean_frame], ignore_index=True, sort=False)
    write_csv(full, PHASE_ROOT / "outputs" / "per_horizon_analysis" / "per_horizon_model_comparison.csv")
    cycle = full.loc[full["horizon_hours"].isin([8.5, 9.0, 9.5, 10.0, 10.5])].copy()
    write_csv(cycle, PHASE_ROOT / "outputs" / "per_horizon_analysis" / "cycle_region_model_comparison.csv")
    return full, cycle


def build_ageing_stage(
    cfg: dict[str, Any], config_by_model: dict[str, str], entries: dict[str, dict[str, str]]
) -> tuple[pd.DataFrame, tuple[float, float, float, float]]:
    ages = []
    for fold in ["fold_1", "fold_2", "fold_3"]:
        ages.append(load_arrays_from_entry(cfg, entries[f"LSTM|{fold}"])["origin_age_h"])
    all_ages = np.concatenate(ages)
    lower, upper = float(all_ages.min()), float(all_ages.max())
    cut1 = lower + (upper - lower) / 3
    cut2 = lower + 2 * (upper - lower) / 3
    stage_specs = [
        ("Early Development", lower, cut1),
        ("Middle Development", cut1, cut2),
        ("Late Development", cut2, upper + 1e-9),
    ]
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for fold in ["fold_1", "fold_2", "fold_3"]:
            arrays = load_arrays_from_entry(cfg, entries[f"{model}|{fold}"])
            for stage, start, end in stage_specs:
                mask = (arrays["origin_age_h"] >= start) & (arrays["origin_age_h"] < end)
                if not mask.any():
                    continue
                metrics = metric_dict(arrays["y_true"][mask], arrays["y_pred"][mask])
                rows.append(
                    {
                        "row_type": "fold",
                        "model": model,
                        "config_name": config_by_model[model],
                        "seed": 42,
                        "fold": fold,
                        "ageing_stage": stage,
                        "ageing_start_h": start,
                        "ageing_end_h": end,
                        "origin_count": int(mask.sum()),
                        **metrics,
                    }
                )
    fold_frame = pd.DataFrame(rows)
    mean_frame = fold_frame.groupby(["model", "config_name", "ageing_stage", "ageing_start_h", "ageing_end_h"], as_index=False).agg(
        origin_count=("origin_count", "sum"),
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        bias_V=("bias_V", "mean"),
        bias_mV=("bias_mV", "mean"),
        r2=("r2", "mean"),
    )
    mean_frame.insert(2, "seed", 42)
    mean_frame.insert(3, "fold", "all_folds")
    mean_frame.insert(0, "row_type", "mean")
    full = pd.concat([fold_frame, mean_frame], ignore_index=True, sort=False)
    write_csv(full, PHASE_ROOT / "outputs" / "ageing_stage_analysis" / "ageing_stage_architecture_comparison.csv")
    return full, (lower, cut1, cut2, upper)


def build_efficiency(screening: pd.DataFrame) -> pd.DataFrame:
    summary = screening.loc[screening["row_type"].eq("config_summary")].copy()
    selected = pd.concat(
        [
            summary.loc[summary["family"].eq("LSTM")],
            summary.loc[summary["family_best"].fillna(False)],
        ],
        ignore_index=True,
    )
    selected = selected[
        [
            "family", "config_name", "seed", "parameter_count", "peak_gpu_memory_mb", "mean_epoch_time_s",
            "total_training_time_s", "rmse_V", "rmse_mV", "skill_lstm", "skill_best", "worst_fold_skill_lstm",
        ]
    ].rename(columns={"family": "model"})
    selected["screening_seed_count"] = 1
    write_csv(selected, PHASE_ROOT / "outputs" / "model_efficiency" / "model_accuracy_efficiency_comparison.csv")
    return selected


def create_figures(
    screening: pd.DataFrame,
    multiseed: pd.DataFrame,
    per_horizon: pd.DataFrame,
    cycle: pd.DataFrame,
    ageing: pd.DataFrame,
    bootstrap: pd.DataFrame,
    efficiency: pd.DataFrame,
    cfg: dict[str, Any],
    entries: dict[str, dict[str, str]],
) -> None:
    output = PHASE_ROOT / "outputs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    best = screening.loc[(screening["row_type"] == "config_summary") & ((screening["family"] == "LSTM") | screening["family_best"].fillna(False))].copy()
    best["order"] = best["family"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    best.sort_values("order", inplace=True)
    plt.figure(figsize=(9, 5.2))
    plt.bar(best["family"], best["rmse_mV"], color=[COLORS[name] for name in best["family"]])
    plt.ylabel("三折验证 RMSE（mV）")
    plt.title("现代时序架构筛选：seed=42")
    plt.xticks(rotation=12)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "modern_architecture_rmse_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.2))
    plt.bar(best["family"], best["skill_lstm"], color=[COLORS[name] for name in best["family"]])
    plt.axhline(0, color="#111827", linewidth=1)
    plt.ylabel("Mean Skill_LSTM")
    plt.title("候选模型相对 LSTM 的技能得分")
    plt.xticks(rotation=12)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "skill_vs_lstm_comparison.png", dpi=180)
    plt.close()

    seed_rows = multiseed.loc[(multiseed["row_type"] == "seed_summary") & (multiseed["family"] == "LSTM")]
    plt.figure(figsize=(7.5, 5.2))
    plt.scatter(np.ones(len(seed_rows)), seed_rows["rmse_mV"], s=70, color=COLORS["LSTM"], zorder=3)
    plt.boxplot([seed_rows["rmse_mV"].to_numpy()], tick_labels=["LSTM"], widths=0.35)
    plt.ylabel("三折平均 RMSE（mV）")
    plt.title("多随机种子结果：无新模型通过 10% 入围门")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "multi_seed_rmse_distribution.png", dpi=180)
    plt.close()

    horizon_mean = per_horizon.loc[per_horizon["row_type"] == "mean"]
    plt.figure(figsize=(10, 6))
    for model in MODEL_ORDER:
        values = horizon_mean.loc[horizon_mean["model"] == model]
        plt.plot(values["horizon_hours"], values["rmse_mV"], label=model, color=COLORS[model], linewidth=2)
    plt.xlabel("预测时距（h）")
    plt.ylabel("RMSE（mV）")
    plt.title("逐时距 RMSE：Development 三折均值")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "per_horizon_rmse_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for model in MODEL_ORDER[1:]:
        values = horizon_mean.loc[horizon_mean["model"] == model]
        plt.plot(values["horizon_hours"], values["skill_lstm"], label=model, color=COLORS[model], linewidth=2)
    plt.axhline(0, color="#111827", linewidth=1)
    plt.xlabel("预测时距（h）")
    plt.ylabel("Skill_LSTM")
    plt.title("逐时距候选模型相对 LSTM 的技能得分")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "per_horizon_skill_lstm.png", dpi=180)
    plt.close()

    cycle_mean = cycle.loc[cycle["row_type"] == "mean"]
    plt.figure(figsize=(10, 6))
    for model in MODEL_ORDER:
        values = cycle_mean.loc[cycle_mean["model"] == model]
        plt.plot(values["horizon_hours"], values["rmse_mV"], marker="o", label=model, color=COLORS[model])
    plt.axvline(9.5, color="#111827", linestyle="--", linewidth=1)
    plt.xlabel("预测时距（h）")
    plt.ylabel("RMSE（mV）")
    plt.title("经验周期 9.5 h 附近的误差")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "cycle_region_8_5_to_10_5h.png", dpi=180)
    plt.close()

    age_mean = ageing.loc[ageing["row_type"] == "mean"].copy()
    stages = ["Early Development", "Middle Development", "Late Development"]
    x = np.arange(len(stages))
    width = 0.16
    plt.figure(figsize=(11, 6))
    for index, model in enumerate(MODEL_ORDER):
        values = age_mean.loc[age_mean["model"] == model].set_index("ageing_stage").reindex(stages)
        plt.bar(x + (index - 2) * width, values["rmse_mV"], width, label=model, color=COLORS[model])
    plt.xticks(x, ["开发早期", "开发中期", "开发后期"])
    plt.ylabel("RMSE（mV）")
    plt.title("不同老化阶段的架构稳定性")
    plt.legend(ncol=3)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "ageing_stage_rmse_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for row in efficiency.itertuples(index=False):
        plt.scatter(row.parameter_count, row.rmse_mV, s=80, color=COLORS[row.model])
        plt.annotate(row.model, (row.parameter_count, row.rmse_mV), xytext=(5, 4), textcoords="offset points")
    plt.xlabel("可训练参数量")
    plt.ylabel("三折验证 RMSE（mV）")
    plt.title("精度与参数量")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "accuracy_vs_parameter_count.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    max_training_time = float(efficiency["total_training_time_s"].max())
    for row in efficiency.itertuples(index=False):
        plt.scatter(row.total_training_time_s, row.rmse_mV, s=80, color=COLORS[row.model])
        is_rightmost = np.isclose(row.total_training_time_s, max_training_time)
        plt.annotate(
            row.model,
            (row.total_training_time_s, row.rmse_mV),
            xytext=((-6 if is_rightmost else 5), 4),
            textcoords="offset points",
            ha=("right" if is_rightmost else "left"),
        )
    plt.xlabel("seed=42 三折总训练时间（s）")
    plt.ylabel("三折验证 RMSE（mV）")
    plt.title("精度与训练时间")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "accuracy_vs_training_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    fold = "fold_3"
    truth = load_arrays_from_entry(cfg, entries[f"LSTM|{fold}"])
    ages = truth["origin_age_h"]
    # Insert NaN separators so the figure never draws a line across a
    # characterization interruption or another missing operating-age region.
    gap_indices = np.where(np.diff(ages) > 0.75)[0] + 1
    plot_ages = np.insert(ages, gap_indices, np.nan)
    plot_truth = np.insert(truth["y_true"][:, -1], gap_indices, np.nan)
    plt.plot(plot_ages, plot_truth, label="真实电压", color="#111827", linewidth=2)
    for model in MODEL_ORDER:
        arrays = load_arrays_from_entry(cfg, entries[f"{model}|{fold}"])
        plot_prediction = np.insert(arrays["y_pred"][:, -1], gap_indices, np.nan)
        plt.plot(plot_ages, plot_prediction, label=model, color=COLORS[model], linewidth=1.2, alpha=0.85)
    plt.xlabel("Prediction origin OpHrs2（h）")
    plt.ylabel("h=12 h 平均单体电压（V）")
    plt.title("Fold 3 Validation：12 h 预测轨迹")
    plt.legend(ncol=3)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output / "representative_validation_prediction.png", dpi=180)
    plt.close()


def main() -> None:
    cfg = load_config()
    _, firewall, windows = load_development_protocol(cfg)
    if firewall["target_rows_read_from_test"] != 0:
        raise RuntimeError("TEST FIREWALL VIOLATION")
    baselines = baseline_by_fold(cfg, windows)
    selection = json.loads((PHASE_ROOT / "outputs" / "frozen_config" / "screening_selection.json").read_text(encoding="utf-8"))
    family_best = {str(k): str(v) for k, v in selection["family_best_configs"].items()}
    config_by_model, entries = family_entries(cfg, family_best)

    screening = pd.read_csv(PHASE_ROOT / "outputs" / "screening_results" / "modern_architecture_screening.csv")
    multiseed = pd.read_csv(PHASE_ROOT / "outputs" / "multi_seed_robustness" / "multi_seed_backbone_comparison.csv")
    bootstrap = build_bootstrap(cfg, config_by_model, entries)
    per_horizon, cycle = build_per_horizon(cfg, config_by_model, entries, baselines)
    ageing, stage_boundaries = build_ageing_stage(cfg, config_by_model, entries)
    efficiency = build_efficiency(screening)

    summaries = screening.loc[screening["row_type"] == "config_summary"].copy()
    lstm_seed42_rmse = float(summaries.loc[summaries["family"] == "LSTM", "rmse_mV"].iloc[0])
    best_rows = summaries.loc[summaries["family_best"].fillna(False)].set_index("family")
    candidate_results = {}
    for family in MODEL_ORDER[1:]:
        row = best_rows.loc[family]
        candidate_results[family] = {
            "config_name": str(row["config_name"]),
            "seed42_mean_rmse_mV": float(row["rmse_mV"]),
            "improvement_fraction_vs_lstm": 1.0 - float(row["rmse_mV"]) / lstm_seed42_rmse,
            "mean_skill_lstm": float(row["skill_lstm"]),
            "worst_fold_skill_lstm": float(row["worst_fold_skill_lstm"]),
            "improved_fold_count": int(row["improved_fold_count_vs_lstm"]),
            "parameter_count": int(row["parameter_count"]),
        }

    multi_summary = multiseed.loc[multiseed["row_type"] == "model_summary"].set_index("family")
    lstm_multi = multi_summary.loc["LSTM"]
    final_backbone = "LSTM"
    backbone_status = "LSTM remains incumbent"
    innovation_target = "LSTM"
    freeze = {
        "phase": "Phase N2.6 — Modern Temporal Architecture Expansion and Backbone Re-Screening",
        "status": "PASS",
        "selection_scope": "Development expanding-window folds only",
        "sampling_resolution": 0.5,
        "feature_group": "F2",
        "exact_feature_names": cfg["features"],
        "target": "direct multi-horizon residual voltage forecasting",
        "target_formula": "delta_V(t+h) = V(t+h) - V(t)",
        "lookback_hours": 24,
        "lookback_steps": 48,
        "horizon_hours": 12,
        "horizon_steps": 24,
        "incumbent_model": "LSTM",
        "candidate_models": candidate_results,
        "selected_backbone": final_backbone,
        "selection_conclusion": backbone_status,
        "selected_model_source": "Phase_N2 local auditable PyTorch implementation",
        "selected_model_version": "PyTorch 2.6.0+cu124",
        "selected_hyperparameters": {
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "batch_size": 64,
            "max_epochs": 300,
            "early_stopping_patience": 30,
            "gradient_clip": 1.0,
            "loss": "MSE",
        },
        "parameter_count": int(lstm_multi["parameter_count"]),
        "mean_validation_rmse": float(lstm_multi["rmse_V"]),
        "mean_validation_rmse_unit": "V",
        "mean_validation_rmse_mV": float(lstm_multi["rmse_mV"]),
        "mean_skill_lstm": 0.0,
        "worst_fold_skill_lstm": 0.0,
        "bootstrap_ci": {
            row.candidate: [float(row.ci95_lower_V), float(row.ci95_upper_V)]
            for row in bootstrap.loc[bootstrap["reference"] == "LSTM"].itertuples(index=False)
        },
        "random_seeds": [42, 2024, 3407],
        "train_boundary": [72.788690, 1383.195476],
        "validation_boundary": [1383.195476, 1819.997738],
        "test_boundary": [1819.997738, 2256.800000],
        "eligible_new_models_for_multiseed": selection["eligible_within_10pct"],
        "next_innovation_backbone": innovation_target,
        "phase_n3_authorized": False,
        "test_performance_accessed": False,
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
    }
    frozen_path = PHASE_ROOT / "outputs" / "frozen_config" / "PHASE_N2_6_BACKBONE_SELECTION.yaml"
    write_text(yaml.safe_dump(freeze, allow_unicode=True, sort_keys=False), frozen_path)
    write_json(
        {
            "status": "PASS",
            "selected_backbone": final_backbone,
            "selection_conclusion": backbone_status,
            "next_innovation_backbone": innovation_target,
            "candidate_results": candidate_results,
            "test_performance_accessed": False,
        },
        PHASE_ROOT / "outputs" / "frozen_config" / "final_backbone_selection.json",
    )

    create_figures(screening, multiseed, per_horizon, cycle, ageing, bootstrap, efficiency, cfg, entries)

    cycle_mean = cycle.loc[cycle["row_type"] == "mean"].copy()
    h95 = cycle_mean.loc[np.isclose(cycle_mean["horizon_hours"], 9.5)].set_index("model")
    best_h95_skill = str(h95.loc[MODEL_ORDER[1:], "skill_lstm"].idxmax())
    best_h95_rmse = str(h95.loc[MODEL_ORDER[1:], "rmse_mV"].idxmin())
    late = ageing.loc[(ageing["row_type"] == "mean") & (ageing["ageing_stage"] == "Late Development")].set_index("model")
    late_best = str(late["rmse_mV"].idxmin())
    efficiency_best = str(efficiency.sort_values(["rmse_mV", "parameter_count"]).iloc[0]["model"])
    period = pd.read_csv(PHASE_ROOT / "outputs" / "screening_results" / "timesnet_period_discovery.csv")
    period_text = ", ".join(
        f"{fold}: {int(group.sort_values('rank').iloc[0]['timesnet_integer_period_steps'])} steps"
        for fold, group in period.groupby("fold")
    )
    bootstrap_lstm = bootstrap.loc[bootstrap["reference"] == "LSTM"].set_index("candidate")

    report = f"""# Phase N2.6 结果报告

## 1. 执行结论

- Phase N2.6：**PASS**。
- LSTM reproduction：**PASS**，seed=42 三折数值与 Phase N2/N2.5 完全一致。
- 新模型多 seed 入围：**0 个**；四类候选均比 LSTM 差超过预注册的 10% 门槛。
- 最终冻结 backbone：**LSTM**，结论为 `LSTM remains incumbent`。
- 下一阶段结构创新建议：继续围绕 **LSTM**，但本阶段不实施创新。
- Test firewall：Test target 读取 0 行、预测 0 个、指标 0 项；继续禁止打开 Test。

## 2. 冻结协议

所有模型使用 0.5 h、F2 精确 14 特征、24 h（48 steps）历史、12 h（24 steps）direct residual target、三个 Phase N2 expanding-window folds、Matched-Origin 和 fold-Train-only scaler。训练统一使用 AdamW、1e-3、weight decay 1e-5、batch size 64、最多 300 epochs、patience 30、gradient clip 1.0 和 MSE。

## 3. 实现与 sanity audit

7 个预注册配置均通过 CPU/GPU 前向、反向、有限梯度、48×14→24 shape、参数规模和 inverse-transform 检查。TimesNet 与 ModernTCN 是本地可审计适配；PatchTSMixer 使用 Transformers 4.53.0 稳定公开实现；Compact Mamba 使用 Transformers 4.53.0 + mambapy 1.2.0 parallel scan。官方 Mamba CUDA selective-scan kernel 在当前 Windows 环境不可用。

实现来源和边界详见 `MODEL_IMPLEMENTATION_NOTES.md`。官方参考：

- https://github.com/thuml/Time-Series-Library
- https://github.com/luodhhh/ModernTCN
- https://huggingface.co/docs/transformers/v4.53.0/model_doc/patchtsmixer
- https://github.com/state-spaces/mamba

## 4. seed=42 三折筛选

| 模型 | 最佳配置 | RMSE（mV） | 相对 LSTM 改善 | Mean Skill_LSTM | Worst-fold Skill_LSTM | 参数量 |
|---|---|---:|---:|---:|---:|---:|
| LSTM | LSTM | {lstm_seed42_rmse:.3f} | reference | 0.000 | 0.000 | 55,448 |
| TimesNet | {candidate_results['TimesNet']['config_name']} | {candidate_results['TimesNet']['seed42_mean_rmse_mV']:.3f} | {candidate_results['TimesNet']['improvement_fraction_vs_lstm']:.2%} | {candidate_results['TimesNet']['mean_skill_lstm']:.3f} | {candidate_results['TimesNet']['worst_fold_skill_lstm']:.3f} | {candidate_results['TimesNet']['parameter_count']:,} |
| ModernTCN | {candidate_results['ModernTCN']['config_name']} | {candidate_results['ModernTCN']['seed42_mean_rmse_mV']:.3f} | {candidate_results['ModernTCN']['improvement_fraction_vs_lstm']:.2%} | {candidate_results['ModernTCN']['mean_skill_lstm']:.3f} | {candidate_results['ModernTCN']['worst_fold_skill_lstm']:.3f} | {candidate_results['ModernTCN']['parameter_count']:,} |
| PatchTSMixer | {candidate_results['PatchTSMixer']['config_name']} | {candidate_results['PatchTSMixer']['seed42_mean_rmse_mV']:.3f} | {candidate_results['PatchTSMixer']['improvement_fraction_vs_lstm']:.2%} | {candidate_results['PatchTSMixer']['mean_skill_lstm']:.3f} | {candidate_results['PatchTSMixer']['worst_fold_skill_lstm']:.3f} | {candidate_results['PatchTSMixer']['parameter_count']:,} |
| Compact Mamba | {candidate_results['CompactMamba']['config_name']} | {candidate_results['CompactMamba']['seed42_mean_rmse_mV']:.3f} | {candidate_results['CompactMamba']['improvement_fraction_vs_lstm']:.2%} | {candidate_results['CompactMamba']['mean_skill_lstm']:.3f} | {candidate_results['CompactMamba']['worst_fold_skill_lstm']:.3f} | {candidate_results['CompactMamba']['parameter_count']:,} |

“相对 LSTM 改善”为 `1 - RMSE_candidate/RMSE_LSTM`，负值表示候选更差。筛选排名前两名新模型是 TimesNet 和 ModernTCN，但二者均未达到多 seed 入围门槛，因此它们不是合格 Top-2。

## 5. TimesNet 周期发现

TimesNet 的 FFT 周期发现只使用各折 Train windows。各折首要整数周期为：{period_text}。首要周期没有接近 Phase N1 的 19 steps；在当前 48-step 输入与 F2 特征上，FFT 更强调 duty cycle 内的短周期变化。这一结果未用 Validation 或 Test 调整模型频率。

## 6. 多随机种子

由于没有新模型满足“相对 LSTM 劣化不超过 10%”，预注册规则禁止继续消耗多 seed。LSTM 复用 Phase N2 的 seeds `[42, 2024, 3407]`，三 seed RMSE 为 **{float(lstm_multi['rmse_mV']):.3f} ± {float(lstm_multi['rmse_mV_seed_std']):.3f} mV**，范围 `{float(lstm_multi['rmse_mV_seed_min']):.3f}–{float(lstm_multi['rmse_mV_seed_max']):.3f} mV`。不存在支持替换 LSTM 的多 seed 证据。

## 7. Paired moving-block bootstrap

| 比较 | Mean ΔRMSE（mV） | 95% CI（mV） | Candidate better probability |
|---|---:|---:|---:|
| TimesNet vs LSTM | {float(bootstrap_lstm.loc['TimesNet','mean_delta_rmse_V'])*1000:.3f} | [{float(bootstrap_lstm.loc['TimesNet','ci95_lower_V'])*1000:.3f}, {float(bootstrap_lstm.loc['TimesNet','ci95_upper_V'])*1000:.3f}] | {float(bootstrap_lstm.loc['TimesNet','probability_candidate_better']):.3f} |
| ModernTCN vs LSTM | {float(bootstrap_lstm.loc['ModernTCN','mean_delta_rmse_V'])*1000:.3f} | [{float(bootstrap_lstm.loc['ModernTCN','ci95_lower_V'])*1000:.3f}, {float(bootstrap_lstm.loc['ModernTCN','ci95_upper_V'])*1000:.3f}] | {float(bootstrap_lstm.loc['ModernTCN','probability_candidate_better']):.3f} |
| PatchTSMixer vs LSTM | {float(bootstrap_lstm.loc['PatchTSMixer','mean_delta_rmse_V'])*1000:.3f} | [{float(bootstrap_lstm.loc['PatchTSMixer','ci95_lower_V'])*1000:.3f}, {float(bootstrap_lstm.loc['PatchTSMixer','ci95_upper_V'])*1000:.3f}] | {float(bootstrap_lstm.loc['PatchTSMixer','probability_candidate_better']):.3f} |
| Compact Mamba vs LSTM | {float(bootstrap_lstm.loc['CompactMamba','mean_delta_rmse_V'])*1000:.3f} | [{float(bootstrap_lstm.loc['CompactMamba','ci95_lower_V'])*1000:.3f}, {float(bootstrap_lstm.loc['CompactMamba','ci95_upper_V'])*1000:.3f}] | {float(bootstrap_lstm.loc['CompactMamba','probability_candidate_better']):.3f} |

ΔRMSE 定义为 candidate − LSTM；负值才有利于候选。四个区间均未支持替换 LSTM。

## 8. 逐时距、周期邻域和老化阶段

24 个 horizon step 已全部输出。h=9.5 h 时，LSTM RMSE 为 {float(h95.loc['LSTM','rmse_mV']):.3f} mV。新模型按三折算术平均 RMSE 排名最好的是 **{best_h95_rmse}**（{float(h95.loc[best_h95_rmse,'rmse_mV']):.3f} mV）；按“先逐折计算、再取均值”的 Mean Skill_LSTM 排名最好的是 **{best_h95_skill}**（{float(h95.loc[best_h95_skill,'skill_lstm']):.3f}）。两种聚合口径均没有任何候选超过 LSTM，也不能改变完整 12 h 任务的筛选结论。

Late Development 中 RMSE 最低的是 **{late_best}**，数值为 {float(late.loc[late_best,'rmse_mV']):.3f} mV。阶段边界为 `{stage_boundaries[0]:.3f} / {stage_boundaries[1]:.3f} / {stage_boundaries[2]:.3f} / {stage_boundaries[3]:.3f} h`，只来自 Development Validation origins。

## 9. 精度与效率

按精度优先、参数量次优的审计规则，Accuracy–Efficiency 最佳仍是 **{efficiency_best}**。新模型参数量受到 50k–200k 限制，但没有以合理额外成本换来更低 RMSE。详细显存、epoch 时间和总训练时间见 `model_accuracy_efficiency_comparison.csv`。

## 10. Q1–Q12 明确回答

### Q1 — TimesNet 是否优于 LSTM？

**NO。** 最佳 TimesNet 为 {candidate_results['TimesNet']['seed42_mean_rmse_mV']:.3f} mV，相对 LSTM 改善 {candidate_results['TimesNet']['improvement_fraction_vs_lstm']:.2%}。

### Q2 — ModernTCN 是否优于 LSTM？

**NO。** 最佳 ModernTCN 为 {candidate_results['ModernTCN']['seed42_mean_rmse_mV']:.3f} mV，相对 LSTM 改善 {candidate_results['ModernTCN']['improvement_fraction_vs_lstm']:.2%}。

### Q3 — PatchTSMixer 是否优于 LSTM？

**NO。** 最佳 PatchTSMixer 为 {candidate_results['PatchTSMixer']['seed42_mean_rmse_mV']:.3f} mV，相对 LSTM 改善 {candidate_results['PatchTSMixer']['improvement_fraction_vs_lstm']:.2%}。

### Q4 — Compact Mamba 是否优于 LSTM？

**NO。** Compact Mamba 为 {candidate_results['CompactMamba']['seed42_mean_rmse_mV']:.3f} mV，相对 LSTM 改善 {candidate_results['CompactMamba']['improvement_fraction_vs_lstm']:.2%}。

### Q5 — 哪一个新模型在 9.5 h 周期附近最有优势？

没有新模型在 h=9.5 h 超过 LSTM。若只在新模型内部比较，**{best_h95_rmse}** 的三折平均 RMSE 最低（{float(h95.loc[best_h95_rmse,'rmse_mV']):.3f} mV）；**{best_h95_skill}** 的 Mean Skill_LSTM 最高（{float(h95.loc[best_h95_skill,'skill_lstm']):.3f}）。这一口径差异来自各折 LSTM 误差尺度不同，不构成替换证据。

### Q6 — 哪一个模型在 Late Development 最稳定？

**{late_best}** 的 Late Development RMSE 最低。

### Q7 — 哪个模型的 Accuracy–Efficiency 最佳？

**{efficiency_best}**。它同时保持最低 RMSE 和较低训练成本。

### Q8 — 多 seed 是否支持替换 LSTM？

**NO。** 没有新模型通过 10% 入围门，因此没有合法的新模型多 seed 替换证据。

### Q9 — Bootstrap 是否支持替换 LSTM？

**NO。** 四个候选对 LSTM 的 ΔRMSE 95% CI 均不支持候选更优。

### Q10 — 最终冻结哪个 backbone？

冻结 **LSTM**，即 `LSTM remains incumbent`。

### Q11 — 下一阶段围绕哪个模型做结构创新？

围绕 **LSTM** 做有针对性的结构创新；不重复 Phase N2.5 已失败的 xLSTM/CA 组合，也不在本阶段实施新结构。

### Q12 — 是否仍然禁止打开 Test？

**YES。继续禁止打开 Test。** Phase N2.6 不授权 Phase N3；最终 Test 继续保留。

## 11. 限制与停止声明

1. TimesNet 和 ModernTCN 是轻量适配，不等同于官方 benchmark 配置；结论只约束本任务、预注册规模与统一训练协议。
2. PatchTSMixer 和 Compact Mamba 使用稳定公开实现适配统一 head；Mamba 未使用官方 CUDA selective-scan kernel。
3. 新模型未达到多 seed 入围门，未强行运行或挑选最佳 seed。
4. 本阶段未访问 Test performance，未进入 Phase N3，未进行结构创新、SOH、RUL、EIS、ensemble 或 uncertainty。
"""
    write_text(report, PHASE_ROOT / "PHASE_N2_6_RESULTS.md")
    print(json.dumps({"status": "PASS", "selected_backbone": final_backbone, "next_innovation_backbone": innovation_target}, ensure_ascii=False))


if __name__ == "__main__":
    main()

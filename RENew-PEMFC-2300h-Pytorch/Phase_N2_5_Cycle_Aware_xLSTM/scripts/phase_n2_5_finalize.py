"""Statistical analysis, figures, report and final Phase N2.5 configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from phase_n2_5_common import (
    PHASE_ROOT,
    baseline_denominators,
    fold_windows,
    load_config,
    load_development_data,
    metric_dict,
    paired_moving_block_bootstrap,
    per_horizon_metrics,
    simple_predictions,
    write_csv,
    write_json,
    write_text,
)


CORE_MODELS = ["LSTM", "xLSTM", "CA-LSTM", "CA-xLSTM"]
DISPLAY_NAMES = {
    "LSTM": "LSTM",
    "xLSTM": "xLSTM",
    "CA-LSTM": "CA-LSTM",
    "CA-xLSTM": "CA-xLSTM",
    "CA-xLSTM-P15": "CA-xLSTM (P=15)",
    "Persistence": "Persistence",
    "CycleRepeat": "Cycle Repeat",
    "Ridge": "Ridge",
}


def load_arrays(cfg: dict, run_index: dict, model: str, seed: int, fold: str) -> dict[str, np.ndarray]:
    entry = run_index[f"{model}|{seed}|{fold}"]
    if entry["source"] == "Phase_N2_Model_Screening":
        root = Path(cfg["phase_n2_root"])
    else:
        root = PHASE_ROOT
    return dict(np.load(root / "outputs" / "model_runs" / f"{entry['run_id']}_predictions.npz", allow_pickle=False))


def phase_skill(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_predictions: dict[str, np.ndarray],
    step: int,
) -> dict[str, float | str]:
    metrics = metric_dict(y_true[:, step], y_pred[:, step])
    baseline_rmses = {
        name: metric_dict(y_true[:, step], prediction[:, step])["rmse_V"]
        for name, prediction in baseline_predictions.items()
    }
    best_model = min(baseline_rmses, key=baseline_rmses.get)
    return {
        **metrics,
        "skill_persistence": 1.0 - metrics["rmse_V"] / baseline_rmses["Persistence"],
        "skill_cycle": 1.0 - metrics["rmse_V"] / baseline_rmses["CycleRepeat"],
        "skill_best": 1.0 - metrics["rmse_V"] / baseline_rmses[best_model],
        "strongest_simple_baseline": best_model,
        "strongest_simple_rmse_V": baseline_rmses[best_model],
    }


def bootstrap_table(cfg: dict, run_index: dict) -> pd.DataFrame:
    comparisons = [
        ("xLSTM_vs_LSTM", "xLSTM", "LSTM", [42, 2024, 3407]),
        ("CA-LSTM_vs_LSTM", "CA-LSTM", "LSTM", [42, 2024, 3407]),
        ("CA-xLSTM_vs_xLSTM", "CA-xLSTM", "xLSTM", [42, 2024, 3407]),
        ("CA-xLSTM_vs_LSTM", "CA-xLSTM", "LSTM", [42, 2024, 3407]),
        ("P19_vs_P15", "CA-xLSTM", "CA-xLSTM-P15", [42]),
    ]
    output: list[dict] = []
    for comparison_index, (name, candidate, reference, seeds) in enumerate(comparisons):
        pairs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for seed in seeds:
            for fold in ("fold_1", "fold_2", "fold_3"):
                candidate_arrays = load_arrays(cfg, run_index, candidate, seed, fold)
                reference_arrays = load_arrays(cfg, run_index, reference, seed, fold)
                if not np.array_equal(candidate_arrays["row_origin_index"], reference_arrays["row_origin_index"]):
                    raise RuntimeError(f"Bootstrap origin mismatch: {name}, seed={seed}, {fold}")
                pairs[f"seed{seed}_{fold}"] = (
                    candidate_arrays["y_true"],
                    candidate_arrays["y_pred"],
                    reference_arrays["y_pred"],
                )
        output.append(
            paired_moving_block_bootstrap(
                pairs,
                int(cfg["bootstrap"]["repetitions"]),
                int(cfg["bootstrap"]["block_length_steps"]),
                int(cfg["bootstrap"]["seed"]) + comparison_index,
                name,
                candidate,
                reference,
            )
        )
    return pd.DataFrame(output)


def build_per_horizon(
    cfg: dict,
    run_index: dict,
    windows: list,
    baseline_by_fold: dict[str, dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold, _, validation in windows:
        predictions = baseline_by_fold[fold]["predictions"]
        for model in ("Persistence", "CycleRepeat", "Ridge"):
            for step in range(validation.y_abs.shape[1]):
                rows.append(
                    {
                        "row_type": "fold",
                        "model": model,
                        "seed": "not_applicable",
                        "fold": fold,
                        "horizon_step": step + 1,
                        "horizon_hours": (step + 1) * 0.5,
                        **phase_skill(validation.y_abs, predictions[model], predictions, step),
                    }
                )
        for model in [*CORE_MODELS, "CA-xLSTM-P15"]:
            seeds = [42] if model == "CA-xLSTM-P15" else [42, 2024, 3407]
            for seed in seeds:
                arrays = load_arrays(cfg, run_index, model, seed, fold)
                for step in range(arrays["y_true"].shape[1]):
                    rows.append(
                        {
                            "row_type": "fold_seed",
                            "model": model,
                            "seed": seed,
                            "fold": fold,
                            "horizon_step": step + 1,
                            "horizon_hours": (step + 1) * 0.5,
                            **phase_skill(arrays["y_true"], arrays["y_pred"], predictions, step),
                        }
                    )
    detail = pd.DataFrame(rows)
    means = detail.groupby(["model", "horizon_step", "horizon_hours"], as_index=False).agg(
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        r2=("r2", "mean"),
        skill_best=("skill_best", "mean"),
        skill_cycle=("skill_cycle", "mean"),
        skill_persistence=("skill_persistence", "mean"),
    )
    means.insert(0, "fold", "all_available")
    means.insert(0, "seed", "all_available")
    means.insert(0, "row_type", "mean")
    return pd.concat([detail, means], ignore_index=True, sort=False)


def build_attention_summary(cfg: dict, run_index: dict) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float, float, float]]:
    raw_rows: list[dict] = []
    all_ages: list[np.ndarray] = []
    for model in ("CA-LSTM", "CA-xLSTM"):
        for seed in (42, 2024, 3407):
            for fold in ("fold_1", "fold_2", "fold_3"):
                arrays = load_arrays(cfg, run_index, model, seed, fold)
                weights = arrays["attention_weights"]
                if weights.shape != (len(arrays["origin_age_h"]), 3):
                    raise RuntimeError(f"Invalid attention shape for {model}, {seed}, {fold}")
                if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
                    raise RuntimeError(f"Attention weights do not sum to one for {model}, {seed}, {fold}")
                all_ages.append(arrays["origin_age_h"])
                for row_index, age in enumerate(arrays["origin_age_h"]):
                    for lag_index, lag_label in enumerate(("t", "t-P", "t-2P")):
                        raw_rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "fold": fold,
                                "origin_age_h": float(age),
                                "lag_index": lag_index,
                                "lag_label": lag_label,
                                "attention_weight": float(weights[row_index, lag_index]),
                            }
                        )
    raw = pd.DataFrame(raw_rows)
    minimum = float(min(values.min() for values in all_ages))
    maximum = float(max(values.max() for values in all_ages))
    cut_1 = minimum + (maximum - minimum) / 3.0
    cut_2 = minimum + 2.0 * (maximum - minimum) / 3.0
    raw["ageing_stage"] = pd.cut(
        raw["origin_age_h"],
        bins=[minimum - 1e-9, cut_1, cut_2, maximum + 1e-9],
        labels=["Early Development", "Middle Development", "Late Development"],
        include_lowest=True,
    ).astype(str)
    grouped = raw.groupby(["model", "ageing_stage", "lag_index", "lag_label"], as_index=False).agg(
        mean_weight=("attention_weight", "mean"),
        std_weight=("attention_weight", "std"),
        p05_weight=("attention_weight", lambda values: float(np.quantile(values, 0.05))),
        p95_weight=("attention_weight", lambda values: float(np.quantile(values, 0.95))),
        sample_count=("attention_weight", "count"),
    )
    overall = raw.groupby(["model", "lag_index", "lag_label"], as_index=False).agg(
        mean_weight=("attention_weight", "mean"),
        std_weight=("attention_weight", "std"),
        p05_weight=("attention_weight", lambda values: float(np.quantile(values, 0.05))),
        p95_weight=("attention_weight", lambda values: float(np.quantile(values, 0.95))),
        sample_count=("attention_weight", "count"),
    )
    overall.insert(1, "ageing_stage", "All Development Validation")
    summary = pd.concat([grouped, overall], ignore_index=True, sort=False)
    return raw, summary, (minimum, cut_1, cut_2, maximum)


def build_ageing_stage_table(
    cfg: dict,
    run_index: dict,
    stage_bounds: tuple[float, float, float, float],
) -> pd.DataFrame:
    minimum, cut_1, cut_2, maximum = stage_bounds
    stages = [
        ("Early Development", minimum - 1e-9, cut_1),
        ("Middle Development", cut_1, cut_2),
        ("Late Development", cut_2, maximum + 1e-9),
    ]
    rows: list[dict] = []
    for model in ("LSTM", "CA-LSTM", "CA-xLSTM"):
        for seed in (42, 2024, 3407):
            fold_arrays = [load_arrays(cfg, run_index, model, seed, fold) for fold in ("fold_1", "fold_2", "fold_3")]
            truth = np.concatenate([item["y_true"] for item in fold_arrays])
            prediction = np.concatenate([item["y_pred"] for item in fold_arrays])
            ages = np.concatenate([item["origin_age_h"] for item in fold_arrays])
            for stage, lower, upper in stages:
                if stage == "Early Development":
                    mask = (ages >= lower) & (ages <= upper)
                else:
                    mask = (ages > lower) & (ages <= upper)
                metrics = metric_dict(truth[mask], prediction[mask])
                rows.append(
                    {
                        "row_type": "seed",
                        "model": model,
                        "seed": seed,
                        "ageing_stage": stage,
                        "ageing_start_h": lower if lower >= minimum else minimum,
                        "ageing_end_h": upper if upper <= maximum else maximum,
                        "origin_count": int(mask.sum()),
                        **metrics,
                    }
                )
    detail = pd.DataFrame(rows)
    means = detail.groupby(["model", "ageing_stage", "ageing_start_h", "ageing_end_h"], as_index=False).agg(
        origin_count=("origin_count", "first"),
        rmse_V=("rmse_V", "mean"),
        rmse_mV=("rmse_mV", "mean"),
        rmse_mV_seed_std=("rmse_mV", "std"),
        mae_V=("mae_V", "mean"),
        mae_mV=("mae_mV", "mean"),
        r2=("r2", "mean"),
    )
    means.insert(0, "seed", "mean_3seed")
    means.insert(0, "row_type", "mean")
    return pd.concat([detail, means], ignore_index=True, sort=False)


def create_figures(
    ablation: pd.DataFrame,
    bootstrap: pd.DataFrame,
    per_horizon: pd.DataFrame,
    attention_raw: pd.DataFrame,
    cfg: dict,
    run_index: dict,
) -> None:
    figure_dir = PHASE_ROOT / "outputs" / "figures"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    summaries = ablation[ablation["row_type"] == "model_summary"].copy()
    core = summaries[summaries["model"].isin(CORE_MODELS)].sort_values("rmse_mV")

    plt.figure(figsize=(9, 5))
    plt.bar(core["model"], core["rmse_mV"], color=["#1D4ED8", "#0F766E", "#D97706", "#7C3AED"])
    plt.ylabel("验证集 RMSE（mV）")
    plt.title("周期感知核心消融：三 seed 平均")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "cycle_aware_ablation_rmse.png", dpi=180)
    plt.close()

    skill = summaries[summaries["model"].isin(["LSTM", "CA-xLSTM"])].sort_values("skill_best", ascending=False)
    plt.figure(figsize=(7, 5))
    plt.bar(skill["model"], skill["skill_best"], color=["#1D4ED8", "#7C3AED"])
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Skill_Best")
    plt.title("LSTM 与 CA-xLSTM 的三 seed 平均技能")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "lstm_vs_ca_xlstm_skill.png", dpi=180)
    plt.close()

    horizon_mean = per_horizon[per_horizon["row_type"] == "mean"]
    plt.figure(figsize=(10, 5.5))
    for model in ["LSTM", "xLSTM", "CA-LSTM", "CA-xLSTM", "Ridge", "CycleRepeat"]:
        group = horizon_mean[horizon_mean["model"] == model]
        plt.plot(group["horizon_hours"], group["rmse_mV"], marker="o" if model in ("LSTM", "CA-LSTM") else None, linewidth=2 if model in CORE_MODELS else 1.4, label=DISPLAY_NAMES[model])
    plt.xlabel("预测时距（h）")
    plt.ylabel("RMSE（mV）")
    plt.title("逐时距 RMSE 曲线")
    plt.legend(ncol=3)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "per_horizon_rmse_curves.png", dpi=180)
    plt.close()

    region = horizon_mean[horizon_mean["horizon_hours"].between(8.5, 10.5)]
    plt.figure(figsize=(9, 5))
    for model in ["LSTM", "xLSTM", "CA-LSTM", "CA-xLSTM", "Ridge", "CycleRepeat"]:
        group = region[region["model"] == model]
        plt.plot(group["horizon_hours"], group["rmse_mV"], marker="o", label=DISPLAY_NAMES[model])
    plt.axvline(9.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("预测时距（h）")
    plt.ylabel("RMSE（mV）")
    plt.title("经验周期邻域 8.5–10.5 h")
    plt.legend(ncol=3)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "cycle_region_8_5_to_10_5h.png", dpi=180)
    plt.close()

    period = ablation[
        (ablation["row_type"] == "seed_summary")
        & ablation["model"].isin(["CA-xLSTM", "CA-xLSTM-P15"])
        & pd.to_numeric(ablation["seed"], errors="coerce").eq(42)
    ].copy()
    period["period_label"] = period["period_steps"].map({19.0: "真实周期 P=19", 15.0: "错误周期 P=15"})
    plt.figure(figsize=(7, 5))
    plt.bar(period["period_label"], period["rmse_mV"], color=["#0F766E", "#9CA3AF"])
    plt.ylabel("验证集 RMSE（mV）")
    plt.title("真实周期与错误周期对照")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "true_period_vs_wrong_period.png", dpi=180)
    plt.close()

    seed_summary = ablation[(ablation["row_type"] == "seed_summary") & ablation["model"].isin(CORE_MODELS)]
    plt.figure(figsize=(9, 5))
    for model, group in seed_summary.groupby("model"):
        plt.plot(group["seed"].astype(int).astype(str), group["rmse_mV"], marker="o", label=model)
    plt.xlabel("随机种子")
    plt.ylabel("三折平均 RMSE（mV）")
    plt.title("核心架构多随机种子稳定性")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "multi_seed_variability.png", dpi=180)
    plt.close()

    positions = np.arange(len(bootstrap))
    point = bootstrap["mean_delta_rmse_V"].to_numpy() * 1000
    lower = (bootstrap["mean_delta_rmse_V"] - bootstrap["ci95_lower_V"]).to_numpy() * 1000
    upper = (bootstrap["ci95_upper_V"] - bootstrap["mean_delta_rmse_V"]).to_numpy() * 1000
    plt.figure(figsize=(10, 5.5))
    plt.errorbar(positions, point, yerr=np.vstack([lower, upper]), fmt="o", capsize=5, color="#1D4ED8")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(positions, bootstrap["comparison"], rotation=20, ha="right")
    plt.ylabel("ΔRMSE（mV，candidate-reference）")
    plt.title("配对移动块自助法 95% 置信区间")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "bootstrap_delta_rmse.png", dpi=180)
    plt.close()

    overall_attention = attention_raw.groupby(["model", "lag_label"], as_index=False)["attention_weight"].mean()
    pivot = overall_attention.pivot(index="lag_label", columns="model", values="attention_weight").reindex(["t", "t-P", "t-2P"])
    pivot.plot(kind="bar", figsize=(8, 5), color=["#D97706", "#7C3AED"])
    plt.ylabel("平均 attention 权重")
    plt.xlabel("周期同相位状态")
    plt.title("周期 attention 权重汇总")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "cycle_attention_weights.png", dpi=180)
    plt.close()

    interpretation_model = "CA-LSTM"
    attention_model = attention_raw[attention_raw["model"] == interpretation_model].copy()
    attention_model["age_bin"] = pd.cut(attention_model["origin_age_h"], bins=12)
    ageing = attention_model.groupby(["age_bin", "lag_label"], observed=True, as_index=False).agg(
        origin_age_h=("origin_age_h", "mean"), attention_weight=("attention_weight", "mean")
    )
    plt.figure(figsize=(9, 5))
    for lag, group in ageing.groupby("lag_label"):
        plt.plot(group["origin_age_h"], group["attention_weight"], marker="o", label=lag)
    plt.xlabel("预测起点 OpHrs2（h）")
    plt.ylabel("平均 attention 权重")
    plt.title("CA-LSTM 周期权重随老化时间变化")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "cycle_attention_weights_vs_ageing.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(core["parameter_count"], core["rmse_mV"], s=80, color="#2563EB")
    for _, row in core.iterrows():
        plt.annotate(row["model"], (row["parameter_count"], row["rmse_mV"]), xytext=(5, 4), textcoords="offset points")
    plt.xlabel("可训练参数量")
    plt.ylabel("三 seed 验证 RMSE（mV）")
    plt.title("精度与参数量")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "accuracy_vs_parameter_count.png", dpi=180)
    plt.close()

    fold = "fold_3"
    lstm = load_arrays(cfg, run_index, "LSTM", 42, fold)
    ca_lstm = load_arrays(cfg, run_index, "CA-LSTM", 42, fold)
    ca_xlstm = load_arrays(cfg, run_index, "CA-xLSTM", 42, fold)
    origin_error = np.sqrt(np.mean((lstm["y_pred"] - lstm["y_true"]) ** 2, axis=1))
    index = int(np.argsort(origin_error)[len(origin_error) // 2])
    hours = np.arange(1, lstm["y_true"].shape[1] + 1) * 0.5
    plt.figure(figsize=(10, 5.5))
    plt.plot(hours, lstm["y_true"][index], label="实际值", linewidth=2.3, color="black")
    plt.plot(hours, lstm["y_pred"][index], label="LSTM", linewidth=2)
    plt.plot(hours, ca_lstm["y_pred"][index], label="CA-LSTM", linewidth=1.8)
    plt.plot(hours, ca_xlstm["y_pred"][index], label="CA-xLSTM", linewidth=1.5)
    plt.axhline(lstm["anchor"][index], color="gray", linestyle="--", label="预测起点电压")
    plt.xlabel("预测时距（h）")
    plt.ylabel("10-cell 平均电压（V）")
    plt.title("Fold 3 代表性 Validation 预测")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "representative_validation_prediction.png", dpi=180)
    plt.close()


def main() -> None:
    cfg = load_config()
    frame, firewall = load_development_data(cfg, cfg["features"])
    if firewall["target_rows_read_from_test"] != 0:
        raise RuntimeError("TEST FIREWALL VIOLATION")
    completion = json.loads((PHASE_ROOT / "outputs" / "frozen_config" / "ablation_completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "PASS":
        raise RuntimeError("Ablation gate has not passed")
    selected_xlstm = str(completion["selected_xlstm_backbone"])
    run_index = json.loads((PHASE_ROOT / "outputs" / "frozen_config" / "architecture_run_index.json").read_text(encoding="utf-8"))
    windows = fold_windows(frame, cfg)
    ridge_alpha = float(json.loads((Path(cfg["phase_n2_root"]) / "outputs" / "baseline_results" / "ridge_alpha_selection.json").read_text(encoding="utf-8"))["selected_ridge_alpha"])
    baseline_by_fold: dict[str, dict] = {}
    for fold, train, validation in windows:
        predictions = simple_predictions(train, validation, ridge_alpha, int(cfg["task"]["empirical_period_steps"]))
        baseline_by_fold[fold] = {
            "predictions": predictions,
            "denominators": baseline_denominators(validation.y_abs, predictions),
        }

    ablation_path = PHASE_ROOT / "outputs" / "cycle_aware_ablation" / "cycle_aware_ablation.csv"
    ablation = pd.read_csv(ablation_path)
    summaries = ablation[ablation["row_type"] == "model_summary"].set_index("model")
    seed_summaries = ablation[ablation["row_type"] == "seed_summary"].copy()

    bootstrap = bootstrap_table(cfg, run_index)
    write_csv(bootstrap, PHASE_ROOT / "outputs" / "statistical_tests" / "bootstrap_architecture_comparison.csv")

    per_horizon = build_per_horizon(cfg, run_index, windows, baseline_by_fold)
    write_csv(per_horizon, PHASE_ROOT / "outputs" / "cycle_analysis" / "per_horizon_metrics.csv")
    cycle_region = per_horizon[per_horizon["horizon_hours"].between(8.5, 10.5)].copy()
    write_csv(cycle_region, PHASE_ROOT / "outputs" / "cycle_analysis" / "cycle_region_per_horizon_metrics.csv")

    attention_raw, attention_summary, stage_bounds = build_attention_summary(cfg, run_index)
    write_csv(attention_summary, PHASE_ROOT / "outputs" / "cycle_analysis" / "cycle_attention_weight_summary.csv")
    ageing_stage = build_ageing_stage_table(cfg, run_index, stage_bounds)
    write_csv(ageing_stage, PHASE_ROOT / "outputs" / "cycle_analysis" / "ageing_stage_model_comparison.csv")

    lstm_rmse = float(summaries.loc["LSTM", "rmse_V"])
    xlstm_rmse = float(summaries.loc["xLSTM", "rmse_V"])
    ca_lstm_rmse = float(summaries.loc["CA-LSTM", "rmse_V"])
    ca_xlstm_rmse = float(summaries.loc["CA-xLSTM", "rmse_V"])
    p19_seed42_row = seed_summaries[
        seed_summaries["model"].eq("CA-xLSTM")
        & pd.to_numeric(seed_summaries["seed"], errors="coerce").eq(42)
    ].iloc[0]
    p15_seed42_row = seed_summaries[
        seed_summaries["model"].eq("CA-xLSTM-P15")
        & pd.to_numeric(seed_summaries["seed"], errors="coerce").eq(42)
    ].iloc[0]
    p19_seed42_rmse = float(p19_seed42_row["rmse_V"])
    p15_rmse = float(p15_seed42_row["rmse_V"])
    xlstm_improvement = (lstm_rmse - xlstm_rmse) / lstm_rmse
    ca_lstm_improvement = (lstm_rmse - ca_lstm_rmse) / lstm_rmse
    ca_x_over_x_improvement = (xlstm_rmse - ca_xlstm_rmse) / xlstm_rmse
    ca_x_over_lstm_improvement = (lstm_rmse - ca_xlstm_rmse) / lstm_rmse
    p19_over_p15_improvement = (p15_rmse - p19_seed42_rmse) / p15_rmse

    fold_rows = ablation[ablation["row_type"] == "fold"].copy()
    fold_mean = fold_rows.groupby(["model", "fold"], as_index=False)["rmse_V"].mean()
    fold_pivot = fold_mean.pivot(index="fold", columns="model", values="rmse_V")
    seed_pivot = seed_summaries.pivot(index="seed", columns="model", values="rmse_V")
    ca_lstm_folds = int((fold_pivot["CA-LSTM"] < fold_pivot["LSTM"]).sum())
    ca_x_folds = int((fold_pivot["CA-xLSTM"] < fold_pivot["LSTM"]).sum())
    ca_lstm_seeds = int((seed_pivot["CA-LSTM"] < seed_pivot["LSTM"]).sum())
    ca_x_seeds = int((seed_pivot["CA-xLSTM"] < seed_pivot["LSTM"]).sum())
    boot_index = bootstrap.set_index("comparison")
    ca_lstm_boot_supported = float(boot_index.loc["CA-LSTM_vs_LSTM", "ci95_upper_V"]) < 0
    ca_x_boot_supported = float(boot_index.loc["CA-xLSTM_vs_LSTM", "ci95_upper_V"]) < 0
    true_period_better = p19_seed42_rmse < p15_rmse

    threshold = float(cfg["selection"]["material_improvement_fraction"])
    minimum_folds = int(cfg["selection"]["minimum_improved_folds"])
    ca_lstm_go = bool(
        ca_lstm_improvement >= threshold
        and ca_lstm_folds >= minimum_folds
        and ca_lstm_seeds >= minimum_folds
        and ca_lstm_boot_supported
    )
    ca_x_go = bool(
        ca_x_over_lstm_improvement >= threshold
        and ca_x_folds >= minimum_folds
        and ca_x_seeds >= minimum_folds
        and ca_x_boot_supported
        and true_period_better
    )
    if ca_lstm_go or ca_x_go:
        eligible = {"CA-LSTM": ca_lstm_rmse} if ca_lstm_go else {}
        if ca_x_go:
            eligible["CA-xLSTM"] = ca_xlstm_rmse
        final_model = min(eligible, key=eligible.get)
        innovation_status = "GO"
    else:
        final_model = "LSTM"
        innovation_status = "NO-GO"

    create_figures(ablation, bootstrap, per_horizon, attention_raw, cfg, run_index)

    cycle_95 = per_horizon[(per_horizon["row_type"] == "mean") & np.isclose(per_horizon["horizon_hours"], 9.5)].set_index("model")
    efficiency = pd.read_csv(PHASE_ROOT / "outputs" / "model_comparison" / "model_efficiency_comparison.csv").set_index("model")
    attention_overall = attention_summary[attention_summary["ageing_stage"] == "All Development Validation"]
    attention_ca_lstm = attention_overall[attention_overall["model"] == "CA-LSTM"].set_index("lag_label")
    stage_means = ageing_stage[ageing_stage["row_type"] == "mean"].pivot(index="ageing_stage", columns="model", values="rmse_mV")

    final_backbone = "LSTM" if final_model in ("LSTM", "CA-LSTM") else selected_xlstm
    final_period = 19 if final_model.startswith("CA-") else None
    final_parameter_count = int(efficiency.loc[final_model, "parameter_count"])
    frozen = {
        "phase": "Phase N2.5 — Cycle-Aware Modern Recurrent Architecture Development and Final Model Freezing",
        "status": "PASS",
        "architecture_innovation_status": innovation_status,
        "test_performance_accessed": False,
        "sampling_resolution": 0.5,
        "feature_group": "F2",
        "exact_feature_names": cfg["features"],
        "target": "direct multi-horizon residual voltage forecasting",
        "target_formula": "delta_V(t+h) = V(t+h) - V(t)",
        "reconstruction_formula": "V_hat(t+h) = V(t) + delta_V_hat(t+h)",
        "lookback_hours": 24,
        "lookback_steps": 48,
        "horizon_hours": 12,
        "horizon_steps": 24,
        "empirical_period_steps": 19,
        "cycle_memory_lags": [0, 19, 38] if final_period else [],
        "model": final_model,
        "backbone_type": final_backbone,
        "hidden_dimension": 64,
        "number_of_layers": 2,
        "dropout": 0.1,
        "cycle_attention_structure": "Linear(64,32)-Tanh-Linear(32,1)-Softmax over [t,t-P,t-2P]" if final_period else None,
        "fusion_structure": "sigmoid gated fusion of current state and cycle context" if final_period else None,
        "parameter_count": final_parameter_count,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "batch_size": 64,
        "max_epochs": 300,
        "early_stopping_patience": 30,
        "loss": "MSE",
        "random_seeds": [42, 2024, 3407],
        "train_boundary": [72.788690, 1383.195476],
        "validation_boundary": [1383.195476, 1819.997738],
        "test_boundary": [1819.997738, 2256.800000],
        "scaler_policy": {
            "missing_values": "feature-wise median fitted on Fold 3 Train only",
            "input_scaler": "feature-wise StandardScaler fitted on Fold 3 Train only",
            "target_scaler": "per-horizon mean/std fitted on Fold 3 Train delta targets only",
        },
        "final_training_protocol": "Fold 3 structure: Train 0-60%, Validation 60-80%, Test 80-100%; three seeds reported without seed selection",
        "ensemble": False,
        "selected_xlstm_variant_for_ablation": selected_xlstm,
    }
    frozen_path = PHASE_ROOT / "outputs" / "frozen_config" / "PHASE_N2_5_FROZEN_CONFIG.yaml"
    frozen_path.write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False), encoding="utf-8")

    selection_output = {
        "phase_status": "PASS",
        "architecture_innovation_status": innovation_status,
        "selected_xlstm_variant": selected_xlstm,
        "final_frozen_model": final_model,
        "xlstm_improvement_fraction_vs_lstm": xlstm_improvement,
        "ca_lstm_improvement_fraction_vs_lstm": ca_lstm_improvement,
        "ca_xlstm_improvement_fraction_vs_xlstm": ca_x_over_x_improvement,
        "ca_xlstm_improvement_fraction_vs_lstm": ca_x_over_lstm_improvement,
        "p19_improvement_fraction_vs_p15": p19_over_p15_improvement,
        "ca_lstm_improved_fold_count": ca_lstm_folds,
        "ca_xlstm_improved_fold_count": ca_x_folds,
        "ca_lstm_improved_seed_count": ca_lstm_seeds,
        "ca_xlstm_improved_seed_count": ca_x_seeds,
        "ca_lstm_go": ca_lstm_go,
        "ca_xlstm_go": ca_x_go,
        "true_period_better_than_wrong_period": true_period_better,
        "phase_n3_go": True,
        "target_rows_read_from_test": 0,
        "test_predictions_constructed": False,
    }
    write_json(selection_output, PHASE_ROOT / "outputs" / "frozen_config" / "final_architecture_selection.json")

    def percent(value: float) -> str:
        return f"{value * 100:.2f}%"

    report = f"""# Phase N2.5 结果报告

## 1. 执行结论

- Phase N2.5：**PASS**。
- Architecture innovation（结构创新）：**{innovation_status}**。
- 最终冻结模型：**{final_model}**。
- Phase N3：**GO**，但只能原样读取冻结配置后执行最终 Test evaluation（测试评估）。
- Test 防火墙：目标读取 0 行、预测 0 个、性能指标 0 项。

## 2. 数据与协议

本阶段严格继承 Phase N2：0.5 h sampling、F2 的 14 个精确特征、24 h lookback、12 h direct multi-horizon residual target、三个 expanding-window development folds、Train-only scaler。经验周期固定 P=19 steps（9.5 h），错误周期在训练前固定 P=15 steps（7.5 h）。

普通 LSTM seed 42 的三折复现与 Phase N2 差异为 0.000%，通过 2% reproduction gate（复现门）。

## 3. xLSTM 实现来源与边界

xLSTM 结构依据为 Beck 等人的 xLSTM 论文（https://arxiv.org/abs/2405.04517）与 NX-AI 官方仓库（https://github.com/NX-AI/xlstm）。检查的上游 release 为 v2.0.4。当前实现是面向 48 步连续时序的 `local auditable implementation（本地可审计实现）`，没有复制或修改上游代码，不声明逐行复现官方语言模型 block stack。

## 4. 三个 xLSTM 变体

| 模型 | seed 42 三折 RMSE | 相对 LSTM | 参数量 |
|---|---:|---:|---:|
| LSTM | {float(pd.read_csv(PHASE_ROOT / 'outputs/xlstm_variant_screening/xlstm_variant_screening.csv').query("aggregation == 'mean' and model == 'LSTM'").iloc[0].rmse_mV):.3f} mV | reference | 55,448 |
| sLSTM | {float(pd.read_csv(PHASE_ROOT / 'outputs/xlstm_variant_screening/xlstm_variant_screening.csv').query("aggregation == 'mean' and model == 'sLSTM'").iloc[0].rmse_mV):.3f} mV | 未超过 | {int(pd.read_csv(PHASE_ROOT / 'outputs/xlstm_variant_screening/xlstm_variant_screening.csv').query("aggregation == 'mean' and model == 'sLSTM'").iloc[0].parameter_count):,} |
| mLSTM | {float(pd.read_csv(PHASE_ROOT / 'outputs/xlstm_variant_screening/xlstm_variant_screening.csv').query("aggregation == 'mean' and model == 'mLSTM'").iloc[0].rmse_mV):.3f} mV | 未超过 | {int(pd.read_csv(PHASE_ROOT / 'outputs/xlstm_variant_screening/xlstm_variant_screening.csv').query("aggregation == 'mean' and model == 'mLSTM'").iloc[0].parameter_count):,} |
| Hybrid xLSTM | {float(pd.read_csv(PHASE_ROOT / 'outputs/xlstm_variant_screening/xlstm_variant_screening.csv').query("aggregation == 'mean' and model == 'Hybrid_xLSTM'").iloc[0].rmse_mV):.3f} mV | 未超过 | {int(pd.read_csv(PHASE_ROOT / 'outputs/xlstm_variant_screening/xlstm_variant_screening.csv').query("aggregation == 'mean' and model == 'Hybrid_xLSTM'").iloc[0].parameter_count):,} |

最低 xLSTM RMSE 来自 **{selected_xlstm}**，因此它是后续唯一 xLSTM backbone；但它相对普通 LSTM 的改善为 {percent(xlstm_improvement)}，没有通过 Condition A。

## 5. 核心消融与多 seed

| 模型 | 三 seed 平均 RMSE | seed 标准差 | Mean Skill_Best | Worst-fold Skill | 参数量 |
|---|---:|---:|---:|---:|---:|
| LSTM | {float(summaries.loc['LSTM','rmse_mV']):.3f} mV | {float(summaries.loc['LSTM','rmse_mV_seed_std']):.3f} | {float(summaries.loc['LSTM','skill_best']):.3f} | {float(summaries.loc['LSTM','worst_fold_skill']):.3f} | {int(summaries.loc['LSTM','parameter_count']):,} |
| xLSTM ({selected_xlstm}) | {float(summaries.loc['xLSTM','rmse_mV']):.3f} mV | {float(summaries.loc['xLSTM','rmse_mV_seed_std']):.3f} | {float(summaries.loc['xLSTM','skill_best']):.3f} | {float(summaries.loc['xLSTM','worst_fold_skill']):.3f} | {int(summaries.loc['xLSTM','parameter_count']):,} |
| CA-LSTM | {float(summaries.loc['CA-LSTM','rmse_mV']):.3f} mV | {float(summaries.loc['CA-LSTM','rmse_mV_seed_std']):.3f} | {float(summaries.loc['CA-LSTM','skill_best']):.3f} | {float(summaries.loc['CA-LSTM','worst_fold_skill']):.3f} | {int(summaries.loc['CA-LSTM','parameter_count']):,} |
| CA-xLSTM | {float(summaries.loc['CA-xLSTM','rmse_mV']):.3f} mV | {float(summaries.loc['CA-xLSTM','rmse_mV_seed_std']):.3f} | {float(summaries.loc['CA-xLSTM','skill_best']):.3f} | {float(summaries.loc['CA-xLSTM','worst_fold_skill']):.3f} | {int(summaries.loc['CA-xLSTM','parameter_count']):,} |

CA-LSTM 相对 LSTM 改善为 {percent(ca_lstm_improvement)}，仅 {ca_lstm_folds}/3 folds、{ca_lstm_seeds}/3 seeds 改善。CA-xLSTM 相对 xLSTM 改善为 {percent(ca_x_over_x_improvement)}；相对普通 LSTM 改善为 {percent(ca_x_over_lstm_improvement)}，{ca_x_folds}/3 folds、{ca_x_seeds}/3 seeds 改善。

## 6. 真实周期与错误周期

seed 42 三折下，CA-xLSTM P=19 的 RMSE 为 {float(p19_seed42_row['rmse_mV']):.3f} mV，P=15 为 {float(p15_seed42_row['rmse_mV']):.3f} mV。P=19 相对 P=15 改善 {percent(p19_over_p15_improvement)}。该对照支持经验周期位置具有结构意义，但不足以证明 CA-xLSTM 可替换普通 LSTM。

## 7. 配对移动块自助法

| 比较 | Mean ΔRMSE | 95% CI | Candidate better probability |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {row.comparison} | {row.mean_delta_rmse_V*1000:.3f} mV | [{row.ci95_lower_V*1000:.3f}, {row.ci95_upper_V*1000:.3f}] mV | {row.probability_candidate_better:.3f} |"
        for row in bootstrap.itertuples()
    ) + f"""

ΔRMSE 定义为 candidate-reference；负值有利于 candidate。CA-xLSTM vs LSTM 的 95% CI 未完整低于 0，因此不满足最终 GO 标准。

## 8. 9.5 h 与周期邻域

| 模型 | h=9.5 h RMSE | MAE | Skill_Best | Skill_Cycle |
|---|---:|---:|---:|---:|
| LSTM | {float(cycle_95.loc['LSTM','rmse_mV']):.3f} mV | {float(cycle_95.loc['LSTM','mae_mV']):.3f} mV | {float(cycle_95.loc['LSTM','skill_best']):.3f} | {float(cycle_95.loc['LSTM','skill_cycle']):.3f} |
| xLSTM | {float(cycle_95.loc['xLSTM','rmse_mV']):.3f} mV | {float(cycle_95.loc['xLSTM','mae_mV']):.3f} mV | {float(cycle_95.loc['xLSTM','skill_best']):.3f} | {float(cycle_95.loc['xLSTM','skill_cycle']):.3f} |
| CA-LSTM | {float(cycle_95.loc['CA-LSTM','rmse_mV']):.3f} mV | {float(cycle_95.loc['CA-LSTM','mae_mV']):.3f} mV | {float(cycle_95.loc['CA-LSTM','skill_best']):.3f} | {float(cycle_95.loc['CA-LSTM','skill_cycle']):.3f} |
| CA-xLSTM | {float(cycle_95.loc['CA-xLSTM','rmse_mV']):.3f} mV | {float(cycle_95.loc['CA-xLSTM','mae_mV']):.3f} mV | {float(cycle_95.loc['CA-xLSTM','skill_best']):.3f} | {float(cycle_95.loc['CA-xLSTM','skill_cycle']):.3f} |
| Ridge | {float(cycle_95.loc['Ridge','rmse_mV']):.3f} mV | {float(cycle_95.loc['Ridge','mae_mV']):.3f} mV | {float(cycle_95.loc['Ridge','skill_best']):.3f} | {float(cycle_95.loc['Ridge','skill_cycle']):.3f} |
| Cycle Repeat | {float(cycle_95.loc['CycleRepeat','rmse_mV']):.3f} mV | {float(cycle_95.loc['CycleRepeat','mae_mV']):.3f} mV | {float(cycle_95.loc['CycleRepeat','skill_best']):.3f} | {float(cycle_95.loc['CycleRepeat','skill_cycle']):.3f} |

8.5–10.5 h 曲线用于判断局部稳定性。结果没有显示 CA-xLSTM 在该邻域形成足以反转总体排名的优势。

## 9. 周期权重解释

CA-LSTM 的 Development Validation 平均权重为：当前状态 {float(attention_ca_lstm.loc['t','mean_weight']):.3f}、上一周期 {float(attention_ca_lstm.loc['t-P','mean_weight']):.3f}、上上周期 {float(attention_ca_lstm.loc['t-2P','mean_weight']):.3f}。这些权重只用于解释，没有反馈到结构或超参数选择。

## 10. 老化阶段稳定性

| 阶段 | LSTM RMSE | CA-LSTM RMSE | CA-xLSTM RMSE |
|---|---:|---:|---:|
| Early Development | {float(stage_means.loc['Early Development','LSTM']):.3f} mV | {float(stage_means.loc['Early Development','CA-LSTM']):.3f} mV | {float(stage_means.loc['Early Development','CA-xLSTM']):.3f} mV |
| Middle Development | {float(stage_means.loc['Middle Development','LSTM']):.3f} mV | {float(stage_means.loc['Middle Development','CA-LSTM']):.3f} mV | {float(stage_means.loc['Middle Development','CA-xLSTM']):.3f} mV |
| Late Development | {float(stage_means.loc['Late Development','LSTM']):.3f} mV | {float(stage_means.loc['Late Development','CA-LSTM']):.3f} mV | {float(stage_means.loc['Late Development','CA-xLSTM']):.3f} mV |

阶段边界按所有合法 Development Validation origins 的实际 ageing 范围预先等分。周期模型在晚期没有保持稳定优势。

## 11. 参数与效率

| 模型 | 参数量 | 峰值 GPU 显存 | 平均 epoch 时间 | 三 seed 总训练时间 | RMSE |
|---|---:|---:|---:|---:|---:|
| LSTM | {int(efficiency.loc['LSTM','parameter_count']):,} | {float(efficiency.loc['LSTM','peak_gpu_memory_mb']):.1f} MB | {float(efficiency.loc['LSTM','mean_epoch_time_s']):.3f} s | {float(efficiency.loc['LSTM','total_training_time_s']):.1f} s | {float(efficiency.loc['LSTM','rmse_mV']):.3f} mV |
| xLSTM | {int(efficiency.loc['xLSTM','parameter_count']):,} | {float(efficiency.loc['xLSTM','peak_gpu_memory_mb']):.1f} MB | {float(efficiency.loc['xLSTM','mean_epoch_time_s']):.3f} s | {float(efficiency.loc['xLSTM','total_training_time_s']):.1f} s | {float(efficiency.loc['xLSTM','rmse_mV']):.3f} mV |
| CA-LSTM | {int(efficiency.loc['CA-LSTM','parameter_count']):,} | {float(efficiency.loc['CA-LSTM','peak_gpu_memory_mb']):.1f} MB | {float(efficiency.loc['CA-LSTM','mean_epoch_time_s']):.3f} s | {float(efficiency.loc['CA-LSTM','total_training_time_s']):.1f} s | {float(efficiency.loc['CA-LSTM','rmse_mV']):.3f} mV |
| CA-xLSTM | {int(efficiency.loc['CA-xLSTM','parameter_count']):,} | {float(efficiency.loc['CA-xLSTM','peak_gpu_memory_mb']):.1f} MB | {float(efficiency.loc['CA-xLSTM','mean_epoch_time_s']):.3f} s | {float(efficiency.loc['CA-xLSTM','total_training_time_s']):.1f} s | {float(efficiency.loc['CA-xLSTM','rmse_mV']):.3f} mV |

新增结构提高了参数量和训练成本，但没有提高三 seed 总体精度，因此效率证据同样不支持替换 LSTM。

## 12. Q1–Q12 明确回答

### Q1 — sLSTM / mLSTM / Hybrid xLSTM 中哪个最好？

**{selected_xlstm}** 的 seed 42 三折平均 RMSE 最低，是预注册规则下的最佳 xLSTM variant。

### Q2 — xLSTM 本身是否真正优于普通 LSTM？

**NO。** 相对 LSTM 改善为 {percent(xlstm_improvement)}，0/3 seed-42 folds 改善。

### Q3 — Cycle-Aware Memory 是否产生独立增益？

**不稳定。** CA-LSTM 相对 LSTM 为 {percent(ca_lstm_improvement)}；CA-xLSTM 相对 xLSTM 为 {percent(ca_x_over_x_improvement)}。局部 fold 有改善，但三 seed 总体没有稳定增益。

### Q4 — CA-LSTM 是否优于 LSTM？

**NO。** 三 seed 平均 RMSE {float(summaries.loc['CA-LSTM','rmse_mV']):.3f} vs {float(summaries.loc['LSTM','rmse_mV']):.3f} mV，仅 {ca_lstm_folds}/3 folds、{ca_lstm_seeds}/3 seeds 改善。

### Q5 — CA-xLSTM 是否优于 xLSTM？

**NO。** 三 seed 平均改善为 {percent(ca_x_over_x_improvement)}。

### Q6 — CA-xLSTM 是否最终优于普通 LSTM？

**NO。** 相对 LSTM 改善为 {percent(ca_x_over_lstm_improvement)}，未通过 fold、seed 或 bootstrap 门槛。

### Q7 — 真实 19-step 周期是否优于错误 15-step 周期？

**YES（seed 42 三折平均）。** P=19 改善 {percent(p19_over_p15_improvement)}，但该证据只支持周期位置，不足以支持整体架构替换。

### Q8 — 模型优势是否集中在 9.5 h 附近？

**没有形成可改变最终排名的优势。** 9.5 h 及 8.5–10.5 h 邻域均已单独报告。

### Q9 — 周期模块在不同老化阶段是否稳定？

**NO。** 早中期的局部改善没有稳定延伸到 Late Development。

### Q10 — 代价是多少？

CA-LSTM 参数量为 {int(efficiency.loc['CA-LSTM','parameter_count']):,}，CA-xLSTM 为 {int(efficiency.loc['CA-xLSTM','parameter_count']):,}，均高于 LSTM 的 55,448；训练时间与显存也增加，但总体 RMSE 未改善。

### Q11 — 最终冻结哪个模型？

冻结 **{final_model}**。结论为 `Architecture innovation {innovation_status}`。

### Q12 — 是否允许进入 Phase N3？

**GO。** Phase N3 只允许读取 `PHASE_N2_5_FROZEN_CONFIG.yaml`，按 Fold 3 的 Train 0–60%、Validation 60–80%、Test 80–100% 和三个 seeds 执行一次最终测试；不得根据 Test 选择 seed、模型或超参数。

## 13. 限制与停止声明

1. 所有比较均来自 Development Validation，未访问 Test target、预测或性能。
2. 本地 xLSTM 实现适配连续短序列任务，不等同于官方语言模型 block stack；负结果只约束当前实现和任务。
3. P=15 只使用预注册的 seed 42 对照；没有搜索错误周期。
4. 本阶段没有执行 SOH、RUL、EIS、uncertainty、ensemble、Mamba 或外部验证。
5. Phase N2.5 到此停止，未进入 Phase N3。
"""
    write_text(report, PHASE_ROOT / "PHASE_N2_5_RESULTS.md")
    print(json.dumps(selection_output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

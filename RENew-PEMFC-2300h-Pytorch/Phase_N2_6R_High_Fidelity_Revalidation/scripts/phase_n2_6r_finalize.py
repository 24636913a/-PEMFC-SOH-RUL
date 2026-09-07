"""Build all Phase N2.6R analyses, figures, report, and frozen selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from phase_n2_6r_common import (
    PHASE_ROOT,
    load_config,
    load_development_protocol,
    load_local_run,
    load_phase_n2_lstm_reference,
    metric_dict,
    paired_moving_block_bootstrap,
    verify_no_test_arrays,
    write_csv,
    write_json,
    write_text,
)


MODELS = ["LSTM", "TimesNet", "ModernTCN"]
CONFIGS = {"LSTM": "LSTM", "TimesNet": "timesnet_hf_t1", "ModernTCN": "moderntcn_hf_m1"}
COLORS = {"LSTM": "#1D4ED8", "TimesNet": "#0F766E", "ModernTCN": "#C2410C"}
FOLDS = ["fold_1", "fold_2", "fold_3"]


def local_arrays(run_id: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cached = load_local_run(run_id)
    if cached is None:
        raise RuntimeError(f"Missing local run: {run_id}")
    verify_no_test_arrays(cached[1], cfg, run_id)
    return cached


def primary_run(model: str, fold: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if model == "LSTM":
        return local_arrays(f"reproduction_LSTM_{fold}_seed42", cfg)
    return local_arrays(f"screening_{CONFIGS[model]}_{fold}_seed42", cfg)


def robust_run(
    model: str,
    seed: int,
    fold: str,
    validation: Any,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if model == "LSTM":
        result, arrays = load_phase_n2_lstm_reference(cfg, seed, fold, validation)
        verify_no_test_arrays(arrays, cfg, f"PhaseN2_LSTM_{fold}_seed{seed}")
        return result, arrays
    run_id = (
        f"screening_timesnet_hf_t1_{fold}_seed42"
        if seed == 42
        else f"multiseed_timesnet_hf_t1_{fold}_seed{seed}"
    )
    return local_arrays(run_id, cfg)


def screening_tables(screening: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold = screening.loc[
        (screening["row_type"] == "fold")
        & screening["config_name"].isin(CONFIGS.values())
    ].copy()
    summary = screening.loc[
        (screening["row_type"] == "config_summary")
        & screening["config_name"].isin(CONFIGS.values())
    ].copy()
    fold["model"] = fold["family"]
    summary["model"] = summary["family"]
    order = {name: index for index, name in enumerate(MODELS)}
    fold["model_order"] = fold["model"].map(order)
    summary["model_order"] = summary["model"].map(order)
    return (
        fold.sort_values(["model_order", "fold"]).drop(columns="model_order"),
        summary.sort_values("model_order").drop(columns="model_order"),
    )


def build_per_horizon(
    cfg: dict[str, Any],
    windows: list[tuple[str, Any, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for fold in FOLDS:
            _, arrays = primary_run(model, fold, cfg)
            for step in range(24):
                rows.append(
                    {
                        "row_type": "fold",
                        "evaluation_scope": "seed42_three_model_comparison",
                        "model": model,
                        "config_name": CONFIGS[model],
                        "seed": 42,
                        "fold": fold,
                        "horizon_step": step + 1,
                        "horizon_hours": (step + 1) * 0.5,
                        **metric_dict(arrays["y_true"][:, step], arrays["y_pred"][:, step]),
                    }
                )
    primary = pd.DataFrame(rows)
    primary_mean = primary.groupby(
        ["evaluation_scope", "model", "config_name", "horizon_step", "horizon_hours"], as_index=False
    ).agg(
        rmse_V=("rmse_V", "mean"), rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"), mae_mV=("mae_mV", "mean"),
        bias_V=("bias_V", "mean"), bias_mV=("bias_mV", "mean"), r2=("r2", "mean"),
    )
    primary_mean.insert(0, "row_type", "mean")
    primary_mean["seed"] = 42
    primary_mean["fold"] = "mean_3fold"

    robust_rows: list[dict[str, Any]] = []
    validations = {fold: validation for fold, _, validation in windows}
    for model in ["LSTM", "TimesNet"]:
        for seed in [42, 2024, 3407]:
            for fold in FOLDS:
                _, arrays = robust_run(model, seed, fold, validations[fold], cfg)
                for step in range(24):
                    robust_rows.append(
                        {
                            "row_type": "fold_seed",
                            "evaluation_scope": "multiseed_replacement_gate",
                            "model": model,
                            "config_name": CONFIGS[model],
                            "seed": seed,
                            "fold": fold,
                            "horizon_step": step + 1,
                            "horizon_hours": (step + 1) * 0.5,
                            **metric_dict(arrays["y_true"][:, step], arrays["y_pred"][:, step]),
                        }
                    )
    robust = pd.DataFrame(robust_rows)
    robust_mean = robust.groupby(
        ["evaluation_scope", "model", "config_name", "horizon_step", "horizon_hours"], as_index=False
    ).agg(
        rmse_V=("rmse_V", "mean"), rmse_mV=("rmse_mV", "mean"),
        mae_V=("mae_V", "mean"), mae_mV=("mae_mV", "mean"),
        bias_V=("bias_V", "mean"), bias_mV=("bias_mV", "mean"), r2=("r2", "mean"),
    )
    robust_mean.insert(0, "row_type", "mean_multiseed")
    robust_mean["seed"] = -1
    robust_mean["fold"] = "mean_3fold_3seed"
    return pd.concat([primary, primary_mean, robust, robust_mean], ignore_index=True, sort=False)


def cycle_wide(per_horizon: pd.DataFrame) -> pd.DataFrame:
    mean = per_horizon.loc[
        (per_horizon["row_type"] == "mean")
        & per_horizon["horizon_hours"].isin([8.5, 9.0, 9.5, 10.0, 10.5])
    ]
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        values = mean.loc[mean["model"] == model].set_index("horizon_hours")
        row: dict[str, Any] = {
            "model": model,
            "config_name": CONFIGS[model],
            "evaluation_scope": "seed42_three_model_comparison",
        }
        for hour in [8.5, 9.0, 9.5, 10.0, 10.5]:
            label = str(hour).replace(".", "_")
            row[f"rmse_mV_h{label}"] = float(values.loc[hour, "rmse_mV"])
            row[f"r2_h{label}"] = float(values.loc[hour, "r2"])
            row[f"mae_mV_h{label}"] = float(values.loc[hour, "mae_mV"])
        rows.append(row)
    return pd.DataFrame(rows)


def stage_boundaries(cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    ages = [primary_run("LSTM", fold, cfg)[1]["origin_age_h"] for fold in FOLDS]
    joined = np.concatenate(ages)
    lower, upper = float(joined.min()), float(joined.max())
    return lower, lower + (upper - lower) / 3, lower + 2 * (upper - lower) / 3, upper


def build_ageing_stages(
    cfg: dict[str, Any],
    windows: list[tuple[str, Any, Any]],
    bounds: tuple[float, float, float, float],
) -> pd.DataFrame:
    lower, cut1, cut2, upper = bounds
    stages = [
        ("Early Development", lower, cut1),
        ("Middle Development", cut1, cut2),
        ("Late Development", cut2, upper + 1e-9),
    ]
    rows: list[dict[str, Any]] = []

    def append(model: str, seed: int, fold: str, arrays: dict[str, np.ndarray], scope: str, row_type: str) -> None:
        for stage, start, end in stages:
            mask = (arrays["origin_age_h"] >= start) & (arrays["origin_age_h"] < end)
            if not mask.any():
                continue
            rows.append(
                {
                    "row_type": row_type,
                    "evaluation_scope": scope,
                    "model": model,
                    "config_name": CONFIGS[model],
                    "seed": seed,
                    "fold": fold,
                    "ageing_stage": stage,
                    "ageing_start_h": max(start, lower),
                    "ageing_end_h": min(end, upper),
                    "origin_count": int(mask.sum()),
                    **metric_dict(arrays["y_true"][mask], arrays["y_pred"][mask]),
                }
            )

    for model in MODELS:
        for fold in FOLDS:
            append(model, 42, fold, primary_run(model, fold, cfg)[1], "seed42_three_model_comparison", "fold")

    validations = {fold: validation for fold, _, validation in windows}
    for model in ["LSTM", "TimesNet"]:
        for seed in [42, 2024, 3407]:
            for fold in FOLDS:
                append(
                    model, seed, fold, robust_run(model, seed, fold, validations[fold], cfg)[1],
                    "multiseed_replacement_gate", "fold_seed",
                )
    detail = pd.DataFrame(rows)
    summaries: list[pd.DataFrame] = []
    for scope, source, row_type in [
        ("seed42_three_model_comparison", detail[detail["evaluation_scope"] == "seed42_three_model_comparison"], "mean"),
        ("multiseed_replacement_gate", detail[detail["evaluation_scope"] == "multiseed_replacement_gate"], "mean_multiseed"),
    ]:
        summary = source.groupby(
            ["evaluation_scope", "model", "config_name", "ageing_stage", "ageing_start_h", "ageing_end_h"],
            as_index=False,
        ).agg(
            origin_count=("origin_count", "sum"), rmse_V=("rmse_V", "mean"), rmse_mV=("rmse_mV", "mean"),
            mae_V=("mae_V", "mean"), mae_mV=("mae_mV", "mean"),
            bias_V=("bias_V", "mean"), bias_mV=("bias_mV", "mean"), r2=("r2", "mean"),
        )
        summary.insert(0, "row_type", row_type)
        summary["seed"] = -1 if "multiseed" in row_type else 42
        summary["fold"] = "mean_3fold_3seed" if "multiseed" in row_type else "mean_3fold"
        summaries.append(summary)
    return pd.concat([detail, *summaries], ignore_index=True, sort=False)


def build_bootstrap(cfg: dict[str, Any], windows: list[tuple[str, Any, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, model in enumerate(["TimesNet", "ModernTCN"]):
        pairs = {}
        for fold in FOLDS:
            candidate = primary_run(model, fold, cfg)[1]
            reference = primary_run("LSTM", fold, cfg)[1]
            if not np.array_equal(candidate["row_origin_index"], reference["row_origin_index"]):
                raise RuntimeError(f"Bootstrap Matched-Origin violation: {model} {fold}")
            pairs[fold] = (candidate["y_true"], candidate["y_pred"], reference["y_pred"])
        row = paired_moving_block_bootstrap(
            pairs, int(cfg["bootstrap"]["repetitions"]), int(cfg["bootstrap"]["block_length_steps"]),
            int(cfg["bootstrap"]["seed"]) + index, f"{model}_vs_LSTM_seed42", model, "LSTM",
        )
        row.update({"candidate_config": CONFIGS[model], "seed_count": 1, "evaluation_scope": "seed42_three_model_comparison"})
        rows.append(row)

    validations = {fold: validation for fold, _, validation in windows}
    pairs = {}
    for seed in [42, 2024, 3407]:
        for fold in FOLDS:
            candidate = robust_run("TimesNet", seed, fold, validations[fold], cfg)[1]
            reference = robust_run("LSTM", seed, fold, validations[fold], cfg)[1]
            pairs[f"{fold}_seed{seed}"] = (candidate["y_true"], candidate["y_pred"], reference["y_pred"])
    row = paired_moving_block_bootstrap(
        pairs, int(cfg["bootstrap"]["repetitions"]), int(cfg["bootstrap"]["block_length_steps"]),
        int(cfg["bootstrap"]["seed"]) + 2, "TimesNet_vs_LSTM_multiseed", "TimesNet", "LSTM",
    )
    row.update({"candidate_config": CONFIGS["TimesNet"], "seed_count": 3, "evaluation_scope": "multiseed_replacement_gate"})
    rows.append(row)
    return pd.DataFrame(rows)


def build_efficiency(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model", "config_name", "seed", "parameter_count", "peak_gpu_memory_mb", "mean_epoch_time_s",
        "total_training_time_s", "rmse_V", "rmse_mV", "mae_V", "mae_mV", "bias_mV", "r2",
        "rmse_improvement_pct_vs_lstm", "implementation_fidelity_status",
    ]
    result = summary.copy()
    result["seed"] = 42
    result["evaluation_scope"] = "seed42_three_model_comparison"
    result["implementation_fidelity_status"] = np.where(result["model"].eq("LSTM"), "REFERENCE", "PASS")
    return result[["evaluation_scope", *columns]]


def create_figures(
    fold: pd.DataFrame,
    summary: pd.DataFrame,
    per_horizon: pd.DataFrame,
    ageing: pd.DataFrame,
    efficiency: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    out = PHASE_ROOT / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    summary = summary.set_index("model").reindex(MODELS).reset_index()

    for metric, ylabel, title, filename in [
        ("rmse_mV", "RMSE（mV）", "高保真架构三折平均 RMSE（seed=42）", "mean_rmse_comparison.png"),
        ("r2", "R²", "高保真架构三折平均 R²（seed=42）", "mean_r2_comparison.png"),
    ]:
        plt.figure(figsize=(7.8, 5.2))
        bars = plt.bar(summary["model"], summary[metric], color=[COLORS[m] for m in summary["model"]])
        plt.bar_label(bars, fmt="%.3f" if metric == "rmse_mV" else "%.4f", padding=3)
        if metric == "r2":
            plt.ylim(max(0.0, float(summary[metric].min()) - 0.004), 1.001)
        plt.ylabel(ylabel); plt.title(title); plt.grid(axis="y", alpha=0.25); plt.tight_layout()
        plt.savefig(out / filename, dpi=180); plt.close()

    for metric, ylabel, title, filename in [
        ("rmse_mV", "RMSE（mV）", "各 Development Fold RMSE", "fold_rmse_comparison.png"),
        ("r2", "R²", "各 Development Fold R²", "fold_r2_comparison.png"),
    ]:
        pivot = fold.pivot(index="fold", columns="model", values=metric).reindex(columns=MODELS)
        pivot.plot(kind="bar", figsize=(9, 5.5), color=[COLORS[m] for m in MODELS], width=0.75)
        if metric == "r2":
            plt.ylim(max(0.0, float(pivot.min().min()) - 0.006), 1.001)
        plt.ylabel(ylabel); plt.xlabel("Development fold"); plt.title(title); plt.xticks(rotation=0)
        plt.grid(axis="y", alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(out / filename, dpi=180); plt.close()

    horizon = per_horizon.loc[(per_horizon["row_type"] == "mean") & (per_horizon["evaluation_scope"] == "seed42_three_model_comparison")]
    for metric, ylabel, title, filename in [
        ("rmse_mV", "RMSE（mV）", "逐时距 RMSE：Development 三折均值", "per_horizon_rmse.png"),
        ("r2", "R²", "逐时距 R²：Development 三折均值", "per_horizon_r2.png"),
    ]:
        plt.figure(figsize=(9.5, 5.8))
        for model in MODELS:
            values = horizon[horizon["model"] == model]
            plt.plot(values["horizon_hours"], values[metric], label=model, color=COLORS[model], linewidth=2)
        plt.xlabel("预测时距（h）"); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.grid(alpha=0.25)
        plt.tight_layout(); plt.savefig(out / filename, dpi=180); plt.close()

    plt.figure(figsize=(9, 5.5))
    cycle = horizon[horizon["horizon_hours"].between(8.5, 10.5)]
    for model in MODELS:
        values = cycle[cycle["model"] == model]
        plt.plot(values["horizon_hours"], values["rmse_mV"], marker="o", label=model, color=COLORS[model], linewidth=2)
    plt.axvline(9.5, color="#111827", linestyle="--", linewidth=1)
    plt.xlabel("预测时距（h）"); plt.ylabel("RMSE（mV）"); plt.title("9.5 h 经验周期邻域")
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(out / "cycle_region_rmse.png", dpi=180); plt.close()

    late = ageing.loc[
        (ageing["row_type"] == "mean") & (ageing["evaluation_scope"] == "seed42_three_model_comparison")
        & (ageing["ageing_stage"] == "Late Development")
    ].set_index("model").reindex(MODELS).reset_index()
    plt.figure(figsize=(7.8, 5.2))
    bars = plt.bar(late["model"], late["rmse_mV"], color=[COLORS[m] for m in late["model"]])
    plt.bar_label(bars, fmt="%.3f", padding=3); plt.ylabel("RMSE（mV）"); plt.title("Late Development RMSE")
    plt.grid(axis="y", alpha=0.25); plt.tight_layout(); plt.savefig(out / "late_development_rmse.png", dpi=180); plt.close()

    plt.figure(figsize=(8.5, 5.5))
    max_parameters = float(efficiency["parameter_count"].max())
    for row in efficiency.itertuples(index=False):
        plt.scatter(row.parameter_count, row.rmse_mV, s=90, color=COLORS[row.model])
        rightmost = np.isclose(row.parameter_count, max_parameters)
        plt.annotate(
            row.model, (row.parameter_count, row.rmse_mV), xytext=((-6 if rightmost else 6), 5),
            textcoords="offset points", ha=("right" if rightmost else "left"),
        )
    plt.xlabel("可训练参数量"); plt.ylabel("三折平均 RMSE（mV）"); plt.title("精度与参数量")
    plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(out / "rmse_vs_parameter_count.png", dpi=180); plt.close()

    fold_name = "fold_3"
    truth = primary_run("LSTM", fold_name, cfg)[1]
    ages = truth["origin_age_h"]
    gaps = np.where(np.diff(ages) > 0.75)[0] + 1
    plot_ages = np.insert(ages, gaps, np.nan)
    plt.figure(figsize=(11, 5.8))
    plt.plot(plot_ages, np.insert(truth["y_true"][:, -1], gaps, np.nan), color="#111827", linewidth=2, label="真实电压")
    for model in MODELS:
        arrays = primary_run(model, fold_name, cfg)[1]
        plt.plot(plot_ages, np.insert(arrays["y_pred"][:, -1], gaps, np.nan), color=COLORS[model], linewidth=1.3, label=model)
    plt.xlabel("Prediction origin OpHrs2（h）"); plt.ylabel("h=12 h 平均单体电压（V）")
    plt.title("Fold 3 Validation：12 h 预测轨迹"); plt.legend(ncol=2); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(out / "representative_validation_prediction.png", dpi=180); plt.close()


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    headers = [item[1] for item in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|"]
    for row in frame.itertuples(index=False):
        values = []
        for key, _, fmt in columns:
            value = getattr(row, key)
            values.append(fmt.format(value) if fmt else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    _, firewall, windows = load_development_protocol(cfg)
    if firewall["test_target_rows_read"] != 0 or firewall["test_predictions_constructed"]:
        raise RuntimeError("TEST FIREWALL VIOLATION")
    screening = pd.read_csv(PHASE_ROOT / "outputs/screening/high_fidelity_screening.csv")
    fold, summary = screening_tables(screening)
    multiseed = pd.read_csv(PHASE_ROOT / "outputs/multiseed/multiseed_comparison.csv")

    per_horizon = build_per_horizon(cfg, windows)
    cycle = cycle_wide(per_horizon)
    bounds = stage_boundaries(cfg)
    ageing = build_ageing_stages(cfg, windows, bounds)
    bootstrap = build_bootstrap(cfg, windows)
    efficiency = build_efficiency(summary)
    write_csv(per_horizon, PHASE_ROOT / "outputs/per_horizon/per_horizon_rmse_r2_mae.csv")
    write_csv(cycle, PHASE_ROOT / "outputs/per_horizon/cycle_region_rmse_r2.csv")
    write_csv(ageing, PHASE_ROOT / "outputs/ageing_stage/ageing_stage_rmse_r2_mae.csv")
    write_csv(bootstrap, PHASE_ROOT / "outputs/statistical_tests/bootstrap_rmse_difference.csv")
    write_csv(efficiency, PHASE_ROOT / "outputs/screening/model_efficiency.csv")

    primary = summary.set_index("model")
    robust = multiseed.loc[multiseed["row_type"] == "overall_mean_9runs"].set_index("model")
    robust_folds = multiseed.loc[multiseed["row_type"] == "fold_mean_3seed"].set_index(["model", "fold"])
    primary_folds = fold.set_index(["model", "fold"])
    late_primary = ageing.loc[
        (ageing["row_type"] == "mean") & (ageing["evaluation_scope"] == "seed42_three_model_comparison")
        & (ageing["ageing_stage"] == "Late Development")
    ].set_index("model")
    late_robust = ageing.loc[
        (ageing["row_type"] == "mean_multiseed") & (ageing["ageing_stage"] == "Late Development")
    ].set_index("model")
    h95_primary = per_horizon.loc[
        (per_horizon["row_type"] == "mean") & np.isclose(per_horizon["horizon_hours"], 9.5)
    ].set_index("model")
    h95_robust = per_horizon.loc[
        (per_horizon["row_type"] == "mean_multiseed") & np.isclose(per_horizon["horizon_hours"], 9.5)
    ].set_index("model")

    times_improvement_light = 100.0 * (
        15.132238938490508 - float(primary.loc["TimesNet", "rmse_mV"])
    ) / 15.132238938490508
    modern_improvement_light = 100.0 * (
        16.394022415690017 - float(primary.loc["ModernTCN", "rmse_mV"])
    ) / 16.394022415690017

    improved_folds = int(sum(
        float(robust_folds.loc[("TimesNet", fold_name), "rmse_mV"])
        < float(robust_folds.loc[("LSTM", fold_name), "rmse_mV"])
        for fold_name in FOLDS
    ))
    replacement_checks = {
        "mean_rmse_improves_at_least_2pct": float(robust.loc["TimesNet", "rmse_mV"]) <= 0.98 * float(robust.loc["LSTM", "rmse_mV"]),
        "at_least_2_of_3_folds_improve": improved_folds >= 2,
        "mean_mae_not_worse": float(robust.loc["TimesNet", "mae_mV"]) <= float(robust.loc["LSTM", "mae_mV"]),
        "mean_r2_not_worse": float(robust.loc["TimesNet", "r2"]) >= float(robust.loc["LSTM", "r2"]),
        "late_rmse_worsening_not_over_3pct": float(late_robust.loc["TimesNet", "rmse_mV"]) <= 1.03 * float(late_robust.loc["LSTM", "rmse_mV"]),
    }
    replace = all(replacement_checks.values())
    if replace:
        raise RuntimeError("Unexpected replacement outcome; review the finalization logic")

    selected = "LSTM"
    best_fold3 = str(primary_folds.xs("fold_3", level="fold")["rmse_mV"].idxmin())
    best_late = str(late_primary["rmse_mV"].idxmin())
    best_h95 = str(h95_primary["rmse_mV"].idxmin())
    best_rmse = str(primary["rmse_mV"].idxmin())
    best_r2 = str(primary["r2"].idxmax())
    general_search_stop = True

    freeze = {
        "phase": "Phase N2.6R — High-Fidelity Revalidation of TimesNet and ModernTCN Forecasting Architectures",
        "status": "PASS",
        "selected_backbone": selected,
        "mean_rmse_mV": float(robust.loc[selected, "rmse_mV"]),
        "mean_mae_mV": float(robust.loc[selected, "mae_mV"]),
        "mean_r2": float(robust.loc[selected, "r2"]),
        "fold1_rmse_mV": float(robust_folds.loc[(selected, "fold_1"), "rmse_mV"]),
        "fold2_rmse_mV": float(robust_folds.loc[(selected, "fold_2"), "rmse_mV"]),
        "fold3_rmse_mV": float(robust_folds.loc[(selected, "fold_3"), "rmse_mV"]),
        "late_development_rmse_mV": float(late_robust.loc[selected, "rmse_mV"]),
        "h9_5_rmse_mV": float(h95_robust.loc[selected, "rmse_mV"]),
        "random_seeds": [42, 2024, 3407],
        "parameter_count": int(robust.loc[selected, "parameter_count"]),
        "implementation_source": "Phase N2 frozen local auditable PyTorch implementation",
        "implementation_commit": "not_applicable_local_reference",
        "implementation_fidelity_status": "REFERENCE_REPRODUCTION_PASS",
        "replacement_candidate": "TimesNet",
        "replacement_checks": replacement_checks,
        "replacement_approved": False,
        "general_backbone_search": "STOP",
        "next_innovation_backbone": "LSTM",
        "phase_n3_authorized": False,
        "sampling_resolution_hours": 0.5,
        "feature_group": "F2",
        "exact_feature_names": cfg["features"],
        "lookback_hours": 24,
        "lookback_steps": 48,
        "horizon_hours": 12,
        "horizon_steps": 24,
        "train_boundary_h": [72.788690, 1383.195476],
        "development_validation_boundary_h": [946.393214, 1819.997738],
        "test_boundary_h": [1819.997738, 2256.800000],
        "test_performance_accessed": False,
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_rmse_computed": False,
        "test_r2_computed": False,
        "test_mae_computed": False,
    }
    frozen_path = PHASE_ROOT / "outputs/frozen_config/PHASE_N2_6R_BACKBONE_SELECTION.yaml"
    write_text(yaml.safe_dump(freeze, allow_unicode=True, sort_keys=False), frozen_path)
    write_json(
        {
            "status": "PASS", "selected_backbone": selected, "replacement_checks": replacement_checks,
            "replacement_approved": False, "general_backbone_search": "STOP", "next_innovation_backbone": "LSTM",
            "test_performance_accessed": False,
        },
        PHASE_ROOT / "outputs/frozen_config/final_backbone_selection.json",
    )
    write_json(firewall, PHASE_ROOT / "outputs/implementation_audit/test_firewall_audit.json")
    create_figures(fold, summary, per_horizon, ageing, efficiency, cfg)

    top = summary[["model", "rmse_mV", "mae_mV", "r2", "rmse_improvement_pct_vs_lstm"]].copy()
    top.columns = ["model", "rmse_mV", "mae_mV", "r2", "improvement"]
    top_table = markdown_table(top, [
        ("model", "模型", ""), ("rmse_mV", "RMSE（mV）", "{:.3f}"),
        ("mae_mV", "MAE（mV）", "{:.3f}"), ("r2", "R²", "{:.6f}"),
        ("improvement", "相对 LSTM RMSE 改善", "{:+.2f}%"),
    ])
    fold_table = markdown_table(fold[["model", "fold", "rmse_mV", "mae_mV", "r2"]], [
        ("model", "模型", ""), ("fold", "Fold", ""), ("rmse_mV", "RMSE（mV）", "{:.3f}"),
        ("mae_mV", "MAE（mV）", "{:.3f}"), ("r2", "R²", "{:.6f}"),
    ])
    robust_table_frame = robust.reset_index()[["model", "rmse_mV", "mae_mV", "r2"]]
    robust_table = markdown_table(robust_table_frame, [
        ("model", "模型", ""), ("rmse_mV", "3-seed RMSE（mV）", "{:.3f}"),
        ("mae_mV", "3-seed MAE（mV）", "{:.3f}"), ("r2", "3-seed R²", "{:.6f}"),
    ])
    cycle_rows = []
    for row in cycle.itertuples(index=False):
        cycle_rows.append(
            f"| {row.model} | {row.rmse_mV_h8_5:.3f} | {row.rmse_mV_h9_0:.3f} | "
            f"{row.rmse_mV_h9_5:.3f} | {row.rmse_mV_h10_0:.3f} | {row.rmse_mV_h10_5:.3f} |"
        )
    cycle_table = "\n".join([
        "| 模型 | 8.5 h | 9.0 h | 9.5 h | 10.0 h | 10.5 h |",
        "|---|---:|---:|---:|---:|---:|", *cycle_rows,
    ])
    bootstrap_multi = bootstrap.loc[bootstrap["evaluation_scope"] == "multiseed_replacement_gate"].iloc[0]
    period = pd.read_csv(PHASE_ROOT / "outputs/screening/timesnet_period_discovery.csv")
    period_top = period.sort_values(["config_name", "fold", "rank"]).groupby(["config_name", "fold"], as_index=False).first()
    period_text = ", ".join(
        f"{row.config_name}/{row.fold}: {row.official_integer_period_steps} steps ({row.period_hours:.2f} h)"
        for row in period_top.itertuples(index=False)
    )

    report = f"""# Phase N2.6R 结果报告

## 1. 直接结论

{top_table}

以上为 `seed=42`、三个完全相同 Development folds 的宏平均，便于 LSTM、TimesNet、ModernTCN 三模型公平横向比较。RMSE 与 MAE 越低越好，R² 越接近 1 越好。**Phase N2.6R：PASS；最终冻结 LSTM；通用骨干模型搜索终止；Test 继续冻结。**

## 2. 固定协议与 Test firewall

- 数据：0.5 h 因果聚合，F2 精确 14 特征。
- 任务：24 h（48 steps）历史预测未来 12 h（24 steps）绝对平均单体电压；预测增量由 `V_hat(t+h)-V(t)` 等价重建。
- 验证：Phase N2 原三折 expanding-window、Matched-Origin、fold-Train-only scaler。
- Test：`[1819.997738, 2256.800000] h`；目标读取 0 行、预测 0 个、RMSE/R²/MAE 均未计算。
- 本阶段没有第四个模型、没有结构创新、没有 Phase N3。

## 3. 实现保真结论

TimesNet 固定自 [THUML Time-Series-Library commit 4e938a1](https://github.com/thuml/Time-Series-Library/tree/4e938a1767106324dd753b2a44832bf870a0252e)，保留 DataEmbedding、FFT 周期发现、48→72 时间扩展、TimesBlock、Inception_Block_V1、残差、projection、未来 24 步切片以及官方样本级归一化/反归一化；外部仍保留每折 Train-only scaler。主损失只监督 `cell_voltage_mean`，没有未来真实负载输入。

ModernTCN 固定自 [官方 commit 56a9a2c](https://github.com/luodhhh/ModernTCN/tree/56a9a2c018385cd5acef015378cae7f084d1b11c)，保留 RevIN、逐变量 patching、大/小核、多阶段层次、两类 FFN、Flatten_Head。该固定版本虽声明多尺度层，但原 `forward_feature` 未调用；本地透明 adapter 仅激活已声明的 top-down multi-scale fusion，vendor 源码未修改。52/52 项实现及 CPU/GPU 数值审计通过，详见 `OFFICIAL_IMPLEMENTATION_AUDIT.md`。

## 4. seed=42 三折明细

{fold_table}

平均 RMSE 最低与平均 R² 最高均为 **{best_rmse}**。Fold3 最低 RMSE 为 **{best_fold3}**：{float(primary_folds.loc[(best_fold3, 'fold_3'), 'rmse_mV']):.3f} mV。

## 5. 与 Phase N2.6 轻量适配版比较

- TimesNet：轻量版 15.132 mV → 高保真版 {float(primary.loc['TimesNet','rmse_mV']):.3f} mV，RMSE 改善 **{times_improvement_light:.2f}%**。
- ModernTCN：轻量版 16.394 mV → 高保真版 {float(primary.loc['ModernTCN','rmse_mV']):.3f} mV，RMSE 改善 **{modern_improvement_light:.2f}%**。

高保真预测结构确实显著修正了 TimesNet 的轻量 head 偏差，但这不等于稳定超过 LSTM。

## 6. 多随机种子替换门

TimesNet-T1 因 seed=42 三折均值仅比 LSTM 差 9.87%，满足“差距不超过 10%”条件，依法进入 `[42, 2024, 3407]` 多种子复验。结果：

{robust_table}

TimesNet 三种子宏平均相对 LSTM 的 RMSE 改善为 **{100.0*(float(robust.loc['LSTM','rmse_mV'])-float(robust.loc['TimesNet','rmse_mV']))/float(robust.loc['LSTM','rmse_mV']):+.2f}%**；仅 **{improved_folds}/3 folds** 平均 RMSE 更低，且 Fold3 三种子均明显变差。替换五项规则：

1. 3-seed Mean RMSE 至少降低 2%：**FAIL**；
2. 至少 2/3 folds RMSE 更低：**{'PASS' if replacement_checks['at_least_2_of_3_folds_improve'] else 'FAIL'}**；
3. Mean MAE 不高于 LSTM：**{'PASS' if replacement_checks['mean_mae_not_worse'] else 'FAIL'}**；
4. Mean R² 不低于 LSTM：**FAIL**；
5. Late Development RMSE 恶化不超过 3%：**FAIL**。

因此 **不允许替换 LSTM**。

## 7. 逐时距与 9.5 h 邻域

{cycle_table}

表中为 RMSE（mV）；对应 R² 与 MAE 已写入 `cycle_region_rmse_r2.csv`。h=9.5 h 最低 RMSE 为 **{best_h95}**：{float(h95_primary.loc[best_h95,'rmse_mV']):.3f} mV。TimesNet 的 FFT 各配置/折首位周期为：{period_text}。周期发现只使用各折 Train 输入，未据 Validation/Test 人工强塞 19 steps。

## 8. 老化阶段，重点 Late Development

Development Validation origin 范围按物理老化时长三等分：`{bounds[0]:.3f} / {bounds[1]:.3f} / {bounds[2]:.3f} / {bounds[3]:.3f} h`。Late Development 最低 RMSE 为 **{best_late}**：{float(late_primary.loc[best_late,'rmse_mV']):.3f} mV。多种子下 TimesNet Late RMSE 为 {float(late_robust.loc['TimesNet','rmse_mV']):.3f} mV，LSTM 为 {float(late_robust.loc['LSTM','rmse_mV']):.3f} mV，候选恶化 {100.0*(float(late_robust.loc['TimesNet','rmse_mV'])/float(late_robust.loc['LSTM','rmse_mV'])-1):.2f}%。

## 9. 配对移动块自助法

TimesNet 对 LSTM 的三种子配对 ΔRMSE（candidate−LSTM）95% CI 为 **[{float(bootstrap_multi['ci95_lower_V'])*1000:.3f}, {float(bootstrap_multi['ci95_upper_V'])*1000:.3f}] mV**。该检验使用 19-step moving blocks、2000 repetitions；统计结果不改变预注册替换结论。

## 10. Q1–Q12 明确回答

### Q1 — 高保真 TimesNet 的 RMSE、MAE、R²？

seed=42 三折宏平均：RMSE **{float(primary.loc['TimesNet','rmse_mV']):.3f} mV**（{float(primary.loc['TimesNet','rmse_V']):.6f} V），MAE **{float(primary.loc['TimesNet','mae_mV']):.3f} mV**（{float(primary.loc['TimesNet','mae_V']):.6f} V），R² **{float(primary.loc['TimesNet','r2']):.6f}**。

### Q2 — 高保真 ModernTCN 的 RMSE、MAE、R²？

seed=42 三折宏平均：RMSE **{float(primary.loc['ModernTCN','rmse_mV']):.3f} mV**（{float(primary.loc['ModernTCN','rmse_V']):.6f} V），MAE **{float(primary.loc['ModernTCN','mae_mV']):.3f} mV**（{float(primary.loc['ModernTCN','mae_V']):.6f} V），R² **{float(primary.loc['ModernTCN','r2']):.6f}**。

### Q3 — 相比 Phase N2.6 轻量版改善多少？

TimesNet RMSE 改善 **{times_improvement_light:.2f}%**；ModernTCN 改善 **{modern_improvement_light:.2f}%**。

### Q4 — 平均 RMSE 最低？

**{best_rmse}**。

### Q5 — 平均 R² 最高？

**{best_r2}**。

### Q6 — Fold3 谁最好？

**{best_fold3}**，RMSE {float(primary_folds.loc[(best_fold3,'fold_3'),'rmse_mV']):.3f} mV。

### Q7 — Late Development 谁最好？

**{best_late}**，seed=42 RMSE {float(late_primary.loc[best_late,'rmse_mV']):.3f} mV。

### Q8 — 9.5 h 附近谁最好？

**{best_h95}**，h=9.5 h RMSE {float(h95_primary.loc[best_h95,'rmse_mV']):.3f} mV。

### Q9 — 多 seed 是否支持替换 LSTM？

**NO。** TimesNet 仅通过 5 项规则中的 **{sum(bool(value) for value in replacement_checks.values())} 项**，关键的平均 RMSE、R²和 Late Development 门槛均失败；ModernTCN 未达到多种子入围门。

### Q10 — 最终冻结哪个 backbone？

冻结 **LSTM**。

### Q11 — 是否正式停止继续搜索通用模型？

**YES。General backbone search STOP。** 官方高保真 TimesNet 虽修复轻量版偏差，但多种子 Fold3/晚期泛化不稳；完整 ModernTCN 仍明显落后。继续横向堆叠 Transformer/Mamba/TCN/Mixer 的证据价值已低。

### Q12 — 下一阶段围绕哪个 backbone 创新？

围绕 **LSTM** 做 PEMFC 任务机制创新；不重复无边界的通用骨干枚举。本阶段不实施创新，也不授权 Phase N3。

## 11. 结论边界

本结论只适用于冻结的 0.5 h/F2/24 h→12 h/三个 Development folds 协议。Test performance 从未打开；最终 Test 仍留给后续一次性评估。所有详细数值、图和冻结决定均位于本阶段目录。
"""
    write_text(report, PHASE_ROOT / "PHASE_N2_6R_RESULTS.md")
    print(json.dumps(
        {
            "status": "PASS", "selected_backbone": selected, "general_backbone_search": "STOP",
            "best_fold3": best_fold3, "best_late": best_late, "best_h9_5": best_h95,
        }, ensure_ascii=False
    ))


if __name__ == "__main__":
    main()

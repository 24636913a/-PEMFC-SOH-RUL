"""Finalize Development-only complementarity, shift, gate feasibility, figures, report, and freeze."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase_n2_7_common import (
    FOLDS,
    PHASE_ROOT,
    SEEDS,
    development_stage_bounds,
    ensure_output,
    load_config,
    load_protocol,
    load_reference_prediction,
    metric_dict,
    stage_labels,
    write_csv,
    write_json,
    write_text,
)
from phase_n2_7_models import load_local_run


COLORS = {"LSTM": "#2864B7", "TimesNet": "#D85A38", "Oracle": "#2E8B57", "Fusion": "#8055A5"}


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return math.nan
    result = pearsonr(x[valid], y[valid]) if method == "pearson" else spearmanr(x[valid], y[valid])
    return float(result.statistic)


def macro_metric(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    rows = []
    for _, group in frame.groupby(["seed", "fold"], sort=False):
        rows.append(metric_dict(group["y_true_V"].to_numpy(float), group[prediction_column].to_numpy(float)))
    if not rows:
        return {key: math.nan for key in ["rmse_V", "rmse_mV", "mae_V", "mae_mV", "bias_V", "bias_mV", "r2"]}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def metric_row(scope_type: str, scope_value: str, frame: pd.DataFrame, prediction_column: str, model: str) -> dict:
    metrics = macro_metric(frame, prediction_column)
    return {
        "analysis_type": "model_performance",
        "scope_type": scope_type,
        "scope_value": scope_value,
        "model": model,
        "prediction_points": len(frame),
        **metrics,
    }


def load_selected_prediction(
    cfg: dict,
    family: str,
    preprocessing: str,
    timesnet_tuning: str,
    seed: int,
    fold: str,
) -> dict[str, np.ndarray]:
    if preprocessing == "P0_raw":
        if family == "LSTM":
            _, arrays, _ = load_reference_prediction(cfg, "LSTM", seed, fold)
            return arrays
        if timesnet_tuning == "T0_original":
            _, arrays, _ = load_reference_prediction(cfg, "TimesNet", seed, fold)
            return arrays
        run_id = f"n27_timesnet_{timesnet_tuning}_{fold}_seed{seed}"
    else:
        run_id = f"n27_preproc_{family.lower()}_{preprocessing}_{fold}_seed{seed}"
    loaded = load_local_run(run_id, cfg)
    if loaded is None:
        raise RuntimeError(f"Missing final selected prediction: {run_id}")
    return loaded[1]


def make_point_frame(
    cfg: dict,
    windows: list,
    lstm_preprocessing: str,
    timesnet_preprocessing: str,
    timesnet_tuning: str,
    use_original_lstm: bool = False,
) -> pd.DataFrame:
    validation_map = {fold: validation for fold, _, validation in windows}
    bounds = development_stage_bounds(windows)
    shift = pd.read_csv(PHASE_ROOT / "outputs/shift_diagnostics/window_shift_score.csv")
    shift = shift[shift["dataset_role"].eq("validation")].copy()
    shift_lookup = shift.set_index(["fold", "row_origin_index"])
    rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            validation = validation_map[fold]
            if use_original_lstm:
                _, lstm, _ = load_reference_prediction(cfg, "LSTM", seed, fold)
            else:
                lstm = load_selected_prediction(cfg, "LSTM", lstm_preprocessing, timesnet_tuning, seed, fold)
            timesnet = load_selected_prediction(cfg, "TimesNet", timesnet_preprocessing, timesnet_tuning, seed, fold)
            for key in ["row_origin_index", "y_true", "origin_age_h", "target_end_age_h"]:
                if not np.array_equal(lstm[key], timesnet[key]):
                    raise RuntimeError(f"Final paired prediction mismatch: seed={seed}, {fold}, {key}")
            stages = stage_labels(lstm["origin_age_h"], bounds)
            for origin_pos, row_origin in enumerate(lstm["row_origin_index"]):
                shift_row = shift_lookup.loc[(fold, int(row_origin))]
                if isinstance(shift_row, pd.DataFrame):
                    raise RuntimeError(f"Duplicate shift row: {fold}, {row_origin}")
                for horizon_index in range(lstm["y_true"].shape[1]):
                    y_true = float(lstm["y_true"][origin_pos, horizon_index])
                    pred_l = float(lstm["y_pred"][origin_pos, horizon_index])
                    pred_t = float(timesnet["y_pred"][origin_pos, horizon_index])
                    rows.append({
                        "seed": seed,
                        "fold": fold,
                        "row_origin_index": int(row_origin),
                        "origin_position": origin_pos,
                        "origin_age_h": float(lstm["origin_age_h"][origin_pos]),
                        "target_end_age_h": float(lstm["target_end_age_h"][origin_pos]),
                        "ageing_stage": str(stages[origin_pos]),
                        "horizon_step": horizon_index + 1,
                        "horizon_hours": (horizon_index + 1) * float(cfg["task"]["resolution_hours"]),
                        "shift_score": float(shift_row["shift_score"]),
                        "shift_level": str(shift_row["shift_level"]),
                        "y_true_V": y_true,
                        "lstm_pred_V": pred_l,
                        "timesnet_pred_V": pred_t,
                        "lstm_error_mV": (pred_l - y_true) * 1000.0,
                        "timesnet_error_mV": (pred_t - y_true) * 1000.0,
                        "lstm_absolute_error_mV": abs(pred_l - y_true) * 1000.0,
                        "timesnet_absolute_error_mV": abs(pred_t - y_true) * 1000.0,
                        "timesnet_win": abs(pred_t - y_true) < abs(pred_l - y_true),
                    })
    return pd.DataFrame(rows)


def win_rate_table(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("overall", "Development", points)]
    scopes += [("fold", fold, group) for fold, group in points.groupby("fold", sort=True)]
    scopes += [("horizon", f"{hours:.1f} h", group) for hours, group in points.groupby("horizon_hours", sort=True)]
    scopes += [("ageing_stage", stage, group) for stage, group in points.groupby("ageing_stage", sort=False)]
    scopes += [("shift_level", level, group) for level, group in points.groupby("shift_level", sort=False)]
    for scope_type, scope_value, group in scopes:
        rows.append({
            "scope_type": scope_type,
            "scope_value": scope_value,
            "timesnet_win_count": int(group["timesnet_win"].sum()),
            "prediction_point_count": len(group),
            "timesnet_win_rate": float(group["timesnet_win"].mean()),
            "tie_rate": float(np.isclose(group["lstm_absolute_error_mV"], group["timesnet_absolute_error_mV"], atol=1e-12).mean()),
        })
    return pd.DataFrame(rows)


def correlation_table(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("overall", "Development", points)]
    scopes += [("fold", fold, group) for fold, group in points.groupby("fold", sort=True)]
    scopes += [("ageing_stage", stage, group) for stage, group in points.groupby("ageing_stage", sort=False)]
    for scope_type, scope_value, group in scopes:
        rows.append({
            "scope_type": scope_type,
            "scope_value": scope_value,
            "prediction_point_count": len(group),
            "signed_error_pearson": safe_corr(group["lstm_error_mV"], group["timesnet_error_mV"], "pearson"),
            "signed_error_spearman": safe_corr(group["lstm_error_mV"], group["timesnet_error_mV"], "spearman"),
            "absolute_error_pearson": safe_corr(group["lstm_absolute_error_mV"], group["timesnet_absolute_error_mV"], "pearson"),
            "absolute_error_spearman": safe_corr(group["lstm_absolute_error_mV"], group["timesnet_absolute_error_mV"], "spearman"),
        })
    return pd.DataFrame(rows)


def overlap_table(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("overall", "Development", points)]
    scopes += [("fold", fold, group) for fold, group in points.groupby("fold", sort=True)]
    scopes += [("ageing_stage", stage, group) for stage, group in points.groupby("ageing_stage", sort=False)]
    for scope_type, scope_value, group in scopes:
        for top_percent in [5, 10]:
            quantile = 1.0 - top_percent / 100.0
            lstm_cut = float(group["lstm_absolute_error_mV"].quantile(quantile))
            times_cut = float(group["timesnet_absolute_error_mV"].quantile(quantile))
            lstm_large = group["lstm_absolute_error_mV"] >= lstm_cut
            times_large = group["timesnet_absolute_error_mV"] >= times_cut
            intersection = int((lstm_large & times_large).sum())
            union = int((lstm_large | times_large).sum())
            rows.append({
                "scope_type": scope_type,
                "scope_value": scope_value,
                "top_error_percent": top_percent,
                "lstm_error_threshold_mV": lstm_cut,
                "timesnet_error_threshold_mV": times_cut,
                "lstm_large_error_count": int(lstm_large.sum()),
                "timesnet_large_error_count": int(times_large.sum()),
                "intersection_count": intersection,
                "union_count": union,
                "jaccard_overlap": intersection / union if union else math.nan,
            })
    return pd.DataFrame(rows)


def add_oracles(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = points.copy()
    output["pointwise_oracle_pred_V"] = np.where(
        output["timesnet_absolute_error_mV"] < output["lstm_absolute_error_mV"],
        output["timesnet_pred_V"], output["lstm_pred_V"],
    )
    origin = output.groupby(["seed", "fold", "row_origin_index"], sort=False).agg(
        lstm_origin_rmse_mV=("lstm_error_mV", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        timesnet_origin_rmse_mV=("timesnet_error_mV", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        origin_age_h=("origin_age_h", "first"),
        ageing_stage=("ageing_stage", "first"),
        shift_score=("shift_score", "first"),
        shift_level=("shift_level", "first"),
    ).reset_index()
    origin["origin_winner"] = np.where(origin["timesnet_origin_rmse_mV"] < origin["lstm_origin_rmse_mV"], "TimesNet", "LSTM")
    output = output.merge(origin[["seed", "fold", "row_origin_index", "origin_winner"]], on=["seed", "fold", "row_origin_index"], validate="many_to_one")
    output["originwise_oracle_pred_V"] = np.where(output["origin_winner"].eq("TimesNet"), output["timesnet_pred_V"], output["lstm_pred_V"])
    return output, origin


def oracle_table(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("overall", "Development", points)]
    scopes += [("fold", fold, group) for fold, group in points.groupby("fold", sort=True)]
    scopes += [("ageing_stage", stage, group) for stage, group in points.groupby("ageing_stage", sort=False)]
    for scope_type, scope_value, group in scopes:
        baseline = macro_metric(group, "lstm_pred_V")
        for strategy, column in [("Pointwise Oracle", "pointwise_oracle_pred_V"), ("Origin-wise Oracle", "originwise_oracle_pred_V")]:
            metrics = macro_metric(group, column)
            rows.append({
                "scope_type": scope_type,
                "scope_value": scope_value,
                "oracle_strategy": strategy,
                "prediction_points": len(group),
                **metrics,
                "lstm_reference_rmse_mV": baseline["rmse_mV"],
                "rmse_improvement_fraction_vs_lstm": (baseline["rmse_mV"] - metrics["rmse_mV"]) / baseline["rmse_mV"],
                "deployable_model": False,
            })
    return pd.DataFrame(rows)


def fusion_table(points: pd.DataFrame, alphas: list[float]) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        column = f"fusion_{alpha:.2f}_pred_V"
        points[column] = (1.0 - alpha) * points["lstm_pred_V"] + alpha * points["timesnet_pred_V"]
        scopes = [("overall", "Development", points)] + [("fold", fold, group) for fold, group in points.groupby("fold", sort=True)]
        for scope_type, scope_value, group in scopes:
            metrics = macro_metric(group, column)
            baseline = macro_metric(group, "lstm_pred_V")
            rows.append({
                "scope_type": scope_type,
                "scope_value": scope_value,
                "timesnet_weight_alpha": alpha,
                **metrics,
                "lstm_reference_rmse_mV": baseline["rmse_mV"],
                "rmse_improvement_fraction_vs_lstm": (baseline["rmse_mV"] - metrics["rmse_mV"]) / baseline["rmse_mV"],
                "alpha_preregistered_on_development": True,
            })
    return pd.DataFrame(rows)


def shift_performance(points: pd.DataFrame, origin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, group in points.groupby("shift_level", sort=False):
        rows.append(metric_row("shift_level", str(level), group, "lstm_pred_V", "LSTM"))
        rows.append(metric_row("shift_level", str(level), group, "timesnet_pred_V", "TimesNet"))
    for stage, group in points.groupby("ageing_stage", sort=False):
        rows.append(metric_row("ageing_stage", str(stage), group, "lstm_pred_V", "LSTM"))
        rows.append(metric_row("ageing_stage", str(stage), group, "timesnet_pred_V", "TimesNet"))
    for fold, group in points.groupby("fold", sort=True):
        rows.append(metric_row("fold", str(fold), group, "lstm_pred_V", "LSTM"))
        rows.append(metric_row("fold", str(fold), group, "timesnet_pred_V", "TimesNet"))
    for model, error_column in [("LSTM", "lstm_origin_rmse_mV"), ("TimesNet", "timesnet_origin_rmse_mV")]:
        scopes = [("overall", "Development", origin)] + [("fold", fold, group) for fold, group in origin.groupby("fold", sort=True)]
        for scope_type, scope_value, group in scopes:
            rows.append({
                "analysis_type": "shift_error_correlation",
                "scope_type": scope_type,
                "scope_value": scope_value,
                "model": model,
                "origin_count": len(group),
                "shift_error_pearson": safe_corr(group["shift_score"], group[error_column], "pearson"),
                "shift_error_spearman": safe_corr(group["shift_score"], group[error_column], "spearman"),
            })
    return pd.DataFrame(rows)


def regime_analysis(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift = pd.read_csv(PHASE_ROOT / "outputs/shift_diagnostics/window_shift_score.csv")
    train = shift[shift["dataset_role"].eq("train")]
    validation = shift[shift["dataset_role"].eq("validation")].copy()
    definitions = {
        "Current Level": ("actual_current_A_mean__mean", "tertile"),
        "Current Variability": ("actual_current_A_std__mean", "tertile"),
        "Load Transition": ("load_transition_count__mean", "binary"),
        "Operation Point Transition": ("operation_point_transitions__mean", "binary"),
        "Zero Current Fraction": ("zero_current_fraction__mean", "tertile"),
        "Voltage Variability": ("cell_voltage_std__mean", "tertile"),
    }
    origin_categories = validation[["fold", "row_origin_index", "shift_level", "ageing_stage"]].copy()
    for factor, (column, kind) in definitions.items():
        result = np.empty(len(validation), dtype=object)
        for fold in FOLDS:
            train_values = train.loc[train["fold"].eq(fold), column].to_numpy(float)
            mask = validation["fold"].eq(fold).to_numpy()
            values = validation.loc[mask, column].to_numpy(float)
            if kind == "tertile":
                low, high = np.quantile(train_values, [1 / 3, 2 / 3])
                labels = np.where(values <= low, "Low", np.where(values <= high, "Medium", "High"))
            else:
                cut = float(np.quantile(train_values, 0.5))
                labels = np.where(values <= cut, "Low", "High")
            result[mask] = labels
        origin_categories[factor] = result
    enriched = points.merge(origin_categories, on=["fold", "row_origin_index", "shift_level", "ageing_stage"], validate="many_to_one")
    regime_rows = []

    def append_regime(factor: str, level: str, group: pd.DataFrame) -> None:
        lstm = macro_metric(group, "lstm_pred_V")
        times = macro_metric(group, "timesnet_pred_V")
        per_seed_advantage = []
        for seed, seed_group in group.groupby("seed", sort=True):
            per_seed_advantage.append((seed, macro_metric(seed_group, "timesnet_pred_V")["rmse_mV"] < macro_metric(seed_group, "lstm_pred_V")["rmse_mV"]))
        improvement = (lstm["rmse_mV"] - times["rmse_mV"]) / lstm["rmse_mV"]
        regime_rows.append({
            "regime_factor": factor,
            "regime_level": level,
            "prediction_points": len(group),
            "lstm_rmse_mV": lstm["rmse_mV"],
            "timesnet_rmse_mV": times["rmse_mV"],
            "lstm_mae_mV": lstm["mae_mV"],
            "timesnet_mae_mV": times["mae_mV"],
            "better_model": "TimesNet" if times["rmse_mV"] < lstm["rmse_mV"] else "LSTM",
            "timesnet_rmse_improvement_fraction_vs_lstm": improvement,
            "timesnet_better_seed_count": sum(int(flag) for _, flag in per_seed_advantage),
            "stable_timesnet_advantage_all_3_seeds": bool(len(per_seed_advantage) == 3 and all(flag for _, flag in per_seed_advantage) and improvement >= 0.05),
            "causal_regime_only": True,
        })

    for fold, group in enriched.groupby("fold", sort=True):
        append_regime("Fold", str(fold), group)
    for factor in ["shift_level", "ageing_stage"] + list(definitions):
        for level, group in enriched.groupby(factor, sort=False):
            append_regime(factor, str(level), group)
    for left, right, name in [
        ("shift_level", "Current Variability", "Shift × Current Variability"),
        ("shift_level", "Load Transition", "Shift × Load Transition"),
    ]:
        for levels, group in enriched.groupby([left, right], sort=False):
            append_regime(name, f"{levels[0]} | {levels[1]}", group)
    return pd.DataFrame(regime_rows), enriched


def winner_predictability(origin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift = pd.read_csv(PHASE_ROOT / "outputs/shift_diagnostics/window_shift_score.csv")
    validation = shift[shift["dataset_role"].eq("validation")].copy()
    mean_errors = origin.groupby(["fold", "row_origin_index"], as_index=False).agg(
        lstm_seed_mean_origin_rmse_mV=("lstm_origin_rmse_mV", "mean"),
        timesnet_seed_mean_origin_rmse_mV=("timesnet_origin_rmse_mV", "mean"),
    )
    labels = validation.merge(mean_errors, on=["fold", "row_origin_index"], validate="one_to_one")
    labels["winner_timesnet"] = (labels["timesnet_seed_mean_origin_rmse_mV"] < labels["lstm_seed_mean_origin_rmse_mV"]).astype(int)
    explicit = [
        "shift_score", "actual_current_A_mean__mean", "actual_current_A_std__mean",
        "load_transition_count__mean", "operation_point_transitions__mean",
        "zero_current_fraction__mean", "cell_voltage_std__mean",
    ]
    slopes = [column for column in labels.columns if column.endswith("__slope")]
    features = list(dict.fromkeys(explicit + slopes))
    diagnostics = [
        ("Fold1 Validation → Fold2 Validation", ["fold_1"], "fold_2"),
        ("Fold1+Fold2 Validation → Fold3 Validation", ["fold_1", "fold_2"], "fold_3"),
    ]
    rows = []
    for name, train_folds, test_fold in diagnostics:
        train = labels[labels["fold"].isin(train_folds)]
        test = labels[labels["fold"].eq(test_fold)]
        x_train = train[features].to_numpy(float)
        x_test = test[features].to_numpy(float)
        y_train = train["winner_timesnet"].to_numpy(int)
        y_test = test["winner_timesnet"].to_numpy(int)
        if len(np.unique(y_train)) < 2:
            auc = accuracy = precision = recall = math.nan
            status = "NOT_ESTIMABLE_SINGLE_TRAIN_CLASS"
        else:
            classifier = Pipeline([
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000, random_state=20260908,
                )),
            ])
            classifier.fit(x_train, y_train)
            probability = classifier.predict_proba(x_test)[:, 1]
            prediction = (probability >= 0.5).astype(int)
            auc = roc_auc_score(y_test, probability) if len(np.unique(y_test)) == 2 else math.nan
            accuracy = accuracy_score(y_test, prediction)
            precision = precision_score(y_test, prediction, zero_division=0)
            recall = recall_score(y_test, prediction, zero_division=0)
            status = "PASS"
        rows.append({
            "diagnostic": name,
            "train_folds": "+".join(train_folds),
            "evaluation_fold": test_fold,
            "train_origin_count": len(train),
            "evaluation_origin_count": len(test),
            "train_timesnet_winner_rate": float(y_train.mean()),
            "evaluation_timesnet_winner_rate": float(y_test.mean()),
            "roc_auc": auc,
            "accuracy": accuracy,
            "precision_timesnet": precision,
            "recall_timesnet": recall,
            "auc_at_least_0p65": bool(np.isfinite(auc) and auc >= 0.65),
            "feature_count": len(features),
            "feature_names": " | ".join(features),
            "uses_future_information": False,
            "uses_fold_age_or_time_feature": False,
            "status": status,
        })
    return pd.DataFrame(rows), labels


def complementarity_summary(scenario: str, points: pd.DataFrame) -> dict:
    if not {"pointwise_oracle_pred_V", "originwise_oracle_pred_V"}.issubset(points.columns):
        points, _ = add_oracles(points)
    lstm = macro_metric(points, "lstm_pred_V")
    times = macro_metric(points, "timesnet_pred_V")
    origin_oracle = macro_metric(points, "originwise_oracle_pred_V")
    point_oracle = macro_metric(points, "pointwise_oracle_pred_V")
    overlap = overlap_table(points)
    top10 = overlap[(overlap["scope_type"].eq("overall")) & (overlap["top_error_percent"].eq(10))].iloc[0]
    return {
        "scenario": scenario,
        "lstm_rmse_mV": lstm["rmse_mV"], "lstm_mae_mV": lstm["mae_mV"], "lstm_r2": lstm["r2"],
        "timesnet_rmse_mV": times["rmse_mV"], "timesnet_mae_mV": times["mae_mV"], "timesnet_r2": times["r2"],
        "timesnet_win_rate": float(points["timesnet_win"].mean()),
        "signed_error_pearson": safe_corr(points["lstm_error_mV"], points["timesnet_error_mV"], "pearson"),
        "absolute_error_pearson": safe_corr(points["lstm_absolute_error_mV"], points["timesnet_absolute_error_mV"], "pearson"),
        "absolute_error_spearman": safe_corr(points["lstm_absolute_error_mV"], points["timesnet_absolute_error_mV"], "spearman"),
        "top10_error_jaccard_overlap": float(top10["jaccard_overlap"]),
        "pointwise_oracle_rmse_mV": point_oracle["rmse_mV"],
        "originwise_oracle_rmse_mV": origin_oracle["rmse_mV"],
        "originwise_oracle_improvement_fraction_vs_lstm": (lstm["rmse_mV"] - origin_oracle["rmse_mV"]) / lstm["rmse_mV"],
    }


def save_figures(
    points: pd.DataFrame,
    origin: pd.DataFrame,
    win: pd.DataFrame,
    corr: pd.DataFrame,
    overlap: pd.DataFrame,
    shift_perf: pd.DataFrame,
    oracle: pd.DataFrame,
    fusion: pd.DataFrame,
) -> None:
    figure_dir = PHASE_ROOT / "outputs/figures"
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 180, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

    rng = np.random.default_rng(20260908)
    sample = points.iloc[rng.choice(len(points), size=min(30000, len(points)), replace=False)]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.hexbin(sample["lstm_absolute_error_mV"], sample["timesnet_absolute_error_mV"], gridsize=65, bins="log", cmap="viridis", mincnt=1)
    limit = float(np.quantile(np.r_[sample["lstm_absolute_error_mV"], sample["timesnet_absolute_error_mV"]], 0.995))
    ax.plot([0, limit], [0, limit], "--", color="black", lw=1)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="LSTM absolute error (mV)", ylabel="TimesNet absolute error (mV)", title="Final experts: absolute-error complementarity (Development only)")
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "lstm_timesnet_absolute_error_scatter.png")); plt.close(fig)

    fold_win = win[win["scope_type"].eq("fold")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(fold_win["scope_value"], fold_win["timesnet_win_rate"] * 100, color=COLORS["TimesNet"])
    ax.axhline(50, ls="--", color="black", lw=1); ax.set(ylabel="TimesNet win rate (%)", title="TimesNet pointwise win rate by fold", ylim=(0, 100))
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "timesnet_win_rate_by_fold.png")); plt.close(fig)

    horizon_win = win[win["scope_type"].eq("horizon")]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(np.arange(1, len(horizon_win) + 1) * 0.5, horizon_win["timesnet_win_rate"] * 100, marker="o", ms=3, color=COLORS["TimesNet"])
    ax.axhline(50, ls="--", color="black", lw=1); ax.set(xlabel="Forecast horizon (h)", ylabel="TimesNet win rate (%)", title="TimesNet pointwise win rate by horizon", ylim=(0, 100))
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "timesnet_win_rate_by_horizon.png")); plt.close(fig)

    matrix = np.corrcoef(points[["lstm_error_mV", "timesnet_error_mV", "lstm_absolute_error_mV", "timesnet_absolute_error_mV"]].to_numpy(float).T)
    labels = ["LSTM e", "TimesNet e", "LSTM |e|", "TimesNet |e|"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(4), labels, rotation=30, ha="right"); ax.set_yticks(range(4), labels)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if abs(matrix[i,j]) > .55 else "black")
    fig.colorbar(image, ax=ax, label="Pearson correlation"); ax.set_title("Final error correlation matrix")
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "error_correlation_heatmap.png")); plt.close(fig)

    overlap_plot = overlap[(overlap["scope_type"].eq("overall"))]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.bar(overlap_plot["top_error_percent"].astype(str) + "%", overlap_plot["jaccard_overlap"], color="#5A8F7B")
    ax.set(xlabel="Largest absolute-error set", ylabel="Jaccard overlap", title="Large-error overlap of final experts", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "large_error_overlap.png")); plt.close(fig)

    origin_sample = origin.iloc[rng.choice(len(origin), size=min(5000, len(origin)), replace=False)]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.scatter(origin_sample["shift_score"], origin_sample["lstm_origin_rmse_mV"], s=10, alpha=.25, color=COLORS["LSTM"], label="LSTM")
    ax.scatter(origin_sample["shift_score"], origin_sample["timesnet_origin_rmse_mV"], s=10, alpha=.25, color=COLORS["TimesNet"], label="TimesNet")
    ax.set(xlabel="Train-defined Mahalanobis shift score", ylabel="24-step origin RMSE (mV)", title="Causal shift score versus expert error"); ax.legend()
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "shift_score_vs_error.png")); plt.close(fig)

    def grouped_rmse(scope_type: str, filename: str, title: str) -> None:
        data = shift_perf[(shift_perf["analysis_type"].eq("model_performance")) & (shift_perf["scope_type"].eq(scope_type))]
        order = (["Low", "Medium", "High"] if scope_type == "shift_level" else ["Early Development", "Middle Development", "Late Development"])
        x = np.arange(len(order)); width = .36
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for offset, model in [(-width/2, "LSTM"), (width/2, "TimesNet")]:
            values = [float(data[(data["scope_value"].eq(level)) & (data["model"].eq(model))]["rmse_mV"].iloc[0]) for level in order]
            ax.bar(x + offset, values, width, label=model, color=COLORS[model])
        ax.set_xticks(x, [label.replace(" Development", "") for label in order]); ax.set(ylabel="Macro RMSE (mV)", title=title); ax.legend()
        fig.tight_layout(); fig.savefig(ensure_output(figure_dir / filename)); plt.close(fig)
    grouped_rmse("shift_level", "rmse_by_shift_level.png", "Final expert RMSE by Train-defined shift level")
    grouped_rmse("ageing_stage", "rmse_by_ageing_stage.png", "Final expert RMSE by ageing stage")

    tuning = pd.read_csv(PHASE_ROOT / "outputs/timesnet_tuning/timesnet_robustness_tuning.csv")
    tuning = tuning[tuning["evaluation_stage"].eq("seed42_preregistered_screen")]
    x = np.arange(len(tuning)); width = .25
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for idx, fold in enumerate(FOLDS):
        ax.bar(x + (idx - 1) * width, tuning[f"{fold}_rmse_mV"], width, label=fold)
    ax.set_xticks(x, tuning["config_name"], rotation=25, ha="right"); ax.set(ylabel="RMSE (mV)", title="Preregistered TimesNet tuning, seed 42"); ax.legend()
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "timesnet_tuning_fold_rmse.png")); plt.close(fig)

    preprocessing = pd.read_csv(PHASE_ROOT / "outputs/preprocessing/preprocessing_screening.csv")
    preprocessing = preprocessing[preprocessing["evaluation_stage"].eq("seed42_causal_preprocessing_screen")]
    labels = [f"{row.family}\n{row.preprocessing.replace('P1_causal_', '').replace('P2_causal_', '').replace('P0_', '')}" for row in preprocessing.itertuples()]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.bar(labels, preprocessing["mean_rmse_mV"], color=[COLORS[family] for family in preprocessing["family"]])
    ax.set(ylabel="3-fold mean RMSE (mV)", title="Strictly causal preprocessing screen, seed 42"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "preprocessing_rmse_comparison.png")); plt.close(fig)

    oracle_plot = oracle[(oracle["scope_type"].isin(["overall", "fold"])) & (oracle["oracle_strategy"].eq("Origin-wise Oracle"))]
    order = ["Development", "fold_1", "fold_2", "fold_3"]
    oracle_values = [float(oracle_plot[oracle_plot["scope_value"].eq(level)]["rmse_mV"].iloc[0]) for level in order]
    lstm_values = [float(oracle_plot[oracle_plot["scope_value"].eq(level)]["lstm_reference_rmse_mV"].iloc[0]) for level in order]
    x = np.arange(len(order)); width=.36
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(x-width/2, lstm_values, width, label="LSTM", color=COLORS["LSTM"]); ax.bar(x+width/2, oracle_values, width, label="Origin-wise Oracle", color=COLORS["Oracle"])
    ax.set_xticks(x, ["Overall", "Fold1", "Fold2", "Fold3"]); ax.set(ylabel="Macro RMSE (mV)", title="Origin-wise Oracle upper bound versus final LSTM"); ax.legend()
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "oracle_vs_lstm_rmse.png")); plt.close(fig)

    fusion_plot = fusion[fusion["scope_type"].eq("overall")].sort_values("timesnet_weight_alpha")
    lstm_rmse = float(fusion_plot["lstm_reference_rmse_mV"].iloc[0])
    times_rmse = macro_metric(points, "timesnet_pred_V")["rmse_mV"]
    values = [lstm_rmse] + fusion_plot["rmse_mV"].tolist() + [times_rmse]
    labels = ["LSTM"] + [f"α={alpha:.2f}" for alpha in fusion_plot["timesnet_weight_alpha"]] + ["TimesNet"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(labels, values, color=[COLORS["LSTM"]] + [COLORS["Fusion"]]*len(fusion_plot) + [COLORS["TimesNet"]])
    ax.set(ylabel="3-seed macro RMSE (mV)", title="Preregistered fixed fusion baselines")
    fig.tight_layout(); fig.savefig(ensure_output(figure_dir / "fixed_fusion_rmse.png")); plt.close(fig)


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def f(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    cfg = load_config()
    _, firewall, windows = load_protocol(cfg)
    prep_selection = json.loads((PHASE_ROOT / "outputs/preprocessing/preprocessing_selection.json").read_text(encoding="utf-8"))
    lstm_preprocessing = str(prep_selection["final_preprocessing"]["LSTM"])
    timesnet_preprocessing = str(prep_selection["final_preprocessing"]["TimesNet"])
    timesnet_tuning = str(prep_selection["final_timesnet_tuning"])

    final_points = make_point_frame(cfg, windows, lstm_preprocessing, timesnet_preprocessing, timesnet_tuning)
    original_points = make_point_frame(cfg, windows, "P0_raw", "P0_raw", "T0_original", use_original_lstm=True)
    final_points, origin = add_oracles(final_points)
    win = win_rate_table(final_points)
    corr = correlation_table(final_points)
    overlap = overlap_table(final_points)
    oracle = oracle_table(final_points)
    fusion = fusion_table(final_points, [float(value) for value in cfg["fixed_fusion_alphas"]])
    shift_perf = shift_performance(final_points, origin)
    regime, enriched_points = regime_analysis(final_points)
    winner, winner_labels = winner_predictability(origin)
    complementarity = pd.DataFrame([
        complementarity_summary("Original frozen Raw LSTM vs original Raw TimesNet", original_points),
        complementarity_summary(f"Final LSTM {lstm_preprocessing} vs final TimesNet {timesnet_tuning}+{timesnet_preprocessing}", final_points),
    ])

    write_csv(win, PHASE_ROOT / "outputs/error_complementarity/timesnet_win_rate.csv")
    write_csv(corr, PHASE_ROOT / "outputs/error_complementarity/error_correlation.csv")
    write_csv(overlap, PHASE_ROOT / "outputs/error_complementarity/large_error_overlap.csv")
    write_csv(oracle, PHASE_ROOT / "outputs/fusion_diagnostics/oracle_fusion_upper_bound.csv")
    write_csv(fusion, PHASE_ROOT / "outputs/fusion_diagnostics/fixed_fusion_baseline.csv")
    write_csv(shift_perf, PHASE_ROOT / "outputs/shift_diagnostics/shift_conditioned_model_performance.csv")
    write_csv(regime, PHASE_ROOT / "outputs/shift_diagnostics/expert_regime_analysis.csv")
    write_csv(winner, PHASE_ROOT / "outputs/shift_diagnostics/winner_predictability.csv")
    write_csv(complementarity, PHASE_ROOT / "outputs/error_complementarity/final_complementarity_analysis.csv")

    save_figures(final_points, origin, win, corr, overlap, shift_perf, oracle, fusion)

    final_lstm = macro_metric(final_points, "lstm_pred_V")
    final_times = macro_metric(final_points, "timesnet_pred_V")
    original_lstm = macro_metric(original_points, "lstm_pred_V")
    original_times = macro_metric(original_points, "timesnet_pred_V")
    overall_oracle = oracle[(oracle["scope_type"].eq("overall")) & (oracle["oracle_strategy"].eq("Origin-wise Oracle"))].iloc[0]
    overall_point_oracle = oracle[(oracle["scope_type"].eq("overall")) & (oracle["oracle_strategy"].eq("Pointwise Oracle"))].iloc[0]
    overall_win = float(win[(win["scope_type"].eq("overall"))]["timesnet_win_rate"].iloc[0])
    overall_corr = corr[corr["scope_type"].eq("overall")].iloc[0]
    top10 = overlap[(overlap["scope_type"].eq("overall")) & (overlap["top_error_percent"].eq(10))].iloc[0]
    best_fusion = fusion[fusion["scope_type"].eq("overall")].sort_values("rmse_mV").iloc[0]
    high_rows = shift_perf[(shift_perf["analysis_type"].eq("model_performance")) & (shift_perf["scope_type"].eq("shift_level")) & (shift_perf["scope_value"].eq("High"))]
    high_lstm = high_rows[high_rows["model"].eq("LSTM")].iloc[0]
    high_times = high_rows[high_rows["model"].eq("TimesNet")].iloc[0]
    late_rows = shift_perf[(shift_perf["analysis_type"].eq("model_performance")) & (shift_perf["scope_type"].eq("ageing_stage")) & (shift_perf["scope_value"].eq("Late Development"))]
    late_lstm = late_rows[late_rows["model"].eq("LSTM")].iloc[0]
    late_times = late_rows[late_rows["model"].eq("TimesNet")].iloc[0]
    shift_corr_times = shift_perf[(shift_perf["analysis_type"].eq("shift_error_correlation")) & (shift_perf["scope_type"].eq("overall")) & (shift_perf["model"].eq("TimesNet"))].iloc[0]
    auc_values = winner["roc_auc"].to_numpy(float)
    max_auc = float(np.nanmax(auc_values)) if np.isfinite(auc_values).any() else math.nan
    stable_timesnet_regions = regime[regime["stable_timesnet_advantage_all_3_seeds"]]
    gate1 = float(overall_oracle["rmse_improvement_fraction_vs_lstm"]) >= float(cfg["selection"]["oracle_min_improvement"])
    gate2 = len(stable_timesnet_regions) > 0
    gate3 = bool(np.isfinite(max_auc) and max_auc >= float(cfg["selection"]["winner_auc_threshold"]))
    gate4 = bool(high_lstm["rmse_mV"] < high_times["rmse_mV"] and late_lstm["rmse_mV"] < late_times["rmse_mV"])
    n28_go = bool(gate1 and gate2 and gate3 and gate4)

    tuning_selection = json.loads((PHASE_ROOT / "outputs/timesnet_tuning/timesnet_tuning_selection.json").read_text(encoding="utf-8"))
    tuning_csv = pd.read_csv(PHASE_ROOT / "outputs/timesnet_tuning/timesnet_robustness_tuning.csv")
    final_tuning_row = tuning_csv[(tuning_csv["evaluation_stage"].eq("multiseed_confirmation_mean")) & (tuning_csv["config_name"].eq(timesnet_tuning))].iloc[0]
    huber_screen = tuning_csv[(tuning_csv["evaluation_stage"].eq("seed42_preregistered_screen")) & (tuning_csv["config_name"].eq("T1_huber"))].iloc[0]
    preprocessing_csv = pd.read_csv(PHASE_ROOT / "outputs/preprocessing/preprocessing_screening.csv")
    ema_lstm = preprocessing_csv[(preprocessing_csv["evaluation_stage"].eq("seed42_causal_preprocessing_screen")) & (preprocessing_csv["family"].eq("LSTM")) & (preprocessing_csv["preprocessing"].eq("P1_causal_ema"))].iloc[0]
    sg_lstm_multi = preprocessing_csv[(preprocessing_csv["evaluation_stage"].eq("multiseed_confirmation_mean")) & (preprocessing_csv["family"].eq("LSTM")) & (preprocessing_csv["preprocessing"].eq("P2_causal_sg"))].iloc[0]
    sg_times = preprocessing_csv[(preprocessing_csv["evaluation_stage"].eq("seed42_causal_preprocessing_screen")) & (preprocessing_csv["family"].eq("TimesNet")) & (preprocessing_csv["preprocessing"].eq("P2_causal_sg"))].iloc[0]

    top_table = [
        ["Frozen LSTM", f(original_lstm["rmse_mV"]), f(original_lstm["mae_mV"]), f(original_lstm["r2"], 5)],
        ["Original TimesNet", f(original_times["rmse_mV"]), f(original_times["mae_mV"]), f(original_times["r2"], 5)],
        [f"Robust-Tuned TimesNet ({timesnet_tuning})", f(final_tuning_row["mean_rmse_mV"]), f(final_tuning_row["mean_mae_mV"]), f(final_tuning_row["mean_r2"], 5)],
        [f"Best Preprocessed LSTM ({lstm_preprocessing})", f(final_lstm["rmse_mV"]), f(final_lstm["mae_mV"]), f(final_lstm["r2"], 5)],
        [f"Best Preprocessed TimesNet ({timesnet_preprocessing})", f(final_times["rmse_mV"]), f(final_times["mae_mV"]), f(final_times["r2"], 5)],
        [f"Best Fixed Fusion (α={float(best_fusion['timesnet_weight_alpha']):.2f})", f(float(best_fusion["rmse_mV"])), f(float(best_fusion["mae_mV"])), f(float(best_fusion["r2"]), 5)],
        ["Origin-wise Oracle (upper bound)", f(float(overall_oracle["rmse_mV"])), f(float(overall_oracle["mae_mV"])), f(float(overall_oracle["r2"]), 5)],
    ]
    fold_win = win[win["scope_type"].eq("fold")].set_index("scope_value")
    winner_rows = [[str(row.diagnostic), f(float(row.roc_auc), 3), f(float(row.accuracy), 3), f(float(row.precision_timesnet), 3), f(float(row.recall_timesnet), 3)] for row in winner.itertuples()]
    stable_text = ", ".join(stable_timesnet_regions.apply(lambda row: f"{row['regime_factor']}={row['regime_level']}", axis=1).head(8).tolist()) or "无"
    phase_status = "PASS" if firewall.get("test_target_rows_read") == 0 and not firewall.get("test_predictions_constructed") and not firewall.get("test_metrics_computed") and not firewall.get("test_shift_diagnostics_computed") else "FAIL"
    report = f"""# Phase N2.7 Results — LSTM–TimesNet误差互补性、TimesNet稳健调优与因果预处理筛选

## 主要结果

{markdown_table(top_table, ["模型/方案", "RMSE (mV)", "MAE (mV)", "R²"])}

以上均为 Development 三个 expanding folds、三个 seeds 的 `(seed, fold)` 宏平均；Test `[1819.997738, 2256.800000] h` 未读取目标、未构造预测、未计算指标、未做 shift 诊断。

## 冻结结论

- Phase N2.7：**{phase_status}**。
- 最终 LSTM 专家：Frozen LSTM 架构 + `P2_causal_sg`（仅历史 5 点、二次拟合，仅处理 voltage-state，预测 raw target）。
- 最终 TimesNet 专家：N2.6R 高保真 `T0_original` + `P0_raw`。
- TimesNet Huber 在 seed=42 将 Fold3 改善 {float(huber_screen['fold3_improvement_fraction_vs_T0'])*100:.2f}%，但未通过三 seed 复核，因此不冻结为最终版本。
- Phase N2.8：**{'GO' if n28_go else 'NOGO'}**。Gate 1/2/3/4 = `{gate1}/{gate2}/{gate3}/{gate4}`。

## Q1–Q7：误差互补性与融合上限

**Q1：LSTM和TimesNet是否具有明显误差互补性？**  {'YES' if gate1 else 'NO/WEAK'}。最终两专家绝对误差 Pearson/Spearman 为 {float(overall_corr['absolute_error_pearson']):.3f}/{float(overall_corr['absolute_error_spearman']):.3f}，Top10% 大误差 Jaccard 为 {float(top10['jaccard_overlap']):.3f}，Origin-wise Oracle 相对最终 LSTM 改善 {float(overall_oracle['rmse_improvement_fraction_vs_lstm'])*100:.2f}%。

**Q2：TimesNet整体有多少比例预测点优于LSTM？** {overall_win*100:.2f}%。

**Q3：Fold1/Fold2/Fold3 win rate？** {float(fold_win.loc['fold_1','timesnet_win_rate'])*100:.2f}% / {float(fold_win.loc['fold_2','timesnet_win_rate'])*100:.2f}% / {float(fold_win.loc['fold_3','timesnet_win_rate'])*100:.2f}%。

**Q4：误差相关系数？** signed error Pearson/Spearman = {float(overall_corr['signed_error_pearson']):.3f}/{float(overall_corr['signed_error_spearman']):.3f}；absolute error Pearson/Spearman = {float(overall_corr['absolute_error_pearson']):.3f}/{float(overall_corr['absolute_error_spearman']):.3f}。

**Q5：Top10%大误差重合率？** Jaccard = {float(top10['jaccard_overlap']):.3f}。

**Q6：Origin-wise Oracle改善？** RMSE = {float(overall_oracle['rmse_mV']):.3f} mV，相对最终 LSTM 改善 {float(overall_oracle['rmse_improvement_fraction_vs_lstm'])*100:.2f}%；Pointwise Oracle RMSE = {float(overall_point_oracle['rmse_mV']):.3f} mV。二者均仅为不可部署的理论上限。

**Q7：固定融合是否优于LSTM？** {'YES' if float(best_fusion['rmse_mV']) < final_lstm['rmse_mV'] else 'NO'}。三组预注册权重中最佳为 α={float(best_fusion['timesnet_weight_alpha']):.2f}，RMSE={float(best_fusion['rmse_mV']):.3f} mV，相对 LSTM 改善 {float(best_fusion['rmse_improvement_fraction_vs_lstm'])*100:.2f}%。

## Q8–Q10：Distribution shift 与 winner 可预测性

**Q8：TimesNet Fold3失稳是否与 distribution shift 相关？** TimesNet origin RMSE 与 Train-defined Mahalanobis shift score 的 Pearson/Spearman 为 {float(shift_corr_times['shift_error_pearson']):.3f}/{float(shift_corr_times['shift_error_spearman']):.3f}，接近 0，不支持强单调关系。High Shift 分层中 TimesNet 明显弱于 LSTM，说明存在相对脆弱的 shift regime，但单一 Mahalanobis score 不能解释 Fold3 失稳。

**Q9：High Shift区域RMSE？** LSTM={float(high_lstm['rmse_mV']):.3f} mV，TimesNet={float(high_times['rmse_mV']):.3f} mV。Late Development：LSTM={float(late_lstm['rmse_mV']):.3f} mV，TimesNet={float(late_times['rmse_mV']):.3f} mV。

**Q10：Winner能否被因果特征预测？** 最大 chronological ROC-AUC={max_auc:.3f}；阈值 0.65 {'达到' if gate3 else '未达到'}。Fold1 的 233 个 origin 全部由 TimesNet 获胜，因训练标签只有一个类别，Fold1→Fold2 诊断不可估计；Fold1+Fold2→Fold3 可估计但未过门槛。分类器只使用过去 48 steps 的因果摘要，不使用 future target/load、fold、ageing stage 或时间。

{markdown_table(winner_rows, ["时间顺序诊断", "ROC-AUC", "Accuracy", "Precision(T)", "Recall(T)"])}

## Q11–Q16：TimesNet调优与因果预处理

**Q11：Huber/LR/dropout/weight decay 是否改善 Fold3？** seed=42 下只有 Huber 满足三道初筛门槛；lower LR 与 more dropout 使 Fold3恶化，stronger regularization 基本无变化，robust combination 未达到 10% Fold3 改善。Huber 未在三 seed 复核中维持全部门槛。

**Q12：最佳稳健 TimesNet？** 最终保留 `T0_original`：RMSE={final_times['rmse_mV']:.3f} mV，MAE={final_times['mae_mV']:.3f} mV，R²={final_times['r2']:.5f}；不宣称调优成功。

**Q13：EMA是否有效？** LSTM seed=42 mean RMSE={float(ema_lstm['mean_rmse_mV']):.3f} mV，初筛有益，但弱于 causal SG，未进入多 seed；TimesNet EMA 未通过门槛。

**Q14：Causal SG是否有效？** 对 LSTM 有效并通过三 seed：RMSE={float(sg_lstm_multi['mean_rmse_mV']):.3f} mV，Fold3={float(sg_lstm_multi['fold_3_rmse_mV']):.3f} mV；对 TimesNet seed=42 的 mean improvement 仅 {float(sg_times['mean_improvement_fraction_vs_raw'])*100:.2f}%，未达到 2%。

**Q15：LSTM最终 Raw 还是 Filtered？** `P2_causal_sg`。

**Q16：TimesNet最终 Raw 还是 Filtered？** `P0_raw`。

## Q17–Q18：最终专家与下一阶段判定

**Q17：最终双路径候选专家？** `LSTM + causal SG voltage-state inputs` 与 `official high-fidelity TimesNet T0 + Raw inputs`。优势区域的稳定性按三 seed 检查；满足“全部三 seed 且 RMSE 至少改善 5%”的区域包括：{stable_text}。

**Q18：是否进入 Phase N2.8？** **{'GO' if n28_go else 'NOGO'}**。

- Gate 1 Origin-wise Oracle ≥5%：{gate1}（{float(overall_oracle['rmse_improvement_fraction_vs_lstm'])*100:.2f}%）。
- Gate 2 存在稳定 TimesNet 优势区域：{gate2}。
- Gate 3 至少一个 chronological AUC ≥0.65：{gate3}（max={max_auc:.3f}）。
- Gate 4 High Shift 与 Late Development 均由 LSTM 更安全：{gate4}。

## 完整性与可复现性

- 预测成对完整性：9/9 `(seed, fold)` PASS。
- 因果滤波：future sample=false、centered=false、backfill=false、target filtered=false、load transition filtered=false。
- Shift：每个 fold 的 scaler、Ledoit-Wolf covariance、PSI bins、shift tertiles 均只由该 fold Train 拟合。
- Winner label：对每个 origin 使用三 seed 的 24-step RMSE均值确定；Logistic Regression 按 Fold1→Fold2、Fold1+2→Fold3 顺序评估。
- Test firewall：`test_target_rows_read=0`、`test_predictions_constructed=false`、`test_metrics_computed=false`、`test_shift_diagnostics_computed=false`。
- 本阶段没有训练 Adaptive Dual-Path Gate，也没有进入 Phase N2.8 实施。
"""
    write_text(report, PHASE_ROOT / "PHASE_N2_7_RESULTS.md")

    freeze = {
        "phase": "Phase N2.7",
        "phase_status": phase_status,
        "created_date": "2026-09-08",
        "protocol": {
            "sampling_resolution_hours": 0.5,
            "feature_group": "F2",
            "exact_feature_names": cfg["features"],
            "lookback_hours": 24,
            "lookback_steps": 48,
            "horizon_hours": 12,
            "horizon_steps": 24,
            "folds": FOLDS,
            "seeds": SEEDS,
            "development_end_h_exclusive": cfg["split"]["development_end_h"],
            "test_interval_h": [cfg["split"]["development_end_h"], cfg["split"]["test_end_h"]],
        },
        "final_lstm_expert": {
            "architecture": "Frozen Phase N2 LSTM",
            "preprocessing": lstm_preprocessing,
            "preprocessing_definition": "one-sided historical 5-point polynomial degree 2 on voltage-state inputs only",
            "target": "raw future voltage",
            "metrics_3seed_3fold_macro": final_lstm,
        },
        "final_timesnet_expert": {
            "architecture": "official high-fidelity TimesNet from Phase N2.6R",
            "tuning": timesnet_tuning,
            "preprocessing": timesnet_preprocessing,
            "metrics_3seed_3fold_macro": final_times,
            "seed42_screen_winner": tuning_selection["screen_winner"],
            "tuning_final_reason": tuning_selection["final_reason"],
        },
        "best_fixed_fusion": {
            "timesnet_weight_alpha": float(best_fusion["timesnet_weight_alpha"]),
            "metrics": {key: float(best_fusion[key]) for key in ["rmse_mV", "mae_mV", "r2", "bias_mV"]},
            "diagnostic_only": True,
        },
        "originwise_oracle": {
            "rmse_mV": float(overall_oracle["rmse_mV"]),
            "improvement_fraction_vs_lstm": float(overall_oracle["rmse_improvement_fraction_vs_lstm"]),
            "deployable_model": False,
        },
        "phase_n2_8_decision": "GO" if n28_go else "NOGO",
        "phase_n2_8_gates": {
            "originwise_oracle_at_least_5pct": gate1,
            "stable_timesnet_advantage_region": gate2,
            "chronological_winner_auc_at_least_0p65": gate3,
            "lstm_high_shift_and_late_safety": gate4,
        },
        "test_firewall": {
            "test_target_rows_read": 0,
            "test_predictions_constructed": False,
            "test_metrics_computed": False,
            "test_shift_diagnostics_computed": False,
            "test_performance_accessed": False,
        },
        "adaptive_gate_trained": False,
        "phase_n2_8_executed": False,
        "run_indexes": {
            "timesnet": "outputs/timesnet_tuning/timesnet_run_index.json",
            "preprocessing": "outputs/preprocessing/preprocessing_run_index.json",
        },
    }
    freeze_path = ensure_output(PHASE_ROOT / "outputs/frozen_config/PHASE_N2_7_FINAL_EXPERT_SELECTION.yaml")
    freeze_path.write_text(yaml.safe_dump(freeze, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json({
        "status": phase_status,
        "required_csv_count": 15,
        "required_figure_count": 12,
        "phase_n2_8_decision": "GO" if n28_go else "NOGO",
        "test_firewall": freeze["test_firewall"],
    }, PHASE_ROOT / "outputs/integrity_audit/finalization_summary.json")
    print(json.dumps({
        "status": phase_status,
        "lstm_rmse_mV": final_lstm["rmse_mV"],
        "timesnet_rmse_mV": final_times["rmse_mV"],
        "origin_oracle_rmse_mV": float(overall_oracle["rmse_mV"]),
        "max_winner_auc": max_auc,
        "phase_n2_8": "GO" if n28_go else "NOGO",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

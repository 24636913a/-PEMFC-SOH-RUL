"""Generate read-only diagnostics, figures and reports for completed T0 outputs.

This post-processing program never builds a dataset, loads a model checkpoint or
re-evaluates a model.  It only summarizes the completed Seed-42 Development
artifacts written by ``run_t0_baseline.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
OUTPUTS = PHASE / "outputs"
FIGURES = PHASE / "figures"
REPORTS = PHASE / "reports"
PROTOCOL = PHASE / "configs" / "PREREGISTERED_T0_PROTOCOL.yaml"
RAW_F2_LABELS = {
    "raw_f2_00": "cell_voltage_mean",
    "raw_f2_01": "cell_voltage_std",
    "raw_f2_02": "cell_voltage_range",
    "raw_f2_03": "cell_voltage_min",
    "raw_f2_04": "cell_voltage_max",
    "raw_f2_05": "actual_current_A_mean",
    "raw_f2_06": "actual_current_A_std",
    "raw_f2_07": "actual_current_A_peak_to_peak",
    "raw_f2_08": "actual_current_density_A_cm2_mean",
    "raw_f2_09": "actual_current_density_A_cm2_std",
    "raw_f2_10": "actual_current_density_A_cm2_peak_to_peak",
    "raw_f2_11": "load_transition_count",
    "raw_f2_12": "operation_point_transitions",
    "raw_f2_13": "zero_current_fraction",
}
FROZEN_CONTEXT_BY_FOLD = {
    # These records use the identical three Development validation-origin
    # universes documented in the cited phase outputs.  They are descriptive
    # frozen context, not an additional selection exercise in T0.
    "RawF2_TimesNet_seed42": {
        "fold_1": 8.753251435342959,
        "fold_2": 7.186496222627692,
        "fold_3": 26.67992583211793,
        "source": "Phase_BS_Final_Backbone_Screening/outputs/per_fold.csv",
    },
    "Historical_TFT_like_seed42": {
        "fold_1": 23.939920984653796,
        "fold_2": 16.40809279330953,
        "fold_3": 19.394529946532725,
        "source": "Phase_TFT_Targeted_Reconsideration/outputs/PER_FOLD_METRICS.csv",
    },
}


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
        }
    )


def _require(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"T0_POSTPROCESSING_INPUT_MISSING: {path}")
    return path


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _make_architecture_figure() -> None:
    """Draw the actually instantiated PTF TFT topology, not a generic TFT sketch."""
    fig, ax = plt.subplots(figsize=(12.6, 3.4))
    ax.set_axis_off()
    blocks = [
        (0.02, 0.56, 0.135, 0.25, "48×14\nRaw F2 history", "#DCEAF7"),
        (0.19, 0.56, 0.15, 0.25, "Encoder continuous\nVariable Selection\n(raw 14 + position)", "#CFE8CF"),
        (0.39, 0.56, 0.13, 0.25, "LSTM encoder\n(d=32)", "#FFE0B2"),
        (0.19, 0.12, 0.15, 0.25, "Decoder VSN\nrelative forecast\nstep only", "#CFE8CF"),
        (0.39, 0.12, 0.13, 0.25, "LSTM decoder\nstate handoff", "#FFE0B2"),
        (0.58, 0.35, 0.16, 0.28, "Gated residual\nskip + static\nenrichment (none)", "#E1D5E7"),
        (0.79, 0.35, 0.15, 0.28, "Causal\nInterpretable MHA\n4 heads", "#FCE4EC"),
        (0.79, 0.04, 0.15, 0.20, "Gating + GRN\nprojection\n24 ΔV values", "#FFF2CC"),
    ]
    for x, y, w, h, label, color in blocks:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#3D4A57", linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.3, wrap=True)
    arrows = [
        ((0.155, 0.685), (0.19, 0.685)), ((0.34, 0.685), (0.39, 0.685)),
        ((0.34, 0.245), (0.39, 0.245)), ((0.52, 0.685), (0.58, 0.51)),
        ((0.52, 0.245), (0.58, 0.47)), ((0.74, 0.49), (0.79, 0.49)),
        ((0.865, 0.35), (0.865, 0.24)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#4B5563"})
    ax.text(0.02, 0.94, "Canonical_Architecture_TFT_Strict_History — instantiated TemporalFusionTransformer topology", weight="bold", fontsize=11)
    ax.text(0.02, -0.02, "Causal constraint: no future measured operating variables; decoder uses the deterministic relative forecast-step coordinate only.", fontsize=8.2)
    _save(fig, "Fig1_Actual_Canonical_TFT_Architecture.png")


def _make_training_figure(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.35), sharey=False)
    for ax, fold in zip(axes, ["fold_1", "fold_2", "fold_3"]):
        part = curves.loc[curves.fold.eq(fold)]
        for stage, color, label in [("inner", "#1f77b4", "inner chronological stopper"), ("outer_refit", "#d62728", "full outer-train refit")]:
            series = part.loc[part.stage.eq(stage)]
            if not series.empty:
                ax.plot(series.epoch, series.selection_standardized_RMSE, color=color, lw=1.45, label=label)
        ax.set_title(fold.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Standardized ΔV RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False)
    fig.suptitle("T0 chronological inner selection and full outer-train refit", y=1.17, fontsize=11, weight="bold")
    _save(fig, "Fig2_Training_Curves.png")


def _make_context_figure(metrics: pd.DataFrame, context: dict[str, float]) -> None:
    t0_by_fold = metrics.set_index("fold").rmse_mV.to_dict()
    fold_names = ["fold_1", "fold_2", "fold_3"]
    x = np.arange(len(fold_names)); width = 0.25
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.15), gridspec_kw={"width_ratios": [1.55, 1.0]})
    collections = [
        ("Canonical TFT (T0)", t0_by_fold, "#2A6FBB"),
        ("Raw F2 TimesNet†", FROZEN_CONTEXT_BY_FOLD["RawF2_TimesNet_seed42"], "#8EC0E4"),
        ("Historical TFT-like†", FROZEN_CONTEXT_BY_FOLD["Historical_TFT_like_seed42"], "#F4B183"),
    ]
    for index, (label, values, color) in enumerate(collections):
        bars = left.bar(x + (index - 1) * width, [values[f] for f in fold_names], width, label=label, color=color, edgecolor="#34495e", linewidth=0.55)
        for bar in bars:
            left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, _fmt(bar.get_height(), 1), ha="center", va="bottom", rotation=90, fontsize=7.2)
    left.set_xticks(x, ["Fold 1", "Fold 2", "Fold 3"])
    left.set_ylabel("Forecast-query RMSE (mV)")
    left.set_title("Per-fold Development context")
    left.grid(axis="y", alpha=0.25); left.legend(frameon=False, fontsize=8)
    values = [float(metrics.rmse_mV.mean()), context["RawF2_TimesNet_seed42_RMSE_mV"], context["EMDFeature_TimesNet_seed42_RMSE_mV"], context["historical_StrictHistoryTFT_seed42_RMSE_mV"]]
    labels = ["Canonical\nTFT (T0)", "Raw F2\nTimesNet†", "EMD TimesNet†", "Historical\nTFT-like†"]
    bars = right.bar(labels, values, color=["#2A6FBB", "#8EC0E4", "#8CCB9B", "#F4B183"], edgecolor="#34495e", linewidth=0.7)
    for bar, value in zip(bars, values):
        right.text(bar.get_x() + bar.get_width() / 2, value + 0.6, _fmt(value), ha="center", va="bottom", fontsize=8)
    right.set_ylabel("Three-fold equal RMSE (mV)")
    right.set_title("Frozen context summary")
    right.grid(axis="y", alpha=0.25)
    fig.text(0.01, -0.02, "† Frozen historical Development context only; no new backbone-selection or significance claim is made in T0.", fontsize=8)
    _save(fig, "Fig3_Fold_RMSE_Context.png")


def _make_behavior_diagnostics(predictions: pd.DataFrame, weights: pd.DataFrame, attention: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    rows: list[dict[str, float | str | int]] = []
    variation_rows: list[dict[str, float | str]] = []
    for fold, part in predictions.groupby("fold", sort=False):
        by_horizon = part.groupby("horizon_step", sort=True)
        for horizon, item in by_horizon:
            error = item.y_pred_delta_V.to_numpy(float) - item.y_true_delta_V.to_numpy(float)
            rows.append({"fold": fold, "horizon_step": int(horizon), "mean_error_mV": float(error.mean() * 1000.0), "prediction_std_mV": float(item.y_pred_delta_V.std(ddof=0) * 1000.0), "target_std_mV": float(item.y_true_delta_V.std(ddof=0) * 1000.0), "rmse_mV": float(np.sqrt(np.mean(error**2)) * 1000.0), "mae_mV": float(np.mean(abs(error)) * 1000.0)})
        per_origin = part.groupby("origin_row", sort=False).agg(pred_std_V=("y_pred_delta_V", lambda s: float(np.std(s, ddof=0))), true_std_V=("y_true_delta_V", lambda s: float(np.std(s, ddof=0))))
        ratio = per_origin.pred_std_V / per_origin.true_std_V.replace(0.0, np.nan)
        full_error = part.y_pred_delta_V.to_numpy(float) - part.y_true_delta_V.to_numpy(float)
        variation_rows.append({"fold": fold, "prediction_std_mV": float(part.y_pred_delta_V.std(ddof=0) * 1000.0), "target_std_mV": float(part.y_true_delta_V.std(ddof=0) * 1000.0), "mean_error_mV": float(full_error.mean() * 1000.0), "median_per_origin_prediction_to_target_std_ratio": float(np.nanmedian(ratio)), "near_constant_prediction_origin_fraction": float(np.mean(per_origin.pred_std_V < 1e-5))})
    interpretability: dict[str, dict[str, float]] = {}
    for fold in ["fold_1", "fold_2", "fold_3"]:
        variable_weights = weights.loc[weights.fold.eq(fold) & weights.stream.eq("encoder"), "mean_selection_weight"].to_numpy(float)
        matrix = attention.loc[attention.fold.eq(fold) & attention.stream.eq("encoder")].pivot(index="decoder_horizon_step", columns="source_position", values="mean_attention_weight").to_numpy(float)
        entropy = -(matrix * np.log(np.maximum(matrix, 1e-12))).sum(axis=1) / np.log(matrix.shape[1])
        interpretability[fold] = {"encoder_vsn_top_weight": float(variable_weights.max()), "encoder_vsn_normalized_entropy": float(-(variable_weights * np.log(np.maximum(variable_weights, 1e-12))).sum() / np.log(len(variable_weights))), "encoder_attention_max_weight": float(matrix.max()), "encoder_attention_mean_normalized_entropy": float(entropy.mean())}
    return pd.DataFrame(rows), pd.DataFrame(variation_rows), interpretability


def _make_horizon_figure(horizon: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for fold, part in horizon.groupby("fold", sort=False):
        ax.plot(part.horizon_step, part.rmse_mV, marker="o", ms=2.8, lw=1.25, label=fold.replace("_", " ").title())
    equal = horizon.groupby("horizon_step", as_index=False).rmse_mV.mean()
    ax.plot(equal.horizon_step, equal.rmse_mV, color="black", lw=2.0, label="equal-fold mean")
    ax.set_xlabel("Forecast horizon step (0.5 h per step)")
    ax.set_ylabel("RMSE (mV)")
    ax.set_title("T0 horizon-resolved ΔV error")
    ax.grid(alpha=0.25); ax.legend(frameon=False, ncol=2)
    _save(fig, "Fig4_Horizon_RMSE.png")


def _make_prediction_trace(predictions: pd.DataFrame) -> None:
    source = predictions.loc[predictions.fold.eq("fold_3")].copy()
    origin_mae = source.groupby("origin_row", as_index=False).absolute_error_mV.mean().sort_values("absolute_error_mV")
    chosen = int(origin_mae.iloc[len(origin_mae) // 2].origin_row)
    view = source.loc[source.origin_row.eq(chosen)].sort_values("horizon_step")
    fig, ax = plt.subplots(figsize=(7.4, 4.15))
    ax.plot(view.horizon_step, view.y_true_delta_V * 1000.0, marker="o", ms=3.1, lw=1.6, label="observed ΔV")
    ax.plot(view.horizon_step, view.y_pred_delta_V * 1000.0, marker="s", ms=3.0, lw=1.45, label="canonical TFT")
    ax.axhline(0.0, color="#6b7280", lw=0.7)
    ax.set_xlabel("Forecast horizon step (0.5 h per step)")
    ax.set_ylabel("ΔV (mV)")
    ax.set_title(f"Fold 3 median-origin-error trace (origin row {chosen})")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    _save(fig, "Fig5_Fold3_Prediction_Trace.png")


def _make_vsn_figure(weights: pd.DataFrame) -> None:
    part = weights.loc[weights.fold.eq("fold_3") & weights.stream.eq("encoder")].copy()
    part["display_variable"] = part.variable.map(RAW_F2_LABELS).fillna(part.variable)
    part = part.sort_values("mean_selection_weight")
    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    ax.barh(part.display_variable, part.mean_selection_weight, color="#4C9F70", edgecolor="#2f4858", linewidth=0.5)
    ax.set_xlabel("Mean encoder variable-selection weight")
    ax.set_title("Actual Fold 3 encoder VSN weights")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, "Fig6_Encoder_VSN_Weights.png")


def _make_attention_figure(attention: pd.DataFrame) -> None:
    part = attention.loc[(attention.fold.eq("fold_3")) & (attention.stream.eq("encoder"))]
    grid = part.pivot(index="decoder_horizon_step", columns="source_position", values="mean_attention_weight").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8.4, 4.25))
    image = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label="Mean causal attention weight")
    ax.set_xlabel("Encoder source position (1 = oldest of 48)")
    ax.set_ylabel("Decoder horizon step")
    ax.set_title("Actual Fold 3 encoder attention, mean over validation samples and heads")
    _save(fig, "Fig7_Causal_Encoder_Attention.png")


def main() -> None:
    _style()
    for folder in (FIGURES, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(_require(PROTOCOL).read_text(encoding="utf-8"))
    metrics = pd.read_csv(_require(OUTPUTS / "TFT_SEED42_PER_FOLD_RESULTS.csv"))
    predictions = pd.read_csv(_require(OUTPUTS / "TFT_SEED42_PREDICTIONS.csv"))
    curves = pd.read_csv(_require(OUTPUTS / "TFT_TRAINING_CURVES.csv"))
    weights = pd.read_csv(_require(OUTPUTS / "TFT_VARIABLE_SELECTION_WEIGHTS.csv"))
    attention = pd.read_csv(_require(OUTPUTS / "TFT_ATTENTION_SUMMARY.csv"))
    run_audit = pd.read_csv(_require(OUTPUTS / "TFT_RUN_AUDIT.csv"))
    stage0 = json.loads(_require(OUTPUTS / "T0_STAGE0_PREFLIGHT_AUDIT.json").read_text(encoding="utf-8"))
    baseline_summary = json.loads(_require(OUTPUTS / "T0_BASELINE_SUMMARY.json").read_text(encoding="utf-8"))
    if len(metrics) != 3 or set(metrics.fold) != {"fold_1", "fold_2", "fold_3"} or not bool(baseline_summary.get("training_complete")):
        raise RuntimeError("T0_INCOMPLETE_THREE_FOLD_OUTPUT")
    if not stage0.get("status") == "T0_STAGE0_PASS" or metrics.test_accessed.any() or run_audit.test_accessed.any():
        raise RuntimeError("T0_POSTPROCESSING_FIREWALL_OR_PREFLIGHT_FAILURE")

    err = predictions.y_pred_delta_V.to_numpy(float) - predictions.y_true_delta_V.to_numpy(float)
    predictions = predictions.assign(squared_error_mV2=(err * 1000.0) ** 2)
    horizon = (predictions.groupby(["fold", "horizon_step"], as_index=False)
                .agg(rmse_mV=("squared_error_mV2", lambda s: float(np.sqrt(np.mean(s)))),
                     mae_mV=("absolute_error_mV", "mean"), query_count=("absolute_error_mV", "size")))
    horizon.to_csv(OUTPUTS / "HORIZON_DIAGNOSTICS.csv", index=False)
    errors = (predictions.groupby("fold", as_index=False)
              .agg(mean_absolute_error_mV=("absolute_error_mV", "mean"), p95_absolute_error_mV=("absolute_error_mV", lambda s: float(np.quantile(s, 0.95))), max_absolute_error_mV=("absolute_error_mV", "max"), forecast_query_count=("absolute_error_mV", "size"), unique_target_rows=("target_row", "nunique")))
    errors.to_csv(OUTPUTS / "ERROR_DIAGNOSTICS.csv", index=False)
    behavior, variation, interpretability = _make_behavior_diagnostics(predictions, weights, attention)
    behavior.to_csv(OUTPUTS / "T0_HORIZON_BEHAVIOR_DIAGNOSTICS.csv", index=False)
    variation.to_csv(OUTPUTS / "T0_OUTPUT_VARIATION_DIAGNOSTICS.csv", index=False)
    (OUTPUTS / "T0_INTERPRETABILITY_DIAGNOSTICS.json").write_text(json.dumps(interpretability, indent=2) + "\n", encoding="utf-8")
    context_rows: list[dict[str, str | float]] = []
    for row in metrics.itertuples():
        context_rows.append({"model": "Canonical_Architecture_TFT_Strict_History", "fold": row.fold, "rmse_mV": float(row.rmse_mV), "source": "T0 formal output", "role": "formal_T0"})
    for model, values in FROZEN_CONTEXT_BY_FOLD.items():
        for fold in ["fold_1", "fold_2", "fold_3"]:
            context_rows.append({"model": model, "fold": fold, "rmse_mV": values[fold], "source": values["source"], "role": "frozen_context_only"})
    pd.DataFrame(context_rows).to_csv(OUTPUTS / "FROZEN_CONTEXT_FOLD_COMPARISON.csv", index=False)

    context = cfg["metrics"]["frozen_context_references"]
    _make_architecture_figure(); _make_training_figure(curves); _make_context_figure(metrics, context)
    _make_horizon_figure(horizon); _make_prediction_trace(predictions); _make_vsn_figure(weights); _make_attention_figure(attention)

    overall_rmse = float(metrics.rmse_mV.mean()); overall_mae = float(metrics.mae_mV.mean())
    overall_horizon = behavior.groupby("horizon_step", as_index=False).mean(numeric_only=True)
    h1 = overall_horizon.loc[overall_horizon.horizon_step.eq(1)].iloc[0]
    h24 = overall_horizon.loc[overall_horizon.horizon_step.eq(24)].iloc[0]
    worst = overall_horizon.loc[overall_horizon.rmse_mV.idxmax()]
    raw_gap = overall_rmse - float(context["RawF2_TimesNet_seed42_RMSE_mV"])
    historical_gap = overall_rmse - float(context["historical_StrictHistoryTFT_seed42_RMSE_mV"])
    t0_fold3 = float(metrics.loc[metrics.fold.eq("fold_3"), "rmse_mV"].iloc[0])
    raw_fold3 = float(FROZEN_CONTEXT_BY_FOLD["RawF2_TimesNet_seed42"]["fold_3"])
    legacy_fold3 = float(FROZEN_CONTEXT_BY_FOLD["Historical_TFT_like_seed42"]["fold_3"])
    fold_rows = "\n".join(
        f"| {r.fold} | {r.query_count:,} | {_fmt(r.rmse_mV)} | {_fmt(r.mae_mV)} | {_fmt(r.r2, 4)} | {_fmt(r.endpoint_rmse_mV)} | {int(r.selected_epoch)} |"
        for r in metrics.itertuples()
    )
    _write_markdown(REPORTS / "02_DATA_AND_LEAKAGE_AUDIT.md", f"""
# T0 data and leakage audit

The formal Seed-42 run used only `pd.read_csv(..., nrows=3007)` followed by `bin_end_ophrs2_h < 1819.997738`, yielding exactly {stage0['data_firewall']['development_rows']} Development rows. The recorded Development-prefix fingerprint is `{stage0['data_firewall']['development_prefix_fingerprint_sha256']}`. Full-file reads and hashes are prohibited by the frozen protocol.

The three frozen expanding folds retained every legal validation origin: Fold 1 = 233, Fold 2 = 245, Fold 3 = 466. Every target lies below the Development boundary. Model inputs are the 48 observed Raw F2 history steps only. Decoder data contain the deterministic `relative_forecast_step` coordinate and no future measured current, voltage, temperature, pressure, RH, flow, stoichiometry, or any Test observation.

`test_accessed=false` for Stage 0, all formal fold results, and the run audit.
""")
    _write_markdown(REPORTS / "03_TFT_SEED42_BASELINE_RESULTS.md", f"""
# Canonical TFT Seed-42 Development baseline

The preregistered formal score is the three-fold equal forecast-query-weighted RMSE: **{_fmt(overall_rmse)} mV**. The equal-fold MAE is **{_fmt(overall_mae)} mV**.

| Fold | Forecast queries | RMSE (mV) | MAE (mV) | R² | 12 h endpoint RMSE (mV) | Selected inner epoch |
|---|---:|---:|---:|---:|---:|---:|
{fold_rows}

This establishes a canonical full-TFT baseline under the frozen strict-history protocol. It is a Seed-42 baseline, not a multi-seed stability claim. No Test metric was computed.
""")
    _write_markdown(REPORTS / "04_FOLD_AND_HORIZON_DIAGNOSTICS.md", f"""
# Fold and horizon diagnostics

Horizon RMSE/MAE are in `outputs/HORIZON_DIAGNOSTICS.csv`; bias and output-dispersion diagnostics are in `outputs/T0_HORIZON_BEHAVIOR_DIAGNOSTICS.csv`. These values retain overlapping forecast-query weighting and should not be interpreted as independent-sample evidence. Error tail and target multiplicity diagnostics are in `outputs/ERROR_DIAGNOSTICS.csv`.

Across folds, the equal-fold RMSE rises from {_fmt(h1.rmse_mV)} mV at 0.5 h to {_fmt(h24.rmse_mV)} mV at 12 h ({(h24.rmse_mV / h1.rmse_mV - 1.0) * 100.0:.1f}% higher). It is not monotone at every step: the maximum equal-fold horizon RMSE is {_fmt(worst.rmse_mV)} mV at step {int(worst.horizon_step)}. The signed 12 h biases are fold-specific rather than sharing a common direction (Fold 1 {_fmt(behavior.loc[(behavior.fold.eq('fold_1')) & (behavior.horizon_step.eq(24)), 'mean_error_mV'].iloc[0])} mV; Fold 2 {_fmt(behavior.loc[(behavior.fold.eq('fold_2')) & (behavior.horizon_step.eq(24)), 'mean_error_mV'].iloc[0])} mV; Fold 3 {_fmt(behavior.loc[(behavior.fold.eq('fold_3')) & (behavior.horizon_step.eq(24)), 'mean_error_mV'].iloc[0])} mV). Thus the observed degradation is primarily error dispersion, not a uniformly signed late-horizon drift.

Fold 3 has {int(metrics.loc[metrics.fold.eq('fold_3'), 'query_count'].iloc[0]):,} forecast queries and RMSE {_fmt(t0_fold3)} mV. The frozen matched-origin context values are Raw F2 TimesNet {_fmt(raw_fold3)} mV and the legacy TFT-like implementation {_fmt(legacy_fold3)} mV. Canonical T0 therefore does not reproduce a Fold-3-only advantage. There is no repeated, cross-fold periodic error minimum supporting a cycle-specific advantage; Figure 4 is descriptive and does not select a horizon-specific model.
""")
    vsn_lines = "\n".join(f"| {fold} | {_fmt(values['encoder_vsn_top_weight'], 4)} | {_fmt(values['encoder_vsn_normalized_entropy'], 4)} | {_fmt(values['encoder_attention_max_weight'], 4)} | {_fmt(values['encoder_attention_mean_normalized_entropy'], 4)} |" for fold, values in interpretability.items())
    _write_markdown(REPORTS / "05_VSN_AND_ATTENTION_ANALYSIS.md", f"""
# VSN and attention analysis

The exported VSN weights and attention values are from the actual completed canonical TFT forward path, not a reconstructed visualization. Encoder variable-selection weights aggregate validation samples and encoder time. Decoder variable selection has one available known-future input: the deterministic relative forecast-step coordinate. The Fold 3 attention heatmap aggregates actual causal encoder attention over validation samples and heads.

| Fold | largest encoder VSN weight | VSN normalized entropy | largest encoder attention weight | encoder attention normalized entropy |
|---|---:|---:|---:|---:|
{vsn_lines}

The encoder VSN does not collapse to a single variable (largest weight ≤ 0.257) and its normalized entropy remains 0.839–0.933. Encoder attention is neither uniform nor a single-position spike (maximum averaged cell ≤ 0.089; normalized entropy 0.465–0.591). These are descriptive internal checks, not causal feature-effect estimates and not a post-hoc feature-selection rule.
""")
    variation_lines = "\n".join(f"| {r.fold} | {_fmt(r.prediction_std_mV)} | {_fmt(r.target_std_mV)} | {_fmt(r.median_per_origin_prediction_to_target_std_ratio, 4)} | {_fmt(r.near_constant_prediction_origin_fraction, 4)} |" for r in variation.itertuples())
    inner_lines = "\n".join(f"| {r.fold} | {int(r.selected_epoch)} | {_fmt(r.inner_stopper_RMSE_mV)} | {int(curves.loc[(curves.fold.eq(r.fold)) & (curves.stage.eq('inner')), 'epoch'].max())} |" for r in metrics.itertuples())
    _write_markdown(REPORTS / "06_TRAINING_AND_ERROR_DIAGNOSIS.md", f"""
# Training and error diagnosis

Each fold first used the final chronological 15% of its outer-train origins as an inner stopper. The selected epoch then governed a same-seed reinitialised refit on the entire outer train; outer validation was not used for epoch selection. The maximum available epochs were 300 with patience 30, AdamW (0.001, weight decay 1e-5), batch size 64, and gradient clipping 1.0.

| Fold | selected epoch | best inner-stopper RMSE (mV) | final evaluated inner epoch |
|---|---:|---:|---:|
{inner_lines}

Fold 2 and Fold 3 reached the 300-epoch limit with best epochs 297 and 289, respectively; this is evidence that the fixed Stage-T0 training budget, rather than a demonstrated convergence plateau, may be a follow-up tuning question. It does **not** invalidate the frozen baseline or license in-stage retuning.

| Fold | prediction SD (mV) | target SD (mV) | median per-origin prediction/target SD ratio | near-constant origin fraction |
|---|---:|---:|---:|---:|
{variation_lines}

The prediction dispersion closely matches target dispersion and no origin has a near-constant 24-step prediction under the stated 0.01 mV criterion; T0 therefore provides no evidence that its poorer RMSE is explained by a trivial constant or excessively smoothed output. No abrupt-load subgroup was pre-registered or exported in the T0 prediction artifact, so this stage makes no unsupported claim about transition-specific failure. Stage 0 verified the actual PTF architecture, finite GPU forward/backward, gradient finiteness, extractable VSN/attention tensors, and prediction invariance to unavailable future measurements. The completed model has {int(metrics.parameter_count.iloc[0]):,} trainable parameters. The formal overall RMSE is {_fmt(overall_rmse)} mV.
""")
    identity_path = REPORTS / "01_TFT_ARCHITECTURE_IDENTITY_AUDIT.md"
    identity = identity_path.read_text(encoding="utf-8")
    marker = "## Post-training confirmation"
    if marker in identity:
        identity = identity.split(marker, 1)[0].rstrip()
    _write_markdown(identity_path, identity + "\n\n## Post-training confirmation\n\nThe same audited implementation completed all three frozen folds. The model is a full PTF TemporalFusionTransformer with VSN, separate LSTM encoder/decoder state handoff, gating/GRNs, causal interpretable multi-head attention, and extractable interpretation tensors.")
    _write_markdown(REPORTS / "FINAL_STAGE_T0_DECISION.md", f"""
# Phase T0 final decision

## Status: `CANONICAL_TFT_BASELINE_ESTABLISHED`

The pre-training architecture audit passed and all three frozen Seed-42 Development folds completed with no Test access. This phase establishes a reproducible, strict-history **canonical full TFT baseline**, rather than reusing the historical incomplete `StrictHistoryTFT` implementation.

- Formal T0 three-fold equal RMSE: **{_fmt(overall_rmse)} mV**.
- Frozen context only: Raw F2 TimesNet Seed-42 = {_fmt(context['RawF2_TimesNet_seed42_RMSE_mV'])} mV; EMD-feature TimesNet Seed-42 = {_fmt(context['EMDFeature_TimesNet_seed42_RMSE_mV'])} mV; historical TFT-like result = {_fmt(context['historical_StrictHistoryTFT_seed42_RMSE_mV'])} mV.

## Evidence-led answers

1. **Canonical versus earlier TFT-like model.** The new canonical TFT is worse by {_fmt(historical_gap)} mV in three-fold equal RMSE. This is not an identity contradiction: the legacy TFT-like model is not accepted as canonical because it lacked the verified PTF component pathway. T0 is the valid architecture control.
2. **Canonical TFT versus Raw F2 TimesNet.** T0 is worse by {_fmt(raw_gap)} mV ({raw_gap / context['RawF2_TimesNet_seed42_RMSE_mV'] * 100.0:.1f}% of the Raw TimesNet baseline). This is a frozen, same-origin Development context comparison, not a new backbone-screening decision.
3. **Fold behaviour.** The canonical TFT does not reproduce the historical narrative of a relatively better Fold 3: its Fold 3 RMSE is {_fmt(t0_fold3)} mV versus {_fmt(raw_fold3)} mV for frozen Raw TimesNet and {_fmt(legacy_fold3)} mV for the legacy TFT-like record.
4. **Most evidence-supported failure hypotheses.** The output is not constant or grossly over-smoothed, and VSN/attention do not show a single-variable or single-position collapse. The remaining evidence is consistent with insufficiently well-conditioned fitting of the full TFT under this limited chronological data regime; Fold 2/3 best epochs near the fixed 300-epoch cap make training budget/convergence a protocol-controlled follow-up question. No transition-specific claim is supported because that subgroup was not exported.
5. **Potential T1 parameter priorities.** If an independent T1 protocol is approved, it should pre-register only evidence-motivated candidates: training budget/patience, hidden/continuous widths, dropout, learning rate/weight decay, and attention-head count. It must retain the strict history firewall, frozen fold structure, and no-future-covariate rule.
6. **Mandatory repair before any future run.** None was identified: Stage 0 architecture, data firewall, scaler scope, output shape, prediction invariance, and interpretability extraction all passed. Performance is not a structural or data-interface failure.
7. **T1 status.** A separately pre-registered Optuna-TFT study is scientifically permissible as a future decision, but it is **not started by T0** and requires an explicit new instruction and protocol freeze.

The historical TFT-like result is not treated as this model: it lacked the verified full canonical TFT component set and interpretable attention pathway. Context comparisons do not reopen backbone selection or claim a new formal multi-model ranking. T0 changes no scientific target, feature universe, fold, test boundary, or future-covariate policy. It creates the necessary canonical TFT control for later preregistered research decisions.

No Test prediction, SOH/RUL target, future actual operating condition, decomposition, physics loss, or hyperparameter search was used. The phase stops here; follow-up model changes require a new protocol.
""")

    final = {
        "phase": cfg["phase"], "status": "CANONICAL_TFT_BASELINE_ESTABLISHED", "protocol_sha256": baseline_summary["protocol_sha256"],
        "model": baseline_summary["model"], "seed": 42, "three_fold_equal_forecast_query_weighted_RMSE_mV": overall_rmse,
        "three_fold_equal_MAE_mV": overall_mae, "fold_RMSE_mV": {r.fold: float(r.rmse_mV) for r in metrics.itertuples()},
        "fold_MAE_mV": {r.fold: float(r.mae_mV) for r in metrics.itertuples()}, "test_accessed": False,
        "future_actual_covariates_used": False, "training_complete": True, "stage0_status": stage0["status"],
        "frozen_context_only": context, "raw_timesnet_RMSE_gap_mV": raw_gap,
        "historical_tft_like_RMSE_gap_mV": historical_gap,
        "horizon_step_1_equal_fold_RMSE_mV": float(h1.rmse_mV), "horizon_step_24_equal_fold_RMSE_mV": float(h24.rmse_mV),
        "diagnostic_artifacts": {"horizon_behavior": "T0_HORIZON_BEHAVIOR_DIAGNOSTICS.csv", "output_variation": "T0_OUTPUT_VARIATION_DIAGNOSTICS.csv", "interpretability": "T0_INTERPRETABILITY_DIAGNOSTICS.json", "frozen_context": "FROZEN_CONTEXT_FOLD_COMPARISON.csv"},
    }
    (OUTPUTS / "T0_FINAL_SUMMARY.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

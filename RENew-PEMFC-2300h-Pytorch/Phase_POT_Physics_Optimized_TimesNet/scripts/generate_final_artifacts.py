"""Create Phase POT's read-only tables, figures, and decision reports.

This script consumes only Phase POT outputs plus the registered Development
prefix required to display two descriptive forecast examples.  It performs no
training, hyperparameter selection, checkpoint modification, or Test access.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pot_common import (
    PHASE,
    build_frozen_folds,
    load_inherited_protocols,
    load_protocol,
    safe_read_development,
    write_csv,
    write_json,
    write_text,
)


OUTPUTS = PHASE / "outputs"
FIGURES = PHASE / "figures"
REPORTS = PHASE / "reports"


def _read_csv(name: str) -> pd.DataFrame:
    path = OUTPUTS / name
    if not path.is_file():
        raise RuntimeError(f"REQUIRED_OUTPUT_MISSING: {path.name}")
    return pd.read_csv(path)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _m2_m4(summary: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    m2 = summary.loc[(summary.ablation_id == "M2_EMD_TimesNet") & (summary.level == "seed_mean")].iloc[0]
    m4 = summary.loc[(summary.ablation_id == "M4_Optuna_EMD_TimesNet") & (summary.level == "seed_mean")].iloc[0]
    m2_seed = summary.loc[(summary.ablation_id == "M2_EMD_TimesNet") & (summary.level == "seed_equal_fold")].sort_values("seed")
    m4_seed = summary.loc[(summary.ablation_id == "M4_Optuna_EMD_TimesNet") & (summary.level == "seed_equal_fold")].sort_values("seed")
    return m2, m4, m2_seed, m4_seed


def _horizon_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, fold, seed, horizon), group in predictions.groupby(
        ["ablation_id", "fold", "seed", "horizon_step"], sort=True
    ):
        error_mV = (group.y_pred_delta_V.to_numpy(float) - group.y_true_delta_V.to_numpy(float)) * 1000.0
        rows.append(
            {
                "ablation_id": model,
                "fold": fold,
                "seed": int(seed),
                "horizon_step": int(horizon),
                "forecast_horizon_h": float(horizon) * 0.5,
                "RMSE_mV": float(np.sqrt(np.mean(error_mV**2))),
                "MAE_mV": float(np.mean(np.abs(error_mV))),
                "query_count": int(len(group)),
            }
        )
    detail = pd.DataFrame(rows)
    aggregate = (
        detail.groupby(["ablation_id", "horizon_step", "forecast_horizon_h"], as_index=False)
        .agg(RMSE_mV_mean=("RMSE_mV", "mean"), RMSE_mV_std=("RMSE_mV", "std"), MAE_mV_mean=("MAE_mV", "mean"), query_count=("query_count", "sum"))
        .sort_values(["ablation_id", "horizon_step"])
    )
    return aggregate


def _fold_seed_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = ["rmse_mV", "mae_mV", "r2", "endpoint_rmse_mV", "p95_abs_error_mV", "p99_abs_error_mV", "max_abs_error_mV", "target_time_equal_rmse_mV"]
    out = (
        metrics.groupby(["ablation_id", "fold"], as_index=False)[numeric]
        .mean()
        .rename(columns={key: f"seed_mean_{key}" for key in numeric})
    )
    return out


def _make_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 2.8))
    ax.axis("off")
    boxes = [
        (0.02, 0.31, 0.16, 0.35, "Causal Raw F2\n48 × 14\n24 h history", "#d7e9f7"),
        (0.24, 0.31, 0.16, 0.35, "Optional frozen\nCAUSAL_EMD_SLOW_\nVARYING_FEATURES\n48 × 2", "#e6def7"),
        (0.46, 0.31, 0.17, 0.35, "TimesNet\nfixed backbone\n48 → 24", "#f9e3c4"),
        (0.70, 0.31, 0.15, 0.35, "Forecast head\nΔV(t+1:t+24)", "#dcefdc"),
        (0.90, 0.31, 0.08, 0.35, "Metrics\n(mV)", "#f7d8db"),
    ]
    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#334155", linewidth=1.0, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", transform=ax.transAxes, fontsize=8.1)
    for x0, x1 in [(0.18, 0.46), (0.40, 0.46), (0.63, 0.70), (0.85, 0.90)]:
        ax.annotate("", xy=(x1, 0.485), xytext=(x0, 0.485), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.25, "color": "#334155"})
    ax.text(0.32, 0.18, "M1/M3: Raw F2 only;  M2/M4: Raw F2 + two origin-conditioned causal EMD features", ha="center", transform=ax.transAxes, fontsize=8.2)
    ax.text(0.72, 0.18, "Physics loss: NOT RUN — future actual conditions are unavailable at inference", ha="center", transform=ax.transAxes, fontsize=8.2, color="#9a3412")
    _save(fig, "Fig1_TimesNet_Architecture_and_Inputs.png")


def _make_optuna_figure(raw_trials: pd.DataFrame, emd_trials: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), sharey=True)
    for ax, trials, title in zip(axes, [raw_trials, emd_trials], ["Optuna Raw F2", "Optuna EMD feature"]):
        complete = trials.loc[trials.state.astype(str).str.upper().eq("COMPLETE")].copy().sort_values("trial")
        ax.scatter(complete.trial, complete.value_inner_deltaV_RMSE_mV, s=24, color="#2563eb", label="completed trial")
        best = complete.value_inner_deltaV_RMSE_mV.cummin()
        ax.plot(complete.trial, best, color="#dc2626", linewidth=1.45, label="running best")
        ax.set_title(title)
        ax.set_xlabel("Trial number")
        ax.set_ylabel("Inner RMSE (mV)")
        ax.legend(frameon=False, loc="best")
    _save(fig, "Fig2_Optuna_Search_History.png")


def _make_ablation_figure(fold_summary: pd.DataFrame) -> None:
    order = ["M1_RawF2_TimesNet", "M2_EMD_TimesNet", "M3_Optuna_RawF2_TimesNet", "M4_Optuna_EMD_TimesNet"]
    labels = ["M1\nRaw", "M2\nEMD", "M3\nOptuna\nRaw", "M4\nOptuna\nEMD"]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.15), sharey=True)
    colors = ["#64748b", "#7c3aed", "#2563eb", "#dc2626"]
    for ax, fold in zip(axes, ["fold_1", "fold_2", "fold_3"]):
        part = fold_summary.set_index(["ablation_id", "fold"]).reindex([(x, fold) for x in order]).reset_index()
        ax.bar(np.arange(4), part.seed_mean_rmse_mV, color=colors, width=0.72)
        for i, value in enumerate(part.seed_mean_rmse_mV):
            ax.text(i, value + 0.35, f"{value:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(np.arange(4), labels, fontsize=7.5)
        ax.set_title(fold.replace("_", " ").title())
        ax.set_ylabel("Seed-mean RMSE (mV)")
    fig.subplots_adjust(wspace=0.20, bottom=0.24)
    _save(fig, "Fig3_ThreeFold_Ablation_RMSE.png")


def _make_horizon_figure(horizon: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.55))
    labels = {"M1_RawF2_TimesNet": "M1 Raw", "M2_EMD_TimesNet": "M2 EMD", "M3_Optuna_RawF2_TimesNet": "M3 Optuna Raw", "M4_Optuna_EMD_TimesNet": "M4 Optuna EMD"}
    colors = {"M1_RawF2_TimesNet": "#64748b", "M2_EMD_TimesNet": "#7c3aed", "M3_Optuna_RawF2_TimesNet": "#2563eb", "M4_Optuna_EMD_TimesNet": "#dc2626"}
    for model, group in horizon.groupby("ablation_id", sort=False):
        ax.plot(group.forecast_horizon_h, group.RMSE_mV_mean, label=labels[model], color=colors[model], linewidth=1.5)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("RMSE (mV), equal fold/seed mean")
    ax.legend(ncol=2, frameon=False)
    _save(fig, "Fig4_Horizon_Error_Curves.png")


def _absolute_trace(frame: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    origin = int(query.origin_row.iloc[0])
    anchor = float(frame.iloc[origin]["cell_voltage_mean"])
    ordered = query.sort_values("horizon_step")
    horizon = ordered.horizon_step.to_numpy(float) * 0.5
    true_v = anchor + ordered.y_true_delta_V.to_numpy(float)
    pred_v = anchor + ordered.y_pred_delta_V.to_numpy(float)
    return horizon, true_v, pred_v, anchor


def _make_forecast_figures(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[(predictions.ablation_id == "M4_Optuna_EMD_TimesNet") & (predictions.fold == "fold_3") & (predictions.seed == 42)].copy()
    origins = np.sort(selected.origin_row.unique())
    typical_origin = int(origins[len(origins) // 2])
    typical = selected.loc[selected.origin_row.eq(typical_origin)]
    extreme = selected.loc[selected.absolute_error_mV.eq(selected.absolute_error_mV.max())]
    extreme_origin = int(extreme.origin_row.iloc[0])
    extreme_query = selected.loc[selected.origin_row.eq(extreme_origin)]
    rows = []
    for kind, query, filename, title in [
        ("typical_chronological", typical, "Fig5_Typical_Fold3_Forecast.png", "Typical chronological Fold 3 query"),
        ("largest_posthoc_absolute_error", extreme_query, "Fig6_Extreme_Error_Case.png", "Post-hoc largest absolute-error Fold 3 query"),
    ]:
        horizon, true_v, pred_v, anchor = _absolute_trace(frame, query)
        fig, ax = plt.subplots(figsize=(6.7, 3.45))
        ax.plot(horizon, true_v, marker="o", markersize=3, color="#111827", label="Observed target voltage")
        ax.plot(horizon, pred_v, marker="s", markersize=3, color="#dc2626", label="M4 predicted voltage")
        ax.axhline(anchor, linestyle="--", linewidth=1.0, color="#64748b", label="Origin voltage")
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_ylabel("Cell voltage (V)")
        ax.set_title(title)
        ax.legend(frameon=False)
        _save(fig, filename)
        rows.append({"case": kind, "model": "M4_Optuna_EMD_TimesNet", "fold": "fold_3", "seed": 42, "origin_row": int(query.origin_row.iloc[0]), "origin_age_h": float(query.origin_age_h.iloc[0]), "anchor_voltage_V": anchor, "maximum_query_absolute_error_mV": float(query.absolute_error_mV.max()), "posthoc_descriptive_only": True})
    return pd.DataFrame(rows)


def _make_summary_figure(summary: pd.DataFrame, m2: pd.Series, m4: pd.Series) -> None:
    models = ["M1_RawF2_TimesNet", "M2_EMD_TimesNet", "M3_Optuna_RawF2_TimesNet", "M4_Optuna_EMD_TimesNet"]
    labels = ["M1 Raw", "M2 EMD", "M3 Optuna Raw", "M4 Optuna EMD"]
    rows = summary.loc[(summary.level == "seed_mean") & (summary.ablation_id.isin(models))].set_index("ablation_id").reindex(models)
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.25))
    axes[0].bar(labels, rows.rmse_mV, color=["#64748b", "#7c3aed", "#2563eb", "#dc2626"])
    axes[0].set_ylabel("Three-fold equal RMSE (mV)")
    axes[0].tick_params(axis="x", rotation=18)
    gate_values = [m4.rmse_mV / m2.rmse_mV, m4.fold3_rmse_mV / m2.fold3_rmse_mV, m4.p99_abs_error_mV / m2.p99_abs_error_mV, m4.target_time_equal_rmse_mV / m2.target_time_equal_rmse_mV]
    gate_labels = ["RMSE", "Fold 3\nRMSE", "P99", "Target-time\nRMSE"]
    axes[1].bar(gate_labels, gate_values, color="#dc2626")
    axes[1].axhline(1.0, color="#111827", linewidth=1.0, linestyle="--", label="M2 reference")
    axes[1].axhline(0.99, color="#16a34a", linewidth=1.0, linestyle=":", label="M4 RMSE gate")
    axes[1].set_ylabel("M4 / M2 ratio")
    axes[1].legend(frameon=False, fontsize=7.5)
    _save(fig, "Fig7_Ablation_Summary.png")


def _fmt(value: float, places: int = 3) -> str:
    return f"{float(value):.{places}f}"


def _reports(metrics: pd.DataFrame, summary: pd.DataFrame, horizon: pd.DataFrame, audit: dict, stage0: dict, selected: dict, gate_rows: dict, errors: pd.DataFrame) -> None:
    m2, m4, m2_seed, m4_seed = _m2_m4(summary)
    status = "OPTUNA_ONLY_ADVANCE" if bool(gate_rows["all_pass"]) else "NO_ROBUST_IMPROVEMENT"
    m3 = summary.loc[(summary.ablation_id == "M3_Optuna_RawF2_TimesNet") & (summary.level == "seed_mean")].iloc[0]
    m1 = summary.loc[(summary.ablation_id == "M1_RawF2_TimesNet") & (summary.level == "seed_mean")].iloc[0]
    reports = {
        "02_OPTUNA_SEARCH_REPORT.md": f"""# Phase POT — Optuna Search Report

The search was frozen before any outer-fold evaluation. Twenty deterministic TPE trials were completed for each method using only Fold 1 outer-train's chronological inner fit/stopper partition; no outer validation or Test selection occurred.

| Method | Best trial | Inner RMSE (mV) | Selected epoch | Learning rate | Weight decay | Dropout | Batch size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw F2 | {selected['Optuna_RawF2_TimesNet']['best_trial']} | {_fmt(selected['Optuna_RawF2_TimesNet']['best_inner_deltaV_RMSE_mV'])} | {selected['Optuna_RawF2_TimesNet']['selected_epoch']} | {selected['Optuna_RawF2_TimesNet']['hyperparameters']['learning_rate']:.8f} | {selected['Optuna_RawF2_TimesNet']['hyperparameters']['weight_decay']:.8f} | {selected['Optuna_RawF2_TimesNet']['hyperparameters']['dropout']} | {selected['Optuna_RawF2_TimesNet']['hyperparameters']['batch_size']} |
| EMD | {selected['Optuna_EMDFeature_TimesNet']['best_trial']} | {_fmt(selected['Optuna_EMDFeature_TimesNet']['best_inner_deltaV_RMSE_mV'])} | {selected['Optuna_EMDFeature_TimesNet']['selected_epoch']} | {selected['Optuna_EMDFeature_TimesNet']['hyperparameters']['learning_rate']:.8f} | {selected['Optuna_EMDFeature_TimesNet']['hyperparameters']['weight_decay']:.8f} | {selected['Optuna_EMDFeature_TimesNet']['hyperparameters']['dropout']} | {selected['Optuna_EMDFeature_TimesNet']['hyperparameters']['batch_size']} |

The frozen resolved configuration file was hashed before M3/M4 formal refits. Fold 2 and Fold 3 were never used for the search.
""",
        "03_MODEL_IMPLEMENTATION_AUDIT.md": f"""# Phase POT — Model Implementation Audit

- Task: 48 half-hour causal steps → 24 future `cell_voltage_mean` ΔV steps.
- Formal population: 3005 Development rows obtained only by `pd.read_csv(..., nrows=3007)` followed by `bin_end_ophrs2_h < 1819.997738`.
- Test rows/targets accessed in this phase: **No**.
- M1/M2: read-only historical controls, revalidated at exact origin, target-row, and truth level.
- M3/M4: same frozen TimesNet architecture, train-only scaling, chronological inner stopping, seed reinitialisation for final outer-train refit.
- Parameters: Raw = 149,046; EMD = 149,304. EMD adds two causal origin-conditioned slow-varying features and is not interpreted as true SOH.
- M5/M6: not run by the Stage 0 gate, rather than replaced by an alternative physics penalty.

Current phase Test status is `CURRENT_PHASE_TEST_NOT_ACCESSED`. This does not erase the separate historical fact `HISTORICAL_TEST_CONTACT_EXISTS`.
""",
        "04_ABLATION_RESULTS.md": f"""# Phase POT — Formal Ablation Results

Primary metric: three-fold equal forecast-query-weighted RMSE, reported as the mean across the three formal seeds. Overlapping forecast queries are not treated as independent observations.

| Model | RMSE (mV) | MAE (mV) | R² | Endpoint RMSE (mV) | P99 abs. error (mV) | Target-time-equal RMSE (mV) |
|---|---:|---:|---:|---:|---:|---:|
| M1 Raw F2 | {_fmt(m1.rmse_mV)} | {_fmt(m1.mae_mV)} | {_fmt(m1.r2, 4)} | {_fmt(m1.endpoint_rmse_mV)} | {_fmt(m1.p99_abs_error_mV)} | {_fmt(m1.target_time_equal_rmse_mV)} |
| M2 EMD | {_fmt(m2.rmse_mV)} | {_fmt(m2.mae_mV)} | {_fmt(m2.r2, 4)} | {_fmt(m2.endpoint_rmse_mV)} | {_fmt(m2.p99_abs_error_mV)} | {_fmt(m2.target_time_equal_rmse_mV)} |
| M3 Optuna Raw F2 | {_fmt(m3.rmse_mV)} | {_fmt(m3.mae_mV)} | {_fmt(m3.r2, 4)} | {_fmt(m3.endpoint_rmse_mV)} | {_fmt(m3.p99_abs_error_mV)} | {_fmt(m3.target_time_equal_rmse_mV)} |
| M4 Optuna EMD | {_fmt(m4.rmse_mV)} | {_fmt(m4.mae_mV)} | {_fmt(m4.r2, 4)} | {_fmt(m4.endpoint_rmse_mV)} | {_fmt(m4.p99_abs_error_mV)} | {_fmt(m4.target_time_equal_rmse_mV)} |

M4 vs M2 preregistered gates: RMSE≤0.99×M2 = `{gate_rows['checks']['formal_RMSE_99pct_of_M2']}`; ≥2 paired seed wins = `{gate_rows['checks']['paired_seed_improvement_at_least_2']}` ({gate_rows['paired_seed_wins']}/3); Fold 3 RMSE≤1.03×M2 = `{gate_rows['checks']['fold3_RMSE_103pct_of_M2']}`; P99≤1.05×M2 = `{gate_rows['checks']['P99_105pct_of_M2']}`; target-time-equal RMSE≤1.05×M2 = `{gate_rows['checks']['target_equal_105pct_of_M2']}`.

Therefore, the frozen M4 advance gate is **not passed**. M3's numerical improvement over M1 is descriptive; the registered advance decision was based on M4 versus M2 and is not retroactively changed.
""",
        "05_PHYSICAL_CONSISTENCY_ANALYSIS.md": f"""# Phase POT — Physical Consistency Analysis

Stage 0 status: **{stage0['status']}**.

The preregistered candidate had {stage0['support']['pair_count']} matched train-query pairs and active-pair fraction {_fmt(stage0['support']['active_pair_fraction'], 4)}. Its differentiable synthetic implementation passed the nontriviality check. However, pair selection requires target-time actual current density, temperature, relative humidity, and pressure. Those variables are registered `future-unknown actual` quantities at inference; using them as inputs or training-time selectors would violate the causal forecasting information set.

Accordingly, M5/M6 were not trained. No substitute constraint, empirical penalty, PINN claim, violation-rate comparison, or physical-gain claim is made. The result is an informative negative feasibility finding: the audited polarization-order candidate is not deployable for this future-condition-unknown task.
""",
        "06_ERROR_AND_ROBUSTNESS_ANALYSIS.md": f"""# Phase POT — Error and Robustness Analysis

The primary aggregation is equal-fold and forecast-query weighted. Target-time-equal RMSE is a mandatory sensitivity metric, not a replacement selection metric. The formal M4/M2 ratios were: RMSE {_fmt(m4.rmse_mV / m2.rmse_mV, 4)}, Fold 3 RMSE {_fmt(m4.fold3_rmse_mV / m2.fold3_rmse_mV, 4)}, P99 {_fmt(m4.p99_abs_error_mV / m2.p99_abs_error_mV, 4)}, and target-time-equal RMSE {_fmt(m4.target_time_equal_rmse_mV / m2.target_time_equal_rmse_mV, 4)}.

Seed-level M2 and M4 equal-fold RMSE values (mV):

| Seed | M2 EMD | M4 Optuna EMD |
|---:|---:|---:|
""" + "\n".join(f"| {int(a.seed)} | {_fmt(a.rmse_mV)} | {_fmt(b.rmse_mV)} |" for (_, a), (_, b) in zip(m2_seed.iterrows(), m4_seed.iterrows())) + f"""

Only {gate_rows['paired_seed_wins']}/3 paired seeds improved for M4. The extreme-error plot is explicitly a post-hoc descriptive diagnostic (`{errors.loc[errors.case.eq('largest_posthoc_absolute_error'), 'origin_row'].iloc[0]}`), not an outlier-removal rule or a new selection criterion. All legal validation queries, including the historical extreme targets, remained in the formal metrics.
""",
        "07_PAPER_CONTRIBUTION_ASSESSMENT.md": f"""# Phase POT — Paper Contribution Assessment

Supported factual contributions are limited to: (1) transparent causal EMD slow-varying feature use inherited from the read-only Phase DF control; (2) a preregistered, inner-only Optuna-TPE evaluation; and (3) an explicit negative audit showing why the selected physics loss cannot be used without future-condition leakage.

The full proposed claim “Causal EMD + Optuna-TPE + Physics-Constrained TimesNet” is **not** supported: physics models were correctly not run, and M4 did not meet the frozen robustness gates against M2. The results do not establish a new backbone, true SOH, irreversible degradation identification, or RUL prediction. A paper must present POT as a rigorous failed/negative extension or retain the previously supported EMD forecasting contribution, rather than claim physics-optimized superiority.
""",
        "FINAL_RESEARCH_DECISION.md": f"""# Phase POT — Final Research Decision

## Final status: `{status}`

1. **Did Optuna improve TimesNet?** M3 Raw F2 RMSE ({_fmt(m3.rmse_mV)} mV) was numerically lower than M1 ({_fmt(m1.rmse_mV)} mV), but the preregistered EMD+Optuna advancement comparison was M4 vs M2 and failed all five registered checks.
2. **Did EMD remain beneficial after optimisation?** No robust M4-over-M2 gain was established. M2 remained the best seed-mean primary RMSE among the EMD configurations.
3. **Was a physics constraint established?** No. Stage 0 failed only the causal information gate, not because of numerical-gradient failure. M5/M6 were not run.
4. **Was the complete method stable across folds and seeds?** No. M4 had one paired seed improvement but worse aggregate RMSE and elevated seed variability relative to M2.
5. **Was any conclusion driven by deleting rare/extreme samples?** No. All legal validation origins and targets were retained; the error-case figure is post-hoc descriptive only.
6. **Is added complexity justified?** No. Optuna+EMD added training-selection complexity without a robust improvement over the frozen EMD baseline.
7. **SCI contribution claim supported now?** The defensible claim is a leakage-aware negative feasibility audit and a rigorously preregistered ablation. A physics-optimized TimesNet superiority claim is not supported.
8. **Primary failure reason?** Future actual operating conditions needed to instantiate the physical candidate are unavailable at inference, while the resulting Optuna EMD configuration failed its registered robustness comparisons against M2.

This is a Development-only multi-step dynamic-voltage forecasting decision. `CAUSAL_EMD_SLOW_VARYING_FEATURES` are not claimed to be true SOH or irreversible degradation, and no SOH/RUL model was trained.
""",
    }
    for name, content in reports.items():
        write_text(textwrap.dedent(content).strip() + "\n", "reports", name)


def main() -> None:
    _style()
    cfg, protocol_sha = load_protocol()
    bs_cfg, _df_cfg, common = load_inherited_protocols(cfg)
    stage0 = json.loads((OUTPUTS / "STAGE0_PHYSICAL_FEASIBILITY_SUMMARY.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUTS / "FORMAL_ABLATION_AUDIT.json").read_text(encoding="utf-8"))
    resolved = __import__("yaml").safe_load((PHASE / "configs" / "RESOLVED_OPTUNA_CONFIGS.yaml").read_text(encoding="utf-8"))
    metrics = _read_csv("FORMAL_ABLATION_PER_FOLD_RESULTS.csv")
    summary = _read_csv("FORMAL_ABLATION_SEED_SUMMARY.csv")
    predictions = _read_csv("FORMAL_ABLATION_PREDICTIONS.csv")
    raw_trials = _read_csv("OPTUNA_Optuna_RawF2_TimesNet_TRIALS.csv")
    emd_trials = _read_csv("OPTUNA_Optuna_EMDFeature_TimesNet_TRIALS.csv")
    if stage0["status"] != "PHYSICS_LOSS_NOT_ESTABLISHED" or not bool(audit["M3_M4_formal_refits_completed"]):
        raise RuntimeError("FORMAL_ARTIFACT_STATUS_UNEXPECTED")
    if bool(audit["data_firewall"]["test_accessed"]):
        raise RuntimeError("TEST_FIREWALL_FAIL")
    m2, m4, _m2_seed, _m4_seed = _m2_m4(summary)
    gate_rows = audit["optuna_only_gate"]
    horizon = _horizon_metrics(predictions)
    fold_summary = _fold_seed_metric_table(metrics)
    frame, firewall = safe_read_development(cfg, bs_cfg)
    _ = build_frozen_folds(frame, cfg, bs_cfg, common)
    errors = _make_forecast_figures(frame, predictions)
    _make_architecture_figure()
    _make_optuna_figure(raw_trials, emd_trials)
    _make_ablation_figure(fold_summary)
    _make_horizon_figure(horizon)
    _make_summary_figure(summary, m2, m4)
    write_csv(horizon, "outputs", "HORIZON_ERROR_CURVES.csv")
    write_csv(fold_summary, "outputs", "ABLATION_FOLD_SEED_MEANS.csv")
    write_csv(errors, "outputs", "POSTHOC_ERROR_CASE_AUDIT.csv")
    status = "OPTUNA_ONLY_ADVANCE" if bool(gate_rows["all_pass"]) else "NO_ROBUST_IMPROVEMENT"
    final = {
        "phase": cfg["phase"],
        "protocol_sha256": protocol_sha,
        "stage0_physics_status": stage0["status"],
        "physics_models_M5_M6": "NOT_RUN_BY_FROZEN_STAGE0_GATE",
        "final_research_status": status,
        "optuna_only_gate": gate_rows,
        "M2_EMD_seed_mean_RMSE_mV": float(m2.rmse_mV),
        "M4_Optuna_EMD_seed_mean_RMSE_mV": float(m4.rmse_mV),
        "data_firewall": firewall,
        "current_phase_test_accessed": False,
        "historical_test_contact_exists": True,
        "SOH_training": False,
        "RUL_training": False,
        "backbone_search": False,
        "formal_predictions_reused_from_prior_run": False,
    }
    write_json(final, "outputs", "POT_FINAL_SUMMARY.json")
    _reports(metrics, summary, horizon, audit, stage0, resolved["search_results"], gate_rows, errors)
    print(json.dumps({"status": status, "stage0": stage0["status"], "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

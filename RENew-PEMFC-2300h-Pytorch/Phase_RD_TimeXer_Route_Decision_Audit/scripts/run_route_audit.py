"""Cache-first Development-only TimeXer route attribution audit.

No model is instantiated or trained.  Historical files are read-only inputs;
all derived reports and tables are written only below Phase_RD.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch")
PHASE = ROOT / "Phase_RD_TimeXer_Route_Decision_Audit"
OUT, REP, FIG = PHASE / "outputs", PHASE / "reports", PHASE / "figures"
OPR = ROOT / "Phase_OPR_TimeXer"
TTD = ROOT / "Phase_TTD_TimeXer_Failure_Diagnosis"
N28F = ROOT / "Phase_N2_8F_Uncertainty_Robust_SA_TimeXer"
CFG = yaml.safe_load((PHASE / "configs" / "frozen_audit_protocol.yaml").read_text(encoding="utf-8"))
EPS = 1e-10
PALETTE = {"blue": "#0F4D92", "green": "#8BCF8B", "red": "#B64342", "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E"}

PATHS = {
    "decomposition": OPR / "outputs" / "decomposition" / "prediction_decomposition.csv",
    "paired": OPR / "outputs" / "error_attribution" / "p1_p2_error_difference.csv",
    "baseline": OPR / "outputs" / "reproduction" / "baseline_reproduction_summary.csv",
    "opr_script": OPR / "scripts" / "phase_opr_diagnostic.py",
    "n28_script": N28F / "scripts" / "phase_n2_8f_run.py",
    "fold1": TTD / "outputs" / "fold1_12h" / "fold1_12h_pointwise.csv",
    "fold1_concentration": TTD / "outputs" / "fold1_12h" / "fold1_12h_loss_concentration.csv",
    "fold3": TTD / "outputs" / "fold3_nonzero" / "fold3_nonzero_pointwise.csv",
    "fold3_segments": TTD / "outputs" / "fold3_nonzero" / "fold3_independent_segments.csv",
    "ttd_results": TTD / "PHASE_TTD_RESULTS.md",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) <= EPS or np.std(b) <= EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(x: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2)))


def mdt(df: pd.DataFrame, limit: int | None = None) -> str:
    d = df.copy() if limit is None else df.head(limit).copy()
    if d.empty:
        return "_No rows._"
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
    return "| " + " | ".join(d.columns) + " |\n|" + "|".join(["---"] * len(d.columns)) + "|\n" + "\n".join("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in d.itertuples(index=False, name=None))


def source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in PATHS.values():
        if not path.exists():
            raise RuntimeError(f"Required read-only evidence missing: {path}")
    decomp = pd.read_csv(PATHS["decomposition"])
    paired = pd.read_csv(PATHS["paired"])
    baseline = pd.read_csv(PATHS["baseline"])
    fold1 = pd.read_csv(PATHS["fold1"])
    fold3 = pd.read_csv(PATHS["fold3"])
    if not set(decomp.model.unique()).issuperset({"P0", "P1", "P2", "RS"}):
        raise RuntimeError("OPR decomposition model cache incomplete")
    if set(paired.fold.unique()) != {"fold_1", "fold_2", "fold_3"}:
        raise RuntimeError("paired P1/P2 cache has unexpected folds")
    if paired.target_age_h.max() >= float(CFG["scope"]["development_age_h"][1]):
        raise RuntimeError("TEST FIREWALL: target age reaches Test boundary")
    return decomp, paired, baseline, fold1, fold3


def validate_contracts(decomp: pd.DataFrame, paired: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    keys = ["fold", "origin_row", "target_row", "horizon_step"]
    p1 = decomp[decomp.model == "P1"].sort_values(keys).reset_index(drop=True)
    p2 = decomp[decomp.model == "P2"].sort_values(keys).reset_index(drop=True)
    if not p1[keys].equals(p2[keys]):
        raise RuntimeError("P1/P2 decomposition keys differ")
    base_delta = float(np.max(np.abs(p1.base_prediction_DeltaV_mV - p2.base_prediction_DeltaV_mV)))
    identity = float(np.max(np.abs(decomp.identity_residual_mV)))
    sq_identity = float(np.max(np.abs(paired.D_identity_residual_mV2)))
    if identity > float(CFG["tolerances"]["decomposition_identity_mV"]):
        raise RuntimeError("B+C decomposition identity failed")
    if base_delta > float(CFG["tolerances"]["p1_p2_base_identity_mV"]):
        raise RuntimeError("P1/P2 base identity failed")
    if sq_identity > float(CFG["tolerances"]["squared_error_identity_mV2"]):
        raise RuntimeError("P1/P2 squared-error identity failed")
    actual = dict(zip(baseline.model, baseline.mean_rmse_delta_mV))
    for label, reference in CFG["frozen_reference_rmse_mV"].items():
        if label not in actual or abs(float(actual[label]) - float(reference)) > 0.002:
            raise RuntimeError(f"Baseline reproduction mismatch: {label}")
    if paired.duplicated(["fold", "origin_row", "horizon_step"]).any():
        raise RuntimeError("origin/horizon to target mapping is not one-to-one")
    fold_origin_sets = [set(g.origin_row) for _, g in paired.groupby("fold")]
    if any(a & b for i, a in enumerate(fold_origin_sets) for b in fold_origin_sets[i + 1:]):
        raise RuntimeError("fold origin rows overlap")
    return {"decomposition_max_residual_mV": identity, "p1_p2_base_max_difference_mV": base_delta, "squared_error_max_residual_mV2": sq_identity, "test_accessed": False}


def common_audit(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, g in paired.groupby("fold", sort=True):
        rows.append({"fold": fold, "origin_count": int(g.origin_row.nunique()), "prediction_point_count": int(len(g)), "unique_target_timestamp_count": int(g.target_time.nunique()), "unique_target_age_count": int(g.target_age_h.nunique()), "min_origin_age_h": float(g.origin_age_h.min()), "max_target_age_h": float(g.target_age_h.max()), "excluded_samples": 0, "exclusion_reason": "none; retained original OPR legal Development set", "test_rows": 0})
    return pd.DataFrame(rows)


def base_correction(decomp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = decomp[decomp.model == "P2"].copy()
    d["base_residual_mV"] = d.true_DeltaV_mV - d.base_prediction_DeltaV_mV
    d["final_residual_mV"] = d.final_prediction_DeltaV_mV - d.true_DeltaV_mV
    d["opposite_sign"] = d.base_prediction_DeltaV_mV * d.condition_correction_mV < 0
    d["abs_pair"] = np.minimum(abs(d.base_prediction_DeltaV_mV), abs(d.condition_correction_mV))
    threshold = d.groupby("fold")["abs_pair"].transform(lambda x: x.quantile(.75))
    d["large_amplitude_cancellation"] = d.opposite_sign & (d.abs_pair >= threshold)
    stat_rows, residual_rows = [], []
    for (fold, step, h), g in d.groupby(["fold", "horizon_step", "horizon_hours"], sort=True):
        b, c, f = g.base_prediction_DeltaV_mV, g.condition_correction_mV, g.final_prediction_DeltaV_mV
        stat_rows.append({"fold": fold, "horizon_step": int(step), "horizon_hours": float(h), "sample_count": len(g), "B_mean_mV": float(b.mean()), "B_std_mV": float(b.std(ddof=0)), "B_abs_p95_mV": float(abs(b).quantile(.95)), "C_mean_mV": float(c.mean()), "C_std_mV": float(c.std(ddof=0)), "C_abs_p95_mV": float(abs(c).quantile(.95)), "B_C_pearson": safe_corr(b, c), "B_plus_C_abs_mean_mV": float(abs(f).mean()), "opposite_sign_fraction": float(g.opposite_sign.mean()), "large_amplitude_cancellation_fraction": float(g.large_amplitude_cancellation.mean()), "final_rmse_mV": rmse(g.final_residual_mV)})
        eb = b - g.true_DeltaV_mV
        lhs = (b + c - g.true_DeltaV_mV) ** 2
        rhs = eb ** 2 + 2 * eb * c + c ** 2
        residual_rows.append({"fold": fold, "horizon_step": int(step), "horizon_hours": float(h), "sample_count": len(g), "B_only_rmse_mV": rmse(eb), "final_rmse_mV": rmse(g.final_residual_mV), "corr_residualB_C": safe_corr(-eb, c), "mean_B_only_squared_error_mV2": float(np.mean(eb ** 2)), "mean_cross_term_mV2": float(np.mean(2 * eb * c)), "mean_C_squared_mV2": float(np.mean(c ** 2)), "mean_final_squared_error_mV2": float(np.mean(lhs)), "squared_expansion_max_residual_mV2": float(np.max(abs(lhs - rhs)))})
    return pd.DataFrame(stat_rows), pd.DataFrame(residual_rows)


def enrich_paired(paired: pd.DataFrame) -> pd.DataFrame:
    d = paired.copy()
    true_mixed, pred_mixed = d.actual_zero_current_fraction > EPS, d.p2_zero_current_fraction > EPS
    d["state_group_q_derived"] = np.select([pred_mixed & ~true_mixed, ~pred_mixed & true_mixed, pred_mixed & true_mixed], ["pred_mixed_true_nonmixed", "pred_nonmixed_true_mixed", "both_mixed"], default="both_nonmixed")
    d["impact"] = np.select([d.D_mV2 < -EPS, d.D_mV2 > EPS], ["P2_improves", "P2_harms"], default="near_neutral")
    d["abs_deltaC_mV"] = abs(d.correction_change_mV)
    d["unique_target_event_id"] = d.fold.astype(str) + "|" + d.target_time.astype(str)
    keep = ["fold", "origin_row", "target_row", "origin_age_h", "target_age_h", "target_time", "horizon_step", "horizon_hours", "regime_strict", "regime_transition", "state_group_q_derived", "impact", "p1_error_mV", "p2_error_mV", "D_mV2", "D_identity_rhs_mV2", "D_identity_residual_mV2", "correction_p1_mV", "correction_p2_mV", "correction_change_mV", "abs_deltaC_mV", "p2_current_abs_error_A", "p2_q_abs_error", "unique_target_event_id"]
    return d[keep]


def attribution_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def aggregate(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"prediction_point_count": len(g), "unique_target_timestamp_count": g.unique_target_event_id.nunique(), "P1_RMSE_mV": rmse(g.p1_error_mV), "P2_RMSE_mV": rmse(g.p2_error_mV), "mean_D_mV2": float(g.D_mV2.mean()), "net_D_sum_mV2": float(g.D_mV2.sum()), "positive_D_sum_mV2": float(g.loc[g.D_mV2 > 0, "D_mV2"].sum()), "negative_D_sum_mV2": float(g.loc[g.D_mV2 < 0, "D_mV2"].sum()), "mean_abs_deltaC_mV": float(g.abs_deltaC_mV.mean()), "p95_abs_deltaC_mV": float(g.abs_deltaC_mV.quantile(.95))})
    regime = d.groupby(["fold", "regime_strict", "state_group_q_derived", "impact"], dropna=False).apply(aggregate, include_groups=False).reset_index()
    horizon = d.groupby(["fold", "horizon_step", "horizon_hours", "impact"], dropna=False).apply(aggregate, include_groups=False).reset_index()
    events = d.groupby(["fold", "target_time"], dropna=False).agg(prediction_point_count=("D_mV2", "size"), net_D_sum_mV2=("D_mV2", "sum"), positive_D_sum_mV2=("D_mV2", lambda x: x[x > 0].sum()), max_abs_deltaC_mV=("abs_deltaC_mV", "max"), impacted_horizon_count=("horizon_step", "nunique")).reset_index()
    events["event_impact"] = np.select([events.net_D_sum_mV2 < -EPS, events.net_D_sum_mV2 > EPS], ["net_P2_improves", "net_P2_harms"], default="near_neutral")
    return regime, horizon, events


def fold_case_tables(fold1: pd.DataFrame, fold3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f1 = fold1.copy(); f1["impact"] = np.where(f1.D_mV2 > EPS, "P2_harms", "P2_improves_or_neutral")
    f1_events = f1.groupby("target_time").agg(prediction_point_count=("D_mV2", "size"), net_D_sum_mV2=("D_mV2", "sum"), actual_q=("actual_q", "first"), predicted_q=("predicted_P2_q", "first"), mixed_probability=("predicted_P2_mixed_probability", "first"), p2_current_error_A=("p2_current_abs_error_A", "first")).reset_index().sort_values("net_D_sum_mV2", ascending=False)
    f1_events["cumulative_positive_D_share"] = f1_events.net_D_sum_mV2.clip(lower=0).cumsum() / max(float(f1_events.net_D_sum_mV2.clip(lower=0).sum()), EPS)
    f3 = fold3.copy(); actual, pred = f3.actual_q > EPS, f3.predicted_P2_mixed_probability >= .5
    f3["state_group"] = np.select([pred & ~actual, ~pred & actual, pred & actual], ["pred_mixed_true_nonmixed", "pred_nonmixed_true_mixed", "both_mixed"], default="both_nonmixed")
    f3["state_correct"] = actual == pred
    f3_summary = f3.groupby(["state_group", "state_correct"], dropna=False).apply(lambda g: pd.Series({"prediction_point_count": len(g), "unique_target_timestamp_count": g.target_time.nunique(), "P1_RMSE_mV": rmse(g.p1_error_mV), "P2_RMSE_mV": rmse(g.p2_error_mV), "mean_D_mV2": float(g.D_mV2.mean()), "net_D_sum_mV2": float(g.D_mV2.sum()), "p95_abs_deltaC_mV": float(abs(g.C_delta_mV).quantile(.95))}), include_groups=False).reset_index()
    return f1_events, f3_summary, pd.read_csv(PATHS["fold3_segments"])


def comparability() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "Strong causal-SG LSTM", "phase": "N2.7", "historical_rmse_mV": 7.786, "dataset": "RealHyFC", "target": "raw target / reported DeltaV context", "history_h": 24, "horizon_h": 12, "folds": "three Development folds", "seeds": "42,2024,3407", "future_condition_protocol": "not P1/P2 frozen-TimeXer protocol", "common_pointwise_cache_with_P2_TimeXer": False, "status": "HISTORICAL_CONTEXT_ONLY"},
        {"model": "TimesNet T0", "phase": "N2.7", "historical_rmse_mV": 9.908, "dataset": "RealHyFC", "target": "raw target / reported DeltaV context", "history_h": 24, "horizon_h": 12, "folds": "three Development folds", "seeds": "42,2024,3407", "future_condition_protocol": "not P1/P2 frozen-TimeXer protocol", "common_pointwise_cache_with_P2_TimeXer": False, "status": "HISTORICAL_CONTEXT_ONLY"},
        {"model": "SA-TimeXer P2", "phase": "OPR/G-STOP", "historical_rmse_mV": 21.346566, "dataset": "RealHyFC", "target": "DeltaV then anchor-restored voltage", "history_h": 24, "horizon_h": 12, "folds": "three Development folds", "seeds": "42", "future_condition_protocol": "P2 physics-consistent predicted 9-D exogenous tensor", "common_pointwise_cache_with_P2_TimeXer": True, "status": "FORMAL_WITHIN_TIMEXER_ONLY"},
    ])


def plot_outputs(stats: pd.DataFrame, horizon: pd.DataFrame, f1_events: pd.DataFrame, segments: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.linewidth": 1.8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, (fold, g) in zip(axes, stats.groupby("fold", sort=True)):
        ax.scatter(g.B_mean_mV, g.C_mean_mV, s=28, c=g.horizon_hours, cmap="viridis", edgecolor="black", linewidth=.25)
        lim = max(abs(g.B_mean_mV).max(), abs(g.C_mean_mV).max()) * 1.08
        ax.plot([-lim, lim], [lim, -lim], color=PALETTE["red"], ls="--", lw=1, label="exact cancellation")
        ax.set(xlabel="B mean (mV)", ylabel="C mean (mV)", title=f"Development only: {fold}")
    fig.colorbar(axes[-1].collections[0], ax=axes, label="Horizon (h)")
    fig.savefig(FIG / "base_correction_means_by_fold.png", dpi=300); fig.savefig(FIG / "base_correction_means_by_fold.pdf"); plt.close(fig)
    h = horizon.groupby(["horizon_hours", "impact"], as_index=False).net_D_sum_mV2.sum()
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for impact, color in [("P2_improves", PALETTE["green"]), ("P2_harms", PALETTE["red"]), ("near_neutral", PALETTE["neutral"])]:
        q = h[h.impact == impact]; ax.plot(q.horizon_hours, q.net_D_sum_mV2 / 1e3, marker="o", label=impact, color=color)
    ax.axhline(0, color="black", lw=1); ax.set(xlabel="Forecast horizon (h)", ylabel="Net squared-error change (10³ mV²)", title="Development only: P2 impact by horizon"); ax.legend(frameon=False)
    fig.savefig(FIG / "p1_p2_impact_by_horizon.png", dpi=300); fig.savefig(FIG / "p1_p2_impact_by_horizon.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    axes[0].bar(np.arange(min(5, len(f1_events))), f1_events.net_D_sum_mV2.head(5) / 1e3, color=PALETTE["red"], edgecolor="black")
    axes[0].set(xlabel="Ranked unique Fold 1 / 12 h target", ylabel="Net D (10³ mV²)", title="Development only: Fold 1 loss concentration")
    axes[1].bar(segments.temporal_segment_id.astype(str), segments.net_D_sum_mV2 / 1e3, color=np.where(segments.net_D_sum_mV2 > 0, PALETTE["red"], PALETTE["green"]), edgecolor="black")
    axes[1].axhline(0, color="black", lw=1); axes[1].set(xlabel="Independent temporal segment", ylabel="Net D (10³ mV²)", title="Development only: Fold 3 nonzero target bins")
    fig.savefig(FIG / "fold_case_error_concentration.png", dpi=300); fig.savefig(FIG / "fold_case_error_concentration.pdf"); plt.close(fig)


def reports(baseline, contracts, common, stats, residual, regime, horizon, events, f1_events, f3_summary, segments, modelcmp) -> None:
    (REP / "01_EXISTING_EVIDENCE_INVENTORY.md").write_text("# Existing evidence inventory\n\n| Evidence | Existing conclusion | Actually tested | Current status | Read-only source | Further calculation here |\n|---|---|---|---|---|---|\n| N2.7 complementarity | LSTM/TimesNet historical context only for this audit | yes | no common TimeXer pair | N2.7 results/cache inventory | comparability restriction |\n| N2.8B adaptation | TimeXer adaptations stopped; not a current frozen P2 comparison | yes | stopped | N2.8B decision | context only |\n| N2.8F | M0 Ridge is formal frozen TimeXer; FEA/CAF no formal gain | yes | pass reproduction / no new gain | N2.8F results/checkpoints | checkpoint hash and contract audit |\n| G-STOP P2 | improves frozen TimeXer average but worsens Fold 3 | yes | conditional | G-STOP results | paired impact map |\n| OPR B+C algebra | exact joint-output decomposition; not an independent B model | yes | pass | OPR decomposition / attribution | common-key numerical audit |\n| TTD cases | Fold 1 sparse exceptional loss; Fold 3 distributed segments | yes | inconclusive mechanism | TTD case caches | state-group reaggregation |\n| R0 rare state | q pathway diagnostic evidence; no internal TimeXer change justified | yes | exploratory only | R0 results | context only |\n| R1 scenarios | calibration internally failed | yes | stopped | R1 results | context only |\n| TR2 correction | 10-min direct q beats simple Ridge only | yes | post-hoc diagnostic | TR2 reports | context only |\n| MR pilot | MR does not beat P2 under frozen condition gate | yes | condition gate fail | MR reports | context only |\n\nNo historical proposal is recorded as a completed model unless its phase saved an actual result.\n", encoding="utf-8")
    (REP / "02_DECOMPOSITION_CONTRACT.md").write_text("# Decomposition contract\n\nFrozen `StepAlignedTimeXer` computes `base = base.head(history encoding)`, then applies horizon-specific FiLM to `history state + horizon state`; `prediction = base + correction(modified)`. Thus B/C are a decomposition of one jointly trained forward pass, not separately trained models. The frozen contract is 48 x 0.5 h = 24 h historical input, 24 x 0.5 h = 12 h forecast, 14 history features, nine future-condition features, DeltaV target, and final voltage recovery `anchor + scaled DeltaV`.\n\n" + mdt(pd.DataFrame([contracts])) + "\n\nBaseline cache reproduction:\n\n" + mdt(baseline) + "\n", encoding="utf-8")
    (REP / "03_ALIGNMENT_AND_LEAKAGE_AUDIT.md").write_text("# Alignment and leakage audit\n\nAll formal attribution rows are the original OPR P1/P2 Development-only pointwise keys. No error-dependent selection was applied. Target ages remain strictly below the sealed 1819.997738 h Development/Test boundary.\n\n" + mdt(common) + "\n\nTest rows read, predictions, metrics and selections: **0 / false / false / false**.\n", encoding="utf-8")
    bc_fold = stats.groupby("fold", as_index=False).agg(opposite_sign_fraction=("opposite_sign_fraction", "mean"), large_amplitude_cancellation_fraction=("large_amplitude_cancellation_fraction", "mean"), mean_B_C_pearson=("B_C_pearson", "mean"))
    (REP / "04_BASE_CORRECTION_NUMERICAL_AUDIT.md").write_text("# Base-correction numerical audit\n\nB and C have large mV-scale values and are opposite-signed at roughly half of within-horizon prediction points; the predeclared large-amplitude cancellation flag occurs at about one tenth. This is numerical compensation within a jointly trained forward pass, not evidence that B is an independently trained history-only model. The verified squared expansion is descriptive only: `(B+C-y)^2=(B-y)^2+2(B-y)C+C^2`; it is not a physical attribution of independent error sources.\n\nB/C numerical relationship:\n\n" + mdt(bc_fold) + "\n\nPer-fold/horizon statistics are in `outputs/base_correction_statistics.csv`; residual matching and squared expansion checks are in `outputs/base_residual_analysis.csv`.\n\n" + mdt(residual.groupby("fold", as_index=False).agg(B_only_rmse_mV=("B_only_rmse_mV", "mean"), final_rmse_mV=("final_rmse_mV", "mean"), corr_residualB_C=("corr_residualB_C", "mean"), squared_expansion_max_residual_mV2=("squared_expansion_max_residual_mV2", "max"))) + "\n", encoding="utf-8")
    (REP / "05_P1_P2_ERROR_ATTRIBUTION.md").write_text("# P1-to-P2 paired error attribution\n\nP1/P2 share the same TimeXer checkpoint, history, origin, target and B term. `DeltaC=C_P2-C_P1` is therefore the only forward-input-related difference. The identity `D=e_P2²-e_P1²=2e_P1DeltaC+DeltaC²` passed under the frozen tolerance. Overlapping windows are reported as prediction points; independent target timestamps are separately counted.\n\n" + mdt(regime.sort_values(["fold", "net_D_sum_mV2"]).head(24)) + "\n", encoding="utf-8")
    (REP / "06_FOLD1_12H_CASE.md").write_text("# Fold 1 / 12 h case\n\nThe official TTD object is retained unchanged. Its top one unique target contributes 93.03% of positive D and its top five contribute 98.36%. The leading event is near 1130 h and is a predicted-mixed / true-nonmixed state error with a very large condition correction. This is a sparse exceptional-state loss, not evidence of global timing failure. Removing it is diagnostic sensitivity only and never replaces the formal total.\n\n" + mdt(f1_events.head(5)) + "\n", encoding="utf-8")
    (REP / "07_FOLD3_NONZERO_CASE.md").write_text("# Fold 3 nonzero-current case\n\nThis formal TTD object means individual 30-minute target bins with nonzero current; it is not a claim that every full 24-step trajectory is continuously nonzero. The harm occurs in four independent temporal segments. In the cached object, state-correct `both_nonmixed` rows favor P2, while 54 predicted-mixed / true-nonmixed rows covering five unique target times dominate loss. This shows state errors co-occur with harmful DeltaC, but cannot isolate probability error, current magnitude error, or FiLM response.\n\nState groups:\n\n" + mdt(f3_summary) + "\n\nSegments:\n\n" + mdt(segments) + "\n", encoding="utf-8")
    (REP / "08_HISTORY_REPRESENTATION_DIAGNOSIS.md").write_text("# History representation diagnosis\n\nB varies across histories and horizons, and P1/P2 have identical B under identical histories. B-only RMSE is 192.383 mV, but this does not measure an independently trained history-only model: B was optimized jointly with C and can be numerically compensated by C. The present evidence therefore neither proves history encoding ineffective nor identifies a specific history-backbone defect. Testing an independent history model would require a separate authorized training protocol.\n", encoding="utf-8")
    (REP / "09_STRONG_BASELINE_COMPARABILITY.md").write_text("# Strong baseline comparability\n\n" + mdt(modelcmp) + "\n\nLSTM, TimesNet and P2-TimeXer historical numbers provide context, but no saved all-model cache verifies the same origins, targets, seeds, preprocessing and future-condition protocol. Therefore no formal paired architecture ranking is generated here.\n", encoding="utf-8")
    (REP / "10_ROUTE_DECISION.md").write_text("# Route decision\n\n## Verified facts\n\n- Frozen OPR B+C and P1/P2 squared-error identities reproduce from Development caches.\n- P2 improves many ordinary/state-correct points, while rare predicted-mixed / true-nonmixed events can create extreme harmful corrections.\n- Fold 1 / 12 h is dominated by a few targets; Fold 3 harm spans four segments and is therefore not the same concentration pattern.\n- Alignment/index error is not supported by TTD; MR did not surpass P2 at its preregistered condition gate.\n\n## Unresolved competition\n\nAt rare state errors, the saved cache cannot separate misclassified state probability, correlated current-magnitude forecast error, and the frozen FiLM correction response. B-only error also cannot identify independent history-only ability.\n\n## Decision\n\n**INCONCLUSIVE.** There is no single separable code-level mechanism that distinguishes all alternatives and no authorization for a new model experiment. A future protocol may test a falsifiable task-information hypothesis only after separately establishing forecastability and a causal target definition; it must not be presented as a proven TimeXer architecture remedy.\n", encoding="utf-8")
    (REP / "FINAL_RESEARCH_DECISION.md").write_text("# Final research decision\n\n**INCONCLUSIVE**\n\nThe evidence supports numerical B/C compensation and rare-state-associated harmful P2 corrections, but does not separate state probability, condition magnitude, and FiLM response as unique mechanisms. No TimeXer change, model training, or Test access is authorized.\n", encoding="utf-8")


def main() -> None:
    for p in (OUT, REP, FIG):
        p.mkdir(parents=True, exist_ok=True)
    decomp, paired, baseline, fold1, fold3 = source_data()
    contracts = validate_contracts(decomp, paired, baseline)
    common = common_audit(paired)
    stats, residual = base_correction(decomp)
    paired_out = enrich_paired(paired)
    regime, horizon, events = attribution_tables(paired_out)
    f1_events, f3_summary, segments = fold_case_tables(fold1, fold3)
    modelcmp = comparability()
    common.to_csv(OUT / "common_origin_audit.csv", index=False)
    stats.to_csv(OUT / "base_correction_statistics.csv", index=False)
    residual.to_csv(OUT / "base_residual_analysis.csv", index=False)
    paired_out.to_csv(OUT / "paired_condition_error_attribution.csv", index=False)
    regime.to_csv(OUT / "regime_attribution.csv", index=False)
    horizon.to_csv(OUT / "horizon_attribution.csv", index=False)
    events.to_csv(OUT / "independent_target_events.csv", index=False)
    f1_events.to_csv(OUT / "fold1_12h_unique_target_contribution.csv", index=False)
    f3_summary.to_csv(OUT / "fold3_nonzero_state_groups.csv", index=False)
    segments.to_csv(OUT / "fold3_independent_segments.csv", index=False)
    modelcmp.to_csv(OUT / "model_comparability.csv", index=False)
    plot_outputs(stats, horizon, f1_events, segments)
    reports(baseline, contracts, common, stats, residual, regime, horizon, events, f1_events, f3_summary, segments, modelcmp)
    hashes = {name: sha(path) for name, path in PATHS.items()}
    ckpts = {p.name: sha(p) for p in sorted((N28F / "checkpoints").glob("M0_SA_TimeXer_Ridge_*_seed42.pt"))}
    audit = {"command": "python scripts/run_route_audit.py", "head": subprocess.check_output(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"], text=True).strip(), "python": platform.python_version(), "torch": torch.__version__, "test_accessed": False, "source_sha256": hashes, "frozen_checkpoint_sha256": ckpts, "contracts": contracts, "decision": "INCONCLUSIVE", "training_performed": False}
    (PHASE / "RUN_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (PHASE / "FINAL_DECISION.yaml").write_text("status: INCONCLUSIVE\ntraining_performed: false\ntest_accessed: false\nreason: rare-state probability, magnitude, and FiLM-response mechanisms remain non-separable from frozen caches\n", encoding="utf-8")
    print(json.dumps({"status": "INCONCLUSIVE", **contracts}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

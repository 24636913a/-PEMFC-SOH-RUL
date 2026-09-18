#!/usr/bin/env python
"""Frozen causal audit for Phase RS-SA-TimeXer; never reads the sealed Test split."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, r2_score


ROOT = Path(__file__).resolve().parent.parent
RENEW_ROOT = ROOT.parent
G_ROOT = RENEW_ROOT / "Phase_G_STOP_Physics_Consistent_Future_Conditions"
N28F_ROOT = RENEW_ROOT / "Phase_N2_8F_Uncertainty_Robust_SA_TimeXer"
FOLDS = ("fold_1", "fold_2", "fold_3")
EPS = 1e-12
EXPECTED_P0 = 24.538245
EXPECTED_P2 = 21.346566


def out(relative: str) -> Path:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    frame.to_csv(out(relative), index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(obj: Any, relative: str) -> None:
    out(relative).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def p2_train_with_probability(p: dict[str, Any], aux_train: np.ndarray, cfg: dict[str, Any], n28f, gs, fold: str):
    """Legal OOF P2 inputs and p_mixed, with target-completion checks per block."""
    n = len(p["htr"])
    start = n28f.first_index_with_completed(p["train"].target_end_age_h, p["train"].origin_age_h, int(cfg["forecast"]["oof_min_completed_windows"]))
    q = np.full((n, 24), np.nan, dtype=float)
    probability = np.full((n, 24), np.nan, dtype=float)
    aux = np.full((n, 24, 2), np.nan, dtype=float)
    audits: list[dict[str, Any]] = []
    for block, (left, right) in enumerate(n28f.block_edges(start, n, int(cfg["forecast"]["oof_blocks"])), start=1):
        origin = float(p["train"].origin_age_h[left])
        fit = n28f.completed_indices(p["train"].target_end_age_h, origin)
        if len(fit) < int(cfg["forecast"]["oof_min_completed_windows"]):
            raise RuntimeError(f"{fold}: P2 OOF warm-up violation")
        q_model = gs.TwoPartQForecaster(cfg).fit(p["htr"][fit], p["ztr"][fit, :, 8])
        q[left:right], probability[left:right] = q_model.predict(p["htr"][left:right])
        aux_model = gs.AuxiliaryRidge(float(cfg["forecast"]["conditional_on_stats_ridge_alpha"])).fit(p["htr"][fit], aux_train[fit])
        aux[left:right] = aux_model.predict(p["htr"][left:right])
        max_end = float(p["train"].target_end_age_h[fit].max())
        if max_end > origin + EPS:
            raise RuntimeError(f"{fold}: RS OOF target-completion leakage")
        audits.append({"fold": fold, "block": block, "fit_count": int(len(fit)), "predict_count": int(right-left), "max_fit_target_end_h": max_end, "min_prediction_origin_h": origin, "target_completion_pass": True, "train_only_scaler": True})
    keep = np.isfinite(q).all(axis=1) & np.isfinite(probability).all(axis=1) & np.isfinite(aux).all(axis=(1, 2))
    candidate = p["z_oof"].copy()
    mu = np.maximum(aux[:, :, 0], 0.0)
    sigma = np.maximum(aux[:, :, 1], 0.0)
    candidate[:, :, 8] = q
    candidate[:, :, 0] = (1.0 - q) * mu
    candidate[:, :, 1] = np.sqrt(np.maximum(0.0, (1.0 - q) * sigma**2 + q * (1.0 - q) * mu**2))
    p2, _ = gs.project_future(candidate, p["area"])
    return p2, probability, keep, audits


def selector_records(fold: str, stage: str, probability: np.ndarray, selected: np.ndarray, p1: np.ndarray, truth: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    true_event = truth[:, :, 8] > EPS
    pred_event = selected.astype(bool)
    result_fold: list[dict[str, Any]] = []
    result_horizon: list[dict[str, Any]] = []
    result_confusion: list[dict[str, Any]] = []
    result_disagree: list[dict[str, Any]] = []
    for scope, step in [("all_horizons", None), *[(f"h{h+1}", h) for h in range(24)]]:
        actual = true_event if step is None else true_event[:, step:step+1]
        decision = pred_event if step is None else pred_event[:, step:step+1]
        p = probability if step is None else probability[:, step:step+1]
        q1 = p1[:, :, 8] if step is None else p1[:, step:step+1, 8]
        tp = int(np.sum(decision & actual)); fp = int(np.sum(decision & ~actual)); fn = int(np.sum(~decision & actual)); tn = int(np.sum(~decision & ~actual))
        precision, recall, f1, _ = precision_recall_fscore_support(actual.ravel(), decision.ravel(), average="binary", zero_division=0)
        base = {"fold": fold, "stage": stage, "scope": scope, "sample_count": int(actual.size), "g0_count": int((~decision).sum()), "g1_count": int(decision.sum()), "g0_fraction": float((~decision).mean()), "g1_fraction": float(decision.mean()), "actual_mixed_fraction": float(actual.mean()), "predicted_mixed_fraction": float(decision.mean()), "mean_predicted_probability": float(p.mean()), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
        if step is None:
            result_fold.append(base)
        else:
            result_horizon.append({**base, "horizon_step": step+1, "horizon_hours": (step+1)*0.5})
        result_confusion.append({"fold": fold, "stage": stage, "scope": scope, "true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp})
        g0 = ~decision
        denom = int(g0.sum())
        incompatible = int(np.sum(g0 & (q1 > EPS)))
        result_disagree.append({"fold": fold, "stage": stage, "scope": scope, "g0_selected_p1_count": denom, "g0_selected_p1_with_nonzero_q_count": incompatible, "selector_input_regime_disagreement_rate": float(incompatible / denom) if denom else np.nan, "definition": "among g=0 selections, fraction whose selected P1 vector retains q>0"})
    return result_fold, result_horizon, result_confusion, result_disagree


def voltage_metrics(gs, p, pred: np.ndarray) -> dict[str, float]:
    return gs.voltage_metrics(p["valid"].y_abs, pred, p["valid"].anchor)


def horizon_records(model: str, fold: str, p, pred: np.ndarray) -> list[dict[str, Any]]:
    target = p["valid"].y_abs - p["valid"].anchor[:, None]
    guess = pred - p["valid"].anchor[:, None]
    rows = []
    for h in range(24):
        error = guess[:, h] - target[:, h]
        rows.append({"model": model, "fold": fold, "horizon_step": h+1, "horizon_hours": (h+1)*0.5, "r2_delta": float(r2_score(target[:, h], guess[:, h])), "rmse_delta_mV": float(np.sqrt(np.mean(error**2))*1000.0), "mae_delta_mV": float(np.mean(np.abs(error))*1000.0)})
    return rows


def tail(model: str, arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    err = np.concatenate([np.abs(x["pred"] - x["truth"]).ravel()*1000.0 for x in arrays.values()])
    return {"model": model, "aggregation": "pooled_prediction_points_across_three_folds", "n_prediction_points": int(err.size), "p50_mV": float(np.quantile(err, .5)), "p90_mV": float(np.quantile(err, .9)), "p95_mV": float(np.quantile(err, .95)), "p99_mV": float(np.quantile(err, .99)), "max_mV": float(err.max())}


def paired_fold_bootstrap(candidate: dict[str, dict[str, np.ndarray]], reference: dict[str, dict[str, np.ndarray]], block: int, repetitions: int, comparison: str) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    deltas = np.empty(repetitions, dtype=float)
    point_deltas: dict[str, float] = {}
    for fold in FOLDS:
        ec = candidate[fold]["pred"] - candidate[fold]["truth"]
        er = reference[fold]["pred"] - reference[fold]["truth"]
        point_deltas[fold] = float((np.sqrt(np.mean(ec**2)) - np.sqrt(np.mean(er**2)))*1000.0)
    for rep in range(repetitions):
        fold_rmse_candidate, fold_rmse_reference = [], []
        for fold in FOLDS:
            c, r, truth = candidate[fold]["pred"], reference[fold]["pred"], candidate[fold]["truth"]
            n = len(truth); blocks = math.ceil(n / block)
            starts = rng.integers(0, n, size=blocks)
            idx = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
            fold_rmse_candidate.append(float(np.sqrt(np.mean((c[idx]-truth[idx])**2))*1000.0))
            fold_rmse_reference.append(float(np.sqrt(np.mean((r[idx]-truth[idx])**2))*1000.0))
        deltas[rep] = float(np.mean(fold_rmse_candidate) - np.mean(fold_rmse_reference))
    lower, upper = float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))
    return {"comparison": comparison, "delta_definition": "unweighted_mean_3fold_RMSE_candidate_minus_reference_mV", "block_length_origins": block, "repetitions": repetitions, **{f"{fold}_rmse_difference_mV": point_deltas[fold] for fold in FOLDS}, "point_estimate_mean_rmse_difference_mV": float(np.mean(list(point_deltas.values()))), "bootstrap_mean_rmse_difference_mV": float(deltas.mean()), "ci95_lower_mV": lower, "ci95_upper_mV": upper, "ci_crosses_zero": bool(lower <= 0.0 <= upper), "interpretation": "IMPROVEMENT_NOT_STATISTICALLY_RESOLVED" if lower <= 0.0 <= upper else "DEVELOPMENT_CI_DOES_NOT_CROSS_ZERO"}


def make_figures(summary: pd.DataFrame, horizon: pd.DataFrame, selector: pd.DataFrame) -> None:
    colors = {"P0_Original_Ridge": "#B64342", "P1_Physics_Projected_Ridge": "#CFCECE", "P2_Zero_Current_Aware": "#8BCF8B", "RS_Regime_Selective": "#0F4D92"}
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 11, "axes.linewidth": 1.8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    models = summary["model"].tolist(); x = np.arange(len(models)); values = summary["mean_rmse_delta_mV"].to_numpy(float)
    bars = ax.bar(x, values, color=[colors[m] for m in models], edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, values): ax.text(b.get_x()+b.get_width()/2, v+.15, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, ["P0", "P1", "P2", "RS"]); ax.set_ylabel("Three-fold mean RMSE (mV)"); ax.set_ylim(max(0, values.min()-2), values.max()+2); fig.tight_layout(pad=1.1); fig.savefig(out("outputs/figures/frozen_rmse_comparison.png"), dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for model, group in horizon[horizon.fold == "mean_3fold"].groupby("model", sort=False): ax.plot(group.horizon_hours, group.rmse_delta_mV, label=model.replace("_", " "), color=colors[model], linewidth=2.2)
    ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("RMSE (mV)"); ax.legend(fontsize=8); fig.tight_layout(pad=1.1); fig.savefig(out("outputs/figures/frozen_rmse_by_horizon.png"), dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    valid = selector[selector.stage == "validation"]
    ax.bar(np.arange(3), valid.g1_fraction, color="#0F4D92", edgecolor="black")
    ax.set_xticks(np.arange(3), ["Fold 1", "Fold 2", "Fold 3"]); ax.set_ylim(0, 1); ax.set_ylabel("P2 selection fraction (g=1)"); fig.tight_layout(pad=1.1); fig.savefig(out("outputs/figures/selector_selection_by_fold.png"), dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off")
    boxes = [(0.02, .38, .16, .24, "X history\n48 × 14"), (.27, .15, .16, .20, "Z P1\n24 × 9"), (.27, .62, .16, .20, "Z P2\n24 × 9"), (.49, .70, .13, .15, "p mixed\n24 × 1"), (.67, .70, .13, .15, "g = 1[p≥0.5]\n24 × 1"), (.55, .34, .20, .20, "Z selected\n24 × 9"), (.84, .38, .14, .24, "Step-Aligned\nFiLM + TimeXer\nΔV: 24 × 1")]
    for left, bottom, width, height, label in boxes:
        ax.add_patch(plt.Rectangle((left,bottom), width,height, fill=False, linewidth=1.8, color="#0F4D92")); ax.text(left+width/2,bottom+height/2,label,ha="center",va="center",fontsize=10)
    for start,end in [((.18,.50),(.27,.25)), ((.18,.50),(.27,.72)), ((.18,.56),(.49,.76)), ((.62,.775),(.67,.775)), ((.43,.25),(.55,.40)), ((.43,.72),(.55,.48)), ((.73,.70),(.69,.54)), ((.18,.50),(.84,.48)), ((.75,.44),(.84,.51))]: ax.annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"->","lw":1.35})
    ax.text(.65,.56,"ZRS=(1−g)ZP1+gZP2",ha="center",fontsize=8.5); fig.tight_layout(pad=.2); fig.savefig(out("outputs/figures/rs_sa_timexer_structure.png"), dpi=300); plt.close(fig)


def write_reports(summary: pd.DataFrame, fold_metrics: pd.DataFrame, violation: pd.DataFrame, boot: pd.DataFrame, gate: dict[str, Any], selector: pd.DataFrame, regime: pd.DataFrame, firewall: dict[str, Any]) -> None:
    rs = summary.set_index("model").loc["RS_Regime_Selective"]
    p0 = summary.set_index("model").loc["P0_Original_Ridge"]
    p2 = summary.set_index("model").loc["P2_Zero_Current_Aware"]
    selector_note = selector[selector.stage == "validation"]
    reports = {
        "REGIME_SELECTOR_AUDIT.md": f"# Regime selector audit\n\nThe selector uses only P2 predicted probability at the frozen threshold 0.5. Validation P2-selection fractions are {', '.join(f'{x:.3f}' for x in selector_note.g1_fraction)} for folds 1–3. `g=0` can select a P1 vector with nonzero q; this input-state disagreement is quantified in `outputs/selector/selector_input_disagreement.csv`, rather than hidden.\n",
        "PHYSICAL_CONSISTENCY_AUDIT.md": "# Physical consistency audit\n\nP1, P2, and RS are projected and audited for non-negativity, J=I/A, q range, and q=1 compatibility. RS is a stepwise mixture of complete valid P1/P2 vectors and is independently re-audited. The machine-readable findings are in `outputs/frozen_model/physical_violation_metrics.csv`.\n",
        "FROZEN_MODEL_RESULTS.md": "# Frozen model results\n\nThe complete fold and unweighted three-fold summaries are in `outputs/frozen_model/frozen_model_fold_metrics.csv` and `outputs/frozen_model/frozen_model_summary.csv`.\n",
        "BOOTSTRAP_AUDIT.md": "# Paired time-block bootstrap\n\nThe statistic is the unweighted mean of three fold RMSEs. Pairs were resampled by common contiguous origin blocks, 2,000 repetitions, seed 42. A CI that includes zero is labeled `IMPROVEMENT_NOT_STATISTICALLY_RESOLVED`. Complete results are in `outputs/bootstrap/bootstrap_main_metric.csv` and `outputs/bootstrap/bootstrap_block_sensitivity.csv`.\n",
        "RETRAINING_RESULTS.md": "# Retraining results\n\nStatus: NOT_RUN pending Gate A. If Gate A is false, no neural-network retraining is authorized.\n",
        "MULTISEED_RESULTS.md": "# Multi-seed results\n\nStatus: NOT_RUN pending Gate B. The 42/2024/3407 protocol is not authorized unless every Gate B criterion passes.\n",
        "PHASE_RS_RESULTS.md": f"# Phase RS-SA-TimeXer results\n\n## Frozen result\n\nRS mean RMSE is {rs.mean_rmse_delta_mV:.6f} mV and R² is {rs.mean_r2_delta:.6f}. Relative to P0, change is {(p0.mean_rmse_delta_mV-rs.mean_rmse_delta_mV)/p0.mean_rmse_delta_mV*100:.3f}%; relative to P2, change is {(p2.mean_rmse_delta_mV-rs.mean_rmse_delta_mV)/p2.mean_rmse_delta_mV*100:.3f}%.\n\n## Gate A\n\nStatus: `{gate['status']}`. Every predeclared condition is recorded in `outputs/frozen_model/gate_a_audit.csv`.\n\n## Test firewall\n\nAll Test access flags are false and target rows read from Test are zero.\n"}
    for filename, content in reports.items(): out(filename).write_text(content, encoding="utf-8")
    decision = {"phase_status": "FROZEN_AUDIT_COMPLETED", "baseline_reproduction": {"P0": float(p0.mean_rmse_delta_mV), "P2": float(p2.mean_rmse_delta_mV)}, "selector_audit": {"fixed_threshold": 0.5, "nonconstant_validation": bool(gate["selector_nonconstant"])}, "physical_consistency": {"RS_mean_trajectory_violation_rate": float(violation.query("model == 'RS_Regime_Selective' and fold == 'mean_3fold'").physical_violation_trajectory_rate.iloc[0])}, "frozen_models": {row.model: {"mean_rmse_delta_mV": float(row.mean_rmse_delta_mV), "mean_r2_delta": float(row.mean_r2_delta)} for row in summary.itertuples()}, "gate_a_passed": bool(gate["passed"]), "retraining": {"status": "NOT_RUN_PENDING_GATE_A" if not gate["passed"] else "AUTHORIZED_NOT_YET_RUN"}, "gate_b_passed": "NOT_RUN", "multiseed": {"status": "NOT_RUN_PENDING_GATE_B"}, "paper_candidate": "NOT_ESTABLISHED", "supported_contributions": ["P1/P2/RS physical-condition audit", "frozen deterministic selector audit"], "unsupported_claims": ["Test performance", "stable neural retraining advantage"], "test_performance_accessed": False, "next_action": "Run matched retraining only if gate_a_passed is true"}
    out("PHASE_RS_DECISION.yaml").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(firewall, "outputs/test_firewall_audit.json")


def main() -> None:
    gs = load_module("phase_g_stop_readonly", G_ROOT / "scripts" / "phase_g_stop_experiment.py")
    cfg = gs.load_config()
    n28f = gs.import_n28f(Path(cfg["n2_8f_root"]))
    ncfg = n28f.load_config()
    common, frame, windows, _ = n28f.load_development(ncfg)
    if frame["split"].eq("test").any(): raise RuntimeError("TEST FIREWALL: development loader contained Test")
    raw_aux = pd.read_csv(G_ROOT / "outputs" / "physical_audit" / "raw_conditional_current_statistics.csv")
    frame = frame.merge(raw_aux[["bin_start_ophrs2_h", "mu_on_A", "sigma_on_A"]], on="bin_start_ophrs2_h", how="left", validate="one_to_one")
    if frame[["mu_on_A", "sigma_on_A"]].isna().any().any(): raise RuntimeError("Missing development auxiliary statistic")
    packages = {fold: n28f.package_fold(common, frame, windows, ncfg, fold) for fold in FOLDS}
    records = {name: {} for name in ("P0_Original_Ridge", "P1_Physics_Projected_Ridge", "P2_Zero_Current_Aware", "RS_Regime_Selective")}
    fold_rows=[]; horizon_rows=[]; regime_rows=[]; violation_rows=[]; selector_fold=[]; selector_horizon=[]; selector_confusion=[]; selector_disagreement=[]; causal_rows=[]
    for fold in FOLDS:
        p = packages[fold]
        aux_train = gs.future_aux(frame, p["train"], ["mu_on_A", "sigma_on_A"])
        aux_valid = gs.future_aux(frame, p["valid"], ["mu_on_A", "sigma_on_A"])
        p2_train, prob_train, p2_keep, audits = p2_train_with_probability(p, aux_train, cfg, n28f, gs, fold)
        causal_rows.extend(audits)
        if not np.array_equal(p2_keep, p["ridge_keep"]): raise RuntimeError(f"{fold}: OOF origin mismatch against P0")
        p0_train = p["z_oof"].copy(); p1_train, _ = gs.project_future(p0_train, p["area"]); g_train = prob_train >= 0.5; rs_train, _ = gs.project_future(np.where(g_train[:, :, None], p2_train, p1_train), p["area"])
        p0 = p["z_ridge"].copy(); p1, _ = gs.project_future(p0, p["area"]); p2, prob_valid, _ = gs.build_p2(p, aux_train, aux_valid, cfg); g_valid = prob_valid >= 0.5; rs, _ = gs.project_future(np.where(g_valid[:, :, None], p2, p1), p["area"])
        for label, values in {"P1_Physics_Projected_Ridge":p1,"P2_Zero_Current_Aware":p2,"RS_Regime_Selective":rs}.items():
            violation_rows.append(gs.violation_metrics(label, fold, values, p["area"], float(cfg["g1"]["area_identity_median_relative_tolerance"])))
        rs_violation = violation_rows[-1]["physical_violation_trajectory_rate"]
        if rs_violation > max(violation_rows[-2]["physical_violation_trajectory_rate"], violation_rows[-3]["physical_violation_trajectory_rate"]) + EPS: raise RuntimeError(f"{fold}: RS introduced physical violation")
        a,b,c,d = selector_records(fold, "oof_train", prob_train[p2_keep], g_train[p2_keep], p1_train[p2_keep], p["ztr"][p2_keep]); selector_fold += a; selector_horizon += b; selector_confusion += c; selector_disagreement += d
        a,b,c,d = selector_records(fold, "validation", prob_valid, g_valid, p1, p["zva"]); selector_fold += a; selector_horizon += b; selector_confusion += c; selector_disagreement += d
        zvalues = {"P0_Original_Ridge":p0,"P1_Physics_Projected_Ridge":p1,"P2_Zero_Current_Aware":p2,"RS_Regime_Selective":rs}
        checkpoint = N28F_ROOT / "checkpoints" / f"M0_SA_TimeXer_Ridge_{fold}_seed42.pt"
        model = n28f.StepAlignedTimeXer(ncfg, caf=False); model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)); model.to(n28f.device()).eval()
        predictions={}
        for label,z in zvalues.items():
            pred = gs.inference(n28f, model, p, z); predictions[label]=pred
            fold_rows.append({"model":label,"fold":fold,**voltage_metrics(gs,p,pred)})
            horizon_rows += horizon_records(label,fold,p,pred)
            records[label][fold] = {"truth":p["valid"].y_abs.astype(float),"pred":pred.astype(float),"anchor":p["valid"].anchor.astype(float)}
            np.savez_compressed(out(f"outputs/frozen_model/{label}_{fold}_predictions.npz"), y_true=p["valid"].y_abs.astype(float), y_pred=pred.astype(float), anchor=p["valid"].anchor.astype(float), origin_age_h=p["valid"].origin_age_h.astype(float), target_end_age_h=p["valid"].target_end_age_h.astype(float), row_origin_index=p["valid"].row_origin_index.astype(int), future_conditions=z.astype(float), p_mixed=prob_valid.astype(float), selector_g=g_valid.astype(np.int8))
        regime_rows += gs.regime_rows(frame,p["valid"],predictions,fold)
    selector_by_fold=pd.DataFrame(selector_fold); selector_by_horizon=pd.DataFrame(selector_horizon); selector_confusion_frame=pd.DataFrame(selector_confusion); selector_disagreement_frame=pd.DataFrame(selector_disagreement)
    violation=pd.DataFrame(violation_rows); violation=pd.concat([violation,violation.groupby("model",as_index=False).mean(numeric_only=True).assign(fold="mean_3fold")],ignore_index=True,sort=False)
    folds=pd.DataFrame(fold_rows); summary=[]
    for model,group in folds.groupby("model",sort=False):
        row={"model":model,"fold_average_definition":"unweighted_mean_of_three_folds"}
        for col in ("r2_delta","rmse_delta_mV","mae_delta_mV","p95_mV","p99_mV"): row[f"mean_{col}"]=float(group[col].mean())
        for fold in FOLDS: row[f"{fold}_rmse_delta_mV"]=float(group.loc[group.fold==fold,"rmse_delta_mV"].iloc[0])
        summary.append(row)
    summary=pd.DataFrame(summary); p0mean=float(summary.query("model == 'P0_Original_Ridge'").mean_rmse_delta_mV.iloc[0]); p2mean=float(summary.query("model == 'P2_Zero_Current_Aware'").mean_rmse_delta_mV.iloc[0])
    if abs(p0mean-EXPECTED_P0)>.001 or abs(p2mean-EXPECTED_P2)>.001: raise RuntimeError(f"Frozen reproduction failed P0={p0mean:.6f}, P2={p2mean:.6f}")
    horizon=pd.DataFrame(horizon_rows); horizon=pd.concat([horizon,horizon.groupby(["model","horizon_step","horizon_hours"],as_index=False).mean(numeric_only=True).assign(fold="mean_3fold")],ignore_index=True,sort=False)
    regimes=pd.DataFrame(regime_rows); tails=pd.DataFrame([tail(model,records[model]) for model in records])
    boot_rows=[]
    for name,candidate,reference in (("RS_vs_P0",records["RS_Regime_Selective"],records["P0_Original_Ridge"]),("RS_vs_P2",records["RS_Regime_Selective"],records["P2_Zero_Current_Aware"])):
        for block in (24,48,96): boot_rows.append(paired_fold_bootstrap(candidate,reference,block,2000,name))
    bootstrap=pd.DataFrame(boot_rows)
    rs=float(summary.query("model == 'RS_Regime_Selective'").mean_rmse_delta_mV.iloc[0]); fold3=float(folds.query("model == 'RS_Regime_Selective' and fold == 'fold_3'").rmse_delta_mV.iloc[0]); p0fold3=float(folds.query("model == 'P0_Original_Ridge' and fold == 'fold_3'").rmse_delta_mV.iloc[0]); p2fold3continuous=float(regimes.query("model == 'P2_Zero_Current_Aware' and fold == 'fold_3' and category == 'transition_state' and regime == 'continuous_nonzero'").rmse_delta_mV.iloc[0]); rsfold3continuous=float(regimes.query("model == 'RS_Regime_Selective' and fold == 'fold_3' and category == 'transition_state' and regime == 'continuous_nonzero'").rmse_delta_mV.iloc[0])
    mixed_checks=[]
    for fold in FOLDS:
        p0mixed=float(regimes.query("model == 'P0_Original_Ridge' and fold == @fold and category == 'strict_q_state' and regime == '0<q<1'").rmse_delta_mV.iloc[0]); rsmixed=float(regimes.query("model == 'RS_Regime_Selective' and fold == @fold and category == 'strict_q_state' and regime == '0<q<1'").rmse_delta_mV.iloc[0]); mixed_checks.append(rsmixed<=p0mixed+EPS)
    h12_rs=float(horizon.query("model == 'RS_Regime_Selective' and fold == 'mean_3fold' and horizon_hours == 12.0").rmse_delta_mV.iloc[0]); h12_p0=float(horizon.query("model == 'P0_Original_Ridge' and fold == 'mean_3fold' and horizon_hours == 12.0").rmse_delta_mV.iloc[0]); p99rs=float(tails.query("model == 'RS_Regime_Selective'").p99_mV.iloc[0]); p99p0=float(tails.query("model == 'P0_Original_Ridge'").p99_mV.iloc[0]); selector_validation=selector_by_fold.query("stage == 'validation'"); selector_nonconstant=bool(((selector_validation.g1_fraction>0)&(selector_validation.g1_fraction<1)).all())
    rsviol=float(violation.query("model == 'RS_Regime_Selective' and fold == 'mean_3fold'").physical_violation_trajectory_rate.iloc[0]); p1viol=float(violation.query("model == 'P1_Physics_Projected_Ridge' and fold == 'mean_3fold'").physical_violation_trajectory_rate.iloc[0]); p2viol=float(violation.query("model == 'P2_Zero_Current_Aware' and fold == 'mean_3fold'").physical_violation_trajectory_rate.iloc[0])
    gate_checks=[("mean_rmse_improves_vs_p0_at_least_5pct",(p0mean-rs)/p0mean*100>=5),("mean_rmse_not_worse_than_p2_by_over_3pct",(rs-p2mean)/p2mean*100<=3),("fold3_overall_rs_no_worse_than_p0",fold3<=27.885315),("fold3_continuous_nonzero_better_than_p2",rsfold3continuous<p2fold3continuous),("mixed_current_each_fold_no_worse_than_p0",all(mixed_checks)),("12h_no_worse_than_p0_by_over_5pct",h12_rs<=h12_p0*1.05),("p99_no_worse_than_p0",p99rs<=p99p0),("physical_violation_no_worse_than_p1_p2",rsviol<=max(p1viol,p2viol)+EPS),("selector_nonconstant_in_each_validation_fold",selector_nonconstant)]
    gate={"status":"GATE_A_PASS" if all(x[1] for x in gate_checks) else "GATE_A_FAIL","passed":bool(all(x[1] for x in gate_checks)),"selector_nonconstant":selector_nonconstant}
    gate_frame=pd.DataFrame([{"criterion":x,"passed":bool(y)} for x,y in gate_checks]); gate_frame["gate_a_status"]=gate["status"]
    firewall={"target_rows_read_from_test":0,"test_predictions_constructed":False,"test_metrics_computed":False,"test_feature_diagnostics_computed":False,"test_performance_accessed":False}
    write_csv(selector_by_fold,"outputs/selector/selector_by_fold.csv"); write_csv(selector_by_horizon,"outputs/selector/selector_by_horizon.csv"); write_csv(selector_confusion_frame,"outputs/selector/selector_confusion_matrix.csv"); write_csv(selector_disagreement_frame,"outputs/selector/selector_input_disagreement.csv"); write_csv(violation,"outputs/frozen_model/physical_violation_metrics.csv"); write_csv(folds,"outputs/frozen_model/frozen_model_fold_metrics.csv"); write_csv(summary,"outputs/frozen_model/frozen_model_summary.csv"); write_csv(horizon,"outputs/horizon/frozen_model_per_horizon.csv"); write_csv(regimes,"outputs/regime/frozen_model_regime_metrics.csv"); write_csv(tails,"outputs/tail/frozen_model_tail_metrics.csv"); write_csv(bootstrap.query("block_length_origins == 48"),"outputs/bootstrap/bootstrap_main_metric.csv"); write_csv(bootstrap,"outputs/bootstrap/bootstrap_block_sensitivity.csv"); write_csv(pd.DataFrame(causal_rows),"outputs/lifecycle/causal_oof_audit.csv"); write_csv(gate_frame,"outputs/frozen_model/gate_a_audit.csv")
    notrun=pd.DataFrame([{ "status":"NOT_RUN_GATE_A_PENDING", "reason":"Retraining is forbidden unless frozen Gate A passes." }]); write_csv(notrun,"outputs/retraining/retraining_fold_metrics.csv"); write_csv(notrun,"outputs/retraining/retraining_summary.csv"); write_csv(pd.DataFrame([{ "status":"NOT_RUN_GATE_B_PENDING", "reason":"Three-seed evaluation is forbidden unless Gate B passes." }]),"outputs/multiseed/multiseed_summary.csv")
    make_figures(summary,horizon,selector_by_fold); write_reports(summary,folds,violation,bootstrap,gate,selector_by_fold,regimes,firewall)
    write_json({"status":"COMPLETED","gate_a":gate,"p0_reproduced":p0mean,"p2_reproduced":p2mean,"test_performance_accessed":False},"outputs/frozen_model/frozen_audit_status.json")
    print(json.dumps({"gate_a":gate,"P0":p0mean,"P2":p2mean,"RS":rs},ensure_ascii=False))


if __name__ == "__main__": main()

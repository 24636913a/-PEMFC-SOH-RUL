"""Feature-group and matched-origin lookback diagnostics for Phase N2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from phase_n2_common import PHASE_ROOT, FoldPreprocessor, build_windows, development_folds, load_config, load_development_data, metric_dict, write_csv, write_json
from phase_n2_models import train_model_run


def ridge_predict(train, val, alpha: float) -> np.ndarray:
    pre = FoldPreprocessor().fit(train.x)
    xt = pre.transform(train.x).reshape(len(train.x), -1)
    xv = pre.transform(val.x).reshape(len(val.x), -1)
    model = Ridge(alpha=alpha).fit(xt, train.y_delta)
    return val.anchor[:, None] + model.predict(xv)


def rule(candidate: list[float], reference: list[float], cfg: dict) -> dict:
    c, r = np.asarray(candidate), np.asarray(reference)
    improvement = 1.0 - float(c.mean() / r.mean())
    improved = int(np.sum(c < r))
    max_degradation = float(np.max((c - r) / r))
    passed = improvement >= float(cfg["selection"]["material_improvement_fraction"]) and improved >= int(cfg["selection"]["minimum_improved_folds"]) and max_degradation <= float(cfg["selection"]["maximum_single_fold_degradation_fraction"])
    return {"mean_improvement_fraction": improvement, "improved_fold_count": improved, "maximum_fold_degradation_fraction": max_degradation, "rule_pass": bool(passed)}


def moving_block_bootstrap(pair_by_fold: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], cfg: dict, comparison: str, category: str, candidate: str, reference: str) -> dict:
    rng = np.random.default_rng(int(cfg["bootstrap"]["seed"]) + sum(ord(c) for c in comparison))
    reps = int(cfg["bootstrap"]["repetitions"]); block = int(cfg["bootstrap"]["block_length_steps"])
    deltas = np.empty(reps)
    for b in range(reps):
        true_parts, cand_parts, ref_parts = [], [], []
        for yt, pc, pr in pair_by_fold.values():
            n = len(yt); indices = []
            max_start = max(1, n - block + 1)
            while len(indices) < n:
                start = int(rng.integers(0, max_start))
                indices.extend(range(start, min(start + block, n)))
            idx = np.asarray(indices[:n])
            true_parts.append(yt[idx]); cand_parts.append(pc[idx]); ref_parts.append(pr[idx])
        yt = np.concatenate(true_parts); pc = np.concatenate(cand_parts); pr = np.concatenate(ref_parts)
        deltas[b] = np.sqrt(np.mean((pc - yt) ** 2)) - np.sqrt(np.mean((pr - yt) ** 2))
    return {
        "category": category, "comparison": comparison, "candidate": candidate, "reference": reference,
        "delta_definition": "RMSE_candidate_minus_RMSE_reference_V", "bootstrap_repetitions": reps, "block_length_steps": block,
        "mean_delta_rmse_V": float(np.mean(deltas)), "median_delta_rmse_V": float(np.median(deltas)),
        "ci95_lower_V": float(np.quantile(deltas, .025)), "ci95_upper_V": float(np.quantile(deltas, .975)),
        "probability_candidate_better": float(np.mean(deltas < 0)),
    }


def main() -> None:
    cfg = load_config()
    gate_path = PHASE_ROOT / "outputs" / "frozen_config" / "baseline_gate.json"
    if not gate_path.is_file() or not json.loads(gate_path.read_text(encoding="utf-8")).get("baselines_complete"):
        raise RuntimeError("Baseline gate missing; deep diagnostics must not start")
    alpha = float(json.loads((PHASE_ROOT / "outputs" / "baseline_results" / "ridge_alpha_selection.json").read_text(encoding="utf-8"))["selected_ridge_alpha"])
    resolution = 0.5; lookback = 48; horizon = 24; seed = int(cfg["training"]["screening_seed"])
    all_features = cfg["feature_groups"]["F3"]
    frame, firewall = load_development_data(cfg, resolution, all_features)
    if firewall["target_rows_read_from_test"] != 0: raise RuntimeError("TEST FIREWALL VIOLATION")

    feature_rows: list[dict] = []; bootstrap_rows: list[dict] = []
    feature_predictions: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {"Ridge": {}, "LSTM": {}}
    for group in ["F1", "F2", "F3"]:
        features = cfg["feature_groups"][group]
        for model_name in ["Ridge", "LSTM"]:
            feature_predictions[model_name][group] = {}
        for spec in development_folds(cfg):
            fold = str(spec["fold"])
            train = build_windows(frame, features, lookback, horizon, float(spec["train_start_h"]), float(spec["train_end_h"]), resolution, cfg)
            val = build_windows(frame, features, lookback, horizon, float(spec["validation_start_h"]), float(spec["validation_end_h"]), resolution, cfg)
            ridge_pred = ridge_predict(train, val, alpha)
            m = metric_dict(val.y_abs, ridge_pred)
            feature_rows.append({"aggregation": "fold", "fold": fold, "reference_model": "Ridge", "feature_group": group, "feature_count": len(features), "origin_count": len(val.x), **m})
            feature_predictions["Ridge"][group][fold] = (val.y_abs, ridge_pred)
            run_id = f"feature_{group}_LSTM_{fold}_seed{seed}"
            result, arrays = train_model_run(run_id, "LSTM", train, val, cfg["architectures"]["LSTM"], cfg["training"], seed, {"stage": "feature_group", "fold": fold, "feature_group": group, "feature_count": len(features), "lookback_hours": 24, "horizon_hours": 12, "resolution_hours": resolution})
            feature_rows.append({"aggregation": "fold", "fold": fold, "reference_model": "LSTM", "feature_group": group, "feature_count": len(features), "origin_count": len(val.x), **{k: result[k] for k in ["rmse_V", "rmse_mV", "mae_V", "mae_mV", "bias_V", "bias_mV", "r2"]}})
            feature_predictions["LSTM"][group][fold] = (arrays["y_true"], arrays["y_pred"])

    feature_df = pd.DataFrame(feature_rows)
    means = feature_df.groupby(["reference_model", "feature_group", "feature_count"], as_index=False).agg(rmse_V=("rmse_V", "mean"), rmse_mV=("rmse_mV", "mean"), mae_V=("mae_V", "mean"), mae_mV=("mae_mV", "mean"), bias_V=("bias_V", "mean"), bias_mV=("bias_mV", "mean"), r2=("r2", "mean"), origin_count=("origin_count", "sum"))
    means.insert(0, "fold", "mean_3fold"); means.insert(0, "aggregation", "mean")
    feature_df = pd.concat([feature_df, means], ignore_index=True)

    feature_rules: dict[str, dict[str, dict]] = {}
    for model in ["Ridge", "LSTM"]:
        feature_rules[model] = {}
        for candidate, reference in [("F2", "F1"), ("F3", "F2")]:
            c = feature_df[(feature_df.aggregation == "fold") & (feature_df.reference_model == model) & (feature_df.feature_group == candidate)].sort_values("fold")["rmse_V"].tolist()
            r = feature_df[(feature_df.aggregation == "fold") & (feature_df.reference_model == model) & (feature_df.feature_group == reference)].sort_values("fold")["rmse_V"].tolist()
            feature_rules[model][f"{candidate}_vs_{reference}"] = rule(c, r, cfg)
            pairs = {}
            for fold in feature_predictions[model][candidate]:
                yt, pc = feature_predictions[model][candidate][fold]; ytr, pr = feature_predictions[model][reference][fold]
                if not np.array_equal(yt, ytr): raise RuntimeError("Feature-group target mismatch")
                pairs[fold] = (yt, pc, pr)
            bootstrap_rows.append(moving_block_bootstrap(pairs, cfg, f"{model}:{candidate}_vs_{reference}", "feature_group", candidate, reference))

    f2_pass = all(feature_rules[m]["F2_vs_F1"]["rule_pass"] for m in ["Ridge", "LSTM"])
    f2_disagree = feature_rules["Ridge"]["F2_vs_F1"]["rule_pass"] != feature_rules["LSTM"]["F2_vs_F1"]["rule_pass"]
    selected_group = "F2" if f2_pass else "F1"
    f3_pass = all(feature_rules[m]["F3_vs_F2"]["rule_pass"] for m in ["Ridge", "LSTM"])
    f3_disagree = feature_rules["Ridge"]["F3_vs_F2"]["rule_pass"] != feature_rules["LSTM"]["F3_vs_F2"]["rule_pass"]
    if selected_group == "F2" and f3_pass: selected_group = "F3"
    feature_status = "resolved" if not (f2_disagree or (selected_group == "F2" and f3_disagree)) else "feature contribution unresolved"

    # Matched-origin lookback comparison using the selected feature group.
    features = cfg["feature_groups"][selected_group]
    lookback_rows: list[dict] = []; audit_rows: list[dict] = []
    lookback_predictions: dict[str, dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]] = {"Ridge": {12: {}, 24: {}, 48: {}}, "LSTM": {12: {}, 24: {}, 48: {}}}
    for spec in development_folds(cfg):
        fold = str(spec["fold"])
        max_train = build_windows(frame, features, 96, horizon, float(spec["train_start_h"]), float(spec["train_end_h"]), resolution, cfg)
        max_val = build_windows(frame, features, 96, horizon, float(spec["validation_start_h"]), float(spec["validation_end_h"]), resolution, cfg)
        sets = {}
        for hours, steps in [(12, 24), (24, 48), (48, 96)]:
            train = build_windows(frame, features, steps, horizon, float(spec["train_start_h"]), float(spec["train_end_h"]), resolution, cfg, set(max_train.row_origin_index))
            val = build_windows(frame, features, steps, horizon, float(spec["validation_start_h"]), float(spec["validation_end_h"]), resolution, cfg, set(max_val.row_origin_index))
            sets[hours] = (train, val)
            audit_rows.append({
                "fold": fold, "lookback_hours": hours, "train_origin_count": len(train.x), "validation_origin_count": len(val.x),
                "train_origins_match_48h": bool(np.array_equal(train.row_origin_index, max_train.row_origin_index)),
                "validation_origins_match_48h": bool(np.array_equal(val.row_origin_index, max_val.row_origin_index)),
                "validation_targets_match_48h": bool(np.array_equal(val.y_delta, max_val.y_delta)),
                "validation_anchor_match_48h": bool(np.array_equal(val.anchor, max_val.anchor)),
                "validation_target_end_match_48h": bool(np.array_equal(val.target_end_age_h, max_val.target_end_age_h)),
            })
            rp = ridge_predict(train, val, alpha); rm = metric_dict(val.y_abs, rp)
            lookback_rows.append({"aggregation": "fold", "fold": fold, "reference_model": "Ridge", "lookback_hours": hours, "origin_count": len(val.x), **rm})
            lookback_predictions["Ridge"][hours][fold] = (val.y_abs, rp)
            run_id = f"lookback_{hours}h_LSTM_{selected_group}_{fold}_seed{seed}"
            result, arrays = train_model_run(run_id, "LSTM", train, val, cfg["architectures"]["LSTM"], cfg["training"], seed, {"stage": "lookback", "fold": fold, "feature_group": selected_group, "lookback_hours": hours, "horizon_hours": 12, "resolution_hours": resolution, "matched_origin": True})
            lookback_rows.append({"aggregation": "fold", "fold": fold, "reference_model": "LSTM", "lookback_hours": hours, "origin_count": len(val.x), **{k: result[k] for k in ["rmse_V", "rmse_mV", "mae_V", "mae_mV", "bias_V", "bias_mV", "r2"]}})
            lookback_predictions["LSTM"][hours][fold] = (arrays["y_true"], arrays["y_pred"])

    audit = pd.DataFrame(audit_rows)
    audit_cols = [c for c in audit.columns if c.endswith("_48h")]
    matched_pass = bool(audit[audit_cols].all().all())
    write_csv(audit, PHASE_ROOT / "outputs" / "lookback_diagnostics" / "matched_origin_audit.csv")
    if not matched_pass: raise RuntimeError("Matched-Origin audit failed")
    lookback_df = pd.DataFrame(lookback_rows)
    lbmeans = lookback_df.groupby(["reference_model", "lookback_hours"], as_index=False).agg(rmse_V=("rmse_V", "mean"), rmse_mV=("rmse_mV", "mean"), mae_V=("mae_V", "mean"), mae_mV=("mae_mV", "mean"), bias_V=("bias_V", "mean"), bias_mV=("bias_mV", "mean"), r2=("r2", "mean"), origin_count=("origin_count", "sum"))
    lbmeans.insert(0, "fold", "mean_3fold"); lbmeans.insert(0, "aggregation", "mean")
    lookback_df = pd.concat([lookback_df, lbmeans], ignore_index=True)

    lookback_rules: dict[str, dict[str, dict]] = {}
    bootstrap_lookup: dict[str, dict] = {}
    for model in ["Ridge", "LSTM"]:
        lookback_rules[model] = {}
        for candidate in [12, 48]:
            c = lookback_df[(lookback_df.aggregation == "fold") & (lookback_df.reference_model == model) & (lookback_df.lookback_hours == candidate)].sort_values("fold")["rmse_V"].tolist()
            r = lookback_df[(lookback_df.aggregation == "fold") & (lookback_df.reference_model == model) & (lookback_df.lookback_hours == 24)].sort_values("fold")["rmse_V"].tolist()
            lookback_rules[model][f"{candidate}_vs_24"] = rule(c, r, cfg)
            pairs = {}
            for fold in lookback_predictions[model][candidate]:
                yt, pc = lookback_predictions[model][candidate][fold]; ytr, pr = lookback_predictions[model][24][fold]
                if not np.array_equal(yt, ytr): raise RuntimeError("Lookback target mismatch")
                pairs[fold] = (yt, pc, pr)
            boot = moving_block_bootstrap(pairs, cfg, f"{model}:{candidate}h_vs_24h", "lookback", f"{candidate}h", "24h")
            bootstrap_rows.append(boot); bootstrap_lookup[f"{model}:{candidate}"] = boot

    selected_lookback = 24
    for candidate in [48, 12]:
        both_rule = all(lookback_rules[m][f"{candidate}_vs_24"]["rule_pass"] for m in ["Ridge", "LSTM"])
        both_boot = all(bootstrap_lookup[f"{m}:{candidate}"]["ci95_upper_V"] < 0 for m in ["Ridge", "LSTM"])
        if both_rule and both_boot:
            selected_lookback = candidate
            break

    write_csv(feature_df, PHASE_ROOT / "outputs" / "feature_group_diagnostics" / "feature_group_comparison.csv")
    write_csv(lookback_df, PHASE_ROOT / "outputs" / "lookback_diagnostics" / "lookback_matched_origin_comparison.csv")
    write_csv(pd.DataFrame(bootstrap_rows), PHASE_ROOT / "outputs" / "statistical_tests" / "bootstrap_pairwise_comparison.csv")
    selection = {
        "status": "PASS", "test_predictions_constructed": False, "selected_feature_group": selected_group,
        "feature_status": feature_status, "feature_rules": feature_rules, "selected_lookback_hours": selected_lookback,
        "lookback_rules": lookback_rules, "matched_origin_audit_pass": matched_pass,
    }
    write_json(selection, PHASE_ROOT / "outputs" / "frozen_config" / "diagnostic_selection.json")

    figdir = PHASE_ROOT / "outputs" / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    p = means.copy();
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = ["F1", "F2", "F3"]; x = np.arange(3); width=.35
    for j, model in enumerate(["Ridge", "LSTM"]):
        vals = [float(p[(p.reference_model == model) & (p.feature_group == g)].rmse_mV.iloc[0]) for g in groups]
        ax.bar(x + (j-.5)*width, vals, width, label=model)
    ax.set_xticks(x, groups); ax.set_ylabel("Validation RMSE (mV)"); ax.set_title("Incremental value of feature groups (development only)"); ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(figdir / "feature_group_incremental_value.png", dpi=180); plt.close(fig)
    p = lbmeans.copy(); fig, ax = plt.subplots(figsize=(9, 5))
    lbs=[12,24,48]; x=np.arange(3)
    for j, model in enumerate(["Ridge", "LSTM"]):
        vals=[float(p[(p.reference_model==model)&(p.lookback_hours==h)].rmse_mV.iloc[0]) for h in lbs]
        ax.bar(x+(j-.5)*width, vals, width, label=model)
    ax.set_xticks(x,[f"{h} h" for h in lbs]); ax.set_ylabel("Matched-origin Validation RMSE (mV)"); ax.set_title("Lookback comparison with identical forecast origins"); ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(figdir / "lookback_12_24_48h_comparison.png", dpi=180); plt.close(fig)
    print(json.dumps(selection, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

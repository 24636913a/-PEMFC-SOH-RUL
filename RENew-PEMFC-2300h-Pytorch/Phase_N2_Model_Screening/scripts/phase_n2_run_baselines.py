"""Run Phase N2 development folds and strong baselines before deep learning."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from phase_n2_common import (
    PHASE_ROOT,
    FoldPreprocessor,
    build_windows,
    development_folds,
    load_config,
    load_development_data,
    metric_dict,
    per_horizon_metrics,
    write_csv,
    write_json,
)


def baseline_predictions(windows, cycle_steps: int) -> dict[str, np.ndarray]:
    h = windows.y_abs.shape[1]
    persistence = np.repeat(windows.anchor[:, None], h, axis=1)
    x_voltage = windows.x[:, :, 0].astype(float)
    t_hist = np.arange(x_voltage.shape[1], dtype=float)
    t_future = np.arange(x_voltage.shape[1], x_voltage.shape[1] + h, dtype=float)
    linear = np.empty_like(windows.y_abs, dtype=float)
    cycle = np.empty_like(windows.y_abs, dtype=float)
    for i, history in enumerate(x_voltage):
        slope, intercept = np.polyfit(t_hist, history, 1)
        linear[i] = intercept + slope * t_future
        period = history[-cycle_steps:]
        cycle[i] = period[np.arange(h) % cycle_steps]
    return {"Persistence": persistence, "LocalLinear": linear, "CycleRepeat": cycle}


def add_skills(rows: list[dict], fold_best: dict[str, float]) -> None:
    for row in rows:
        if row.get("aggregation") != "fold":
            continue
        fold = row["fold"]
        denom_p = fold_best[fold + "|Persistence"]
        denom_c = fold_best[fold + "|CycleRepeat"]
        denom_b = fold_best[fold + "|best"]
        row["skill_persistence"] = 1.0 - row["rmse_V"] / denom_p
        row["skill_cycle"] = 1.0 - row["rmse_V"] / denom_c
        row["skill_best"] = 1.0 - row["rmse_V"] / denom_b


def main() -> None:
    cfg = load_config()
    features = cfg["feature_groups"]["F1"]
    resolution = float(cfg["task"]["primary_resolution_hours"])
    lookback = round(cfg["task"]["default_lookback_hours"] / resolution)
    horizon = round(cfg["task"]["primary_horizon_hours"] / resolution)
    cycle_steps = int(cfg["task"]["cycle_period_steps_30min"])
    frame, firewall = load_development_data(cfg, resolution, features)
    write_json(firewall, PHASE_ROOT / "outputs" / "frozen_config" / "test_firewall_audit.json")
    if firewall["target_rows_read_from_test"] != 0:
        raise RuntimeError("TEST FIREWALL VIOLATION")

    fold_manifest: list[dict] = []
    fold_predictions: dict[str, dict[str, np.ndarray]] = {}
    fold_true: dict[str, np.ndarray] = {}
    fold_meta: dict[str, dict[str, np.ndarray]] = {}
    ridge_candidates: dict[float, list[float]] = {float(a): [] for a in cfg["baselines"]["ridge_alphas"]}

    for spec in development_folds(cfg):
        fold = str(spec["fold"])
        train = build_windows(frame, features, lookback, horizon, float(spec["train_start_h"]), float(spec["train_end_h"]), resolution, cfg)
        val = build_windows(frame, features, lookback, horizon, float(spec["validation_start_h"]), float(spec["validation_end_h"]), resolution, cfg)
        preds = baseline_predictions(val, cycle_steps)
        pre = FoldPreprocessor().fit(train.x)
        x_train = pre.transform(train.x).reshape(len(train.x), -1)
        x_val = pre.transform(val.x).reshape(len(val.x), -1)
        for alpha in ridge_candidates:
            ridge = Ridge(alpha=alpha)
            ridge.fit(x_train, train.y_delta)
            pred = val.anchor[:, None] + ridge.predict(x_val)
            ridge_candidates[alpha].append(metric_dict(val.y_abs, pred)["rmse_V"])
            preds[f"Ridge_alpha_{alpha:g}"] = pred
        fold_predictions[fold] = preds
        fold_true[fold] = val.y_abs
        fold_meta[fold] = {
            "origin_age_h": val.origin_age_h,
            "origin_datetime": val.origin_datetime,
            "target_end_age_h": val.target_end_age_h,
            "anchor": val.anchor,
        }
        for role, start, end, windows in [
            ("train", float(spec["train_start_h"]), float(spec["train_end_h"]), train),
            ("validation", float(spec["validation_start_h"]), float(spec["validation_end_h"]), val),
        ]:
            subset = frame[(frame[cfg["data"]["bin_start_column"]] >= start - 1e-9) & (frame[cfg["data"]["bin_end_column"]] <= end + 1e-9)]
            fold_manifest.append({
                "fold": fold,
                "role": role,
                "ageing_start_h": start,
                "ageing_end_h": end,
                "datetime_start": subset[cfg["data"]["datetime_column"]].min(),
                "datetime_end": subset[cfg["data"]["datetime_column"]].max(),
                "bin_count": len(subset),
                "legal_window_count_24h_to_12h": len(windows.x),
                "gap_affected_bin_count": int(subset["gap_before_flag"].astype(bool).sum()) if "gap_before_flag" in subset else np.nan,
                "mean_coverage_ratio": float(subset["coverage_ratio"].mean()),
                "valid_bin_fraction": float(subset[cfg["data"]["valid_column"]].astype(bool).mean()),
                "scaler_fit_policy": "this fold Train only" if role == "train" else "fitted on corresponding fold Train",
            })

    selected_alpha = min(ridge_candidates, key=lambda a: float(np.mean(ridge_candidates[a])))
    summary_rows: list[dict] = []
    per_h_rows: list[dict] = []
    prediction_records: list[dict] = []
    models = ["Persistence", "LocalLinear", "CycleRepeat", f"Ridge_alpha_{selected_alpha:g}"]
    display = {models[-1]: "Ridge"}
    fold_best: dict[str, float] = {}

    for fold in fold_predictions:
        yt = fold_true[fold]
        for model in models:
            name = display.get(model, model)
            yp = fold_predictions[fold][model]
            metrics = metric_dict(yt, yp)
            summary_rows.append({"aggregation": "fold", "fold": fold, "model": name, "ridge_alpha": selected_alpha if name == "Ridge" else np.nan, "validation_origin_count": len(yt), **metrics})
            for row in per_horizon_metrics(yt, yp, resolution):
                per_h_rows.append({"fold": fold, "model": name, **row})
            meta = fold_meta[fold]
            for i in range(len(yt)):
                for h in range(horizon):
                    prediction_records.append({
                        "fold": fold,
                        "model": name,
                        "origin_age_h": meta["origin_age_h"][i],
                        "origin_datetime": meta["origin_datetime"][i],
                        "horizon_step": h + 1,
                        "horizon_hours": (h + 1) * resolution,
                        "target_age_h": meta["origin_age_h"][i] + (h + 1) * resolution,
                        "anchor_voltage_V": meta["anchor"][i],
                        "actual_voltage_V": yt[i, h],
                        "predicted_voltage_V": yp[i, h],
                        "error_V": yp[i, h] - yt[i, h],
                    })
        rmse_map = {display.get(m, m): metric_dict(yt, fold_predictions[fold][m])["rmse_V"] for m in models}
        for name, value in rmse_map.items():
            fold_best[fold + "|" + name] = value
        fold_best[fold + "|best"] = min(rmse_map.values())

    add_skills(summary_rows, fold_best)
    per_h = pd.DataFrame(per_h_rows)
    # Per-horizon skills use the strongest simple baseline at each fold and step.
    for (fold, step), group in per_h.groupby(["fold", "horizon_step"]):
        best = group["rmse_V"].min()
        p = float(group.loc[group["model"] == "Persistence", "rmse_V"].iloc[0])
        c = float(group.loc[group["model"] == "CycleRepeat", "rmse_V"].iloc[0])
        idx = group.index
        per_h.loc[idx, "skill_persistence"] = 1.0 - per_h.loc[idx, "rmse_V"] / p
        per_h.loc[idx, "skill_cycle"] = 1.0 - per_h.loc[idx, "rmse_V"] / c
        per_h.loc[idx, "skill_best"] = 1.0 - per_h.loc[idx, "rmse_V"] / best

    fold_df = pd.DataFrame(summary_rows)
    overall = fold_df.groupby("model", as_index=False).agg(
        rmse_V=("rmse_V", "mean"), rmse_mV=("rmse_mV", "mean"), mae_V=("mae_V", "mean"), mae_mV=("mae_mV", "mean"),
        bias_V=("bias_V", "mean"), bias_mV=("bias_mV", "mean"), r2=("r2", "mean"),
        skill_persistence=("skill_persistence", "mean"), skill_cycle=("skill_cycle", "mean"), skill_best=("skill_best", "mean"),
        validation_origin_count=("validation_origin_count", "sum"),
    )
    overall.insert(0, "fold", "mean_3fold")
    overall.insert(0, "aggregation", "mean")
    overall["ridge_alpha"] = np.where(overall["model"].eq("Ridge"), selected_alpha, np.nan)
    baseline_summary = pd.concat([fold_df, overall], ignore_index=True, sort=False)

    out = PHASE_ROOT / "outputs" / "baseline_results"
    write_csv(pd.DataFrame(fold_manifest), PHASE_ROOT / "outputs" / "frozen_config" / "development_fold_manifest.csv")
    write_csv(baseline_summary, out / "baseline_summary.csv")
    write_csv(per_h, out / "baseline_per_horizon_metrics.csv")
    write_csv(pd.DataFrame(prediction_records), out / "baseline_validation_predictions.csv")
    write_json({"selected_ridge_alpha": selected_alpha, "candidate_fold_rmse_V": ridge_candidates, "selection_scope": "development folds only"}, out / "ridge_alpha_selection.json")

    figdir = PHASE_ROOT / "outputs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plot = overall.sort_values("rmse_mV")
    plt.figure(figsize=(8, 5))
    plt.bar(plot["model"], plot["rmse_mV"], color=["#2563EB", "#6B7280", "#F59E0B", "#10B981"])
    plt.ylabel("Validation RMSE (mV)"); plt.title("Strong baselines: three-fold mean RMSE"); plt.grid(axis="y", alpha=.25); plt.tight_layout()
    plt.savefig(figdir / "baseline_rmse_comparison.png", dpi=180); plt.close()
    skill = overall.sort_values("skill_cycle", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(skill["model"], skill["skill_cycle"], color="#2563EB")
    plt.axhline(0, color="black", linewidth=1); plt.ylabel("Skill_Cycle"); plt.title("Skill relative to Cycle Repeat baseline"); plt.grid(axis="y", alpha=.25); plt.tight_layout()
    plt.savefig(figdir / "baseline_skill_comparison.png", dpi=180); plt.close()

    marker = {"status": "PASS", "baselines_complete": True, "test_predictions_constructed": False, "selected_ridge_alpha": selected_alpha, "models": ["Persistence", "LocalLinear", "CycleRepeat", "Ridge"]}
    write_json(marker, PHASE_ROOT / "outputs" / "frozen_config" / "baseline_gate.json")
    print(json.dumps(marker, ensure_ascii=False))


if __name__ == "__main__":
    main()

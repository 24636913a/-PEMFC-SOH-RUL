#!/usr/bin/env python
"""Conditional G4 retraining under the frozen Phase G-STOP protocol."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PHASE_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_PATH = Path(__file__).resolve().parent / "phase_g_stop_experiment.py"
FOLDS = ("fold_1", "fold_2", "fold_3")


def load_experiment():
    spec = importlib.util.spec_from_file_location("phase_g_stop_experiment_module", EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import Phase G-STOP experiment helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    path = PHASE_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(obj: Any, relative: str) -> None:
    path = PHASE_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def p2_training_future(p: dict[str, Any], q_oof: np.ndarray, aux_oof: np.ndarray, exp) -> np.ndarray:
    candidate = p["z_oof"].copy()
    mu_on = np.maximum(aux_oof[:, :, 0], 0.0)
    sigma_on = np.maximum(aux_oof[:, :, 1], 0.0)
    candidate[:, :, 8] = q_oof
    candidate[:, :, 0] = (1.0 - q_oof) * mu_on
    candidate[:, :, 1] = np.sqrt(np.maximum(0.0, (1.0 - q_oof) * sigma_on**2 + q_oof * (1.0 - q_oof) * mu_on**2))
    projected, _ = exp.project_future(candidate, p["area"])
    return projected


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), group in metrics.groupby(["model", "seed"], sort=False):
        row = {"model": model, "seed": int(seed)}
        for name in ("r2_delta", "rmse_delta_mV", "mae_delta_mV", "p95_mV", "p99_mV"):
            row[f"mean_{name}"] = float(group[name].mean())
        for fold in FOLDS:
            row[f"{fold}_rmse_delta_mV"] = float(group.loc[group["fold"] == fold, "rmse_delta_mV"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    exp = load_experiment()
    cfg = exp.load_config()
    g3 = json.loads((PHASE_ROOT / "outputs" / "frozen_timexer" / "g3_result.json").read_text(encoding="utf-8"))
    if not g3.get("gate_b_passed") or g3.get("gate_b_winner") != "P2_Zero_Current_Aware":
        write_json({"status": "NOT_RUN_GATE_B_FAILED"}, "outputs/frozen_timexer/g4_result.json")
        print("G4 NOT_RUN_GATE_B_FAILED")
        return

    n28f = exp.import_n28f(Path(cfg["n2_8f_root"]))
    ncfg = n28f.load_config()
    common, frame, windows, _ = n28f.load_development(ncfg)
    raw_aux = pd.read_csv(PHASE_ROOT / "outputs" / "physical_audit" / "raw_conditional_current_statistics.csv")
    frame = frame.merge(raw_aux[["bin_start_ophrs2_h", "mu_on_A", "sigma_on_A"]], on="bin_start_ophrs2_h", how="left", validate="one_to_one")
    packages = {fold: n28f.package_fold(common, frame, windows, ncfg, fold) for fold in FOLDS}
    packs = {}
    for fold in FOLDS:
        p = packages[fold]
        aux_train = exp.future_aux(frame, p["train"], ["mu_on_A", "sigma_on_A"])
        aux_valid = exp.future_aux(frame, p["valid"], ["mu_on_A", "sigma_on_A"])
        q_oof, aux_oof, p2_keep, _ = exp.causal_p2_oof(p, aux_train, cfg, n28f, fold)
        if not np.array_equal(p2_keep, p["ridge_keep"]):
            raise RuntimeError("G4 P2 origin mismatch")
        p2_train = p2_training_future(p, q_oof, aux_oof, exp)
        p2_valid, _, _ = exp.build_p2(p, aux_train, aux_valid, cfg)
        packs[fold] = {
            "P0_Retrained": (p["z_oof"], p["z_ridge"]),
            "P2_Retrained": (p2_train, p2_valid),
        }

    # Redirect N2.8F's checkpoint writer into the new phase before training.
    n28f.ROOT = PHASE_ROOT
    metric_rows = []
    history_rows = []

    def run_seed(seed: int) -> None:
        for label in ("P0_Retrained", "P2_Retrained"):
            for fold in FOLDS:
                p = packages[fold]
                keep = p["fair_keep"]
                z_train, z_valid = packs[fold][label]
                if not np.isfinite(z_train[keep]).all():
                    raise RuntimeError(f"{label}/{fold}: non-finite OOF future conditions")
                n28f.set_seed(seed)
                model = n28f.StepAlignedTimeXer(ncfg, caf=False)
                model, pred_scaled, _, info, history = n28f.train_model(
                    model,
                    p["xtr"][keep], p["ytr"][keep], p["z_scaler"].transform(z_train[keep]), None,
                    p["xva"], p["yva"], p["z_scaler"].transform(z_valid), None,
                    ncfg, label, fold, seed, None,
                )
                pred = p["valid"].anchor[:, None] + pred_scaled * p["ystd"] + p["ymean"]
                metrics = exp.voltage_metrics(p["valid"].y_abs, pred, p["valid"].anchor)
                metric_rows.append({"model": label, "fold": fold, "seed": seed, "fair_train_windows": int(keep.sum()), **metrics, **info})
                for item in history:
                    history_rows.append({"model": label, "fold": fold, "seed": seed, **item})
                np.savez_compressed(
                    PHASE_ROOT / "outputs" / "frozen_timexer" / f"G4_{label}_{fold}_seed{seed}_predictions.npz",
                    y_true=p["valid"].y_abs.astype(float), y_pred=pred.astype(float), anchor=p["valid"].anchor.astype(float),
                    row_origin_index=p["valid"].row_origin_index.astype(int), origin_age_h=p["valid"].origin_age_h.astype(float),
                )

    run_seed(42)
    seed42 = summarize(pd.DataFrame(metric_rows))
    p0 = seed42.loc[(seed42["model"] == "P0_Retrained") & (seed42["seed"] == 42)].iloc[0]
    p2 = seed42.loc[(seed42["model"] == "P2_Retrained") & (seed42["seed"] == 42)].iloc[0]
    improvement = (p0["mean_rmse_delta_mV"] - p2["mean_rmse_delta_mV"]) / p0["mean_rmse_delta_mV"] * 100.0
    improved_folds = sum(p2[f"{fold}_rmse_delta_mV"] < p0[f"{fold}_rmse_delta_mV"] for fold in FOLDS)
    fold3_worsening = (p2["fold_3_rmse_delta_mV"] - p0["fold_3_rmse_delta_mV"]) / p0["fold_3_rmse_delta_mV"] * 100.0
    p95_worsening = (p2["mean_p95_mV"] - p0["mean_p95_mV"]) / p0["mean_p95_mV"] * 100.0
    p99_worsening = (p2["mean_p99_mV"] - p0["mean_p99_mV"]) / p0["mean_p99_mV"] * 100.0
    trigger = cfg["g4_multiseed_trigger"]
    multiseed = bool(
        improvement >= float(trigger["minimum_seed42_rmse_improvement_pct"])
        and improved_folds >= int(trigger["minimum_improved_folds"])
        and fold3_worsening <= float(trigger["maximum_fold3_worsening_pct"])
        and p95_worsening <= float(trigger["maximum_p95_worsening_pct"])
        and p99_worsening <= float(trigger["maximum_p99_worsening_pct"])
    )
    if multiseed:
        for seed in (2024, 3407):
            run_seed(seed)

    metrics = pd.DataFrame(metric_rows)
    summary = summarize(metrics)
    write_csv(metrics, "outputs/frozen_timexer/g4_retraining_fold_metrics.csv")
    write_csv(summary, "outputs/frozen_timexer/g4_retraining_summary.csv")
    write_csv(pd.DataFrame(history_rows), "logs/g4_training_history.csv")
    result = {
        "status": "COMPLETED",
        "seed42_rmse_improvement_pct": float(improvement),
        "seed42_improved_fold_count": int(improved_folds),
        "seed42_fold3_worsening_pct": float(fold3_worsening),
        "seed42_p95_worsening_pct": float(p95_worsening),
        "seed42_p99_worsening_pct": float(p99_worsening),
        "multiseed_trigger_passed": multiseed,
        "seeds_run": sorted(metrics["seed"].unique().astype(int).tolist()),
        "p2_surpassed_fair_p0": bool(improvement > 0.0),
        "test_performance_accessed": False,
    }
    write_json(result, "outputs/frozen_timexer/g4_result.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

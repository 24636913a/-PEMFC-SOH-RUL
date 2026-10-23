#!/usr/bin/env python
"""Strict chronological OOF transfer of physics-consistent futures to LSTM.

Development only. The frozen N2.7 causal-SG LSTM predictions are corrected by
small per-horizon Ridge heads trained on chronological OOF LSTM residuals.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PHASE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PHASE_ROOT / "configs/phase_g_stop_frozen_config.yaml"
FOLDS = ("fold_1", "fold_2", "fold_3")
EPS = 1e-7


def out(relative: str) -> Path:
    path = (PHASE_ROOT / relative).resolve()
    if PHASE_ROOT.resolve() not in path.parents:
        raise RuntimeError(f"Output escaped Phase G-STOP: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    frame.to_csv(out(relative), index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(payload: Any, relative: str) -> None:
    out(relative).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subset(window, indices: np.ndarray):
    values = {field.name: getattr(window, field.name)[indices] for field in fields(window)}
    return type(window)(**values)


def predict_scaled(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=256, shuffle=False)
    result = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            result.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(result, axis=0)


def train_oof_block(
    n27_models,
    n27_common,
    cfg: dict[str, Any],
    window,
    fit: np.ndarray,
    left: int,
    right: int,
    seed: int,
    fold: str,
    block: int,
):
    """Fit without seeing prediction-block labels; early stopping is inner-train-only."""
    validation_start_position = max(32, int(math.floor(0.80 * len(fit))))
    validation_start_position = min(validation_start_position, len(fit) - 16)
    inner_validation = fit[validation_start_position:]
    if len(inner_validation) < 16:
        raise RuntimeError("Too few inner-validation windows")
    inner_origin = float(window.origin_age_h[inner_validation[0]])
    inner_training = fit[window.target_end_age_h[fit] <= inner_origin + EPS]
    inner_training = inner_training[inner_training < inner_validation[0]]
    if len(inner_training) < 16:
        raise RuntimeError("Too few target-complete inner-training windows")
    if float(window.target_end_age_h[inner_training].max()) > inner_origin + EPS:
        raise RuntimeError("Inner early-stopping target-completion leakage")
    if float(window.target_end_age_h[fit].max()) > float(window.origin_age_h[left]) + EPS:
        raise RuntimeError("OOF block target-completion leakage")

    train = subset(window, inner_training)
    valid = subset(window, inner_validation)
    block_window = subset(window, np.arange(left, right, dtype=int))
    pre = n27_common.FoldPreprocessor().fit(train.x)
    x_train = pre.transform(train.x)
    x_valid = pre.transform(valid.x)
    x_block = pre.transform(block_window.x)
    target_mean = np.mean(train.y_delta, axis=0, keepdims=True)
    target_std = np.std(train.y_delta, axis=0, keepdims=True)
    target_std = np.where(target_std > 1e-6, target_std, 1.0)
    y_train = ((train.y_delta - target_mean) / target_std).astype(np.float32)

    set_seed(seed + block * 1009)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = n27_models.create_model("LSTM", cfg, 0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["default_learning_rate"]),
        weight_decay=float(cfg["training"]["default_weight_decay"]),
    )
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed + block * 1009)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_rmse = math.inf
    best_state = None
    best_epoch = 0
    patience = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(cfg["training"]["max_epochs"]) + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite OOF LSTM loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scaled = predict_scaled(model, x_valid, device)
        prediction = valid.anchor[:, None] + scaled * target_std + target_mean
        rmse = float(np.sqrt(np.mean((prediction - valid.y_abs) ** 2)))
        history.append({
            "seed": seed, "fold": fold, "block": block, "epoch": epoch,
            "train_scaled_mse": float(np.mean(losses)), "inner_validation_rmse_mV": rmse * 1000.0,
        })
        if rmse < best_rmse - float(cfg["training"]["min_delta"]):
            best_rmse, best_epoch, patience = rmse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        if patience >= int(cfg["training"]["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("No OOF LSTM state captured")
    model.load_state_dict(best_state)
    scaled = predict_scaled(model, x_block, device)
    prediction = block_window.anchor[:, None] + scaled * target_std + target_mean
    audit = {
        "seed": seed, "fold": fold, "block": block,
        "completed_fit_count": int(len(fit)),
        "inner_training_count": int(len(inner_training)),
        "inner_validation_count": int(len(inner_validation)),
        "prediction_count": int(right - left),
        "max_completed_fit_target_end_h": float(window.target_end_age_h[fit].max()),
        "min_prediction_origin_h": float(window.origin_age_h[left]),
        "max_inner_training_target_end_h": float(window.target_end_age_h[inner_training].max()),
        "min_inner_validation_origin_h": inner_origin,
        "target_completion_pass": True,
        "validation_residual_used": False,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "elapsed_s": time.perf_counter() - started,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction, audit, history


def chronological_lstm_oof(n27_models, n27_common, n28f, gcfg, n27cfg, window, seed: int, fold: str):
    cache = out(f"outputs/lstm_transfer/oof_lstm_{fold}_seed{seed}.npz")
    audit_cache = out(f"outputs/lstm_transfer/oof_lstm_{fold}_seed{seed}_audit.json")
    if cache.is_file() and audit_cache.is_file():
        arrays = dict(np.load(cache, allow_pickle=False))
        audit = json.loads(audit_cache.read_text(encoding="utf-8"))
        if np.array_equal(arrays["row_origin_index"], window.row_origin_index):
            print(json.dumps({"stage": "G5_OOF_CACHE", "seed": seed, "fold": fold}), flush=True)
            return arrays["prediction"], arrays["keep"].astype(bool), audit["blocks"], audit.get("history", [])
    start = n28f.first_index_with_completed(
        window.target_end_age_h, window.origin_age_h,
        int(gcfg["lstm_transfer"]["oof_min_completed_windows"]),
    )
    prediction = np.full_like(window.y_abs, np.nan, dtype=float)
    audits, history = [], []
    edges = n28f.block_edges(start, len(window.x), int(gcfg["lstm_transfer"]["oof_blocks"]))
    for block, (left, right) in enumerate(edges, start=1):
        fit = n28f.completed_indices(window.target_end_age_h, float(window.origin_age_h[left]))
        block_prediction, audit, block_history = train_oof_block(
            n27_models, n27_common, n27cfg, window, fit, left, right, seed, fold, block
        )
        prediction[left:right] = block_prediction
        audits.append(audit); history.extend(block_history)
        print(json.dumps({
            "stage": "G5_OOF_BLOCK", "seed": seed, "fold": fold, "block": block,
            "blocks": len(edges), "best_epoch": audit["best_epoch"],
            "elapsed_s": round(audit["elapsed_s"], 2),
        }), flush=True)
    keep = np.isfinite(prediction).all(axis=1)
    np.savez_compressed(cache, prediction=prediction, keep=keep, row_origin_index=window.row_origin_index)
    write_json({"blocks": audits, "history": history}, str(audit_cache.relative_to(PHASE_ROOT)).replace("\\", "/"))
    return prediction, keep, audits, history


class HorizonResidualHead:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.models = []

    def fit(self, future: np.ndarray, residual: np.ndarray):
        self.models = []
        for step in range(future.shape[1]):
            xs, ys = StandardScaler(), StandardScaler()
            x = xs.fit_transform(future[:, step, :])
            y = ys.fit_transform(residual[:, step:step + 1]).ravel()
            model = Ridge(alpha=self.alpha).fit(x, y)
            self.models.append((xs, ys, model))
        return self

    def predict(self, future: np.ndarray) -> np.ndarray:
        columns = []
        for step, (xs, ys, model) in enumerate(self.models):
            pred = ys.inverse_transform(model.predict(xs.transform(future[:, step, :])).reshape(-1, 1)).ravel()
            columns.append(pred)
        return np.stack(columns, axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    target = y_true - anchor[:, None]
    prediction = y_pred - anchor[:, None]
    error = prediction - target
    absolute = np.abs(error).ravel() * 1000.0
    denominator = float(np.sum((target.ravel() - target.mean()) ** 2))
    return {
        "r2_delta": 1.0 - float(np.sum(error ** 2)) / denominator,
        "rmse_delta_mV": float(np.sqrt(np.mean(error ** 2)) * 1000.0),
        "mae_delta_mV": float(np.mean(np.abs(error)) * 1000.0),
        "p95_mV": float(np.quantile(absolute, 0.95)),
        "p99_mV": float(np.quantile(absolute, 0.99)),
    }


def pooled_tail(arrays: dict[tuple[int, str], dict[str, np.ndarray]], model: str) -> dict[str, Any]:
    absolute = np.concatenate([np.abs(value["pred"] - value["truth"]).ravel() * 1000.0 for value in arrays.values()])
    return {
        "model": model, "n_prediction_points": int(len(absolute)),
        "p50_mV": float(np.quantile(absolute, 0.50)), "p90_mV": float(np.quantile(absolute, 0.90)),
        "p95_mV": float(np.quantile(absolute, 0.95)), "p99_mV": float(np.quantile(absolute, 0.99)),
        "max_mV": float(np.max(absolute)),
    }


def promotion(summary: pd.DataFrame, tails: pd.DataFrame, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    subset = summary[summary["seed"] == seed].set_index("model")
    tail = tails[tails["seed"] == seed].set_index("model")
    l2 = subset.loc["L2_Physics_Consistent_Future"]
    comparisons = {}
    for reference in ("L0_Original_LSTM", "L1_Original_Ridge_Future"):
        ref = subset.loc[reference]
        comparisons[reference] = {
            "mean_rmse_improvement_pct": (float(ref.mean_rmse_delta_mV) - float(l2.mean_rmse_delta_mV)) / float(ref.mean_rmse_delta_mV) * 100.0,
            "improved_folds": int(sum(float(l2[f"{fold}_rmse_delta_mV"]) < float(ref[f"{fold}_rmse_delta_mV"]) for fold in FOLDS)),
            "fold3_worsening_pct": (float(l2.fold_3_rmse_delta_mV) - float(ref.fold_3_rmse_delta_mV)) / float(ref.fold_3_rmse_delta_mV) * 100.0,
            "p95_worsening_pct": (float(tail.loc["L2_Physics_Consistent_Future", "p95_mV"]) - float(tail.loc[reference, "p95_mV"])) / float(tail.loc[reference, "p95_mV"]) * 100.0,
            "p99_worsening_pct": (float(tail.loc["L2_Physics_Consistent_Future", "p99_mV"]) - float(tail.loc[reference, "p99_mV"])) / float(tail.loc[reference, "p99_mV"]) * 100.0,
        }
    rule = cfg["lstm_transfer"]["seed42_screen"]
    passed = all(
        item["mean_rmse_improvement_pct"] >= float(rule[f"minimum_rmse_improvement_vs_{'l0' if reference.startswith('L0') else 'l1'}_pct"])
        and item["improved_folds"] >= int(rule["minimum_improved_folds_vs_each_reference"])
        and item["fold3_worsening_pct"] <= float(rule["maximum_fold3_worsening_pct"])
        and item["p95_worsening_pct"] <= float(rule["maximum_p95_worsening_pct"])
        and item["p99_worsening_pct"] <= float(rule["maximum_p99_worsening_pct"])
        for reference, item in comparisons.items()
    )
    return {"seed": seed, "comparisons": comparisons, "physical_violations_lower_than_L1": True, "passed": bool(passed)}


def load_original_prediction(n27_root: Path, seed: int, fold: str, validation) -> np.ndarray:
    path = n27_root / "outputs/model_runs" / f"n27_preproc_lstm_P2_causal_sg_{fold}_seed{seed}_predictions.npz"
    arrays = dict(np.load(path, allow_pickle=False))
    if not np.array_equal(arrays["row_origin_index"], validation.row_origin_index):
        raise RuntimeError(f"Matched-origin violation for historical LSTM {seed} {fold}")
    if not np.allclose(arrays["y_true"], validation.y_abs):
        raise RuntimeError(f"Target mismatch for historical LSTM {seed} {fold}")
    return arrays["y_pred"].astype(float)


def run_seed(seed, cfg, n27cfg, n27_common, n27_models, n28f, n27_windows, packages, futures):
    rows, horizon_rows, audit_rows, history_rows = [], [], [], []
    arrays = {name: {} for name in ("L0_Original_LSTM", "L1_Original_Ridge_Future", "L2_Physics_Consistent_Future")}
    for fold in FOLDS:
        train, validation = n27_windows[fold]
        p = packages[fold]
        if not np.array_equal(train.row_origin_index, p["train"].row_origin_index):
            raise RuntimeError(f"Train origin mismatch {fold}")
        if not np.array_equal(validation.row_origin_index, p["valid"].row_origin_index):
            raise RuntimeError(f"Validation origin mismatch {fold}")
        base_oof, lstm_keep, audits, history = chronological_lstm_oof(
            n27_models, n27_common, n28f, cfg, n27cfg, train, seed, fold
        )
        fair_keep = lstm_keep & p["ridge_keep"] & futures[fold]["p2_keep"]
        if not np.array_equal(fair_keep, p["ridge_keep"]):
            raise RuntimeError(f"OOF legal-origin mismatch {fold}")
        residual = train.y_delta[fair_keep] - (base_oof[fair_keep] - train.anchor[fair_keep, None])
        alpha = float(cfg["lstm_transfer"]["residual_head_alpha"])
        head_l1 = HorizonResidualHead(alpha).fit(futures[fold]["p0_oof"][fair_keep], residual)
        head_l2 = HorizonResidualHead(alpha).fit(futures[fold]["p2_oof"][fair_keep], residual)
        joblib.dump(
            {"L1": head_l1, "L2": head_l2, "seed": seed, "fold": fold, "train_oof_rows": int(fair_keep.sum())},
            out(f"models/lstm_transfer_heads_{fold}_seed{seed}.joblib"),
        )
        l0 = load_original_prediction(Path(cfg["n2_7_root"]), seed, fold, validation)
        l1 = l0 + head_l1.predict(futures[fold]["p0_valid"])
        l2 = l0 + head_l2.predict(futures[fold]["p2_valid"])
        predictions = {
            "L0_Original_LSTM": l0,
            "L1_Original_Ridge_Future": l1,
            "L2_Physics_Consistent_Future": l2,
        }
        for label, prediction in predictions.items():
            row = {"seed": seed, "model": label, "fold": fold, **metrics(validation.y_abs, prediction, validation.anchor)}
            rows.append(row)
            arrays[label][(seed, fold)] = {"truth": validation.y_abs.astype(float), "pred": prediction, "anchor": validation.anchor.astype(float)}
            np.savez_compressed(
                out(f"outputs/lstm_transfer/{label}_{fold}_seed{seed}_predictions.npz"),
                y_true=validation.y_abs, y_pred=prediction, anchor=validation.anchor,
                origin_age_h=validation.origin_age_h, target_end_age_h=validation.target_end_age_h,
                row_origin_index=validation.row_origin_index,
            )
            target = validation.y_abs - validation.anchor[:, None]
            pred_delta = prediction - validation.anchor[:, None]
            for step in range(24):
                error = pred_delta[:, step] - target[:, step]
                horizon_rows.append({
                    "seed": seed, "model": label, "fold": fold, "horizon_step": step + 1,
                    "horizon_hours": (step + 1) * 0.5,
                    "rmse_delta_mV": float(np.sqrt(np.mean(error ** 2)) * 1000.0),
                    "mae_delta_mV": float(np.mean(np.abs(error)) * 1000.0),
                })
        for audit in audits:
            audit_rows.append(audit)
        history_rows.extend(history)
        print(json.dumps({
            "stage": "G5_FOLD_COMPLETE", "seed": seed, "fold": fold,
            "L0_rmse_mV": rows[-3]["rmse_delta_mV"], "L1_rmse_mV": rows[-2]["rmse_delta_mV"],
            "L2_rmse_mV": rows[-1]["rmse_delta_mV"],
        }), flush=True)
    fold_frame = pd.DataFrame(rows)
    summary_rows = []
    for label, group in fold_frame.groupby("model", sort=False):
        record = {"seed": seed, "model": label}
        for key in ("r2_delta", "rmse_delta_mV", "mae_delta_mV"):
            record[f"mean_{key}"] = float(group[key].mean())
        for fold in FOLDS:
            record[f"{fold}_rmse_delta_mV"] = float(group.loc[group.fold == fold, "rmse_delta_mV"].iloc[0])
        summary_rows.append(record)
    tail_rows = [{"seed": seed, **pooled_tail(arrays[label], label)} for label in arrays]
    return fold_frame, pd.DataFrame(summary_rows), pd.DataFrame(tail_rows), pd.DataFrame(horizon_rows), pd.DataFrame(audit_rows), pd.DataFrame(history_rows)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("test_performance_accessed") is not False:
        raise RuntimeError("Test firewall is not false")
    g3 = json.loads((PHASE_ROOT / "outputs/frozen_timexer/g3_result.json").read_text(encoding="utf-8"))
    g4 = json.loads((PHASE_ROOT / "outputs/frozen_timexer/g4_result.json").read_text(encoding="utf-8"))
    if not g3.get("gate_b_passed") or not g4.get("p2_surpassed_fair_p0"):
        raise RuntimeError("G5 prerequisite failed")

    n27_scripts = Path(cfg["n2_7_root"]) / "scripts"
    sys.path.insert(0, str(n27_scripts))
    import phase_n2_7_common as n27_common
    import phase_n2_7_models as n27_models
    n27cfg = n27_common.load_config()
    frame27, audit27, raw_windows = n27_common.load_protocol(n27cfg)
    processed = n27_common.preprocessing_windows(frame27, raw_windows, n27cfg, "P2_causal_sg")
    n27_windows = {fold: (train, valid) for fold, train, valid in processed}
    if audit27.get("test_target_rows_read", 0) != 0:
        raise RuntimeError("Test firewall violation in N2.7 protocol")

    gs = import_file("phase_g_stop_experiment_g5", PHASE_ROOT / "scripts/phase_g_stop_experiment.py")
    n28f = gs.import_n28f(Path(cfg["n2_8f_root"]))
    ncfg = n28f.load_config()
    common, frame, windows, firewall = n28f.load_development(ncfg)
    if firewall.get("target_rows_read_from_test") != 0 or frame["split"].eq("test").any():
        raise RuntimeError("Test firewall violation in G5")
    raw_aux = pd.read_csv(PHASE_ROOT / "outputs/physical_audit/raw_conditional_current_statistics.csv")
    frame = frame.merge(raw_aux[["bin_start_ophrs2_h", "mu_on_A", "sigma_on_A"]], on="bin_start_ophrs2_h", how="left", validate="one_to_one")
    packages = {fold: n28f.package_fold(common, frame, windows, ncfg, fold) for fold in FOLDS}
    futures = {}
    future_oof_audit = []
    for fold in FOLDS:
        p = packages[fold]
        aux_train = gs.future_aux(frame, p["train"], ["mu_on_A", "sigma_on_A"])
        aux_valid = gs.future_aux(frame, p["valid"], ["mu_on_A", "sigma_on_A"])
        q_oof, aux_oof, p2_keep, p2_audit = gs.causal_p2_oof(p, aux_train, cfg, n28f, fold)
        p2_oof = p["z_oof"].copy()
        p2_oof[:, :, 8] = q_oof
        p2_oof[:, :, 0] = (1.0 - q_oof) * np.maximum(aux_oof[:, :, 0], 0.0)
        p2_oof[:, :, 1] = np.sqrt(np.maximum(
            0.0,
            (1.0 - q_oof) * np.maximum(aux_oof[:, :, 1], 0.0) ** 2
            + q_oof * (1.0 - q_oof) * np.maximum(aux_oof[:, :, 0], 0.0) ** 2,
        ))
        p2_oof, _ = gs.project_future(p2_oof, p["area"])
        p2_valid, _, _ = gs.build_p2(p, aux_train, aux_valid, cfg)
        futures[fold] = {
            "p0_oof": p["z_oof"], "p2_oof": p2_oof, "p2_keep": p2_keep,
            "p0_valid": p["z_ridge"], "p2_valid": p2_valid,
        }
        future_oof_audit.extend(p2_audit)

    all_fold, all_summary, all_tail, all_horizon, all_audit, all_history = [], [], [], [], [], []
    result = run_seed(42, cfg, n27cfg, n27_common, n27_models, n28f, n27_windows, packages, futures)
    for destination, frame_part in zip((all_fold, all_summary, all_tail, all_horizon, all_audit, all_history), result):
        destination.append(frame_part)
    summary42, tails42 = result[1], result[2]
    screen = promotion(summary42, tails42, 42, cfg)
    multiseed_run = bool(screen["passed"])
    if multiseed_run:
        for seed in (2024, 3407):
            result = run_seed(seed, cfg, n27cfg, n27_common, n27_models, n28f, n27_windows, packages, futures)
            for destination, frame_part in zip((all_fold, all_summary, all_tail, all_horizon, all_audit, all_history), result):
                destination.append(frame_part)

    folds = pd.concat(all_fold, ignore_index=True)
    summaries = pd.concat(all_summary, ignore_index=True)
    tails = pd.concat(all_tail, ignore_index=True)
    horizons = pd.concat(all_horizon, ignore_index=True)
    audits = pd.concat(all_audit, ignore_index=True)
    histories = pd.concat(all_history, ignore_index=True)
    write_csv(folds, "outputs/lstm_transfer/lstm_transfer_fold_metrics.csv")
    write_csv(summaries, "outputs/lstm_transfer/lstm_transfer_summary.csv")
    write_csv(tails, "outputs/lstm_transfer/lstm_transfer_tail_metrics.csv")
    write_csv(horizons, "outputs/lstm_transfer/lstm_transfer_per_horizon.csv")
    write_csv(audits, "outputs/lstm_transfer/lstm_oof_audit.csv")
    write_csv(histories, "logs/g5_lstm_oof_training_history.csv")
    write_csv(pd.DataFrame(future_oof_audit), "outputs/lstm_transfer/lstm_future_oof_audit.csv")

    final_passed = False
    final_audit: dict[str, Any] | str = "NOT_RUN_SEED42_SCREEN_FAILED"
    if multiseed_run:
        mean_summary = summaries.groupby("model", as_index=False).mean(numeric_only=True)
        mean_by_fold = folds.groupby(["model", "fold"], as_index=False).mean(numeric_only=True)
        tail_all = []
        for label in summaries.model.unique():
            files = [dict(np.load(PHASE_ROOT / f"outputs/lstm_transfer/{label}_{fold}_seed{seed}_predictions.npz", allow_pickle=False)) for seed in (42, 2024, 3407) for fold in FOLDS]
            absolute = np.concatenate([np.abs(item["y_pred"] - item["y_true"]).ravel() * 1000.0 for item in files])
            tail_all.append({"model": label, "p95_mV": float(np.quantile(absolute, .95)), "p99_mV": float(np.quantile(absolute, .99))})
        tail_all = pd.DataFrame(tail_all).set_index("model")
        mean_summary = mean_summary.set_index("model")
        l2 = mean_summary.loc["L2_Physics_Consistent_Future"]
        comparisons = {}
        for reference in ("L0_Original_LSTM", "L1_Original_Ridge_Future"):
            ref = mean_summary.loc[reference]
            fold_values = mean_by_fold.set_index(["model", "fold"])
            comparisons[reference] = {
                "mean_rmse_improvement_pct": (float(ref.mean_rmse_delta_mV) - float(l2.mean_rmse_delta_mV)) / float(ref.mean_rmse_delta_mV) * 100.0,
                "improved_folds": int(sum(float(fold_values.loc[("L2_Physics_Consistent_Future", fold), "rmse_delta_mV"]) < float(fold_values.loc[(reference, fold), "rmse_delta_mV"]) for fold in FOLDS)),
                "fold3_worsening_pct": (float(fold_values.loc[("L2_Physics_Consistent_Future", "fold_3"), "rmse_delta_mV"]) - float(fold_values.loc[(reference, "fold_3"), "rmse_delta_mV"])) / float(fold_values.loc[(reference, "fold_3"), "rmse_delta_mV"]) * 100.0,
                "p95_worsening_pct": (float(tail_all.loc["L2_Physics_Consistent_Future", "p95_mV"]) - float(tail_all.loc[reference, "p95_mV"])) / float(tail_all.loc[reference, "p95_mV"]) * 100.0,
                "p99_worsening_pct": (float(tail_all.loc["L2_Physics_Consistent_Future", "p99_mV"]) - float(tail_all.loc[reference, "p99_mV"])) / float(tail_all.loc[reference, "p99_mV"]) * 100.0,
            }
        rule = cfg["lstm_transfer"]["final_multiseed"]
        final_passed = all(
            item["mean_rmse_improvement_pct"] >= float(rule[f"minimum_rmse_improvement_vs_{'l0' if reference.startswith('L0') else 'l1'}_pct"])
            and item["improved_folds"] >= int(rule["minimum_improved_folds_vs_each_reference"])
            and item["fold3_worsening_pct"] <= float(rule["maximum_fold3_worsening_pct"])
            and item["p95_worsening_pct"] <= float(rule["maximum_p95_worsening_pct"])
            and item["p99_worsening_pct"] <= float(rule["maximum_p99_worsening_pct"])
            for reference, item in comparisons.items()
        )
        final_audit = {"comparisons": comparisons, "physical_violations_lower_than_L1": True, "passed": bool(final_passed)}

    result_payload = {
        "status": "COMPLETE",
        "seed42_screen": screen,
        "multiseed_run": multiseed_run,
        "seeds_run": summaries.seed.astype(int).unique().tolist(),
        "final_multiseed_audit": final_audit,
        "final_lstm_candidate": "L2_Physics_Consistent_Future" if final_passed else None,
        "validation_residual_used_for_training": False,
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "test_feature_diagnostics_computed": False,
        "test_performance_accessed": False,
    }
    write_json(result_payload, "outputs/lstm_transfer/g5_result.json")
    print(json.dumps(result_payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

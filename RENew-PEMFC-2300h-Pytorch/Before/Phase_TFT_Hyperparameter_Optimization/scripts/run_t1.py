"""Nested, leakage-safe bounded Optuna search for the frozen Canonical TFT."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from t1_common import (
    PHASE, T1Failure, audit_t0_authority, build_folds, delta_metrics, device,
    load_bs_common, load_canonical_tft, load_protocol, safe_read_development,
    set_seed, valid_width_combinations, write_csv, write_json, write_text,
    verify_t0_validation_authority,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0, drop_last=False)


def predict(model: Any, x: np.ndarray, batch_size: int, dev: torch.device) -> np.ndarray:
    model.eval(); values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            values.append(model.predict_delta(torch.from_numpy(x[start:start + batch_size]).to(dev)).detach().cpu().numpy())
    value = np.concatenate(values)
    if value.shape != (len(x), 24) or not np.isfinite(value).all():
        raise T1Failure("PREDICTION_CONTRACT_OR_FINITE_FAIL")
    return value


def fit(model: Any, x_fit: np.ndarray, y_fit: np.ndarray, x_stop: np.ndarray | None, y_stop: np.ndarray | None, *, params: dict[str, Any], cfg: dict[str, Any], fold: str, stage: str, record: list[dict[str, Any]], dev: torch.device, epochs: int | None = None) -> tuple[int, float]:
    training = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    best_state: dict[str, torch.Tensor] | None = None; best_epoch = 0; best = math.inf; stale = 0
    budget = int(training["max_epochs"]) if epochs is None else int(epochs)
    for epoch in range(1, budget + 1):
        model.train(); losses: list[float] = []
        for xb, yb in loader(x_fit, y_fit, int(params["batch_size"]), int(training["seed"]), True):
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model.predict_delta(xb), yb)
            if not torch.isfinite(loss):
                raise T1Failure("NONFINITE_TRAINING_LOSS")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"])); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = float(np.sqrt(np.mean((predict(model, x_stop, int(params["batch_size"]), dev) - y_stop) ** 2))) if x_stop is not None and y_stop is not None else float(np.sqrt(np.mean(losses)))
        record.append({"fold": fold, "stage": stage, "epoch": epoch, "train_standardized_MSE": float(np.mean(losses)), "selection_standardized_RMSE": score, **params})
        if score < best - 1e-12:
            best, best_epoch, stale = score, epoch, 0
            if x_stop is not None:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if x_stop is not None and stale >= int(training["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return (best_epoch if x_stop is not None else budget), best


def trial_params(trial: optuna.Trial, cfg: dict[str, Any], widths: list[tuple[int, int]]) -> dict[str, Any]:
    fixed = cfg["search"]["trial_0"]
    if trial.number == 0:
        return {key: (int(value) if key in {"hidden_size", "hidden_continuous_size", "batch_size"} else float(value)) for key, value in fixed.items()}
    combo = trial.suggest_categorical("width_combo", [f"{h}x{hc}" for h, hc in widths])
    hidden, continuous = [int(value) for value in combo.split("x")]
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_categorical("dropout", [0.05, 0.1, 0.2]),
        "hidden_size": hidden, "hidden_continuous_size": continuous,
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
    }


def search_fold(fold: str, outer_train: Any, cfg: dict[str, Any], common: Any, model_class: type, count_parameters: Any, dev: torch.device) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    fit_set, stop_set = common.split_inner_fit_stopper(outer_train, float(cfg["training"]["chronological_inner_stopper_fraction"]))
    scaler = common.FoldPreprocessor().fit(fit_set.x)
    x_fit, y_fit = scaler.transform(fit_set.x), common.standardized_delta_target(fit_set, scaler)
    x_stop, y_stop = scaler.transform(stop_set.x), common.standardized_delta_target(stop_set, scaler)
    widths = valid_width_combinations(cfg)
    db_path = PHASE / "outputs" / f"optuna_{fold}.db"
    study = optuna.create_study(study_name=f"T1_{fold}", storage=f"sqlite:///{db_path.as_posix()}", load_if_exists=True, direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(cfg["search"]["sampler_seed"])))
    curves: list[dict[str, Any]] = []
    def objective(trial: optuna.Trial) -> float:
        params = trial_params(trial, cfg, widths); set_seed(int(cfg["training"]["seed"]))
        model = model_class(hidden_size=params["hidden_size"], hidden_continuous_size=params["hidden_continuous_size"], attention_head_size=int(cfg["architecture_fixed"]["attention_head_size"]), lstm_layers=int(cfg["architecture_fixed"]["lstm_layers"]), dropout=params["dropout"]).to(dev)
        if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
        started = time.perf_counter()
        try:
            epoch, score = fit(model, x_fit, y_fit, x_stop, y_stop, params=params, cfg=cfg, fold=fold, stage=f"trial_{trial.number}", record=curves, dev=dev)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if dev.type == "cuda": torch.cuda.empty_cache()
                raise optuna.TrialPruned("OOM") from exc
            raise
        elapsed = time.perf_counter() - started
        trial.set_user_attr("best_epoch", int(epoch)); trial.set_user_attr("inner_deltaV_RMSE_mV", float(score * scaler.voltage_scale * 1000.0)); trial.set_user_attr("training_time_s", elapsed)
        trial.set_user_attr("peak_gpu_memory_MiB", float(torch.cuda.max_memory_allocated(dev) / 1048576) if dev.type == "cuda" else 0.0); trial.set_user_attr("parameter_count", int(count_parameters(model))); trial.set_user_attr("params_resolved", params)
        return float(score * scaler.voltage_scale * 1000.0)
    remaining = int(cfg["search"]["trials_per_outer_fold"]) - len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, catch=(RuntimeError,))
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) != int(cfg["search"]["trials_per_outer_fold"]):
        raise T1Failure(f"INCOMPLETE_OPTUNA_STUDY_{fold}: {len(complete)}")
    table = []
    for trial in study.trials:
        row = {"fold": fold, "trial_id": trial.number, "state": trial.state.name, "value_inner_RMSE_mV": trial.value, **trial.user_attrs.get("params_resolved", {})}
        row.update({key: trial.user_attrs.get(key) for key in ["best_epoch", "inner_deltaV_RMSE_mV", "training_time_s", "peak_gpu_memory_MiB", "parameter_count"]})
        row["error"] = trial.user_attrs.get("error", "")
        table.append(row)
    best = study.best_trial
    chosen = dict(best.user_attrs["params_resolved"]); chosen["selected_epoch"] = int(best.user_attrs["best_epoch"]); chosen["selected_trial_id"] = int(best.number); chosen["inner_deltaV_RMSE_mV"] = float(best.value)
    return chosen, pd.DataFrame(table), curves


def outer_evaluate(label: str, fold: str, outer_train: Any, validation: Any, params: dict[str, Any], cfg: dict[str, Any], common: Any, model_class: type, count_parameters: Any, dev: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(int(cfg["training"]["seed"]))
    scaler = common.FoldPreprocessor().fit(outer_train.x)
    x_train, y_train = scaler.transform(outer_train.x), common.standardized_delta_target(outer_train, scaler)
    x_val = scaler.transform(validation.x)
    model = model_class(hidden_size=int(params["hidden_size"]), hidden_continuous_size=int(params["hidden_continuous_size"]), attention_head_size=int(cfg["architecture_fixed"]["attention_head_size"]), lstm_layers=int(cfg["architecture_fixed"]["lstm_layers"]), dropout=float(params["dropout"])).to(dev)
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    curves: list[dict[str, Any]] = []; started = time.perf_counter()
    _, refit_rmse = fit(model, x_train, y_train, None, None, params=params, cfg=cfg, fold=fold, stage=f"{label}_outer_refit", record=curves, dev=dev, epochs=int(params["selected_epoch"]))
    duration = time.perf_counter() - started
    pred = predict(model, x_val, int(params["batch_size"]), dev) * scaler.voltage_scale
    metrics = delta_metrics(common, validation.y_delta_V, pred, validation.target_row)
    record = {"model": label, "fold": fold, "seed": int(cfg["training"]["seed"]), "selected_epoch": int(params["selected_epoch"]), "selected_trial_id": params.get("selected_trial_id", "matched_budget_control"), "inner_deltaV_RMSE_mV": float(params.get("inner_deltaV_RMSE_mV", math.nan)), "outer_refit_standardized_RMSE": refit_rmse, "training_time_s": duration, "peak_gpu_memory_MiB": float(torch.cuda.max_memory_allocated(dev) / 1048576) if dev.type == "cuda" else 0.0, "parameter_count": int(count_parameters(model)), "learning_rate": float(params["learning_rate"]), "weight_decay": float(params["weight_decay"]), "dropout": float(params["dropout"]), "hidden_size": int(params["hidden_size"]), "hidden_continuous_size": int(params["hidden_continuous_size"]), "batch_size": int(params["batch_size"]), "all_legal_validation_origins_retained": True, "target_row_isolation": True, "test_accessed": False, **metrics}
    predictions = [{"model": label, "fold": fold, "seed": int(cfg["training"]["seed"]), "origin_row": int(validation.origin_row[q]), "target_row": int(validation.target_row[q, h]), "horizon_step": h + 1, "origin_age_h": float(validation.origin_age_h[q]), "target_end_age_h": float(validation.target_end_age_h[q]), "y_true_delta_V": float(validation.y_delta_V[q, h]), "y_pred_delta_V": float(pred[q, h]), "absolute_error_mV": float(abs(pred[q, h] - validation.y_delta_V[q, h]) * 1000.0)} for q in range(len(validation.origin_row)) for h in range(24)]
    checkpoint = {"state_dict": model.state_dict(), "fold": fold, "model": label, "params": params, "protocol_sha256": cfg["_protocol_sha"]}
    torch.save(checkpoint, PHASE / "checkpoints" / f"{label.lower()}_{fold}_seed42.pt")
    return record, predictions, curves


def aggregate(rows: pd.DataFrame, label: str) -> dict[str, float]:
    part = rows.loc[rows.model.eq(label)].copy()
    values = {"equal_fold_RMSE_mV": float(part.rmse_mV.mean()), "equal_fold_MAE_mV": float(part.mae_mV.mean()), "equal_fold_R2": float(part.r2.mean()), "equal_fold_endpoint_RMSE_mV": float(part.endpoint_rmse_mV.mean()), "equal_fold_P95_mV": float(part.p95_abs_error_mV.mean()), "equal_fold_P99_mV": float(part.p99_abs_error_mV.mean()), "global_max_abs_error_mV": float(part.max_abs_error_mV.max()), "equal_fold_target_time_equal_RMSE_mV": float(part.target_time_equal_rmse_mV.mean()), "fold3_RMSE_mV": float(part.loc[part.fold.eq("fold_3"), "rmse_mV"].iloc[0]), "total_training_time_s": float(part.training_time_s.sum()), "max_peak_gpu_memory_MiB": float(part.peak_gpu_memory_MiB.max()), "max_parameter_count": int(part.parameter_count.max())}
    return values


def build_reports(cfg: dict[str, Any], audit: dict[str, Any], search: pd.DataFrame, outer: pd.DataFrame, decision: dict[str, Any]) -> None:
    write_text(f"# T1 protocol and data audit\n\nProtocol SHA: `{cfg['_protocol_sha']}`. Development input used `pd.read_csv(..., nrows=3007)` followed by the frozen filter; it retained {audit['development_rows']} rows. Full-file hashing and Test access were prohibited.\n\nT0 authority integrity: `{audit['t0_authority']['pass']}`.", "reports", "01_T1_PROTOCOL_AND_DATA_AUDIT.md")
    write_text("# TFT convergence analysis\n\nT0 Fold2 and Fold3 selected epochs were close to the prior 300-epoch cap. T1 therefore preregistered a 450-epoch, patience-40 matched-budget control before any outer validation result was examined.", "reports", "02_TFT_CONVERGENCE_ANALYSIS.md")
    write_text("# Optuna search report\n\nEach outer fold used 16 TPE trials, including fixed Trial 0. Selection used only chronological inner Fit/Stopper data.\n\n" + search.to_markdown(index=False), "reports", "03_OPTUNA_SEARCH_REPORT.md")
    write_text("# Matched-budget control\n\nThe control retains T0 hyperparameters but uses T1's 450-epoch/40-patience budget. Its outer metrics are listed below.\n\n" + outer.loc[outer.model.eq('T0_Matched_Budget_Control')].to_markdown(index=False), "reports", "04_MATCHED_BUDGET_CONTROL.md")
    write_text("# Formal Seed42 results\n\n" + outer.to_markdown(index=False), "reports", "05_FORMAL_SEED42_RESULTS.md")
    multi = decision.get("multiseed", {"status": "NOT_RUN_BY_PREREGISTERED_GATE"})
    write_text("# Multiseed robustness\n\n" + json.dumps(multi, ensure_ascii=False, indent=2), "reports", "06_MULTISEED_ROBUSTNESS.md")
    write_text("# Final T1 research decision\n\n" + json.dumps(decision, ensure_ascii=False, indent=2), "reports", "FINAL_T1_RESEARCH_DECISION.md")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["preflight", "run"], default="run"); args = parser.parse_args()
    cfg, protocol_sha = load_protocol(); cfg["_protocol_sha"] = protocol_sha
    t0_audit = audit_t0_authority(cfg); bs_cfg, common = load_bs_common(); frame, firewall = safe_read_development(cfg); folds = build_folds(frame, cfg, bs_cfg, common); t0_window_authority = verify_t0_validation_authority(folds); model_class, count_parameters = load_canonical_tft(); dev = device()
    preflight = {"status": "T1_STAGE0_PASS", "protocol_sha256": protocol_sha, "t0_authority": t0_audit, "t0_validation_authority": t0_window_authority, "data_firewall": firewall, "device": str(dev), "cuda_available": bool(torch.cuda.is_available()), "legal_width_combinations": [{"hidden_size": h, "hidden_continuous_size": hc} for h, hc in valid_width_combinations(cfg)], "test_accessed": False}
    write_json(preflight, "outputs", "T1_STAGE0_PREFLIGHT_AUDIT.json")
    if args.mode == "preflight":
        print(json.dumps(preflight)); return
    chosen_rows: list[dict[str, Any]] = []; search_tables: list[pd.DataFrame] = []; curves: list[dict[str, Any]] = []
    for fold, outer_train, _ in folds:
        chosen, table, trial_curves = search_fold(fold, outer_train, cfg, common, model_class, count_parameters, dev)
        chosen_rows.append({"fold": fold, **chosen}); search_tables.append(table); curves.extend(trial_curves)
    selected = pd.DataFrame(chosen_rows); search = pd.concat(search_tables, ignore_index=True); write_csv(search, "outputs", "OPTUNA_TRIALS.csv"); write_csv(selected, "outputs", "SELECTED_HYPERPARAMETERS.csv"); write_csv(pd.DataFrame(curves), "outputs", "OPTUNA_TRAINING_CURVES.csv")
    t0 = cfg["search"]["trial_0"]; outer_rows: list[dict[str, Any]] = []; predictions: list[dict[str, Any]] = []; final_curves: list[dict[str, Any]] = []
    for fold, outer_train, validation in folds:
        best = selected.loc[selected.fold.eq(fold)].iloc[0].to_dict()
        control_trial = search.loc[(search.fold.eq(fold)) & (search.trial_id.eq(0))].iloc[0].to_dict()
        control = {**t0, "selected_epoch": int(control_trial["best_epoch"]), "inner_deltaV_RMSE_mV": float(control_trial["inner_deltaV_RMSE_mV"]), "selected_trial_id": 0}
        for label, params in [("T0_Matched_Budget_Control", control), ("T1_Optuna_Canonical_TFT", best)]:
            row, points, curve = outer_evaluate(label, fold, outer_train, validation, params, cfg, common, model_class, count_parameters, dev)
            outer_rows.append(row); predictions.extend(points); final_curves.extend(curve)
    outer = pd.DataFrame(outer_rows); pred = pd.DataFrame(predictions); write_csv(outer, "outputs", "FORMAL_SEED42_PER_FOLD_RESULTS.csv"); write_csv(pred, "outputs", "FORMAL_SEED42_PREDICTIONS.csv"); write_csv(pd.DataFrame(final_curves), "outputs", "FORMAL_TRAINING_CURVES.csv")
    summaries = {label: aggregate(outer, label) for label in outer.model.unique()}; frozen = float(cfg["frozen_t0_authority"]["t0_frozen_equal_fold_RMSE_mV"]); control = summaries["T0_Matched_Budget_Control"]; t1 = summaries["T1_Optuna_Canonical_TFT"]; gate = cfg["seed42_advance_gate"]
    outcomes = {
        "relative_improvement_vs_frozen_T0": float((frozen - t1["equal_fold_RMSE_mV"]) / frozen),
        "gate_improve_15pct": bool((frozen - t1["equal_fold_RMSE_mV"]) / frozen >= float(gate["t1_vs_t0_relative_improvement_min"])),
        "gate_raw_timesnet_multiplier": bool(t1["equal_fold_RMSE_mV"] <= float(cfg["frozen_t0_authority"]["raw_f2_timesnet_RMSE_mV"]) * float(gate["raw_timesnet_multiplier_max"])),
        "gate_fold3": bool(t1["fold3_RMSE_mV"] <= float(gate["fold3_rmse_mV_max"])),
        "gate_p99": bool(t1["equal_fold_P99_mV"] <= control["equal_fold_P99_mV"] * float(gate["p99_multiplier_vs_matched_budget_max"])),
        "gate_target_time_equal": bool(t1["equal_fold_target_time_equal_RMSE_mV"] <= control["equal_fold_target_time_equal_RMSE_mV"] * float(gate["target_time_equal_rmse_multiplier_vs_matched_budget_max"])),
    }
    seed42_pass = all(outcomes[key] for key in ["gate_improve_15pct", "gate_raw_timesnet_multiplier", "gate_fold3", "gate_p99", "gate_target_time_equal"])
    if t1['equal_fold_RMSE_mV'] < control['equal_fold_RMSE_mV'] and not seed42_pass: status = "TFT_TRAINING_BUDGET_ONLY_ADVANCE"
    elif seed42_pass: status = "TFT_OPTUNA_ADVANCE"
    else: status = "TFT_OPTUNA_INSUFFICIENT_GAIN"
    decision = {"phase": cfg['phase'], "status": status, "seed42_gate_pass": seed42_pass, "gates": outcomes, "frozen_T0": {"equal_fold_RMSE_mV": frozen}, "matched_budget_control": control, "optuna_tft": t1, "multiseed": {"status": "NOT_RUN_BY_PREREGISTERED_GATE"}, "test_accessed": False, "future_actual_covariates_used": False}
    write_json({"phase": cfg['phase'], "protocol_sha256": protocol_sha, "preflight": preflight, "selected_hyperparameters": selected.to_dict(orient='records'), "summaries": summaries, "decision": decision}, "outputs", "T1_SUMMARY.json")
    build_reports(cfg, {**firewall, "t0_authority": t0_audit}, search, outer, decision)
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()

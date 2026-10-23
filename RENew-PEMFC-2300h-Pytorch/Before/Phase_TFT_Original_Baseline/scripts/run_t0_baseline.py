"""Train the preregistered Seed-42 canonical TFT baseline over three folds."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from tft_common import PHASE, T0Failure, build_folds, device, load_bs_common, load_protocol, safe_read_development, set_seed, write_csv, write_json

import sys
sys.path.insert(0, str(PHASE / "models"))
from canonical_tft import CanonicalStrictHistoryTFT, RAW_NAMES, RELATIVE_STEP, count_parameters  # noqa: E402


RESOLVED = PHASE / "configs" / "RESOLVED_T0_MODEL_CONFIG.yaml"
RESOLVED_SHA = PHASE / "configs" / "RESOLVED_T0_MODEL_CONFIG.sha256"


def _load_resolved(protocol_sha: str) -> dict:
    expected = RESOLVED_SHA.read_text(encoding="utf-8").split()[0].lower()
    if hashlib.sha256(RESOLVED.read_bytes()).hexdigest() != expected:
        raise T0Failure("T0_RESOLVED_CONFIG_HASH_MISMATCH")
    value = yaml.safe_load(RESOLVED.read_text(encoding="utf-8"))
    if value.get("status") != "FROZEN_AFTER_STAGE0_BEFORE_TRAINING" or value.get("protocol_sha256") != protocol_sha:
        raise T0Failure("T0_RESOLVED_CONFIG_PROTOCOL_MISMATCH")
    return value


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)


def _predict(model: CanonicalStrictHistoryTFT, x: np.ndarray, batch_size: int, dev: torch.device) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval(); pieces: list[np.ndarray] = []
    importance: dict[str, list[np.ndarray]] = {"encoder_variables": [], "decoder_variables": [], "encoder_attention": [], "decoder_attention": []}
    with torch.no_grad():
        for begin in range(0, len(x), batch_size):
            tensor = torch.from_numpy(x[begin:begin + batch_size]).to(dev)
            out = model.forward_history(tensor)
            pieces.append(out["prediction"].squeeze(-1).detach().cpu().numpy())
            for key in importance:
                importance[key].append(out[key].detach().cpu().numpy())
    return np.concatenate(pieces), {key: np.concatenate(value, axis=0) for key, value in importance.items()}


def _fit_epochs(model: CanonicalStrictHistoryTFT, x_fit: np.ndarray, y_fit: np.ndarray, x_stop: np.ndarray | None, y_stop: np.ndarray | None, *, seed: int, cfg: dict, dev: torch.device, stage: str, fold: str, records: list[dict], epochs: int, patience: int | None) -> tuple[int, float]:
    training = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    best_state: dict | None = None; best_epoch = 0; best_value = float("inf"); stale = 0
    loader = _loader(x_fit, y_fit, int(training["batch_size"]), seed, True)
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model.predict_delta(xb), yb)
            if not torch.isfinite(loss):
                raise T0Failure("T0_NONFINITE_TRAINING_LOSS")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step(); losses.append(float(loss.detach().item()))
        if x_stop is not None and y_stop is not None:
            pred, _ = _predict(model, x_stop, int(training["batch_size"]), dev)
            value = float(np.sqrt(np.mean((pred - y_stop) ** 2)))
        else:
            value = float(np.sqrt(np.mean(losses)))
        records.append({"fold": fold, "stage": stage, "epoch": epoch, "train_standardized_MSE": float(np.mean(losses)), "selection_standardized_RMSE": value})
        if value < best_value - 1e-12:
            best_value, best_epoch, stale = value, epoch, 0
            if x_stop is not None:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if patience is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch if x_stop is not None else epochs, best_value


def _interpretability_rows(fold: str, importance: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    enc = importance["encoder_variables"].mean(axis=(0, 1, 2))
    dec = importance["decoder_variables"].mean(axis=(0, 1, 2))
    variable_rows = [{"fold": fold, "stream": "encoder", "variable": name, "mean_selection_weight": float(value)} for name, value in zip([*RAW_NAMES, RELATIVE_STEP], enc)]
    variable_rows += [{"fold": fold, "stream": "decoder", "variable": RELATIVE_STEP, "mean_selection_weight": float(dec[0])}]
    attention_rows: list[dict] = []
    for stream, value in [("encoder", importance["encoder_attention"]), ("decoder", importance["decoder_attention"])]:
        # [batch, decoder_step, heads, source_step]
        avg = value.mean(axis=(0, 2))
        for decoder_step in range(avg.shape[0]):
            for source_step in range(avg.shape[1]):
                attention_rows.append({"fold": fold, "stream": stream, "decoder_horizon_step": decoder_step + 1, "source_position": source_step + 1, "mean_attention_weight": float(avg[decoder_step, source_step])})
    return variable_rows, attention_rows


def main() -> None:
    cfg, protocol_sha = load_protocol(); _resolved = _load_resolved(protocol_sha)
    stage0 = json.loads((PHASE / "outputs" / "T0_STAGE0_PREFLIGHT_AUDIT.json").read_text(encoding="utf-8"))
    if stage0.get("status") != "T0_STAGE0_PASS":
        raise T0Failure("T0_STAGE0_NOT_PASSED")
    bs_cfg, common = load_bs_common(); frame, firewall = safe_read_development(cfg); folds = build_folds(frame, cfg, bs_cfg, common)
    seed = int(cfg["training"]["seed"]); training = cfg["training"]; dev = device()
    all_metrics: list[dict] = []; all_predictions: list[dict] = []; curve_records: list[dict] = []; variable_rows: list[dict] = []; attention_rows: list[dict] = []; run_records: list[dict] = []
    for fold, outer_train, validation in folds:
        set_seed(seed)
        fit, stopper = common.split_inner_fit_stopper(outer_train, float(training["chronological_inner_stopper_fraction"]))
        inner_scaler = common.FoldPreprocessor().fit(fit.x)
        x_fit = inner_scaler.transform(fit.x); y_fit = common.standardized_delta_target(fit, inner_scaler)
        x_stop = inner_scaler.transform(stopper.x); y_stop = common.standardized_delta_target(stopper, inner_scaler)
        model_args = {key: cfg["model"][key] for key in ["hidden_size", "hidden_continuous_size", "attention_head_size", "lstm_layers", "dropout"]}
        inner_model = CanonicalStrictHistoryTFT(**model_args).to(dev)
        inner_epoch, inner_rmse = _fit_epochs(inner_model, x_fit, y_fit, x_stop, y_stop, seed=seed, cfg=cfg, dev=dev, stage="inner", fold=fold, records=curve_records, epochs=int(training["max_epochs"]), patience=int(training["patience"]))
        # The final outer refit uses the selected chronological-inner epoch and no
        # outer-validation observation for epoch or parameter selection.
        set_seed(seed)
        final_scaler = common.FoldPreprocessor().fit(outer_train.x)
        x_train = final_scaler.transform(outer_train.x); y_train = common.standardized_delta_target(outer_train, final_scaler)
        x_val = final_scaler.transform(validation.x)
        model = CanonicalStrictHistoryTFT(**model_args).to(dev)
        if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
        started = time.perf_counter()
        _, refit_value = _fit_epochs(model, x_train, y_train, None, None, seed=seed, cfg=cfg, dev=dev, stage="outer_refit", fold=fold, records=curve_records, epochs=inner_epoch, patience=None)
        duration = time.perf_counter() - started
        pred_std, importance = _predict(model, x_val, int(training["batch_size"]), dev)
        pred_v = pred_std * final_scaler.voltage_scale
        metrics = common.delta_metrics(validation.y_delta_V, pred_v, validation.target_row)
        peak = float(torch.cuda.max_memory_allocated(dev) / (1024**2)) if dev.type == "cuda" else 0.0
        all_metrics.append({"model": "Canonical_Architecture_TFT_Strict_History", "fold": fold, "seed": seed, "parameter_count": count_parameters(model), "selected_epoch": int(inner_epoch), "inner_stopper_RMSE_mV": float(inner_rmse * inner_scaler.voltage_scale * 1000.0), "outer_refit_standardized_RMSE": refit_value, "training_time_s": duration, "peak_gpu_memory_MiB": peak, "input_dimension": 14, "decoder_actual_covariates_used": False, "test_accessed": False, **metrics})
        for q in range(len(validation.origin_row)):
            for h in range(24):
                all_predictions.append({"fold": fold, "seed": seed, "origin_row": int(validation.origin_row[q]), "target_row": int(validation.target_row[q, h]), "horizon_step": h + 1, "origin_age_h": float(validation.origin_age_h[q]), "target_end_age_h": float(validation.target_end_age_h[q]), "anchor_voltage_V": float(validation.anchor_V[q]), "y_true_delta_V": float(validation.y_delta_V[q, h]), "y_pred_delta_V": float(pred_v[q, h]), "absolute_error_mV": float(abs(pred_v[q, h] - validation.y_delta_V[q, h]) * 1000.0)})
        variables, attention = _interpretability_rows(fold, importance); variable_rows.extend(variables); attention_rows.extend(attention)
        np.savez_compressed(PHASE / "outputs" / "interpretability" / f"{fold}_seed42.npz", encoder_variables=importance["encoder_variables"], decoder_variables=importance["decoder_variables"], encoder_attention=importance["encoder_attention"], decoder_attention=importance["decoder_attention"], validation_origin_row=validation.origin_row)
        np.savez_compressed(PHASE / "outputs" / "scalers" / f"{fold}_seed42.npz", medians=final_scaler.medians, mean=final_scaler.mean, scale=final_scaler.scale)
        torch.save({"state_dict": model.state_dict(), "model_args": model_args, "protocol_sha256": protocol_sha, "fold": fold, "seed": seed}, PHASE / "checkpoints" / f"canonical_tft_{fold}_seed42.pt")
        run_records.append({"fold": fold, "inner_fit_origins": len(fit.origin_row), "inner_stopper_origins": len(stopper.origin_row), "outer_train_origins": len(outer_train.origin_row), "outer_validation_origins": len(validation.origin_row), "all_legal_validation_origins_retained": True, "target_row_isolation": True, "max_validation_target_age_h": float(validation.target_end_age_h.max()), "test_accessed": False})
    metrics_df = pd.DataFrame(all_metrics); predictions_df = pd.DataFrame(all_predictions)
    write_csv(metrics_df, "outputs", "TFT_SEED42_PER_FOLD_RESULTS.csv"); write_csv(predictions_df, "outputs", "TFT_SEED42_PREDICTIONS.csv"); write_csv(pd.DataFrame(curve_records), "outputs", "TFT_TRAINING_CURVES.csv"); write_csv(pd.DataFrame(variable_rows), "outputs", "TFT_VARIABLE_SELECTION_WEIGHTS.csv"); write_csv(pd.DataFrame(attention_rows), "outputs", "TFT_ATTENTION_SUMMARY.csv"); write_csv(pd.DataFrame(run_records), "outputs", "TFT_RUN_AUDIT.csv")
    summary = {"phase": cfg["phase"], "protocol_sha256": protocol_sha, "model": "Canonical_Architecture_TFT_Strict_History", "seed": seed, "three_fold_equal_RMSE_mV": float(metrics_df.rmse_mV.mean()), "three_fold_equal_MAE_mV": float(metrics_df.mae_mV.mean()), "fold_RMSE_mV": {row.fold: float(row.rmse_mV) for row in metrics_df.itertuples()}, "data_firewall": firewall, "test_accessed": False, "future_actual_covariates_used": False, "static_covariates_used": False, "training_complete": True}
    write_json(summary, "outputs", "T0_BASELINE_SUMMARY.json")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

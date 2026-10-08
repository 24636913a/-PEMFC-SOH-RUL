"""Shared nested-training implementation for preregistered POT M3/M4 runs."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pot_common import (
    PHASE, POTFailure, TimesNetVoltageAdapter, augment_emd, device, git_head,
    load_emd_cache_additions, model_delta_prediction, parameter_count, set_seed,
)


METHODS = {
    "Optuna_RawF2_TimesNet": {"input_dimension": 14, "uses_emd": False},
    "Optuna_EMDFeature_TimesNet": {"input_dimension": 16, "uses_emd": True},
}


@dataclass(frozen=True)
class HyperParameters:
    learning_rate: float
    weight_decay: float
    dropout: float
    batch_size: int


def method_windows(method: str, raw_windows: Any) -> Any:
    if method not in METHODS:
        raise POTFailure("METHOD_NOT_PREREGISTERED")
    if not METHODS[method]["uses_emd"]:
        return raw_windows
    return augment_emd(raw_windows, load_emd_cache_additions(raw_windows))


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.float32))),
        batch_size=int(batch_size), shuffle=True, generator=generator, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def predict_delta(model: nn.Module, x: np.ndarray, dev: torch.device) -> np.ndarray:
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            bx = torch.from_numpy(np.asarray(x[start:start + 256], dtype=np.float32)).to(dev)
            pieces.append(model_delta_prediction(model, bx).detach().cpu().numpy())
    output = np.concatenate(pieces, axis=0)
    if output.shape != (len(x), 24) or not np.isfinite(output).all():
        raise POTFailure("PREDICTION_CONTRACT_OR_FINITE_FAIL")
    return output


def _one_epoch(model: nn.Module, opt: torch.optim.Optimizer, x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, dev: torch.device, clip: float) -> float:
    model.train()
    losses: list[float] = []
    for bx, by in loader(x, y, batch_size, seed):
        bx, by = bx.to(dev, non_blocking=True), by.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        pred = model_delta_prediction(model, bx)
        loss = torch.mean((pred - by) ** 2)
        if not torch.isfinite(loss):
            raise POTFailure("NONFINITE_TRAINING_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip))
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise POTFailure("NONFINITE_TRAINING_GRADIENT")
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def select_inner_epoch(method: str, fit: Any, stopper: Any, hp: HyperParameters, run_seed: int, bs_cfg: dict[str, Any], common: Any, max_epochs: int, patience: int, clip: float) -> dict[str, Any]:
    """Train on the chronology-safe inner fit population only."""
    set_seed(run_seed)
    dev = device()
    scaler = common.FoldPreprocessor().fit(fit.x)
    x_fit, x_stop = scaler.transform(fit.x), scaler.transform(stopper.x)
    y_fit = common.standardized_delta_target(fit, scaler)
    model = TimesNetVoltageAdapter(bs_cfg, METHODS[method]["input_dimension"], hp.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    best_epoch, best_rmse, wait = 0, float("inf"), 0
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, int(max_epochs) + 1):
        train_mse = _one_epoch(model, opt, x_fit, y_fit, hp.batch_size, run_seed + epoch, dev, clip)
        pred = predict_delta(model, x_stop, dev) * scaler.voltage_scale
        metrics = common.delta_metrics(stopper.y_delta_V, pred, stopper.target_row)
        rmse = float(metrics["rmse_mV"])
        rows.append({"epoch": epoch, "train_standardized_delta_mse": train_mse, "inner_deltaV_RMSE_mV": rmse})
        if rmse < best_rmse - 1e-12:
            best_epoch, best_rmse, wait = epoch, rmse, 0
        else:
            wait += 1
        if wait >= int(patience):
            break
    elapsed = time.perf_counter() - started
    if best_epoch <= 0 or not np.isfinite(best_rmse):
        raise POTFailure("INNER_STOPPER_SELECTION_FAIL")
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"selected_epoch": int(best_epoch), "best_inner_RMSE_mV": float(best_rmse), "epochs_ran": int(len(rows)), "inner_time_s": float(elapsed), "inner_log": rows}


def _checkpoint_paths(method: str, fold: str, seed: int) -> tuple[Path, Path]:
    stem = f"pot_{method.lower()}_{fold}_seed{seed}"
    return PHASE / "checkpoints" / f"{stem}.pt", PHASE / "checkpoints" / f"{stem}.json"


def _signature(method: str, fold: str, seed: int, hp: HyperParameters, protocol_sha: str) -> dict[str, Any]:
    return {"method": method, "fold": fold, "seed": int(seed), "hyperparameters": asdict(hp), "protocol_sha256": protocol_sha, "git_head_at_start": git_head()}


def formal_refit_and_predict(method: str, fold: str, outer_train: Any, outer_validation: Any, hp: HyperParameters, seed: int, bs_cfg: dict[str, Any], common: Any, protocol_sha: str, max_epochs: int, patience: int, clip: float) -> tuple[dict[str, Any], np.ndarray]:
    """Select with inner data, then reinitialize/refit on the full outer train."""
    train = method_windows(method, outer_train)
    validation = method_windows(method, outer_validation)
    fit, stopper = common.split_inner_fit_stopper(train, 0.15)
    if set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)):
        raise POTFailure("TARGET_ROW_ISOLATION_FAIL")
    signature = _signature(method, fold, seed, hp, protocol_sha)
    checkpoint, checkpoint_json = _checkpoint_paths(method, fold, seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    dev = device()
    if checkpoint.exists() and checkpoint_json.exists():
        saved_signature = json.loads(checkpoint_json.read_text(encoding="utf-8"))
        if {key: saved_signature[key] for key in signature} == signature and bool(saved_signature.get("complete")):
            state = torch.load(checkpoint, map_location=dev, weights_only=False)
            model = TimesNetVoltageAdapter(bs_cfg, METHODS[method]["input_dimension"], hp.dropout).to(dev)
            model.load_state_dict(state["model_state_dict"])
            scaler = common.FoldPreprocessor()
            scaler.mean = np.asarray(state["preprocessor"]["mean"])
            scaler.scale = np.asarray(state["preprocessor"]["scale"])
            scaler.medians = np.asarray(state["preprocessor"]["medians"])
            prediction = predict_delta(model, scaler.transform(validation.x), dev) * scaler.voltage_scale
            record = dict(state["record"])
            record["resumed_from_checkpoint"] = True
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            return record, prediction
    selected = select_inner_epoch(method, fit, stopper, hp, int(seed), bs_cfg, common, max_epochs, patience, clip)
    set_seed(int(seed))
    scaler = common.FoldPreprocessor().fit(train.x)
    x_train, x_valid = scaler.transform(train.x), scaler.transform(validation.x)
    y_train = common.standardized_delta_target(train, scaler)
    model = TimesNetVoltageAdapter(bs_cfg, METHODS[method]["input_dimension"], hp.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    for epoch in range(1, int(selected["selected_epoch"]) + 1):
        _one_epoch(model, opt, x_train, y_train, hp.batch_size, int(seed) + epoch, dev, clip)
    outer_time = time.perf_counter() - started
    prediction = predict_delta(model, x_valid, dev) * scaler.voltage_scale
    metrics = common.delta_metrics(validation.y_delta_V, prediction, validation.target_row)
    peak = float(torch.cuda.max_memory_allocated(dev) / (1024 ** 2)) if dev.type == "cuda" else float("nan")
    if np.any(validation.target_end_age_h >= 1819.997738):
        raise POTFailure("TEST_FIREWALL_FAIL")
    record = {
        "run_id": checkpoint.stem, "model": method, "fold": fold, "seed": int(seed), "status": "complete",
        "selected_epoch": int(selected["selected_epoch"]), "inner_fit_origins": len(fit.origin_row), "inner_stopper_origins": len(stopper.origin_row),
        "inner_target_overlap_count": 0, "parameter_count": parameter_count(model), "peak_gpu_memory_MiB": peak,
        "inner_time_s": selected["inner_time_s"], "outer_refit_time_s": outer_time,
        "total_inner_plus_refit_time_s": selected["inner_time_s"] + outer_time,
        "max_target_end_age_h": float(np.max(validation.target_end_age_h)), "input_dimension": METHODS[method]["input_dimension"],
        "future_covariates_used": False, "physics_loss_used": False, "test_accessed": False,
        "all_legal_validation_origins_retained": True, **metrics,
    }
    state = {
        "signature": signature,
        "selected": selected,
        "model_state_dict": model.state_dict(),
        "preprocessor": {"mean": scaler.mean, "scale": scaler.scale, "medians": scaler.medians},
        "record": record,
    }
    torch.save(state, checkpoint)
    checkpoint_json.write_text(json.dumps({**signature, "selected_epoch": selected["selected_epoch"], "complete": True}, indent=2) + "\n", encoding="utf-8")
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return record, prediction

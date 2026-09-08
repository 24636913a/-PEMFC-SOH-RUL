"""Frozen LSTM/official TimesNet wrappers and Development-only training for Phase N2.7."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from phase_n2_7_common import (
    FoldPreprocessor,
    PHASE_ROOT,
    WindowSet,
    ensure_output,
    metric_dict,
    set_seed,
    verify_prediction_arrays,
    write_csv,
    write_json,
)


sys.dont_write_bytecode = True
PROJECT_ROOT = PHASE_ROOT.parent
N26R_SCRIPTS = PROJECT_ROOT / "Phase_N2_6R_High_Fidelity_Revalidation/scripts"
if str(N26R_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(N26R_SCRIPTS))

import phase_n2_6r_models as n26r_models  # noqa: E402


# The historical Phase N2.6R directory is read-only.  Its archived, hash-verified
# official source is restored under this phase so imports and any Python caches
# cannot write back into the older phase.
N27_TIMESNET_ROOT = PHASE_ROOT / "vendor" / "Time-Series-Library-4e938a1767106324dd753b2a44832bf870a0252e"
if not (N27_TIMESNET_ROOT / "models" / "TimesNet.py").is_file():
    raise RuntimeError(f"Phase N2.7 TimesNet source is missing: {N27_TIMESNET_ROOT}")
n26r_models.TIMESNET_ROOT = N27_TIMESNET_ROOT
n26r_models._TIMESNET_MODULE = None
PEMFCTimesNetWrapper = n26r_models.PEMFCTimesNetWrapper
StandardLSTMForecaster = n26r_models.StandardLSTMForecaster


def create_model(family: str, cfg: dict[str, Any], dropout: float) -> nn.Module:
    if family == "LSTM":
        if not math.isclose(dropout, 0.1):
            raise RuntimeError("Frozen LSTM dropout cannot change")
        return StandardLSTMForecaster(14, 24)
    if family == "TimesNet":
        spec = dict(cfg["timesnet_architecture"])
        spec["dropout"] = float(dropout)
        return PEMFCTimesNetWrapper(14, 24, spec)
    raise KeyError(family)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def load_local_run(run_id: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    result_path = PHASE_ROOT / "outputs/model_runs" / f"{run_id}.json"
    prediction_path = PHASE_ROOT / "outputs/model_runs" / f"{run_id}_predictions.npz"
    if not result_path.is_file() or not prediction_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(prediction_path, allow_pickle=False))
    if result.get("status") != "complete":
        return None
    verify_prediction_arrays(arrays, cfg, run_id)
    return result, arrays


def predict_scaled(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            outputs.append(model(batch_x.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def target_setup(
    family: str,
    train: WindowSet,
    preprocessor: FoldPreprocessor,
) -> tuple[np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray], dict[str, Any]]:
    if family == "LSTM":
        target_mean = np.mean(train.y_delta, axis=0, keepdims=True)
        target_std = np.std(train.y_delta, axis=0, keepdims=True)
        target_std = np.where(target_std > 1e-6, target_std, 1.0)
        y_train = ((train.y_delta - target_mean) / target_std).astype(np.float32)

        def inverse(values: np.ndarray, anchor: np.ndarray) -> np.ndarray:
            return anchor[:, None] + values * target_std + target_mean

        return y_train, inverse, {
            "target_training_space": "standardized_raw_delta_voltage",
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
        }

    voltage_mean = float(preprocessor.scaler.mean_[0])
    voltage_scale = float(preprocessor.scaler.scale_[0])
    if not math.isfinite(voltage_scale) or voltage_scale <= 0:
        raise RuntimeError("Invalid fold-Train voltage scale")
    y_train = ((train.y_abs - voltage_mean) / voltage_scale).astype(np.float32)

    def inverse(values: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        del anchor
        return values * voltage_scale + voltage_mean

    return y_train, inverse, {
        "target_training_space": "scaled_raw_absolute_voltage_with_official_internal_normalization",
        "voltage_feature_mean": voltage_mean,
        "voltage_feature_scale": voltage_scale,
    }


def train_model_run(
    run_id: str,
    family: str,
    train: WindowSet,
    validation: WindowSet,
    cfg: dict[str, Any],
    seed: int,
    training_spec: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cached = load_local_run(run_id, cfg)
    if cached is not None:
        return cached

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = FoldPreprocessor().fit(train.x)
    x_train = preprocessor.transform(train.x)
    x_validation = preprocessor.transform(validation.x)
    y_train, inverse, target_metadata = target_setup(family, train, preprocessor)
    dropout = 0.1 if family == "LSTM" else float(training_spec["dropout"])
    model = create_model(family, cfg, dropout).to(device)
    count = parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_spec["learning_rate"]),
        weight_decay=float(training_spec["weight_decay"]),
    )
    if training_spec["loss"] == "MSE":
        criterion: nn.Module = nn.MSELoss()
    elif training_spec["loss"] == "Huber":
        criterion = nn.HuberLoss(delta=float(training_spec["huber_delta"]))
    else:
        raise KeyError(training_spec["loss"])

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_rmse = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    epoch_rows: list[dict[str, float | int]] = []
    epoch_times: list[float] = []
    start_total = time.perf_counter()

    for epoch in range(1, int(cfg["training"]["max_epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_epoch = time.perf_counter()
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            if prediction.shape != batch_y.shape:
                raise RuntimeError(f"Output shape mismatch: {run_id}")
            loss = criterion(prediction, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {run_id}")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            if not torch.isfinite(norm):
                raise RuntimeError(f"Non-finite gradient: {run_id}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        scaled_validation = predict_scaled(model, x_validation, device)
        prediction_abs = inverse(scaled_validation, validation.anchor)
        metrics = metric_dict(validation.y_abs, prediction_abs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_epoch
        epoch_times.append(elapsed)
        epoch_rows.append({
            "epoch": epoch,
            "train_scaled_loss": float(np.mean(losses)),
            "validation_rmse_V": metrics["rmse_V"],
            "validation_mae_V": metrics["mae_V"],
            "validation_r2": metrics["r2"],
            "epoch_time_s": elapsed,
        })
        if epoch % 50 == 0:
            print(json.dumps({"run_id": run_id, "epoch": epoch, "validation_rmse_mV": metrics["rmse_mV"]}), flush=True)
        if metrics["rmse_V"] < best_rmse - float(cfg["training"]["min_delta"]):
            best_rmse = metrics["rmse_V"]
            best_epoch = epoch
            patience = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
        if patience >= int(cfg["training"]["early_stopping_patience"]):
            break

    total_time = time.perf_counter() - start_total
    if best_state is None:
        raise RuntimeError(f"No valid model state: {run_id}")
    model.load_state_dict(best_state)
    scaled_validation = predict_scaled(model, x_validation, device)
    prediction_abs = inverse(scaled_validation, validation.anchor)
    metrics = metric_dict(validation.y_abs, prediction_abs)
    result = {
        "run_id": run_id,
        "family": family,
        "seed": seed,
        "fold": metadata["fold"],
        "status": "complete",
        "best_epoch": best_epoch,
        "epochs_ran": len(epoch_rows),
        "parameter_count": count,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "mean_epoch_time_s": float(np.mean(epoch_times)),
        "total_training_time_s": total_time,
        "device": str(device),
        "torch_version": torch.__version__,
        "loss": training_spec["loss"],
        "learning_rate": float(training_spec["learning_rate"]),
        "weight_decay": float(training_spec["weight_decay"]),
        "dropout": dropout,
        "huber_delta": training_spec.get("huber_delta"),
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "test_shift_diagnostics_computed": False,
        **metadata,
        **target_metadata,
        **metrics,
    }
    write_csv(pd.DataFrame(epoch_rows), PHASE_ROOT / "logs" / f"{run_id}_epochs.csv")
    write_json(result, PHASE_ROOT / "outputs/model_runs" / f"{run_id}.json")
    prediction_path = ensure_output(PHASE_ROOT / "outputs/model_runs" / f"{run_id}_predictions.npz")
    np.savez_compressed(
        prediction_path,
        y_true=validation.y_abs,
        y_pred=prediction_abs,
        y_pred_delta=prediction_abs - validation.anchor[:, None],
        anchor=validation.anchor,
        origin_age_h=validation.origin_age_h,
        target_end_age_h=validation.target_end_age_h,
        row_origin_index=validation.row_origin_index,
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "family": family,
            "training_spec": training_spec,
            "feature_medians": preprocessor.medians,
            "feature_scaler_mean": preprocessor.scaler.mean_,
            "feature_scaler_scale": preprocessor.scaler.scale_,
            "target_metadata": target_metadata,
            "metadata": metadata,
        },
        ensure_output(PHASE_ROOT / "checkpoints" / f"{run_id}.pt"),
    )
    arrays = dict(np.load(prediction_path, allow_pickle=False))
    verify_prediction_arrays(arrays, cfg, run_id)
    print(json.dumps({"run_id": run_id, "rmse_mV": metrics["rmse_mV"], "mae_mV": metrics["mae_mV"], "r2": metrics["r2"], "best_epoch": best_epoch}, ensure_ascii=False), flush=True)
    return result, arrays

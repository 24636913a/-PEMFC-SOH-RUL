"""Auditable local PyTorch models and one shared training pipeline for Phase N2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from phase_n2_common import PHASE_ROOT, FoldPreprocessor, WindowSet, metric_dict, model_device, parameter_count, set_seed, write_csv, write_json


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, num_layers: int, dropout: float, **_: Any) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.head(y[:, -1])


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation), nn.GELU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x) + self.skip(x))


class TCNForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, num_layers: int, kernel_size: int, dropout: float, **_: Any) -> None:
        super().__init__()
        blocks = []
        channels = input_dim
        for i in range(num_layers):
            blocks.append(TCNBlock(channels, hidden_dim, kernel_size, 2**i, dropout))
            channels = hidden_dim
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tcn(x.transpose(1, 2)))


class ITransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, lookback: int, horizon: int, d_model: int, num_layers: int, nhead: int, dropout: float, **_: Any) -> None:
        super().__init__()
        self.temporal_projection = nn.Linear(lookback, d_model)
        self.variable_embedding = nn.Parameter(torch.zeros(1, input_dim, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.temporal_projection(x.transpose(1, 2)) + self.variable_embedding
        return self.head(self.encoder(tokens).mean(dim=1))


class TimeMixerForecaster(nn.Module):
    def __init__(self, input_dim: int, lookback: int, horizon: int, hidden_dim: int, scales: list[int], dropout: float, **_: Any) -> None:
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleList()
        for scale in scales:
            length = math.ceil(lookback / scale)
            self.branches.append(nn.Sequential(nn.Linear(length * input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
        self.mixer = nn.Sequential(nn.Linear(hidden_dim * len(scales), hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps = []
        for scale, branch in zip(self.scales, self.branches):
            if scale == 1:
                pooled = x
            else:
                pooled = torch.nn.functional.avg_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale, ceil_mode=True).transpose(1, 2)
            reps.append(branch(pooled.flatten(1)))
        return self.mixer(torch.cat(reps, dim=1))


class PatchTSTForecaster(nn.Module):
    def __init__(self, input_dim: int, lookback: int, horizon: int, d_model: int, num_layers: int, nhead: int, patch_len: int, stride: int, dropout: float, **_: Any) -> None:
        super().__init__()
        if lookback < patch_len:
            patch_len = lookback
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = 1 + (lookback - patch_len) // stride
        self.patch_projection = nn.Linear(input_dim * patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model * self.n_patches), nn.Linear(d_model * self.n_patches, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(1, self.patch_len, self.stride).permute(0, 1, 3, 2).flatten(2)
        tokens = self.patch_projection(patches) + self.position
        return self.head(self.encoder(tokens).flatten(1))


MODEL_CLASSES = {
    "LSTM": LSTMForecaster,
    "TCN": TCNForecaster,
    "iTransformer": ITransformerForecaster,
    "TimeMixer": TimeMixerForecaster,
    "PatchTST": PatchTSTForecaster,
}


def create_model(name: str, input_dim: int, lookback: int, horizon: int, architecture: dict[str, Any]) -> nn.Module:
    if name not in MODEL_CLASSES:
        raise KeyError(name)
    return MODEL_CLASSES[name](input_dim=input_dim, lookback=lookback, horizon=horizon, **architecture)


def load_run(run_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    result_path = PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}.json"
    pred_path = PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}_predictions.npz"
    if result_path.is_file() and pred_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8")), dict(np.load(pred_path, allow_pickle=False))
    return None


def train_model_run(
    run_id: str,
    model_name: str,
    train: WindowSet,
    val: WindowSet,
    architecture: dict[str, Any],
    training_cfg: dict[str, Any],
    seed: int,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cached = load_run(run_id)
    if cached is not None:
        return cached
    set_seed(seed)
    # Small deterministic Conv1d kernels are substantially faster on this host's
    # CPU than under deterministic cuDNN.  This changes only execution device,
    # not model, batch size, optimizer, epochs, origins or early stopping.
    device = torch.device("cpu") if model_name == "TCN" else model_device()
    pre = FoldPreprocessor().fit(train.x)
    x_train = pre.transform(train.x)
    x_val = pre.transform(val.x)
    target_mean = np.mean(train.y_delta, axis=0, keepdims=True)
    target_std = np.std(train.y_delta, axis=0, keepdims=True)
    target_std = np.where(target_std > 1e-6, target_std, 1.0)
    y_train_scaled = ((train.y_delta - target_mean) / target_std).astype(np.float32)
    y_val_scaled = ((val.y_delta - target_mean) / target_std).astype(np.float32)

    model = create_model(model_name, x_train.shape[-1], x_train.shape[1], y_train_scaled.shape[1], architecture).to(device)
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg["learning_rate"]), weight_decay=float(training_cfg["weight_decay"]))
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_scaled)), batch_size=int(training_cfg["batch_size"]), shuffle=True, generator=generator, num_workers=int(training_cfg.get("num_workers", 0)))
    val_x_tensor = torch.from_numpy(x_val).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_rmse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience = 0
    logs: list[dict[str, float | int]] = []
    import time
    total_start = time.perf_counter()
    epoch_times: list[float] = []
    for epoch in range(1, int(training_cfg["max_epochs"]) + 1):
        if device.type == "cuda": torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred_scaled = model(val_x_tensor).cpu().numpy()
        pred_delta = pred_scaled * target_std + target_mean
        pred_abs = val.anchor[:, None] + pred_delta
        val_rmse = metric_dict(val.y_abs, pred_abs)["rmse_V"]
        if device.type == "cuda": torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        logs.append({"epoch": epoch, "train_scaled_mse": float(np.mean(losses)), "validation_rmse_V": val_rmse, "epoch_time_s": epoch_time})
        if epoch % 50 == 0:
            print(json.dumps({"run_id": run_id, "epoch": epoch, "validation_rmse_mV": val_rmse * 1000.0}), flush=True)
        if val_rmse < best_rmse - float(training_cfg["min_delta"]):
            best_rmse, best_epoch, patience = val_rmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= int(training_cfg["early_stopping_patience"]):
            break
    total_time = time.perf_counter() - total_start
    if best_state is None:
        raise RuntimeError("No best state captured")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred_scaled = model(val_x_tensor).cpu().numpy()
    pred_delta = pred_scaled * target_std + target_mean
    pred_abs = val.anchor[:, None] + pred_delta
    metrics = metric_dict(val.y_abs, pred_abs)
    peak_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    result = {
        "run_id": run_id, "model": model_name, "seed": seed, "status": "complete", "best_epoch": best_epoch,
        "epochs_ran": len(logs), "parameter_count": params, "peak_gpu_memory_mb": peak_mb,
        "mean_epoch_time_s": float(np.mean(epoch_times)), "total_training_time_s": total_time,
        "device": str(device), "torch_version": torch.__version__, "implementation": "local_auditable_phase_n2_v1",
        **metadata, **metrics,
    }
    run_dir = PHASE_ROOT / "outputs" / "model_runs"; run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PHASE_ROOT / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = PHASE_ROOT / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(logs), log_dir / f"{run_id}_epochs.csv")
    write_json(result, run_dir / f"{run_id}.json")
    np.savez_compressed(run_dir / f"{run_id}_predictions.npz", y_true=val.y_abs, y_pred=pred_abs, anchor=val.anchor, origin_age_h=val.origin_age_h, target_end_age_h=val.target_end_age_h, row_origin_index=val.row_origin_index)
    torch.save({
        "model_state_dict": best_state, "model_name": model_name, "architecture": architecture,
        "feature_medians": pre.medians, "feature_scaler_mean": pre.scaler.mean_, "feature_scaler_scale": pre.scaler.scale_,
        "target_mean": target_mean, "target_std": target_std, "metadata": metadata,
    }, checkpoint_dir / f"{run_id}.pt")
    arrays = {"y_true": val.y_abs, "y_pred": pred_abs, "anchor": val.anchor, "origin_age_h": val.origin_age_h, "target_end_age_h": val.target_end_age_h, "row_origin_index": val.row_origin_index}
    print(json.dumps({"run_id": run_id, "rmse_mV": metrics["rmse_mV"], "best_epoch": best_epoch, "epochs": len(logs)}, ensure_ascii=False), flush=True)
    return result, arrays

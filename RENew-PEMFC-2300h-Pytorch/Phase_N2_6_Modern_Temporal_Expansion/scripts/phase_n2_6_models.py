"""Auditable Phase N2.6 model adapters and unified training loop."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from phase_n2_6_common import (
    PHASE_ROOT,
    FoldPreprocessor,
    WindowSet,
    ensure_output,
    load_local_run,
    metric_dict,
    model_device,
    set_seed,
    write_csv,
    write_json,
)


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
VENDOR = PHASE_ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


class StandardLSTMForecaster(nn.Module):
    """Exact Phase N2 incumbent architecture."""

    def __init__(self, input_dim: int, horizon: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1])


def fft_for_period(x: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Trainable-forward period discovery following the TimesNet FFT rule."""
    spectrum = torch.fft.rfft(x, dim=1)
    average_amplitude = torch.abs(spectrum).mean(0).mean(-1)
    average_amplitude = average_amplitude.clone()
    average_amplitude[0] = 0.0
    available = max(1, min(top_k, average_amplitude.numel() - 1))
    frequencies = torch.topk(average_amplitude, available).indices
    periods = torch.div(x.shape[1], frequencies, rounding_mode="floor").clamp_min(1)
    sample_weights = torch.abs(spectrum).mean(-1)[:, frequencies]
    return periods, sample_weights


class Inception2D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_kernels: int) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(input_dim, output_dim, kernel_size=2 * index + 1, padding=index)
                for index in range(num_kernels)
            ]
        )
        for layer in self.kernels:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([layer(x) for layer in self.kernels], dim=-1).mean(dim=-1)


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, top_k: int, num_kernels: int, dropout: float) -> None:
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            Inception2D(d_model, d_ff, num_kernels),
            nn.GELU(),
            nn.Dropout(dropout),
            Inception2D(d_ff, d_model, num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        periods, weights = fft_for_period(x, self.top_k)
        outputs: list[torch.Tensor] = []
        for period_tensor in periods:
            period = int(period_tensor.item())
            padded_length = math.ceil(length / period) * period
            if padded_length > length:
                padding = torch.zeros(batch, padded_length - length, channels, device=x.device, dtype=x.dtype)
                values = torch.cat([x, padding], dim=1)
            else:
                values = x
            values = values.reshape(batch, padded_length // period, period, channels).permute(0, 3, 1, 2)
            values = self.conv(values)
            values = values.permute(0, 2, 3, 1).reshape(batch, padded_length, channels)
            outputs.append(values[:, :length])
        stacked = torch.stack(outputs, dim=-1)
        normalized_weights = torch.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)
        return x + torch.sum(stacked * normalized_weights, dim=-1)


class TimesNetForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, spec: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(spec["d_model"])
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(float(spec["dropout"]))
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model,
                    int(spec["d_ff"]),
                    int(spec["top_k"]),
                    int(spec["num_kernels"]),
                    float(spec["dropout"]),
                )
                for _ in range(int(spec["e_layers"]))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in self.blocks])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.embedding(x))
        for block, norm in zip(self.blocks, self.norms):
            hidden = norm(block(hidden))
        return self.head(hidden[:, -1])


class ModernTCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.norm = nn.LayerNorm(d_model)
        self.channel_mixing = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.depthwise(x)
        mixed = self.channel_mixing(self.norm(temporal.transpose(1, 2))).transpose(1, 2)
        return x + mixed


class ModernTCNForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, spec: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(spec["d_model"])
        self.patch_embedding = nn.Conv1d(
            input_dim,
            d_model,
            kernel_size=int(spec["patch_length"]),
            stride=int(spec["stride"]),
        )
        self.blocks = nn.Sequential(
            *[
                ModernTCNBlock(
                    d_model,
                    int(spec["large_kernel"]),
                    int(spec["expansion"]),
                    float(spec["dropout"]),
                )
                for _ in range(int(spec["num_blocks"]))
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.patch_embedding(x.transpose(1, 2)))
        return self.head(hidden.mean(dim=-1))


class PatchTSMixerForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, spec: dict[str, Any]) -> None:
        super().__init__()
        from transformers import PatchTSMixerConfig, PatchTSMixerModel

        d_model = int(spec["d_model"])
        config = PatchTSMixerConfig(
            context_length=48,
            patch_length=int(spec["patch_length"]),
            patch_stride=int(spec["stride"]),
            num_input_channels=input_dim,
            d_model=d_model,
            num_layers=int(spec["num_layers"]),
            expansion_factor=int(spec["expansion"]),
            dropout=float(spec["dropout"]),
            mode=str(spec["mode"]),
            gated_attn=bool(spec["gated_attention"]),
            self_attn=bool(spec["self_attention"]),
            scaling=None,
            post_init=True,
        )
        self.backbone = PatchTSMixerModel(config)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        observed = torch.ones_like(x)
        hidden = self.backbone(past_values=x, observed_mask=observed, return_dict=True).last_hidden_state
        state = hidden.mean(dim=(1, 2))
        return self.head(state)


class CompactMambaForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, spec: dict[str, Any]) -> None:
        super().__init__()
        from transformers import MambaConfig, MambaModel

        d_model = int(spec["d_model"])
        self.input_projection = nn.Linear(input_dim, d_model)
        config = MambaConfig(
            hidden_size=d_model,
            state_size=int(spec["state_size"]),
            num_hidden_layers=int(spec["num_layers"]),
            conv_kernel=int(spec["conv_kernel"]),
            expand=int(spec["expand"]),
            vocab_size=1,
            use_mambapy=True,
            use_cache=False,
        )
        self.backbone = MambaModel(config)
        self.dropout = nn.Dropout(float(spec["dropout"]))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(x)
        hidden = self.backbone(inputs_embeds=embedded, use_cache=False, return_dict=True).last_hidden_state
        return self.head(self.dropout(hidden[:, -1]))


def create_model(config_name: str, input_dim: int, horizon: int, cfg: dict[str, Any]) -> nn.Module:
    if config_name == "LSTM":
        return StandardLSTMForecaster(input_dim, horizon)
    spec = cfg["architectures"][config_name]
    family = str(spec["family"])
    if family == "TimesNet":
        return TimesNetForecaster(input_dim, horizon, spec)
    if family == "ModernTCN":
        return ModernTCNForecaster(input_dim, horizon, spec)
    if family == "PatchTSMixer":
        return PatchTSMixerForecaster(input_dim, horizon, spec)
    if family == "CompactMamba":
        return CompactMambaForecaster(input_dim, horizon, spec)
    raise KeyError(f"Unknown model family: {family}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def train_model_run(
    run_id: str,
    config_name: str,
    train: WindowSet,
    validation: WindowSet,
    cfg: dict[str, Any],
    seed: int,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cached = load_local_run(run_id)
    if cached is not None:
        return cached
    set_seed(seed)
    device = model_device()
    preprocessor = FoldPreprocessor().fit(train.x)
    x_train = preprocessor.transform(train.x)
    x_validation = preprocessor.transform(validation.x)
    target_mean = np.mean(train.y_delta, axis=0, keepdims=True)
    target_std = np.std(train.y_delta, axis=0, keepdims=True)
    target_std = np.where(target_std > 1e-6, target_std, 1.0)
    y_train = ((train.y_delta - target_mean) / target_std).astype(np.float32)

    model = create_model(config_name, x_train.shape[-1], y_train.shape[-1], cfg).to(device)
    count = parameter_count(model)
    if config_name != "LSTM" and not 50_000 <= count <= 200_000:
        raise RuntimeError(f"Parameter count outside frozen range for {config_name}: {count}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    validation_tensor = torch.from_numpy(x_validation).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_rmse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
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
                raise RuntimeError(f"Output shape mismatch in {run_id}: {prediction.shape} vs {batch_y.shape}")
            loss = criterion(prediction, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss in {run_id}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite gradient in {run_id}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            scaled_prediction = model(validation_tensor).cpu().numpy()
        prediction_delta = scaled_prediction * target_std + target_mean
        prediction_abs = validation.anchor[:, None] + prediction_delta
        validation_rmse = metric_dict(validation.y_abs, prediction_abs)["rmse_V"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_epoch
        epoch_times.append(elapsed)
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_scaled_mse": float(np.mean(losses)),
                "validation_rmse_V": validation_rmse,
                "epoch_time_s": elapsed,
            }
        )
        if epoch % 50 == 0:
            print(json.dumps({"run_id": run_id, "epoch": epoch, "validation_rmse_mV": validation_rmse * 1000}), flush=True)
        if validation_rmse < best_rmse - float(cfg["training"]["min_delta"]):
            best_rmse = validation_rmse
            best_epoch = epoch
            patience = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        if patience >= int(cfg["training"]["early_stopping_patience"]):
            break
    total_time = time.perf_counter() - start_total
    if best_state is None:
        raise RuntimeError(f"No finite state captured for {run_id}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scaled_prediction = model(validation_tensor).cpu().numpy()
    prediction_delta = scaled_prediction * target_std + target_mean
    prediction_abs = validation.anchor[:, None] + prediction_delta
    if not np.isfinite(prediction_abs).all():
        raise RuntimeError(f"Non-finite inverse-transformed predictions in {run_id}")
    metrics = metric_dict(validation.y_abs, prediction_abs)
    peak_memory = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    result = {
        "run_id": run_id,
        "config_name": config_name,
        "seed": seed,
        "status": "complete",
        "best_epoch": best_epoch,
        "epochs_ran": len(epoch_rows),
        "parameter_count": count,
        "peak_gpu_memory_mb": peak_memory,
        "mean_epoch_time_s": float(np.mean(epoch_times)),
        "total_training_time_s": total_time,
        "device": str(device),
        "torch_version": torch.__version__,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        **metadata,
        **metrics,
    }
    write_csv(pd.DataFrame(epoch_rows), PHASE_ROOT / "logs" / f"{run_id}_epochs.csv")
    write_json(result, PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}.json")
    ensure_output(PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}_predictions.npz")
    np.savez_compressed(
        PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}_predictions.npz",
        y_true=validation.y_abs,
        y_pred=prediction_abs,
        anchor=validation.anchor,
        origin_age_h=validation.origin_age_h,
        target_end_age_h=validation.target_end_age_h,
        row_origin_index=validation.row_origin_index,
    )
    ensure_output(PHASE_ROOT / "checkpoints" / f"{run_id}.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "config_name": config_name,
            "feature_medians": preprocessor.medians,
            "feature_scaler_mean": preprocessor.scaler.mean_,
            "feature_scaler_scale": preprocessor.scaler.scale_,
            "target_mean": target_mean,
            "target_std": target_std,
            "metadata": metadata,
        },
        PHASE_ROOT / "checkpoints" / f"{run_id}.pt",
    )
    arrays = {
        "y_true": validation.y_abs,
        "y_pred": prediction_abs,
        "anchor": validation.anchor,
        "origin_age_h": validation.origin_age_h,
        "target_end_age_h": validation.target_end_age_h,
        "row_origin_index": validation.row_origin_index,
    }
    print(json.dumps({"run_id": run_id, "rmse_mV": metrics["rmse_mV"], "best_epoch": best_epoch, "epochs": len(epoch_rows)}), flush=True)
    return result, arrays

"""Local auditable xLSTM-style and cycle-aware PyTorch models for Phase N2.5."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from phase_n2_5_common import (
    PHASE_ROOT,
    FoldPreprocessor,
    WindowSet,
    metric_dict,
    model_device,
    set_seed,
    write_csv,
    write_json,
)


class StandardLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return sequence


class StandardLSTMForecaster(nn.Module):
    """Exact Phase N2 reference architecture."""

    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder = StandardLSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)[:, -1])


class ScalarMemoryBlock(nn.Module):
    """Stabilized exponential-gated scalar memory with residual mixing."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.gates = nn.Linear(dimension, 4 * dimension)
        self.dropout = nn.Dropout(dropout)
        self.norm_memory = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dimension, dimension),
        )
        self.norm_output = nn.LayerNorm(dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate, input_logit, forget_logit, output_logit = self.gates(x).chunk(4, dim=-1)
        candidate = torch.tanh(candidate)
        forget_log = F.logsigmoid(forget_logit)
        output_gate = torch.sigmoid(output_logit)
        length = x.shape[1]
        cumulative_forget = torch.cumsum(forget_log, dim=1)
        # Query-time t, source-time k log contribution:
        # i_k + sum_{j=k+1..t} log(f_j).  A causal softmax is the
        # log-sum-exp-stabilized form of the scalar numerator/normalizer memory.
        log_weight = (
            input_logit.unsqueeze(1)
            + cumulative_forget.unsqueeze(2)
            - cumulative_forget.unsqueeze(1)
        )
        causal_mask = torch.tril(torch.ones(length, length, device=x.device, dtype=torch.bool))
        log_weight = log_weight.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(-1), -1.0e30)
        weight = torch.softmax(log_weight, dim=2)
        memory = torch.sum(weight * candidate.unsqueeze(1), dim=2) * output_gate
        mixed = self.norm_memory(x + self.dropout(memory))
        return self.norm_output(mixed + self.dropout(self.feed_forward(mixed)))


class MatrixMemoryBlock(nn.Module):
    """Parallel multi-head matrix-memory read with covariance-style updates."""

    def __init__(self, dimension: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dimension % num_heads:
            raise ValueError("dimension must be divisible by num_heads")
        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.gates = nn.Linear(dimension, 2 * num_heads + dimension)
        self.output_projection = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)
        self.norm_memory = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dimension, dimension),
        )
        self.norm_output = nn.LayerNorm(dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = query.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        gate_values = self.gates(x)
        input_logit = gate_values[..., : self.num_heads].transpose(1, 2)
        forget_log = F.logsigmoid(
            gate_values[..., self.num_heads : 2 * self.num_heads]
        ).transpose(1, 2)
        output_gate = torch.sigmoid(gate_values[..., 2 * self.num_heads :])

        cumulative_forget = torch.cumsum(forget_log, dim=-1)
        log_decay = (
            input_logit.unsqueeze(-2)
            + cumulative_forget.unsqueeze(-1)
            - cumulative_forget.unsqueeze(-2)
        )
        causal_mask = torch.tril(torch.ones(length, length, device=x.device, dtype=torch.bool))
        log_decay = log_decay.masked_fill(~causal_mask, -1.0e30)
        row_max = log_decay.max(dim=-1, keepdim=True).values
        decay = torch.exp(torch.clamp(log_decay - row_max, min=-30.0, max=0.0)) * causal_mask
        content = torch.einsum("bhtd,bhsd->bhts", query, key) / math.sqrt(self.head_dim)
        signed_weight = decay * content
        numerator = torch.einsum("bhts,bhsd->bhtd", signed_weight, value)
        denominator = signed_weight.sum(dim=-1, keepdim=True).abs().clamp_min(1.0)
        memory = (numerator / denominator).transpose(1, 2).reshape(batch, length, self.dimension)
        memory = output_gate * self.output_projection(memory)
        mixed = self.norm_memory(x + self.dropout(memory))
        return self.norm_output(mixed + self.dropout(self.feed_forward(mixed)))


class SLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ScalarMemoryBlock(hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.input_projection(x)
        for block in self.blocks:
            sequence = block(sequence)
        return sequence


class MLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [MatrixMemoryBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.input_projection(x)
        for block in self.blocks:
            sequence = block(sequence)
        return sequence


class HybridXLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.scalar_block = ScalarMemoryBlock(hidden_dim, dropout)
        self.matrix_block = MatrixMemoryBlock(hidden_dim, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.matrix_block(self.scalar_block(self.input_projection(x)))


class EncoderForecaster(nn.Module):
    def __init__(self, encoder: nn.Module, horizon: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.LayerNorm(encoder.output_dim), nn.Linear(encoder.output_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)[:, -1])


class CycleAwareForecaster(nn.Module):
    def __init__(self, encoder: nn.Module, horizon: int, period_steps: int, attention_hidden_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.period_steps = period_steps
        dimension = encoder.output_dim
        self.attention_scorer = nn.Sequential(
            nn.Linear(dimension, attention_hidden_dim), nn.Tanh(), nn.Linear(attention_hidden_dim, 1)
        )
        self.fusion_gate = nn.Linear(2 * dimension, dimension)
        self.head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, horizon))

    def _fuse(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = sequence.shape[1]
        if 2 * self.period_steps >= length:
            raise RuntimeError("Cycle lags do not fit inside the lookback")
        states = torch.stack(
            [
                sequence[:, -1],
                sequence[:, -1 - self.period_steps],
                sequence[:, -1 - 2 * self.period_steps],
            ],
            dim=1,
        )
        weights = torch.softmax(self.attention_scorer(states).squeeze(-1), dim=1)
        context = torch.sum(weights.unsqueeze(-1) * states, dim=1)
        current = states[:, 0]
        gate = torch.sigmoid(self.fusion_gate(torch.cat([current, context], dim=-1)))
        fused = gate * current + (1.0 - gate) * context
        return fused, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused, _ = self._fuse(self.encoder(x))
        return self.head(fused)

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused, weights = self._fuse(self.encoder(x))
        return self.head(fused), weights


def create_encoder(name: str, input_dim: int, architecture: dict[str, Any]) -> nn.Module:
    if name == "LSTM":
        return StandardLSTMEncoder(input_dim, int(architecture["hidden_dim"]), int(architecture["num_layers"]), float(architecture["dropout"]))
    if name == "sLSTM":
        return SLSTMEncoder(input_dim, int(architecture["hidden_dim"]), int(architecture["num_layers"]), float(architecture["dropout"]))
    if name == "mLSTM":
        return MLSTMEncoder(input_dim, int(architecture["hidden_dim"]), int(architecture["num_layers"]), int(architecture["num_heads"]), float(architecture["dropout"]))
    if name == "Hybrid_xLSTM":
        return HybridXLSTMEncoder(input_dim, int(architecture["hidden_dim"]), int(architecture["num_heads"]), float(architecture["dropout"]))
    raise KeyError(name)


def create_model(
    name: str,
    input_dim: int,
    horizon: int,
    architectures: dict[str, Any],
    period_steps: int | None = None,
) -> nn.Module:
    if name == "LSTM" and period_steps is None:
        return StandardLSTMForecaster(input_dim, horizon, **architectures["LSTM"])
    encoder = create_encoder(name, input_dim, architectures[name])
    if period_steps is None:
        return EncoderForecaster(encoder, horizon)
    return CycleAwareForecaster(
        encoder,
        horizon,
        period_steps,
        int(architectures["cycle_attention"]["attention_hidden_dim"]),
    )


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def load_run(run_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    result_path = PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}.json"
    prediction_path = PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}_predictions.npz"
    if result_path.is_file() and prediction_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8")), dict(np.load(prediction_path, allow_pickle=False))
    return None


def train_model_run(
    run_id: str,
    model_name: str,
    train: WindowSet,
    validation: WindowSet,
    cfg: dict[str, Any],
    seed: int,
    metadata: dict[str, Any],
    period_steps: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cached = load_run(run_id)
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

    model = create_model(model_name, x_train.shape[-1], y_train.shape[-1], cfg["architectures"], period_steps).to(device)
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
            loss = criterion(prediction, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss in {run_id}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
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
        if isinstance(model, CycleAwareForecaster):
            scaled_prediction_tensor, attention_tensor = model.forward_with_attention(validation_tensor)
            attention = attention_tensor.cpu().numpy()
        else:
            scaled_prediction_tensor = model(validation_tensor)
            attention = np.empty((len(validation.x), 0), dtype=np.float32)
    prediction_delta = scaled_prediction_tensor.cpu().numpy() * target_std + target_mean
    prediction_abs = validation.anchor[:, None] + prediction_delta
    metrics = metric_dict(validation.y_abs, prediction_abs)
    peak_memory = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    result = {
        "run_id": run_id,
        "model": model_name,
        "seed": seed,
        "status": "complete",
        "best_epoch": best_epoch,
        "epochs_ran": len(epoch_rows),
        "parameter_count": parameter_count(model),
        "peak_gpu_memory_mb": peak_memory,
        "mean_epoch_time_s": float(np.mean(epoch_times)),
        "total_training_time_s": total_time,
        "device": str(device),
        "torch_version": torch.__version__,
        "implementation": "local_auditable_phase_n2_5_v1",
        "period_steps": period_steps,
        **metadata,
        **metrics,
    }
    write_csv(pd.DataFrame(epoch_rows), PHASE_ROOT / "logs" / f"{run_id}_epochs.csv")
    write_json(result, PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}.json")
    np.savez_compressed(
        PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}_predictions.npz",
        y_true=validation.y_abs,
        y_pred=prediction_abs,
        anchor=validation.anchor,
        origin_age_h=validation.origin_age_h,
        target_end_age_h=validation.target_end_age_h,
        row_origin_index=validation.row_origin_index,
        attention_weights=attention,
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": model_name,
            "period_steps": period_steps,
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
        "attention_weights": attention,
    }
    print(json.dumps({"run_id": run_id, "rmse_mV": metrics["rmse_mV"], "best_epoch": best_epoch, "epochs": len(epoch_rows)}), flush=True)
    return result, arrays

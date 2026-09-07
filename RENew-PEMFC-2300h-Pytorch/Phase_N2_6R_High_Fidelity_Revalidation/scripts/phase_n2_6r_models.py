"""Official-source model wrappers and unified Development-only training loop."""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from phase_n2_6r_common import (
    PHASE_ROOT,
    FoldPreprocessor,
    WindowSet,
    ensure_output,
    family_for_config,
    load_local_run,
    metric_dict,
    model_device,
    set_seed,
    verify_no_test_arrays,
    write_csv,
    write_json,
)


sys.dont_write_bytecode = True
TIMESNET_ROOT = PHASE_ROOT / "vendor" / "TimesNet_official"
MODERNTCN_ROOT = PHASE_ROOT / "vendor" / "ModernTCN_official" / "ModernTCN-Long-term-forecasting"
_TIMESNET_MODULE: Any | None = None
_MODERNTCN_MODULE: Any | None = None


def _clear_official_absolute_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "layers" or name.startswith("layers."):
            del sys.modules[name]


def _load_timesnet_module() -> Any:
    global _TIMESNET_MODULE
    if _TIMESNET_MODULE is None:
        _clear_official_absolute_modules()
        sys.path.insert(0, str(TIMESNET_ROOT))
        try:
            _TIMESNET_MODULE = importlib.import_module("models.TimesNet")
        finally:
            sys.path.remove(str(TIMESNET_ROOT))
    return _TIMESNET_MODULE


def _load_moderntcn_module() -> Any:
    global _MODERNTCN_MODULE
    if _MODERNTCN_MODULE is None:
        _clear_official_absolute_modules()
        sys.path.insert(0, str(MODERNTCN_ROOT))
        try:
            _MODERNTCN_MODULE = importlib.import_module("models.ModernTCN")
        finally:
            sys.path.remove(str(MODERNTCN_ROOT))
    return _MODERNTCN_MODULE


class StandardLSTMForecaster(nn.Module):
    """Exact incumbent architecture from Phase N2/N2.6."""

    def __init__(self, input_dim: int, horizon: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1])


class PEMFCTimesNetWrapper(nn.Module):
    """Shape/target wrapper around the unmodified official TimesNet Model."""

    def __init__(self, input_dim: int, horizon: int, spec: dict[str, Any]) -> None:
        super().__init__()
        official = _load_timesnet_module()
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=48,
            label_len=0,
            pred_len=horizon,
            e_layers=int(spec["e_layers"]),
            d_model=int(spec["d_model"]),
            d_ff=int(spec["d_ff"]),
            top_k=int(spec["top_k"]),
            num_kernels=int(spec["num_kernels"]),
            enc_in=input_dim,
            c_out=input_dim,
            embed="fixed",
            freq="h",
            dropout=float(spec["dropout"]),
            num_class=0,
        )
        self.official_model = official.Model(configs)
        self.voltage_channel = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        all_features = self.official_model(x, None, None, None)
        if all_features.ndim != 3 or all_features.shape[1:] != (24, 14):
            raise RuntimeError(f"Unexpected official TimesNet output: {tuple(all_features.shape)}")
        return all_features[:, :, self.voltage_channel]


def _build_multiscale_moderntcn_class(official_base: type[nn.Module]) -> type[nn.Module]:
    class MultiScaleActivatedModernTCN(official_base):
        """Activates the FPN modules declared by the frozen official commit.

        The upstream long-term file constructs lateral, smoothing and transposed
        convolution layers and sizes Flatten_Head for the fused stage-0 length,
        but its forward_feature returns the last stage before calling them.  This
        adapter preserves every official stem/stage/block and activates those
        already-declared fusion modules in the standard top-down order.
        """

        multiscale_activation_patch = True

        @staticmethod
        def _apply_per_variable(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
            batch, variables, channels, length = x.shape
            output = module(x.reshape(batch * variables, channels, length))
            return output.reshape(batch, variables, output.shape[1], output.shape[2])

        def forward_feature(self, x: torch.Tensor, te: torch.Tensor | None = None) -> torch.Tensor:
            batch, variables, _ = x.shape
            x = x.unsqueeze(-2)
            stage_outputs: list[torch.Tensor] = []
            for stage_index in range(self.num_stage):
                batch, variables, channels, length = x.shape
                flattened = x.reshape(batch * variables, channels, length)
                if stage_index == 0:
                    if self.patch_size != self.patch_stride:
                        pad_length = self.patch_size - self.patch_stride
                        flattened = torch.cat(
                            [flattened, flattened[:, :, -1:].repeat(1, 1, pad_length)], dim=-1
                        )
                elif length % self.downsample_ratio != 0:
                    pad_length = self.downsample_ratio - (length % self.downsample_ratio)
                    flattened = torch.cat([flattened, flattened[:, :, -pad_length:]], dim=-1)
                flattened = self.downsample_layers[stage_index](flattened)
                x = flattened.reshape(batch, variables, flattened.shape[1], flattened.shape[2])
                x = self.stages[stage_index](x)
                stage_outputs.append(x)

            if not self.use_multi_scale:
                return stage_outputs[-1]

            fused = self._apply_per_variable(self.lat_layer[-1], stage_outputs[-1])
            fused = self._apply_per_variable(self.smooth_layer[-1], fused)
            for level in range(self.num_stage - 2, -1, -1):
                fused = self._apply_per_variable(self.up_sample_conv[level], fused)
                lateral = self._apply_per_variable(self.lat_layer[level], stage_outputs[level])
                if fused.shape[-1] != lateral.shape[-1]:
                    raise RuntimeError(
                        f"Official multi-scale length mismatch at level {level}: "
                        f"{fused.shape[-1]} vs {lateral.shape[-1]}"
                    )
                fused = self._apply_per_variable(self.smooth_layer[level], fused + lateral)
            return fused

    return MultiScaleActivatedModernTCN


class PEMFCModernTCNWrapper(nn.Module):
    """Input/target wrapper around the official ModernTCN forecasting path."""

    def __init__(self, input_dim: int, horizon: int, spec: dict[str, Any]) -> None:
        super().__init__()
        official = _load_moderntcn_module()
        activated_class = _build_multiscale_moderntcn_class(official.ModernTCN)
        self.official_model = activated_class(
            patch_size=int(spec["patch_size"]),
            patch_stride=int(spec["patch_stride"]),
            stem_ratio=int(spec["stem_ratio"]),
            downsample_ratio=int(spec["downsample_ratio"]),
            ffn_ratio=int(spec["ffn_ratio"]),
            num_blocks=list(spec["num_blocks"]),
            large_size=list(spec["large_size"]),
            small_size=list(spec["small_size"]),
            dims=list(spec["dims"]),
            dw_dims=list(spec["dw_dims"]),
            nvars=input_dim,
            small_kernel_merged=bool(spec["small_kernel_merged"]),
            backbone_dropout=float(spec["dropout"]),
            head_dropout=float(spec["head_dropout"]),
            use_multi_scale=bool(spec["use_multi_scale"]),
            revin=bool(spec["revin"]),
            affine=bool(spec["affine"]),
            subtract_last=bool(spec["subtract_last"]),
            freq="h",
            seq_len=48,
            c_in=input_dim,
            individual=bool(spec["individual"]),
            target_window=horizon,
        )
        self.voltage_channel = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        all_features = self.official_model(x.transpose(1, 2), None)
        if all_features.ndim != 3 or all_features.shape[1:] != (14, 24):
            raise RuntimeError(f"Unexpected official ModernTCN output: {tuple(all_features.shape)}")
        return all_features[:, self.voltage_channel, :]


def create_model(config_name: str, input_dim: int, horizon: int, cfg: dict[str, Any]) -> nn.Module:
    if config_name == "LSTM":
        return StandardLSTMForecaster(input_dim, horizon)
    spec = cfg["architectures"][config_name]
    family = str(spec["family"])
    if family == "TimesNet":
        return PEMFCTimesNetWrapper(input_dim, horizon, spec)
    if family == "ModernTCN":
        return PEMFCModernTCNWrapper(input_dim, horizon, spec)
    raise KeyError(f"Phase N2.6R forbids model family: {family}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _predict_scaled(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            rows.append(model(batch_x.to(device)).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _target_setup(
    config_name: str,
    train: WindowSet,
    preprocessor: FoldPreprocessor,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    if config_name == "LSTM":
        target_mean = np.mean(train.y_delta, axis=0, keepdims=True)
        target_std = np.std(train.y_delta, axis=0, keepdims=True)
        target_std = np.where(target_std > 1e-6, target_std, 1.0)
        y_train = ((train.y_delta - target_mean) / target_std).astype(np.float32)

        def inverse(values: np.ndarray, anchor: np.ndarray) -> np.ndarray:
            return anchor[:, None] + values * target_std + target_mean

        metadata = {"target_training_space": "standardized_delta_voltage", "target_mean": target_mean, "target_std": target_std}
        return y_train, inverse, metadata

    voltage_mean = float(preprocessor.scaler.mean_[0])
    voltage_scale = float(preprocessor.scaler.scale_[0])
    if not math.isfinite(voltage_scale) or voltage_scale <= 0:
        raise RuntimeError("Invalid fold-Train voltage feature scale")
    y_train = ((train.y_abs - voltage_mean) / voltage_scale).astype(np.float32)

    def inverse(values: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        del anchor
        return values * voltage_scale + voltage_mean

    metadata = {
        "target_training_space": "externally_scaled_absolute_voltage_with_official_internal_normalization",
        "voltage_feature_mean": voltage_mean,
        "voltage_feature_scale": voltage_scale,
    }
    return y_train, inverse, metadata


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
        verify_no_test_arrays(cached[1], cfg, run_id)
        return cached

    set_seed(seed)
    device = model_device()
    preprocessor = FoldPreprocessor().fit(train.x)
    x_train = preprocessor.transform(train.x)
    x_validation = preprocessor.transform(validation.x)
    y_train, inverse_prediction, target_metadata = _target_setup(config_name, train, preprocessor)
    model = create_model(config_name, x_train.shape[-1], y_train.shape[-1], cfg).to(device)
    count = parameter_count(model)
    if config_name != "LSTM" and count >= int(cfg["selection"]["parameter_upper_bound"]):
        raise RuntimeError(f"High-fidelity parameter bound exceeded for {config_name}: {count}")

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

        scaled_validation = _predict_scaled(model, x_validation, device)
        prediction_abs = inverse_prediction(scaled_validation, validation.anchor)
        validation_metrics = metric_dict(validation.y_abs, prediction_abs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_epoch
        epoch_times.append(elapsed)
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_scaled_mse": float(np.mean(losses)),
                "validation_rmse_V": validation_metrics["rmse_V"],
                "validation_mae_V": validation_metrics["mae_V"],
                "validation_r2": validation_metrics["r2"],
                "epoch_time_s": elapsed,
            }
        )
        if epoch % 25 == 0:
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "epoch": epoch,
                        "validation_rmse_mV": validation_metrics["rmse_mV"],
                        "validation_r2": validation_metrics["r2"],
                    }
                ),
                flush=True,
            )
        if validation_metrics["rmse_V"] < best_rmse - float(cfg["training"]["min_delta"]):
            best_rmse = validation_metrics["rmse_V"]
            best_epoch = epoch
            patience = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
        if patience >= int(cfg["training"]["early_stopping_patience"]):
            break

    total_time = time.perf_counter() - start_total
    if best_state is None:
        raise RuntimeError(f"No finite state captured for {run_id}")
    model.load_state_dict(best_state)
    scaled_validation = _predict_scaled(model, x_validation, device)
    prediction_abs = inverse_prediction(scaled_validation, validation.anchor)
    prediction_delta = prediction_abs - validation.anchor[:, None]
    if not np.isfinite(prediction_abs).all():
        raise RuntimeError(f"Non-finite inverse-transformed predictions in {run_id}")
    metrics = metric_dict(validation.y_abs, prediction_abs)
    peak_memory = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    result = {
        "run_id": run_id,
        "config_name": config_name,
        "family": family_for_config(cfg, config_name),
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
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_rmse_computed": False,
        "test_r2_computed": False,
        "test_mae_computed": False,
        **metadata,
        **{key: value for key, value in target_metadata.items() if not isinstance(value, np.ndarray)},
        **metrics,
    }
    write_csv(pd.DataFrame(epoch_rows), PHASE_ROOT / "logs" / f"{run_id}_epochs.csv")
    write_json(result, PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}.json")
    prediction_path = ensure_output(PHASE_ROOT / "outputs" / "model_runs" / f"{run_id}_predictions.npz")
    np.savez_compressed(
        prediction_path,
        y_true=validation.y_abs,
        y_pred=prediction_abs,
        y_pred_delta=prediction_delta,
        anchor=validation.anchor,
        origin_age_h=validation.origin_age_h,
        target_end_age_h=validation.target_end_age_h,
        row_origin_index=validation.row_origin_index,
    )
    checkpoint_path = ensure_output(PHASE_ROOT / "checkpoints" / f"{run_id}.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "config_name": config_name,
            "feature_medians": preprocessor.medians,
            "feature_scaler_mean": preprocessor.scaler.mean_,
            "feature_scaler_scale": preprocessor.scaler.scale_,
            "target_metadata": target_metadata,
            "metadata": metadata,
        },
        checkpoint_path,
    )
    arrays = dict(np.load(prediction_path, allow_pickle=False))
    verify_no_test_arrays(arrays, cfg, run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "rmse_mV": metrics["rmse_mV"],
                "mae_mV": metrics["mae_mV"],
                "r2": metrics["r2"],
                "best_epoch": best_epoch,
                "epochs": len(epoch_rows),
            }
        ),
        flush=True,
    )
    return result, arrays

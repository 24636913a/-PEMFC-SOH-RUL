"""RealHyFC F2: one self-contained project program; third-party TFT is unchanged.

    python -B RealHyFC_F2_Main.py --mode evaluate --seed 42
    python -B RealHyFC_F2_Main.py --mode evaluate --seed all
    python -B RealHyFC_F2_Main.py --mode train --seed 42

Evaluate is the default. Train creates a NEW instance in a unique output run.
The console reports one evaluation partition (Validation by default), with
RMSE/MAE in V and the historical DeltaV R2. All 24 steps enter the metrics;
the voltage time plot uses the fixed 24th step, never stitched forecasts.
No project Python module is imported or executed. Historical JSON/YAML, CSV,
NPZ and checkpoint assets remain read-only. Assessment/late-life are exploratory.
"""
from __future__ import annotations

# 1. 环境配置、依赖与文件路径
import argparse
import ast
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paper-folder copy: resolve the unchanged research assets two levels above.
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
HISTORY = ROOT / 'Phase_H_Determinism_MultiSeed_Stability'
FULL_EVALUATION = ROOT / 'Phase_SCI_RealHyFC_Full_Chronological_Evaluation'
FROZEN_DATA = ROOT / 'Phase_SCI_F2_Seed_Screening_Round2/outputs/audits'
SEED_ASSETS = Path(__file__).resolve().with_name('RealHyFC_F2_Main_Seed_Assets.json')
SEED_ASSETS_SHA256 = '266215dcadf07fad0d6b73e6255626823537559b2b48cbafe3c5996c03acc187'
SUPPORTED_SEEDS = (42, 103, 193, 202, 210, 212)
HYPERPARAMETER_SOURCE = ROOT / 'Phase_TFT_Hyperparameter_Optimization_Restart/outputs/T1R_SELECTED_CONFIGURATIONS.json'
OUTPUT_ROOT = ROOT / 'RealHyFC_F2_Main_Outputs'
LOOKBACK, HORIZON, RESOLUTION_H = 48, 24, 0.5
FEATURES = ['cell_voltage_mean', 'cell_voltage_std', 'cell_voltage_range',
            'cell_voltage_min', 'cell_voltage_max', 'actual_current_A_mean',
            'actual_current_A_std', 'actual_current_A_peak_to_peak',
            'actual_current_density_A_cm2_mean', 'actual_current_density_A_cm2_std',
            'actual_current_density_A_cm2_peak_to_peak', 'load_transition_count',
            'operation_point_transitions', 'zero_current_fraction']
PARTITIONS = ('TRAIN', 'VALIDATION', 'HISTORICAL_ASSESSMENT', 'LATE_LIFE')
DEFAULT_EVALUATION_PARTITION = 'VALIDATION'
PLOT_HORIZON_STEP = 24  # Fixed before inference; plotting only, not metric selection.

def sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()

def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, str(value.dtype))).encode() + value.tobytes()).hexdigest()

def read_json(file: Path) -> dict:
    return json.loads(file.read_text(encoding='utf-8'))

def read_csv(file: Path) -> pd.DataFrame:
    return pd.read_csv(file, float_precision='round_trip')

def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)

def output_path(run: Path, name: str) -> Path:
    file = (run / name).resolve()
    require(run.resolve().is_relative_to(OUTPUT_ROOT.resolve()) and
            file.is_relative_to(run.resolve()), 'HISTORICAL_OUTPUT_WRITE_FORBIDDEN')
    file.parent.mkdir(parents=True, exist_ok=True)
    return file

def atomic_text(run: Path, name: str, text: str) -> None:
    file = output_path(run, name); temporary = file.with_suffix(file.suffix + '.writing')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    for delay in [.02, .05, .1, .2, .5, 1., None]:
        try: os.replace(temporary, file); return
        except PermissionError:
            if delay is None: raise
            time.sleep(delay)

def save_json(run: Path, name: str, value: Any) -> None:
    atomic_text(run, name, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def save_csv(run: Path, name: str, value: Any) -> None:
    atomic_text(run, name, pd.DataFrame(value).to_csv(index=False, float_format='%.17g', lineterminator='\n'))

def configure_environment(device: str) -> dict:
    frozen = yaml.safe_load((HISTORY / 'configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
    expected = frozen['normal_runtime']
    require(torch.cuda.is_available(), 'CUDA_REQUIRED_FOR_EXACT_HISTORICAL_REPLAY')
    require(device == 'cuda', 'EXACT_REPLAY_REQUIRES_HISTORICAL_CUDA_DEVICE')
    torch.set_num_threads(expected['CPU_threads'])
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.backends.cudnn.deterministic = expected['cudnn_deterministic']
    torch.backends.cudnn.benchmark = expected['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = expected['TF32_matmul']
    torch.backends.cudnn.allow_tf32 = expected['TF32_cudnn']
    import platform
    import pytorch_forecasting
    actual = dict(python=platform.python_version(), torch=str(torch.__version__),
                  pytorch_forecasting=pytorch_forecasting.__version__, CUDA=torch.version.cuda,
                  cuDNN=torch.backends.cudnn.version(), GPU=torch.cuda.get_device_name(0),
                  CPU_threads=torch.get_num_threads(), cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                  TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  TF32_cudnn=torch.backends.cudnn.allow_tf32,
                  CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    for key, value in expected.items():
        if key != 'claim': require(actual[key] == value, 'HISTORICAL_RUNTIME_MISMATCH:' + key)
    # Third-party constructor messages are irrelevant to the concise CLI output.
    warnings.filterwarnings('ignore', message="Attribute '(loss|logging_metrics)' is an instance of")
    return actual


# 2. RealHyFC完整数据读取
def load_full_realhyfc() -> tuple[pd.DataFrame, dict]:
    # Full read is isolated here; do not change any historical Development loader.
    frame = pd.read_csv(SOURCE)  # Historical default parser preserves Float32 input identity.
    require(frame.index.equals(pd.RangeIndex(len(frame))), 'PHYSICAL_ROW_ORDER_CHANGED')
    require(frame.columns.is_unique and set(FEATURES).issubset(frame.columns), 'RAW14_SCHEMA_MISMATCH')
    require(frame.bin_end_ophrs2_h.is_monotonic_increasing and frame.bin_end_ophrs2_h.is_unique,
            'OPERATING_TIME_ORDER_OR_DUPLICATE_ERROR')
    require(not frame.bin_valid_for_windows.isna().any() and
            set(frame.bin_valid_for_windows.unique()).issubset({0, 1}), 'VALIDITY_FLAG_ERROR')
    historical = read_json(FULL_EVALUATION / 'outputs/audits/FULL_SOURCE.json')
    require(len(frame) == historical['rows'] == 3843, 'FULL_FEATURE_ROW_COUNT_CHANGED')
    require(sha256(SOURCE) == historical['sha256'], 'FULL_FEATURE_SHA_CHANGED')
    prefix = frame.iloc[:3007].loc[lambda x: x.bin_end_ophrs2_h < 1819.997738]
    require(len(prefix) == 3005 and np.array_equal(prefix.index, np.arange(3005)),
            'HISTORICAL_DEVELOPMENT_IDENTITY_CHANGED')
    return frame, historical


# 3. 时间划分及合法窗口构建
@dataclass
class Windows:
    x_raw: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    targets: np.ndarray
    origin_age_h: np.ndarray
    target_age_h: np.ndarray
    origin_raw_voltage_V: np.ndarray
    target_raw_voltage_V: np.ndarray

def chronological_partitions(n: int) -> dict[str, tuple[int, int]]:
    require(isinstance(n, int) and n > 3005, 'INVALID_FULL_SERIES_LENGTH')
    return dict(zip(PARTITIONS, [(0, 2103), (2103, 2404), (2404, 3005), (3005, n)]))

def build_legal_windows(frame: pd.DataFrame, start: int, end: int) -> tuple[Windows, pd.DataFrame]:
    values = frame[FEATURES].to_numpy(float); voltage = frame.cell_voltage_mean.to_numpy(float)
    ages = frame.bin_end_ophrs2_h.to_numpy(float)
    valid = frame.bin_valid_for_windows.astype(bool).to_numpy(); segment = frame.temporal_segment_id.to_numpy()
    xs, ys, origins, targets, exclusions = [], [], [], [], []
    for origin in range(start, end):
        first, last = origin - LOOKBACK + 1, origin + HORIZON
        reasons = []
        if first < 0: reasons.append('HISTORY_BEFORE_DATASET')
        if last >= end: reasons.append('24_TARGETS_OUTSIDE_PARTITION')
        if not reasons:
            block = slice(first, last + 1)
            if not valid[block].all(): reasons.append('INVALID_BIN')
            if np.any(segment[block] != segment[origin]): reasons.append('TEMPORAL_SEGMENT_BREAK')
            if not np.allclose(np.diff(ages[block]), .5, atol=1e-6, rtol=0): reasons.append('OPERATING_TIME_GAP')
            if not np.isfinite(values[block]).all(): reasons.append('NONFINITE_RAW14_OR_VOLTAGE')
        if reasons:
            exclusions.append(dict(origin_row=origin, primary_reason=reasons[0], all_reasons='|'.join(reasons)))
            continue
        # Future covariates above certify eligibility ONLY; x ends at origin.
        xs.append(values[first:origin + 1]); ys.append(voltage[origin + 1:last + 1] - voltage[origin])
        origins.append(origin); targets.append(np.arange(origin + 1, last + 1))
    o = np.asarray(origins, dtype=np.int64); t = np.asarray(targets, dtype=np.int64).reshape(-1, HORIZON)
    windows = Windows(np.asarray(xs, np.float32).reshape(-1, LOOKBACK, 14),
                      np.asarray(ys, np.float32).reshape(-1, HORIZON), o, t,
                      ages[o], ages[t], voltage[o], voltage[t])
    return windows, pd.DataFrame(exclusions)

def verify_window_identity(windows: dict[str, Windows], frame: pd.DataFrame) -> list[dict]:
    results = []; gap = (pd.to_datetime(frame.first_datetime) - pd.to_datetime(frame.last_datetime).shift()).dt.total_seconds()
    for name, w in windows.items():
        frozen = np.load(FULL_EVALUATION / f'outputs/audits/{name}_WINDOWS.npz', allow_pickle=False)
        for field in Windows.__dataclass_fields__:
            value = getattr(w, field)
            require(value.dtype == frozen[field].dtype and np.array_equal(value, frozen[field]),
                    'FROZEN_WINDOW_ARRAY_MISMATCH:' + name + ':' + field)
        pairs = np.unique((w.origins[:, None] + np.arange(-46, 25)).ravel())
        require(frame.loc[pairs, 'gap_before_flag'].sum() == 0 and gap.iloc[pairs].max() <= 1.01,
                'LEGAL_WINDOW_CALENDAR_GAP:' + name)
        results.append(dict(partition=name, records=chronological_partitions(len(frame))[name][1] -
                            chronological_partitions(len(frame))[name][0], origins=len(w.origins),
                            queries=w.y.size, unique_targets=len(np.unique(w.targets)),
                            origins_sha256=array_sha256(w.origins), targets_sha256=array_sha256(w.targets),
                            x_raw_sha256=array_sha256(w.x_raw), y_sha256=array_sha256(w.y), exact=True))
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            require(not np.intersect1d(windows[a].targets, windows[b].targets).size, 'PARTITION_TARGET_OVERLAP')
    return results


# 4. Train-only标准化、Spearman特征筛选、Top10掩码及因果EWMA
class TrainOnlyScaler:
    def fit(self, x: np.ndarray) -> 'TrainOnlyScaler':
        flat = x.astype(np.float64).reshape(-1, 14)
        self.mean = flat.mean(0); self.scale = flat.std(0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    @property
    def target_scale(self) -> float: return float(self.scale[0])

def spearman_train_audit(train: Windows, frozen_top10: list[int]) -> pd.DataFrame:
    # Reproduce historical Train-only scoring, never select against late targets.
    rows = []
    for index, name in enumerate(FEATURES):
        origin = pd.Series(train.x_raw[:, -1, index])
        rho = [origin.corr(pd.Series(train.y[:, h]), method='spearman') for h in range(HORIZON)]
        rows.append(dict(feature=name, raw_feature_index=index, mean_abs_spearman=float(np.nanmean(np.abs(rho))),
                         **{f'spearman_h{h + 1:02d}': float(r) for h, r in enumerate(rho)}))
    scores = pd.DataFrame(rows).sort_values(['mean_abs_spearman', 'raw_feature_index'],
                                          ascending=[False, True], kind='stable').reset_index(drop=True)
    selected = scores.head(10).raw_feature_index.tolist()
    if 0 not in selected: selected[-1] = 0
    scores['selected_top10'] = scores.raw_feature_index.isin(selected)
    observed = scores.loc[scores.selected_top10, 'raw_feature_index'].tolist()
    require(observed == frozen_top10, 'TRAIN_SPEARMAN_FROZEN_TOP10_MISMATCH')
    return scores

def causal_ewma(x: np.ndarray) -> np.ndarray:
    result = x.copy()
    for t in range(1, LOOKBACK): result[:, t, 0] = .3 * x[:, t, 0] + .7 * result[:, t - 1, 0]
    return result

def preprocess_frozen(windows: dict[str, Windows]) -> tuple[TrainOnlyScaler, dict[str, np.ndarray], pd.DataFrame]:
    frozen = yaml.safe_load((HISTORY / 'configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
    require(frozen['task']['raw_f2_feature_order'] == FEATURES, 'FROZEN_FEATURE_SCHEMA_MISMATCH')
    mask = frozen['preprocessing']['Top10_indices']
    require(mask == [0, 3, 13, 11, 4, 9, 6, 7, 10, 8], 'FROZEN_TOP10_CHANGED')
    scaler = TrainOnlyScaler().fit(windows['TRAIN'].x_raw)  # unfiltered legal Train histories only
    authority = read_json(FROZEN_DATA / 'SCALER.json')
    require(np.array_equal(scaler.mean, authority['mean']) and np.array_equal(scaler.scale, authority['scale']),
            'TRAIN_ONLY_SCALER_MISMATCH')
    scores = spearman_train_audit(windows['TRAIN'], mask)
    prepared = {}
    for name, w in windows.items():
        z = scaler.transform(causal_ewma(w.x_raw))
        z[:, :, [index for index in range(14) if index not in mask]] = 0.
        frozen_x = np.load(FULL_EVALUATION / f'outputs/audits/{name}_WINDOWS.npz', allow_pickle=False)['x']
        require(np.array_equal(z, frozen_x), 'FROZEN_PREPROCESSING_MISMATCH:' + name)
        prepared[name] = z
    return scaler, prepared, scores


# 5. TFT基础结构和F2双路径网络定义
RAW_NAMES = [f"raw_f2_{i:02d}" for i in range(14)]
RELATIVE_STEP = "relative_forecast_step"
X_REALS = [*RAW_NAMES, RELATIVE_STEP]
RAW_F2_FEATURE_NAMES = tuple(FEATURES)
REGIME_FEATURE_NAMES = ("load_transition_count", "zero_current_fraction", "actual_current_density_A_cm2_std", "actual_current_A_std", "actual_current_density_A_cm2_mean")


class CanonicalStrictHistoryTFT(TemporalFusionTransformer):
    """Canonical TFT with a direct standardized-DeltaV point-forecast head."""

    def __init__(self, *, hidden_size: int, hidden_continuous_size: int, attention_head_size: int, lstm_layers: int, dropout: float) -> None:
        super().__init__(
            hidden_size=hidden_size,
            hidden_continuous_size=hidden_continuous_size,
            attention_head_size=attention_head_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            output_size=1,
            max_encoder_length=48,
            static_categoricals=[],
            static_reals=[],
            time_varying_categoricals_encoder=[],
            time_varying_categoricals_decoder=[],
            time_varying_reals_encoder=X_REALS,
            time_varying_reals_decoder=[RELATIVE_STEP],
            x_reals=X_REALS,
            x_categoricals=[],
            causal_attention=True,
            learning_rate=0.001,
            log_interval=-1,
        )

    def transform_output(self, prediction: torch.Tensor, target_scale: torch.Tensor, loss: Any = None) -> torch.Tensor:
        """Retain the upstream TFT body but expose its raw one-channel head.

        The task's target is already standardized DeltaV and uses MSE outside
        pytorch-forecasting's probabilistic/quantile loss interface.  Therefore
        no target normalizer or rescaling is applied here.
        """
        return prediction

    @staticmethod
    def build_input(x_history_scaled: torch.Tensor) -> dict[str, torch.Tensor]:
        """Construct a strict information-set input dictionary for the TFT body."""
        if x_history_scaled.ndim != 3 or x_history_scaled.shape[1:] != (48, 14):
            raise RuntimeError(f"Expected [B,48,14] history, got {tuple(x_history_scaled.shape)}")
        batch, _, _ = x_history_scaled.shape
        dtype, device = x_history_scaled.dtype, x_history_scaled.device
        # A relative coordinate is deterministic at the origin. It is the only
        # decoder variable and carries no future operating observation.
        encoder_relative = torch.arange(-47, 1, device=device, dtype=dtype).view(1, 48, 1).expand(batch, -1, -1) / 24.0
        decoder_relative = torch.arange(1, 25, device=device, dtype=dtype).view(1, 24, 1).expand(batch, -1, -1) / 24.0
        encoder_cont = torch.cat([x_history_scaled, encoder_relative], dim=-1)
        # The 14 raw decoder channels are deliberately fixed placeholders and
        # not included in the decoder variable-selection network.
        decoder_cont = torch.cat([torch.zeros((batch, 24, 14), device=device, dtype=dtype), decoder_relative], dim=-1)
        return {
            "encoder_cat": torch.empty((batch, 48, 0), device=device, dtype=torch.long),
            "decoder_cat": torch.empty((batch, 24, 0), device=device, dtype=torch.long),
            "encoder_cont": encoder_cont,
            "decoder_cont": decoder_cont,
            "encoder_lengths": torch.full((batch,), 48, device=device, dtype=torch.long),
            "decoder_lengths": torch.full((batch,), 24, device=device, dtype=torch.long),
            "target_scale": torch.zeros((batch, 2), device=device, dtype=dtype),
        }

    def forward_history(self, x_history_scaled: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.forward(self.build_input(x_history_scaled))
        prediction = out["prediction"]
        if prediction.shape != (x_history_scaled.shape[0], 24, 1):
            raise RuntimeError(f"Canonical TFT output contract fail: {tuple(prediction.shape)}")
        return out

    def predict_delta(self, x_history_scaled: torch.Tensor) -> torch.Tensor:
        return self.forward_history(x_history_scaled)["prediction"].squeeze(-1)


class _LeftPaddedDepthwiseConv1d(nn.Module):
    """Depthwise Conv1d with explicit left-only padding and fixed output length."""

    def __init__(self, channels: int, *, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError(f"Expected [B,C,L], got {tuple(x.shape)}")
        output = self.conv(F.pad(x, (self.left_padding, 0)))
        if output.shape != x.shape:
            raise RuntimeError(
                "Causal depthwise convolution changed shape: "
                f"{tuple(x.shape)} -> {tuple(output.shape)}"
            )
        return output


class CausalMultiScaleHistoryBranch(nn.Module):
    """Two frozen-specification causal receptive fields over VSN history.

    Input and output both have shape ``[B, L, H]``.  The branch never receives
    decoder positions; causality is supplied by explicit left-only padding.
    """

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.short = _LeftPaddedDepthwiseConv1d(
            self.hidden_size, kernel_size=3, dilation=1
        )
        self.medium = _LeftPaddedDepthwiseConv1d(
            self.hidden_size, kernel_size=3, dilation=4
        )
        self.pointwise = nn.Conv1d(
            2 * self.hidden_size, self.hidden_size, kernel_size=1, bias=True
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, encoder_vsn: torch.Tensor) -> torch.Tensor:
        if encoder_vsn.ndim != 3 or encoder_vsn.shape[-1] != self.hidden_size:
            raise RuntimeError(
                "Expected encoder VSN [B,L,hidden_size], got "
                f"{tuple(encoder_vsn.shape)} for hidden_size={self.hidden_size}"
            )
        channels_first = encoder_vsn.transpose(1, 2)
        short = self.short(channels_first)
        medium = self.medium(channels_first)
        mixed = self.pointwise(torch.cat((short, medium), dim=1)).transpose(1, 2)
        output = self.norm(self.dropout(mixed) + encoder_vsn)
        if output.shape != encoder_vsn.shape:
            raise RuntimeError("History branch output shape contract failed")
        return output


class DualPathStrictHistoryTFT(CanonicalStrictHistoryTFT):
    """F1: native TFT plus a causal convolutional encoder history branch.

    ``dual_path_enabled=False`` delegates exactly to the inherited upstream
    forward method. It exists solely for the synthetic native-parity contract.
    The enabled branch changes only the post-LSTM encoder tensor that is
    concatenated with the untouched decoder tensor before static enrichment.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        hidden_continuous_size: int,
        attention_head_size: int,
        lstm_layers: int,
        dropout: float,
        dual_path_enabled: bool = True,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            hidden_continuous_size=hidden_continuous_size,
            attention_head_size=attention_head_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
        self.dual_path_enabled = bool(dual_path_enabled)
        if self.dual_path_enabled:
            self.history_branch = CausalMultiScaleHistoryBranch(
                hidden_size=self.hparams.hidden_size,
                dropout=self.hparams.dropout,
            )

    def fuse_encoder(
        self,
        encoder_lstm: torch.Tensor,
        encoder_conv: torch.Tensor,
        encoder_cont: torch.Tensor,
    ) -> torch.Tensor:
        """F1's parameter-free equal mixture; overridden by F2."""
        del encoder_cont
        return 0.5 * encoder_lstm + 0.5 * encoder_conv

    def encode_history_branch(self, encoder_vsn: torch.Tensor) -> torch.Tensor:
        if not self.dual_path_enabled:
            raise RuntimeError("The history branch is disabled for native parity mode")
        return self.history_branch(encoder_vsn)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.dual_path_enabled:
            return super().forward(x)

        # This block mirrors pytorch-forecasting 1.8.0 _tft.py:530–678. The
        # only intentional delta is labelled below at ``encoder_fused``.
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cont.size(1)
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        if len(self.static_variables) > 0:
            static_embedding = {
                name: input_vectors[name][:, 0] for name in self.static_variables
            }
            static_embedding, static_variable_selection = self.static_variable_selection(
                static_embedding
            )
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            static_variable_selection = torch.zeros(
                (x_cont.size(0), 0), dtype=self.dtype, device=self.device
            )

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )
        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length]
            for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = (
            self.encoder_variable_selection(
                embeddings_varying_encoder,
                static_context_variable_selection[:, :max_encoder_length],
            )
        )
        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:]
            for name in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = (
            self.decoder_variable_selection(
                embeddings_varying_decoder,
                static_context_variable_selection[:, max_encoder_length:],
            )
        )

        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder,
            (input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False,
        )
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            lstm_output_encoder, embeddings_varying_encoder
        )
        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            lstm_output_decoder, embeddings_varying_decoder
        )

        encoder_conv = self.encode_history_branch(embeddings_varying_encoder)
        encoder_fused = self.fuse_encoder(
            lstm_output_encoder, encoder_conv, x["encoder_cont"]
        )
        if encoder_fused.shape != lstm_output_encoder.shape:
            raise RuntimeError("Encoder fusion changed the native encoder shape")
        lstm_output = torch.cat([encoder_fused, lstm_output_decoder], dim=1)

        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output,
            self.expand_static_context(static_context_enrichment, timesteps),
        )
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths
            ),
        )
        attn_output = self.post_attn_gate_norm(
            attn_output, attn_input[:, max_encoder_length:]
        )
        output = self.pos_wise_ff(attn_output)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if self.n_targets > 1:
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)
        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )


class RegimeFusionGate(nn.Module):
    """The frozen 5→8→1 historical-regime gate."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(5, 8)
        self.activation = nn.ELU()
        self.output = nn.Linear(8, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, regime_context: torch.Tensor) -> torch.Tensor:
        if regime_context.ndim != 2 or regime_context.shape[1] != 5:
            raise RuntimeError(f"Expected [B,5] regime context, got {tuple(regime_context.shape)}")
        return torch.sigmoid(self.output(self.activation(self.hidden(regime_context))))


class RegimeAwareDualPathTFT(DualPathStrictHistoryTFT):
    """F2: F1 with only its fixed encoder fusion replaced by a regime gate."""

    regime_feature_names = REGIME_FEATURE_NAMES

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.dual_path_enabled:
            raise RuntimeError("F2 requires the enabled F1 convolutional branch")
        if tuple(RAW_NAMES) != tuple(f"raw_f2_{i:02d}" for i in range(14)):
            raise RuntimeError("Frozen Raw F2 schema identity mismatch")
        self.regime_indices = tuple(RAW_F2_FEATURE_NAMES.index(name) for name in REGIME_FEATURE_NAMES)
        self.regime_gate = RegimeFusionGate()

    def regime_context(self, encoder_cont: torch.Tensor) -> torch.Tensor:
        if encoder_cont.ndim != 3 or encoder_cont.shape[-1] < 14:
            raise RuntimeError("Expected scaled/masked encoder continuous tensor with 14 Raw F2 slots")
        index = torch.as_tensor(self.regime_indices, dtype=torch.long, device=encoder_cont.device)
        return encoder_cont.index_select(dim=-1, index=index).mean(dim=1)

    def regime_weight(self, encoder_cont: torch.Tensor) -> torch.Tensor:
        return self.regime_gate(self.regime_context(encoder_cont))

    def fuse_encoder(
        self,
        encoder_lstm: torch.Tensor,
        encoder_conv: torch.Tensor,
        encoder_cont: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.regime_weight(encoder_cont).view(-1, 1, 1)
        return gate * encoder_lstm + (1.0 - gate) * encoder_conv


MODEL_SOURCE_SHA256 = {'Phase_TFT_Original_Baseline/models/canonical_tft.py': 'b6540cdb109565eb40fa9d631984699640a74233d3e869bca163e37f02db3618', 'Phase_F_TFT_DualPath_RegimeFusion/models/causal_multiscale_branch.py': 'd70bbb0093a0e06b41325a09de48ac18a97a223bb6a4259941ae17a2ea1b220c', 'Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py': 'a8e979c00a753c7a65de3e200ff0e7cb9562888d9555c63b94cef97f28fececc', 'Phase_F_TFT_DualPath_RegimeFusion/models/regime_aware_dualpath_tft.py': 'bbc219e5d22a664f1a45dbe7ac5d8a561a0bfc6d63097a7f847612e8af352c57'}


MODEL_CLASS_AST_SHA256 = {'CanonicalStrictHistoryTFT': 'b68e1a0a0e9236d75b3b969300ebfa536d6256835d87931f6c0b83f82f0c2c5b', '_LeftPaddedDepthwiseConv1d': '7f460c56d88c1d30d6d68b65aa821f4f65f19bb186beda943d487322b6776127', 'CausalMultiScaleHistoryBranch': '7e061c13d798f42ccf22af3a7b1348f6897ad7b4c1a20387681f1be98d1b747c', 'DualPathStrictHistoryTFT': '152795fa73b26a5cb9f5b15ba951b666b8942bec0547a4e7569775f22ac0efeb', 'RegimeFusionGate': '1a1e7b579b27bacb8809b3495b3314854e198277f6ac63999abb60e39d251e42', 'RegimeAwareDualPathTFT': '3b10169740cb49bed34381f66be63f7dd71669f322ccef78c38bbc259dfd829b'}


# 6. 历史Optuna超参数结果读取、配置选择与冻结
def load_historical_hyperparameters() -> dict:
    """No Optuna execution: read historical selected base-TFT Fold 2 trial 13.

    F2 INHERITS this configuration; it has not undergone an independent F2
    hyperparameter search. Assessment and late-life never select parameters.
    """
    selected = read_json(HYPERPARAMETER_SOURCE)
    fold = selected['folds']['fold_2']
    require(fold['pass'] and fold['study_name'] == 'T1R_fold_2' and fold['best_trial_id'] == 13,
            'HISTORICAL_OPTUNA_FOLD2_IDENTITY_MISMATCH')
    parameters = dict(fold['best_params'])
    frozen = yaml.safe_load((HISTORY / 'configs/H_FROZEN_PROTOCOL.yaml').read_text(encoding='utf-8'))
    for name, value in parameters.items():
        require(value == frozen['hyperparameters'][name], 'INHERITED_FROZEN_HYPERPARAMETER_MISMATCH:' + name)
    # Training settings are inherited separately from the frozen H protocol,
    # not falsely represented as new Optuna discoveries.
    for name in ['max_epochs', 'patience', 'gradient_clip_norm', 'loss', 'optimizer']:
        parameters[name] = frozen['hyperparameters'][name]
    require(parameters['max_epochs'] == 450 and parameters['patience'] == 40 and
            parameters['gradient_clip_norm'] == 1.0 and parameters['loss'] == 'MSE' and
            parameters['optimizer'] == 'AdamW', 'FROZEN_TRAINING_PROTOCOL_CHANGED')
    return dict(parameters=parameters, source=str(HYPERPARAMETER_SOURCE),
                source_sha256=sha256(HYPERPARAMETER_SOURCE), fold='fold_2',
                study=fold['study_name'], selected_trial=fold['best_trial_id'],
                historical_optimized_model='base TFT', inherited_by='F2',
                F2_specific_search_performed=False, optuna_executed=False,
                selection_uses_Assessment_or_late_life=False,
                historical_H_protocol_sha256=sha256(HISTORY / 'configs/H_FROZEN_PROTOCOL.yaml'))


# 7. F2模型初始化
def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def initialize_f2(seed: int, parameters: dict, device: torch.device) -> RegimeAwareDualPathTFT:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = RegimeAwareDualPathTFT(hidden_size=parameters['hidden_size'],
                                  hidden_continuous_size=parameters['hidden_continuous_size'],
                                  attention_head_size=4, lstm_layers=1,
                                  dropout=parameters['dropout']).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(count == 332436, 'F2_PARAMETER_COUNT_MISMATCH')
    authority = frozen_seed_asset(seed)
    require(state_sha256(model) == authority['initialization_sha256'],
            'ORIGINAL_F2_INITIALIZATION_MISMATCH')
    return model


# 8. F2模型训练、Validation与Early Stopping
def train_new_f2(model: RegimeAwareDualPathTFT, seed: int, parameters: dict,
                 train: Windows, validation: Windows, tx: np.ndarray, vx: np.ndarray,
                 scaler: TrainOnlyScaler, run: Path, provenance: dict,
                 device: torch.device) -> tuple[Path, pd.DataFrame, dict]:
    """Only explicit --mode train enters here. No Assessment or late arguments.

    Fresh initialization; MSE/AdamW, Train-only scaling, Validation RMSE early
    stopping. Never warm-start from historical best weights. A new unique run
    directory is used, so historical checkpoints cannot be overwritten.
    """
    initial = state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['learning_rate'],
                                  weight_decay=parameters['weight_decay'])
    ty, vy = train.y / scaler.target_scale, validation.y / scaler.target_scale
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty.astype(np.float32))),
                        batch_size=parameters['batch_size'], shuffle=True, generator=generator, num_workers=0)
    best, best_epoch, stale, steps = math.inf, 0, 0, 0
    history = []; started = time.perf_counter()
    checkpoint = output_path(run, f'seed{seed}/checkpoints/F2_seed{seed}_NEW_best.pt')
    require(not checkpoint.exists(), 'NEW_TRAINING_CHECKPOINT_ALREADY_EXISTS')
    save_json(run, f'seed{seed}/TRAINING_STATE.json', dict(status='TRAINING', initialization_sha256=initial))
    for epoch in range(1, parameters['max_epochs'] + 1):
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model.predict_delta(xb), yb)
            require(bool(torch.isfinite(loss)), 'NONFINITE_TRAINING_LOSS')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['gradient_clip_norm'])
            require(bool(torch.isfinite(norm)) and all(bool(torch.isfinite(p.grad).all())
                    for p in model.parameters() if p.grad is not None), 'NONFINITE_GRADIENT')
            optimizer.step(); steps += 1; losses.append(float(loss.detach().cpu()))
        prediction = predict_standardized(model, vx, parameters['batch_size'], device)
        rmse = float(np.sqrt(np.mean((prediction - vy) ** 2)))
        require(math.isfinite(rmse), 'NONFINITE_VALIDATION_RMSE')
        if rmse < best - 1e-12:
            best, best_epoch, stale = rmse, epoch, 0
            save_new_checkpoint(checkpoint, model, seed, epoch, scaler, provenance, initial, prediction)
        else: stale += 1
        history.append(dict(epoch=epoch, train_standardized_MSE=float(np.mean(losses)),
                            validation_standardized_RMSE=rmse, validation_RMSE_mV=rmse * scaler.target_scale * 1000,
                            best_epoch=best_epoch, stale_epochs=stale))
        save_csv(run, f'seed{seed}/training_history.csv', history)
        save_json(run, f'seed{seed}/TRAINING_STATE.json', dict(status='TRAINING', completed_epoch=epoch,
                  best_epoch=best_epoch, optimizer_steps=steps, stale_epochs=stale))
        if stale >= parameters['patience']: break
    result = dict(status='COMPLETED_NEW_INSTANCE', seed=seed, best_epoch=best_epoch,
                  stop_epoch=epoch, optimizer_steps=steps, training_seconds=time.perf_counter() - started,
                  initialization_sha256=initial, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    save_json(run, f'seed{seed}/TRAINING_STATE.json', result)
    return checkpoint, pd.DataFrame(history), result


# 9. 最佳Checkpoint保存及加载
def save_new_checkpoint(file: Path, model: nn.Module, seed: int, epoch: int,
                        scaler: TrainOnlyScaler, provenance: dict, initial: str,
                        validation_prediction: np.ndarray) -> None:
    require(file.resolve().is_relative_to(OUTPUT_ROOT.resolve()), 'HISTORICAL_CHECKPOINT_WRITE_FORBIDDEN')
    payload = dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   model='F2', seed=seed, best_epoch=epoch, target_scale_V=scaler.target_scale,
                   main_program_sha256=sha256(Path(__file__)), hyperparameter_source=provenance,
                   initialization_sha256=initial,
                   full_validation_prediction=torch.from_numpy(validation_prediction.copy()),
                   scientific_identity='NEW_F2_INSTANCE_NOT_A_REPLACEMENT_FOR_HISTORICAL_RESULT')
    temporary = file.with_suffix('.pt.writing')
    torch.save(payload, temporary)
    for delay in [.02, .05, .1, .2, .5, 1., None]:
        try: os.replace(temporary, file); return
        except PermissionError:
            if delay is None: raise
            time.sleep(delay)

def frozen_seed_asset(seed: int) -> dict:
    """Explicit portable asset map, frozen from the actual six-candidate record."""
    require(seed in SUPPORTED_SEEDS, 'UNSUPPORTED_F2_SEED')
    require(sha256(SEED_ASSETS) == SEED_ASSETS_SHA256, 'F2_SEED_ASSET_MAP_SHA_MISMATCH')
    inventory = read_json(SEED_ASSETS)
    require(inventory['seeds'] == list(SUPPORTED_SEEDS) and
            [row['seed'] for row in inventory['models']] == list(SUPPORTED_SEEDS), 'F2_SEED_ORDER_MISMATCH')
    require(sha256(ROOT.parent / inventory['authority']) == inventory['authority_sha256'],
            'F2_SIX_CANDIDATE_AUTHORITY_SHA_MISMATCH')
    return next(row for row in inventory['models'] if row['seed'] == seed)


def historical_checkpoint(seed: int) -> pd.Series:
    row = pd.Series(frozen_seed_asset(seed))
    require(sha256(ROOT.parent / row.checkpoint) == row.checkpoint_sha256, 'F2_CHECKPOINT_SHA_MISMATCH')
    require(sha256(ROOT.parent / row.reference_prediction) == row.prediction_sha256, 'F2_REFERENCE_PREDICTION_SHA_MISMATCH')
    require(sha256(ROOT.parent / row.training_history) == row.training_history_sha256,
            'F2_TRAINING_HISTORY_SHA_MISMATCH')
    require(sha256(ROOT.parent / row.protocol_source) == row.historical_protocol_sha256,
            'F2_HISTORICAL_PROTOCOL_SHA_MISMATCH')
    return row

def load_checkpoint(model: nn.Module, file: Path, scaler: TrainOnlyScaler,
                    expected_seed: int, historical_row: pd.Series | None = None) -> dict:
    checkpoint = torch.load(file, map_location='cpu', weights_only=True)
    require(checkpoint['seed'] == expected_seed and checkpoint['model'] == 'F2', 'F2_CHECKPOINT_MODEL_SEED_MISMATCH')
    require(checkpoint['target_scale_V'] == scaler.target_scale, 'F2_CHECKPOINT_TARGET_SCALE_MISMATCH')
    if historical_row is not None:
        require(checkpoint['protocol_sha256'] == historical_row.historical_protocol_sha256,
                'F2_CHECKPOINT_PROTOCOL_MISMATCH')
        require(checkpoint['best_epoch'] == historical_row.best_epoch, 'F2_BEST_EPOCH_MISMATCH')
        require(checkpoint['initialization_sha256'] == historical_row.initialization_sha256,
                'F2_CHECKPOINT_INITIALIZATION_MISMATCH')
        require(checkpoint['input_fingerprints'] == historical_row.input_fingerprints and
                checkpoint['input_fingerprints'] == read_json(FROZEN_DATA / 'DATA_FINGERPRINTS.json'),
                'F2_CHECKPOINT_INPUT_FINGERPRINT_MISMATCH')
    model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
    require(set(model.state_dict()) == set(checkpoint['state_dict']), 'STATE_DICT_KEYS_MISMATCH')
    return checkpoint


# 10. Train/Validation/Assessment/后段预测
def predict_standardized(model: nn.Module, x: np.ndarray, batch_size: int,
                         device: torch.device) -> np.ndarray:
    model.eval(); chunks = []
    with torch.no_grad():
        for first in range(0, len(x), batch_size):
            history = torch.from_numpy(x[first:first + batch_size]).to(device)
            chunks.append(model.predict_delta(history).cpu().numpy())
    result = np.concatenate(chunks)
    require(result.shape == (len(x), HORIZON) and np.isfinite(result).all(), 'INVALID_F2_PREDICTION')
    return result

def prediction_frame(partition: str, seed: int, windows: Windows, delta_v: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(dict(model='F2', seed=seed, partition=partition,
                  origin_row=np.repeat(windows.origins, HORIZON), target_row=windows.targets.ravel(),
                  horizon_step=np.tile(np.arange(1, HORIZON + 1), len(windows.origins)),
                  origin_age_h=np.repeat(windows.origin_age_h, HORIZON),
                  target_age_h=windows.target_age_h.ravel(),
                  origin_raw_voltage_V=np.repeat(windows.origin_raw_voltage_V, HORIZON),
                  true_raw_target_voltage_V=windows.target_raw_voltage_V.ravel(),
                  y_true_delta_V=windows.y.ravel(), y_pred_delta_V=delta_v.ravel()))
    frame['predicted_voltage_V'] = frame.origin_raw_voltage_V + frame.y_pred_delta_V
    frame['true_label_voltage_V'] = frame.origin_raw_voltage_V + frame.y_true_delta_V
    return frame

def verify_historical_validation(frame: pd.DataFrame, reference: Path) -> dict:
    old = read_csv(reference)
    require(len(frame) == 4920 and frame.origin_row.nunique() == 205, 'VALIDATION_COUNT_MISMATCH')
    for name in ['origin_row', 'target_row', 'horizon_step', 'y_true_delta_V', 'y_pred_delta_V']:
        require(np.array_equal(frame[name].to_numpy(), old[name].to_numpy()),
                'VALIDATION_NOT_POINTWISE_EXACT:' + name)
    return dict(pointwise_exact=True, origins=205, queries=4920, max_prediction_difference_V=0.,
                reference=str(reference), reference_sha256=sha256(reference))

def predict_partitions(model: nn.Module, seed: int, windows: dict[str, Windows],
                        prepared: dict[str, np.ndarray], scaler: TrainOnlyScaler,
                        parameters: dict, run: Path, device: torch.device,
                        historical_row: pd.Series | None, checkpoint: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    # Validation is checked BEFORE any Train/Assessment/late-life predictions.
    before = state_sha256(model); frames = {}
    standardized = predict_standardized(model, prepared['VALIDATION'], parameters['batch_size'], device)
    v = prediction_frame('VALIDATION', seed, windows['VALIDATION'], standardized * scaler.target_scale)
    if historical_row is not None:
        parity = verify_historical_validation(v, ROOT.parent / historical_row.reference_prediction)
        require(np.array_equal(standardized[:32], checkpoint['fixed_validation_prediction'].numpy()),
                'HISTORICAL_FIXED_BATCH_PREDICTION_MISMATCH')
    else:
        require(np.array_equal(standardized, checkpoint['full_validation_prediction'].numpy()),
                'NEW_CHECKPOINT_RELOAD_PREDICTION_MISMATCH')
        parity = dict(new_checkpoint_reload_exact=True, historical_prediction_equality='NOT_APPLICABLE_NEW_TRAINING')
    frames['VALIDATION'] = v
    for name in ['TRAIN', 'HISTORICAL_ASSESSMENT', 'LATE_LIFE']:
        delta = predict_standardized(model, prepared[name], parameters['batch_size'], device) * scaler.target_scale
        frames[name] = prediction_frame(name, seed, windows[name], delta)
    reference_checks = {}
    for name in PARTITIONS:
        frame = frames[name]
        if historical_row is not None and historical_row.full_domain_historical_reference_available:
            # Also independently require exact parity with the full-domain frozen results.
            old = read_csv(FULL_EVALUATION / f'outputs/predictions/{name}_F2_seed{seed}.csv')
            for key in ['origin_row', 'target_row', 'horizon_step', 'y_true_delta_V', 'y_pred_delta_V',
                        'origin_raw_voltage_V', 'predicted_voltage_V']:
                require(np.array_equal(frame[key], old[key]), 'FULL_PARTITION_PARITY_MISMATCH:' + name + ':' + key)
            reference_checks[name] = 'POINTWISE_EXACT_FULL_DOMAIN_CACHE'
        elif historical_row is not None:
            reference_checks[name] = ('POINTWISE_EXACT_FROZEN_VALIDATION_CACHE' if name == 'VALIDATION' else
                                     'READ_ONLY_INFERENCE_NO_HISTORICAL_FULL_DOMAIN_CACHE')
        save_csv(run, f'seed{seed}/predictions/{name}.csv', frame)
    require(before == state_sha256(model), 'MODEL_PARAMETERS_CHANGED_DURING_INFERENCE')
    return frames, dict(**parity, state_unchanged=True, state_sha256=before,
                        partition_reference_checks=reference_checks,
                        all_four_partitions_pointwise_exact=historical_row is not None and
                            bool(historical_row.full_domain_historical_reference_available))


# 11. RMSE、MAE、R²计算
def calculate_metrics(frame: pd.DataFrame) -> dict:
    truth = frame.y_true_delta_V.to_numpy(float); pred = frame.y_pred_delta_V.to_numpy(float)
    errors = pred - truth; n = len(errors)
    sse = math.fsum(float(e) ** 2 for e in errors)
    mean = math.fsum(map(float, truth)) / n
    sst = math.fsum((float(y) - mean) ** 2 for y in truth)
    return dict(RMSE_mV=math.sqrt(sse / n) * 1000,
                MAE_mV=math.fsum(abs(float(e)) for e in errors) / n * 1000,
                R2=1 - sse / sst if sst > 0 else None,
                R2_target='raw DeltaV; same definition as historical H',
                origins=int(frame.origin_row.nunique()), queries=n,
                unique_target_times=int(frame.target_row.nunique()),
                scientific_status='in_sample' if frame.partition.iloc[0] == 'TRAIN' else
                  'reused_validation' if frame.partition.iloc[0] == 'VALIDATION' else 'EXPLORATORY_NOT_INDEPENDENT_TEST')


# 12. 电压预测曲线与训练曲线绘制
def plot_outputs(run: Path, seed: int, frames: dict[str, pd.DataFrame], history: pd.DataFrame,
                 best_epoch: int, mode: str, partition: str = DEFAULT_EVALUATION_PARTITION) -> None:
    """FC2-style three figures. The time curve is a fixed-lead rolling forecast.

    Historical logs supply the loss curves in evaluate mode, with no training.
    Scatter uses all 24-step queries; the time curve uses exactly one fixed lead
    per origin and breaks at missing half-hours. Neither selects metric samples.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': True, 'axes.spines.right': True, 'pdf.fonttype': 42})
    def save(fig: Any, name: str, data: pd.DataFrame) -> None:
        save_csv(run, f'seed{seed}/{name}_plot_data.csv', data)
        fig.savefig(output_path(run, f'seed{seed}/{name}.png'), dpi=300, facecolor='white')
        plt.close(fig)
    require(not history.empty, 'TRAINING_HISTORY_UNAVAILABLE_NO_SYNTHETIC_LOSS_CURVE')
    h = history[['epoch', 'train_standardized_MSE', 'validation_standardized_RMSE']].copy()
    h['validation_standardized_MSE'] = h.validation_standardized_RMSE ** 2
    save_csv(run, f'seed{seed}/training_history.csv', history)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(h.epoch, h.train_standardized_MSE, label='Train Loss', linewidth=1.5)
    ax.plot(h.epoch, h.validation_standardized_MSE, label='Validation Loss', linewidth=1.5)
    source = 'historical log; no retraining' if mode == 'evaluate' else 'new training instance'
    ax.set(xlabel='Epoch', ylabel='Standardized MSE Loss',
           title=f'Training History — F2 Seed {seed} ({source}; best epoch {best_epoch})')
    ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout(); save(fig, 'training_history', h)

    frame = frames[partition]
    require(frame.partition.eq(partition).all(), 'PLOT_PARTITIONS_MIXED')
    fixed = fixed_horizon_plot_data(frame)
    lines = voltage_plot_segments(fixed)
    status = 'reused Development Validation' if partition == 'VALIDATION' else 'exploratory; not independent Test'
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lines.target_age_h, lines.true_raw_target_voltage_V, label='True Voltage', linewidth=1.5, alpha=.8)
    ax.plot(lines.target_age_h, lines.predicted_voltage_V, label='Predicted Voltage',
            linewidth=1.5, alpha=.8, linestyle='--')
    ax.set(xlabel='Actual Operating Time (hours)', ylabel='Cell Voltage (V)',
           title=f'F2 Seed {seed} — {status}\nFixed step {PLOT_HORIZON_STEP} rolling predictions; n={len(fixed)}')
    ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout(); save(fig, 'prediction_vs_true', fixed)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(frame.true_raw_target_voltage_V, frame.predicted_voltage_V, alpha=.5, s=10)
    low = min(frame.true_raw_target_voltage_V.min(), frame.predicted_voltage_V.min())
    high = max(frame.true_raw_target_voltage_V.max(), frame.predicted_voltage_V.max())
    ax.plot([low, high], [low, high], 'r--', linewidth=1)
    ax.set(xlabel='True Cell Voltage (V)', ylabel='Predicted Cell Voltage (V)',
           title=f'F2 Seed {seed} — True vs Predicted\n{status}; all {len(frame)} queries (24 steps)')
    ax.grid(True, alpha=.3); fig.tight_layout(); save(fig, 'scatter_true_vs_pred', frame)


def fixed_horizon_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """One predetermined horizon per origin, preserving physical time identity."""
    fixed = frame.loc[frame.horizon_step == PLOT_HORIZON_STEP].sort_values('target_row').copy()
    require(len(fixed) == frame.origin_row.nunique() and
            fixed.origin_row.is_unique and fixed.target_row.is_unique, 'FIXED_LEAD_PLOT_IDENTITY_ERROR')
    require(fixed.partition.nunique() == 1, 'FIXED_LEAD_PLOT_PARTITIONS_MIXED')
    return fixed


def voltage_plot_segments(fixed: pd.DataFrame) -> pd.DataFrame:
    """Insert only plotting NaNs at real gaps; never interpolate predictions."""
    columns = ['target_age_h', 'true_raw_target_voltage_V', 'predicted_voltage_V']
    lines = []
    for index, (_, row) in enumerate(fixed.iterrows()):
        if index and (row.target_row != previous.target_row + 1 or
                      not math.isclose(row.target_age_h - previous.target_age_h, RESOLUTION_H,
                                       rel_tol=0, abs_tol=1e-8)):
            lines.append({name: math.nan for name in columns})
        lines.append({name: row[name] for name in columns}); previous = row
    return pd.DataFrame(lines, columns=columns)


def evaluation_summary(results: list[dict], partition: str, seeds: list[int]) -> dict:
    """One partition only; --seed all means equal per-seed means, no ensemble."""
    selected = [row for row in results if row['partition'] == partition]
    require(len(selected) == len(seeds) and sorted(row['seed'] for row in selected) == sorted(seeds),
            'SUMMARY_SEED_OR_PARTITION_MISMATCH')
    return dict(model='F2', partition=partition, seeds=seeds,
                aggregation='equal mean of per-seed metrics; no ensemble' if len(seeds) > 1 else 'single seed',
                RMSE_V=math.fsum(row['RMSE_mV'] for row in selected) / len(seeds) / 1000,
                MAE_V=math.fsum(row['MAE_mV'] for row in selected) / len(seeds) / 1000,
                R2=math.fsum(row['R2'] for row in selected) / len(seeds),
                R2_target='raw DeltaV; historical definition, not absolute-voltage R2',
                queries_per_seed=[row['queries'] for row in selected],
                scientific_status=selected[0]['scientific_status'])


def save_result_bundle(run: Path, seed: int, checkpoint: Path, scaler: TrainOnlyScaler,
                       provenance: dict, frame: pd.DataFrame) -> None:
    """FC2-style filenames in a fresh run; original assets stay read-only."""
    copied = output_path(run, f'seed{seed}/best_model.pth')
    require(not copied.exists(), 'OUTPUT_MODEL_ALREADY_EXISTS')
    shutil.copyfile(checkpoint, copied)
    require(sha256(checkpoint) == sha256(copied), 'COPIED_CHECKPOINT_IDENTITY_MISMATCH')
    save_csv(run, f'seed{seed}/predictions.csv', frame)
    save_json(run, f'seed{seed}/preprocess_info.json', dict(
        features=FEATURES, mean=scaler.mean.tolist(), scale=scaler.scale.tolist(),
        target_scale_V=scaler.target_scale, lookback=LOOKBACK, horizon=HORIZON,
        frozen_top10_indices=yaml.safe_load((HISTORY / 'configs/H_FROZEN_PROTOCOL.yaml').read_text(
            encoding='utf-8'))['preprocessing']['Top10_indices'],
        causal_EWMA_alpha=.3, voltage_recovery='raw origin V + predicted DeltaV',
        evaluation_partition=frame.partition.iloc[0], metric_steps='all 24; all legal queries',
        curve_horizon_step=PLOT_HORIZON_STEP, checkpoint_source=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), hyperparameters=provenance))


# 13. main()运行入口
def integration_self_checks() -> dict:
    text = Path(__file__).read_text(encoding='utf-8'); tree = ast.parse(text)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    for name, expected in MODEL_CLASS_AST_SHA256.items():
        require(hashlib.sha256(ast.dump(classes[name], include_attributes=False).encode()).hexdigest() == expected,
                'INTEGRATED_MODEL_CLASS_AST_CHANGED:' + name)
    allowed = {'__future__', 'argparse', 'ast', 'hashlib', 'json', 'math', 'os', 'random', 'sys',
               'time', 'traceback', 'warnings', 'dataclasses', 'datetime', 'pathlib', 'typing', 'shutil',
               'numpy', 'pandas', 'torch', 'yaml', 'pytorch_forecasting', 'matplotlib', 'platform'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names: require(name.name.split('.')[0] in allowed, 'PROJECT_PYTHON_IMPORT_FORBIDDEN')
        if isinstance(node, ast.ImportFrom):
            require(not node.level and node.module.split('.')[0] in allowed, 'PROJECT_PYTHON_IMPORT_FORBIDDEN')
    zero = torch.zeros((2, 48, 14))
    decoder = CanonicalStrictHistoryTFT.build_input(zero)['decoder_cont']
    require(bool((decoder[:, :, :14] == 0).all()), 'FUTURE_PHYSICAL_DECODER_VARIABLES_FOUND')
    require(torch.equal(decoder[0, :, -1], torch.arange(1, 25, dtype=torch.float32) / 24), 'DECODER_RELATIVE_TIME_CHANGED')
    x = np.ones((1, 48, 14), np.float32); changed = x.copy(); changed[:, 24:, 0] += 10
    require(np.array_equal(causal_ewma(x)[:, :24], causal_ewma(changed)[:, :24]), 'EWMA_NONCAUSAL')
    require(chronological_partitions(3843)['LATE_LIFE'] == (3005, 3843), 'PARTITION_BOUNDARY_ERROR')
    require(chronological_partitions(3999)['LATE_LIFE'] == (3005, 3999), 'N_WAS_HARDCODED_IN_PARTITIONS')
    # Check no future data enters x while its finite eligibility flag stays fixed.
    synthetic = pd.DataFrame({name: np.linspace(.5, .8, 200) for name in FEATURES})
    synthetic['bin_end_ophrs2_h'] = np.arange(200) * .5
    synthetic['bin_valid_for_windows'] = 1; synthetic['temporal_segment_id'] = 1
    original, _ = build_legal_windows(synthetic, 100, 200)
    synthetic.loc[101:, 'actual_current_A_std'] += 100
    mutated, _ = build_legal_windows(synthetic, 100, 200)
    require(np.array_equal(original.x_raw[0], mutated.x_raw[0]), 'FUTURE_CONDITIONS_ENTERED_INPUT')
    require(original.origins[0] == 100 and original.targets.max() == 199, 'SYNTHETIC_BOUNDARIES_FAILED')
    return dict(model_classes_AST_exact=True, model_class_count=len(MODEL_CLASS_AST_SHA256),
                no_project_Python_imports=True, relative_coordinate_only_decoder=True,
                causal_EWMA=True, synthetic_window_causality=True, no_training_executed_by_self_checks=True)

def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Frozen RealHyFC F2 single-file training/evaluation; default evaluate.')
    parser.add_argument('--mode', choices=['evaluate', 'train'], default='evaluate')
    parser.add_argument('--seed', choices=[*(str(seed) for seed in SUPPORTED_SEEDS), 'all'], default='42')
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--partition', choices=list(PARTITIONS[1:]), default=DEFAULT_EVALUATION_PARTITION,
                        help='One displayed/plotted evaluation partition; all partition results stay in files.')
    if argv is None:
        argv = sys.argv[1:]
        if 'ipykernel' in sys.modules:
            # VS Code Interactive/Jupyter supplies its own kernel connection file.
            # Strip only that option; keep model options and reject other typos.
            filtered = []
            index = 0
            while index < len(argv):
                argument = argv[index]
                if argument.startswith(('--f=', '-f=')):
                    index += 1
                    continue
                if argument in ('--f', '-f') and index + 1 < len(argv):
                    index += 2
                    continue
                filtered.append(argument)
                index += 1
            argv = filtered
    args = parser.parse_args(argv)
    if args.mode == 'train' and args.seed == 'all': parser.error('Train mode requires one seed; no automatic search.')
    return args


def main() -> None:
    # Windows redirected GBK consoles cannot encode the FC2-style superscript 2.
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    args = parse_arguments()
    seeds = list(SUPPORTED_SEEDS) if args.seed == 'all' else [int(args.seed)]
    run = OUTPUT_ROOT / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + f'_F2_{args.mode}_seed{args.seed}')
    run.mkdir(parents=True, exist_ok=False)
    try:
        environment = configure_environment(args.device); checks = integration_self_checks()
        frame, source = load_full_realhyfc(); windows = {}; exclusions = []
        for name, (start, end) in chronological_partitions(len(frame)).items():
            windows[name], excluded = build_legal_windows(frame, start, end)
            excluded['partition'] = name; exclusions.append(excluded)
        window_audit = verify_window_identity(windows, frame)
        scaler, prepared, scores = preprocess_frozen(windows)
        provenance = load_historical_hyperparameters(); parameters = provenance['parameters']
        save_json(run, 'FROZEN_RUN_CONFIGURATION.json', dict(mode=args.mode, seeds=seeds,
                  supported_seeds=list(SUPPORTED_SEEDS), seed_assets_sha256=sha256(SEED_ASSETS),
                  selected_seed_assets=[frozen_seed_asset(seed) for seed in seeds],
                  seed_population='Validation-selected instances; not unbiased random-seed population',
                  displayed_evaluation_partition=args.partition, formal_metric_horizons=list(range(1, HORIZON + 1)),
                  plot_rule=dict(fixed_horizon_step=PLOT_HORIZON_STEP, gap_handling='break line; no interpolation',
                                 selection='all legal origins; no error-based selection'),
                  full_source=source, hyperparameters=provenance, environment=environment,
                  main_program_sha256=sha256(Path(__file__)), model_sources=MODEL_SOURCE_SHA256,
                  scaler=dict(mean=scaler.mean.tolist(), scale=scaler.scale.tolist(), target_scale_V=scaler.target_scale),
                  preprocessing='Train-only unfiltered scaler; frozen Top10; causal per-window EWMA alpha=0.3',
                  scientific_limits='Historical Assessment and late-life exploratory; Phase L unchanged'))
        save_csv(run, 'WINDOW_COUNTS_AND_IDENTITIES.csv', window_audit)
        save_csv(run, 'EXCLUDED_WINDOWS.csv', pd.concat(exclusions, ignore_index=True))
        save_csv(run, 'TRAIN_ONLY_SPEARMAN_AUDIT.csv', scores)
        results = []; audits = []
        for seed in seeds:
            model = initialize_f2(seed, parameters, torch.device(args.device))
            if args.mode == 'evaluate':
                row = historical_checkpoint(seed); file = ROOT.parent / row.checkpoint
                checkpoint = load_checkpoint(model, file, scaler, seed, row)
                history_file = ROOT.parent / row.training_history
                history = read_csv(history_file); training = dict(training_performed=False, optimizer_steps=0)
            else:
                row = None
                file, history, training = train_new_f2(model, seed, parameters, windows['TRAIN'],
                       windows['VALIDATION'], prepared['TRAIN'], prepared['VALIDATION'], scaler,
                       run, provenance, torch.device(args.device))
                checkpoint = load_checkpoint(model, file, scaler, seed)
            before_file = sha256(file)
            frames, parity = predict_partitions(model, seed, windows, prepared, scaler, parameters,
                                  run, torch.device(args.device), row, checkpoint)
            require(sha256(file) == before_file, 'CHECKPOINT_FILE_CHANGED')
            for name in PARTITIONS:
                metrics = calculate_metrics(frames[name]); results.append(dict(model='F2', seed=seed, partition=name, **metrics))
            save_result_bundle(run, seed, file, scaler, provenance, frames[args.partition])
            plot_outputs(run, seed, frames, history, checkpoint['best_epoch'], args.mode, args.partition)
            audits.append(dict(seed=seed, checkpoint=str(file), checkpoint_sha256=before_file,
                               best_epoch=checkpoint['best_epoch'], training_history=str(history_file) if row is not None else None,
                               prediction_sha256=row.prediction_sha256 if row is not None else None,
                               trainable_parameters=332436, **parity, **training))
            del model, checkpoint; torch.cuda.empty_cache()
        save_csv(run, 'F2_PARTITION_METRICS.csv', results)
        summary = evaluation_summary(results, args.partition, seeds)
        save_json(run, 'evaluation_results.json', summary)
        save_csv(run, 'evaluation_results.csv', [{**summary, 'seeds': ','.join(map(str, seeds))}])
        save_json(run, 'INTEGRATION_ACCEPTANCE.json', dict(status='PASS', mode=args.mode,
                  self_checks=checks, windows=window_audit, seeds=audits,
                  historical_results_modified=False, optuna_executed=False,
                  new_training_executed=args.mode == 'train', program_sha256=sha256(Path(__file__))))
        files = [dict(path=f.relative_to(run).as_posix(), bytes=f.stat().st_size, sha256=sha256(f))
                 for f in sorted(run.rglob('*')) if f.is_file()]
        save_csv(run, 'ARTIFACT_MANIFEST.csv', files)
        print(f'RMSE: {summary["RMSE_V"]:.6f} V', flush=True)
        print(f'MAE : {summary["MAE_V"]:.6f} V', flush=True)
        print(f'R²  : {summary["R2"]:.4f}', flush=True)
    except Exception:
        atomic_text(run, 'FAILURE.txt', traceback.format_exc()); raise

if __name__ == '__main__':
    main()

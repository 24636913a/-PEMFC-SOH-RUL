"""Side-effect-free shared data/model definitions, copied by AST from frozen F2.
No old interactive program is imported or executed. Only Top10 is run here.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import ast
import hashlib
import json
import math
import os
import random
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OFFICIAL=ROOT/'RealHyFC_Raw1s_Rolling_300to120/archives/F2_legacy_9ba5daaf/F2_Train_Interactive.py'
REFERENCE_RUN=ROOT/'F2_LSTM_DualPath_Comparison/outputs/20261019T090430_205588Z_Top10_LSTM_F2'
SOURCE=ROOT/'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
SOURCE_CODE_SHA='3dbdc602eb2229a54ee7d5eaa841a6dd94798e8227dbfc6f60db1e78f66419de'
SUPPORTED_SEEDS = (42, 103, 193, 202, 210, 212)

LOOKBACK, HORIZON, RESOLUTION_H = 48, 24, .5

FEATURES = ['cell_voltage_mean', 'cell_voltage_std', 'cell_voltage_range',
            'cell_voltage_min', 'cell_voltage_max', 'actual_current_A_mean',
            'actual_current_A_std', 'actual_current_A_peak_to_peak',
            'actual_current_density_A_cm2_mean', 'actual_current_density_A_cm2_std',
            'actual_current_density_A_cm2_peak_to_peak', 'load_transition_count',
            'operation_point_transitions', 'zero_current_fraction']

MASKS = {'top10': [0, 3, 13, 11, 4, 9, 6, 7, 10, 8],
         }

HYPERPARAMETERS = dict(learning_rate=0.0009605449687795866,
    weight_decay=6.95942232671908e-06, batch_size=32, dropout=.1,
    hidden_size=64, hidden_continuous_size=32, attention_head_size=4,
    lstm_layers=1, gradient_clip_norm=1., optimizer='AdamW', loss='MSE')

def require(condition, message):
    if not condition: raise RuntimeError(message)

def array_sha(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, str(array.dtype))).encode() + array.tobytes()).hexdigest()

def state_sha(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()

def configure_environment():
    require(torch.cuda.is_available(), 'CUDA_GPU_REQUIRED_FOR_FORMAL_TRAINING')
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(False)
    warnings.filterwarnings('ignore', message="Attribute '(loss|logging_metrics)' is an instance of")
    return dict(python=sys.version, torch=str(torch.__version__), CUDA=torch.version.cuda,
                GPU=torch.cuda.get_device_name(0), bitwise_reproducibility_promised=False)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def partitions(n):
    require(n == 3843, 'EXPECTED_3843_REAL_BINS')
    return dict(TRAIN=(0, int(.7 * n)), VALIDATION=(int(.7 * n), int(.8 * n)), TEST=(int(.8 * n), n))

@dataclass
class Windows:
    x_raw: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    targets: np.ndarray
    origin_age_h: np.ndarray
    target_age_h: np.ndarray
    origin_voltage: np.ndarray
    target_voltage: np.ndarray

def build_windows(frame, start, end):
    values = frame[FEATURES].to_numpy(float); ages = frame.bin_end_ophrs2_h.to_numpy(float)
    voltage = frame.cell_voltage_mean.to_numpy(float)
    valid = frame.bin_valid_for_windows.astype(bool).to_numpy()
    segment = frame.temporal_segment_id.to_numpy()
    xs, ys, origins, targets, excluded = [], [], [], [], []
    for origin in range(start, end):
        first, last = origin - LOOKBACK + 1, origin + HORIZON
        reasons = []
        if first < 0: reasons.append('HISTORY_BEFORE_DATASET')
        if last >= end: reasons.append('24_TARGETS_OUTSIDE_PARTITION')
        if not reasons:
            block = slice(first, last + 1)
            if not valid[block].all(): reasons.append('INVALID_BIN')
            if np.any(segment[block] != segment[origin]): reasons.append('TEMPORAL_SEGMENT_BREAK')
            if not np.allclose(np.diff(ages[block]), .5, rtol=0, atol=1e-6): reasons.append('OPERATING_TIME_GAP')
            if not np.isfinite(values[block]).all(): reasons.append('NONFINITE_RAW14_OR_VOLTAGE')
        if reasons:
            excluded.append(dict(origin_row=origin, reasons='|'.join(reasons))); continue
        xs.append(values[first:origin + 1]); ys.append(voltage[origin + 1:last + 1] - voltage[origin])
        origins.append(origin); targets.append(np.arange(origin + 1, last + 1))
    o = np.asarray(origins, np.int64); t = np.asarray(targets, np.int64).reshape(-1, HORIZON)
    return Windows(np.asarray(xs, np.float32).reshape(-1, LOOKBACK, 14),
        np.asarray(ys, np.float32).reshape(-1, HORIZON), o, t, ages[o], ages[t], voltage[o], voltage[t]), pd.DataFrame(excluded)

class TrainScaler:
    def fit(self, x):
        flat = x.astype('float64').reshape(-1, 14)
        self.mean = flat.mean(0); self.scale = flat.std(0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.)
        return self

    def transform(self, x): return ((x - self.mean) / self.scale).astype(np.float32)
    @property
    def target_scale(self): return float(self.scale[0])

def causal_ewma(x):
    result = x.copy()
    for step in range(1, LOOKBACK): result[:, step, 0] = .3 * x[:, step, 0] + .7 * result[:, step - 1, 0]
    return result

def masked_inputs(x, scaler, mode):
    require(mode in MASKS, 'ONLY_FIXED_TOP10_INPUTS')
    result = scaler.transform(causal_ewma(x))
    result[:, :, [i for i in range(14) if i not in MASKS[mode]]] = 0.
    return result

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

MODEL_CLASS_AST_SHA256 = {'CanonicalStrictHistoryTFT': 'b68e1a0a0e9236d75b3b969300ebfa536d6256835d87931f6c0b83f82f0c2c5b', '_LeftPaddedDepthwiseConv1d': '7f460c56d88c1d30d6d68b65aa821f4f65f19bb186beda943d487322b6776127', 'CausalMultiScaleHistoryBranch': '7e061c13d798f42ccf22af3a7b1348f6897ad7b4c1a20387681f1be98d1b747c', 'DualPathStrictHistoryTFT': '152795fa73b26a5cb9f5b15ba951b666b8942bec0547a4e7569775f22ac0efeb', 'RegimeFusionGate': '1a1e7b579b27bacb8809b3495b3314854e198277f6ac63999abb60e39d251e42', 'RegimeAwareDualPathTFT': '3b10169740cb49bed34381f66be63f7dd71669f322ccef78c38bbc259dfd829b'}

def calculate_metrics(points):
    truth = points.y_true_delta_V.to_numpy(float); error = points.y_pred_delta_V.to_numpy(float) - truth
    n = len(error); sse = math.fsum(float(x) ** 2 for x in error)
    mean = math.fsum(map(float, truth)) / n
    return dict(RMSE_V=math.sqrt(sse / n), MAE_V=math.fsum(abs(float(x)) for x in error) / n,
                R2=1 - sse / math.fsum((float(x) - mean) ** 2 for x in truth),
                R2_target='raw DeltaV', queries=n, origins=int(points.origin_row.nunique()))

def prediction_frame(frame, windows, delta, seed, mode, partition):
    result = pd.DataFrame(dict(seed=seed, feature_mode=mode, partition=partition,
        origin_row=np.repeat(windows.origins, 24), target_row=windows.targets.ravel(),
        horizon_step=np.tile(np.arange(1, 25), len(windows.origins)),
        origin_age_h=np.repeat(windows.origin_age_h, 24), target_age_h=windows.target_age_h.ravel(),
        origin_raw_voltage_V=np.repeat(windows.origin_voltage, 24), true_voltage_V=windows.target_voltage.ravel(),
        y_true_delta_V=windows.y.ravel(), y_pred_delta_V=delta.ravel()))
    result['predicted_voltage_V'] = result.origin_raw_voltage_V + result.y_pred_delta_V
    result['target_actual_current_A'] = frame.actual_current_A_mean.to_numpy()[result.target_row]
    result['target_actual_current_density_A_cm2'] = frame.actual_current_density_A_cm2_mean.to_numpy()[result.target_row]
    return result


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):result.update(block)
    return result.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def read_csv(path):return pd.read_csv(path,float_precision='round_trip')

def model_identity():
    require(sha(OFFICIAL)==SOURCE_CODE_SHA,'OFFICIAL_MODEL_SOURCE_CHANGED')
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    nodes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    for name,expected in MODEL_CLASS_AST_SHA256.items():
        require(hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest()==expected,'MODEL_AST_CHANGED:'+name)
    return dict(F2_classes_exact=True,F2_parameters=332436,native_F2_only=True)




def load_selection_data():
    """3843-row contract, but build/model/preprocess Train+Validation ONLY."""
    frame=pd.read_csv(SOURCE)
    require(len(frame)==3843 and frame.index.equals(pd.RangeIndex(3843)), '3843_BIN_IDENTITY')
    require(frame.bin_end_ophrs2_h.is_unique and frame.bin_end_ophrs2_h.is_monotonic_increasing,'TIME_ORDER')
    windows={};excluded=[]
    for part,(start,end) in {'TRAIN':(0,2690),'VALIDATION':(2690,3074)}.items():
        w,e=build_windows(frame,start,end)
        require(len(w.origins)=={'TRAIN':1750,'VALIDATION':218}[part],'WINDOW_COUNT')
        require(w.targets.min()>=start and w.targets.max()<end,'TARGET_BOUNDARY')
        windows[part]=w;e['partition']=part;excluded.append(e)
    require(windows['TRAIN'].targets.max()<windows['VALIDATION'].origins.min(),'TRAIN_LABEL_CAUSALITY')
    scaler=TrainScaler().fit(windows['TRAIN'].x_raw)
    prepared={part:masked_inputs(w.x_raw,scaler,'top10') for part,w in windows.items()}
    prior=read_json(REFERENCE_RUN/'FROZEN_PROTOCOL.json')
    for part,w in windows.items():
        ref=next(r for r in prior['partitions'] if r['partition']==part)
        require(array_sha(prepared[part])==ref['inputs_sha256'],'HISTORICAL_INPUT_IDENTITY')
        old=read_csv(REFERENCE_RUN/f'windows/{part}.csv')
        require(np.array_equal(old.origin_row,np.repeat(w.origins,24)),'HISTORICAL_ORIGIN_IDENTITY')
        require(np.array_equal(old.target_row,w.targets.ravel()),'HISTORICAL_TARGET_IDENTITY')
        require(np.array_equal(old.horizon_step,np.tile(np.arange(1,25),len(w.origins))),'HISTORICAL_STEP_IDENTITY')
        # Independent calendar continuity check on the complete legal window.
        first=pd.to_datetime(frame.first_datetime).astype('int64').to_numpy()
        last=pd.to_datetime(frame.last_datetime).astype('int64').to_numpy()
        for origin in w.origins:
            lo=origin-47;hi=origin+24
            gaps=(first[lo+1:hi+1]-last[lo:hi])/1e9
            require((gaps>=0).all() and (gaps<=1.01).all(),'CALENDAR_BREAK')
    return frame,windows,scaler,prepared,pd.concat(excluded,ignore_index=True)

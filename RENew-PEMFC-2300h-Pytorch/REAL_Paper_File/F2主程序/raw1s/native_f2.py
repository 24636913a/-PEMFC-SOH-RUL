"""Native F2 class bodies copied statically; no imports of historical programs.
Provenance: root RealHyFC_F2_Main.py sha256
64cde85cac939806641921083bcf2c2e5354ee2bcd19dc4315949459305660f9.
Adaptations: 24 raw-second slots, 300 history, name-based gate semantics;
legacy 48-step helper removed. Wrapper supplies 120 relative-only decoder queries.
"""
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from .features import FEATURE_NAMES, GATE_NAMES
RAW_NAMES = [f"raw_f2_{i:02d}" for i in range(24)]
RELATIVE_STEP = "relative_forecast_step"
X_REALS = [*RAW_NAMES, RELATIVE_STEP]
RAW_F2_FEATURE_NAMES = tuple(FEATURE_NAMES)
REGIME_FEATURE_NAMES = tuple(GATE_NAMES)
class CanonicalStrictHistoryTFT(TemporalFusionTransformer):
    """Canonical TFT with a direct standardized point-voltage forecast head."""

    def __init__(self, *, hidden_size: int, hidden_continuous_size: int, attention_head_size: int, lstm_layers: int, dropout: float) -> None:
        super().__init__(
            hidden_size=hidden_size,
            hidden_continuous_size=hidden_continuous_size,
            attention_head_size=attention_head_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            output_size=1,
            max_encoder_length=300,
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

        The task's target is already standardized V and uses MSE outside
        pytorch-forecasting's probabilistic/quantile loss interface.  Therefore
        no target normalizer or rescaling is applied here.
        """
        return prediction

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

        # This block mirrors pytorch-forecasting 1.8.0 _tft.py:530–678.0 The
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
        if tuple(RAW_NAMES) != tuple(f"raw_f2_{i:02d}" for i in range(24)):
            raise RuntimeError("Frozen Raw F2 schema identity mismatch")
        self.regime_indices = tuple(RAW_F2_FEATURE_NAMES.index(name) for name in REGIME_FEATURE_NAMES)
        self.regime_gate = RegimeFusionGate()

    def regime_context(self, encoder_cont: torch.Tensor) -> torch.Tensor:
        if encoder_cont.ndim != 3 or encoder_cont.shape[-1] < 24:
            raise RuntimeError("Expected scaled/masked encoder continuous tensor with 24 raw-second slots")
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

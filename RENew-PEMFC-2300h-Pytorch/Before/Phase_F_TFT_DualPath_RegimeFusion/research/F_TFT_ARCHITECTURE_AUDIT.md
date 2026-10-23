# F — Actual Canonical TFT Architecture Audit

## Audited runtime

- Adapter: `Phase_TFT_Original_Baseline/models/canonical_tft.py` (`CanonicalStrictHistoryTFT`).
- Upstream package: `pytorch-forecasting 1.8.0`.
- Upstream sources inspected: `_tft.py` SHA-256 `6ad5f6c806d314cfec53851ce0914c6d3476c4b05bd136a2764f4a562f1b7343`; `sub_modules.py` SHA-256 `9ef892c565f6891ddf6d21c52c4d7f90393ec731924851aee872468cffe2cf9e`.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124. The planned Phase-F width is the Phase-E frozen width `hidden_size=64`, `hidden_continuous_size=32`, four attention heads and one LSTM layer.

A synthetic `[B=2,48,14]` call through the existing adapter (no data file read and no training) returned `prediction=[2,24,1]`, `encoder_attention=[2,24,4,48]`, `decoder_attention=[2,24,4,24]`, `encoder_variables=[2,48,1,15]`, and `decoder_variables=[2,24,1,1]`. Thus the only selected decoder variable is the deterministic relative step.

## Strict-history task adapter

`CanonicalStrictHistoryTFT.build_input()` creates:

- encoder continuous tensor `[B,48,15]`: 14 standardized historical Raw F2 channels plus `relative_forecast_step` from -47/24 to 0;
- decoder continuous tensor `[B,24,15]`: fourteen zero-valued schema placeholders plus deterministic steps 1/24 to 24/24;
- empty categorical and static tensors.

The 14 decoder placeholders are not included in the decoder VSN. `time_varying_reals_decoder=[relative_forecast_step]`; hence no measured future current, voltage, temperature, pressure, RH, flow, or target can reach the decoder. The formal model output is a raw standardized-DeltaV head `[B,24,1]`, exposed as `[B,24]` by `predict_delta()`.

## Actual forward graph

The following is a source-derived summary of `_tft.py:530–678`, not a generic TFT sketch.

1. The implementation concatenates encoder and decoder time axes, maps each continuous real independently through `prescalers`, and creates `input_vectors`.
2. There are no static covariates in this task. Upstream therefore creates an all-zero `static_embedding [B,H]`; the static-context GRNs are still instantiated but receive zero information.
3. `encoder_variable_selection` receives the 15 encoder variables and the zero static-selection context. It returns `E_vsn [B,48,H]` and instance/time-specific sparse weights `[B,48,1,15]`. `decoder_variable_selection` processes only the one relative-step variable, returning `D_vsn [B,24,H]` and `[B,24,1,1]` weights.
4. `lstm_encoder(E_vsn, initial_hidden, initial_cell)` returns encoder output plus `(hidden, cell)`. `lstm_decoder(D_vsn, (hidden, cell))` is a separate module and consumes those terminal encoder states. Encoder and decoder do **not** share LSTM weights.
5. Each LSTM path applies the same `post_lstm_gate_encoder` GLU object and the same `post_lstm_add_norm_encoder` object (the decoder attributes alias the encoder objects in this installed version). The skip tensors are `E_vsn` and `D_vsn` respectively.
6. Concatenated post-LSTM encoder/decoder representations receive static enrichment. Interpretable attention uses decoder timesteps as queries and all 72 encoder-plus-decoder positions as keys and values. Its output is gated/residual-normalized against the decoder portion of the attention input, passed through a position-wise GRN, then gated/residual-normalized against the decoder portion of the concatenated post-LSTM tensor before the linear output head.

## Variable selection network (VSN)

For each continuous real, `VariableSelectionNetwork` applies a learned prescaler and a variable-specific GRN to create an `H`-wide representation. A flattened GRN produces a softmax over variables and the output is their weighted sum. Therefore it is **instance- and time-dependent variable weighting**, not a fixed Spearman selector. Phase E's Top10 mechanism is external: after common Train-only scaling it sets four fixed input channels to zero. The VSN still receives all 15 encoder slots; the zeroed channels can be downweighted but their positions remain structurally present.

The VSN selects variables. It does not construct a separate local receptive-field representation after selection. A future F2 regime gate must therefore not reimplement a 14-variable selector; it should consume only a small fixed operational-summary vector and decide between already-produced temporal paths.

## GRN, GLU, residual and static paths

`GatedResidualNetwork` is `Linear -> optional context Linear addition -> ELU -> Linear -> GateAddNorm`. `GateAddNorm` is GLU followed by residual addition and LayerNorm; GLU is a linear expansion to twice the output width followed by `F.glu`. These are learned transform/skip controls, not an explicit operating-regime mixture of two encoders.

Distinct GRNs are created for variable selection, initial LSTM hidden state, initial LSTM cell state, static enrichment, position-wise processing, and individual variables. They are not globally shared. In this strict-history task the nominal static contexts are all zero; they must not be misdescribed as observed physical context.

## Attention and causal mask

`InterpretableMultiHeadAttention` uses independent query/key projections per head, a shared value projection, averages heads, then projects back to `H`. Its queries are `[B,24,H]`; keys and values are `[B,72,H]`. With `causal_attention=True`, `_tft.py:get_attention_mask()` masks a decoder query from attending to itself and later decoder positions, while allowing encoder positions and earlier decoder positions. It is temporal causality inside the decoder attention, not permission to insert future measured covariates.

## Safest F1 insertion point

The recommended insertion is **after** the existing post-LSTM encoder gate/add-norm and **before** the 72-step concatenation sent to static enrichment and attention:

`E_vsn -> native encoder LSTM -> E_lstm` in parallel with `E_vsn -> causal convolution branch -> E_conv`, then `E_fused = 0.5*(E_lstm + E_conv)`. Decoder execution, encoder-to-decoder state handoff, attention mask, attention block, output head and all existing skips are retained. This preserves dimensions, decoder information restrictions, and the native output contract.

It is not safe to insert before VSN: a convolution over raw variables would change the semantics of variable selection and obscure the F0→F1 contrast. Replacing the LSTM would discard its state handoff to the decoder. Adding a branch after attention would not be a temporal encoder comparison and would disrupt attention interpretability.

## Implementation consequence

The upstream `forward()` is monolithic. A future project-local subclass must override its `forward()` and copy the audited upstream flow exactly, substituting only the encoder representation immediately before `torch.cat([encoder, decoder], dim=1)`. It must call the inherited modules by name and must not monkey-patch the installed `pytorch-forecasting` package. A source-hash guard and synthetic parity test are required before training: F0-style subclass behavior with the new branch disabled must match the native adapter on a fixed synthetic input.

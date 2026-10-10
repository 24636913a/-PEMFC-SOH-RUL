# F1 Recommended Architecture — Native TFT + Causal Multi-Scale History Branch

## Frozen recommendation

Implement a project-local subclass named, for example, `DualPathStrictHistoryTFT`. It retains the complete `CanonicalStrictHistoryTFT` contract and inherited TFT modules. Its only new F1 module is `CausalMultiScaleHistoryBranch`; F1 then replaces the native encoder tensor entering attention with a fixed equal mixture of native and convolutional encoder representations.

This is an implementation specification, not a model implementation and not a training authorization.

## Exact tensor contract

Let `H=64`, `L=48`, `P=24`, and `B` be batch size. The normal adapter produces `X=[B,L,14]`, `E_vsn=[B,L,H]`, and `D_vsn=[B,P,H]`.

1. Preserve native operations exactly:
   - `E_lstm = post_lstm_add_norm_encoder(post_lstm_gate_encoder(lstm_encoder(E_vsn)), E_vsn)`, shape `[B,48,64]`.
   - `D_lstm = post_lstm_add_norm_decoder(post_lstm_gate_decoder(lstm_decoder(D_vsn, native_hidden_cell)), D_vsn)`, shape `[B,24,64]`.
2. New history branch consumes **only** `E_vsn`:
   - transpose to `[B,64,48]`;
   - depthwise left-padded causal `Conv1d(64,64,kernel_size=3,dilation=1,groups=64)` -> `[B,64,48]`;
   - independent depthwise left-padded causal `Conv1d(64,64,kernel_size=3,dilation=4,groups=64)` -> `[B,64,48]`;
   - concatenate -> `[B,128,48]`, apply pointwise `Conv1d(128,64,kernel_size=1)`, transpose -> `[B,48,64]`;
   - apply the existing configured dropout rate and a new `LayerNorm(64)` after a residual addition with `E_vsn`, yielding `E_conv=[B,48,64]`.
3. F1 fusion is immutable and parameter-free:

`E_f1 = 0.5 * E_lstm + 0.5 * E_conv`, shape `[B,48,64]`.

4. Restore the native downstream contract:

`lstm_output = cat([E_f1, D_lstm], time)` -> `[B,72,64]`;
`static_enrichment -> attention -> post_attn_gate_norm -> pos_wise_ff -> pre_output_gate_norm -> output_layer` remains unchanged; prediction remains `[B,24,1]` then `[B,24]`.

The branch is causal because each convolution uses only left padding and clips no future value into a history position. The branch does not receive decoder tensors. No convolution runs across the encoder/decoder boundary.

## Why this exact location

The new branch must see the same VSN-selected historical tensor as the native encoder. It must not use raw unselected variables before VSN, and it must not replace the native LSTM because that LSTM supplies the `(hidden, cell)` used by the decoder. Fusion after the LSTM residual path preserves both the native LSTM path and the decoder state handoff while retaining the native attention and output modules without a shape or mask change.

## Required future source layout

Create only in the next implementation phase:

```text
Phase_F_TFT_DualPath_RegimeFusion/
  models/
    causal_multiscale_branch.py
    dualpath_strict_history_tft.py
  tests/
    test_dualpath_shapes_and_causality.py
    test_f0_parity_when_branch_disabled.py
    test_decoder_future_covariate_firewall.py
```

`dualpath_strict_history_tft.py` must copy the audited PTF `forward()` flow into a subclass and record exact upstream source hashes. It may alter only the construction of `E_f1` and add branch modules. It must not modify `canonical_tft.py`, `pytorch_forecasting`, VSN, LSTM modules, decoder placeholders, attention mask, output head, loss, or optimizer.

## Initialization and parameter accounting

Use PyTorch's declared Conv1d initialization, initialize the new LayerNorm with unit weight and zero bias, and record every new tensor hash after global seeding. At `H=64`, the proposed branch adds 8,896 trainable parameters: two depthwise convolutions (512 total), a 128-to-64 pointwise convolution (8,256), and LayerNorm (128). If F0 has 323,483 trainable parameters under the Phase-E configuration, F1 is expected to have 332,379. The actual count must be computed and recorded at construction time; it may not be asserted without measurement.

## Mandatory causal and compatibility tests

1. Shape: `[B,48,14] -> [B,24]` for arbitrary legal `B`.
2. Branch causal perturbation: perturbing input at history time `u` must not change `E_conv[:, :u, :]`.
3. Decoder firewall: perturbing the fourteen zero decoder placeholders must not change prediction; no future physical tensor may be accepted by the input API.
4. Upstream parity: a no-branch/native-control path must equal `CanonicalStrictHistoryTFT` on a fixed synthetic input under `eval()` and identical weights.
5. State-handoff audit: hooks must verify that native encoder terminal `(hidden,cell)` are passed unchanged to native decoder.
6. Attention audit: query/key/value widths and `[B,24,4,48]`/`[B,24,4,24]` returned attention shapes remain valid.

## F0→F1→F2 interpretation

- F0 tests the newly run native baseline with all frozen Phase-E information.
- F1 minus F0 tests the fixed dual-path encoder, not a condition gate.
- F2 minus F1 tests only the gate described in `F2_REGIME_AWARE_FUSION_DESIGN.md`.

No Assessment result may select the candidate, branch width, kernels, dilation, gate dimension, or fusion type. Those choices are frozen here before F0–F2 execution.

# T0 architecture identity audit

The runtime is `pytorch-forecasting` TemporalFusionTransformer 1.8.0, used through a strict-history task adapter. It exposes continuous variable selection, GRNs, distinct LSTM encoder/decoder modules with state handoff, static enrichment with an explicitly empty static context, interpretable multi-head attention, causal attention masking, gated residual normalization, position-wise GRN, and a 24-step output. The only decoder variable is the deterministic relative forecast step. The tensor's fourteen raw placeholder positions are schema placeholders and are not decoder-selection variables; no future measured covariate or target is read.

Synthetic [64,48,14] forward/backward passed with parameter count 86,883, finite gradient norm 12.367393, and GPU peak 90.7 MiB.

## Post-training confirmation

The same audited implementation completed all three frozen folds. The model is a full PTF TemporalFusionTransformer with VSN, separate LSTM encoder/decoder state handoff, gating/GRNs, causal interpretable multi-head attention, and extractable interpretation tensors.

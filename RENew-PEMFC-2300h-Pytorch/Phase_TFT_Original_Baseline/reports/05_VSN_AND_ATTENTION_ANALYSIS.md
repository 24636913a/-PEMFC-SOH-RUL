# VSN and attention analysis

The exported VSN weights and attention values are from the actual completed canonical TFT forward path, not a reconstructed visualization. Encoder variable-selection weights aggregate validation samples and encoder time. Decoder variable selection has one available known-future input: the deterministic relative forecast-step coordinate. The Fold 3 attention heatmap aggregates actual causal encoder attention over validation samples and heads.

These weights are model-internal importance summaries, not causal feature-effect estimates and not a post-hoc feature-selection rule.

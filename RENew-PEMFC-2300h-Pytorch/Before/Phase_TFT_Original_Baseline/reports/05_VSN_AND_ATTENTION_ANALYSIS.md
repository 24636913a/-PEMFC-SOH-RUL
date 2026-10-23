# VSN and attention analysis

The exported VSN weights and attention values are from the actual completed canonical TFT forward path, not a reconstructed visualization. Encoder variable-selection weights aggregate validation samples and encoder time. Decoder variable selection has one available known-future input: the deterministic relative forecast-step coordinate. The Fold 3 attention heatmap aggregates actual causal encoder attention over validation samples and heads.

| Fold | largest encoder VSN weight | VSN normalized entropy | largest encoder attention weight | encoder attention normalized entropy |
|---|---:|---:|---:|---:|
| fold_1 | 0.2573 | 0.8390 | 0.0887 | 0.4650 |
| fold_2 | 0.1853 | 0.9335 | 0.0440 | 0.5905 |
| fold_3 | 0.2516 | 0.8839 | 0.0370 | 0.5610 |

The encoder VSN does not collapse to a single variable (largest weight ≤ 0.257) and its normalized entropy remains 0.839–0.933. Encoder attention is neither uniform nor a single-position spike (maximum averaged cell ≤ 0.089; normalized entropy 0.465–0.591). These are descriptive internal checks, not causal feature-effect estimates and not a post-hoc feature-selection rule.

# xLSTMTime author-implementation mapping

| Paper/architecture responsibility | Author source | Forward-path fact | Gate result |
|---|---|---|---|
| Input and native trend/seasonal separation | `xlstm.py:xlstm.forward`; `series_decomp2` | A 25-point native moving average separates the raw input before projection. | Present |
| 48→24 forecast projection | `Linear_Seasonal`, `Linear_Trend`, `xlstm.forward` | Each variable is projected from `context_points=48` to `target_points=24`. | Present |
| xLSTM representation | `xlstm.py:mm`, `xlstm1/xlstm_block_stack.py:xLSTMBlockStack` | The projected 24-step representation is mapped to `n2=256` then processed by the stack. | Present |
| mLSTM matrix-memory mechanism | `xlstm1/blocks/mlstm/backends.py` | Causal matrix combination and stabilized exponential gates are in the active mLSTM backend. | Present |
| Forecast head | `xlstm.py:mm2`, final `permute` | Maps 256 to 24 and returns `[B,24,enc_in]`. | Present |

## Critical interpretation

At the frozen author commit, `xLSTMBlockStackConfig.__post_init__()` derives the
effective block map from `slstm_at=[]`; despite the literal `_block_map=1` in
`xlstm.py`, the active three blocks are mLSTM. The xLSTM stack receives the
post-projection `[B, variables, 256]` representation, so this author forecaster
does **not** recurrently consume the original 48 time positions. Phase BG keeps
this behavior intact rather than rewriting it into a time-axis LSTM.

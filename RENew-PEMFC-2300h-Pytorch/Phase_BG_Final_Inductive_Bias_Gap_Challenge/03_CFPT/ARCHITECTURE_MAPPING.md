# CFPT architecture mapping

The source audit confirms that the official body contains the paper's two
named mechanisms. This is an architecture result, not a performance result.

| Claimed responsibility | Verified official source location | Forward-path evidence |
|---|---|---|
| Cross-frequency decomposition | `models/CFPT.py:FrequencyDecomposition.forward` | `rfft` separates low and high spectra. |
| Cross-frequency interaction | `FrequencyModel.forward`, `InteractionBlock.forward` | Low/high complex components are embedded separately, concatenated, then processed by exponential affine interaction blocks. |
| Frequency prediction/reconstruction | `FrequencyModel.forward` | Separate low/high frequency predictors are recombined through inverse real FFT to `[B,pred_len,N]`. |
| Periodic-aware timestamp representation | `Model.time_enc` | A timestamp vector is transformed through MLP layers, period unfolding, `Conv2d`, fold-back, and output projection. |
| Intra/inter-period operation | `UnfoldAndReshape`, `Conv2d`, `FoldBackToSequence` | The timestamp branch forms a 2-D periodic representation before convolution. |
| Fusion and forecast head | `Model.forecast` | The frequency and time forecasts are blended, then de-normalized. |
| Decoder timestamps in the actual forward | `Model.forward` → `Model.forecast` | `x_mark_dec` is passed to `forecast`, encoded by `time_enc`, and contributes to the blend. |

At commit `1ded0ef…`, `x_mark_enc` and `x_dec` appear in the public function
signature but the forecast path uses `x_enc` and `x_mark_dec`. The decoder
timestamp branch is not an optional, dead argument.

`CFPT_PROTOCOL_DECISION.md` records why that verified architecture cannot be
trained under either permitted Phase BG protocol without an impermissible
change.

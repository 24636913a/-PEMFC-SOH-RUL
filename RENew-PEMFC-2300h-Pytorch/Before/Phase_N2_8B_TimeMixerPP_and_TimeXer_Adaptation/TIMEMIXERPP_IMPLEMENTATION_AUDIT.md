# TimeMixer++ implementation audit

## Identity and fixed source

- Paper: Wang et al., *TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis*, ICLR 2025, OpenReview `1CLzLXSFNn`.
- Repository: `https://github.com/WenjieDu/PyPOTS`.
- Fixed vendor commit: `22fe29a3ef02da054c8ac6d56f4d6dc2eac2359c` (the requested `22fe29a` lineage; release v1.2).
- Status: **maintained public reproduction / maintained implementation**. It is not labelled official author code. The original anonymous author repository is no longer accessible; PyPOTS documents its implementation as inspired by that repository.
- Audited source paths: `pypots/forecasting/timemixerpp/` and `pypots/nn/modules/timemixerpp/`.

## Core-mechanism mapping

| Paper mechanism | Pinned PyPOTS mapping |
|---|---|
| MRTI | `MixerBlock.time_imaging` creates period-shaped 2-D time images. |
| TID | `MixerBlock.dual_axis_attn`, `RowAttention`, and `ColAttention` split trend and seasonal patterns. |
| MCM | `MixerBlock.multi_scale_mixing`, `MultiScaleSeasonCross`, and `MultiScaleTrendCross` exchange patterns across scales. |
| MRM | `MixerBlock.multi_reso_mixing` frequency-weights multiple resolutions and adds the residual. |
| Multi-scale input | `BackboneTimeMixerPP.__multi_scale_process_inputs` uses the fixed average downsampler. |
| Multi-resolution prediction / head | one fixed `predict_layers[i]` and shared `projection_layer` per resolution. |

## Output-fusion audit

The pinned maintained `forecast()` builds `dec_out_list` for every resolution but then returns only the final `dec_out` produced by the loop. Therefore the unchanged `TimeMixer++-PyPOTS` run is a faithful execution of the maintained implementation but does **not** combine every element of `dec_out_list` at the final output.

The paper describes integration of predictions from multiple resolutions, but the accessible paper source does not provide reproducible learned fusion weights. `TimeMixer++-PaperFaithful` changes only that terminal return: it applies an equal mean over the same already-computed resolution-specific prediction tensors. This is explicitly a **paper-guided equal-weight reconstruction**, not official author code and not a new architecture. No mixer block, input feature, loss, optimizer, scaler, hidden size, or training schedule was otherwise changed.

## Frozen small configuration

`n_layers=2`, `d_model=32`, `d_ffn=64`, `top_k=3`, `n_heads=4`, `n_kernels=4`, average downsampling with two levels (48/24/12 steps), channel mixing and channel independence enabled, external train-only scaler, and internal RevIN disabled. Parameter counts are recorded in the seed-42 comparison table.

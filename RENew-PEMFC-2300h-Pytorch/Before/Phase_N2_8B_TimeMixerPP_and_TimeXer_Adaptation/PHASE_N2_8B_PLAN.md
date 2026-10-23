# Phase N2.8B execution plan

## Frozen task

- Resolution: 0.5 h; lookback: 24 h / 48 steps; horizon: 12 h / 24 steps.
- Feature group: exact frozen F2 (14 columns); target: future `Delta V` relative to the known origin voltage.
- Development protocol: the established chronological three-fold development folds, with train-only feature and target scalers.
- Test firewall: no Test targets, predictions, metrics, distributions, plots, or model-selection information may be accessed.

## Scope

1. Vendor and pin `WenjieDu/PyPOTS` at the requested `22fe29a3ef02da054c8ac6d56f4d6dc2eac2359c` commit. Audit its forecasting and neural-module TimeMixer++ paths.
2. Run exactly two TimeMixer++ variants at seed 42: unchanged PyPOTS forward (`TM++-P`) and a narrowly scoped PaperFaithful equal-weight fusion of `dec_out_list` (`TM++-F`) if the maintained forward does not fuse all resolutions.
3. Re-run exactly two allowed TimeXer input-role checks at seed 42: PEMFC-A (one endogenous voltage channel and 13 historical exogenous F2 channels) and PEMFC-Life (the same, plus only a train-standardized origin-known scalar lifecycle context repeated across the past input window). No future age sequence is constructed.
4. Compare the four new runs with frozen LSTM, TimesNet HF, and original TimeXer results. Advance only a model satisfying the frozen promotion conditions to seeds 2024 and 3407.
5. Freeze a development-only model decision from R2_delta, RMSE_delta, Fold3, tails, lifecycle ratio, seed stability, and complexity.

## Fixed experimental settings

- Loss / optimizer: MSE / AdamW; learning rate 0.001; weight decay 0.0001; batch size 64; maximum 100 epochs; patience 15; gradient clipping 1.0.
- TM++ small maintained-implementation configuration: two mixer blocks, `d_model=32`, `d_ffn=64`, `top_k=3`, four heads, four kernels, three resolutions (`downsampling_layers=2`, window 2), channel independence and channel mixing enabled, and no internal RevIN because the frozen external train-only scaling is used.
- TimeXer uses its fixed official vendor commit and the N2.8A small configuration. It receives only historical F2 values. The Life context is the standardized origin age scalar, not future covariates.

## Stop conditions

- No architectures beyond TimeMixer++ and the two specified TimeXer adaptations.
- TimeXer permanently stops unless an adaptation has mean RMSE_delta <= 11.9 mV and Fold3 RMSE below original TimeXer. The stricter >25 mV and 20–25 mV stopping rules are also recorded.
- A TimeMixer++ version runs multi-seed only if it satisfies any frozen N2.8B promotion rule versus TimesNet HF.
- Stop after seed-42 comparison, only eligible multiseed repeats, diagnostics, and the frozen decision. Do not enter a new phase.

# F2 Regime-Aware Fusion Design

## Sole change relative to F1

F2 retains the exact F1 data, scaler, VSN, LSTM encoder/decoder, causal convolution branch, attention, output head, loss and optimizer. It changes only

`E_f1 = 0.5 E_lstm + 0.5 E_conv`

to

`E_f2 = g E_lstm + (1-g) E_conv`, where `g` is one scalar per sample, broadcast to `[B,48,64]`.

## Regime context

The gate may use only five fixed, historical quantities already among Phase-E Top10:

`J = {load_transition_count, zero_current_fraction, actual_current_density_A_cm2_std, actual_current_A_std, actual_current_density_A_cm2_mean}`.

From the 48-step **scaled, causally transformed historical** input, compute a temporal mean for each listed channel:

`r = mean_t(X[:,t,J])`, `r=[B,5]`.

This is an observed-past regime summary: it receives neither decoder tensors nor `t+1:t+24` physical observations. EWMA only affects historical voltage and does not add future values. It cannot observe the target DeltaV.

## Frozen gate

`g = sigmoid(W2 * ELU(W1*r + b1) + b2)`, with `W1: 5->8` and `W2: 8->1`.

Initialize `W2` and `b2` to zero so `g=0.5` at initialization. This ensures the initial F2 fusion equals F1 fusion and makes the causal distinction experimentally narrow. F2 adds 57 trainable parameters relative to F1. No entropy, balance, auxiliary, monotonicity, physics, or gate-regularization loss is permitted: adding one would change more than fusion.

## Why sample-level scalar gating

A channel gate `[B,64]` or time-step gate `[B,48,1]` is more expressive but creates an additional feature/temporal-selection mechanism whose effect cannot be cleanly separated from VSN and attention in this small-sample setting. The scalar gate answers a narrower question: does the observed historical operating regime change the relative value of the native LSTM and causal-convolution histories?

This is not a physical constraint or a PINN. Gate values are data-driven mixing weights. They may be summarized after training (mean, standard deviation, 5th/50th/95th percentiles and fraction below 0.1/above 0.9) only as diagnostics. They cannot be called degradation mechanisms. This F2 delta is an exploratory TFT architecture change and does not reopen the frozen TimesNet/main-backbone route.

## Failure diagnostics, not tuning rules

- If `g` saturates near 0 or 1 for nearly every sample, report branch collapse; do not change the gate or loss during the registered run.
- If F2 does not improve F1, conclude only that this frozen regime-fusion form did not show added value under the single split.
- If gate inputs have a constant channel in Train, retain the channel definition but record its zero variance and the resulting limited information; do not substitute another feature post hoc.

## Required tests

1. A future-only perturbation must leave `r`, `g`, and predictions unchanged.
2. Altering a non-regime channel at fixed regime channels must not alter `g` directly (but may alter paths and final prediction).
3. At initialization `g` equals 0.5 to numerical tolerance.
4. Replacing the gate by exactly 0.5 reproduces F1 fusion before later native TFT processing.
5. Gate shape is `[B,1]`, and fusion shape is `[B,48,64]`.

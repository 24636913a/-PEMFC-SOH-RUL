# Frozen baseline reproduction

The exact OPR frozen summary was reused. Mean three-fold DeltaV RMSE is P0 24.538245 mV, P1 24.442883 mV, P2 21.346566 mV, and RS 24.201795 mV. Each agrees with the prescribed reference within ±0.001 mV.

The pointwise audit also retains the OPR identities: P1 and P2 share B in scaled space, `Y_P2 - Y_P1 = C_P2 - C_P1`, and the mV² D identity residual remains within the OPR float64 audit tolerance. This phase did not recreate model predictions, load a checkpoint, or use Test data.

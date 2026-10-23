# P1–P2 error attribution

At every Development prediction point, the analysis verifies `D=e_P2²−e_P1² = 2e_P1ΔC+ΔC²` in mV². The maximum numerical residual is 0.027660480 mV², attributable to float32 checkpoint output followed by mV conversion and squaring.

Fold 3 continuous-nonzero current is the principal localized loss: P1 RMSE 26.684831 mV versus P2 32.812165 mV, with mean D 364.557952 mV². Its largest positive-D horizons are 11.5 h, 0.5 h, 2.0 h, 12.0 h, 1.5 h. This attributes the output difference to the changed predicted-condition input at the frozen checkpoint; it is not evidence of an electrochemical causal mechanism.

At 12 h, the aggregate deterioration is primarily Fold 1: P1/P2 RMSE 31.682133/41.296982 mV, while Fold 3 is actually improved at that exact horizon (38.170891/35.528181 mV). Thus “12 h degradation” must not be attributed to Fold 3 alone.

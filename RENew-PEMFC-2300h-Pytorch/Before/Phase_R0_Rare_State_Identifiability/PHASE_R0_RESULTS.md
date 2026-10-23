# Phase R0 — Results

**Status: PASS. Decision: `Q_ERROR_SUPPORTED`.** This is an **EXPLORATORY_DEVELOPMENT_DIAGNOSIS** with Test closed and no new model training.

1. P0/P1/P2/RS reproduce at 24.538245/24.442883/21.346566/24.201795 mV (`baseline_reproduction.csv`).
2. Test Firewall is PASS (`test_firewall_audit.json`).
3. P2 training OOF is strict causal: 15 completed-target block checks pass.
4–6. Train global independent events: E1=66, E2=9; Validation E1=9, E2=2.
7–9. Train/Validation E1 patterns are similar; event support is SUFFICIENT across 7 segments; probabilities overpredict on aggregate and are not declared calibrated.
10–12. True conditional statistics and CF00 reconstruction pass. The frozen formal P2 mean remains 21.346566 mV; Oracle comparisons use common valid samples only.
13–18. Fold1/12 h CF00/CF10/CF01/CF11=41.297/16.204/41.529/16.392 mV; Fold3/continuous-nonzero=32.812/13.144/32.648/13.274 mV.
19–21. q/conditional/interaction Shapley-style attribution is 1449.43/-12.64/-13.03 mV² in Fold1 and 896.80/3.63/14.16 mV² in Fold3.
22–26. q improvements recur over state-error points; normal points are nearly neutral for q but conditional-only correction is mildly adverse. Other P2 peak-to-peak/transition errors remain; B is invariant, so the mechanism is propagation through C, not a proof that TimeXer is intrinsically defective.
27–31. A future q representation is supported for preregistration only; conditional-statistics-only and TimeXer-internal modifications are unsupported. Deterministic future-condition identifiability is not established. The only candidate is probabilistic physics-consistent future-condition modeling.
32. **New model training: NO** in R0.
33. **Test remains closed.**

All numerical evidence is under `outputs/`; Oracle CFs are explicitly not deployable results.

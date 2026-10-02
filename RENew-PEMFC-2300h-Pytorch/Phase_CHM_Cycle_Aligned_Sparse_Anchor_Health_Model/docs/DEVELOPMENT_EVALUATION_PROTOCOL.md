# Development Evaluation Protocol

## Baselines

Three low-complexity baselines are mandatory; none may be replaced by a modern
backbone.

- **B0 — raw P1S:** `H_c=R_c`.
- **B1 — isotonic calibration:** a monotone P1S-to-anchor calibration fitted
  only on training anchors within the relevant evaluation fold.
- **B2 — low-order monotone fit:** a constrained low-order monotone P1S-to-
  anchor calibration. Its exact implementation is an open technical decision;
  it must be fixed before any fit and then used identically in every fold.

No baseline receives age, cycle index, or a held-out anchor value.

## Two non-random chronological evaluations

1. **LOAO retrospective diagnostic.** Hold out each non-BoL anchor at 300,
   600, 850, 1150, 1500, and 1750 h in turn. Train only on the other five
   non-BoL Development anchors plus the fixed BoL reference. The held-out
   polarization value cannot enter training, lambda selection, early stopping,
   anchor-noise tuning, or any calibration decision. Its cycle operating
   observation remains a legal unlabeled causal observation.
2. **Causal-online chronological-prefix diagnostic.** For anchor `k`, use
   only the fixed BoL reference and non-BoL anchors strictly earlier than `k`.
   The held-out anchor and every later anchor are unavailable. This test is
   reported separately and controls the claim hierarchy for an online health
   estimator. The 300 h case must be explicitly labeled as a BoL-only sparse-
   supervision condition, not silently excluded.

The state recurrence is causal in both tracks. The retrospective LOAO score is
not presented as online deployment performance.

## Metrics and reporting

Each held-out checkpoint must be reported individually. The minimum report is:

1. anchor MAE and RMSE;
2. Spearman rho, with the small-anchor count stated;
3. latent monotonic-violation count (expected structural zero);
4. cycle-health roughness;
5. calibration slope and bias; and
6. 95% uncertainty coverage, interval width, and each anchor's inclusion flag.

Mean-only tables are prohibited. No success claim may rely on Spearman alone.

## Development qualification gate

The primary model may receive `DEVELOPMENT_HEALTH_MODEL_QUALIFIED` only when
all of the following are met using frozen Development protocol choices:

1. all six LOAO anchors have legal predictions;
2. aggregate LOAO MAE is better than B0;
3. it is no worse than the best interpretable monotone baseline;
4. there is no structural rise in latent health;
5. bootstrap uncertainty has reasonable documented coverage; and
6. S1 does not reverse the conclusion.

A developmental PASS does not open Test and does not authorize RUL.

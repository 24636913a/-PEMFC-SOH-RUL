# Stage 0 — physical-loss feasibility

## Frozen candidate

The only candidate was `TARGET_FUTURE_ACTUAL_CONDITIONED_POLARIZATION_ORDERING`:
for two same-horizon forecast queries with matched target-time conditions and
`j_high-j_low >= 0.20 A cm^-2`, penalize
`ReLU(Vhat_high-j - Vhat_low-j)`.  It directly differentiates through predicted
absolute voltage and the synthetic gradient contract passed.

## Information-set finding

The candidate needs target-time **actual** current density, temperature,
relative humidity, and pressure to select pairs.  These variables are labelled
`future-unknown actual` by the frozen N1 dictionary.  They are unavailable at a
prediction origin and are not authorized as inference inputs or training-pair
selectors.  Therefore empirical Fold-1 outer-train pair support cannot repair
the information-set failure.

## Decision

**PHYSICS_LOSS_NOT_ESTABLISHED**.  No Physics-TimesNet branch is trained, and the protocol forbids
substituting an easier regularizer.  Broad voltage bounds and input identities
such as `J=I/A` are not adopted: the former is not a condition-specific
forecasting law and the latter does not constrain TimesNet voltage output.

## Audit facts

- Fold-1 outer-train candidate pairs: `9452`.
- Active-pair fraction of candidate queries: `0.375795`.
- Minimum pair support requirement: `200`.
- Future-condition information gate: **FAIL**.
- Direct differentiability/finite-gradient contract: **True**.

This is a forecasting feasibility decision only.  It does not make an SOH,
irreversible-degradation, EOL, or RUL claim.

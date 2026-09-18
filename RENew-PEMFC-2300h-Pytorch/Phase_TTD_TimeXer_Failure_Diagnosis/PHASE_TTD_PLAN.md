# Phase TTD — TimeXer Failure Diagnosis

## Frozen scope

- Start commit: `8a30f68346088b82511a3f71c1de9f59b316464a`.
- Development task: 0.5 h causal bins, 48 h history, 24 h horizon, DeltaV target, 14 historical features and 9 future condition features.
- Folds: the three pre-registered development folds. Final Test remains sealed.
- Sources are read-only: Phase N2.8F, Phase G-STOP, Phase RS and Phase OPR.

## Permitted work

This phase reuses the OPR pointwise P0/P1/P2 decomposition and performs frozen causal auxiliary-condition reconstruction only where OPR did not retain P2 mixed-state probability. It will audit timing, future-condition magnitude, FiLM response and index alignment. No neural-network model is trained, retrained, modified or saved.

## Decision framework

The diagnosis distinguishes: (A) transition-timing error, (B) future-condition magnitude error, (C) excessive FiLM condition response with condition-state/magnitude support, (D) index/data/label error, or (E) mixed/inconclusive. Evidence must be reported separately for Fold 1 at 12 h and Fold 3 `continuous_nonzero`; no mechanism is declared dominant without the pre-specified concentration evidence in both objects.

## Audit rules

- All comparison metrics are in mV or mV² and retain the exact OPR pointwise definitions.
- A sequence is indexed by the same origin and strictly future target step. Cross-segment and boundary rows are flagged, never silently removed.
- State matching uses actual mixed state `q > 0` and P2 predicted mixed state `probability >= 0.5`; event matching uses a fixed ±2-step, same-origin, same-type rule.
- Condition amplitude groups use fold-specific train-only, causal P2 OOF current-error quantiles. No voltage error selects the threshold.
- Any follow-up direction is a plan only. Model-experiment authorization is not granted by this phase.

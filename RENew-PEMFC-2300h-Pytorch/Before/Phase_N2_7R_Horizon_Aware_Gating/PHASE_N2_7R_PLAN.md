# Phase N2.7R Plan — Horizon-Aware Adaptive Gating Feasibility Reassessment

## Frozen scope

- Experts are frozen: N2.7 `LSTM + P2_causal_sg` and high-fidelity `TimesNet T0 + P0_raw`.
- The protocol remains 0.5 h sampling, F2 (14 features), 48-step history, and 24-step/12 h raw-voltage target.
- Only lightweight meta-gates are trained from cached Development predictions and causal origin features. Neither expert is run or fine-tuned during gate training.
- Final Test `[1819.997738, 2256.800000] h` is excluded at loading time. No Test target, expert prediction, gate output, weight, metric, feature distribution, or operating-age analysis is created.

## Meta-learning design

Each record is `(seed, fold, origin, horizon)`.

- Meta evaluation 1: gate train = Fold1 validation; OOS = Fold2 validation.
- Meta evaluation 2: gate train = Fold1+Fold2 validation; OOS = Fold3 validation.
- Meta-OOS aggregates Fold2 and Fold3 only. Gate training folds are never reported as OOS performance.
- Gate A is Logistic Regression on causal state summaries plus normalized horizon.
- Gate B is Logistic Regression on Gate A inputs plus deployable expert-prediction/disagreement features.
- Gate C is the fixed lightweight `32 → GELU → Dropout(0.1) → 16 → GELU → 1 → Sigmoid` MLP, optimized only for raw-voltage fusion MSE.
- Logistic gates use probability as a continuous TimesNet weight. Gate C also emits a continuous weight. There is no hard switch.
- The main gate excludes absolute `OpHrs2`; a separate best-gate diagnostic adds current origin `OpHrs2`, scaled only on the corresponding meta-training segment.

## Causality and scaling

- G0 uses only the historical 48-step F2 window at the prediction origin.
- G1 is current forecast horizon information.
- G2 contains only cached frozen expert predictions, their signed/absolute disagreement, and deltas to the current known origin voltage.
- All gate feature medians/scalers and disagreement tertiles are fitted on the current meta-training segment only.
- Ageing stage, future load, future target, future operating hours, fold identity, and Test-derived statistics are not gate inputs.

## Comparators and selection

For the same OOS records, compare LSTM, TimesNet, fixed `g=0.25`, Horizon-Static Fusion with pre-registered per-horizon weights `{0,0.25,0.50,0.75,1}`, Gate A, Gate B, and Gate C.

- Seed 42 executes all three pre-registered gates.
- All three gate types then execute with paired expert seeds `[42, 2024, 3407]`; the final main choice is the lowest 3-seed Meta-OOS RMSE gate without `OpHrs2`.
- The same final gate architecture is repeated with current `OpHrs2` only for the operating-age diagnostic. It cannot replace the main gate by default.
- Paired moving-block bootstrap uses 2,000 repetitions and a 19-origin (9.5 h) circular block, retaining all 24 horizons inside each sampled origin.

## Decision rules

Phase N2.8 is GO only if the best main adaptive gate:

1. improves Meta-OOS RMSE over fixed `g=0.25` by at least 2%;
2. does not worsen Fold3 relative to fixed fusion;
3. improves Meta-OOS RMSE over LSTM by at least 5%; and
4. has non-degenerate weights, documented by distribution and horizon-wise summaries.

The pointwise-winner AUC is supporting evidence, not an independent veto. If Horizon-Static Fusion is within 1% of the best adaptive gate, the report will prefer the simpler horizon-dependent explanation.

## Required artifacts

The 18 required CSV files, Test firewall audit, 10 figures, full results report, and `PHASE_N2_7R_GATE_DECISION.yaml` are generated only in this phase directory. Old phases are read-only.

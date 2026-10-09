# Phase T0 final decision

## Status: `CANONICAL_TFT_BASELINE_ESTABLISHED`

The pre-training architecture audit passed and all three frozen Seed-42 Development folds completed with no Test access. This phase establishes a reproducible, strict-history **canonical full TFT baseline**, rather than reusing the historical incomplete `StrictHistoryTFT` implementation.

- Formal T0 three-fold equal RMSE: **26.158 mV**.
- Frozen context only: Raw F2 TimesNet Seed-42 = 14.207 mV; EMD-feature TimesNet Seed-42 = 13.335 mV; historical TFT-like result = 19.914 mV.

## Evidence-led answers

1. **Canonical versus earlier TFT-like model.** The new canonical TFT is worse by 6.244 mV in three-fold equal RMSE. This is not an identity contradiction: the legacy TFT-like model is not accepted as canonical because it lacked the verified PTF component pathway. T0 is the valid architecture control.
2. **Canonical TFT versus Raw F2 TimesNet.** T0 is worse by 11.951 mV (84.1% of the Raw TimesNet baseline). This is a frozen, same-origin Development context comparison, not a new backbone-screening decision.
3. **Fold behaviour.** The canonical TFT does not reproduce the historical narrative of a relatively better Fold 3: its Fold 3 RMSE is 33.453 mV versus 26.680 mV for frozen Raw TimesNet and 19.395 mV for the legacy TFT-like record.
4. **Most evidence-supported failure hypotheses.** The output is not constant or grossly over-smoothed, and VSN/attention do not show a single-variable or single-position collapse. The remaining evidence is consistent with insufficiently well-conditioned fitting of the full TFT under this limited chronological data regime; Fold 2/3 best epochs near the fixed 300-epoch cap make training budget/convergence a protocol-controlled follow-up question. No transition-specific claim is supported because that subgroup was not exported.
5. **Potential T1 parameter priorities.** If an independent T1 protocol is approved, it should pre-register only evidence-motivated candidates: training budget/patience, hidden/continuous widths, dropout, learning rate/weight decay, and attention-head count. It must retain the strict history firewall, frozen fold structure, and no-future-covariate rule.
6. **Mandatory repair before any future run.** None was identified: Stage 0 architecture, data firewall, scaler scope, output shape, prediction invariance, and interpretability extraction all passed. Performance is not a structural or data-interface failure.
7. **T1 status.** A separately pre-registered Optuna-TFT study is scientifically permissible as a future decision, but it is **not started by T0** and requires an explicit new instruction and protocol freeze.

The historical TFT-like result is not treated as this model: it lacked the verified full canonical TFT component set and interpretable attention pathway. Context comparisons do not reopen backbone selection or claim a new formal multi-model ranking. T0 changes no scientific target, feature universe, fold, test boundary, or future-covariate policy. It creates the necessary canonical TFT control for later preregistered research decisions.

No Test prediction, SOH/RUL target, future actual operating condition, decomposition, physics loss, or hyperparameter search was used. The phase stops here; follow-up model changes require a new protocol.

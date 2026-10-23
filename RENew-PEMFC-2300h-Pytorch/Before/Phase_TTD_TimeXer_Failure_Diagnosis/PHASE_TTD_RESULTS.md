# Phase TTD results

## Scope and reproduction

Start commit is `8a30f68346088b82511a3f71c1de9f59b316464a`. The development-only frozen task is 0.5 h bins, 48 h history, 24 steps/12 h horizon, DeltaV, 14 historical features, 9 future-condition features and three development folds. Test was not loaded, predicted, diagnosed, or used for any decision.

OPR frozen baselines reproduce exactly: P0/P1/P2/RS mean RMSE are 24.538245, 24.442883, 21.346566 and 24.201795 mV. The B+C decomposition, P1/P2 shared B property and D identity are inherited from the OPR pointwise audit and used unchanged.

## Object A: Fold 1, 12 h

P1/P2 RMSE is 31.682133/41.296982 mV. Net D is +163492.180 mV², but 93.03% of positive D comes from one mixed-state false positive at 1130.0 h and 98.36% from the top five unique targets. The remaining state-correct low-amplitude group favors P2. Object A is a sparse exceptional-state failure, not evidence of global timing or label misalignment.

## Object B: Fold 3 continuous-nonzero

The official pointwise object has 6502 points and P1/P2 RMSE 26.684831/32.812165 mV. It has positive net D +2.370 million mV². Its predeclared shortest-horizon unique-target diagnostic has 328 targets and P1/P2 RMSE 29.776766/49.117095 mV. The loss is distributed over four temporal segments; it is not the Fold 1 single-target phenomenon.

## Mechanism decision

Timing is nearly exact under same-origin ±2-step matching, and zero-shift future-condition error is far smaller than any ±1/±2 shift. Alignment is PASS for all 70 high-loss records. Thus neither A as a global timing mechanism nor D as an index/data/label mechanism is supported.

State-error groups account for the harmful portions in both objects: two Fold 1 and 54 Fold 3 `predicted mixed / true nonmixed` points. Yet frozen evidence cannot separate state probability error, associated current-magnitude error, and FiLM response at those same rare events. State-correct groups are generally favorable to P2, so C cannot be claimed as harm under accurate conditioning.

**Final decision: INCONCLUSIVE (mixed rare-state failure; not a global temporal-shift or alignment error).** No new model training is authorized.

## Output map

- Reproduction: `outputs/reproduction/baseline_reproduction.csv`
- Objects A/B: `outputs/fold1_12h/` and `outputs/fold3_nonzero/`
- Timing: `outputs/transitions/`
- Magnitude and FiLM: `outputs/amplitude/` and `outputs/film_response/`
- Alignment and firewall: `outputs/time_alignment/` and `outputs/audit/test_firewall_audit.json`

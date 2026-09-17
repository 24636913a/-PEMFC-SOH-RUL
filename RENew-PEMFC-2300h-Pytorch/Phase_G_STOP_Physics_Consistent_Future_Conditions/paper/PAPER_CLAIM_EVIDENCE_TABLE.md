# Claim–Evidence Table

| Claim | Status | Direct evidence | Boundary |
|---|---|---|---|
| Current/current density obey one fixed-area relation | Supported | raw audit slopes/correlations and identity errors | Development data only |
| P2 predicts mixed-current occurrence better than P0 | Supported | mixed-event F1 0.990 vs 0.510 | q=1 absent |
| P2 removes audited future-condition violations | Supported | trajectory violation 0% vs P0 100% | only audited constraints |
| P2 improves frozen SA-TimeXer mean RMSE | Supported by point-estimate Gate B | 24.538→21.347 mV; 2 folds | bootstrap CI crosses zero; Fold3 worsens |
| P2 retraining improves seed42 SA-TimeXer | Supported for seed42 | 24.538→21.553 mV | multiseed trigger failed |
| Physics-consistent futures improve the strong LSTM | Refuted for tested head | 7.926→21.467 mV | one preregistered transfer design |
| q is a shutdown label | Unsupported | no shutdown log and no q=1 bins | use operating-regime terminology |
| New method is a stable final model | Unsupported | no new three-seed result | incumbent LSTM remains reference |
| Model predicts SOH or RUL | Unsupported | task is dynamic voltage prediction | no dense health/RUL ground truth |
| Test or cross-stack generalization | Unsupported | Test sealed; one experimental system | separate authorization/study needed |

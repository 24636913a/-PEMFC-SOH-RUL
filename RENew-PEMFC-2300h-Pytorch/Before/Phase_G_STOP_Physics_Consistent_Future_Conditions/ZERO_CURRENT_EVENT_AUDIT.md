# Zero-Current Event Audit

Status: **SUFFICIENT FOR MIXED-CURRENT ANALYSIS; q=1 NOT ESTIMABLE**.

Across Development there are **624** mixed-current bins forming **624** independent segments, plus **624** nonzero-to-mixed and **621** mixed-to-nonzero transitions. Validation support is 77, 67, and 156 mixed segments in Folds 1–3. The mixed-bin proportion is 20.90%.

No q=1 bin was observed. Therefore this work uses the terms *zero-current fraction*, *mixed-current operating regime*, and *zero-current transition*; it does not claim measured shutdown/startup state. Event counts are based on unique bins or state runs rather than treating overlapping 24-step targets as independent events.

P2 reduces frozen-TimeXer mixed-current RMSE in all folds (31.87→17.59, 35.26→23.72, and 39.98→35.93 mV for P0→P2), but Fold3 continuous-nonzero RMSE worsens (26.89→32.81 mV). This localization explains why the overall Fold3 metric does not improve.

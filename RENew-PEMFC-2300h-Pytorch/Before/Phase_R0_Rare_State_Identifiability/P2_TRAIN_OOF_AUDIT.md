# P2 train OOF audit

**PASS — strict causal OOF.** Each block fit uses only targets completed on or before the next prediction origin (`max_fit_target_end_h <= min_prediction_origin_h`); all 15 block checks pass. The OOF table contains 86,256 prediction points, 24 horizons and all nine frozen future-condition dimensions with matching truth labels.

Evidence: `outputs/train_oof/p2_train_oof_predictions.csv`, `p2_train_oof_audit.csv`, and `p2_train_oof_reproduction.json`.

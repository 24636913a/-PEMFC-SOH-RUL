"""Development-only protocol identity tests, run only after Stage 0 passes."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PI = Path(__file__).resolve().parents[1]
if str(PI) not in sys.path:
    sys.path.insert(0, str(PI))

from scripts.protocol import frozen_timesnet_summary, recover_windows_and_timesnet


class ProtocolIdentityTests(unittest.TestCase):
    def test_window_identity_and_registered_prefix_counts(self) -> None:
        _cfg, _helpers, _frame, _folds, records, audit = recover_windows_and_timesnet()
        self.assertEqual(audit["raw_prefix_rows_read"], 3007)
        self.assertEqual(audit["development_rows_after_filter"], 3005)
        self.assertEqual(audit["boundary_rows_filtered"], 2)
        self.assertEqual(audit["post_boundary_rows_entered_pipeline"], 0)
        self.assertTrue(records[["origin_row_match", "target_row_match", "truth_match", "anchor_match", "target_end_match", "horizon_match"]].all().all())

    def test_scaler_population_and_metric_recomputation(self) -> None:
        cfg, helpers, _frame, folds, _records, _audit = recover_windows_and_timesnet()
        fold, train, valid = folds[0]
        fit, _stopper = helpers.split_inner_fit_stopper(train, .15)
        scaler = helpers.FoldPreprocessor().fit(fit.x)
        flat = np.asarray(fit.x, dtype=np.float64).reshape(-1, 14)
        expected_mean = np.where(np.isfinite(flat), flat, np.nanmedian(flat, axis=0)).mean(axis=0)
        self.assertTrue(np.allclose(scaler.mean, expected_mean))
        cache = PI.parent / "Phase_BS_Final_Backbone_Screening" / "outputs" / "predictions" / f"bs_timesnet_{fold}_seed42.npz"
        with np.load(cache, allow_pickle=False) as z:
            metrics = helpers.delta_metrics(z["y_true_delta_V"], z["y_pred_delta_V"], z["target_row"])
        self.assertGreater(metrics["rmse_mV"], 0)
        self.assertEqual(metrics["query_count"], len(valid.origin_row) * 24)
        self.assertEqual(set(frozen_timesnet_summary().seed), {42, 2024, 3407})


if __name__ == "__main__":
    unittest.main()

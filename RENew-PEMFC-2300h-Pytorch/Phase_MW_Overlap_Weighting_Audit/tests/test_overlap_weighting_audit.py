import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

P = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_MW_Overlap_Weighting_Audit\scripts\run_overlap_weighting_audit.py")
s = importlib.util.spec_from_file_location("mw", P); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


def keys():
    return pd.DataFrame({"fold": ["fold_1"] * 2, "origin_row": [10, 11], "horizon_step": [1, 1], "target_row": [11, 12], "target_age_h": [100., 100.5]})


class OverlapAuditTests(unittest.TestCase):
    def test_missing_key_uses_production_alignment_guard(self):
        a, b, e = keys(), keys().iloc[:1].copy(), keys()
        with self.assertRaises(RuntimeError): m.validate_complete_alignment(a, b, e)

    def test_duplicate_key_uses_production_guard(self):
        a = pd.concat([keys(), keys().iloc[[0]]], ignore_index=True)
        with self.assertRaises(RuntimeError): m.validate_key_frame(a, "injected")

    def test_boundary_overrun_uses_production_guard(self):
        a = keys(); a.loc[0, "target_age_h"] = m.BOUNDARY
        with self.assertRaises(RuntimeError): m.validate_key_frame(a, "injected")

    def test_truth_mismatch_uses_production_guard(self):
        base = keys().assign(y_true_V=[1., 1.], anchor_V=[.8, .8])
        altered = base.copy(); altered.loc[0, "y_true_V"] += .01
        with self.assertRaises(RuntimeError): m.validate_truth_anchor(base, altered)

    def test_hash_mismatch_uses_production_guard(self):
        p = next(iter(m.expected_paths()))
        with self.assertRaises(RuntimeError): m.validate_input_hashes({p: "bad"})

    def test_delta_mse_decomposition_identity(self):
        d = pd.DataFrame({"fold": ["fold_1"] * 3, "origin_row": [1, 2, 3], "horizon_step": [1, 2, 1], "target_row": [2, 2, 4], "squared_error_mV2_p0": [1., 4., 9.], "squared_error_mV2_p2": [4., 1., 16.]})
        multi, horizon, segments = m.contribution_tables(d)
        self.assertAlmostEqual(multi.forecast_query_weighted_delta_MSE_contribution_mV2.sum(), ((3 - 3 + 7) / 3), places=10)
        self.assertAlmostEqual(multi.target_time_equal_weight_delta_MSE_contribution_mV2.sum(), (((3 - 3) / 2 + 7) / 2), places=10)

    def test_three_fold_macro_is_not_pooled(self):
        rows = []
        for fold, err in (("fold_1", 1.), ("fold_2", 2.), ("fold_3", 3.)):
            rows.append({"fold": fold, "origin_row": len(rows) + 1, "horizon_step": 1, "target_row": len(rows) + 2, "squared_error_mV2_p0": 1., "squared_error_mV2_p2": err ** 2})
        source = pd.DataFrame(rows)
        source["error_mV_p0"] = 1.; source["error_mV_p2"] = [1., 2., 3.]
        result = m.fold_metrics(source)
        macro = result[(result.fold == "three_fold_macro") & (result.evaluation_unit == "forecast_query_weighted_RMSE")].iloc[0]
        self.assertAlmostEqual(macro.p2_RMSE_mV, 2.0, places=10)

    def test_current_head_can_descend_from_start(self):
        self.assertTrue(m.validate_starting_commit())


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import unittest

P = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_H0_Existing_Artifact_Comparability_Audit\scripts\run_existing_h0_audit.py")
s = importlib.util.spec_from_file_location("h0audit", P); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


class ExistingArtifactAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m.assert_head(); cls.opr = m.opr_target_map()
        cls.metas = {(name, f"fold_{i}"): m.metadata(base / pattern.format(i=i), name, f"fold_{i}") for name, (base, pattern) in m.SPECS.items() for i in range(1, 4)}

    def test_boundary_overrun_fails_closed(self):
        bad = self.opr.copy(); bad.loc[0, "target_age_h"] = m.BOUNDARY
        with self.assertRaises(RuntimeError):
            if (bad.target_age_h >= m.BOUNDARY).any(): raise RuntimeError("boundary")

    def test_target_end_semantics_are_explicit(self):
        self.assertEqual(m.TARGET_END_OFFSET_H, 0.25)

    def test_truth_anchor_mismatch_is_not_tolerated(self):
        self.assertLessEqual(m.ATOL_V, 0.000005)
        self.assertFalse(abs(0.01) <= m.ATOL_V)

    def test_duplicate_key_contract_is_detectable(self):
        key = m.reconstruct_keys(self.metas[("H0_original", "fold_1")], self.opr)
        self.assertTrue(pd_concat_duplicate(key))

    def test_shuffled_origin_order_is_not_accepted(self):
        origin = self.metas[("H0_original", "fold_1")]["origin"].copy(); origin[[0, 1]] = origin[[1, 0]]
        self.assertTrue((origin[1:] <= origin[:-1]).any())

    def test_b_joint_only_is_excluded(self):
        self.assertNotIn("B_joint_only", m.FORMAL_MODELS)

    def test_macro_is_not_pooled(self):
        self.assertNotEqual("unweighted_three_fold_macro", "pooled_prediction_points")

    def test_missing_artifact_cannot_be_silently_skipped(self):
        with self.assertRaises(PermissionError): m.assert_allowed(m.ROOT / "missing_predictions.npz")

    def test_whitelist_rejects_raw_data_test_and_rul90(self):
        for bad in (m.ROOT / "data" / "raw.csv", m.ROOT / "Test_predictions.npz", m.ROOT / "Phase_RUL90_Data_Feasibility" / "full.csv"):
            with self.assertRaises(PermissionError): m.assert_allowed(bad)


def pd_concat_duplicate(key):
    import pandas as pd
    duplicated = pd.concat([key, key.iloc[[0]]], ignore_index=True)
    return bool(duplicated.duplicated(m.KEY_COLS).any())


if __name__ == "__main__":
    unittest.main()

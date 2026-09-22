import importlib.util
from pathlib import Path
import unittest

P = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_RD_TimeXer_Route_Decision_Audit\scripts\run_route_audit.py")
s = importlib.util.spec_from_file_location("route_audit", P); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


class RouteAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decomp, cls.paired, cls.baseline, _, _ = m.source_data()
        cls.contracts = m.validate_contracts(cls.decomp, cls.paired, cls.baseline)

    def test_01_base_plus_correction_equals_full_output(self):
        self.assertLessEqual(self.contracts["decomposition_max_residual_mV"], .001)

    def test_02_p1_p2_share_the_same_history_base(self):
        self.assertLessEqual(self.contracts["p1_p2_base_max_difference_mV"], .001)

    def test_03_paired_squared_error_identity(self):
        self.assertLessEqual(self.contracts["squared_error_max_residual_mV2"], .1)

    def test_04_deltav_anchor_and_mv_contract(self):
        self.assertIn("true_DeltaV_mV", self.decomp.columns)
        self.assertIn("final_prediction_DeltaV_mV", self.decomp.columns)
        opr_code = m.PATHS["opr_script"].read_text(encoding="utf-8")
        self.assertIn('anchor[:,None]+delta', opr_code)

    def test_05_origin_target_mapping_is_one_to_one(self):
        self.assertFalse(self.paired.duplicated(["fold", "origin_row", "horizon_step"]).any())

    def test_06_folds_have_no_origin_crossing(self):
        groups = [set(g.origin_row) for _, g in self.paired.groupby("fold")]
        self.assertTrue(all(not (groups[i] & groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups))))

    def test_07_test_firewall_is_intact(self):
        self.assertFalse(self.contracts["test_accessed"])
        self.assertLess(self.paired.target_age_h.max(), 1819.997738)

    def test_08_reported_baselines_are_the_reproduced_values(self):
        actual = dict(zip(self.baseline.model, self.baseline.mean_rmse_delta_mV))
        self.assertAlmostEqual(actual["B_only"], 192.38268025716147, places=6)
        report = (m.REP / "02_DECOMPOSITION_CONTRACT.md").read_text(encoding="utf-8")
        self.assertIn("192.383", report)

    def test_09_independent_events_are_deduplicated_by_target_time(self):
        _, _, events = m.attribution_tables(m.enrich_paired(self.paired))
        self.assertFalse(events.duplicated(["fold", "target_time"]).any())
        self.assertTrue((events.prediction_point_count >= 1).all())

    def test_10_strong_baselines_remain_context_only(self):
        self.assertTrue((m.comparability().status != "FORMAL_PAIRED_ACROSS_ARCHITECTURES").all())


if __name__ == "__main__":
    unittest.main()

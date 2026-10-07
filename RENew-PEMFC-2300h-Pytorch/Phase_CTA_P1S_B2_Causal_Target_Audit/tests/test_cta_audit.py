import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cta_audit.py"
SPEC = importlib.util.spec_from_file_location("cta_runner", SCRIPT)
cta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cta)


class CTAContracts(unittest.TestCase):
    def test_phases_have_expected_p1s_event_coverage(self):
        data = cta.load_trajectory()
        self.assertEqual(len(data), 141)
        self.assertEqual(int((data.condition_support_status == "IN_SUPPORT").sum()), 82)
        self.assertEqual(int((data.condition_support_status == "OUT_OF_SUPPORT").sum()), 59)

    def test_frozen_fold_calibrations_exclude_future_anchors(self):
        params = cta.fold_parameters()
        self.assertAlmostEqual(float(params.loc["fold_1", "a_fold"]), 1.246206115099345, places=15)
        self.assertTrue((params.future_anchor_used.astype(str).str.lower() == "false").all())

    def test_no_raw_late_life_target_access_code(self):
        cta.assert_no_target_reopening()


if __name__ == "__main__":
    unittest.main()

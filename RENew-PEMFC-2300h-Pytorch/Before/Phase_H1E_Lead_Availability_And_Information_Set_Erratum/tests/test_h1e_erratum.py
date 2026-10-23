import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h1e_test", ROOT / "scripts" / "run_h1e_erratum.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class H1EErratumTests(unittest.TestCase):
    def test_lead_statistics_preserve_all_requested_quantities(self):
        result = MODULE.statistics(np.array([9.52, 9.53, 10.02]))
        self.assertEqual(result["N"], 3)
        self.assertEqual(result["min_h"], 9.52)
        self.assertEqual(result["max_h"], 10.02)
        self.assertIn("P05_h", result)
        self.assertIn("IQR_h", result)

    def test_method_audit_marks_only_actual_lead_methods_non_strict(self):
        audit = MODULE.build_method_information_audit().set_index("method")
        self.assertTrue(bool(audit.loc["PERSISTENCE", "strictly_available_at_prediction_origin"]))
        self.assertFalse(bool(audit.loc["RIDGE_CAUSAL_STATE", "strictly_available_at_prediction_origin"]))
        self.assertFalse(bool(audit.loc["TIMESNET_RAW_F2", "uses_future_target_measurement"]))


if __name__ == "__main__":
    unittest.main()

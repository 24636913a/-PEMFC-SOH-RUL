import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_phi_consolidation.py"
SPEC = importlib.util.spec_from_file_location("phi_runner", SCRIPT)
phi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phi)


class PHIContracts(unittest.TestCase):
    def test_frozen_b2_matches_authority(self):
        a, b = phi.parse_frozen_models()
        self.assertAlmostEqual(a, 1.1988114938324186, places=15)
        self.assertAlmostEqual(b, -0.20903624631182127, places=15)

    def test_raw_target_files_not_configured(self):
        phi.assert_no_raw_target_access()

    def test_complete_cycle_source_uses_frozen_middle_window(self):
        data = phi.load_cycle_trajectory(*phi.parse_frozen_models())
        self.assertEqual(len(data), 141)
        self.assertTrue((data["ophrs_h"] < phi.DEVELOPMENT_BOUNDARY).all())
        self.assertTrue(data["cycle_interval_id"].is_unique)


if __name__ == "__main__":
    unittest.main()

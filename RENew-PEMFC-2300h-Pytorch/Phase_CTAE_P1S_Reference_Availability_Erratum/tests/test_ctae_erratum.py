import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ctae_erratum.py"
SPEC = importlib.util.spec_from_file_location("ctae_runner", SCRIPT)
ctae = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ctae)


class CTAEErratumContracts(unittest.TestCase):
    def test_reference_voltage_and_cycle_identity(self):
        ref = ctae.verify_reference()
        self.assertEqual(len(ref), 10)
        self.assertEqual(ref.iloc[-1].cycle_interval_id, "COMPLETE_010")

    def test_warmup_and_causal_target_separation(self):
        data = ctae.load_availability()
        self.assertEqual(int(data.reference_warmup_record.sum()), 10)
        self.assertEqual(int(data.causal_target_eligible.sum()), 131)
        self.assertEqual(data.loc[data.causal_target_eligible].iloc[0].cycle_interval_id, "COMPLETE_011")
        self.assertTrue(data.loc[data.reference_warmup_record, "R_P1S_causal"].isna().all())

    def test_post_reference_slope_replay_is_prefix_invariant(self):
        data = ctae.load_availability()
        _, _, verdict, _ = ctae.make_slopes(data)
        self.assertTrue(all(verdict.values()))


if __name__ == "__main__":
    unittest.main()

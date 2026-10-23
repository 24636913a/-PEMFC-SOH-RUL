from __future__ import annotations

import sys
import unittest
from pathlib import Path


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from run_final_preopen_verification import formal_stage_p_development_runtime_replay  # noqa: E402


class FormalStagePDevelopmentEquivalenceTests(unittest.TestCase):
    """Exercise the actual future Stage-P runtime, never a local shortcut."""

    def test_six_development_prefixes_match_frozen_paa_at_tight_tolerance(self) -> None:
        result = formal_stage_p_development_runtime_replay()
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["pass_count"], 6)
        rows = result["rows"]
        self.assertEqual(len(rows), 18)
        self.assertTrue(rows["formal_runtime_path"].eq("ahc_tv_stage_p authoritative runtime core").all())
        self.assertTrue((rows["synthetic_test_mode"] == False).all())  # noqa: E712
        for column in (
            "interval_id_match", "datetime_match", "R_match", "voltage_match", "ophrs2_match",
            "conditions_match", "support_match", "eligible_dwell_count_match", "formal_anchor_median_match",
        ):
            with self.subTest(column=column):
                self.assertTrue(bool(rows[column].all()))


if __name__ == "__main__":
    unittest.main()

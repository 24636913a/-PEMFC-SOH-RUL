from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from mq_access import DataAccessGuard
from run_attempt_2_local import calibrate, effective_protocol


class TestAttempt2ProductionDefences(unittest.TestCase):
    def test_effective_protocol_hashes_and_gap_after_override(self):
        _, _, columns = effective_protocol()
        self.assertNotIn("gap_after_flag", columns)
        self.assertIn("cell_voltage_mean", columns)

    def test_guard_rejects_wrong_n1_scope_and_forbidden_polarization(self):
        _, _, columns = effective_protocol()
        guard = DataAccessGuard(columns)
        with self.assertRaises(RuntimeError):
            guard._check(guard.n1, nrows=3006, usecols=columns)
        with self.assertRaises(RuntimeError):
            guard._check(Path("https://example.invalid/data.csv"), nrows=None, usecols=None)
        with self.assertRaises(RuntimeError):
            guard._check(guard.n1.parent / "polarization_checkpoint_jstar_measurements.csv", nrows=None, usecols=None)

    def test_calibration_fails_closed_without_within_segment_lag_pairs(self):
        rows=[]
        for i in range(20):
            rows.append({"bin_start_ophrs2_h":72.8+i*.5,"temporal_segment_id":i,"split":"train","age_stratum":"EARLY","eligible":True,"token":"OP1|1.000"})
        period, template, scores, prefix = calibrate(pd.DataFrame(rows))
        self.assertIsNone(period)
        self.assertEqual(template, [])
        self.assertTrue(all(r["lag_pairs"] == 0 for r in scores))
        self.assertEqual(len(prefix), 20)


if __name__ == "__main__":
    unittest.main()

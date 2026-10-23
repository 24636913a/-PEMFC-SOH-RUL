from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from ahc_tv_alignment import TARGET_COLUMNS  # noqa: E402
from ahc_tv_stage_t import extract_jstar_voltage_from_table  # noqa: E402
from run_final_preopen_verification import stage_t_development_target_replay  # noqa: E402


class StageTDevelopmentTargetEquivalenceTests(unittest.TestCase):
    def test_allowlisted_bol_and_six_development_targets_replay(self) -> None:
        result = stage_t_development_target_replay()
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["pass_count"], 7)
        self.assertTrue(bool(result["rows"]["pass"].all()))

    def test_active_cell_nanmean_keeps_partially_observed_rows_and_drops_all_nan_rows(self) -> None:
        values = {column: [0.5, np.nan] for column in TARGET_COLUMNS[1:]}
        values[TARGET_COLUMNS[1]][0] = np.nan
        values[TARGET_COLUMNS[0]] = [1.0, 1.0]
        voltage, count = extract_jstar_voltage_from_table(pd.DataFrame(values).loc[:, list(TARGET_COLUMNS)])
        self.assertEqual(count, 1)
        self.assertAlmostEqual(voltage, 0.5)


if __name__ == "__main__":
    unittest.main()

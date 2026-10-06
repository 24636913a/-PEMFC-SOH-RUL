from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from run_final_preopen_verification import stage_p_development_raw_replay  # noqa: E402
from ahc_tv_stage_p import ACTIVE_CELLS, _nanmean_active_voltage  # noqa: E402


class StagePDevelopmentRawEquivalenceTests(unittest.TestCase):
    def test_active_cell_nanmean_matches_frozen_partial_and_all_missing_semantics(self) -> None:
        data = pd.DataFrame({column: [0.6, np.nan] for column in ACTIVE_CELLS})
        data.loc[0, ACTIVE_CELLS[0]] = np.nan
        values = _nanmean_active_voltage(data)
        self.assertAlmostEqual(float(values.iloc[0]), 0.6)
        self.assertTrue(np.isnan(values.iloc[1]))

    def test_six_frozen_cutoffs_reconstruct_all_eighteen_selected_intervals(self) -> None:
        result = stage_p_development_raw_replay()
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["pass_count"], 6)
        rows = result["rows"]
        self.assertEqual(len(rows), 18)
        for column in (
            "interval_id_match", "datetime_match", "R_match", "ophrs2_match",
            "conditions_match", "support_match", "eligible_dwell_count_match",
        ):
            with self.subTest(column=column):
                self.assertTrue(bool(rows[column].all()))


if __name__ == "__main__":
    unittest.main()

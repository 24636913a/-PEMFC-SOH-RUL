"""Synthetic contract tests for the production RCR audit functions."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "run_raw_cycle_reconstruction_audit.py"
SPEC = importlib.util.spec_from_file_location("rcr", MODULE)
assert SPEC and SPEC.loader
rcr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rcr
SPEC.loader.exec_module(rcr)


def frame(rows: list[tuple[float, str, str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=rcr.REQUIRED_COLUMNS)


class RawCycleReconstructionTests(unittest.TestCase):
    def test_voltage_column_is_rejected_by_production_contract(self) -> None:
        with self.assertRaises(rcr.AuditFailure):
            rcr.assert_operating_only_columns((*rcr.REQUIRED_COLUMNS, "U1 [V]"))

    def test_boundary_rows_are_dropped_before_compression(self) -> None:
        raw = frame([
            (1819.9, "2020-01-01 00:00:00", "A", 1.0, 1.0, 0.1),
            (1819.998, "2020-01-01 00:00:01", "A", 1.0, 1.0, 0.1),
        ])
        development, discarded = rcr.normalise_chunk(raw)
        self.assertEqual(discarded, 1)
        self.assertEqual(len(development), 1)
        self.assertTrue((development["OpHrs2 [h]"] < rcr.BOUNDARY_H).all())

    def test_thousands_formatted_operating_hours_remain_development_rows(self) -> None:
        raw = frame([
            ("1,225.01", "2020-01-01 00:00:00", "A", 1.0, 1.0, 0.1),
        ])
        development, discarded = rcr.normalise_chunk(raw)
        self.assertEqual(discarded, 0)
        self.assertEqual(len(development), 1)
        self.assertAlmostEqual(float(development.iloc[0]["OpHrs2 [h]"]), 1225.01)

    def test_zero_stop_state_is_preserved(self) -> None:
        raw = frame([
            (1.0, "2020-01-01 00:00:00", "STOP", 0.0, 0.0, 0.0),
            (1.0 + 1 / 3600, "2020-01-01 00:00:01", "STOP", 0.0, 0.0, 0.0),
            (1.0 + 2 / 3600, "2020-01-01 00:00:02", "LOAD", 2.0, 2.0, 0.2),
        ])
        runs = rcr.compress_rows(raw)
        self.assertEqual(len(runs), 2)
        self.assertEqual(int(runs.iloc[0]["stop_or_zero_current_flag"]), 1)
        self.assertEqual(runs.iloc[0]["operation_point"], "STOP")

    def test_fixed_setpoint_rounding_merges_float_jitter(self) -> None:
        raw = frame([
            (1.0, "2020-01-01 00:00:00", "A", 1.0001, 1.0, 0.1),
            (1.0 + 1 / 3600, "2020-01-01 00:00:01", "A", 1.0004, 1.0, 0.1),
        ])
        runs = rcr.compress_rows(raw)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs.iloc[0]["state_token"], "A||1.000")

    def test_order_lcs_does_not_elastically_align(self) -> None:
        self.assertEqual(rcr.lcs_length(["A", "B", "C"], ["A", "X", "B", "C"]), 3)
        self.assertEqual(rcr.lcs_length(["A", "B", "C"], ["C", "B", "A"]), 1)

    def test_cross_stratum_cycle_is_not_complete(self) -> None:
        self.assertIsNone(rcr.age_stratum(650.0, 660.0))


if __name__ == "__main__":
    unittest.main()

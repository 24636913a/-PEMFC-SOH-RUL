from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from ahc_tv_alignment import CONDITIONS, align_latest_three_intervals  # noqa: E402


def bounds():
    return {name: (0.0, 10.0) for name in CONDITIONS}


def rows():
    records = []
    for index, (identifier, time, age) in enumerate((
        ("I4", "2030-01-01 12:04:00", 147.9),
        ("I3", "2030-01-01 12:03:00", 147.0),
        ("I2", "2030-01-01 12:02:00", 146.0),
        ("I1", "2030-01-01 12:01:00", 100.0),
    )):
        record = {
            "cycle_interval_id": identifier,
            "interval_kind": "COMPLETE_CYCLE_INTERVAL",
            "measurement_end_datetime": time,
            "measurement_end_ophrs2_h": age,
            "P1S_operating_voltage_retention": 0.99 - index / 1000,
            "has_complete_middle50_dwell": True,
            "cycle_start_anchor_present": True,
        }
        record.update({name: 1.0 for name in CONDITIONS})
        records.append(record)
    return pd.DataFrame(records)


class TestAlignmentSemantics(unittest.TestCase):
    def align(self, table, cutoff=148.0):
        return align_latest_three_intervals(table, pd.Timestamp("2030-01-01 13:00:00"), cutoff, bounds())

    def test_legal_selection_order_and_voltage_invariance(self) -> None:
        table = rows()
        first = self.align(table)
        table.loc[table.cycle_interval_id.eq("I1"), "P1S_operating_voltage_retention"] = 99.0
        second = self.align(table)
        self.assertEqual(first.selected.cycle_interval_id.tolist(), ["I4", "I3", "I2"])
        self.assertEqual(second.selected.cycle_interval_id.tolist(), ["I4", "I3", "I2"])

    def test_datetime_then_age_then_interval_id_tie_break(self) -> None:
        table = rows().iloc[:3].copy()
        table["measurement_end_datetime"] = "2030-01-01 12:03:00"
        table["measurement_end_ophrs2_h"] = 147.0
        table.loc[:, "cycle_interval_id"] = ["C", "A", "B"]
        result = self.align(table)
        self.assertEqual(result.selected.cycle_interval_id.tolist(), ["A", "B", "C"])

    def test_support_cannot_replace_third_most_recent_interval(self) -> None:
        table = rows()
        table.loc[table.cycle_interval_id.eq("I2"), "anode_inlet_RH_pct"] = 99.0
        table.loc[table.cycle_interval_id.eq("I1"), "anode_inlet_RH_pct"] = 1.0
        result = self.align(table)
        self.assertEqual(result.selected.cycle_interval_id.tolist(), ["I4", "I3", "I2"])
        self.assertEqual(result.support_count, 2)
        self.assertEqual(result.status, "ALIGNED")

    def test_operating_hour_and_strict_time_boundaries(self) -> None:
        table = rows().iloc[:3].copy()
        table.loc[0, "measurement_end_ophrs2_h"] = 100.0  # exactly 48 h: included
        result = self.align(table)
        self.assertEqual(len(result.selected), 3)
        table.loc[0, "measurement_end_ophrs2_h"] = 99.999
        self.assertEqual(self.align(table).status, "TEST_ALIGNMENT_INCONCLUSIVE")
        table = rows().iloc[:3].copy()
        table.loc[0, "measurement_end_datetime"] = "2030-01-01 13:00:00"
        self.assertEqual(self.align(table).status, "TEST_ALIGNMENT_INCONCLUSIVE")

    def test_partial_requires_anchor_and_less_than_three_is_inconclusive(self) -> None:
        table = rows().iloc[:3].copy()
        table.loc[0, "interval_kind"] = "PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL"
        table.loc[0, "cycle_start_anchor_present"] = False
        self.assertEqual(self.align(table).status, "TEST_ALIGNMENT_INCONCLUSIVE")
        self.assertEqual(len(self.align(rows().iloc[:2].copy()).selected), 2)

    def test_one_of_three_support_is_inconclusive_and_two_of_three_is_legal(self) -> None:
        table = rows().iloc[:3].copy()
        table.loc[[0, 1], "current_density"] = 99.0
        self.assertEqual(self.align(table).status, "TEST_ALIGNMENT_INCONCLUSIVE")
        table.loc[1, "current_density"] = 1.0
        result = self.align(table)
        self.assertEqual(result.status, "ALIGNED")
        self.assertEqual(result.support_count, 2)

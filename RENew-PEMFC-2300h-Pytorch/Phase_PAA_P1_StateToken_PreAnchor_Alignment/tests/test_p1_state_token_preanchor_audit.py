"""Synthetic tests for PAA production functions; no raw or polarization reads."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_p1_state_token_preanchor_audit.py"
SPEC = importlib.util.spec_from_file_location("paa", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PAA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PAA
SPEC.loader.exec_module(PAA)

CONDITIONS = list(PAA.COND_RAW)


def token_states() -> pd.DataFrame:
    tokens = ["221||221.200"] * 93 + ["221||156.800"] * 48 + ["221||313.600"] * 25
    return pd.DataFrame({"run_id": np.arange(len(tokens)), "operation_point": ["221"] * len(tokens), "stop_or_zero_current_flag": [0] * len(tokens), "median_actual_current_A": [1.0] * len(tokens), "state_token": tokens})


def minimal_config(anchor_count: int = 1) -> dict:
    anchors = []
    for checkpoint in range(anchor_count):
        anchors.append({"checkpoint_h": 300 + checkpoint, "incomplete_raw_cycle_id": 25 + checkpoint, "cycle_start_run_id": 1, "pre_characterization_last_run_id": 3, "pre_characterization_cutoff_datetime": "2025-01-01T00:10:00", "pre_characterization_cutoff_ophrs2_h": 100.0})
    return {"raw_firewall": {"maximum_required_precharacterization_cutoff_h": 200.0}, "partial_interval_characterization_metadata": {"anchors": anchors}, "conditions": CONDITIONS, "support_and_drift": {"complete_cycle_measurements_min_per_stratum": 20, "seven_condition_complete_rate_min": .90, "joint_common_support_min_per_stratum": 15}}


class PAAProductionTests(unittest.TestCase):
    def test_primary_token_comes_only_from_template_frequency(self) -> None:
        states = token_states()
        template = {"template_start_run_id": 0, "template_end_run_id_exclusive": len(states)}
        baseline = PAA.select_p1s_token(states, template)
        states["invented_voltage"] = np.linspace(.1, 1.1, len(states))
        states["invented_polarization"] = np.linspace(1.1, .1, len(states))
        changed = PAA.select_p1s_token(states, template)
        self.assertEqual(baseline["selected"], "221||221.200")
        self.assertEqual(baseline, changed)
        self.assertEqual(baseline["state_token_frequencies"]["221||221.200"], 93)

    def test_other_p1_tokens_never_replace_primary(self) -> None:
        selected = PAA.select_p1s_token(token_states(), {"template_start_run_id": 0, "template_end_run_id_exclusive": 166})
        self.assertEqual(selected["selected"], PAA.P1_TOKEN)
        self.assertNotEqual(selected["selected"], "221||156.800")
        self.assertNotEqual(selected["selected"], "221||313.600")

    def test_one_interval_with_multiple_dwells_has_one_median_measurement(self) -> None:
        rows = []
        for voltage in [.70, .72, .74]:
            rows.append({"stable_window": "middle_50", "cycle_interval_id": "COMPLETE_001", "interval_kind": "COMPLETE_CYCLE_INTERVAL", "raw_cycle_id": 1, "age_stratum": "EARLY", "nominal_checkpoint_h": np.nan, "end_ophrs2_h": 100.0, "end_datetime": "2025-01-01T00:00:00", "P1S_voltage_V": voltage, **{name: 1.0 for name in CONDITIONS}})
        result = PAA.interval_measurements(pd.DataFrame(rows))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result.iloc[0].P1S_voltage_V), .72)
        self.assertEqual(int(result.iloc[0].eligible_dwell_count), 3)

    def test_partial_interval_requires_anchor_and_excludes_post_cutoff_dwell(self) -> None:
        states = pd.DataFrame({"run_id": [1, 2, 3, 4], "state_token": ["15||14.000", "15||0.000", PAA.P1_TOKEN, PAA.P1_TOKEN], "duration_wall_s": [1.0, 1.0, 180.0, 180.0], "stop_or_zero_current_flag": [0, 0, 0, 0], "end_datetime": ["2025-01-01T00:00:01", "2025-01-01T00:00:02", "2025-01-01T00:05:00", "2025-01-01T00:15:00"], "start_datetime": ["2025-01-01T00:00:00"] * 4, "operation_point": ["15", "15", "221", "221"], "median_set_current_A": [14.0, 0.0, 221.2, 221.2], "end_ophrs2_h": [1., 1., 2., 3.]})
        cycles = pd.DataFrame({"raw_cycle_id": [25], "start_run_id": [1]})
        complete = pd.DataFrame(columns=["end_ophrs2_h", "start_run_id", "end_run_id_exclusive"])
        config = minimal_config(); config["partial_interval_characterization_metadata"]["anchors"][0]["pre_characterization_cutoff_datetime"] = "2025-01-01T00:10:00"
        result = PAA.target_intervals(complete, cycles, states, config)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].cycle_interval_id, "PARTIAL_PRE_300")
        bad = states.copy(); bad.loc[0, "state_token"] = "x"
        with self.assertRaises(PAA.AuditFailure):
            PAA.target_intervals(complete, cycles, bad, config)

    def test_strict_48h_and_three_distinct_intervals_define_alignment(self) -> None:
        config = minimal_config(); config["partial_interval_characterization_metadata"]["anchors"] = [{"checkpoint_h": 300, "pre_characterization_cutoff_datetime": "2025-01-01T00:10:00", "pre_characterization_cutoff_ophrs2_h": 100.0}]
        rows = []
        for index, age in enumerate([52.0, 60.0, 99.0]):
            rows.append({"stable_window": "middle_50", "cycle_interval_id": f"COMPLETE_{index}", "interval_kind": "COMPLETE_CYCLE_INTERVAL", "measurement_end_ophrs2_h": age, "measurement_end_datetime": "2025-01-01T00:00:00", "P1S_operating_voltage_retention": .99, **{name: 1.0 for name in CONDITIONS}})
        alignment, _ = PAA.anchor_alignment(pd.DataFrame(rows), {name: (0.0, 2.0) for name in CONDITIONS}, config)
        summary = alignment.loc[alignment.record_type == "anchor_summary"].iloc[0]
        self.assertEqual(summary.status, "ALIGNED")
        self.assertEqual(int(summary.eligible_interval_count_within_48h), 3)
        rows.pop()
        alignment, _ = PAA.anchor_alignment(pd.DataFrame(rows), {name: (0.0, 2.0) for name in CONDITIONS}, config)
        self.assertEqual(alignment.loc[alignment.record_type == "anchor_summary"].iloc[0].status, "UNRESOLVED")

    def test_test_raw_and_test_polarization_are_rejected(self) -> None:
        with self.assertRaises(PAA.AuditFailure):
            PAA.enforce_raw_firewall(pd.Series([1819.99, 1820.0]), 1719.31)
        with self.assertRaises(PAA.AuditFailure):
            PAA.validate_polarization_path(Path("Polcurve_Nominal_RHFC_2000h.csv"))
        with self.assertRaises(PAA.AuditFailure):
            PAA.validate_polarization_path(Path("Polcurve_Nominal_RHFC_2300h_afterAirAirStart.csv"))


if __name__ == "__main__":
    unittest.main()

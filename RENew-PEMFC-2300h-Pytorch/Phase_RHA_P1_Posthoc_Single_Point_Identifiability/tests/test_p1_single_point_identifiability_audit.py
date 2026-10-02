"""Synthetic tests for Phase RHA-P1 production helpers; no raw data access."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_p1_single_point_identifiability_audit.py"
SPEC = importlib.util.spec_from_file_location("p1_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
P1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P1
SPEC.loader.exec_module(P1)


class P1AuditProductionHelperTests(unittest.TestCase):
    def test_contiguous_segments_split_at_cycle_interruption_and_stratum(self) -> None:
        data = pd.DataFrame(
            {
                "age_stratum": ["EARLY", "EARLY", "EARLY", "LATE", "LATE"],
                "raw_cycle_id": [1, 2, 4, 8, 9],
                "P1_cycle_operating_voltage_retention": [1.0, 0.99, 0.98, 0.94, 0.93],
            }
        )
        segments = P1.contiguous_segments(data)
        self.assertEqual([segment.tolist() for segment in segments], [[1.0, 0.99], [0.98], [0.94, 0.93]])

    def test_moving_block_sampler_uses_contiguous_candidate_blocks(self) -> None:
        segments = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 11.0, 12.0, 13.0])]
        output = P1.moving_block_sample(segments, n=12, block=2, rng=np.random.default_rng(2024))
        self.assertEqual(len(output), 12)
        valid_pairs = {(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (10.0, 11.0), (11.0, 12.0), (12.0, 13.0)}
        self.assertTrue(all(tuple(output[index : index + 2]) in valid_pairs for index in range(0, len(output), 2)))

    def test_deterministic_matching_uses_condition_variables_not_voltage(self) -> None:
        conditions = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]
        def row(cycle: int, stratum: str, value: float, h: float, ref: bool = False) -> dict[str, object]:
            return {"raw_cycle_id": cycle, "age_stratum": stratum, "is_reference_cycle": ref, "is_evaluation_cycle": not ref, "P1_cycle_operating_voltage_retention": h, **{name: value for name in conditions}}
        primary = pd.DataFrame([row(1, "EARLY", 1.0, 1.0, True), row(2, "EARLY", 2.0, 1.0), row(3, "EARLY", 3.0, 0.99), row(4, "LATE", 2.0, 0.96), row(5, "LATE", 3.0, 0.95)])
        joint = primary.loc[(primary["is_evaluation_cycle"])].copy()
        joint["in_joint_common_support"] = True
        matches, summary = P1.deterministic_match(primary, joint, conditions)
        self.assertEqual(summary["matched_pair_count"], 2)
        self.assertTrue((matches["late_minus_early_retention"] < 0).all())
        self.assertNotIn("P1_cycle_operating_voltage_retention", summary["matching_variables"])

    def test_final_status_priority_is_fail_closed(self) -> None:
        self.assertEqual(P1.final_status(False, True, True, True, "PASS", True), "P1_SUPPORT_REPRODUCTION_FAIL")
        self.assertEqual(P1.final_status(True, False, True, True, "PASS", True), "P1_VOLTAGE_DRIFT_NOT_IDENTIFIABLE")
        self.assertEqual(P1.final_status(True, True, False, True, "PASS", True), "P1_VOLTAGE_DRIFT_CONDITION_CONFOUNDED")
        self.assertEqual(P1.final_status(True, True, True, False, "PASS", True), "P1_STABLE_DWELL_SENSITIVITY_UNRESOLVED")
        self.assertEqual(P1.final_status(True, True, True, True, "P1_ANCHOR_ALIGNMENT_INCONCLUSIVE", False), "P1_ANCHOR_ALIGNMENT_INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()

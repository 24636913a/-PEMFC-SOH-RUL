from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chm_v2a_execution_semantics_audit.py"
SPEC = importlib.util.spec_from_file_location("chm_v2a", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CHMV2AExecutionSemanticsAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hou = pd.read_csv(MODULE.HOU)

    def test_01_production_audit_passes_without_training_or_test(self) -> None:
        result = MODULE.run_execution_semantics_audit()
        self.assertEqual(result["execution_semantics_audit"], "PASS")
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["Test_accessed"])
        self.assertEqual(json.loads(MODULE.AUDIT_OUTPUT.read_text(encoding="utf-8"))["remaining_execution_ambiguity"], "NONE")

    def test_02_final_profile_function_accepts_only_six_selected_profile_labels(self) -> None:
        self.assertEqual(MODULE.choose_final_profile(["A_BALANCED", "A_BALANCED", "B_ANCHOR_EMPHASIS", "C_OBSERVATION_EMPHASIS", "A_BALANCED", "B_ANCHOR_EMPHASIS"]), "A_BALANCED")
        with self.assertRaises(MODULE.ExecutionSemanticsFailure):
            MODULE.choose_final_profile(["A_BALANCED"] * 5)

    def test_03_majority_tie_breaks_are_frozen(self) -> None:
        self.assertEqual(MODULE.choose_final_profile(["A_BALANCED"] * 3 + ["B_ANCHOR_EMPHASIS"] * 3), "A_BALANCED")
        self.assertEqual(MODULE.choose_final_profile(["A_BALANCED"] * 2 + ["B_ANCHOR_EMPHASIS"] * 2 + ["C_OBSERVATION_EMPHASIS"] * 2), "A_BALANCED")
        self.assertEqual(MODULE.choose_final_profile(["B_ANCHOR_EMPHASIS"] * 3 + ["C_OBSERVATION_EMPHASIS"] * 3), "B_ANCHOR_EMPHASIS")

    def test_04_inner_score_is_mean_of_15_absolute_errors_not_prediction_median(self) -> None:
        residuals = np.array([-1.0, 1.0] + [0.5] * 13)
        self.assertAlmostEqual(MODULE.inner_profile_score(residuals), np.abs(residuals).mean())
        with self.assertRaises(MODULE.ExecutionSemanticsFailure):
            MODULE.inner_profile_score(residuals[:-1])

    def test_05_outer_prediction_is_exact_three_seed_median(self) -> None:
        self.assertEqual(MODULE.aggregate_outer_seed_predictions([0.1, 0.9, 0.2]), 0.2)

    def test_06_a_seed_failure_cannot_be_silently_removed_or_replaced(self) -> None:
        for invalid in ([0.1, 0.2], [0.1, None, 0.3], [0.1, np.nan, 0.3]):
            with self.assertRaises(MODULE.ExecutionSemanticsFailure):
                MODULE.aggregate_outer_seed_predictions(invalid)

    def test_07_bootstrap_seed_tuple_and_optimizer_seed_are_frozen(self) -> None:
        self.assertEqual(MODULE.seed_tuple(101, 2, 7), (90210, 101, 2, 7))
        self.assertEqual([MODULE.optimizer_seed_for_replicate(index) for index in range(6)], [42, 2024, 3407, 42, 2024, 3407])

    def test_08_same_seed_tuple_repeats_identical_weights_and_next_replicate_changes_stream(self) -> None:
        first = MODULE.build_bootstrap_observation_weights(self.hou, 4, 101, 0, 11)["bootstrap_observation_weight"].tolist()
        repeat = MODULE.build_bootstrap_observation_weights(self.hou, 4, 101, 0, 11)["bootstrap_observation_weight"].tolist()
        next_replicate = MODULE.build_bootstrap_observation_weights(self.hou, 4, 101, 0, 12)["bootstrap_observation_weight"].tolist()
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, next_replicate)

    def test_09_frozen_hou_segments_match_protocol(self) -> None:
        segments = MODULE.assert_expected_segments(self.hou)
        self.assertEqual(tuple((segment[0], segment[-1], len(segment)) for segment in segments), MODULE.EXPECTED_SEGMENTS)

    def test_10_effective_block_lengths_handle_short_segments(self) -> None:
        self.assertEqual(MODULE.effective_block_length(4, 3), 3)
        self.assertEqual(MODULE.effective_block_length(8, 5), 5)

    def test_11_each_segment_multiplicity_sums_to_original_length(self) -> None:
        for segment in MODULE.assert_expected_segments(self.hou):
            weights = MODULE.segment_multiplicity_weights(segment, 4, MODULE.bootstrap_generator(101, 0, len(segment)))
            self.assertEqual(sum(weights.values()), len(segment))

    def test_12_bootstrap_preserves_all_rows_original_order_and_partial_weight_one(self) -> None:
        weighted = MODULE.build_bootstrap_observation_weights(self.hou, 8, 301, 0, 3)
        self.assertEqual(weighted["hou_id"].tolist(), self.hou["hou_id"].tolist())
        partial = weighted[weighted["hou_kind"].eq("PREANCHOR_PARTIAL_HOU")]
        self.assertEqual(len(partial), 6)
        self.assertTrue(partial["bootstrap_observation_weight"].eq(1).all())

    def test_13_complete_weight_zero_does_not_remove_hou_or_anchor_contract(self) -> None:
        weighted = MODULE.build_bootstrap_observation_weights(self.hou, 4, 101, 0, 0)
        self.assertEqual(len(weighted), 152)
        self.assertGreater(int(weighted.loc[weighted["hou_kind"].eq("COMPLETE_CYCLE_HOU"), "bootstrap_observation_weight"].eq(0).sum()), 0)
        supplement = MODULE.load_yaml(MODULE.SUPPLEMENT)
        self.assertEqual(supplement["bootstrap_execution_semantics"]["anchor_loss"]["remove_anchor_when_corresponding_complete_weight_zero"], "prohibited")

    def test_14_drawn_blocks_are_not_a_synthetic_time_axis(self) -> None:
        weighted = MODULE.build_bootstrap_observation_weights(self.hou, 4, 201, 0, 5)
        self.assertEqual(weighted.index.tolist(), self.hou.index.tolist())
        self.assertEqual(weighted["measurement_end_datetime"].tolist(), self.hou["measurement_end_datetime"].tolist())

    def test_15_invalid_replicates_are_not_replaced(self) -> None:
        accounting = MODULE.replicate_accounting(200, [37])
        self.assertEqual(accounting, {"scheduled": 200, "invalid_indices": [37], "legal_count": 199, "replacement_performed": False})

    def test_16_bootstrap_cannot_select_profiles(self) -> None:
        supplement = MODULE.load_yaml(MODULE.SUPPLEMENT)
        self.assertEqual(supplement["bootstrap_execution_semantics"]["profile_selection_inside_bootstrap"], "prohibited")

    def test_17_deterministic_gate_predictions_cannot_be_bootstrap_p50(self) -> None:
        supplement = MODULE.load_yaml(MODULE.SUPPLEMENT)
        self.assertEqual(supplement["seeds_and_deterministic_prediction_semantics"]["bootstrap_P50_for_deterministic_point_prediction"], "prohibited")

    def test_18_S1_remains_nonqualification_diagnostic(self) -> None:
        supplement = MODULE.load_yaml(MODULE.SUPPLEMENT)
        self.assertFalse(supplement["S1_reporting_contract"]["primary_gate"])
        self.assertEqual(supplement["S1_reporting_contract"]["role"], "PRESPECIFIED_SECONDARY_ROBUSTNESS_DIAGNOSTIC")

    def test_19_production_validator_rejects_execution_semantic_drift(self) -> None:
        supplement = copy.deepcopy(MODULE.load_yaml(MODULE.SUPPLEMENT))
        supplement["bootstrap_execution_semantics"]["invalid_replicate_replacement"] = "allowed"
        with self.assertRaises(MODULE.ExecutionSemanticsFailure):
            MODULE.validate_supplement(supplement)

    def test_20_production_validator_rejects_hou_order_change(self) -> None:
        corrupt = self.hou.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(MODULE.ExecutionSemanticsFailure):
            MODULE.complete_cycle_segments(corrupt)


if __name__ == "__main__":
    unittest.main()

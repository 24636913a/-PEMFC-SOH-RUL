from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chm_candidate_protocol_static.py"
SPEC = importlib.util.spec_from_file_location("chm_static", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CHMCandidateStaticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MODULE.load_candidate()

    def test_01_complete_cycle_hou_count_is_frozen_at_146(self) -> None:
        self.assertEqual(len(MODULE.frozen_complete_cycles()), 146)

    def test_02_only_six_frozen_partial_hou_exist(self) -> None:
        partial = MODULE.frozen_partial_hou()
        self.assertEqual(set(partial["cycle_interval_id"]), set(MODULE.EXPECTED_PARTIAL_IDS))

    def test_03_known_hou_measurements_are_strictly_datetime_sorted(self) -> None:
        ordered = MODULE.known_measured_hou_time_order()
        MODULE.assert_strict_datetime_order(ordered)

    def test_04_600_anchor_contains_frozen_partial_pre_600(self) -> None:
        self.assertIn("PARTIAL_PRE_600", MODULE.selected_anchor_ids()[600])

    def test_05_anchor_prediction_is_median_of_three_states(self) -> None:
        self.assertEqual(MODULE.anchor_prediction_from_three_hou_states([0.99, 0.97, 0.98]), 0.98)

    def test_06_bol_is_not_first_hou(self) -> None:
        self.assertFalse(self.config["p1s_observation"]["BoL_is_HOU"])
        self.assertEqual(self.config["initial_state_and_transition"]["first_observable_state"], "H_1_equals_sigmoid_eta")

    def test_07_h1_is_not_fixed_one(self) -> None:
        initial = self.config["initial_state_and_transition"]
        self.assertEqual(initial["eta_initialization"], "logit_0_995")
        self.assertFalse(initial["eta_initialization_is_health_label"])

    def test_08_age_is_not_a_model_input(self) -> None:
        forbidden = set(self.config["inputs"]["forbidden"])
        self.assertTrue({"OpHrs2", "cycle_index", "DateTime", "nominal_checkpoint_age"}.issubset(forbidden))

    def test_09_partial_hou_never_changes_complete_cycle_count(self) -> None:
        self.assertEqual(self.config["HOU_definition"]["complete_cycle_HOU"]["count"], 146)
        self.assertEqual(self.config["HOU_definition"]["preanchor_partial_HOU"]["count"], 6)
        self.assertEqual(self.config["HOU_definition"]["primary_trajectory_report"], "complete_cycle_HOU_only_146")

    def test_10_bootstrap_attaches_weights_without_reordering_hou(self) -> None:
        ids = ["COMPLETE_001", "PARTIAL_PRE_300", "COMPLETE_024"]
        weighted = MODULE.attach_block_weights_without_reordering(ids, {"COMPLETE_001": 2, "COMPLETE_024": 1})
        self.assertEqual(weighted["hou_id"].tolist(), ids)
        self.assertEqual(weighted["bootstrap_weight"].tolist(), [2, 0, 1])

    def test_11_outer_holdout_is_excluded_from_outer_fit(self) -> None:
        training = MODULE.outer_training_anchor_ids(850)
        self.assertNotIn(850, training)
        self.assertEqual(len(training), 5)

    def test_12_per_anchor_spearman_is_prohibited(self) -> None:
        self.assertEqual(self.config["retrospective_LOAO"]["per_anchor_Spearman"], "prohibited")

    def test_13_aggregate_spearman_uses_six_heldout_predictions_only(self) -> None:
        self.assertEqual(self.config["retrospective_LOAO"]["aggregate_metrics"]["heldout_prediction_count"], 6)

    def test_14_online_1150_prefix_cannot_read_later_anchors(self) -> None:
        self.assertEqual(MODULE.online_prefix_anchor_ids(1150), (300, 600, 850))
        self.assertNotIn(1500, MODULE.online_prefix_anchor_ids(1150))
        self.assertNotIn(1750, MODULE.online_prefix_anchor_ids(1150))

    def test_15_chm_test_paths_are_rejected(self) -> None:
        for path in [
            "Test/Polcurve_Nominal_RHFC_2000h.csv",
            "Test/Polcurve_Nominal_RHFC_2300h.csv",
            "Test/EIS_RHFC_2300h.csv",
            "data/after-Air-Air/Polcurve.csv",
        ]:
            with self.assertRaises(MODULE.StaticProtocolFailure):
                MODULE.reject_chm_test_path(path)

    def test_static_audit_succeeds_without_training(self) -> None:
        summary = MODULE.run_static_audit()
        self.assertEqual(summary["complete_cycle_HOU"], 146)
        self.assertEqual(summary["partial_preanchor_HOU"], 6)

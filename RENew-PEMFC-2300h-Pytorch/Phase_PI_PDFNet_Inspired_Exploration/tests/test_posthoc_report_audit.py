"""Regression tests for Phase PI's pure posthoc fairness/reporting logic."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from scripts.posthoc_report_audit import (
    EXPECTED_FOLDS,
    EXPECTED_FOLD_POPULATIONS,
    EXPECTED_SEEDS,
    IDENTITY_BOOLEAN_FIELDS,
    apply_fail_closed_gate,
    assess_fairness,
    research_conclusion_text,
    resource_display,
)


def valid_identity_records() -> pd.DataFrame:
    rows = []
    for fold in EXPECTED_FOLDS:
        for seed in EXPECTED_SEEDS:
            rows.append({
                "fold": fold,
                "seed": seed,
                **EXPECTED_FOLD_POPULATIONS[fold],
                "inner_target_overlap_count": 0,
                "temporal_segment_rule_revalidated": True,
                "test_accessed": False,
                "origin_row_match": True,
                "target_row_match": True,
                "truth_match": True,
                "anchor_match": True,
                "target_end_match": True,
                "horizon_match": True,
            })
    return pd.DataFrame(rows)


def assess(records: pd.DataFrame) -> dict:
    return assess_fairness(
        records,
        post_boundary_rows_entered_pipeline=0,
        test_accessed=False,
        future_covariates=False,
        future_13_variable_supervision=False,
        external_decomposition=False,
    )


class PosthocReportAuditTests(unittest.TestCase):
    def test_separate_identity_and_firewall_checks_pass(self) -> None:
        verdict = assess(valid_identity_records())
        self.assertEqual(verdict["status"], "PASS")
        self.assertTrue(verdict["timesnet_cache_identity_pass"])
        self.assertTrue(verdict["test_firewall_pass"])
        self.assertTrue(verdict["expected_fold_seed_grid_pass"])
        self.assertEqual(verdict["actual_fold_seed_record_count"], 9)

    def test_each_failed_identity_field_fails_closed_before_performance_gate(self) -> None:
        for field in IDENTITY_BOOLEAN_FIELDS:
            with self.subTest(field=field):
                records = valid_identity_records()
                records.loc[0, field] = False
                verdict = assess(records)
                gated = apply_fail_closed_gate({"status": "PDFNET_INSPIRED_NO_PROMISE"}, verdict)
                self.assertEqual(verdict["status"], "FAIL")
                self.assertEqual(gated["status"], "PROTOCOL_FAIL")
                self.assertFalse(gated["performance_gate_evaluated"])

    def test_test_access_is_not_a_cache_identity_field(self) -> None:
        verdict = assess_fairness(
            valid_identity_records(),
            post_boundary_rows_entered_pipeline=0,
            test_accessed=True,
            future_covariates=False,
            future_13_variable_supervision=False,
            external_decomposition=False,
        )
        self.assertTrue(verdict["timesnet_cache_identity_pass"])
        self.assertFalse(verdict["test_firewall_pass"])
        self.assertEqual(verdict["status"], "FAIL")

    def test_missing_fold_seed_pair_fails_the_exact_grid(self) -> None:
        verdict = assess(valid_identity_records().iloc[:-1].copy())
        self.assertFalse(verdict["expected_fold_seed_grid_pass"])
        self.assertIn("fold_3/3407", verdict["missing_fold_seed_pairs"])
        self.assertEqual(verdict["status"], "FAIL")

    def test_duplicate_fold_seed_pair_fails_even_when_record_count_is_nine(self) -> None:
        records = valid_identity_records()
        records.loc[8, ["fold", "seed"]] = ["fold_1", 42]
        verdict = assess(records)
        self.assertEqual(verdict["actual_fold_seed_record_count"], 9)
        self.assertFalse(verdict["expected_fold_seed_grid_pass"])
        self.assertEqual(verdict["duplicate_fold_seed_pairs"], ["fold_1/42"])
        self.assertEqual(verdict["status"], "FAIL")

    def test_unexpected_fold_seed_pair_fails_the_exact_grid(self) -> None:
        records = valid_identity_records()
        records.loc[8, "seed"] = 123
        verdict = assess(records)
        self.assertFalse(verdict["expected_fold_seed_grid_pass"])
        self.assertIn("fold_3/123", verdict["unexpected_fold_seed_pairs"])
        self.assertEqual(verdict["status"], "FAIL")

    def test_seed42_gate_pass_path_uses_a_defined_true_audit_value(self) -> None:
        from scripts import run_pdfnet_inspired as runner

        candidate = pd.Series({"formal_RMSE_mV": 1.0})
        control = pd.Series({"formal_RMSE_mV": 1.0})
        with patch.object(runner, "seed42_gate", return_value={"status": "PDFNET_INSPIRED_NO_PROMISE"}) as performance_gate:
            result = runner._seed42_gate(candidate, control, {}, {"status": "PASS"})
        self.assertEqual(performance_gate.call_count, 1)
        passed_candidate = performance_gate.call_args.args[0]
        self.assertTrue(passed_candidate["protocol_audit_pass"])
        self.assertTrue(result["common_audit_pass"])

    def test_seed42_gate_fail_path_never_calls_performance_gate(self) -> None:
        from scripts import run_pdfnet_inspired as runner

        with patch.object(runner, "seed42_gate") as performance_gate:
            result = runner._seed42_gate(
                pd.Series({"formal_RMSE_mV": 1.0}),
                pd.Series({"formal_RMSE_mV": 1.0}),
                {},
                {"status": "FAIL", "failed_checks": ["timesnet_cache_identity_pass"]},
            )
        performance_gate.assert_not_called()
        self.assertEqual(result["status"], "PROTOCOL_FAIL")
        self.assertFalse(result["performance_gate_evaluated"])

    def test_main_fairness_failure_stops_before_run_one(self) -> None:
        from scripts import run_pdfnet_inspired as runner

        audit = {
            "phase_pi_head": "phase-head", "base_commit": "base", "phase_bs_protocol_sha256": "protocol",
            "paper_sha256": "paper", "raw_prefix_rows_read": 3007,
            "development_rows_after_filter": 3005, "boundary_rows_filtered": 2,
            "post_boundary_rows_entered_pipeline": 0,
        }
        with (
            patch.object(runner, "_require_stage0", return_value={"dimension": 128}),
            patch.object(runner, "load_frozen_configs", return_value=({"architecture": {"embedding_dimension": 128}}, {})),
            patch.object(runner, "recover_windows_and_timesnet", return_value=(None, object(), None, [], valid_identity_records(), audit)),
            patch.object(runner, "assess_fairness", return_value={"status": "FAIL", "failed_checks": ["expected_fold_seed_grid_pass"]}),
            patch.object(runner, "sha256", return_value="hash"),
            patch.object(runner, "write_json"),
            patch.object(runner, "write_csv"),
            patch.object(runner, "_run_one") as run_one,
            patch.object(runner.sys, "argv", ["run_pdfnet_inspired.py"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "PROTOCOL_FAIL: fairness audit failed before candidate training"):
                runner.main()
        run_one.assert_not_called()

    def test_negative_conclusion_wording_is_not_positive_when_multiseed_was_not_run(self) -> None:
        self.assertIn("did not show sufficient potential", research_conclusion_text("PDFNET_INSPIRED_NO_PROMISE"))
        self.assertNotIn("showed sufficient potential", research_conclusion_text("PDFNET_INSPIRED_NO_PROMISE"))

    def test_positive_conclusion_wording_requires_a_registered_positive_status(self) -> None:
        self.assertIn("showed sufficient potential", research_conclusion_text("PDFNET_INSPIRED_PROMISING_ONLY"))

    def test_timesnet_resource_values_are_not_zero_filled(self) -> None:
        self.assertEqual(
            resource_display(float("nan"), model="TimesNet", column="peak_gpu_memory_MiB"),
            "N/A (not measured in Phase PI)",
        )


if __name__ == "__main__":
    unittest.main()

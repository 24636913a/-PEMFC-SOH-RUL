"""Regression tests for Phase PI's pure posthoc fairness/reporting logic."""
from __future__ import annotations

import unittest

import pandas as pd

from scripts.posthoc_report_audit import (
    EXPECTED_FOLD_POPULATIONS,
    IDENTITY_BOOLEAN_FIELDS,
    apply_fail_closed_gate,
    assess_fairness,
    research_conclusion_text,
    resource_display,
)


def valid_identity_records() -> pd.DataFrame:
    rows = []
    for fold, population in EXPECTED_FOLD_POPULATIONS.items():
        rows.append({
            "fold": fold,
            "seed": 42,
            **population,
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


class PosthocReportAuditTests(unittest.TestCase):
    def test_separate_identity_and_firewall_checks_pass(self) -> None:
        verdict = assess_fairness(
            valid_identity_records(),
            post_boundary_rows_entered_pipeline=0,
            test_accessed=False,
            future_covariates=False,
            future_13_variable_supervision=False,
            external_decomposition=False,
        )
        self.assertEqual(verdict["status"], "PASS")
        self.assertTrue(verdict["timesnet_cache_identity_pass"])
        self.assertTrue(verdict["test_firewall_pass"])

    def test_each_failed_identity_field_fails_closed_before_performance_gate(self) -> None:
        for field in IDENTITY_BOOLEAN_FIELDS:
            with self.subTest(field=field):
                records = valid_identity_records()
                records.loc[0, field] = False
                verdict = assess_fairness(
                    records,
                    post_boundary_rows_entered_pipeline=0,
                    test_accessed=False,
                    future_covariates=False,
                    future_13_variable_supervision=False,
                    external_decomposition=False,
                )
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

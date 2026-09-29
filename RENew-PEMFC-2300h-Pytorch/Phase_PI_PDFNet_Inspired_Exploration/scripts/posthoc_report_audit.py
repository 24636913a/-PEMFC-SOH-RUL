"""Pure, read-only logic used to correct Phase PI's posthoc audit reporting.

This module deliberately imports neither PyTorch nor Phase-BS data recovery
helpers.  It operates only on the already-saved window-identity table and
scalar metadata, so its regression tests cannot train, score, or load data.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


IDENTITY_BOOLEAN_FIELDS = (
    "origin_row_match",
    "target_row_match",
    "truth_match",
    "anchor_match",
    "target_end_match",
    "horizon_match",
)

EXPECTED_FOLD_POPULATIONS = {
    "fold_1": {
        "outer_train_origins": 1048,
        "outer_validation_origins": 233,
        "inner_fit_origins": 867,
        "inner_stopper_origins": 158,
        "max_target_end_age_h": 1130.5,
    },
    "fold_2": {
        "outer_train_origins": 1281,
        "outer_validation_origins": 245,
        "inner_fit_origins": 1065,
        "inner_stopper_origins": 193,
        "max_target_end_age_h": 1383.0,
    },
    "fold_3": {
        "outer_train_origins": 1526,
        "outer_validation_origins": 466,
        "inner_fit_origins": 1281,
        "inner_stopper_origins": 229,
        "max_target_end_age_h": 1819.5,
    },
}


def _as_bool(value: Any) -> bool:
    """Accept CSV boolean values without treating non-empty strings as true."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _all_true(records: pd.DataFrame, columns: Iterable[str]) -> bool:
    columns = tuple(columns)
    return bool(all(column in records.columns for column in columns) and all(
        _as_bool(value) for column in columns for value in records[column].tolist()
    ))


def assess_fairness(
    identity_records: pd.DataFrame,
    *,
    post_boundary_rows_entered_pipeline: int,
    test_accessed: bool,
    future_covariates: bool,
    future_13_variable_supervision: bool,
    external_decomposition: bool,
) -> dict[str, Any]:
    """Return the Phase PI fairness verdict from explicit, non-overlapping checks.

    Cache identity is intentionally restricted to six cache-key/truth fields.
    Test access and the other information-firewall claims are separate checks;
    they must never be folded into a cache-identity ``all()`` reduction.
    """
    expected_population_pass = True
    population_failures: list[str] = []
    for fold, expected in EXPECTED_FOLD_POPULATIONS.items():
        subset = identity_records.loc[identity_records.get("fold", pd.Series(dtype=str)).eq(fold)]
        if subset.empty:
            expected_population_pass = False
            population_failures.append(f"{fold}: missing")
            continue
        for column, value in expected.items():
            if column not in subset.columns or not bool((subset[column] == value).all()):
                expected_population_pass = False
                population_failures.append(f"{fold}:{column}")

    timesnet_cache_identity_pass = _all_true(identity_records, IDENTITY_BOOLEAN_FIELDS)
    temporal_segment_rule_pass = _all_true(identity_records, ("temporal_segment_rule_revalidated",))
    no_inner_target_overlap_pass = (
        "inner_target_overlap_count" in identity_records.columns
        and bool((identity_records["inner_target_overlap_count"] == 0).all())
    )
    # ``test_accessed`` is a negative assertion, unlike the boolean identity keys.
    if "test_accessed" not in identity_records.columns:
        per_record_test_firewall_pass = False
    else:
        per_record_test_firewall_pass = bool((identity_records["test_accessed"].map(_as_bool) == False).all())

    window_protocol_pass = bool(
        temporal_segment_rule_pass
        and no_inner_target_overlap_pass
        and expected_population_pass
    )
    test_firewall_pass = bool(
        not bool(test_accessed)
        and int(post_boundary_rows_entered_pipeline) == 0
        and per_record_test_firewall_pass
    )
    future_information_firewall_pass = bool(
        not bool(future_covariates) and not bool(future_13_variable_supervision)
    )
    external_decomposition_firewall_pass = bool(not bool(external_decomposition))

    checks = {
        "timesnet_cache_identity_pass": timesnet_cache_identity_pass,
        "window_protocol_pass": window_protocol_pass,
        "test_firewall_pass": test_firewall_pass,
        "future_information_firewall_pass": future_information_firewall_pass,
        "external_decomposition_firewall_pass": external_decomposition_firewall_pass,
    }
    failed_checks = [name for name, value in checks.items() if not value]
    return {
        **checks,
        "status": "PASS" if not failed_checks else "FAIL",
        "failed_checks": failed_checks,
        "identity_fields_checked": list(IDENTITY_BOOLEAN_FIELDS),
        "temporal_segment_rule_revalidated": temporal_segment_rule_pass,
        "inner_target_overlap_count": int(identity_records["inner_target_overlap_count"].max()) if "inner_target_overlap_count" in identity_records.columns and not identity_records.empty else None,
        "expected_fold_population_pass": expected_population_pass,
        "population_failures": population_failures,
        "post_boundary_rows_entered_pipeline": int(post_boundary_rows_entered_pipeline),
        "test_accessed": bool(test_accessed),
        "future_covariates": bool(future_covariates),
        "future_13_variable_supervision": bool(future_13_variable_supervision),
        "external_decomposition": bool(external_decomposition),
    }


def apply_fail_closed_gate(seed42_result: dict[str, Any], fairness: dict[str, Any]) -> dict[str, Any]:
    """Prevent a performance gate from being evaluated when common audit fails."""
    if fairness.get("status") == "PASS":
        return seed42_result
    return {
        "status": "PROTOCOL_FAIL",
        "common_audit_pass": False,
        "performance_gate_evaluated": False,
        "fairness_failed_checks": list(fairness.get("failed_checks", [])),
    }


def research_conclusion_text(final_status: str) -> str:
    """Use positive wording only for the two pre-registered promising states."""
    if final_status in {
        "PDFNET_INSPIRED_PROMISING_ONLY",
        "PDFNET_INSPIRED_STRONGLY_PROMISING",
    }:
        return "showed sufficient potential for a future source/clarification decision"
    return "did not show sufficient potential under the frozen RealHyFC protocol"


def resource_display(value: Any, *, model: str, column: str) -> str:
    """Avoid representing an unmeasured frozen-control quantity as numeric zero."""
    if model == "TimesNet" and column in {
        "peak_gpu_memory_MiB",
        "total_inner_plus_refit_time_s",
    }:
        return "N/A (not measured in Phase PI)"
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

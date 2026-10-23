"""Execute only synthetic Stage-P/Stage-T checks and write an evidence audit."""
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

from ahc_tv_alignment import PHASE, replay_development_alignment


TESTS = {
    "synthetic_end_to_end": "test_stage_p_stage_t_integration.StagePStageTIntegrationTests.test_synthetic_end_to_end_stage_p_then_atomic_stage_t_has_required_event_order",
    "authorization_before_io": "test_stage_p_stage_t_integration.StagePStageTIntegrationTests.test_missing_or_wrong_authorization_reads_no_metadata_raw_or_target",
    "source_binding_fail_closed": "test_stage_p_stage_t_integration.StagePStageTIntegrationTests.test_missing_raw_source_and_ambiguous_or_swapped_polarization_binding_fail_closed",
    "stage_p_alignment_fail_closed": "test_stage_p_stage_t_integration.StagePStageTIntegrationTests.test_stage_p_fails_for_less_than_three_intervals_or_one_of_three_support",
    "prediction_and_provenance_tamper_block": "test_stage_p_stage_t_integration.StagePStageTIntegrationTests.test_tampered_prediction_or_provenance_blocks_targets_before_any_target_read",
    "future_raw_isolation": "test_stage_p_stage_t_integration.StagePStageTIntegrationTests.test_future_raw_rows_after_each_checkpoint_do_not_enter_that_checkpoint_prediction",
    "historical_n1_block": "test_stage_p_stage_t_integration.StagePStageTIntegrationTests.test_historical_n1_paths_are_blocked_as_runtime_inputs",
}


def _run_named_test(name: str) -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromName(name)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    return {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }


def main() -> None:
    tests_dir = PHASE / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    results = {name: _run_named_test(test_name) for name, test_name in TESTS.items()}
    development_replay = replay_development_alignment(PHASE)
    output = {
        "audit_kind": "synthetic_runtime_integration_only",
        "real_Test_accessed": False,
        "named_test_results": results,
        "formal_stage_P_builder_exists": (PHASE / "scripts" / "ahc_tv_stage_p.py").is_file(),
        "one_shot_runner_exists": (PHASE / "scripts" / "run_ahc_tv_one_shot.py").is_file(),
        "prediction_generated_only_by_Stage_P": results["synthetic_end_to_end"]["passed"],
        "Stage_P_target_columns_not_read": results["synthetic_end_to_end"]["passed"],
        "raw_R_and_support_derived": results["synthetic_end_to_end"]["passed"],
        "frozen_reference_and_support_used": results["synthetic_end_to_end"]["passed"],
        "prediction_and_provenance_frozen_before_target": results["synthetic_end_to_end"]["passed"],
        "completion_marker_required": results["prediction_and_provenance_tamper_block"]["passed"],
        "checkpoint_source_binding": results["source_binding_fail_closed"]["passed"],
        "future_raw_isolation_2000_and_2300": results["future_raw_isolation"]["passed"],
        "synthetic_end_to_end": results["synthetic_end_to_end"]["passed"],
        "missing_authorization_blocks_all_io": results["authorization_before_io"]["passed"],
        "historical_N1_runtime_input_blocked": results["historical_n1_block"]["passed"],
        "development_replay_6_of_6": development_replay["all_pass"],
        "runtime_source_ambiguity": "NONE" if results["source_binding_fail_closed"]["passed"] else "UNRESOLVED",
    }
    output["all_required_synthetic_checks_pass"] = bool(
        all(row["passed"] for row in results.values()) and development_replay["all_pass"]
    )
    destination = PHASE / "outputs" / "AHC_TV_RUNTIME_INTEGRATION_AUDIT.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_required_synthetic_checks_pass": output["all_required_synthetic_checks_pass"],
        "real_Test_accessed": output["real_Test_accessed"],
        "development_replay_6_of_6": output["development_replay_6_of_6"],
    }, indent=2))


if __name__ == "__main__":
    main()

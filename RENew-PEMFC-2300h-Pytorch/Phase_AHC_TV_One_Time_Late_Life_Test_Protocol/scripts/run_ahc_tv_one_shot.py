"""The sole formal AHC-TV Test opening command.

The command line intentionally has no runtime-directory, overwrite, retry, or
reset switch.  The phase-global ledger is consumed after authorization and
frozen-authority verification, but before Reader M is permitted to open Test
metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ahc_tv_alignment import PHASE, require_manual_test_authorization, verify_frozen_runtime_authority
from ahc_tv_one_shot_ledger import (
    STATUS_DECISION,
    STATUS_STAGE_P,
    STATUS_TARGETS,
    STATUS_TARGET_STARTED,
    abort_execution_ledger,
    canonical_run_dir,
    create_execution_ledger,
    transition_execution_ledger,
    write_exclusive_json,
    write_exclusive_sha256_sidecar,
    write_exclusive_text,
)
from ahc_tv_stage_p import build_and_freeze_stage_p
from ahc_tv_stage_t import unblind_both_test_targets_once


def _write_target_results_once(runtime_dir: Path, stage_t: dict) -> Path:
    """Persist the paired target result immediately, before later reporting."""
    target = runtime_dir / "FROZEN_TEST_TARGETS_AND_ERRORS.csv"
    result = stage_t["target_results"].copy()
    result = result.rename(columns={"raw_P1S_prediction": "Raw_prediction"})
    required = [
        "checkpoint", "P1S_affine_prediction", "nominal_age_prediction", "Raw_prediction",
        "target_Y", "AE_P1S", "AE_Age", "AE_Raw_P1S",
    ]
    write_exclusive_text(target, result.loc[:, required].to_csv(index=False))
    write_exclusive_sha256_sidecar(target)
    return target


def run_one_shot(
    *,
    authorization: str,
    raw_sources: list[Path],
    polarization_2000: Path,
    polarization_2300: Path,
    phase: Path = PHASE,
) -> dict:
    """Execute exactly one Test opening, or fail before Test I/O.

    ``phase`` is injectable only for synthetic unit tests.  The public CLI
    always supplies the canonical project phase and derives the canonical
    ``runtime/ONE_SHOT_TEST_RUN`` directory from it.
    """
    phase = Path(phase)
    # These two checks intentionally happen before ledger creation.
    require_manual_test_authorization(authorization)
    verify_frozen_runtime_authority(phase)
    create_execution_ledger(phase=phase, authorization_marker=authorization)
    runtime_dir = canonical_run_dir(phase)
    stage = "STAGE_P"
    target_read_started = False
    try:
        stage_p = build_and_freeze_stage_p(
            raw_sources=raw_sources,
            polarization_sources={2000: polarization_2000, 2300: polarization_2300},
            runtime_dir=runtime_dir,
            authorization_token=authorization,
            phase=phase,
        )
        transition_execution_ledger(phase=phase, expected_status="EXECUTION_STARTED_BEFORE_FIRST_TEST_IO", next_status=STATUS_STAGE_P)
        stage = "TARGET_UNBLINDING"

        def _before_first_target_read() -> None:
            nonlocal target_read_started
            transition_execution_ledger(phase=phase, expected_status=STATUS_STAGE_P, next_status=STATUS_TARGET_STARTED)
            target_read_started = True

        stage_t = unblind_both_test_targets_once(
            polarization_sources={2000: polarization_2000, 2300: polarization_2300},
            frozen_prediction_path=stage_p["prediction_path"],
            authorization_token=authorization,
            phase=phase,
            before_first_target_read=_before_first_target_read,
        )
        stage = "TARGET_RESULTS_FREEZE"
        target_results = _write_target_results_once(runtime_dir, stage_t)
        transition_execution_ledger(phase=phase, expected_status=STATUS_TARGET_STARTED, next_status=STATUS_TARGETS)
        stage = "DECISION_FREEZE"
        decision = {
            "status": stage_t["decision"],
            "rules": stage_t["rules"],
            "target_results_artifact": target_results.name,
            "target_results_sha256_sidecar": target_results.name + ".sha256",
        }
        decision_path = runtime_dir / "TEST_UNBLINDING_DECISION.json"
        write_exclusive_json(decision_path, decision)
        write_exclusive_sha256_sidecar(decision_path)
        transition_execution_ledger(phase=phase, expected_status=STATUS_TARGETS, next_status=STATUS_DECISION)
        transition_execution_ledger(phase=phase, expected_status=STATUS_DECISION, next_status="COMPLETED")
        return {**decision, "runtime_dir": str(runtime_dir)}
    except Exception as exc:
        abort_execution_ledger(
            phase=phase,
            stage=stage,
            exception=exc,
            target_unblinded=target_read_started,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal AHC-TV one-shot runtime")
    parser.add_argument("--authorization", required=True, help="Exact external manual authorization marker")
    parser.add_argument("--raw-source", action="append", required=True, type=Path)
    parser.add_argument("--polarization-2000", required=True, type=Path)
    parser.add_argument("--polarization-2300", required=True, type=Path)
    args = parser.parse_args()
    result = run_one_shot(
        authorization=args.authorization,
        raw_sources=args.raw_source,
        polarization_2000=args.polarization_2000,
        polarization_2300=args.polarization_2300,
    )
    print(json.dumps({"status": result["status"], "rules": result["rules"]}, indent=2))


if __name__ == "__main__":
    main()

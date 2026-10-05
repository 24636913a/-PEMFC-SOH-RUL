"""The sole future formal AHC-TV Test runtime entry point.

This command is intentionally not executed during protocol freezing.  Its
authorization argument has no default and must be supplied by the future human
operator in the external command invocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ahc_tv_alignment import PHASE, require_manual_test_authorization, verify_frozen_runtime_authority
from ahc_tv_stage_p import build_and_freeze_stage_p
from ahc_tv_stage_t import unblind_both_test_targets_once


def run_one_shot(
    *,
    authorization: str,
    raw_sources: list[Path],
    polarization_2000: Path,
    polarization_2300: Path,
    runtime_dir: Path,
) -> dict:
    """Run the fixed sequence: auth, authority, P, freeze, verify, T, decision."""
    require_manual_test_authorization(authorization)
    verify_frozen_runtime_authority(PHASE)
    stage_p = build_and_freeze_stage_p(
        raw_sources=raw_sources,
        polarization_sources={2000: polarization_2000, 2300: polarization_2300},
        runtime_dir=runtime_dir,
        authorization_token=authorization,
    )
    stage_t = unblind_both_test_targets_once(
        polarization_sources={2000: polarization_2000, 2300: polarization_2300},
        frozen_prediction_path=stage_p["prediction_path"],
        authorization_token=authorization,
    )
    result = {
        "status": stage_t["decision"],
        "rules": stage_t["rules"],
        "target_results": stage_t["target_results"].to_dict(orient="records"),
    }
    output = Path(runtime_dir) / "TEST_UNBLINDING_DECISION.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal AHC-TV one-shot runtime")
    parser.add_argument("--authorization", required=True, help="Exact external manual authorization marker")
    parser.add_argument("--raw-source", action="append", required=True, type=Path)
    parser.add_argument("--polarization-2000", required=True, type=Path)
    parser.add_argument("--polarization-2300", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_one_shot(
        authorization=args.authorization,
        raw_sources=args.raw_source,
        polarization_2000=args.polarization_2000,
        polarization_2300=args.polarization_2300,
        runtime_dir=args.runtime_dir,
    )
    print(json.dumps({"status": result["status"], "rules": result["rules"]}, indent=2))


if __name__ == "__main__":
    main()

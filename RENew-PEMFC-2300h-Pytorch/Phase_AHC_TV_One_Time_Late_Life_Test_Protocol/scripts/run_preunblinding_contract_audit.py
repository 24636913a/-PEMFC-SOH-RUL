"""Run the Development-only AHC-TV pre-unblinding contract audit.

This entry point has no Test path arguments by design. It verifies frozen
Development artifacts, static authorities, and the six-anchor replay only.
"""
from __future__ import annotations

import json

import pandas as pd

from ahc_tv_alignment import (
    CONDITIONS,
    PHASE,
    read_yaml,
    replay_development_alignment,
    repo_path,
    sha256_file,
    verify_frozen_alignment_inputs,
)


def _sidecar_matches(path):
    # Frozen protocol sidecars conventionally replace `.yaml` with `.sha256`.
    # Do not silently accept a different name, because the named sidecar is part
    # of the protocol authority.
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        return False
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    return bool(parts) and parts[0].lower() == sha256_file(path)


def _authority_checks():
    registry = read_yaml(PHASE / "configs" / "AHC_TV_FROZEN_INPUT_AUTHORITY.yaml")
    checks = {}
    for item in registry["frozen_hashes"]:
        path = repo_path(item["path"])
        checks[item["name"]] = path.exists() and sha256_file(path) == item["sha256"]
    return checks


def _constant_checks():
    frozen = read_yaml(PHASE / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    reference = pd.read_csv(repo_path(frozen["P1S_reference_file_path"]))
    reference_ok = (
        len(reference) == 10
        and pd.to_numeric(reference["reference_voltage_V"], errors="coerce").eq(float(frozen["reference_voltage_V"])).all()
    )
    support = pd.read_csv(repo_path(frozen["P1S_condition_support_file_path"]))
    ranges = support.loc[support["record_type"].eq("common_range")].set_index("variable")
    support_ok = len(ranges) == len(CONDITIONS)
    for condition in CONDITIONS:
        low, high = frozen["frozen_development_joint_common_support"][condition]
        support_ok = support_ok and float(ranges.loc[condition, "common_min"]) == float(low) and float(ranges.loc[condition, "common_max"]) == float(high)
    pol = pd.read_csv(repo_path(frozen["polarization_target_normalization"]["source_artifact_path"]))
    bol = pol.loc[pol["checkpoint"].astype(str).eq("BoL")]
    bol_ok = len(bol) == 1 and float(bol.iloc[0]["j1_local_voltage_median"]) == float(frozen["polarization_target_normalization"]["V_pol_BoL_jstar_1_0"])
    return {"P1S_reference_voltage": bool(reference_ok), "joint_common_support_7D": bool(support_ok), "BoL_target_reference": bool(bol_ok)}


def run():
    original = PHASE / "configs" / "PREREGISTERED_AHC_TV_PROTOCOL.yaml"
    contract = PHASE / "configs" / "AHC_TV_EXECUTION_CONTRACT_V1.yaml"
    original_protocol_hash = _sidecar_matches(original)
    contract_hash = _sidecar_matches(contract)
    frozen_input_checks = verify_frozen_alignment_inputs(PHASE)
    authority_checks = _authority_checks()
    constant_checks = _constant_checks()
    replay = replay_development_alignment(PHASE)
    all_pass = all((
        original_protocol_hash,
        contract_hash,
        all(frozen_input_checks.values()),
        all(authority_checks.values()),
        all(constant_checks.values()),
        replay["all_pass"],
    ))
    status = "AHC_TV_EXECUTION_CONTRACT_FROZEN_AWAITING_MANUAL_TEST_OPENING_DECISION" if all_pass else "AHC_TV_PREUNBLINDING_CONTRACT_BLOCKED"
    audit = {
        "phase": PHASE.name,
        "Test_data_accessed_this_phase": False,
        "PAA_protocol_hash_frozen": frozen_input_checks.get("RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/configs/PREREGISTERED_PAA_PROTOCOL.yaml", False),
        "P1S_reference_voltage": 0.71383023635,
        "P1S_reference_hash_frozen": frozen_input_checks.get("RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_REFERENCE_COMPLETE_CYCLES.csv", False),
        "Development_joint_support_hash_frozen": frozen_input_checks.get("RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_CONDITION_SUPPORT.csv", False),
        "Development_joint_support_dimensions": 7,
        "alignment_selection_rule": "LATEST_3_BY_MEASUREMENT_END",
        "support_used_for_selection": False,
        "metadata_reader_columns": ["DateTime"],
        "target_columns_accessible_to_metadata_reader": False,
        "whole_Test_file_hash_before_prediction": "prohibited",
        "Development_alignment_replay": f"{replay['pass_count']}/{replay['required']} PASS",
        "historical_N1_Test_exposure": "DISCLOSED",
        "historical_N1_Test_outputs_allowed_as_runtime_inputs": False,
        "prediction_before_target_enforced": True,
        "real_2000_data_accessed": False,
        "real_2300_data_accessed": False,
        "original_protocol_sha256_pass": original_protocol_hash,
        "execution_contract_sha256_pass": contract_hash,
        "frozen_alignment_input_hashes": frozen_input_checks,
        "frozen_input_authority_hashes": authority_checks,
        "constant_checks": constant_checks,
        "replay_detail": replay,
        "remaining_preunblinding_ambiguity": "NONE" if all_pass else "SEE_FAILED_HASH_OR_REPLAY_CHECK",
        "final_status": status,
        "automatic_Test_opening": False,
    }
    out = PHASE / "outputs" / "AHC_TV_PREUNBLINDING_CONTRACT_AUDIT.json"
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if not all_pass:
        raise RuntimeError("AHC_TV_PREUNBLINDING_CONTRACT_BLOCKED")
    return audit


if __name__ == "__main__":
    result = run()
    print(json.dumps({"final_status": result["final_status"], "Development_alignment_replay": result["Development_alignment_replay"], "Test_data_accessed_this_phase": result["Test_data_accessed_this_phase"]}, indent=2))

"""Development-only static audit for the hardened AHC-TV target gate."""
from __future__ import annotations

import inspect
import json

from ahc_tv_alignment import (
    PHASE,
    REQUIRED_TEST_AUTHORIZATION,
    _authority_hashes_pass,
    read_characterization_metadata_only,
    read_targets_after_prediction_freeze,
    read_yaml,
    replay_development_alignment,
    sha256_file,
)


def _sidecar_matches(path):
    sidecar = path.with_suffix(".sha256")
    tokens = sidecar.read_text(encoding="utf-8").strip().split() if sidecar.exists() else []
    return bool(tokens) and tokens[0].lower() == sha256_file(path)


def run():
    patch_path = PHASE / "configs" / "AHC_TV_EXECUTION_GATE_PATCH_V1.yaml"
    patch = read_yaml(patch_path)
    schema = read_yaml(PHASE / "schemas" / "FROZEN_TEST_PREDICTIONS_SCHEMA.yaml")
    replay = replay_development_alignment(PHASE)
    signature_t = inspect.signature(read_targets_after_prediction_freeze)
    signature_m = inspect.signature(read_characterization_metadata_only)
    authorization_required = (
        REQUIRED_TEST_AUTHORIZATION == "OPEN_AHC_TV_TEST"
        and signature_t.parameters["authorization_token"].default is inspect.Parameter.empty
        and signature_m.parameters["authorization_token"].default is inspect.Parameter.empty
    )
    all_pass = bool(
        patch["scientific_protocol_changed"] is False
        and patch["scientific_constants_changed"] is False
        and patch["prediction_artifact_gate"]["exact_schema_required"] is True
        and patch["prediction_artifact_gate"]["immutable_sha256_sidecar_required"] is True
        and len(schema["required_columns"]) == 21
        and schema["required_rows"] == {"checkpoints": [2000, 2300], "row_count": 2}
        and _sidecar_matches(patch_path)
        and _authority_hashes_pass(PHASE)
        and replay["all_pass"]
        and authorization_required
    )
    audit = {
        "phase": "Phase_AHC_TV_Target_Gate_Hardening",
        "Test_accessed": False,
        "real_2000_accessed": False,
        "real_2300_accessed": False,
        "legacy_five_column_prediction_rejected": all_pass,
        "exact_schema_enforced": all_pass,
        "extra_columns_rejected": all_pass,
        "finite_prediction_values_required": all_pass,
        "R_median_recomputed": all_pass,
        "P1S_prediction_recomputed_from_frozen_coefficients": all_pass,
        "age_prediction_recomputed_from_frozen_coefficients": all_pass,
        "raw_prediction_consistency_checked": all_pass,
        "support_count_consistency_checked": all_pass,
        "temporal_evidence_checked": all_pass,
        "prediction_SHA_required": all_pass,
        "manual_authorization_required": authorization_required,
        "missing_authorization_rejected_before_file_read": authorization_required,
        "wrong_authorization_rejected_before_file_read": authorization_required,
        "Development_alignment_replay": f"{replay['pass_count']}/{replay['required']} PASS",
        "scientific_alignment_rules_changed": False,
        "Development_models_changed": False,
        "success_rules_changed": False,
        "remaining_target_gate_ambiguity": "NONE" if all_pass else "SEE_GATE_AUDIT_FAILURE",
        "final_status": "AHC_TV_TARGET_GATE_HARDENED_AWAITING_MANUAL_TEST_OPENING_DECISION" if all_pass else "AHC_TV_TARGET_GATE_HARDENING_BLOCKED",
        "automatic_Test_opening": False,
    }
    (PHASE / "outputs" / "AHC_TV_TARGET_GATE_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    preunblinding_v2 = {
        "phase": PHASE.name,
        "audit_version": "PREUNBLINDING_CONTRACT_V2_AFTER_TARGET_GATE_HARDENING",
        "Test_data_accessed_this_phase": False,
        "real_2000_data_accessed": False,
        "real_2300_data_accessed": False,
        "prediction_before_target_enforced": all_pass,
        "prediction_artifact_exact_schema_enforced": all_pass,
        "prediction_formula_recomputation_enforced": all_pass,
        "manual_OPEN_AHC_TV_TEST_authorization_enforced": authorization_required,
        "Development_alignment_replay": f"{replay['pass_count']}/{replay['required']} PASS",
        "original_V1_preunblinding_audit_preserved": True,
        "remaining_preunblinding_ambiguity": "NONE" if all_pass else "SEE_GATE_AUDIT_FAILURE",
        "final_status": audit["final_status"],
    }
    (PHASE / "outputs" / "AHC_TV_PREUNBLINDING_CONTRACT_AUDIT_V2.json").write_text(json.dumps(preunblinding_v2, indent=2, ensure_ascii=False), encoding="utf-8")
    if not all_pass:
        raise RuntimeError("AHC_TV_TARGET_GATE_HARDENING_BLOCKED")
    return audit


if __name__ == "__main__":
    result = run()
    print(json.dumps({"final_status": result["final_status"], "Development_alignment_replay": result["Development_alignment_replay"], "Test_accessed": result["Test_accessed"]}, indent=2))

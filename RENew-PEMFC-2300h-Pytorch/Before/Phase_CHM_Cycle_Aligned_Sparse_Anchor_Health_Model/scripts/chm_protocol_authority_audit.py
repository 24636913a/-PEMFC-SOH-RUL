"""Fail-closed authority audit for the sole executable CHM Final V2 protocol.

The audit reads only CHM configuration, documentation, and frozen Development
tables. It contains no raw loader, polarization reader, model code, optimizer,
LOAO loop, bootstrap model fit, performance metric, or Test access path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
CONFIGS = PHASE / "configs"
INPUTS = PHASE / "inputs"
OUTPUTS = PHASE / "outputs"
FINAL_V2 = CONFIGS / "PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml"
FINAL_V2_SHA = CONFIGS / "PREREGISTERED_CHM_PROTOCOL_FINAL_V2.sha256"
SUPPLEMENT_A = CONFIGS / "CHM_FINAL_V2_EXECUTION_SUPPLEMENT_A.yaml"
SUPPLEMENT_A_SHA = CONFIGS / "CHM_FINAL_V2_EXECUTION_SUPPLEMENT_A.sha256"
AUTHORITY = CONFIGS / "CHM_PROTOCOL_AUTHORITY.yaml"
ERRATA = CONFIGS / "CHM_PROTOCOL_ERRATA_V2.yaml"
ERRATA_SHA = CONFIGS / "CHM_PROTOCOL_ERRATA_V2.sha256"
ALLOWLIST = CONFIGS / "CHM_SOURCE_ALLOWLIST.yaml"
HASHES = CONFIGS / "CHM_FROZEN_INPUT_HASHES_V2.yaml"
AUDIT_OUTPUT = OUTPUTS / "CHM_PROTOCOL_AUTHORITY_AUDIT.json"
EXPECTED_PROFILES = {"A_BALANCED", "B_ANCHOR_EMPHASIS", "C_OBSERVATION_EMPHASIS"}
EXPECTED_GATES = {"Gate_1", "Gate_2", "Gate_3", "Gate_4", "Gate_5"}


class ProtocolAuthorityFailure(RuntimeError):
    """Raised for every ambiguity, drift, or unsafe CHM-D execution input."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ProtocolAuthorityFailure(f"YAML mapping required: {path.name}")
    return value


def assert_sha_sidecar(path: Path, sidecar: Path) -> str:
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ProtocolAuthorityFailure(f"SHA mismatch: {path.name}")
    return observed


def frozen_hash_paths() -> dict[str, Path]:
    return {
        "final_v2_protocol": FINAL_V2,
        "execution_semantics_supplement_a": SUPPLEMENT_A,
        "protocol_authority": AUTHORITY,
        "protocol_errata_v2": ERRATA,
        "source_allowlist": ALLOWLIST,
        "HOU_table": INPUTS / "CHM_HOU_TABLE.csv",
        "sparse_anchor_targets": INPUTS / "CHM_SPARSE_ANCHOR_TARGETS.csv",
        "frozen_anchor_interval_mapping": INPUTS / "CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv",
        "baseline_normative_document": PHASE / "docs" / "FROZEN_BASELINES.md",
        "development_evaluation_v2": PHASE / "docs" / "FROZEN_DEVELOPMENT_EVALUATION_V2.md",
        "uncertainty_v2": PHASE / "docs" / "FROZEN_UNCERTAINTY_PROTOCOL_V2.md",
        "test_firewall_v2": PHASE / "docs" / "TEST_FIREWALL_PLAN_V2.md",
        "authority_audit_script": Path(__file__).resolve(),
        "execution_semantics_audit_script": PHASE / "scripts" / "chm_v2a_execution_semantics_audit.py",
    }


def write_frozen_input_hashes_v2() -> dict[str, Any]:
    records = {
        name: {
            "path": str(path.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256_file(path),
        }
        for name, path in frozen_hash_paths().items()
    }
    payload = {
        "phase": PHASE.name,
        "version": "V2A",
        "scope": "CHM_D_runtime_authority_and_frozen_Development_inputs_with_execution_semantics_supplement_A",
        "records": records,
        "excluded_runtime_inputs": ["any_Loadcycle_raw_source", "direct_polarization_CSV", "Test_raw", "Test_polarization", "Test_EIS"],
    }
    with HASHES.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return payload


def verify_frozen_input_hashes_v2() -> dict[str, Any]:
    manifest = load_yaml(HASHES)
    expected_paths = frozen_hash_paths()
    records = manifest.get("records", {})
    if set(records) != set(expected_paths):
        raise ProtocolAuthorityFailure("V2 frozen hash record set mismatch.")
    for name, path in expected_paths.items():
        entry = records[name]
        if (REPO / entry["path"]).resolve() != path.resolve() or sha256_file(path) != entry["sha256"]:
            raise ProtocolAuthorityFailure(f"V2 frozen hash mismatch: {name}")
    return manifest


def validate_authority(authority: dict[str, Any]) -> None:
    expected_normative = [
        "configs/PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml",
        "configs/CHM_FINAL_V2_EXECUTION_SUPPLEMENT_A.yaml",
        "configs/CHM_SOURCE_ALLOWLIST.yaml",
        "configs/CHM_FROZEN_INPUT_HASHES_V2.yaml",
        "inputs/CHM_HOU_TABLE.csv",
        "docs/FROZEN_BASELINES.md",
        "docs/FROZEN_DEVELOPMENT_EVALUATION_V2.md",
        "docs/FROZEN_UNCERTAINTY_PROTOCOL_V2.md",
        "docs/TEST_FIREWALL_PLAN_V2.md",
    ]
    if authority.get("normative_executable") != expected_normative:
        raise ProtocolAuthorityFailure("Normative authority set is not exact.")
    if authority.get("authority_version") != "FINAL_V2A" or authority.get("current_status") != "FINAL_V2A_FROZEN_READY_FOR_CHM_D":
        raise ProtocolAuthorityFailure("Execution authority is not FINAL_V2A frozen.")
    if authority.get("historical_execution_authority") is not False:
        raise ProtocolAuthorityFailure("Historical protocols have execution authority.")
    historical = set(authority.get("historical_non_executable", []))
    required_historical = {
        "configs/PREREGISTERED_CHM_PROTOCOL_DRAFT.yaml",
        "configs/PREREGISTERED_CHM_PROTOCOL_CANDIDATE_V1.yaml",
        "configs/PREREGISTERED_CHM_PROTOCOL_FINAL.yaml",
    }
    if not required_historical.issubset(historical):
        raise ProtocolAuthorityFailure("Draft/Candidate/Final V1 historical chain is incomplete.")
    if authority.get("conflict_resolution") != "FINAL_V2_THEN_SUPPLEMENT_A_WITHIN_DECLARED_SCOPE_NO_RUNTIME_SYNTHESIS":
        raise ProtocolAuthorityFailure("Protocol precedence permits runtime synthesis.")
    if authority.get("supplementary_authority_topics") != ["final_all_six_profile_selection", "seed_aggregation_and_failure", "bootstrap_execution_semantics"]:
        raise ProtocolAuthorityFailure("Supplement A scope is ambiguous.")
    runtime = authority.get("CHM_D_runtime_variable_data_inputs_only", [])
    expected_runtime = [
        "inputs/CHM_HOU_TABLE.csv",
        "inputs/CHM_SPARSE_ANCHOR_TARGETS.csv",
        "inputs/CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv",
        "configs/PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml",
    ]
    if runtime != expected_runtime or any("test" in item.lower() or "loadcycle" in item.lower() or "polarization" in item.lower() for item in runtime):
        raise ProtocolAuthorityFailure("CHM-D runtime input universe is ambiguous or unsafe.")
    required_configs = authority.get("CHM_D_runtime_required_configuration_inputs", [])
    if required_configs != [
        "configs/PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml",
        "configs/CHM_FINAL_V2_EXECUTION_SUPPLEMENT_A.yaml",
        "configs/CHM_PROTOCOL_AUTHORITY.yaml",
        "configs/CHM_FROZEN_INPUT_HASHES_V2.yaml",
    ]:
        raise ProtocolAuthorityFailure("CHM-D required configuration hash-validation set is ambiguous.")


def validate_historical_nonexecutability(authority: dict[str, Any]) -> None:
    """Verify preserved historical files exist but are excluded from execution."""
    normative = set(authority["normative_executable"])
    historical = set(authority["historical_non_executable"])
    required = {
        "configs/PREREGISTERED_CHM_PROTOCOL_DRAFT.yaml",
        "configs/PREREGISTERED_CHM_PROTOCOL_CANDIDATE_V1.yaml",
        "configs/PREREGISTERED_CHM_PROTOCOL_FINAL.yaml",
    }
    if normative.intersection(historical) or not required.issubset(historical):
        raise ProtocolAuthorityFailure("Historical protocol authority overlap.")
    records = {relative: load_yaml(PHASE / relative) for relative in required}
    if records["configs/PREREGISTERED_CHM_PROTOCOL_CANDIDATE_V1.yaml"].get("execution_authorized"):
        raise ProtocolAuthorityFailure("Candidate V1 remains execution-authorized.")
    if records["configs/PREREGISTERED_CHM_PROTOCOL_FINAL.yaml"].get("execution_authorized"):
        raise ProtocolAuthorityFailure("Final V1 remains execution-authorized.")
    if records["configs/PREREGISTERED_CHM_PROTOCOL_DRAFT.yaml"].get("execution_authority") is True:
        raise ProtocolAuthorityFailure("Draft has execution authority.")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_version") != "FINAL_V2" or protocol.get("status") != "FINAL_V2_FROZEN_AWAITING_CHM_D_EXECUTION":
        raise ProtocolAuthorityFailure("Final V2 protocol/status mismatch.")
    if not protocol.get("execution_authority") or protocol.get("execution_authorized_by_this_phase"):
        raise ProtocolAuthorityFailure("Execution authority/authorization distinction is invalid.")
    profiles = protocol.get("loss_profiles", {}).get("profiles", {})
    if set(profiles) != EXPECTED_PROFILES or protocol["loss_profiles"].get("profile_count") != 3:
        raise ProtocolAuthorityFailure("CHM profile count is not exactly three.")
    if protocol["loss_profiles"].get("27_combination_grid") != "SUPERSEDED_NOT_EXECUTABLE":
        raise ProtocolAuthorityFailure("27-profile grid remains executable.")
    gates = protocol.get("primary_development_qualification", {})
    if gates.get("gate_count") != 5 or set(key for key in gates if key.startswith("Gate_")) != EXPECTED_GATES:
        raise ProtocolAuthorityFailure("Primary qualification gates are not exactly five.")
    s1 = protocol.get("S1_robustness_diagnostic", {})
    if s1.get("role") != "PRESPECIFIED_SECONDARY_ROBUSTNESS_DIAGNOSTIC" or gates.get("S1_is_qualification_gate") is not False:
        raise ProtocolAuthorityFailure("S1 role is not a secondary diagnostic.")
    thresholds = s1.get("diagnostic_reference_thresholds", {})
    if thresholds != {"trajectory_Spearman_minimum": 0.95, "median_absolute_difference_maximum": 0.01, "terminal_absolute_difference_maximum": 0.01}:
        raise ProtocolAuthorityFailure("S1 diagnostic thresholds drifted.")
    if protocol["polarization_reliability"].get("anchor_target_gaussian_perturbation") != "DISABLED":
        raise ProtocolAuthorityFailure("Anchor Gaussian perturbation is enabled.")
    baselines = protocol.get("baselines", {})
    if baselines.get("B0", {}).get("name") != "RAW_P1S_ANCHOR_FUNCTIONAL":
        raise ProtocolAuthorityFailure("B0 definition missing.")
    if baselines.get("B1", {}).get("name") != "ISOTONIC_P1S_CALIBRATION" or baselines["B1"].get("implementation") != "sklearn.isotonic.IsotonicRegression":
        raise ProtocolAuthorityFailure("B1 definition missing.")
    if baselines.get("B2", {}).get("name") != "NONNEGATIVE_AFFINE_P1S_CALIBRATION" or baselines["B2"].get("constraint") != "a_greater_or_equal_zero":
        raise ProtocolAuthorityFailure("B2 definition missing.")
    forbidden = protocol.get("runtime_authority", {})
    if not forbidden.get("direct_raw_reconstruction") == "prohibited" or forbidden.get("direct_polarization_csv_reading") != "prohibited":
        raise ProtocolAuthorityFailure("CHM-D direct source reconstruction remains allowed.")


def validate_frozen_tables(protocol: dict[str, Any]) -> None:
    hou = pd.read_csv(INPUTS / "CHM_HOU_TABLE.csv")
    targets = pd.read_csv(INPUTS / "CHM_SPARSE_ANCHOR_TARGETS.csv")
    mapping = pd.read_csv(INPUTS / "CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv")
    inputs = protocol["frozen_runtime_inputs"]
    if sha256_file(INPUTS / "CHM_HOU_TABLE.csv") != inputs["HOU_table"]["sha256"] or len(hou) != 152:
        raise ProtocolAuthorityFailure("Frozen HOU hash/count mismatch.")
    if int(hou["hou_kind"].eq("COMPLETE_CYCLE_HOU").sum()) != 146 or int(hou["hou_kind"].eq("PREANCHOR_PARTIAL_HOU").sum()) != 6:
        raise ProtocolAuthorityFailure("Frozen HOU kind count mismatch.")
    if sha256_file(INPUTS / "CHM_SPARSE_ANCHOR_TARGETS.csv") != inputs["sparse_anchor_targets"]["sha256"]:
        raise ProtocolAuthorityFailure("Frozen sparse-target hash mismatch.")
    if set(targets["checkpoint"].astype(int)) != {300, 600, 850, 1150, 1500, 1750} or len(targets) != 6:
        raise ProtocolAuthorityFailure("Frozen sparse target table mismatch.")
    if sha256_file(INPUTS / "CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv") != inputs["frozen_anchor_interval_mapping"]["sha256"] or len(mapping) != 18:
        raise ProtocolAuthorityFailure("Frozen anchor mapping hash/count mismatch.")


def run_authority_audit() -> dict[str, Any]:
    """Run all protocol-authority checks without a training or Test code path."""
    protocol_hash = assert_sha_sidecar(FINAL_V2, FINAL_V2_SHA)
    supplement_a_hash = assert_sha_sidecar(SUPPLEMENT_A, SUPPLEMENT_A_SHA)
    errata_hash = assert_sha_sidecar(ERRATA, ERRATA_SHA)
    authority = load_yaml(AUTHORITY)
    protocol = load_yaml(FINAL_V2)
    validate_authority(authority)
    validate_historical_nonexecutability(authority)
    validate_protocol(protocol)
    validate_frozen_tables(protocol)
    verify_frozen_input_hashes_v2()
    checks = {
        "execution_protocol_is_FINAL_V2_with_Supplement_A": True,
        "draft_not_executable": True,
        "candidate_v1_not_executable": True,
        "FINAL_V1_superseded_preserved": True,
        "profiles_exactly_3": True,
        "primary_gate_count_exactly_5": True,
        "S1_not_primary_qualification_gate": True,
        "S1_diagnostic_thresholds_present": True,
        "anchor_Gaussian_perturbation_disabled": True,
        "B0_exact_definition_present": True,
        "B1_exact_definition_present": True,
        "B2_exact_definition_present": True,
        "HOU_SHA_matches": True,
        "sparse_anchor_target_SHA_matches": True,
        "Test_not_in_allowed_training_inputs": True,
        "raw_Loadcycle_not_in_CHM_D_runtime_inputs": True,
        "direct_polarization_CSV_not_in_CHM_D_runtime_inputs": True,
        "Supplement_A_hash_matches": True,
    }
    result = {
        "phase": PHASE.name,
        "base_commit": "002936e0d995d7cc1db5e8edffca35b612f271b5",
        "status": "FINAL_V2A_FROZEN_READY_FOR_CHM_D",
        "training_executed": False,
        "LOAO_executed": False,
        "bootstrap_model_fitting_executed": False,
        "MAE_RMSE_generated": False,
        "Test_accessed": False,
        "checks": checks,
        "protocol_authority_audit": "PASS",
        "final_v2_sha256": protocol_hash,
        "execution_semantics_supplement_a_sha256": supplement_a_hash,
        "errata_v2_sha256": errata_hash,
        "remaining_protocol_ambiguity": "NONE",
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with AUDIT_OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def print_summary(result: dict[str, Any]) -> None:
    print("Phase CHM Final V2A Authority Freeze")
    print("===================================")
    print(f"Base: {result['base_commit']}")
    print("Training executed: NO")
    print("Authoritative protocol: FINAL_V2 + EXECUTION_SUPPLEMENT_A")
    print("Historical FINAL_V1 preserved: YES")
    print("Execution sources ambiguous: NO")
    print("Loss profiles: 3")
    print("Primary qualification gates: 5")
    print("S1 role: SECONDARY_ROBUSTNESS_DIAGNOSTIC")
    print("S1 is qualification gate: NO")
    print("Anchor Gaussian perturbation: DISABLED")
    print("B0: RAW_P1S_ANCHOR_FUNCTIONAL")
    print("B1: ISOTONIC_P1S_CALIBRATION")
    print("B2: NONNEGATIVE_AFFINE_P1S_CALIBRATION")
    print("Complete HOU: 146")
    print("Partial HOU: 6")
    print("Frozen sparse anchors: 6")
    print("CHM-D raw-data reading allowed: NO")
    print("CHM-D direct polarization CSV reading allowed: NO")
    print(f"Protocol authority audit: {result['protocol_authority_audit']}")
    print(f"Final V2 hash: {result['final_v2_sha256']}")
    print(f"Supplement A hash: {result['execution_semantics_supplement_a_sha256']}")
    print(f"Errata hash: {result['errata_v2_sha256']}")
    print(f"Remaining protocol ambiguity: {result['remaining_protocol_ambiguity']}")
    print(f"Final status: {result['status']}")
    print("Next action: CHM_D_EXECUTION")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-hashes", action="store_true")
    args = parser.parse_args()
    if args.write_hashes:
        print(json.dumps(write_frozen_input_hashes_v2(), indent=2, ensure_ascii=False))
    else:
        print_summary(run_authority_audit())

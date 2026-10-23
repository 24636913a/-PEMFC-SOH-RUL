"""Build the only Development-only polarization inputs allowed for PSDM.

This script has an intentionally closed source allowlist.  It never searches
the N1 tree and never opens a mixed polarization table.  The one measurement
source is pinned by its Git blob SHA before parsing.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
INPUT = PHASE / "inputs"
CONFIG = PHASE / "configs"
SOURCE = PROJECT / "Phase_RUL90_Data_Feasibility" / "outputs" / "polarization_points_development.csv"
CHM = PROJECT / "Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model" / "inputs" / "CHM_SPARSE_ANCHOR_TARGETS.csv"
PAA = PROJECT / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs" / "P1S_ANCHOR_ALIGNMENT.csv"
EXPECTED_BLOB = "1da6d582d92e50576072f53575725e6e2d57c66f"
BASE_COMMIT = "7bd758ce9553850fa808a3a994532c1e9e4c5a72"
STAGES = ("BoL", "300h", "600h", "850h", "1150h", "1500h", "1750h")
CHECKPOINTS = (300, 600, 850, 1150, 1500, 1750)
JSTARS = (1.0, 1.6)
REFS = {1.0: 0.6989533066, 1.6: 0.6373789965}
VREF_P1S = 0.71383023635


class AnchorAuthorityFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", "--", str(path.relative_to(REPO))], cwd=REPO, text=True).strip()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def canonical_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip() in {"True", "False"}:
        return value.strip() == "True"
    raise AnchorAuthorityFailure(f"INVALID_AFTER_AIR_AIR_BOOLEAN:{value!r}")


def validate_anchor_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"filename", "stage", "after_air_air_start", "target_jstar_A_cm2", "active_voltage_median_V", "valid_sample_count", "selection_scope"}
    if required - set(raw.columns):
        raise AnchorAuthorityFailure("POLARIZATION_SOURCE_SCHEMA_MISMATCH")
    if len(raw) != 40:
        raise AnchorAuthorityFailure(f"POLARIZATION_SOURCE_ROW_COUNT_UNEXPECTED:{len(raw)}")
    parsed = raw.copy()
    parsed["after_air_air_start"] = parsed["after_air_air_start"].map(canonical_bool)
    if not set(parsed.stage.astype(str)).issubset(set(STAGES) | {"1500h_afterAirAirStart"}):
        raise AnchorAuthorityFailure("UNEXPECTED_POLARIZATION_STAGE")
    if not parsed.selection_scope.astype(str).eq("development-only").all():
        raise AnchorAuthorityFailure("NON_DEVELOPMENT_SELECTION_SCOPE")
    selected = parsed.loc[
        parsed.stage.astype(str).isin(STAGES)
        & ~parsed.after_air_air_start
        & parsed.target_jstar_A_cm2.astype(float).isin(JSTARS)
        & parsed.selection_scope.astype(str).eq("development-only")
    ].copy()
    if selected.duplicated(["stage", "target_jstar_A_cm2"]).any() or len(selected) != 14:
        raise AnchorAuthorityFailure("ANCHOR_STAGE_JSTAR_DUPLICATE_OR_COUNT_FAILURE")
    if not np.isfinite(selected.active_voltage_median_V.to_numpy(float)).all() or not (selected.active_voltage_median_V.to_numpy(float) > 0).all():
        raise AnchorAuthorityFailure("INVALID_POLARIZATION_VOLTAGE")
    refs = selected.loc[selected.stage.eq("BoL")].copy()
    anchors = selected.loc[~selected.stage.eq("BoL")].copy()
    if len(refs) != 2 or len(anchors) != 12:
        raise AnchorAuthorityFailure("REFERENCE_OR_ANCHOR_COUNT_FAILURE")
    if set(refs.target_jstar_A_cm2.astype(float)) != set(JSTARS):
        raise AnchorAuthorityFailure("BOL_REFERENCE_JSTAR_FAILURE")
    expected_pairs = {(f"{checkpoint}h", j) for checkpoint in CHECKPOINTS for j in JSTARS}
    observed_pairs = set(zip(anchors.stage.astype(str), anchors.target_jstar_A_cm2.astype(float)))
    if observed_pairs != expected_pairs:
        raise AnchorAuthorityFailure("DEVELOPMENT_ANCHOR_GRID_FAILURE")
    for jstar, reference in REFS.items():
        observed = float(refs.loc[np.isclose(refs.target_jstar_A_cm2.astype(float), jstar), "active_voltage_median_V"].iloc[0])
        if not np.isclose(observed, reference, atol=1e-10, rtol=0.0):
            raise AnchorAuthorityFailure(f"BOL_REFERENCE_VOLTAGE_MISMATCH_J{jstar}")
    return refs, anchors


def frozen_protocol(source_hash: str, source_blob: str, chm_hash: str, paa_hash: str, anchor_hash: str, reference_hash: str) -> dict[str, Any]:
    return {
        "phase": "Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility",
        "status": "FROZEN_BEFORE_PSDM_RESULTS",
        "base_commit": BASE_COMMIT,
        "scope": {"development_only": True, "test_access_prohibited": True, "raw_F2_modeling": False, "TimesNet_training": False, "EMD": False, "VMD": False, "SG": False, "PINN": False, "RUL": False},
        "historical_psdm_attempt": {"mixed_n1_test_rows_accessed": True, "model_training_performed": False, "results_generated": False, "git_changed": False},
        "stage_A_anchor_authority": {"exact_source": str(SOURCE.relative_to(REPO)).replace("\\", "/"), "expected_git_blob_sha": EXPECTED_BLOB, "observed_git_blob_sha": source_blob, "source_sha256": source_hash, "frozen_anchor_table_sha256": anchor_hash, "frozen_reference_table_sha256": reference_hash, "allowed_stages": list(STAGES), "allowed_jstar_A_cm2": list(JSTARS), "after_air_air_start": False, "selection_scope": "development-only", "BoL_reference_voltage_V": {"j1.0": REFS[1.0], "j1.6": REFS[1.6]}, "cross_checks": {"CHM_sparse_anchor_targets_sha256": chm_hash, "PAA_anchor_alignment_sha256": paa_hash}},
        "P1S_observation": {"source": "Phase_CTAE_P1S_Reference_Availability_Erratum/outputs/P1S_REFERENCE_AVAILABILITY_AUDIT.csv", "criterion": "causal_target_eligible == true", "expected_count": 131, "reference_voltage_V": VREF_P1S, "identity_source": "Phase_PHI_P1S_B2_Health_Index_Consolidation/outputs/P1S_B2_FULL_HEALTH_TRAJECTORY.csv"},
        "evaluation_authorities": {"H3": "Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/inputs/H3_FROZEN_SAMPLE_AUTHORITY.csv", "H6": "Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/inputs/H6_FROZEN_SAMPLE_AUTHORITY.csv", "H6_primary": True},
        "models": {"persistence": "Rhat_target=R_origin", "gaussian_persistence": "mu=R_origin; sigma=std(outer_train_DeltaR_H,ddof=1), floor=1e-8", "ULLT": {"observation": "R=H+epsilon", "transition": "H_next=H+s; s_next=s+eta", "state": ["H", "s"], "Q": "diag(0,q_s)", "parameters": ["log_q_s", "log_r", "s0"]}, "PSDM": {"observation": "R=H+epsilon", "transition": "H_next=H-softplus(z); z_next=z+eta", "state": ["H", "z"], "Q": "diag(0,q_z)", "parameters": ["log_q_z", "log_r", "z0"], "jacobian": "[[1,-sigmoid(z)],[0,1]]", "posterior": "Gaussian_EKF_approximation"}},
        "parameter_fit": {"optimizer": "scipy.optimize.minimize L-BFGS-B", "log_q_log_r_bounds": [-25, -5], "ULLT_s0_bounds": [-0.01, 0.01], "PSDM_softplus_z0_bounds": [1e-8, 0.01], "initial_state": "H0=first_outer_train_R; P0=diag(r,max(q,1e-8))", "deterministic_training_only_multistarts": 4, "validation_fit_prohibited": True},
        "walk_forward": {"fixed_parameters_after_outer_train_fit": True, "past_validation_P1S_assimilation_allowed": True, "future_P1S_for_current_origin_prohibited": True, "raw_cycle_propagation_exact": True},
        "metrics": {"point": ["MAE_R", "RMSE_R", "MAE_mV_equivalent", "RMSE_mV_equivalent", "R2", "Spearman", "P95_absolute_error_mV_equivalent", "max_absolute_error_mV_equivalent"], "probabilistic": ["Gaussian_NLL", "Gaussian_CRPS", "coverage_80", "coverage_95", "mean_width_80", "mean_width_95"], "aggregation": "equal_fold_mean_primary; pooled_descriptive_only"},
        "gates": {"A": "PSDM_H6_equal_fold_MAE_mV <= 1.05*Persistence_H6_equal_fold_MAE_mV", "B": "PSDM_H6_equal_fold_CRPS <= 0.95*GaussianPersistence_H6_equal_fold_CRPS", "C": "0.85 <= pooled_H6_coverage_95 <= 1.00 and finite_mean_width_95", "D": "0.65 <= pooled_H6_coverage_80 <= 0.95", "E": "PSDM_H6_fold3_CRPS <= 1.10*GaussianPersistence_H6_fold3_CRPS"},
        "stop_rules": {"stage_A_failure_prevents_stage_B": True, "no_posthoc_parameter_search": True, "no_automatic_second_round": True, "no_RUL_or_threshold_definition": True},
    }


def run() -> dict[str, Any]:
    INPUT.mkdir(parents=True, exist_ok=True); CONFIG.mkdir(parents=True, exist_ok=True)
    if git_blob(SOURCE) != EXPECTED_BLOB:
        raise AnchorAuthorityFailure("DEVELOPMENT_POLARIZATION_BLOB_MISMATCH")
    raw = pd.read_csv(SOURCE)
    refs, anchors = validate_anchor_frame(raw)
    source_blob = git_blob(SOURCE); source_hash = sha256_file(SOURCE)
    chm = pd.read_csv(CHM); paa = pd.read_csv(PAA)
    required_chm = {"checkpoint", "Y"}; required_paa = {"record_type", "checkpoint_h", "precharacterization_cutoff_h", "P1S_anchor_retention"}
    if required_chm - set(chm) or required_paa - set(paa):
        raise AnchorAuthorityFailure("CROSSCHECK_SOURCE_SCHEMA_MISMATCH")
    chm = chm.loc[chm.checkpoint.astype(int).isin(CHECKPOINTS), ["checkpoint", "Y"]].sort_values("checkpoint")
    paa = paa.loc[(paa.record_type.astype(str) == "anchor_summary") & (paa.checkpoint_h.astype(int).isin(CHECKPOINTS)), ["checkpoint_h", "precharacterization_cutoff_h", "P1S_anchor_retention"]].sort_values("checkpoint_h")
    if len(chm) != 6 or chm.checkpoint.duplicated().any() or len(paa) != 6 or paa.checkpoint_h.duplicated().any():
        raise AnchorAuthorityFailure("CROSSCHECK_CHECKPOINT_GRID_FAILURE")
    ref_map = refs.set_index(refs.target_jstar_A_cm2.astype(float))["active_voltage_median_V"].to_dict()
    anchors["checkpoint_h"] = anchors.stage.str.replace("h", "", regex=False).astype(int)
    anchors["jstar_A_cm2"] = anchors.target_jstar_A_cm2.astype(float)
    anchors["BoL_reference_voltage_V"] = anchors.jstar_A_cm2.map(ref_map).astype(float)
    anchors["polarization_voltage_V"] = anchors.active_voltage_median_V.astype(float)
    anchors["polarization_retention"] = anchors.polarization_voltage_V / anchors.BoL_reference_voltage_V
    anchors["source_stage"] = anchors.stage.astype(str); anchors["source_filename"] = anchors.filename.astype(str); anchors["source_valid_sample_count"] = anchors.valid_sample_count.astype(int)
    freeze = anchors[["checkpoint_h", "jstar_A_cm2", "polarization_voltage_V", "BoL_reference_voltage_V", "polarization_retention", "source_stage", "source_filename", "source_valid_sample_count", "selection_scope"]].sort_values(["checkpoint_h", "jstar_A_cm2"]).reset_index(drop=True)
    j1 = freeze.loc[np.isclose(freeze.jstar_A_cm2, 1.0), ["checkpoint_h", "polarization_retention"]].merge(chm, left_on="checkpoint_h", right_on="checkpoint", validate="one_to_one")
    if not np.all(np.abs(j1.polarization_retention.to_numpy(float) - j1.Y.to_numpy(float)) < 1e-5):
        raise AnchorAuthorityFailure("CHM_J1_CROSSCHECK_MISMATCH")
    if set(freeze.checkpoint_h.astype(int)) != set(paa.checkpoint_h.astype(int)):
        raise AnchorAuthorityFailure("PAA_CHECKPOINT_ALIGNMENT_MISMATCH")
    refs_out = refs.assign(jstar_A_cm2=refs.target_jstar_A_cm2.astype(float), BoL_reference_voltage_V=refs.active_voltage_median_V.astype(float), source_stage=refs.stage.astype(str), source_filename=refs.filename.astype(str), source_valid_sample_count=refs.valid_sample_count.astype(int))[ ["jstar_A_cm2", "BoL_reference_voltage_V", "source_stage", "source_filename", "source_valid_sample_count", "selection_scope"]].sort_values("jstar_A_cm2")
    anchor_path = INPUT / "DEV_POLARIZATION_ANCHORS_J1_J16_FROZEN.csv"; reference_path = INPUT / "DEV_POLARIZATION_REFERENCE_FROZEN.csv"
    write_csv(freeze, anchor_path); write_csv(refs_out, reference_path)
    anchor_hash, reference_hash = sha256_file(anchor_path), sha256_file(reference_path)
    protocol = frozen_protocol(source_hash, source_blob, sha256_file(CHM), sha256_file(PAA), anchor_hash, reference_hash)
    protocol_path = CONFIG / "PSDM_FEASIBILITY_FROZEN_PROTOCOL.yaml"; protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8")
    protocol_sha = sha256_file(protocol_path); (CONFIG / "PSDM_FEASIBILITY_FROZEN_PROTOCOL.sha256").write_text(f"{protocol_sha}  PSDM_FEASIBILITY_FROZEN_PROTOCOL.yaml\n", encoding="utf-8")
    chm_difference = np.abs(j1.polarization_retention.to_numpy(float) - j1.Y.to_numpy(float))
    manifest = {
        "stage": "A", "exact_polarization_source": str(SOURCE.relative_to(REPO)).replace("\\", "/"),
        "expected_git_blob_sha": EXPECTED_BLOB, "observed_git_blob_sha": source_blob, "actual_source_sha256": source_hash,
        "frozen_anchor_table_sha256": anchor_hash, "frozen_reference_table_sha256": reference_hash,
        "source_record_count": int(len(raw)), "frozen_anchor_count": int(len(freeze)), "frozen_reference_count": int(len(refs_out)),
        "allowed_stages": list(STAGES), "allowed_jstar_A_cm2": list(JSTARS),
        "excluded_stage_examples": ["1500h_afterAirAirStart", "2000h", "2300h"],
        "CHM_j1_crosscheck_max_abs_difference": float(np.max(chm_difference)), "PAA_checkpoint_alignment_pass": True,
        "historical_mixed_n1_access": True, "current_stage_A_test_accessed": False,
        "current_stage_A_late_life_target_accessed": False,
    }
    (INPUT / "DEV_ANCHOR_SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"protocol_sha": protocol_sha, "anchor_rows": len(freeze), "reference_rows": len(refs_out), "source_blob": source_blob}


if __name__ == "__main__":
    try:
        print(json.dumps(run(), ensure_ascii=False))
    except AnchorAuthorityFailure as exc:
        print(json.dumps({"status": "STAGE_A_FAILED", "reason": str(exc)}, ensure_ascii=False))
        raise

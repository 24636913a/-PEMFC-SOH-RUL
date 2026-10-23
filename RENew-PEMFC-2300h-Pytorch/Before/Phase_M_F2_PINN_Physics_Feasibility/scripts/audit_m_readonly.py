"""Metadata-only Phase M audit. Never import a historical model or runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
PREFIX = "RENew-PEMFC-2300h-Pytorch/"
BASE = "82ee5ad35ada7dfe1e7a7c0add63de30fee3af43"
STAGE = PREFIX + PHASE.name + "/"
SOURCES = (
    "Phase_TFT_FullFeature_CausalFilter_Comparison/reports/E_SERIES_EXPERIMENT_REPORT.md",
    "Phase_TFT_FullFeature_CausalFilter_Comparison/configs/PREREGISTERED_E_COMPARISON.yaml",
    "Phase_TFT_FullFeature_CausalFilter_Comparison/outputs/E_COMMON_SCALER.json",
    "Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py",
    "Phase_TFT_SingleSplit_Feature_Filter_Ablation/scripts/run_c_ablation.py",
    "Phase_TFT_Original_Baseline/models/canonical_tft.py",
    "Phase_F_TFT_DualPath_RegimeFusion/reports/F_FINAL_EXPERIMENT_SUMMARY.md",
    "Phase_F_TFT_DualPath_RegimeFusion/configs/F0_F2_EXECUTION_PROTOCOL.yaml",
    "Phase_F_TFT_DualPath_RegimeFusion/models/causal_multiscale_branch.py",
    "Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py",
    "Phase_F_TFT_DualPath_RegimeFusion/models/regime_aware_dualpath_tft.py",
    "Phase_F_TFT_DualPath_RegimeFusion/models/frozen_feature_schema.py",
    "Phase_H_Determinism_MultiSeed_Stability/reports/H_FINAL_CONCLUSION.md",
    "Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml",
    "Phase_IR_Validation_Reload_Parity_Audit/reports/IR_FINAL_RECOMMENDATION.md",
    "Phase_I2_Frozen_Assessment_Evaluation/reports/I2_FINAL_SCIENTIFIC_CONCLUSION.md",
    "Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_FINAL_ARCHITECTURE_DECISION.md",
    "Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_FINAL_RESEARCH_RECOMMENDATION.md",
    "Phase_L_Test_Eligibility_and_Reproducibility_Freeze/reports/L_FINAL_EVALUATION_READINESS_DECISION.md",
    "Phase_L_Test_Eligibility_and_Reproducibility_Freeze/manifests/L_FROZEN_MODEL_MANIFEST.json",
    "Phase_POT_Physics_Optimized_TimesNet/reports/STAGE0_PHYSICAL_FEASIBILITY_REPORT.md",
    "Phase_G_STOP_Physics_Consistent_Future_Conditions/PHYSICAL_RELATION_AUDIT.md",
    "Phase_G_STOP_Physics_Consistent_Future_Conditions/FUTURE_CONDITION_FORECAST_AUDIT.md",
    "Phase_HA_EMD_Health_Representation_Audit/reports/FINAL_HEALTH_REPRESENTATION_AUDIT.md",
    "Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_feature_dictionary.csv",
    "Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_feature_availability.csv",
    "Phase_N1_Data_Foundation/scripts/phase_n1_build_foundation.py",
    "Phase_N1_Data_Foundation/configs/phase_n1_config.yaml",
)
MODEL_PATHS = tuple(
    PREFIX + f"Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-{model}-Seed{seed}_best.pt"
    for model in ("F0", "F1", "F2") for seed in (42, 52, 62)
)
ALLOWED = frozenset(PREFIX + name for name in SOURCES) | frozenset(MODEL_PATHS)
FORBIDDEN_BASENAMES = {"causal_features_30min.csv", "causal_features_1h.csv"}
MANIFEST = PHASE / "evidence/M_SOURCE_IDENTITY_MANIFEST.json"
FIELDS = (
    "candidate_id", "classification", "equation", "literature_source", "variables",
    "units", "unknown_parameters", "data_source", "applicable_conditions",
    "train_computability", "inference_computability", "future_actual_required",
    "extra_state_required", "nontrivial_output_constraint", "parameter_gradient",
    "current_validation_evidence",
)


def allowed_path(root: Path, relative: str, allowed=ALLOWED) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("PATH_ESCAPE")
    canonical = path.relative_to(root.resolve()).as_posix()
    if canonical not in allowed or path.name in FORBIDDEN_BASENAMES:
        raise RuntimeError("SOURCE_NOT_ALLOWED")
    return path


def verify_digest(payload: bytes, expected: str) -> str:
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise RuntimeError("SOURCE_HASH_MISMATCH")
    return observed


def source_bytes(relative: str) -> bytes:
    return allowed_path(ROOT, relative).read_bytes()


def source_digest(relative: str) -> str:
    return hashlib.sha256(source_bytes(relative)).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "core.quotepath=false", *args], cwd=ROOT
    ).decode("utf-8").strip()


def write_json(relative: str, payload) -> None:
    path = (PHASE / relative).resolve()
    if not path.is_relative_to(PHASE):
        raise RuntimeError("WRITE_OUTSIDE_PHASE_M")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def candidate(cid, grade, equation, variables, units, unknowns, source, conditions,
              train, inference, future, state, output, gradient, evidence):
    values = (cid, grade, equation, "DOE Handbook §§2.1–2.6; project mathematical audit",
              variables, units, unknowns, source, conditions, train, inference,
              future, state, output, gradient, evidence)
    return dict(zip(FIELDS, values, strict=True))


CANDIDATES = [
    candidate("A_FUTURE_POLARIZATION", "FUTURE_INFORMATION_REQUIRED",
              "V=Erev-eta_act-j*ASR-eta_conc", "V,j,T,aH2,aO2,aH2O",
              "V; A/cm2; K; dimensionless activities; ohm*cm2", "j0,alpha,ASR,jlim,membrane water",
              "N1 metadata only; future actual prohibited", "same configuration; validated quasi-steady domain",
              "NOT_CLOSED", "NOT_COMPUTABLE_FROM_ORIGIN", "YES", "YES",
              "conditional YES if context closed", "dresidual/dDelta_std=sV", "no joint identification; POT information failure"),
    candidate("A_HISTORICAL_POLARIZATION", "CONDITIONALLY_ADMISSIBLE",
              "Vpast-f(jpast,upast,parameters)", "V,j,T,p,RH,interface state", "V; A/cm2; K; absolute pressure",
              "j0,alpha,ASR,jlim,interface activities", "authorized future Train evidence needed; none read in M",
              "steady dwell and independently verified parameters", "EVIDENCE_REQUIRED", "not a future forecast residual",
              "NO", "YES", "auxiliary only if separately defined", "conditional via auxiliary head/encoder",
              "schema existence only; no numerical validation in M"),
    candidate("B_INPUT_CURRENT_IDENTITY", "NOT_ADMISSIBLE", "j-I/A=0", "I,j,A", "A; A/cm2; cm2",
              "A historically estimated", "G-STOP physical relation report", "same raw/bin population and units",
              "identity check possible; not voltage loss", "historical check possible", "NO", "NO", "NO",
              "ZERO with respect to voltage output", "historical area 280.001183 cm2; mean I masked in F2"),
    candidate("C_FUTURE_PAIR_ORDERING", "FUTURE_INFORMATION_REQUIRED", "max(0,Vhat_highj-Vhat_lowj)",
              "future predicted V; actual future j,T,RH,p", "V; A/cm2; K; humidity; pressure", "unobserved state equivalence",
              "POT frozen candidate report", "matched conditions and stable state", "not authorized as future pair selector",
              "future context unknown", "YES", "YES", "YES if legal pairs existed", "piecewise nonzero",
              "9452 historical pairs do not repair information gate"),
    candidate("C_HISTORICAL_AUX_ORDERING", "CONDITIONALLY_ADMISSIBLE", "max(0,q_highj-q_lowj)^2",
              "historical auxiliary V; past j,T,RH,p,state", "V2 or Train-normalized dimensionless loss",
              "material state and quasi-steady equivalence", "future authorized RealHyFC Train evidence; no real rows read",
              "joint support; stable dwell; auxiliary measurement isolation; anti-collapse supervision",
              "ONLY_AFTER_EVIDENCE", "auxiliary head absent from future inference", "NO", "YES",
              "indirect encoder constraint; no proven forecast benefit", "conditional nonzero via new auxiliary head",
              "theory reasonable; joint support/validity not established; preferred P3 only"),
    candidate("D_THERMODYNAMIC_BOUND", "FUTURE_INFORMATION_REQUIRED", "Vhat<=Erev(T,a)", "V,T,reactant activities",
              "V; K; dimensionless", "future interface state", "DOE theory; no project future-state closure",
              "normal fuel-cell domain; known reversible state", "conditional if state known", "not origin-computable",
              "YES", "YES", "YES if bound known and active", "piecewise nonzero", "no validated per-step bound"),
    candidate("D_ENGINEERING_RANGE", "PHYSICS_INSPIRED_ONLY", "max(0,Vhat-U)+max(0,L-Vhat)", "Vhat,L,U",
              "V", "valid work-domain bounds not frozen", "future engineering evidence or Train-only empirical quantiles",
              "explicit equipment/work domain; not universal electrochemical law", "conditional numerical calculation",
              "conditional if bounds valid", "NO", "NO", "weak; zero within wide interval", "nonzero only outside bounds",
              "not registered or deployed; Train extrema are not physical laws"),
    candidate("E_OBSERVED_TIME_MONOTONIC", "NOT_ADMISSIBLE", "dV/dt<=0", "observed/predicted V,t", "V/h",
              "load and reversible-state effects omitted", "HA and dynamic task semantics", "not applicable across dynamic conditions",
              "calculable but scientifically invalid", "invalid dynamic constraint", "NO", "YES",
              "nontrivial but wrong restriction", "possible does not imply validity", "recovery/load confounding; no health proof"),
    candidate("E_IRREVERSIBLE_LATENT", "NOT_IDENTIFIABLE", "V=g(H,u,w)+epsilon; dH/dt<=0 only after definition",
              "H,u,w,V,t", "V; H units undefined; H/h", "H definition,observation map,reversible dynamics",
              "HA report; no new health data", "reference normalization and identified state separation",
              "NOT_IDENTIFIABLE", "NOT_IDENTIFIABLE", "potentially YES for voltage map", "YES",
              "not closed; no SOH labels constructed", "hypothetical; not evidence", "HEALTH_EVIDENCE_NOT_IDENTIFIABLE"),
]


def validate_candidates(rows) -> None:
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise RuntimeError("DUPLICATE_CANDIDATE")
    allowed_grades = {"PHYSICS_EQUATION_ADMISSIBLE", "CONDITIONALLY_ADMISSIBLE",
                      "PHYSICS_INSPIRED_ONLY", "NOT_IDENTIFIABLE",
                      "FUTURE_INFORMATION_REQUIRED", "NOT_ADMISSIBLE"}
    for row in rows:
        if set(row) != set(FIELDS) or any(not str(row[key]).strip() for key in FIELDS):
            raise RuntimeError("INCOMPLETE_FOURTEEN_FIELD_PROOF")
        if row["classification"] not in allowed_grades:
            raise RuntimeError("UNKNOWN_CLASSIFICATION")
        if row["classification"] == "PHYSICS_EQUATION_ADMISSIBLE":
            raise RuntimeError("NO_FULLY_VALIDATED_CONSTRAINT_IN_THIS_PHASE")


def audit() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["base_commit"] != BASE or set(manifest["sources"]) != set(ALLOWED):
        raise RuntimeError("SOURCE_MANIFEST_SET_MISMATCH")
    if git("merge-base", "--is-ancestor", BASE, "HEAD") != "":
        raise RuntimeError("BASE_NOT_ANCESTOR")
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if code_hash != manifest["audit_script_sha256"]:
        raise RuntimeError("AUDIT_CODE_IDENTITY_CHANGED")
    before = {}
    for relative, info in manifest["sources"].items():
        before[relative] = verify_digest(source_bytes(relative), info["sha256"])
    frozen = json.loads(source_bytes(PREFIX + SOURCES[19]))
    if len(frozen["models"]) != 9 or set(m["path"] for m in frozen["models"]) != set(MODEL_PATHS):
        raise RuntimeError("FROZEN_MODEL_SET_MISMATCH")
    for model in frozen["models"]:
        if before[model["path"]] != model["sha256"]:
            raise RuntimeError("FROZEN_CHECKPOINT_IDENTITY_MISMATCH")
    validate_candidates(CANDIDATES)
    result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                            cwd=PHASE, text=True, encoding="utf-8", capture_output=True)
    (PHASE / "outputs/M_SYNTHETIC_TEST_LOG.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("SYNTHETIC_TEST_FAILURE")
    after = {relative: source_digest(relative) for relative in ALLOWED}
    if before != after:
        raise RuntimeError("READONLY_SOURCE_CHANGED")
    changed = git("diff", "--name-only", BASE).splitlines()
    if any(not path.startswith(STAGE) for path in changed):
        raise RuntimeError("CHANGE_OUTSIDE_PHASE_M")
    pending = git("ls-files", "--others", "--exclude-standard").splitlines()
    if any(not path.startswith(STAGE) for path in pending):
        raise RuntimeError("UNTRACKED_OUTSIDE_PHASE_M")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(CANDIDATES)
    (PHASE / "outputs/M_PHYSICS_CANDIDATE_MATRIX.csv").write_text(stream.getvalue(), encoding="utf-8")
    return {
        "phase": PHASE.name, "base_commit": BASE, "execution_head": git("rev-parse", "HEAD"),
        "checked_utc": datetime.now(timezone.utc).isoformat(), "technical_audit_status": "PASS",
        "scientific_status": "NO_ADMISSIBLE_PINN_CONSTRAINT_YET", "preferred_route": "P3_CONDITIONAL_HISTORY_AUXILIARY",
        "source_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "audit_script_sha256": code_hash, "source_count": len(ALLOWED), "checkpoint_count": 9,
        "all_source_hashes_match": True, "all_source_hashes_unchanged_after_audit": True,
        "checkpoint_operation": "FILE_BYTE_HASH_ONLY_NO_DESERIALIZATION",
        "model_loaded": False, "model_forward": False, "new_training": False,
        "f2_modified": False, "hyperparameter_search": False,
        "local_real_timeseries_accessed": False, "real_development_rows_read": 0,
        "test_numeric_accessed": False, "mixed_source_full_read_or_hash": False,
        "new_soh_rul_labels": False, "physics_loss_deployed": False, "automatic_next_experiment": False,
        "only_realhyfc": True, "candidate_count": len(CANDIDATES), "candidate_proof_field_count": 14,
        "synthetic_tests_exit_code": result.returncode, "synthetic_test_log": "outputs/M_SYNTHETIC_TEST_LOG.txt",
        "test_result_summary": result.stderr.strip().splitlines()[-3:],
        "audit_scope": "Explicit whitelist broker + selected source/checkpoint identities + Git write-scope; not OS-wide historical I/O proof",
        "safety_basis": "No data/model library imported; selected CSVs are dictionaries only; prior reports are Development/static provenance",
        "inherit": ["PHASE_J_F2_ARCHITECTURE_FROZEN", "PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED",
                    "PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE", "PHYSICS_LOSS_NOT_ESTABLISHED",
                    "HEALTH_EVIDENCE_NOT_IDENTIFIABLE"],
        "stopped": True, "next_training_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-sources", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    (PHASE / "outputs").mkdir(exist_ok=True)
    if args.freeze_sources:
        if MANIFEST.exists():
            raise RuntimeError("SOURCE_MANIFEST_ALREADY_FROZEN")
        if git("rev-parse", "HEAD") != BASE:
            raise RuntimeError("WRONG_INITIAL_BASE")
        records = {}
        for path in sorted(ALLOWED):
            records[path] = {
                "sha256": source_digest(path), "baseline_repository_commit": BASE,
                "last_modified_commit": git("log", "-1", "--format=%H", "--", path),
                "read_class": ("CHECKPOINT_BYTES_ONLY" if path in MODEL_PATHS else
                               "SCHEMA_ONLY_NO_OBSERVATIONS" if path.endswith(".csv") else
                               "STATIC_CODE_CONFIG_OR_SAVED_DEVELOPMENT_REPORT"),
                "runtime_imported": False,
            }
        write_json("evidence/M_SOURCE_IDENTITY_MANIFEST.json", {
            "base_commit": BASE, "sources": records,
            "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "access_contract": "NO_REAL_DATA_NO_MODEL_EXECUTION_EXACT_ALLOWLIST",
        })
    try:
        result = audit()
    except Exception as exc:
        write_json("outputs/M_READONLY_AUDIT.json", {
            "technical_audit_status": "FAIL", "exception": str(exc), "next_training_authorized": False,
        })
        raise
    write_json("outputs/M_READONLY_AUDIT.json", result)
    print("PHASE_M_READONLY_AUDIT_PASS")
    print(result["scientific_status"])
    print("TRAINING_AUTHORIZED=NO")


if __name__ == "__main__":
    main()

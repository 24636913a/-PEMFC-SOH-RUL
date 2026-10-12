"""Only read explicit metadata/code; never import historical model/runners."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

PHASE = Path(__file__).resolve().parents[1]
RESEARCH = PHASE.parent
REPO = RESEARCH.parent
BASE = "42d8b4587be715e5a55b034e6d5351126b8350cd"


def approved_source(relative, allowlist):
    p = Path(relative)
    if relative not in allowlist or p.is_absolute() or ".." in p.parts or p.suffix not in {".md", ".py", ".json", ".yaml"}:
        raise ValueError("N_STATIC_SOURCE_NOT_ALLOWLISTED")
    resolved = (RESEARCH / p).resolve()
    if not resolved.is_relative_to(RESEARCH.resolve()):
        raise ValueError("N_SOURCE_ESCAPE")
    return relative


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def verify_static_source_bytes(local, historical):
    # Git text checkout may convert LF to CRLF. No other normalization is allowed.
    if local.replace(b"\r\n", b"\n") != historical.replace(b"\r\n", b"\n"):
        raise ValueError("HISTORICAL_SOURCE_CONTENT_CHANGED")
    return {"base_blob_byte_equal": local == historical,
            "base_blob_LF_normalized_equal": True,
            "git_base_blob_sha256": digest(historical),
            "comparison_normalization": "CRLF_to_LF_only; preserve all other bytes"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")


def main():
    sys.dont_write_bytecode = True
    protocol_path = PHASE / "configs/N_CONVERGENCE_PROTOCOL.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["base_commit"] != BASE:
        raise ValueError("BASE_CHANGED")
    subprocess.run(["git","merge-base","--is-ancestor",BASE,"HEAD"],cwd=REPO,check=True)
    for field in ("training_enabled","real_data_read_allowed","checkpoint_load_allowed",
                  "physics_parameter_fit_allowed","prediction_allowed","test_access_allowed"):
        if protocol[field] is not False:
            raise ValueError("N_SCOPE_BREACH")
    previous_manifest = PHASE / "evidence/N_SOURCE_MANIFEST.json"
    previous = json.loads(previous_manifest.read_text(encoding="utf-8")) if previous_manifest.exists() else None
    records = []
    contents = {}
    for relative in protocol["source_whitelist"]:
        approved_source(relative, protocol["source_whitelist"])
        path = RESEARCH / relative
        # Closed whitelist contains only small reports, configuration/scaler metadata and code.
        payload = path.read_bytes()
        repo_relative = path.relative_to(REPO).as_posix()
        historical = git("show", BASE+":"+repo_relative)
        comparison = verify_static_source_bytes(payload, historical)
        contents[relative] = payload.decode("utf-8-sig")
        records.append({"path":repo_relative,"sha256":digest(payload),"bytes":len(payload),
                        "source_commit":BASE,"access":"STATIC_REPORT_CONFIG_CODE_ONLY",
                        **comparison})
    if previous and previous["sources"] != records:
        raise ValueError("FROZEN_SOURCE_MANIFEST_CHANGED")
    manifest = {"base_commit":BASE,"sources":records,"real_timeseries_or_weight_bytes_read":False}
    for tag in protocol["inherited_states"]:
        if not any(tag in text for text in contents.values()):
            raise ValueError("INHERITED_STATE_UNPROVEN: "+tag)
    if any(all(g.values()) for g in protocol["gate_evidence"].values()):
        raise ValueError("UNSUPPORTED_QUALIFICATION")
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(PHASE/"tests"),pattern="test_*.py")
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (PHASE/"outputs").mkdir(exist_ok=True)
    (PHASE/"outputs/N_SYNTHETIC_TEST_LOG.txt").write_text(log.getvalue(),encoding="utf-8",newline="\n")
    if not result.wasSuccessful() or result.testsRun < 18:
        raise ValueError("N_SYNTHETIC_TEST_FAILED")
    changed = git("diff", BASE,"--name-only").decode().splitlines()
    prefix = PHASE.relative_to(REPO).as_posix()+"/"
    if any(not p.startswith(prefix) for p in changed):
        raise ValueError("HISTORICAL_REPO_MODIFICATION")
    required = ("N_PHYSICS_METHOD_COMPARISON.md","N_PHYSICS_EQUATION_AND_PARAMETER_AUDIT.md",
                "N_TRAIN_ONLY_PRIVILEGED_INFORMATION_PROTOCOL.md","N_RELATED_WORK_AND_NOVELTY_MATRIX.md",
                "N_PHYSICS_F2_ARCHITECTURE_BLUEPRINT.md","N_FAIR_ABLATION_AND_STOP_RULES.md",
                "N_FINAL_TRAINING_READINESS_DECISION.md")
    for filename in required:
        if not (PHASE/filename).is_file() or (PHASE/filename).stat().st_size < 200:
            raise ValueError("MISSING_N_DELIVERABLE: "+filename)
    # Re-read only the same authorized static sources, to check no audit-side mutation.
    if any(digest((REPO/r["path"]).read_bytes()) != r["sha256"] for r in records):
        raise ValueError("STATIC_SOURCE_MUTATED_DURING_AUDIT")
    write_json(previous_manifest,manifest)
    (PHASE/"configs/N_CONVERGENCE_PROTOCOL.sha256").write_text(digest(protocol_bytes)+"\n",encoding="ascii",newline="\n")
    identity = {p.relative_to(PHASE).as_posix():digest(p.read_bytes()) for p in sorted(PHASE.rglob("*.py"))}
    audit = {"technical_audit":"PASS","scientific_status":protocol["final_status"],
             "execution_head":git("rev-parse","HEAD").decode().strip(),"base_commit":BASE,
             "protocol_sha256":digest(protocol_bytes),"code_sha256":identity,
             "source_count":len(records),"synthetic_tests_run":result.testsRun,
             "synthetic_failures":len(result.failures),"synthetic_errors":len(result.errors),
             "inherited_states":protocol["inherited_states"],"training_qualified":False,
             "test_numeric_content_read":False,"new_real_timeseries_read":False,
             "mixed_source_hash_performed":False,"checkpoints_loaded_or_modified":False,
             "neural_training_or_forward":False,"physical_parameter_fitting":False,
             "audit_access_scope":"closed static whitelist; own files; git metadata/whitelisted base blobs",
             "access_audit_limit":"Not an OS-wide historical I/O forensic claim",
             "command":"python -B "+(PHASE/"scripts/run_n_static_audit.py").relative_to(REPO).as_posix()}
    write_json(PHASE/"outputs/N_STATIC_AND_SYNTHETIC_AUDIT.json",audit)
    write_json(PHASE/"FINAL_DECISION.json",{"status":protocol["final_status"],"training_allowed":False,
               "A_concept":"CONDITIONALLY_ADMISSIBLE","A_current":"PHYSICAL_PARAMETER_NOT_IDENTIFIABLE",
               "B_current":"NOT_ADMISSIBLE","most_promising_concept":"A_cross_fitted_train_only_physics_teacher",
               "next_training_implementation_allowed":False,"P3_paused":True,
               "retain_independent_F2_voltage_paper":True,"test_accessed":False,
               "technical_audit":"PASS","protocol_sha256":digest(protocol_bytes)})
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

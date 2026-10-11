"""Evidence identity audit only. No pandas, NumPy, torch, model import or inference."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
CONFIG_PATH = PHASE / "configs/K_READONLY_ACCESS_PROTOCOL.json"
READ_EVENTS = []

class AccessDenied(RuntimeError):
    pass

def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()

def git(*args, binary=False):
    result = subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode("utf-8").strip()

def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def require_full_read_allowed(relative, cfg, checkpoint_paths=()):
    allowed = {r["path"] for r in cfg["full_read_allowlist"]}
    if relative not in allowed and relative not in checkpoint_paths:
        raise AccessDenied("FULL_CONTENT_READ_NOT_ALLOWED:" + relative)
    candidate = (REPO / relative).resolve()
    candidate.relative_to(REPO.resolve())
    return candidate

def safe_full_hash(relative, cfg, checkpoint_paths=()):
    path = require_full_read_allowed(relative, cfg, checkpoint_paths)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    READ_EVENTS.append({"path": relative, "mode": "allowlisted_file_bytes_identity_only"})
    return h.hexdigest()

def bounded_line(handle, max_bytes=65536):
    # An unbuffered binary handle reads EXACTLY to LF, not a prefetched suffix.
    line = bytearray()
    for _ in range(max_bytes):
        byte = handle.read(1)
        if not byte:
            if not line:
                raise AccessDenied("REQUIRED_PREFIX_RECORD_MISSING")
            return bytes(line)
        line += byte
        if byte == b"\n":
            return bytes(line)
    raise AccessDenied("PREFIX_LINE_TOO_LONG")

def parse_development_anchor_prefix(payload, cfg):
    records = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    rules = cfg["bounded_anchor_source"]
    if len(records) != rules["maximum_data_records"]:
        raise AccessDenied("ANCHOR_PREFIX_RECORD_COUNT_MISMATCH")
    for row in records:
        if row["split"] not in rules["allowed_splits"]:
            raise AccessDenied("NON_DEVELOPMENT_ANCHOR")
        if not float(row["ageing_hour"]) < rules["maximum_age_h_exclusive"]:
            raise AccessDenied("ANCHOR_BOUNDARY_BREACH")
        if float(row["jstar_A_cm2"]) != 1.0:
            raise AccessDenied("PRIMARY_ANCHOR_IDENTITY_MISMATCH")
    nominal_ages = sorted(float(r["ageing_hour"]) for r in records
                          if "afterAirAir" not in r["filename"])
    if nominal_ages != [0., 300., 600., 850., 1150., 1500., 1750.]:
        raise AccessDenied("NOMINAL_CHECKPOINT_UNIVERSE_MISMATCH")
    schema = list(records[0])
    canonical = json.dumps({"columns": schema, "records": records},
                           ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"fingerprint_type": "ALLOWED_DEVELOPMENT_ANCHOR_LOGICAL_PREFIX",
            "row_count": len(records), "nominal_count": len(nominal_ages),
            "column_order": schema, "serialization": "ordered compact JSON / UTF-8",
            "logical_prefix_fingerprint_sha256": digest_bytes(canonical),
            "full_file_hash_performed": False, "records": records}

def anchor_prefix(cfg):
    relative = cfg["bounded_anchor_source"]["path"]
    path = (REPO / relative).resolve()
    path.relative_to(REPO.resolve())
    with path.open("rb", buffering=0) as handle:
        lines = [bounded_line(handle)
                 for _ in range(cfg["bounded_anchor_source"]["maximum_data_records"] + 1)]
    READ_EVENTS.append({"path": relative, "mode": "unbuffered_exact_9_lines",
                        "data_records": 8, "full_file_hash": False})
    return parse_development_anchor_prefix(b"".join(lines), cfg)

def checkpoint_entries(cfg):
    relative = next(r["path"] for r in cfg["full_read_allowlist"]
                    if r["id"] == cfg["checkpoint_map_id"])
    path = require_full_read_allowed(relative, cfg)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if len(doc) != 9:
        raise AccessDenied("NINE_CHECKPOINTS_REQUIRED")
    for identity, row in doc.items():
        expected = ("RENew-PEMFC-2300h-Pytorch/"
                    "Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/"
                    f"H-{row['model']}-Seed{row['seed']}_best.pt")
        if row["path"].replace("\\", "/") != expected:
            raise AccessDenied("CHECKPOINT_NOT_FROZEN_H_IDENTITY")
        if identity != f"H-{row['model']}-Seed{row['seed']}":
            raise AccessDenied("CHECKPOINT_IDENTITY_MISMATCH")
    return doc

def changed_paths():
    raw = git("status", "--porcelain=v1", "-z", "--untracked-files=all", binary=True)
    paths = []
    for record in raw.split(b"\0"):
        if record:
            # Renames are not allowed here; all old files must be unchanged.
            if record[:2].find(b"R") >= 0:
                raise AccessDenied("RENAME_NOT_ALLOWED")
            paths.append(record[3:].decode("utf-8").replace("\\", "/"))
    return paths

def check_scope(cfg):
    git("merge-base", "--is-ancestor", cfg["base_commit"], "HEAD")
    if git("branch", "--show-current") != "main":
        raise AccessDenied("MAIN_BRANCH_REQUIRED")
    prefix = PHASE.relative_to(REPO).as_posix() + "/"
    paths = changed_paths()
    outside = [p for p in paths if not p.startswith(prefix)]
    if outside:
        raise AccessDenied("HISTORICAL_OR_OUTSIDE_CHANGE:" + repr(outside))
    return paths

def source_identities(cfg):
    rows = []
    for entry in cfg["full_read_allowlist"]:
        relative = entry["path"]
        observed = safe_full_hash(relative, cfg)
        frozen = git("show", cfg["base_commit"] + ":" + relative, binary=True)
        expected = digest_bytes(frozen)
        local = require_full_read_allowed(relative, cfg).read_bytes()
        canonical_local = digest_bytes(local.replace(b"\r\n", b"\n"))
        canonical_base = digest_bytes(frozen.replace(b"\r\n", b"\n"))
        if canonical_local != canonical_base:
            raise AccessDenied("SOURCE_DIFFERS_FROM_BASE:" + relative)
        rows.append({**entry, "sha256": observed, "base_sha256": expected,
                     "base_blob_bytes_equal": observed == expected,
                     "comparison": "CRLF_to_LF_only_for_text_Git_checkout_identity",
                     "canonical_sha256": canonical_local,
                     "canonical_base_sha256": canonical_base,
                     "git_blob": git("rev-parse", cfg["base_commit"] + ":" + relative),
                     "source_last_commit": git("log", "-1", "--format=%H",
                                               cfg["base_commit"], "--", relative),
                     "access_scope": "small_frozen_document_or_static_source",
                     "source_code_executed": False})
    return rows

def checkpoint_hashes(cfg):
    entries = checkpoint_entries(cfg)
    allowed = [row["path"] for row in entries.values()]
    rows = []
    for identity, row in entries.items():
        observed = safe_full_hash(row["path"], cfg, allowed)
        if observed != row["sha256"]:
            raise AccessDenied("CHECKPOINT_HASH_CHANGED:" + identity)
        rows.append({"identity": identity, "path": row["path"],
                     "expected_sha256": row["sha256"], "observed_sha256": observed,
                     "unchanged": True, "loaded_as_model": False})
    return rows

def write_json(name, data):
    target = PHASE / "outputs" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.verify:
        parser.error("Specify exactly one of --freeze / --verify")
    cfg = load_config()
    manifest = PHASE / "outputs/K_SOURCE_MANIFEST.json"
    if args.verify:
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior["access_protocol_sha256"] != digest_bytes(CONFIG_PATH.read_bytes()):
            raise AccessDenied("ACCESS_PROTOCOL_CHANGED_BEFORE_SOURCE_IO")
    paths = check_scope(cfg)
    rows = source_identities(cfg)
    checkpoints = checkpoint_hashes(cfg)
    prefix = anchor_prefix(cfg)
    snapshot = {"base_commit": cfg["base_commit"], "source_identities": rows,
                "checkpoint_identities": checkpoints, "bounded_anchor_prefix": prefix,
                "access_protocol_sha256": digest_bytes(CONFIG_PATH.read_bytes())}
    if args.freeze:
        if manifest.exists():
            raise AccessDenied("EVIDENCE_SNAPSHOT_ALREADY_EXISTS_NO_OVERWRITE")
        write_json("K_SOURCE_MANIFEST.json", snapshot)
    else:
        frozen = json.loads(manifest.read_text(encoding="utf-8"))
        if frozen != snapshot:
            raise AccessDenied("EVIDENCE_CHANGED_AFTER_K_SNAPSHOT")
        for report in cfg["required_reports"]:
            p = PHASE / "reports" / report
            if not p.exists() or p.stat().st_size == 0:
                raise AccessDenied("REQUIRED_REPORT_MISSING:" + report)
        ledger = PHASE / "evidence/K_SOURCE_EVIDENCE_LEDGER.md"
        if not ledger.exists() or not ledger.stat().st_size:
            raise AccessDenied("EVIDENCE_LEDGER_MISSING")
        test_record = json.loads((PHASE / "outputs/K_FIREWALL_TEST_RESULTS.json").read_text(encoding="utf-8"))
        if (test_record["returncode"] != 0
                or test_record["production_audit_sha256"] != digest_bytes(Path(__file__).read_bytes())
                or test_record["test_module_sha256"] != digest_bytes((PHASE / "tests/test_readonly_firewall.py").read_bytes())):
            raise AccessDenied("FIREWALL_TEST_EVIDENCE_FAILED_OR_STALE")
        write_json("K_READONLY_INTEGRITY_AUDIT.json", {
            "status": "PASS", "phase": PHASE.name, "base_commit": cfg["base_commit"],
            "execution_head": git("rev-parse", "HEAD"), "branch": "main",
            "method": "explicit allowlist + exact bounded lines + base Git blobs + byte identity hashes + Git status",
            "not_an_os_level_global_IO_monitor": True,
            "allowed_document_count": len(rows), "source_identity_unchanged": True,
            "checkpoint_identity_count": len(checkpoints), "nine_checkpoints_unchanged": True,
            "production_firewall_test_exitcode": test_record["returncode"],
            "test_evidence": "outputs/K_FIREWALL_TEST_RESULTS.json",
            "checkpoint_model_load_performed": False,
            "bounded_anchor_prefix": {k: v for k, v in prefix.items() if k != "records"},
            "mixed_anchor_full_file_hash": False, "causal_features_file_opened": False,
            "raw_data_opened": False, "Test_numerical_rows_read": False,
            "Test_polarization_values_read": False, "Test_labels_read": False,
            "historical_Test_metadata_may_be_documented": True,
            "new_training": False, "new_inference": False, "new_Assessment_inference": False,
            "model_imports": False, "seed_selection": False, "SOH_labels_created": False,
            "EOL_defined": False, "RUL_labels_created": False,
            "historical_phase_files_changed": False, "working_changes": paths,
            "source_manifest_sha256": digest_bytes(manifest.read_bytes()),
            "audit_script_sha256": digest_bytes(Path(__file__).read_bytes()),
            "data_access_events": READ_EVENTS, "automatic_next_phase": False})
    print("PHASE_K_READONLY_" + ("SNAPSHOT" if args.freeze else "AUDIT") + "_PASS")
    print("Documents:", len(rows), "Checkpoint identities:", len(checkpoints),
          "Development anchor records:", prefix["row_count"])
    print("New training / inference / Test content: NO")

if __name__ == "__main__":
    main()

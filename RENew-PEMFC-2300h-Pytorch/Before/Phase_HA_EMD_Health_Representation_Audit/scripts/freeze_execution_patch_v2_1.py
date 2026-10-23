"""Freeze the non-statistical V2.1 runtime patch after its regression test."""
from __future__ import annotations

import json

from ha_common import (HA_DIR, V1_PROTOCOL_SHA256, assert_history_provenance,
                       execution_code_hashes, sha256_file, write_json)

AMEND = HA_DIR / "configs" / "PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.yaml"
AMEND_SHA = HA_DIR / "configs" / "PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.sha256"
BLOCKER = HA_DIR / "TECHNICAL_BLOCKER_V2.md"


def main() -> None:
    assert_history_provenance()
    if not AMEND.exists() or not AMEND_SHA.exists() or not BLOCKER.exists():
        raise RuntimeError("preserved V2 amendment or V2 blocker record is missing")
    amendment_sha = AMEND_SHA.read_text(encoding="utf-8").strip().split()[0]
    if sha256_file(AMEND) != amendment_sha:
        raise RuntimeError("preserved V2 amendment SHA mismatch")
    amend = json.loads(AMEND.read_text(encoding="utf-8"))
    patch = {
        "phase": "Phase_HA_EMD_Health_Representation_Audit",
        "patch": "PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_1",
        "status": "FROZEN_AFTER_V2_RUNTIME_BLOCKER_BEFORE_FIRST_PERSISTED_ENDPOINT_STATISTICS",
        "v1_protocol_sha256": V1_PROTOCOL_SHA256,
        "predecessor_amendment_v2_sha256": amendment_sha,
        "predecessor_execution_code_hashes": amend["execution_code_hashes"],
        "v2_runtime_blocker_record": {
            "path": "TECHNICAL_BLOCKER_V2.md",
            "sha256": sha256_file(BLOCKER),
            "observed_error": "AttributeError: empty ASR table has no stratum column",
        },
        "sole_implementation_change": {
            "location": "final_decision ASR gate evaluation",
            "behavior": "an empty ASR table from no matched-support stratum is interpreted as a failed stability gate instead of dereferencing a missing dataframe column",
        },
        "statistical_protocol_changes": [],
        "explicitly_unchanged": [
            "N1 prefix-only access", "Test prohibition", "cache dual-hash semantics",
            "EMD parameters and units", "matching variables/scales/calipers", "24h/48h blocks",
            "2000 iterations", "seed 2024", "moving-block bootstrap", "RDR definition",
            "recovery definition", "Phase DF frozen results", "model-training prohibition",
        ],
        "execution_code_hashes": execution_code_hashes(),
    }
    path = HA_DIR / "configs" / "PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_1.yaml"
    write_json(path, patch)
    digest = sha256_file(path)
    (HA_DIR / "configs" / "PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_1.sha256").write_text(
        f"{digest}  PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_1.yaml\n", encoding="utf-8"
    )
    print(json.dumps({"patch": str(path), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()

"""Freeze the V2.2 no-statistical-change runtime patch."""
from __future__ import annotations

import json

from ha_common import HA_DIR, assert_history_provenance, execution_code_hashes, sha256_file, write_json

AMEND = HA_DIR / "configs" / "PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.yaml"
AMEND_SHA = HA_DIR / "configs" / "PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.sha256"
PATCH1 = HA_DIR / "configs" / "PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_1.yaml"
PATCH1_SHA = HA_DIR / "configs" / "PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_1.sha256"
BLOCKER1 = HA_DIR / "TECHNICAL_BLOCKER_V2_1.md"


def frozen_digest(path, sidecar):
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"missing frozen artifact: {path.name}")
    digest=sidecar.read_text(encoding="utf-8").strip().split()[0]
    if sha256_file(path)!=digest:
        raise RuntimeError(f"frozen artifact SHA mismatch: {path.name}")
    return digest


def main() -> None:
    assert_history_provenance()
    amendment_sha=frozen_digest(AMEND,AMEND_SHA); patch1_sha=frozen_digest(PATCH1,PATCH1_SHA)
    if not BLOCKER1.exists():
        raise RuntimeError("V2.1 blocker record is missing")
    amend=json.loads(AMEND.read_text(encoding="utf-8")); patch1=json.loads(PATCH1.read_text(encoding="utf-8"))
    patch={
        "phase":"Phase_HA_EMD_Health_Representation_Audit",
        "patch":"PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_2",
        "status":"FROZEN_AFTER_V2_1_RUNTIME_BLOCKER_BEFORE_FIRST_PERSISTED_ENDPOINT_STATISTICS",
        "predecessor_amendment_v2_sha256":amendment_sha,
        "predecessor_execution_patch_v2_1_sha256":patch1_sha,
        "predecessor_execution_code_hashes":patch1["execution_code_hashes"],
        "v2_1_runtime_blocker_record":{"path":"TECHNICAL_BLOCKER_V2_1.md","sha256":sha256_file(BLOCKER1),"observed_error":"AttributeError: empty matched-contrast table has no stratum column in RDR audit"},
        "sole_implementation_change":{"location":"rdr_audit empty-contrast handling","behavior":"an empty contrast table from no matched-support stratum produces RDR_NOT_IDENTIFIABLE_IN_STABLE_SUBSET for each stratum instead of dereferencing a missing dataframe column"},
        "statistical_protocol_changes":[],
        "explicitly_unchanged":["N1 prefix-only access","Test prohibition","cache dual-hash semantics","EMD parameters and units","matching variables/scales/calipers","24h/48h blocks","2000 iterations","seed 2024","moving-block bootstrap","RDR definition and 30-event rule","recovery definition","Phase DF frozen results","model-training prohibition"],
        "execution_code_hashes":execution_code_hashes(),
    }
    path=HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_2.yaml";write_json(path,patch);digest=sha256_file(path)
    (HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_2.sha256").write_text(f"{digest}  PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_2.yaml\n",encoding="utf-8")
    print(json.dumps({"patch":str(path),"sha256":digest},indent=2))


if __name__=="__main__": main()

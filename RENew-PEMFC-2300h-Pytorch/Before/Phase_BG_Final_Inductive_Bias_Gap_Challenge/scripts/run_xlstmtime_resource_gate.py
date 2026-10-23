"""Run the xLSTMTime source and synthetic-resource gate before Development scoring.

This entry point deliberately does not invoke the development-prefix reader or
the window builder.  It makes the resource decision solely from the frozen
configuration, the author-source fingerprint, and a synthetic Raw-F2 tensor.
"""
from __future__ import annotations

import json

from legacy_protocol import load_legacy_contract, write_csv, write_json
from run_xlstmtime import CANDIDATE, canonical_hashes, load_config, resource_gate


def main() -> None:
    cfg = load_config()
    _legacy_cfg, helpers, protocol = load_legacy_contract()
    source = canonical_hashes(cfg)
    resource = resource_gate(cfg, helpers)
    payload = {
        **resource,
        "legacy_protocol_sha256": protocol["legacy_protocol_sha256"],
        "legacy_protected_bs_drift": protocol["protected_bs_drift"],
        "development_performance_computed": False,
        "test_accessed": False,
    }
    write_csv(source, "02_xLSTMTime", "SOURCE_FINGERPRINTS.csv")
    write_json(payload, "02_xLSTMTime", "RESOURCE_AUDIT.json")
    print(json.dumps({"candidate": CANDIDATE, "status": payload["status"]}, ensure_ascii=False))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

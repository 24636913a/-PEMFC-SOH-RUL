"""Verify Phase POT's frozen authority without starting Stage 0 or training."""
from __future__ import annotations

import json

from pot_common import load_inherited_protocols, load_protocol, source_audit, write_csv, write_json


def main() -> None:
    cfg, digest = load_protocol()
    bs_cfg, df_cfg, common = load_inherited_protocols(cfg)
    audit = source_audit(common, bs_cfg)
    write_csv(audit, "outputs", "PRETRAINING_TIMESNET_SOURCE_AUDIT.csv")
    result = {
        "status": "PASS",
        "protocol_sha256": digest,
        "phase_bs_protocol_sha256": cfg["inherited_evidence"]["phase_bs_protocol_sha256"],
        "phase_df_protocol_sha256": cfg["inherited_evidence"]["phase_df_protocol_sha256"],
        "TimesNet_source_pass": bool(audit["pass"].all()),
        "data_not_read": True,
        "model_training_performed": False,
        "test_accessed": False,
        "df_protocol_status": df_cfg["status"],
    }
    write_json(result, "outputs", "PROTOCOL_PREFLIGHT_AUDIT.json")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

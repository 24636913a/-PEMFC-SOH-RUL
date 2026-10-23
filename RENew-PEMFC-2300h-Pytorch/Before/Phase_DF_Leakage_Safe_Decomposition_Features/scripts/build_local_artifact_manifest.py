"""Record ignored local Phase DF artifacts without reading data or running models."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from df_protocol_adapter import DF, sha256_file, write_csv, write_json


RUN_PATTERN = re.compile(r"^df_(?P<route>.+)_fold_(?P<fold>\d+)_seed(?P<seed>\d+)$")
ROUTE_TO_METHOD = {
    "simpletrend_timesnet": "SimpleTrend_TimesNet",
    "vmdfeature_timesnet": "VMDFeature_TimesNet",
    "emdfeature_timesnet": "EMDFeature_TimesNet",
}


def _run_metadata(path: Path) -> dict[str, Any]:
    matched = RUN_PATTERN.match(path.stem.removesuffix("_epochs"))
    if not matched:
        return {"method": "not_applicable", "fold": "not_applicable", "seed": "not_applicable"}
    fields = matched.groupdict()
    return {
        "method": ROUTE_TO_METHOD.get(fields["route"], fields["route"]),
        "fold": f"fold_{fields['fold']}",
        "seed": int(fields["seed"]),
    }


def _row(path: Path, kind: str, protocol_sha: str) -> dict[str, Any]:
    metadata = _run_metadata(path)
    if kind == "origin_window_feature_cache":
        metadata = {
            "method": path.parent.name,
            "fold": "membership_recorded_in_CACHE_PROVENANCE_MANIFEST",
            "seed": "not_applicable",
        }
    command = "D:/ruanjian/python3.11.9/python.exe scripts/run_stage0.py" if kind == "origin_window_feature_cache" else "D:/ruanjian/python3.11.9/python.exe scripts/run_stage1.py"
    return {
        "artifact_kind": kind,
        "local_absolute_path": str(path.resolve()),
        "relative_phase_path": str(path.relative_to(DF)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "method": metadata["method"],
        "fold": metadata["fold"],
        "seed": metadata["seed"],
        "df_protocol_sha256": protocol_sha,
        "rebuild_command": command,
        "test_accessed": False,
    }


def main() -> None:
    protocol_sha = (DF / "configs" / "PREREGISTERED_DF_PROTOCOL.sha256").read_text(encoding="utf-8").split()[0]
    groups = {
        "origin_window_feature_cache": DF / "cache",
        "checkpoint": DF / "checkpoints",
        "prediction_cache": DF / "outputs" / "predictions",
        "epoch_log": DF / "logs",
    }
    rows = []
    for kind, root in groups.items():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append(_row(path, kind, protocol_sha))
    counts = pd.Series([row["artifact_kind"] for row in rows]).value_counts().to_dict() if rows else {}
    write_csv(pd.DataFrame(rows), "outputs", "LOCAL_ARTIFACT_MANIFEST.csv")

    resource = pd.read_csv(DF / "outputs" / "RESOURCE_AUDIT.csv")
    resource_payload = {
        "status": "PASS" if not resource.empty and bool((resource["projected_complete_origin_cache_hours"] <= 2.0).all()) else "RESOURCE_STOP",
        "method_rows": resource.to_dict(orient="records"),
        "registered_method_limit_hours": 2.0,
        "test_accessed": False,
        "derived_from": "outputs/RESOURCE_AUDIT.csv",
    }
    write_json(resource_payload, "outputs", "RESOURCE_AUDIT.json")
    write_json({
        "status": "PASS",
        "artifact_counts": counts,
        "manifest_rows": len(rows),
        "df_protocol_sha256": protocol_sha,
        "test_accessed": False,
        "model_training_performed": False,
        "input_scope": "local_already_written_phase_df_artifacts_only",
    }, "outputs", "LOCAL_ARTIFACT_MANIFEST_AUDIT.json")
    print(json.dumps({"rows": len(rows), "artifact_counts": counts, "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

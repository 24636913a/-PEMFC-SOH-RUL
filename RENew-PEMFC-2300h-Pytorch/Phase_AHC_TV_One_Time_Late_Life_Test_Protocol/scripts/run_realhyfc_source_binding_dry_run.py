"""Filename-only RealHyFC source-binding audit before any AHC-TV Test read.

This module enumerates directory entries and evaluates frozen filename rules.
It must never open, hash, parse, or otherwise inspect a real Test CSV.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from ahc_tv_alignment import PHASE, read_yaml
from ahc_tv_stage_p import bind_both_polarization_sources


OUTPUTS = PHASE / "outputs"
FILENAME_AUDIT = OUTPUTS / "AHC_TV_TEST_SOURCE_FILENAME_AUDIT.csv"
DRY_RUN_AUDIT = OUTPUTS / "AHC_TV_REAL_FILENAME_BINDING_DRY_RUN.json"
AFTER_AIR_AIR_PATTERN = re.compile(r"after(?:[_-]?air){2}", flags=re.IGNORECASE)
CHECKPOINT_PATTERN = {
    2000: re.compile(r"(?<![0-9])2000(?:h)?(?![0-9])", flags=re.IGNORECASE),
    2300: re.compile(r"(?<![0-9])2300(?:h)?(?![0-9])", flags=re.IGNORECASE),
}


class FilenameBindingDryRunError(RuntimeError):
    """The source root or its filename-only binding did not meet the contract."""


def _registry(phase: Path) -> dict[str, Any]:
    return read_yaml(Path(phase) / "configs" / "AHC_TV_RUNTIME_SOURCE_REGISTRY.yaml")


def _has_checkpoint(name: str, checkpoint: int) -> bool:
    return CHECKPOINT_PATTERN[int(checkpoint)].search(name) is not None


def _audit_role(path: Path, registry: dict[str, Any]) -> str:
    """Classify only the filename.  No file content is ever accessed."""
    name = path.name
    lowered = name.lower()
    forbidden = any(token.lower() in lowered for token in registry["forbidden_source_name_tokens"])
    if AFTER_AIR_AIR_PATTERN.search(name) is not None:
        return "AFTER_AIR_AIR_EXCLUDED"
    if forbidden:
        return "OTHER"
    if re.search(registry["polarization_checkpoint_sources"]["2000"]["filename_regex"], name):
        return "NOMINAL_2000_CANDIDATE"
    if re.search(registry["polarization_checkpoint_sources"]["2300"]["filename_regex"], name):
        return "NOMINAL_2300_CANDIDATE"
    return "OTHER"


def build_filename_audit_rows(*, phase: Path = PHASE) -> tuple[Path, list[dict[str, object]]]:
    """Enumerate only CSV names and relative paths below the frozen local root."""
    registry = _registry(Path(phase))
    root = Path(registry["local_dataset_root"]["path"])
    if not root.is_dir():
        raise FilenameBindingDryRunError("AUTHORITATIVE_LOCAL_DATASET_ROOT_MISSING")
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.csv"), key=lambda item: str(item.relative_to(root)).lower()):
        name = path.name
        contains_after = AFTER_AIR_AIR_PATTERN.search(name) is not None
        if not (contains_after or _has_checkpoint(name, 2000) or _has_checkpoint(name, 2300)):
            continue
        rows.append({
            "basename": name,
            "relative_path": str(path.relative_to(root)),
            "contains_2000": _has_checkpoint(name, 2000),
            "contains_2300": _has_checkpoint(name, 2300),
            "contains_afterairair": contains_after,
            "classified_role": _audit_role(path, registry),
        })
    return root, rows


def run_real_filename_binding_dry_run(*, phase: Path = PHASE) -> dict[str, Any]:
    """Write the permitted filename-only audit and require a unique nominal pair."""
    phase = Path(phase)
    root, rows = build_filename_audit_rows(phase=phase)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with FILENAME_AUDIT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "basename", "relative_path", "contains_2000", "contains_2300",
            "contains_afterairair", "classified_role",
        ])
        writer.writeheader()
        writer.writerows(rows)

    candidates = [root / str(row["relative_path"]) for row in rows]
    try:
        bound = bind_both_polarization_sources(candidates, phase=phase)
    except Exception as error:
        audit = {
            "dataset_root": str(root),
            "file_content_opened": False,
            "nominal_2000_match_count": sum(row["classified_role"] == "NOMINAL_2000_CANDIDATE" for row in rows),
            "nominal_2300_match_count": sum(row["classified_role"] == "NOMINAL_2300_CANDIDATE" for row in rows),
            "after_air_air_2300_excluded": any(bool(row["contains_2300"]) and bool(row["contains_afterairair"]) for row in rows),
            "binding_status": "FAIL",
            "binding_error": type(error).__name__,
        }
        DRY_RUN_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise FilenameBindingDryRunError("TEST_SOURCE_BINDING_AMBIGUOUS") from error

    nominal_2000 = [row for row in rows if row["classified_role"] == "NOMINAL_2000_CANDIDATE"]
    nominal_2300 = [row for row in rows if row["classified_role"] == "NOMINAL_2300_CANDIDATE"]
    after_air_air_2300 = [
        row for row in rows
        if bool(row["contains_2300"]) and bool(row["contains_afterairair"])
    ]
    passed = len(nominal_2000) == 1 and len(nominal_2300) == 1 and bool(after_air_air_2300)
    audit = {
        "dataset_root": str(root),
        "file_content_opened": False,
        "nominal_2000_match_count": len(nominal_2000),
        "nominal_2300_match_count": len(nominal_2300),
        "after_air_air_2300_excluded": bool(after_air_air_2300) and bound[2300].name != after_air_air_2300[0]["basename"],
        "matched_2000_basename": bound[2000].name,
        "matched_2300_basename": bound[2300].name,
        "binding_status": "PASS" if passed else "FAIL",
    }
    DRY_RUN_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not passed or not audit["after_air_air_2300_excluded"]:
        raise FilenameBindingDryRunError("TEST_SOURCE_BINDING_AMBIGUOUS")
    return audit


if __name__ == "__main__":
    print(json.dumps(run_real_filename_binding_dry_run(), indent=2, ensure_ascii=False))

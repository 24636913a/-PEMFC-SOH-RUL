"""Read the immutable Raw F2 order from the frozen Phase-E protocol.

This is configuration provenance only. It never opens a data table, label,
prediction cache, Assessment artifact, or Test artifact.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


PHASE_ROOT = Path(__file__).resolve().parents[1]
PHASE_E_PROTOCOL = (
    PHASE_ROOT.parent
    / "Phase_TFT_FullFeature_CausalFilter_Comparison"
    / "configs"
    / "PREREGISTERED_E_COMPARISON.yaml"
)


def load_frozen_raw_f2_feature_order() -> tuple[str, ...]:
    payload = yaml.safe_load(PHASE_E_PROTOCOL.read_text(encoding="utf-8"))
    order = tuple(payload["task"]["raw_f2_feature_order"])
    if len(order) != 14 or len(set(order)) != 14:
        raise RuntimeError("Frozen Phase-E Raw F2 feature schema is invalid")
    return order


def protocol_sha256() -> str:
    return hashlib.sha256(PHASE_E_PROTOCOL.read_bytes()).hexdigest()


RAW_F2_FEATURE_NAMES = load_frozen_raw_f2_feature_order()

"""Fail-closed CHM-D runtime authority and frozen-table loading."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
CONFIGS = PHASE / "configs"
INPUTS = PHASE / "inputs"
FINAL_V2 = CONFIGS / "PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml"
SUPPLEMENT_A = CONFIGS / "CHM_FINAL_V2_EXECUTION_SUPPLEMENT_A.yaml"
AUTHORITY = CONFIGS / "CHM_PROTOCOL_AUTHORITY.yaml"
HASHES = CONFIGS / "CHM_FROZEN_INPUT_HASHES_V2.yaml"
HOU_PATH = INPUTS / "CHM_HOU_TABLE.csv"
TARGETS_PATH = INPUTS / "CHM_SPARSE_ANCHOR_TARGETS.csv"
MAPPING_PATH = INPUTS / "CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv"
CONDITION_COLUMNS = (
    "current_density",
    "anode_inlet_temperature_C",
    "anode_inlet_pressure_kPag",
    "anode_inlet_RH_pct",
    "anode_hydrogen_flow_nl_min",
    "anode_stoichiometry",
    "cathode_stoichiometry",
)
FORBIDDEN_MODEL_INPUTS = {
    "OpHrs2",
    "measurement_end_ophrs2_h",
    "DateTime",
    "measurement_end_datetime",
    "raw_cycle_id",
    "cycle_index",
    "nominal_checkpoint_age",
    "future_anchor",
    "future_cycle",
    "Test",
}
ANCHORS = (300, 600, 850, 1150, 1500, 1750)


class CHMRuntimeFailure(RuntimeError):
    """Raised before training when a frozen runtime contract cannot be proven."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise CHMRuntimeFailure(f"YAML mapping required: {path.name}")
    return payload


def verify_runtime_authority() -> dict[str, Any]:
    """Verify every frozen file hash before a caller may create an optimizer."""
    authority = load_yaml(AUTHORITY)
    if authority.get("authority_version") != "FINAL_V2A" or authority.get("current_status") != "FINAL_V2A_FROZEN_READY_FOR_CHM_D":
        raise CHMRuntimeFailure("CHM_PROTOCOL_RUNTIME_MISMATCH: authority is not Final V2A-ready.")
    expected_runtime = [
        "inputs/CHM_HOU_TABLE.csv",
        "inputs/CHM_SPARSE_ANCHOR_TARGETS.csv",
        "inputs/CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv",
        "configs/PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml",
    ]
    if authority.get("CHM_D_runtime_variable_data_inputs_only") != expected_runtime:
        raise CHMRuntimeFailure("CHM_PROTOCOL_RUNTIME_MISMATCH: runtime table allowlist drift.")
    manifest = load_yaml(HASHES)
    required = {
        "final_v2_protocol",
        "execution_semantics_supplement_a",
        "protocol_authority",
        "HOU_table",
        "sparse_anchor_targets",
        "frozen_anchor_interval_mapping",
        "baseline_normative_document",
        "development_evaluation_v2",
        "uncertainty_v2",
        "test_firewall_v2",
    }
    records = manifest.get("records", {})
    if not required.issubset(records):
        raise CHMRuntimeFailure("CHM_PROTOCOL_RUNTIME_MISMATCH: frozen hash manifest incomplete.")
    for name, entry in records.items():
        path = (REPO / entry["path"]).resolve()
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise CHMRuntimeFailure(f"CHM_PROTOCOL_RUNTIME_MISMATCH: SHA256 mismatch for {name}.")
    if not FINAL_V2.is_file() or not SUPPLEMENT_A.is_file() or not HOU_PATH.is_file() or not TARGETS_PATH.is_file() or not MAPPING_PATH.is_file():
        raise CHMRuntimeFailure("CHM_PROTOCOL_RUNTIME_MISMATCH: required frozen artifact missing.")
    return {"authority": authority, "protocol": load_yaml(FINAL_V2), "supplement": load_yaml(SUPPLEMENT_A), "hash_manifest": manifest}


@dataclass(frozen=True)
class ReferenceScaler:
    median: dict[str, float]
    scale: dict[str, float]
    constant_features: tuple[str, ...]
    input_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class FrozenCHMData:
    hou: pd.DataFrame
    targets: pd.DataFrame
    mapping: pd.DataFrame
    scaler: ReferenceScaler
    model_inputs: np.ndarray
    anchor_indices: dict[int, tuple[int, int, int]]
    continuous_complete_triples: tuple[tuple[int, int, int, int, int, int], ...]


def _validate_hou(hou: pd.DataFrame) -> pd.DataFrame:
    required = {"hou_id", "hou_kind", "source_interval_id", "raw_cycle_id", "measurement_end_datetime", "P1S_operating_voltage_retention", *CONDITION_COLUMNS}
    if not required.issubset(hou.columns) or len(hou) != 152:
        raise CHMRuntimeFailure("Frozen HOU schema/count mismatch.")
    ordered = hou.copy()
    ordered["measurement_end_datetime"] = pd.to_datetime(ordered["measurement_end_datetime"], errors="raise")
    if ordered["measurement_end_datetime"].isna().any() or not ordered["measurement_end_datetime"].is_unique:
        raise CHMRuntimeFailure("HOU timestamps are missing or non-unique.")
    if not ordered["measurement_end_datetime"].is_monotonic_increasing:
        raise CHMRuntimeFailure("Frozen HOU chronology must already be strictly ascending; reordering is prohibited.")
    if ordered["P1S_operating_voltage_retention"].isna().any() or ordered[list(CONDITION_COLUMNS)].isna().any().any():
        raise CHMRuntimeFailure("Frozen HOU contains missing retention or required condition values.")
    complete = ordered["hou_kind"].eq("COMPLETE_CYCLE_HOU")
    partial = ordered["hou_kind"].eq("PREANCHOR_PARTIAL_HOU")
    if int(complete.sum()) != 146 or int(partial.sum()) != 6 or int((complete | partial).sum()) != 152:
        raise CHMRuntimeFailure("Frozen HOU kind counts mismatch.")
    cycle_ids = ordered.loc[complete, "raw_cycle_id"]
    if cycle_ids.isna().any() or cycle_ids.astype(int).duplicated().any() or len(cycle_ids) != 146:
        raise CHMRuntimeFailure("Frozen complete-cycle IDs are invalid.")
    return ordered


def build_reference_scaler(hou: pd.DataFrame) -> ReferenceScaler:
    reference = hou.loc[hou["hou_kind"].eq("COMPLETE_CYCLE_HOU")].head(10)
    if len(reference) != 10:
        raise CHMRuntimeFailure("Earliest ten legal complete HOU are unavailable.")
    median: dict[str, float] = {}
    scale: dict[str, float] = {}
    constants: list[str] = []
    retained: list[str] = ["P1S_retention_minus_1"]
    for column in CONDITION_COLUMNS:
        values = reference[column].astype(float).to_numpy()
        med = float(np.median(values))
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        std = float(np.std(values, ddof=0))
        chosen = iqr if iqr > 0 else std
        median[column] = med
        scale[column] = chosen
        if chosen == 0:
            constants.append(column)
        else:
            retained.append(column)
    return ReferenceScaler(median=median, scale=scale, constant_features=tuple(constants), input_feature_names=tuple(retained))


def model_inputs_from_hou(hou: pd.DataFrame, scaler: ReferenceScaler) -> np.ndarray:
    columns: list[np.ndarray] = [(hou["P1S_operating_voltage_retention"].astype(float).to_numpy() - 1.0)]
    for column in CONDITION_COLUMNS:
        if column in scaler.constant_features:
            continue
        columns.append((hou[column].astype(float).to_numpy() - scaler.median[column]) / scaler.scale[column])
    result = np.column_stack(columns).astype(np.float64, copy=False)
    if not np.isfinite(result).all() or any(name in FORBIDDEN_MODEL_INPUTS for name in scaler.input_feature_names):
        raise CHMRuntimeFailure("Forbidden or non-finite model input.")
    return result


def _anchor_indices(hou: pd.DataFrame, targets: pd.DataFrame, mapping: pd.DataFrame) -> dict[int, tuple[int, int, int]]:
    if targets["checkpoint"].astype(int).tolist() != list(ANCHORS) or len(targets) != 6 or len(mapping) != 18:
        raise CHMRuntimeFailure("Frozen sparse anchor universe mismatch.")
    index_by_interval = {value: int(index) for index, value in hou["source_interval_id"].items()}
    result: dict[int, tuple[int, int, int]] = {}
    for checkpoint in ANCHORS:
        rows = mapping.loc[mapping["checkpoint"].astype(int).eq(checkpoint)].sort_values("selection_rank")
        if len(rows) != 3 or rows["selection_rank"].astype(int).tolist() != [1, 2, 3]:
            raise CHMRuntimeFailure(f"Frozen anchor mapping invalid for {checkpoint}.")
        interval_ids = rows["source_interval_id"].tolist()
        if any(interval_id not in index_by_interval for interval_id in interval_ids):
            raise CHMRuntimeFailure(f"Mapped HOU unavailable for anchor {checkpoint}.")
        result[checkpoint] = tuple(index_by_interval[interval_id] for interval_id in interval_ids)
    return result


def _continuous_complete_triples(hou: pd.DataFrame) -> tuple[tuple[int, int, int, int, int, int], ...]:
    complete = hou.loc[hou["hou_kind"].eq("COMPLETE_CYCLE_HOU"), ["raw_cycle_id"]].copy()
    complete["row_index"] = complete.index
    triples: list[tuple[int, int, int, int, int, int]] = []
    rows = list(complete.itertuples(index=False))
    for left, middle, right in zip(rows, rows[1:], rows[2:]):
        if int(middle.raw_cycle_id) == int(left.raw_cycle_id) + 1 and int(right.raw_cycle_id) == int(middle.raw_cycle_id) + 1:
            triples.append((int(left.row_index), int(middle.row_index), int(right.row_index), int(left.raw_cycle_id), int(middle.raw_cycle_id), int(right.raw_cycle_id)))
    if not triples:
        raise CHMRuntimeFailure("No legal complete-cycle smoothness triples.")
    return tuple(triples)


def load_frozen_chm_data() -> FrozenCHMData:
    """Read only the three V2A-authorized variable Development input tables."""
    hou = _validate_hou(pd.read_csv(HOU_PATH))
    targets = pd.read_csv(TARGETS_PATH)
    mapping = pd.read_csv(MAPPING_PATH)
    scaler = build_reference_scaler(hou)
    return FrozenCHMData(
        hou=hou,
        targets=targets,
        mapping=mapping,
        scaler=scaler,
        model_inputs=model_inputs_from_hou(hou, scaler),
        anchor_indices=_anchor_indices(hou, targets, mapping),
        continuous_complete_triples=_continuous_complete_triples(hou),
    )


def target_for_checkpoint(data: FrozenCHMData, checkpoint: int) -> float:
    rows = data.targets.loc[data.targets["checkpoint"].astype(int).eq(int(checkpoint)), "Y"]
    if len(rows) != 1:
        raise CHMRuntimeFailure(f"Unique target missing for checkpoint {checkpoint}.")
    return float(rows.iloc[0])


def training_anchor_frame(data: FrozenCHMData, checkpoints: tuple[int, ...] | list[int]) -> pd.DataFrame:
    allowed = tuple(int(value) for value in checkpoints)
    if not allowed or any(value not in ANCHORS for value in allowed):
        raise CHMRuntimeFailure("Illegal training anchor set.")
    selected = data.targets[data.targets["checkpoint"].astype(int).isin(allowed)].copy()
    if len(selected) != len(allowed):
        raise CHMRuntimeFailure("Training anchor set has duplicate or missing checkpoint.")
    selected["checkpoint"] = selected["checkpoint"].astype(int)
    return selected.sort_values("checkpoint")

"""Final CHM Development-only feasibility audit and protocol-freeze checks.

This module deliberately contains no model, optimizer, LOAO, bootstrap-fit, or
Test-data path.  Every source read goes through one of the three allowlisted
readers below.  In particular, the only newly parsed raw source is a parser
bounded Development prefix of ``Loadcycle_151-162.csv``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
CONFIG = PHASE / "configs" / "PREREGISTERED_CHM_PROTOCOL_FINAL.yaml"
CONFIG_SHA = PHASE / "configs" / "PREREGISTERED_CHM_PROTOCOL_FINAL.sha256"
ALLOWLIST = PHASE / "configs" / "CHM_SOURCE_ALLOWLIST.yaml"
INPUT_HASHES = PHASE / "configs" / "CHM_FROZEN_INPUT_HASHES.yaml"
INPUTS = PHASE / "inputs"
OUTPUTS = PHASE / "outputs"

BOUNDARY_H = 1819.997738
PREFIX_ROWS_151_162 = 268_684
P1_TOKEN = "221||221.200"
P1_OPERATION_POINT = "221"
P1_SET_CURRENT = 221.200
ACTIVE_CELLS = tuple(f"U{i} [V]" for i in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11))
CONDITION_SOURCES = {
    "current_density": "j.S.act [A/cm2]",
    "anode_inlet_temperature_C": "T.Si.A [°C]",
    "anode_inlet_pressure_kPag": "p.Si.A [kPag]",
    "anode_inlet_RH_pct": "RH.Si.A  [%]",
    "anode_hydrogen_flow_nl_min": "F.Si.H.A [nl/min]",
    "anode_stoichiometry": "Stoic.S.A [-]",
    "cathode_stoichiometry": "Stoic.S.C [-]",
}
CONDITION_FIELDS = tuple(CONDITION_SOURCES)
EXPECTED_PARTIAL_IDS = (
    "PARTIAL_PRE_300",
    "PARTIAL_PRE_600",
    "PARTIAL_PRE_850",
    "PARTIAL_PRE_1150",
    "PARTIAL_PRE_1500",
    "PARTIAL_PRE_1750",
)
NEW_CYCLE_IDS = (152, 153, 154, 155, 156)
POLARIZATION_USECOLS = ("DateTime", "j.S.act [A/cm2]", *ACTIVE_CELLS)
RAW_USECOLS = (
    "OpHrs2 [h]",
    "DateTime",
    "operation_point",
    "I.S.set [A]",
    "I.S.act [A]",
    "j.S.act [A/cm2]",
    *ACTIVE_CELLS,
    *CONDITION_SOURCES.values(),
)


class CHMPreexecutionFailure(RuntimeError):
    """Fail-closed error for an invalid input, firewall, or protocol contract."""


class CHMFirewallFailure(CHMPreexecutionFailure):
    """Raised before an unregistered or post-boundary source can be parsed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise CHMPreexecutionFailure(f"YAML mapping required: {path}")
    return loaded


def allowlist_config() -> dict[str, Any]:
    return load_yaml(ALLOWLIST)


def final_protocol() -> dict[str, Any]:
    expected = CONFIG_SHA.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(CONFIG)
    if observed != expected:
        raise CHMPreexecutionFailure("FINAL_PROTOCOL_HASH_MISMATCH")
    protocol = load_yaml(CONFIG)
    if protocol.get("status") != "FINAL_PROTOCOL_FROZEN_AWAITING_EXECUTION_COMMAND":
        raise CHMPreexecutionFailure("Final protocol status is not frozen-awaiting-command.")
    if protocol.get("training_executed") or protocol.get("execution_authorized"):
        raise CHMPreexecutionFailure("Preexecution audit must not authorize or execute training.")
    return protocol


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve()


def _artifact_entry(name: str) -> tuple[dict[str, Any], Path]:
    registry = allowlist_config().get("repository_frozen_artifacts", {})
    if name not in registry:
        raise CHMFirewallFailure(f"UNREGISTERED_FROZEN_ARTIFACT: {name}")
    entry = registry[name]
    return entry, (REPO / entry["relative_path"]).resolve()


def safe_read_frozen_artifact(name: str, path: str | Path | None = None) -> pd.DataFrame:
    """Read one hash-verified frozen repository CSV through the CHM registry."""
    entry, expected = _artifact_entry(name)
    actual_path = expected if path is None else _resolved(path)
    if actual_path != expected:
        raise CHMFirewallFailure(f"FROZEN_ARTIFACT_PATH_NOT_ALLOWLISTED: {actual_path}")
    observed = sha256_file(actual_path)
    if observed != entry["sha256"]:
        raise CHMPreexecutionFailure(f"FROZEN_ARTIFACT_HASH_MISMATCH: {name}")
    return pd.read_csv(actual_path)


def safe_read_development_raw_prefix(
    path: str | Path | None = None,
    usecols: Iterable[str] | None = None,
    requested_rows: int = PREFIX_ROWS_151_162,
) -> pd.DataFrame:
    """Parse exactly the frozen Development prefix; never parse row 268685."""
    entry = allowlist_config()["development_raw_source"]
    expected = _resolved(entry["path"])
    actual_path = expected if path is None else _resolved(path)
    if actual_path != expected:
        raise CHMFirewallFailure(f"UNREGISTERED_RAW_SOURCE: {actual_path}")
    if requested_rows != PREFIX_ROWS_151_162:
        raise CHMFirewallFailure("RAW_PREFIX_ROW_LIMIT_VIOLATION")
    requested = list(usecols or RAW_USECOLS)
    header = pd.read_csv(actual_path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
    missing = sorted(set(requested).difference(header))
    if missing:
        raise CHMPreexecutionFailure(f"RAW_PREFIX_SCHEMA_MISSING: {missing}")
    frame = pd.read_csv(
        actual_path,
        sep=";",
        encoding="utf-8-sig",
        usecols=requested,
        nrows=PREFIX_ROWS_151_162,
        low_memory=False,
    )
    if len(frame) != PREFIX_ROWS_151_162:
        raise CHMPreexecutionFailure("CHM_DEVELOPMENT_PREFIX_FIREWALL_FAIL: parsed row count")
    ages = normalize_numeric(frame["OpHrs2 [h]"])
    if ages.isna().any() or not bool((ages < BOUNDARY_H).all()):
        raise CHMPreexecutionFailure("CHM_DEVELOPMENT_PREFIX_FIREWALL_FAIL: age boundary")
    frame["OpHrs2 [h]"] = ages
    return frame


def _polarization_entry(file_name: str | Path) -> tuple[dict[str, Any], Path, str]:
    supplied = str(file_name)
    name = Path(supplied).name
    if supplied != name:
        raise CHMFirewallFailure(f"POLARIZATION_BASENAME_REQUIRED: {file_name}")
    registry = allowlist_config()["development_polarization"]
    entries = registry["allowed_files"]
    if name not in entries:
        raise CHMFirewallFailure(f"UNREGISTERED_OR_TEST_POLARIZATION: {name}")
    return entries[name], (_resolved(registry["root"]) / name).resolve(), name


def safe_read_development_polarization(
    file_name: str | Path, usecols: Iterable[str] | None = None
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Read only one exact, hash-verified Development polarization artifact."""
    entry, path, _ = _polarization_entry(file_name)
    if sha256_file(path) != entry["sha256"]:
        raise CHMPreexecutionFailure(f"POLARIZATION_HASH_MISMATCH: {path.name}")
    header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()
    requested = list(usecols) if usecols is not None else header
    missing = sorted(set(requested).difference(header))
    if missing:
        raise CHMPreexecutionFailure(f"POLARIZATION_SCHEMA_MISSING: {path.name}: {missing}")
    return (
        pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=requested, low_memory=False),
        header,
        entry,
    )


def normalize_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype("string").str.replace(",", "", regex=False), errors="coerce")


def normalize_operation(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = values.astype("string").str.strip()
    valid = numeric.notna()
    result.loc[valid] = numeric.loc[valid].map(lambda value: f"{float(value):g}")
    return result.astype(str)


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin(("true", "1"))


def audit_polarization_structure() -> pd.DataFrame:
    """Audit layout and local j=1 window scatter without asserting repeats."""
    rows: list[dict[str, Any]] = []
    allowed = allowlist_config()["development_polarization"]["allowed_files"]
    for name, entry in allowed.items():
        frame, columns, _ = safe_read_development_polarization(name, POLARIZATION_USECOLS)
        local = frame.loc[
            np.isclose(normalize_numeric(frame["j.S.act [A/cm2]"]), 1.0, atol=0.01, rtol=0.0)
        ].copy()
        j = normalize_numeric(local["j.S.act [A/cm2]"])
        voltage = np.nanmean(local.loc[:, ACTIVE_CELLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), axis=1)
        datetimes = pd.to_datetime(local["DateTime"], errors="coerce")
        id_columns = [column for column in columns if re.search(r"replicate|trial|sweep", column, flags=re.IGNORECASE)]
        explicit_replicate = any(re.search(r"replicate|trial", column, flags=re.IGNORECASE) for column in id_columns)
        explicit_sweep = any(re.search(r"sweep", column, flags=re.IGNORECASE) for column in id_columns)
        rows.append(
            {
                "checkpoint": entry["checkpoint"],
                "source_file": name,
                "column_names": json.dumps(columns, ensure_ascii=False),
                "explicit_replicate_id_present": bool(explicit_replicate),
                "explicit_sweep_id_present": bool(explicit_sweep),
                "j1_local_row_count": int(len(local)),
                "j1_local_j_min": float(j.min()) if j.notna().any() else np.nan,
                "j1_local_j_max": float(j.max()) if j.notna().any() else np.nan,
                "j1_local_voltage_median": float(np.nanmedian(voltage)) if np.isfinite(voltage).any() else np.nan,
                "j1_local_voltage_sd": float(np.nanstd(voltage, ddof=1)) if np.isfinite(voltage).sum() >= 2 else np.nan,
                "DateTime_start": str(datetimes.min()) if datetimes.notna().any() else "",
                "DateTime_end": str(datetimes.max()) if datetimes.notna().any() else "",
                "independently_identifiable_replicates": False,
                "variance_semantics": "LOCAL_WINDOW_SCATTER_PROXY",
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit) != 7 or (audit["j1_local_row_count"] < 2).any() or audit["j1_local_voltage_sd"].isna().any():
        raise CHMPreexecutionFailure("POLARIZATION_LOCAL_WINDOW_NOT_QUALIFIED")
    if audit["independently_identifiable_replicates"].any():
        raise CHMPreexecutionFailure("Unexpected independent replicate classification.")
    return audit


def _new_cycle_target_states(catalog: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    complete = catalog.loc[_truthy(catalog["complete_cycle"])].copy()
    new_cycles = complete.loc[complete["raw_cycle_id"].isin(NEW_CYCLE_IDS)].copy()
    if tuple(new_cycles["raw_cycle_id"].astype(int)) != NEW_CYCLE_IDS:
        raise CHMPreexecutionFailure("New cycle IDs are not the frozen five complete Development cycles.")
    if not bool((new_cycles["end_ophrs2_h"] < BOUNDARY_H).all()):
        raise CHMPreexecutionFailure("New cycles cross the Development/Test boundary.")
    records: list[dict[str, Any]] = []
    for cycle in new_cycles.itertuples(index=False):
        subset = states.loc[
            (states["run_id"] >= int(cycle.start_run_id))
            & (states["run_id"] < int(cycle.end_run_id_exclusive))
            & (states["state_token"].astype(str) == P1_TOKEN)
            & (pd.to_numeric(states["duration_wall_s"], errors="coerce") >= 120.0)
            & (~_truthy(states["stop_or_zero_current_flag"]))
        ].copy()
        if subset.empty:
            raise CHMPreexecutionFailure(f"No legal P1S dwell in COMPLETE_{int(cycle.raw_cycle_id):03d}.")
        for run in subset.itertuples(index=False):
            start = pd.Timestamp(run.start_datetime)
            duration = float(run.duration_wall_s)
            records.append(
                {
                    "raw_cycle_id": int(cycle.raw_cycle_id),
                    "age_stratum": str(cycle.age_stratum),
                    "run_id": int(run.run_id),
                    "window_start": start + pd.to_timedelta(duration * 0.25, unit="s"),
                    "window_end": start + pd.to_timedelta(duration * 0.75, unit="s"),
                    "measurement_end_datetime": str(run.end_datetime),
                    "measurement_end_ophrs2_h": float(run.end_ophrs2_h),
                }
            )
    return pd.DataFrame(records).sort_values(["raw_cycle_id", "run_id"], kind="stable").reset_index(drop=True)


def build_new_complete_hou(catalog: pd.DataFrame, states: pd.DataFrame, reference_voltage: float) -> pd.DataFrame:
    """Build only COMPLETE_152 through COMPLETE_156 from a parser-bounded prefix."""
    target = _new_cycle_target_states(catalog, states)
    raw = safe_read_development_raw_prefix(usecols=RAW_USECOLS)
    raw["DateTime"] = pd.to_datetime(raw["DateTime"], errors="coerce")
    raw["operation_point"] = normalize_operation(raw["operation_point"])
    for column in ("I.S.set [A]", *ACTIVE_CELLS, *CONDITION_SOURCES.values()):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw["DateTime"].isna().any():
        raise CHMPreexecutionFailure("Development prefix has invalid DateTime in required rows.")
    dwell_records: list[dict[str, Any]] = []
    for window in target.itertuples(index=False):
        selected = raw.loc[(raw["DateTime"] >= window.window_start) & (raw["DateTime"] <= window.window_end)].copy()
        selected = selected.loc[
            (selected["operation_point"] == P1_OPERATION_POINT)
            & np.isclose(selected["I.S.set [A]"], P1_SET_CURRENT, atol=0.0005, rtol=0.0)
        ]
        if selected.empty:
            raise CHMPreexecutionFailure(f"P1S middle-50 raw selection empty for run {window.run_id}.")
        mean_voltage = np.nanmean(selected.loc[:, ACTIVE_CELLS].to_numpy(dtype=float), axis=1)
        record: dict[str, Any] = {
            "raw_cycle_id": int(window.raw_cycle_id),
            "age_stratum": window.age_stratum,
            "run_id": int(window.run_id),
            "raw_sample_count": int(len(selected)),
            "measurement_end_datetime": window.measurement_end_datetime,
            "measurement_end_ophrs2_h": float(window.measurement_end_ophrs2_h),
            "P1S_voltage_V": float(np.nanmedian(mean_voltage)) if np.isfinite(mean_voltage).any() else np.nan,
        }
        for field, source_column in CONDITION_SOURCES.items():
            values = selected[source_column].to_numpy(dtype=float)
            record[field] = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
        dwell_records.append(record)
    dwell = pd.DataFrame(dwell_records)
    records: list[dict[str, Any]] = []
    for raw_cycle_id, group in dwell.groupby("raw_cycle_id", sort=True):
        p1s = pd.to_numeric(group["P1S_voltage_V"], errors="coerce")
        if p1s.isna().any():
            raise CHMPreexecutionFailure(f"Missing P1S voltage in COMPLETE_{int(raw_cycle_id):03d}.")
        row: dict[str, Any] = {
            "hou_id": f"COMPLETE_CYCLE_HOU_{int(raw_cycle_id):03d}",
            "hou_kind": "COMPLETE_CYCLE_HOU",
            "source_interval_id": f"COMPLETE_{int(raw_cycle_id):03d}",
            "raw_cycle_id": int(raw_cycle_id),
            "age_stratum": str(group["age_stratum"].iloc[0]),
            "stable_window": "middle_50",
            "eligible_dwell_count": int(len(group)),
            "valid_dwell_count": int(p1s.notna().sum()),
            "measurement_end_ophrs2_h": float(group["measurement_end_ophrs2_h"].max()),
            "measurement_end_datetime": str(pd.to_datetime(group["measurement_end_datetime"]).max()),
            "P1S_voltage_V": float(p1s.median()),
            "reference_voltage_V": float(reference_voltage),
            "P1S_operating_voltage_retention": float(p1s.median() / reference_voltage),
        }
        for field in CONDITION_FIELDS:
            values = pd.to_numeric(group[field], errors="coerce")
            row[field] = float(values.median()) if values.notna().any() else np.nan
        records.append(row)
    result = pd.DataFrame(records)
    if tuple(result["raw_cycle_id"].astype(int)) != NEW_CYCLE_IDS:
        raise CHMPreexecutionFailure("New HOU builder did not produce exactly cycles 152-156.")
    return result


def _reuse_existing_hou(complete: pd.DataFrame, partial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = complete.loc[complete["stable_window"].eq("middle_50")].copy()
    if len(existing) != 141:
        raise CHMPreexecutionFailure("Expected exactly 141 hash-frozen PAA complete HOU.")
    existing["raw_cycle_id"] = pd.to_numeric(existing["raw_cycle_id"], errors="raise").astype(int)
    existing["hou_id"] = existing["raw_cycle_id"].map(lambda value: f"COMPLETE_CYCLE_HOU_{value:03d}")
    existing["hou_kind"] = "COMPLETE_CYCLE_HOU"
    existing["source_interval_id"] = existing["cycle_interval_id"].astype(str)
    existing = existing.loc[
        :, [
            "hou_id", "hou_kind", "source_interval_id", "raw_cycle_id", "age_stratum", "stable_window",
            "eligible_dwell_count", "valid_dwell_count", "measurement_end_ophrs2_h", "measurement_end_datetime",
            "P1S_voltage_V", "reference_voltage_V", "P1S_operating_voltage_retention", *CONDITION_FIELDS,
        ]
    ]
    frozen_partial = partial.loc[partial["stable_window"].eq("middle_50")].copy()
    if set(frozen_partial["cycle_interval_id"].astype(str)) != set(EXPECTED_PARTIAL_IDS) or len(frozen_partial) != 6:
        raise CHMPreexecutionFailure("Frozen six partial HOU source is not exact.")
    frozen_partial["hou_id"] = "PREANCHOR_PARTIAL_HOU_" + frozen_partial["nominal_checkpoint_h"].astype(int).astype(str)
    frozen_partial["hou_kind"] = "PREANCHOR_PARTIAL_HOU"
    frozen_partial["source_interval_id"] = frozen_partial["cycle_interval_id"].astype(str)
    frozen_partial = frozen_partial.loc[
        :, [
            "hou_id", "hou_kind", "source_interval_id", "raw_cycle_id", "age_stratum", "stable_window",
            "eligible_dwell_count", "valid_dwell_count", "measurement_end_ophrs2_h", "measurement_end_datetime",
            "P1S_voltage_V", "reference_voltage_V", "P1S_operating_voltage_retention", *CONDITION_FIELDS,
        ]
    ]
    return existing, frozen_partial


def build_chm_hou_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the 152-row final HOU table without recalculating PAA's 141 rows."""
    catalog = safe_read_frozen_artifact("rcr_cycle_catalog")
    states = safe_read_frozen_artifact("rcr_state_runs")
    reference = safe_read_frozen_artifact("paa_reference_cycles")
    complete = safe_read_frozen_artifact("paa_existing_complete_hou")
    partial = safe_read_frozen_artifact("paa_partial_hou")
    ref_values = pd.to_numeric(reference["reference_voltage_V"], errors="coerce").dropna().unique()
    if len(ref_values) != 1:
        raise CHMPreexecutionFailure("Frozen PAA reference voltage is not unique.")
    existing, frozen_partial = _reuse_existing_hou(complete, partial)
    new = build_new_complete_hou(catalog, states, float(ref_values[0]))
    table = pd.concat([existing, new, frozen_partial], ignore_index=True)
    table["measurement_end_datetime"] = pd.to_datetime(table["measurement_end_datetime"], errors="coerce")
    table = table.sort_values("measurement_end_datetime", kind="stable").reset_index(drop=True)
    table["measurement_end_datetime"] = table["measurement_end_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    validate_hou_table(table, catalog, safe_read_frozen_artifact("paa_anchor_alignment"))
    return table, new, catalog


def validate_hou_table(table: pd.DataFrame, catalog: pd.DataFrame, alignment: pd.DataFrame) -> None:
    """Static complete-HOU, partial-HOU, chronology, and anchor-membership gate."""
    if len(table) != 152:
        raise CHMPreexecutionFailure("CHM_HOU_TABLE_NOT_QUALIFIED: total row count")
    complete = table.loc[table["hou_kind"].eq("COMPLETE_CYCLE_HOU")]
    partial = table.loc[table["hou_kind"].eq("PREANCHOR_PARTIAL_HOU")]
    if len(complete) != 146 or len(partial) != 6:
        raise CHMPreexecutionFailure("CHM_HOU_TABLE_NOT_QUALIFIED: HOU-kind counts")
    if table["hou_id"].duplicated().any():
        raise CHMPreexecutionFailure("CHM_HOU_TABLE_NOT_QUALIFIED: duplicate HOU ID")
    frozen_complete_ids = set(pd.to_numeric(catalog.loc[_truthy(catalog["complete_cycle"]), "raw_cycle_id"], errors="raise").astype(int))
    observed_complete_ids = set(pd.to_numeric(complete["raw_cycle_id"], errors="raise").astype(int))
    if observed_complete_ids != frozen_complete_ids:
        raise CHMPreexecutionFailure("CHM_HOU_TABLE_NOT_QUALIFIED: complete raw-cycle set")
    if set(partial["source_interval_id"].astype(str)) != set(EXPECTED_PARTIAL_IDS):
        raise CHMPreexecutionFailure("CHM_HOU_TABLE_NOT_QUALIFIED: partial interval set")
    times = pd.to_datetime(table["measurement_end_datetime"], errors="coerce")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise CHMPreexecutionFailure("CHM_HOU_TABLE_NOT_QUALIFIED: strict datetime order")
    if not np.isfinite(pd.to_numeric(table["P1S_operating_voltage_retention"], errors="coerce")).all():
        raise CHMPreexecutionFailure("CHM_HOU_TABLE_NOT_QUALIFIED: nonfinite retention")
    for field in CONDITION_FIELDS:
        if field not in table or not np.isfinite(pd.to_numeric(table[field], errors="coerce")).all():
            raise CHMPreexecutionFailure(f"CHM_HOU_TABLE_NOT_QUALIFIED: condition {field}")
    present_sources = set(table["source_interval_id"].astype(str))
    summaries = alignment.loc[alignment["record_type"].eq("anchor_summary")]
    selected = set()
    for value in summaries["selected_cycle_interval_ids"].dropna():
        selected.update(str(value).split("|"))
    if len(summaries) != 6 or not selected.issubset(present_sources):
        raise CHMPreexecutionFailure("CHM_HOU_TABLE_NOT_QUALIFIED: frozen anchor interval membership")


def build_hou_provenance(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paa_complete = "RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv"
    paa_partial = "RENew-PEMFC-2300h-Pytorch/Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/PREANCHOR_PARTIAL_CYCLE_INTERVALS.csv"
    for row in table.itertuples(index=False):
        raw_id = None if pd.isna(row.raw_cycle_id) else int(row.raw_cycle_id)
        if row.hou_kind == "COMPLETE_CYCLE_HOU" and raw_id in NEW_CYCLE_IDS:
            source = "NEW_DEVELOPMENT_COMPLETION"
            reused = "NEW_DEVELOPMENT_COMPLETION"
            source_file = "Loadcycle_151-162.csv"
            raw_rows = PREFIX_ROWS_151_162
        elif row.hou_kind == "COMPLETE_CYCLE_HOU":
            source = "REUSED_FROZEN_PAA"
            reused = "REUSED_FROZEN_PAA"
            source_file = paa_complete
            raw_rows = 0
        else:
            source = "REUSED_FROZEN_PAA_PARTIAL"
            reused = "REUSED_FROZEN_PAA_PARTIAL"
            source_file = paa_partial
            raw_rows = 0
        rows.append(
            {
                "hou_id": row.hou_id,
                "source_interval_id": row.source_interval_id,
                "source": source,
                "reused_or_new": reused,
                "raw_cycle_id": raw_id,
                "source_file": source_file,
                "raw_rows_parsed": raw_rows,
                "P1S_dwell_count": int(row.valid_dwell_count),
                "stable_window": row.stable_window,
                "reference_source": "PAA_P1S_REFERENCE_COMPLETE_CYCLES.csv",
                "Test_rows_parsed": 0,
            }
        )
    return pd.DataFrame(rows)


def run_firewall_integration() -> dict[str, bool]:
    """Exercise actual allowlisted reader functions, including rejection before I/O."""
    results: dict[str, bool] = {}
    try:
        artifact = _artifact_entry("rcr_cycle_catalog")[1]
        safe_read_frozen_artifact("rcr_cycle_catalog", artifact)
        results["allowed_dev_artifact_tests_passed"] = True
        results["allowed_project_path_with_2300h_name_passed"] = "PEMFC-2300h-Pytorch" in str(artifact)
    except Exception:
        results["allowed_dev_artifact_tests_passed"] = False
        results["allowed_project_path_with_2300h_name_passed"] = False
    try:
        safe_read_development_raw_prefix(usecols=["OpHrs2 [h]"])
        results["development_raw_prefix_test_passed"] = True
    except Exception:
        results["development_raw_prefix_test_passed"] = False
    polar_ok = True
    for name in allowlist_config()["development_polarization"]["allowed_files"]:
        try:
            safe_read_development_polarization(name, usecols=["DateTime", "j.S.act [A/cm2]"])
        except Exception:
            polar_ok = False
    results["development_polarization_allowlist_passed"] = polar_ok

    def rejected(callable_: Any) -> bool:
        try:
            callable_()
        except CHMFirewallFailure:
            return True
        return False

    raw = _resolved(allowlist_config()["development_raw_source"]["path"])
    results["row_268685_rejected"] = rejected(lambda: safe_read_development_raw_prefix(requested_rows=PREFIX_ROWS_151_162 + 1))
    results["2000_pol_rejected"] = rejected(lambda: safe_read_development_polarization("Polcurve_Nominal_RHFC_2000h.csv"))
    results["2300_pol_rejected"] = rejected(lambda: safe_read_development_polarization("Polcurve_Nominal_RHFC_2300h.csv"))
    results["after_air_air_rejected"] = rejected(lambda: safe_read_development_polarization("Polcurve_Nominal_RHFC_1500h_afterAirAirStart.csv"))
    results["test_eis_rejected"] = rejected(lambda: safe_read_development_polarization("EIS_RHFC_2300h.csv"))
    results["unregistered_raw_rejected"] = rejected(lambda: safe_read_development_raw_prefix(raw.with_name("Loadcycle_134-150.csv")))
    return results


def write_frozen_input_hashes() -> dict[str, Any]:
    """Freeze every actual CHM input/output dependency, never the mixed raw file."""
    registry = allowlist_config()
    records: dict[str, dict[str, Any]] = {}
    for name, entry in registry["repository_frozen_artifacts"].items():
        records[name] = {"path": entry["relative_path"], "sha256": entry["sha256"]}
    local_files = {
        "completed_CHM_HOU_table_152_rows": INPUTS / "CHM_HOU_TABLE.csv",
        "newly_completed_HOU_152_156": INPUTS / "CHM_NEW_COMPLETE_HOU_152_156.csv",
        "development_polarization_structure_metadata": OUTPUTS / "POLARIZATION_STRUCTURE_AUDIT.csv",
        "final_source_allowlist": ALLOWLIST,
        "final_protocol": CONFIG,
    }
    for name, path in local_files.items():
        records[name] = {"path": str(path.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(path)}
    payload = {
        "phase": PHASE.name,
        "raw_source_not_full_hashed": "Loadcycle_151-162.csv contains Test rows and is accessed only by the frozen parser prefix.",
        "records": records,
    }
    with INPUT_HASHES.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return payload


def verify_frozen_input_hashes() -> dict[str, Any]:
    """Recheck every frozen record after writing the input-hash manifest."""
    manifest = load_yaml(INPUT_HASHES)
    records = manifest.get("records", {})
    required = {
        "rcr_cycle_catalog", "rcr_state_runs", "rcr_timeline_audit", "paa_reference_cycles",
        "paa_existing_complete_hou", "paa_partial_hou", "paa_anchor_alignment", "pav_anchor_provenance",
        "completed_CHM_HOU_table_152_rows", "newly_completed_HOU_152_156",
        "development_polarization_structure_metadata", "final_source_allowlist", "final_protocol",
    }
    if set(records) != required:
        raise CHMPreexecutionFailure("FROZEN_INPUT_HASH_MANIFEST_SCHEMA_MISMATCH")
    for name, entry in records.items():
        path = (REPO / entry["path"]).resolve()
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise CHMPreexecutionFailure(f"FROZEN_INPUT_HASH_MISMATCH: {name}")
    return manifest


def final_preexecution_audit() -> dict[str, Any]:
    """Build auditable artifacts, run static/firewall gates, and write no model outputs."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    INPUTS.mkdir(parents=True, exist_ok=True)
    protocol = final_protocol()
    polarization = audit_polarization_structure()
    polarization.to_csv(OUTPUTS / "POLARIZATION_STRUCTURE_AUDIT.csv", index=False)
    hou, new_hou, catalog = build_chm_hou_table()
    hou.to_csv(INPUTS / "CHM_HOU_TABLE.csv", index=False)
    new_hou.to_csv(INPUTS / "CHM_NEW_COMPLETE_HOU_152_156.csv", index=False)
    provenance = build_hou_provenance(hou)
    provenance.to_csv(OUTPUTS / "CHM_HOU_BUILD_AUDIT.csv", index=False)
    firewall = run_firewall_integration()
    with (OUTPUTS / "CHM_FIREWALL_INTEGRATION_AUDIT.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(firewall, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    input_hashes = write_frozen_input_hashes()
    verify_frozen_input_hashes()
    actual_complete = int(hou["hou_kind"].eq("COMPLETE_CYCLE_HOU").sum())
    actual_partial = int(hou["hou_kind"].eq("PREANCHOR_PARTIAL_HOU").sum())
    all_firewall = all(firewall.values())
    status = (
        "FINAL_PROTOCOL_FROZEN_AWAITING_EXECUTION_COMMAND"
        if all_firewall and actual_complete == 146 and actual_partial == 6 and len(polarization) == 7
        else "CHM_FINAL_FREEZE_BLOCKED"
    )
    result = {
        "phase": PHASE.name,
        "base_commit": protocol["base_commit"],
        "status": status,
        "training_executed": False,
        "LOAO_executed": False,
        "bootstrap_model_fitting_executed": False,
        "CHM_MAE_or_RMSE_generated": False,
        "test_polarization_accessed": False,
        "test_EIS_accessed": False,
        "polarization_explicit_independent_replicates": False,
        "polarization_uncertainty_semantics": "LOCAL_WINDOW_SCATTER_PROXY",
        "anchor_target_gaussian_perturbation": "DISABLED",
        "existing_complete_PAA_HOU": 141,
        "new_complete_Development_HOU": int(len(new_hou)),
        "new_cycle_ids": list(NEW_CYCLE_IDS),
        "final_complete_HOU": actual_complete,
        "partial_HOU": actual_partial,
        "total_HOU": int(len(hou)),
        "Loadcycle_151_162_contains_Test_rows": True,
        "development_prefix_parsed_rows": PREFIX_ROWS_151_162,
        "parsed_post_boundary_rows": 0,
        "used_post_boundary_rows": 0,
        "loss_profiles": 3,
        "inner_selection_fits": 270,
        "outer_deterministic_refits": 18,
        "firewall_integration": "PASS" if all_firewall else "FAIL",
        "frozen_input_hashes_generated": bool(input_hashes),
        "blocking_technical_decisions": "NO_BLOCKING_TECHNICAL_DECISIONS_REMAIN" if status.startswith("FINAL") else "TECHNICAL_FIX_REQUIRED",
        "next_action": "CHM_DEVELOPMENT_EXECUTION" if status.startswith("FINAL") else "TECHNICAL_FIX_REQUIRED",
        "rcr_complete_cycle_count": int(_truthy(catalog["complete_cycle"]).sum()),
    }
    with (OUTPUTS / "FINAL_PREEXECUTION_AUDIT.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if status != "FINAL_PROTOCOL_FROZEN_AWAITING_EXECUTION_COMMAND":
        raise CHMPreexecutionFailure("CHM_FINAL_FREEZE_BLOCKED")
    return result


def print_terminal_summary(result: dict[str, Any]) -> None:
    print("Phase CHM Final Feasibility and Freeze")
    print("======================================")
    print(f"Base commit: {result['base_commit']}")
    print("Training executed: NO")
    print("Polarization explicit independent replicates: NO")
    print(f"Polarization uncertainty semantics: {result['polarization_uncertainty_semantics']}")
    print("Anchor Gaussian perturbation: DISABLED")
    print("Existing complete PAA HOU: 141")
    print("New complete Development HOU: 5")
    print("New cycle IDs: 152,153,154,155,156")
    print("Final complete HOU: 146")
    print("Partial HOU: 6")
    print("Total HOU: 152")
    print("Loadcycle_151-162 contains Test rows: YES")
    print("Development prefix parsed rows: 268684")
    print("Post-boundary rows parsed: 0")
    print("Loss profiles: 3")
    print("Inner selection fits: 270")
    print(f"Firewall integration: {result['firewall_integration']}")
    print("Project parent directory containing 2300h false-positive: FIXED")
    print("Test polarization accessed: NO")
    print("Test EIS accessed: NO")
    print("Frozen input hashes: YES")
    print(f"Blocking technical decisions: {result['blocking_technical_decisions']}")
    print(f"Final status: {result['status']}")
    print(f"Next action: {result['next_action']}")


if __name__ == "__main__":
    print_terminal_summary(final_preexecution_audit())

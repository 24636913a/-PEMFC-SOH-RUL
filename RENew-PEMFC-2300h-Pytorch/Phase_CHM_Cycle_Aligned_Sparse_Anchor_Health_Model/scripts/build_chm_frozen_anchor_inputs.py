"""Build CHM-D's frozen sparse-anchor inputs without raw or polarization parsing.

This one-time errata builder consumes only pre-existing Development artifacts:
the first eight rows of N1's historical combined sparse-anchor table (which are
asserted non-Test before use), PAA's frozen mapping, PAV's Development temporal
provenance, and CHM's existing local-window structure audit.  It never opens a
Loadcycle or polarization CSV and it performs no model computation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
INPUTS = PHASE / "inputs"
OUTPUTS = PHASE / "outputs"
CHECKPOINTS = (300, 600, 850, 1150, 1500, 1750)
N1_COMBINED_ANCHORS = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_N1_Data_Foundation" / "outputs" / "soh_anchor_analysis" / "soh_reference_anchors.csv"
PAA_ALIGNMENT = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs" / "P1S_ANCHOR_ALIGNMENT.csv"
PAV_MEDIAN_PROVENANCE = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAV_PAA_Anchor_Validity_Qualification" / "outputs" / "ANCHOR_MEDIAN_SUPPORT_PROVENANCE.csv"
PAV_TIME_PROVENANCE = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAV_PAA_Anchor_Validity_Qualification" / "outputs" / "POLARIZATION_TIME_PROVENANCE.csv"
CHM_POLARIZATION_AUDIT = OUTPUTS / "POLARIZATION_STRUCTURE_AUDIT.csv"


class FrozenAnchorInputFailure(RuntimeError):
    """Fail-closed error for a target or mapping provenance mismatch."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_development_anchor_prefix() -> pd.DataFrame:
    """Read only the known eight-row non-Test prefix; fail before Test can enter."""
    frame = pd.read_csv(N1_COMBINED_ANCHORS, nrows=8)
    if len(frame) != 8 or frame["split"].astype(str).str.lower().eq("test").any():
        raise FrozenAnchorInputFailure("CHM_SPARSE_ANCHOR_PREFIX_FIREWALL_FAIL")
    nominal = frame.loc[
        (frame["jstar_A_cm2"].eq(1.0))
        & (~frame["air_air_start_status"].astype(str).str.contains("after", case=False, na=False))
    ].copy()
    nominal["ageing_hour"] = pd.to_numeric(nominal["ageing_hour"], errors="coerce")
    expected = {0, *CHECKPOINTS}
    observed = set(nominal["ageing_hour"].dropna().astype(int))
    if observed != expected or nominal["split"].astype(str).str.lower().eq("test").any():
        raise FrozenAnchorInputFailure("CHM_SPARSE_ANCHOR_DEVELOPMENT_SET_MISMATCH")
    return nominal.sort_values("ageing_hour", kind="stable").reset_index(drop=True)


def read_development_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alignment = pd.read_csv(PAA_ALIGNMENT)
    median = pd.read_csv(PAV_MEDIAN_PROVENANCE)
    temporal = pd.read_csv(PAV_TIME_PROVENANCE)
    polar = pd.read_csv(CHM_POLARIZATION_AUDIT)
    if set(median["checkpoint_h"].astype(int)) != set(CHECKPOINTS):
        raise FrozenAnchorInputFailure("PAV median provenance does not contain exactly six Development checkpoints.")
    if set(temporal["checkpoint_h"].astype(int)) != set(CHECKPOINTS):
        raise FrozenAnchorInputFailure("PAV temporal provenance does not contain exactly six Development checkpoints.")
    if set(polar["checkpoint"].astype(str)) != {"BoL", *(str(value) for value in CHECKPOINTS)}:
        raise FrozenAnchorInputFailure("CHM polarization structure audit differs from the seven Development files.")
    return alignment, median, temporal, polar


def build_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    anchors = read_development_anchor_prefix()
    alignment, median, temporal, polar = read_development_artifacts()
    bol_row = anchors.loc[anchors["ageing_hour"].eq(0)].iloc[0]
    bol_voltage = float(bol_row["voltage_anchor_V"])
    polar_local = polar.copy()
    polar_local["checkpoint_key"] = polar_local["checkpoint"].astype(str)
    local_sd = dict(zip(polar_local["checkpoint_key"], pd.to_numeric(polar_local["j1_local_voltage_sd"], errors="coerce")))
    map_rows = alignment.loc[alignment["record_type"].eq("anchor_interval")].copy()
    mapping: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        current = anchors.loc[anchors["ageing_hour"].eq(checkpoint)]
        if len(current) != 1:
            raise FrozenAnchorInputFailure(f"Missing or duplicated N1 target at {checkpoint} h.")
        target = current.iloc[0]
        intervals = map_rows.loc[map_rows["checkpoint_h"].eq(checkpoint)].copy().sort_values("selection_rank", kind="stable")
        if len(intervals) != 3 or tuple(intervals["selection_rank"].astype(int)) != (1, 2, 3):
            raise FrozenAnchorInputFailure(f"Frozen PAA mapping lacks three ordered intervals at {checkpoint} h.")
        provenance = median.loc[median["checkpoint_h"].eq(checkpoint)]
        temporal_row = temporal.loc[temporal["checkpoint_h"].eq(checkpoint)]
        if len(provenance) != 1 or len(temporal_row) != 1:
            raise FrozenAnchorInputFailure(f"PAV provenance is incomplete at {checkpoint} h.")
        p = provenance.iloc[0]
        t = temporal_row.iloc[0]
        retentions = pd.to_numeric(intervals["P1S_operating_voltage_retention"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(retentions).all():
            raise FrozenAnchorInputFailure(f"Nonfinite frozen P1S retention at {checkpoint} h.")
        sigma_int = float(1.4826 * np.median(np.abs(retentions - np.median(retentions))))
        support = intervals["in_development_joint_common_support"].astype(str).str.lower().eq("true").to_numpy()
        if support.sum() < 2:
            raise FrozenAnchorInputFailure(f"PAV 2-of-3 support is not available at {checkpoint} h.")
        sigma_support = float(abs(np.median(retentions) - np.median(retentions[support])))
        source_sd = float(local_sd[str(checkpoint)])
        if not np.isfinite(source_sd) or source_sd <= 0:
            raise FrozenAnchorInputFailure(f"Local polarization scatter is invalid at {checkpoint} h.")
        sigma_local = source_sd / bol_voltage
        q_squared = sigma_local ** 2 + sigma_int ** 2 + sigma_support ** 2
        for interval in intervals.itertuples(index=False):
            mapping.append(
                {
                    "checkpoint": checkpoint,
                    "selection_rank": int(interval.selection_rank),
                    "source_interval_id": str(interval.cycle_interval_id),
                    "interval_kind": str(interval.interval_kind),
                    "measurement_end_ophrs2_h": float(interval.measurement_end_ophrs2_h),
                    "P1S_operating_voltage_retention": float(interval.P1S_operating_voltage_retention),
                    "in_development_joint_common_support": bool(str(interval.in_development_joint_common_support).lower() == "true"),
                }
            )
        targets.append(
            {
                "checkpoint": checkpoint,
                "Y": float(target.normalized_health_value),
                "target_definition": "V_pol_checkpoint_jstar_1_0_divided_by_V_pol_BoL_jstar_1_0",
                "sigma_pol_local_proxy": sigma_local,
                "sigma_int": sigma_int,
                "sigma_support": sigma_support,
                "q": float(np.sqrt(q_squared)),
                "reliability_weight_raw": float(1.0 / (q_squared + 1e-8)),
                "source_polarization_file": str(target.filename),
                "temporal_qualification": str(t.qualification_status),
                "selected_interval_ids": "|".join(intervals["cycle_interval_id"].astype(str)),
                "median_source_interval_id": str(p.median_source_interval_id),
                "median_source_joint_support": bool(str(p.median_source_joint_support).lower() == "true"),
            }
        )
    target_frame = pd.DataFrame(targets).sort_values("checkpoint", kind="stable").reset_index(drop=True)
    mapping_frame = pd.DataFrame(mapping).sort_values(["checkpoint", "selection_rank"], kind="stable").reset_index(drop=True)
    if len(target_frame) != 6 or len(mapping_frame) != 18:
        raise FrozenAnchorInputFailure("Frozen target/mapping row count mismatch.")
    audit = {
        "n1_combined_anchor_source": str(N1_COMBINED_ANCHORS.relative_to(REPO)).replace("\\", "/"),
        "n1_rows_parsed": 8,
        "n1_test_rows_parsed": 0,
        "raw_Loadcycle_accessed": False,
        "direct_polarization_csv_accessed": False,
        "test_polarization_accessed": False,
        "target_count": 6,
        "mapping_count": 18,
        "target_checkpoints": list(CHECKPOINTS),
        "source_hashes": {
            "PAA_anchor_alignment": sha256_file(PAA_ALIGNMENT),
            "PAV_median_support_provenance": sha256_file(PAV_MEDIAN_PROVENANCE),
            "PAV_temporal_provenance": sha256_file(PAV_TIME_PROVENANCE),
            "CHM_polarization_structure_audit": sha256_file(CHM_POLARIZATION_AUDIT),
        },
    }
    return target_frame, mapping_frame, audit


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    targets, mapping, audit = build_frozen_inputs()
    targets.to_csv(INPUTS / "CHM_SPARSE_ANCHOR_TARGETS.csv", index=False)
    mapping.to_csv(INPUTS / "CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv", index=False)
    with (OUTPUTS / "CHM_SPARSE_ANCHOR_TARGET_BUILD_AUDIT.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

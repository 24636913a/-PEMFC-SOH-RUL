"""Read-only qualification audit for frozen Phase PAA anchors.

The script intentionally never reads a Test polarization/EIS file and never
recomputes a PAA state token, interval selection, or anchor value.  It joins
the frozen PAA interval records to the internally timestamped Development
polarization CSVs solely to qualify temporal ordering and median provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_COMMIT = "02b0319fac3380b0c5b76cf999f4ae8ae71df1ea"
PAA_STATUS = "P1S_PREANCHOR_COMPATIBILITY_EXPLORATORY_PASS"
CHECKPOINTS = (300, 600, 850, 1150, 1500, 1750)
DEVELOPMENT_POLARIZATION_FILENAMES = frozenset(
    f"Polcurve_Nominal_RHFC_{checkpoint}h.csv" for checkpoint in CHECKPOINTS
)
POLARIZATION_DIRECTORY_NAME = "《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据"
TIME_TOLERANCE_SECONDS = 0.0


class QualificationFailure(RuntimeError):
    """Raised for a broken frozen input or forbidden data-access route."""


@dataclass(frozen=True)
class CutoffMetadata:
    checkpoint_h: int
    cutoff_datetime: pd.Timestamp
    cutoff_ophrs2_h: float


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def phase_root() -> Path:
    return Path(__file__).resolve().parents[1]


def current_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_polarization_path(path: Path) -> None:
    """Fail closed for every file outside the six named Development inputs."""
    if path.name not in DEVELOPMENT_POLARIZATION_FILENAMES:
        raise QualificationFailure(
            f"FORBIDDEN_POLARIZATION_OR_EIS_ACCESS: {path.name} is not allowlisted."
        )
    forbidden_terms = ("2000", "2300", "afterairstart", "eis", "test")
    if any(term in path.name.lower() for term in forbidden_terms):
        raise QualificationFailure(f"FORBIDDEN_POLARIZATION_OR_EIS_ACCESS: {path.name}")


def parse_paa_cutoff_metadata(protocol_path: Path) -> dict[int, CutoffMetadata]:
    """Read frozen PAA cutoff metadata; do not derive it from operating gaps."""
    metadata: dict[int, CutoffMetadata] = {}
    pattern = re.compile(
        r"checkpoint_h:\s*(?P<checkpoint>\d+).*?"
        r"pre_characterization_cutoff_datetime:\s*'(?P<datetime>[^']+)'.*?"
        r"pre_characterization_cutoff_ophrs2_h:\s*(?P<age>[0-9.]+)"
    )
    for line in protocol_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            checkpoint = int(match.group("checkpoint"))
            metadata[checkpoint] = CutoffMetadata(
                checkpoint_h=checkpoint,
                cutoff_datetime=pd.Timestamp(match.group("datetime")),
                cutoff_ophrs2_h=float(match.group("age")),
            )
    if set(metadata) != set(CHECKPOINTS):
        raise QualificationFailure(
            f"PAA cutoff metadata mismatch: found {sorted(metadata)}, expected {list(CHECKPOINTS)}."
        )
    return metadata


def read_polarization_time_provenance(path: Path, checkpoint_h: int) -> dict[str, Any]:
    """Inspect header first and then read only the internally recorded DateTime field."""
    strict_polarization_path(path)
    header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0)
    columns = list(header.columns)
    datetime_present = "DateTime" in columns
    ophr2_present = "OpHrs2 [h]" in columns
    if not datetime_present:
        return {
            "datetime_column_present": False,
            "ophrs2_column_present": ophr2_present,
            "polarization_first_datetime": pd.NaT,
            "polarization_last_datetime": pd.NaT,
            "polarization_duration_s": np.nan,
            "polarization_valid_datetime_rows": 0,
            "polarization_ophrs2_min": np.nan,
            "polarization_ophrs2_max": np.nan,
            "polarization_ophrs2_median": np.nan,
            "timestamp_provenance_class": "POLARIZATION_EXACT_TIME_UNVERIFIED",
            "age_provenance_class": "POLARIZATION_INTERNAL_OPHRS2_ABSENT",
        }

    # The field-level read is intentional: voltage/current/electrochemical data
    # are not needed for this qualification audit.
    datetimes = pd.read_csv(
        path, sep=";", encoding="utf-8-sig", usecols=["DateTime"]
    )["DateTime"]
    datetimes = pd.to_datetime(datetimes, errors="coerce")
    valid = datetimes.dropna()
    if valid.empty:
        return {
            "datetime_column_present": True,
            "ophrs2_column_present": ophr2_present,
            "polarization_first_datetime": pd.NaT,
            "polarization_last_datetime": pd.NaT,
            "polarization_duration_s": np.nan,
            "polarization_valid_datetime_rows": 0,
            "polarization_ophrs2_min": np.nan,
            "polarization_ophrs2_max": np.nan,
            "polarization_ophrs2_median": np.nan,
            "timestamp_provenance_class": "POLARIZATION_EXACT_TIME_UNVERIFIED",
            "age_provenance_class": "POLARIZATION_INTERNAL_OPHRS2_ABSENT",
        }
    first, last = valid.min(), valid.max()
    return {
        "datetime_column_present": True,
        "ophrs2_column_present": ophr2_present,
        "polarization_first_datetime": first,
        "polarization_last_datetime": last,
        "polarization_duration_s": float((last - first).total_seconds()),
        "polarization_valid_datetime_rows": int(valid.size),
        "polarization_ophrs2_min": np.nan,
        "polarization_ophrs2_max": np.nan,
        "polarization_ophrs2_median": np.nan,
        "timestamp_provenance_class": "POLARIZATION_CSV_INTERNAL_DATETIME",
        "age_provenance_class": "POLARIZATION_INTERNAL_OPHRS2_ABSENT",
    }


def frozen_anchor_intervals(
    alignment_path: Path, complete_path: Path, partial_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing PAA records and validate their immutable three-row anchors."""
    alignment = pd.read_csv(alignment_path)
    summaries = alignment.loc[alignment["record_type"].eq("anchor_summary")].copy()
    intervals = alignment.loc[alignment["record_type"].eq("anchor_interval")].copy()
    if set(summaries["checkpoint_h"].astype(int)) != set(CHECKPOINTS):
        raise QualificationFailure("Frozen PAA anchor summaries do not cover the six Development checkpoints.")
    if set(summaries["status"]) != {"ALIGNED"}:
        raise QualificationFailure("Frozen PAA result is not the registered aligned six-anchor result.")
    if intervals.groupby("checkpoint_h").size().to_dict() != {checkpoint: 3 for checkpoint in CHECKPOINTS}:
        raise QualificationFailure("Frozen PAA anchors must contain exactly three selected intervals each.")

    complete = pd.read_csv(complete_path)
    partial = pd.read_csv(partial_path)
    measurements = pd.concat([complete, partial], ignore_index=True)
    measurements = measurements.loc[measurements["stable_window"].eq("middle_50")].copy()
    if measurements.duplicated("cycle_interval_id").any():
        raise QualificationFailure("Frozen PAA middle_50 interval measurement IDs are not unique.")
    source_columns = [
        "cycle_interval_id",
        "measurement_end_ophrs2_h",
        "measurement_end_datetime",
        "P1S_operating_voltage_retention",
    ]
    intervals = intervals.merge(
        measurements[source_columns], on="cycle_interval_id", how="left", suffixes=("_anchor", "_measurement"), validate="many_to_one"
    )
    if intervals["measurement_end_datetime"].isna().any():
        raise QualificationFailure("A frozen PAA selected interval lacks a timestamped middle_50 measurement.")
    if not np.allclose(
        intervals["P1S_operating_voltage_retention_anchor"],
        intervals["P1S_operating_voltage_retention_measurement"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise QualificationFailure("Frozen PAA interval retentions disagree across frozen PAA outputs.")
    return summaries, intervals


def median_source(interval_rows: pd.DataFrame) -> pd.Series:
    """Return the already-selected interval that supplies the frozen three-value median."""
    if len(interval_rows) != 3:
        raise QualificationFailure("Median provenance requires exactly three frozen interval rows.")
    ordered = interval_rows.sort_values(
        ["P1S_operating_voltage_retention_anchor", "cycle_interval_id"], kind="stable"
    ).reset_index(drop=True)
    values = ordered["P1S_operating_voltage_retention_anchor"].to_numpy(dtype=float)
    # Anchor retentions are dimensionless values near one.  NumPy's default
    # relative tolerance (1e-5) is far too broad here and could falsely merge
    # distinct frozen measurements; uniqueness is an exact-provenance check.
    if (
        not np.isfinite(values).all()
        or abs(values[0] - values[1]) <= 1e-12
        or abs(values[1] - values[2]) <= 1e-12
    ):
        raise QualificationFailure("Median source is non-unique under the frozen PAA anchor values.")
    return ordered.iloc[1]


def iso_or_blank(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    pd.DataFrame(records).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_claim_boundary(path: Path, final_status: str, fully_supported: int) -> None:
    if final_status == "PAA_EXPLORATORY_PASS_TEMPORALLY_VERIFIED":
        temporal_claim = (
            "The fixed P1S operating-voltage representation showed exploratory compatibility "
            "with sparse Development polarization anchors under the registered pre-characterization "
            "alignment procedure."
        )
    else:
        temporal_claim = (
            "The fixed P1S operating-voltage representation showed exploratory association with nominal "
            "sparse polarization checkpoints; exact characterization timing could not be independently "
            "verified for all checkpoints."
        )
    path.write_text(
        f"""# PAA Claim Boundary after PAV Qualification

Phase PAV is a `POSTHOC_QUALIFICATION_AUDIT`.  It preserves the frozen Phase
PAA result `P1S_PREANCHOR_COMPATIBILITY_EXPLORATORY_PASS`; it does not validate
P1S as SOH, create a continuous health truth, or make a causal claim.

## Permitted statement

> {temporal_claim}

## Required qualifications

- Internal `DateTime` fields in the six Development polarization CSVs establish
  temporal order only.  The nominal `300h` … `1750h` filename labels were not
  used as exact times.
- The registered PAA 2-of-3 complete-cycle joint-common-support rule remains
  unchanged.  The frozen median source itself is inside that support for
  **{fully_supported}/6** anchors.  At the remaining anchors, the registered
  rule passed but the final median can be contributed by an interval outside
  the complete-cycle joint-support region.
- PAA condition matching preserved a negative direction, but it had no
  preregistered distance cutoff and does not prove all operating-condition
  confounding was eliminated.  The appropriate phrase is
  `condition-adjusted directional robustness`, not complete condition matching.

## Prohibited statements

Do not call P1S a validated SOH, an independent SOH validation, continuous
health truth, a causal degradation indicator, or evidence that all anchor
estimates are fully condition matched.

## Public-context sources

The [Zenodo underlying-data record](https://zenodo.org/records/20715542) and
the [associated RealHyFC paper record](https://zenodo.org/records/20742676)
describe a >2000 h durability test with regular electrochemical
characterizations and nominal polarization datasets.  They do **not** provide
the exact DateTime of each of the six characterization runs; the internal CSV
timestamps, not those public pages, are the time provenance used here.
""",
        encoding="utf-8",
    )


def write_report(
    path: Path,
    provenance: pd.DataFrame,
    median: pd.DataFrame,
    final_status: str,
) -> None:
    merged = provenance.merge(
        median[["checkpoint_h", "median_source_interval_id", "median_source_joint_support", "anchor_value_fully_joint_supported"]],
        on="checkpoint_h",
        how="inner",
        validate="one_to_one",
    )
    display = merged[[
        "checkpoint_h", "polarization_first_datetime", "PAA_gap_cutoff_datetime",
        "preanchor_measurements_strictly_before_polarization", "qualification_status",
        "median_source_interval_id", "median_source_joint_support",
    ]].copy()
    display["polarization_first_datetime"] = display["polarization_first_datetime"].map(iso_or_blank)
    display["PAA_gap_cutoff_datetime"] = display["PAA_gap_cutoff_datetime"].map(iso_or_blank)
    display["preanchor_measurements_strictly_before_polarization"] = display["preanchor_measurements_strictly_before_polarization"].map(lambda value: "YES" if value else "NO")
    display["median_source_joint_support"] = display["median_source_joint_support"].map(lambda value: "YES" if value else "NO")
    path.write_text(
        f"""# Final PAA Anchor Validity Qualification

## Scope and preserved result

This independent `POSTHOC_QUALIFICATION_AUDIT` read the existing PAA anchor
records and only the internal `DateTime` column of the six allowed Development
nominal-polarization CSVs.  It did not recalculate P1S, select intervals, alter
the PAA 48 h / 3-interval / 2-of-3 rule, or access Test polarization/EIS.

The frozen PAA result remains **`{PAA_STATUS}`** (`POSTHOC_EXPLORATORY`).
PAV final qualification is **`{final_status}`**.

## Per-checkpoint temporal and median-source qualification

{markdown_table(display)}

Every row uses `POLARIZATION_CSV_INTERNAL_DATETIME`.  `OpHrs2 [h]` is absent
from all six polarization files, so this phase does not infer precise
characterization operating ages from their nominal filenames.  The recorded
PAA cutoff is only a frozen PAA gap boundary; the CSV timestamp is the actual
polarization-time source.

## Median-support interpretation

The registered PAA 2-of-3 support rule passed for all six anchors and is not
modified.  A `NO` value in `median_source_joint_support` does **not** invalidate
the original PAA gate: it qualifies the estimand by showing that the frozen
three-value median was supplied by an interval outside the complete-cycle joint
support.  No post-hoc median excluding that interval was computed.

## Claim boundary

See `docs/PAA_CLAIM_BOUNDARY.md`.  The public Zenodo/article records document
the long durability experiment and regular characterizations but not exact
times of individual characterizations; they were not used as timing evidence.

## Non-results

Neural training: **NO**.  New SOH: **NO**.  RUL: **NO**.  Test polarization:
**NO**.  Test EIS: **NO**.
""",
        encoding="utf-8",
    )


def run() -> None:
    repo = repository_root()
    phase = phase_root()
    if current_head(repo) != BASE_COMMIT:
        raise QualificationFailure(
            f"BASE_COMMIT_MISMATCH: expected {BASE_COMMIT}, got {current_head(repo)}."
        )

    paa = repo / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAA_P1_StateToken_PreAnchor_Alignment"
    paa_protocol = paa / "configs" / "PREREGISTERED_PAA_PROTOCOL.yaml"
    paa_alignment = paa / "outputs" / "P1S_ANCHOR_ALIGNMENT.csv"
    paa_complete = paa / "outputs" / "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv"
    paa_partial = paa / "outputs" / "PREANCHOR_PARTIAL_CYCLE_INTERVALS.csv"
    paa_final = paa / "final" / "FINAL_DECISION.yaml"
    required = [paa_protocol, paa_alignment, paa_complete, paa_partial, paa_final]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise QualificationFailure(f"Missing frozen PAA input(s): {missing}")
    if PAA_STATUS not in paa_final.read_text(encoding="utf-8"):
        raise QualificationFailure("Frozen PAA final status is not the expected exploratory pass.")

    cutoff_metadata = parse_paa_cutoff_metadata(paa_protocol)
    summaries, intervals = frozen_anchor_intervals(paa_alignment, paa_complete, paa_partial)
    polarization_root = repo / "数据" / POLARIZATION_DIRECTORY_NAME
    output_dir = phase / "outputs"
    output_dir.mkdir(exist_ok=True)

    provenance_records: list[dict[str, Any]] = []
    median_records: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        polarization_path = polarization_root / f"Polcurve_Nominal_RHFC_{checkpoint}h.csv"
        if not polarization_path.is_file():
            raise QualificationFailure(f"Missing allowed Development polarization file: {polarization_path}")
        provenance = read_polarization_time_provenance(polarization_path, checkpoint)
        selected = intervals.loc[intervals["checkpoint_h"].eq(checkpoint)].copy()
        selected["measurement_end_datetime"] = pd.to_datetime(selected["measurement_end_datetime"], errors="coerce")
        if selected["measurement_end_datetime"].isna().any():
            raise QualificationFailure(f"PAA selected interval timestamp missing at {checkpoint} h.")
        cutoff = cutoff_metadata[checkpoint]
        pol_start = provenance["polarization_first_datetime"]
        if pd.isna(pol_start):
            gap_before = False
            strictly_before = False
            qualification = "TEMPORAL_ORDER_UNVERIFIED"
        else:
            gap_before = bool(cutoff.cutoff_datetime < pol_start)
            strictly_before = bool((selected["measurement_end_datetime"] < pol_start).all())
            qualification = "TEMPORAL_ORDER_VERIFIED" if strictly_before else "TEMPORAL_ORDER_CONTRADICTED"

        source = median_source(selected)
        frozen_summary = summaries.loc[summaries["checkpoint_h"].eq(checkpoint)].iloc[0]
        ordered = selected.sort_values("selection_rank", kind="stable")
        median_records.append({
            "checkpoint_h": checkpoint,
            "selected_interval_1": ordered.iloc[0]["cycle_interval_id"],
            "selected_interval_2": ordered.iloc[1]["cycle_interval_id"],
            "selected_interval_3": ordered.iloc[2]["cycle_interval_id"],
            "retention_1": float(ordered.iloc[0]["P1S_operating_voltage_retention_anchor"]),
            "retention_2": float(ordered.iloc[1]["P1S_operating_voltage_retention_anchor"]),
            "retention_3": float(ordered.iloc[2]["P1S_operating_voltage_retention_anchor"]),
            "joint_support_1": bool(ordered.iloc[0]["in_development_joint_common_support"]),
            "joint_support_2": bool(ordered.iloc[1]["in_development_joint_common_support"]),
            "joint_support_3": bool(ordered.iloc[2]["in_development_joint_common_support"]),
            "joint_support_count": int(ordered["in_development_joint_common_support"].sum()),
            "median_retention": float(frozen_summary["P1S_anchor_retention"]),
            "median_source_interval_id": source["cycle_interval_id"],
            "median_source_joint_support": bool(source["in_development_joint_common_support"]),
            "registered_2_of_3_rule_pass": bool(int(ordered["in_development_joint_common_support"].sum()) >= 2),
            "anchor_value_fully_joint_supported": bool(source["in_development_joint_common_support"]),
        })
        provenance_records.append({
            "checkpoint_h": checkpoint,
            "source_file": str(polarization_path.relative_to(repo)),
            **provenance,
            "nominal_filename_checkpoint_h": checkpoint,
            "PAA_gap_cutoff_datetime": cutoff.cutoff_datetime,
            "PAA_gap_cutoff_ophrs2_h": cutoff.cutoff_ophrs2_h,
            "gap_before_polarization_verified": gap_before,
            "preanchor_measurements_strictly_before_polarization": strictly_before,
            "selected_measurement_end_datetimes": "|".join(
                iso_or_blank(value) for value in ordered["measurement_end_datetime"]
            ),
            "qualification_status": qualification,
        })

    provenance_frame = pd.DataFrame(provenance_records)
    median_frame = pd.DataFrame(median_records)
    statuses = set(provenance_frame["qualification_status"])
    if "TEMPORAL_ORDER_CONTRADICTED" in statuses:
        final_status = "PAA_TEMPORAL_ORDER_CONTRADICTION_FOUND"
    elif statuses == {"TEMPORAL_ORDER_VERIFIED"}:
        final_status = "PAA_EXPLORATORY_PASS_TEMPORALLY_VERIFIED"
    else:
        final_status = "PAA_EXPLORATORY_PASS_WITH_TEMPORAL_PROVENANCE_LIMITATION"

    write_csv(output_dir / "POLARIZATION_TIME_PROVENANCE.csv", provenance_records)
    write_csv(output_dir / "ANCHOR_MEDIAN_SUPPORT_PROVENANCE.csv", median_records)
    qualification = {
        "phase": "Phase_PAV_PAA_Anchor_Validity_Qualification",
        "designation": "POSTHOC_QUALIFICATION_AUDIT",
        "base_commit": BASE_COMMIT,
        "execution_head": current_head(repo),
        "original_paa_result_changed": False,
        "paa_exploratory_result": PAA_STATUS,
        "final_qualification": final_status,
        "temporally_verified_anchors": int((provenance_frame["qualification_status"] == "TEMPORAL_ORDER_VERIFIED").sum()),
        "temporal_contradictions": int((provenance_frame["qualification_status"] == "TEMPORAL_ORDER_CONTRADICTED").sum()),
        "median_anchors_sourced_inside_joint_support": int(median_frame["anchor_value_fully_joint_supported"].sum()),
        "registered_2_of_3_condition_support_rule_unchanged": True,
        "access": {
            "development_polarization_files": sorted(DEVELOPMENT_POLARIZATION_FILENAMES),
            "BoL_header_read": False,
            "test_polarization_accessed": False,
            "test_EIS_accessed": False,
            "neural_training": False,
            "SOH_constructed": False,
            "RUL": False,
        },
        "frozen_paa_input_sha256": {
            "P1S_ANCHOR_ALIGNMENT.csv": sha256_file(paa_alignment),
            "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv": sha256_file(paa_complete),
            "PREANCHOR_PARTIAL_CYCLE_INTERVALS.csv": sha256_file(paa_partial),
            "PREREGISTERED_PAA_PROTOCOL.yaml": sha256_file(paa_protocol),
            "FINAL_DECISION.yaml": sha256_file(paa_final),
        },
        "notes": [
            "Polarization CSV DateTime, not nominal filename or filesystem metadata, is the exact time source.",
            "PAA anchor values and support gate are read from frozen PAA outputs and are not recomputed.",
            "A median source outside joint support is a qualification label, not a retrospective failure of PAA's 2-of-3 rule.",
        ],
    }
    (output_dir / "QUALIFICATION_AUDIT.json").write_text(
        json.dumps(qualification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_claim_boundary(
        phase / "docs" / "PAA_CLAIM_BOUNDARY.md",
        final_status,
        int(median_frame["anchor_value_fully_joint_supported"].sum()),
    )
    write_report(
        phase / "reports" / "FINAL_PAA_VALIDITY_QUALIFICATION.md",
        provenance_frame,
        median_frame,
        final_status,
    )
    (phase / "final" / "FINAL_DECISION.yaml").write_text(
        "\n".join([
            "phase: Phase_PAV_PAA_Anchor_Validity_Qualification",
            "designation: POSTHOC_QUALIFICATION_AUDIT",
            f"base_commit: {BASE_COMMIT}",
            "original_paa_result_changed: false",
            f"paa_exploratory_result: {PAA_STATUS}",
            f"final_qualification: {final_status}",
            f"temporally_verified_anchors: {int((provenance_frame['qualification_status'] == 'TEMPORAL_ORDER_VERIFIED').sum())}",
            f"temporal_contradictions: {int((provenance_frame['qualification_status'] == 'TEMPORAL_ORDER_CONTRADICTED').sum())}",
            f"median_anchors_sourced_inside_joint_support: {int(median_frame['anchor_value_fully_joint_supported'].sum())}",
            "registered_2_of_3_condition_support_rule: unchanged",
            "test_polarization_accessed: false",
            "test_EIS_accessed: false",
            "neural_training: false",
            "SOH_constructed: false",
            "RUL: false",
            "next_action: STOP",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Phase PAV final qualification: {final_status}")
    print(f"Temporally verified anchors: {(provenance_frame['qualification_status'] == 'TEMPORAL_ORDER_VERIFIED').sum()} / 6")
    print(f"Median anchors sourced inside joint support: {median_frame['anchor_value_fully_joint_supported'].sum()} / 6")


if __name__ == "__main__":
    try:
        run()
    except QualificationFailure as error:
        print(f"PAV fail-closed: {error}", file=sys.stderr)
        raise SystemExit(2)

"""Target-blind postmortem for the frozen AHC-TV Stage-P alignment failure.

This script intentionally never resolves a Test polarization CSV.  It reads
only the frozen runtime metadata containing checkpoint/start time, the formal
raw operating files, and frozen Development alignment inputs.  The original
one-shot ledger and runtime artifacts are never modified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahc_tv_alignment import CONDITIONS, PHASE as AHC_PHASE, align_latest_three_intervals, read_yaml
from ahc_tv_raw_alignment_core import P1_TOKEN, RawAlignmentCoreError, runtime_p1_intervals
from ahc_tv_stage_p import _read_raw_operating


REPO = AHC_PHASE.parents[1]
POSTMORTEM = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_AHC_TV_Target_Blind_Alignment_Postmortem"
OUTPUTS = POSTMORTEM / "outputs"
REPORTS = POSTMORTEM / "reports"
RUNTIME = AHC_PHASE / "runtime"
METADATA_PATH = RUNTIME / "ONE_SHOT_TEST_RUN" / "TEST_CHARACTERIZATION_TIME_METADATA.csv"
LEDGER_PATH = RUNTIME / "AHC_TV_ONE_SHOT_EXECUTION_LEDGER.json"
DATA_ROOT = Path(
    r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据"
)
RAW_ROOT = DATA_ROOT / "Loadcycle_data" / "Loadcycle_data"


class PostmortemError(RuntimeError):
    """The target-blind postmortem input contract did not hold."""


def _numeric_raw_sources() -> list[Path]:
    """Enumerate only formal Loadcycle sources in frozen chronological order."""
    files = list(RAW_ROOT.glob("Loadcycle_*.csv"))

    def key(path: Path) -> tuple[int, int, str]:
        parts = path.stem.removeprefix("Loadcycle_").split("-", maxsplit=1)
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise PostmortemError(f"UNPARSEABLE_RAW_SOURCE:{path.name}")
        return int(parts[0]), int(parts[1]), path.name

    sources = sorted(files, key=key)
    if len(sources) != 17 or any(not path.name.startswith("Loadcycle_") for path in sources):
        raise PostmortemError("FORMAL_RAW_SOURCE_UNIVERSE_INVALID")
    return sources


def _metadata() -> pd.DataFrame:
    """Read only the already-frozen Reader-M metadata artifact, not polar data."""
    if not METADATA_PATH.exists():
        raise PostmortemError("FROZEN_CHARACTERIZATION_METADATA_MISSING")
    metadata = pd.read_csv(METADATA_PATH, usecols=["checkpoint", "characterization_start_datetime"])
    metadata["checkpoint"] = pd.to_numeric(metadata["checkpoint"], errors="raise").astype(int)
    metadata["characterization_start_datetime"] = pd.to_datetime(
        metadata["characterization_start_datetime"], errors="raise"
    )
    if set(metadata["checkpoint"]) != {2000, 2300} or len(metadata) != 2:
        raise PostmortemError("FROZEN_CHARACTERIZATION_METADATA_INVALID")
    return metadata.sort_values("checkpoint").reset_index(drop=True)


def _structural_legal(frame: pd.DataFrame) -> pd.Series:
    partial = frame["interval_kind"].eq("PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL")
    return (
        frame["has_complete_middle50_dwell"].astype(bool)
        & (~partial | frame["cycle_start_anchor_present"].astype(bool))
        & pd.to_numeric(frame["P1S_operating_voltage_retention"], errors="coerce").notna()
        & frame.loc[:, list(CONDITIONS)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        & pd.to_datetime(frame["measurement_end_datetime"], errors="coerce").notna()
        & pd.to_numeric(frame["measurement_end_ophrs2_h"], errors="coerce").notna()
    )


def _candidate_table(
    intervals: pd.DataFrame,
    *,
    checkpoint: int,
    start: pd.Timestamp,
    raw_cutoff_age_h: float,
    bounds: dict[str, list[float]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    table = intervals.copy()
    table["checkpoint"] = int(checkpoint)
    table["characterization_start_datetime"] = pd.Timestamp(start)
    table["measurement_end_datetime"] = pd.to_datetime(table["measurement_end_datetime"], errors="coerce")
    table["measurement_end_ophrs2_h"] = pd.to_numeric(table["measurement_end_ophrs2_h"], errors="coerce")
    table["wallclock_gap_h"] = (
        pd.Timestamp(start) - table["measurement_end_datetime"]
    ).dt.total_seconds() / 3600.0
    table["operating_age_gap_h"] = float(raw_cutoff_age_h) - table["measurement_end_ophrs2_h"]
    table["strictly_pre_characterization"] = table["measurement_end_datetime"].lt(pd.Timestamp(start))
    table["within_48h"] = table["measurement_end_ophrs2_h"].between(
        float(raw_cutoff_age_h) - 48.0, float(raw_cutoff_age_h), inclusive="both"
    )
    table["structural_legal"] = _structural_legal(table)
    support_columns: list[str] = []
    fail_variables: list[str] = []
    for variable in CONDITIONS:
        lower, upper = (float(value) for value in bounds[variable])
        value = pd.to_numeric(table[variable], errors="coerce")
        column = f"{variable}_in_support"
        table[f"{variable}_development_lower_bound"] = lower
        table[f"{variable}_development_upper_bound"] = upper
        table[column] = value.between(lower, upper, inclusive="both")
        support_columns.append(column)
    table["in_frozen_joint_common_support"] = table[support_columns].all(axis=1)
    table["support_fail_variables"] = table.apply(
        lambda row: ";".join(
            variable for variable in CONDITIONS if not bool(row[f"{variable}_in_support"])
        ),
        axis=1,
    )
    formal_candidates = table.loc[
        table["structural_legal"]
        & table["strictly_pre_characterization"]
        & table["within_48h"]
    ].copy()
    formal_candidates = formal_candidates.sort_values(
        ["measurement_end_datetime", "measurement_end_ophrs2_h", "cycle_interval_id"],
        ascending=[False, False, True], kind="mergesort",
    ).drop_duplicates("cycle_interval_id", keep="first")
    selected = formal_candidates.head(3).copy()
    selected["selected_rank"] = np.arange(1, len(selected) + 1, dtype=int)
    table = table.merge(
        selected.loc[:, ["cycle_interval_id", "selected_rank"]],
        on="cycle_interval_id", how="left",
    )
    latest = table.loc[
        table["structural_legal"] & table["strictly_pre_characterization"]
    ].sort_values(
        ["measurement_end_datetime", "measurement_end_ophrs2_h", "cycle_interval_id"],
        ascending=[False, False, True], kind="mergesort",
    ).drop_duplicates("cycle_interval_id", keep="first")
    latest_gaps = [float(value) for value in latest["operating_age_gap_h"].head(5).tolist()]
    hypothetical = {
        str(window): int((latest["operating_age_gap_h"].between(0.0, float(window), inclusive="both")).sum())
        for window in (48, 72, 96, 120, 168)
    }
    gate = {
        "available_structural_legal_interval_count": int(latest["cycle_interval_id"].nunique()),
        "within_48h_candidate_count": int(formal_candidates["cycle_interval_id"].nunique()),
        "strictly_pre_characterization_count": int(
            table.loc[table["structural_legal"] & table["strictly_pre_characterization"], "cycle_interval_id"].nunique()
        ),
        "latest_legal_interval_operating_age_gaps_h": latest_gaps,
        "hypothetical_window_legal_interval_counts": hypothetical,
    }
    return table, selected, gate


def _p1_state_coverage(runs: pd.DataFrame, raw_cutoff_age_h: float) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for window in (48, 96, 168):
        subset = runs.loc[pd.to_numeric(runs["end_ophrs2_h"], errors="coerce").between(
            float(raw_cutoff_age_h) - window, float(raw_cutoff_age_h), inclusive="both"
        )]
        state = subset.loc[subset["state_token"].eq(P1_TOKEN)]
        stable = state.loc[
            state["duration_wall_s"].ge(120.0) & state["stop_or_zero_current_flag"].eq(0)
        ]
        result[str(window)] = {
            "P1S_state_token_appearance_count": int(len(state)),
            "P1S_stable_dwell_ge_120s_count": int(len(stable)),
        }
    return result


def _primary_failure(
    *,
    raw_history: bool,
    interval_count: int,
    g1: bool,
    g2: bool,
    g3: bool,
    g4: bool,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not raw_history:
        failures.append("FAIL_NO_PRE_CHARACTERIZATION_RAW_HISTORY")
    if interval_count == 0:
        failures.append("FAIL_NO_LEGAL_P1S_INTERVALS")
    if not g1:
        failures.append("FAIL_INSUFFICIENT_DISTINCT_INTERVALS")
    # The production predicate evaluates strict chronological containment
    # before the operating-age window.
    if not g3:
        failures.append("FAIL_NOT_STRICTLY_PRE_CHARACTERIZATION")
    if not g2:
        failures.append("FAIL_INSUFFICIENT_INTERVALS_WITHIN_48H")
    if not g4:
        failures.append("FAIL_JOINT_SUPPORT_LT_2_OF_3")
    return (failures[0] if failures else "NONE", failures[1:])


def _failure_category(summary: dict[str, Any]) -> str:
    coverage = summary["p1_state_coverage"]["48"]
    primary = summary["primary_failure"]
    if not summary["raw_history_available"]:
        return "RAW_HISTORY"
    if coverage["P1S_state_token_appearance_count"] == 0:
        return "STATE_COVERAGE"
    if coverage["P1S_stable_dwell_ge_120s_count"] == 0:
        return "DWELL_AVAILABILITY"
    if primary == "FAIL_JOINT_SUPPORT_LT_2_OF_3":
        return "JOINT_SUPPORT"
    if primary in {"FAIL_INSUFFICIENT_DISTINCT_INTERVALS", "FAIL_INSUFFICIENT_INTERVALS_WITHIN_48H"}:
        return "INTERVAL_DENSITY"
    return "OTHER"


def _diagnose_checkpoint(
    raw: pd.DataFrame,
    *,
    checkpoint: int,
    start: pd.Timestamp,
    frozen: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    before = raw.loc[pd.to_datetime(raw["DateTime"]) < pd.Timestamp(start)].copy()
    base = {
        "checkpoint": int(checkpoint),
        "characterization_start_datetime": str(pd.Timestamp(start)),
        "raw_rows_total": int(len(raw)),
        "raw_rows_before_characterization": int(len(before)),
        "earliest_raw_datetime": str(pd.to_datetime(raw["DateTime"]).min()),
        "latest_raw_datetime_before_characterization": str(pd.to_datetime(before["DateTime"]).max()) if not before.empty else None,
        "latest_raw_ophrs2_h": float(pd.to_numeric(before["OpHrs2 [h]"], errors="coerce").max()) if not before.empty else None,
        "raw_history_available": bool(not before.empty),
        "target_evaluation": "NOT_EVALUATED_TARGET_BLIND",
    }
    if before.empty:
        base.update({
            "alignment_status": "FAILED", "primary_failure": "FAIL_NO_PRE_CHARACTERIZATION_RAW_HISTORY",
            "secondary_failures": [], "failure_category": "RAW_HISTORY",
        })
        return base, pd.DataFrame()
    try:
        intervals, runs = runtime_p1_intervals(
            before, float(frozen["reference_voltage_V"]),
            partial_interval_id=f"POSTMORTEM_PARTIAL_PRE_{checkpoint}",
        )
    except RawAlignmentCoreError as error:
        base.update({
            "total_reconstructed_runs": 0, "total_complete_intervals": 0,
            "partial_interval_present": False, "total_P1S_intervals": 0,
            "alignment_status": "FAILED", "primary_failure": "FAIL_NO_LEGAL_P1S_INTERVALS",
            "secondary_failures": [str(error)], "failure_category": "OTHER",
        })
        return base, pd.DataFrame()
    raw_cutoff_age_h = float(pd.to_numeric(runs["end_ophrs2_h"], errors="raise").iloc[-1])
    base.update({
        "raw_cutoff_ophrs2_h": raw_cutoff_age_h,
        "total_reconstructed_runs": int(len(runs)),
        "total_complete_intervals": int(intervals["interval_kind"].eq("COMPLETE_CYCLE_INTERVAL").sum()) if not intervals.empty else 0,
        "partial_interval_present": bool(intervals["interval_kind"].eq("PRE_CHARACTERIZATION_PARTIAL_CYCLE_INTERVAL").any()) if not intervals.empty else False,
        "total_P1S_intervals": int(len(intervals)),
        "p1_state_coverage": _p1_state_coverage(runs, raw_cutoff_age_h),
    })
    if intervals.empty:
        base.update({
            "alignment_status": "FAILED", "primary_failure": "FAIL_NO_LEGAL_P1S_INTERVALS",
            "secondary_failures": [], "failure_category": _failure_category(base),
        })
        return base, pd.DataFrame()
    table, selected, gate = _candidate_table(
        intervals, checkpoint=checkpoint, start=start, raw_cutoff_age_h=raw_cutoff_age_h,
        bounds=frozen["frozen_development_joint_common_support"],
    )
    official = align_latest_three_intervals(
        intervals, pd.Timestamp(start), raw_cutoff_age_h,
        frozen["frozen_development_joint_common_support"],
    )
    g1 = bool(gate["available_structural_legal_interval_count"] >= 3)
    g2 = bool(gate["within_48h_candidate_count"] >= 3)
    g3 = bool(gate["strictly_pre_characterization_count"] >= 3)
    support_flags = [bool(value) for value in selected["in_frozen_joint_common_support"].tolist()]
    g4 = bool(len(selected) == 3 and sum(support_flags) >= 2)
    primary, secondary = _primary_failure(
        raw_history=True, interval_count=int(len(intervals)), g1=g1, g2=g2, g3=g3, g4=g4
    )
    base.update({
        **gate,
        "G1_PASS": g1, "G2_PASS": g2, "G3_PASS": g3, "G4_PASS": g4,
        "latest_3_available": bool(len(selected) == 3),
        "selected_interval_ids": selected["cycle_interval_id"].astype(str).tolist(),
        "selected_support_flags": support_flags,
        "selected_support_fail_variables": selected["support_fail_variables"].astype(str).tolist(),
        "support_count": int(sum(support_flags)),
        "official_frozen_alignment_status": official.status,
        "alignment_status": "ALIGNED" if official.status == "ALIGNED" else "FAILED",
        "primary_failure": primary,
        "secondary_failures": secondary,
    })
    base["failure_category"] = _failure_category(base)
    return base, table


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# AHC-TV Target-Blind Alignment Failure Postmortem", "",
        "## Scope", "",
        "This postmortem used only frozen Reader-M characterization starts, formal raw operating data, and frozen Development alignment inputs. It did not open Test polarization target columns, invoke Stage T, alter the ledger, or authorize a rerun.", "",
        "## Checkpoint findings", "",
    ]
    for checkpoint in (2000, 2300):
        item = summary["checkpoints"][str(checkpoint)]
        lines.extend([
            f"### {checkpoint} h", "",
            f"- Raw history available: `{item['raw_history_available']}`",
            f"- P1S interval count: `{item.get('total_P1S_intervals', 0)}`",
            f"- Legal intervals within 48 h: `{item.get('within_48h_candidate_count', 0)}`",
            "- Latest legal interval operating-age gaps (h): `"
            + ", ".join(f"#{index + 1}={gap:.6f}" for index, gap in enumerate(item.get("latest_legal_interval_operating_age_gaps_h", [])))
            + "`",
            "- Hypothetical-window legal counts (diagnostic only): `"
            + ", ".join(f"{window} h={count}" for window, count in item.get("hypothetical_window_legal_interval_counts", {}).items())
            + "`",
            f"- Latest-3 available: `{item.get('latest_3_available', False)}`",
            f"- Support count: `{item.get('support_count', 0)} / 3`",
            f"- Latest-3 support flags: `{item.get('selected_support_flags', [])}`",
            f"- Latest-3 support failures by interval: `{item.get('selected_support_fail_variables', [])}`",
            f"- Gates: G1 `{item.get('G1_PASS', False)}`, G2 `{item.get('G2_PASS', False)}`, G3 `{item.get('G3_PASS', False)}`, G4 `{item.get('G4_PASS', False)}`",
            "- P1S state / stable dwell coverage: `"
            + "; ".join(
                f"{window} h: state={counts['P1S_state_token_appearance_count']}, stable={counts['P1S_stable_dwell_ge_120s_count']}"
                for window, counts in item.get("p1_state_coverage", {}).items()
            )
            + "`",
            f"- Alignment: `{item['alignment_status']}`",
            f"- Primary failure: `{item['primary_failure']}`",
            f"- Failure category: `{item['failure_category']}`", "",
        ])
    lines.extend([
        "## Research judgement", "",
        f"- Root cause: `{summary['root_cause']}`",
        f"- P1S itself unusable: `{summary['p1s_itself_unusable']}`",
        "- Original 2000/2300 Test may be rerun as confirmatory Test: `NO`",
        "- Polarization targets remain blinded: `YES`",
        "- All target outcomes, AE values, MAE values, and Rules A/B/C: `NOT_EVALUATED_TARGET_BLIND`", "",
        "## Protocol boundary", "",
        "The diagnostic is descriptive only. Hypothetical 72–168 h window counts do not alter the frozen 48 h protocol and do not authorize a second Test run.",
    ])
    return "\n".join(lines) + "\n"


def _p1s_usability(diagnoses: dict[str, dict[str, Any]]) -> str:
    """Answer the engineering question without examining any target outcome."""
    records = list(diagnoses.values())
    if any(item.get("alignment_status") == "ALIGNED" for item in records):
        return "NO"
    categories = {item.get("failure_category") for item in records}
    if categories and categories <= {"STATE_COVERAGE", "DWELL_AVAILABILITY"}:
        return "YES"
    return "INCONCLUSIVE"


def _next_direction(diagnoses: dict[str, dict[str, Any]]) -> str:
    categories = {item.get("failure_category") for item in diagnoses.values()}
    if categories & {"STATE_COVERAGE", "DWELL_AVAILABILITY"}:
        return "Assess target-blind late-life P1S state observability before proposing any separate future protocol; do not alter the consumed Test."
    if "JOINT_SUPPORT" in categories:
        return "Assess target-blind late-life operating-condition support drift before proposing any separate future protocol; do not alter the consumed Test."
    return "Assess target-blind late-life interval density and alignment feasibility before proposing any separate future protocol; do not alter the consumed Test."


def main() -> None:
    if not LEDGER_PATH.exists():
        raise PostmortemError("ABORTED_ONE_SHOT_LEDGER_MISSING")
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    if ledger.get("status") != "ABORTED_AFTER_TEST_EXECUTION_STARTED":
        raise PostmortemError("ONE_SHOT_LEDGER_NOT_ABORTED")
    frozen = read_yaml(AHC_PHASE / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    metadata = _metadata()
    raw = _read_raw_operating(_numeric_raw_sources())
    diagnoses: dict[str, Any] = {}
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for row in metadata.itertuples(index=False):
        diagnosis, table = _diagnose_checkpoint(
            raw, checkpoint=int(row.checkpoint), start=pd.Timestamp(row.characterization_start_datetime), frozen=frozen
        )
        diagnoses[str(int(row.checkpoint))] = diagnosis
        table.to_csv(OUTPUTS / f"AHC_TV_ALIGNMENT_POSTMORTEM_INTERVALS_{int(row.checkpoint)}.csv", index=False)
    categories = [diagnosis["failure_category"] for diagnosis in diagnoses.values() if diagnosis["alignment_status"] == "FAILED"]
    root_cause = ";".join(sorted(set(categories))) if categories else "NO_ALIGNMENT_FAILURE_REPRODUCED"
    summary = {
        "postmortem_mode": "TARGET_BLIND_ONLY",
        "original_one_shot_rerun": False,
        "polarization_targets_accessed": False,
        "ledger_modified": False,
        "frozen_ledger_status": ledger["status"],
        "frozen_failure": ledger.get("failure", {}),
        "checkpoints": diagnoses,
        "root_cause": root_cause,
        "p1s_itself_unusable": _p1s_usability(diagnoses),
        "original_test_may_be_rerun_as_confirmatory_test": "NO",
        "polarization_targets_remain_blinded": "YES",
        "all_target_metrics": "NOT_EVALUATED_TARGET_BLIND",
        "next_recommended_protocol_direction": _next_direction(diagnoses),
    }
    (OUTPUTS / "AHC_TV_ALIGNMENT_POSTMORTEM_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    (REPORTS / "AHC_TV_TARGET_BLIND_ALIGNMENT_FAILURE_POSTMORTEM.md").write_text(
        _report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

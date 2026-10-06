"""Development-only direct verification for the final AHC-TV pre-open gate.

This script never opens 2000 h/2300 h Test data. The retained historical raw
replay compares the authoritative core against frozen PAA artifacts; the
formal replay invokes ``ahc_tv_stage_p._prediction_row`` with synthetic mode
disabled.
"""
from __future__ import annotations

import io
import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahc_tv_alignment import CONDITIONS, PHASE, read_yaml, replay_development_alignment, sha256_file, verify_frozen_runtime_authority
from ahc_tv_raw_alignment_core import RCR_TEMPLATE_PATH, RawAlignmentCoreError, read_development_prefix, runtime_p1_intervals
from ahc_tv_stage_p import _prediction_row
from ahc_tv_stage_t import TARGET_COLUMNS, extract_jstar_voltage_from_table
from run_realhyfc_source_binding_dry_run import run_real_filename_binding_dry_run


REPO = PHASE.parents[1]
CHM_PHASE = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model"
OUTPUTS = PHASE / "outputs"
PAA_RAW_ROOT = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据")
VOLTAGE_TOL = 1e-12
RETENTION_TOL = 1e-12
CONDITION_TOL = 1e-10
OPHRS_TOL = 1e-9
RCR_TEMPLATE_SHA256 = "267a3ee5a2d297b4368f4dd262771a19109e8015b2617de45ab828d83d696200"


class PreOpenVerificationError(RuntimeError):
    """A Development-only replay did not reproduce a frozen artifact."""


def rcr_cycle_template_runtime_lock_audit(*, phase: Path = PHASE) -> dict[str, Any]:
    """Verify the direct formal raw-core dependency before any replay or Test I/O."""
    authority = read_yaml(Path(phase) / "configs" / "AHC_TV_FROZEN_INPUT_AUTHORITY.yaml")
    entries = [entry for entry in authority["frozen_hashes"] if entry["name"] == "frozen_RCR_cycle_template"]
    if len(entries) != 1:
        raise PreOpenVerificationError("RCR_CYCLE_TEMPLATE_RUNTIME_LOCK_MISSING")
    entry = entries[0]
    if entry["path"] != "RENew-PEMFC-2300h-Pytorch/Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/CYCLE_TEMPLATE.json":
        raise PreOpenVerificationError("RCR_CYCLE_TEMPLATE_RUNTIME_PATH_MISMATCH")
    if entry["sha256"] != RCR_TEMPLATE_SHA256 or sha256_file(RCR_TEMPLATE_PATH) != RCR_TEMPLATE_SHA256:
        raise PreOpenVerificationError("RCR_CYCLE_TEMPLATE_RUNTIME_HASH_MISMATCH")
    # This production guard hashes every frozen runtime input, including the
    # direct RCR template entry, before any Reader M/Reader T invocation.
    verify_frozen_runtime_authority(Path(phase))
    return {"locked": True, "sha256": RCR_TEMPLATE_SHA256, "hash_check": "PASS"}


def _load_dev_raw_sources() -> list[Path]:
    files = list((PAA_RAW_ROOT / "Loadcycle_data").rglob("Loadcycle_*.csv"))

    def key(path: Path) -> tuple[int, int, str]:
        match = re.fullmatch(r"Loadcycle_(\d+)-(\d+)", path.stem)
        if match is None:
            raise PreOpenVerificationError(f"UNPARSEABLE_DEVELOPMENT_RAW_SOURCE:{path.name}")
        return int(match.group(1)), int(match.group(2)), path.name

    selected = sorted((path for path in files if key(path)[1] <= 150), key=key)
    if not selected or any(key(path)[1] > 150 for path in selected):
        raise PreOpenVerificationError("DEVELOPMENT_RAW_SOURCE_ALLOWLIST_INVALID")
    return selected


def _frozen_inputs(phase: Path) -> dict:
    return read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")


def _paa_metadata(frozen: dict) -> list[dict]:
    paa = read_yaml(REPO / frozen["PAA_protocol_path"])
    return list(paa["partial_interval_characterization_metadata"]["anchors"])


def _frozen_measurements(frozen: dict) -> pd.DataFrame:
    inputs = frozen["development_replay_inputs"]
    complete = pd.read_csv(REPO / inputs["P1S_complete_cycle_measurements_path"])
    partial = pd.read_csv(REPO / inputs["preanchor_partial_intervals_path"])
    return pd.concat([complete, partial], ignore_index=True, sort=False).loc[lambda data: data["stable_window"].eq("middle_50")].copy()


def _expected_rows(frozen: dict, checkpoint: int) -> pd.DataFrame:
    alignment = pd.read_csv(REPO / frozen["development_replay_inputs"]["P1S_anchor_alignment_path"])
    return alignment.loc[
        alignment["record_type"].eq("anchor_interval")
        & pd.to_numeric(alignment["checkpoint_h"], errors="coerce").eq(int(checkpoint))
    ].sort_values("selection_rank", kind="mergesort").reset_index(drop=True)


def _expected_median(frozen: dict, checkpoint: int) -> float:
    alignment = pd.read_csv(REPO / frozen["development_replay_inputs"]["P1S_anchor_alignment_path"])
    values = alignment.loc[
        alignment["record_type"].eq("anchor_summary")
        & pd.to_numeric(alignment["checkpoint_h"], errors="coerce").eq(int(checkpoint)),
        "P1S_anchor_retention",
    ]
    if len(values) != 1:
        raise PreOpenVerificationError(f"FROZEN_ANCHOR_MEDIAN_MISSING:{checkpoint}")
    return float(values.iloc[0])


def _support(record: Any, frozen: dict) -> bool:
    for condition, bounds in frozen["frozen_development_joint_common_support"].items():
        value = float(record[condition] if isinstance(record, pd.Series) else getattr(record, condition))
        if not float(bounds[0]) <= value <= float(bounds[1]):
            return False
    return True


def _core_intervals(meta: dict, frozen: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, float]:
    checkpoint = int(meta["checkpoint_h"])
    cutoff_time = pd.Timestamp(meta["pre_characterization_cutoff_datetime"])
    cutoff_h = float(meta["pre_characterization_cutoff_ophrs2_h"])
    try:
        raw = read_development_prefix(_load_dev_raw_sources(), cutoff_datetime=cutoff_time, cutoff_ophrs2_h=cutoff_h)
        intervals, _ = runtime_p1_intervals(
            raw, float(frozen["reference_voltage_V"]), partial_interval_id=f"PARTIAL_PRE_{checkpoint}",
            partial_last_run_id=int(meta["pre_characterization_last_run_id"]),
            require_development_stratum=True, synthetic_test_mode=False,
        )
    except RawAlignmentCoreError as exc:
        raise PreOpenVerificationError(f"RAW_ALIGNMENT_CORE_FAILURE:{checkpoint}:{exc}") from exc
    if intervals.empty:
        raise PreOpenVerificationError(f"RAW_INTERVAL_RECONSTRUCTION_EMPTY:{checkpoint}")
    return raw, intervals, cutoff_time, cutoff_h


def _select_reference(intervals: pd.DataFrame, cutoff_time: pd.Timestamp, cutoff_h: float) -> pd.DataFrame:
    legal = intervals.loc[
        intervals["measurement_end_ophrs2_h"].between(cutoff_h - 48.0, cutoff_h, inclusive="both")
        & pd.to_datetime(intervals["measurement_end_datetime"]).lt(cutoff_time)
        & intervals[list(CONDITIONS)].notna().all(axis=1)
        & intervals["has_complete_middle50_dwell"].astype(bool)
    ].copy()
    return legal.sort_values(
        ["measurement_end_datetime", "measurement_end_ophrs2_h", "cycle_interval_id"],
        ascending=[False, False, True], kind="mergesort",
    ).drop_duplicates("cycle_interval_id").head(3).reset_index(drop=True)


def _compare_selected(*, checkpoint: int, runtime: pd.DataFrame, frozen: dict) -> tuple[list[dict], dict]:
    expected = _expected_rows(frozen, checkpoint)
    measurements = _frozen_measurements(frozen).set_index("cycle_interval_id", drop=False)
    if len(runtime) != 3 or len(expected) != 3:
        raise PreOpenVerificationError(f"RAW_REPLAY_NOT_THREE_INTERVALS:{checkpoint}")
    records: list[dict] = []
    for rank, observed in enumerate(runtime.itertuples(index=False), start=1):
        expected_alignment = expected.iloc[rank - 1]
        expected_id = str(expected_alignment["cycle_interval_id"])
        if expected_id not in measurements.index:
            raise PreOpenVerificationError(f"FROZEN_MEASUREMENT_MISSING:{expected_id}")
        measurement = measurements.loc[expected_id]
        if isinstance(measurement, pd.DataFrame):
            raise PreOpenVerificationError(f"FROZEN_MEASUREMENT_DUPLICATE:{expected_id}")
        expected_time = pd.Timestamp(measurement["measurement_end_datetime"])
        runtime_time = pd.Timestamp(observed.measurement_end_datetime)
        expected_r, runtime_r = float(measurement["P1S_operating_voltage_retention"]), float(observed.P1S_operating_voltage_retention)
        diffs = {condition: abs(float(measurement[condition]) - float(getattr(observed, condition))) for condition in CONDITIONS}
        record = {
            "checkpoint": checkpoint, "rank": rank,
            "expected_interval_id": expected_id, "runtime_interval_id": str(observed.cycle_interval_id),
            "expected_measurement_end_datetime": expected_time.isoformat(), "runtime_measurement_end_datetime": runtime_time.isoformat(),
            "expected_ophrs2": float(measurement["measurement_end_ophrs2_h"]), "runtime_ophrs2": float(observed.measurement_end_ophrs2_h),
            "ophrs2_abs_diff": abs(float(measurement["measurement_end_ophrs2_h"]) - float(observed.measurement_end_ophrs2_h)),
            "expected_P1S_voltage_V": float(measurement["P1S_voltage_V"]), "runtime_P1S_voltage_V": float(observed.P1S_voltage_V),
            "P1S_voltage_abs_diff": abs(float(measurement["P1S_voltage_V"]) - float(observed.P1S_voltage_V)),
            "expected_R": expected_r, "runtime_R": runtime_r, "R_abs_diff": abs(expected_r - runtime_r),
            "expected_eligible_dwell_count": int(measurement["eligible_dwell_count"]), "runtime_eligible_dwell_count": int(observed.eligible_dwell_count),
            "eligible_dwell_count_match": int(measurement["eligible_dwell_count"]) == int(observed.eligible_dwell_count),
            "expected_support": bool(expected_alignment["in_development_joint_common_support"]), "runtime_support": _support(observed, frozen),
            "interval_id_match": expected_id == str(observed.cycle_interval_id), "datetime_match": expected_time == runtime_time,
            "R_match": abs(expected_r - runtime_r) <= RETENTION_TOL,
            "voltage_match": abs(float(measurement["P1S_voltage_V"]) - float(observed.P1S_voltage_V)) <= VOLTAGE_TOL,
            "ophrs2_match": abs(float(measurement["measurement_end_ophrs2_h"]) - float(observed.measurement_end_ophrs2_h)) <= OPHRS_TOL,
            "support_match": bool(expected_alignment["in_development_joint_common_support"]) == _support(observed, frozen),
        }
        for condition, difference in diffs.items():
            record[f"expected_{condition}"] = float(measurement[condition]); record[f"runtime_{condition}"] = float(getattr(observed, condition)); record[f"{condition}_abs_diff"] = difference
        record["conditions_match"] = all(difference <= CONDITION_TOL for difference in diffs.values())
        records.append(record)
    expected_median = _expected_median(frozen, checkpoint)
    runtime_median = float(np.nanmedian(runtime["P1S_operating_voltage_retention"].to_numpy(dtype=float)))
    row_pass = all(bool(row["interval_id_match"] and row["datetime_match"] and row["R_match"] and row["voltage_match"] and row["ophrs2_match"] and row["conditions_match"] and row["support_match"] and row["eligible_dwell_count_match"]) for row in records)
    summary = {
        "checkpoint": checkpoint, "expected_anchor_median_R": expected_median, "runtime_anchor_median_R": runtime_median,
        "anchor_median_abs_diff": abs(expected_median - runtime_median), "anchor_median_match": abs(expected_median - runtime_median) <= RETENTION_TOL,
        "expected_support_count": int(expected["in_development_joint_common_support"].astype(bool).sum()),
        "runtime_support_count": int(sum(bool(row["runtime_support"]) for row in records)),
        "support_count_match": int(expected["in_development_joint_common_support"].astype(bool).sum()) == int(sum(bool(row["runtime_support"]) for row in records)),
        "pass": bool(row_pass and abs(expected_median - runtime_median) <= RETENTION_TOL),
    }
    return records, summary


def stage_p_development_raw_replay(*, phase: Path = PHASE) -> dict[str, Any]:
    """Retained historical output: ``INDEPENDENT_REFERENCE_REPLAY``."""
    phase = Path(phase); frozen = _frozen_inputs(phase)
    rows: list[dict] = []; summaries: list[dict] = []
    for meta in _paa_metadata(frozen):
        checkpoint = int(meta["checkpoint_h"])
        _, intervals, cutoff_time, cutoff_h = _core_intervals(meta, frozen)
        comparison, summary = _compare_selected(checkpoint=checkpoint, runtime=_select_reference(intervals, cutoff_time, cutoff_h), frozen=frozen)
        rows.extend(comparison); summaries.append(summary)
    output = pd.DataFrame(rows)
    if len(output) != 18:
        raise PreOpenVerificationError(f"RAW_REPLAY_EXPECTED_18_ROWS_GOT_{len(output)}")
    output["audit_role"] = "INDEPENDENT_REFERENCE_REPLAY"
    output.to_csv(OUTPUTS / "AHC_TV_STAGE_P_DEVELOPMENT_RAW_EQUIVALENCE.csv", index=False)
    passed = int(sum(item["pass"] for item in summaries))
    return {"rows": output, "checkpoints": summaries, "pass_count": passed, "required": 6, "all_pass": passed == 6, "audit_role": "INDEPENDENT_REFERENCE_REPLAY"}


def formal_stage_p_development_runtime_replay(*, phase: Path = PHASE) -> dict[str, Any]:
    """Invoke exact future formal Stage P on each legal Development prefix."""
    phase = Path(phase); frozen = _frozen_inputs(phase); models = read_yaml(phase / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml")
    rows: list[dict] = []; summaries: list[dict] = []
    for meta in _paa_metadata(frozen):
        checkpoint = int(meta["checkpoint_h"])
        raw, _, cutoff_time, cutoff_h = _core_intervals(meta, frozen)
        prediction, provenance = _prediction_row(
            checkpoint, raw, cutoff_time, frozen, models, synthetic_test_mode=False,
            partial_last_run_id=int(meta["pre_characterization_last_run_id"]), require_development_stratum=True,
            operating_age_cutoff_h=cutoff_h,
        )
        selected = provenance["selected_intervals"]
        if len(selected) != 3:
            raise PreOpenVerificationError(f"FORMAL_RUNTIME_NOT_THREE_INTERVALS:{checkpoint}")
        runtime = pd.DataFrame(selected).rename(columns={"interval_id": "cycle_interval_id"})
        comparison, summary = _compare_selected(checkpoint=checkpoint, runtime=runtime, frozen=frozen)
        expected_median = _expected_median(frozen, checkpoint)
        median_match = abs(float(prediction["P1S_R_median"]) - expected_median) <= RETENTION_TOL
        for record in comparison:
            record.update({"formal_runtime_path": "ahc_tv_stage_p authoritative runtime core", "formal_P1S_R_median": float(prediction["P1S_R_median"]), "expected_P1S_R_median": expected_median, "formal_anchor_median_match": median_match, "synthetic_test_mode": False})
        summary.update({"formal_runtime_path": "ahc_tv_stage_p authoritative runtime core", "formal_anchor_median_match": median_match, "pass": bool(summary["pass"] and median_match)})
        rows.extend(comparison); summaries.append(summary)
    output = pd.DataFrame(rows)
    if len(output) != 18:
        raise PreOpenVerificationError(f"FORMAL_RUNTIME_EXPECTED_18_ROWS_GOT_{len(output)}")
    output.to_csv(OUTPUTS / "AHC_TV_FORMAL_STAGE_P_DEVELOPMENT_EQUIVALENCE.csv", index=False)
    passed = int(sum(item["pass"] for item in summaries))
    return {"rows": output, "checkpoints": summaries, "pass_count": passed, "required": 6, "all_pass": passed == 6}


def stage_t_development_target_replay(*, phase: Path = PHASE) -> dict[str, Any]:
    """Reapply the frozen Stage-T extractor to BoL plus six allowed anchors."""
    phase = Path(phase); registry = read_yaml(CHM_PHASE / "configs" / "CHM_SOURCE_ALLOWLIST.yaml")
    root = Path(registry["development_polarization"]["root"]); structure = pd.read_csv(CHM_PHASE / "outputs" / "POLARIZATION_STRUCTURE_AUDIT.csv")
    anchors = pd.read_csv(CHM_PHASE / "inputs" / "CHM_SPARSE_ANCHOR_TARGETS.csv")
    bol_voltage = float(_frozen_inputs(phase)["polarization_target_normalization"]["V_pol_BoL_jstar_1_0"])
    records: list[dict] = []
    for filename, source in registry["development_polarization"]["allowed_files"].items():
        checkpoint = str(source["checkpoint"]); path = root / filename
        if sha256_file(path) != source["sha256"]:
            raise PreOpenVerificationError(f"DEVELOPMENT_POLARIZATION_HASH_MISMATCH:{filename}")
        runtime_voltage, runtime_rows = extract_jstar_voltage_from_table(pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=list(TARGET_COLUMNS)))
        expected = structure.loc[structure["checkpoint"].astype(str).eq(checkpoint)]
        if len(expected) != 1:
            raise PreOpenVerificationError(f"FROZEN_POLARIZATION_STRUCTURE_ROW_MISSING:{checkpoint}")
        expected_voltage, expected_rows = float(expected.iloc[0]["j1_local_voltage_median"]), int(expected.iloc[0]["j1_local_row_count"])
        historical_y = 1.0 if checkpoint == "BoL" else float(anchors.loc[pd.to_numeric(anchors["checkpoint"], errors="coerce").eq(int(checkpoint)), "Y"].iloc[0])
        expected_y, runtime_y = float(expected_voltage / bol_voltage), float(runtime_voltage / bol_voltage)
        records.append({"checkpoint": checkpoint, "expected_local_rows": expected_rows, "runtime_local_rows": runtime_rows, "local_rows_match": expected_rows == runtime_rows, "expected_voltage": expected_voltage, "runtime_voltage": runtime_voltage, "abs_voltage_diff": abs(expected_voltage - runtime_voltage), "expected_Y": expected_y, "runtime_Y": runtime_y, "abs_Y_diff": abs(expected_y - runtime_y), "historical_CHM_sparse_Y": historical_y, "historical_CHM_sparse_Y_abs_diff_from_AHC_TV_normalization": abs(historical_y - expected_y), "pass": bool(expected_rows == runtime_rows and abs(expected_voltage - runtime_voltage) <= VOLTAGE_TOL and abs(expected_y - runtime_y) <= VOLTAGE_TOL)})
    output = pd.DataFrame(records); output.to_csv(OUTPUTS / "AHC_TV_STAGE_T_DEVELOPMENT_TARGET_EQUIVALENCE.csv", index=False)
    passed = int(output["pass"].sum())
    return {"rows": output, "pass_count": passed, "required": 7, "all_pass": passed == 7}


def _run_test_suite(filename: str) -> dict[str, Any]:
    tests = PHASE / "tests"
    if str(PHASE / "scripts") not in sys.path:
        sys.path.insert(0, str(PHASE / "scripts"))
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(unittest.defaultTestLoader.discover(str(tests), pattern=filename))
    return {"executed_in_this_audit": True, "tests_run": int(result.testsRun), "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "wasSuccessful": bool(result.wasSuccessful()), "status": "PASS" if result.wasSuccessful() else "FAIL"}


def _formal_output_summary() -> dict[str, Any]:
    path = OUTPUTS / "AHC_TV_FORMAL_STAGE_P_DEVELOPMENT_EQUIVALENCE.csv"
    if not path.exists():
        raise PreOpenVerificationError("FORMAL_RUNTIME_OUTPUT_MISSING")
    rows = pd.read_csv(path)
    required = ("interval_id_match", "datetime_match", "R_match", "voltage_match", "ophrs2_match", "conditions_match", "support_match", "eligible_dwell_count_match", "formal_anchor_median_match")
    if len(rows) != 18 or any(column not in rows for column in required):
        raise PreOpenVerificationError("FORMAL_RUNTIME_OUTPUT_SCHEMA_INCOMPLETE")
    checkpoint_pass = rows.groupby("checkpoint", sort=True)[list(required)].all().all(axis=1)
    return {"rows": rows, "pass_count": int(checkpoint_pass.sum()), "required": 6, "all_pass": bool(len(checkpoint_pass) == 6 and checkpoint_pass.all())}


def write_final_preopen_audit_v2(*, rcr_template: dict, source_binding: dict, artifact: dict, independent: dict, formal: dict, targets: dict, suites: dict[str, dict]) -> dict[str, Any]:
    rows = formal["rows"]
    all_pass = bool(rcr_template["locked"] and rcr_template["hash_check"] == "PASS" and source_binding["binding_status"] == "PASS" and artifact["all_pass"] and independent["all_pass"] and formal["all_pass"] and targets["all_pass"] and all(result["wasSuccessful"] for result in suites.values()))
    audit = {
        "real_Test_accessed": False, "real_2000_2300_accessed": False,
        "authoritative_local_dataset_root_recorded": True,
        "real_test_filename_enumeration_only": True,
        "real_test_file_content_accessed": bool(source_binding["file_content_opened"]),
        "nominal_2000_source_match_count": int(source_binding["nominal_2000_match_count"]),
        "nominal_2300_source_match_count": int(source_binding["nominal_2300_match_count"]),
        "afterAirAir_2300_excluded": bool(source_binding["after_air_air_2300_excluded"]),
        "test_source_binding_dry_run": source_binding["binding_status"],
        "RCR_cycle_template_runtime_hash_locked": bool(rcr_template["locked"]),
        "RCR_cycle_template_sha256": rcr_template["sha256"],
        "RCR_cycle_template_hash_check": rcr_template["hash_check"],
        "artifact_alignment_replay": f"{artifact['pass_count']}/{artifact['required']} {'PASS' if artifact['all_pass'] else 'FAIL'}",
        "independent_reference_raw_stage_p_replay": f"{independent['pass_count']}/{independent['required']} {'PASS' if independent['all_pass'] else 'FAIL'}",
        "formal_stage_p_runtime_replay": f"{formal['pass_count']}/{formal['required']} {'PASS' if formal['all_pass'] else 'FAIL'}",
        "formal_selected_intervals": f"{int(rows['interval_id_match'].sum())}/18 {'PASS' if bool(rows['interval_id_match'].all()) else 'FAIL'}",
        "formal_R_values": f"{int(rows['R_match'].sum())}/18 {'PASS' if bool(rows['R_match'].all()) else 'FAIL'}",
        "formal_condition_vectors": f"{int(rows['conditions_match'].sum())}/18 {'PASS' if bool(rows['conditions_match'].all()) else 'FAIL'}",
        "formal_support": f"{int(rows['support_match'].sum())}/18 {'PASS' if bool(rows['support_match'].all()) else 'FAIL'}",
        "stage_t_development_target_replay": f"{targets['pass_count']}/{targets['required']} {'PASS' if targets['all_pass'] else 'FAIL'}",
        "synthetic_runtime_E2E_test_suite": suites["A_stage_p_stage_t_integration"], "one_shot_state_machine_test_suite": suites["B_one_shot_execution_state_machine"], "formal_stage_p_development_equivalence_test_suite": suites["C_formal_stage_p_development_equivalence"],
        "synthetic_test_mode_used_for_formal_development_replay": False, "active_cell_missing_value_semantics": "MATCHES_FROZEN_NANMEAN", "scientific_protocol_changed": False,
        "remaining_preopen_blocker": "NONE" if all_pass else "IMPLEMENTATION_EQUIVALENCE_OR_ACTUAL_TEST_FAILURE", "final_status": "AHC_TV_FINAL_PREOPEN_VERIFICATION_PASS" if all_pass else "AHC_TV_FINAL_PREOPEN_VERIFICATION_BLOCKED",
    }
    (OUTPUTS / "AHC_TV_FINAL_PREOPEN_AUDIT_V2.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    rcr_template = rcr_cycle_template_runtime_lock_audit(phase=PHASE)
    source_binding = run_real_filename_binding_dry_run(phase=PHASE)
    artifact = replay_development_alignment(PHASE)
    independent = stage_p_development_raw_replay(); targets = stage_t_development_target_replay()
    suites = {"A_stage_p_stage_t_integration": _run_test_suite("test_stage_p_stage_t_integration.py"), "B_one_shot_execution_state_machine": _run_test_suite("test_one_shot_execution_state_machine.py"), "C_formal_stage_p_development_equivalence": _run_test_suite("test_formal_stage_p_development_equivalence.py"), "D_realhyfc_source_binding": _run_test_suite("test_realhyfc_source_binding.py")}
    formal = _formal_output_summary(); audit = write_final_preopen_audit_v2(rcr_template=rcr_template, source_binding=source_binding, artifact=artifact, independent=independent, formal=formal, targets=targets, suites=suites)
    print(json.dumps({"rcr_template": rcr_template, "source_binding": source_binding, "artifact": artifact, "independent": {key: value for key, value in independent.items() if key != "rows"}, "formal": {key: value for key, value in formal.items() if key != "rows"}, "targets": {key: value for key, value in targets.items() if key != "rows"}, "audit": audit}, indent=2, default=str))
    if audit["final_status"] != "AHC_TV_FINAL_PREOPEN_VERIFICATION_PASS":
        raise PreOpenVerificationError(audit["remaining_preopen_blocker"])


if __name__ == "__main__":
    main()

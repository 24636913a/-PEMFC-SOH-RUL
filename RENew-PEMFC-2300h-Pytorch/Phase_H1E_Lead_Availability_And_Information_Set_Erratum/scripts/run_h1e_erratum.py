"""Read-only H1 lead-availability and information-set erratum."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
OUT = PHASE / "outputs"
REPORT = PHASE / "reports"
H1 = PROJECT / "Phase_CPRF_H1_Causal_P1S_Retention_Forecasting"
H1_SUMMARY = H1 / "outputs" / "CPRF_H1_SUMMARY.json"
H1_AUTHORITY = H1 / "inputs" / "H1_FROZEN_SAMPLE_AUTHORITY.csv"
RCR = PROJECT / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit"
RCR_PROTOCOL = RCR / "configs" / "PREREGISTERED_RCR_PROTOCOL.yaml"
RCR_TEMPLATE = RCR / "outputs" / "CYCLE_TEMPLATE.json"
PAA = PROJECT / "Phase_PAA_P1_StateToken_PreAnchor_Alignment"
PAA_SCRIPT = PAA / "scripts" / "run_p1_state_token_preanchor_audit.py"
BASE = "fb1489d94333315a1b07c57030387b244a05e02b"
EXPECTED = {
    "legal_unique_H1_samples": 122,
    "persistence_equal_fold_mean_MAE_mV": 0.15006652420634772,
    "ridge_equal_fold_mean_MAE_mV": 0.12043668140196866,
    "timesnet_three_seed_equal_fold_mean_MAE_mV": 2.064150563241723,
    "gate_A": "FAIL", "gate_B": "FAIL", "gate_C": "FAIL", "gate_D": "FAIL",
}


class ErratumFailure(RuntimeError):
    pass


def git(command: list[str]) -> str:
    return subprocess.check_output(["git", *command], cwd=REPO, text=True).strip()


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, lineterminator="\n")


def write_json(payload: dict[str, Any], name: str) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def preflight() -> None:
    head, remote = git(["rev-parse", "HEAD"]), git(["rev-parse", "origin/main"])
    dirty = git(["status", "--short"]).splitlines()
    if head != BASE or remote != BASE or head != remote:
        raise ErratumFailure("REMOTE_MAIN_ADVANCED")
    allowed_prefix = str(PHASE.relative_to(REPO)).replace("\\", "/")
    disallowed = [line for line in dirty if allowed_prefix not in line.replace("\\", "/")]
    if disallowed:
        raise ErratumFailure("WORKING_TREE_NOT_CLEAN_OUTSIDE_H1E")


def load_frozen_h1() -> tuple[dict[str, Any], pd.DataFrame]:
    summary = json.loads(H1_SUMMARY.read_text(encoding="utf-8"))
    for key, expected in EXPECTED.items():
        observed = summary.get(key)
        if isinstance(expected, float):
            if not isinstance(observed, (float, int)) or not math.isclose(float(observed), expected, abs_tol=1e-15, rel_tol=0.0):
                raise ErratumFailure(f"H1_RESULT_MISMATCH:{key}")
        elif observed != expected:
            raise ErratumFailure(f"H1_RESULT_MISMATCH:{key}")
    authority = pd.read_csv(H1_AUTHORITY)
    required = {"fold", "role", "sample_index", "origin_p1s_measurement_end_ophrs2_h", "target_p1s_measurement_end_ophrs2_h", "lead_to_target_end_h", "origin_raw_cycle_id", "target_raw_cycle_id"}
    if required - set(authority.columns) or len(authority) != 292:
        raise ErratumFailure("H1_FROZEN_AUTHORITY_SCHEMA_OR_SIZE_MISMATCH")
    if not (authority.target_raw_cycle_id.to_numpy(int) == authority.origin_raw_cycle_id.to_numpy(int) + 1).all():
        raise ErratumFailure("H1_CONTIGUITY_AUTHORITY_MISMATCH")
    if authority.target_p1s_measurement_end_ophrs2_h.to_numpy(float).max() >= 1819.997738:
        raise ErratumFailure("TEST_RANGE_IN_H1_AUTHORITY")
    return summary, authority


def statistics(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    if not len(arr) or not np.isfinite(arr).all():
        raise ErratumFailure("INVALID_LEAD_DISTRIBUTION")
    return {"N": int(len(arr)), "mean_h": float(np.mean(arr)), "median_h": float(np.median(arr)),
            "std_h": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, "min_h": float(np.min(arr)),
            "P05_h": float(np.quantile(arr, .05)), "P95_h": float(np.quantile(arr, .95)), "max_h": float(np.max(arr)),
            "IQR_h": float(np.quantile(arr, .75) - np.quantile(arr, .25))}


def build_lead_outputs(authority: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    distribution = authority.copy()
    distribution["actual_lead_h"] = distribution.target_p1s_measurement_end_ophrs2_h.to_numpy(float) - distribution.origin_p1s_measurement_end_ophrs2_h.to_numpy(float)
    distribution["stored_vs_recomputed_lead_difference_h"] = distribution.actual_lead_h - distribution.lead_to_target_end_h.to_numpy(float)
    if not np.allclose(distribution.stored_vs_recomputed_lead_difference_h, 0.0, atol=1e-9, rtol=0.0):
        raise ErratumFailure("H1_LEAD_RECOMPUTATION_MISMATCH")
    distribution["distribution_record_type"] = "fold_assigned_authority_row"
    unique = distribution.sort_values(["sample_index", "fold", "role"]).drop_duplicates("sample_index", keep="first").copy()
    global_stats = statistics(unique.actual_lead_h)
    rows = [{"scope": "UNIQUE_FROZEN_AUTHORITY_H1_PAIRS", "note": "No duplicate weighting across expanding folds; one H1 legal candidate is absent because its target measurement interval crossed a fold boundary.", **global_stats}]
    for fold in ("fold_1", "fold_2", "fold_3"):
        current = distribution.loc[distribution.fold.eq(fold)]
        rows.append({"scope": f"{fold}_ALL_ASSIGNED_ROLES", "note": "Each H1 pair once within this fold, across train and validation roles.", **statistics(current.drop_duplicates("sample_index").actual_lead_h)})
        for role in ("train", "validation"):
            sub = current.loc[current.role.eq(role)]
            rows.append({"scope": f"{fold}_{role.upper()}", "note": "Target-interval fold assignment; descriptive only.", **statistics(sub.actual_lead_h)})
    summary = pd.DataFrame(rows)
    return distribution, summary, global_stats


def build_method_information_audit() -> pd.DataFrame:
    actual_lead_available = "NOT_ESTABLISHED"
    records = [
        {"method": "PERSISTENCE", "uses_R_origin": True, "uses_past_retention_slope": False, "uses_actual_target_end_lead": False, "uses_Raw_F2_dynamic_history": False, "uses_future_target_measurement": False, "strictly_available_at_prediction_origin": True, "formal_role_after_erratum": "PRINCIPAL_STRICTLY_CAUSAL_H1_BASELINE", "information_set": "{R_origin}"},
        {"method": "CAUSAL_LINEAR_96H", "uses_R_origin": True, "uses_past_retention_slope": True, "uses_actual_target_end_lead": True, "uses_Raw_F2_dynamic_history": False, "uses_future_target_measurement": False, "strictly_available_at_prediction_origin": False, "formal_role_after_erratum": "DESCRIPTIVE_ACTUAL_LEAD_LINEAR_BASELINE", "information_set": "{R_origin, slope_96h, actual_target_end_lead}"},
        {"method": "RIDGE_CAUSAL_STATE", "uses_R_origin": True, "uses_past_retention_slope": True, "uses_actual_target_end_lead": True, "uses_Raw_F2_dynamic_history": False, "uses_future_target_measurement": False, "strictly_available_at_prediction_origin": False, "formal_role_after_erratum": "DESCRIPTIVE_ACTUAL_LEAD_RIDGE_BASELINE", "information_set": "{R_origin, slope_24h, slope_48h, slope_96h, actual_target_end_lead}"},
        {"method": "TIMESNET_RAW_F2", "uses_R_origin": False, "uses_past_retention_slope": False, "uses_actual_target_end_lead": False, "uses_Raw_F2_dynamic_history": True, "uses_future_target_measurement": False, "strictly_available_at_prediction_origin": True, "formal_role_after_erratum": "RAW_F2_ONLY_TIMESNET_H1_NOT_SUPPORTED", "information_set": "{48x14 Raw F2 dynamic history}"},
    ]
    audit = pd.DataFrame(records)
    audit["actual_target_end_lead_availability_status"] = actual_lead_available
    return audit


def protocol_evidence() -> dict[str, Any]:
    template = json.loads(RCR_TEMPLATE.read_text(encoding="utf-8"))
    rcr_text = RCR_PROTOCOL.read_text(encoding="utf-8")
    paa_text = PAA_SCRIPT.read_text(encoding="utf-8")
    if "earliest_consecutive_calibration_anchor_interval_in_loaded_range" not in rcr_text:
        raise ErratumFailure("RCR_TEMPLATE_PROVENANCE_TEXT_MISSING")
    if "def target_intervals" not in paa_text or "def make_windows" not in paa_text:
        raise ErratumFailure("PAA_ACTUAL_MEASUREMENT_REPLAY_TEXT_MISSING")
    return {
        "actual_target_end_lead_causally_available": "NOT_ESTABLISHED",
        "nominal_lead_prospectively_available": "NOT_ESTABLISHED",
        "retrospective_nominal_pattern": "RCR CYCLE_TEMPLATE template_loaded_duration_h",
        "template_loaded_duration_h": float(template["template_loaded_duration_h"]),
        "template_wall_duration_h": float(template["template_wall_duration_h"]),
        "template_selection_rule": str(template["selection_rule"]),
        "evidence_paths": [str(RCR_PROTOCOL.relative_to(REPO)).replace("\\", "/"), str(RCR_TEMPLATE.relative_to(REPO)).replace("\\", "/"), str(PAA_SCRIPT.relative_to(REPO)).replace("\\", "/")],
        "reason": "The RCR template is a retrospectively selected first-100-h calibration object and its duration is a qualification/descriptive construct. Neither it nor the PAA reconstruction specifies every next raw-cycle start, realized duration, P1 dwell sequence, or P1S measurement end prospectively at the previous P1S origin. PAA determines P1S dwell membership from actual recorded cycle/state timing.",
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    head = [str(col).replace("|", "\\|") for col in view.columns]
    body = ["| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "| " + " | ".join(head) + " |\n|" + "|".join(["---"] * len(head)) + "|\n" + "\n".join(body)


def report(summary: dict[str, Any], lead_summary: pd.DataFrame, methods: pd.DataFrame, evidence: dict[str, Any]) -> None:
    global_row = lead_summary.loc[lead_summary.scope.eq("UNIQUE_FROZEN_AUTHORITY_H1_PAIRS")].iloc[0]
    text = f"""# H1 lead-availability and information-set erratum

## Scope and preserved H1 result

This is a read-only erratum. No H1 model, Ridge model, slope extrapolation, score, gate, sample, or Test artifact was recomputed. The committed H1 numbers remain: Persistence `{summary['persistence_equal_fold_mean_MAE_mV']:.13g}` mV-equivalent, historical Ridge `{summary['ridge_equal_fold_mean_MAE_mV']:.13g}` mV-equivalent, and TimesNet `{summary['timesnet_three_seed_equal_fold_mean_MAE_mV']:.13g}` mV-equivalent. Gates A–D remain FAIL.

`mV-equivalent = abs(retention error) × V_ref × 1000`; it is not the Phase BS dynamic-voltage RMSE.

## Lead availability

**ACTUAL_TARGET_END_LEAD_CAUSALLY_AVAILABLE = NOT_ESTABLISHED.**

The RCR artifact records a retrospective calibration template with loaded duration `{evidence['template_loaded_duration_h']:.6g}` h and wall duration `{evidence['template_wall_duration_h']:.6g}` h. It was selected as `{evidence['template_selection_rule']}`, rather than as a prospective controller schedule fixing each future P1S end. The PAA implementation identifies P1 dwell windows from actual complete-cycle/state metadata, then derives measurement windows from those realized dwell durations. No existing Development-side source proved that the next actual P1S measurement end was known at the preceding P1S prediction origin.

This supports neither Case A nor a prospective nominal-schedule claim. A 9.523 h calibration-template duration is retained only as a retrospective nominal pattern—not as a `PROSPECTIVE_NOMINAL_LEAD_CANDIDATE` and not as a performance-recalculation input.

Across the unique fold-assigned H1 pairs, actual lead has N={int(global_row.N)}, mean={global_row.mean_h:.6g} h, median={global_row.median_h:.6g} h, SD={global_row.std_h:.6g} h, min={global_row.min_h:.6g} h, P05={global_row.P05_h:.6g} h, P95={global_row.P95_h:.6g} h, max={global_row.max_h:.6g} h, and IQR={global_row.IQR_h:.6g} h. Thus actual lead is not a fixed constant in the frozen authority. H1 declares 122 legal candidates; 121 occur in the fold-assigned authority used here because one candidate's target P1S measurement interval crossed a frozen fold boundary and was excluded before fitting/evaluation.

{markdown_table(lead_summary, ['scope', 'N', 'mean_h', 'median_h', 'std_h', 'min_h', 'P05_h', 'P95_h', 'max_h', 'IQR_h'])}

## Information sets and formal roles

{markdown_table(methods, ['method', 'uses_R_origin', 'uses_past_retention_slope', 'uses_actual_target_end_lead', 'uses_Raw_F2_dynamic_history', 'strictly_available_at_prediction_origin', 'formal_role_after_erratum'])}

Persistence uses `{{R_origin}}`; the observation is complete at its prediction origin, so it is the **PRINCIPAL_STRICTLY_CAUSAL_H1_BASELINE**. Linear and Ridge use the realized target-end lead; their historical values are retained but their formal roles are respectively `DESCRIPTIVE_ACTUAL_LEAD_LINEAR_BASELINE` and `DESCRIPTIVE_ACTUAL_LEAD_RIDGE_BASELINE`, not strictly causal deployment baselines.

TimesNet uses only `{{48×14 Raw F2}}`, while Persistence uses `{{R_origin}}` and Ridge uses `{{R_origin, historical slopes, actual target-end lead}}`. Hence `TIMESNET_VS_PERSISTENCE = DEVELOPMENT_RELEVANT_BUT_NOT_EQUAL_INFORMATION_SET_COMPARISON`: Persistence is a practically available opponent, but this is not a matched-information-set comparison.

The corrected conclusion is: **Under the frozen Raw-F2-only H1 protocol, TimesNet did not recover sufficient next-cycle P1S-retention information to outperform the directly observable origin-retention persistence baseline.** This does not establish that complex models cannot improve P1S forecasting under another preregistered information set.

## Status and stop

* `RAW_F2_ONLY_TIMESNET_H1_NOT_SUPPORTED`
* `PERSISTENCE_STRONG_CAUSAL_BASELINE`
* `AUTO_H3_H6 = PROHIBITED`

Recommended research direction: **STOP_H1_MODEL_DEVELOPMENT**. A Raw-F2-plus-origin-retention or longer-horizon experiment would be a distinct preregistered question, not an amendment or automatic continuation of H1. No Test was accessed and no model training was performed.
"""
    (REPORT / "H1_LEAD_AND_INFORMATION_SET_ERRATUM.md").write_text(text, encoding="utf-8")


def run() -> dict[str, Any]:
    ensure_dirs(); preflight()
    h1_summary, authority = load_frozen_h1()
    distribution, lead_summary, global_stats = build_lead_outputs(authority)
    methods = build_method_information_audit(); evidence = protocol_evidence()
    write_csv(methods, "H1_METHOD_INFORMATION_SET_AUDIT.csv")
    write_csv(distribution, "H1_ACTUAL_LEAD_DISTRIBUTION.csv")
    write_csv(lead_summary, "H1_LEAD_SUMMARY.csv")
    output = {
        "phase": "Phase_H1E_Lead_Availability_And_Information_Set_Erratum", "base_commit": BASE,
        "legal_H1_samples": int(h1_summary["legal_unique_H1_samples"]), "fold_assigned_H1_pairs_in_lead_distribution": int(global_stats["N"]),
        "persistence_MAE_mV_equivalent": h1_summary["persistence_equal_fold_mean_MAE_mV"],
        "ridge_historical_MAE_mV_equivalent": h1_summary["ridge_equal_fold_mean_MAE_mV"],
        "timesnet_MAE_mV_equivalent": h1_summary["timesnet_three_seed_equal_fold_mean_MAE_mV"],
        "actual_target_end_lead_causally_available": evidence["actual_target_end_lead_causally_available"],
        "nominal_lead_prospectively_available": evidence["nominal_lead_prospectively_available"],
        "actual_lead_mean_h": global_stats["mean_h"], "actual_lead_std_h": global_stats["std_h"], "actual_lead_min_h": global_stats["min_h"], "actual_lead_max_h": global_stats["max_h"],
        "persistence_uses_future_target_timing": False, "ridge_uses_actual_target_timing": True, "causal_linear_uses_actual_target_timing": True, "timesnet_uses_future_target_timing": False,
        "persistence_formal_role": "PRINCIPAL_STRICTLY_CAUSAL_H1_BASELINE", "ridge_formal_role": "DESCRIPTIVE_ACTUAL_LEAD_RIDGE_BASELINE", "causal_linear_formal_role": "DESCRIPTIVE_ACTUAL_LEAD_LINEAR_BASELINE", "timesnet_formal_status": "RAW_F2_ONLY_TIMESNET_H1_NOT_SUPPORTED",
        "equal_information_set_comparison": False, "H3_H6_automatically_authorized": False, "recommended_research_direction": "STOP_H1_MODEL_DEVELOPMENT",
        "retrospective_template_loaded_duration_h": evidence["template_loaded_duration_h"], "lead_evidence_paths": evidence["evidence_paths"],
        "test_accessed": False, "model_training_performed": False, "H1_results_recomputed": False,
    }
    write_json(output, "H1E_SUMMARY.json")
    report(h1_summary, lead_summary, methods, evidence)
    return output


if __name__ == "__main__":
    try:
        result = run()
        print(json.dumps({"status": "H1E_COMPLETE", "actual_target_end_lead_causally_available": result["actual_target_end_lead_causally_available"], "model_training_performed": False, "test_accessed": False}))
    except ErratumFailure as exc:
        print(json.dumps({"status": "H1E_FAILED_CLOSED", "reason": str(exc)}))
        raise

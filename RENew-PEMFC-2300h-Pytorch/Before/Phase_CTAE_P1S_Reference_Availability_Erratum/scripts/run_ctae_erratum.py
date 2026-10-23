#!/usr/bin/env python3
"""Correct the causal availability boundary of fixed-reference P1S retention."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
OUT = PHASE / "outputs"
REPORT = PHASE / "reports"
PAA = PROJECT / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs"
PHI = PROJECT / "Phase_PHI_P1S_B2_Health_Index_Consolidation" / "outputs"

REFERENCE = PAA / "P1S_REFERENCE_COMPLETE_CYCLES.csv"
COMPLETE = PAA / "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv"
TRAJECTORY = PHI / "P1S_B2_FULL_HEALTH_TRAJECTORY.csv"

V_REF = 0.71383023635
V_REF_AVAIL = 171.82091
TOL = 1e-12
REFERENCE_IDS = [f"COMPLETE_{i:03d}" for i in range(1, 11)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_reference() -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE).sort_values("raw_cycle_id", kind="stable").reset_index(drop=True)
    if len(ref) != 10 or ref["cycle_interval_id"].tolist() != REFERENCE_IDS:
        raise RuntimeError("REFERENCE_CYCLE_SET_REPRODUCTION_FAILED")
    if not ref["is_reference_complete_interval"].all() or ref["is_evaluation_complete_interval"].any():
        raise RuntimeError("REFERENCE_EVALUATION_SEPARATION_FAILED")
    reproduced = float(ref["P1S_voltage_V"].median())
    if abs(reproduced - V_REF) > TOL:
        raise RuntimeError("REFERENCE_VOLTAGE_REPRODUCTION_FAILED")
    if abs(float(ref.iloc[-1].measurement_end_ophrs2_h) - V_REF_AVAIL) > TOL:
        raise RuntimeError("REFERENCE_AVAILABILITY_TIME_REPRODUCTION_FAILED")
    return ref


def load_availability() -> pd.DataFrame:
    trajectory = pd.read_csv(TRAJECTORY).sort_values("ophrs_h", kind="stable").reset_index(drop=True)
    complete = pd.read_csv(COMPLETE)
    complete = complete.loc[complete["stable_window"].eq("middle_50")].copy()
    flags = complete[["cycle_interval_id", "is_reference_complete_interval", "is_evaluation_complete_interval", "measurement_end_ophrs2_h", "P1S_voltage_V"]]
    data = trajectory.drop(columns=["P1S_voltage_V"], errors="ignore").merge(flags, on="cycle_interval_id", validate="one_to_one")
    if len(data) != 141:
        raise RuntimeError("PHI_EVENT_COUNT_REPRODUCTION_FAILED")
    if not np.allclose(data["ophrs_h"], data["measurement_end_ophrs2_h"], rtol=0, atol=TOL):
        raise RuntimeError("PHI_PAA_TIME_ALIGNMENT_FAILED")
    data["V_ref_V"] = V_REF
    data["V_ref_availability_ophrs_h"] = V_REF_AVAIL
    data["V_ref_available_at_event"] = data["ophrs_h"] >= V_REF_AVAIL
    data["reference_warmup_record"] = data["is_reference_complete_interval"].astype(bool)
    data["causal_target_eligible"] = data["is_evaluation_complete_interval"].astype(bool) & (data["ophrs_h"] > V_REF_AVAIL)
    data["R_P1S_retrospective"] = data["P1S_voltage_V"] / V_REF
    data["R_P1S_causal"] = np.where(data["causal_target_eligible"], data["R_P1S_retrospective"], np.nan)
    warm = data.loc[data["reference_warmup_record"]]
    if len(warm) != 10 or warm["causal_target_eligible"].any():
        raise RuntimeError("REFERENCE_WARMUP_ELIGIBILITY_VIOLATION")
    if warm.iloc[:-1]["V_ref_available_at_event"].any() or not bool(warm.iloc[-1]["V_ref_available_at_event"]):
        raise RuntimeError("REFERENCE_AVAILABILITY_FLAG_VIOLATION")
    return data


def past_slope(t: np.ndarray, y: np.ndarray, window_h: float) -> tuple[float, int]:
    now = t[-1]
    use = (t <= now) & (t >= now - window_h)
    n = int(use.sum())
    if n < 3 or np.ptp(t[use]) == 0:
        return np.nan, n
    return float(np.polyfit(t[use], y[use], 1)[0]), n


def full_chronological_slope_at(t: np.ndarray, y: np.ndarray, index: int, window_h: float) -> tuple[float, int]:
    """Compute at one endpoint from the full sequence while explicitly excluding future rows."""
    now = t[index]
    use = (t <= now) & (t >= now - window_h)
    n = int(use.sum())
    if n < 3 or np.ptp(t[use]) == 0:
        return np.nan, n
    return float(np.polyfit(t[use], y[use], 1)[0]), n


def make_slopes(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool], dict[str, float | None]]:
    eligible = data.loc[data["causal_target_eligible"]].copy().reset_index(drop=True)
    t = eligible["ophrs_h"].to_numpy(float)
    r = eligible["R_P1S_causal"].to_numpy(float)
    rows = []
    replay_rows = []
    for i in range(len(eligible)):
        row = {"cycle_interval_id": eligible.loc[i, "cycle_interval_id"], "ophrs_h": t[i], "R_P1S_causal": r[i]}
        replay = {"prefix_event_count": i + 1, "cycle_interval_id": eligible.loc[i, "cycle_interval_id"], "ophrs_h": t[i]}
        for hours in (24, 48, 96):
            slope, count = past_slope(t[:i + 1], r[:i + 1], float(hours))
            row[f"slope_{hours}h"] = slope
            row[f"slope_{hours}h_available"] = bool(np.isfinite(slope))
            row[f"n_points_{hours}h"] = count
            full_chronological, _ = full_chronological_slope_at(t, r, i, float(hours))
            replay[f"slope_{hours}h_prefix"] = slope
            replay[f"slope_{hours}h_full_chronological"] = full_chronological
            replay[f"slope_{hours}h_abs_difference"] = np.nan if (np.isnan(slope) and np.isnan(full_chronological)) else abs(slope - full_chronological)
            replay[f"n_points_{hours}h"] = count
        rows.append(row)
        replay_rows.append(replay)
    slopes = pd.DataFrame(rows)
    replay_df = pd.DataFrame(replay_rows)
    verdict, first = {}, {}
    for hours in (24, 48, 96):
        diffs = replay_df[f"slope_{hours}h_abs_difference"].dropna()
        verdict[f"{hours}h"] = bool((diffs <= TOL).all())
        available = slopes.loc[slopes[f"slope_{hours}h_available"]]
        first[f"{hours}h"] = None if available.empty else float(available.iloc[0].ophrs_h)
    return slopes, replay_df, verdict, first


def manifest() -> dict[str, object]:
    files = [REFERENCE, COMPLETE, TRAJECTORY]
    return {
        "base_commit": "1f689dbcdd988326fbab4c6114c085c4c266d7a5",
        "sources": [{"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)} for path in files],
        "test_accessed": False,
        "model_training_performed": False,
        "RUL_training_performed": False,
        "raw_polarization_files_read": False,
    }


def write_report(data: pd.DataFrame, invariance: dict[str, bool], first: dict[str, float | None]) -> None:
    first_eval = data.loc[data["causal_target_eligible"]].iloc[0]
    text = f"""# CTA Reference Availability Erratum

CTA correctly selected T2, but its temporal scope was overbroad. The fixed denominator is `V_ref = median(P1S voltage of COMPLETE_001 through COMPLETE_010)`. The ten frozen reference voltages reproduce `{V_REF:.11f} V`; the final required measurement ends at `{V_REF_AVAIL:.5f} h`. Before this time, fixed-reference retention can only be reconstructed retrospectively.

PAA remains stricter than numerical denominator availability: all ten cycles are `REFERENCE_WARMUP_ONLY` and never formal prediction targets. The first automatically identified causal evaluation cycle is `{first_eval.cycle_interval_id}` at `{first_eval.ophrs_h:.5f} h`.

## Corrected target

`POST_REFERENCE_RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET`: `R_P1S(t) = V_P1S(t) / V_ref` only for non-reference evaluation cycles after `{V_REF_AVAIL:.5f} h`.

Post-reference 24 h, 48 h and 96 h past-only retention slope prefix invariance is `{invariance['24h']}`, `{invariance['48h']}`, and `{invariance['96h']}`. First available times are {first['24h']}, {first['48h']}, and {first['96h']} h. This establishes calculation-level causality only, not forecasting utility.

## Preserved CTA conclusions

`CTA_T2_DIRECTION_PRESERVED`; `CTA_T2_TEMPORAL_SCOPE_CORRECTED`.

Global B2: `RETROSPECTIVE_ONLY`. Full-sequence trends: `RETROSPECTIVE_ONLY`. Support: `STRATIFIED_MODELING_REQUIRED`.

## CTA ERRATUM DECISION

T2 direction: **PRESERVED**

Corrected target: **POST_REFERENCE_RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET**

Reference warm-up: **FIRST_10_COMPLETE_P1S_CYCLES**

Fixed reference available after: **171.82091 h**

Reference cycles used as prediction targets: **NO**

Global B2: **RETROSPECTIVE_ONLY**

Full-sequence trends: **RETROSPECTIVE_ONLY**

Post-reference raw-retention slope: **CAUSAL_FEATURE_CANDIDATE_IF_PREFIX_REPLAY_PASS**

Support: **STRATIFIED_MODELING_REQUIRED**

Next phase: **CAUSAL_P1S_RETENTION_FORECASTING**
"""
    (REPORT / "CTA_REFERENCE_AVAILABILITY_ERRATUM.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    reference = verify_reference()
    data = load_availability()
    slopes, replay, invariance, first = make_slopes(data)
    first_eval = data.loc[data["causal_target_eligible"]].iloc[0]
    fields = [
        "cycle_interval_id", "cycle_index", "ophrs_h", "P1S_voltage_V", "is_reference_complete_interval", "is_evaluation_complete_interval",
        "V_ref_V", "V_ref_availability_ophrs_h", "V_ref_available_at_event", "reference_warmup_record", "causal_target_eligible",
        "R_P1S_retrospective", "R_P1S_causal", "condition_support_status",
    ]
    data[fields].to_csv(OUT / "P1S_REFERENCE_AVAILABILITY_AUDIT.csv", index=False)
    slopes.to_csv(OUT / "POST_REFERENCE_CAUSAL_SLOPE_AVAILABILITY.csv", index=False)
    replay.to_csv(OUT / "POST_REFERENCE_CAUSAL_SLOPE_PREFIX_REPLAY.csv", index=False)
    (OUT / "CTAE_SOURCE_MANIFEST.json").write_text(json.dumps(manifest(), indent=2), encoding="utf-8")
    summary = {
        "phase": "Phase_CTAE_P1S_Reference_Availability_Erratum", "base_commit": "1f689dbcdd988326fbab4c6114c085c4c266d7a5",
        "reference_cycle_count": int(len(reference)), "reference_cycle_ids": REFERENCE_IDS, "V_ref": V_REF, "V_ref_reproduced": True, "V_ref_availability_ophrs_h": V_REF_AVAIL,
        "reference_warmup_count": int(data.reference_warmup_record.sum()), "reference_warmup_target_count": 0,
        "first_causal_evaluation_cycle": str(first_eval.cycle_interval_id), "first_causal_evaluation_ophrs_h": float(first_eval.ophrs_h),
        "total_phi_events": int(len(data)), "causal_target_eligible_event_count": int(data.causal_target_eligible.sum()), "causal_target_ineligible_event_count": int((~data.causal_target_eligible).sum()),
        "eligible_in_support_count": int((data.causal_target_eligible & data.condition_support_status.eq("IN_SUPPORT")).sum()),
        "eligible_out_of_support_count": int((data.causal_target_eligible & data.condition_support_status.eq("OUT_OF_SUPPORT")).sum()),
        "formal_target": "POST_REFERENCE_RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET",
        "formal_target_formula": "R_P1S(t) = V_P1S(t) / median(first_10_reference_cycle_P1S_voltages), evaluation cycles only after 171.82091 h",
        "global_B2_role": "RETROSPECTIVE_REFERENCE_ONLY", "full_sequence_trend_role": "RETROSPECTIVE_ONLY",
        "post_reference_24h_prefix_invariance": "PASS" if invariance["24h"] else "FAIL", "post_reference_48h_prefix_invariance": "PASS" if invariance["48h"] else "FAIL", "post_reference_96h_prefix_invariance": "PASS" if invariance["96h"] else "FAIL",
        "first_24h_slope_available_ophrs_h": first["24h"], "first_48h_slope_available_ophrs_h": first["48h"], "first_96h_slope_available_ophrs_h": first["96h"],
        "support_handling": "STRATIFIED_MODELING_REQUIRED", "test_accessed": False, "model_training_performed": False, "RUL_training_performed": False,
    }
    (OUT / "CTAE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(data, invariance, first)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

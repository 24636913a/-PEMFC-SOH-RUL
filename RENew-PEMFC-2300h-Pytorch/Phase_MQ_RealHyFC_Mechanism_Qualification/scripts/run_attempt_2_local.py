"""Local, no-training Attempt-2 execution; stops closed at the first MQ gate failure."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mq_access import DataAccessGuard

PHASE = Path(__file__).resolve().parents[1]
OUT = PHASE / "outputs"
CFG = PHASE / "configs"
ORIGINAL = CFG / "PREREGISTERED_MQ_PROTOCOL.yaml"
AMENDMENT = CFG / "PRE_EXECUTION_AMENDMENT_V1.yaml"
EXPECTED_HEAD = "9acc1d584fec49555f122d2fac6a208bf0cd10ad"


def write_json(name: str, data: Any) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ["status"]
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def protocol_sha256(path: Path) -> str:
    """Match repository LF protocol bytes despite Windows core.autocrlf checkout."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def effective_protocol() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    if protocol_sha256(ORIGINAL) != "1e934216843a8a72fe7d2ebe9975db2b573c0548605cee0496fdd445505e6c63":
        raise RuntimeError("Original MQ protocol SHA mismatch")
    if protocol_sha256(AMENDMENT) != "79b78925ef1f0bbeaabc0b7d41b0ae3d3c14b6f249c7b44c0c2fd7a39b22a69e":
        raise RuntimeError("MQ amendment V1 SHA mismatch")
    text = ORIGINAL.read_text(encoding="utf-8")
    bad = "  active_cells: [U1 [V], U2 [V], U3 [V], U4 [V], U5 [V], U7 [V], U8 [V], U9 [V], U10 [V], U11 [V]]"
    good = "  active_cells: [\"U1 [V]\", \"U2 [V]\", \"U3 [V]\", \"U4 [V]\", \"U5 [V]\", \"U7 [V]\", \"U8 [V]\", \"U9 [V]\", \"U10 [V]\", \"U11 [V]\"]"
    if text.count(bad) != 1: raise RuntimeError("Original protocol parser compatibility contract failed")
    base = yaml.safe_load(text.replace(bad, good)); amend = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))
    cols = [x for x in base["data_firewall"]["frozen_mq_columns"] if x != "gap_after_flag"]
    return base, amend, cols


def assign_stratum(age: float) -> str:
    if 72.788690 <= age < 655.191706: return "EARLY"
    if 655.191706 <= age < 1237.594722: return "MIDDLE"
    if 1237.594722 <= age < 1819.997738: return "LATE"
    return "OUT"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.loc[df.bin_end_ophrs2_h < 1819.997738].copy()
    if len(d) != 3005: raise RuntimeError(f"DATA_FIREWALL_FAIL: got {len(d)} Development rows")
    d["source_row_index"] = d.index
    d["age_stratum"] = d.bin_end_ophrs2_h.map(assign_stratum)
    d["set_token"] = d.set_current_A_mean.map(lambda v: f"{float(v):.3f}" if pd.notna(v) else "NA")
    transition = (d.number_of_operation_points > 1) | (d.operation_point_transitions > 0)
    d["op_token"] = np.where(transition, "TRANSITION", d.operation_point_mode.map(lambda v: f"OP{float(v):.0f}" if pd.notna(v) else "OPNA"))
    d["token"] = d.op_token + "|" + d.set_token
    d["eligible"] = (d.coverage_ratio >= 0.80) & d.bin_valid_for_windows.astype(bool) & ~d.gap_before_flag.astype(bool) & ~d.startup_flag.astype(bool) & ~d.shutdown_flag.astype(bool) & (d.age_stratum != "OUT")
    d["pure"] = d.eligible & (d.number_of_operation_points == 1) & (d.operation_point_transitions == 0)
    d["positive_pure"] = d.pure & (d.zero_current_fraction == 0) & (d.actual_current_density_A_cm2_mean > 0)
    return d


def continuous_runs(d: pd.DataFrame, mask: pd.Series) -> list[pd.DataFrame]:
    rows = d.loc[mask].sort_values("bin_start_ophrs2_h")
    out: list[list[int]] = []; cur: list[int] = []; prev = None
    for i, r in rows.iterrows():
        linked = prev is not None and r.temporal_segment_id == prev.temporal_segment_id and r.split == prev.split and r.age_stratum == prev.age_stratum and abs(r.bin_start_ophrs2_h - prev.bin_start_ophrs2_h - .5) < 1e-6
        if not linked:
            if cur: out.append(cur)
            cur = []
        cur.append(i); prev = r
    if cur: out.append(cur)
    return [d.loc[x].copy() for x in out]


def calibrate(d: pd.DataFrame, token_column: str = "token") -> tuple[int | None, list[str], list[dict[str, Any]], pd.DataFrame]:
    early = d.loc[(d.age_stratum == "EARLY") & d.eligible].sort_values("bin_start_ophrs2_h").head(300).copy()
    runs = continuous_runs(early, pd.Series(True, index=early.index))
    scores=[]
    for p in range(12,31):
        pairs=match=0
        for run in runs:
            t=run[token_column].to_numpy()
            if len(t)>p: pairs += len(t)-p; match += int((t[p:]==t[:-p]).sum())
        scores.append({"period_steps":p,"lag_pairs":pairs,"exact_matches":match,"exact_match_rate":match/pairs if pairs else np.nan})
    valid=[r for r in scores if r["lag_pairs"] and r["exact_match_rate"]>=.70]
    if not valid: return None, [], scores, early
    best=sorted(valid,key=lambda r:(-r["exact_match_rate"],r["period_steps"]))[0]
    p=int(best["period_steps"])
    for run in runs:
        if len(run)>=p: return p, run[token_column].iloc[:p].tolist(), scores, early
    return None, [], scores, early


def scan_cycles(d: pd.DataFrame, p: int, template: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cycles=[]; exclusions=[]
    for run in continuous_runs(d, d.eligible):
        toks=run.token.tolist(); pos=0
        while pos+p <= len(run):
            block=run.iloc[pos:pos+p]
            if block.token.tolist()==template:
                cycles.append({"segment":str(block.temporal_segment_id.iloc[0]),"split":str(block.split.iloc[0]),"age_stratum":str(block.age_stratum.iloc[0]),"start_row":int(block.source_row_index.iloc[0]),"end_row":int(block.source_row_index.iloc[-1]),"start_age_h":float(block.bin_start_ophrs2_h.iloc[0]),"end_age_h":float(block.bin_end_ophrs2_h.iloc[-1]),"summary_available_at":str(block.last_datetime.iloc[-1]),"member_rows":"|".join(map(str,block.source_row_index.tolist()))})
                pos += p
            else:
                exclusions.append({"reason":"expected_template_window_mismatch","start_row":int(block.source_row_index.iloc[0]),"age_stratum":str(block.age_stratum.iloc[0])}); pos += p
    return cycles, exclusions


def gate_a(d: pd.DataFrame, p: int, template: list[str], cycles: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    avail=max((c["end_age_h"] for c in cycles[:10]), default=np.inf)
    after=[c for c in cycles if c["start_age_h"] > avail]
    rows=[]
    for st in ("EARLY","MIDDLE","LATE"):
        n=sum(c["age_stratum"]==st for c in after); eligible_bins=int(((d.age_stratum==st)&d.eligible).sum()); frac=n*p/eligible_bins if eligible_bins else 0
        rows.append({"age_stratum":st,"post_calibration_cycles":n,"eligible_bins":eligible_bins,"post_calibration_recurrence_fraction":frac})
    a0=all(r["post_calibration_recurrence_fraction"]>=.70 for r in rows)
    a1=all(r["post_calibration_cycles"]>=({"EARLY":5,"MIDDLE":8,"LATE":8}[r["age_stratum"]]) for r in rows)
    ref=cycles[:10]; ref_rows={int(x) for c in ref for x in c["member_rows"].split("|")}
    point_counts=d.loc[d.source_row_index.isin(ref_rows)&d.positive_pure].operation_point_mode.value_counts()
    points=[float(x) for x in sorted(point_counts.index, key=lambda x:(-point_counts[x],float(x)))[:2]]
    a2=len(ref)==10 and len(points)==2
    return {"A0_coarse_set_current_recurrence":a0,"A1_operation_point_sequence_recurrence":a1,"A2_pure_measurement_support":a2,"template_first_available_at_age_h":avail,"selected_positive_operation_points":points}, rows


def main() -> None:
    repo=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    if head!=EXPECTED_HEAD: raise RuntimeError(f"HEAD mismatch {head} != {EXPECTED_HEAD}")
    base, amend, cols=effective_protocol(); guard=DataAccessGuard(cols)
    n1=Path(base["data_firewall"]["n1_table"])
    if not n1.is_absolute(): n1=(PHASE/n1).resolve()
    raw=guard.read_csv(n1,nrows=3007,usecols=cols)
    d=prepare(raw)
    guard.write_audit(OUT/"ATTEMPT_2_DATA_FIREWALL_AUDIT.json",attempt_id="MQ_ATTEMPT_2_LOCAL_ISOLATED",historical_planning_layer_test_contact=True,historical_attempt_1_test_contact=True,attempt_2_github_data_content_accessed=False,attempt_2_mixed_anchor_summary_accessed=False,attempt_2_test_accessed=False,n1_prefix_rows_read=3007,development_rows_retained=3005,post_prefix_rows_read=0,full_n1_hash_performed=False,development_polarization_files_available=7,neural_training=False)
    a0_period, _, a0_scores, _=calibrate(d, "set_token")
    p, template, scores, prefix=calibrate(d, "token")
    write_csv("COARSE_CYCLE_RECURRENCE.csv",[{"gate":"A0","token":"set_current_A_mean_rounded_3_decimals",**r} for r in a0_scores])
    write_csv("OP_SEQUENCE_RECURRENCE.csv",[{"gate":"A1","token":"operation_point_or_transition_plus_set_current",**r} for r in scores])
    write_csv("CALIBRATION_PREFIX_AUDIT.csv",[{"token_family":"A0_coarse_set_current",**r} for r in a0_scores]+[{"token_family":"A1_operation_point_sequence",**r} for r in scores])
    if a0_period is None:
        status="CYCLE_REPRODUCIBILITY_NOT_IDENTIFIABLE"; write_json("GATE_EVALUATION.json",{"Gate_A":"FAIL","Gate_B":"NOT_RUN","Gate_C":"NOT_RUN","Gate_D":"NOT_RUN","Gate_E":"NOT_RUN","final_status":status}); return
    if p is None:
        status="AGGREGATION_RESOLUTION_LIMITATION"; write_json("AGGREGATION_RESOLUTION_AUDIT.json",{"A0_coarse_period_steps":a0_period,"A1_operation_point_sequence_period":"UNRESOLVED","final_status":status,"stop_before_D_E":True}); write_json("GATE_EVALUATION.json",{"Gate_A":"FAIL","Gate_B":"NOT_RUN","Gate_C":"NOT_RUN","Gate_D":"NOT_RUN","Gate_E":"NOT_RUN","final_status":status}); return
    cycles, exclusions=scan_cycles(d,p,template); ga, recurrence=gate_a(d,p,template,cycles)
    write_json("CYCLE_TEMPLATE.json",{"primary_definition":"OPERATION_POINT_SEQUENCE","period_steps":p,"duration_h":p*.5,"tokens":template,"fixed_19_sensitivity_not_primary":True})
    write_csv("CYCLE_SEGMENTS.csv",cycles); write_csv("CYCLE_EXCLUSIONS.csv",exclusions); write_csv("CYCLE_REPRODUCIBILITY.csv",recurrence)
    write_json("TEMPLATE_AVAILABILITY_AUDIT.json",{"period_steps":p,"template_first_available_at_age_h":ga["template_first_available_at_age_h"],"reference_cycles":10})
    if not (ga["A0_coarse_set_current_recurrence"] and ga["A1_operation_point_sequence_recurrence"] and ga["A2_pure_measurement_support"]):
        status="AGGREGATION_RESOLUTION_LIMITATION" if ga["A0_coarse_set_current_recurrence"] else "CYCLE_REPRODUCIBILITY_NOT_IDENTIFIABLE"
        write_json("AGGREGATION_RESOLUTION_AUDIT.json",{**ga,"final_status":status,"stop_before_D_E":True})
        write_json("GATE_EVALUATION.json",{"Gate_A":"FAIL","Gate_B":"NOT_RUN","Gate_C":"NOT_RUN","Gate_D":"NOT_RUN","Gate_E":"NOT_RUN","final_status":status}); return
    write_json("GATE_EVALUATION.json",{"Gate_A":"PASS","Gate_B":"NOT_IMPLEMENTED_ATTEMPT_2_STOP","Gate_C":"NOT_RUN","Gate_D":"NOT_RUN","Gate_E":"NOT_RUN","final_status":"REALHYFC_MECHANISM_NOT_IDENTIFIABLE"})

if __name__ == "__main__": main()

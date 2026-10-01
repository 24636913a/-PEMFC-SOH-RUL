"""Narrow A0 check: segment-first gap handling only; no template or later gates."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mq_access import DataAccessGuard
from run_attempt_2_local import effective_protocol

PHASE = Path(__file__).resolve().parents[1]
OUT = PHASE / "outputs" / "a0_implementation_audit"
EXPECTED_HEAD = "075bb1cbdd571cc721e059b7136fadfc4b9c52f9"


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def age_stratum(age: float) -> str:
    return "EARLY" if 72.788690 <= age < 655.191706 else "OTHER"


def base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    dev = frame.loc[frame.bin_end_ophrs2_h < 1819.997738].copy()
    if len(dev) != 3005:
        raise RuntimeError(f"DATA_FIREWALL_FAIL: expected 3005, observed {len(dev)}")
    dev["source_row"] = dev.index
    dev["age_stratum"] = dev.bin_end_ophrs2_h.map(age_stratum)
    dev["a0_token"] = dev.set_current_A_mean.map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "NA")
    dev["base_eligible"] = (dev.coverage_ratio >= .80) & dev.bin_valid_for_windows.astype(bool) & ~dev.startup_flag.astype(bool) & ~dev.shutdown_flag.astype(bool) & (dev.age_stratum == "EARLY")
    dev["segment_first_row"] = dev.groupby("temporal_segment_id", sort=False)["source_row"].transform("min") == dev.source_row
    return dev


def prefix_and_runs(dev: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, list[pd.DataFrame], list[dict[str, Any]]]:
    if variant == "CURRENT":
        eligible = dev.base_eligible & ~dev.gap_before_flag.astype(bool)
    elif variant == "COMPLIANT":
        # All valid bins are retained. An internal gap starts a new run but is not deleted.
        eligible = dev.base_eligible
    else:
        raise ValueError(variant)
    prefix = dev.loc[eligible].sort_values("bin_start_ophrs2_h").head(300).copy()
    runs: list[list[int]] = []; current: list[int] = []; previous = None
    for idx, row in prefix.iterrows():
        linked = previous is not None and row.temporal_segment_id == previous.temporal_segment_id and abs(row.bin_start_ophrs2_h - previous.bin_start_ophrs2_h - .5) < 1e-9
        if variant == "COMPLIANT" and bool(row.gap_before_flag) and not bool(row.segment_first_row):
            linked = False
        if not linked:
            if current: runs.append(current)
            current=[]
        current.append(idx); previous=row
    if current: runs.append(current)
    blocks=[prefix.loc[indexes].copy() for indexes in runs]
    audit=[]
    for block in blocks:
        first=block.iloc[0]
        audit.append({"variant":variant,"temporal_segment_id":first.temporal_segment_id,"start_row":int(first.source_row),"end_row":int(block.source_row.iloc[-1]),"start_age_h":float(first.bin_start_ophrs2_h),"end_age_h":float(block.bin_end_ophrs2_h.iloc[-1]),"run_length":len(block),"first_bin_gap_before_flag":bool(first.gap_before_flag),"first_bin_removed_in_current":bool(first.gap_before_flag and first.segment_first_row),"startup_exclusions":int((~dev.startup_flag.astype(bool) & ~dev.base_eligible).sum()),"shutdown_exclusions":int((~dev.shutdown_flag.astype(bool) & ~dev.base_eligible).sum()),"invalid_bin_exclusions":int((~dev.bin_valid_for_windows.astype(bool) & ~dev.base_eligible).sum())})
    return prefix, blocks, audit


def lag_rows(blocks: list[pd.DataFrame]) -> list[dict[str, Any]]:
    rows=[]
    for p in range(12,31):
        pairs=matches=0
        for block in blocks:
            values=block.a0_token.to_numpy(); count=max(len(values)-p,0)
            pairs += count
            if count: matches += int((values[p:] == values[:-p]).sum())
        rows.append({"period_steps":p,"max_run_length":max((len(b) for b in blocks),default=0),"lag_pairs":pairs,"exact_matches":matches,"exact_match_rate":matches/pairs if pairs else np.nan})
    return rows


def classify(current: list[dict[str, Any]], compliant: list[dict[str, Any]]) -> tuple[str,str]:
    c_pairs=[r["lag_pairs"] for r in current]; k_pairs=[r["lag_pairs"] for r in compliant]
    if any(c_pairs): return "ORIGINAL_ATTEMPT_NOT_REPRODUCIBLE", "TECHNICAL_REVIEW"
    transitions=any(k>0 for k in k_pairs)
    if not transitions: return "ZERO_LAG_PAIRS_ROBUST", "STOP_0P5H_CYCLE_ROUTE_AND_START_RAW_RESOLUTION_AUDIT"
    rates=[r["exact_match_rate"] for r in compliant if r["lag_pairs"]>0]
    if rates and all(r<.70 for r in rates): return "IMPLEMENTATION_DIFFERENCE_PRESENT_BUT_A0_DECISION_UNCHANGED", "TECHNICAL_REVIEW"
    return "ZERO_LAG_PAIRS_NOT_ROBUST", "MQ_ATTEMPT_3_IMPLEMENTATION_CORRECTED"


def main() -> None:
    repo=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    if head != EXPECTED_HEAD: raise RuntimeError(f"HEAD mismatch: {head}")
    base, _, columns=effective_protocol(); guard=DataAccessGuard(columns)
    path=(PHASE / base["data_firewall"]["n1_table"]).resolve()
    raw=guard.read_csv(path,nrows=3007,usecols=columns)
    dev=base_frame(raw)
    cur_prefix, cur_blocks, cur_audit=prefix_and_runs(dev,"CURRENT")
    cmp_prefix, cmp_blocks, cmp_audit=prefix_and_runs(dev,"COMPLIANT")
    current=lag_rows(cur_blocks); compliant=lag_rows(cmp_blocks)
    if any(r["lag_pairs"] for r in current):
        result, action="ORIGINAL_ATTEMPT_NOT_REPRODUCIBLE","TECHNICAL_REVIEW"
    else:
        result, action=classify(current,compliant)
    combined=[]
    for a,b in zip(current,compliant):
        combined.append({"period_steps":a["period_steps"],"current_max_run_length":a["max_run_length"],"current_lag_pairs":a["lag_pairs"],"current_exact_matches":a["exact_matches"],"current_exact_match_rate":a["exact_match_rate"],"compliant_max_run_length":b["max_run_length"],"compliant_lag_pairs":b["lag_pairs"],"compliant_exact_matches":b["exact_matches"],"compliant_exact_match_rate":b["exact_match_rate"],"lag_pair_delta":b["lag_pairs"]-a["lag_pairs"]})
    write_csv("A0_SEGMENT_LENGTH_AUDIT.csv",cur_audit+cmp_audit)
    write_csv("A0_LAG_PAIR_COMPARISON.csv",combined)
    payload={"base_commit":EXPECTED_HEAD,"current_implementation_reproduced":not any(r["lag_pairs"] for r in current),"calibration_bins":{"CURRENT":len(cur_prefix),"COMPLIANT":len(cmp_prefix)},"current_max_continuous_run":max((len(b) for b in cur_blocks),default=0),"compliant_max_continuous_run":max((len(b) for b in cmp_blocks),default=0),"current_periods_with_lag_pairs":[r["period_steps"] for r in current if r["lag_pairs"]>0],"compliant_periods_with_lag_pairs":[r["period_steps"] for r in compliant if r["lag_pairs"]>0],"largest_lag_pair_delta":max((r["lag_pair_delta"] for r in combined),default=0),"any_zero_to_positive_transition":any(r["current_lag_pairs"]==0 and r["compliant_lag_pairs"]>0 for r in combined),"final_technical_result":result,"recommended_next_action":action,"neural_training":False,"polarization_read":False,"test_accessed":False,"gates_B_to_E":"NOT_RUN","data_access":guard.events}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"A0_IMPLEMENTATION_AUDIT.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report=f"# Phase MQ A0 Implementation Audit\n\nResult: **{result}**.\n\nThe CURRENT path reproduces the Attempt 2 all-zero lag-pair result. The COMPLIANT path retains a temporal segment's first valid `gap_before_flag=1` bin, while an internal gap starts a new run without deleting that row. No period selection, template, A1/A2, Gate B–E, voltage residual, polarization or Test analysis was executed.\n\n- CURRENT max run: {payload['current_max_continuous_run']}\n- COMPLIANT max run: {payload['compliant_max_continuous_run']}\n- Largest lag-pair delta: {payload['largest_lag_pair_delta']}\n- Recommended action: `{action}`\n"
    (OUT/"A0_IMPLEMENTATION_AUDIT.md").write_text(report,encoding="utf-8")

if __name__=="__main__": main()

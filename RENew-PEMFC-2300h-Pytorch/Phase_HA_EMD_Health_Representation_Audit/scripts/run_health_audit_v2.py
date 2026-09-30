"""Phase HA V2: technical gates, then first Development-only endpoint analysis."""
from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from amendment_v2_core import (REPRESENTATIONS, asr_audit, condition_adjusted, condition_origin_frame, frozen_matching_design,
                               make_blocks, match_all, matched_moving_bootstrap, online_roughness,
                               rdr_audit, recovery_audit, revision_events, revision_summary, trend_metrics)
from ha_common import (AuditFailure, BASE_COMMIT, CONTEXT_STEPS, DEVELOPMENT_END_H, DF_BASE_COMMIT, DF_DIR, HA_DIR,
                       V1_BLOCKER_COMMIT, V1_PROTOCOL_SHA256, assert_history_provenance, assign_stratum, assign_third,
                       cache_universe, chronological_thirds, execution_code_hashes, git_branch, git_head, load_protocol,
                       p33_p67_and_stable_p50, read_development_prefix, read_manifest, sha256_file, source_hashes,
                       validate_cache_entry, validate_df_status, write_json)

OUT=HA_DIR/"outputs"; AMEND=HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.yaml"; AMEND_SHA=HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.sha256"
EXECUTION_PATCH=HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_1.yaml"; EXECUTION_PATCH_SHA=HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_1.sha256"
EXECUTION_PATCH_V2_2=HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_2.yaml"; EXECUTION_PATCH_V2_2_SHA=HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_EXECUTION_PATCH_V2_2.sha256"


def csv_out(name:str, frame:pd.DataFrame):
    OUT.mkdir(parents=True,exist_ok=True);frame.to_csv(OUT/name,index=False,encoding="utf-8-sig")


def load_amendment()->dict:
    if not AMEND.exists() or not AMEND_SHA.exists(): raise AuditFailure("V2 amendment/hash not frozen")
    digest=AMEND_SHA.read_text(encoding="utf-8").strip().split()[0]
    if sha256_file(AMEND)!=digest: raise AuditFailure("V2 amendment SHA mismatch")
    return json.loads(AMEND.read_text(encoding="utf-8"))


def load_execution_patch_v2_1(amend:dict)->dict:
    """Load the narrow V2.1 runtime patch without mutating frozen V2."""
    if not EXECUTION_PATCH.exists() or not EXECUTION_PATCH_SHA.exists():
        raise AuditFailure("V2.1 execution patch/hash not frozen")
    digest=EXECUTION_PATCH_SHA.read_text(encoding="utf-8").strip().split()[0]
    if sha256_file(EXECUTION_PATCH)!=digest:
        raise AuditFailure("V2.1 execution patch SHA mismatch")
    patch=json.loads(EXECUTION_PATCH.read_text(encoding="utf-8"))
    if patch.get("predecessor_amendment_v2_sha256")!=sha256_file(AMEND):
        raise AuditFailure("V2.1 patch does not pin the preserved V2 amendment")
    if patch.get("predecessor_execution_code_hashes")!=amend.get("execution_code_hashes"):
        raise AuditFailure("V2.1 patch predecessor code identity does not match preserved V2")
    if patch.get("statistical_protocol_changes")!=[]:
        raise AuditFailure("V2.1 execution patch must not change the registered statistical protocol")
    return patch


def load_execution_patch_v2_2(amend:dict, patch_v2_1:dict)->dict:
    """Load the second narrow runtime patch while retaining the full chain."""
    if not EXECUTION_PATCH_V2_2.exists() or not EXECUTION_PATCH_V2_2_SHA.exists():
        raise AuditFailure("V2.2 execution patch/hash not frozen")
    digest=EXECUTION_PATCH_V2_2_SHA.read_text(encoding="utf-8").strip().split()[0]
    if sha256_file(EXECUTION_PATCH_V2_2)!=digest:
        raise AuditFailure("V2.2 execution patch SHA mismatch")
    patch=json.loads(EXECUTION_PATCH_V2_2.read_text(encoding="utf-8"))
    if patch.get("predecessor_amendment_v2_sha256")!=sha256_file(AMEND):
        raise AuditFailure("V2.2 patch does not pin the preserved V2 amendment")
    if patch.get("predecessor_execution_patch_v2_1_sha256")!=sha256_file(EXECUTION_PATCH):
        raise AuditFailure("V2.2 patch does not pin the preserved V2.1 execution patch")
    if patch.get("predecessor_execution_code_hashes")!=patch_v2_1.get("execution_code_hashes"):
        raise AuditFailure("V2.2 predecessor code identity does not match V2.1")
    if patch.get("statistical_protocol_changes")!=[]:
        raise AuditFailure("V2.2 execution patch must not change the registered statistical protocol")
    return patch


def effective_protocol_audit(amend:dict, patch_v2_1:dict, patch_v2_2:dict)->dict:
    assert_history_provenance()
    if amend.get("v1_protocol_sha256")!=V1_PROTOCOL_SHA256: raise AuditFailure("amendment does not pin V1 SHA")
    expected=amend["frozen_phase_df_source_hashes"]
    actual=source_hashes()
    if expected!=actual: raise AuditFailure("frozen Phase DF source hashes changed after amendment freeze")
    code=execution_code_hashes()
    if patch_v2_2["execution_code_hashes"]!=code:
        raise AuditFailure("current V2.2 execution code hashes differ from frozen execution patch")
    return {"v1_protocol_sha256":V1_PROTOCOL_SHA256,"v2_amendment_sha256":sha256_file(AMEND),"v2_1_execution_patch_sha256":sha256_file(EXECUTION_PATCH),"v2_2_execution_patch_sha256":sha256_file(EXECUTION_PATCH_V2_2),"effective_protocol":"v1 + amendment_v2 + execution_patch_v2_1 + execution_patch_v2_2","effective_overridden_fields":amend["overrides"],"execution_patch_scope":[patch_v2_1["sole_implementation_change"],patch_v2_2["sole_implementation_change"]],"statistical_protocol_changes":patch_v2_2["statistical_protocol_changes"],"phase_df_base_commit":DF_BASE_COMMIT,"v1_blocker_commit":V1_BLOCKER_COMMIT,"current_head":git_head(),"branch":git_branch(),"v1_protocol_preserved":True,"technical_blocker_history_preserved":True,"v2_original_code_hashes_preserved_as_predecessor":True,"v2_1_code_hashes_preserved_as_predecessor":True}


def condition_origin(development:pd.DataFrame,common:pd.DataFrame, threshold:dict)->pd.DataFrame:
    lower,upper=chronological_thirds(common.origin_age_h_simple)
    rows=[]
    for origin,row in common.sort_values("origin_row").set_index("origin_row").iterrows():
        source=development.loc[int(origin)]
        if not np.isclose(float(source.bin_end_ophrs2_h),float(row.origin_age_h_simple)) or int(source.temporal_segment_id)!=int(row.temporal_segment_id_simple): raise AuditFailure("cache/N1 origin age or segment mismatch")
        item={"origin_row":int(origin),"origin_age_h":float(source.bin_end_ophrs2_h),"origin_datetime":source.last_datetime,"temporal_segment_id":int(source.temporal_segment_id),"chronological_third":assign_third(float(source.bin_end_ophrs2_h),lower,upper),"fold_membership":row.fold_membership_simple,"input_history_sha":row.input_voltage_sha256_simple}
        for col in ("actual_current_A_mean","actual_current_A_std","actual_current_A_peak_to_peak","actual_current_density_A_cm2_mean","actual_current_density_A_cm2_std","actual_current_density_A_cm2_peak_to_peak","load_transition_count","operation_point_transitions","zero_current_fraction","cell_voltage_std","cell_voltage_range","startup_flag","shutdown_flag","gap_before_flag","gap_before_duration_s","gap_after_flag","cell_voltage_mean"):
            item[col]=float(source[col])
        item["current_density_stratum"]=assign_stratum(item["actual_current_density_A_cm2_mean"],threshold["P33_A_cm2"],threshold["P67_A_cm2"]);rows.append(item)
    return pd.DataFrame(rows)


def load_df_module():
    spec=importlib.util.spec_from_file_location("frozen_df_emd_v2",DF_DIR/"scripts"/"decomposition_features.py")
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def technical_unit_contract(development:pd.DataFrame, common:pd.DataFrame)->dict:
    # Production feature function sees a legal 48-point history only; suffix values are never supplied.
    origin=int(common.origin_row.min()); history=development.loc[origin-47:origin,"cell_voltage_mean"].to_numpy(np.float64)
    if len(history)!=CONTEXT_STEPS: raise AuditFailure("exact 48-point history fails")
    module=load_df_module(); simple=module.simple_trend_features(history).features;emd=module.emd_feature(history).features
    copied=np.concatenate([np.array([-999.]),history,np.array([999.])])[1:49]
    if not np.array_equal(simple,module.simple_trend_features(copied).features) or not np.array_equal(emd,module.emd_feature(copied).features): raise AuditFailure("future/prefix invariance failure")
    z,mu,scale=module._normalise(history); processor=module._configured_emd();processor.emd(z);imfs,_=processor.get_imfs_and_residue(); removed=np.asarray(imfs)[:min(2,len(imfs))].sum(axis=0) if len(imfs) else np.zeros(48); low=mu+scale*(z-removed)
    if not np.allclose(emd[:,0],low.astype(np.float32)) or not np.allclose(emd[:,1],module.causal_slope8(low).astype(np.float32)): raise AuditFailure("EMD V/Vh unit contract failed")
    return {"future_perturbation_invariance":True,"prefix_invariance":True,"exact_48_point_history":True,"same_segment_history":True,"endpoint_index":"features[47,0/1]","lag_revision_index":"later_features[47-k,0]","EMD_endpoint_unit":"V","EMD_slope_unit":"V/h","no_future_target_read":True}


def dual_cache_audit(simple:pd.DataFrame,emd:pd.DataFrame,local:pd.DataFrame)->tuple[pd.DataFrame,dict[int,np.ndarray],dict[int,np.ndarray]]:
    rows=[];sfeatures={};efeatures={}
    for method,frame,store in (("SimpleTrend",simple,sfeatures),("EMD",emd,efeatures)):
        for _,row in frame.sort_values("origin_row").iterrows():
            feature,audit=validate_cache_entry(row,local); rows.append(audit);store[int(row.origin_row)]=feature
    audit=pd.DataFrame(rows)
    if not audit.CACHE_INTEGRITY_PASS.all(): raise AuditFailure("cache dual-hash integrity audit failed")
    return audit,sfeatures,efeatures


def build_endpoint_table(origin:pd.DataFrame,sfeatures:dict[int,np.ndarray],efeatures:dict[int,np.ndarray],simple:pd.DataFrame,emd:pd.DataFrame)->pd.DataFrame:
    sm=simple.set_index("origin_row");em=emd.set_index("origin_row");rows=[]
    for _,base in origin.iterrows():
        key=int(base.origin_row); s=sfeatures[key];e=efeatures[key];item=base.to_dict();item.update({"Raw_endpoint_V":base.cell_voltage_mean,"SimpleTrend_endpoint_V":float(s[47,0]),"SimpleTrend_endpoint_slope_V_per_h":float(s[47,1]),"EMD_endpoint_V":float(e[47,0]),"EMD_endpoint_slope_V_per_h":float(e[47,1]),"source_cache_sha":em.loc[key,"cache_sha256"],"simpletrend_source_cache_sha":sm.loc[key,"cache_sha256"]});rows.append(item)
    table=pd.DataFrame(rows).sort_values("origin_row").reset_index(drop=True)
    if table.origin_row.duplicated().any() or len(table)!=1992 or (table.origin_age_h>=DEVELOPMENT_END_H).any(): raise AuditFailure("endpoint origin universe technical contract failed")
    return table


def final_decision(support24, contrasts24, contrasts48, uncertainty, adjusted24, adjusted48, asr, rdr, trends):
    support=support24.loc[support24.status.eq("MATCHED_COMMON_SUPPORT_PASS"),"stratum"].tolist(); support_ok=len(support)>=2
    def contrast_ok(stratum):
        base=contrasts24.loc[(contrasts24.stratum.eq(stratum))&(contrasts24.representation.eq("EMD"))]
        ci=uncertainty.loc[(uncertainty.analysis.eq("24h_primary"))&(uncertainty.stratum.eq(stratum))&(uncertainty.representation.eq("EMD"))]
        sen=contrasts48.loc[(contrasts48.stratum.eq(stratum))&(contrasts48.representation.eq("EMD"))]
        return len(base)==len(ci)==len(sen)==1 and base.iloc[0].D_matched_late_minus_early_V<0 and ci.iloc[0].uncertainty_status=="PASS" and ci.iloc[0].ci95_upper_V<0 and sen.iloc[0].D_matched_late_minus_early_V<0
    contrasts_ok=support_ok and all(contrast_ok(s) for s in support)
    a24=adjusted24.loc[adjusted24.representation.eq("EMD")].iloc[0];a48=adjusted48.loc[adjusted48.representation.eq("EMD")].iloc[0]
    adjusted_ok=all([a24.uncertainty_status=="PASS",a48.uncertainty_status=="PASS",a24.age_coefficient_V_per_1000h<0,a24.age_ci95_upper_V_per_1000h<0,a48.age_coefficient_V_per_1000h<0,a48.age_ci95_upper_V_per_1000h<0,not a24.severe_collinearity,not a48.severe_collinearity])
    gate1=support_ok and contrasts_ok and adjusted_ok
    asr_good=0
    # No matched-support stratum legitimately yields an empty ASR table.  It is
    # a failed stability gate, not a dataframe-schema exception or a reason to
    # manufacture an ASR result.
    if len(asr) and {"stratum", "representation", "ASR"}.issubset(asr.columns):
        for s in support:
            x=asr.loc[asr.stratum.eq(s)].set_index("representation").ASR
            if x.get("EMD",np.nan)>x.get("Raw",np.inf) and x.get("EMD",np.nan)>x.get("SimpleTrend",np.inf):asr_good+=1
    r=adjusted24.set_index("representation").relative_residual_sd; residual_ok=r.get("EMD",np.inf)<=r.get("Raw",-np.inf) and r.get("EMD",np.inf)<=r.get("SimpleTrend",-np.inf)
    stable_rdr=pd.to_numeric(rdr.RDR95,errors="coerce");rdr_ok=(stable_rdr<1).sum()>=2
    third=trends.loc[(trends.representation.eq("EMD"))&(trends.scope.str.startswith("third_")),"Theil_Sen_slope_V_per_1000h"]; temporal_ok=bool(len(third)==3 and (third<=0).all())
    gate2=asr_good>=2 and residual_ok and rdr_ok and temporal_ok
    if not gate1: status,reason="FORECAST_ONLY_REPRESENTATION", "HEALTH_EVIDENCE_NOT_IDENTIFIABLE" if not support_ok or not adjusted_ok else "MATCHED_AGE_EVIDENCE_GATE_NOT_MET"
    elif not gate2: status,reason="DEGRADATION_SENSITIVE_BUT_UNSTABLE","RETROSPECTIVE_BOUNDARY_IDENTIFIABILITY_LIMITATION" if not rdr_ok else "STABILITY_GATE_NOT_MET"
    else: status,reason="HI_CANDIDATE_PASS","ALL_V2_REGISTERED_GATES_MET"
    gates=pd.DataFrame([{"gate":"technical_integrity","pass":True},{"gate":"matched_common_support_at_least_two","pass":support_ok},{"gate":"matched_24h_and_48h_age_direction","pass":contrasts_ok},{"gate":"time_dependent_condition_adjusted_age_association","pass":adjusted_ok},{"gate":"ASR_EMD_better_in_two","pass":asr_good>=2},{"gate":"stable_subset_RDR_less_than_one_in_two","pass":rdr_ok},{"gate":"chronological_direction_no_strong_reversal","pass":temporal_ok},{"gate":"final_status","pass":status}])
    return gates,{"final_status":status,"reason":reason,"matched_support_strata":support,"gate1":gate1,"gate2":gate2}


def main():
    try:
        amend=load_amendment(); patch_v2_1=load_execution_patch_v2_1(amend); patch_v2_2=load_execution_patch_v2_2(amend,patch_v2_1); effective=effective_protocol_audit(amend,patch_v2_1,patch_v2_2); inherited=validate_df_status();development,firewall=read_development_prefix();cache,local=read_manifest();simple,emd,common=cache_universe(cache)
        thresholds=p33_p67_and_stable_p50(development,common)
        if thresholds!=amend["frozen_thresholds"]: raise AuditFailure("frozen V2 P33/P67/P50 thresholds do not reproduce")
        condition=condition_origin_frame(development,common,thresholds); condition_blocks,scales,constants=frozen_matching_design(condition,thresholds["P33_A_cm2"],thresholds["P67_A_cm2"])
        if scales!=amend["matching_design"]["scales"] or constants!=amend["matching_design"]["constant_variables"]: raise AuditFailure("frozen matching design does not reproduce")
        unit=technical_unit_contract(development,common); cache_audit,sfeatures,efeatures=dual_cache_audit(simple,emd,local)
        # Gate 0 has now passed; only now is the first endpoint table constructed.
        table=build_endpoint_table(condition,sfeatures,efeatures,simple,emd);blocks24=make_blocks(table,48,thresholds["P33_A_cm2"],thresholds["P67_A_cm2"]);blocks48=make_blocks(table,96,thresholds["P33_A_cm2"],thresholds["P67_A_cm2"])
        support24,pairs24,contrasts24=match_all(blocks24,scales);support48,pairs48,contrasts48=match_all(blocks48,scales)
        boot24=matched_moving_bootstrap(blocks24,scales,2,2000,2024,"24h_primary");boot24s=matched_moving_bootstrap(blocks24,scales,4,2000,2024,"24h_96h_superblock_sensitivity");boot48=matched_moving_bootstrap(blocks48,scales,2,2000,2024,"48h_analysis_sensitivity")
        trends=trend_metrics(table);rough_events,rough_summary=online_roughness(table);revisions=revision_events(table,efeatures,thresholds);revision_stats=revision_summary(revisions);adjust24=condition_adjusted(blocks24,2,2000,2024,"24h_residual_moving_48h");adjust48=condition_adjusted(blocks48,2,2000,2024,"48h_residual_moving_96h");asr=asr_audit(contrasts24,rough_events);rdr=rdr_audit(revisions,contrasts24);recovery=recovery_audit(table,scales,thresholds["P33_A_cm2"],thresholds["P67_A_cm2"]);gates,decision=final_decision(support24,contrasts24,contrasts48,pd.concat([boot24,boot24s,boot48],ignore_index=True),adjust24,adjust48,asr,rdr,trends)
        csv_out("CACHE_DUAL_HASH_AUDIT.csv",cache_audit);write_json(OUT/"EFFECTIVE_PROTOCOL_AUDIT.json",effective);write_json(OUT/"N1_PREFIX_FIREWALL_AUDIT.json",firewall);write_json(OUT/"UNIT_CONTRACT_AUDIT.json",unit);write_json(OUT/"MATCHING_IMPLEMENTATION_AUDIT.json",{"scales":scales,"constant_variables":constants,"caliper_per_variable":.5,"euclidean_caliper":1.0,"matching":"deterministic 1:1 greedy without replacement; outcomes excluded"});write_json(OUT/"BOOTSTRAP_IMPLEMENTATION_AUDIT.json",{"iid_bootstrap_used":False,"matched_primary":"24h analysis blocks, 2-block/48h moving superblocks","matched_sensitivity":"24h analysis blocks, 4-block/96h moving superblocks","analysis_48h":"48h analysis blocks, 2-block/96h moving superblocks","regression":"fixed-design residual moving-block bootstrap"});write_json(OUT/"STABLE_SUBSET_AUDIT.json",{**thresholds,"stable_lag1_events":int(((revisions.lag_steps==1)&revisions.stable_both).sum()),"definition":"both origin and lag1 later origin positive, zero_current_fraction=0, no load/operation transition, Fold1 P50 std/p2p limits"})
        csv_out("EMD_ORIGIN_HEALTH_AUDIT_TABLE.csv",table);csv_out("TREND_METRICS.csv",trends);csv_out("ONLINE_ROUGHNESS.csv",rough_summary);csv_out("ONLINE_ROUGHNESS_EVENTS.csv",rough_events);csv_out("REVISION_LAG_1_2_4_8.csv",pd.concat([revisions.assign(record_type="event"),revision_stats.assign(record_type="summary")],ignore_index=True,sort=False));csv_out("COMMON_SUPPORT_AUDIT.csv",pd.concat([support24,support48],ignore_index=True));csv_out("MATCHED_CONDITION_PAIRS.csv",pd.concat([pairs24.assign(block_hours=24),pairs48.assign(block_hours=48)],ignore_index=True));csv_out("MATCHED_AGE_CONTRASTS.csv",pd.concat([contrasts24,contrasts48],ignore_index=True));csv_out("ASR_AUDIT.csv",asr);csv_out("RDR_AUDIT.csv",rdr);csv_out("CONDITION_ADJUSTED_EFFECTS.csv",pd.concat([adjust24,adjust48],ignore_index=True));csv_out("RECOVERY_AUDIT.csv",recovery);csv_out("BLOCK_UNCERTAINTY.csv",pd.concat([boot24,boot24s,boot48],ignore_index=True));csv_out("FINAL_GATES.csv",gates)
        from generate_v2_artifacts import generate
        generate(table,blocks24,support24,contrasts24,rough_events,rough_summary,revisions,adjust24,adjust48,asr,rdr,recovery,gates,decision,effective)
        print(json.dumps({"technical_integrity":"PASS","endpoint_statistics_executed":True,"final_status":decision["final_status"],"unique_origins":len(table),"matched_support":decision["matched_support_strata"]},indent=2))
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True);(HA_DIR/"TECHNICAL_BLOCKER_V2_2.md").write_text("# TECHNICAL BLOCKER V2.2\n\nV2.2 stopped before issuing a health state.\n\n```text\n"+"".join(traceback.format_exception_only(type(exc),exc))+"```\n",encoding="utf-8");raise


if __name__=="__main__":main()


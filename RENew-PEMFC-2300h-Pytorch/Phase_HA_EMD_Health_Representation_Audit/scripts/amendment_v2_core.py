"""Outcome-free matching and time-dependent uncertainty implementation for HA V2."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ha_common import (AuditFailure, CONDITION_COLUMNS, RESOLUTION_H, assign_stratum, assign_third, chronological_thirds,
                       joint_match, matching_scales, moving_block_indices, quantile_summary,
                       valid_runs)

REPRESENTATIONS = ("Raw", "SimpleTrend", "EMD")


def condition_origin_frame(development: pd.DataFrame, common: pd.DataFrame, threshold: dict) -> pd.DataFrame:
    """Cache-universe origins plus N1 operating conditions, without feature endpoints."""
    lower, upper = chronological_thirds(common.origin_age_h_simple)
    rows=[]
    for origin,row in common.sort_values("origin_row").set_index("origin_row").iterrows():
        source=development.loc[int(origin)]
        if not np.isclose(float(source.bin_end_ophrs2_h),float(row.origin_age_h_simple)) or int(source.temporal_segment_id)!=int(row.temporal_segment_id_simple):
            raise AuditFailure("cache/N1 origin age or segment mismatch")
        item={"origin_row":int(origin),"origin_age_h":float(source.bin_end_ophrs2_h),"origin_datetime":source.last_datetime,"temporal_segment_id":int(source.temporal_segment_id),"chronological_third":assign_third(float(source.bin_end_ophrs2_h),lower,upper),"fold_membership":row.fold_membership_simple,"input_history_sha":row.input_voltage_sha256_simple}
        for col in ("actual_current_A_mean","actual_current_A_std","actual_current_A_peak_to_peak","actual_current_density_A_cm2_mean","actual_current_density_A_cm2_std","actual_current_density_A_cm2_peak_to_peak","load_transition_count","operation_point_transitions","zero_current_fraction","cell_voltage_std","cell_voltage_range","startup_flag","shutdown_flag","gap_before_flag","gap_before_duration_s","gap_after_flag","cell_voltage_mean"):
            item[col]=float(source[col])
        item["current_density_stratum"]=assign_stratum(item["actual_current_density_A_cm2_mean"],threshold["P33_A_cm2"],threshold["P67_A_cm2"]);rows.append(item)
    return pd.DataFrame(rows)


def make_blocks(origin: pd.DataFrame, n_origins: int, p33: float, p67: float) -> pd.DataFrame:
    rows=[]
    for number, run in enumerate(valid_runs(origin, n_origins), 1):
        item={"block_id":number,"block_hours":n_origins*RESOLUTION_H,"origin_start_row":int(run.origin_row.iloc[0]),
              "origin_end_row":int(run.origin_row.iloc[-1]),"age_start_h":float(run.origin_age_h.iloc[0]),"age_end_h":float(run.origin_age_h.iloc[-1]),
              "age_midpoint_h":float(run.origin_age_h.median()),"chronological_third":run.chronological_third.iloc[0],"temporal_segment_id":int(run.temporal_segment_id.iloc[0]),"origin_count":len(run)}
        item["current_density"]=float(run.actual_current_density_A_cm2_mean.median())
        item["current_density_std"]=float(run.actual_current_density_A_cm2_std.median())
        item["load_transition_count"]=float(run.load_transition_count.median())
        item["zero_current_fraction"]=float(run.zero_current_fraction.median())
        item["current_density_stratum"]=assign_stratum(item["current_density"],p33,p67)
        for rep in REPRESENTATIONS:
            col=f"{rep}_endpoint_V"
            if col in run: item[f"{rep}_endpoint_median_V"]=float(run[col].median())
        rows.append(item)
    return pd.DataFrame(rows)


def frozen_matching_design(condition_origin: pd.DataFrame, p33: float, p67: float) -> tuple[pd.DataFrame,dict,list[str]]:
    blocks=make_blocks(condition_origin,48,p33,p67)
    fold1=blocks.loc[(blocks.age_midpoint_h>=72.788690)&(blocks.age_midpoint_h<946.393214)&(blocks.current_density>0)]
    if len(fold1)<5: raise AuditFailure("insufficient Fold1 positive-current 24h condition blocks for matching scales")
    scales, constants=matching_scales(fold1)
    return blocks,scales,constants


def _pair_outcomes(early: pd.DataFrame, late: pd.DataFrame, matched: pd.DataFrame, stratum: str) -> pd.DataFrame:
    records=[]
    for pair_id, pair in matched.reset_index(drop=True).iterrows():
        e=early.loc[pair.early_index]; l=late.loc[pair.late_index]
        row={"stratum":stratum,"pair_id":pair_id+1,**pair.to_dict()}
        for col in CONDITION_COLUMNS: row[f"early_{col}"]=float(e[col]); row[f"late_{col}"]=float(l[col])
        for rep in REPRESENTATIONS:
            row[f"pair_delta_{rep}_V"]=float(l[f"{rep}_endpoint_median_V"]-e[f"{rep}_endpoint_median_V"])
        records.append(row)
    return pd.DataFrame(records)


def match_all(blocks: pd.DataFrame, scales: dict[str,float]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    support_rows=[]; pair_frames=[]; contrast_rows=[]
    for stratum in ("LOW","MID","HIGH"):
        early=blocks.loc[(blocks.current_density_stratum.eq(stratum))&(blocks.chronological_third.eq("EARLY"))]
        late=blocks.loc[(blocks.current_density_stratum.eq(stratum))&(blocks.chronological_third.eq("LATE"))]
        if not len(early) or not len(late):
            info={"early_raw_blocks":len(early),"late_raw_blocks":len(late),"early_joint_support_blocks":0,"late_joint_support_blocks":0,"matched_pair_count":0,"unmatched_early":0,"unmatched_late":0,"joint_bounds":{},"MATCHED_COMMON_SUPPORT_PASS":False}
            pairs=pd.DataFrame()
        else: pairs,info=joint_match(early,late,scales)
        support_rows.append({"stratum":stratum,"block_hours":int(blocks.block_hours.iloc[0]),**{k:v for k,v in info.items() if k!="joint_bounds"},"joint_bounds":str(info["joint_bounds"]),"status":"MATCHED_COMMON_SUPPORT_PASS" if info["MATCHED_COMMON_SUPPORT_PASS"] else "INSUFFICIENT_MATCHED_COMMON_SUPPORT"})
        if info["MATCHED_COMMON_SUPPORT_PASS"]:
            paired=_pair_outcomes(early,late,pairs,stratum); pair_frames.append(paired)
            for rep in REPRESENTATIONS:
                values=paired[f"pair_delta_{rep}_V"]
                contrast_rows.append({"block_hours":int(blocks.block_hours.iloc[0]),"stratum":stratum,"representation":rep,"matched_pair_count":len(values),"D_matched_late_minus_early_V":float(values.median()),"D_matched_late_minus_early_mV":float(values.median()*1000)})
    pairs=pd.concat(pair_frames,ignore_index=True) if pair_frames else pd.DataFrame()
    return pd.DataFrame(support_rows),pairs,pd.DataFrame(contrast_rows)


def matched_moving_bootstrap(blocks: pd.DataFrame, scales: dict[str,float], superblock_length: int, iterations: int, seed: int, label: str) -> pd.DataFrame:
    """Resample contiguous blocks, then recompute joint support and 1:1 matching each replicate."""
    rng=np.random.default_rng(seed); samples={(s,r):[] for s in ("LOW","MID","HIGH") for r in REPRESENTATIONS}; successful={k:0 for k in samples}
    early0=blocks.loc[blocks.chronological_third.eq("EARLY")].reset_index(drop=True); late0=blocks.loc[blocks.chronological_third.eq("LATE")].reset_index(drop=True)
    for _ in range(iterations):
        try:
            es=early0.loc[moving_block_indices(early0,superblock_length,rng)].reset_index(drop=True)
            ls=late0.loc[moving_block_indices(late0,superblock_length,rng)].reset_index(drop=True)
        except AuditFailure:
            break
        for stratum in ("LOW","MID","HIGH"):
            early=es.loc[es.current_density_stratum.eq(stratum)]; late=ls.loc[ls.current_density_stratum.eq(stratum)]
            if not len(early) or not len(late): continue
            pairs,info=joint_match(early,late,scales)
            if not info["MATCHED_COMMON_SUPPORT_PASS"]: continue
            paired=_pair_outcomes(early,late,pairs,stratum)
            for rep in REPRESENTATIONS:
                key=(stratum,rep); samples[key].append(float(paired[f"pair_delta_{rep}_V"].median())); successful[key]+=1
    records=[]
    for (stratum,rep),values in samples.items():
        rate=len(values)/iterations
        records.append({"analysis":label,"analysis_block_hours":int(blocks.block_hours.iloc[0]),"moving_superblock_analysis_blocks":superblock_length,
                        "moving_superblock_hours":int(blocks.block_hours.iloc[0]*superblock_length),"stratum":stratum,"representation":rep,
                        "bootstrap_iterations":iterations,"bootstrap_valid_replicates":len(values),"bootstrap_success_rate":rate,
                        "ci95_lower_V":float(np.quantile(values,.025)) if rate>=.8 else np.nan,"ci95_upper_V":float(np.quantile(values,.975)) if rate>=.8 else np.nan,
                        "uncertainty_status":"PASS" if rate>=.8 else "UNCERTAINTY_NOT_IDENTIFIABLE"})
    return pd.DataFrame(records)


def trend_metrics(origin: pd.DataFrame) -> pd.DataFrame:
    out=[]
    scopes=[("all_origins",origin)]+[(f"third_{x}",origin.loc[origin.chronological_third.eq(x)]) for x in ("EARLY","MIDDLE","LATE")]
    for scope,part in scopes:
        for rep in REPRESENTATIONS:
            y=part[f"{rep}_endpoint_V"].to_numpy(float); x=part.origin_age_h.to_numpy(float)
            if len(x)<3: continue
            lr=stats.linregress(x,y); sen=stats.theilslopes(y,x)[0]
            out.append({"scope":scope,"representation":rep,"n_origins":len(x),"pearson_r":stats.pearsonr(x,y).statistic,"spearman_rho":stats.spearmanr(x,y).statistic,"kendall_tau":stats.kendalltau(x,y).statistic,"OLS_slope_V_per_1000h":lr.slope*1000,"Theil_Sen_slope_V_per_1000h":sen*1000,"OLS_r_squared":lr.rvalue**2,"evidence_label":"PROJECT_EVIDENCE_DESCRIPTIVE"})
    return pd.DataFrame(out)


def online_roughness(origin: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    ordered=origin.sort_values("origin_row").reset_index(drop=True); valid=ordered.origin_row.diff().eq(1)&ordered.temporal_segment_id.eq(ordered.temporal_segment_id.shift())&np.isclose(ordered.origin_age_h.diff(),.5)
    events=[];summary=[]
    for rep in REPRESENTATIONS:
        delta=ordered[f"{rep}_endpoint_V"].diff().loc[valid].to_numpy(float); part=ordered.loc[valid,["origin_row","origin_age_h","temporal_segment_id","current_density_stratum"]].copy(); part["representation"]=rep;part["delta_V"]=delta;part["abs_delta_V"]=np.abs(delta);events.append(part)
        q=quantile_summary(delta);second=np.diff(delta)
        summary.append({"representation":rep,"n_adjacent_pairs":len(delta),"median_abs_delta_V":q["abs_median"],"median_abs_delta_mV":q["abs_median"]*1000,"p95_abs_delta_V":q["p95_abs"],"p99_abs_delta_V":q["p99_abs"],"std_delta_V":float(np.std(delta,ddof=1)),"total_variation_V_per_h":float(np.abs(delta).sum()/(len(delta)*.5)),"second_difference_std_V":float(np.std(second,ddof=1))})
    return pd.concat(events,ignore_index=True),pd.DataFrame(summary)


def revision_events(origin: pd.DataFrame, emd_features: dict[int,np.ndarray], stable: dict) -> pd.DataFrame:
    by=origin.set_index("origin_row");records=[]
    for t,feature in emd_features.items():
        for k in (1,2,4,8):
            ids=[t+i for i in range(k+1)]
            if not all(i in emd_features and i in by.index for i in ids):continue
            rows=by.loc[ids]
            if rows.temporal_segment_id.nunique()!=1 or not np.allclose(np.diff(rows.origin_age_h.to_numpy(float)),.5):continue
            later=rows.iloc[-1]; first=rows.iloc[0]
            stable_origin=(first.actual_current_density_A_cm2_mean>0 and first.zero_current_fraction==0 and first.load_transition_count==0 and first.operation_point_transitions==0 and first.actual_current_density_A_cm2_std<=stable["stable_current_density_std_P50_A_cm2"] and first.actual_current_density_A_cm2_peak_to_peak<=stable["stable_current_density_peak_to_peak_P50_A_cm2"])
            stable_later=(later.actual_current_density_A_cm2_mean>0 and later.zero_current_fraction==0 and later.load_transition_count==0 and later.operation_point_transitions==0 and later.actual_current_density_A_cm2_std<=stable["stable_current_density_std_P50_A_cm2"] and later.actual_current_density_A_cm2_peak_to_peak<=stable["stable_current_density_peak_to_peak_P50_A_cm2"])
            u=float(emd_features[t+k][47-k,0]-feature[47,0]);records.append({"origin_row":t,"origin_age_h":float(first.origin_age_h),"lag_steps":k,"lag_h":k*.5,"temporal_segment_id":int(first.temporal_segment_id),"chronological_third":first.chronological_third,"current_density_stratum":first.current_density_stratum,"load_transition_group":"transition" if first.load_transition_count>0 else "no_transition","U_reestimate_minus_online_V":u,"U_reestimate_minus_online_mV":u*1000,"abs_U_V":abs(u),"stable_origin":stable_origin,"stable_later":stable_later,"stable_both":stable_origin and stable_later})
    return pd.DataFrame(records)


def revision_summary(events: pd.DataFrame) -> pd.DataFrame:
    records=[]
    for group,col in (("all","lag_steps"),("stratum","current_density_stratum"),("load_transition","load_transition_group")):
        for key,part in events.groupby(col):
            for lag,sub in part.groupby("lag_steps"):
                records.append({"group_type":group,"group":str(key),"lag_steps":lag,**quantile_summary(sub.U_reestimate_minus_online_V)})
    return pd.DataFrame(records)


def _design(blocks: pd.DataFrame, endpoint: str):
    d=blocks.dropna(subset=[endpoint,"age_midpoint_h","current_density","current_density_std","load_transition_count","zero_current_fraction"]).sort_values(["chronological_third","temporal_segment_id","origin_start_row"]).reset_index(drop=True)
    x=np.column_stack([np.ones(len(d)),d.age_midpoint_h.to_numpy(float)/1000,d.current_density.to_numpy(float),d.current_density.to_numpy(float)**2,d.current_density_std.to_numpy(float),d.load_transition_count.to_numpy(float),d.zero_current_fraction.to_numpy(float)])
    return d,x,d[endpoint].to_numpy(float)


def residual_moving_sample(data: pd.DataFrame,residual: np.ndarray,length:int,rng:np.random.Generator)->np.ndarray:
    result=np.empty(len(data),dtype=float)
    for _,part in data.groupby(["chronological_third","temporal_segment_id"],sort=True):
        idx=list(part.index); chunks=[idx[i:i+length] for i in range(len(idx)-length+1)]
        if not chunks: raise AuditFailure("residual moving bootstrap has no legal contiguous superblock")
        sampled=[]
        while len(sampled)<len(idx): sampled.extend(chunks[int(rng.choice(len(chunks)))])
        result[idx]=residual[np.asarray(sampled[:len(idx)],dtype=int)]
    return result


def condition_adjusted(blocks: pd.DataFrame, residual_length: int, iterations:int, seed:int, label:str)->pd.DataFrame:
    rng=np.random.default_rng(seed); rows=[]
    for rep in REPRESENTATIONS:
        d,x,y=_design(blocks,f"{rep}_endpoint_median_V"); beta=np.linalg.lstsq(x,y,rcond=None)[0];resid=y-x@beta; samples=[]
        for _ in range(iterations):
            try: samples.append(float(np.linalg.lstsq(x,x@beta+residual_moving_sample(d,resid,residual_length,rng),rcond=None)[0][1]))
            except (AuditFailure,np.linalg.LinAlgError): pass
        rate=len(samples)/iterations;resid_sd=float(np.std(resid,ddof=1)); endpoint_sd=float(np.std(y,ddof=1))
        rows.append({"analysis":label,"block_hours":int(blocks.block_hours.iloc[0]),"residual_superblock_analysis_blocks":residual_length,"residual_superblock_hours":int(blocks.block_hours.iloc[0]*residual_length),"representation":rep,"formula":"endpoint ~ age_per_1000h + current_density + current_density_squared + current_density_std + load_transition_count + zero_current_fraction","n_blocks":len(d),"age_coefficient_V_per_1000h":float(beta[1]),"age_ci95_lower_V_per_1000h":float(np.quantile(samples,.025)) if rate>=.8 else np.nan,"age_ci95_upper_V_per_1000h":float(np.quantile(samples,.975)) if rate>=.8 else np.nan,"bootstrap_success_rate":rate,"uncertainty_status":"PASS" if rate>=.8 else "UNCERTAINTY_NOT_IDENTIFIABLE","condition_number":float(np.linalg.cond(x)),"severe_collinearity":float(np.linalg.cond(x))>=1000,"residual_sd_V":resid_sd,"relative_residual_sd":resid_sd/endpoint_sd if endpoint_sd>0 else np.nan,"interpretation":"condition-adjusted association, not causal effect"})
    return pd.DataFrame(rows)


def rdr_audit(events:pd.DataFrame, contrasts:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for stratum in ("LOW","MID","HIGH"):
        # With no matched-support stratum, ``contrasts`` is a deliberately
        # empty dataframe without outcome columns.  That is an
        # identifiability failure, not an exceptional schema to dereference.
        has_contrast_schema={"stratum","representation","D_matched_late_minus_early_V"}.issubset(contrasts.columns)
        c=contrasts.loc[(contrasts["stratum"].eq(stratum))&(contrasts["representation"].eq("EMD"))] if has_contrast_schema else pd.DataFrame()
        values=events.loc[(events.lag_steps.eq(1))&(events.current_density_stratum.eq(stratum))&(events.stable_both),"abs_U_V"]
        d=float(c.D_matched_late_minus_early_V.iloc[0]) if len(c) else np.nan
        status="PASS" if len(values)>=30 and np.isfinite(d) and abs(d)>1e-12 else "RDR_NOT_IDENTIFIABLE_IN_STABLE_SUBSET"
        p95=float(values.quantile(.95)) if status=="PASS" else np.nan
        rows.append({"stratum":stratum,"stable_lag1_event_count":len(values),"D_EMD_matched_V":d,"lag1_stable_revision_p95_V":p95,"RDR95":p95/abs(d) if status=="PASS" else np.nan,"status":status})
    return pd.DataFrame(rows)


def asr_audit(contrasts:pd.DataFrame, rough:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for _,c in contrasts.iterrows():
        values=rough.loc[(rough.current_density_stratum.eq(c.stratum))&(rough.representation.eq(c.representation)),"abs_delta_V"]
        med=float(values.median()) if len(values) else np.nan
        rows.append({"stratum":c.stratum,"representation":c.representation,"D_matched_late_minus_early_V":c.D_matched_late_minus_early_V,"median_abs_online_delta_V":med,"ASR":abs(c.D_matched_late_minus_early_V)/med if np.isfinite(med) and med>0 else np.nan})
    return pd.DataFrame(rows)


def recovery_audit(origin:pd.DataFrame,scales:dict[str,float],p33:float,p67:float)->pd.DataFrame:
    by=origin.set_index("origin_row");rows=[]
    for event_row,event in by.iterrows():
        candidate=(event.startup_flag>0 or event.gap_before_flag>0) and (event.gap_before_flag>0 or (event_row-1 in by.index and (by.loc[event_row-1,"shutdown_flag"]>0 or by.loc[event_row-1,"gap_after_flag"]>0)))
        pre_ids=list(range(event_row-4,event_row));post_ids=list(range(event_row,event_row+4))
        if not candidate or not all(x in by.index for x in pre_ids+post_ids):continue
        pre,post=by.loc[pre_ids],by.loc[post_ids]
        if pre.actual_current_density_A_cm2_mean.le(0).any() or post.actual_current_density_A_cm2_mean.le(0).any():continue
        if pre.temporal_segment_id.nunique()!=1 or post.temporal_segment_id.nunique()!=1 or not np.allclose(np.diff(pre.origin_age_h),.5) or not np.allclose(np.diff(post.origin_age_h),.5):continue
        pcond={"current_density":float(pre.actual_current_density_A_cm2_mean.median()),"current_density_std":float(pre.actual_current_density_A_cm2_std.median()),"load_transition_count":float(pre.load_transition_count.median()),"zero_current_fraction":float(pre.zero_current_fraction.median())}
        qcond={"current_density":float(post.actual_current_density_A_cm2_mean.median()),"current_density_std":float(post.actual_current_density_A_cm2_std.median()),"load_transition_count":float(post.load_transition_count.median()),"zero_current_fraction":float(post.zero_current_fraction.median())}
        std={c:abs(pcond[c]-qcond[c])/s for c,s in scales.items()};dist=np.sqrt(sum(x*x for x in std.values())); same_stratum=event.current_density_stratum==assign_stratum(qcond["current_density"],p33,p67)
        matched=bool(same_stratum and (not std or (max(std.values())<=.5 and dist<=1)))
        row={"event_origin_row":event_row,"event_age_h":float(event.origin_age_h),"event_metadata":"startup" if event.startup_flag>0 else "gap_return","gap_before_duration_s":float(event.gap_before_duration_s),"same_current_density_stratum":same_stratum,"condition_distance":dist,"matched_recovery_event":matched,**{f"standardized_absdiff_{c}":std.get(c,0.) for c in CONDITION_COLUMNS}}
        for rep in REPRESENTATIONS: row[f"G_{rep}_V"]=float(post[f"{rep}_endpoint_V"].median()-pre[f"{rep}_endpoint_V"].median()) if matched else np.nan
        rows.append(row)
    result=pd.DataFrame(rows)
    if len(result): result["recovery_status"]=np.where(result.matched_recovery_event.sum()>=3,"RECOVERY_CANDIDATE_ANALYSIS_AVAILABLE","RECOVERY_NOT_IDENTIFIABLE_FROM_AVAILABLE_METADATA")
    return result


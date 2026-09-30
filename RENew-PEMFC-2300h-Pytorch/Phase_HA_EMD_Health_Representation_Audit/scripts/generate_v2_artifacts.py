"""V2 publication-style audit figures and final health-representation report."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ha_common import DF_BASE_COMMIT, HA_DIR, V1_BLOCKER_COMMIT, git_branch, git_head, write_json

PALETTE={"Raw":"#0F4D92","SimpleTrend":"#42949E","EMD":"#9A4D8E","warning":"#B64342","neutral":"#CFCECE"}
FIG=HA_DIR/"figures"


def style(ax):
    ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",color="#d9d9d9",lw=.6);ax.set_axisbelow(True)


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(FIG/f"{name}.png",dpi=300,bbox_inches="tight");fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight");plt.close(fig)


def note(ax,text="No eligible data under frozen V2 rule"):
    ax.text(.5,.5,text,transform=ax.transAxes,ha="center",va="center");style(ax)


def generate(table,blocks,support,contrasts,rough_events,rough_summary,revisions,adjust24,adjust48,asr,rdr,recovery,gates,decision,effective):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.linewidth":1.3,"pdf.fonttype":42,"ps.fonttype":42})
    # 1 endpoint trajectories
    fig,axes=plt.subplots(3,1,figsize=(8.2,8),sharex=True)
    for ax,rep in zip(axes,("Raw","SimpleTrend","EMD")):
        ax.scatter(table.origin_age_h,table[f"{rep}_endpoint_V"],s=4,color=PALETTE[rep],alpha=.36,linewidths=0);ax.set_ylabel(f"{rep}\nendpoint (V)");style(ax)
    axes[-1].set_xlabel("Operating age (h)");fig.suptitle("Development-only online endpoint trajectories")
    save(fig,"01_endpoint_vs_operating_hours_v2")
    # 2 block trajectories
    fig,axes=plt.subplots(3,1,figsize=(8.2,8),sharex=True)
    for ax,rep in zip(axes,("Raw","SimpleTrend","EMD")):
        ax.plot(blocks.age_midpoint_h,blocks[f"{rep}_endpoint_median_V"],"o-",ms=3,lw=1.2,color=PALETTE[rep]);ax.set_ylabel(f"{rep}\n24 h median (V)");style(ax)
    axes[-1].set_xlabel("Operating age (h)");fig.suptitle("Non-overlapping 24 h block medians")
    save(fig,"02_block_median_trajectories_v2")
    # 3 strata
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.6),sharex=True)
    for ax,rep in zip(axes,("Raw","SimpleTrend","EMD")):
        for s,m in zip(("LOW","MID","HIGH"),("o","s","^")):
            p=table.loc[table.current_density_stratum.eq(s)];ax.scatter(p.origin_age_h,p[f"{rep}_endpoint_V"],s=5,alpha=.4,label=s,marker=m)
        ax.set_title(rep);ax.set_xlabel("Age (h)");style(ax)
    axes[0].set_ylabel("Endpoint (V)");axes[-1].legend(frameon=False,fontsize=8);fig.suptitle("Frozen current-density strata")
    save(fig,"03_endpoint_by_current_density_strata_v2")
    # 4 roughness
    fig,ax=plt.subplots(figsize=(7.2,4.2)); groups=[rough_events.loc[rough_events.representation.eq(r),"abs_delta_V"] for r in ("Raw","SimpleTrend","EMD")]
    if all(len(x) for x in groups):
        b=ax.boxplot(groups,labels=("Raw","SimpleTrend","EMD"),showfliers=False,patch_artist=True)
        for p,r in zip(b["boxes"],("Raw","SimpleTrend","EMD")):p.set_facecolor(PALETTE[r]);p.set_alpha(.7)
    else:note(ax)
    ax.set_ylabel("|adjacent online Δ endpoint| (V)");ax.set_title("Online roughness; same-segment 0.5 h pairs");style(ax);save(fig,"04_online_roughness_distribution_v2")
    # 5 revisions
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    if len(revisions):
        b=ax.boxplot([revisions.loc[revisions.lag_steps.eq(k),"U_reestimate_minus_online_V"] for k in (1,2,4,8)],labels=("1","2","4","8"),showfliers=False,patch_artist=True)
        for p in b["boxes"]:p.set_facecolor(PALETTE["EMD"]);p.set_alpha(.7)
        ax.axhline(0,color="black",lw=.8)
    else:note(ax)
    ax.set_xlabel("Retrospective lag (0.5 h steps)");ax.set_ylabel("U (V)");ax.set_title("Retrospective EMD boundary revisions");style(ax);save(fig,"05_revision_lag_distribution_v2")
    # 6 revision-age
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    if len(revisions):
        for k in (1,2,4,8):
            p=revisions.loc[revisions.lag_steps.eq(k)];ax.scatter(p.origin_age_h,p.abs_U_V,s=4,alpha=.35,label=f"k={k}")
        ax.legend(frameon=False,ncol=4,fontsize=8)
    else:note(ax)
    ax.set_xlabel("Operating age (h)");ax.set_ylabel("|U| (V)");ax.set_title("Revision magnitude vs age");style(ax);save(fig,"06_revision_vs_age_v2")
    # 7 revision operating markers
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    if len(revisions):
        for ax,col,label in ((axes[0],"current_density_stratum","Current-density stratum"),(axes[1],"load_transition_group","Load transition")):
            labels=list(revisions[col].unique());ax.boxplot([revisions.loc[revisions[col].eq(x),"abs_U_V"] for x in labels],labels=labels,showfliers=False);ax.set_xlabel(label);ax.set_ylabel("|U| (V)");style(ax)
    else:
        for ax in axes:note(ax)
    fig.suptitle("Revisions by pre-existing operating markers");save(fig,"07_revision_vs_operating_markers_v2")
    # 8 support
    fig,ax=plt.subplots(figsize=(7.2,4.2));x=np.arange(len(support));
    if len(support):
        ax.bar(x-.18,support.early_raw_blocks,.36,label="EARLY",color=PALETTE["Raw"],edgecolor="black");ax.bar(x+.18,support.late_raw_blocks,.36,label="LATE",color=PALETTE["EMD"],edgecolor="black")
        ax.set_xticks(x,support.stratum);ax.legend(frameon=False)
        for i,row in support.reset_index(drop=True).iterrows():ax.text(i,max(row.early_raw_blocks,row.late_raw_blocks)+.2,"PASS" if row.status=="MATCHED_COMMON_SUPPORT_PASS" else "FAIL",ha="center",fontsize=8)
    else:note(ax)
    ax.set_ylabel("24 h blocks");ax.set_title("Joint support and matched-pair eligibility");style(ax);save(fig,"08_joint_common_support_v2")
    # 9 contrasts
    fig,ax=plt.subplots(figsize=(8,4.5))
    if len(contrasts):
        y=np.arange(len(contrasts));vals=contrasts.D_matched_late_minus_early_mV;labels=[f"{r.stratum}\n{r.representation}" for _,r in contrasts.iterrows()];ax.scatter(vals,y,c=[PALETTE[r] for r in contrasts.representation],s=36);ax.axvline(0,color="black",lw=.8);ax.set_yticks(y,labels)
    else:note(ax)
    ax.set_xlabel("Matched late − early endpoint (mV)");ax.set_title("1:1 matched age contrasts; point estimates");style(ax);save(fig,"09_matched_age_contrasts_v2")
    # 10 adjusted effects
    fig,ax=plt.subplots(figsize=(8,4.2));adjust=pd.concat([adjust24,adjust48],ignore_index=True)
    y=np.arange(len(adjust));vals=adjust.age_coefficient_V_per_1000h*1000
    if len(adjust):
        for i,row in adjust.reset_index(drop=True).iterrows():
            lo=row.age_ci95_lower_V_per_1000h*1000;hi=row.age_ci95_upper_V_per_1000h*1000
            ax.errorbar(vals.iloc[i],i,xerr=[[vals.iloc[i]-lo],[hi-vals.iloc[i]]],fmt="o",color=PALETTE[row.representation],capsize=3)
        ax.axvline(0,color="black",lw=.8);ax.set_yticks(y,[f"{r.representation} {int(r.block_hours)}h" for _,r in adjust.iterrows()])
    else:note(ax)
    ax.set_xlabel("Age association (mV per 1000 h)");ax.set_title("Fixed condition-adjusted association; not causal effect");style(ax);save(fig,"10_condition_adjusted_age_effect_v2")
    # 11 ASR
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    if len(asr):asr.pivot(index="stratum",columns="representation",values="ASR").reindex(columns=["Raw","SimpleTrend","EMD"]).plot.bar(ax=ax,color=[PALETTE["Raw"],PALETTE["SimpleTrend"],PALETTE["EMD"]],edgecolor="black");ax.legend(frameon=False,fontsize=8)
    else:note(ax)
    ax.set_ylabel("ASR");ax.set_title("Age-signal-to-local-variation ratio");style(ax);save(fig,"11_asr_comparison_v2")
    # 12 RDR
    fig,ax=plt.subplots(figsize=(7.2,4.2));values=pd.to_numeric(rdr.RDR95,errors="coerce")
    if len(values.dropna()):ax.bar(rdr.stratum,values,color=PALETTE["EMD"],edgecolor="black");ax.axhline(1,color=PALETTE["warning"],ls="--",label="RDR95=1");ax.legend(frameon=False)
    else:note(ax,"RDR undefined in stable subset")
    ax.set_ylabel("RDR95");ax.set_title("Stable positive-current revision-to-drift ratio");style(ax);save(fig,"12_stable_subset_rdr_v2")
    report(table,support,contrasts,rough_summary,revisions,adjust24,adjust48,asr,rdr,recovery,gates,decision,effective)


def report(table,support,contrasts,rough,rev,a24,a48,asr,rdr,recovery,gates,decision,effective):
    common=support.loc[support.status.eq("MATCHED_COMMON_SUPPORT_PASS"),"stratum"].tolist();e24=a24.loc[a24.representation.eq("EMD")].iloc[0];e48=a48.loc[a48.representation.eq("EMD")].iloc[0]
    rec_n=int(recovery.matched_recovery_event.sum()) if len(recovery) else 0;lag1=rev.loc[rev.lag_steps.eq(1),"abs_U_V"]
    text=f"""# Phase HA — EMD Health Representation Audit (V2)

## Provenance and technical integrity

V1 protocol SHA `{effective['v1_protocol_sha256']}` and its technical-blocker
history remain preserved.  V2 amendment SHA `{effective['v2_amendment_sha256']}`
was frozen after that blocker and before any Phase HA endpoint statistic.
Phase DF remains `EMD_SPECIFIC_ADVANCE`; `TIMESNET_FREEZE` and backbone-search
closure remain unchanged.  N1 access was limited to its registered 3,007-row
logical prefix; no full N1 hash, post-prefix value, Test row, target, prediction,
model training, EMD parameter change, SOH, EOL, or RUL was used.

Both cache contracts passed separately: feature-payload SHA and full NPZ file
SHA.  EMD endpoint `feature[47,0]` is restored voltage in **V** and its slope
is **V/h**, as confirmed against the frozen production function.

## Registered endpoint evidence

- Unique Development cache origins: **{len(table)}**.
- Positive-current strata passing joint support and 1:1 matching: **{', '.join(common) if common else 'none'}**.
- EMD condition-adjusted age association: 24 h **{e24.age_coefficient_V_per_1000h:.4g} V/1000 h** (CI [{e24.age_ci95_lower_V_per_1000h:.4g}, {e24.age_ci95_upper_V_per_1000h:.4g}]); 48 h **{e48.age_coefficient_V_per_1000h:.4g} V/1000 h**.  These are associations, not causal effects.
- Lag-1 retrospective revision P95: **{(lag1.quantile(.95)*1000 if len(lag1) else float('nan')):.4g} mV**.  It is distinct from online roughness.
- Metadata-defined matched recovery events: **{rec_n}**; recovery is supportive only and never a hard HI gate.

## Final state

**{decision['final_status']}** — `{decision['reason']}`.

This single-trajectory Development-only endpoint decision neither constructs
SOH nor validates irreversible degradation, a physical degradation mode,
cross-stack generalization, EOL, or RUL.  Full-block Phase DF forecasting value
and endpoint HI qualification answer different questions and may differ.

## Evidence labels

- **PROJECT_EVIDENCE:** source-pinned caches and Development-only outputs.
- **LITERATURE_SUPPORTED_METHOD:** EMD and descriptive trend methods.
- **PROJECT_ANALYSIS_DESIGN:** matching calipers, blocks, moving bootstrap, ASR,
  RDR, and gates; they are not universal PHM standards.
"""
    path=HA_DIR/"reports"/"FINAL_HEALTH_REPRESENTATION_AUDIT.md";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
    write_json(HA_DIR/"final"/"FINAL_DECISION.yaml",{"phase":"Phase_HA_EMD_Health_Representation_Audit","final_status":decision["final_status"],"reason":decision["reason"],"phase_df_base_commit":DF_BASE_COMMIT,"v1_blocker_commit":V1_BLOCKER_COMMIT,"v1_protocol_preserved":True,"technical_blocker_history_preserved":True,"test_accessed":False,"full_N1_file_hash_performed":False,"neural_model_trained":False,"emd_parameters_changed":False,"Phase_DF_result_changed":False,"SOH_constructed":False,"RUL_model":False,"branch":git_branch(),"head_at_generation":git_head(),"automatic_next_phase":"PROHIBITED; await technical review"})


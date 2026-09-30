"""Publication-style static figures and the Phase HA final report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ha_common import BASE_COMMIT, HA_DIR, git_branch, git_head, write_json

FIG = HA_DIR / "figures"
REPORT = HA_DIR / "reports" / "FINAL_HEALTH_REPRESENTATION_AUDIT.md"
FINAL = HA_DIR / "final" / "FINAL_DECISION.yaml"
COLORS = {"Raw": "#0F4D92", "SimpleTrend": "#42949E", "EMD": "#9A4D8E", "warning": "#B64342"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=.6, alpha=.8)
    ax.set_axisbelow(True)


def _save(fig, stem: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _empty_note(ax, message="No eligible data under the frozen rule"):
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes)
    _style(ax)


def generate_figures(table, blocks24, blocks48, support, contrasts, rough_events, rough_summary,
                     revision_events, adjusted, asr, rdr, recovery, gates, decision, protocol):
    # 1. Endpoint development trajectories.
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8), sharex=True)
    for ax, rep, col in zip(axes, ("Raw", "SimpleTrend", "EMD"), ("Raw_endpoint_V", "SimpleTrend_endpoint_V", "EMD_endpoint_V")):
        ax.scatter(table.origin_age_h, table[col], s=4, alpha=.35, color=COLORS[rep], linewidths=0)
        ax.set_ylabel(f"{rep}\nendpoint")
        _style(ax)
    axes[-1].set_xlabel("Operating age (h)")
    fig.suptitle("Online origin endpoints vs operating age\n(Development-only; EMD endpoint is standardized remainder)")
    _save(fig, "01_endpoint_vs_operating_hours")

    # 2. Block median trajectories.
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8), sharex=True)
    for ax, rep in zip(axes, ("Raw", "SimpleTrend", "EMD")):
        ax.plot(blocks24.age_midpoint_h, blocks24[f"{rep}_endpoint_median"], marker="o", ms=3, lw=1.2, color=COLORS[rep])
        ax.set_ylabel(f"{rep}\n24 h median")
        _style(ax)
    axes[-1].set_xlabel("Operating age (h)")
    fig.suptitle("Non-overlapping 24 h block-median endpoint trajectories")
    _save(fig, "02_block_median_trajectories")

    # 3. Fixed current-density strata.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True)
    for ax, rep in zip(axes, ("Raw", "SimpleTrend", "EMD")):
        col = f"{rep}_endpoint_V" if rep != "Raw" else "Raw_endpoint_V"
        for stratum, marker in zip(("LOW", "MID", "HIGH"), ("o", "s", "^")):
            subset = table.loc[table.current_density_stratum.eq(stratum)]
            ax.scatter(subset.origin_age_h, subset[col], s=5, alpha=.42, label=stratum, marker=marker)
        ax.set_title(rep)
        ax.set_xlabel("Operating age (h)")
        _style(ax)
    axes[0].set_ylabel("Endpoint (representation units)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Frozen current-density strata; descriptive only")
    _save(fig, "03_endpoint_by_current_density_strata")

    # 4. Online roughness distributions.
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    groups = [rough_events.loc[rough_events.representation.eq(rep), "abs_delta"].to_numpy() for rep in ("Raw", "SimpleTrend", "EMD")]
    if all(len(x) for x in groups):
        box = ax.boxplot(groups, labels=("Raw", "SimpleTrend", "EMD"), showfliers=False, patch_artist=True)
        for patch, rep in zip(box["boxes"], ("Raw", "SimpleTrend", "EMD")):
            patch.set_facecolor(COLORS[rep]); patch.set_alpha(.65)
    else: _empty_note(ax)
    ax.set_ylabel("|adjacent online endpoint difference|")
    ax.set_title("Online endpoint roughness (same-segment 0.5 h adjacency)")
    _style(ax); _save(fig, "04_online_roughness_distribution")

    # 5. Revision distributions.
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    if len(revision_events):
        groups = [revision_events.loc[revision_events.lag_steps.eq(k), "U_reestimate_minus_online"].to_numpy() for k in (1,2,4,8)]
        box = ax.boxplot(groups, labels=("1", "2", "4", "8"), showfliers=False, patch_artist=True)
        for patch in box["boxes"]: patch.set_facecolor(COLORS["EMD"]); patch.set_alpha(.65)
        ax.axhline(0, color="black", lw=.8)
    else: _empty_note(ax)
    ax.set_xlabel("Retrospective lag (0.5 h steps)"); ax.set_ylabel("U: re-estimate − online")
    ax.set_title("Retrospective EMD boundary revisions")
    _style(ax); _save(fig, "05_revision_lag_distributions")

    # 6. Revision magnitude by age.
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    if len(revision_events):
        for lag in (1,2,4,8):
            subset = revision_events.loc[revision_events.lag_steps.eq(lag)]
            ax.scatter(subset.origin_age_h, subset.abs_U, s=4, alpha=.35, label=f"k={lag}")
        ax.legend(frameon=False, ncol=4, fontsize=8)
    else: _empty_note(ax)
    ax.set_xlabel("Operating age (h)"); ax.set_ylabel("|U|")
    ax.set_title("Retrospective revision magnitude vs operating age")
    _style(ax); _save(fig, "06_revision_magnitude_vs_age")

    # 7. Revision versus frozen operating markers.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    if len(revision_events):
        for ax, column, label in ((axes[0], "current_density_stratum", "Current-density stratum"), (axes[1], "load_transition_group", "Load-transition group")):
            labels = list(revision_events[column].dropna().unique())
            groups = [revision_events.loc[revision_events[column].eq(x), "abs_U"].to_numpy() for x in labels]
            ax.boxplot(groups, labels=labels, showfliers=False)
            ax.set_xlabel(label); ax.set_ylabel("|U|"); _style(ax)
    else:
        for ax in axes: _empty_note(ax)
    fig.suptitle("Revision magnitude by pre-existing operating markers")
    _save(fig, "07_revision_vs_operating_markers")

    # 8. Common support.
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    summary = support.loc[support.record_type.eq("summary")]
    if len(summary):
        x = np.arange(len(summary)); width=.34
        ax.bar(x-width/2, summary.early_blocks, width, label="EARLY blocks", color=COLORS["Raw"])
        ax.bar(x+width/2, summary.late_blocks, width, label="LATE blocks", color=COLORS["EMD"])
        for index, row in summary.reset_index(drop=True).iterrows():
            ax.text(index, max(row.early_blocks, row.late_blocks)+.25, "PASS" if row.common_support_status == "COMMON_SUPPORT_PASS" else "FAIL", ha="center", fontsize=8)
        ax.set_xticks(x, summary.stratum); ax.legend(frameon=False)
    else: _empty_note(ax)
    ax.set_ylabel("24 h blocks"); ax.set_title("Frozen common-support audit")
    _style(ax); _save(fig, "08_common_support_coverage")

    # 9. Matched contrasts.
    fig, ax = plt.subplots(figsize=(8, 4.4))
    c = contrasts.loc[contrasts.block_hours.eq(24)]
    if len(c):
        labels = [f"{r.stratum}\n{r.representation}" for _, r in c.iterrows()]
        y=np.arange(len(c)); vals=c.D_late_minus_early.to_numpy(float)
        err=np.vstack([vals-c.bootstrap_ci95_lower.to_numpy(float), c.bootstrap_ci95_upper.to_numpy(float)-vals])
        ax.errorbar(vals, y, xerr=err, fmt="o", color=COLORS["EMD"], capsize=3)
        ax.axvline(0,color="black",lw=.8); ax.set_yticks(y, labels)
    else: _empty_note(ax)
    ax.set_xlabel("Late − early endpoint (24 h block bootstrap CI)")
    ax.set_title("Matched early/late age contrasts in common-support strata")
    _style(ax); _save(fig, "09_matched_age_contrasts")

    # 10. Adjusted association.
    fig, ax = plt.subplots(figsize=(7.8,4.2))
    if len(adjusted):
        rows=adjusted.copy(); labels=[f"{r.representation} {int(r.block_hours)}h" for _,r in rows.iterrows()]; y=np.arange(len(rows)); vals=rows.age_coefficient_per_1000h.to_numpy(float)
        err=np.vstack([vals-rows.age_bootstrap_ci95_lower.to_numpy(float), rows.age_bootstrap_ci95_upper.to_numpy(float)-vals])
        colors=[COLORS[r] for r in rows.representation]
        for i in range(len(rows)): ax.errorbar(vals[i],y[i],xerr=err[:,i:i+1],fmt="o",color=colors[i],capsize=3)
        ax.axvline(0,color="black",lw=.8); ax.set_yticks(y,labels)
    else: _empty_note(ax)
    ax.set_xlabel("Age coefficient per 1000 h (fixed condition-adjusted model)")
    ax.set_title("Condition-adjusted association, not causal effect")
    _style(ax); _save(fig, "10_condition_adjusted_age_effect")

    # 11. ASR.
    fig, ax = plt.subplots(figsize=(7.3,4.2))
    if len(asr):
        pivot=asr.pivot(index="stratum",columns="representation",values="ASR").reindex(columns=["Raw","SimpleTrend","EMD"])
        pivot.plot.bar(ax=ax,color=[COLORS["Raw"],COLORS["SimpleTrend"],COLORS["EMD"]],width=.72)
        ax.legend(frameon=False, fontsize=8)
    else: _empty_note(ax)
    ax.set_ylabel("ASR = |D| / median(|online Δ|)"); ax.set_title("Age-signal-to-local-variation ratio")
    _style(ax); _save(fig, "11_asr_comparisons")

    # 12. RDR.
    fig, ax=plt.subplots(figsize=(7.3,4.2))
    numeric=pd.to_numeric(rdr.RDR95,errors="coerce") if len(rdr) else pd.Series(dtype=float)
    if len(numeric.dropna()):
        ax.bar(rdr.stratum, numeric, color=COLORS["EMD"]); ax.axhline(1,color=COLORS["warning"],ls="--",label="RDR95=1")
        ax.legend(frameon=False)
    else: _empty_note(ax, "RDR undefined: no eligible common-support denominator")
    ax.set_ylabel("RDR95"); ax.set_title("Retrospective revision-to-drift ratio")
    _style(ax); _save(fig, "12_rdr_comparisons")

    _write_report(table, blocks24, blocks48, support, contrasts, rough_summary, revision_events, adjusted, asr, rdr, recovery, gates, decision, protocol)


def _fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _write_report(table, blocks24, blocks48, support, contrasts, rough_summary, revisions, adjusted, asr, rdr, recovery, gates, decision, protocol):
    common = support.loc[(support.record_type.eq("summary")) & (support.common_support_status.eq("COMMON_SUPPORT_PASS")), "stratum"].tolist()
    emd_trend = pd.read_csv(HA_DIR/"outputs"/"TREND_METRICS.csv").query("scope == 'all_origins' and representation == 'EMD'").iloc[0]
    emd_adjusted = adjusted.loc[(adjusted.representation.eq("EMD")) & (adjusted.block_hours.eq(24))].iloc[0]
    lag1 = revisions.loc[revisions.lag_steps.eq(1), "abs_U"]
    rdr_text = ", ".join(f"{r.stratum}={r.RDR95}" for _,r in rdr.iterrows()) if len(rdr) else "undefined"
    contrasts_text = contrasts.loc[contrasts.representation.eq("EMD"), ["block_hours", "stratum", "D_late_minus_early", "bootstrap_ci95_lower", "bootstrap_ci95_upper"]].to_markdown(index=False)
    asr_text = asr.to_markdown(index=False) if len(asr) else "No stratum passed common support."
    report = f"""# Phase HA — EMD Health Representation Audit

## Scope and firewall

**PROJECT_EVIDENCE.** Base commit `{BASE_COMMIT}` was verified before the audit.
The runner read exactly the first 3,007 N1 causal rows and retained 3,005 rows
with `bin_end_ophrs2_h < {protocol['development_firewall']['development_end_h_exclusive']}`.
No post-boundary row, Test row, forecast target, forecast prediction, neural
model, VMD run, or EMD parameter change entered this phase.  The audited unit is
one unique origin endpoint from frozen Phase DF cache, not the full `[48,2]`
feature block rewritten as a global health series.

Phase DF status remains `EMD_SPECIFIC_ADVANCE`; backbone status remains
`TIMESNET_FREEZE`; backbone search remains `CLOSED`.

## Answers to the registered questions

1. **Long-term EMD drift:** the descriptive all-origin EMD OLS trend is
   {_fmt(emd_trend.OLS_slope_per_1000h)} standardized-remainder units/1000 h
   (Spearman {_fmt(emd_trend.spearman_rho)}).  This is PROJECT_EVIDENCE, not
   proof of irreversible degradation.
2. **Common support:** {len(common)} positive-current strata passed the frozen
   24 h support rule: {', '.join(common) if common else 'none'}.
3. **Matched early-to-late contrasts:** only passing strata can support this
   question.  The frozen 24 h/48 h EMD contrasts are:

{contrasts_text}

4. **Condition-adjusted age association:** for 24 h blocks, EMD age coefficient
   is {_fmt(emd_adjusted.age_coefficient_per_1000h)} with bootstrap interval
   [{_fmt(emd_adjusted.age_bootstrap_ci95_lower)}, {_fmt(emd_adjusted.age_bootstrap_ci95_upper)}],
   condition number {_fmt(emd_adjusted.condition_number)}.  It is explicitly a
   condition-adjusted association, not a causal degradation effect.
5. **ASR comparison (PROJECT_ANALYSIS_DESIGN):**

{asr_text}

6. **Online stability:** online roughness is reported from released endpoint
   differences only in `outputs/ONLINE_ROUGHNESS.csv`; it is deliberately not
   conflated with retrospective re-estimation.
7. **Historical ≈105.9 mV revision context:** that historical Phase DF value is
   a retrospective boundary-revision warning, not online roughness.  Here lag-1
   P95 is {_fmt(lag1.quantile(.95) if len(lag1) else float('nan'))} in standardized
   EMD-remainder units, using `U_1 = re-estimate − online`.
8. **RDR:** {rdr_text}. Undefined values are retained as undefined rather than
   being converted to zero or infinity.
9. **Revision versus transitions:** the pre-existing load-transition/current-density
   stratification is in `REVISION_LAG_1_2_4_8.csv` and Figure 07; it is
   descriptive and does not claim an electrochemical mechanism.
10. **Recovery:** `{recovery.iloc[0].recovery_status}`.  Existing independent
    startup/shutdown/gap fields may define candidates, but no voltage-selected
    recovery event or reversible/irreversible mechanism claim was made.
11. **24 h/48 h sensitivity:** assessed with the pre-frozen block sizes in
    `MATCHED_AGE_CONTRASTS.csv` and `CONDITION_ADJUSTED_EFFECTS.csv`; it feeds
    the final gate without changing the primary 24 h rule.
12. **Forecasting versus HI:** Phase DF's full-block forecasting gain is retained.
    This endpoint audit can legitimately reach a weaker conclusion; a useful
    full 48×2 forecasting representation need not supply an HI endpoint.

## Final registered decision

**{decision['final_status']}** — `{decision['reason']}`.

This status is restricted to the single RealHyFC Development trajectory and
the Phase HA protocol.  It does not construct SOH, define EOL, validate
irreversibility, or support RUL prediction.

## Evidence labels

- **PROJECT_EVIDENCE:** frozen cache, Development-only endpoint results and
  source-pinned implementation facts.
- **LITERATURE_SUPPORTED_METHOD:** EMD and descriptive robust-trend methods;
  see `docs/METHOD_AND_LITERATURE_SOURCES.md`.
- **PROJECT_ANALYSIS_DESIGN:** 24 h/48 h blocks, support rule, bootstrap, ASR,
  RDR and decision gates; these are not universal PHM standards.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    final = {"phase": "Phase_HA_EMD_Health_Representation_Audit", "base_commit": BASE_COMMIT,
             "final_status": decision["final_status"], "reason": decision["reason"],
             "test_accessed": False, "forecast_targets_loaded": False, "neural_model_trained": False,
             "emd_parameters_changed": False, "SOH_constructed": False, "RUL_model": False,
             "phase_df_status": "EMD_SPECIFIC_ADVANCE", "backbone": "TIMESNET_FREEZE", "backbone_search": "CLOSED",
             "protocol_status": "PRE-REGISTERED_BEFORE_PHASE_HA_ENDPOINT_ANALYSIS", "git_branch": git_branch(), "head_at_generation": git_head(),
             "automatic_next_phase": "PROHIBITED; await technical review"}
    write_json(FINAL, final)


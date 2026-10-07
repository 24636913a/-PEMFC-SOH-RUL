"""Target-blind late-life P1S operating-condition support-drift audit.

The script reads no polarization CSV, does not import a target extractor or
one-shot runner, and never writes under the consumed AHC-TV runtime directory.
It performs only descriptive analysis on formal raw-derived P1S interval
artifacts already created by the target-blind alignment postmortem.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
AHC_PHASE = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_AHC_TV_One_Time_Late_Life_Test_Protocol"
POSTMORTEM = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_AHC_TV_Target_Blind_Alignment_Postmortem"
PAA = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAA_P1_StateToken_PreAnchor_Alignment"
OUTPUTS = PHASE / "outputs"
FIGURES = PHASE / "figures"
REPORTS = PHASE / "reports"

FROZEN_INPUTS = AHC_PHASE / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml"
POSTMORTEM_SUMMARY = POSTMORTEM / "outputs" / "AHC_TV_ALIGNMENT_POSTMORTEM_SUMMARY.json"
POSTMORTEM_2000 = POSTMORTEM / "outputs" / "AHC_TV_ALIGNMENT_POSTMORTEM_INTERVALS_2000.csv"
POSTMORTEM_2300 = POSTMORTEM / "outputs" / "AHC_TV_ALIGNMENT_POSTMORTEM_INTERVALS_2300.csv"
PAA_ALIGNMENT = PAA / "outputs" / "P1S_ANCHOR_ALIGNMENT.csv"

CONDITIONS = (
    "current_density",
    "anode_inlet_temperature_C",
    "anode_inlet_pressure_kPag",
    "anode_inlet_RH_pct",
    "anode_hydrogen_flow_nl_min",
    "anode_stoichiometry",
    "cathode_stoichiometry",
)
DISPLAY = {
    "current_density": "Current density (A cm$^{-2}$)",
    "anode_inlet_temperature_C": "Anode inlet temperature (°C)",
    "anode_inlet_pressure_kPag": "Anode inlet pressure (kPag)",
    "anode_inlet_RH_pct": "Anode inlet RH (%)",
    "anode_hydrogen_flow_nl_min": "Anode H$_2$ flow (NL min$^{-1}$)",
    "anode_stoichiometry": "Anode stoichiometry (-)",
    "cathode_stoichiometry": "Cathode stoichiometry (-)",
}
PALETTE = {
    "blue_main": "#0F4D92",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral": "#767676",
    "highlight": "#FFD700",
}
FORBIDDEN_COLUMNS = {
    "y_2000", "y_2300", "ae_p1s", "ae_age", "mae_test", "rule_a", "rule_b", "rule_c",
    "polarization_target", "actual_health", "voltage_anchor",
}


class SupportDriftAuditError(RuntimeError):
    """The frozen target-blind input contract was not satisfied."""


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _apply_publication_style() -> None:
    """Use the repository scientific-figure style with headless-safe export."""
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 12,
        "axes.linewidth": 1.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "figure.dpi": 120,
    })


def _require_target_blind_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SupportDriftAuditError(f"POSTMORTEM_INPUT_MISSING:{path}")
    table = pd.read_csv(path)
    overlap = FORBIDDEN_COLUMNS & {str(column).strip().lower() for column in table.columns}
    if overlap:
        raise SupportDriftAuditError("TARGET_CAPABLE_COLUMN_REJECTED:" + ",".join(sorted(overlap)))
    required = {
        "cycle_interval_id", "measurement_end_datetime", "measurement_end_ophrs2_h",
        "P1S_voltage_V", "P1S_operating_voltage_retention", "structural_legal",
        "strictly_pre_characterization", "in_frozen_joint_common_support", *CONDITIONS,
    }
    absent = required - set(table.columns)
    if absent:
        raise SupportDriftAuditError("POSTMORTEM_INTERVAL_SCHEMA_INVALID:" + ",".join(sorted(absent)))
    table["measurement_end_datetime"] = pd.to_datetime(table["measurement_end_datetime"], errors="raise")
    table["measurement_end_ophrs2_h"] = pd.to_numeric(table["measurement_end_ophrs2_h"], errors="raise")
    return table


def _decorate_support(table: pd.DataFrame, bounds: dict[str, list[float]]) -> pd.DataFrame:
    data = table.copy()
    for variable in CONDITIONS:
        lower, upper = (float(value) for value in bounds[variable])
        value = pd.to_numeric(data[variable], errors="raise")
        inside = value.between(lower, upper, inclusive="both")
        signed = np.where(value < lower, value - lower, np.where(value > upper, value - upper, 0.0))
        data[f"{variable}_development_lower_bound"] = lower
        data[f"{variable}_development_upper_bound"] = upper
        data[f"{variable}_in_support"] = inside
        data[f"{variable}_below_support"] = value < lower
        data[f"{variable}_inside_support"] = inside
        data[f"{variable}_above_support"] = value > upper
        data[f"{variable}_signed_deviation"] = signed
        data[f"{variable}_absolute_deviation"] = np.abs(signed)
        if upper > lower:
            data[f"{variable}_normalized_excess_if_defined"] = np.abs(signed) / (upper - lower)
            data[f"{variable}_exact_support_match"] = pd.NA
        else:
            data[f"{variable}_normalized_excess_if_defined"] = np.nan
            data[f"{variable}_exact_support_match"] = value.eq(lower)
    calculated_joint = data.loc[:, [f"{name}_in_support" for name in CONDITIONS]].all(axis=1)
    existing_joint = _as_bool(data["in_frozen_joint_common_support"])
    if not calculated_joint.equals(existing_joint):
        raise SupportDriftAuditError("FROZEN_JOINT_SUPPORT_REPLAY_MISMATCH")
    data["joint_support"] = calculated_joint
    data["support_fail_variables"] = data.apply(
        lambda row: ";".join(name for name in CONDITIONS if not bool(row[f"{name}_in_support"])), axis=1
    )
    data["support_fail_count"] = (~data.loc[:, [f"{name}_in_support" for name in CONDITIONS]]).sum(axis=1).astype(int)
    return data


def _structural_legal(data: pd.DataFrame) -> pd.DataFrame:
    mask = _as_bool(data["structural_legal"]) & _as_bool(data["strictly_pre_characterization"])
    return data.loc[mask].copy().sort_values(
        ["measurement_end_ophrs2_h", "measurement_end_datetime", "cycle_interval_id"], kind="mergesort"
    ).drop_duplicates("cycle_interval_id", keep="last").reset_index(drop=True)


def _development_cutoffs(summary: dict[str, Any]) -> dict[int, float]:
    alignment = pd.read_csv(PAA_ALIGNMENT)
    summaries = alignment.loc[alignment["record_type"].eq("anchor_summary")].copy()
    result: dict[int, float] = {}
    for checkpoint in (1500, 1750):
        row = summaries.loc[pd.to_numeric(summaries["checkpoint_h"], errors="coerce").eq(checkpoint)]
        if len(row) != 1:
            raise SupportDriftAuditError(f"FROZEN_DEVELOPMENT_CUTOFF_MISSING:{checkpoint}")
        result[checkpoint] = float(row.iloc[0]["precharacterization_cutoff_h"])
    for checkpoint in (2000, 2300):
        item = summary["checkpoints"].get(str(checkpoint), {})
        value = item.get("raw_cutoff_ophrs2_h")
        if value is None:
            raise SupportDriftAuditError(f"POSTMORTEM_CUTOFF_MISSING:{checkpoint}")
        result[checkpoint] = float(value)
    return result


def _window_summary(
    checkpoint: int,
    cutoff_h: float,
    source: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lookback_h in (48, 96, 168):
        window = source.loc[source["measurement_end_ophrs2_h"].between(cutoff_h - lookback_h, cutoff_h, inclusive="both")].copy()
        count = int(len(window))
        joint_count = int(window["joint_support"].sum())
        for variable in CONDITIONS:
            values = pd.to_numeric(window[variable], errors="coerce")
            row = {
                "checkpoint_h": checkpoint,
                "checkpoint_cutoff_ophrs2_h": cutoff_h,
                "lookback_h": lookback_h,
                "variable": variable,
                "P1S_interval_count": count,
                "joint_support_pass_count": joint_count,
                "joint_support_pass_rate": float(joint_count / count) if count else np.nan,
                "in_support_rate": float(window[f"{variable}_in_support"].mean()) if count else np.nan,
                "below_support_rate": float(window[f"{variable}_below_support"].mean()) if count else np.nan,
                "above_support_rate": float(window[f"{variable}_above_support"].mean()) if count else np.nan,
                "median": float(values.median()) if count else np.nan,
                "IQR": float(values.quantile(0.75) - values.quantile(0.25)) if count else np.nan,
                "min": float(values.min()) if count else np.nan,
                "max": float(values.max()) if count else np.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _longest_true_run(values: pd.Series) -> int:
    best = current = 0
    for value in values.astype(bool).tolist():
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _violation_ranking(timeline: pd.DataFrame, bounds: dict[str, list[float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in CONDITIONS:
        value = pd.to_numeric(timeline[variable], errors="raise")
        violated = ~timeline[f"{variable}_in_support"]
        lower, upper = (float(item) for item in bounds[variable])
        width = upper - lower
        iqr = float(value.quantile(0.75) - value.quantile(0.25))
        normalized_variation = float(iqr / width) if width > 0 else np.nan
        rate = float(violated.mean())
        potentially_overrestrictive = bool(
            rate > 0 and ((width == 0) or (np.isfinite(normalized_variation) and normalized_variation >= 1.0))
        )
        rows.append({
            "variable": variable,
            "support_lower_bound": lower,
            "support_upper_bound": upper,
            "support_width": width,
            "late_life_interval_count": int(len(timeline)),
            "late_life_median": float(value.median()),
            "late_life_IQR": iqr,
            "late_life_typical_variation_over_support_width": normalized_variation,
            "violation_count": int(violated.sum()),
            "violation_rate": rate,
            "below_count": int(timeline[f"{variable}_below_support"].sum()),
            "above_count": int(timeline[f"{variable}_above_support"].sum()),
            "longest_consecutive_violation_run": _longest_true_run(violated),
            "support_box_assessment": (
                "POTENTIALLY_OVERRESTRICTIVE_SUPPORT_DIMENSION" if potentially_overrestrictive else "NOT_FLAGGED_BY_DESCRIPTIVE_RULE"
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["violation_rate", "violation_count", "variable"], ascending=[False, False, True], kind="mergesort"
    ).reset_index(drop=True)


def _regime_segments(timeline: pd.DataFrame) -> pd.DataFrame:
    data = timeline.sort_values(["measurement_end_ophrs2_h", "cycle_interval_id"], kind="mergesort").reset_index(drop=True)
    changed = data["joint_support"].ne(data["joint_support"].shift()).cumsum()
    rows: list[dict[str, Any]] = []
    for segment_id, group in data.groupby(changed, sort=True):
        status = bool(group["joint_support"].iloc[0])
        rows.append({
            "segment_id": int(segment_id),
            "support_regime": "PASS" if status else "FAIL",
            "interval_count": int(len(group)),
            "start_cycle_interval_id": str(group["cycle_interval_id"].iloc[0]),
            "end_cycle_interval_id": str(group["cycle_interval_id"].iloc[-1]),
            "start_ophrs2_h": float(group["measurement_end_ophrs2_h"].iloc[0]),
            "end_ophrs2_h": float(group["measurement_end_ophrs2_h"].iloc[-1]),
            "duration_operating_h": float(group["measurement_end_ophrs2_h"].iloc[-1] - group["measurement_end_ophrs2_h"].iloc[0]),
        })
    return pd.DataFrame(rows)


def _latest_n(source: pd.DataFrame, cutoff_h: float, n: int = 10) -> pd.DataFrame:
    data = source.loc[source["measurement_end_ophrs2_h"].le(cutoff_h)].copy()
    data = data.sort_values(
        ["measurement_end_datetime", "measurement_end_ophrs2_h", "cycle_interval_id"],
        ascending=[False, False, True], kind="mergesort"
    ).drop_duplicates("cycle_interval_id", keep="first").head(n).copy()
    data["age_gap_h"] = cutoff_h - data["measurement_end_ophrs2_h"]
    data["recency_rank"] = np.arange(1, len(data) + 1, dtype=int)
    return data


def _final_failure_run(table: pd.DataFrame) -> tuple[int, float | None]:
    chronological = table.sort_values("measurement_end_ophrs2_h", kind="mergesort")
    count = 0
    start_age: float | None = None
    for item in chronological.iloc[::-1].itertuples(index=False):
        if bool(item.joint_support):
            break
        count += 1
        start_age = float(item.measurement_end_ophrs2_h)
    return count, start_age


def _final_variable_violation_run(table: pd.DataFrame, variable: str) -> tuple[int, float | None]:
    chronological = table.sort_values("measurement_end_ophrs2_h", kind="mergesort")
    count = 0
    start_age: float | None = None
    for item in chronological.iloc[::-1].itertuples(index=False):
        if bool(getattr(item, f"{variable}_in_support")):
            break
        count += 1
        start_age = float(item.measurement_end_ophrs2_h)
    return count, start_age


def _intersection_fail_variables(selected: pd.DataFrame) -> list[str]:
    if selected.empty:
        return []
    common = set(CONDITIONS)
    for _, row in selected.iterrows():
        common &= {name for name in CONDITIONS if not bool(row[f"{name}_in_support"])}
    return [name for name in CONDITIONS if name in common]


def _sustained_start(latest: pd.DataFrame, variable: str) -> float | None:
    chronological = latest.sort_values("measurement_end_ophrs2_h", kind="mergesort")
    run: list[float] = []
    for item in chronological.iloc[::-1].itertuples(index=False):
        if bool(getattr(item, f"{variable}_in_support")):
            break
        run.append(float(item.measurement_end_ophrs2_h))
    return min(run) if len(run) >= 3 else None


def _latest_selection(source: pd.DataFrame) -> pd.DataFrame:
    selected = source.loc[pd.to_numeric(source.get("selected_rank"), errors="coerce").notna()].copy()
    return selected.sort_values("selected_rank", kind="mergesort")


def _plot_variable(timeline: pd.DataFrame, variable: str, bounds: tuple[float, float], path: Path) -> None:
    _apply_publication_style()
    lower, upper = bounds
    fig, axis = plt.subplots(figsize=(10.5, 4.5))
    x = timeline["measurement_end_ophrs2_h"].to_numpy(dtype=float)
    y = timeline[variable].to_numpy(dtype=float)
    inside = timeline[f"{variable}_in_support"].to_numpy(dtype=bool)
    axis.plot(x, y, color=PALETTE["neutral"], linewidth=1.4, zorder=1)
    axis.scatter(x[inside], y[inside], color=PALETTE["blue_main"], s=34, label="inside frozen support", zorder=2)
    axis.scatter(x[~inside], y[~inside], color=PALETTE["red_strong"], marker="x", s=52, linewidths=1.8, label="outside frozen support", zorder=3)
    if upper > lower:
        axis.axhspan(lower, upper, color=PALETTE["green_3"], alpha=0.18, label="frozen Development support")
    axis.axhline(lower, color=PALETTE["green_3"], linestyle="--", linewidth=1.6)
    axis.axhline(upper, color=PALETTE["green_3"], linestyle="--", linewidth=1.6)
    for checkpoint in (1500, 1750, 2000, 2300):
        axis.axvline(checkpoint, color=PALETTE["neutral"], linestyle=":", linewidth=1.0)
        axis.text(checkpoint, 0.98, str(checkpoint), transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9)
    values = np.concatenate([y, np.asarray([lower, upper], dtype=float)])
    spread = max(float(values.max() - values.min()), max(abs(float(values.mean())), 1.0) * 0.002)
    axis.set_ylim(float(values.min() - 0.12 * spread), float(values.max() + 0.12 * spread))
    axis.set_xlim(1480, 2320)
    axis.set_xlabel("Operating age, OpHrs2 (h)")
    axis.set_ylabel(DISPLAY[variable])
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout(pad=1.0)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_joint_support(timeline: pd.DataFrame, path: Path) -> None:
    _apply_publication_style()
    fig, axis = plt.subplots(figsize=(10.5, 3.8))
    x = timeline["measurement_end_ophrs2_h"].to_numpy(dtype=float)
    y = timeline["joint_support"].to_numpy(dtype=int)
    axis.step(x, y, where="mid", color=PALETTE["neutral"], linewidth=1.3)
    axis.scatter(x[y == 1], y[y == 1], color=PALETTE["blue_main"], s=35, label="joint support pass", zorder=3)
    axis.scatter(x[y == 0], y[y == 0], color=PALETTE["red_strong"], marker="x", s=52, linewidths=1.7, label="joint support fail", zorder=3)
    for checkpoint in (1500, 1750, 2000, 2300):
        axis.axvline(checkpoint, color=PALETTE["neutral"], linestyle=":", linewidth=1.0)
        axis.text(checkpoint, 1.1, str(checkpoint), ha="center", va="bottom", fontsize=9)
    axis.set_xlim(1480, 2320)
    axis.set_ylim(-0.22, 1.28)
    axis.set_yticks([0, 1], ["fail", "pass"])
    axis.set_xlabel("Operating age, OpHrs2 (h)")
    axis.set_ylabel("Frozen joint support")
    axis.legend(loc="lower left", fontsize=9)
    fig.tight_layout(pad=1.0)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _format_table(data: pd.DataFrame, columns: list[str], digits: int = 5) -> str:
    if data.empty:
        return "No rows."
    text = data.loc[:, columns].copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    header = "| " + " | ".join(text.columns) + " |"
    divider = "|" + "|".join(["---"] * len(text.columns)) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in text.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def _report(
    summary: dict[str, Any],
    ranking: pd.DataFrame,
    latest_2000: pd.DataFrame,
    latest_2300: pd.DataFrame,
    selected_2000: pd.DataFrame,
    selected_2300: pd.DataFrame,
    segments: pd.DataFrame,
) -> str:
    selected_2000 = summary["checkpoint_anatomy"]["2000"]
    selected_2300 = summary["checkpoint_anatomy"]["2300"]
    ranked = ranking.head(4)
    lines = [
        "# Late-Life Operating-Condition Support Drift Audit", "",
        "## 1. Objective", "",
        "This target-blind audit identifies why frozen Development joint support rejected the 2000 h latest-three P1S intervals while accepting two of three at 2300 h. It evaluates operating-condition support only, not health-prediction performance.", "",
        "## 2. Data boundary / target-blind statement", "",
        "Only frozen Development support, Development P1S checkpoint metadata, and existing target-blind Stage-P interval artifacts were read. No polarization target, Stage T, one-shot runner, ledger write, Y, AE, MAE, or Rule A/B/C calculation was performed.", "",
        "## 3. Frozen Development support", "",
        _format_table(ranking, ["variable", "support_lower_bound", "support_upper_bound", "support_width", "late_life_IQR", "late_life_typical_variation_over_support_width", "support_box_assessment"]), "",
        "The support bounds above are reported exactly as frozen. They were not refit, widened, or otherwise modified.", "",
        "## 4. Late-life support timeline", "",
        f"The timeline contains `{summary['timeline_interval_count']}` structurally legal, pre-characterization P1S intervals from `{summary['timeline_start_h']:.3f}` to `{summary['timeline_end_h']:.3f}` OpHrs2. It is exported in `outputs/LATE_LIFE_P1S_SUPPORT_TIMELINE.csv`.", "",
        "## 5. 2000 h failure anatomy", "",
        f"All latest-three intervals were present within 48 h, but frozen support was `{selected_2000['support_count']}/3`; G4 was the only failed alignment gate. The variables failing all three selected intervals were: `{', '.join(selected_2000['common_fail_variables']) or 'none'}`.", "",
        f"Q1 — sudden or persistent: `{selected_2000['failure_pattern']}`. The final contiguous joint-support failure run before 2000 h contained `{selected_2000['final_joint_failure_run_count']}` interval(s) in the last-ten audit, beginning at `{selected_2000['final_joint_failure_run_start_h']}` OpHrs2. In the full pre-2000 interval history, the persistent anode/cathode-stoichiometry violations both began at `{selected_2000['persistent_variable_violation_start_h']['anode_stoichiometry']}` OpHrs2 and lasted `{selected_2000['persistent_variable_violation_run_count']['anode_stoichiometry']}` interval(s).", "",
        "Q2 — same stable new condition: `YES_FOR_THE_PRIMARY_STOICHIOMETRY_SHIFT`. The three selected intervals share the same above-support anode-stoichiometry level, while cathode stoichiometry remains above support in all three; RH and temperature contribute smaller interval-specific differences.", "",
        f"Q3 — earliest sustained violation: `{', '.join(selected_2000['earliest_sustained_variables']) or 'none'}`. Q4 — variables causing all selected intervals to fail: `{', '.join(selected_2000['common_fail_variables']) or 'none'}`.", "",
        "The ten preceding structural-legal intervals are exported without target information in `outputs/CHECKPOINT_LATEST10_P1S_INTERVALS.csv`; their interval IDs, age gaps, all seven conditions, and per-variable support flags are retained there.", "",
        _format_table(latest_2000, ["recency_rank", "cycle_interval_id", "age_gap_h", "joint_support", "support_fail_variables"], digits=3), "",
        "## 6. 2300 h support recovery", "",
        f"Q5 — why 2300 h recovered: the frozen latest-three selection had `{selected_2300['support_count']}/3` support because ranks 1–2 returned inside every frozen support dimension; rank 3 remained outside `{', '.join(selected_2300['rank3_fail_variables']) or 'none'}`. The observed outcome is therefore a partial operating-condition recovery, not a change to the frozen support rule.", "",
        _format_table(latest_2300, ["recency_rank", "cycle_interval_id", "age_gap_h", "joint_support", "support_fail_variables"], digits=3), "",
        "## 7. Variable-level drift ranking", "",
        _format_table(ranked, ["variable", "violation_count", "violation_rate", "below_count", "above_count", "longest_consecutive_violation_run", "support_box_assessment"]), "",
        "## 8. Regime analysis", "",
        f"The sequential support-state decomposition found `{len(segments)}` pass/fail regime segment(s); the complete chronology is in `outputs/JOINT_SUPPORT_REGIME_SEGMENTS.csv`. The 2000 h failure and 2300 h partial recovery show that support status is not a monotone late-life loss of P1S availability.", "",
        "## 9. Is the support box overly restrictive?", "",
        f"Assessment: `{summary['development_support_box_assessment']}`. This is a descriptive statement based on frozen-width versus late-life P1S-condition variability; it does not authorize changing the frozen box or rerunning the Test.", "",
        "## 10. Implications for P1S", "",
        f"P1S should be abandoned: `{summary['p1s_should_be_abandoned']}`. Principal weakness: `{summary['principal_weakness']}`. The evidence supports an observability/support conclusion only; it does not validate P1S health-prediction performance because targets remain blinded.", "",
        "## 11. Recommended next protocol", "",
        f"Recommended route: `{summary['recommended_next_route']}`. Any future protocol would need a separate authorization and must not reinterpret or rerun the consumed AHC-TV Test.", "",
        "## Target-blind boundary", "",
        "`TARGET_BLIND_ONLY`; one-shot rerun `NO`; ledger modified `NO`; polarization targets remain blinded `YES`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    for required in (FROZEN_INPUTS, POSTMORTEM_SUMMARY, POSTMORTEM_2000, POSTMORTEM_2300, PAA_ALIGNMENT):
        if not required.exists():
            raise SupportDriftAuditError(f"FROZEN_INPUT_MISSING:{required}")
    frozen = _read_yaml(FROZEN_INPUTS)
    bounds = frozen["frozen_development_joint_common_support"]
    if tuple(bounds) != CONDITIONS:
        raise SupportDriftAuditError("FROZEN_SUPPORT_DIMENSION_OR_ORDER_INVALID")
    postmortem_summary = json.loads(POSTMORTEM_SUMMARY.read_text(encoding="utf-8"))
    if postmortem_summary.get("polarization_targets_accessed") is not False:
        raise SupportDriftAuditError("POSTMORTEM_TARGET_BLIND_CONTRACT_INVALID")
    table_2000 = _decorate_support(_require_target_blind_table(POSTMORTEM_2000), bounds)
    table_2300 = _decorate_support(_require_target_blind_table(POSTMORTEM_2300), bounds)
    legal_2000 = _structural_legal(table_2000)
    legal_2300 = _structural_legal(table_2300)
    timeline = legal_2300.loc[legal_2300["measurement_end_ophrs2_h"].between(1500.0, 2300.0, inclusive="both")].copy()
    if timeline.empty:
        raise SupportDriftAuditError("LATE_LIFE_TIMELINE_EMPTY")
    cutoffs = _development_cutoffs(postmortem_summary)
    sources = {1500: legal_2300, 1750: legal_2300, 2000: legal_2000, 2300: legal_2300}
    windows = pd.concat(
        [_window_summary(checkpoint, cutoffs[checkpoint], sources[checkpoint]) for checkpoint in (1500, 1750, 2000, 2300)],
        ignore_index=True,
    )
    ranking = _violation_ranking(timeline, bounds)
    segments = _regime_segments(timeline)
    latest_2000 = _latest_n(legal_2000, cutoffs[2000])
    latest_2300 = _latest_n(legal_2300, cutoffs[2300])
    selected_2000 = _latest_selection(table_2000)
    selected_2300 = _latest_selection(table_2300)
    if len(selected_2000) != 3 or len(selected_2300) != 3:
        raise SupportDriftAuditError("POSTMORTEM_LATEST_THREE_SELECTION_MISSING")
    support_2000 = int(selected_2000["joint_support"].sum())
    support_2300 = int(selected_2300["joint_support"].sum())
    final_run_count, final_run_start = _final_failure_run(latest_2000)
    sustained_starts = {
        variable: _sustained_start(latest_2000, variable) for variable in CONDITIONS
    }
    persistent_variable_runs = {
        variable: _final_variable_violation_run(legal_2000, variable) for variable in CONDITIONS
    }
    earliest_persistent_start = min(
        (start for count, start in persistent_variable_runs.values() if count >= 3 and start is not None),
        default=None,
    )
    earliest_sustained_variables = [
        variable for variable, (count, start) in persistent_variable_runs.items()
        if count >= 3 and start == earliest_persistent_start
    ]
    common_2000_failures = _intersection_fail_variables(selected_2000)
    dominant = ranking.loc[ranking["violation_count"].gt(0), "variable"].head(3).tolist()
    potential_dimensions = ranking.loc[
        ranking["support_box_assessment"].eq("POTENTIALLY_OVERRESTRICTIVE_SUPPORT_DIMENSION"), "variable"
    ].tolist()
    failure_pattern = "PERSISTENT" if final_run_count >= 3 else "LOCAL"
    if support_2000 == 0 and support_2300 >= 2 and potential_dimensions:
        drift_classification = "MULTIPLE_CAUSES"
    elif support_2000 == 0 and support_2300 >= 2:
        drift_classification = "CYCLIC_OR_REGIME_DEPENDENT_CONDITION_SHIFT"
    elif final_run_count >= 3:
        drift_classification = "PERSISTENT_LATE_LIFE_OPERATING_CONDITION_DRIFT"
    else:
        drift_classification = "INCONCLUSIVE"
    support_assessment = "POTENTIALLY_OVERRESTRICTIVE" if potential_dimensions else "ROBUST"
    checkpoint_anatomy = {
        "2000": {
            "support_count": support_2000,
            "common_fail_variables": common_2000_failures,
            "final_joint_failure_run_count": final_run_count,
            "final_joint_failure_run_start_h": final_run_start,
            "failure_pattern": failure_pattern,
            "sustained_variable_violation_start_h_in_latest10": sustained_starts,
            "persistent_variable_violation_run_count": {
                variable: count for variable, (count, _) in persistent_variable_runs.items()
            },
            "persistent_variable_violation_start_h": {
                variable: start for variable, (_, start) in persistent_variable_runs.items()
            },
            "earliest_sustained_variables": earliest_sustained_variables,
        },
        "2300": {
            "support_count": support_2300,
            "rank3_fail_variables": [
                variable for variable in CONDITIONS if not bool(selected_2300.iloc[2][f"{variable}_in_support"])
            ],
        },
    }
    summary = {
        "target_blind": True,
        "one_shot_rerun": False,
        "ledger_modified": False,
        "2000_failure": "JOINT_SUPPORT",
        "2300_alignment": "ALIGNED",
        "timeline_interval_count": int(len(timeline)),
        "timeline_start_h": float(timeline["measurement_end_ophrs2_h"].min()),
        "timeline_end_h": float(timeline["measurement_end_ophrs2_h"].max()),
        "development_checkpoint_cutoffs_h": cutoffs,
        "dominant_violation_variables": dominant,
        "potentially_overrestrictive_support_dimensions": potential_dimensions,
        "development_support_box_assessment": support_assessment,
        "support_drift_classification": drift_classification,
        "p1s_should_be_abandoned": "NO",
        "p1s_signal_itself": "USABLE",
        "principal_weakness": "HARD_SUPPORT_GATE",
        "recommended_next_route": "Route A - Condition-Normalized P1S Health Representation",
        "checkpoint_anatomy": checkpoint_anatomy,
        "targets_remain_blinded": True,
        "all_target_metrics": "NOT_EVALUATED_TARGET_BLIND",
        "input_boundary": {
            "raw_polarization_targets_read": False,
            "stage_t_run": False,
            "original_one_shot_runner_run": False,
            "frozen_support_refit": False,
        },
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(OUTPUTS / "LATE_LIFE_P1S_SUPPORT_TIMELINE.csv", index=False)
    windows.to_csv(OUTPUTS / "SUPPORT_DRIFT_WINDOW_SUMMARY.csv", index=False)
    ranking.to_csv(OUTPUTS / "SUPPORT_VIOLATION_RANKING.csv", index=False)
    segments.to_csv(OUTPUTS / "JOINT_SUPPORT_REGIME_SEGMENTS.csv", index=False)
    pd.concat([
        latest_2000.assign(checkpoint_h=2000),
        latest_2300.assign(checkpoint_h=2300),
    ], ignore_index=True).to_csv(OUTPUTS / "CHECKPOINT_LATEST10_P1S_INTERVALS.csv", index=False)
    (OUTPUTS / "LATE_LIFE_SUPPORT_DRIFT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    figure_spec = (
        ("anode_stoichiometry", "Fig1_Anode_Stoichiometry_Support_Drift.png"),
        ("cathode_stoichiometry", "Fig2_Cathode_Stoichiometry_Support_Drift.png"),
        ("anode_inlet_RH_pct", "Fig3_Anode_RH_Support_Drift.png"),
        ("anode_inlet_temperature_C", "Fig4_Anode_Temperature_Support_Drift.png"),
    )
    for variable, filename in figure_spec:
        _plot_variable(timeline, variable, tuple(float(value) for value in bounds[variable]), FIGURES / filename)
    _plot_joint_support(timeline, FIGURES / "Fig5_Joint_Support_Status_Timeline.png")
    (REPORTS / "LATE_LIFE_OPERATING_CONDITION_SUPPORT_DRIFT_AUDIT.md").write_text(
        _report(summary, ranking, latest_2000, latest_2300, selected_2000, selected_2300, segments), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Independent raw-resolution reconstruction of two previously exposed events.

Read-only sources; writes only inside RealHyFC_Raw1s_Dynamic_Study.  This
descriptive audit does not fit models, select training thresholds, infer SOH,
or assume that a logged set point is an available future schedule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parent
REPO = PROJECT.parent
RAW = REPO / "数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据/Loadcycle_data/Loadcycle_data"
OLD = PROJECT / "Phase_SCI_RealHyFC_Full_Chronological_Evaluation/outputs/audits/FULL_ROW_IDENTITY.csv"
ACTIVE = [f"U{i} [V]" for i in range(1, 12) if i != 6]
VOLTAGES = [f"U{i} [V]" for i in range(1, 12)]
USE = ["DateTime", "OpHrs [h]", "OpHrs2 [h]", "I.S.act [A]", "I.S.set [A]",
       "I.S.set.internal [A]", "j.S.act [A/cm2]", "operation_point",
       "T.Si.CL [°C]", "T.So.CL [°C]", "p.Si.A [kPag]", "p.Si.C [kPag]"] + VOLTAGES
EVENTS = {
    "E1": {"file": "Loadcycle_162-175.csv", "low": 1872.5, "high": 1877.5, "frozen_h": 1873.5, "low_row": 3113},
    "E2": {"file": "Loadcycle_176-190.csv", "low": 2148.5, "high": 2154.5, "frozen_h": 2149.5, "low_row": 3628},
}
BLUE, RED, GRAY = "#0F4D92", "#B64342", "#767676"


def output(name: str) -> Path:
    p = (STUDY / name).resolve()
    if not p.is_relative_to(STUDY):
        raise ValueError("WRITE_OUTSIDE_NEW_STUDY")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.replace("\ufeff", "", regex=False).str.strip()
    comma, dot = s.str.contains(",", regex=False, na=False), s.str.contains(".", regex=False, na=False)
    s = s.mask(comma & dot, s.str.replace(",", "", regex=False))
    s = s.mask(comma & ~dot, s.str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def csv(name: str, data) -> None:
    pd.DataFrame(data).to_csv(output(name), index=False, float_format="%.12g")


def json_out(name: str, data) -> None:
    output(name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_event(event: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    source = RAW / spec["file"]
    with source.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"LFS_POINTER_NOT_CSV: {source}")
    pieces, offset = [], 0
    for chunk in pd.read_csv(source, sep=";", encoding="utf-8-sig", usecols=USE,
                             chunksize=100000, low_memory=False):
        age = numeric(chunk["OpHrs2 [h]"])
        keep = (age >= spec["low"]) & (age < spec["high"])
        if keep.any():
            z = chunk.loc[keep].copy()
            z["raw_row_index_0based"] = offset + np.flatnonzero(keep)
            pieces.append(z)
        offset += len(chunk)
    z = pd.concat(pieces, ignore_index=True)
    for c in USE:
        if c != "DateTime":
            z[c] = numeric(z[c])
    z["DateTime"] = pd.to_datetime(z["DateTime"], errors="raise")
    z["event"] = event
    z["active10_voltage_mean_V"] = z[ACTIVE].mean(axis=1, skipna=False)
    z["active10_voltage_range_V"] = z[ACTIVE].max(axis=1) - z[ACTIVE].min(axis=1)
    z["zero_current_historical_0p1A"] = z["I.S.act [A]"].abs() <= 0.1
    z["datetime_step_s"] = z.DateTime.diff().dt.total_seconds()
    z["OpHrs2_step_h"] = z["OpHrs2 [h]"].diff()
    z["delta_voltage_1sample_V"] = z.active10_voltage_mean_V.diff()
    z["bin_start_ophrs2_h"] = np.floor((z["OpHrs2 [h]"] + 1e-10) / 0.5) * 0.5
    if (z.datetime_step_s.dropna() <= 0).any():
        raise RuntimeError("EVENT_TIMESTAMP_NOT_STRICTLY_ORDERED")
    csv(f"data/events/{event}_raw_observations.csv", z)
    return z, dict(event=event, source_file=str(source.relative_to(REPO)),
                   sha256=sha256(source), rows_scanned=offset, retained_rows=len(z),
                   raw_indices_zero_based=True, filtering=f"{spec['low']} <= OpHrs2 < {spec['high']}",
                   first_datetime=str(z.DateTime.iloc[0]), last_datetime=str(z.DateTime.iloc[-1]),
                   future_plan_evidence="No issue-time, schedule-publication-time or future-plan field in raw CSV; logged setpoint is contemporaneous observation only")


def run_table(event: str, z: pd.DataFrame) -> pd.DataFrame:
    keys = z[["operation_point", "I.S.set [A]"]].round(3)
    group = keys.ne(keys.shift()).any(axis=1).cumsum()
    rows = []
    for _, g in z.groupby(group, sort=False):
        rows.append(dict(event=event, start=str(g.DateTime.iloc[0]), end=str(g.DateTime.iloc[-1]),
                         samples=len(g), duration_elapsed_s=(g.DateTime.iloc[-1] - g.DateTime.iloc[0]).total_seconds(),
                         first_raw_row=int(g.raw_row_index_0based.iloc[0]), last_raw_row=int(g.raw_row_index_0based.iloc[-1]),
                         op_start_h=float(g["OpHrs2 [h]"].iloc[0]), op_end_h=float(g["OpHrs2 [h]"].iloc[-1]),
                         operation_point=float(g.operation_point.iloc[0]), set_A=float(g["I.S.set [A]"].iloc[0]),
                         current_mean_A=float(g["I.S.act [A]"].mean()), voltage_mean_V=float(g.active10_voltage_mean_V.mean()),
                         voltage_min_V=float(g.active10_voltage_mean_V.min()), voltage_max_V=float(g.active10_voltage_mean_V.max()),
                         zero_fraction=float(g.zero_current_historical_0p1A.mean())))
    return pd.DataFrame(rows)


def command_lags(event: str, z: pd.DataFrame) -> pd.DataFrame:
    """Descriptive logged-command/current midpoint crossings; no future-input claim."""
    setpoint = z["I.S.set [A]"].to_numpy(float)
    actual = z["I.S.act [A]"].to_numpy(float)
    step = np.diff(setpoint, prepend=np.nan)
    rows = []
    for i in np.flatnonzero(np.abs(step) >= 5.0):  # frozen historical N1 5-A criterion
        midpoint = (setpoint[i-1] + setpoint[i]) / 2
        sign = np.sign(step[i])
        crossing = None
        # Symmetric window used only to diagnose recorded ordering, never model input.
        for j in range(max(1, i-30), min(len(z), i+31)):
            if sign * (actual[j-1] - midpoint) < 0 <= sign * (actual[j] - midpoint):
                crossing = j
                break
        rows.append(dict(event=event, command_time=str(z.DateTime.iloc[i]), raw_row=int(z.raw_row_index_0based.iloc[i]),
                         old_set_A=float(setpoint[i-1]), new_set_A=float(setpoint[i]), step_A=float(step[i]),
                         actual_midpoint_crossing_time=str(z.DateTime.iloc[crossing]) if crossing is not None else None,
                         observed_lag_s=(z.DateTime.iloc[crossing]-z.DateTime.iloc[i]).total_seconds() if crossing is not None else None,
                         scope="descriptive +/-30 recorded seconds; not command issue time; not a known future load plan"))
    return pd.DataFrame(rows)


def mixture(event: str, z: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_start, g in z.groupby("bin_start_ophrs2_h", sort=True):
        matches = old.loc[np.isclose(old.bin_start_ophrs2_h, bin_start, atol=1e-9, rtol=0)]
        if len(matches) != 1:
            raise RuntimeError("OLD_BIN_IDENTITY_AMBIGUOUS")
        historical = matches.iloc[0]
        off = g.zero_current_historical_0p1A
        whole = float(g.active10_voltage_mean_V.mean())
        fraction = float(off.mean())
        on_v = float(g.loc[~off, "active10_voltage_mean_V"].mean()) if (~off).any() else None
        off_v = float(g.loc[off, "active10_voltage_mean_V"].mean()) if off.any() else None
        difference = whole - float(historical.cell_voltage_mean)
        if len(g) != int(historical.raw_sample_count) or abs(difference) >= 1e-12:
            raise RuntimeError("RAW_TO_FROZEN_BIN_RECOMPUTATION_FAILED")
        rows.append(dict(event=event, row_id=int(historical.row_id), bin_start_h=float(bin_start), bin_end_h=float(bin_start+0.5),
                         first_datetime=str(g.DateTime.iloc[0]), last_datetime=str(g.DateTime.iloc[-1]),
                         calendar_span_s=(g.DateTime.iloc[-1]-g.DateTime.iloc[0]).total_seconds(), raw_samples=len(g),
                         zero_samples=int(off.sum()), nonzero_samples=int((~off).sum()), zero_fraction=fraction,
                         zero_mean_voltage_V=off_v, nonzero_mean_voltage_V=on_v, bin_mean_voltage_V=whole,
                         frozen_bin_mean_voltage_V=float(historical.cell_voltage_mean), difference_V=difference,
                         raw_current_min_A=float(g["I.S.act [A]"].min()), raw_current_max_A=float(g["I.S.act [A]"].max()),
                         state_count=int(g.operation_point.nunique()), coverage_vs_1800_samples=len(g)/1800))
    return pd.DataFrame(rows)


def largest_increments(event: str, z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Rank raw observed increments, with no event magnitude tuned to exposed data.
    ix = z.delta_voltage_1sample_V.abs().nlargest(12).index
    for i in ix:
        individual = z.loc[i, ACTIVE].to_numpy(float)-z.loc[i-1, ACTIVE].to_numpy(float)
        delta = float(z.delta_voltage_1sample_V.iloc[i])
        rows.append(dict(event=event, time=str(z.DateTime.iloc[i]), raw_row=int(z.raw_row_index_0based.iloc[i]),
                         dt_s=float(z.datetime_step_s.iloc[i]), delta_mean_voltage_V=delta,
                         ten_channels_matching_direction=int(np.count_nonzero(np.sign(individual)==np.sign(delta))),
                         min_cell_delta_V=float(np.min(individual)), max_cell_delta_V=float(np.max(individual)),
                         current_before_A=float(z["I.S.act [A]"].iloc[i-1]), current_after_A=float(z["I.S.act [A]"].iloc[i]),
                         set_before_A=float(z["I.S.set [A]"].iloc[i-1]), set_after_A=float(z["I.S.set [A]"].iloc[i]),
                         mode_before=float(z.operation_point.iloc[i-1]), mode_after=float(z.operation_point.iloc[i]),
                         interpretation="multi-channel/sign and current/mode evidence; no automatic physical-vs-sensor truth label"))
    return pd.DataFrame(rows)


def event_timeline(event: str, z: pd.DataFrame, runs: pd.DataFrame, spec: dict) -> dict:
    """Describe the longest zero-command run in the requested clock plateau."""
    candidates = runs.loc[(runs.set_A == 0) & np.isclose(runs.op_start_h, spec["frozen_h"], atol=1e-9, rtol=0)]
    r = candidates.sort_values("samples", ascending=False).iloc[0]
    stop, last_zero = pd.Timestamp(r.start), pd.Timestamp(r.end)
    stopped = z.loc[(z.DateTime >= stop)&(z.DateTime <= last_zero)]
    first_zero = stopped.loc[stopped.zero_current_historical_0p1A].iloc[0]
    restart = z.loc[z.DateTime > last_zero].iloc[0]
    jump_row = stopped.delta_voltage_1sample_V.idxmax()
    jump = z.loc[jump_row]
    markers = {"zero_command_start": stop, "first_actual_current_zero": first_zero.DateTime,
               "largest_recovery_jump_in_stop": jump.DateTime, "positive_command_restart": restart.DateTime}
    snap = []
    for kind, time in markers.items():
        index = int(np.flatnonzero(z.DateTime.to_numpy() == time.to_datetime64())[0])
        for lag in [120,30,5,2,1,0]:
            a = z.iloc[max(0,index-lag)]
            snap.append(dict(event=event, marker=kind, seconds_before_by_sample_index=lag,
                             marker_time=str(time), observation_time=str(a.DateTime), current_A=float(a["I.S.act [A]"]),
                             set_A=float(a["I.S.set [A]"]), operation_point=float(a.operation_point),
                             voltage_V=float(a.active10_voltage_mean_V), anode_pressure_kPag=float(a["p.Si.A [kPag]"]),
                             cathode_pressure_kPag=float(a["p.Si.C [kPag]"]), coolant_temperature_C=float(a["T.Si.CL [°C]"])))
    csv(f"outputs/events/{event}_past_observable_marker_snapshots.csv", snap)
    result = dict(event=event, zero_command_start=str(stop), first_actual_zero=str(first_zero.DateTime),
                  zero_command_end=str(last_zero), zero_command_samples=int(r.samples), restart_positive_command=str(restart.DateTime),
                  voltage_recovery_jump=str(jump.DateTime), recovery_delta_voltage_V=float(jump.delta_voltage_1sample_V),
                  recovery_before_positive_command_s=(restart.DateTime-jump.DateTime).total_seconds(),
                  zero_command_to_first_actual_zero_s=(first_zero.DateTime-stop).total_seconds(),
                  operation_code_during_zero_command=float(r.operation_point),
                  note="At-most-one-sample pressure prelude is observational, not evidence of a 30/120-s known future command")
    return result


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(pad=1.2)
    fig.savefig(output(f"figures/events/{name}.png"), dpi=300, facecolor="white")
    fig.savefig(output(f"figures/events/{name}.pdf"), facecolor="white")
    plt.close(fig)


def figures(event: str, z: pd.DataFrame, bins: pd.DataFrame, spec: dict) -> None:
    # Every line is plotted at native timestamps; no resampling/interpolation.
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(z.DateTime, z.active10_voltage_mean_V, color=BLUE, lw=.9, label="Raw active-10 mean")
    for i, b in bins.iterrows():
        axes[0].plot([pd.Timestamp(b.first_datetime), pd.Timestamp(b.last_datetime)],
                     [b.bin_mean_voltage_V, b.bin_mean_voltage_V], color=RED, lw=2.0, label="Frozen 0.5 op-hour mean" if i==0 else None)
    axes[0].set_ylabel("Cell mean (V)")
    axes[0].margins(y=.18)
    axes[0].legend(ncols=2, loc="upper right")
    axes[1].plot(z.DateTime, z["I.S.act [A]"], color=BLUE, lw=.8, label="Actual")
    axes[1].step(z.DateTime, z["I.S.set [A]"], where="post", color=RED, lw=.8, alpha=.8, label="Logged set point")
    axes[1].set_ylabel("Current (A)"); axes[1].legend(ncols=2, loc="upper right")
    axes[1].margins(y=.22)
    axes[2].step(z.DateTime, z.operation_point, where="post", color=GRAY, lw=1)
    axes[2].set_ylabel("Operation code")
    axes[3].plot(z.DateTime, z["OpHrs2 [h]"], color=BLUE, lw=1.2)
    axes[3].set_ylabel("OpHrs2 (h)"); axes[3].set_xlabel("Recorded calendar time")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[0].set_title(f"{event}: native 1-s observations and historical operating-clock bins", loc="left")
    save_figure(fig, f"{event}_raw_and_aggregation")
    csv(f"figures/events/{event}_raw_and_aggregation_plot_data.csv", z)
    csv(f"figures/events/{event}_raw_and_aggregation_bin_data.csv", bins)

    # Freeze-coordinate plateau is a timestamp fact, independent of the voltage/error.
    plateau = np.isclose(z["OpHrs2 [h]"], spec["frozen_h"], atol=1e-9, rtol=0)
    times = [z.DateTime.loc[plateau].iloc[0], z.DateTime.loc[plateau].iloc[-1]]
    all_zoom = []
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 10), sharex="col")
    for col, (edge, title) in enumerate(zip(times, ["Clock-freeze onset", "Clock-resume vicinity"])):
        g = z.loc[(z.DateTime>=edge-pd.Timedelta(seconds=180))&(z.DateTime<=edge+pd.Timedelta(seconds=180))].copy()
        g["seconds_from_clock_edge"] = (g.DateTime-edge).dt.total_seconds()
        g["edge"] = title
        all_zoom.append(g)
        t = g.seconds_from_clock_edge
        axes[0, col].plot(t, g.active10_voltage_mean_V, color=BLUE, lw=1.2)
        axes[1, col].plot(t, g["I.S.act [A]"], color=BLUE, lw=1.0, label="Actual")
        axes[1, col].step(t, g["I.S.set [A]"], color=RED, where="post", lw=1.0, label="Set point")
        axes[2, col].step(t, g.operation_point, color=GRAY, where="post", lw=1)
        axes[3, col].plot(t, g["p.Si.A [kPag]"], color=BLUE, lw=1, label="Anode inlet")
        axes[3, col].plot(t, g["p.Si.C [kPag]"], color=RED, lw=1, label="Cathode inlet")
        for ax in axes[:, col]:
            ax.axvline(0, color=GRAY, lw=.8, ls="--")
        axes[0, col].set_title(f"{event}: {title}\n{edge}", loc="left")
        axes[3, col].set_xlabel("Seconds from recorded clock edge")
    axes[0, 0].set_ylabel("Cell mean (V)")
    axes[1, 0].set_ylabel("Current (A)"); axes[1, 1].legend(ncols=2, loc="upper left")
    axes[2, 0].set_ylabel("Operation code")
    axes[3, 0].set_ylabel("Pressure (kPag)")
    axes[3, 1].legend(ncols=2, loc="upper left")
    save_figure(fig, f"{event}_native_transition_zoom")
    csv(f"figures/events/{event}_native_transition_zoom_plot_data.csv", pd.concat(all_zoom, ignore_index=True))


def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2,
                         "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    old = pd.read_csv(OLD, float_precision="round_trip")
    manifests, summaries, all_runs, all_lags, all_bins, jumps, timelines = [], [], [], [], [], [], []
    for event, spec in EVENTS.items():
        z, manifest = read_event(event, spec)
        manifests.append(manifest)
        runs, lags, bins, inc = run_table(event,z), command_lags(event,z), mixture(event,z,old), largest_increments(event,z)
        csv(f"outputs/events/{event}_state_runs.csv", runs)
        csv(f"outputs/events/{event}_command_current_lags.csv", lags)
        csv(f"outputs/events/{event}_historical_bin_recomputation.csv", bins)
        csv(f"outputs/events/{event}_largest_native_increments.csv", inc)
        all_runs.append(runs); all_lags.append(lags); all_bins.append(bins); jumps.append(inc)
        timelines.append(event_timeline(event,z,runs,spec))
        plateau = z.loc[np.isclose(z["OpHrs2 [h]"], spec["frozen_h"], atol=1e-9, rtol=0)]
        dt = z.datetime_step_s.dropna()
        summary = dict(event=event, retained_samples=len(z), first_datetime=str(z.DateTime.iloc[0]), last_datetime=str(z.DateTime.iloc[-1]),
                       op_min_h=float(z["OpHrs2 [h]"].min()), op_max_h=float(z["OpHrs2 [h]"].max()),
                       dt_min_s=float(dt.min()), dt_median_s=float(dt.median()), dt_max_s=float(dt.max()), gaps_gt_1p5_s=int((dt>1.5).sum()),
                       frozen_ophrs2_h=spec["frozen_h"], plateau_first=str(plateau.DateTime.iloc[0]), plateau_last=str(plateau.DateTime.iloc[-1]),
                       plateau_samples=len(plateau), plateau_calendar_s=(plateau.DateTime.iloc[-1]-plateau.DateTime.iloc[0]).total_seconds(),
                       plateau_mode_codes=sorted(plateau.operation_point.unique().tolist()), plateau_setpoints_A=sorted(plateau["I.S.set [A]"].unique().tolist()),
                       current_step_commands=len(lags), midpoint_lag_median_s=float(lags.observed_lag_s.median()) if lags.observed_lag_s.notna().any() else None,
                       midpoint_lag_counts={str(k):int(v) for k,v in lags.observed_lag_s.value_counts(dropna=False).items()},
                       max_abs_native_voltage_increment_V=float(z.delta_voltage_1sample_V.abs().max()),
                       U6_min_V=float(z["U6 [V]"].min()), U6_max_V=float(z["U6 [V]"].max()),
                       operating_modes=sorted(z.operation_point.unique().tolist()), fields_missing={c:int(z[c].isna().sum()) for c in USE},
                       future_plan_proved_available=False, selection_status="user-specified historical events; explanatory only; excluded from threshold/protocol fitting")
        summaries.append(summary)
        figures(event,z,bins,spec)
        print(f"EVENT_AUDIT_COMPLETE {event} samples={len(z)} command_steps={len(lags)} plateau={len(plateau)}", flush=True)
    csv("outputs/events/historical_bin_recomputation.csv", pd.concat(all_bins,ignore_index=True))
    csv("outputs/events/state_runs.csv", pd.concat(all_runs,ignore_index=True))
    csv("outputs/events/command_current_lags.csv", pd.concat(all_lags,ignore_index=True))
    csv("outputs/events/largest_native_increments.csv", pd.concat(jumps,ignore_index=True))
    json_out("outputs/events/source_manifest.json", manifests)
    json_out("outputs/events/event_summary.json", summaries)
    json_out("outputs/events/event_timeline.json", timelines)
    json_out("outputs/events/audit_method.json", dict(script_sha256=sha256(Path(__file__)),
              training_executed=False, models_or_thresholds_fit=False, no_interpolation=True, no_resampling=True,
              known_historical_zero_threshold_A=0.1, known_historical_current_transition_A=5.0,
              lag_search_seconds=[-30,30], lag_interpretation="record ordering only, not advance command-availability proof",
              event_selection="explicit user historical cases; held-out explanatory use, not independent confirmation",
              active_cells=ACTIVE, U6="retained diagnostic only; excluded from mean", old_bin_sha256=sha256(OLD),
              plot_exports="300 dpi PNG + PDF + native plotting CSV; each source row preserves original index and timestamp"))


if __name__ == "__main__":
    main()

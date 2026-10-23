"""Publication exports for predeclared event diagnostics, figures 06--09.

Every chart has its exact plotting CSV. No training, GPU or error-selected
case selection is performed. Run only after completed prediction diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from diagnose_events import DIAG, OUT, STUDY, available_runs, cache, read_json, sha

FIG = STUDY / "figures"
COLORS = {"Persistence": "#767676", "A1": "#3775BA", "A2": "#0F4D92",
          "A3": "#3775BA", "A4": "#0F4D92", "B1": "#42949E", "B2": "#42949E",
          "truth": "#272727", "current": "#B64342", "command": "#3775BA",
          "LSTM_path": "#0F4D92", "CNN_path": "#B64342"}
LABELS = {"A1": "LSTM", "A2": "Native F2", "A3": "LSTM", "A4": "Native F2",
          "B1": "Event-weighted F2", "B2": "Event-weighted F2"}


def style() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    families = [name for name in ("Arial", "Helvetica", "DejaVu Sans") if name in installed]
    plt.rcParams.update({"font.family": families or ["sans-serif"],
                         "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": 1.4, "legend.frameon": False,
                         "lines.linewidth": 1.8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})


def export(fig: plt.Figure, stem: str, data: pd.DataFrame, note: str) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIG / f"{stem}_plot_data.csv", index=False, lineterminator="\n")
    fig.text(.01, .009, note, ha="left", va="bottom", fontsize=8, color="#4D4D4D")
    fig.tight_layout(pad=1.7, rect=(0, .035, 1, .96))
    fig.savefig(FIG / f"{stem}.png", dpi=300)
    fig.savefig(FIG / f"{stem}.pdf", dpi=300)
    plt.close(fig)
    return {"stem": stem, "plot_rows": len(data), "PNG_dpi": 300,
            "PNG_sha256": sha(FIG / f"{stem}.png"), "PDF_sha256": sha(FIG / f"{stem}.pdf"),
            "plot_data_sha256": sha(FIG / f"{stem}_plot_data.csv")}


def cases() -> pd.DataFrame:
    selected = pd.read_csv(DIAG / "predeclared_branch_response_events.csv")
    selected = selected[selected.partition == "VALIDATION"].sort_values("first_row")
    assert len(selected) == 3, "Three preregistered chronological Validation episodes are required"
    return selected


def synchronized(arrays: dict) -> dict:
    selected = cases()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex="col")
    data = []
    for column, event in enumerate(selected.itertuples()):
        rows = np.arange(max(0, int(event.first_row) - 180),
                         min(len(arrays["time_ms"]), int(event.first_row) + 181))
        rows = rows[arrays["segment"][rows] == event.segment_id]
        relative = (arrays["time_ms"][rows] - event.first_ms) / 1000.0
        voltage = np.asarray(arrays["voltage_V"][rows])
        actual = np.asarray(arrays["x"][rows, 5]); command = np.asarray(arrays["x"][rows, 14])
        axes[0, column].plot(relative, actual, color=COLORS["current"], label="Measured current")
        axes[0, column].plot(relative, command, color=COLORS["command"], linestyle="--", label="Recorded setpoint")
        axes[1, column].plot(relative, voltage, color=COLORS["truth"], label="Raw ten-cell mean")
        for ax in axes[:, column]:
            ax.axvline(0, color="#767676", lw=1, linestyle=":")
        axes[0, column].set_title(f"Declared episode {event.event_id}")
        axes[0, column].set_ylabel("Current (A)")
        axes[1, column].set_ylabel("Mean cell voltage (V)")
        axes[1, column].set_xlabel("Seconds from first voltage trigger")
        if column == 0:
            axes[0, column].legend(loc="best", fontsize=8)
            axes[1, column].legend(loc="best", fontsize=8)
        for row, second, v, current, setpoint in zip(rows, relative, voltage, actual, command):
            data.append({"episode_id": int(event.event_id), "raw_row": int(row),
                         "original_time_ms": int(arrays["time_ms"][row]),
                         "seconds_from_first_trigger": float(second), "mean10_voltage_V": float(v),
                         "actual_current_A": float(current), "recorded_set_current_A": float(setpoint),
                         "operation_point": float(arrays["x"][row, 22])})
    fig.suptitle("06  Raw current and voltage at three predeclared Validation episodes", fontsize=13)
    return export(fig, "06_current_voltage_sync", pd.DataFrame(data),
                  "First three eligible chronological Validation episodes; no error-based case selection. Zero current is not a hardware shutdown label.")


def gate_paths(runs: list) -> dict:
    f2 = [r for r in runs if r[1] == "F2"]
    assert {r[0] for r in f2}.issuperset({"A2", "A4"})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    records = []
    for column, horizon in enumerate((30, 120)):
        matching = [r for r in f2 if r[2] == horizon]
        for name, model, h, run in matching:
            responses = pd.read_csv(DIAG / f"{name}_native_path_response.csv")
            val = responses[responses.partition == "VALIDATION"]
            average = val.groupby("first_trigger_offset_seconds", sort=True).mean(numeric_only=True)
            axes[0, column].plot(average.index, average.LSTM_gate_weight, marker="o", ms=4,
                                 label=LABELS[name], color=COLORS[name])
            for rec in val.to_dict("records"):
                records.append({"record_type": "event_response", **rec})
            gates = pd.read_csv(DIAG / f"{name}_all_validation_gate.csv")
            axes[2, column].hist(gates.LSTM_gate_weight, bins=30, density=True, alpha=.35,
                                 color=COLORS[name], label=LABELS[name])
            for rec in gates.to_dict("records"):
                records.append({"record_type": "all_validation_gate", **rec})
            if name in ("A2", "A4"):
                for branch in ("LSTM", "CNN"):
                    field = f"{branch}_hidden_change_from_minus120_RMS"
                    axes[1, column].plot(average.index, average[field], marker="o", ms=4,
                                         color=COLORS[f"{branch}_path"], label=f"{branch} path")
        axes[0, column].axvline(0, color="#767676", ls=":", lw=1)
        axes[0, column].axhline(.5, color="#767676", ls="--", lw=1)
        axes[0, column].set_ylim(0, 1)
        axes[0, column].set_title(f"H={horizon}s: gate response (3 declared episodes)")
        axes[0, column].set_ylabel("LSTM-path fusion weight")
        axes[1, column].axvline(0, color="#767676", ls=":", lw=1)
        axes[1, column].set_title(f"Native F2 H={horizon}s: hidden-path response")
        axes[1, column].set_ylabel("Hidden change RMS from t=-120s")
        axes[1, column].set_xlabel("Seconds from first voltage trigger")
        axes[2, column].set_xlabel("LSTM-path fusion weight")
        axes[2, column].set_ylabel("Density over all Validation origins")
        axes[2, column].set_title(f"H={horizon}s: full Validation gate distribution")
        for row in range(3):
            axes[row, column].legend(fontsize=8, loc="best")
    fig.suptitle("07  F2 historical gate and native hidden-path responses", fontsize=13)
    return export(fig, "07_F2_gate_and_path_response", pd.DataFrame(records),
                  "Episode curves are descriptive means over 3 fixed episodes, not independent repeats. Hidden changes are not branch-specific voltage contributions.")


def extremes(runs: list) -> dict:
    curve = pd.read_csv(DIAG / "error_contribution_curve.csv")
    episodes = pd.read_csv(DIAG / "error_by_independent_episode.csv")
    summary = pd.read_csv(DIAG / "extreme_error_summary.csv")
    fig = plt.figure(figsize=(12, 8.6))
    layout = fig.add_gridspec(3, 2, height_ratios=[1, 1, .12])
    axes = np.array([[fig.add_subplot(layout[row, column]) for column in range(2)]
                     for row in range(2)])
    records = []
    for row, horizon in enumerate((30, 120)):
        matching = [r for r in runs if r[2] == horizon]
        for name, model, h, run in matching:
            data = curve[curve.run == name]
            assert len(data)
            c = COLORS.get(name, COLORS["Persistence"])
            label = LABELS.get(name, "Persistence")
            axes[row, 0].plot(100 * data.largest_error_query_fraction,
                              100 * data.cumulative_SSE_fraction,
                              color=c, marker="o", ms=3, label=label,
                              linestyle="--" if model == "Persistence" else "-")
            for rec in data.to_dict("records"):
                records.append({"record_type": "query_rank_contribution", **rec})
        axes[row, 0].set_xscale("log")
        axes[row, 0].set_xlim(.1, 100); axes[row, 0].set_ylim(0, 103)
        axes[row, 0].axvline(1, color="#767676", linestyle=":", lw=1)
        axes[row, 0].set_xlabel("Largest-error queries retained (%)")
        axes[row, 0].set_ylabel("Cumulative share of all squared error (%)")
        axes[row, 0].set_title(f"H={horizon}s: concentration (all queries retained)")
        axes[row, 0].legend(fontsize=8)
        labels = []; ordinary = []; leading = []; other = []
        for name, model, h, run in matching:
            data = episodes[episodes.run == name]
            base = float(data.loc[data.event_id == 0, "SSE_V2"].sum())
            event = data[data.event_id > 0].sort_values("SSE_V2", ascending=False)
            dominant = float(event.head(10).SSE_V2.sum())
            remaining = float(event.iloc[10:].SSE_V2.sum())
            total = base + dominant + remaining
            ordinary.append(100 * base / total); leading.append(100 * dominant / total)
            other.append(100 * remaining / total)
            labels.append(LABELS.get(name, "Persistence"))
            for rec in data.to_dict("records"):
                records.append({"record_type": "operational_episode_contribution", **rec})
        y = np.arange(len(labels))
        axes[row, 1].barh(y, ordinary, color="#CFCECE", edgecolor="black", lw=.6, label="Ordinary / no event ID")
        axes[row, 1].barh(y, leading, left=ordinary, color="#B64342", edgecolor="black", lw=.6,
                          label="10 largest-error episodes")
        axes[row, 1].barh(y, other, left=np.array(ordinary) + np.array(leading), color="#42949E",
                          edgecolor="black", lw=.6, label="Other voltage episodes")
        axes[row, 1].set_yticks(y, labels); axes[row, 1].invert_yaxis()
        axes[row, 1].set_xlim(0, 100); axes[row, 1].set_xlabel("Share of all squared error (%)")
        axes[row, 1].set_title(f"H={horizon}s: operational episode attribution")
    legend_axis = fig.add_subplot(layout[2, :])
    legend_axis.set_axis_off()
    handles, labels = axes[0, 1].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", ncol=3, fontsize=9)
    for rec in summary.to_dict("records"):
        records.append({"record_type": "top1pct_summary", **rec})
    fig.suptitle("08  Extreme-query and episode contributions to total Validation error", fontsize=13)
    return export(fig, "08_extreme_error_contribution", pd.DataFrame(records),
                  "All origin x lead queries are counted once. H120 targets overlap across origins; episodes are quiet-gap proxies, not independent trials.")


def all_metrics(run: Path) -> dict:
    with np.load(run / "predictions_VALIDATION_60s.npz", allow_pickle=False) as data:
        e = data["prediction_delta_V"] - data["truth_delta_V"]
        event = data["event_query"].astype(bool)
        return {"all_RMSE_mV": float(np.sqrt(np.mean(e ** 2)) * 1000),
                "event_RMSE_mV": float(np.sqrt(np.mean(e[event] ** 2)) * 1000),
                "normal_MAE_mV": float(np.mean(np.abs(e[~event])) * 1000)}


def weighted_comparison(arrays: dict, runs: list) -> dict:
    by_name = {r[0]: r for r in runs}
    assert "B1" in by_name, "Do not generate a weighted comparison before B1 completes"
    acceptance = read_json(OUT / "B1_acceptance.json")
    b2 = "B2" in by_name
    assert b2 or not acceptance["B2_allowed"], "B2 is authorized but not completed; final figure must wait"
    nrows = 4 if b2 else 2
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 5.5 * nrows / 2), squeeze=False)
    selected = cases()
    records = []
    for block, horizon in enumerate((30, 120) if b2 else (30,)):
        native, weighted = ("A2", "B1") if horizon == 30 else ("A4", "B2")
        metrics = {name: all_metrics(by_name[name][3]) for name in (native, weighted)}
        for column, (field, title) in enumerate((("all_RMSE_mV", "All-query RMSE"),
                                                  ("event_RMSE_mV", "Event-query RMSE"),
                                                  ("normal_MAE_mV", "Ordinary-query MAE"))):
            ax = axes[2 * block, column]
            values = [metrics[name][field] for name in (native, weighted)]
            bars = ax.bar([0, 1], values, color=[COLORS[native], COLORS[weighted]],
                          edgecolor="black", lw=.8, width=.65)
            ax.set_xticks([0, 1], ["Native F2", "Weighted F2"])
            ax.set_ylabel("mV"); ax.set_title(f"H={horizon}s: {title}")
            ax.set_ylim(0, max(values) * 1.22)
            for bar, name, value in zip(bars, (native, weighted), values):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
                records.append({"record_type": "all_validation_metric", "run": name,
                                "horizon_seconds": horizon, "metric": field, "value_mV": value,
                                "query_scope": "ALL_ORIGIN_TIMES_LEAD"})
        with np.load(by_name[native][3] / "predictions_VALIDATION_60s.npz", allow_pickle=False) as native_data, \
                np.load(by_name[weighted][3] / "predictions_VALIDATION_60s.npz", allow_pickle=False) as weighted_data:
            assert np.array_equal(native_data["origins"], weighted_data["origins"])
            assert np.array_equal(native_data["targets"], weighted_data["targets"])
            origins = native_data["origins"]
            for column, event in enumerate(selected.itertuples()):
                # Latest pre-trigger origin is deterministic even if the first
                # trigger falls outside the short forecast on a 60s origin grid.
                index = int(np.searchsorted(origins, int(event.first_row), side="left") - 1)
                assert index >= 0
                targets = native_data["targets"][index]
                origin = int(origins[index])
                x = (native_data["target_time_ms"][index] - event.first_ms) / 1000.0
                truth = native_data["origin_voltage_V"][index] + native_data["truth_delta_V"][index]
                ax = axes[2 * block + 1, column]
                ax.plot(x, truth, color=COLORS["truth"], label="True raw voltage")
                for name, data in ((native, native_data), (weighted, weighted_data)):
                    prediction = data["origin_voltage_V"][index] + data["prediction_delta_V"][index]
                    ax.plot(x, prediction, color=COLORS[name], linestyle="--" if name == weighted else "-", label=LABELS[name])
                    for lead, row, second, y, p in zip(range(1, horizon + 1), targets, x, truth, prediction):
                        records.append({"record_type": "predeclared_episode_prediction", "run": name,
                                        "horizon_seconds": horizon, "event_id": int(event.event_id),
                                        "origin_row": origin, "target_row": int(row), "lead_seconds": lead,
                                        "original_target_time_ms": int(arrays["time_ms"][row]),
                                        "seconds_from_first_trigger": float(second),
                                        "truth_voltage_V": float(y), "prediction_voltage_V": float(p)})
                if x.min() <= 0 <= x.max():
                    ax.axvline(0, color="#767676", linestyle=":", lw=1)
                covered = int(targets[0]) <= int(event.first_row) <= int(targets[-1])
                suffix = "trigger inside horizon" if covered else "trigger outside horizon"
                ax.set_title(f"Episode {event.event_id}: {suffix}", fontsize=9)
                ax.set_xlabel("Target seconds from first trigger"); ax.set_ylabel("Mean cell voltage (V)")
                if column == 0:
                    ax.legend(fontsize=7)
    if b2:
        subtitle = "H30 and conditionally admitted H120 experiments"
    else:
        subtitle = "H120 weighted model: NOT_RUN because B1 acceptance gate failed (not plotted as zero)"
        records.append({"record_type": "experiment_status", "run": "B2", "horizon_seconds": 120,
                        "status": "NOT_RUN_B1_ACCEPTANCE_FAILED"})
    fig.suptitle(f"09  Event-weighted versus native F2\n{subtitle}", fontsize=12)
    return export(fig, "09_event_weighted_comparison", pd.DataFrame(records),
                  "Cases use the same latest pre-trigger 60s-grid origin and every forecast lead. These fixed examples are not representative-performance claims.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("intermediate", "final"), default="final")
    args = parser.parse_args()
    style()
    _, arrays = cache()
    runs = available_runs()
    assert {r[0] for r in runs}.issuperset({"A1", "A2", "A3", "A4"}), "Wait for all native runs"
    outputs = [synchronized(arrays), gate_paths(runs), extremes(runs)]
    if args.phase == "final":
        outputs.append(weighted_comparison(arrays, runs))
    (DIAG / "diagnostic_figure_manifest.json").write_text(
        json.dumps({"phase": args.phase, "figures": outputs, "GPU_calls": 0,
                    "case_selection": "predeclared_first_three_eligible_validation_episodes"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"stage": "PLOT_DIAGNOSTICS", "figures": [r["stem"] for r in outputs],
                      "GPU_calls": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

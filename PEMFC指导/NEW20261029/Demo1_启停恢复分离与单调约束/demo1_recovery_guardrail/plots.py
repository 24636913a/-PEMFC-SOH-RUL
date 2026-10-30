"""Plot observations/results; optional truth overlays are evaluation only."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import METHODS


COLORS = {"raw_voltage": "#c76b4e", "recovery_separated": "#479c99", "recovery_guardrail": "#244d69"}
LABELS = {"raw_voltage": "Raw-voltage baseline", "recovery_separated": "Recovery separated", "recovery_guardrail": "+ Monotonic guardrail"}


def make_plots(result, observations, truth, output_dir: Path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.grid": True, "grid.alpha": .15,
                         "savefig.bbox": "tight", "savefig.dpi": 180})
    history = result["history"]
    x = history.time_h
    cutoff = result["cutoff_h"]
    threshold = result["threshold_v"]
    status = "SYNTHETIC teaching data" if result["is_synthetic"] else "User-supplied observations"
    seen_events = history.loc[(history.restart == 1) & (history.time_h > result["history_start_h"]), "time_h"]
    selected_event = (809.0 if np.isclose(seen_events, 809).any()
                      else (float(seen_events.iloc[-1]) if len(seen_events) else None))
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10), sharex=True, constrained_layout=True)
    fig.suptitle(f"Demo 1 | Recovery is separate from long-term degradation\n{status}; fixed forecast origin = {cutoff:g} h", fontweight="bold")
    axes[0].plot(x, history.standardized_observed_v, color="#aaaaaa", lw=.7, label="Observed voltage (reference conditions)")
    if truth is not None:
        past = truth.loc[truth.time_h <= cutoff]
        axes[0].plot(past.time_h, past.latent_voltage_v, color="black", lw=2, label="True latent voltage (evaluation only)")
    axes[0].axhline(threshold, ls="--", color="#b87158", label=f"{100*(1-result['voltage_loss_fraction']):g}% V0 threshold")
    axes[0].set_ylabel("Cell voltage (V)")
    axes[0].legend(ncol=2, loc="lower left", fontsize=9)
    axes[1].plot(x, history.estimated_q_rev_v * 1000, color=COLORS["raw_voltage"], lw=1.7, label="Estimated remaining reversible loss q")
    if truth is not None:
        axes[1].plot(past.time_h, past.q_rev_v * 1000, "k--", lw=1, label="True remaining reversible loss")
    axes[1].set_ylabel("Reversible loss (mV)")
    axes[1].legend(fontsize=9, loc="lower left")
    for method in METHODS:
        axes[2].plot(x, history[f"{method}_estimate_v"], color=COLORS[method], lw=1.5, label=LABELS[method])
    if truth is not None:
        axes[2].plot(past.time_h, past.latent_voltage_v, "k--", lw=1.4, label="True latent voltage")
    axes[2].set_ylabel("Latent-state estimate (V)")
    axes[2].set_xlabel("Operating time (h)")
    axes[2].legend(fontsize=9, ncol=2, loc="lower left")
    for ax in axes:
        ax.axvspan(0, result["history_start_h"], color="#eeeeee", alpha=.6)
        if selected_event is not None:
            ax.axvline(selected_event, ls=":", color="#855c4d", alpha=.6)
        ax.set_xlim(0, cutoff)
    axes[0].text(16, .715, "Initial calibration only", fontsize=9, color="#555555")
    fig.savefig(output_dir / "01_recovery_decomposition.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    if truth is not None:
        ax.plot(truth.time_h, truth.latent_voltage_v, color="black", lw=2, label="True latent voltage (evaluation only)")
    future = result["forecast"]
    for method in METHODS:
        ax.plot(history.time_h, history[f"{method}_estimate_v"], color=COLORS[method], lw=1, alpha=.65)
        ax.plot(future.time_h, future[f"{method}_forecast_v"], color=COLORS[method], lw=2,
                ls="--", label=LABELS[method])
    ax.axvline(cutoff, color="#333333", ls=":", label=f"Forecast origin: {cutoff:g} h")
    ax.axhline(threshold, color="#a5654a", ls="-.", label=f"Latent threshold: {100*(1-result['voltage_loss_fraction']):g}% V0")
    ax.axvspan(0, cutoff, color="#eeeeee", alpha=.5)
    ax.set_xlim(result["history_start_h"], result["forecast"].time_h.iloc[-1])
    ax.set_xlabel("Operating time (h)")
    ax.set_ylabel("Reference-condition cell voltage (V)")
    ax.set_title(f"Fixed-origin prognosis | No held-out voltage or future restart flags used\n{status}; transparent teaching-model outputs", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    fig.savefig(output_dir / "02_fixed_origin_forecast.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    if selected_event is None:
        for ax in axes:
            ax.set_axis_off()
        axes[0].text(.5, .5, "No observed restart after calibration and before the forecast cutoff", ha="center", transform=axes[0].transAxes)
        fig.savefig(output_dir / "03_restart_at_809h.png")
        plt.close(fig)
        return
    mask = (history.time_h >= selected_event - 29) & (history.time_h <= selected_event + 31)
    zoom = history.loc[mask]
    axes[0].plot(zoom.time_h, zoom.standardized_observed_v, color=COLORS["raw_voltage"], marker=".", ms=3, label="Observed voltage")
    axes[0].axhline(threshold, color="#666666", ls="--", label=f"{100*(1-result['voltage_loss_fraction']):g}% V0 threshold")
    if truth is not None:
        ztruth = truth.loc[(truth.time_h >= selected_event - 29) & (truth.time_h <= selected_event + 31)]
        axes[0].plot(ztruth.time_h, ztruth.latent_voltage_v, color="black", lw=1.8, label="True latent voltage")
        axes[1].plot(ztruth.time_h, ztruth.latent_voltage_v, color="black", lw=1.8, label="True latent voltage")
    axes[1].plot(zoom.time_h, zoom.recovery_corrected_measurement_v, color="#bbbbbb", lw=.8, label="Observed voltage + estimated q")
    axes[1].plot(zoom.time_h, zoom.recovery_separated_estimate_v, color=COLORS["recovery_separated"], label="Recovery separated")
    axes[1].plot(zoom.time_h, zoom.recovery_guardrail_estimate_v, color=COLORS["recovery_guardrail"], label="+ Guardrail")
    axes[0].set_title(f"Observed restart at {selected_event:g} h | {status}", fontweight="bold")
    for ax in axes:
        ax.axvline(selected_event, color="#855c4d", ls=":")
        ax.set_ylabel("Cell voltage (V)")
        ax.legend(loc="best", fontsize=9, ncol=2)
    axes[1].set_xlabel("Operating time (h)")
    fig.savefig(output_dir / "03_restart_at_809h.png")
    plt.close(fig)

"""Truth is allowed only here, after inference has completed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from model import METHODS


def first_crossing(time, values, threshold):
    time, values = np.asarray(time), np.asarray(values)
    idx = np.flatnonzero(values <= threshold)
    if len(idx) == 0:
        return None
    k = int(idx[0])
    if k == 0:
        return float(time[k])
    fraction = (values[k - 1] - threshold) / (values[k - 1] - values[k])
    return float(time[k - 1] + fraction * (time[k] - time[k - 1]))


def evaluate(result: dict, truth: pd.DataFrame | None) -> tuple[dict, pd.DataFrame]:
    metrics = {
        "data_status": ("SYNTHETIC; no real FC1 data are included" if result["is_synthetic"]
                        else "User-supplied observations; provenance not verified by this demo"),
        "is_synthetic": result["is_synthetic"],
        "forecast_cutoff_h": result["cutoff_h"],
        "threshold_v": result["threshold_v"],
        "reference_voltage_v": result["reference_voltage_v"],
        "voltage_loss_fraction": result["voltage_loss_fraction"],
        "forecast_fit_window_h": result["forecast_fit_window_h"],
        "threshold_definition": "(1 - voltage_loss_fraction) times initial reference voltage",
        "evaluation_available": truth is not None,
        "methods": {key: value.copy() for key, value in result["forecast_summary"].items()},
    }
    history = result["history"]
    events = []
    for i in range(1, len(history)):
        if not history.restart.iloc[i]:
            continue
        row, previous = history.iloc[i], history.iloc[i - 1]
        if np.isnan(row.estimated_q_rev_v) or np.isnan(previous.estimated_q_rev_v):
            continue
        events.append({
            "restart_time_h": row.time_h,
            "observed_jump_v": row.standardized_observed_v - previous.standardized_observed_v,
            "estimated_recovered_loss_v": previous.estimated_q_rev_v - row.estimated_q_rev_v,
            "raw_voltage_before_v": previous.standardized_observed_v,
            "raw_voltage_after_v": row.standardized_observed_v,
        })
    event_df = pd.DataFrame(events)
    if truth is None:
        return metrics, event_df
    truth = truth.sort_values("time_h").reset_index(drop=True)
    required = {"time_h", "latent_voltage_v", "q_rev_v", "noise_free_operating_voltage_v"}
    if not required.issubset(truth.columns):
        raise ValueError(f"Truth table must have columns: {sorted(required)}")
    merged_history = history.merge(truth[["time_h", "latent_voltage_v"]], on="time_h", validate="one_to_one")
    merged_future = result["forecast"].merge(truth[["time_h", "latent_voltage_v"]], on="time_h", validate="one_to_one")
    # Exclude the observed cutoff point from the held-out forecast score.
    merged_future = merged_future.loc[merged_future.time_h > result["cutoff_h"]]
    true_eol = first_crossing(truth.time_h, truth.latent_voltage_v, result["threshold_v"])
    metrics["true_latent_threshold_time_h"] = true_eol
    metrics["true_rul_h"] = None if true_eol is None else max(true_eol - result["cutoff_h"], 0.0)
    metrics["noise_free_operating_voltage_first_threshold_time_h"] = first_crossing(
        truth.time_h, truth.noise_free_operating_voltage_v, result["threshold_v"])
    metrics["heldout_n"] = int(len(merged_future))
    metrics["history_evaluation_window_h"] = [result["history_start_h"], result["cutoff_h"]]
    for name in METHODS:
        valid = merged_history[f"{name}_estimate_v"].notna()
        error = (merged_history.loc[valid, f"{name}_estimate_v"]
                 - merged_history.loc[valid, "latent_voltage_v"]).to_numpy(float)
        future_error = (merged_future[f"{name}_forecast_v"]
                        - merged_future.latent_voltage_v).to_numpy(float)
        m = metrics["methods"][name]
        m["history_latent_rmse_mv"] = float(1000 * np.sqrt(np.mean(error ** 2)))
        m["heldout_latent_rmse_mv"] = float(1000 * np.sqrt(np.mean(future_error ** 2)))
        m["rul_absolute_error_h"] = (None if metrics["true_rul_h"] is None or m["rul_h"] is None
                                      else abs(m["rul_h"] - metrics["true_rul_h"]))
    for k, row in event_df.iterrows():
        prev = truth.loc[truth.time_h < row.restart_time_h].iloc[-1]
        now = truth.loc[np.isclose(truth.time_h, row.restart_time_h)].iloc[0]
        event_df.loc[k, "true_recovered_loss_v"] = prev.q_rev_v - now.q_rev_v
        event_df.loc[k, "true_latent_before_v"] = prev.latent_voltage_v
        event_df.loc[k, "true_latent_after_v"] = now.latent_voltage_v
    return metrics, event_df

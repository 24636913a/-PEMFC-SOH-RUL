"""Observation-only models. This module never opens or receives a truth table."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


METHODS = ("raw_voltage", "recovery_separated", "recovery_guardrail")


def validate_observations(df: pd.DataFrame) -> pd.DataFrame:
    required = ["time_h", "stack_voltage_v", "n_cells", "restart", "condition_offset_v"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing observation columns: {missing}")
    out = df.copy().sort_values("time_h").reset_index(drop=True)
    if len(out) < 30 or out[required].isna().any().any():
        raise ValueError("At least 30 complete observations are required.")
    if not np.isfinite(out[required].to_numpy(dtype=float)).all():
        raise ValueError("Observation columns must contain finite numeric values.")
    if (out.n_cells <= 0).any() or (out.n_cells != np.floor(out.n_cells)).any():
        raise ValueError("n_cells must be positive integers.")
    if not out.restart.isin([0, 1]).all():
        raise ValueError("restart must be 0 or 1.")
    if (out.time_h.diff().dropna() <= 0).any():
        raise ValueError("time_h must be unique and strictly increasing.")
    if not np.allclose(out.time_h.diff().dropna(), out.time_h.diff().dropna().iloc[0]):
        raise ValueError("This teaching filter requires a uniformly spaced time series.")
    out["cell_voltage_v"] = out.stack_voltage_v / out.n_cells
    out["standardized_observed_v"] = out.cell_voltage_v - out.condition_offset_v
    return out


def recovery_age(time: np.ndarray, restart: np.ndarray) -> np.ndarray:
    """Uses only restart flags at or before each time, never future flags."""
    last_restart = float(time[0])
    age = np.zeros(len(time), dtype=float)
    for k, (t, flag) in enumerate(zip(time, restart)):
        if flag:
            last_restart = float(t)
        age[k] = float(t) - last_restart
    return age


def fit_recovery_model(observed: pd.DataFrame, cfg: dict) -> dict:
    """Fit v=a-beta*t-A*(1-exp(-age/tau)) using calibration history only.

    For each predeclared tau, solve bounded least squares for a, beta and A.
    Selection is based only on calibration residuals, not holdout accuracy.
    At least one restart is required to help separate beta from q(age).
    """
    time = observed.time_h.to_numpy(float)
    age = recovery_age(time, observed.restart.to_numpy(int))
    mask = time <= cfg["calibration_end_h"]
    if mask.sum() < 30 or observed.loc[mask & (time > time[0]), "restart"].sum() < 1:
        raise ValueError("Calibration requires >=30 observations and at least one restart.")
    lo, hi, step = cfg["tau_candidates_h"]
    if lo <= 0 or step <= 0 or hi < lo:
        raise ValueError("Invalid positive tau grid.")
    lower = [cfg["initial_voltage_bounds_v"][0], cfg["decay_bounds_v_per_h"][0],
             cfg["reversible_amplitude_bounds_v"][0]]
    upper = [cfg["initial_voltage_bounds_v"][1], cfg["decay_bounds_v_per_h"][1],
             cfg["reversible_amplitude_bounds_v"][1]]
    y = observed.standardized_observed_v.to_numpy(float)[mask]
    best = None
    for tau in np.arange(lo, hi + 0.5 * step, step):
        basis = 1 - np.exp(-age[mask] / tau)
        design = np.column_stack([np.ones(mask.sum()), -time[mask], -basis])
        fit = lsq_linear(design, y, bounds=(lower, upper), tol=1e-12)
        if not fit.success:
            continue
        mse = float(np.mean((design @ fit.x - y) ** 2))
        if best is None or mse < best[0]:
            best = (mse, float(tau), fit.x)
    if best is None:
        raise RuntimeError("No bounded least-squares fit succeeded.")
    mse, tau, coef = best
    return {
        "initial_voltage_fit_v": float(coef[0]),
        "decay_fit_v_per_h": float(coef[1]),
        "reversible_amplitude_fit_v": float(coef[2]),
        "reversible_tau_fit_h": tau,
        "calibration_mse_v2": mse,
        "calibration_last_used_h": float(time[mask][-1]),
        "calibration_n": int(mask.sum()),
        "tau_grid_used_h": [lo, hi, step],
    }


def calculate_rul(current_v: float, slope_v_per_h: float, threshold_v: float):
    if current_v <= threshold_v:
        return 0.0
    if slope_v_per_h >= -1e-12:
        return None  # No finite crossing under this linear extrapolation.
    return float((current_v - threshold_v) / -slope_v_per_h)


def forecast_from_observations(observations: pd.DataFrame, model_config: dict,
                               reference_config: dict) -> dict:
    """Causal entry point. Receives no simulator parameters and no truth.

    All future observations and future restart flags are removed before fitting.
    Model parameters are frozen after initial calibration. History estimates
    begin at calibration_end_h; earlier rows are not retrospective estimates.
    """
    cfg = model_config
    cutoff = float(cfg["forecast_cutoff_h"])
    # Truncate before validation/calibration: held-out rows cannot affect fit.
    train = observations.loc[observations.time_h <= cutoff].copy()
    observed = validate_observations(train)
    time = observed.time_h.to_numpy(float)
    if not np.isclose(time[-1], cutoff):
        raise ValueError("The forecast cutoff must be an observed time point.")
    if cutoff <= cfg["calibration_end_h"]:
        raise ValueError("Forecast cutoff must be after calibration end.")
    parameters = fit_recovery_model(observed, cfg)
    age = recovery_age(time, observed.restart.to_numpy(int))
    q = parameters["reversible_amplitude_fit_v"] * (
        1 - np.exp(-age / parameters["reversible_tau_fit_h"]))
    corrected = observed.standardized_observed_v.to_numpy(float) + q
    y = observed.standardized_observed_v.to_numpy(float)
    start = int(np.searchsorted(time, cfg["calibration_end_h"], side="left"))
    alpha = float(cfg["smoothing_alpha"])
    if not 0 < alpha <= 1:
        raise ValueError("smoothing_alpha must be in (0,1].")
    hist = observed.copy()
    hist["restart_age_h"] = age
    hist["estimated_q_rev_v"] = q
    # Prior-to-calibration q estimates are calibration diagnostics only;
    # suppress them here to avoid presenting retrospective estimates as online.
    hist.loc[: start - 1, "estimated_q_rev_v"] = np.nan
    hist["recovery_corrected_measurement_v"] = corrected
    hist.loc[: start - 1, "recovery_corrected_measurement_v"] = np.nan
    series = {name: np.full(len(time), np.nan) for name in METHODS}
    init = (time >= time[start] - cfg["initialization_window_h"]) & (time <= time[start])
    series["raw_voltage"][start] = float(np.mean(y[init]))
    series["recovery_separated"][start] = float(np.mean(corrected[init]))
    series["recovery_guardrail"][start] = series["recovery_separated"][start]
    for k in range(start + 1, len(time)):
        series["raw_voltage"][k] = ((1 - alpha) * series["raw_voltage"][k - 1]
                                           + alpha * y[k])
        series["recovery_separated"][k] = (
            (1 - alpha) * series["recovery_separated"][k - 1] + alpha * corrected[k])
        previous = series["recovery_guardrail"][k - 1]
        candidate = (1 - alpha) * previous + alpha * corrected[k]
        # One-step Euclidean projection onto x_t <= x_(t-1).
        # This changes only the latent-state estimate, never the raw observation.
        series["recovery_guardrail"][k] = min(previous, candidate)
    for name in METHODS:
        hist[f"{name}_estimate_v"] = series[name]
    fitmask = (time > cutoff - cfg["forecast_fit_window_h"]) & (time >= time[start])
    if fitmask.sum() < 10:
        raise ValueError("Not enough history for the forecast-fit window.")
    future_time = np.arange(cutoff, cfg["forecast_end_h"] + 0.5 * cfg["forecast_step_h"],
                            cfg["forecast_step_h"], dtype=float)
    if len(future_time) < 2:
        raise ValueError("Forecast horizon must extend beyond cutoff.")
    future = pd.DataFrame({"time_h": future_time, "remaining_time_h": future_time - cutoff})
    threshold = reference_config["initial_reference_cell_voltage_v"] * (
        1 - reference_config["voltage_loss_fraction"])
    forecasts = {}
    for name in METHODS:
        slope, _ = np.polyfit(time[fitmask] - cutoff, series[name][fitmask], 1)
        if name == "recovery_guardrail":
            slope = min(float(slope), 0.0)
        current = float(series[name][-1])
        future[f"{name}_forecast_v"] = current + float(slope) * (future_time - cutoff)
        rul = calculate_rul(current, float(slope), float(threshold))
        forecasts[name] = {
            "current_estimate_v": current, "slope_v_per_h": float(slope),
            "rul_h": rul, "threshold_crossing_time_h": None if rul is None else cutoff + rul,
            "crossing_within_plotted_horizon": bool(rul is not None and cutoff + rul <= future_time[-1]),
            "forecast_fit_first_h": float(time[fitmask][0]),
            "forecast_fit_last_h": float(time[fitmask][-1]),
            "history_upward_steps": int(np.sum(np.diff(series[name][start:]) > 1e-12)),
        }
    return {"history": hist, "forecast": future, "parameters": parameters,
            "forecast_summary": forecasts, "threshold_v": float(threshold),
            "cutoff_h": cutoff, "history_start_h": float(time[start]),
            "reference_voltage_v": float(reference_config["initial_reference_cell_voltage_v"]),
            "voltage_loss_fraction": float(reference_config["voltage_loss_fraction"]),
            "forecast_fit_window_h": float(cfg["forecast_fit_window_h"]),
            "is_synthetic": bool("is_synthetic" in observed and
                                 observed.is_synthetic.astype(str).str.lower().eq("true").all())}

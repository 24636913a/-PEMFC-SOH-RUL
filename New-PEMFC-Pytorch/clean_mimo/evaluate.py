"""Per-horizon metrics, lawful baselines, and saved prediction curves."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def persistence_forecast(X_raw: np.ndarray, utot_feature_index: int, horizon: int) -> np.ndarray:
    """Forecast every future hour as the last observed hourly Utot."""
    origin_voltage = X_raw[:, -1, utot_feature_index]
    return np.repeat(origin_voltage[:, None], horizon, axis=1)


def linear_trend_forecast(X_raw: np.ndarray, utot_feature_index: int, horizon: int) -> np.ndarray:
    """Least-squares extrapolation from the preceding 15 legal voltage observations."""
    x = np.arange(X_raw.shape[1], dtype=float)
    future_x = np.arange(X_raw.shape[1], X_raw.shape[1] + horizon, dtype=float)
    output = np.empty((len(X_raw), horizon), dtype=np.float32)
    for row, history in enumerate(X_raw[:, :, utot_feature_index]):
        slope, intercept = np.polyfit(x, history, deg=1)
        output[row] = slope * future_x + intercept
    return output


def metric_rows(y_true: np.ndarray, y_pred: np.ndarray, model: str, persistence: np.ndarray | None = None) -> pd.DataFrame:
    """Report MAE/RMSE/R² separately for h=1..H and their arithmetic mean."""
    rows = []
    for idx in range(y_true.shape[1]):
        mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
        rmse = mean_squared_error(y_true[:, idx], y_pred[:, idx]) ** 0.5
        r2 = r2_score(y_true[:, idx], y_pred[:, idx])
        row = {"model": model, "horizon": idx + 1, "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
        if persistence is not None:
            pers_rmse = mean_squared_error(y_true[:, idx], persistence[:, idx]) ** 0.5
            pers_mse = mean_squared_error(y_true[:, idx], persistence[:, idx])
            mse = mean_squared_error(y_true[:, idx], y_pred[:, idx])
            row["persistence_RMSE"] = float(pers_rmse)
            row["RMSE_improvement_vs_persistence"] = float(1.0 - rmse / pers_rmse) if pers_rmse else np.nan
            row["MSE_skill_vs_persistence"] = float(1.0 - mse / pers_mse) if pers_mse else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    average = {"model": model, "horizon": "mean_h1_h4"}
    for col in table.columns:
        if col not in {"model", "horizon"}:
            average[col] = float(table[col].mean())
    return pd.concat([table, pd.DataFrame([average])], ignore_index=True)


def prediction_frame(origin_times: np.ndarray, target_times: np.ndarray, y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for sample in range(len(origin_times)):
        for h in range(y_true.shape[1]):
            row = {
                "sample": sample,
                "origin_time_h": float(origin_times[sample]),
                "target_time_h": float(target_times[sample, h]),
                "horizon": h + 1,
                "actual_Utot_mean": float(y_true[sample, h]),
            }
            row.update({f"pred_{name}": float(value[sample, h]) for name, value in predictions.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def save_prediction_curve(frame: pd.DataFrame, output_dir: Path, model: str, max_points: int = 250) -> None:
    """Save an h=4 representative future-voltage curve without selecting on test R²."""
    view = frame.loc[frame["horizon"] == 4].iloc[:max_points]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(view["target_time_h"], view["actual_Utot_mean"], label="observed Utot_mean", linewidth=1.8, color="black")
    ax.plot(view["target_time_h"], view[f"pred_{model}"], label=f"{model} h=4", linewidth=1.4)
    ax.set_xlabel("Target interval end time (h)")
    ax.set_ylabel("Hourly mean total voltage (V)")
    ax.set_title(f"Clean MIMO test prediction — {model}, horizon h=4")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"prediction_curve_{model}_h4.png", dpi=180)
    plt.close(fig)

"""Frozen B0/B1/B2 outer-LOAO baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from chm_data import ANCHORS, FrozenCHMData, target_for_checkpoint


def anchor_retention_median(data: FrozenCHMData, checkpoint: int) -> float:
    return float(np.median(data.hou.loc[list(data.anchor_indices[checkpoint]), "P1S_operating_voltage_retention"].astype(float).to_numpy()))


def nonnegative_affine_prediction(x_train: np.ndarray, y_train: np.ndarray, x_value: float) -> float:
    centered_x = x_train - x_train.mean()
    denominator = float(np.dot(centered_x, centered_x))
    slope = 0.0 if denominator == 0 else float(np.dot(centered_x, y_train - y_train.mean()) / denominator)
    slope = max(0.0, slope)
    intercept = float(y_train.mean() - slope * x_train.mean())
    return float(slope * x_value + intercept)


def run_baseline_loao(data: FrozenCHMData) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    values = {checkpoint: anchor_retention_median(data, checkpoint) for checkpoint in ANCHORS}
    for heldout in ANCHORS:
        training = [checkpoint for checkpoint in ANCHORS if checkpoint != heldout]
        x_train = np.asarray([values[checkpoint] for checkpoint in training], dtype=float)
        y_train = np.asarray([target_for_checkpoint(data, checkpoint) for checkpoint in training], dtype=float)
        b0 = values[heldout]
        isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x_train, y_train)
        b1 = float(isotonic.predict([values[heldout]])[0])
        b2 = nonnegative_affine_prediction(x_train, y_train, values[heldout])
        target = target_for_checkpoint(data, heldout)
        rows.append({"checkpoint": heldout, "target": target, "B0_prediction": b0, "B1_prediction": b1, "B2_prediction": b2, "B0_absolute_error": abs(b0-target), "B1_absolute_error": abs(b1-target), "B2_absolute_error": abs(b2-target), "B0_squared_error": (b0-target)**2, "B1_squared_error": (b1-target)**2, "B2_squared_error": (b2-target)**2})
    return pd.DataFrame(rows)


def baseline_metrics(frame: pd.DataFrame, name: str) -> dict[str, float]:
    return {"MAE": float(frame[f"{name}_absolute_error"].mean()), "RMSE": float(np.sqrt(frame[f"{name}_squared_error"].mean()))}

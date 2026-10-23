"""Unchanged pure window/scaler/EWMA definitions extracted from C/E authority."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    targets: np.ndarray
    ages: np.ndarray

def build_windows(frame: pd.DataFrame, cfg: dict[str, Any], start: int, end: int) -> Windows:
    features = cfg["task"]["raw_f2_feature_order"]
    values, voltage = frame[features].to_numpy(float), frame["cell_voltage_mean"].to_numpy(float)
    valid, segment = frame["bin_valid_for_windows"].astype(bool).to_numpy(), frame["temporal_segment_id"].to_numpy()
    ages = frame["bin_end_ophrs2_h"].to_numpy(float)
    lookback, horizon = 48, 24
    xs: list[np.ndarray] = []; ys: list[np.ndarray] = []; origins: list[int] = []; targets: list[np.ndarray] = []; origin_ages: list[float] = []
    for origin in range(start, end - horizon):
        first, last = origin - lookback + 1, origin + horizon
        if first < 0 or last >= end: continue
        block = slice(first, last + 1)
        if not valid[block].all() or np.any(segment[block] != segment[origin]): continue
        if not np.allclose(np.diff(ages[block]), 0.5, atol=1e-6, rtol=0): continue
        if not np.isfinite(values[first:last + 1]).all() or not np.isfinite(voltage[first:last + 1]).all(): continue
        xs.append(values[first:origin + 1]); ys.append(voltage[origin + 1:last + 1] - voltage[origin]); origins.append(origin); targets.append(np.arange(origin + 1, last + 1)); origin_ages.append(ages[origin])
    if not xs: raise RuntimeError(f"NO_LEGAL_WINDOWS_{start}_{end}")
    return Windows(np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(origins), np.asarray(targets), np.asarray(origin_ages))

def ewma(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for t in range(1, x.shape[1]): out[:, t, 0] = 0.3 * x[:, t, 0] + 0.7 * out[:, t - 1, 0]
    return out

class CommonScaler:
    def fit(self, x: np.ndarray) -> "CommonScaler":
        flat = x.astype(np.float64).reshape(-1, x.shape[-1])
        self.mean = flat.mean(axis=0)
        self.scale = flat.std(axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        self.samples = int(flat.shape[0])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    @property
    def target_scale(self) -> float:
        return float(self.scale[0])

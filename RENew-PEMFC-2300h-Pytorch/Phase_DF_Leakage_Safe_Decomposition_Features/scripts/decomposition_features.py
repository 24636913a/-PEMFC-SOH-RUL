"""Origin-conditioned, 48-step-only feature transforms for Phase DF.

The public production functions accept a single voltage history of exactly 48
points.  They intentionally cannot receive a dataframe, a target, a suffix, or
an arbitrary full-series array.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyEMD import EMD
from vmdpy import VMD


CONTEXT = 48
RESOLUTION_H = 0.5


@dataclass(frozen=True)
class FeatureResult:
    features: np.ndarray
    audit: dict[str, Any]


def _history(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=np.float64)
    if values.shape != (CONTEXT,):
        raise ValueError(f"Origin-conditioned feature input must be exactly ({CONTEXT},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Origin-conditioned feature input contains non-finite voltage")
    return np.ascontiguousarray(values)


def input_voltage_sha256(history: np.ndarray) -> str:
    return hashlib.sha256(_history(history).tobytes()).hexdigest()


def _normalise(history: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = _history(history)
    mean = float(values.mean())
    scale = max(float(values.std()), 0.001)
    return (values - mean) / scale, mean, scale


def causal_slope8(values: np.ndarray) -> np.ndarray:
    """OLS slopes at 0.5 h intervals, with no later in-block position used."""
    values = _history(values)
    output = np.zeros(CONTEXT, dtype=np.float64)
    for j in range(1, CONTEXT):
        start = max(0, j - 7)
        y = values[start:j + 1]
        x = np.arange(len(y), dtype=np.float64) * RESOLUTION_H
        centered_x = x - x.mean()
        denominator = float(np.dot(centered_x, centered_x))
        output[j] = 0.0 if denominator == 0.0 else float(np.dot(centered_x, y - y.mean()) / denominator)
    return output


def simple_trend_features(history: np.ndarray) -> FeatureResult:
    values = _history(history)
    means = np.asarray([values[max(0, j - 7):j + 1].mean() for j in range(CONTEXT)], dtype=np.float64)
    slopes = causal_slope8(values)
    return FeatureResult(
        features=np.column_stack([means, slopes]).astype(np.float32),
        audit={"method": "SimpleTrend", "first_slope_V_per_h": float(slopes[0])},
    )


def vmd_feature(history: np.ndarray, *, alpha: int, tau: float = 0.0, K: int = 2, DC: int = 0, init: int = 1, tol: float = 1e-7) -> FeatureResult:
    if K != 2 or DC != 0 or tau != 0.0 or init != 1 or tol != 1e-7:
        raise ValueError("Frozen VMD configuration violation")
    z, mean, scale = _normalise(history)
    modes, _spectra, omega = VMD(z, alpha, tau, K, DC, init, tol)
    if modes.shape != (2, CONTEXT) or omega.ndim != 2 or omega.shape[1] != 2:
        raise RuntimeError("IMPLEMENTATION_FAIL: vmdpy did not return two 48-point modes")
    final_frequencies = np.asarray(omega[-1], dtype=np.float64)
    order = np.argsort(final_frequencies, kind="stable")
    low = mean + scale * np.asarray(modes[order[0]], dtype=np.float64)
    reconstruction = np.asarray(modes.sum(axis=0), dtype=np.float64)
    residual = z - reconstruction
    slopes = causal_slope8(low)
    return FeatureResult(
        features=np.column_stack([low, slopes]).astype(np.float32),
        audit={
            "method": "VMD", "alpha": int(alpha), "mu_V": mean, "scale_V": scale,
            "final_center_frequencies_cycles_per_sample": final_frequencies[order].tolist(),
            "frequency_separation_cycles_per_sample": float(final_frequencies[order[1]] - final_frequencies[order[0]]),
            "normalized_reconstruction_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "normalized_reconstruction_max_abs": float(np.max(np.abs(residual))),
            "first_slope_V_per_h": float(slopes[0]),
        },
    )


def _configured_emd() -> EMD:
    emd = EMD(spline_kind="cubic", nbsym=2, extrema_detection="simple")
    params = {
        "FIXE": 0, "FIXE_H": 0, "MAX_ITERATION": 1000, "std_thr": 0.2,
        "svar_thr": 0.001, "energy_ratio_thr": 0.2, "total_power_thr": 0.005,
        "range_thr": 0.001,
    }
    missing = [name for name in params if not hasattr(emd, name)]
    if missing or not hasattr(emd, "get_imfs_and_residue"):
        raise RuntimeError(f"IMPLEMENTATION_FAIL: EMD-signal API mismatch: {missing}")
    for name, value in params.items():
        setattr(emd, name, value)
    return emd


def emd_feature(history: np.ndarray) -> FeatureResult:
    z, mean, scale = _normalise(history)
    emd = _configured_emd()
    # Full decomposition is required so that ``remainder2`` can retain any IMF
    # beyond the first two as well as the residue. No target/suffix is supplied.
    emd.emd(z.astype(np.float64))
    imfs, residue = emd.get_imfs_and_residue()
    imfs = np.asarray(imfs, dtype=np.float64)
    residue = np.asarray(residue, dtype=np.float64)
    if imfs.size == 0:
        imfs = np.empty((0, CONTEXT), dtype=np.float64)
    if residue.shape != (CONTEXT,) or imfs.ndim != 2 or imfs.shape[1] != CONTEXT:
        raise RuntimeError("IMPLEMENTATION_FAIL: EMD IMF/residue output shape changed")
    removed = imfs[: min(2, len(imfs))].sum(axis=0) if len(imfs) else np.zeros(CONTEXT, dtype=np.float64)
    remainder = z - removed
    reconstructed = imfs.sum(axis=0) + residue if len(imfs) else residue
    reconstruction_error = z - reconstructed
    low = mean + scale * remainder
    slopes = causal_slope8(low)
    energies = (np.mean(imfs ** 2, axis=1).tolist() if len(imfs) else [])
    return FeatureResult(
        features=np.column_stack([low, slopes]).astype(np.float32),
        audit={
            "method": "EMD", "mu_V": mean, "scale_V": scale, "imf_count": int(len(imfs)),
            "imf_energy_normalized": energies,
            "normalized_reconstruction_rmse": float(np.sqrt(np.mean(reconstruction_error ** 2))),
            "normalized_reconstruction_max_abs": float(np.max(np.abs(reconstruction_error))),
            "first_slope_V_per_h": float(slopes[0]),
        },
    )


def method_feature(method: str, history: np.ndarray, *, vmd_alpha: int | None = None) -> FeatureResult:
    """The sole production entry point; it has no target/full-series argument."""
    if method == "SimpleTrend":
        return simple_trend_features(history)
    if method == "VMD":
        if vmd_alpha is None:
            raise ValueError("Resolved VMD alpha is required")
        return vmd_feature(history, alpha=int(vmd_alpha))
    if method == "EMD":
        return emd_feature(history)
    raise ValueError(f"Unregistered Phase DF method: {method}")


def source_api_snapshot() -> dict[str, str]:
    return {
        "vmd_import_path": "vmdpy.VMD",
        "vmd_signature": str(inspect.signature(VMD)),
        "emd_import_path": "PyEMD.EMD",
        "emd_signature": str(inspect.signature(EMD)),
        "emd_class_module": EMD.__module__,
    }

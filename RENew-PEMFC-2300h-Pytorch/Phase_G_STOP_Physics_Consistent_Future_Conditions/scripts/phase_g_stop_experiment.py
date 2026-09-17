#!/usr/bin/env python
"""G2/G3 experiment for physics-consistent future operating conditions.

This script is development-only. It loads the frozen N2.8F M0 checkpoints and
changes only their future-condition input. No SA-TimeXer parameter is trained.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import precision_recall_fscore_support, r2_score
from sklearn.preprocessing import StandardScaler


PHASE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PHASE_ROOT / "configs" / "phase_g_stop_frozen_config.yaml"
FOLDS = ("fold_1", "fold_2", "fold_3")
EPS = 1e-7


def out(relative: str) -> Path:
    path = (PHASE_ROOT / relative).resolve()
    if PHASE_ROOT.resolve() not in path.parents:
        raise RuntimeError(f"Output escaped Phase G-STOP: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    frame.to_csv(out(relative), index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(obj: Any, relative: str) -> None:
    out(relative).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if Path(cfg["phase_root"]).resolve() != PHASE_ROOT.resolve():
        raise RuntimeError("Frozen phase root mismatch")
    if cfg.get("test_performance_accessed") is not False:
        raise RuntimeError("Test firewall is not false")
    return cfg


def import_n28f(root: Path):
    path = root / "scripts" / "phase_n2_8f_run.py"
    spec = importlib.util.spec_from_file_location("phase_g_stop_n28f", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen N2.8F implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def future_aux(frame: pd.DataFrame, windows, columns: list[str], horizon: int = 24) -> np.ndarray:
    rows = []
    for origin in windows.row_origin_index.astype(int):
        block = frame.iloc[origin + 1 : origin + 1 + horizon].loc[:, columns].to_numpy(float)
        if block.shape != (horizon, len(columns)):
            raise RuntimeError("Incomplete auxiliary future window")
        rows.append(block)
    return np.asarray(rows, dtype=float)


class TwoPartQForecaster:
    """Fixed binary occurrence plus conditional-q Ridge model, horizon by horizon."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.x_scaler = StandardScaler()
        self.classifiers: list[Any] = []
        self.conditional: list[Any] = []

    def fit(self, history: np.ndarray, future_q: np.ndarray):
        x = np.nan_to_num(history.reshape(len(history), -1), nan=0.0)
        xz = self.x_scaler.fit_transform(x)
        self.classifiers = []
        self.conditional = []
        for step in range(future_q.shape[1]):
            target = future_q[:, step]
            event = target > 1e-12
            if np.unique(event).size == 2:
                clf = LogisticRegression(
                    C=float(self.cfg["forecast"]["logistic_C"]),
                    class_weight=self.cfg["forecast"]["logistic_class_weight"],
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ).fit(xz, event.astype(int))
            else:
                clf = {"constant_probability": float(event.mean())}
            positive = np.flatnonzero(event)
            if len(positive) >= 10:
                reg = Ridge(alpha=float(self.cfg["forecast"]["conditional_q_ridge_alpha"])).fit(xz[positive], target[positive])
            else:
                reg = {"constant": float(np.median(target[positive])) if len(positive) else 0.0}
            self.classifiers.append(clf)
            self.conditional.append(reg)
        return self

    def predict(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.nan_to_num(history.reshape(len(history), -1), nan=0.0)
        xz = self.x_scaler.transform(x)
        q = np.zeros((len(history), 24), dtype=float)
        probability = np.zeros_like(q)
        threshold = float(self.cfg["forecast"]["logistic_threshold"])
        for step, (clf, reg) in enumerate(zip(self.classifiers, self.conditional)):
            if isinstance(clf, dict):
                prob = np.full(len(history), clf["constant_probability"], dtype=float)
            else:
                prob = clf.predict_proba(xz)[:, 1]
            if isinstance(reg, dict):
                conditional = np.full(len(history), reg["constant"], dtype=float)
            else:
                conditional = reg.predict(xz)
            conditional = np.clip(conditional, 0.0, 1.0)
            event = prob >= threshold
            q[:, step] = np.where(event, conditional, 0.0)
            probability[:, step] = prob
        return q, probability


class AuxiliaryRidge:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, history: np.ndarray, future: np.ndarray):
        x = np.nan_to_num(history.reshape(len(history), -1), nan=0.0)
        y = np.nan_to_num(future.reshape(len(future), -1), nan=0.0)
        self.model.fit(self.x_scaler.fit_transform(x), self.y_scaler.fit_transform(y))
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(history.reshape(len(history), -1), nan=0.0)
        y = self.y_scaler.inverse_transform(self.model.predict(self.x_scaler.transform(x)))
        return y.reshape(len(history), 24, 2)


def project_future(values: np.ndarray, area: float) -> tuple[np.ndarray, dict[str, float]]:
    before = values.copy()
    arr = values.copy()
    arr[..., 0:3] = np.maximum(arr[..., 0:3], 0.0)
    arr[..., 3] = arr[..., 0] / area
    arr[..., 4] = arr[..., 1] / area
    arr[..., 5] = arr[..., 2] / area
    arr[..., 6:8] = np.maximum(arr[..., 6:8], 0.0)
    arr[..., 8] = np.clip(arr[..., 8], 0.0, 1.0)
    full_zero = arr[..., 8] >= 1.0 - 1e-12
    arr[..., 0:6][full_zero] = 0.0
    change = np.abs(arr - before)
    altered = change > 1e-10
    audit = {
        "modified_trajectory_fraction": float(np.mean(np.any(altered, axis=(1, 2)))),
        "modified_element_fraction": float(np.mean(altered)),
        "mean_absolute_change_all_variables": float(np.mean(change)),
        "max_absolute_change_all_variables": float(np.max(change)),
    }
    return arr, audit


def build_p2(p: dict[str, Any], aux_train: np.ndarray, aux_valid: np.ndarray, cfg: dict[str, Any]):
    q_model = TwoPartQForecaster(cfg).fit(p["htr"], p["ztr"][:, :, 8])
    q_pred, q_probability = q_model.predict(p["hva"])
    aux_model = AuxiliaryRidge(float(cfg["forecast"]["conditional_on_stats_ridge_alpha"])).fit(p["htr"], aux_train)
    on_pred = aux_model.predict(p["hva"])
    mu_on = np.maximum(on_pred[:, :, 0], 0.0)
    sigma_on = np.maximum(on_pred[:, :, 1], 0.0)
    candidate = p["z_ridge"].copy()
    candidate[:, :, 8] = q_pred
    candidate[:, :, 0] = (1.0 - q_pred) * mu_on
    candidate[:, :, 1] = np.sqrt(np.maximum(0.0, (1.0 - q_pred) * sigma_on**2 + q_pred * (1.0 - q_pred) * mu_on**2))
    # Peak-to-peak is not determined by q, mean, and variance; preserve its independent Ridge prediction.
    projected, audit = project_future(candidate, p["area"])
    return projected, q_probability, audit


def causal_p2_oof(p: dict[str, Any], aux_train: np.ndarray, cfg: dict[str, Any], n28f, fold: str):
    n = len(p["htr"])
    start = n28f.first_index_with_completed(
        p["train"].target_end_age_h,
        p["train"].origin_age_h,
        int(cfg["forecast"]["oof_min_completed_windows"]),
    )
    q_pred = np.full((n, 24), np.nan, dtype=float)
    aux_pred = np.full((n, 24, 2), np.nan, dtype=float)
    audits = []
    for block, (left, right) in enumerate(n28f.block_edges(start, n, int(cfg["forecast"]["oof_blocks"])), start=1):
        min_origin = float(p["train"].origin_age_h[left])
        fit = n28f.completed_indices(p["train"].target_end_age_h, min_origin)
        if len(fit) < int(cfg["forecast"]["oof_min_completed_windows"]):
            raise RuntimeError("P2 OOF warm-up violated")
        q_model = TwoPartQForecaster(cfg).fit(p["htr"][fit], p["ztr"][fit, :, 8])
        q_pred[left:right], _ = q_model.predict(p["htr"][left:right])
        aux_model = AuxiliaryRidge(float(cfg["forecast"]["conditional_on_stats_ridge_alpha"])).fit(p["htr"][fit], aux_train[fit])
        aux_pred[left:right] = aux_model.predict(p["htr"][left:right])
        max_target = float(p["train"].target_end_age_h[fit].max())
        violation = max_target > min_origin + EPS
        if violation:
            raise RuntimeError("P2 OOF target-completion leakage")
        audits.append(
            {
                "stage": "P2_two_part_oof",
                "fold": fold,
                "block": block,
                "fit_count": int(len(fit)),
                "predict_count": int(right - left),
                "max_fit_target_end_h": max_target,
                "min_prediction_origin_h": min_origin,
                "target_completion_pass": True,
                "train_only_scaler": True,
            }
        )
    keep = np.isfinite(q_pred).all(axis=1) & np.isfinite(aux_pred).all(axis=(1, 2))
    return q_pred, aux_pred, keep, audits


def voltage_metrics(y_true: np.ndarray, y_pred: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    target = y_true - anchor[:, None]
    pred = y_pred - anchor[:, None]
    error = pred - target
    ae = np.abs(error).ravel() * 1000.0
    return {
        "r2_delta": float(r2_score(target.ravel(), pred.ravel())),
        "rmse_delta_mV": float(np.sqrt(np.mean(error**2)) * 1000.0),
        "mae_delta_mV": float(np.mean(np.abs(error)) * 1000.0),
        "p95_mV": float(np.quantile(ae, 0.95)),
        "p99_mV": float(np.quantile(ae, 0.99)),
    }


def future_metrics(model: str, fold: str, pred: np.ndarray, truth: np.ndarray, q_probability: np.ndarray | None = None) -> dict[str, Any]:
    q_true = truth[:, :, 8]
    q_pred = pred[:, :, 8]
    true_event = q_true > 1e-12
    if q_probability is None:
        pred_event = q_pred > 1e-12
    else:
        pred_event = q_probability >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(true_event.ravel(), pred_event.ravel(), average="binary", zero_division=0)
    return {
        "model": model,
        "fold": fold,
        "q_mae": float(np.mean(np.abs(q_pred - q_true))),
        "q_rmse": float(np.sqrt(np.mean((q_pred - q_true) ** 2))),
        "mixed_event_precision": float(precision),
        "mixed_event_recall": float(recall),
        "mixed_event_f1": float(f1),
        "q1_precision": "NOT_ESTIMABLE",
        "q1_recall": "NOT_ESTIMABLE",
        "q1_f1": "NOT_ESTIMABLE",
        "current_mean_mae_A": float(np.mean(np.abs(pred[:, :, 0] - truth[:, :, 0]))),
        "current_std_mae_A": float(np.mean(np.abs(pred[:, :, 1] - truth[:, :, 1]))),
        "current_peak_to_peak_mae_A": float(np.mean(np.abs(pred[:, :, 2] - truth[:, :, 2]))),
        "density_mean_mae_A_cm2": float(np.mean(np.abs(pred[:, :, 3] - truth[:, :, 3]))),
        "density_std_mae_A_cm2": float(np.mean(np.abs(pred[:, :, 4] - truth[:, :, 4]))),
        "density_peak_to_peak_mae_A_cm2": float(np.mean(np.abs(pred[:, :, 5] - truth[:, :, 5]))),
    }


def violation_metrics(model: str, fold: str, values: np.ndarray, area: float, tolerance: float) -> dict[str, Any]:
    negative = values[:, :, 0:8] < -1e-12
    q_range = (values[:, :, 8] < -1e-12) | (values[:, :, 8] > 1.0 + 1e-12)
    relative = []
    for current_idx, density_idx in ((0, 3), (1, 4), (2, 5)):
        error = np.abs(values[:, :, current_idx] - area * values[:, :, density_idx])
        relative.append(error / np.maximum(np.abs(values[:, :, current_idx]), 1.0) > tolerance)
    identity = np.stack(relative, axis=2)
    q1 = values[:, :, 8] >= 1.0 - 1e-12
    q1_incompatible = q1 & np.any(np.abs(values[:, :, 0:6]) > 1e-8, axis=2)
    trajectory_violation = np.any(negative, axis=(1, 2)) | np.any(q_range, axis=1) | np.any(identity, axis=(1, 2)) | np.any(q1_incompatible, axis=1)
    return {
        "model": model,
        "fold": fold,
        "negative_stat_element_rate": float(np.mean(negative)),
        "q_range_element_rate": float(np.mean(q_range)),
        "current_density_identity_element_rate": float(np.mean(identity)),
        "q1_incompatible_element_rate": float(np.mean(q1_incompatible)),
        "physical_violation_trajectory_rate": float(np.mean(trajectory_violation)),
    }


def transition_metrics(model: str, fold: str, valid, pred: np.ndarray, truth: np.ndarray, probability: np.ndarray | None) -> dict[str, Any]:
    # One h=1 prediction per unique target bin avoids counting overlapping horizons as independent events.
    true_state = truth[:, 0, 8] > 1e-12
    pred_state = (probability[:, 0] >= 0.5) if probability is not None else (pred[:, 0, 8] > 1e-12)
    target_rows = valid.row_origin_index.astype(int) + 1
    contiguous = (target_rows[1:] == target_rows[:-1] + 1)
    true_transition = contiguous & (true_state[1:] != true_state[:-1])
    pred_transition = contiguous & (pred_state[1:] != pred_state[:-1])
    tp = int(np.sum(true_transition & pred_transition))
    fp = int(np.sum(~true_transition & pred_transition))
    fn = int(np.sum(true_transition & ~pred_transition))
    return {
        "model": model,
        "fold": fold,
        "unique_target_bins": int(len(target_rows)),
        "true_transition_events": int(true_transition.sum()),
        "predicted_transition_events": int(pred_transition.sum()),
        "transition_precision": tp / (tp + fp) if tp + fp else "NOT_ESTIMABLE",
        "transition_recall": tp / (tp + fn) if tp + fn else "NOT_ESTIMABLE",
    }


def inference(n28f, model, p: dict[str, Any], z_raw: np.ndarray) -> np.ndarray:
    scaled, _ = n28f.predict_scaled(model, p["xva"], p["z_scaler"].transform(z_raw))
    return p["valid"].anchor[:, None] + scaled * p["ystd"] + p["ymean"]


def pooled_tail(arrays: dict[str, dict[str, np.ndarray]], model: str) -> dict[str, Any]:
    ae = np.concatenate([np.abs(item["pred"] - item["truth"]).ravel() * 1000.0 for item in arrays.values()])
    return {
        "model": model,
        "aggregation": "pooled_prediction_points_across_three_folds",
        "n_prediction_points": int(len(ae)),
        "p95_mV": float(np.quantile(ae, 0.95)),
        "p99_mV": float(np.quantile(ae, 0.99)),
        "max_mV": float(np.max(ae)),
    }


def moving_block_bootstrap(candidate: dict[str, dict[str, np.ndarray]], reference: dict[str, dict[str, np.ndarray]], cfg: dict[str, Any], name: str) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg["bootstrap"]["seed"]))
    repetitions = int(cfg["bootstrap"]["repetitions"])
    block = int(cfg["bootstrap"]["block_length_origins"])
    deltas = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sse_c = sse_r = 0.0
        count = 0
        for fold in FOLDS:
            truth = candidate[fold]["truth"]
            pc = candidate[fold]["pred"]
            pr = reference[fold]["pred"]
            n = len(truth)
            blocks = math.ceil(n / block)
            starts = rng.integers(0, n, size=blocks)
            index = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
            ec = pc[index] - truth[index]
            er = pr[index] - truth[index]
            sse_c += float(np.sum(ec**2)); sse_r += float(np.sum(er**2)); count += int(ec.size)
        deltas[repetition] = (math.sqrt(sse_c / count) - math.sqrt(sse_r / count)) * 1000.0
    return {
        "comparison": name,
        "delta_definition": "RMSE_candidate_minus_P0_mV",
        "block_length_origins": block,
        "repetitions": repetitions,
        "mean_delta_rmse_mV": float(deltas.mean()),
        "median_delta_rmse_mV": float(np.median(deltas)),
        "ci95_lower_mV": float(np.quantile(deltas, 0.025)),
        "ci95_upper_mV": float(np.quantile(deltas, 0.975)),
        "probability_candidate_better": float(np.mean(deltas < 0.0)),
    }


def regime_rows(frame: pd.DataFrame, valid, predictions: dict[str, np.ndarray], fold: str) -> list[dict[str, Any]]:
    q = frame["zero_current_fraction"].to_numpy(float)
    segment = frame["temporal_segment_id"].to_numpy()
    records = []
    for model, pred in predictions.items():
        truth = valid.y_abs
        for category in ("strict_q_state", "transition_state"):
            labels = []
            row_ids = []
            errors = []
            for origin_index, origin_row in enumerate(valid.row_origin_index.astype(int)):
                for step in range(24):
                    row = origin_row + 1 + step
                    previous = row - 1
                    current_q = q[row]
                    previous_q = q[previous] if previous >= 0 and segment[previous] == segment[row] else math.nan
                    if category == "strict_q_state":
                        label = "q=0" if abs(current_q) <= 1e-12 else ("q=1" if abs(current_q - 1.0) <= 1e-12 else "0<q<1")
                    else:
                        if not np.isfinite(previous_q):
                            label = "segment_boundary"
                        elif previous_q <= 1e-12 and current_q <= 1e-12:
                            label = "continuous_nonzero"
                        elif previous_q <= 1e-12 and 0.0 < current_q < 1.0:
                            label = "nonzero_to_mixed"
                        elif 0.0 < previous_q < 1.0 and current_q <= 1e-12:
                            label = "mixed_to_nonzero"
                        elif 0.0 < previous_q < 1.0 and 0.0 < current_q < 1.0:
                            label = "continuous_mixed"
                        elif abs(previous_q - 1.0) <= 1e-12 and abs(current_q - 1.0) <= 1e-12:
                            label = "continuous_zero"
                        elif previous_q <= 1e-12 and abs(current_q - 1.0) <= 1e-12:
                            label = "nonzero_to_zero"
                        elif abs(previous_q - 1.0) <= 1e-12 and current_q <= 1e-12:
                            label = "zero_to_nonzero"
                        else:
                            label = "other"
                    labels.append(label); row_ids.append(row); errors.append((pred[origin_index, step] - truth[origin_index, step]) * 1000.0)
            work = pd.DataFrame({"regime": labels, "target_row": row_ids, "error_mV": errors})
            for label, group in work.groupby("regime", sort=True):
                absolute = np.abs(group["error_mV"].to_numpy(float))
                records.append(
                    {
                        "model": model,
                        "fold": fold,
                        "category": category,
                        "regime": label,
                        "sample_count": int(len(group)),
                        "unique_target_bin_count": int(group["target_row"].nunique()),
                        "event_count": int(group["target_row"].drop_duplicates().shape[0]),
                        "rmse_delta_mV": float(np.sqrt(np.mean(group["error_mV"].to_numpy(float) ** 2))),
                        "mae_delta_mV": float(np.mean(absolute)),
                        "p95_mV": float(np.quantile(absolute, 0.95)),
                        "p99_mV": float(np.quantile(absolute, 0.99)),
                    }
                )
    return records


def main() -> None:
    cfg = load_config()
    raw_summary = json.loads((PHASE_ROOT / "outputs" / "physical_audit" / "g1_raw_audit_summary.json").read_text(encoding="utf-8"))
    if not raw_summary["physical_relation_status"]["overall_g1_physical_pass"]:
        raise RuntimeError("G1_INCONCLUSIVE: physical relation audit did not pass")
    if not raw_summary["event_sufficiency"]["sufficient_for_mixed_current_analysis"]:
        raise RuntimeError("G1_INCONCLUSIVE: mixed-current events are insufficient")

    n28f = import_n28f(Path(cfg["n2_8f_root"]))
    ncfg = n28f.load_config()
    common, frame, windows, firewall = n28f.load_development(ncfg)
    if frame["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL: Test entered G2/G3")
    raw_aux = pd.read_csv(PHASE_ROOT / "outputs" / "physical_audit" / "raw_conditional_current_statistics.csv")
    frame = frame.merge(raw_aux[["bin_start_ophrs2_h", "mu_on_A", "sigma_on_A"]], on="bin_start_ophrs2_h", how="left", validate="one_to_one")
    if frame[["mu_on_A", "sigma_on_A"]].isna().any().any():
        raise RuntimeError("Missing raw conditional statistics in Development")
    packages = {fold: n28f.package_fold(common, frame, windows, ncfg, fold) for fold in FOLDS}

    future_rows = []
    violation_rows = []
    projection_rows = []
    transition_rows = []
    oof_rows = []
    fold_rows = []
    horizon_rows = []
    oracle_rows = []
    regime_error_rows = []
    model_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {name: {} for name in ("P0_Original_Ridge", "P1_Physics_Projected_Ridge", "P2_Zero_Current_Aware")}
    forecast_by_fold: dict[str, dict[str, np.ndarray]] = {}
    p2_probability_by_fold: dict[str, np.ndarray] = {}

    for fold in FOLDS:
        p = packages[fold]
        aux_train = future_aux(frame, p["train"], ["mu_on_A", "sigma_on_A"])
        aux_valid = future_aux(frame, p["valid"], ["mu_on_A", "sigma_on_A"])
        q_oof, aux_oof, p2_keep, causal_audit = causal_p2_oof(p, aux_train, cfg, n28f, fold)
        if not np.array_equal(p2_keep, p["ridge_keep"]):
            raise RuntimeError(f"{fold}: P2 OOF legal-origin set differs from P0")
        oof_rows.extend(causal_audit)

        p0 = p["z_ridge"].copy()
        p1, p1_audit = project_future(p0, p["area"])
        p2, p2_probability, p2_audit = build_p2(p, aux_train, aux_valid, cfg)
        forecasts = {"P0_Original_Ridge": p0, "P1_Physics_Projected_Ridge": p1, "P2_Zero_Current_Aware": p2}
        forecast_by_fold[fold] = forecasts
        p2_probability_by_fold[fold] = p2_probability
        for label, audit in (("P1_Physics_Projected_Ridge", p1_audit), ("P2_Zero_Current_Aware", p2_audit)):
            row = {"model": label, "fold": fold, **audit}
            for index, name in enumerate(ncfg["future_exogenous_features"]):
                difference = np.abs(forecasts[label][:, :, index] - p0[:, :, index])
                row[f"{name}_mean_abs_change"] = float(np.mean(difference))
                row[f"{name}_max_abs_change"] = float(np.max(difference))
            projection_rows.append(row)

        checkpoint = Path(cfg["n2_8f_root"]) / "checkpoints" / f"M0_SA_TimeXer_Ridge_{fold}_seed42.pt"
        model = n28f.StepAlignedTimeXer(ncfg, caf=False)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(n28f.device()).eval()
        predictions = {}
        for label, z_values in forecasts.items():
            pred = inference(n28f, model, p, z_values)
            predictions[label] = pred
            metrics = voltage_metrics(p["valid"].y_abs, pred, p["valid"].anchor)
            fold_rows.append({"model": label, "fold": fold, **metrics})
            model_arrays[label][fold] = {"truth": p["valid"].y_abs.astype(float), "pred": pred.astype(float), "anchor": p["valid"].anchor.astype(float)}
            np.savez_compressed(
                out(f"outputs/frozen_timexer/{label}_{fold}_predictions.npz"),
                y_true=p["valid"].y_abs.astype(float), y_pred=pred.astype(float),
                anchor=p["valid"].anchor.astype(float), origin_age_h=p["valid"].origin_age_h.astype(float),
                target_end_age_h=p["valid"].target_end_age_h.astype(float), row_origin_index=p["valid"].row_origin_index.astype(int),
                future_conditions=z_values.astype(float),
            )
            target_delta = p["valid"].y_abs - p["valid"].anchor[:, None]
            pred_delta = pred - p["valid"].anchor[:, None]
            for step in range(24):
                error = pred_delta[:, step] - target_delta[:, step]
                horizon_rows.append(
                    {"model": label, "fold": fold, "horizon_step": step + 1, "horizon_hours": (step + 1) * 0.5,
                     "r2_delta": float(r2_score(target_delta[:, step], pred_delta[:, step])),
                     "rmse_delta_mV": float(np.sqrt(np.mean(error**2)) * 1000.0),
                     "mae_delta_mV": float(np.mean(np.abs(error)) * 1000.0)}
                )
        regime_error_rows.extend(regime_rows(frame, p["valid"], predictions, fold))

        for label, values in forecasts.items():
            probability = p2_probability if label == "P2_Zero_Current_Aware" else None
            future_rows.append(future_metrics(label, fold, values, p["zva"], probability))
            violation_rows.append(violation_metrics(label, fold, values, p["area"], float(cfg["g1"]["area_identity_median_relative_tolerance"])))
            transition_rows.append(transition_metrics(label, fold, p["valid"], values, p["zva"], probability))

        oracle_variants = {
            "P0_Ridge_formal": p0,
            "true_q_only_DIAGNOSTIC": np.concatenate([p0[:, :, :8], p["zva"][:, :, 8:9]], axis=2),
            "true_physical_current_group_DIAGNOSTIC": p0.copy(),
            "all_true_future_conditions_DIAGNOSTIC": p["zva"].copy(),
        }
        oracle_variants["true_physical_current_group_DIAGNOSTIC"][:, :, 0:6] = p["zva"][:, :, 0:6]
        oracle_variants["true_physical_current_group_DIAGNOSTIC"][:, :, 8] = p["zva"][:, :, 8]
        for label, values in oracle_variants.items():
            pred = inference(n28f, model, p, values)
            oracle_rows.append({"variant": label, "fold": fold, "eligibility": "FORMAL" if label == "P0_Ridge_formal" else "DIAGNOSTIC_ONLY", **voltage_metrics(p["valid"].y_abs, pred, p["valid"].anchor)})

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for label, group in fold_metrics.groupby("model", sort=False):
        row = {"model": label}
        for metric in ("r2_delta", "rmse_delta_mV", "mae_delta_mV", "p95_mV", "p99_mV"):
            row[f"mean_{metric}"] = float(group[metric].mean())
        for fold in FOLDS:
            row[f"{fold}_rmse_delta_mV"] = float(group.loc[group["fold"] == fold, "rmse_delta_mV"].iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    p0_rmse = float(summary.loc[summary["model"] == "P0_Original_Ridge", "mean_rmse_delta_mV"].iloc[0])
    if abs(p0_rmse - 24.538245) > 0.001:
        raise RuntimeError(f"P0 reproduction failed: {p0_rmse:.6f} mV")

    future_frame = pd.DataFrame(future_rows)
    future_mean = future_frame.groupby("model", as_index=False).agg(
        q_mae=("q_mae", "mean"), q_rmse=("q_rmse", "mean"),
        mixed_event_precision=("mixed_event_precision", "mean"), mixed_event_recall=("mixed_event_recall", "mean"), mixed_event_f1=("mixed_event_f1", "mean"),
        current_mean_mae_A=("current_mean_mae_A", "mean"), current_std_mae_A=("current_std_mae_A", "mean"),
        current_peak_to_peak_mae_A=("current_peak_to_peak_mae_A", "mean"),
    )
    future_out = pd.concat([future_frame, future_mean.assign(fold="mean_3fold")], ignore_index=True, sort=False)

    violation = pd.DataFrame(violation_rows)
    violation_mean = violation.groupby("model", as_index=False).mean(numeric_only=True).assign(fold="mean_3fold")
    violation_out = pd.concat([violation, violation_mean], ignore_index=True, sort=False)
    tails = pd.DataFrame([pooled_tail(model_arrays[label], label) for label in model_arrays])
    horizon = pd.DataFrame(horizon_rows)
    horizon_mean = horizon.groupby(["model", "horizon_step", "horizon_hours"], as_index=False).mean(numeric_only=True).assign(fold="mean_3fold")
    horizon_out = pd.concat([horizon, horizon_mean], ignore_index=True, sort=False)
    bootstrap = pd.DataFrame([
        moving_block_bootstrap(model_arrays["P1_Physics_Projected_Ridge"], model_arrays["P0_Original_Ridge"], cfg, "P1_vs_P0"),
        moving_block_bootstrap(model_arrays["P2_Zero_Current_Aware"], model_arrays["P0_Original_Ridge"], cfg, "P2_vs_P0"),
    ])

    violation_lookup = violation_mean.set_index("model")["physical_violation_trajectory_rate"].to_dict()
    tail_lookup = tails.set_index("model")["p99_mV"].to_dict()
    h95 = horizon_mean.loc[np.isclose(horizon_mean["horizon_hours"], 9.5)].set_index("model")["rmse_delta_mV"].to_dict()
    gate_rows = []
    for candidate in ("P1_Physics_Projected_Ridge", "P2_Zero_Current_Aware"):
        c = summary.loc[summary["model"] == candidate].iloc[0]
        base = summary.loc[summary["model"] == "P0_Original_Ridge"].iloc[0]
        improvement = (float(base["mean_rmse_delta_mV"]) - float(c["mean_rmse_delta_mV"])) / float(base["mean_rmse_delta_mV"]) * 100.0
        improved_folds = sum(float(c[f"{fold}_rmse_delta_mV"]) < float(base[f"{fold}_rmse_delta_mV"]) for fold in FOLDS)
        p99_change = (tail_lookup[candidate] - tail_lookup["P0_Original_Ridge"]) / tail_lookup["P0_Original_Ridge"] * 100.0
        h95_change = (h95[candidate] - h95["P0_Original_Ridge"]) / h95["P0_Original_Ridge"] * 100.0
        lower_violation = violation_lookup[candidate] < violation_lookup["P0_Original_Ridge"]
        passed = (
            improvement >= float(cfg["gate_b"]["minimum_mean_rmse_improvement_pct"])
            and improved_folds >= int(cfg["gate_b"]["minimum_improved_folds"])
            and p99_change <= float(cfg["gate_b"]["maximum_p99_worsening_pct"])
            and h95_change <= float(cfg["gate_b"]["maximum_9p5h_worsening_pct"])
            and lower_violation
        )
        gate_rows.append({
            "candidate": candidate, "mean_rmse_improvement_pct": improvement, "improved_fold_count": improved_folds,
            "p99_change_pct": p99_change, "horizon_9p5h_change_pct": h95_change,
            "physical_violation_rate_lower": bool(lower_violation), "gate_b_passed": bool(passed),
        })
    gate = pd.DataFrame(gate_rows)
    winners = gate.loc[gate["gate_b_passed"]]
    winner = None if winners.empty else str(winners.sort_values("mean_rmse_improvement_pct", ascending=False).iloc[0]["candidate"])

    write_csv(future_out, "outputs/future_forecast/future_condition_forecast_metrics.csv")
    write_csv(violation_out, "outputs/future_forecast/physical_violation_rates.csv")
    write_csv(pd.DataFrame(projection_rows), "outputs/future_forecast/projection_change_audit.csv")
    write_csv(pd.DataFrame(transition_rows), "outputs/zero_current/transition_event_metrics.csv")
    write_csv(pd.DataFrame(oof_rows), "outputs/future_forecast/p2_causal_oof_audit.csv")
    write_csv(fold_metrics, "outputs/frozen_timexer/frozen_timexer_fold_metrics.csv")
    write_csv(summary, "outputs/frozen_timexer/frozen_timexer_summary.csv")
    write_csv(horizon_out, "outputs/horizon/frozen_timexer_per_horizon.csv")
    write_csv(tails, "outputs/tail/frozen_timexer_tail_metrics.csv")
    write_csv(bootstrap, "outputs/frozen_timexer/paired_block_bootstrap.csv")
    write_csv(pd.DataFrame(oracle_rows), "outputs/ablation/oracle_replacement_audit.csv")
    write_csv(pd.DataFrame(regime_error_rows), "outputs/zero_current/regime_voltage_errors.csv")
    write_csv(gate, "outputs/frozen_timexer/gate_b_audit.csv")

    result = {
        "g1_passed": True,
        "p0_reproduction_rmse_mV": p0_rmse,
        "gate_b_passed": bool(not winners.empty),
        "gate_b_winner": winner,
        "phase_stop_status": "GATE_B_PASSED_REQUIRES_G4" if winner else "G_STOP_NO_DOWNSTREAM_GAIN",
        "timexer_retraining": "PENDING_G4" if winner else "NOT_RUN_GATE_B_FAILED",
        "lstm_transfer": "NOT_RUN_G4_NOT_COMPLETE" if winner else "NOT_RUN_GATE_B_FAILED",
        "test_performance_accessed": False,
    }
    write_json(result, "outputs/frozen_timexer/g3_result.json")
    write_json(
        {
            "target_rows_read_from_test": 0,
            "test_predictions_constructed": False,
            "test_metrics_computed": False,
            "test_feature_diagnostics_computed": False,
            "test_performance_accessed": False,
        },
        "outputs/test_firewall_audit.json",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

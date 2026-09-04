"""Development-only FC1/FC2 horizon-scale diagnosis for causal residual MIMO."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from data_pipeline import PipelineConfig, WindowSplit, aggregate_to_hourly, load_ageing_data
from evaluate import linear_trend_forecast, persistence_forecast
from models import MIMOLSTM
from train import TrainConfig, _fit_model, _predict, set_seed


HORIZONS = (4, 8, 12, 24)
SEED = 42


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_end_fraction: float
    validation_end_fraction: float


FOLD_SPECS = (
    FoldSpec("fold_1", 0.30, 0.40),
    FoldSpec("fold_2", 0.40, 0.50),
    FoldSpec("fold_3", 0.50, 0.60),
)


@dataclass
class HorizonFoldData:
    fc: str
    config: PipelineConfig
    fold: FoldSpec
    total_hourly_rows: int
    development_end_row: int
    hourly_development: pd.DataFrame
    split_rows: Dict[str, tuple[int, int]]
    splits: Dict[str, WindowSplit]
    inputs_raw: Dict[str, np.ndarray]
    anchors: Dict[str, np.ndarray]
    residual_labels: Dict[str, np.ndarray]
    X_scaler: StandardScaler
    residual_scaler: StandardScaler

    def transformed_inputs(self, split: str) -> np.ndarray:
        values = self.inputs_raw[split]
        samples, history, features = values.shape
        return self.X_scaler.transform(values.reshape(samples * history, features)).reshape(values.shape).astype(np.float32)

    def transformed_residuals(self, split: str) -> np.ndarray:
        values = self.residual_labels[split]
        return self.residual_scaler.transform(values.reshape(-1, 1)).reshape(values.shape).astype(np.float32)

    def restore_prediction(self, scaled: np.ndarray, split: str) -> np.ndarray:
        residual = self.residual_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)
        return (residual + self.anchors[split]).astype(np.float32)


def _windows_for_bounds(values: np.ndarray, target: np.ndarray, times: np.ndarray, start: int, end: int, config: PipelineConfig) -> WindowSplit:
    features, labels, origins, feature_times, target_times = [], [], [], [], []
    for origin in range(start + config.history_hours - 1, end - config.horizon_hours):
        history_index = np.arange(origin - config.history_hours + 1, origin + 1)
        target_index = np.arange(origin + 1, origin + config.horizon_hours + 1)
        if history_index.min() < start or target_index.max() >= end:
            raise AssertionError("Phase-3B window crossed a fold boundary")
        features.append(values[history_index])
        labels.append(target[target_index])
        origins.append(origin)
        feature_times.append(times[history_index])
        target_times.append(times[target_index])
    if not features:
        raise ValueError("No complete Phase-3B window for this N/H fold")
    return WindowSplit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), np.asarray(origins, dtype=np.int64), np.asarray(feature_times, dtype=np.float64), np.asarray(target_times, dtype=np.float64))


def build_horizon_fold(fc: str, fold: FoldSpec, horizon_hours: int, data_root: Path | None = None) -> HorizonFoldData:
    if fc not in {"FC1", "FC2"}:
        raise ValueError("FC must be FC1 or FC2")
    if horizon_hours not in HORIZONS:
        raise ValueError(f"Horizon must be one of {HORIZONS}")
    config = PipelineConfig(history_hours=15, horizon_hours=horizon_hours)
    raw = load_ageing_data(fc, data_root)
    hourly, _ = aggregate_to_hourly(raw)
    total_rows = len(hourly)
    development_end = int(total_rows * 0.60)
    train_end = int(total_rows * fold.train_end_fraction)
    validation_end = int(total_rows * fold.validation_end_fraction)
    if not (0 < train_end < validation_end <= development_end < total_rows):
        raise ValueError("Invalid frozen Phase-3B development/fold boundaries")
    development = hourly.iloc[:development_end].copy().reset_index(drop=True)
    values = development[[config.target_name]].to_numpy(dtype=np.float32)
    target = values[:, 0]
    times = development["origin_time_h"].to_numpy(dtype=float)
    split_rows = {"train": (0, train_end), "validation": (train_end, validation_end)}
    splits = {name: _windows_for_bounds(values, target, times, start, end, config) for name, (start, end) in split_rows.items()}
    inputs = {name: split.X_raw.copy() for name, split in splits.items()}
    anchors = {name: split.X_raw[:, -1, 0, None].copy() for name, split in splits.items()}
    residuals = {name: (split.y_raw - anchors[name]).astype(np.float32) for name, split in splits.items()}
    scaler_x = StandardScaler().fit(inputs["train"].reshape(-1, 1))
    scaler_y = StandardScaler().fit(residuals["train"].reshape(-1, 1))
    return HorizonFoldData(fc, config, fold, total_rows, development_end, development, split_rows, splits, inputs, anchors, residuals, scaler_x, scaler_y)


def _fit_residual(data: HorizonFoldData) -> tuple[np.ndarray, list[dict[str, float]], MIMOLSTM]:
    config = TrainConfig()
    set_seed(SEED)
    model, history = _fit_model(
        data.transformed_inputs("train"), data.transformed_residuals("train"),
        data.transformed_inputs("validation"), data.transformed_residuals("validation"),
        input_size=1, horizon=data.config.horizon_hours, config=config,
    )
    prediction = _predict(model, data.transformed_inputs("validation"), config.batch_size, config.device)
    return data.restore_prediction(prediction, "validation"), history, model


def _baselines(data: HorizonFoldData) -> dict[str, np.ndarray]:
    values = data.inputs_raw["validation"]
    return {
        "persistence": persistence_forecast(values, 0, data.config.horizon_hours),
        "linear_trend": linear_trend_forecast(values, 0, data.config.horizon_hours),
    }


def _metric_rows(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    persistence = predictions["persistence"]
    linear = predictions["linear_trend"]
    rows = []
    for model, prediction in predictions.items():
        for h in range(y_true.shape[1]):
            rmse = mean_squared_error(y_true[:, h], prediction[:, h]) ** 0.5
            rmse_p = mean_squared_error(y_true[:, h], persistence[:, h]) ** 0.5
            rmse_l = mean_squared_error(y_true[:, h], linear[:, h]) ** 0.5
            best_name, best_rmse = ("persistence", rmse_p) if rmse_p <= rmse_l else ("linear_trend", rmse_l)
            rows.append({
                "model": model,
                "horizon": h + 1,
                "MAE": float(mean_absolute_error(y_true[:, h], prediction[:, h])),
                "RMSE": float(rmse),
                "R2": float(r2_score(y_true[:, h], prediction[:, h])),
                "persistence_RMSE": float(rmse_p),
                "linear_RMSE": float(rmse_l),
                "best_baseline": best_name,
                "best_baseline_RMSE": float(best_rmse),
                "Skill_Persistence": float(1.0 - rmse / rmse_p) if rmse_p else np.nan,
                "Skill_Linear": float(1.0 - rmse / rmse_l) if rmse_l else np.nan,
                "Skill_Best": float(1.0 - rmse / best_rmse) if best_rmse else np.nan,
            })
    table = pd.DataFrame(rows)
    means = []
    for model, group in table.groupby("model", sort=False):
        mean = {"model": model, "horizon": "mean_h1_hH", "best_baseline": "per-horizon minimum"}
        for column in table.columns:
            if column not in {"model", "horizon", "best_baseline"}:
                mean[column] = float(group[column].mean())
        means.append(mean)
    return pd.concat([table, pd.DataFrame(means)], ignore_index=True)


def _metadata(data: HorizonFoldData, output_dir: Path) -> dict:
    return {
        "phase": "3B",
        "fc": data.fc,
        "seed": SEED,
        "input_feature_set": ["Utot_mean"],
        "input_mode": "absolute_history",
        "target": "Utot(t+h)-observed Utot(t), restored with origin anchor",
        "history_hours": data.config.history_hours,
        "horizon_hours": data.config.horizon_hours,
        "fold": data.fold.name,
        "split_rows": {name: list(bounds) for name, bounds in data.split_rows.items()},
        "development_row_range": [0, data.development_end_row],
        "final_test_row_range": [data.development_end_row, data.total_hourly_rows],
        "test_partition_accessed": False,
        "input_scaler_fit": "flattened current-fold train Utot windows only",
        "residual_scaler_fit": "flattened current-fold train residual labels only",
        "train_config": asdict(TrainConfig()),
        "output_dir": str(output_dir.resolve()),
    }


def _write_run(data: HorizonFoldData, output_dir: Path) -> pd.DataFrame:
    from phase3b_audit import write_phase3b_audit
    write_phase3b_audit(data, output_dir)
    residual_prediction, history, model = _fit_residual(data)
    predictions = _baselines(data)
    predictions["residual_mimo_lstm"] = residual_prediction
    validation = data.splits["validation"]
    metrics = _metric_rows(validation.y_raw, predictions)
    metrics.to_csv(output_dir / "validation_metrics_per_horizon.csv", index=False)
    rows = []
    for sample in range(len(validation.origins)):
        for h in range(data.config.horizon_hours):
            row = {"sample": sample, "origin_time_h": float(validation.feature_times[sample, -1]), "target_time_h": float(validation.target_times[sample, h]), "horizon": h + 1, "actual_Utot": float(validation.y_raw[sample, h])}
            row.update({f"pred_{name}": float(value[sample, h]) for name, value in predictions.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "validation_predictions.csv", index=False)
    torch.save(model.state_dict(), output_dir / "residual_mimo_lstm.pt")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "pipeline_metadata.json").write_text(json.dumps(_metadata(data, output_dir), indent=2), encoding="utf-8")
    return metrics


def run_phase3b(output_root: Path, data_root: Path | None = None) -> dict:
    output_dir = output_root / "seed_42"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-3B output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    all_metrics = []
    for fc in ("FC1", "FC2"):
        for horizon in HORIZONS:
            for fold in FOLD_SPECS:
                data = build_horizon_fold(fc, fold, horizon, data_root)
                run_dir = output_dir / fc / f"H_{horizon}" / fold.name
                run_dir.mkdir(parents=True, exist_ok=False)
                metrics = _write_run(data, run_dir)
                metrics.insert(0, "fold", fold.name)
                metrics.insert(0, "H", horizon)
                metrics.insert(0, "fc", fc)
                all_metrics.append(metrics)
    table = pd.concat(all_metrics, ignore_index=True)
    table.to_csv(output_dir / "all_validation_metrics.csv", index=False)
    residual = table.loc[(table["model"] == "residual_mimo_lstm") & (table["horizon"] != "mean_h1_hH")].copy()
    summary = residual.groupby(["fc", "H", "horizon"], sort=False)[["MAE", "RMSE", "R2", "Skill_Persistence", "Skill_Linear", "Skill_Best"]].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "residual_skill_by_horizon.csv", index=False)
    result = {
        "phase": "3B",
        "seed": SEED,
        "stacks": ["FC1", "FC2"],
        "horizons": list(HORIZONS),
        "folds": [asdict(fold) for fold in FOLD_SPECS],
        "input_feature_set": "Utot_only",
        "test_partition_accessed": False,
        "stop_boundary": "No final test, N sweep, relative input, new architecture, multi-seed, or RUL run was performed.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

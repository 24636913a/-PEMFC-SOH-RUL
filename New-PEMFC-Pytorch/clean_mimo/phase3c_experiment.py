"""Development-only H=24 history-length diagnosis for causal residual MIMO."""
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
from phase3b_experiment import FOLD_SPECS, FoldSpec
from train import TrainConfig, _fit_model, _predict, set_seed


HORIZON_HOURS = 24
HISTORY_LENGTHS = (15, 24, 48, 72)
SEED = 42


@dataclass
class HistoryFoldData:
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
        n, history, features = values.shape
        return self.X_scaler.transform(values.reshape(n * history, features)).reshape(values.shape).astype(np.float32)

    def transformed_residuals(self, split: str) -> np.ndarray:
        values = self.residual_labels[split]
        return self.residual_scaler.transform(values.reshape(-1, 1)).reshape(values.shape).astype(np.float32)

    def restore_prediction(self, scaled: np.ndarray, split: str) -> np.ndarray:
        residual = self.residual_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)
        return (residual + self.anchors[split]).astype(np.float32)


def _windows(values: np.ndarray, target: np.ndarray, times: np.ndarray, start: int, end: int, config: PipelineConfig) -> WindowSplit:
    records_x, records_y, origins, feature_times, target_times = [], [], [], [], []
    for origin in range(start + config.history_hours - 1, end - config.horizon_hours):
        x_index = np.arange(origin - config.history_hours + 1, origin + 1)
        y_index = np.arange(origin + 1, origin + config.horizon_hours + 1)
        if x_index.min() < start or y_index.max() >= end:
            raise AssertionError("Phase-3C window crossed a fold boundary")
        records_x.append(values[x_index])
        records_y.append(target[y_index])
        origins.append(origin)
        feature_times.append(times[x_index])
        target_times.append(times[y_index])
    if not records_x:
        raise ValueError("No complete N/H windows in frozen Phase-3C fold")
    return WindowSplit(np.asarray(records_x, dtype=np.float32), np.asarray(records_y, dtype=np.float32), np.asarray(origins, dtype=np.int64), np.asarray(feature_times, dtype=np.float64), np.asarray(target_times, dtype=np.float64))


def build_history_fold(fc: str, fold: FoldSpec, history_hours: int, data_root: Path | None = None) -> HistoryFoldData:
    if fc not in {"FC1", "FC2"}:
        raise ValueError("FC must be FC1 or FC2")
    if history_hours not in HISTORY_LENGTHS:
        raise ValueError(f"N must be one of {HISTORY_LENGTHS}")
    config = PipelineConfig(history_hours=history_hours, horizon_hours=HORIZON_HOURS)
    raw = load_ageing_data(fc, data_root)
    hourly, _ = aggregate_to_hourly(raw)
    total_rows = len(hourly)
    development_end = int(total_rows * 0.60)
    train_end = int(total_rows * fold.train_end_fraction)
    validation_end = int(total_rows * fold.validation_end_fraction)
    if not (0 < train_end < validation_end <= development_end < total_rows):
        raise ValueError("Invalid Phase-3C fold boundaries")
    development = hourly.iloc[:development_end].copy().reset_index(drop=True)
    values = development[[config.target_name]].to_numpy(dtype=np.float32)
    times = development["origin_time_h"].to_numpy(dtype=float)
    split_rows = {"train": (0, train_end), "validation": (train_end, validation_end)}
    splits = {name: _windows(values, values[:, 0], times, start, end, config) for name, (start, end) in split_rows.items()}
    inputs = {name: split.X_raw.copy() for name, split in splits.items()}
    anchors = {name: split.X_raw[:, -1, 0, None].copy() for name, split in splits.items()}
    residuals = {name: (split.y_raw - anchors[name]).astype(np.float32) for name, split in splits.items()}
    input_scaler = StandardScaler().fit(inputs["train"].reshape(-1, 1))
    residual_scaler = StandardScaler().fit(residuals["train"].reshape(-1, 1))
    return HistoryFoldData(fc, config, fold, total_rows, development_end, development, split_rows, splits, inputs, anchors, residuals, input_scaler, residual_scaler)


def _fit_residual(data: HistoryFoldData) -> tuple[np.ndarray, list[dict[str, float]], MIMOLSTM]:
    config = TrainConfig()
    set_seed(SEED)
    model, history = _fit_model(
        data.transformed_inputs("train"), data.transformed_residuals("train"),
        data.transformed_inputs("validation"), data.transformed_residuals("validation"),
        input_size=1, horizon=HORIZON_HOURS, config=config,
    )
    prediction = _predict(model, data.transformed_inputs("validation"), config.batch_size, config.device)
    return data.restore_prediction(prediction, "validation"), history, model


def _metrics(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    persistence, linear = predictions["persistence"], predictions["linear_trend"]
    rows = []
    for model, prediction in predictions.items():
        for horizon in range(HORIZON_HOURS):
            rmse = mean_squared_error(y_true[:, horizon], prediction[:, horizon]) ** 0.5
            p_rmse = mean_squared_error(y_true[:, horizon], persistence[:, horizon]) ** 0.5
            l_rmse = mean_squared_error(y_true[:, horizon], linear[:, horizon]) ** 0.5
            best_name, best_rmse = ("persistence", p_rmse) if p_rmse <= l_rmse else ("linear_trend", l_rmse)
            rows.append({
                "model": model, "horizon": horizon + 1,
                "MAE": float(mean_absolute_error(y_true[:, horizon], prediction[:, horizon])),
                "RMSE": float(rmse), "R2": float(r2_score(y_true[:, horizon], prediction[:, horizon])),
                "persistence_RMSE": float(p_rmse), "linear_RMSE": float(l_rmse),
                "best_baseline": best_name, "best_baseline_RMSE": float(best_rmse),
                "Skill_Persistence": float(1 - rmse / p_rmse) if p_rmse else np.nan,
                "Skill_Linear": float(1 - rmse / l_rmse) if l_rmse else np.nan,
                "Skill_Best": float(1 - rmse / best_rmse) if best_rmse else np.nan,
            })
    table = pd.DataFrame(rows)
    means = []
    for model, group in table.groupby("model", sort=False):
        row = {"model": model, "horizon": "mean_h1_h24", "best_baseline": "per-horizon minimum"}
        for column in table.columns:
            if column not in {"model", "horizon", "best_baseline"}:
                row[column] = float(group[column].mean())
        means.append(row)
    return pd.concat([table, pd.DataFrame(means)], ignore_index=True)


def _metadata(data: HistoryFoldData, output_dir: Path) -> dict:
    return {
        "phase": "3C", "fc": data.fc, "seed": SEED,
        "history_hours": data.config.history_hours, "horizon_hours": HORIZON_HOURS,
        "feature_set": ["Utot_mean"], "target": "Utot(t+h)-observed Utot(t)",
        "split_rows": {name: list(bounds) for name, bounds in data.split_rows.items()},
        "development_row_range": [0, data.development_end_row],
        "final_test_row_range": [data.development_end_row, data.total_hourly_rows],
        "test_partition_accessed": False,
        "input_scaler_fit": "flattened current-fold train Utot windows only",
        "residual_scaler_fit": "flattened current-fold train residuals only",
        "train_config": asdict(TrainConfig()), "output_dir": str(output_dir.resolve()),
    }


def _write_run(data: HistoryFoldData, output_dir: Path) -> pd.DataFrame:
    from phase3c_audit import write_phase3c_audit
    write_phase3c_audit(data, output_dir)
    prediction, history, model = _fit_residual(data)
    validation = data.splits["validation"]
    predictions = {
        "persistence": persistence_forecast(data.inputs_raw["validation"], 0, HORIZON_HOURS),
        "linear_trend": linear_trend_forecast(data.inputs_raw["validation"], 0, HORIZON_HOURS),
        "residual_mimo_lstm": prediction,
    }
    metrics = _metrics(validation.y_raw, predictions)
    metrics.to_csv(output_dir / "validation_metrics_per_horizon.csv", index=False)
    records = []
    for sample in range(len(validation.origins)):
        for horizon in range(HORIZON_HOURS):
            item = {"sample": sample, "origin_time_h": float(validation.feature_times[sample, -1]), "target_time_h": float(validation.target_times[sample, horizon]), "horizon": horizon + 1, "actual_Utot": float(validation.y_raw[sample, horizon])}
            item.update({f"pred_{name}": float(value[sample, horizon]) for name, value in predictions.items()})
            records.append(item)
    pd.DataFrame(records).to_csv(output_dir / "validation_predictions.csv", index=False)
    torch.save(model.state_dict(), output_dir / "residual_mimo_lstm.pt")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "pipeline_metadata.json").write_text(json.dumps(_metadata(data, output_dir), indent=2), encoding="utf-8")
    return metrics


def _history_summary(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual = table.loc[(table["model"] == "residual_mimo_lstm") & (table["horizon"] != "mean_h1_h24")].copy()
    residual["horizon"] = residual["horizon"].astype(int)
    raw = residual.assign(positive=residual["Skill_Best"] > 0)
    per_horizon = raw.groupby(["fc", "N", "horizon"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"), Skill_Best_mean=("Skill_Best", "mean"),
        Skill_Best_std=("Skill_Best", "std"), positive_folds=("positive", "sum"),
    ).reset_index()
    rows = []
    for (fc, history), group in raw.groupby(["fc", "N"], sort=False):
        stable = per_horizon.loc[(per_horizon["fc"] == fc) & (per_horizon["N"] == history)]
        all_h = float(group["Skill_Best"].mean())
        late = float(group.loc[group["horizon"] >= 13, "Skill_Best"].mean())
        stable_horizons = stable.loc[stable["positive_folds"] == 3, "horizon"].astype(int).tolist()
        flags = set(stable_horizons)
        crossover = None
        for horizon in range(1, HORIZON_HOURS - 1):
            if {horizon, horizon + 1, horizon + 2}.issubset(flags):
                crossover = horizon
                break
        rows.append({
            "fc": fc, "N": int(history), "MeanSkill_all": all_h, "MeanSkill_late_h13_h24": late,
            "stable_positive_horizon_count": len(stable_horizons),
            "stable_positive_horizons": ", ".join(map(str, stable_horizons)) if stable_horizons else "none",
            "crossover_h_c": crossover if crossover is not None else "none",
        })
    return pd.DataFrame(rows), per_horizon


def run_phase3c(output_root: Path, data_root: Path | None = None) -> dict:
    output_dir = output_root / "seed_42"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-3C output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    all_metrics = []
    for fc in ("FC1", "FC2"):
        for history in HISTORY_LENGTHS:
            for fold in FOLD_SPECS:
                data = build_history_fold(fc, fold, history, data_root)
                run_dir = output_dir / fc / f"N_{history}" / fold.name
                run_dir.mkdir(parents=True, exist_ok=False)
                metrics = _write_run(data, run_dir)
                metrics.insert(0, "fold", fold.name)
                metrics.insert(0, "N", history)
                metrics.insert(0, "fc", fc)
                all_metrics.append(metrics)
    table = pd.concat(all_metrics, ignore_index=True)
    table.to_csv(output_dir / "all_validation_metrics.csv", index=False)
    summary, per_horizon = _history_summary(table)
    summary.to_csv(output_dir / "history_length_summary.csv", index=False)
    per_horizon.to_csv(output_dir / "history_length_skill_by_horizon.csv", index=False)
    result = {
        "phase": "3C", "seed": SEED, "stacks": ["FC1", "FC2"], "H": HORIZON_HOURS,
        "history_lengths": list(HISTORY_LENGTHS), "folds": [asdict(fold) for fold in FOLD_SPECS],
        "test_partition_accessed": False,
        "stop_boundary": "No M-LSTM, multi-scale, relative input, uncertainty, multi-seed, final test, or RUL run was performed.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

"""Phase-2b FC2-only expanding-validation feature ladder.

This module intentionally creates no final-test split or test predictions.  It
uses only the first 60% of the hourly record and only residual MIMO-LSTM plus
the lawful analytical baselines.
"""
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
from evaluate import linear_trend_forecast, persistence_forecast, prediction_frame
from models import MIMOLSTM
from train import TrainConfig, _fit_model, _predict, set_seed


VOLTAGE_FEATURES = ("Utot_mean", "U1_mean", "U2_mean", "U3_mean", "U4_mean", "U5_mean")
FEATURE_LADDER: Dict[str, tuple[str, ...]] = {
    "R1_utot_only": ("Utot_mean",),
    "R2_voltage_state": VOLTAGE_FEATURES,
    "R3_state_load": VOLTAGE_FEATURES + ("J_mean", "I_mean", "I_std", "I_p2p"),
}


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_end_fraction: float
    validation_end_fraction: float


FOLD_SPECS = (
    FoldSpec("fold_1", 0.35, 0.40),
    FoldSpec("fold_2", 0.45, 0.50),
    FoldSpec("fold_3", 0.55, 0.60),
)


@dataclass
class Phase2bFoldData:
    fc: str
    config: PipelineConfig
    fold: FoldSpec
    total_hourly_rows: int
    development_end_row: int
    hourly_development: pd.DataFrame
    all_feature_names: list[str]
    feature_set: str
    feature_names: list[str]
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

    def restore_residual_prediction(self, scaled_prediction: np.ndarray, split: str) -> np.ndarray:
        residual = self.residual_scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).reshape(scaled_prediction.shape)
        return (residual + self.anchors[split]).astype(np.float32)


def _windows_for_bounds(
    values: np.ndarray,
    target: np.ndarray,
    times: np.ndarray,
    start: int,
    end: int,
    config: PipelineConfig,
) -> WindowSplit:
    features, labels, origins, feature_times, target_times = [], [], [], [], []
    for origin in range(start + config.history_hours - 1, end - config.horizon_hours):
        history_index = np.arange(origin - config.history_hours + 1, origin + 1)
        target_index = np.arange(origin + 1, origin + config.horizon_hours + 1)
        if history_index.min() < start or target_index.max() >= end:
            raise AssertionError("Phase-2b window crossed a fold boundary")
        features.append(values[history_index])
        labels.append(target[target_index])
        origins.append(origin)
        feature_times.append(times[history_index])
        target_times.append(times[target_index])
    if not features:
        raise ValueError("No complete Phase-2b windows inside fold boundary")
    return WindowSplit(
        X_raw=np.asarray(features, dtype=np.float32),
        y_raw=np.asarray(labels, dtype=np.float32),
        origins=np.asarray(origins, dtype=np.int64),
        feature_times=np.asarray(feature_times, dtype=np.float64),
        target_times=np.asarray(target_times, dtype=np.float64),
    )


def build_phase2b_fold(
    fc: str,
    fold: FoldSpec,
    feature_set: str,
    data_root: Path | None = None,
) -> Phase2bFoldData:
    """Build one fold from FC2's development prefix without a test split."""
    if fc != "FC2":
        raise ValueError("Phase 2b is frozen to FC2 only")
    if feature_set not in FEATURE_LADDER:
        raise ValueError(f"Unknown feature set: {feature_set}")
    config = PipelineConfig()
    raw = load_ageing_data(fc, data_root)
    hourly, all_feature_names = aggregate_to_hourly(raw)
    total_rows = len(hourly)
    development_end = int(total_rows * 0.60)
    train_end = int(total_rows * fold.train_end_fraction)
    validation_end = int(total_rows * fold.validation_end_fraction)
    if not (0 < train_end < validation_end <= development_end < total_rows):
        raise ValueError("Invalid frozen Phase-2b development/fold bounds")
    hourly_development = hourly.iloc[:development_end].copy().reset_index(drop=True)
    names = list(FEATURE_LADDER[feature_set])
    missing = [name for name in names if name not in all_feature_names]
    if missing:
        raise KeyError(f"Unavailable Phase-2b feature(s): {missing}")
    values = hourly_development[all_feature_names].to_numpy(dtype=np.float32)
    target = hourly_development[config.target_name].to_numpy(dtype=np.float32)
    times = hourly_development["origin_time_h"].to_numpy(dtype=float)
    split_rows = {"train": (0, train_end), "validation": (train_end, validation_end)}
    all_splits = {
        split_name: _windows_for_bounds(values, target, times, start, end, config)
        for split_name, (start, end) in split_rows.items()
    }
    selected_indices = np.asarray([all_feature_names.index(name) for name in names], dtype=np.int64)
    utot_index = all_feature_names.index(config.target_name)
    inputs_raw: Dict[str, np.ndarray] = {}
    anchors: Dict[str, np.ndarray] = {}
    residual_labels: Dict[str, np.ndarray] = {}
    for split_name, split in all_splits.items():
        inputs = split.X_raw[:, :, selected_indices].copy().astype(np.float32)
        anchor = split.X_raw[:, -1, utot_index, None].copy().astype(np.float32)
        inputs_raw[split_name] = inputs
        anchors[split_name] = anchor
        residual_labels[split_name] = (split.y_raw - anchor).astype(np.float32)
    train_inputs = inputs_raw["train"]
    X_scaler = StandardScaler().fit(train_inputs.reshape(-1, len(names)))
    residual_scaler = StandardScaler().fit(residual_labels["train"].reshape(-1, 1))
    return Phase2bFoldData(
        fc=fc,
        config=config,
        fold=fold,
        total_hourly_rows=total_rows,
        development_end_row=development_end,
        hourly_development=hourly_development,
        all_feature_names=all_feature_names,
        feature_set=feature_set,
        feature_names=names,
        split_rows=split_rows,
        splits=all_splits,
        inputs_raw=inputs_raw,
        anchors=anchors,
        residual_labels=residual_labels,
        X_scaler=X_scaler,
        residual_scaler=residual_scaler,
    )


def _metric_rows(y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    persistence = predictions["persistence"]
    linear = predictions["linear_trend"]
    rows = []
    for model, prediction in predictions.items():
        for h in range(y_true.shape[1]):
            rmse = mean_squared_error(y_true[:, h], prediction[:, h]) ** 0.5
            persistence_rmse = mean_squared_error(y_true[:, h], persistence[:, h]) ** 0.5
            linear_rmse = mean_squared_error(y_true[:, h], linear[:, h]) ** 0.5
            rows.append(
                {
                    "model": model,
                    "horizon": h + 1,
                    "MAE": float(mean_absolute_error(y_true[:, h], prediction[:, h])),
                    "RMSE": float(rmse),
                    "R2": float(r2_score(y_true[:, h], prediction[:, h])),
                    "persistence_RMSE": float(persistence_rmse),
                    "linear_RMSE": float(linear_rmse),
                    "Skill_P": float(1.0 - rmse / persistence_rmse) if persistence_rmse else np.nan,
                    "Skill_L": float(1.0 - rmse / linear_rmse) if linear_rmse else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    means = []
    for model, group in table.groupby("model", sort=False):
        mean = {"model": model, "horizon": "mean_h1_h4"}
        for column in table.columns:
            if column not in {"model", "horizon"}:
                mean[column] = float(group[column].mean())
        means.append(mean)
    return pd.concat([table, pd.DataFrame(means)], ignore_index=True)


def _train_residual(data: Phase2bFoldData, seed: int, config: TrainConfig) -> tuple[np.ndarray, list[dict[str, float]], MIMOLSTM]:
    set_seed(seed)
    model, history = _fit_model(
        data.transformed_inputs("train"),
        data.transformed_residuals("train"),
        data.transformed_inputs("validation"),
        data.transformed_residuals("validation"),
        input_size=len(data.feature_names),
        horizon=data.config.horizon_hours,
        config=config,
    )
    scaled = _predict(model, data.transformed_inputs("validation"), config.batch_size, config.device)
    return data.restore_residual_prediction(scaled, "validation"), history, model


def _validation_predictions(data: Phase2bFoldData) -> Dict[str, np.ndarray]:
    validation_inputs = data.inputs_raw["validation"]
    utot_index = data.feature_names.index(data.config.target_name)
    return {
        "persistence": persistence_forecast(validation_inputs, utot_index, data.config.horizon_hours),
        "linear_trend": linear_trend_forecast(validation_inputs, utot_index, data.config.horizon_hours),
    }


def _metadata(data: Phase2bFoldData, output_dir: Path, seed: int, config: TrainConfig) -> dict:
    return {
        "phase": "2b",
        "fc": data.fc,
        "seed": seed,
        "fold": data.fold.name,
        "feature_set": data.feature_set,
        "feature_names": data.feature_names,
        "input_mode": "absolute_inputs",
        "target_definition": "Utot(t+h) - observed Utot(t), h=1..4",
        "total_hourly_rows": data.total_hourly_rows,
        "development_row_range": [0, data.development_end_row],
        "final_test_row_range": [data.development_end_row, data.total_hourly_rows],
        "split_rows": {name: list(bounds) for name, bounds in data.split_rows.items()},
        "window_counts": {name: len(split.X_raw) for name, split in data.splits.items()},
        "test_partition_accessed": False,
        "input_scaler_fit": "flattened current-fold train windows only",
        "residual_scaler_fit": "flattened current-fold train residuals only",
        "train_config": asdict(config),
        "output_dir": str(output_dir.resolve()),
    }


def _aggregate_residual_validation(all_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    residual = all_metrics.loc[
        (all_metrics["model"] == "residual_mimo_lstm") & (all_metrics["horizon"] != "mean_h1_h4")
    ].copy()
    by_horizon = (
        residual.groupby(["feature_set", "horizon"], sort=False)[["MAE", "RMSE", "R2", "Skill_P", "Skill_L"]]
        .agg(["mean", "std", "min", "max"])
    )
    by_horizon.columns = [f"{metric}_{stat}" for metric, stat in by_horizon.columns]
    by_horizon = by_horizon.reset_index()
    summary = (
        residual.groupby("feature_set", sort=False)[["MAE", "RMSE", "R2", "Skill_P", "Skill_L"]]
        .mean()
        .rename(columns={
            "MAE": "mean_validation_MAE",
            "RMSE": "mean_validation_RMSE",
            "R2": "mean_validation_R2",
            "Skill_P": "mean_Skill_P",
            "Skill_L": "mean_Skill_L",
        })
        .reset_index()
    )
    ordering = {name: index for index, name in enumerate(FEATURE_LADDER)}
    summary["predeclared_order"] = summary["feature_set"].map(ordering)
    ranked = summary.sort_values(
        ["mean_validation_RMSE", "mean_Skill_L", "predeclared_order"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected = str(ranked.iloc[0]["feature_set"])
    ranked.insert(0, "validation_rank", np.arange(1, len(ranked) + 1))
    return by_horizon, ranked, selected


def run_phase2b_fc2(
    output_root: Path,
    data_root: Path | None = None,
    seed: int = 42,
) -> dict:
    """Run all predeclared FC2 folds/R1-R3 configurations exactly once."""
    output_dir = output_root / "FC2" / f"seed_{seed}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-2b output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = TrainConfig()
    all_metrics, run_manifest = [], []
    from phase2b_audit import write_phase2b_audit  # avoid import cycle during source inspection

    for fold in FOLD_SPECS:
        for feature_set in FEATURE_LADDER:
            data = build_phase2b_fold("FC2", fold, feature_set, data_root)
            run_dir = output_dir / fold.name / feature_set
            run_dir.mkdir(parents=True, exist_ok=False)
            write_phase2b_audit(data, run_dir)
            predictions = _validation_predictions(data)
            residual_prediction, history, model = _train_residual(data, seed, config)
            predictions["residual_mimo_lstm"] = residual_prediction
            validation = data.splits["validation"]
            metrics = _metric_rows(validation.y_raw, predictions)
            metrics.insert(0, "feature_set", feature_set)
            metrics.insert(0, "fold", fold.name)
            metrics.to_csv(run_dir / "validation_metrics_per_horizon.csv", index=False)
            frame = prediction_frame(
                validation.feature_times[:, -1], validation.target_times, validation.y_raw, predictions
            )
            frame.to_csv(run_dir / "validation_predictions.csv", index=False)
            torch.save(model.state_dict(), run_dir / "residual_mimo_lstm.pt")
            (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            (run_dir / "pipeline_metadata.json").write_text(
                json.dumps(_metadata(data, run_dir, seed, config), indent=2), encoding="utf-8"
            )
            all_metrics.append(metrics)
            run_manifest.append(
                {
                    "fold": fold.name,
                    "feature_set": feature_set,
                    "output_dir": str(run_dir.resolve()),
                    "validation_windows": int(len(validation.X_raw)),
                    "test_partition_accessed": False,
                }
            )
    metric_table = pd.concat(all_metrics, ignore_index=True)
    metric_table.to_csv(output_dir / "all_validation_metrics.csv", index=False)
    horizon_summary, selection_summary, selected = _aggregate_residual_validation(metric_table)
    horizon_summary.to_csv(output_dir / "residual_validation_by_horizon.csv", index=False)
    selection_summary.to_csv(output_dir / "feature_selection_validation_only.csv", index=False)
    result = {
        "phase": "2b",
        "fc": "FC2",
        "seed": seed,
        "folds": [asdict(fold) for fold in FOLD_SPECS],
        "feature_sets": list(FEATURE_LADDER),
        "selection_rule": "Lowest mean residual validation RMSE over 3 folds x 4 horizons; exact ties: higher Skill_L then R1/R2/R3 order.",
        "validation_selected_feature_set": selected,
        "test_partition_accessed": False,
        "runs": run_manifest,
        "stop_boundary": "No final-test evaluation, OCM, decomposition, horizon extension, or multi-seed run was performed.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

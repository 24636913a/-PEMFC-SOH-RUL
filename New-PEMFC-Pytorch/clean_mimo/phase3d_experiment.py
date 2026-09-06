"""Phase 3D: prediction-origin relative-history diagnostic at fixed H=24/N=15."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from data_pipeline import PipelineConfig, WindowSplit, aggregate_to_hourly, load_ageing_data
from evaluate import linear_trend_forecast, persistence_forecast
from models import MIMOLSTM
from phase3b_experiment import FOLD_SPECS, FoldSpec
from phase3cr_experiment import MAX_HISTORY_HOURS, _max_history_windows
from train import TrainConfig, _fit_model, _predict, set_seed


HISTORY_HOURS = 15
HORIZON_HOURS = 24
SEED = 42
REPRESENTATIONS = ("absolute", "relative")


@dataclass
class RepresentationFoldData:
    fc: str
    representation: Literal["absolute", "relative"]
    config: PipelineConfig
    fold: FoldSpec
    total_hourly_rows: int
    development_end_row: int
    hourly_development: pd.DataFrame
    split_rows: Dict[str, tuple[int, int]]
    max_splits: Dict[str, WindowSplit]
    splits: Dict[str, WindowSplit]
    absolute_inputs_raw: Dict[str, np.ndarray]
    inputs_raw: Dict[str, np.ndarray]
    anchors: Dict[str, np.ndarray]
    residual_labels: Dict[str, np.ndarray]
    X_scaler: StandardScaler
    residual_scaler: StandardScaler

    def transformed_inputs(self, split: str) -> np.ndarray:
        values = self.inputs_raw[split]
        samples, history, features = values.shape
        transformed = self.X_scaler.transform(values.reshape(samples * history, features))
        return transformed.reshape(samples, history, features).astype(np.float32)

    def transformed_residuals(self, split: str) -> np.ndarray:
        values = self.residual_labels[split]
        return self.residual_scaler.transform(values.reshape(-1, 1)).reshape(values.shape).astype(np.float32)

    def restore_prediction(self, scaled: np.ndarray, split: str) -> np.ndarray:
        residual = self.residual_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)
        return (residual + self.anchors[split]).astype(np.float32)


def _slice_last_n(max_split: WindowSplit) -> WindowSplit:
    return WindowSplit(
        max_split.X_raw[:, -HISTORY_HOURS:, :].copy(),
        max_split.y_raw.copy(),
        max_split.origins.copy(),
        max_split.feature_times[:, -HISTORY_HOURS:].copy(),
        max_split.target_times.copy(),
    )


def _representation_inputs(absolute: np.ndarray, anchors: np.ndarray, representation: str) -> np.ndarray:
    if representation == "absolute":
        return absolute.copy()
    if representation == "relative":
        return (absolute - anchors[:, None, :]).astype(np.float32)
    raise ValueError(f"Unknown representation: {representation}")


def build_representation_fold(
    fc: str,
    fold: FoldSpec,
    representation: Literal["absolute", "relative"],
    data_root: Path | None = None,
) -> RepresentationFoldData:
    if fc not in {"FC1", "FC2"}:
        raise ValueError("FC must be FC1 or FC2")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Representation must be one of {REPRESENTATIONS}")
    config = PipelineConfig(history_hours=HISTORY_HOURS, horizon_hours=HORIZON_HOURS)
    raw = load_ageing_data(fc, data_root)
    hourly, _ = aggregate_to_hourly(raw)
    total_rows = len(hourly)
    development_end = int(total_rows * 0.60)
    train_end = int(total_rows * fold.train_end_fraction)
    validation_end = int(total_rows * fold.validation_end_fraction)
    if not (0 < train_end < validation_end <= development_end < total_rows):
        raise ValueError("Invalid frozen Phase-3D development/fold boundaries")
    development = hourly.iloc[:development_end].copy().reset_index(drop=True)
    values = development[[config.target_name]].to_numpy(dtype=np.float32)
    times = development["origin_time_h"].to_numpy(dtype=float)
    split_rows = {"train": (0, train_end), "validation": (train_end, validation_end)}
    max_splits = {
        name: _max_history_windows(values, values[:, 0], times, start, end)
        for name, (start, end) in split_rows.items()
    }
    splits = {name: _slice_last_n(max_split) for name, max_split in max_splits.items()}
    absolute_inputs = {name: split.X_raw.copy() for name, split in splits.items()}
    anchors = {name: split.X_raw[:, -1, 0, None].copy() for name, split in splits.items()}
    inputs = {name: _representation_inputs(absolute_inputs[name], anchors[name], representation) for name in splits}
    residuals = {name: (split.y_raw - anchors[name]).astype(np.float32) for name, split in splits.items()}
    x_scaler = StandardScaler().fit(inputs["train"].reshape(-1, 1))
    residual_scaler = StandardScaler().fit(residuals["train"].reshape(-1, 1))
    return RepresentationFoldData(
        fc, representation, config, fold, total_rows, development_end, development, split_rows,
        max_splits, splits, absolute_inputs, inputs, anchors, residuals, x_scaler, residual_scaler,
    )


def _fit_residual(data: RepresentationFoldData) -> tuple[np.ndarray, list[dict[str, float]], MIMOLSTM]:
    train_config = TrainConfig()
    set_seed(SEED)
    model, history = _fit_model(
        data.transformed_inputs("train"), data.transformed_residuals("train"),
        data.transformed_inputs("validation"), data.transformed_residuals("validation"),
        input_size=1, horizon=HORIZON_HOURS, config=train_config,
    )
    prediction = _predict(model, data.transformed_inputs("validation"), train_config.batch_size, train_config.device)
    return data.restore_prediction(prediction, "validation"), history, model


def _baselines(data: RepresentationFoldData) -> dict[str, np.ndarray]:
    absolute = data.absolute_inputs_raw["validation"]
    return {
        "persistence": persistence_forecast(absolute, 0, HORIZON_HOURS),
        "linear_trend": linear_trend_forecast(absolute, 0, HORIZON_HOURS),
    }


def _metric_rows(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
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
                "Skill_Persistence": float(1.0 - rmse / p_rmse) if p_rmse else np.nan,
                "Skill_Linear": float(1.0 - rmse / l_rmse) if l_rmse else np.nan,
                "Skill_Best": float(1.0 - rmse / best_rmse) if best_rmse else np.nan,
            })
    table = pd.DataFrame(rows)
    means = []
    for model, group in table.groupby("model", sort=False):
        mean = {"model": model, "horizon": "mean_h1_h24", "best_baseline": "per-horizon minimum"}
        for column in table.columns:
            if column not in {"model", "horizon", "best_baseline"}:
                mean[column] = float(group[column].mean())
        means.append(mean)
    return pd.concat([table, pd.DataFrame(means)], ignore_index=True)


def representation_shift_rows(data: RepresentationFoldData) -> pd.DataFrame:
    """Wasserstein shift between train and validation at every legal input lag."""
    train = data.inputs_raw["train"][:, :, 0]
    validation = data.inputs_raw["validation"][:, :, 0]
    rows = []
    for input_index in range(HISTORY_HOURS):
        lag = HISTORY_HOURS - 1 - input_index
        rows.append({
            "fc": data.fc,
            "fold": data.fold.name,
            "representation": data.representation,
            "lag_hours": lag,
            "wasserstein_distance_V": float(wasserstein_distance(train[:, input_index], validation[:, input_index])),
            "train_mean_V": float(train[:, input_index].mean()),
            "validation_mean_V": float(validation[:, input_index].mean()),
            "train_std_V": float(train[:, input_index].std(ddof=0)),
            "validation_std_V": float(validation[:, input_index].std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values("lag_hours").reset_index(drop=True)


def _metadata(data: RepresentationFoldData, output_dir: Path) -> dict:
    return {
        "phase": "3D",
        "fc": data.fc,
        "representation": data.representation,
        "seed": SEED,
        "history_hours": HISTORY_HOURS,
        "horizon_hours": HORIZON_HOURS,
        "feature_set": ["Utot_mean"],
        "target": "Utot(t+h)-observed_Utot(t), restored with legal origin anchor",
        "origin_schedule": "Phase-3C-R Nmax=48 legal origins retained; model receives only final N=15 input rows.",
        "input_definition": "[U(t-14),...,U(t)]" if data.representation == "absolute" else "[U(t-14)-U(t),...,U(t-1)-U(t),0]",
        "split_rows": {name: list(bounds) for name, bounds in data.split_rows.items()},
        "development_row_range": [0, data.development_end_row],
        "final_test_row_range": [data.development_end_row, data.total_hourly_rows],
        "test_partition_accessed": False,
        "input_scaler_fit": "flattened current-representation/current-fold train inputs only",
        "residual_scaler_fit": "flattened current-fold train residual labels only",
        "train_config": asdict(TrainConfig()),
        "output_dir": str(output_dir.resolve()),
    }


def _prediction_frame(data: RepresentationFoldData, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    validation = data.splits["validation"]
    rows = []
    for sample in range(len(validation.origins)):
        for horizon in range(HORIZON_HOURS):
            row = {
                "sample": sample, "origin_row": int(validation.origins[sample]),
                "origin_time_h": float(validation.feature_times[sample, -1]),
                "target_time_h": float(validation.target_times[sample, horizon]),
                "horizon": horizon + 1, "actual_Utot": float(validation.y_raw[sample, horizon]),
            }
            row.update({f"pred_{name}": float(value[sample, horizon]) for name, value in predictions.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _write_run(data: RepresentationFoldData, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from phase3d_audit import write_phase3d_audit

    write_phase3d_audit(data, output_dir)
    prediction, history, model = _fit_residual(data)
    predictions = _baselines(data)
    predictions["residual_mimo_lstm"] = prediction
    metrics = _metric_rows(data.splits["validation"].y_raw, predictions)
    frame = _prediction_frame(data, predictions)
    shifts = representation_shift_rows(data)
    metrics.to_csv(output_dir / "validation_metrics_per_horizon.csv", index=False)
    frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    shifts.to_csv(output_dir / "representation_shift_by_lag.csv", index=False)
    torch.save(model.state_dict(), output_dir / "residual_mimo_lstm.pt")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "pipeline_metadata.json").write_text(json.dumps(_metadata(data, output_dir), indent=2), encoding="utf-8")
    return metrics, frame, shifts


def _representation_summary(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual = table.loc[(table["model"] == "residual_mimo_lstm") & (table["horizon"] != "mean_h1_h24")].copy()
    residual["horizon"] = residual["horizon"].astype(int)
    raw = residual.assign(positive=residual["Skill_Best"] > 0)
    per_horizon = raw.groupby(["fc", "representation", "horizon"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        Skill_Best_mean=("Skill_Best", "mean"), Skill_Best_std=("Skill_Best", "std"),
        positive_folds=("positive", "sum"),
    ).reset_index()
    rows = []
    for (fc, representation), group in raw.groupby(["fc", "representation"], sort=False):
        stable = per_horizon.loc[(per_horizon["fc"] == fc) & (per_horizon["representation"] == representation)]
        stable_horizons = stable.loc[stable["positive_folds"] == 3, "horizon"].astype(int).tolist()
        flags = set(stable_horizons)
        crossover = next((h for h in range(1, HORIZON_HOURS - 1) if {h, h + 1, h + 2}.issubset(flags)), None)
        rows.append({
            "fc": fc,
            "representation": representation,
            "mean_RMSE": float(group["RMSE"].mean()),
            "mean_RMSE_std_across_folds": float(group.groupby("horizon")["RMSE"].mean().std(ddof=0)),
            "MeanSkill_Best_all": float(group["Skill_Best"].mean()),
            "LateSkill_Best_h13_h24": float(group.loc[group["horizon"] >= 13, "Skill_Best"].mean()),
            "stable_positive_horizon_count": len(stable_horizons),
            "stable_positive_horizons": ", ".join(map(str, stable_horizons)) if stable_horizons else "none",
            "crossover_h_c": crossover if crossover is not None else "none",
        })
    return pd.DataFrame(rows), per_horizon


def run_phase3d(output_root: Path, data_root: Path | None = None) -> dict:
    output_dir = output_root / "seed_42"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-3D output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_rows, prediction_rows, shift_rows = [], [], []
    for fc in ("FC1", "FC2"):
        for fold in FOLD_SPECS:
            data_by_representation = {
                representation: build_representation_fold(fc, fold, representation, data_root)
                for representation in REPRESENTATIONS
            }
            from phase3d_audit import write_representation_pair_audit
            write_representation_pair_audit(data_by_representation, output_dir / fc / "representation_pair_audits" / f"{fold.name}.json")
            for representation, data in data_by_representation.items():
                run_dir = output_dir / fc / representation / fold.name
                run_dir.mkdir(parents=True, exist_ok=False)
                metrics, predictions, shifts = _write_run(data, run_dir)
                for frame in (metrics, predictions):
                    frame.insert(0, "fold", fold.name)
                    frame.insert(0, "representation", representation)
                    frame.insert(0, "fc", fc)
                metrics_rows.append(metrics)
                prediction_rows.append(predictions)
                shift_rows.append(shifts)
    metrics = pd.concat(metrics_rows, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    shifts = pd.concat(shift_rows, ignore_index=True)
    metrics.to_csv(output_dir / "all_validation_metrics.csv", index=False)
    predictions.to_csv(output_dir / "all_validation_predictions.csv", index=False)
    shifts.to_csv(output_dir / "all_representation_shift_by_lag.csv", index=False)
    summary, per_horizon = _representation_summary(metrics)
    summary.to_csv(output_dir / "representation_summary.csv", index=False)
    per_horizon.to_csv(output_dir / "representation_skill_by_horizon.csv", index=False)
    result = {
        "phase": "3D", "seed": SEED, "stacks": ["FC1", "FC2"],
        "H": HORIZON_HOURS, "N": HISTORY_HOURS,
        "representations": list(REPRESENTATIONS),
        "folds": [asdict(fold) for fold in FOLD_SPECS],
        "test_partition_accessed": False,
        "stop_boundary": "No final test, additional representation, memory architecture, multi-seed, probability model, or RUL experiment was performed.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

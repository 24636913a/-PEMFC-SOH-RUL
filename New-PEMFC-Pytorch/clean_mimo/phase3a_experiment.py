"""Development-only causal EWMA decomposition for FC2 Phase 3A."""
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
from models import MIMOLSTM
from phase2b_experiment import FOLD_SPECS, FoldSpec
from train import TrainConfig, _fit_model, _predict, set_seed


TAUS = (3, 6, 12)
LOAD_FEATURES = ("J_mean", "I_mean", "I_std", "I_p2p")
SEED = 42


@dataclass
class Phase3FoldData:
    fc: str
    config: PipelineConfig
    fold: FoldSpec
    tau_hours: int
    alpha: float
    total_hourly_rows: int
    development_end_row: int
    hourly_development: pd.DataFrame
    feature_names: list[str]
    split_rows: Dict[str, tuple[int, int]]
    splits: Dict[str, WindowSplit]
    slow_component: np.ndarray
    transient_component: np.ndarray


@dataclass
class TaskData:
    name: str
    target_description: str
    feature_names: list[str]
    inputs_raw: Dict[str, np.ndarray]
    labels_raw: Dict[str, np.ndarray]
    X_scaler: StandardScaler
    y_scaler: StandardScaler

    def transformed_inputs(self, split: str) -> np.ndarray:
        values = self.inputs_raw[split]
        n, history, features = values.shape
        return self.X_scaler.transform(values.reshape(n * history, features)).reshape(values.shape).astype(np.float32)

    def transformed_labels(self, split: str) -> np.ndarray:
        values = self.labels_raw[split]
        return self.y_scaler.transform(values.reshape(-1, 1)).reshape(values.shape).astype(np.float32)

    def restore_prediction(self, scaled: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape).astype(np.float32)


def alpha_for_tau(tau_hours: int) -> float:
    if tau_hours not in TAUS:
        raise ValueError(f"Tau must be one of {TAUS}; got {tau_hours}")
    return float(1.0 - np.exp(-1.0 / tau_hours))


def causal_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    """One-sided EWMA: each D[t] uses U[t] and already computed D[t-1] only."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or len(source) == 0:
        raise ValueError("EWMA requires a non-empty one-dimensional voltage series")
    output = np.empty_like(source)
    output[0] = source[0]
    for index in range(1, len(source)):
        output[index] = alpha * source[index] + (1.0 - alpha) * output[index - 1]
    return output.astype(np.float32)


def _windows_for_bounds(values: np.ndarray, target: np.ndarray, times: np.ndarray, start: int, end: int, config: PipelineConfig) -> WindowSplit:
    features, labels, origins, feature_times, target_times = [], [], [], [], []
    for origin in range(start + config.history_hours - 1, end - config.horizon_hours):
        history_index = np.arange(origin - config.history_hours + 1, origin + 1)
        target_index = np.arange(origin + 1, origin + config.horizon_hours + 1)
        if history_index.min() < start or target_index.max() >= end:
            raise AssertionError("Phase-3A window crossed a fold boundary")
        features.append(values[history_index])
        labels.append(target[target_index])
        origins.append(origin)
        feature_times.append(times[history_index])
        target_times.append(times[target_index])
    if not features:
        raise ValueError("No complete Phase-3A windows inside a fold")
    return WindowSplit(
        X_raw=np.asarray(features, dtype=np.float32),
        y_raw=np.asarray(labels, dtype=np.float32),
        origins=np.asarray(origins, dtype=np.int64),
        feature_times=np.asarray(feature_times, dtype=np.float64),
        target_times=np.asarray(target_times, dtype=np.float64),
    )


def build_phase3_fold(fc: str, fold: FoldSpec, tau_hours: int, data_root: Path | None = None) -> Phase3FoldData:
    """Build derived causal series and train/validation windows; never a test split."""
    if fc != "FC2":
        raise ValueError("Phase 3A is frozen to FC2")
    config = PipelineConfig()
    raw = load_ageing_data(fc, data_root)
    hourly, feature_names = aggregate_to_hourly(raw)
    total_rows = len(hourly)
    development_end = int(total_rows * 0.60)
    train_end = int(total_rows * fold.train_end_fraction)
    validation_end = int(total_rows * fold.validation_end_fraction)
    if not (0 < train_end < validation_end <= development_end < total_rows):
        raise ValueError("Invalid Phase-3A frozen fold bounds")
    development = hourly.iloc[:development_end].copy().reset_index(drop=True)
    values = development[feature_names].to_numpy(dtype=np.float32)
    target = development[config.target_name].to_numpy(dtype=np.float32)
    times = development["origin_time_h"].to_numpy(dtype=float)
    split_rows = {"train": (0, train_end), "validation": (train_end, validation_end)}
    splits = {name: _windows_for_bounds(values, target, times, start, end, config) for name, (start, end) in split_rows.items()}
    alpha = alpha_for_tau(tau_hours)
    slow = causal_ewma(target, alpha)
    transient = (target - slow).astype(np.float32)
    return Phase3FoldData(
        fc=fc,
        config=config,
        fold=fold,
        tau_hours=tau_hours,
        alpha=alpha,
        total_hourly_rows=total_rows,
        development_end_row=development_end,
        hourly_development=development,
        feature_names=feature_names,
        split_rows=split_rows,
        splits=splits,
        slow_component=slow,
        transient_component=transient,
    )


def _component_windows(component: np.ndarray, split: WindowSplit, history_hours: int, horizon_hours: int) -> tuple[np.ndarray, np.ndarray]:
    histories, futures = [], []
    for origin in split.origins:
        histories.append(component[origin - history_hours + 1:origin + 1])
        futures.append(component[origin + 1:origin + horizon_hours + 1])
    return np.asarray(histories, dtype=np.float32), np.asarray(futures, dtype=np.float32)


def _task_from_arrays(name: str, description: str, feature_names: list[str], inputs: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]) -> TaskData:
    train_x = inputs["train"]
    scaler_x = StandardScaler().fit(train_x.reshape(-1, train_x.shape[-1]))
    scaler_y = StandardScaler().fit(labels["train"].reshape(-1, 1))
    return TaskData(name, description, feature_names, inputs, labels, scaler_x, scaler_y)


def make_t1_slow_task(data: Phase3FoldData) -> TaskData:
    inputs, labels = {}, {}
    for split_name, split in data.splits.items():
        history, future = _component_windows(data.slow_component, split, data.config.history_hours, data.config.horizon_hours)
        inputs[split_name] = history[:, :, None]
        labels[split_name] = future - history[:, -1, None]
    return _task_from_arrays("T1_slow_residual_mimo_lstm", "Delta D(t+h) = D(t+h)-D(t)", ["D_causal_ewma"], inputs, labels)


def make_t2_transient_task(data: Phase3FoldData, with_load: bool) -> TaskData:
    inputs, labels = {}, {}
    load_indices = [data.feature_names.index(name) for name in LOAD_FEATURES]
    for split_name, split in data.splits.items():
        history, future = _component_windows(data.transient_component, split, data.config.history_hours, data.config.horizon_hours)
        if with_load:
            load = split.X_raw[:, :, load_indices]
            inputs[split_name] = np.concatenate([history[:, :, None], load], axis=2)
        else:
            inputs[split_name] = history[:, :, None]
        labels[split_name] = future
    suffix = "R_history_plus_load" if with_load else "R_history_only"
    features = ["R_causal_ewma"] + (list(LOAD_FEATURES) if with_load else [])
    return _task_from_arrays(f"T2_{suffix}_mimo_lstm", "R(t+h)", features, inputs, labels)


def _fit_task(task: TaskData, config: TrainConfig) -> tuple[np.ndarray, list[dict[str, float]], MIMOLSTM]:
    set_seed(SEED)
    model, history = _fit_model(
        task.transformed_inputs("train"), task.transformed_labels("train"),
        task.transformed_inputs("validation"), task.transformed_labels("validation"),
        input_size=len(task.feature_names), horizon=4, config=config,
    )
    prediction = _predict(model, task.transformed_inputs("validation"), config.batch_size, config.device)
    return task.restore_prediction(prediction), history, model


def _slow_baselines(task: TaskData) -> dict[str, np.ndarray]:
    history = task.inputs_raw["validation"][:, :, 0]
    h = task.labels_raw["validation"].shape[1]
    anchor = history[:, -1, None]
    x = np.arange(history.shape[1], dtype=float)
    future_x = np.arange(history.shape[1], history.shape[1] + h, dtype=float)
    linear = np.empty((len(history), h), dtype=np.float32)
    for row, series in enumerate(history):
        slope, intercept = np.polyfit(x, series, deg=1)
        linear[row] = slope * future_x + intercept - anchor[row]
    return {"slow_persistence": np.zeros_like(task.labels_raw["validation"]), "slow_linear_trend": linear}


def _transient_baselines(task: TaskData) -> dict[str, np.ndarray]:
    history = task.inputs_raw["validation"][:, :, 0]
    h = task.labels_raw["validation"].shape[1]
    return {
        "zero_transient": np.zeros_like(task.labels_raw["validation"]),
        "transient_persistence": np.repeat(history[:, -1, None], h, axis=1),
    }


def _metric_rows(y_true: np.ndarray, predictions: dict[str, np.ndarray], baseline_a: str, baseline_b: str) -> pd.DataFrame:
    reference_a, reference_b = predictions[baseline_a], predictions[baseline_b]
    rows = []
    for model, prediction in predictions.items():
        for horizon in range(y_true.shape[1]):
            actual = y_true[:, horizon]
            estimate = prediction[:, horizon]
            rmse = mean_squared_error(actual, estimate) ** 0.5
            std = float(np.std(actual, ddof=0))
            rmse_a = mean_squared_error(actual, reference_a[:, horizon]) ** 0.5
            rmse_b = mean_squared_error(actual, reference_b[:, horizon]) ** 0.5
            rows.append({
                "model": model,
                "horizon": horizon + 1,
                "MAE": float(mean_absolute_error(actual, estimate)),
                "RMSE": float(rmse),
                "R2": float(r2_score(actual, estimate)),
                "target_std": std,
                "NRMSE": float(rmse / std) if std else np.nan,
                f"Skill_{baseline_a}": float(1.0 - rmse / rmse_a) if rmse_a else np.nan,
                f"Skill_{baseline_b}": float(1.0 - rmse / rmse_b) if rmse_b else np.nan,
            })
    table = pd.DataFrame(rows)
    means = []
    for model, group in table.groupby("model", sort=False):
        mean = {"model": model, "horizon": "mean_h1_h4"}
        for column in table.columns:
            if column not in {"model", "horizon"}:
                mean[column] = float(group[column].mean())
        means.append(mean)
    return pd.concat([table, pd.DataFrame(means)], ignore_index=True)


def _task_metadata(data: Phase3FoldData, task: TaskData, output_dir: Path, config: TrainConfig) -> dict:
    return {
        "phase": "3A",
        "fc": data.fc,
        "fold": data.fold.name,
        "tau_hours": data.tau_hours,
        "alpha": data.alpha,
        "task": task.name,
        "target": task.target_description,
        "feature_names": task.feature_names,
        "split_rows": {name: list(bounds) for name, bounds in data.split_rows.items()},
        "development_row_range": [0, data.development_end_row],
        "final_test_row_range": [data.development_end_row, data.total_hourly_rows],
        "test_partition_accessed": False,
        "input_scaler_fit": "flattened current-fold training representations only",
        "target_scaler_fit": "flattened current-fold training labels only",
        "train_config": asdict(config),
        "output_dir": str(output_dir.resolve()),
    }


def _write_task_run(data: Phase3FoldData, task: TaskData, baseline_predictions: dict[str, np.ndarray], baseline_a: str, baseline_b: str, output_dir: Path) -> pd.DataFrame:
    from phase3a_audit import write_phase3a_audit
    config = TrainConfig()
    write_phase3a_audit(data, task, output_dir)
    prediction, history, model = _fit_task(task, config)
    predictions = dict(baseline_predictions)
    predictions[task.name] = prediction
    metrics = _metric_rows(task.labels_raw["validation"], predictions, baseline_a, baseline_b)
    metrics.to_csv(output_dir / "validation_metrics_per_horizon.csv", index=False)
    split = data.splits["validation"]
    rows = []
    for sample, origin in enumerate(split.origins):
        for horizon in range(data.config.horizon_hours):
            row = {
                "sample": sample,
                "origin_time_h": float(split.feature_times[sample, -1]),
                "target_time_h": float(split.target_times[sample, horizon]),
                "horizon": horizon + 1,
                "actual_target": float(task.labels_raw["validation"][sample, horizon]),
            }
            row.update({f"pred_{name}": float(value[sample, horizon]) for name, value in predictions.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "validation_predictions.csv", index=False)
    torch.save(model.state_dict(), output_dir / f"{task.name}.pt")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "pipeline_metadata.json").write_text(json.dumps(_task_metadata(data, task, output_dir, config), indent=2), encoding="utf-8")
    return metrics


def _t0_reference(data: Phase3FoldData, root: Path) -> pd.DataFrame:
    """Read saved Phase-2b R1 predictions only; never train T0 again."""
    path = root / "outputs_phase2b" / "FC2" / "seed_42" / data.fold.name / "R1_utot_only" / "validation_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Saved Phase-2b T0 reference is required but missing: {path}")
    frame = pd.read_csv(path)
    split = data.splits["validation"]
    expected_actual = split.y_raw
    actual = frame.pivot(index="sample", columns="horizon", values="actual_Utot_mean").sort_index(axis=1).to_numpy(dtype=np.float32)
    prediction = frame.pivot(index="sample", columns="horizon", values="pred_residual_mimo_lstm").sort_index(axis=1).to_numpy(dtype=np.float32)
    if actual.shape != expected_actual.shape or not np.allclose(actual, expected_actual):
        raise ValueError("Saved Phase-2b T0 predictions do not align with this frozen fold")
    anchor = split.X_raw[:, -1, data.feature_names.index(data.config.target_name), None]
    actual_delta = actual - anchor
    predicted_delta = prediction - anchor
    rows = []
    for horizon in range(data.config.horizon_hours):
        rmse = mean_squared_error(actual_delta[:, horizon], predicted_delta[:, horizon]) ** 0.5
        std = float(np.std(actual_delta[:, horizon], ddof=0))
        rows.append({"fold": data.fold.name, "source": "Phase2b_saved_R1_raw_residual", "horizon": horizon + 1, "RMSE": float(rmse), "target_std": std, "NRMSE": float(rmse / std) if std else np.nan})
    return pd.DataFrame(rows)


def _select_tau(t1_metrics: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    model = t1_metrics.loc[(t1_metrics["model"] == "T1_slow_residual_mimo_lstm") & (t1_metrics["horizon"] != "mean_h1_h4")].copy()
    summary = model.groupby("tau_hours", sort=True)[["NRMSE", "Skill_slow_persistence", "Skill_slow_linear_trend"]].mean().rename(columns={"NRMSE": "mean_NRMSE", "Skill_slow_persistence": "mean_Skill_SlowPersistence", "Skill_slow_linear_trend": "mean_Skill_SlowLinear"}).reset_index()
    fold_summary = model.groupby(["tau_hours", "fold"], sort=True)[["NRMSE", "Skill_slow_linear_trend"]].mean().reset_index()
    stability = fold_summary.groupby("tau_hours", sort=True).agg(fold_mean_NRMSE_std=("NRMSE", "std"), positive_SlowLinear_folds=("Skill_slow_linear_trend", lambda values: int((values > 0).sum()))).reset_index()
    summary = summary.merge(stability, on="tau_hours", how="left")
    best_nrmse = float(summary["mean_NRMSE"].min())
    summary["within_3pct_best_NRMSE"] = summary["mean_NRMSE"] <= best_nrmse * 1.03
    candidates = summary.loc[summary["within_3pct_best_NRMSE"]].copy()
    candidates = candidates.sort_values(["mean_Skill_SlowLinear", "fold_mean_NRMSE_std", "tau_hours"], ascending=[False, True, True], kind="mergesort")
    selected = int(candidates.iloc[0]["tau_hours"])
    tau6 = summary.loc[summary["tau_hours"] == 6].iloc[0]
    tau12 = summary.loc[summary["tau_hours"] == 12].iloc[0]
    if bool(tau6["within_3pct_best_NRMSE"]) and bool(tau12["within_3pct_best_NRMSE"]) and abs(float(tau6["mean_Skill_SlowLinear"]) - float(tau12["mean_Skill_SlowLinear"])) <= 0.01:
        selected = 6
    summary["validation_selected_tau"] = summary["tau_hours"] == selected
    return summary, selected


def run_phase3a_fc2(output_root: Path, data_root: Path | None = None) -> dict:
    """Run the predeclared development-only T1 ladder, then T2 at selected tau."""
    output_dir = output_root / "FC2" / "seed_42"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-3A output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    all_t1, t0 = [], []
    for tau in TAUS:
        for fold in FOLD_SPECS:
            data = build_phase3_fold("FC2", fold, tau, data_root)
            task = make_t1_slow_task(data)
            run_dir = output_dir / "T1_slow" / f"tau_{tau}h" / fold.name
            run_dir.mkdir(parents=True, exist_ok=False)
            metrics = _write_task_run(data, task, _slow_baselines(task), "slow_persistence", "slow_linear_trend", run_dir)
            metrics.insert(0, "tau_hours", tau)
            metrics.insert(0, "fold", fold.name)
            all_t1.append(metrics)
            if tau == TAUS[0]:
                t0.append(_t0_reference(data, root))
    t1_metrics = pd.concat(all_t1, ignore_index=True)
    t0_reference = pd.concat(t0, ignore_index=True)
    t1_metrics.to_csv(output_dir / "T1_all_validation_metrics.csv", index=False)
    t0_reference.to_csv(output_dir / "T0_saved_raw_residual_reference.csv", index=False)
    tau_selection, selected_tau = _select_tau(t1_metrics)
    tau_selection.to_csv(output_dir / "T1_tau_selection_validation_only.csv", index=False)

    all_t2 = []
    for fold in FOLD_SPECS:
        data = build_phase3_fold("FC2", fold, selected_tau, data_root)
        for with_load in (False, True):
            task = make_t2_transient_task(data, with_load)
            mode = "T2A_R_history_only" if not with_load else "T2B_R_history_plus_load"
            run_dir = output_dir / "T2_transient" / f"tau_{selected_tau}h" / fold.name / mode
            run_dir.mkdir(parents=True, exist_ok=False)
            metrics = _write_task_run(data, task, _transient_baselines(task), "zero_transient", "transient_persistence", run_dir)
            metrics.insert(0, "tau_hours", selected_tau)
            metrics.insert(0, "fold", fold.name)
            metrics.insert(0, "variant", mode)
            all_t2.append(metrics)
    t2_metrics = pd.concat(all_t2, ignore_index=True)
    t2_metrics.to_csv(output_dir / "T2_all_validation_metrics.csv", index=False)
    result = {
        "phase": "3A",
        "fc": "FC2",
        "seed": SEED,
        "development_only": True,
        "test_partition_accessed": False,
        "taus": list(TAUS),
        "validation_selected_tau_hours": selected_tau,
        "t1_task": "Delta D(t+h) from D history only",
        "t2_tasks": ["R history only", "R history plus J/I/I_std/I_p2p"],
        "t0_reference": "Saved Phase-2b R1 validation forecasts; no T0 retraining.",
        "stop_boundary": "No final test, OCM, horizon expansion, decomposition follow-on, other seed, or RUL experiment was run.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

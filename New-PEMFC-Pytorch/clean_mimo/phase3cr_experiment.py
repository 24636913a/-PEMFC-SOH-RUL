"""Matched-origin H=24 history-length diagnostic for causal residual MIMO.

Phase 3C-R deliberately forms every legal origin with Nmax=48 first.  N=15
and N=24 then receive the final 15 or 24 observations from exactly that same
window.  This holds both the train and validation prediction origins fixed so
that history length is the only experimental variable.
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
from evaluate import linear_trend_forecast, persistence_forecast
from models import MIMOLSTM
from phase3b_experiment import FOLD_SPECS, FoldSpec
from train import TrainConfig, _fit_model, _predict, set_seed


HORIZON_HOURS = 24
MAX_HISTORY_HOURS = 48
HISTORY_LENGTHS = (15, 24, 48)
SEED = 42


@dataclass
class MatchedHistoryFoldData:
    fc: str
    config: PipelineConfig
    fold: FoldSpec
    total_hourly_rows: int
    development_end_row: int
    hourly_development: pd.DataFrame
    split_rows: Dict[str, tuple[int, int]]
    max_splits: Dict[str, WindowSplit]
    splits: Dict[str, WindowSplit]
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


def _max_history_windows(
    values: np.ndarray,
    target: np.ndarray,
    times: np.ndarray,
    start: int,
    end: int,
) -> WindowSplit:
    """Construct the sole legal origin set, using Nmax=48 for every model."""
    features, labels, origins, feature_times, target_times = [], [], [], [], []
    for origin in range(start + MAX_HISTORY_HOURS - 1, end - HORIZON_HOURS):
        history_index = np.arange(origin - MAX_HISTORY_HOURS + 1, origin + 1)
        target_index = np.arange(origin + 1, origin + HORIZON_HOURS + 1)
        if history_index.min() < start or target_index.max() >= end:
            raise AssertionError("Phase-3C-R maximum-history window crossed a fold boundary")
        features.append(values[history_index])
        labels.append(target[target_index])
        origins.append(origin)
        feature_times.append(times[history_index])
        target_times.append(times[target_index])
    if not features:
        raise ValueError("No complete Nmax=48/H=24 windows in frozen Phase-3C-R fold")
    return WindowSplit(
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(origins, dtype=np.int64),
        np.asarray(feature_times, dtype=np.float64),
        np.asarray(target_times, dtype=np.float64),
    )


def _slice_max_split(max_split: WindowSplit, history_hours: int) -> WindowSplit:
    """Use only the final N legal observations, without changing targets/origins."""
    if history_hours not in HISTORY_LENGTHS:
        raise ValueError(f"N must be one of {HISTORY_LENGTHS}")
    return WindowSplit(
        max_split.X_raw[:, -history_hours:, :].copy(),
        max_split.y_raw.copy(),
        max_split.origins.copy(),
        max_split.feature_times[:, -history_hours:].copy(),
        max_split.target_times.copy(),
    )


def build_matched_history_fold(
    fc: str,
    fold: FoldSpec,
    history_hours: int,
    data_root: Path | None = None,
) -> MatchedHistoryFoldData:
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
        raise ValueError("Invalid frozen Phase-3C-R development/fold boundaries")
    development = hourly.iloc[:development_end].copy().reset_index(drop=True)
    values = development[[config.target_name]].to_numpy(dtype=np.float32)
    times = development["origin_time_h"].to_numpy(dtype=float)
    split_rows = {"train": (0, train_end), "validation": (train_end, validation_end)}
    max_splits = {
        name: _max_history_windows(values, values[:, 0], times, start, end)
        for name, (start, end) in split_rows.items()
    }
    splits = {name: _slice_max_split(max_split, history_hours) for name, max_split in max_splits.items()}
    inputs = {name: split.X_raw.copy() for name, split in splits.items()}
    anchors = {name: split.X_raw[:, -1, 0, None].copy() for name, split in splits.items()}
    residuals = {name: (split.y_raw - anchors[name]).astype(np.float32) for name, split in splits.items()}
    input_scaler = StandardScaler().fit(inputs["train"].reshape(-1, 1))
    residual_scaler = StandardScaler().fit(residuals["train"].reshape(-1, 1))
    return MatchedHistoryFoldData(
        fc, config, fold, total_rows, development_end, development, split_rows,
        max_splits, splits, inputs, anchors, residuals, input_scaler, residual_scaler,
    )


def _fit_residual(data: MatchedHistoryFoldData) -> tuple[np.ndarray, list[dict[str, float]], MIMOLSTM]:
    config = TrainConfig()
    set_seed(SEED)
    model, history = _fit_model(
        data.transformed_inputs("train"), data.transformed_residuals("train"),
        data.transformed_inputs("validation"), data.transformed_residuals("validation"),
        input_size=1, horizon=HORIZON_HOURS, config=config,
    )
    prediction = _predict(model, data.transformed_inputs("validation"), config.batch_size, config.device)
    return data.restore_prediction(prediction, "validation"), history, model


def _baselines(data: MatchedHistoryFoldData) -> dict[str, np.ndarray]:
    inputs = data.inputs_raw["validation"]
    return {
        "persistence": persistence_forecast(inputs, 0, HORIZON_HOURS),
        "linear_trend": linear_trend_forecast(inputs, 0, HORIZON_HOURS),
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


def _metadata(data: MatchedHistoryFoldData, output_dir: Path) -> dict:
    return {
        "phase": "3C-R",
        "fc": data.fc,
        "seed": SEED,
        "history_hours": data.config.history_hours,
        "max_history_hours_for_matched_origins": MAX_HISTORY_HOURS,
        "horizon_hours": HORIZON_HOURS,
        "feature_set": ["Utot_mean"],
        "input_mode": "absolute_history_only",
        "target": "Utot(t+h)-observed_Utot(t), restored with the legal origin anchor",
        "matching_strategy": "Generate legal origins once with Nmax=48; N=15/24 use only the last N timestamps of each same Nmax window.",
        "split_rows": {name: list(bounds) for name, bounds in data.split_rows.items()},
        "development_row_range": [0, data.development_end_row],
        "final_test_row_range": [data.development_end_row, data.total_hourly_rows],
        "test_partition_accessed": False,
        "input_scaler_fit": "flattened current-fold train windows only",
        "residual_scaler_fit": "flattened current-fold train residual labels only",
        "train_config": asdict(TrainConfig()),
        "output_dir": str(output_dir.resolve()),
    }


def _prediction_frame(data: MatchedHistoryFoldData, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    validation = data.splits["validation"]
    rows = []
    for sample in range(len(validation.origins)):
        for horizon in range(HORIZON_HOURS):
            row = {
                "sample": sample,
                "origin_row": int(validation.origins[sample]),
                "origin_time_h": float(validation.feature_times[sample, -1]),
                "target_time_h": float(validation.target_times[sample, horizon]),
                "horizon": horizon + 1,
                "actual_Utot": float(validation.y_raw[sample, horizon]),
            }
            row.update({f"pred_{name}": float(value[sample, horizon]) for name, value in predictions.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _write_run(data: MatchedHistoryFoldData, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from phase3cr_audit import write_phase3cr_audit

    write_phase3cr_audit(data, output_dir)
    residual_prediction, history, model = _fit_residual(data)
    predictions = _baselines(data)
    predictions["residual_mimo_lstm"] = residual_prediction
    metrics = _metric_rows(data.splits["validation"].y_raw, predictions)
    metrics.to_csv(output_dir / "validation_metrics_per_horizon.csv", index=False)
    frame = _prediction_frame(data, predictions)
    frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    torch.save(model.state_dict(), output_dir / "residual_mimo_lstm.pt")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "pipeline_metadata.json").write_text(json.dumps(_metadata(data, output_dir), indent=2), encoding="utf-8")
    return metrics, frame


def _history_summary(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual = table.loc[(table["model"] == "residual_mimo_lstm") & (table["horizon"] != "mean_h1_h24")].copy()
    residual["horizon"] = residual["horizon"].astype(int)
    raw = residual.assign(positive=residual["Skill_Best"] > 0)
    per_horizon = raw.groupby(["fc", "N", "horizon"], sort=False).agg(
        RMSE_mean=("RMSE", "mean"),
        Skill_Best_mean=("Skill_Best", "mean"),
        Skill_Best_std=("Skill_Best", "std"),
        positive_folds=("positive", "sum"),
    ).reset_index()
    rows = []
    for (fc, history), group in raw.groupby(["fc", "N"], sort=False):
        stable = per_horizon.loc[(per_horizon["fc"] == fc) & (per_horizon["N"] == history)]
        stable_horizons = stable.loc[stable["positive_folds"] == 3, "horizon"].astype(int).tolist()
        flags = set(stable_horizons)
        crossover = next((h for h in range(1, HORIZON_HOURS - 1) if {h, h + 1, h + 2}.issubset(flags)), None)
        rows.append({
            "fc": fc,
            "N": int(history),
            "MeanSkill_all": float(group["Skill_Best"].mean()),
            "MeanSkill_late_h13_h24": float(group.loc[group["horizon"] >= 13, "Skill_Best"].mean()),
            "stable_positive_horizon_count": len(stable_horizons),
            "stable_positive_horizons": ", ".join(map(str, stable_horizons)) if stable_horizons else "none",
            "crossover_h_c": crossover if crossover is not None else "none",
        })
    return pd.DataFrame(rows), per_horizon


def _is_recoverable_audit_only_run(output_dir: Path) -> bool:
    """Allow only a fail-closed audit-only restart; never replace completed results."""
    if not output_dir.exists() or (output_dir / "result.json").exists():
        return False
    files = [path.name for path in output_dir.rglob("*") if path.is_file()]
    allowed_audit_files = {
        "leakage_audit.json", "leakage_audit.md",
        "fold_1.json", "fold_1.md", "fold_2.json", "fold_2.md", "fold_3.json", "fold_3.md",
    }
    return bool(files) and set(files).issubset(allowed_audit_files)


def run_phase3cr(output_root: Path, data_root: Path | None = None) -> dict:
    """Run the frozen matched-origin development diagnostic, without a final test."""
    output_dir = output_root / "seed_42"
    if output_dir.exists():
        if not _is_recoverable_audit_only_run(output_dir):
            raise FileExistsError(f"Refusing to overwrite Phase-3C-R output: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    all_metrics, all_predictions, sample_counts = [], [], []
    for fc in ("FC1", "FC2"):
        for fold in FOLD_SPECS:
            data_by_history = {
                history: build_matched_history_fold(fc, fold, history, data_root)
                for history in HISTORY_LENGTHS
            }
            from phase3cr_audit import write_matched_origin_audit
            write_matched_origin_audit(data_by_history, output_dir / fc / "matched_origin_audits" / f"{fold.name}.json")
            for history, data in data_by_history.items():
                run_dir = output_dir / fc / f"N_{history}" / fold.name
                if run_dir.exists() and not _is_recoverable_audit_only_run(run_dir):
                    raise FileExistsError(f"Refusing to replace non-audit Phase-3C-R run data: {run_dir}")
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics, predictions = _write_run(data, run_dir)
                metrics.insert(0, "fold", fold.name)
                metrics.insert(0, "N", history)
                metrics.insert(0, "fc", fc)
                predictions.insert(0, "fold", fold.name)
                predictions.insert(0, "N", history)
                predictions.insert(0, "fc", fc)
                all_metrics.append(metrics)
                all_predictions.append(predictions)
                sample_counts.append({
                    "fc": fc, "fold": fold.name, "N": history,
                    "train_windows": len(data.splits["train"].origins),
                    "validation_windows": len(data.splits["validation"].origins),
                })
    metric_table = pd.concat(all_metrics, ignore_index=True)
    prediction_table = pd.concat(all_predictions, ignore_index=True)
    metric_table.to_csv(output_dir / "all_validation_metrics.csv", index=False)
    prediction_table.to_csv(output_dir / "all_validation_predictions.csv", index=False)
    counts = pd.DataFrame(sample_counts)
    counts.to_csv(output_dir / "matched_origin_window_counts.csv", index=False)
    summary, per_horizon = _history_summary(metric_table)
    summary.to_csv(output_dir / "history_length_summary.csv", index=False)
    per_horizon.to_csv(output_dir / "history_length_skill_by_horizon.csv", index=False)
    result = {
        "phase": "3C-R",
        "seed": SEED,
        "stacks": ["FC1", "FC2"],
        "H": HORIZON_HOURS,
        "Nmax": MAX_HISTORY_HOURS,
        "history_lengths": list(HISTORY_LENGTHS),
        "folds": [asdict(fold) for fold in FOLD_SPECS],
        "test_partition_accessed": False,
        "stop_boundary": "No final test, relative input, matrix-memory encoder, multi-scale model, uncertainty model, multi-seed run, or RUL experiment was performed.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

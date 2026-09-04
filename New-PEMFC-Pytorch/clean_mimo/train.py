"""Run the Phase-1 baselines under one fixed, strictly causal protocol."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from data_pipeline import PreparedData, prepare_data
from evaluate import (
    linear_trend_forecast,
    metric_rows,
    persistence_forecast,
    prediction_frame,
    save_prediction_curve,
)
from leakage_audit import write_audit
from models import MIMOLSTM


@dataclass(frozen=True)
class TrainConfig:
    hidden_size: int = 64
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 300
    early_stopping_patience: int = 35
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    input_size: int,
    horizon: int,
    config: TrainConfig,
) -> Tuple[MIMOLSTM, List[Dict[str, float]]]:
    device = torch.device(config.device)
    model = MIMOLSTM(input_size, horizon, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    train_loader = _loader(X_train, y_train, config.batch_size, shuffle=True)
    validation_loader = _loader(X_validation, y_validation, config.batch_size, shuffle=False)
    best_state, best_validation, stale = None, float("inf"), 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        training_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for x, y in validation_loader:
                validation_losses.append(float(criterion(model(x.to(device)), y.to(device)).cpu()))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_MSE": float(np.mean(training_losses)), "validation_MSE": validation_loss})
        if validation_loss < best_validation - 1e-10:
            best_validation, stale = validation_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def _predict(model: MIMOLSTM, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    values = []
    loader = _loader(X, np.zeros((len(X), model.head.out_features), dtype=np.float32), batch_size, shuffle=False)
    with torch.no_grad():
        for x, _ in loader:
            values.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(values, axis=0)


def _inverse_targets(data: PreparedData, values: np.ndarray) -> np.ndarray:
    return data.y_scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def _run_single_model(
    data: PreparedData,
    model_name: str,
    seed: int,
    train_config: TrainConfig,
    residual: bool,
) -> Tuple[np.ndarray, List[Dict[str, float]], MIMOLSTM]:
    """Train direct voltage MIMO or the legal Utot(t+h)-Utot(t) variant."""
    set_seed(seed)
    X_train, y_train = data.transformed("train")
    X_validation, y_validation = data.transformed("validation")
    X_test, _ = data.transformed("test")
    utot_idx = data.feature_names.index(data.config.target_name)
    residual_scaler = None
    if residual:
        train_base = data.splits["train"].X_raw[:, -1, utot_idx, None]
        validation_base = data.splits["validation"].X_raw[:, -1, utot_idx, None]
        residual_scaler = StandardScaler().fit((data.splits["train"].y_raw - train_base).reshape(-1, 1))
        y_train = residual_scaler.transform((data.splits["train"].y_raw - train_base).reshape(-1, 1)).reshape(y_train.shape)
        y_validation = residual_scaler.transform((data.splits["validation"].y_raw - validation_base).reshape(-1, 1)).reshape(y_validation.shape)
    model, history = _fit_model(
        X_train, y_train, X_validation, y_validation,
        input_size=X_train.shape[-1], horizon=data.config.horizon_hours, config=train_config,
    )
    pred_scaled = _predict(model, X_test, train_config.batch_size, train_config.device)
    if residual:
        test_base = data.splits["test"].X_raw[:, -1, utot_idx, None]
        prediction = residual_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape) + test_base
    else:
        prediction = _inverse_targets(data, pred_scaled)
    return prediction.astype(np.float32), history, model


def run_seed(data: PreparedData, seed: int, output_dir: Path, train_config: TrainConfig, include_residual: bool) -> Dict[str, object]:
    """Run two analytical baselines and one/optionally two neural MIMO models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_audit(data, output_dir)
    test = data.splits["test"]
    utot_idx = data.feature_names.index(data.config.target_name)
    persistence = persistence_forecast(test.X_raw, utot_idx, data.config.horizon_hours)
    linear = linear_trend_forecast(test.X_raw, utot_idx, data.config.horizon_hours)
    predictions: Dict[str, np.ndarray] = {"persistence": persistence, "linear_trend": linear}
    metrics = [
        metric_rows(test.y_raw, persistence, "persistence", persistence),
        metric_rows(test.y_raw, linear, "linear_trend", persistence),
    ]
    histories = {}
    vanilla, history, model = _run_single_model(data, "mimo_lstm", seed, train_config, residual=False)
    predictions["mimo_lstm"] = vanilla
    metrics.append(metric_rows(test.y_raw, vanilla, "mimo_lstm", persistence))
    histories["mimo_lstm"] = history
    torch.save(model.state_dict(), output_dir / "mimo_lstm.pt")
    if include_residual:
        residual, residual_history, residual_model = _run_single_model(data, "residual_mimo_lstm", seed, train_config, residual=True)
        predictions["residual_mimo_lstm"] = residual
        metrics.append(metric_rows(test.y_raw, residual, "residual_mimo_lstm", persistence))
        histories["residual_mimo_lstm"] = residual_history
        torch.save(residual_model.state_dict(), output_dir / "residual_mimo_lstm.pt")

    metric_table = pd.concat(metrics, ignore_index=True)
    metric_table.to_csv(output_dir / "metrics_per_horizon.csv", index=False)
    frame = prediction_frame(test.feature_times[:, -1], test.target_times, test.y_raw, predictions)
    frame.to_csv(output_dir / "test_predictions.csv", index=False)
    save_prediction_curve(frame, output_dir, "mimo_lstm")
    if include_residual:
        save_prediction_curve(frame, output_dir, "residual_mimo_lstm")
    (output_dir / "training_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    summary = {
        "seed": seed,
        "fc": data.fc,
        "train_config": asdict(train_config),
        "include_residual": include_residual,
        "metrics": metric_table.to_dict(orient="records"),
        "best_validation_MSE": {name: min(item["validation_MSE"] for item in history) for name, history in histories.items()},
        "selection_statement": "This run uses validation-loss early stopping only. No test-result ranking, Top-K selection, or ensemble is used.",
    }
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_seeds(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one seed is required")
    return values


def run_experiment(fc: str, output_root: Path, seeds: Iterable[int], data_root: Path | None = None, include_residual: bool = False) -> pd.DataFrame:
    data = prepare_data(fc, data_root)
    train_config = TrainConfig()
    run_summaries = []
    for seed in seeds:
        result = run_seed(data, int(seed), output_root / fc / f"seed_{seed}", train_config, include_residual)
        for row in result["metrics"]:
            row = dict(row)
            row["seed"] = seed
            run_summaries.append(row)
    all_metrics = pd.DataFrame(run_summaries)
    all_metrics.to_csv(output_root / fc / "all_seed_metrics.csv", index=False)
    if len(set(seeds)) > 1:
        numeric = ["MAE", "RMSE", "R2", "RMSE_improvement_vs_persistence", "MSE_skill_vs_persistence"]
        summary = all_metrics.groupby(["model", "horizon"], dropna=False)[numeric].agg(["min", "mean", "max", "std", "median"])
        summary.columns = [f"{left}_{right}" for left, right in summary.columns]
        summary.reset_index().to_csv(output_root / fc / "seed_stability_summary.csv", index=False)
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 clean causal N=15/H=4 MIMO experiment.")
    parser.add_argument("--fc", choices=("FC1", "FC2"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds. Start with one; use ten only after review.")
    parser.add_argument("--include-residual", action="store_true", help="Run the optional legal residual MIMO model after vanilla MIMO is working.")
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    table = run_experiment(args.fc, args.output_root, seeds, args.data_root, args.include_residual)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

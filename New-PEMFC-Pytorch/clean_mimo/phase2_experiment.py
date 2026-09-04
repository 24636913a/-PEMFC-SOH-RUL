"""Phase-2 causal absolute/residual MIMO runner.

This module is intentionally separate from the Phase-1 runner so Phase-1
artifacts remain immutable.  It reuses only the Phase-1 raw causal windows and
plain one-layer MIMO-LSTM architecture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal
import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from data_pipeline import PreparedData, prepare_data
from evaluate import (
    linear_trend_forecast,
    metric_rows,
    persistence_forecast,
    prediction_frame,
    save_prediction_curve,
)
from models import MIMOLSTM
from train import TrainConfig, _fit_model, _predict, set_seed


InputMode = Literal["absolute_inputs", "relative_voltage_inputs"]
FeatureSet = Literal["utot_only", "voltage_state", "state_load", "full_26"]

VOLTAGE_FEATURES = ("Utot_mean", "U1_mean", "U2_mean", "U3_mean", "U4_mean", "U5_mean")
FEATURE_LADDER: Dict[str, tuple[str, ...]] = {
    "utot_only": ("Utot_mean",),
    "voltage_state": VOLTAGE_FEATURES,
    "state_load": VOLTAGE_FEATURES + ("J_mean", "I_mean", "I_std", "I_p2p"),
    "full_26": (),  # Resolved to the exact Phase-1 list for a fail-closed equality check.
}


@dataclass
class Phase2Data:
    """Causal Phase-2 representations plus scalers with explicit fit scope."""

    base: PreparedData
    feature_set: FeatureSet
    input_mode: InputMode
    feature_names: list[str]
    voltage_feature_indices: np.ndarray
    inputs_raw: Dict[str, np.ndarray]
    inputs_repr: Dict[str, np.ndarray]
    anchors: Dict[str, np.ndarray]
    residual_labels: Dict[str, np.ndarray]
    X_scaler: StandardScaler
    absolute_target_scaler: StandardScaler
    residual_target_scaler: StandardScaler

    def transformed_inputs(self, split: str) -> np.ndarray:
        value = self.inputs_repr[split]
        n, history, features = value.shape
        return self.X_scaler.transform(value.reshape(n * history, features)).reshape(value.shape).astype(np.float32)

    def transformed_labels(self, split: str, formulation: Literal["absolute", "residual"]) -> np.ndarray:
        labels = self.base.splits[split].y_raw if formulation == "absolute" else self.residual_labels[split]
        scaler = self.absolute_target_scaler if formulation == "absolute" else self.residual_target_scaler
        return scaler.transform(labels.reshape(-1, 1)).reshape(labels.shape).astype(np.float32)

    def inverse_prediction(self, scaled: np.ndarray, formulation: Literal["absolute", "residual"], split: str) -> np.ndarray:
        scaler = self.absolute_target_scaler if formulation == "absolute" else self.residual_target_scaler
        prediction = scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)
        if formulation == "residual":
            prediction = prediction + self.anchors[split]
        return prediction.astype(np.float32)


def resolve_feature_names(data: PreparedData, feature_set: FeatureSet) -> list[str]:
    if feature_set not in FEATURE_LADDER:
        raise ValueError(f"Unknown feature set: {feature_set}")
    selected = list(data.feature_names if feature_set == "full_26" else FEATURE_LADDER[feature_set])
    missing = [name for name in selected if name not in data.feature_names]
    if missing:
        raise KeyError(f"Phase-2 feature set cannot be resolved from Phase-1 data: {missing}")
    if feature_set == "full_26" and selected != data.feature_names:
        raise AssertionError("R4 must be exactly the Phase-1 full-26 input list")
    return selected


def build_phase2_data(
    fc: str,
    feature_set: FeatureSet = "full_26",
    input_mode: InputMode = "absolute_inputs",
    data_root: Path | None = None,
) -> Phase2Data:
    """Build legal feature representations and train-only absolute/residual scalers."""
    if input_mode not in {"absolute_inputs", "relative_voltage_inputs"}:
        raise ValueError(f"Unknown input mode: {input_mode}")
    base = prepare_data(fc, data_root)
    names = resolve_feature_names(base, feature_set)
    indices = np.asarray([base.feature_names.index(name) for name in names], dtype=np.int64)
    voltage_indices = np.asarray([idx for idx, name in enumerate(names) if name in VOLTAGE_FEATURES], dtype=np.int64)
    origin_utot_index = base.feature_names.index(base.config.target_name)
    inputs_raw: Dict[str, np.ndarray] = {}
    inputs_repr: Dict[str, np.ndarray] = {}
    anchors: Dict[str, np.ndarray] = {}
    residuals: Dict[str, np.ndarray] = {}

    for split_name, split in base.splits.items():
        raw = split.X_raw[:, :, indices].copy()
        anchor = split.X_raw[:, -1, origin_utot_index].reshape(-1, 1).copy()
        representation = raw.copy()
        if input_mode == "relative_voltage_inputs":
            # The reference is the last historical row, never a target-time row.
            for idx in voltage_indices:
                representation[:, :, idx] = representation[:, :, idx] - raw[:, -1, idx, None]
        inputs_raw[split_name] = raw.astype(np.float32)
        inputs_repr[split_name] = representation.astype(np.float32)
        anchors[split_name] = anchor.astype(np.float32)
        residuals[split_name] = (split.y_raw - anchor).astype(np.float32)

    train_repr = inputs_repr["train"]
    X_scaler = StandardScaler().fit(train_repr.reshape(-1, len(names)))
    absolute_target_scaler = StandardScaler().fit(base.splits["train"].y_raw.reshape(-1, 1))
    residual_target_scaler = StandardScaler().fit(residuals["train"].reshape(-1, 1))
    return Phase2Data(
        base=base,
        feature_set=feature_set,
        input_mode=input_mode,
        feature_names=names,
        voltage_feature_indices=voltage_indices,
        inputs_raw=inputs_raw,
        inputs_repr=inputs_repr,
        anchors=anchors,
        residual_labels=residuals,
        X_scaler=X_scaler,
        absolute_target_scaler=absolute_target_scaler,
        residual_target_scaler=residual_target_scaler,
    )


def phase2_metadata(data: Phase2Data, output_dir: Path) -> dict:
    return {
        "phase": 2,
        "fc": data.base.fc,
        "feature_set": data.feature_set,
        "input_mode": data.input_mode,
        "feature_names": data.feature_names,
        "history_hours": data.base.config.history_hours,
        "horizon_hours": data.base.config.horizon_hours,
        "split_rows": {name: list(bounds) for name, bounds in data.base.split_rows.items()},
        "window_counts": {name: len(split.X_raw) for name, split in data.base.splits.items()},
        "input_scaler_fit": "flattened train representations only",
        "absolute_target_scaler_fit": "flattened train absolute labels only",
        "residual_target_scaler_fit": "flattened train residual labels only",
        "residual_definition": "Utot(t+h) - observed Utot(t), h=1..4",
        "anchor_definition": "last historical Utot_mean at the prediction origin",
        "output_dir": str(output_dir.resolve()),
    }


def _train_formulation(
    data: Phase2Data,
    formulation: Literal["absolute", "residual"],
    seed: int,
    config: TrainConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], MIMOLSTM]:
    set_seed(seed)
    X_train = data.transformed_inputs("train")
    X_validation = data.transformed_inputs("validation")
    X_test = data.transformed_inputs("test")
    y_train = data.transformed_labels("train", formulation)
    y_validation = data.transformed_labels("validation", formulation)
    model, history = _fit_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        input_size=len(data.feature_names),
        horizon=data.base.config.horizon_hours,
        config=config,
    )
    validation_scaled = _predict(model, X_validation, config.batch_size, config.device)
    test_scaled = _predict(model, X_test, config.batch_size, config.device)
    return (
        data.inverse_prediction(validation_scaled, formulation, "validation"),
        data.inverse_prediction(test_scaled, formulation, "test"),
        history,
        model,
    )


def _analytical_predictions(data: Phase2Data, split: str) -> dict[str, np.ndarray]:
    base_split = data.base.splits[split]
    index = data.base.feature_names.index(data.base.config.target_name)
    persistence = persistence_forecast(base_split.X_raw, index, data.base.config.horizon_hours)
    return {
        "persistence": persistence,
        "linear_trend": linear_trend_forecast(base_split.X_raw, index, data.base.config.horizon_hours),
    }


def _metrics(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    persistence = predictions["persistence"]
    return pd.concat(
        [metric_rows(y_true, prediction, name, persistence) for name, prediction in predictions.items()],
        ignore_index=True,
    )


def run_phase2_seed42(
    fc: str,
    output_root: Path,
    feature_set: FeatureSet = "full_26",
    input_mode: InputMode = "absolute_inputs",
    data_root: Path | None = None,
    seed: int = 42,
) -> dict:
    """Run the predeclared seed-42 absolute/residual comparison once.

    Existing output directories are refused so reruns cannot silently overwrite
    a result that has already been inspected.
    """
    data = build_phase2_data(fc, feature_set, input_mode, data_root)
    output_dir = output_root / fc / f"seed_{seed}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-2 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    train_config = TrainConfig()
    from phase2_audit import write_phase2_audit  # avoid circular import during module loading

    write_phase2_audit(data, output_dir)
    validation_predictions = _analytical_predictions(data, "validation")
    test_predictions = _analytical_predictions(data, "test")
    absolute_validation, absolute_test, absolute_history, absolute_model = _train_formulation(
        data, "absolute", seed, train_config
    )
    residual_validation, residual_test, residual_history, residual_model = _train_formulation(
        data, "residual", seed, train_config
    )
    validation_predictions.update({"absolute_mimo_lstm": absolute_validation, "residual_mimo_lstm": residual_validation})
    test_predictions.update({"absolute_mimo_lstm": absolute_test, "residual_mimo_lstm": residual_test})

    validation_metrics = _metrics(data.base.splits["validation"].y_raw, validation_predictions)
    test_metrics = _metrics(data.base.splits["test"].y_raw, test_predictions)
    validation_metrics.to_csv(output_dir / "validation_metrics_per_horizon.csv", index=False)
    test_metrics.to_csv(output_dir / "metrics_per_horizon.csv", index=False)

    test_split = data.base.splits["test"]
    frame = prediction_frame(test_split.feature_times[:, -1], test_split.target_times, test_split.y_raw, test_predictions)
    frame.to_csv(output_dir / "test_predictions.csv", index=False)
    save_prediction_curve(frame, output_dir, "absolute_mimo_lstm")
    save_prediction_curve(frame, output_dir, "residual_mimo_lstm")
    torch.save(absolute_model.state_dict(), output_dir / "absolute_mimo_lstm.pt")
    torch.save(residual_model.state_dict(), output_dir / "residual_mimo_lstm.pt")
    histories = {"absolute_mimo_lstm": absolute_history, "residual_mimo_lstm": residual_history}
    (output_dir / "training_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    metadata = phase2_metadata(data, output_dir)
    metadata["train_config"] = asdict(train_config)
    (output_dir / "pipeline_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = {
        "phase": 2,
        "seed": seed,
        "fc": fc,
        "feature_set": feature_set,
        "input_mode": input_mode,
        "train_config": asdict(train_config),
        "test_metrics": test_metrics.to_dict(orient="records"),
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "best_validation_MSE": {
            "absolute_mimo_lstm": min(row["validation_MSE"] for row in absolute_history),
            "residual_mimo_lstm": min(row["validation_MSE"] for row in residual_history),
        },
        "selection_statement": "This is the predeclared seed-42 diagnostic. No test-based ranking, model selection, Top-K choice, or ensemble was performed.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

"""Execute the single authorized FC2 R3 final-test evaluation exactly once."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from evaluate import linear_trend_forecast, persistence_forecast, prediction_frame, save_prediction_curve
from phase2_experiment import build_phase2_data
from train import TrainConfig, _fit_model, _predict, set_seed


CLEAN_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = CLEAN_DIR / "outputs_phase2b_final_test"
COMPLETION_MARKER = CLEAN_DIR / "FROZEN_R3_FINAL_TEST_COMPLETED.json"
SEED = 42


def _metric_rows(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    persistence = predictions["persistence"]
    linear = predictions["linear_trend"]
    rows = []
    for model, prediction in predictions.items():
        for h in range(y_true.shape[1]):
            rmse = mean_squared_error(y_true[:, h], prediction[:, h]) ** 0.5
            persistence_rmse = mean_squared_error(y_true[:, h], persistence[:, h]) ** 0.5
            linear_rmse = mean_squared_error(y_true[:, h], linear[:, h]) ** 0.5
            rows.append({
                "model": model,
                "horizon": h + 1,
                "MAE": float(mean_absolute_error(y_true[:, h], prediction[:, h])),
                "RMSE": float(rmse),
                "R2": float(r2_score(y_true[:, h], prediction[:, h])),
                "persistence_RMSE": float(persistence_rmse),
                "linear_RMSE": float(linear_rmse),
                "Skill_P": float(1.0 - rmse / persistence_rmse) if persistence_rmse else np.nan,
                "Skill_L": float(1.0 - rmse / linear_rmse) if linear_rmse else np.nan,
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


def run_once(data_root: Path | None = None) -> dict:
    """Run locked R3 once, then permanently mark the completed evaluation."""
    if COMPLETION_MARKER.exists():
        raise RuntimeError(f"Frozen R3 final test already completed; repeat evaluation is blocked: {COMPLETION_MARKER}")
    output_dir = OUTPUT_ROOT / "FC2" / f"seed_{SEED}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen final-test output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    data = build_phase2_data("FC2", "state_load", "absolute_inputs", data_root)
    from phase2b_final_audit import write_frozen_r3_audit
    write_frozen_r3_audit(data, output_dir, COMPLETION_MARKER)

    config = TrainConfig()
    set_seed(SEED)
    model, history = _fit_model(
        data.transformed_inputs("train"),
        data.transformed_labels("train", "residual"),
        data.transformed_inputs("validation"),
        data.transformed_labels("validation", "residual"),
        input_size=len(data.feature_names),
        horizon=data.base.config.horizon_hours,
        config=config,
    )
    test_scaled = _predict(model, data.transformed_inputs("test"), config.batch_size, config.device)
    residual_prediction = data.inverse_prediction(test_scaled, "residual", "test")
    test_split = data.base.splits["test"]
    utot_index = data.base.feature_names.index(data.base.config.target_name)
    predictions = {
        "persistence": persistence_forecast(test_split.X_raw, utot_index, data.base.config.horizon_hours),
        "linear_trend": linear_trend_forecast(test_split.X_raw, utot_index, data.base.config.horizon_hours),
        "R3_residual_mimo_lstm": residual_prediction,
    }
    metrics = _metric_rows(test_split.y_raw, predictions)
    metrics.to_csv(output_dir / "metrics_per_horizon.csv", index=False)
    frame = prediction_frame(test_split.feature_times[:, -1], test_split.target_times, test_split.y_raw, predictions)
    frame.to_csv(output_dir / "test_predictions.csv", index=False)
    save_prediction_curve(frame, output_dir, "R3_residual_mimo_lstm")
    torch.save(model.state_dict(), output_dir / "R3_residual_mimo_lstm.pt")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = {
        "phase": "2b_frozen_final_test",
        "fc": "FC2",
        "seed": SEED,
        "feature_set": "R3_state_load",
        "feature_names": data.feature_names,
        "input_mode": "absolute_inputs",
        "history_hours": data.base.config.history_hours,
        "horizon_hours": data.base.config.horizon_hours,
        "split_rows": {name: list(bounds) for name, bounds in data.base.split_rows.items()},
        "window_counts": {name: len(split.X_raw) for name, split in data.base.splits.items()},
        "residual_definition": "Utot(t+h) - observed Utot(t), h=1..4",
        "input_scaler_fit": "flattened train representations only",
        "residual_scaler_fit": "flattened train residual labels only",
        "train_config": asdict(config),
        "single_run_guard": str(COMPLETION_MARKER.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "pipeline_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = {
        "phase": "2b_frozen_final_test",
        "fc": "FC2",
        "seed": SEED,
        "feature_set": "R3_state_load",
        "test_evaluation_count": 1,
        "selection_statement": "R3 was frozen by Phase-2b validation-only mean RMSE. This is the single authorized final 40% test evaluation.",
        "metrics": metrics.to_dict(orient="records"),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    COMPLETION_MARKER.write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "fc": "FC2",
        "seed": SEED,
        "feature_set": "R3_state_load",
        "output_dir": str(output_dir.resolve()),
        "test_evaluation_count": 1,
    }, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run_once()
    print(f"Frozen R3 final test complete exactly once; rows={len(result['metrics'])}")

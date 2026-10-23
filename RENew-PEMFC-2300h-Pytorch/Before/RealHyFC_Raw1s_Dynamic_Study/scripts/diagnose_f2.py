"""CPU-only post-completion F2 gate and recorded convergence diagnostics.

Never calls a full TFT forward, an optimizer, backward, or a CUDA API.  Uses
only frozen TRAIN/VALIDATION origins and completed checkpoint/log artifacts.
This diagnostic cannot change training or justify extra runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

from audit_raw import STUDY, OUT, save, sha
from freeze_protocol import load_cache
from models import MODEL_PARAMETERS, REGIME_INDICES, _native_classes, source_identity


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    assert len(values) and np.isfinite(values).all()
    quantiles = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    names = ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"]
    result = {name: float(value) for name, value in zip(names, quantiles)}
    result.update(n=len(values), mean=float(values.mean()), std=float(values.std()),
                  saturation_fraction_lt_0p05=float((values < .05).mean()),
                  saturation_fraction_gt_0p95=float((values > .95).mean()),
                  extreme_fraction_lt_0p01=float((values < .01).mean()),
                  extreme_fraction_gt_0p99=float((values > .99).mean()),
                  mean_absolute_change_from_initial_half=float(np.mean(np.abs(values - .5))))
    return result


def completed_run(kind: str, horizon: int, config: dict) -> tuple[Path, dict, pd.DataFrame]:
    run = STUDY / f"runs/H{horizon}_{kind}_Seed202"
    state = json.loads((run / "training_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "COMPLETE", f"Refuse diagnostic on incomplete {run.name}"
    assert state["protocol_sha256"] == sha(OUT / "frozen_protocol.json")
    assert state["seed"] == config["seed"] == 202 and state["horizon"] == horizon
    assert sha(run / "best.pt") == state["checkpoint_sha256"]
    history = pd.read_csv(run / "training_history.csv")
    assert len(history) == state["stop_epoch"] and np.isfinite(
        history[["train_standardized_MSE", "validation_RMSE_V"]].to_numpy()).all()
    assert history.epoch.tolist() == list(range(1, len(history) + 1))
    assert int(history.iloc[-1].optimizer_steps) == state["optimizer_steps"] > 0
    return run, state, history


def curve_evidence(run: Path, state: dict, history: pd.DataFrame, config: dict) -> dict:
    train = history.train_standardized_MSE.to_numpy(dtype=float)
    validation = history.validation_RMSE_V.to_numpy(dtype=float)
    best = int(state["best_epoch"])
    assert np.isclose(validation[best - 1], state["best_validation_RMSE_V"], rtol=1e-12)
    tail = history.tail(min(3, len(history)))
    slope = float(np.polyfit(tail.epoch, tail.train_standardized_MSE, 1)[0]) if len(tail) > 1 else None
    return {
        "run": str(run.relative_to(STUDY)),
        "training_state_sha256": sha(run / "training_state.json"),
        "training_history_sha256": sha(run / "training_history.csv"),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "trainable_parameters_from_manifest": state["parameters"],
        "optimizer_steps": state["optimizer_steps"],
        "stop_epoch": state["stop_epoch"], "best_epoch": best,
        "budget_max_epochs": config["training"]["max_epochs"],
        "reached_epoch_budget": state["stop_epoch"] == config["training"]["max_epochs"],
        "best_is_final_epoch": best == state["stop_epoch"],
        "first_train_standardized_MSE": float(train[0]),
        "final_train_standardized_MSE": float(train[-1]),
        "minimum_train_standardized_MSE": float(train.min()),
        "train_standardized_MSE_at_selected_epoch": float(train[best - 1]),
        "first_validation_RMSE_V": float(validation[0]),
        "best_validation_RMSE_V": state["best_validation_RMSE_V"],
        "final_validation_RMSE_V": float(validation[-1]),
        "last_up_to_three_epoch_train_MSE_slope": slope,
        "final_vs_first_train_MSE_change": float(train[-1] - train[0]),
        "zero_optimizer_step_startup_failures": state.get("zero_optimizer_step_startup_failures", 0),
        "recorded_curve": history[["epoch", "train_standardized_MSE", "validation_RMSE_V",
                                    "best_epoch", "optimizer_steps"]].to_dict("records"),
    }


def diagnose(horizon: int) -> dict:
    output = OUT / f"f2_diagnostics_H{horizon}.json"
    assert not output.exists(), "Refuse to overwrite recorded post-completion diagnostics"
    config = json.loads((OUT / "frozen_protocol.json").read_text(encoding="utf-8"))
    origins_file = OUT / f"origins_H{horizon}.npz"
    assert sha(origins_file) == config["sampling_hashes"][str(horizon)]
    assert config["history_steps"] == 300 and config["n_features"] == 24
    runs = {kind: completed_run(kind, horizon, config) for kind in ("LSTM", "F2")}
    f2_run, f2_state, _ = runs["F2"]
    checkpoint = torch.load(f2_run / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == f2_state["best_epoch"]
    assert checkpoint["seed"] == 202 and checkpoint["horizon"] == horizon
    assert checkpoint["protocol_sha256"] == sha(OUT / "frozen_protocol.json")
    tensors = checkpoint["state_dict"]
    assert all(t.device.type == "cpu" and bool(torch.isfinite(t).all())
               for t in tensors.values() if torch.is_tensor(t))
    # Instantiate only the 57-parameter native gate, not the full F2 network.
    gate = _native_classes(24).RegimeFusionGate().cpu().eval()
    prefix = "native.regime_gate."
    gate_state = {name[len(prefix):]: value for name, value in tensors.items()
                  if name.startswith(prefix)}
    gate.load_state_dict(gate_state, strict=True)
    assert sum(p.numel() for p in gate.parameters()) == 57
    output_weight = gate.output.weight.detach().numpy()
    output_bias = gate.output.bias.detach().numpy()
    parameter_evidence = {
        "gate_parameters": 57,
        "original_output_weight_and_bias_are_exact_zeros": True,
        "output_weight_nonzero_elements": int(np.count_nonzero(output_weight)),
        "output_bias_nonzero_elements": int(np.count_nonzero(output_bias)),
        "output_weight_L2_change_from_initial_zero": float(np.linalg.norm(output_weight)),
        "output_bias_L2_change_from_initial_zero": float(np.linalg.norm(output_bias)),
        "output_weight_or_bias_updated": bool(np.any(output_weight != 0) or np.any(output_bias != 0)),
        "hidden_weight_L2_selected_checkpoint": float(gate.hidden.weight.detach().norm()),
        "hidden_bias_L2_selected_checkpoint": float(gate.hidden.bias.detach().norm()),
        "hidden_parameters_initial_update_not_inferred": "The initial full gate hidden tensor was not separately checkpointed; aggregate initialization hashes cannot establish individual hidden-parameter movement.",
        "state_dict_tensor_elements_not_trainable_parameter_claim": sum(
            int(t.numel()) for t in tensors.values() if torch.is_tensor(t)),
        "all_checkpoint_tensors_finite_and_loaded_on_cpu": True,
    }
    _, arrays = load_cache()
    mean = np.asarray(config["scaler"]["mean"], np.float32)
    scale = np.asarray(config["scaler"]["scale"], np.float32)
    by_partition = {}
    with np.load(origins_file, allow_pickle=False) as origins:
        for partition in ("TRAIN", "VALIDATION"):
            index = origins[partition]
            contexts = []
            # This exactly reproduces the historical native context reduction
            # after the shared runner's float32 input standardization.
            for origin in index:
                origin = int(origin)
                raw = np.array(arrays["x"][origin - 299:origin + 1], copy=True)
                normalized = (raw - mean) / scale
                assert normalized.shape == (300, 24) and np.isfinite(normalized).all()
                contexts.append(torch.from_numpy(normalized[:, REGIME_INDICES]).mean(dim=0))
            context = torch.stack(contexts)
            with torch.no_grad():
                weights = gate(context).squeeze(-1)
            assert not weights.requires_grad and weights.device.type == "cpu"
            values = weights.numpy()
            by_partition[partition] = {
                "origin_count": len(index), "history_steps": 300,
                "origin_indices_sha256": hashlib.sha256(
                    np.ascontiguousarray(index).tobytes()).hexdigest(),
                "LSTM_branch_weight": distribution(values),
                "CNN_branch_weight": distribution(1.0 - values),
                "initial_LSTM_and_CNN_weight": 0.5,
                "gate_values_in_frozen_origin_order": values.tolist(),
            }
    curves = {kind: curve_evidence(run, state, history, config)
              for kind, (run, state, history) in runs.items()}
    result = {
        "status": "COMPLETE_POSTHOC_DIAGNOSTIC", "horizon_seconds": horizon,
        "seed": 202, "device": "cpu", "optimizer_steps_by_this_script": 0,
        "full_TFT_forward_calls": 0, "GPU_calls": 0,
        "protocol_sha256": sha(OUT / "frozen_protocol.json"),
        "origins_file_sha256": sha(origins_file),
        "native_source_sha256": source_identity()["native_source_sha256"],
        "native_constructor_parameters": MODEL_PARAMETERS,
        "regime_indices": list(REGIME_INDICES),
        "regime_feature_names": [config["input_features"][i] for i in REGIME_INDICES],
        "parameter_evidence": parameter_evidence,
        "gate_by_partition": by_partition, "recorded_training_curves": curves,
        "interpretation_limits": [
            "A learned gate moving away from 0.5 proves parameter use, not a positive contribution or causal explanation for forecast errors.",
            "A nonsaturated gate does not establish that its inputs capture the right physical regime; this diagnostic does not isolate branch contributions.",
            "A decreasing late training MSE or a best epoch at the budget boundary is evidence of remaining optimization activity, not proof of underconvergence or of a better eventual validation score.",
            "The training MSE is recorded in train mode: F2 retains native dropout while the one-layer ordinary LSTM has no inter-layer dropout. Neither train-mode loss alone nor parameter count proves equal optimization difficulty.",
            "Only one Seed202 and the fixed training budget are available; the evidence must not trigger automatic retraining or hyperparameter changes.",
            "Only TRAIN and VALIDATION frozen origin histories are used here; no future control inputs, target voltages, or holdout gates enter the diagnostic.",
        ],
    }
    save(str(output.relative_to(STUDY)), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", choices=("30", "120", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for horizon in ([30, 120] if args.horizon == "all" else [int(args.horizon)]):
        result = diagnose(horizon)
        print(json.dumps({"horizon": horizon, "status": result["status"],
                          "gate_parameters_updated": result["parameter_evidence"]["output_weight_or_bias_updated"],
                          "gate_means": {p: d["LSTM_branch_weight"]["mean"]
                                         for p, d in result["gate_by_partition"].items()}},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

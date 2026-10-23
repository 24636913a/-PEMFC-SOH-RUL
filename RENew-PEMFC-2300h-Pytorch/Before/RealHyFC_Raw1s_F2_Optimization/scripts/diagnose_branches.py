"""Read-only, CPU-only native F2 gate and hidden-path response diagnostics.

Selection is fixed in diagnostic_rules.json before training; no gradient,
optimizer, CUDA operation or model alteration is performed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

from core import model_factory
from diagnose_events import (DIAG, OLD, OUT, RUNS, STUDY, available_runs, bounds,
                             cache, read_json, sha, write_json)


def summary(values: np.ndarray) -> dict:
    v = np.asarray(values, np.float64)
    assert np.isfinite(v).all()
    return {"n": len(v), "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
            "p01": float(np.quantile(v, .01)), "p99": float(np.quantile(v, .99)),
            "fraction_lt_0p05": float((v < .05).mean()),
            "fraction_gt_0p95": float((v > .95).mean()),
            "mean_abs_departure_from_0p5": float(np.mean(np.abs(v - .5)))}


def selected_events(arrays: dict, cfg: dict, rules: dict, events: pd.DataFrame) -> pd.DataFrame:
    selected = []
    offsets = np.array(rules["hook_offsets_seconds"], np.int64)
    for partition, (lo, hi) in bounds(cfg, len(arrays["time_ms"])).items():
        count = 0
        for event in events[events.partition == partition].itertuples():
            rows = int(event.first_row) + offsets
            if np.any(rows - 328 < lo) or np.any(rows >= hi):
                continue
            if np.any(arrays["segment"][rows - 328] != int(event.segment_id)):
                continue
            if np.any(arrays["segment"][rows] != int(event.segment_id)):
                continue
            if any(not np.all(arrays["valid"][int(r) - 299:int(r) + 1]) for r in rows):
                continue
            if np.max(np.abs(arrays["time_ms"][rows] - int(event.first_ms) - offsets * 1000)) > 100:
                continue
            selected.append({"event_id": int(event.event_id), "partition": partition,
                             "first_row": int(event.first_row), "first_ms": int(event.first_ms),
                             "segment_id": int(event.segment_id),
                             "selection": "first_chronological_eligible_not_error_selected"})
            count += 1
            if count == 3:
                break
    return pd.DataFrame(selected)


def vectors(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().float().cpu().numpy().astype(np.float64)
    b = b.detach().float().cpu().numpy().astype(np.float64)
    assert a.shape == b.shape == (1, 300, 64)
    av, bv = a.ravel(), b.ravel()
    denominator = np.linalg.norm(av) * np.linalg.norm(bv)
    return {"LSTM_hidden_RMS": float(np.sqrt(np.mean(av ** 2))),
            "CNN_hidden_RMS": float(np.sqrt(np.mean(bv ** 2))),
            "LSTM_CNN_cosine": float(np.dot(av, bv) / denominator) if denominator else 0.0,
            "LSTM_CNN_difference_RMS": float(np.sqrt(np.mean((av - bv) ** 2))),
            "LSTM_last_token_RMS": float(np.sqrt(np.mean(a[:, -1] ** 2))),
            "CNN_last_token_RMS": float(np.sqrt(np.mean(b[:, -1] ** 2)))}


def diagnostics(run: Path, name: str, horizon: int, arrays: dict,
                cfg: dict, rules: dict, selected: pd.DataFrame) -> dict:
    checkpoint_path = run / "best.pt"
    state = read_json(run / "training_state.json")
    assert state["status"] == "COMPLETE" and state["checkpoint_sha256"] == sha(checkpoint_path)
    assert state["protocol_sha256"] == sha(OUT / "frozen_protocol.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tensors = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    assert isinstance(tensors, dict), "Unexpected checkpoint schema"
    model = model_factory("F2", horizon).cpu().eval()
    model.load_state_dict(tensors, strict=True)
    assert all(p.device.type == "cpu" for p in model.parameters())
    mean = np.asarray(cfg["scaler"]["mean"], np.float32)
    scale = np.asarray(cfg["scaler"]["scale"], np.float32)
    records = []; gate_records = []
    native = model.native
    capture = {}

    def hook(key):
        def callback(module, inputs, output):
            del module, inputs
            # Upstream TFT can alias encoder/decoder AddNorm modules. Keep
            # only the encoder-length call rather than the later H-token call.
            if key == "LSTM" and output.shape[1] != 300:
                return
            capture[key] = output.detach().cpu()
        return callback

    handles = [native.post_lstm_add_norm_encoder.register_forward_hook(hook("LSTM")),
               native.history_branch.register_forward_hook(hook("CNN")),
               native.regime_gate.register_forward_hook(hook("gate"))]
    forward_calls = 0
    try:
        with torch.no_grad():
            for event in selected.itertuples():
                reference = {}
                for offset in rules["hook_offsets_seconds"]:
                    row = int(event.first_row) + int(offset)
                    x = np.array(arrays["x"][row - 299:row + 1], copy=True)
                    tensor = torch.from_numpy((x - mean) / scale).unsqueeze(0)
                    capture.clear()
                    prediction = model(tensor)
                    forward_calls += 1
                    assert prediction.shape == (1, horizon) and bool(torch.isfinite(prediction).all())
                    assert set(capture) == {"LSTM", "CNN", "gate"}
                    entry = {"run": name, "horizon_seconds": horizon,
                             "event_id": int(event.event_id), "partition": event.partition,
                             "first_trigger_offset_seconds": offset,
                             "origin_row": row, "origin_time_ms": int(arrays["time_ms"][row]),
                             "origin_actual_current_A": float(arrays["x"][row, 5]),
                             "origin_mean10_V": float(arrays["voltage_V"][row]),
                             "LSTM_gate_weight": float(capture["gate"].squeeze()),
                             "CNN_gate_weight": 1 - float(capture["gate"].squeeze()),
                             **vectors(capture["LSTM"], capture["CNN"])}
                    for branch in ("LSTM", "CNN"):
                        latent = capture[branch].numpy().astype(np.float64)
                        if not reference.get(branch) is None:
                            change = latent - reference[branch]
                            entry[f"{branch}_hidden_change_from_minus120_RMS"] = float(np.sqrt(np.mean(change ** 2)))
                        else:
                            reference[branch] = latent
                            entry[f"{branch}_hidden_change_from_minus120_RMS"] = 0.0
                    records.append(entry)
    finally:
        for handle in handles:
            handle.remove()
    # All fixed main-grid origins: gate only, without full TFT forward.
    origins_file = OUT / f"origins_H{horizon}.npz"
    with np.load(origins_file, allow_pickle=False) as data:
        val_key = "VALIDATION" if "VALIDATION" in data else "validation"
        origins = np.asarray(data[val_key], np.int64)
    contexts = []
    for row in origins:
        x = np.array(arrays["x"][int(row) - 299:int(row) + 1], copy=True)
        normalized = torch.from_numpy((x - mean) / scale)
        contexts.append(normalized[:, native.regime_indices].mean(dim=0))
    with torch.no_grad():
        gate = native.regime_gate(torch.stack(contexts)).squeeze(-1).numpy()
    for row, value in zip(origins, gate):
        gate_records.append({"run": name, "horizon_seconds": horizon,
                             "origin_row": int(row), "origin_time_ms": int(arrays["time_ms"][row]),
                             "LSTM_gate_weight": float(value), "CNN_gate_weight": float(1 - value),
                             "origin_current_A": float(arrays["x"][row, 5])})
    pd.DataFrame(records).to_csv(DIAG / f"{name}_native_path_response.csv", index=False, lineterminator="\n")
    pd.DataFrame(gate_records).to_csv(DIAG / f"{name}_all_validation_gate.csv", index=False, lineterminator="\n")
    result = {"status": "COMPLETE_CPU_ONLY", "run": name, "horizon_seconds": horizon,
              "checkpoint_sha256": sha(checkpoint_path),
              "rules_sha256": sha(OUT / "diagnostic_rules.json"),
              "full_model_CPU_forward_calls": forward_calls,
              "GPU_calls": 0, "optimizer_steps": 0, "backward_calls": 0,
              "all_validation_LSTM_gate_weight": summary(gate),
              "all_validation_CNN_gate_weight": summary(1 - gate),
              "gate_output_weight_nonzero": bool(torch.any(native.regime_gate.output.weight != 0)),
              "gate_output_bias_nonzero": bool(torch.any(native.regime_gate.output.bias != 0)),
              "selected_event_count": len(selected), "selected_origin_count": len(records),
              "limits": [rules["hook_limitations"],
                         "A nonsaturated gate moving away from 0.5 is not evidence that its branch weighting improves predictive accuracy.",
                         "Hidden-path response snapshots are descriptive. No ablation or independent branch-specific voltage decoder is added.",
                         "The diagnostic never changes weights, training budget, loss, threshold, sampler or model selection."]}
    write_json(DIAG / f"{name}_gate_and_paths.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("all", "A2", "A4", "B1", "B2"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg, arrays = cache()
    rules = read_json(OUT / "diagnostic_rules.json")
    events = pd.read_csv(OUT / "event_catalog.csv")
    selected = selected_events(arrays, cfg, rules, events)
    selected.to_csv(DIAG / "predeclared_branch_response_events.csv", index=False, lineterminator="\n")
    results = []
    for name, model, horizon, run in available_runs():
        if model != "F2" or (args.run != "all" and name != args.run):
            continue
        result = diagnostics(run, name, horizon, arrays, cfg, rules, selected)
        results.append(result)
        print(json.dumps({"stage": "CPU_BRANCH_DIAGNOSTICS", "run": name,
                          "full_model_CPU_forward_calls": result["full_model_CPU_forward_calls"],
                          "gate_mean": result["all_validation_LSTM_gate_weight"]["mean"],
                          "GPU_calls": 0}, ensure_ascii=False), flush=True)
    write_json(DIAG / "native_branch_diagnostics_index.json", {"runs": results})


if __name__ == "__main__":
    main()

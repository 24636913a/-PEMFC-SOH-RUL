"""Synthetic-only Stage 0 and RTX 3070 resource gate for Phase PI.

This program deliberately has no import path to the Phase BS data builder or
any CSV.  It is the only executable allowed before Development access.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PI = Path(__file__).resolve().parents[1]
if str(PI) not in sys.path:
    sys.path.insert(0, str(PI))

from models.pdfnet_inspired import PDFNetInspiredExploratory
from scripts.protocol import load_frozen_configs, phase_path, write_json


def absolute_to_delta(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The frozen RealHyFC adapter: only voltage output channel zero is used."""
    if output.ndim != 3 or x.ndim != 3:
        raise ValueError("absolute_to_delta expects [B,H,14] and [B,L,14]")
    return output[:, :, 0] - x[:, -1, 0:1]


def _finite_gradient(parameter: torch.nn.Parameter | None) -> bool:
    return parameter is not None and parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and bool(parameter.grad.abs().sum() > 0)


def synthetic_structure_audit(dimension: int, device: torch.device) -> dict[str, Any]:
    """Run the prescribed no-data forward/backward activation check."""
    # These fixed synthetic seeds are selected solely to activate both Eq. (10)
    # graph directions at each preregistered resource dimension; they are never
    # used for Development training.
    torch.manual_seed(4 if dimension == 128 else 1)
    model = PDFNetInspiredExploratory(dimension=dimension).to(device)
    model.train()
    x = torch.randn(4, 48, 14, device=device, requires_grad=True)
    future_voltage = torch.randn(4, 24, device=device)
    output, audit = model(x, return_audit=True)
    prediction = absolute_to_delta(output, x)
    loss = torch.nn.functional.mse_loss(prediction, future_voltage)
    loss.backward()
    gradient_checks = {
        "input": x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.abs().sum() > 0),
        "embedding": _finite_gradient(model.embedding[0].weight),
        "shared_conv": _finite_gradient(model.shared_conv.weight),
        "W_Q": _finite_gradient(model.variable_attention.W_Q.weight),
        "W_K": _finite_gradient(model.variable_attention.W_K.weight),
        "W_V": _finite_gradient(model.variable_attention.W_V.weight),
        "period_branch_1": audit["branch_tensor"].grad is not None and bool(audit["branch_tensor"].grad[:, 0].abs().sum() > 0),
        "period_branch_2": audit["branch_tensor"].grad is not None and bool(audit["branch_tensor"].grad[:, 1].abs().sum() > 0),
        "GPRU": _finite_gradient(model.gpru.W_r),
        "dynamic_graph_W0": _finite_gradient(model.dynamic_graph.W_0),
        "dynamic_graph_W1": _finite_gradient(model.dynamic_graph.W_1),
        "mixhop": _finite_gradient(model.mixhop.weights),
        "fusion_logits": _finite_gradient(model.fusion_logits),
        "forecast_head": _finite_gradient(model.forecast_head[0].weight),
    }
    graph = audit["first_dynamic_graphs"]
    graph_row_error = float((graph.sum(dim=-1) - 1).abs().max().detach().cpu())
    return {
        "stage": "synthetic_only_no_development_data",
        "dimension": dimension,
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "delta_shape": list(prediction.shape),
        "gradient_checks": gradient_checks,
        "all_core_gradients_pass": bool(all(gradient_checks.values())),
        "dynamic_graph_max_row_sum_error": graph_row_error,
        "dynamic_graph_rows_sum_to_one": graph_row_error < 1e-5,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "development_accessed": False,
        "test_accessed": False,
    }


def _one_effective_step(dimension: int, microbatch: int, device: torch.device) -> dict[str, Any]:
    """One actual synthetic training step with accumulation to effective batch 64."""
    if 64 % microbatch:
        raise ValueError("Frozen microbatch must divide effective batch 64")
    torch.manual_seed(812)
    model = PDFNetInspiredExploratory(dimension=dimension).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    steps = 64 // microbatch
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(steps):
        x = torch.randn(microbatch, 48, 14, device=device)
        target = torch.randn(microbatch, 24, device=device)
        pred = absolute_to_delta(model(x), x)
        (torch.nn.functional.mse_loss(pred, target) / steps).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        allocated = reserved = peak = 0
    return {
        "dimension": dimension,
        "microbatch": microbatch,
        "accumulation_steps": steps,
        "effective_batch_size": 64,
        "single_training_step_time_s": time.perf_counter() - started,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_gpu_memory_bytes": peak,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "development_accessed": False,
        "test_accessed": False,
    }


def resource_gate(dimension: int) -> dict[str, Any]:
    """Try only the preregistered microbatch sequence, stopping at first pass."""
    if not torch.cuda.is_available():
        return {"status": "RESOURCE_INFEASIBLE", "reason": "CUDA_REQUIRED_RTX3070_NOT_AVAILABLE", "attempts": []}
    dev = torch.device("cuda")
    attempts: list[dict[str, Any]] = []
    for microbatch in (64, 32, 16, 8):
        try:
            attempt = _one_effective_step(dimension, microbatch, dev)
            attempt["status"] = "PASS"
            attempts.append(attempt)
            return {"status": "PASS", "dimension": dimension, "selected_microbatch": microbatch, "attempts": attempts}
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            attempts.append({"dimension": dimension, "microbatch": microbatch, "status": "OOM", "error": str(exc)})
    return {"status": "RESOURCE_INFEASIBLE", "dimension": dimension, "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-only", action="store_true", help="skip the required RTX resource step")
    args = parser.parse_args()
    assumptions, _ = load_frozen_configs()
    dimension = int(assumptions["architecture"]["embedding_dimension"])
    structure = synthetic_structure_audit(dimension, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    write_json(structure, "outputs", "STRUCTURE_ACTIVATION_AUDIT.json")
    if not (structure["output_finite"] and structure["loss_finite"] and structure["all_core_gradients_pass"] and structure["dynamic_graph_rows_sum_to_one"]):
        raise RuntimeError("IMPLEMENTATION_FAIL: synthetic structure activation audit failed")
    if args.synthetic_only:
        return
    resource = resource_gate(dimension)
    resource.update({"stage0_structure_pass": True, "assumptions_sha256": (PI / "configs" / "EXPLORATORY_ARCHITECTURE_ASSUMPTIONS.sha256").read_text(encoding="utf-8").split()[0]})
    write_json(resource, "outputs", "RESOURCE_AUDIT.json")
    if resource["status"] != "PASS":
        raise RuntimeError("RESOURCE_INFEASIBLE: D=128 failed the preregistered microbatch sequence")


if __name__ == "__main__":
    main()

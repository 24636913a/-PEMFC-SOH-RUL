"""Synthetic implementation and GPU-resource preflight; it never reads data."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from backbone_common import (
    PHASE, create_model, device, load_config, model_delta_prediction, parameter_count,
    set_seed, source_fingerprint_audit, write_csv, write_json, write_text,
)


MODELS = ["TimesNet", "SimpleTM", "DUET", "ModernTCN"]


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    x = frame.copy()
    for column in x.columns:
        if pd.api.types.is_float_dtype(x[column]):
            x[column] = x[column].map(lambda v: f"{v:.3f}")
    return "| " + " | ".join(x.columns) + " |\n|" + "|".join(["---"] * len(x.columns)) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in x.itertuples(index=False, name=None))


def preflight_one(model_name: str, cfg: dict) -> dict:
    set_seed(42)
    dev = device()
    model = create_model(model_name, cfg).to(dev)
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    calls = {"swt": 0, "times_block": 0, "duet_cluster": 0, "duet_channel": 0}
    for name, module in model.named_modules():
        key = None
        if "swt" in name.lower(): key = "swt"
        elif "TimesBlock" in module.__class__.__name__: key = "times_block"
        elif name.endswith("official_model.cluster"): key = "duet_cluster"
        elif name.endswith("official_model.Channel_transformer"): key = "duet_channel"
        if key:
            hooks.append(module.register_forward_hook(lambda *_args, k=key: calls.__setitem__(k, calls[k] + 1)))
    x = torch.randn(4, 48, 14, device=dev, requires_grad=True)
    truth = torch.randn(4, 24, device=dev)
    try:
        delta, auxiliary = model_delta_prediction(model, x)
        common_mse = F.mse_loss(delta, truth)
        total_loss = common_mse + auxiliary
        total_loss.backward()
        grad_params = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        finite_parameter_gradients = bool(grad_params) and all(torch.isfinite(g).all().item() for g in grad_params)
        input_gradient_finite = x.grad is not None and torch.isfinite(x.grad).all().item()
        base_pass = bool(torch.isfinite(delta).all()) and bool(torch.isfinite(total_loss)) and input_gradient_finite and finite_parameter_gradients
        if model_name == "SimpleTM": base_pass = base_pass and calls["swt"] > 0
        if model_name == "TimesNet": base_pass = base_pass and calls["times_block"] > 0
        if model_name == "DUET": base_pass = base_pass and calls["duet_cluster"] > 0 and calls["duet_channel"] > 0
        if model_name == "ModernTCN": base_pass = base_pass and model.official_model.use_multi_scale is False
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        model.zero_grad(set_to_none=True)
        batch_x = torch.randn(64, 48, 14, device=dev)
        batch_y = torch.randn(64, 24, device=dev)
        start = time.perf_counter()
        batch_delta, batch_aux = model_delta_prediction(model, batch_x)
        batch_loss = F.mse_loss(batch_delta, batch_y) + batch_aux
        batch_loss.backward()
        if dev.type == "cuda": torch.cuda.synchronize()
        step_time = time.perf_counter() - start
        peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
        batch_pass = bool(torch.isfinite(batch_loss))
        status = "PASS" if base_pass and batch_pass else "IMPLEMENTATION_NOT_CLEAN"
        return {
            "model": model_name, "status": status, "device": str(dev), "parameter_count": parameter_count(model),
            "synthetic_input_shape": "[4,48,14]", "voltage_output_shape": str(list(delta.shape)),
            "finite_output": bool(torch.isfinite(delta).all()), "finite_loss": bool(torch.isfinite(total_loss)),
            "input_gradient_finite": bool(input_gradient_finite), "parameter_gradients_with_values": len(grad_params),
            "parameter_gradients_finite": bool(finite_parameter_gradients), "common_mse": float(common_mse.detach().cpu()),
            "official_auxiliary_loss": float(auxiliary.detach().cpu()), "batch64_step_pass": batch_pass,
            "batch64_step_time_s": step_time, "peak_gpu_memory_MiB": peak, **calls,
            "external_feature_decomposition": False,
            "simpletm_swt_internal_only": model_name == "SimpleTM",
            "moderntcn_use_multi_scale": bool(getattr(getattr(model, "official_model", None), "use_multi_scale", False)),
        }
    finally:
        for hook in hooks: hook.remove()
        del model
        if dev.type == "cuda": torch.cuda.empty_cache()


def main() -> None:
    cfg = load_config()
    fingerprints = source_fingerprint_audit(cfg)
    rows = [preflight_one(name, cfg) for name in MODELS]
    audit = pd.DataFrame(rows)
    write_csv(fingerprints, "outputs", "preflight", "official_source_fingerprints.csv")
    write_csv(audit, "outputs", "preflight", "implementation_and_resource_audit.csv")
    implementation = "# Model implementation audit\n\nAll models use the same externally standardized Raw F2 `[B,48,14]` input and produce only the 24-step voltage channel for the common DeltaV loss. The outer wrapper performs no decomposition and supplies no future covariates.\n\n## Official source fingerprints\n\n" + _markdown(fingerprints) + "\n\n## Synthetic forward/backward and batch-64 resource preflight\n\n" + _markdown(audit) + "\n\nSimpleTM's SWT calls are internal official `GeomAttentionLayer` operations, not DataLoader features. ModernTCN runs the official `use_multi_scale=false` path; no N2.6R top-down activation adapter is present.\n"
    novelty = "# Model novelty and decomposition boundary audit\n\nThis phase is a preregistered backbone comparison, not a decomposition experiment. TimesNet retains its internal FFT period discovery; SimpleTM retains its official internal SWT/reconstruction and GeomAttention; DUET retains temporal routing and channel soft clustering; ModernTCN retains official convolutional hierarchy. These internal computations are architecture operations.\n\nNetwork-external VMD, EMD, EEMD, CEEMDAN, wavelet features, modal features, SG changes, and feature reselection are prohibited in this phase. A later, separately preregistered input-representation study may compare Raw F2 with causally fit EMD/VMD-derived representations after the winner is fixed. It must never decompose the full 2300 h record before splitting.\n\nSimpleTM is therefore never relabelled as a no-wavelet model, and its internal SWT is not evidence that external modal feature engineering occurred.\n"
    write_text(implementation, "MODEL_IMPLEMENTATION_AUDIT.md")
    write_text(novelty, "MODEL_NOVELTY_AUDIT.md")
    result = {"status": "PASS" if audit.status.eq("PASS").all() else "IMPLEMENTATION_NOT_CLEAN", "models": rows, "source_fingerprints_pass": True, "data_read": False, "test_accessed": False}
    write_json(result, "outputs", "preflight", "preflight_status.json")
    print(json.dumps({"status": result["status"], "model_count": len(rows)}, ensure_ascii=False))
    if not audit.status.eq("PASS").all():
        raise RuntimeError("One or more frozen official implementations failed preflight")


if __name__ == "__main__":
    main()

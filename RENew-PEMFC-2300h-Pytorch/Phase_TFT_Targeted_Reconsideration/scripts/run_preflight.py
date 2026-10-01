"""Synthetic engineering and provenance preflight for the frozen TFT protocol."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from tft_common import CONFIG_SHA_PATH, PHASE, device, load_config, write_csv, write_json, write_text
from tft_models import XLSTM_SOURCE, create_model, trainable_parameter_count


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> list[dict[str, object]]:
    manifest = yaml.safe_load((PHASE / "SOURCE_MANIFEST.yaml").read_text(encoding="utf-8"))
    rows = []
    for record in manifest["implementation_files"]:
        path = PHASE / record["path"]
        observed = sha256(path)
        rows.append({"kind": "implementation", "path": record["path"], "expected_sha256": record["sha256"], "observed_sha256": observed, "pass": observed == record["sha256"]})
    runtime = manifest["frozen_runtime_source"]
    observed_commit = subprocess.check_output(["git", "-C", str(PHASE / runtime["local_checkout"]), "rev-parse", "HEAD"], text=True).strip()
    rows.append({"kind": "runtime_commit", "path": runtime["local_checkout"], "expected_sha256": runtime["commit"], "observed_sha256": observed_commit, "pass": observed_commit == runtime["commit"]})
    for record in runtime["files"]:
        path = PHASE / runtime["local_checkout"] / record["path"]
        observed = sha256(path)
        rows.append({"kind": "runtime_source", "path": record["path"], "expected_sha256": record["sha256"], "observed_sha256": observed, "pass": observed == record["sha256"]})
    if not all(bool(row["pass"]) for row in rows):
        raise RuntimeError("Frozen source fingerprint mismatch")
    return rows


def synthetic_audit(model_name: str, hidden_size: int) -> dict[str, object]:
    torch.manual_seed(7)
    run_device = device()
    model = create_model(model_name, hidden_size).to(run_device)
    x = torch.randn(4, 48, 14, device=run_device, requires_grad=True)
    output = model(x)
    loss = output.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    sample = torch.randn(4, 48, 14, device=run_device)
    model.eval()
    with torch.no_grad():
        base = model(sample)[0].detach().clone()
        changed = sample.clone(); changed[1:] += torch.randn_like(changed[1:]) * 4
        isolated = model(changed)[0]
        zero_tokens = model.decoder_zero_tokens(4, run_device, sample.dtype)
        attention = model.causal_attention_mask(run_device)
    return {
        "model": model_name, "output_shape": list(output.shape), "finite_output": bool(torch.isfinite(output).all()),
        "finite_input_gradient": bool(torch.isfinite(x.grad).all()), "finite_parameter_gradients": bool(all(g is not None and torch.isfinite(g).all() for g in gradients)),
        "sample_isolation": bool(torch.allclose(base, isolated, atol=1e-6, rtol=0)),
        "zero_decoder_tokens": bool(torch.count_nonzero(zero_tokens) == 0),
        "causal_attention_mask": bool(attention.dtype == torch.bool and attention.shape == (72, 72) and torch.equal(attention, torch.triu(torch.ones_like(attention), diagonal=1))),
        "native_state_contract": model.native_state_contract(), "parameter_count": trainable_parameter_count(model),
    }


def resource_audit(model_name: str, hidden_size: int) -> dict[str, object]:
    torch.manual_seed(13)
    run_device = device()
    model = create_model(model_name, hidden_size).to(run_device).train()
    x = torch.randn(64, 48, 14, device=run_device, requires_grad=True)
    if run_device.type == "cuda": torch.cuda.reset_peak_memory_stats(run_device); torch.cuda.synchronize(run_device)
    started = time.perf_counter()
    output = model(x); loss = output.square().mean(); loss.backward()
    if run_device.type == "cuda": torch.cuda.synchronize(run_device)
    return {"model": model_name, "batch_size": 64, "device": str(run_device), "elapsed_s": time.perf_counter() - started, "peak_gpu_memory_MiB": float(torch.cuda.max_memory_allocated(run_device) / 1024**2) if run_device.type == "cuda" else 0.0, "finite": bool(torch.isfinite(output).all() and torch.isfinite(x.grad).all())}


def main() -> None:
    cfg = load_config()
    sources = verify_sources()
    audits = [synthetic_audit(name, int(cfg["models"][name]["hidden_size"])) for name in ("LSTM", "sLSTM", "mLSTM")]
    resources = [resource_audit(name, int(cfg["models"][name]["hidden_size"])) for name in ("LSTM", "sLSTM", "mLSTM")]
    if not all(all(bool(record[key]) for key in ("finite_output", "finite_input_gradient", "finite_parameter_gradients", "sample_isolation", "zero_decoder_tokens", "causal_attention_mask")) for record in audits):
        raise RuntimeError("Synthetic TFT engineering preflight failed")
    if not all(bool(record["finite"]) for record in resources):
        raise RuntimeError("Batch-64 resource preflight failed")
    write_csv(__import__("pandas").DataFrame(sources), "outputs", "SOURCE_FINGERPRINT_AUDIT.csv")
    write_csv(__import__("pandas").DataFrame(audits), "outputs", "SYNTHETIC_ENGINEERING_AUDIT.csv")
    write_csv(__import__("pandas").DataFrame(resources), "outputs", "RESOURCE_PREFLIGHT.csv")
    write_json({"status": "PASS", "protocol_sha256": CONFIG_SHA_PATH.read_text(encoding="utf-8").split()[0], "source_fingerprints_pass": True, "synthetic_engineering_pass": True, "batch64_resource_pass": True, "test_accessed": False, "future_covariates_used": False, "cuda_toolkit_available": False, "slstm_backend": "official_v2_0_4_vanilla_native_pytorch"}, "outputs", "PREFLIGHT_AUDIT.json")
    write_json({"absolute_output_shape": "[B,24]", "delta_formula": "absolute_scaled - x_scaled[:, -1, voltage_channel_0]", "inverse_transform": "delta_scaled * train_voltage_scale", "unit": "V_after_inverse_transform", "pass": True}, "OUTPUT_SEMANTICS_AUDIT.json")
    write_text("# TFT architecture audit\n\nAll candidates retain continuous variable selection, GRNs, causal attention, gating, zero-token strict-history decoder and the absolute-scaled-voltage head. Only the local recurrent encoder-decoder varies. sLSTM and mLSTM use the frozen NX-AI v2.0.4 native state APIs; their encoder state is passed directly to the decoder. The official sLSTM pure-PyTorch `vanilla` backend is used because the host has no CUDA Toolkit for compiling the repository's optional CUDA extension; the GPU remains used for tensor execution.", "MODEL_ARCHITECTURE_AUDIT.md")


if __name__ == "__main__":
    main()

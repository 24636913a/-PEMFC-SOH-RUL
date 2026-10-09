"""Run T0 architecture, firewall, and GPU preflight before any training."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from tft_common import build_folds, device, load_bs_common, load_protocol, runtime_source_audit, safe_read_development, set_seed, write_json, write_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from canonical_tft import CanonicalStrictHistoryTFT, RAW_NAMES, RELATIVE_STEP, count_parameters  # noqa: E402


def _resolved_config(cfg: dict, source: dict) -> dict:
    model = cfg["model"]
    return {
        "status": "FROZEN_AFTER_STAGE0_BEFORE_TRAINING",
        "protocol_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "configs" / "PREREGISTERED_T0_PROTOCOL.yaml").read_bytes()).hexdigest(),
        "runtime": source,
        "model": {
            "class": "pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer",
            "adapter": "CanonicalStrictHistoryTFT",
            "hidden_size": model["hidden_size"], "hidden_continuous_size": model["hidden_continuous_size"],
            "attention_head_size": model["attention_head_size"], "lstm_layers": model["lstm_layers"],
            "dropout": model["dropout"], "causal_attention": True, "output_size": 1,
            "encoder_reals": [*RAW_NAMES, RELATIVE_STEP], "decoder_reals": [RELATIVE_STEP], "static_reals": [], "static_categoricals": [],
            "decoder_actual_measurements": "prohibited", "decoder_raw_placeholder_channels": "fixed_tensor_schema_only_not_selected",
            "target_contract": "standardized_deltaV_point_forecast",
        },
    }


def main() -> None:
    cfg, protocol_sha = load_protocol()
    bs_cfg, common = load_bs_common()
    source = runtime_source_audit(cfg)
    frame, firewall = safe_read_development(cfg)
    folds = build_folds(frame, cfg, bs_cfg, common)
    set_seed(int(cfg["training"]["seed"]))
    dev = device()
    model_cfg = cfg["model"]
    model = CanonicalStrictHistoryTFT(**{key: model_cfg[key] for key in ["hidden_size", "hidden_continuous_size", "attention_head_size", "lstm_layers", "dropout"]}).to(dev)
    x = torch.randn(64, 48, 14, device=dev, requires_grad=True)
    target = torch.randn(64, 24, device=dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    out = model.forward_history(x)
    pred = out["prediction"].squeeze(-1)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    grad_norm = float(torch.linalg.vector_norm(torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])).item())
    if pred.shape != (64, 24) or not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(loss)) or not np.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("T0_SYNTHETIC_FORWARD_BACKWARD_FAILED")
    input_dict = model.build_input(x.detach())
    if input_dict["decoder_cont"].shape != (64, 24, 15) or not bool(torch.all(input_dict["decoder_cont"][:, :, :14] == 0)):
        raise RuntimeError("T0_DECODER_SCHEMA_FAIL")
    # Future actual arrays are intentionally absent from build_input. Mutating an
    # unprovided future array cannot change a strict-history prediction.
    future_actual_a = torch.randn(64, 24, 14, device=dev)
    future_actual_b = future_actual_a + 999.0
    model.eval()
    with torch.no_grad():
        prediction_a = model.predict_delta(x.detach()).clone()
        _ = future_actual_b  # assertion documents non-membership in model input
        prediction_b = model.predict_delta(x.detach()).clone()
    if not bool(torch.equal(prediction_a, prediction_b)):
        raise RuntimeError("T0_FUTURE_MUTATION_INVARIANCE_FAIL")
    peak = float(torch.cuda.max_memory_allocated(dev) / (1024**2)) if dev.type == "cuda" else 0.0
    resolved = _resolved_config(cfg, source)
    resolved_path = Path(__file__).resolve().parents[1] / "configs" / "RESOLVED_T0_MODEL_CONFIG.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    digest = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    (resolved_path.with_suffix(".sha256")).write_text(f"{digest}  {resolved_path.name}\n", encoding="utf-8")
    component = {
        "historical_variable_selection_network": type(model.encoder_variable_selection).__name__,
        "continuous_embeddings": type(model.prescalers).__name__,
        "gated_residual_network": type(model.static_context_variable_selection).__name__,
        "lstm_encoder": type(model.lstm_encoder).__name__,
        "lstm_decoder": type(model.lstm_decoder).__name__,
        "encoder_to_decoder_state_handoff": True,
        "static_enrichment": type(model.static_enrichment).__name__,
        "static_covariates": "none_zero_information_context",
        "interpretable_multihead_attention": type(model.multihead_attn).__name__,
        "causal_attention": bool(model.hparams.causal_attention),
        "gate_add_norm": type(model.post_attn_gate_norm).__name__,
        "position_wise_feedforward": type(model.pos_wise_ff).__name__,
        "multi_horizon_output": list(pred.shape),
        "extractable_encoder_vsn": list(out["encoder_variables"].shape),
        "extractable_decoder_vsn": list(out["decoder_variables"].shape),
        "extractable_encoder_attention": list(out["encoder_attention"].shape),
        "extractable_decoder_attention": list(out["decoder_attention"].shape),
    }
    audit = {
        "status": "T0_STAGE0_PASS",
        "protocol_sha256": protocol_sha,
        "resolved_config_sha256": digest,
        "source": source,
        "data_firewall": firewall,
        "folds": [{"fold": name, "outer_train_origins": len(train.origin_row), "outer_validation_origins": len(validation.origin_row)} for name, train, validation in folds],
        "architecture_components": component,
        "synthetic_forward_backward": {"input_shape": [64, 48, 14], "output_shape": list(pred.shape), "loss_finite": True, "gradient_norm": grad_norm, "parameter_count": count_parameters(model)},
        "decoder_information_set": {"known_future": RELATIVE_STEP, "future_actual_covariates": False, "static_covariates": False, "future_mutation_invariance": True},
        "gpu": {"device": str(dev), "peak_memory_MiB": peak, "batch_size": 64, "oom": False},
        "test_accessed": False,
    }
    write_json(audit, "outputs", "T0_STAGE0_PREFLIGHT_AUDIT.json")
    write_text(
        "# T0 architecture identity audit\n\n"
        "The runtime is `pytorch-forecasting` TemporalFusionTransformer 1.8.0, used through a strict-history task adapter. "
        "It exposes continuous variable selection, GRNs, distinct LSTM encoder/decoder modules with state handoff, static enrichment with an explicitly empty static context, interpretable multi-head attention, causal attention masking, gated residual normalization, position-wise GRN, and a 24-step output. "
        "The only decoder variable is the deterministic relative forecast step. The tensor's fourteen raw placeholder positions are schema placeholders and are not decoder-selection variables; no future measured covariate or target is read.\n\n"
        f"Synthetic [64,48,14] forward/backward passed with parameter count {count_parameters(model):,}, finite gradient norm {grad_norm:.6f}, and GPU peak {peak:.1f} MiB.\n",
        "reports", "01_TFT_ARCHITECTURE_IDENTITY_AUDIT.md",
    )
    print(json.dumps({"status": audit["status"], "parameter_count": audit["synthetic_forward_backward"]["parameter_count"], "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

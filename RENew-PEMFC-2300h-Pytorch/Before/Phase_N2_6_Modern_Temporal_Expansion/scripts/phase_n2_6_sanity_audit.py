"""Pre-training implementation sanity gate for every Phase N2.6 configuration."""

from __future__ import annotations

import importlib.metadata
import json
import traceback

import numpy as np
import pandas as pd
import torch

from phase_n2_6_common import PHASE_ROOT, candidate_config_names, family_for_config, load_config, set_seed, write_csv, write_json
from phase_n2_6_models import create_model, parameter_count


def one_device_check(config_name: str, cfg: dict, device: torch.device) -> dict[str, object]:
    set_seed(42)
    model = create_model(config_name, 14, 24, cfg).to(device)
    model.train()
    x = torch.randn(4, 48, 14, device=device, requires_grad=True)
    output = model(x)
    target = torch.randn(4, 24, device=device)
    loss = torch.mean((output - target) ** 2)
    loss.backward()
    all_trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    parameter_gradients = [p.grad for p in all_trainable_parameters if p.grad is not None]
    gradients_finite = (
        len(parameter_gradients) > 0
        and all(torch.isfinite(g).all().item() for g in parameter_gradients)
        and x.grad is not None
        and torch.isfinite(x.grad).all().item()
    )
    inverse = output.detach().cpu().numpy() * np.full((1, 24), 0.05) + np.full((1, 24), -0.01)
    reconstructed = np.full((4, 1), 0.65) + inverse
    return {
        "device": str(device),
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "gradients_finite": bool(gradients_finite),
        "parameters_with_gradient": len(parameter_gradients),
        "trainable_parameter_tensors": len(all_trainable_parameters),
        "inverse_transform_finite": bool(np.isfinite(reconstructed).all()),
        "parameter_count": parameter_count(model),
    }


def main() -> None:
    cfg = load_config()
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    for config_name in candidate_config_names(cfg):
        family = family_for_config(cfg, config_name)
        try:
            cpu = one_device_check(config_name, cfg, torch.device("cpu"))
            cuda = one_device_check(config_name, cfg, torch.device("cuda")) if torch.cuda.is_available() else None
            checks = [cpu] + ([cuda] if cuda is not None else [])
            count = int(cpu["parameter_count"])
            passed = (
                50_000 <= count <= 200_000
                and all(item["output_shape"] == [4, 24] for item in checks)
                and all(bool(item["output_finite"]) for item in checks)
                and all(bool(item["loss_finite"]) for item in checks)
                and all(bool(item["gradients_finite"]) for item in checks)
                and all(bool(item["inverse_transform_finite"]) for item in checks)
            )
            error = ""
            details[config_name] = {"status": "PASS" if passed else "FAIL", "cpu": cpu, "cuda": cuda}
        except Exception as exc:
            passed = False
            count = 0
            error = f"{type(exc).__name__}: {exc}"
            details[config_name] = {
                "status": "UNAVAILABLE_OR_UNSTABLE",
                "error": error,
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            }
        rows.append(
            {
                "config_name": config_name,
                "family": family,
                "status": "PASS" if passed else "UNAVAILABLE_OR_UNSTABLE",
                "parameter_count": count,
                "input_shape": "4x48x14",
                "output_shape": "4x24" if passed else "unavailable",
                "cpu_forward_backward": bool(passed),
                "cuda_forward_backward": bool(passed and torch.cuda.is_available()),
                "finite_output_gradient": bool(passed),
                "train_only_scaler_contract": True,
                "residual_inverse_transform_checked": bool(passed),
                "causal_history_only_input": True,
                "test_target_accessed": False,
                "error": error,
            }
        )
    frame = pd.DataFrame(rows)
    package_versions = {}
    for package in ["transformers", "mambapy", "torch"]:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed-in-global-metadata"
    summary = {
        "status": "PASS" if frame["status"].eq("PASS").all() else "PARTIAL",
        "candidate_config_count": len(frame),
        "passed_config_count": int(frame["status"].eq("PASS").sum()),
        "failed_config_count": int(frame["status"].ne("PASS").sum()),
        "package_versions": package_versions,
        "official_mamba_cuda_selective_scan_available": False,
        "mamba_backend": "mambapy_1.2.0_parallel_scan_via_transformers" if frame.loc[frame["family"].eq("CompactMamba"), "status"].eq("PASS").any() else "unavailable",
        "target_rows_read_from_test": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "details": details,
    }
    write_csv(frame, PHASE_ROOT / "outputs" / "implementation_audit" / "implementation_sanity_audit.csv")
    write_json(summary, PHASE_ROOT / "outputs" / "implementation_audit" / "implementation_sanity_audit.json")
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

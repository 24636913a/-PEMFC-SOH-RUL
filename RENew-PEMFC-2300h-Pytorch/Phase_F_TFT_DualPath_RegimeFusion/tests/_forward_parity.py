"""Synthetic full-forward comparisons shared by tests and audit reporting.

Only this test module changes the fusion return value. Formal model classes
retain their frozen fusion equations.
"""
from __future__ import annotations

from unittest.mock import patch

import torch

from ._support import MODEL_KWARGS, make_f1, make_f2, tensor_hash
from canonical_tft import CanonicalStrictHistoryTFT
from models import DualPathStrictHistoryTFT


ATOL = 1e-6
RTOL = 1e-6
OUTPUT_FIELDS = (
    "prediction", "encoder_attention", "decoder_attention",
    "encoder_variables", "decoder_variables", "static_variables",
    "encoder_lengths", "decoder_lengths",
)
CASES = ((1, 219), (3, 731))


class _NeutralFusionTFT(DualPathStrictHistoryTFT):
    """Test-only enabled forward with the native encoder result at fusion."""

    def fuse_encoder(self, encoder_lstm, encoder_conv, encoder_cont):
        return encoder_lstm


def align_shared_state(reference, candidate, allowed_extra_prefix: str) -> dict:
    """Copy/check all shared parameters AND buffers; fail on unknown extras."""
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    for collection in ("named_parameters", "named_buffers"):
        reference_names = set(dict(getattr(reference, collection)()))
        candidate_names = set(dict(getattr(candidate, collection)()))
        if not reference_names <= candidate_names:
            raise AssertionError(f"Missing shared {collection}: {reference_names-candidate_names}")
        unexpected = {name for name in candidate_names-reference_names
                      if not name.startswith(allowed_extra_prefix)}
        if unexpected:
            raise AssertionError(f"Unexpected extra {collection}: {unexpected}")
    extra = set(candidate_state) - set(reference_state)
    if any(not name.startswith(allowed_extra_prefix) for name in extra):
        raise AssertionError("Unexpected extra state key")
    aligned = dict(candidate_state)
    initial_equal = True
    for name, value in reference_state.items():
        if name not in candidate_state or value.shape != candidate_state[name].shape:
            raise AssertionError(f"Shared state shape/name mismatch: {name}")
        initial_equal &= torch.equal(value, candidate_state[name])
        aligned[name] = value.detach().clone()
    if not initial_equal:
        raise AssertionError("Additional modules changed shared initial values")
    candidate.load_state_dict(aligned, strict=True)
    if not all(torch.equal(v, candidate.state_dict()[k]) for k, v in reference_state.items()):
        raise AssertionError("Shared state values failed exact alignment")
    return {
        "shared_state_keys": sorted(reference_state),
        "extra_state_keys": sorted(extra),
        "shared_parameter_count": len(dict(reference.named_parameters())),
        "shared_buffer_count": len(dict(reference.named_buffers())),
        "shared_initial_values_equal_before_copy": bool(initial_equal),
        "shared_names_shapes_values_aligned": True,
    }


def compare_full_outputs(reference, candidate) -> dict:
    result = {}
    if set(reference.keys()) != set(candidate.keys()) or set(reference.keys()) != set(OUTPUT_FIELDS):
        raise AssertionError("Full forward output field set changed")
    for field in OUTPUT_FIELDS:
        a, b = reference[field], candidate[field]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"Output shape/dtype mismatch: {field}")
        floating = a.is_floating_point()
        if floating and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise AssertionError(f"Nonfinite output: {field}")
        exact = torch.equal(a, b)
        max_abs = float((a.double()-b.double()).abs().max()) if a.numel() else 0.0
        if floating:
            torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)
        elif not exact:
            raise AssertionError(f"Non-floating output must be exact: {field}")
        result[field] = {
            "shape": list(a.shape), "dtype": str(a.dtype),
            "exact_equal": exact, "max_absolute_difference": max_abs,
            "atol": ATOL if floating else 0.0,
            "rtol": RTOL if floating else 0.0, "pass": True,
        }
    return result


def _synthetic_history(batch: int, seed: int) -> torch.Tensor:
    return torch.randn(batch, 48, 14, generator=torch.Generator().manual_seed(seed))


def _enabled_forward(model, source) -> tuple[object, dict]:
    """Count real branch/gate execution and forbid inherited forward fallback."""
    if not model.dual_path_enabled:
        raise AssertionError("Parity candidate must enable its custom forward")
    counts = {"history_branch": 0}
    gate_values = []

    def branch_hook(_module, _args, _output):
        counts["history_branch"] += 1

    def gate_hook(_module, _args, output):
        counts["regime_gate"] += 1
        gate_values.append(output.detach().clone())

    handles = [model.history_branch.register_forward_hook(branch_hook)]
    if hasattr(model, "regime_gate"):
        counts["regime_gate"] = 0
        handles.append(model.regime_gate.register_forward_hook(gate_hook))
    try:
        # Calling the inherited native forward would invalidate coverage even
        # if it happens to return equal predictions.
        with patch.object(CanonicalStrictHistoryTFT, "forward",
                          side_effect=AssertionError("Inherited forward fallback forbidden")):
            with torch.no_grad():
                output = model.forward_history(source)
    finally:
        for handle in handles:
            handle.remove()
    if counts["history_branch"] != 1:
        raise AssertionError("Expected one actual history-branch call")
    if "regime_gate" in counts:
        if counts["regime_gate"] != 1:
            raise AssertionError("Expected one actual regime-gate call")
        for gate in gate_values:
            torch.testing.assert_close(gate, torch.full_like(gate, 0.5), atol=0.0, rtol=0.0)
        counts["gate_shape"] = list(gate_values[0].shape)
        counts["gate_half_max_absolute_difference"] = float((gate_values[0]-0.5).abs().max())
    counts["inherited_forward_fallback_forbidden"] = True
    return output, counts


def enabled_neutral_fusion_audit() -> dict:
    torch.manual_seed(101)
    native = CanonicalStrictHistoryTFT(**MODEL_KWARGS).eval()
    torch.manual_seed(101)
    candidate = _NeutralFusionTFT(**MODEL_KWARGS, dual_path_enabled=True).eval()
    state = align_shared_state(native, candidate, "history_branch.")
    cases = []
    for batch, seed in CASES:
        source = _synthetic_history(batch, seed)
        with torch.no_grad():
            expected = native.forward_history(source)
        observed, calls = _enabled_forward(candidate, source)
        cases.append({"input_shape": list(source.shape), "input_seed": seed,
                      "input_sha256": tensor_hash(source), "execution": calls,
                      "outputs": compare_full_outputs(expected, observed), "pass": True})
    return {"mode": "eval", "comparison": "F1 enabled neutral fusion vs F0",
            "shared_state": state, "cases": cases, "pass": True}


def initial_f1_f2_full_forward_audit() -> dict:
    f1, f2 = make_f1().eval(), make_f2().eval()
    state = align_shared_state(f1, f2, "regime_gate.")
    cases = []
    for batch, seed in CASES:
        source = _synthetic_history(batch, seed)
        expected, f1_calls = _enabled_forward(f1, source)
        observed, f2_calls = _enabled_forward(f2, source)
        cases.append({"input_shape": list(source.shape), "input_seed": seed,
                      "input_sha256": tensor_hash(source),
                      "F1_execution": f1_calls, "F2_execution": f2_calls,
                      "outputs": compare_full_outputs(expected, observed), "pass": True})
    return {"mode": "eval", "comparison": "F2 initial g=0.5 full forward vs F1",
            "shared_state": state, "cases": cases, "pass": True}

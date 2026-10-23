from __future__ import annotations

import unittest

import torch

from ._support import make_f1, make_f2, make_history
from models.regime_aware_dualpath_tft import REGIME_FEATURE_NAMES


class TestF2RegimeGate(unittest.TestCase):
    def test_gate_contract_and_fixed_half_fusion(self) -> None:
        f1 = make_f1().eval()
        f2 = make_f2().eval()
        enc = f2.build_input(make_history())["encoder_cont"]
        with torch.no_grad():
            gate = f2.regime_weight(enc)
        self.assertEqual(tuple(gate.shape), (2, 1))
        torch.testing.assert_close(gate, torch.full_like(gate, 0.5), rtol=0.0, atol=0.0)
        a = torch.randn(2, 48, 64)
        b = torch.randn(2, 48, 64)
        torch.testing.assert_close(
            f1.fuse_encoder(a, b, enc), f2.fuse_encoder(a, b, enc), rtol=0.0, atol=0.0
        )

    def test_nonregime_change_does_not_directly_change_gate_context(self) -> None:
        model = make_f2().eval()
        enc = model.build_input(make_history())["encoder_cont"]
        changed = enc.clone()
        # raw_f2_00 is cell_voltage_mean and is not a frozen regime input.
        changed[..., 0] += 91.0
        with torch.no_grad():
            torch.testing.assert_close(model.regime_context(enc), model.regime_context(changed))
            torch.testing.assert_close(model.regime_weight(enc), model.regime_weight(changed))
        self.assertEqual(len(model.regime_indices), len(REGIME_FEATURE_NAMES))

    def test_regime_channel_change_reaches_the_gate_context(self) -> None:
        model = make_f2().eval()
        enc = model.build_input(make_history())["encoder_cont"]
        changed = enc.clone()
        changed[..., model.regime_indices[0]] += 3.0
        with torch.no_grad():
            context_delta = model.regime_context(changed) - model.regime_context(enc)
        self.assertGreater(float(context_delta[:, 0].abs().max()), 0.0)
        # The final gate projection is deliberately zero at initialization, so
        # a changed context need not yet change g. Gradient tests cover learning.

    def test_gate_receives_finite_gradients_after_synthetic_smoke_step(self) -> None:
        model = make_f2()
        optimizer = torch.optim.SGD(model.regime_gate.parameters(), lr=0.1)
        context = torch.randn(3, 5)
        # Step one moves the deliberately zero-initialized final projection.
        loss = (model.regime_gate(context) - 0.1).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.regime_gate.output.weight.grad).all())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss2 = (model.regime_gate(context) - 0.9).square().mean()
        loss2.backward()
        self.assertTrue(torch.isfinite(model.regime_gate.hidden.weight.grad).all())
        self.assertGreater(float(model.regime_gate.hidden.weight.grad.abs().sum()), 0.0)

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

from ._support import MODEL_KWARGS, make_f1, make_f2, make_history, named_state_hash

CANONICAL = Path(__file__).resolve().parents[2] / "Phase_TFT_Original_Baseline" / "models"
if str(CANONICAL) not in sys.path:
    sys.path.insert(0, str(CANONICAL))
from canonical_tft import CanonicalStrictHistoryTFT  # noqa: E402


class TestInitializationAndGradients(unittest.TestCase):
    def test_shared_initialization_and_parameter_contracts(self) -> None:
        torch.manual_seed(101)
        f0 = CanonicalStrictHistoryTFT(**MODEL_KWARGS)
        f1 = make_f1(101)
        f2 = make_f2(101)
        for name, value in f0.state_dict().items():
            self.assertTrue(torch.equal(value, f1.state_dict()[name]), name)
            self.assertTrue(torch.equal(value, f2.state_dict()[name]), name)
        self.assertEqual(
            named_state_hash(f1, "history_branch."), named_state_hash(f2, "history_branch.")
        )
        self.assertGreater(sum(p.numel() for p in f1.parameters()), sum(p.numel() for p in f0.parameters()))
        self.assertEqual(
            sum(p.numel() for p in f2.parameters()) - sum(p.numel() for p in f1.parameters()), 57
        )

    def test_synthetic_backward_connects_f1_branch_and_f2_gate(self) -> None:
        for model in (make_f1(), make_f2()):
            prediction = model.predict_delta(make_history(batch=1))
            target = torch.zeros_like(prediction)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(model.history_branch.pointwise.weight.grad).all())
            self.assertGreater(float(model.history_branch.pointwise.weight.grad.abs().sum()), 0.0)
        f2 = make_f2()
        # Gate output layer is zero-initialized but still receives a finite
        # gradient through the fusion loss.
        loss = torch.nn.functional.mse_loss(f2.predict_delta(make_history(batch=1)), torch.zeros(1, 24))
        loss.backward()
        self.assertTrue(torch.isfinite(f2.regime_gate.output.weight.grad).all())

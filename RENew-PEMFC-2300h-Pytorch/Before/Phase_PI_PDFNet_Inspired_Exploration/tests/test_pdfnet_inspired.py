"""Synthetic-only unit tests for the preregistered Phase PI architecture."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

PI = Path(__file__).resolve().parents[1]
if str(PI) not in sys.path:
    sys.path.insert(0, str(PI))

from models.embedding_periods import half_up_period, right_pad_and_segment, select_distinct_periods
from models.gated_graph import GatedDynamicGraph
from models.gpru import GatedPeriodicRecurrentUnit
from models.mixhop import BidirectionalMixHop
from models.pdfnet_inspired import PDFNetInspiredExploratory
from scripts.potential_gate import seed42_gate
from scripts.protocol import phase_path
from scripts.stage0_preflight import absolute_to_delta, synthetic_structure_audit


class PDFNetInspiredStructureTests(unittest.TestCase):
    def test_period_rules_exclude_dc_and_remove_duplicate_integer_periods(self) -> None:
        length = 128
        t = torch.arange(length, dtype=torch.float32)
        # DC has deliberately much larger amplitude, while f=50/51 both map
        # to p=3. The selector must skip DC and continue to an unrepresented p.
        wave = 100.0 + 10.0 * torch.cos(2 * math.pi * 50 * t / length) + 9.0 * torch.cos(2 * math.pi * 51 * t / length) + 8.0 * torch.cos(2 * math.pi * 32 * t / length)
        selected = select_distinct_periods(wave.reshape(1, 1, length).repeat(1, 14, 1))
        self.assertTrue(torch.all(selected.frequencies > 0))
        self.assertEqual(len(set(selected.periods[0].tolist())), 2)
        self.assertEqual(half_up_period(128, 50), 3)
        self.assertEqual(half_up_period(128, 51), 3)
        self.assertEqual(half_up_period(128, 32), 4)

    def test_right_padding_and_mask(self) -> None:
        values = torch.arange(1, 6, dtype=torch.float32).reshape(1, 5)
        segment, mask = right_pad_and_segment(values, torch.ones(5, dtype=torch.bool), 3)
        self.assertEqual(tuple(segment.shape), (2, 1, 3))
        self.assertEqual(segment[1, 0, 2].item(), 0.0)
        self.assertTrue(torch.equal(mask[1], torch.tensor([True, True, False])))

    def test_batch_independence_and_graph_response(self) -> None:
        torch.manual_seed(12)
        model = PDFNetInspiredExploratory(dimension=64).eval()
        x = torch.randn(1, 48, 14)
        other = torch.randn(1, 48, 14)
        with torch.no_grad():
            alone, alone_audit = model(x, return_audit=True)
            together, together_audit = model(torch.cat([x, other]), return_audit=True)
            changed, changed_audit = model(x + 0.5, return_audit=True)
        self.assertTrue(torch.allclose(alone[0], together[0], rtol=1e-5, atol=1e-5))
        self.assertLess(float((alone_audit["first_dynamic_graphs"].sum(-1) - 1).abs().max()), 1e-5)
        self.assertGreater(float((alone_audit["first_dynamic_graphs"] - changed_audit["first_dynamic_graphs"]).abs().sum()), 0.0)
        self.assertEqual(tuple(changed.shape), (1, 24, 14))

    def test_gpru_equation_9_manual(self) -> None:
        unit = GatedPeriodicRecurrentUnit(variables=2, dimension=2)
        with torch.no_grad():
            unit.W_r.copy_(torch.tensor([[.1, .2], [.3, .4], [.5, .6], [.7, .8]]))
            unit.W_u.copy_(torch.tensor([[.2, .1], [.4, .3], [.6, .5], [.8, .7]]))
            unit.W_o.copy_(torch.tensor([[.3, .2], [.5, .4], [.7, .6], [.9, .8]]))
            unit.b_r.copy_(torch.tensor([[.01, .02], [.03, .04]]))
            unit.b_u.copy_(torch.tensor([[.02, .01], [.04, .03]]))
            unit.b_o.copy_(torch.tensor([[.03, .02], [.01, .04]]))
        signal = torch.tensor([[.2, -.1], [.3, .4]])
        hidden = torch.tensor([[.5, .1], [-.2, .6]])
        joined = torch.cat([signal, hidden], dim=-1)
        reset = torch.sigmoid(joined @ unit.W_r + unit.b_r)
        update = torch.sigmoid(joined @ unit.W_u + unit.b_u)
        candidate = torch.tanh(torch.cat([signal, reset * hidden], dim=-1) @ unit.W_o + unit.b_o)
        expected = update * hidden + (1 - update) * candidate
        self.assertTrue(torch.allclose(unit(signal, hidden), expected, atol=1e-7, rtol=0))

    def test_graph_and_bidirectional_mixhop_manual(self) -> None:
        graph = GatedDynamicGraph(variables=2, dimension=2)
        with torch.no_grad():
            graph.W_0.copy_(torch.tensor([[.2], [.1], [.3], [.4]]))
            graph.W_1.copy_(torch.tensor([[.4], [.3], [.2], [.1]]))
        base = torch.tensor([[.7, .3], [.2, .8]])
        value = torch.tensor([[.4, .1], [.2, .6]])
        hidden = torch.tensor([[.5, .2], [.1, .7]])
        z0, z1 = base @ value, base.T @ value
        tau0 = torch.relu(torch.tanh(torch.cat([z0, hidden], -1) @ graph.W_0))
        tau1 = torch.relu(torch.tanh(torch.cat([z1, hidden], -1) @ graph.W_1))
        manual = torch.softmax(torch.relu((z0 + tau0 * hidden) @ (z1 + tau1 * hidden).T), dim=-1)
        observed, _, _ = graph(base, value, hidden)
        self.assertTrue(torch.allclose(observed, manual, atol=1e-7, rtol=0))
        mix = BidirectionalMixHop(2)
        with torch.no_grad():
            mix.weights.copy_(torch.eye(2).repeat(3, 1, 1))
        signal = torch.tensor([[.1, .2], [.3, .4]])
        def path(a: torch.Tensor) -> torch.Tensor:
            current, states = signal, [signal]
            for hop in (1, 2):
                current = .5 * signal + .5 * (torch.matrix_power(a, hop) @ current)
                states.append(current)
            return sum(states)
        expected = (path(observed) + path(observed.T)) / 2
        actual, _, _ = mix(observed, signal)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=0))

    def test_synthetic_gradients_and_delta_adapter(self) -> None:
        audit = synthetic_structure_audit(64, torch.device("cpu"))
        self.assertTrue(audit["all_core_gradients_pass"])
        output = torch.randn(2, 24, 14)
        x = torch.randn(2, 48, 14)
        baseline = absolute_to_delta(output, x)
        output[:, :, 1:] += 999.0
        self.assertTrue(torch.equal(baseline, absolute_to_delta(output, x)))

    def test_output_path_firewall(self) -> None:
        self.assertTrue(str(phase_path("outputs", "ok.json")).startswith(str(PI)))
        with self.assertRaises(RuntimeError):
            phase_path("..", "Phase_BS_Final_Backbone_Screening", "forbidden.txt")

    def test_potential_gate_boundary(self) -> None:
        fields = {"formal_RMSE_mV": 100.0, "Fold3_RMSE_mV": 100.0, "P99_mV": 100.0, "target_time_equal_RMSE_mV": 100.0, "Fold3_target_time_equal_RMSE_mV": 100.0, "endpoint_RMSE_mV": 100.0}
        flags = {"protocol_audit_pass": True, "core_gradient_pass": True, "no_leakage": True, "all_legal_origins_retained": True, "no_future_13_variable_supervision": True}
        control = torch.tensor(0)  # ensure this test has no Development dependency
        del control
        import pandas as pd
        candidate = pd.Series({**fields, **flags})
        control = pd.Series(fields)
        gate = {"seed42_gate": {"strongly_promising": {"formal_RMSE_fraction_of_TimesNet": .97, "fold3_RMSE_max_fraction": 1.1, "p99_max_fraction": 1.1, "target_time_equal_max_fraction": 1.1, "fold3_target_time_equal_max_fraction": 1.15, "endpoint_12h_max_fraction": 1.15}, "promising_A": {"formal_RMSE_max_fraction": 1.1, "fold3_RMSE_max_fraction": 1.25, "p99_max_fraction": 1.25, "target_time_equal_max_fraction": 1.2, "fold3_target_time_equal_max_fraction": 1.25, "endpoint_12h_max_fraction": 1.25}, "promising_B": {"formal_RMSE_lower_exclusive_fraction": 1.1, "formal_RMSE_upper_inclusive_fraction": 1.2, "required_improvement_fraction_on_one_preordered_axis": .1}}}
        candidate.formal_RMSE_mV = 97.0
        self.assertEqual(seed42_gate(candidate, control, gate)["status"], "PDFNET_INSPIRED_STRONGLY_PROMISING")


if __name__ == "__main__":
    unittest.main()

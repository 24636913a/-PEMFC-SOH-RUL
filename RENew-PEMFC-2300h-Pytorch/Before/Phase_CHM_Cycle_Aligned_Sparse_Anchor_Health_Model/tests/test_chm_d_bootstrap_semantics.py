from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))

from chm_bootstrap import bootstrap_weights, optimizer_seed, seed_tuple
from chm_data import load_frozen_chm_data, load_yaml, SUPPLEMENT_A
from chm_losses import prepare_smoothness_design, smoothness_loss


class CHMDBootstrapSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load_frozen_chm_data()

    def test_01_seed_tuple_and_optimizer_seed_are_frozen(self) -> None:
        self.assertEqual(seed_tuple(101, 2, 7), (90210, 101, 2, 7))
        self.assertEqual([optimizer_seed(index) for index in range(6)], [42, 2024, 3407, 42, 2024, 3407])

    def test_02_bootstrap_preserves_hou_order_and_partial_weight_one(self) -> None:
        weights, multiplicity = bootstrap_weights(self.data, 4, 101, 0, 0)
        self.assertEqual(weights.shape, (152,))
        self.assertTrue(np.all(weights[self.data.hou["hou_kind"].eq("PREANCHOR_PARTIAL_HOU").to_numpy()] == 1))
        self.assertEqual(sum(multiplicity.values()), 146)

    def test_03_same_seed_tuple_is_reproducible(self) -> None:
        one, _ = bootstrap_weights(self.data, 8, 301, 0, 3)
        two, _ = bootstrap_weights(self.data, 8, 301, 0, 3)
        self.assertTrue(np.array_equal(one, two))

    def test_04_bootstrap_p50_cannot_replace_deterministic_point_prediction(self) -> None:
        supplement = load_yaml(SUPPLEMENT_A)
        self.assertEqual(supplement["seeds_and_deterministic_prediction_semantics"]["bootstrap_P50_for_deterministic_point_prediction"], "prohibited")

    def test_05_s1_cannot_change_primary_pass_fail(self) -> None:
        supplement = load_yaml(SUPPLEMENT_A)
        self.assertFalse(supplement["S1_reporting_contract"]["primary_gate"])

    def test_06_vectorized_smoothness_matches_direct_weighted_second_differences(self) -> None:
        triples = ((0, 1, 2, 10, 11, 12), (1, 2, 3, 11, 12, 13), (2, 3, 4, 12, 13, 14))
        weights = {10: 2, 11: 1, 12: 3, 13: 4, 14: 2}
        states = torch.tensor([1.0, .95, .87, .84, .80], dtype=torch.float64)
        expected_terms = []
        expected_weights = []
        for left, middle, right, c_left, c_middle, c_right in triples:
            weight = min(weights[c_left], weights[c_middle], weights[c_right])
            expected_terms.append((states[right] - 2 * states[middle] + states[left]).square())
            expected_weights.append(weight)
        expected = (torch.stack(expected_terms) * torch.tensor(expected_weights, dtype=torch.float64)).sum() / sum(expected_weights)
        observed = smoothness_loss(states, prepare_smoothness_design(triples, weights))
        self.assertTrue(torch.allclose(observed, expected, rtol=0.0, atol=1e-14))

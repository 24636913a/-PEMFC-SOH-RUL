from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))

from chm_bootstrap import bootstrap_weights, optimizer_seed, seed_tuple
from chm_data import load_frozen_chm_data, load_yaml, SUPPLEMENT_A


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

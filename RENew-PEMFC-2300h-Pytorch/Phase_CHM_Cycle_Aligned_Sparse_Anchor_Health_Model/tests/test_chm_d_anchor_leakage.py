from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))

from chm_baselines import anchor_retention_median, nonnegative_affine_prediction, run_baseline_loao
from chm_data import ANCHORS, load_frozen_chm_data, target_for_checkpoint


class CHMDAnchorLeakageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load_frozen_chm_data()

    def test_01_every_frozen_anchor_maps_to_exactly_three_existing_hou_rows(self) -> None:
        for checkpoint in ANCHORS:
            indices = self.data.anchor_indices[checkpoint]
            self.assertEqual(len(indices), 3)
            self.assertEqual(len(set(indices)), 3)

    def test_02_b1_and_b2_reproduce_from_training_anchors_only(self) -> None:
        heldout = 1150
        baseline = run_baseline_loao(self.data).set_index("checkpoint").loc[heldout]
        training = [checkpoint for checkpoint in ANCHORS if checkpoint != heldout]
        x = np.array([anchor_retention_median(self.data, checkpoint) for checkpoint in training])
        y = np.array([target_for_checkpoint(self.data, checkpoint) for checkpoint in training])
        self.assertAlmostEqual(float(baseline["B1_prediction"]), float(IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y).predict([anchor_retention_median(self.data, heldout)])[0]))
        self.assertAlmostEqual(float(baseline["B2_prediction"]), nonnegative_affine_prediction(x, y, anchor_retention_median(self.data, heldout)))

    def test_03_heldout_target_value_is_not_part_of_baseline_fit_arrays(self) -> None:
        heldout = 1500
        training = [checkpoint for checkpoint in ANCHORS if checkpoint != heldout]
        self.assertNotIn(heldout, training)
        self.assertEqual(len(training), 5)

    def test_04_partial_hou_is_not_counted_as_complete_cycle(self) -> None:
        self.assertEqual(int(self.data.hou["hou_kind"].eq("COMPLETE_CYCLE_HOU").sum()), 146)
        self.assertEqual(int(self.data.hou["hou_kind"].eq("PREANCHOR_PARTIAL_HOU").sum()), 6)

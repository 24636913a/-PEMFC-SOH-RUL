from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


PHASE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ahc", PHASE / "scripts" / "run_affine_characterization.py")
assert SPEC and SPEC.loader
AHC = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AHC)


class AffineCharacterizationTests(unittest.TestCase):
    def test_nonnegative_affine_constraint(self) -> None:
        slope, intercept = AHC.affine_parameters(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
        self.assertEqual(slope, 0.0); self.assertEqual(intercept, 0.5)

    def test_prefix_sets_cannot_use_future_anchors(self) -> None:
        targets, mapping, baselines, _ = AHC.read_frozen_inputs(); frame = AHC.anchor_frame(targets, mapping, baselines)
        for checkpoint in (1150, 1500, 1750):
            prior = frame.loc[frame.checkpoint.lt(checkpoint), "checkpoint"].astype(int).tolist()
            self.assertTrue(all(anchor < checkpoint for anchor in prior))

    def test_frozen_b2_loao_reproduces(self) -> None:
        targets, mapping, baselines, _ = AHC.read_frozen_inputs(); frame = AHC.anchor_frame(targets, mapping, baselines); result = AHC.loao(frame)
        self.assertTrue(np.allclose(result.prediction, baselines.B2_prediction, rtol=0.0, atol=1e-10))

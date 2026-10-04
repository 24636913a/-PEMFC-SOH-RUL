from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((PHASE / "configs" / "PREREGISTERED_AHC_TV_PROTOCOL.yaml").read_text(encoding="utf-8"))
MODELS = yaml.safe_load((PHASE / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml").read_text(encoding="utf-8"))


class AHCTVProtocolStaticTests(unittest.TestCase):
    def test_01_test_checkpoints_and_after_air_air_are_frozen(self) -> None:
        self.assertEqual(PROTOCOL["primary_Test_checkpoints"], [2000, 2300]); self.assertEqual(PROTOCOL["after_Air_Air"], "EXCLUDED")

    def test_02_p1s_coefficients_and_state_are_exact(self) -> None:
        self.assertEqual(MODELS["P1S_affine"]["slope"], 1.1988114938324186); self.assertEqual(MODELS["P1S_affine"]["intercept"], -0.20903624631182127); self.assertEqual(PROTOCOL["P1S_input"]["state_token"], "221||221.200")

    def test_03_age_control_is_six_anchor_nonpositive_affine(self) -> None:
        age = MODELS["nominal_age_affine"]; checkpoints = [300, 600, 850, 1150, 1500, 1750]
        self.assertEqual(age["fitted_on_nominal_checkpoints"], checkpoints); self.assertLessEqual(age["slope_beta"], 0.0)
        targets = pd.read_csv(PHASE.parent / "Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model" / "inputs" / "CHM_SPARSE_ANCHOR_TARGETS.csv")
        x, y = np.asarray(checkpoints, dtype=float), targets["Y"].to_numpy(dtype=float)
        beta = min(0.0, float(np.dot(x-x.mean(), y-y.mean()) / np.dot(x-x.mean(), x-x.mean())))
        alpha = float(y.mean() - beta*x.mean())
        self.assertAlmostEqual(age["slope_beta"], beta); self.assertAlmostEqual(age["intercept_alpha"], alpha)

    def test_04_alignment_requirements_cannot_drift(self) -> None:
        p = PROTOCOL["P1S_input"]; self.assertEqual(p["pre_characterization_window_h"], 48); self.assertEqual(p["distinct_intervals"], 3); self.assertEqual(p["joint_support_requirement"], "2_of_3")

    def test_05_one_shot_prediction_precedes_target_read(self) -> None:
        sequence = PROTOCOL["one_shot_unblinding"]["sequence"]; self.assertLess(sequence.index("write_and_sha256_freeze_prediction_artifact"), sequence.index("read_polarization_targets"))

    def test_06_all_primary_rules_and_firewall_are_required(self) -> None:
        self.assertTrue(PROTOCOL["success_rule"]["all_required"]); self.assertEqual(PROTOCOL["secondary_reference"], "RAW_P1S_B0"); self.assertFalse(PROTOCOL["Test_accessed"]); self.assertFalse(PROTOCOL["automatic_actions"]["RUL_training"]); self.assertTrue(PROTOCOL["one_shot_unblinding"]["target_before_prediction_prohibited"])

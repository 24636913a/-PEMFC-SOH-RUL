from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))

from run_tft_reconsideration import phase_bs_alignment, vanilla_status
from tft_common import FoldPreprocessor, build_outer_folds, delta_from_absolute_scaled, load_config, read_development_prefix, split_inner_fit_stopper
from tft_models import create_model


class TestFrozenTFTProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()

    def test_data_firewall_and_f2_order(self) -> None:
        frame, audit = read_development_prefix(self.cfg)
        self.assertEqual(audit["prefix_rows_read"], 3007)
        self.assertEqual(audit["development_rows_retained"], 3005)
        self.assertFalse(audit["test_accessed"])
        self.assertEqual(len(self.cfg["data_firewall"]["f2_raw_feature_order"]), 14)
        self.assertListEqual(self.cfg["data_firewall"]["f2_raw_feature_order"], list(self.cfg["data_firewall"]["required_columns"][-14:]))
        self.assertTrue((frame.bin_end_ophrs2_h.astype(float) < self.cfg["development_end_h_exclusive"]).all())

    def test_windows_align_phase_bs_and_obey_target_rules(self) -> None:
        frame, _ = read_development_prefix(self.cfg)
        folds = build_outer_folds(frame, self.cfg)
        audit = phase_bs_alignment(folds)
        self.assertTrue(audit.target_rows_match_phase_bs.all())
        for _, train, valid in folds:
            self.assertFalse(set(train.origin_row) & set(valid.origin_row))
            self.assertTrue(np.array_equal(valid.target_row, valid.origin_row[:, None] + np.arange(1, 25)))
            self.assertLess(float(valid.target_end_age_h.max()), self.cfg["development_end_h_exclusive"])
            fit, stopper = split_inner_fit_stopper(train, .15)
            self.assertFalse(set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)))

    def test_scaler_scope_uses_fit_only(self) -> None:
        x_fit = np.zeros((2, 48, 14), dtype=np.float32)
        x_future = np.full((2, 48, 14), 1000.0, dtype=np.float32)
        scaler = FoldPreprocessor().fit(x_fit)
        self.assertTrue(np.allclose(scaler.mean, 0.0))
        transformed = scaler.transform(x_future)
        self.assertGreater(float(transformed.mean()), 100.0)

    def test_absolute_voltage_delta_semantics(self) -> None:
        x = torch.zeros(2, 48, 14)
        x[:, -1, 0] = torch.tensor([2.0, -3.0])
        absolute = torch.tensor([[2.5] * 24, [-2.0] * 24])
        delta = delta_from_absolute_scaled(absolute, x)
        self.assertTrue(torch.allclose(delta[0], torch.full((24,), .5)))
        self.assertTrue(torch.allclose(delta[1], torch.full((24,), 1.0)))

    def test_strict_history_engineering_and_native_handoff(self) -> None:
        torch.manual_seed(9)
        x = torch.randn(4, 48, 14, requires_grad=True)
        for name in ("LSTM", "sLSTM", "mLSTM"):
            model = create_model(name, 32)
            output = model(x)
            self.assertEqual(tuple(output.shape), (4, 24))
            self.assertTrue(torch.isfinite(output).all())
            output.mean().backward(retain_graph=True)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertIn("handoff", model.native_state_contract())
            self.assertEqual(int(torch.count_nonzero(model.decoder_zero_tokens(4, x.device, x.dtype))), 0)
            self.assertTrue(torch.equal(model.causal_attention_mask(x.device), torch.triu(torch.ones(72, 72, dtype=torch.bool), diagonal=1)))
            with self.assertRaises(TypeError):
                model(x.detach(), torch.randn(4, 24, 14))

    def test_gate_stops_nonviable_vanilla_route(self) -> None:
        candidate = pd.DataFrame([{"model": "LSTM", "seed": 42, "formal_RMSE_mV": 30., "Fold3_RMSE_mV": 30., "P99_mV": 30., "endpoint_RMSE_mV": 30., "target_time_equal_RMSE_mV": 30.}])
        raw = pd.DataFrame([{"seed": 42, "formal_RMSE_mV": 10., "Fold3_RMSE_mV": 10., "P99_mV": 10., "endpoint_RMSE_mV": 10., "target_time_equal_RMSE_mV": 10.}])
        status, _ = vanilla_status(candidate, raw)
        self.assertEqual(status, "TFT_NOT_VIABLE")


if __name__ == "__main__":
    unittest.main()

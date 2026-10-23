from __future__ import annotations

import unittest

import torch

from ._support import make_f1, make_history


class TestF1ShapesAndCausality(unittest.TestCase):
    def test_full_model_output_shape_and_attention_contract(self) -> None:
        model = make_f1().eval()
        with torch.no_grad():
            result = model.forward_history(make_history())
        self.assertEqual(tuple(result["prediction"].shape), (2, 24, 1))
        self.assertEqual(tuple(result["encoder_attention"].shape), (2, 24, 4, 48))
        self.assertEqual(tuple(result["decoder_attention"].shape), (2, 24, 4, 24))

    def test_branch_is_positionwise_causal_and_length_preserving(self) -> None:
        model = make_f1().eval()
        generator = torch.Generator().manual_seed(22)
        source = torch.randn(2, 48, 64, generator=generator)
        changed = source.clone()
        cutoff = 21
        changed[:, cutoff + 1 :, :] += 100.0
        with torch.no_grad():
            original_out = model.encode_history_branch(source)
            changed_out = model.encode_history_branch(changed)
        self.assertEqual(tuple(original_out.shape), (2, 48, 64))
        torch.testing.assert_close(
            original_out[:, : cutoff + 1], changed_out[:, : cutoff + 1], rtol=0.0, atol=0.0
        )
        self.assertEqual(model.history_branch.short.left_padding, 2)
        self.assertEqual(model.history_branch.medium.left_padding, 8)

    def test_native_encoder_state_is_handed_unchanged_to_native_decoder(self) -> None:
        model = make_f1().eval()
        captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def encoder_hook(_module, _inputs, output):
            captured["encoder_state"] = (output[1][0].detach().clone(), output[1][1].detach().clone())

        def decoder_pre_hook(_module, inputs):
            captured["decoder_state"] = (inputs[1][0].detach().clone(), inputs[1][1].detach().clone())

        encoder_handle = model.lstm_encoder.register_forward_hook(encoder_hook)
        decoder_handle = model.lstm_decoder.register_forward_pre_hook(decoder_pre_hook)
        try:
            with torch.no_grad():
                model.predict_delta(make_history())
        finally:
            encoder_handle.remove()
            decoder_handle.remove()
        self.assertIn("encoder_state", captured)
        self.assertIn("decoder_state", captured)
        for encoder_state, decoder_state in zip(captured["encoder_state"], captured["decoder_state"]):
            torch.testing.assert_close(encoder_state, decoder_state, rtol=0.0, atol=0.0)

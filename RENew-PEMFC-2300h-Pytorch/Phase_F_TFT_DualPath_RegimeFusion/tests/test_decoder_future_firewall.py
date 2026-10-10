from __future__ import annotations

import unittest

import torch

from ._support import make_f1, make_f2, make_history


class TestDecoderFutureFirewall(unittest.TestCase):
    def _assert_decoder_placeholders_inert(self, model) -> None:
        model.eval()
        source = make_history()
        x = model.build_input(source)
        modified = {name: value.clone() for name, value in x.items()}
        # The raw decoder schema slots exist only for tensor shape; decoder VSN
        # selects relative_forecast_step alone.
        modified["decoder_cont"][..., :14] = 999.0
        with torch.no_grad():
            base = model.forward(x)["prediction"]
            altered = model.forward(modified)["prediction"]
        torch.testing.assert_close(base, altered, rtol=0.0, atol=0.0)

    def test_f1_future_placeholder_firewall(self) -> None:
        self._assert_decoder_placeholders_inert(make_f1())

    def test_f2_future_placeholder_firewall(self) -> None:
        self._assert_decoder_placeholders_inert(make_f2())

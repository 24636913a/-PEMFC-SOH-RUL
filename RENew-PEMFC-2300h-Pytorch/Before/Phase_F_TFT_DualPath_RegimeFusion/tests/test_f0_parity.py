from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

from ._support import MODEL_KWARGS, make_history

CANONICAL = Path(__file__).resolve().parents[2] / "Phase_TFT_Original_Baseline" / "models"
if str(CANONICAL) not in sys.path:
    sys.path.insert(0, str(CANONICAL))
from canonical_tft import CanonicalStrictHistoryTFT  # noqa: E402
from models import DualPathStrictHistoryTFT  # noqa: E402


class TestF0Parity(unittest.TestCase):
    def test_disabled_dualpath_delegates_exactly_to_native_tft(self) -> None:
        torch.manual_seed(17)
        native = CanonicalStrictHistoryTFT(**MODEL_KWARGS).eval()
        torch.manual_seed(17)
        disabled = DualPathStrictHistoryTFT(**MODEL_KWARGS, dual_path_enabled=False).eval()
        self.assertEqual(native.state_dict().keys(), disabled.state_dict().keys())
        for name, value in native.state_dict().items():
            self.assertTrue(torch.equal(value, disabled.state_dict()[name]), name)
        x = make_history()
        with torch.no_grad():
            expected = native.predict_delta(x)
            observed = disabled.predict_delta(x)
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(observed.shape), (2, 24))

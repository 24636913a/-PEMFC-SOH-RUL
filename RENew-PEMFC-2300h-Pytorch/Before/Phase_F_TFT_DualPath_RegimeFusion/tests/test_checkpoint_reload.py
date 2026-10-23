from __future__ import annotations

import tempfile
import unittest

import torch

from ._support import make_f2, make_history


class TestCheckpointReload(unittest.TestCase):
    def test_synthetic_checkpoint_reload_parity(self) -> None:
        model = make_f2().eval()
        source = make_history()
        with torch.no_grad():
            expected = model.predict_delta(source)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = f"{temporary}/synthetic_f2_state.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = make_f2().eval()
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            with torch.no_grad():
                observed = restored.predict_delta(source)
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

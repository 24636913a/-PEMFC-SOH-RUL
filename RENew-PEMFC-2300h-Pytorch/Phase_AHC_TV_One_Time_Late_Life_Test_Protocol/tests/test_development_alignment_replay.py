from __future__ import annotations

import sys
import unittest
from pathlib import Path


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from ahc_tv_alignment import replay_development_alignment  # noqa: E402


class DevelopmentAlignmentReplayTests(unittest.TestCase):
    def test_all_six_frozen_development_anchors_replay_exactly(self) -> None:
        replay = replay_development_alignment(PHASE)
        self.assertTrue(replay["all_pass"])
        self.assertEqual(replay["pass_count"], 6)
        self.assertEqual([row["checkpoint"] for row in replay["anchors"]], [300, 600, 850, 1150, 1500, 1750])
        for row in replay["anchors"]:
            self.assertTrue(row["interval_ids_pass"])
            self.assertTrue(row["retention_pass_at_1e12"])
            self.assertTrue(row["support_count_pass"])

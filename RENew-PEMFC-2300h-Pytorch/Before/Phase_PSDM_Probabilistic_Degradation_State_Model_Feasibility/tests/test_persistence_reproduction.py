"""Reproduce CMRF Persistence before PSDM scoring is permitted."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_psdm_feasibility.py"
SPEC = importlib.util.spec_from_file_location("psdm_runner", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MOD; SPEC.loader.exec_module(MOD)


class PersistenceReproductionTests(unittest.TestCase):
    def test_frozen_cmrf_authorities_reproduce_persistence(self) -> None:
        events = MOD.load_p1_events(); authorities = {h: MOD.load_authority(h, events) for h in (3, 6)}
        MOD.assert_persistence_reproduction(authorities)
        self.assertAlmostEqual(MOD.persistence_equal_fold(authorities[3]), 0.36629848657406594, places=10)
        self.assertAlmostEqual(MOD.persistence_equal_fold(authorities[6]), 0.6041042435185089, places=10)


if __name__ == "__main__": unittest.main()

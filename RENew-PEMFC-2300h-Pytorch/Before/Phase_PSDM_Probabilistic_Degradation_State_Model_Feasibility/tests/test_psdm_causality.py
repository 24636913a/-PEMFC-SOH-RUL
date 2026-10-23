"""Production-path causality and physical-cycle propagation tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_psdm_feasibility.py"
SPEC = importlib.util.spec_from_file_location("psdm_runner", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MOD; SPEC.loader.exec_module(MOD)


class PSDMCausalityTests(unittest.TestCase):
    def test_raw_cycle_gap_uses_exact_number_of_transitions(self) -> None:
        x = np.array([1.0, 0.01]); p = np.eye(2) * 1e-6
        advanced, _ = MOD.transition("ULLT", x, p, 1e-8, 3)
        self.assertAlmostEqual(float(advanced[0]), 1.03, places=12)

    def test_history_cutoff_excludes_future_observation(self) -> None:
        events = pd.DataFrame({"raw_cycle_id": [11, 12, 13], "p1s_measurement_start_ophrs2_h": [100., 110., 120.], "p1s_measurement_end_ophrs2_h": [101., 111., 121.], "R_P1S_causal": [.99, .98, .97]})
        past = MOD.observations_for_time(events, 72.788690, 111.)
        self.assertEqual(past.raw_cycle_id.tolist(), [11, 12])

    def test_horizon_is_not_replaced_by_future_event_count(self) -> None:
        x = np.array([.99, -6.0]); p = np.eye(2) * 1e-6
        h3, _ = MOD.transition("PSDM", x, p, 1e-8, 3)
        h6, _ = MOD.transition("PSDM", x, p, 1e-8, 6)
        self.assertLess(float(h6[0]), float(h3[0]))


if __name__ == "__main__": unittest.main()

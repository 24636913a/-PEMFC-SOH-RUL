from __future__ import annotations

import sys
import unittest
from pathlib import Path


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from ahc_tv_alignment import HistoricalTestDerivedInputRejected, assert_runtime_input_allowed  # noqa: E402


class HistoricalTestOutputBlocklistTests(unittest.TestCase):
    def test_n1_test_derived_paths_are_rejected_without_reading_contents(self) -> None:
        root = PHASE.parents[1] / "RENew-PEMFC-2300h-Pytorch" / "Phase_N1_Data_Foundation" / "outputs" / "soh_anchor_analysis"
        for name in ("soh_reference_anchors.csv", "polarization_checkpoint_jstar_measurements.csv", "figures/example.png"):
            with self.assertRaises(HistoricalTestDerivedInputRejected) as caught:
                assert_runtime_input_allowed(root / name)
            self.assertIn("HISTORICAL_TEST_DERIVED_INPUT_REJECTED", str(caught.exception))

    def test_non_tainted_frozen_paa_input_is_not_blocked(self) -> None:
        paa = PHASE.parents[1] / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs" / "P1S_REFERENCE_COMPLETE_CYCLES.csv"
        self.assertEqual(assert_runtime_input_allowed(paa), paa.resolve())

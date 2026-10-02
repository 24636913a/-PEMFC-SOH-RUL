from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_paa_anchor_validity_qualification.py"
SPEC = importlib.util.spec_from_file_location("pav_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PAVProductionGuardTests(unittest.TestCase):
    def test_test_polarization_and_eis_paths_are_rejected(self) -> None:
        for name in [
            "Polcurve_Nominal_RHFC_2000h.csv",
            "Polcurve_Nominal_RHFC_2300h_afterAirAirStart.csv",
            "EIS_RHFC_1750h.csv",
            "Polcurve_Nominal_RHFC_test.csv",
        ]:
            with self.assertRaises(MODULE.QualificationFailure):
                MODULE.strict_polarization_path(Path(name))

    def test_allowlisted_development_polarization_is_accepted(self) -> None:
        MODULE.strict_polarization_path(Path("Polcurve_Nominal_RHFC_850h.csv"))

    def test_median_source_is_the_middle_frozen_value_not_recomputed_anchor(self) -> None:
        rows = pd.DataFrame(
            {
                "cycle_interval_id": ["FIRST", "SECOND", "THIRD"],
                "P1S_operating_voltage_retention_anchor": [0.995, 0.991, 0.993],
            }
        )
        result = MODULE.median_source(rows)
        self.assertEqual(result["cycle_interval_id"], "THIRD")

    def test_nonunique_median_source_fails_closed(self) -> None:
        rows = pd.DataFrame(
            {
                "cycle_interval_id": ["FIRST", "SECOND", "THIRD"],
                "P1S_operating_voltage_retention_anchor": [0.995, 0.993, 0.993],
            }
        )
        with self.assertRaises(MODULE.QualificationFailure):
            MODULE.median_source(rows)

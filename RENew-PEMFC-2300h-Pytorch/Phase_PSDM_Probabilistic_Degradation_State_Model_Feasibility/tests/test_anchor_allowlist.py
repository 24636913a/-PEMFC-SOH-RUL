"""Regression tests for the closed Development-only polarization allowlist."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dev_anchor_authority.py"
SPEC = importlib.util.spec_from_file_location("psdm_anchor_builder", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_source() -> pd.DataFrame:
    rows = []
    reference = {0.8: 0.718128, 1.0: 0.6989533066, 1.2: 0.676958, 1.4: 0.657588, 1.6: 0.6373789965}
    for stage in ("BoL", "300h", "600h", "850h", "1150h", "1500h", "1500h_afterAirAirStart", "1750h"):
        for jstar, voltage in reference.items():
            rows.append({"filename": f"{stage}.csv", "stage": stage, "after_air_air_start": stage == "1500h_afterAirAirStart", "target_jstar_A_cm2": jstar, "active_voltage_median_V": voltage if stage == "BoL" else voltage * 0.98, "valid_sample_count": 10, "selection_scope": "development-only"})
    return pd.DataFrame(rows)


class AnchorAllowlistTests(unittest.TestCase):
    def test_invalid_stage_is_rejected(self) -> None:
        frame = valid_source(); frame.loc[0, "stage"] = "2000h"
        with self.assertRaisesRegex(MODULE.AnchorAuthorityFailure, "UNEXPECTED_POLARIZATION_STAGE"):
            MODULE.validate_anchor_frame(frame)

    def test_invalid_boolean_is_rejected(self) -> None:
        frame = valid_source(); frame["after_air_air_start"] = frame["after_air_air_start"].astype(object); frame.loc[0, "after_air_air_start"] = "maybe"
        with self.assertRaisesRegex(MODULE.AnchorAuthorityFailure, "INVALID_AFTER_AIR_AIR_BOOLEAN"):
            MODULE.validate_anchor_frame(frame)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))

from ahc_tv_stage_p import RuntimeSourceBindingError, bind_both_polarization_sources, bind_unique_polarization_source  # noqa: E402


class RealHyFCSourceBindingTests(unittest.TestCase):
    """Exercise the production filename binder without opening any CSV."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def candidate(self, name: str) -> Path:
        return self.root / name

    def bind(self, checkpoint: int, candidates: list[Path]) -> Path:
        return bind_unique_polarization_source(
            checkpoint, candidates, phase=PHASE, synthetic_test_mode=True
        )

    def test_nominal_2000_and_2300_each_bind(self) -> None:
        source_2000 = self.candidate("Polcurve_Nominal_RHFC_2000h.csv")
        source_2300 = self.candidate("Polcurve_Nominal_RHFC_2300h.csv")
        self.assertEqual(self.bind(2000, [source_2000]).name, source_2000.name)
        self.assertEqual(self.bind(2300, [source_2300]).name, source_2300.name)

    def test_after_air_air_name_variants_are_excluded(self) -> None:
        for suffix in ("afterAirAirStart", "after_air_air", "after-air-air"):
            with self.subTest(suffix=suffix):
                after = self.candidate(f"Polcurve_Nominal_RHFC_2300h_{suffix}.csv")
                with self.assertRaisesRegex(RuntimeSourceBindingError, "TEST_SOURCE_BINDING_AMBIGUOUS"):
                    self.bind(2300, [after])

    def test_nominal_plus_after_air_air_selects_unique_nominal(self) -> None:
        nominal = self.candidate("Polcurve_Nominal_RHFC_2300h.csv")
        after = self.candidate("Polcurve_Nominal_RHFC_2300h_afterAirAirStart.csv")
        self.assertEqual(self.bind(2300, [nominal, after]), nominal.resolve())

    def test_two_nominal_files_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeSourceBindingError, "TEST_SOURCE_BINDING_AMBIGUOUS"):
            self.bind(2300, [
                self.candidate("Polcurve_Nominal_RHFC_2300h.csv"),
                self.candidate("Polcurve_Nominal_RHFC_2300h_duplicate.csv"),
            ])

    def test_explicit_forbidden_mapping_and_wrong_checkpoint_mapping_fail_closed(self) -> None:
        source_2000 = self.candidate("Polcurve_Nominal_RHFC_2000h.csv")
        source_2300 = self.candidate("Polcurve_Nominal_RHFC_2300h.csv")
        after = self.candidate("Polcurve_Nominal_RHFC_2300h_after_air_air.csv")
        with self.assertRaisesRegex(RuntimeSourceBindingError, "TEST_SOURCE_BINDING_AMBIGUOUS"):
            bind_both_polarization_sources(
                {2000: source_2000, 2300: after}, phase=PHASE, synthetic_test_mode=True
            )
        with self.assertRaisesRegex(RuntimeSourceBindingError, "TEST_SOURCE_BINDING_AMBIGUOUS"):
            bind_both_polarization_sources(
                {2000: source_2300, 2300: source_2000}, phase=PHASE, synthetic_test_mode=True
            )


if __name__ == "__main__":
    unittest.main()

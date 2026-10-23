from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))

from ahc_tv_alignment import REQUIRED_TEST_AUTHORIZATION, read_yaml  # noqa: E402
from ahc_tv_stage_p import (  # noqa: E402
    RuntimeSourceBindingError,
    bind_both_polarization_sources,
    bind_unique_polarization_source,
    verify_frozen_polarization_source_identity,
)
from run_ahc_tv_one_shot import run_one_shot  # noqa: E402


class RealHyFCSourceBindingTests(unittest.TestCase):
    """Exercise the production filename binder without opening any CSV."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def candidate(self, name: str) -> Path:
        return self.root / name

    def frozen_candidate(self, checkpoint: int) -> Path:
        registry = read_yaml(PHASE / "configs" / "AHC_TV_RUNTIME_SOURCE_REGISTRY.yaml")
        return Path(registry["local_dataset_root"]["path"]) / registry["polarization_checkpoint_sources"][str(checkpoint)]["frozen_relative_path"]

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

    def test_exact_frozen_paths_pass_identity_validation(self) -> None:
        for checkpoint in (2000, 2300):
            with self.subTest(checkpoint=checkpoint):
                path = self.frozen_candidate(checkpoint)
                self.assertEqual(
                    verify_frozen_polarization_source_identity(checkpoint, path, phase=PHASE),
                    path.resolve(),
                )

    def test_path_identity_rejects_alternate_swap_subdirectory_and_outside_root(self) -> None:
        source_2000 = self.frozen_candidate(2000)
        source_2300 = self.frozen_candidate(2300)
        alternatives = (
            self.frozen_candidate(2300).with_name("Polcurve_Nominal_RHFC_2300h_alternate.csv"),
            self.frozen_candidate(2300).with_name("Polcurve_Nominal_RHFC_2300h_afterAirAirStart.csv"),
            self.frozen_candidate(2300).parent / "alternate" / source_2300.name,
            self.root / source_2300.name,
            source_2000,
        )
        for candidate in alternatives:
            with self.subTest(candidate=str(candidate)):
                with self.assertRaisesRegex(RuntimeSourceBindingError, "TEST_SOURCE_IDENTITY_MISMATCH"):
                    verify_frozen_polarization_source_identity(2300, candidate, phase=PHASE)

    def test_identity_mismatch_prevents_ledger_runtime_and_all_readers(self) -> None:
        raw = self.frozen_candidate(2000).parent / "Loadcycle_preflight_only.csv"
        wrong_2000 = self.frozen_candidate(2000).parent / "alternate" / self.frozen_candidate(2000).name
        correct_2300 = self.frozen_candidate(2300)
        with (
            mock.patch("run_ahc_tv_one_shot.create_execution_ledger") as create_ledger,
            mock.patch("run_ahc_tv_one_shot.build_and_freeze_stage_p") as reader_m_and_raw,
            mock.patch("run_ahc_tv_one_shot.unblind_both_test_targets_once") as reader_t,
        ):
            with self.assertRaisesRegex(RuntimeSourceBindingError, "TEST_SOURCE_IDENTITY_MISMATCH"):
                run_one_shot(
                    authorization=REQUIRED_TEST_AUTHORIZATION,
                    raw_sources=[raw],
                    polarization_2000=wrong_2000,
                    polarization_2300=correct_2300,
                    phase=PHASE,
                )
        create_ledger.assert_not_called()
        reader_m_and_raw.assert_not_called()
        reader_t.assert_not_called()


if __name__ == "__main__":
    unittest.main()

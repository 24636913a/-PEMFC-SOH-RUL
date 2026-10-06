from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from ahc_tv_alignment import (  # noqa: E402
    HistoricalTestDerivedInputRejected,
    REQUIRED_TEST_AUTHORIZATION,
    TargetUnblindingBlocked,
)
from ahc_tv_stage_p import (  # noqa: E402
    RuntimeSourceBindingError,
    StagePError,
    build_and_freeze_stage_p,
)
from ahc_tv_stage_t import unblind_both_test_targets_once  # noqa: E402
from run_ahc_tv_one_shot import run_one_shot  # noqa: E402


ACTIVE = [f"U{i} [V]" for i in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)]
CONDITIONS = {
    "j.S.act [A/cm2]": 0.789229989,
    "T.Si.A [°C]": 75.08,
    "p.Si.A [kPag]": 164.44,
    "RH.Si.A  [%]": 40.25,
    "F.Si.H.A [nl/min]": 23.305,
    "Stoic.S.A [-]": 1.512,
    "Stoic.S.C [-]": 1.7115,
}


class StagePStageTIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "Loadcycle_synthetic.csv"
        self.polar_2000 = self.root / "Polarization_2000h.csv"
        self.polar_2300 = self.root / "Polarization_2300h.csv"
        self._write_polarization(self.polar_2000, "2025-07-01 00:00:00", 0.690)
        self._write_polarization(self.polar_2300, "2026-01-01 00:00:00", 0.675)
        self._write_raw(self.raw)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_polarization(self, path: Path, start: str, voltage: float) -> None:
        data = {
            "DateTime": [start, str(pd.Timestamp(start) + pd.Timedelta(minutes=1))],
            "j.S.act [A/cm2]": [1.0, 1.0],
            "TARGET_SENTINEL_MUST_NOT_REACH_STAGE_P": [987654.0, 987654.0],
        }
        data.update({column: [voltage, voltage - 0.001] for column in ACTIVE})
        pd.DataFrame(data).to_csv(path, sep=";", encoding="utf-8-sig", index=False)

    def _cycle_rows(self, start: pd.Timestamp, age: float, voltage: float, *, supported: bool = True) -> list[dict]:
        rows: list[dict] = []
        current_density = CONDITIONS["j.S.act [A/cm2]"] if supported else 0.20
        for second, operation, set_current in ((0, 15, 14.0), (1, 15, 0.0)):
            row = {
                "DateTime": str(start + pd.Timedelta(seconds=second)),
                "OpHrs2 [h]": age + second / 3600.0,
                "operation_point": operation,
                "I.S.set [A]": set_current,
                "I.S.act [A]": set_current,
                **CONDITIONS,
            }
            row["j.S.act [A/cm2]"] = current_density
            row.update({column: voltage for column in ACTIVE})
            rows.append(row)
        for second in range(2, 123):
            row = {
                "DateTime": str(start + pd.Timedelta(seconds=second)),
                "OpHrs2 [h]": age + second / 3600.0,
                "operation_point": 221,
                "I.S.set [A]": 221.2,
                "I.S.act [A]": 221.2,
                **CONDITIONS,
            }
            row["j.S.act [A/cm2]"] = current_density
            row.update({column: voltage for column in ACTIVE})
            rows.append(row)
        return rows

    def _write_raw(
        self,
        path: Path,
        *,
        cycles_2000: int = 4,
        support_only_one_2000: bool = False,
        include_abnormal_after_2000: bool = True,
        include_abnormal_after_2300: bool = True,
    ) -> None:
        rows: list[dict] = []
        for index in range(cycles_2000):
            rows.extend(self._cycle_rows(
                pd.Timestamp("2025-06-30 14:00:00") + pd.Timedelta(hours=2 * index),
                1990.0 + index,
                0.707 - 0.001 * index,
                supported=not (support_only_one_2000 and index >= max(1, cycles_2000 - 2)),
            ))
        if include_abnormal_after_2000:
            rows.extend(self._cycle_rows(pd.Timestamp("2025-07-01 02:00:00"), 2001.0, 0.100, supported=True))
        for index in range(4):
            rows.extend(self._cycle_rows(
                pd.Timestamp("2025-12-31 14:00:00") + pd.Timedelta(hours=2 * index),
                2290.0 + index,
                0.695 - 0.001 * index,
                supported=True,
            ))
        if include_abnormal_after_2300:
            rows.extend(self._cycle_rows(pd.Timestamp("2026-01-01 02:00:00"), 2301.0, 0.100, supported=True))
        pd.DataFrame(rows).to_csv(path, sep=";", encoding="utf-8-sig", index=False)

    def _stage_p(self, runtime_name: str = "runtime", **kwargs) -> tuple[dict, list[str]]:
        events: list[str] = []
        result = build_and_freeze_stage_p(
            raw_sources=[kwargs.pop("raw_source", self.raw)],
            polarization_sources=kwargs.pop("polarization_sources", {2000: self.polar_2000, 2300: self.polar_2300}),
            runtime_dir=self.root / runtime_name,
            authorization_token=kwargs.pop("authorization_token", REQUIRED_TEST_AUTHORIZATION),
            event_log=events,
            synthetic_test_mode=True,
            **kwargs,
        )
        return result, events

    def test_synthetic_end_to_end_stage_p_then_atomic_stage_t_has_required_event_order(self) -> None:
        stage_p, events = self._stage_p()
        observed_stage_p_usecols: list[list[str]] = []

        def metadata_reader(path, **kwargs):
            observed_stage_p_usecols.append(kwargs["usecols"])
            self.assertEqual(kwargs["usecols"], ["DateTime"])
            return pd.read_csv(path, **kwargs)

        stage_p, events = self._stage_p("runtime_spy", metadata_read_csv=metadata_reader)
        self.assertEqual(observed_stage_p_usecols, [["DateTime"], ["DateTime"]])
        stage_t = unblind_both_test_targets_once(
            polarization_sources={2000: self.polar_2000, 2300: self.polar_2300},
            frozen_prediction_path=stage_p["prediction_path"],
            authorization_token=REQUIRED_TEST_AUTHORIZATION,
            event_log=events,
        )
        self.assertEqual(
            events,
            [
                "AUTHORIZATION_OK", "METADATA_2000_READ", "METADATA_2300_READ",
                "RAW_STAGE_P_READ", "PREDICTION_WRITTEN", "PREDICTION_HASH_FROZEN",
                "PROVENANCE_WRITTEN", "PROVENANCE_HASH_FROZEN", "STRICT_VERIFY_PASS",
                "STAGE_P_COMPLETION_FROZEN", "TARGET_2000_READ", "TARGET_2300_READ",
                "DECISION_COMPUTED",
            ],
        )
        self.assertEqual(stage_t["target_results"]["checkpoint"].tolist(), [2000, 2300])
        self.assertTrue((stage_t["target_results"]["target_Y"] > 0).all())
        self.assertTrue(stage_p["provenance_path"].exists())
        self.assertTrue(stage_p["completion_path"].exists())

    def test_missing_or_wrong_authorization_reads_no_metadata_raw_or_target(self) -> None:
        calls: list[str] = []

        def forbidden(*args, **kwargs):
            calls.append("read")
            raise AssertionError("no Test source may be opened")

        with self.assertRaises(TargetUnblindingBlocked):
            build_and_freeze_stage_p(
                raw_sources=[self.raw],
                polarization_sources={2000: self.polar_2000, 2300: self.polar_2300},
                runtime_dir=self.root / "no_auth",
                authorization_token="wrong",
                raw_read_csv=forbidden,
                metadata_read_csv=forbidden,
            )
        self.assertEqual(calls, [])
        with self.assertRaises(TargetUnblindingBlocked):
            run_one_shot(
                authorization="wrong", raw_sources=[self.raw], polarization_2000=self.polar_2000,
                polarization_2300=self.polar_2300,
            )

    def test_stage_p_fails_for_less_than_three_intervals_or_one_of_three_support(self) -> None:
        too_few = self.root / "Loadcycle_too_few.csv"
        self._write_raw(too_few, cycles_2000=2)
        with self.assertRaisesRegex(StagePError, "TEST_ALIGNMENT_INCONCLUSIVE"):
            self._stage_p("few", raw_source=too_few)
        one_support = self.root / "Loadcycle_one_support.csv"
        self._write_raw(one_support, support_only_one_2000=True)
        with self.assertRaisesRegex(StagePError, "TEST_ALIGNMENT_INCONCLUSIVE"):
            self._stage_p("one_support", raw_source=one_support)

    def test_missing_raw_source_and_ambiguous_or_swapped_polarization_binding_fail_closed(self) -> None:
        with self.assertRaisesRegex(StagePError, "TEST_RAW_SOURCE_MISSING"):
            self._stage_p("missing_raw", raw_source=self.root / "Loadcycle_missing.csv")
        duplicate_2000 = self.root / "Polarization_2000h_duplicate.csv"
        self._write_polarization(duplicate_2000, "2025-07-01 00:00:00", 0.690)
        with self.assertRaisesRegex(RuntimeSourceBindingError, "TEST_SOURCE_BINDING_AMBIGUOUS"):
            self._stage_p("ambiguous", polarization_sources=[self.polar_2000, duplicate_2000, self.polar_2300])
        stage_p, _ = self._stage_p("swapped_base")
        target_called = False

        def target_reader(*args, **kwargs):
            nonlocal target_called
            target_called = True
            raise AssertionError("target should remain unopened")

        with self.assertRaisesRegex(RuntimeSourceBindingError, "TEST_SOURCE_BINDING_AMBIGUOUS"):
            unblind_both_test_targets_once(
                polarization_sources={2000: self.polar_2300, 2300: self.polar_2000},
                frozen_prediction_path=stage_p["prediction_path"],
                authorization_token=REQUIRED_TEST_AUTHORIZATION,
                read_csv=target_reader,
            )
        self.assertFalse(target_called)

    def test_tampered_prediction_or_provenance_blocks_targets_before_any_target_read(self) -> None:
        stage_p, _ = self._stage_p("tamper_prediction")
        prediction = stage_p["prediction_path"]
        text = prediction.read_text(encoding="utf-8")
        prediction.write_text(text.replace("P1S_affine_prediction", "P1S_affine_prediction_x", 1), encoding="utf-8")
        called = False

        def forbidden(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("target reader must not run")

        with self.assertRaises(TargetUnblindingBlocked):
            unblind_both_test_targets_once(
                polarization_sources={2000: self.polar_2000, 2300: self.polar_2300},
                frozen_prediction_path=prediction, authorization_token=REQUIRED_TEST_AUTHORIZATION,
                read_csv=forbidden,
            )
        self.assertFalse(called)
        stage_p, _ = self._stage_p("tamper_provenance")
        stage_p["provenance_path"].write_text("{}", encoding="utf-8")
        with self.assertRaises(TargetUnblindingBlocked):
            unblind_both_test_targets_once(
                polarization_sources={2000: self.polar_2000, 2300: self.polar_2300},
                frozen_prediction_path=stage_p["prediction_path"], authorization_token=REQUIRED_TEST_AUTHORIZATION,
                read_csv=forbidden,
            )
        self.assertFalse(called)

    def test_future_raw_rows_after_each_checkpoint_do_not_enter_that_checkpoint_prediction(self) -> None:
        clean = self.root / "Loadcycle_clean.csv"
        self._write_raw(clean, include_abnormal_after_2000=False, include_abnormal_after_2300=False)
        future = self.root / "Loadcycle_future.csv"
        self._write_raw(future, include_abnormal_after_2000=True, include_abnormal_after_2300=True)
        before_2300 = self.root / "Loadcycle_before_2300.csv"
        self._write_raw(before_2300, include_abnormal_after_2000=True, include_abnormal_after_2300=False)
        clean_p, _ = self._stage_p("clean", raw_source=clean)
        future_p, _ = self._stage_p("future", raw_source=future)
        before_2300_p, _ = self._stage_p("before_2300", raw_source=before_2300)
        clean_table = pd.read_csv(clean_p["prediction_path"]).set_index("checkpoint")
        future_table = pd.read_csv(future_p["prediction_path"]).set_index("checkpoint")
        before_2300_table = pd.read_csv(before_2300_p["prediction_path"]).set_index("checkpoint")
        self.assertEqual(clean_table.loc[2000, "selected_interval_1"], future_table.loc[2000, "selected_interval_1"])
        self.assertAlmostEqual(clean_table.loc[2000, "P1S_R_median"], future_table.loc[2000, "P1S_R_median"])
        self.assertEqual(before_2300_table.loc[2300, "selected_interval_1"], future_table.loc[2300, "selected_interval_1"])
        self.assertAlmostEqual(before_2300_table.loc[2300, "P1S_R_median"], future_table.loc[2300, "P1S_R_median"])

    def test_historical_n1_paths_are_blocked_as_runtime_inputs(self) -> None:
        tainted = (
            PHASE.parents[1] / "RENew-PEMFC-2300h-Pytorch" / "Phase_N1_Data_Foundation" /
            "outputs" / "soh_anchor_analysis" / "soh_reference_anchors.csv"
        )
        with self.assertRaises(HistoricalTestDerivedInputRejected):
            self._stage_p("n1_raw", raw_source=tainted)
        with self.assertRaises(HistoricalTestDerivedInputRejected):
            self._stage_p("n1_polar", polarization_sources={2000: tainted, 2300: self.polar_2300})

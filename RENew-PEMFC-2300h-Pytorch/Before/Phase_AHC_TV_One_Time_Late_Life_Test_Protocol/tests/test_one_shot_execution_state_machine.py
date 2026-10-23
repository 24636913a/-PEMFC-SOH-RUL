from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))

from ahc_tv_one_shot_ledger import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_DECISION,
    STATUS_STAGE_P,
    STATUS_STARTED,
    STATUS_TARGETS,
    STATUS_TARGET_STARTED,
    OneShotLedgerError,
    abort_execution_ledger,
    canonical_run_dir,
    create_execution_ledger,
    ledger_path,
    load_frozen_results_for_offline_postprocessing,
    read_ledger,
    transition_execution_ledger,
    write_exclusive_json,
    write_exclusive_sha256_sidecar,
    write_exclusive_text,
)
from run_ahc_tv_one_shot import run_one_shot  # noqa: E402


class OneShotExecutionStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.phase = Path(self.temp.name) / "Phase_AHC_TV_One_Time_Late_Life_Test_Protocol"
        self.phase.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _start(self) -> None:
        create_execution_ledger(phase=self.phase, authorization_marker="OPEN_AHC_TV_TEST", base_commit="synthetic")

    def test_no_or_wrong_authorization_creates_no_ledger(self) -> None:
        with self.assertRaises(Exception):
            run_one_shot(
                authorization="wrong", raw_sources=[self.phase / "raw.csv"],
                polarization_2000=self.phase / "p2000.csv", polarization_2300=self.phase / "p2300.csv", phase=self.phase,
            )
        self.assertFalse(ledger_path(self.phase).exists())

    def test_authority_failure_creates_no_ledger(self) -> None:
        with mock.patch("run_ahc_tv_one_shot.verify_frozen_runtime_authority", side_effect=RuntimeError("hash mismatch")):
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                run_one_shot(
                    authorization="OPEN_AHC_TV_TEST", raw_sources=[],
                    polarization_2000=self.phase / "p2000.csv", polarization_2300=self.phase / "p2300.csv", phase=self.phase,
                )
        self.assertFalse(ledger_path(self.phase).exists())

    def test_exclusive_first_start_second_start_and_alternate_directory_block(self) -> None:
        self._start()
        self.assertEqual(read_ledger(self.phase)["status"], STATUS_STARTED)
        with self.assertRaisesRegex(OneShotLedgerError, "AHC_TV_ONE_SHOT_ALREADY_CONSUMED"):
            self._start()
        arbitrary_second_runtime = self.phase / "runtime" / "ANOTHER_RUNTIME_DIRECTORY"
        arbitrary_second_runtime.mkdir(parents=True)
        with self.assertRaisesRegex(OneShotLedgerError, "AHC_TV_ONE_SHOT_ALREADY_CONSUMED"):
            self._start()

    def test_forward_transitions_and_backward_transition_rejected(self) -> None:
        self._start()
        transition_execution_ledger(phase=self.phase, expected_status=STATUS_STARTED, next_status=STATUS_STAGE_P)
        transition_execution_ledger(phase=self.phase, expected_status=STATUS_STAGE_P, next_status=STATUS_TARGET_STARTED)
        transition_execution_ledger(phase=self.phase, expected_status=STATUS_TARGET_STARTED, next_status=STATUS_TARGETS)
        transition_execution_ledger(phase=self.phase, expected_status=STATUS_TARGETS, next_status=STATUS_DECISION)
        final = transition_execution_ledger(phase=self.phase, expected_status=STATUS_DECISION, next_status=STATUS_COMPLETED)
        self.assertEqual(final["status"], STATUS_COMPLETED)
        with self.assertRaisesRegex(OneShotLedgerError, "INVALID_STATE_TRANSITION"):
            transition_execution_ledger(phase=self.phase, expected_status=STATUS_COMPLETED, next_status=STATUS_STAGE_P)

    def test_ledger_overwrite_is_rejected(self) -> None:
        self._start()
        with self.assertRaises(FileExistsError):
            write_exclusive_json(ledger_path(self.phase), {"status": "forbidden"})
        self.assertEqual(read_ledger(self.phase)["status"], STATUS_STARTED)

    def test_target_read_crash_is_permanent_but_offline_frozen_results_are_readable(self) -> None:
        self._start()
        transition_execution_ledger(phase=self.phase, expected_status=STATUS_STARTED, next_status=STATUS_STAGE_P)
        transition_execution_ledger(phase=self.phase, expected_status=STATUS_STAGE_P, next_status=STATUS_TARGET_STARTED)
        run_dir = canonical_run_dir(self.phase)
        run_dir.mkdir(parents=True)
        result_path = run_dir / "FROZEN_TEST_TARGETS_AND_ERRORS.csv"
        write_exclusive_text(result_path, "checkpoint,target_Y\n2000,0.97\n2300,0.96\n")
        write_exclusive_sha256_sidecar(result_path)
        abort_execution_ledger(
            phase=self.phase, stage="DECISION_FREEZE", exception=RuntimeError("synthetic decision write crash"), target_unblinded=True,
        )
        ledger = read_ledger(self.phase)
        self.assertEqual(ledger["status"], "ABORTED_AFTER_TEST_EXECUTION_STARTED")
        self.assertTrue(ledger["target_unblinded"])
        with self.assertRaisesRegex(OneShotLedgerError, "AHC_TV_ONE_SHOT_ALREADY_CONSUMED"):
            self._start()
        self.assertEqual(load_frozen_results_for_offline_postprocessing(phase=self.phase), [
            {"checkpoint": "2000", "target_Y": "0.97"}, {"checkpoint": "2300", "target_Y": "0.96"},
        ])

    def test_runtime_path_cannot_preexist_before_first_start(self) -> None:
        canonical_run_dir(self.phase).mkdir(parents=True)
        with self.assertRaisesRegex(OneShotLedgerError, "RUNTIME_PATH_ALREADY_EXISTS"):
            self._start()
        self.assertFalse(ledger_path(self.phase).exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from ahc_tv_alignment import (  # noqa: E402
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    TargetUnblindingBlocked,
    read_characterization_metadata_only,
    read_targets_after_prediction_freeze,
)


class MetadataTargetIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.polarization = self.root / "mock_polarization.csv"
        columns = {"DateTime": ["2030-01-01 00:00:00"], "j.S.act [A/cm2]": [1.0]}
        columns.update({f"U{i} [V]": [9000.0 + i] for i in range(1, 12)})
        pd.DataFrame(columns).to_csv(self.polarization, sep=";", index=False, encoding="utf-8-sig")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def freeze_legal_prediction(self) -> Path:
        prediction = self.root / "predictions.csv"
        pd.DataFrame({
            "checkpoint": [2000, 2300],
            "joint_support_count": [2, 3],
            "selected_interval_1": ["A", "D"],
            "selected_interval_2": ["B", "E"],
            "selected_interval_3": ["C", "F"],
        }).to_csv(prediction, index=False)
        digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
        prediction.with_name(prediction.name + ".sha256").write_text(digest + "  predictions.csv\n", encoding="utf-8")
        return prediction

    def test_metadata_reader_requests_only_datetime(self) -> None:
        calls = []

        def strict_reader(path, **kwargs):
            calls.append(kwargs.get("usecols"))
            self.assertEqual(kwargs.get("usecols"), list(METADATA_COLUMNS))
            return pd.read_csv(path, **kwargs)

        metadata = read_characterization_metadata_only(self.polarization, read_csv=strict_reader)
        self.assertEqual(list(metadata.columns), ["DateTime"])
        self.assertEqual(calls, [["DateTime"]])
        self.assertNotIn("j.S.act [A/cm2]", metadata.columns)

    def test_target_reader_blocks_before_file_read_if_prediction_missing(self) -> None:
        called = False

        def forbidden_reader(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("polarization target file was opened")

        with self.assertRaises(TargetUnblindingBlocked):
            read_targets_after_prediction_freeze(self.polarization, self.root / "missing.csv", read_csv=forbidden_reader)
        self.assertFalse(called)

    def test_target_reader_blocks_before_file_read_if_prediction_hash_mismatches(self) -> None:
        prediction = self.root / "predictions.csv"
        pd.DataFrame({"checkpoint": [2000, 2300]}).to_csv(prediction, index=False)
        prediction.with_name(prediction.name + ".sha256").write_text("0" * 64 + "  predictions.csv\n", encoding="utf-8")
        called = False

        def forbidden_reader(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("polarization target file was opened")

        with self.assertRaises(TargetUnblindingBlocked):
            read_targets_after_prediction_freeze(self.polarization, prediction, read_csv=forbidden_reader)
        self.assertFalse(called)
    def test_target_reader_requests_only_frozen_active_cell_target_columns_after_valid_freeze(self) -> None:
        prediction = self.freeze_legal_prediction()
        calls = []

        def spy_reader(path, **kwargs):
            calls.append(kwargs.get("usecols"))
            return pd.read_csv(path, **kwargs)

        target = read_targets_after_prediction_freeze(self.polarization, prediction, read_csv=spy_reader)
        self.assertEqual(calls, [list(TARGET_COLUMNS)])
        self.assertEqual(tuple(target.columns), TARGET_COLUMNS)
        self.assertNotIn("U6 [V]", target.columns)

    def test_target_reader_blocks_before_file_read_if_alignment_evidence_is_missing(self) -> None:
        prediction = self.root / "predictions.csv"
        pd.DataFrame({"checkpoint": [2000, 2300]}).to_csv(prediction, index=False)
        digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
        prediction.with_name(prediction.name + ".sha256").write_text(digest + "  predictions.csv\n", encoding="utf-8")
        called = False

        def forbidden_reader(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("polarization target file was opened")

        with self.assertRaises(TargetUnblindingBlocked):
            read_targets_after_prediction_freeze(self.polarization, prediction, read_csv=forbidden_reader)
        self.assertFalse(called)

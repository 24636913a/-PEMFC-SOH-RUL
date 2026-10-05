from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from ahc_tv_alignment import (  # noqa: E402
    METADATA_COLUMNS,
    REQUIRED_TEST_AUTHORIZATION,
    TARGET_COLUMNS,
    TargetUnblindingBlocked,
    read_characterization_metadata_only,
    read_targets_after_prediction_freeze,
)


class MetadataTargetIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.schema = yaml.safe_load((PHASE / "schemas" / "FROZEN_TEST_PREDICTIONS_SCHEMA.yaml").read_text(encoding="utf-8"))
        self.models = yaml.safe_load((PHASE / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml").read_text(encoding="utf-8"))
        self.polarization = self.root / "synthetic_mock_polarization.csv"
        columns = {"DateTime": ["2030-01-01 00:00:00"], "j.S.act [A/cm2]": [1.0]}
        columns.update({f"U{i} [V]": [9000.0 + i] for i in range(1, 12)})
        pd.DataFrame(columns).to_csv(self.polarization, sep=";", index=False, encoding="utf-8-sig")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def full_prediction(self) -> pd.DataFrame:
        rows = []
        p1s = self.models["P1S_affine"]
        age = self.models["nominal_age_affine"]
        for checkpoint, r_values, support in ((2000, (0.989, 0.990, 0.991), (True, True, False)), (2300, (0.984, 0.985, 0.986), (True, True, True))):
            r_median = float(np.median(r_values))
            rows.append({
                "checkpoint": checkpoint,
                "actual_characterization_start": "2030-01-01 15:00:00",
                "raw_cutoff_datetime": "2030-01-01 14:00:00",
                "raw_cutoff_ophrs2_h": float(checkpoint) - 0.25,
                "selected_interval_1": f"S{checkpoint}_1",
                "selected_interval_2": f"S{checkpoint}_2",
                "selected_interval_3": f"S{checkpoint}_3",
                "measurement_end_datetime_1": "2030-01-01 13:00:00",
                "measurement_end_datetime_2": "2030-01-01 13:15:00",
                "measurement_end_datetime_3": "2030-01-01 13:30:00",
                "joint_support_1": "true" if support[0] else "false",
                "joint_support_2": "true" if support[1] else "false",
                "joint_support_3": "true" if support[2] else "false",
                "joint_support_count": sum(support),
                "R1": r_values[0], "R2": r_values[1], "R3": r_values[2],
                "P1S_R_median": r_median,
                "P1S_affine_prediction": float(p1s["slope"]) * r_median + float(p1s["intercept"]),
                "nominal_age_prediction": float(age["intercept_alpha"]) + float(age["slope_beta"]) * checkpoint,
                "raw_P1S_prediction": r_median,
            })
        return pd.DataFrame(rows).loc[:, self.schema["required_columns"]]

    def freeze(self, table: pd.DataFrame) -> Path:
        prediction = self.root / "FROZEN_TEST_PREDICTIONS_BEFORE_TARGET_UNBLINDING.csv"
        table.to_csv(prediction, index=False)
        digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
        prediction.with_name(prediction.name + ".sha256").write_text(digest + "  " + prediction.name + "\n", encoding="utf-8")
        provenance = {
            "prediction_sha256": digest,
            "checkpoints": [
                {
                    "checkpoint": int(row.checkpoint),
                    "selected_interval_ids": [str(row.selected_interval_1), str(row.selected_interval_2), str(row.selected_interval_3)],
                }
                for row in table.itertuples(index=False)
            ],
        }
        provenance_path = self.root / "FROZEN_TEST_PREDICTION_PROVENANCE.json"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        provenance_digest = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
        provenance_path.with_name(provenance_path.name + ".sha256").write_text(provenance_digest + "  " + provenance_path.name + "\n", encoding="utf-8")
        completion = {
            "status": "STAGE_P_FROZEN_COMPLETE", "authorization_marker_present": True,
            "both_checkpoints_built": True, "both_alignments_legal": True,
            "prediction_sha256_frozen": True, "provenance_sha256_frozen": True,
            "strict_prediction_verification": "PASS", "target_columns_read_so_far": False,
            "prediction_sha256": digest, "provenance_sha256": provenance_digest,
        }
        completion_path = self.root / "STAGE_P_COMPLETION_MARKER.json"
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        completion_digest = hashlib.sha256(completion_path.read_bytes()).hexdigest()
        completion_path.with_name(completion_path.name + ".sha256").write_text(completion_digest + "  " + completion_path.name + "\n", encoding="utf-8")
        return prediction

    def assert_reader_rejected_before_open(self, table: pd.DataFrame, *, authorization_token: str = REQUIRED_TEST_AUTHORIZATION) -> None:
        prediction = self.freeze(table)
        called = False

        def forbidden_reader(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("synthetic target file was opened")

        with self.assertRaises(TargetUnblindingBlocked):
            read_targets_after_prediction_freeze(self.polarization, prediction, authorization_token=authorization_token, read_csv=forbidden_reader)
        self.assertFalse(called)

    def test_metadata_reader_requires_exact_authorization_and_requests_only_datetime(self) -> None:
        called = False

        def strict_reader(path, **kwargs):
            nonlocal called
            called = True
            self.assertEqual(kwargs.get("usecols"), list(METADATA_COLUMNS))
            return pd.read_csv(path, **kwargs)

        with self.assertRaises(TypeError):
            read_characterization_metadata_only(self.polarization, read_csv=strict_reader)
        self.assertFalse(called)
        with self.assertRaises(TargetUnblindingBlocked):
            read_characterization_metadata_only(self.polarization, authorization_token="OPEN_AHC_TV_TEST_WRONG", read_csv=strict_reader)
        self.assertFalse(called)
        metadata = read_characterization_metadata_only(self.polarization, authorization_token=REQUIRED_TEST_AUTHORIZATION, read_csv=strict_reader)
        self.assertEqual(list(metadata.columns), ["DateTime"])
        self.assertTrue(called)

    def test_legacy_five_column_artifact_is_rejected(self) -> None:
        legacy = self.full_prediction().loc[:, ["checkpoint", "joint_support_count", "selected_interval_1", "selected_interval_2", "selected_interval_3"]]
        self.assert_reader_rejected_before_open(legacy)

    def test_missing_required_prediction_columns_are_rejected(self) -> None:
        for name in ("P1S_affine_prediction", "nominal_age_prediction", "R1", "R2", "R3"):
            with self.subTest(name=name):
                self.assert_reader_rejected_before_open(self.full_prediction().drop(columns=[name]))

    def test_nonfinite_p1s_prediction_values_are_rejected(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                table = self.full_prediction()
                table.loc[0, "P1S_affine_prediction"] = value
                self.assert_reader_rejected_before_open(table)

    def test_semantically_tampered_predictions_with_fresh_hash_are_rejected(self) -> None:
        mutations = {
            "P1S_affine_prediction": lambda table: table.__setitem__("P1S_affine_prediction", table["P1S_affine_prediction"] + 1e-4),
            "nominal_age_prediction": lambda table: table.__setitem__("nominal_age_prediction", table["nominal_age_prediction"] + 1e-4),
            "P1S_R_median": lambda table: table.__setitem__("P1S_R_median", table["P1S_R_median"] + 1e-4),
            "raw_P1S_prediction": lambda table: table.__setitem__("raw_P1S_prediction", table["raw_P1S_prediction"] + 1e-4),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                table = self.full_prediction()
                mutate(table)
                self.assert_reader_rejected_before_open(table)

    def test_support_interval_and_temporal_evidence_tampering_is_rejected(self) -> None:
        cases = []
        table = self.full_prediction(); table.loc[0, "joint_support_count"] = 3; cases.append(table)
        table = self.full_prediction(); table.loc[0, "selected_interval_3"] = table.loc[0, "selected_interval_1"]; cases.append(table)
        table = self.full_prediction(); table.loc[0, "measurement_end_datetime_1"] = table.loc[0, "actual_characterization_start"]; cases.append(table)
        table = self.full_prediction(); table["joint_support_1"] = "not_a_boolean"; cases.append(table)
        for index, table in enumerate(cases):
            with self.subTest(case=index):
                self.assert_reader_rejected_before_open(table)

    def test_target_like_extra_column_is_rejected(self) -> None:
        table = self.full_prediction()
        table["target"] = 0.0
        self.assert_reader_rejected_before_open(table)

    def test_missing_and_wrong_authorization_are_rejected_before_target_open(self) -> None:
        prediction = self.freeze(self.full_prediction())
        called = False

        def forbidden_reader(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("synthetic target file was opened")

        with self.assertRaises(TypeError):
            read_targets_after_prediction_freeze(self.polarization, prediction, read_csv=forbidden_reader)
        self.assertFalse(called)
        with self.assertRaises(TargetUnblindingBlocked):
            read_targets_after_prediction_freeze(self.polarization, prediction, authorization_token="OPEN_AHC_TV_TEST_WRONG", read_csv=forbidden_reader)
        self.assertFalse(called)

    def test_exact_full_schema_with_frozen_formulae_and_authorization_allows_mock_target_reader(self) -> None:
        prediction = self.freeze(self.full_prediction())
        calls = []

        def spy_reader(path, **kwargs):
            calls.append(kwargs.get("usecols"))
            return pd.read_csv(path, **kwargs)

        target = read_targets_after_prediction_freeze(self.polarization, prediction, authorization_token=REQUIRED_TEST_AUTHORIZATION, read_csv=spy_reader)
        self.assertEqual(calls, [list(TARGET_COLUMNS)])
        self.assertEqual(tuple(target.columns), TARGET_COLUMNS)
        self.assertNotIn("U6 [V]", target.columns)

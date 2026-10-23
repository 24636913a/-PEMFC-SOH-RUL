"""Production guard tests for Phase HA; no Test data or model paths are used."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))

from ha_common import (AuditFailure, DEVELOPMENT_END_H, SAFE_N1_COLUMNS, cache_universe,
                       load_npz_checked, load_protocol, read_development_prefix, read_manifest)
from run_health_audit import pre_statistics_tests


class PhaseHAProductionGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.development, cls.firewall = read_development_prefix()
        cls.manifest, cls.local_artifact_manifest = read_manifest()
        cls.simple, cls.emd, cls.joined = cache_universe(cls.manifest)

    def test_exact_prefix_boundary_and_no_targets(self):
        self.assertEqual(self.firewall["raw_prefix_rows_read"], 3007)
        self.assertEqual(self.firewall["development_rows_after_filter"], 3005)
        self.assertEqual(self.firewall["boundary_rows_filtered"], 2)
        self.assertEqual(self.firewall["post_boundary_rows_entered_pipeline"], 0)
        self.assertTrue((self.development.bin_end_ophrs2_h < DEVELOPMENT_END_H).all())
        self.assertEqual(self.firewall["loaded_columns"], SAFE_N1_COLUMNS)
        self.assertTrue(self.firewall["future_target_loaded"] is False)

    def test_production_pre_statistics_guards(self):
        audit = pre_statistics_tests(self.development, self.simple, self.emd, self.joined)
        self.assertTrue(audit["future_perturbation_invariance"])
        self.assertTrue(audit["prefix_invariance"])
        self.assertTrue(audit["same_segment_history"])
        self.assertEqual(audit["lag_revision_index"], "later_features[47-k, 0] for k=1,2,4,8")

    def test_duplicate_origin_is_rejected_by_production_cache_universe(self):
        duplicate = pd.concat([self.manifest, self.manifest.iloc[[0]]], ignore_index=True)
        with self.assertRaises(AuditFailure):
            cache_universe(duplicate)

    def test_invalid_cache_parameter_is_rejected_by_production_loader(self):
        row = self.emd.iloc[0].copy()
        with self.assertRaises(AuditFailure):
            load_npz_checked(row, "deliberately-wrong-parameter-hash")

    def test_frozen_protocol_hash_is_required(self):
        self.assertIn("frozen_source_hashes", self.protocol)
        self.assertEqual(self.protocol["protocol_status"], "PRE-REGISTERED_BEFORE_PHASE_HA_ENDPOINT_ANALYSIS")


if __name__ == "__main__":
    unittest.main(verbosity=2)


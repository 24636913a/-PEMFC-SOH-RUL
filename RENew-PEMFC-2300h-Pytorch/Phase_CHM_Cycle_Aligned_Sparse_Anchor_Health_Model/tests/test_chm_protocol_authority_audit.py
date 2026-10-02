from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chm_protocol_authority_audit.py"
SPEC = importlib.util.spec_from_file_location("chm_authority", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CHMProtocolAuthorityAuditTests(unittest.TestCase):
    def test_01_v2_and_errata_sidecars_match(self) -> None:
        self.assertEqual(len(MODULE.assert_sha_sidecar(MODULE.FINAL_V2, MODULE.FINAL_V2_SHA)), 64)
        self.assertEqual(len(MODULE.assert_sha_sidecar(MODULE.ERRATA, MODULE.ERRATA_SHA)), 64)

    def test_02_authority_has_one_normative_protocol_and_historical_v1(self) -> None:
        authority = MODULE.load_yaml(MODULE.AUTHORITY)
        MODULE.validate_authority(authority)
        MODULE.validate_historical_nonexecutability(authority)
        self.assertEqual(authority["normative_executable"][0], "configs/PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml")
        self.assertIn("configs/PREREGISTERED_CHM_PROTOCOL_FINAL.yaml", authority["historical_non_executable"])

    def test_03_protocol_has_three_profiles_and_five_primary_gates(self) -> None:
        protocol = MODULE.load_yaml(MODULE.FINAL_V2)
        MODULE.validate_protocol(protocol)
        self.assertEqual(protocol["loss_profiles"]["profile_count"], 3)
        self.assertEqual(protocol["primary_development_qualification"]["gate_count"], 5)
        self.assertFalse(protocol["primary_development_qualification"]["S1_is_qualification_gate"])

    def test_04_production_validator_rejects_a_fourth_profile(self) -> None:
        protocol = copy.deepcopy(MODULE.load_yaml(MODULE.FINAL_V2))
        protocol["loss_profiles"]["profiles"]["D_ILLEGAL"] = {"lambda_anchor": 1, "lambda_obs": 1, "lambda_smooth": 1}
        with self.assertRaises(MODULE.ProtocolAuthorityFailure):
            MODULE.validate_protocol(protocol)

    def test_05_production_validator_rejects_s1_as_primary_gate(self) -> None:
        protocol = copy.deepcopy(MODULE.load_yaml(MODULE.FINAL_V2))
        protocol["primary_development_qualification"]["S1_is_qualification_gate"] = True
        with self.assertRaises(MODULE.ProtocolAuthorityFailure):
            MODULE.validate_protocol(protocol)

    def test_06_frozen_hou_target_mapping_and_hashes_pass(self) -> None:
        protocol = MODULE.load_yaml(MODULE.FINAL_V2)
        MODULE.validate_frozen_tables(protocol)
        manifest = MODULE.verify_frozen_input_hashes_v2()
        self.assertEqual(len(manifest["records"]), 12)
        self.assertIn("sparse_anchor_targets", manifest["records"])

    def test_07_sparse_targets_and_mapping_have_only_six_development_anchors(self) -> None:
        targets = pd.read_csv(MODULE.INPUTS / "CHM_SPARSE_ANCHOR_TARGETS.csv")
        mapping = pd.read_csv(MODULE.INPUTS / "CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv")
        self.assertEqual(targets["checkpoint"].astype(int).tolist(), [300, 600, 850, 1150, 1500, 1750])
        self.assertEqual(len(mapping), 18)
        self.assertFalse(targets["source_polarization_file"].str.contains("2000|2300|after", case=False).any())

    def test_08_authority_audit_exercises_all_required_checks(self) -> None:
        result = MODULE.run_authority_audit()
        self.assertEqual(result["protocol_authority_audit"], "PASS")
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["Test_accessed"])
        saved = json.loads(MODULE.AUDIT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(saved["remaining_protocol_ambiguity"], "NONE")


if __name__ == "__main__":
    unittest.main()

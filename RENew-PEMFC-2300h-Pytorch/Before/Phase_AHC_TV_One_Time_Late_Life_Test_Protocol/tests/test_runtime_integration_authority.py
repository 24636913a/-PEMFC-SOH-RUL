from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
from ahc_tv_alignment import TargetUnblindingBlocked, sha256_file, verify_frozen_runtime_authority  # noqa: E402
from ahc_tv_raw_alignment_core import RCR_TEMPLATE_PATH  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RuntimeIntegrationAuthorityTests(unittest.TestCase):
    def test_runtime_integration_has_exact_execution_precedence(self) -> None:
        authority = yaml.safe_load((PHASE / "configs" / "AHC_TV_PROTOCOL_AUTHORITY.yaml").read_text(encoding="utf-8"))
        self.assertEqual(
            authority["precedence"],
            [
                "PREREGISTERED_AHC_TV_PROTOCOL.yaml",
                "AHC_TV_EXECUTION_CONTRACT_V1.yaml",
                "AHC_TV_EXECUTION_GATE_PATCH_V1.yaml",
                "AHC_TV_RUNTIME_INTEGRATION_V1.yaml",
                "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml",
                "AHC_TV_FROZEN_INPUT_AUTHORITY.yaml",
            ],
        )
        self.assertEqual(authority["execution_authority"]["runtime_integration"]["path"], "configs/AHC_TV_RUNTIME_INTEGRATION_V1.yaml")

    def test_runtime_integration_contract_and_hash_sidecar_are_frozen(self) -> None:
        contract = PHASE / "configs" / "AHC_TV_RUNTIME_INTEGRATION_V1.yaml"
        config = yaml.safe_load(contract.read_text(encoding="utf-8"))
        self.assertEqual(config["sole_formal_runtime_entry"], "scripts/run_ahc_tv_one_shot.py")
        self.assertEqual(config["runtime_sequence"], ["AUTHORIZATION", "AUTHORITY_VERIFY", "STAGE_P", "PREDICTION_FREEZE", "HARDENED_VERIFY", "STAGE_T", "DECISION"])
        self.assertTrue(config["authorization"]["external_command_argument_required"])
        self.assertFalse(config["stage_P"]["target_columns_permitted"])
        self.assertEqual((PHASE / "configs" / "AHC_TV_RUNTIME_INTEGRATION_V1.sha256").read_text(encoding="utf-8").split()[0], sha256(contract))

    def test_every_authority_hash_and_source_registry_constraint_is_current(self) -> None:
        authority = yaml.safe_load((PHASE / "configs" / "AHC_TV_FROZEN_INPUT_AUTHORITY.yaml").read_text(encoding="utf-8"))
        for record in authority["frozen_hashes"]:
            with self.subTest(record=record["name"]):
                self.assertEqual(sha256(REPO / record["path"]), record["sha256"])
        registry = yaml.safe_load((PHASE / "configs" / "AHC_TV_RUNTIME_SOURCE_REGISTRY.yaml").read_text(encoding="utf-8"))
        self.assertTrue(registry["binding_uses_checkpoint_filename_registry_only"])
        self.assertFalse(registry["binding_uses_voltage_or_target_values"])
        self.assertEqual(registry["binding_failure_status"], "TEST_SOURCE_BINDING_AMBIGUOUS")
        self.assertIn("after_Air_Air_polarization", registry["forbidden_source_classes"])
        self.assertIn("Test_EIS", registry["forbidden_source_classes"])

    def test_rcr_template_hash_mismatch_fails_closed_before_any_test_reader(self) -> None:
        authority = yaml.safe_load((PHASE / "configs" / "AHC_TV_FROZEN_INPUT_AUTHORITY.yaml").read_text(encoding="utf-8"))
        template = [item for item in authority["frozen_hashes"] if item["name"] == "frozen_RCR_cycle_template"]
        self.assertEqual(len(template), 1)
        self.assertEqual(template[0]["sha256"], "267a3ee5a2d297b4368f4dd262771a19109e8015b2617de45ab828d83d696200")
        verify_frozen_runtime_authority(PHASE)
        real_sha256 = sha256_file

        def tampered_hash(path: Path) -> str:
            if Path(path).resolve() == RCR_TEMPLATE_PATH.resolve():
                return "0" * 64
            return real_sha256(path)

        with mock.patch("ahc_tv_alignment.sha256_file", side_effect=tampered_hash), mock.patch("pandas.read_csv") as test_reader:
            with self.assertRaises(TargetUnblindingBlocked):
                verify_frozen_runtime_authority(PHASE)
            test_reader.assert_not_called()

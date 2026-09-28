"""Source-fingerprint and synthetic adapter tests; no Development data is read."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
import legacy_protocol as legacy  # noqa: E402
from run_xlstmtime import canonical_hashes, load_config, main as xlstm_main, resource_gate  # noqa: E402


class XLSTMTimeFrozenContractTest(unittest.TestCase):
    def test_frozen_config_and_canonical_author_source(self) -> None:
        cfg = load_config()
        fingerprints = canonical_hashes(cfg)
        self.assertEqual(len(fingerprints), 3)
        self.assertTrue(fingerprints["pass"].all())

    def test_production_resource_gate_is_synthetic_and_fail_closed(self) -> None:
        cfg = load_config()
        _legacy_cfg, helpers, _audit = legacy.load_legacy_contract()
        result = resource_gate(cfg, helpers)
        self.assertIn(result["status"], {"PASS", "RESOURCE_INFEASIBLE"})
        self.assertEqual(result["synthetic_input_shape"], [4, 48, 14])
        self.assertFalse(result["test_accessed"])
        self.assertEqual(result["development_rows_read"], 0)
        if result["status"] == "RESOURCE_INFEASIBLE":
            self.assertIn(result["failure_stage"], {"author_source_import_or_construction", "synthetic_forward_backward"})

    def test_terminal_final_decision_blocks_training_restart(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "terminally closed"):
            xlstm_main()


if __name__ == "__main__":
    unittest.main(verbosity=2)

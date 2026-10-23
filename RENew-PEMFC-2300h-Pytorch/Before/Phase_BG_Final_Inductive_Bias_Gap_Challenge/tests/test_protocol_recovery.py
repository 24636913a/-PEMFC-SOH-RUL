"""Non-training Phase BG contract tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
import legacy_protocol as legacy  # noqa: E402


class ProtocolRecoveryTest(unittest.TestCase):
    def test_readonly_recovery_and_exact_window_identity(self) -> None:
        _cfg, _helpers, _frame, _folds, audit, record = legacy.recover_windows_and_controls()
        self.assertTrue(record["window_identity_pass"])
        self.assertEqual(record["data_audit"]["pandas_read_csv_nrows"], 3007)
        self.assertEqual(record["data_audit"]["development_rows_retained"], 3005)
        self.assertEqual(len(audit), 9)
        self.assertTrue(audit[["origin_row_match", "target_row_match", "anchor_match", "truth_match", "target_end_match"]].all().all())
        self.assertFalse(record["test_accessed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

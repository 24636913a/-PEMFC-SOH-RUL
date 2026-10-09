from __future__ import annotations
import sys, unittest
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PHASE / "scripts"))
from t1r_common import load_protocol
from t1r_decision import ADVANCE, INSUFFICIENT, PENDING, multiseed_decision, seed42_decision


def m(rmse, fold3=30., p99=10., target=10.):
    return {"rmse": rmse, "fold3_rmse": fold3, "p99": p99, "target_equal_rmse": target}


class DecisionContractTests(unittest.TestCase):
    def setUp(self): self.cfg, _ = load_protocol()
    def test_seed42_is_protocol_derived_and_pending_only(self):
        result = seed42_decision(m(26), m(20), m(18), cfg=self.cfg, identity_audited=True)
        self.assertEqual(result["status"], PENDING)
    def test_seed42_rejects_manual_unaudited_metrics(self):
        self.assertNotEqual(seed42_decision(m(26), m(20), m(18), cfg=self.cfg, identity_audited=False)["status"], PENDING)
    def test_multiseed_requires_identity_and_all_seeds(self):
        first = {"status": PENDING}; data = {s: {**m(18), "b_rmse": 20} for s in (42, 2024, 3407)}
        self.assertEqual(multiseed_decision(first, {"pass": True}, data, cfg=self.cfg)["status"], ADVANCE)
        self.assertEqual(multiseed_decision(first, {"pass": False}, data, cfg=self.cfg)["status"], "TFT_T1_TECHNICAL_FAILURE")
        self.assertEqual(multiseed_decision(first, {"pass": True}, {42: data[42]}, cfg=self.cfg)["status"], "TFT_T1_TECHNICAL_FAILURE")
    def test_seed42_extended_branch_is_preserved(self):
        self.assertEqual(seed42_decision(m(26), m(18), m(19), cfg=self.cfg, identity_audited=True)["status"], "TFT_EXTENDED_BUDGET_CONFIG_ADVANCE")


if __name__ == "__main__": unittest.main()

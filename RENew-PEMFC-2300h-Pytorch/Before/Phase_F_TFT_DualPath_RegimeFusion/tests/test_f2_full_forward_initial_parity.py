import unittest

from ._forward_parity import initial_f1_f2_full_forward_audit


class TestF2FullForwardInitialParity(unittest.TestCase):
    def test_half_gate_matches_f1_complete_enabled_forward(self):
        evidence = initial_f1_f2_full_forward_audit()
        self.assertTrue(evidence["pass"])
        self.assertEqual(len(evidence["cases"]), 2)
        for case in evidence["cases"]:
            with self.subTest(shape=case["input_shape"]):
                self.assertEqual(case["F1_execution"]["history_branch"], 1)
                self.assertEqual(case["F2_execution"]["history_branch"], 1)
                self.assertEqual(case["F2_execution"]["regime_gate"], 1)
                self.assertTrue(all(item["pass"] for item in case["outputs"].values()))

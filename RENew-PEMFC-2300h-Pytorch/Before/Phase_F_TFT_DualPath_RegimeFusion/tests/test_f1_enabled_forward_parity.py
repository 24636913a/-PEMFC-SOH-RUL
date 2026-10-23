import unittest

from ._forward_parity import enabled_neutral_fusion_audit


class TestF1EnabledForwardParity(unittest.TestCase):
    def test_enabled_neutral_fusion_matches_all_native_outputs(self):
        evidence = enabled_neutral_fusion_audit()
        self.assertTrue(evidence["pass"])
        self.assertEqual(len(evidence["cases"]), 2)
        for case in evidence["cases"]:
            with self.subTest(shape=case["input_shape"]):
                self.assertEqual(case["execution"]["history_branch"], 1)
                self.assertTrue(all(item["pass"] for item in case["outputs"].values()))

"""Automated causal aggregation and parent-label identity tests."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd

SCRIPT=Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_TR2_Fair_Resolution_Correction\scripts\run_fair_resolution_correction.py")
spec=importlib.util.spec_from_file_location("tr2",SCRIPT); tr2=importlib.util.module_from_spec(spec); spec.loader.exec_module(tr2)

class CorrectionTests(unittest.TestCase):
    def test_prediction_equal_aggregation_ignores_future_counts(self):
        prediction=[0.2,0.5,0.8]
        first=tr2.equal_prediction_aggregate(prediction)
        future_counts_a=[600,600,600]
        future_counts_b=[1,9999,2]
        self.assertEqual(future_counts_a!=future_counts_b,True)
        self.assertEqual(first,tr2.equal_prediction_aggregate(prediction))

    def test_parent_mixed_identity_cases(self):
        def children(zero, valid=(600,600,600), flags=(True,True,True)):
            return pd.DataFrame({"zero_current_count":zero,"valid_current_count":valid,"valid_bin":flags})
        self.assertEqual(tr2.parent_mixed_from_children(children([0,0,0])),0)
        self.assertEqual(tr2.parent_mixed_from_children(children([600,600,600])),0)
        self.assertEqual(tr2.parent_mixed_from_children(children([0,600,0])),1)
        self.assertEqual(tr2.parent_mixed_from_children(children([1,0,0])),1)
        self.assertIsNone(tr2.parent_mixed_from_children(children([0,0,0],flags=(True,False,True))))

    def test_incomplete_children_are_rejected(self):
        complete=pd.DataFrame({"valid_bin":[True,True,True]})
        incomplete=pd.DataFrame({"valid_bin":[True,False,True]})
        self.assertTrue(tr2.complete_children(complete))
        self.assertFalse(tr2.complete_children(incomplete))

if __name__=="__main__":
    unittest.main()

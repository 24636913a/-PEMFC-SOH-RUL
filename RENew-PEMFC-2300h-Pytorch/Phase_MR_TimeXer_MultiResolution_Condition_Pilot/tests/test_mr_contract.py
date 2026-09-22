import importlib.util
from pathlib import Path
import unittest
import numpy as np

P=Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_MR_TimeXer_MultiResolution_Condition_Pilot\scripts\run_mr_condition_pilot.py")
s=importlib.util.spec_from_file_location("mr",P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class MRTests(unittest.TestCase):
    def test_mixed_definition(self):
        self.assertEqual(m.mixed(np.array([[0.,1.,.2]])).tolist(),[[0,0,1]])
    def test_q_bounds_are_representable(self):
        q=np.clip(np.array([[-1.,.2,2.]]),0,1)
        self.assertTrue(np.all((q>=0)&(q<=1)))
if __name__=="__main__":
    unittest.main()

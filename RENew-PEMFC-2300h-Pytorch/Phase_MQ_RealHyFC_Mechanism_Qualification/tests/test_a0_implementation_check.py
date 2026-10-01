from __future__ import annotations
import sys
import unittest
from pathlib import Path
import pandas as pd

PHASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PHASE/"scripts"))
from run_a0_implementation_check import lag_rows, classify

class TestA0ImplementationCheck(unittest.TestCase):
    def test_lag_pair_formula_boundary(self):
        thirteen=pd.DataFrame({"a0_token":["x"]*13})
        twelve=pd.DataFrame({"a0_token":["x"]*12})
        self.assertEqual(next(x for x in lag_rows([thirteen]) if x["period_steps"]==12)["lag_pairs"],1)
        self.assertEqual(next(x for x in lag_rows([twelve]) if x["period_steps"]==12)["lag_pairs"],0)

    def test_zero_to_positive_requires_attempt_three(self):
        cur=[{"lag_pairs":0}]
        compliant=[{"lag_pairs":1,"exact_match_rate":.8}]
        self.assertEqual(classify(cur,compliant)[0],"ZERO_LAG_PAIRS_NOT_ROBUST")

if __name__=="__main__": unittest.main()

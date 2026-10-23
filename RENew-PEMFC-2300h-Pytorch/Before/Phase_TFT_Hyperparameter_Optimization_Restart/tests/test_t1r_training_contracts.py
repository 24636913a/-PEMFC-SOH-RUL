from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np
PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/'scripts'))
from t1r_training import make_loader

class TrainingContractTests(unittest.TestCase):
    def test_loader_is_deterministic_for_new_fit_but_advances_between_epochs(self):
        x=np.arange(80,dtype=np.float32).reshape(40,2); y=np.arange(40,dtype=np.float32).reshape(40,1)
        a=make_loader(x,y,4,42); b=make_loader(x,y,4,42)
        first_a=[int(v) for xb,_ in a for v in xb[:,0]]; first_b=[int(v) for xb,_ in b for v in xb[:,0]]
        second_a=[int(v) for xb,_ in a for v in xb[:,0]]
        self.assertEqual(first_a,first_b); self.assertNotEqual(first_a,second_a)

if __name__=='__main__': unittest.main()

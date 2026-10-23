import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
from preprocess_frozen_inputs import scaler,preprocess
class ScalerTests(unittest.TestCase):
    def test_historical_scaler_exact(self):
        s=scaler();old=read(B/'outputs/audits/SCALER.json');self.assertEqual(s,old)
        d=data('TRAIN');raw=d['x_raw'].astype(float).reshape(-1,14)
        self.assertTrue(np.array_equal(raw.mean(0),s['mean']));self.assertTrue(np.array_equal(raw.std(0),s['scale']))
    def test_mask_and_no_mutation(self):
        x=data('VALIDATION')['x_raw'];saved=x.copy();z=preprocess(x)
        self.assertTrue(np.array_equal(x,saved));self.assertTrue(np.all(z[:,:,[1,2,5,12]]==0))
    def test_ewma_causality(self):
        x=np.ones((2,48,14),np.float32);a=preprocess(x);x[:,24:,0]+=10;b=preprocess(x)
        self.assertTrue(np.array_equal(a[:,:24],b[:,:24]));self.assertFalse(np.array_equal(a[:,24:],b[:,24:]))

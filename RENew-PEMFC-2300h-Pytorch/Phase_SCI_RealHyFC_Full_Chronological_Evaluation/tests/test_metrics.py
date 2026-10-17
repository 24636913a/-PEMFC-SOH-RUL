import unittest,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
from calculate_partition_metrics import metrics
class MetricTests(unittest.TestCase):
    def test_analytical(self):
        f=pd.DataFrame(dict(origin_row=[0,0,1],target_row=[2,2,3],horizon_step=[1,24,24],y_true_delta_V=[0.,1.,2.],y_pred_delta_V=[.001,1.003,2.002]))
        m=metrics(f)
        self.assertAlmostEqual(m['RMSE_mV'],math.sqrt(14/3),10)
        self.assertAlmostEqual(m['MAE_mV'],2.,10)
        self.assertAlmostEqual(m['endpoint_RMSE_mV'],math.sqrt(13/2),10)
        self.assertAlmostEqual(m['target_time_equal_RMSE_mV'],math.sqrt(4.5),10)
        self.assertAlmostEqual(m['P95_mV'],2.9,10)
        self.assertAlmostEqual(m['R2'],1-14e-6/2,10)
    def test_constant_truth_undefined_R2(self):
        f=pd.DataFrame(dict(origin_row=[0,0],target_row=[1,2],horizon_step=[1,24],y_true_delta_V=[0.,0.],y_pred_delta_V=[.001,-.001]))
        self.assertIsNone(metrics(f)['R2'])

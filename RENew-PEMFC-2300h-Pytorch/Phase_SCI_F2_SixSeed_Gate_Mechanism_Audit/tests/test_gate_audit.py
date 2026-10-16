import sys,unittest,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
import torch
from run_inference import gate_hook
from verify import independent
class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.regime_gate=torch.nn.Sigmoid()
    def forward(self,x):return self.regime_gate(x)
class AuditTests(unittest.TestCase):
    def test_derangement_distribution_and_no_self(self):
        f=derangements(205,[11,12]);a=f[f.permutation_seed==11].source_window.to_numpy()
        self.assertTrue(np.all(a!=np.arange(205)));self.assertTrue(np.array_equal(np.sort(a),np.arange(205)))
    def test_derangement_repeatable(self):self.assertTrue(derangements(205,[11]).equals(derangements(205,[11])))
    def test_hook_assigned_in_order(self):
        m=Dummy();x=torch.tensor([[0.],[1.],[2.]]);a=np.array([.2,.3,.4],dtype='float32')
        with torch.no_grad():
            before=m(x).numpy().copy()
            with gate_hook(m,a) as (g,c,o):
                y=torch.cat([m(x[:2]),m(x[2:])]).numpy().ravel();self.assertTrue(np.array_equal(y,a));self.assertEqual(o[0],3)
            self.assertTrue(np.array_equal(before,m(x).numpy()));self.assertFalse(m.regime_gate._forward_hooks)
    def test_hook_removes_on_error(self):
        m=Dummy()
        with self.assertRaises(ValueError):
            with gate_hook(m):raise ValueError('synthetic-hook-cleanup-test')
        self.assertFalse(m.regime_gate._forward_hooks)
    def test_target_equal_is_not_average_prediction(self):
        f=pd.DataFrame(dict(origin_row=[1,2,3],target_row=[4,4,5],horizon_step=[24,24,24],y_true_delta_V=[0.,0.,0.],y_pred_delta_V=[.001,-.001,.003]))
        self.assertAlmostEqual(metrics(f)['target_time_equal_RMSE_mV'],np.sqrt(5))
    def test_independent_calculator(self):
        f=pd.DataFrame(dict(origin_row=[1,2,3],target_row=[4,4,5],horizon_step=[24,24,24],y_true_delta_V=[.1,.2,.3],y_pred_delta_V=[.101,.199,.303]))
        for k,v in independent(f).items():self.assertAlmostEqual(v,metrics(f)[k],places=10)
    def test_no_training_calls_in_executed_scripts(self):
        for p in (P/'scripts').glob('*.py'):
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,['backward','step','fit_condition','fit','run_training'])
    def test_historical_write_guard(self):
        with self.assertRaises(AssertionError):dest('../historical-forbidden.txt')
    def test_quantile_linear(self):
        from verify import quantile
        for q in [.05,.25,.95,.99]:self.assertAlmostEqual(quantile([1.,2.,4.,9.],q),np.quantile([1.,2.,4.,9.],q))
if __name__=='__main__':unittest.main()

"""只对低阶数值与时间安全做针对性测试，不执行历史阶段。"""
import json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from 迁移核心 import prepare,fit,build_models,predict,propagate_to_cut,project,slope
TASK=Path(__file__).resolve().parents[1]
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=json.loads((TASK/'配置/阶段2C冻结协议.json').read_text(encoding='utf-8'))
        cls.fc,cls.r=prepare(pd.read_csv(TASK/'数据准备/FC1小时观测.csv'),pd.read_csv(TASK/'数据准备/RealHyFC循环观测.csv'))
        cls.source=fit(cls.fc[cls.fc.valid&(cls.fc.partition=='Train')],'FC1')
    def test_ols_unit_conversion(self):
        x=np.arange(50)*12.;y=.001+2e-5*x
        self.assertAlmostEqual(slope(x,y)[1],2e-5,places=15)
        self.assertAlmostEqual(slope(x,.74*y)[1],.74*2e-5,places=15)
    def test_cutoff_counts(self):
        self.assertEqual([len(self.r[self.r.valid&(self.r.h<=c)]) for c in [300,600,629.2038]],[23,47,48])
    def test_future_hist_rejected(self):
        h=self.r[self.r.valid&(self.r.h<=629.2038)]
        with self.assertRaises(ValueError):build_models(h,self.source,300,self.p)
    def test_rate_projection(self):
        p={'rate_cap':1e-3,'acceleration_cap':1e-5}
        self.assertAlmostEqual(project(1,1e-4,2,p),1.2e-4)
        self.assertGreaterEqual(project(-1,1e-4,2,p),0)
    def test_nominal_clock_not_negative_at_600(self):
        anchor=int(self.r[self.r.h<=600].id.max());lead=12*(52-anchor)
        self.assertEqual(anchor,49);self.assertEqual(lead,36);self.assertGreater(600+lead,600)
    def test_future_data_not_input(self):
        h=self.r[self.r.valid&(self.r.h<=300)].copy();a=build_models(h,self.source,300,self.p)
        changed=self.r.copy();changed.loc[changed.h>300,['v','SOH','D','d','available_h','h']]=-9999
        # 起点历史身份由原冻结截止清单给出；改变未来标签不进入此清单。
        b=build_models(changed[changed.id.isin(h.id)].copy(),self.source,300,self.p)
        for m in a[3]:np.testing.assert_array_equal(predict(a[3][m]['at_cutoff'],[12,360,1200])[0],predict(b[3][m]['at_cutoff'],[12,360,1200])[0])
    def test_no_midway_true_state_update(self):
        h=self.r[self.r.valid&(self.r.h<=300)];s=build_models(h,self.source,300,self.p)[3]['D_Target2B_SlowReversible']['at_cutoff']
        later=propagate_to_cut(s,629.2038)
        np.testing.assert_allclose(predict(s,1000)[0],predict(later,1000-(629.2038-300))[0],rtol=0,atol=1e-15)
    def test_keep_recovery_sign(self):
        s=dict(L=.01,q=0.,R=.005,tau_h=12,last_h=0.)
        self.assertLess(float(predict(s,24)[0]),float(predict(s,0)[0]))
        s['R']=-.005;self.assertGreater(float(predict(s,24)[0]),float(predict(s,0)[0]))
    def test_missing_not_filled(self):
        good=self.r[self.r.valid&self.r.id.between(52,76)]
        self.assertNotIn(75,good.id.tolist());self.assertLess(len(good),25)
    def test_source_only_original_train(self):
        self.assertEqual(self.source['n_fit'],669);self.assertEqual(self.source['last_h'],692)
    def test_source_weight_frozen(self):
        h=self.r[self.r.valid&(self.r.h<=300)];p,_,_,states=build_models(h,self.source,300,self.p)
        expected=.25*self.source['q_bounded']+.75*p['q_bounded'];self.assertEqual(states['C_FC1PriorAdapted']['at_cutoff']['q'],expected)
if __name__=='__main__':unittest.main(verbosity=2)

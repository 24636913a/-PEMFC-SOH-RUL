import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from health_common import load_sources,VREF
from history_models import *

class MathTests(unittest.TestCase):
    def test_irregular_actual_times_and_units(self):
        ages=np.array([0.,17.,38.,70.,90.]);r=1.+.0001*ages
        p=pd.DataFrame({'measurement_end_h':ages,'R':r,'cycle_interval_id':['C'+str(i) for i in range(5)]})
        x,a=historical_features(p,90.,r[-1],96)
        self.assertTrue(a['eligible']);self.assertAlmostEqual(x[3],.0001)
        self.assertAlmostEqual(x[2],r.std(ddof=0));self.assertAlmostEqual(x[4],r[-1]-np.median(r))
    def test_coverage_and_count_both_required(self):
        p=pd.DataFrame({'measurement_end_h':[0,90],'R':[1,1],'cycle_interval_id':['A','B']})
        _,a=historical_features(p,90.,1.,96);self.assertFalse(a['eligible']);self.assertEqual(a['reason'],'COUNT_BELOW_MINIMUM')
        p=pd.DataFrame({'measurement_end_h':np.arange(10.),'R':np.ones(10),'cycle_interval_id':list(map(str,range(10)))})
        _,a=historical_features(p,9.,1.,192);self.assertFalse(a['eligible']);self.assertEqual(a['reason'],'SPAN_BELOW_70_PERCENT')
    def test_future_suffix_never_used(self):
        ages=np.array([0.,17.,38.,70.,90.]);p=pd.DataFrame({'measurement_end_h':ages,'R':1.+.0001*ages,'cycle_interval_id':list(map(str,range(5)))})
        a=historical_features(p,90.,p.iloc[-1].R,96)
        full=pd.concat([p,pd.DataFrame([{'measurement_end_h':100.,'R':1000.,'cycle_interval_id':'future'}])])
        b=historical_features(full,90.,p.iloc[-1].R,96)
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
    def test_no_out_of_window_fill(self):
        p=pd.DataFrame({'measurement_end_h':[-100.,0.,90.],'R':[100.,1.,1.],'cycle_interval_id':['old','a','b']})
        _,a=historical_features(p,90.,1.,96);self.assertEqual(a['cycle_ids'],['a','b'])
    def test_millivolt_conversion(self):
        m=error_metrics([1],[1.001],VREF);self.assertAlmostEqual(m['MAE_mV_equivalent'],VREF)
    def test_empty_validation_not_zero_score(self):
        self.assertIsNone(error_metrics([],[],VREF)['MAE_R'])
    def test_soh_bridge_requires_actual_release(self):
        self.assertEqual(causal_soh_bridge(.98,[{'result_release_datetime':None}],'2025-01-01')['status'],'NO_CERTIFIED_ANCHOR_RELEASE_TIME')
        self.assertIsNone(causal_soh_bridge(.98,[],'2025-01-01')['predicted_SOH_proxy'])
        a=[{'result_release_datetime':'2024-01-01','R_anchor':x,'SOH_anchor':2*x-.9} for x in [.99,.98,.97]]
        got=causal_soh_bridge(.96,a,'2025-01-01');self.assertAlmostEqual(got['predicted_SOH_proxy'],1.02);self.assertIsNone(got['true_SOH_MAE'])

class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.p,cls.authorities,cls.old=load_sources()
    def test_physical_ids_and_reference_count(self):
        self.assertEqual(len(self.p),131)
        for h,step in [('H1',1),('H3',3)]:self.assertTrue((self.authorities[h].target_raw_cycle_id-self.authorities[h].origin_raw_cycle_id==step).all())
    def test_fold2_has_no_common_validation_under_frozen_rules(self):
        for h in ['H1','H3']:
            f,_=build_features(self.p,self.authorities[h],h)
            self.assertEqual(len(f[(f.fold=='fold_2')&(f.role=='validation')&f.common_history_eligible]),0)
    def test_production_future_mutation_prefix_invariance(self):
        row=self.authorities['H1'].iloc[-1];t=row.origin_p1s_measurement_end_ophrs2_h
        for L in [96,192]:
            before=historical_features(self.p,t,row.origin_R_P1S_causal,L)
            changed=self.p.copy();changed.loc[changed.measurement_end_h>t,'R']+=1000
            after=historical_features(changed,t,row.origin_R_P1S_causal,L)
            np.testing.assert_array_equal(before[0],after[0]);self.assertEqual(before[1],after[1])
    def test_target_label_does_not_enter_features(self):
        a,_=build_features(self.p,self.authorities['H1'],'H1');modified=self.authorities['H1'].copy()
        modified['target_R_P1S_causal']+=100
        b,_=build_features(self.p,modified,'H1')
        cols=[f'L{L}_{x}' for L in [96,192] for x in FEATURES]
        np.testing.assert_array_equal(a[cols].to_numpy(),b[cols].to_numpy())
    def test_validation_labels_do_not_choose_alpha_or_scaler(self):
        f,_=build_features(self.p,self.authorities['H1'],'H1');train,val,_,_=train_validation(f,'fold_1')
        modified=val.copy();modified.target_R_P1S_causal+=1000
        for L in [96,192]:
            p,a=ridge_fit(train,val,L);q,b=ridge_fit(train,modified,L)
            np.testing.assert_array_equal(p,q);self.assertEqual(a,b)
    def test_causal_parameter_fit_cutoffs(self):
        for h in ['H1','H3']:
            f,_=build_features(self.p,self.authorities[h],h)
            for fold in ['fold_1','fold_2','fold_3']:
                train,val,_,cut=train_validation(f,fold)
                self.assertTrue((train.target_p1s_measurement_end_ophrs2_h<=cut+1e-9).all())
                fit,selection,_,innercut=inner_split(train)
                self.assertTrue((fit.target_p1s_measurement_end_ophrs2_h<=innercut+1e-9).all())
                self.assertTrue(set(fit.sample_index).isdisjoint(selection.sample_index))
    def test_scaler_train_only_five_columns(self):
        f,_=build_features(self.p,self.authorities['H1'],'H1');train,val,_,_=train_validation(f,'fold_1')
        _,record=ridge_fit(train,val,192)
        cols=[f'L192_{x}' for x in FEATURES]
        np.testing.assert_allclose(record['train_scaler_mean'],train[cols].mean().to_numpy(),rtol=0,atol=1e-12)
        self.assertEqual(len(record['coef_standardized_X']),5);self.assertFalse(record['target_standardized'])

if __name__=='__main__':unittest.main()

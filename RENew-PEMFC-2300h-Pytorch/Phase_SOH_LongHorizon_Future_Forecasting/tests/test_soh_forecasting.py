import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from soh_methods import *
from soh_sources import load_sources,VREF

class MathTests(unittest.TestCase):
    def test_ols_exact_line_and_positive_slope(self):
        b,a=ols([0,1,2],[1,3,5]);self.assertEqual((b,a),(2.,1.))
    def test_affine_positive_and_constrained_negative(self):
        self.assertEqual(affine_parameters([0,1,2],[1,3,5]),(2.,1.))
        self.assertEqual(affine_parameters([0,1,2],[3,2,1]),(0.,2.))
    def test_affine_constant_predictor(self):
        self.assertEqual(affine_parameters([1,1],[2,4]),(0.,3.))
    def test_unit_percentage_points(self):
        self.assertAlmostEqual((.975-.970)*100,.5)
    def test_threshold_all_cases(self):
        self.assertEqual(crossing(.96,0,250)['status'],'ALREADY_BELOW_AT_SCENARIO_ORIGIN')
        self.assertEqual(crossing(.98,0,250)['status'],'NO_FINITE_DOWNCROSSING_NONDECLINING')
        self.assertEqual(crossing(.98,-.0001,250)['status'],'CROSSING_BOUNDARY_WITHIN_HORIZON')
        self.assertEqual(crossing(.98,-.00001,250)['status'],'NO_CROSSING_WITHIN_HORIZON')
    def test_ols_bad_data_rejected(self):
        with self.assertRaises(ValueError):ols([1],[2])
        with self.assertRaises(ValueError):ols([1,1],[2,3])

class ProductionIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.refs,cls.mapping,cls.p1s,cls.timing,cls.bol,cls.ahc=load_sources()
    def prepare(self,o,refs=None,mapping=None,p1s=None):
        return origin_inputs(o,self.refs if refs is None else refs,self.mapping if mapping is None else mapping,self.p1s if p1s is None else p1s,self.timing)
    def test_bol_and_causal_count(self):
        self.assertEqual(len(self.p1s),131);self.assertEqual(float(self.bol['normalized_health_value']),1.)
        self.assertEqual(float(self.bol['reference_voltage_V']),.6989533305168152)
    def test_future_target_and_mapping_mutation_no_prediction_change(self):
        for _,o,t in TASKS:
            before=forecasts(o,t,self.prepare(o))[0]
            refs=self.refs.copy();refs.loc[refs.nominal_h>o,'Y']+=100
            mapping=self.mapping.copy();mapping.loc[mapping.checkpoint>o,'P1S_operating_voltage_retention']+=100
            after=forecasts(o,t,self.prepare(o,refs=refs,mapping=mapping))[0]
            self.assertEqual(before,after)
    def test_future_p1s_suffix_mutation_no_prediction_change(self):
        for _,o,t in TASKS:
            inp=self.prepare(o); before=forecasts(o,t,inp)[0]
            p=self.p1s.copy();p.loc[p.available_datetime>=inp['cutoff'],'R']+=100
            self.assertEqual(before,forecasts(o,t,self.prepare(o,p1s=p))[0])
    def test_prefix_only_matches_full_filtered_history(self):
        for _,o,t in TASKS:
            inp=self.prepare(o);prefix=self.p1s.loc[self.p1s.available_datetime<inp['cutoff']]
            self.assertEqual(forecasts(o,t,inp)[0],forecasts(o,t,self.prepare(o,p1s=prefix))[0])
    def test_window_96h_and_unique_cycles_and_mapping(self):
        for _,o,t in TASKS:
            inp=self.prepare(o);w=inp['window'];last=inp['latest']
            self.assertGreaterEqual(len(w),4);self.assertEqual(w.raw_cycle_id.nunique(),len(w))
            self.assertTrue((w.actual_measurement_end_h>=last.actual_measurement_end_h-96).all())
            self.assertTrue((w.available_datetime<inp['cutoff']).all())
            self.assertEqual(inp['calibration_checkpoints'],[x for x in [300,600,850,1150,1500,1750] if x<=o])
    def test_nominal_850_not_actual_850(self):
        inp=self.prepare(850)
        self.assertGreater(float(inp['mapping'].measurement_end_ophrs2_h.max()),850)
        self.assertLess(float(inp['mapping'].measurement_end_ophrs2_h.max()),float(inp['timing'].PAA_gap_cutoff_ophrs2_h))
    def test_missing_cycles_m3_not_fabricated(self):
        inp=self.prepare(850);inp['window']=inp['window'].iloc[-3:]
        self.assertIsNone(forecasts(850,1150,inp)[0]['M3'])
    def test_m2_m3_share_affine_and_no_future_R(self):
        for _,o,t in TASKS:
            inp=self.prepare(o);pred,details=forecasts(o,t,inp)
            self.assertAlmostEqual(pred['M2'],inp['a']*details['R_latest']+inp['b'])
            self.assertAlmostEqual(pred['M3'],inp['a']*details['R_pred_target']+inp['b'])

if __name__=='__main__':unittest.main()

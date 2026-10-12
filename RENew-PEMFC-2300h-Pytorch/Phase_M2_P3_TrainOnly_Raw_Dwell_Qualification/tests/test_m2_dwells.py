import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from m2_dwells import measured_hold,split_complete_runs,CELLS,COND,apply_stability_thresholds

class DwellTests(unittest.TestCase):
    def data(self,n=130):
        f=pd.DataFrame({'OpHrs2 [h]':100.+np.arange(n)/3600.,'_dt':pd.date_range('2025-01-01',periods=n,freq='s'),
             '_record':np.arange(n),'operation_point':'221','I.S.set [A]':221.2,
             'I.S.act [A]':221.2,'j.S.act [A/cm2]':.79})
        for c in CELLS: f[c]=.7
        for c in COND.values(): f[c]=50.
        bins=pd.DataFrame({'bin_start_ophrs2_h':[100.],'bin_end_ophrs2_h':[100.5],
           'temporal_segment_id':[1],'source_file_names':['fake.csv']})
        return f,bins
    def test_true_measurement_record_window(self):
        f,b=self.data(); row=measured_hold(f,'fake.csv',b)
        self.assertEqual(row['raw_record_start_inclusive'],0)
        self.assertEqual(row['raw_record_end_exclusive'],130)
        self.assertGreaterEqual(row['measurement_raw_record_start'],32)
        self.assertLessEqual(row['measurement_raw_record_end_exclusive'],98)
        self.assertAlmostEqual(row['voltage_median_V'],.7)
        self.assertFalse(row['dwell_quasi_steady_certified'])
    def test_short_hold_not_a_dwell(self):
        f,b=self.data(60); self.assertIsNone(measured_hold(f,'fake.csv',b))
    def test_RH_over100_retained_and_flagged(self):
        f,b=self.data(); f[COND['RH_anode']]=130.
        row=measured_hold(f,'fake.csv',b)
        self.assertEqual(row['RH_anode_median'],130.)
        self.assertTrue(row['RH_physical_domain_pending'])
    def test_load_switch_splits(self):
        f,_=self.data(); f.loc[65:,'I.S.set [A]']=156.8
        complete,tail=split_complete_runs(f)
        self.assertEqual(len(complete),1); self.assertEqual(len(complete[0]),65); self.assertEqual(len(tail),65)
    def test_time_gap_splits(self):
        f,_=self.data(); f.loc[65:,'_dt']+=pd.Timedelta(seconds=10)
        complete,tail=split_complete_runs(f)
        self.assertEqual(len(complete),1)
    def test_stability_does_not_certify_physics(self):
        f,b=self.data(); rows=[measured_hold(f,'fake.csv',b) for _ in range(20)]
        output,limits=apply_stability_thresholds(pd.DataFrame(rows))
        self.assertEqual(limits['calibration_hold_count'],20)
        self.assertTrue(output.relative_quiet_candidate.all())
        self.assertFalse(output.dwell_quasi_steady_certified.any())
    def test_no_encoder_imports_in_production(self):
        import ast
        scripts=Path(__file__).resolve().parents[1]/'scripts'
        for p in scripts.glob('*.py'):
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertFalse(any(x.name.startswith(('torch','tensorflow','sklearn','optuna')) for x in node.names))
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('torch','Phase_','phase_')))

if __name__=='__main__': unittest.main(verbosity=2)

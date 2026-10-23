import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
from build_legal_windows import build
def synthetic(n=200):
    f=pd.DataFrame({k:np.linspace(.5,.8,n) for k in FEATURES});f['bin_end_ophrs2_h']=np.arange(n)*.5
    f['bin_valid_for_windows']=1;f['temporal_segment_id']=1;return f
class WindowTests(unittest.TestCase):
    def test_partition_targets_and_history(self):
        d,legal,excluded=build(synthetic(),100,200)
        self.assertEqual(d['origins'][0],100);self.assertEqual(d['origins'][-1],175)
        self.assertEqual(d['targets'].min(),101);self.assertEqual(d['targets'].max(),199)
        self.assertEqual(legal.history_first_row.iloc[0],53);self.assertEqual(len(legal)+len(excluded),100)
    def test_future_conditions_not_inputs(self):
        f=synthetic();d,_,_=build(f,100,200);f.loc[101:,'actual_current_A_std']+=100
        changed,_,_=build(f,100,200);self.assertTrue(np.array_equal(d['x_raw'][0],changed['x_raw'][0]))
    def test_future_nonfinite_is_eligibility_only(self):
        f=synthetic();f.loc[110,'cell_voltage_std']=np.nan
        _,_,e=build(f,100,200);self.assertIn('NONFINITE_RAW14_OR_VOLTAGE',e.loc[e.origin_row==100,'all_reasons'].iloc[0])
    def test_gap_segment_and_invalid_reasons(self):
        for kind,field,value,reason in [('bad','bin_valid_for_windows',0,'INVALID_BIN'),('seg','temporal_segment_id',2,'TEMPORAL_SEGMENT_BREAK'),('gap','bin_end_ophrs2_h',999,'OPERATING_TIME_GAP')]:
            f=synthetic();f.loc[110,field]=value;_,_,e=build(f,100,200)
            self.assertIn(reason,e.loc[e.origin_row==100,'all_reasons'].iloc[0])
    def test_history_and_end_reasons(self):
        _,_,e=build(synthetic(),0,200)
        self.assertEqual(e.iloc[0].primary_reason,'HISTORY_BEFORE_DATASET')
        self.assertEqual(e.iloc[-1].primary_reason,'24_TARGETS_OUTSIDE_PARTITION')

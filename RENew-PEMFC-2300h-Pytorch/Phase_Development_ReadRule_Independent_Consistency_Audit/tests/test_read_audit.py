import json,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from audit_read_rule import same_numeric,independent_selector,guarded_feature_extent,CUTOFF
class ReadContracts(unittest.TestCase):
 def meta(self):
  x=np.full(3007,100.);x[-2:]=[CUTOFF,CUTOFF+1]
  return pd.DataFrame({'bin_end_ophrs2_h':x,'split':['validation']*3005+['boundary_excluded','test']})
 def test_strict_cutoff_and_original_order(self):
  d=self.meta();ids=independent_selector(d);self.assertTrue(np.array_equal(ids,np.arange(3005)));self.assertEqual(guarded_feature_extent(ids,d),3005)
 def test_noncontiguous_not_assumed_first3005(self):
  d=self.meta();d.loc[10,'bin_end_ophrs2_h']=CUTOFF+1
  ids=independent_selector(d)
  with self.assertRaises(PermissionError):guarded_feature_extent(ids,d)
 def test_Test_inside_mask_denies_feature_parser(self):
  d=self.meta();d.loc[10,'split']='test'
  with self.assertRaises(PermissionError):guarded_feature_extent(independent_selector(d),d)
 def test_single_ULP_difference_is_rejected(self):
  self.assertFalse(same_numeric([1.],[np.nextafter(1.,2.)])[0])
 def test_missing_and_signed_zero_are_explicit(self):
  self.assertTrue(same_numeric([np.nan],[np.nan])[0]);self.assertFalse(same_numeric([np.nan],[0.])[0]);self.assertFalse(same_numeric([0.],[-0.])[0])
 def test_calendar_nanosecond_not_tolerated(self):
  a=pd.Timestamp('2025-01-01 00:00:00.005000');b=a+pd.Timedelta(1,'ns');self.assertNotEqual(a.value,b.value)
 def test_saved_result_and_no_performance_work(self):
  r=json.loads((P/'outputs/FINAL_AUDIT.json').read_text());self.assertEqual(r['status'],'PASS_EXACT');self.assertEqual(r['compared_cells'],105175)
  self.assertEqual(r['maximum_numeric_absolute_difference'],0);self.assertFalse(r['excluded_Test_numeric_fields_parsed']);self.assertFalse(r['GPU']);self.assertFalse(r['trained'])
if __name__=='__main__':unittest.main()

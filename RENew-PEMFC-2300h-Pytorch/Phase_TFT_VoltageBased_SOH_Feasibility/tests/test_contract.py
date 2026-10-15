"""Safety/identity boundaries with regression checks on the real saved results."""
import sys,json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from common import selected_extent,read
class ContractTests(unittest.TestCase):
 def test_cutoff_strict_and_authority(self):
  meta=read(P/'data_contracts/HISTORICAL_3007_METADATA_MASK.csv')
  ids=selected_extent(meta);self.assertEqual(len(ids),3005)
  self.assertFalse((meta.iloc[ids].bin_end_ophrs2_h>=1819.997738).any())
 def test_never_silently_prefix_on_noncontiguous_mask(self):
  m=read(P/'data_contracts/HISTORICAL_3007_METADATA_MASK.csv');m.loc[2,'bin_end_ophrs2_h']=1819.997738
  with self.assertRaises(PermissionError): selected_extent(m)
 def test_Test_identity_rejected(self):
  m=read(P/'data_contracts/HISTORICAL_3007_METADATA_MASK.csv');m.loc[2,'split']='Test'
  with self.assertRaises(PermissionError): selected_extent(m)
 def test_missing_authority_rows_rejected(self):
  m=read(P/'data_contracts/HISTORICAL_3007_METADATA_MASK.csv').iloc[:3005]
  with self.assertRaises(ValueError):selected_extent(m)
 def test_no_unqualified_ratios(self):
  d=read(P/'outputs/CONDITIONAL_RETENTION_DIAGNOSTIC.csv')
  self.assertTrue(d.HI_BoL_percent.isna().all())
  self.assertTrue(d.loc[~d.conditional_retention_supported,'conditional_voltage_retention_pred_percent'].isna().all())
  self.assertFalse(d.online_SOH_qualified.any())
 def test_no_voltage_in_matching_fields(self):
  cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text())
  self.assertTrue(all('voltage' not in c.lower() for c in cfg['condition_fields']))
  self.assertEqual(cfg['primary_caliper_IQR'],.5)
  self.assertEqual(cfg['sensitivity_calipers_IQR'],[.25,1.])
 def test_only_qualified_reference_cycles_in_denominator(self):
  d=read(P/'outputs/MATCHING_BY_TARGET_AND_CALIPER.csv')
  self.assertTrue(d.loc[d.conditional_retention_supported,'reference_cycles'].ge(3).all())
  self.assertTrue(d.loc[~d.conditional_retention_supported,'reference_mean_V'].isna().all())
 def test_cycle_not_hour_identity(self):
  p=read(P/'outputs/FROZEN_CYCLE_PERFORMANCE_PREDICTIONS.csv')
  self.assertTrue((p.target_raw_cycle_id-p.origin_raw_cycle_id==p.H).all())
  self.assertTrue((p.target_actual_end_h-p.origin_actual_end_h>9*p.H).all())
 def test_unsupported_comparisons_not_zero_errors(self):
  a=json.loads((P/'outputs/AUXILIARY_QUALIFICATION.json').read_text())
  self.assertIsNone(a['correlation_or_direction_scores'])
  self.assertEqual(a['qualified_common_origin_and_operator_pairs'],0)
 def test_no_training_dependencies(self):
  import ast
  for f in (P/'scripts').glob('*.py'):
   tree=ast.parse(f.read_text(encoding='utf-8'))
   names=[]
   for node in ast.walk(tree):
    if isinstance(node,ast.Import):names += [n.name.split('.')[0] for n in node.names]
    if isinstance(node,ast.ImportFrom):names.append((node.module or '').split('.')[0])
   self.assertFalse(set(names)&{'torch','sklearn','optuna'},f.name)
if __name__=='__main__':unittest.main()

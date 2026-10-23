from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE=Path(__file__).resolve().parents[1]/"scripts"/"run_raw_operation_point_health_audit.py"
SPEC=importlib.util.spec_from_file_location("rha",MODULE); assert SPEC and SPEC.loader
rha=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=rha; SPEC.loader.exec_module(rha)


class RhaProductionContractTests(unittest.TestCase):
    def test_boundary_filter_counts_access_and_never_keeps_post_boundary(self):
        frame=pd.DataFrame({"OpHrs2 [h]":["1,819.90","1,820.00"],"DateTime":["2020-01-01","2020-01-01"]})
        kept, accessed, crossing=rha.boundary_filter(frame)
        self.assertEqual(accessed,1); self.assertTrue(crossing); self.assertEqual(len(kept),1); self.assertTrue((kept["OpHrs2 [h]"]<rha.BOUNDARY).all())

    def test_operation_sort_is_numeric_then_lexical(self):
        self.assertLess(rha.op_sort("12"),rha.op_sort("221")); self.assertLess(rha.op_sort("221"),rha.op_sort("A"))

    def test_frozen_dwell_windows_require_valid_kind_and_middle_definition(self):
        target=pd.DataFrame({"start_datetime":["2020-01-01 00:00:00"],"duration_wall_s":[120.0],"run_id":[1]})
        middle=rha.window_table(target,"middle_50")
        self.assertEqual((middle.window_end.iloc[0]-middle.window_start.iloc[0]).total_seconds(),60.0)
        with self.assertRaises(ValueError): rha.window_table(target,"centered_future")

    def test_window_end_filter_excludes_following_run_sample(self):
        starts=np.array([10],dtype=np.int64); ends=np.array([20],dtype=np.int64); dt=np.array([15,21],dtype=np.int64)
        _,valid=rha.frozen_window_membership(dt,starts,ends)
        self.assertEqual(valid.tolist(),[True,False])

    def test_moving_block_sampler_uses_contiguous_source_chunks(self):
        rng=np.random.default_rng(2024); sample=rha.moving_block_sample([np.array([1.,2.,3.,4.,5.])],8,4,rng)
        self.assertEqual(len(sample),8)
        self.assertTrue(all(abs(a-b)<=3 for a,b in zip(sample[:-1],sample[1:])))

    def test_final_status_rejects_condition_confounded_drift(self):
        drift=pd.DataFrame({"raw_negative":[True,True],"signal_gt_2S":[True,True],"bootstrap4_upper_negative":[True,True],"bootstrap8_no_direction_reversal":[True,True],"matched_negative":[False,True]})
        self.assertEqual(rha.final_status({"A":True,"B":True,"joint":True,"C":True,"D":False},drift,"NOT_RUN",False),"VOLTAGE_DRIFT_CONDITION_CONFOUNDED")


if __name__=="__main__": unittest.main()

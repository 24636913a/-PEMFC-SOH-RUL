"""Synthetic regression tests for the V2 amendment, independent of endpoint analysis."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/"scripts"))
import ha_common
from amendment_v2_core import (condition_adjusted, joint_match, matched_moving_bootstrap,
                               recovery_audit, residual_moving_sample, rdr_audit)
from run_health_audit_v2 import final_decision


def blocks(points, endpoint=True):
    rows=[]
    for i,values in enumerate(points):
        row={"origin_start_row":i*48,"origin_end_row":i*48+47,"age_midpoint_h":i*24.,"chronological_third":"EARLY" if i<5 else "LATE","temporal_segment_id":1 if i<5 else 2,"current_density_stratum":"LOW","block_hours":24,
             "current_density":values[0],"current_density_std":values[1],"load_transition_count":values[2],"zero_current_fraction":values[3]}
        if endpoint:
            for rep in ("Raw","SimpleTrend","EMD"): row[f"{rep}_endpoint_median_V"]=values[4] if len(values)>4 else 0.
        rows.append(row)
    return pd.DataFrame(rows)


class AmendmentV2Tests(unittest.TestCase):
    def test_dual_hash_semantics_pass_when_values_differ(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"origin_9.npz"; features=np.arange(96,dtype=np.float32).reshape(48,2)
            np.savez(path,features=features,origin_row=np.array(9),input_voltage_sha256=np.array("in"),method_parameter_sha256=np.array("param"))
            payload=hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest(); file_hash=ha_common.sha256_file(path)
            self.assertNotEqual(payload,file_hash)
            row=pd.Series({"cache_path":str(path),"method":"EMD","origin_row":9,"cache_sha256":payload,"input_voltage_sha256":"in","method_parameter_sha256":"param"})
            local=pd.DataFrame([{ "artifact_kind":"origin_window_feature_cache","method":"EMD","local_absolute_path":str(path),"relative_phase_path":"cache/EMD/origin_9.npz","sha256":file_hash}])
            _,audit=ha_common.validate_cache_entry(row,local)
            self.assertTrue(audit["CACHE_INTEGRITY_PASS"])
            self.assertTrue(audit["feature_payload_hash_pass"] and audit["artifact_file_hash_pass"])

    def test_wrong_payload_or_file_hash_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"x.npz"; f=np.ones((48,2),np.float32);np.savez(path,features=f,origin_row=np.array(1),input_voltage_sha256=np.array("a"),method_parameter_sha256=np.array("b"))
            row=pd.Series({"cache_path":str(path),"method":"EMD","origin_row":1,"cache_sha256":"wrong","input_voltage_sha256":"a","method_parameter_sha256":"b"})
            local=pd.DataFrame([{ "artifact_kind":"origin_window_feature_cache","method":"EMD","local_absolute_path":str(path),"relative_phase_path":"x","sha256":ha_common.sha256_file(path)}])
            with self.assertRaises(ha_common.AuditFailure): ha_common.validate_cache_entry(row,local)
            row.cache_sha256=ha_common.sha256_payload(f);local.loc[0,"sha256"]="wrong"
            with self.assertRaises(ha_common.AuditFailure): ha_common.validate_cache_entry(row,local)

    def test_prefix_fingerprint_excludes_suffix_and_source_hashes_never_hash_n1(self):
        frame=pd.DataFrame({c:["x"]*3007 if c=="last_datetime" else np.arange(3007,dtype=float) for c in ha_common.SAFE_N1_COLUMNS})
        frame.index=pd.RangeIndex(3007); first=ha_common.allowed_prefix_fingerprint(frame.copy()); suffix=pd.concat([frame,pd.DataFrame([frame.iloc[-1]])],ignore_index=True)
        self.assertEqual(first,ha_common.allowed_prefix_fingerprint(suffix.iloc[:3007].copy()))
        original=ha_common.sha256_file
        def guarded(path):
            if Path(path)==ha_common.N1_TABLE: raise AssertionError("forbidden full N1 hash")
            return original(path)
        ha_common.sha256_file=guarded
        try: ha_common.source_hashes()
        finally: ha_common.sha256_file=original

    def test_emd_feature_is_restored_voltage_and_slope_is_v_per_h(self):
        spec=importlib.util.spec_from_file_location("dfemd",ha_common.DF_DIR/"scripts"/"decomposition_features.py");module=importlib.util.module_from_spec(spec);sys.modules["dfemd"]=module;spec.loader.exec_module(module)
        history=np.linspace(.61,.72,48)+.005*np.sin(np.arange(48)); result=module.emd_feature(history); z,mu,scale=module._normalise(history); emd=module._configured_emd();emd.emd(z);imfs,_=emd.get_imfs_and_residue(); removed=np.asarray(imfs)[:min(2,len(imfs))].sum(axis=0) if len(imfs) else np.zeros(48); low=mu+scale*(z-removed)
        self.assertTrue(np.allclose(result.features[:,0],low.astype(np.float32)))
        self.assertTrue(np.allclose(result.features[:,1],module.causal_slope8(low).astype(np.float32)))

    def test_joint_overlap_without_joint_match_fails(self):
        pts=[(0,0,0,0,0)]*3+[(1,1,1,1,0)]*2+[(0,1,0,1,0)]*3+[(1,0,1,0,0)]*2
        data=blocks(pts); early=data.iloc[:5];late=data.iloc[5:]; pairs,info=joint_match(early,late,{"current_density":1,"current_density_std":1,"load_transition_count":1,"zero_current_fraction":1})
        self.assertFalse(info["MATCHED_COMMON_SUPPORT_PASS"]);self.assertEqual(len(pairs),0)

    def test_only_matched_pairs_enter_contrast_and_matching_is_deterministic(self):
        pts=[(.1,.1,0,0,1),(.2,.1,0,0,2),(.3,.1,0,0,3),(.4,.1,0,0,4),(.5,.1,0,0,5),(.1,.1,0,0,0),(.2,.1,0,0,0),(.3,.1,0,0,0),(.4,.1,0,0,0),(.5,.1,0,0,0),(3,.1,0,0,999)]
        data=blocks(pts);e,l=data.iloc[:5],data.iloc[5:];sc={"current_density":.25,"current_density_std":1,"load_transition_count":1,"zero_current_fraction":1}; p1,i1=joint_match(e,l,sc);p2,i2=joint_match(e,l,sc)
        self.assertTrue(i1["MATCHED_COMMON_SUPPORT_PASS"]);self.assertEqual(p1.to_json(),p2.to_json());self.assertEqual(len(p1),5);self.assertNotIn(10,p1.late_index.to_list())

    def test_moving_bootstrap_and_residual_sampler_are_contiguous_and_not_iid(self):
        data=blocks([(1,1,0,0,1)]*5+[(1,1,0,0,0)]*5); data.loc[:4,"temporal_segment_id"]=1;data.loc[5:,"temporal_segment_id"]=2
        sample=ha_common.moving_block_indices(data,2,np.random.default_rng(2024));self.assertEqual(len(sample),len(data));self.assertTrue(all(i<5 for i in sample[:5]) and all(i>=5 for i in sample[5:]))
        residual=np.arange(len(data),dtype=float);r=residual_moving_sample(data,residual,2,np.random.default_rng(2024));self.assertEqual(len(r),len(data));self.assertFalse("integers(0, n, n)" in Path(PHASE/"scripts"/"amendment_v2_core.py").read_text(encoding="utf-8"))

    def test_stable_rdr_excludes_unstable_and_requires_30(self):
        events=pd.DataFrame([{ "lag_steps":1,"current_density_stratum":"LOW","stable_both":True,"abs_U_V":.01} for _ in range(29)]+[{"lag_steps":1,"current_density_stratum":"LOW","stable_both":False,"abs_U_V":99}])
        c=pd.DataFrame([{ "stratum":"LOW","representation":"EMD","D_matched_late_minus_early_V":-.1}])
        out=rdr_audit(events,c);self.assertEqual(out.loc[out.stratum.eq("LOW"),"status"].iloc[0],"RDR_NOT_IDENTIFIABLE_IN_STABLE_SUBSET")
        # Exercise the production empty-contrast route: no matched support
        # makes RDR undefined, not a dataframe-column exception.
        empty_out=rdr_audit(events,pd.DataFrame())
        self.assertTrue(empty_out.status.eq("RDR_NOT_IDENTIFIABLE_IN_STABLE_SUBSET").all())

    def test_metadata_recovery_returns_amplitude_and_rejects_condition_mismatch(self):
        rows=[]
        for i in range(8): rows.append({"origin_row":i,"origin_age_h":float(i)/2,"temporal_segment_id":1 if i<4 else 2,"current_density_stratum":"LOW","actual_current_density_A_cm2_mean":.4,"actual_current_density_A_cm2_std":.1,"load_transition_count":0,"operation_point_transitions":0,"zero_current_fraction":0,"startup_flag":1 if i==4 else 0,"shutdown_flag":0,"gap_before_flag":1 if i==4 else 0,"gap_after_flag":0,"gap_before_duration_s":10,"Raw_endpoint_V":.6+.01*(i>=4),"SimpleTrend_endpoint_V":.6+.01*(i>=4),"EMD_endpoint_V":.6+.01*(i>=4)})
        out=recovery_audit(pd.DataFrame(rows),{"current_density":1,"current_density_std":1,"load_transition_count":1,"zero_current_fraction":1},.5,.8);self.assertTrue(out.matched_recovery_event.iloc[0]);self.assertGreater(out.G_EMD_V.iloc[0],0)
        bad=pd.DataFrame(rows);bad.loc[4:7,"actual_current_density_A_cm2_mean"]=2.;out2=recovery_audit(bad,{"current_density":1,"current_density_std":1,"load_transition_count":1,"zero_current_fraction":1},.5,.8);self.assertFalse(out2.matched_recovery_event.iloc[0])

    def test_empty_asr_from_no_matched_support_is_a_gate_failure_not_schema_error(self):
        support=pd.DataFrame(columns=["status","stratum"]); empty=pd.DataFrame(); effects=pd.DataFrame([
            {"representation":r,"age_coefficient_V_per_1000h":-1.,"age_ci95_upper_V_per_1000h":-0.1,"uncertainty_status":"PASS","severe_collinearity":False,"relative_residual_sd":1.} for r in ("Raw","SimpleTrend","EMD")])
        trends=pd.DataFrame([{ "representation":"EMD","scope":f"third_{x}","Theil_Sen_slope_V_per_1000h":-1.} for x in ("EARLY","MIDDLE","LATE")])
        rdr=pd.DataFrame([{ "stratum":s,"RDR95":float("nan")} for s in ("LOW","MID","HIGH")])
        _,decision=final_decision(support,empty,empty,empty,effects,effects,empty,rdr,trends)
        self.assertEqual(decision["final_status"],"FORECAST_ONLY_REPRESENTATION")


if __name__=="__main__": unittest.main(verbosity=2)

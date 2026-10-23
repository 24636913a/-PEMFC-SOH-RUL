"""Freeze V2 after synthetic tests and before the first endpoint statistics."""
from __future__ import annotations

import json

from amendment_v2_core import condition_origin_frame, frozen_matching_design
from ha_common import (DF_BASE_COMMIT, HA_DIR, V1_BLOCKER_COMMIT, V1_PROTOCOL_SHA256,
                       assert_history_provenance, cache_universe, execution_code_hashes,
                       p33_p67_and_stable_p50, read_development_prefix, read_manifest,
                       sha256_file, source_hashes, validate_df_status, write_json)


def main():
    assert_history_provenance(); inherited=validate_df_status(); development,firewall=read_development_prefix();cache,_=read_manifest();_,_,common=cache_universe(cache)
    thresholds=p33_p67_and_stable_p50(development,common);condition=condition_origin_frame(development,common,thresholds);blocks,scales,constants=frozen_matching_design(condition,thresholds["P33_A_cm2"],thresholds["P67_A_cm2"])
    amendment={
        "phase":"Phase_HA_EMD_Health_Representation_Audit","amendment":"PRE_ENDPOINT_ANALYSIS_TECHNICAL_AMENDMENT_V2",
        "status":"FROZEN_AFTER_TECHNICAL_BLOCKER_BEFORE_FIRST_ENDPOINT_STATISTICS","phase_df_base_commit":DF_BASE_COMMIT,"v1_blocker_commit":V1_BLOCKER_COMMIT,"v1_protocol_sha256":V1_PROTOCOL_SHA256,
        "inherited_phase_df_status":inherited,"frozen_phase_df_source_hashes":source_hashes(),"execution_code_hashes":execution_code_hashes(),
        "allowed_prefix_firewall":{**firewall,"V1_N1_FULL_FILE_HASH_NOT_USED_BY_EFFECTIVE_PROTOCOL":True,"canonical_serialization":"SAFE_N1_COLUMNS ordered; index 0..3006; UTF-8; LF; <NA>; %.17g"},
        "frozen_thresholds":thresholds,"matching_design":{"analysis_block_hours":24,"fold1_positive_current_block_scale_source":"Fold1 outer-train condition blocks","scales":scales,"constant_variables":constants,"joint_support_variables":["current_density","current_density_std","load_transition_count","zero_current_fraction"],"per_variable_caliper":.5,"euclidean_caliper":1.0,"minimum_raw_early_blocks":5,"minimum_raw_late_blocks":5,"minimum_matched_pairs":5,"matching":"deterministic greedy 1:1 without replacement; outcome excluded"},
        "uncertainty":{"iterations":2000,"seed":2024,"minimum_bootstrap_success_rate":.8,"matched_primary":"24h blocks; 2 consecutive analysis-block moving superblock","matched_24h_sensitivity":"24h blocks; 4 consecutive analysis-block moving superblock","analysis_48h":"48h blocks; 2 consecutive analysis-block moving superblock","regression":"fixed-design residual moving-block bootstrap: 2 analysis blocks"},
        "rdr_stable_subset":{"minimum_stable_lag1_events":30,"definition":"origin and lag1 later origin: positive density, zero_current_fraction=0, load_transition_count=0, operation_point_transitions=0, Fold1 P50 current-density std and peak-to-peak limits"},
        "recovery":{"event_metadata":["startup_flag","shutdown_flag","gap_before_flag","gap_after_flag","gap_before_duration_s"],"pre_post_origins":4,"positive_current_required":True,"matching":"same current-density stratum plus registered condition calipers","minimum_matched_events_for_identifiability":3,"hard_HI_gate":False},
        "units":{"Raw_endpoint_V":"V","SimpleTrend_endpoint_V":"V","SimpleTrend_endpoint_slope_V_per_h":"V/h","EMD_endpoint_V":"V; frozen restored-voltage first-two-IMF-removed EMD remainder feature[47,0]","EMD_endpoint_slope_V_per_h":"V/h; causal slope of restored-voltage remainder feature[47,1]"},
        "overrides":["cache_integrity_semantics","N1_allowed_prefix_fingerprint","EMD_units","joint_common_support_and_matching","time_dependent_uncertainty","stable_current_RDR","recovery_amplitude_audit","associated_regression_tests"],
        "forbidden":["Test_access","full_N1_file_hash","TimesNet_training","Phase_DF_Stage1","EMD_parameter_change","VMD","SOH","EOL","RUL"],
    }
    path=HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.yaml";write_json(path,amendment);digest=sha256_file(path);(HA_DIR/"configs"/"PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.sha256").write_text(f"{digest}  PRE_ENDPOINT_ANALYSIS_AMENDMENT_V2.yaml\n",encoding="utf-8")
    print(json.dumps({"amendment":str(path),"sha256":digest,"P33":thresholds["P33_A_cm2"],"P67":thresholds["P67_A_cm2"],"matching_scale_blocks":len(blocks)},indent=2))


if __name__=="__main__":main()


"""Freeze the Phase HA protocol before loading endpoint feature values."""
from __future__ import annotations

import json
from pathlib import Path

from ha_common import (BASE_COMMIT, CONTEXT_STEPS, DEVELOPMENT_END_H, HA_DIR,
                       RESOLUTION_H, assert_protocol_sources, cache_universe,
                       ensure_base_commit, p33_p67_from_fold1_training,
                       read_development_prefix, read_manifest, sha256_file,
                       source_hashes, validate_df_status, write_json)


def main() -> None:
    ensure_base_commit()
    status = validate_df_status()
    development, firewall = read_development_prefix()
    manifest = read_manifest()
    _, _, common = cache_universe(manifest)
    p33, p67, positive_count = p33_p67_from_fold1_training(development, common)
    protocol = {
        "phase": "Phase_HA_EMD_Health_Representation_Audit",
        "protocol_status": "PRE-REGISTERED_BEFORE_PHASE_HA_ENDPOINT_ANALYSIS",
        "base_commit": BASE_COMMIT,
        "inherited_status": status,
        "frozen_source_hashes": source_hashes(),
        "development_firewall": {**firewall, "development_end_h_exclusive": DEVELOPMENT_END_H},
        "origin_universe": {"expected_unique_common_cached_origins": 1992, "context_steps": CONTEXT_STEPS, "resolution_h": RESOLUTION_H, "cache_methods": ["SimpleTrend", "EMD"]},
        "endpoint_definitions": {
            "Raw_endpoint_V": "cell_voltage_mean at origin",
            "SimpleTrend_endpoint_V": "frozen SimpleTrend features[47,0]",
            "SimpleTrend_endpoint_slope_V_per_h": "frozen SimpleTrend features[47,1]",
            "EMD_endpoint_V": "frozen first-two-IMF-removed EMD remainder features[47,0]; standardized remainder, dimensionless despite inherited field suffix",
            "EMD_endpoint_slope_V_per_h": "frozen EMD features[47,1]; standardized-remainder slope per h",
        },
        "chronological_thirds": {"method": "equal-length intervals over unique legal origin-age range", "labels": ["EARLY", "MIDDLE", "LATE"]},
        "statistical_blocks": {"primary_h": 24, "primary_origins": 48, "sensitivity_h": 48, "sensitivity_origins": 96, "must_not_cross": ["temporal_segment", "chronological_third", "nonconsecutive_origin"]},
        "current_density_strata": {"source": "Fold1 outer-train cached positive-current origins only", "fold1_outer_train_age_h": [72.788690, 946.393214], "positive_origin_count": positive_count, "P33_A_cm2": p33, "P67_A_cm2": p67, "labels": ["LOW", "MID", "HIGH"], "zero_current_label": "ZERO_CURRENT"},
        "common_support_rule": {"minimum_early_blocks": 5, "minimum_late_blocks": 5, "overlap_variables": ["current_density_std", "load_transition_count", "zero_current_fraction"], "overlap": "max(min_early,min_late) <= min(max_early,max_late) for each variable"},
        "matched_age_contrast": {"definition": "median(late blocks) - median(early blocks)", "representations": ["Raw", "SimpleTrend", "EMD"], "bootstrap_iterations": 2000, "bootstrap_seed": 2024, "joint_resample": True},
        "condition_adjusted_association": {"formula": "endpoint ~ age_per_1000h + current_density + current_density_squared + current_density_std + load_transition_count + zero_current_fraction", "uncertainty": "time-aware non-overlapping block bootstrap, 2000 iterations, seed 2024", "severe_condition_number_threshold": 1000, "interpretation": "condition-adjusted association; not a causal degradation effect"},
        "online_roughness": {"adjacency": "same temporal segment, origin-row increment 1, age increment 0.5 h", "metrics": ["median_abs_delta", "p75_abs_delta", "p90_abs_delta", "p95_abs_delta", "p99_abs_delta", "std_delta", "total_variation_per_h", "second_difference_std"]},
        "revision": {"lags": [1, 2, 4, 8], "physical_timestamp_index": "later feature index 47-k", "require_all_intermediate_steps": True},
        "ASR": {"label": "PROJECT_ANALYSIS_DESIGN", "formula": "abs(D_X) / median(abs(adjacent online delta_X))", "required_common_support_positive_strata": 2},
        "RDR": {"label": "PROJECT_ANALYSIS_DESIGN", "formula": "P95(abs(U_1)) / abs(D_EMD)", "undefined_when": ["no common support", "zero/nonidentifiable denominator"], "HI_gate": "RDR95 < 1 in stable positive-current subset"},
        "recovery": {"metadata_only": ["startup_flag", "shutdown_flag", "gap_before_flag", "gap_after_flag", "gap_before_duration_s"], "not_a_hard_HI_gate": True, "prohibit_voltage_selected_events": True},
        "final_decision_logic": {"allowed_statuses": ["FORECAST_ONLY_REPRESENTATION", "DEGRADATION_SENSITIVE_BUT_UNSTABLE", "HI_CANDIDATE_PASS"], "health_evidence_requires": ["at_least_two_common_support_positive_strata", "negative_D_EMD", "24h_CI_excludes_zero", "negative_48h_sensitivity", "negative_condition_adjusted_age_association_without_24h_48h_direction_reversal"], "HI_extra_requires": ["EMD_ASR_greater_than_Raw_and_SimpleTrend_in_at_least_two_strata", "relative_conditional_residual_sd_not_worse_than_Raw_and_SimpleTrend", "RDR95_less_than_one", "no_strong_chronological_or_fold_direction_reversal", "24h_48h_consistency"]},
        "forbidden": ["Test_access", "forecasting_RMSE_revalidation", "EMD_parameter_change", "VMD", "neural_model_training", "backbone_search", "SOH_construction", "EOL_definition", "RUL_model"],
    }
    config = HA_DIR / "configs" / "PREREGISTERED_HA_PROTOCOL.yaml"
    write_json(config, protocol)  # JSON is valid YAML; deterministic and parser-independent.
    digest = sha256_file(config)
    (HA_DIR / "configs" / "PREREGISTERED_HA_PROTOCOL.sha256").write_text(f"{digest}  PREREGISTERED_HA_PROTOCOL.yaml\n", encoding="utf-8")
    assert_protocol_sources(protocol)
    print(json.dumps({"protocol": str(config), "sha256": digest, "P33": p33, "P67": p67}, indent=2))


if __name__ == "__main__":
    main()


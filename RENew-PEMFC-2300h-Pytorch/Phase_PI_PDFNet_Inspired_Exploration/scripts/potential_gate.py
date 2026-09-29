"""Frozen potential classifications for PDFNetInspired_Exploratory."""
from __future__ import annotations

from typing import Any

import pandas as pd


def seed42_gate(candidate: pd.Series, control: pd.Series, gate: dict[str, Any]) -> dict[str, Any]:
    strong, a, b = gate["seed42_gate"]["strongly_promising"], gate["seed42_gate"]["promising_A"], gate["seed42_gate"]["promising_B"]
    values = {
        "formal": float(candidate.formal_RMSE_mV / control.formal_RMSE_mV),
        "fold3": float(candidate.Fold3_RMSE_mV / control.Fold3_RMSE_mV),
        "p99": float(candidate.P99_mV / control.P99_mV),
        "target_equal": float(candidate.target_time_equal_RMSE_mV / control.target_time_equal_RMSE_mV),
        "fold3_target_equal": float(candidate.Fold3_target_time_equal_RMSE_mV / control.Fold3_target_time_equal_RMSE_mV),
        "endpoint": float(candidate.endpoint_RMSE_mV / control.endpoint_RMSE_mV),
    }
    common = bool(candidate.protocol_audit_pass and candidate.core_gradient_pass and candidate.no_leakage and candidate.all_legal_origins_retained and candidate.no_future_13_variable_supervision)
    strong_pass = common and values["formal"] <= strong["formal_RMSE_fraction_of_TimesNet"] and values["fold3"] <= strong["fold3_RMSE_max_fraction"] and values["p99"] <= strong["p99_max_fraction"] and values["target_equal"] <= strong["target_time_equal_max_fraction"] and values["fold3_target_equal"] <= strong["fold3_target_time_equal_max_fraction"] and values["endpoint"] <= strong["endpoint_12h_max_fraction"]
    a_pass = common and values["formal"] <= a["formal_RMSE_max_fraction"] and values["fold3"] <= a["fold3_RMSE_max_fraction"] and values["p99"] <= a["p99_max_fraction"] and values["target_equal"] <= a["target_time_equal_max_fraction"] and values["fold3_target_equal"] <= a["fold3_target_time_equal_max_fraction"] and values["endpoint"] <= a["endpoint_12h_max_fraction"]
    ordered_axis = [("fold3_RMSE", values["fold3"]), ("endpoint_12h_RMSE", values["endpoint"]), ("p99", values["p99"])]
    advantage_axis = next((name for name, ratio in ordered_axis if ratio <= 1.0 - b["required_improvement_fraction_on_one_preordered_axis"]), None)
    b_pass = common and b["formal_RMSE_lower_exclusive_fraction"] < values["formal"] <= b["formal_RMSE_upper_inclusive_fraction"] and advantage_axis is not None and values["fold3"] <= a["fold3_RMSE_max_fraction"] and values["p99"] <= a["p99_max_fraction"] and values["target_equal"] <= a["target_time_equal_max_fraction"] and values["fold3_target_equal"] <= a["fold3_target_time_equal_max_fraction"] and values["endpoint"] <= a["endpoint_12h_max_fraction"]
    status = "PDFNET_INSPIRED_STRONGLY_PROMISING" if strong_pass else "PDFNET_INSPIRED_PROMISING_A" if a_pass else "PDFNET_INSPIRED_PROMISING_B" if b_pass else "PDFNET_INSPIRED_NO_PROMISE"
    return {"status": status, "ratios": values, "common_audit_pass": common, "B_advantage_axis": advantage_axis, "strong_pass": strong_pass, "promising_A_pass": a_pass, "promising_B_pass": b_pass}

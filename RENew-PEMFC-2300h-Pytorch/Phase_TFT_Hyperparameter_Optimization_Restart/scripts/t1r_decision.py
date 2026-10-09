"""Protocol-derived, identity-gated T1-R decisions."""
from __future__ import annotations

import math
from statistics import median
from typing import Any

TECHNICAL = "TFT_T1_TECHNICAL_FAILURE"
INSUFFICIENT = "TFT_OPTUNA_INSUFFICIENT_GAIN"
PENDING = "SEED42_GATE_PASS_PENDING_MULTISEED"
ADVANCE = "TFT_OPTUNA_ADVANCE"
EXTENDED = "TFT_EXTENDED_BUDGET_CONFIG_ADVANCE"


def thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    seed42, multi, a = cfg["decision"]["seed42_optuna_conditions"], cfg["decision"]["multiseed"], cfg["comparators"]["A_frozen_t0"]
    return {"a_rmse": float(a["RMSE_mV"]), "a_fold3": float(a["fold3_RMSE_mV"]), "improve": float(seed42["c_vs_a_relative_improvement_min"]), "competitive": float(seed42["c_rmse_mV_max"]), "fold3": float(seed42["c_fold3_rmse_mV_max"]), "p99_mult": float(seed42["c_p99_vs_b_multiplier_max"]), "target_mult": float(seed42["c_target_time_equal_vs_b_multiplier_max"]), "multi_comp": float(multi["per_seed_c_rmse_mV_max"]), "multi_fold3": float(multi["median_fold3_rmse_mV_max"]), "multi_min": int(multi["min_paired_c_beats_b"])}


def _finite(x: dict[str, Any]) -> bool:
    try:
        return all(math.isfinite(float(x[k])) for k in ("rmse", "fold3_rmse", "p99", "target_equal_rmse"))
    except (KeyError, ValueError, TypeError):
        return False


def seed42_decision(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any], *, cfg: dict[str, Any], identity_audited: bool) -> dict[str, Any]:
    """A formal caller must supply metrics produced after query identity auditing."""
    if not identity_audited or not all(_finite(x) for x in (a, b, c)):
        return {"status": TECHNICAL, "reason": "AUDIT_OR_METRIC_FAILURE"}
    t = thresholds(cfg)
    conditions = {"c_beats_b_strict": c["rmse"] < b["rmse"], "c_improves_a": c["rmse"] <= t["a_rmse"] * (1 - t["improve"]), "c_competitive": c["rmse"] <= t["competitive"], "c_fold3": c["fold3_rmse"] <= t["fold3"], "c_p99": c["p99"] <= b["p99"] * t["p99_mult"], "c_target_equal": c["target_equal_rmse"] <= b["target_equal_rmse"] * t["target_mult"]}
    return {"status": PENDING if all(conditions.values()) else INSUFFICIENT, "reason": "ALL_SEED42_OPTUNA_CONDITIONS_PASS" if all(conditions.values()) else "SEED42_FULL_OPTUNA_GATE_NOT_MET", "conditions": conditions}


def multiseed_decision(seed42: dict[str, Any], identity: dict[str, Any], metrics_by_seed: dict[int, dict[str, float]], *, cfg: dict[str, Any]) -> dict[str, Any]:
    if seed42.get("status") != PENDING:
        return {"status": seed42.get("status", TECHNICAL), "reason": "MULTISEED_NOT_AUTHORIZED_BY_SEED42"}
    if not identity.get("pass") or set(metrics_by_seed) != {42, 2024, 3407}:
        return {"status": TECHNICAL, "reason": "MULTISEED_IDENTITY_AUDIT_FAILURE"}
    if any(not _finite(v) or "b_rmse" not in v or not math.isfinite(float(v["b_rmse"])) for v in metrics_by_seed.values()):
        return {"status": TECHNICAL, "reason": "MULTISEED_METRIC_INVALID"}
    t = thresholds(cfg)
    c = [float(v["rmse"]) for v in metrics_by_seed.values()]
    b = [float(v["b_rmse"]) for v in metrics_by_seed.values()]
    f = [float(v["fold3_rmse"]) for v in metrics_by_seed.values()]
    conditions = {"each_seed_competitive": all(x <= t["multi_comp"] for x in c), "median_c_competitive": median(c) <= t["multi_comp"], "median_fold3": median(f) <= t["multi_fold3"], "paired_c_beats_b": sum(x < y for x, y in zip(c, b)) >= t["multi_min"], "median_c_beats_b": median(c) < median(b)}
    return {"status": ADVANCE if all(conditions.values()) else INSUFFICIENT, "reason": "MULTISEED_PASS" if all(conditions.values()) else "MULTISEED_GATE_NOT_MET", "conditions": conditions}

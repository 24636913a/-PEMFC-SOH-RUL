"""Pure, fail-closed T1-R decision state machine; no model or database I/O."""
from __future__ import annotations

from statistics import median
from typing import Any


TECHNICAL = "TFT_T1_TECHNICAL_FAILURE"
INSUFFICIENT = "TFT_OPTUNA_INSUFFICIENT_GAIN"
PENDING = "SEED42_GATE_PASS_PENDING_MULTISEED"
ADVANCE = "TFT_OPTUNA_ADVANCE"
EXTENDED = "TFT_EXTENDED_BUDGET_CONFIG_ADVANCE"


def _finite_metrics(metrics: dict[str, Any], keys: tuple[str, ...]) -> bool:
    try:
        return all(float(metrics[k]) == float(metrics[k]) and abs(float(metrics[k])) != float("inf") for k in keys)
    except (KeyError, TypeError, ValueError):
        return False


def check_trial_budget(attempted: int, completed: int, states: list[str] | None = None, minimum_completed: int = 12, budget: int = 16) -> dict[str, Any]:
    states = states or []
    if attempted != budget or completed < minimum_completed or completed > budget or len(states) not in {0, attempted}:
        return {"pass": False, "reason": "TRIAL_BUDGET_NOT_CLOSED"}
    if any(state in {"RUNNING", "WAITING"} for state in states):
        return {"pass": False, "reason": "UNRESOLVED_TRIAL_STATE"}
    return {"pass": True, "reason": "PASS"}


def seed42_decision(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any], *, audits_pass: bool) -> dict[str, Any]:
    keys = ("rmse", "fold3_rmse", "p99", "target_equal_rmse")
    if not audits_pass or not (_finite_metrics(a, keys) and _finite_metrics(b, keys) and _finite_metrics(c, keys)):
        return {"status": TECHNICAL, "reason": "AUDIT_OR_METRIC_FAILURE"}
    c_conditions = {
        "c_beats_b_strict": c["rmse"] < b["rmse"],
        "c_improves_a_15pct": c["rmse"] <= a["rmse"] * 0.85,
        "c_competitive": c["rmse"] <= 19.8892,
        "c_fold3": c["fold3_rmse"] <= 33.4534,
        "c_p99": c["p99"] <= b["p99"] * 1.10,
        "c_target_equal": c["target_equal_rmse"] <= b["target_equal_rmse"] * 1.10,
    }
    if all(c_conditions.values()):
        return {"status": PENDING, "reason": "ALL_SEED42_OPTUNA_CONDITIONS_PASS", "conditions": c_conditions}
    b_conditions = {
        "b_beats_a_15pct": b["rmse"] <= a["rmse"] * 0.85,
        "b_competitive": b["rmse"] <= 19.8892,
        "b_fold3": b["fold3_rmse"] <= 33.4534,
        "b_p99": b["p99"] <= a["p99"] * 1.10,
        "b_target_equal": b["target_equal_rmse"] <= a["target_equal_rmse"] * 1.10,
        "c_not_better_than_b": not c_conditions["c_beats_b_strict"],
    }
    if all(b_conditions.values()):
        return {"status": EXTENDED, "reason": "EXTENDED_BUDGET_CONFIGURATION_DESCRIPTIVE_ADVANCE", "conditions": {**c_conditions, **b_conditions}}
    return {"status": INSUFFICIENT, "reason": "SEED42_FULL_OPTUNA_GATE_NOT_MET", "conditions": {**c_conditions, **b_conditions}}


def multiseed_decision(seed42: dict[str, Any], pairs: list[dict[str, float]] | None, *, audits_pass: bool) -> dict[str, Any]:
    if seed42.get("status") != PENDING:
        return {"status": seed42.get("status", TECHNICAL), "reason": "MULTISEED_NOT_AUTHORIZED_BY_SEED42"}
    if not audits_pass:
        return {"status": TECHNICAL, "reason": "MULTISEED_AUDIT_FAILURE"}
    if pairs is None:
        return {"status": PENDING, "reason": "MULTISEED_REQUIRED_NOT_RUN"}
    if len(pairs) != 3 or any(not _finite_metrics(pair, ("b_rmse", "c_rmse", "c_fold3_rmse")) for pair in pairs):
        return {"status": TECHNICAL, "reason": "MULTISEED_INPUT_INVALID"}
    c_values = [float(pair["c_rmse"]) for pair in pairs]
    b_values = [float(pair["b_rmse"]) for pair in pairs]
    fold3 = [float(pair["c_fold3_rmse"]) for pair in pairs]
    conditions = {
        "each_seed_competitive": all(value <= 19.889180962041335 for value in c_values),
        "median_c_competitive": median(c_values) <= 19.889180962041335,
        "median_fold3": median(fold3) <= 33.4533564032138,
        "at_least_two_paired_c_beats_b": sum(c < b for c, b in zip(c_values, b_values)) >= 2,
        "median_c_beats_b_strict": median(c_values) < median(b_values),
    }
    return {"status": ADVANCE if all(conditions.values()) else INSUFFICIENT, "reason": "MULTISEED_PASS" if all(conditions.values()) else "MULTISEED_GATE_NOT_MET", "conditions": conditions}

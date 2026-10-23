"""Stage 0: test whether the single preregistered output physics loss is legal.

The audit deliberately distinguishes empirical support from information-set
availability.  Target-time actual operating conditions may be examined only
inside Fold-1 outer training to demonstrate what the candidate would require;
they are never accepted as model inputs or training pair selectors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pot_common import (
    POTFailure, build_frozen_folds, candidate_polarization_order_loss, finite_gradient_norm,
    load_inherited_protocols, load_protocol, safe_read_development, source_audit,
    write_csv, write_json, write_text,
)


EXTRA_COLUMNS = [
    "actual_cathode_inlet_temperature_C_mean",
    "actual_cathode_inlet_rh_pct_mean",
    "actual_cathode_inlet_pressure_kPag_mean",
]


def build_candidate_pairs(frame: pd.DataFrame, windows: object, candidate: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct deterministic Fold-1-train support pairs without a model.

    All values below are intentionally target-time *actual* values.  They
    quantify potential quasi-steady support but also establish why the loss is
    outside the forecasting information set.
    """
    cal = candidate["confounder_calipers"]
    rows: list[dict] = []
    for query_idx, origin in enumerate(windows.origin_row.tolist()):
        for h, target_row in enumerate(windows.target_row[query_idx].tolist(), start=1):
            target = frame.iloc[int(target_row)]
            rows.append({
                "query_index": query_idx,
                "origin_row": int(origin),
                "horizon_step": h,
                "target_row": int(target_row),
                "target_age_h": float(target["bin_end_ophrs2_h"]),
                "j_target_A_cm2": float(target["actual_current_density_A_cm2_mean"]),
                "temperature_target_C": float(target["actual_cathode_inlet_temperature_C_mean"]),
                "rh_target_pct": float(target["actual_cathode_inlet_rh_pct_mean"]),
                "pressure_target_kPag": float(target["actual_cathode_inlet_pressure_kPag_mean"]),
                "zero_current_fraction_target": float(target["zero_current_fraction"]),
                "flat_prediction_index": int(query_idx * 24 + (h - 1)),
            })
    queries = pd.DataFrame(rows)
    required = ["j_target_A_cm2", "temperature_target_C", "rh_target_pct", "pressure_target_kPag", "zero_current_fraction_target"]
    usable = queries.loc[np.isfinite(queries[required].to_numpy(float)).all(axis=1) & (queries["zero_current_fraction_target"] <= float(cal["target_zero_current_fraction_max"]))].copy()
    pairs: list[dict] = []
    summary: list[dict] = []
    for horizon, group in usable.groupby("horizon_step", sort=True):
        group = group.reset_index(drop=True)
        selected = 0
        high_count = 0
        for high_idx, high in group.iterrows():
            lower = group.loc[
                (high["j_target_A_cm2"] - group["j_target_A_cm2"] >= float(candidate["high_low_condition"].split(">=")[-1]))
                & (abs(high["temperature_target_C"] - group["temperature_target_C"]) <= float(cal["target_actual_cathode_inlet_temperature_C_abs_difference_max"]))
                & (abs(high["rh_target_pct"] - group["rh_target_pct"]) <= float(cal["target_actual_cathode_inlet_RH_pct_abs_difference_max"]))
                & (abs(high["pressure_target_kPag"] - group["pressure_target_kPag"]) <= float(cal["target_actual_cathode_inlet_pressure_kPag_abs_difference_max"]))
                & (abs(high["target_age_h"] - group["target_age_h"]) <= float(cal["target_age_h_abs_difference_max"]))
            ].copy()
            if lower.empty:
                continue
            high_count += 1
            # Deterministic nearest condition match; no voltage or model output enters ranking.
            lower["condition_distance"] = (
                abs(high["temperature_target_C"] - lower["temperature_target_C"]) / float(cal["target_actual_cathode_inlet_temperature_C_abs_difference_max"])
                + abs(high["rh_target_pct"] - lower["rh_target_pct"]) / float(cal["target_actual_cathode_inlet_RH_pct_abs_difference_max"])
                + abs(high["pressure_target_kPag"] - lower["pressure_target_kPag"]) / float(cal["target_actual_cathode_inlet_pressure_kPag_abs_difference_max"])
                + abs(high["target_age_h"] - lower["target_age_h"]) / float(cal["target_age_h_abs_difference_max"])
            )
            low = lower.sort_values(["condition_distance", "target_row", "origin_row"], kind="stable").iloc[0]
            pairs.append({
                "horizon_step": int(horizon),
                "high_origin_row": int(high.origin_row), "low_origin_row": int(low.origin_row),
                "high_target_row": int(high.target_row), "low_target_row": int(low.target_row),
                "high_prediction_index": int(high.flat_prediction_index), "low_prediction_index": int(low.flat_prediction_index),
                "j_difference_A_cm2": float(high.j_target_A_cm2 - low.j_target_A_cm2),
                "temperature_difference_C": float(abs(high.temperature_target_C - low.temperature_target_C)),
                "rh_difference_pct": float(abs(high.rh_target_pct - low.rh_target_pct)),
                "pressure_difference_kPag": float(abs(high.pressure_target_kPag - low.pressure_target_kPag)),
                "age_difference_h": float(abs(high.target_age_h - low.target_age_h)),
                "condition_distance": float(low.condition_distance),
            })
            selected += 1
        summary.append({"horizon_step": int(horizon), "usable_queries": int(len(group)), "queries_with_eligible_lower_j_match": int(high_count), "selected_pairs": int(selected)})
    return pd.DataFrame(pairs), pd.DataFrame(summary)


def synthetic_loss_contract() -> dict:
    predicted = torch.tensor([0.61, 0.73, 0.69], dtype=torch.float64, requires_grad=True)
    high = torch.tensor([1], dtype=torch.long)
    low = torch.tensor([0], dtype=torch.long)
    loss = candidate_polarization_order_loss(predicted, high, low)
    grad = finite_gradient_norm(loss, [predicted])
    zero = candidate_polarization_order_loss(predicted, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    if not torch.isfinite(loss) or not np.isfinite(grad) or grad <= 0 or float(zero) != 0.0:
        raise POTFailure("PHYSICS_LOSS_CONTRACT_FAIL")
    return {"differentiable": True, "finite_loss": True, "nonzero_gradient_on_violation": True, "synthetic_gradient_norm": grad, "zero_support_returns_zero": True}


def report(status: str, support: dict, loss_contract: dict) -> str:
    return f"""# Stage 0 — physical-loss feasibility

## Frozen candidate

The only candidate was `TARGET_FUTURE_ACTUAL_CONDITIONED_POLARIZATION_ORDERING`:
for two same-horizon forecast queries with matched target-time conditions and
`j_high-j_low >= 0.20 A cm^-2`, penalize
`ReLU(Vhat_high-j - Vhat_low-j)`.  It directly differentiates through predicted
absolute voltage and the synthetic gradient contract passed.

## Information-set finding

The candidate needs target-time **actual** current density, temperature,
relative humidity, and pressure to select pairs.  These variables are labelled
`future-unknown actual` by the frozen N1 dictionary.  They are unavailable at a
prediction origin and are not authorized as inference inputs or training-pair
selectors.  Therefore empirical Fold-1 outer-train pair support cannot repair
the information-set failure.

## Decision

**{status}**.  No Physics-TimesNet branch is trained, and the protocol forbids
substituting an easier regularizer.  Broad voltage bounds and input identities
such as `J=I/A` are not adopted: the former is not a condition-specific
forecasting law and the latter does not constrain TimesNet voltage output.

## Audit facts

- Fold-1 outer-train candidate pairs: `{support['pair_count']}`.
- Active-pair fraction of candidate queries: `{support['active_pair_fraction']:.6f}`.
- Minimum pair support requirement: `{support['minimum_pair_support']}`.
- Future-condition information gate: **FAIL**.
- Direct differentiability/finite-gradient contract: **{loss_contract['differentiable']}**.

This is a forecasting feasibility decision only.  It does not make an SOH,
irreversible-degradation, EOL, or RUL claim.
"""


def main() -> None:
    cfg, protocol_sha = load_protocol()
    bs_cfg, _df_cfg, common = load_inherited_protocols(cfg)
    source = source_audit(common, bs_cfg)
    frame, firewall = safe_read_development(cfg, bs_cfg, EXTRA_COLUMNS)
    folds = build_frozen_folds(frame, cfg, bs_cfg, common)
    fold1_train = next(train for fold, train, _validation in folds if fold == "fold_1")
    candidate = cfg["stage0_physics_candidate"]
    pairs, per_horizon = build_candidate_pairs(frame, fold1_train, candidate)
    query_count = len(fold1_train.origin_row) * 24
    support = {
        "pair_count": int(len(pairs)),
        "candidate_query_count": int(query_count),
        "active_pair_fraction": float(len(pairs) / query_count),
        "minimum_pair_support": int(candidate["minimum_support"]["fold1_outer_train_candidate_pairs"]),
        "minimum_active_pair_fraction": float(candidate["minimum_support"]["active_pair_fraction_of_queries"]),
    }
    support["empirical_support_pass"] = bool(support["pair_count"] >= support["minimum_pair_support"] and support["active_pair_fraction"] >= support["minimum_active_pair_fraction"])
    loss_contract = synthetic_loss_contract()
    information = {
        "target_actual_current_density_origin_available": False,
        "target_actual_temperature_origin_available": False,
        "target_actual_relative_humidity_origin_available": False,
        "target_actual_pressure_origin_available": False,
        "future_actual_conditions_used_as_model_inputs": False,
        "future_actual_conditions_allowed_as_training_pair_selectors": False,
        "information_gate_pass": False,
        "reason": "FUTURE_ACTUAL_CONDITION_INFORMATION_UNAVAILABLE_AT_INFERENCE",
    }
    status = cfg["stage0_physics_candidate"]["fail_status"]
    decision = {
        "status": status,
        "protocol_sha256": protocol_sha,
        "candidate": candidate["name"],
        "reason": information["reason"],
        "support": support,
        "loss_contract": loss_contract,
        "information_gate": information,
        "physics_training_authorized": False,
        "alternative_constraint_substituted": False,
        "current_phase_test_not_accessed": True,
        "historical_test_contact_exists": True,
        "model_training_performed": False,
    }
    write_csv(source, "outputs", "TIMESNET_SOURCE_AUDIT.csv")
    write_csv(per_horizon, "outputs", "STAGE0_PHYSICS_PAIR_SUPPORT_BY_HORIZON.csv")
    write_csv(pairs, "outputs", "STAGE0_PHYSICS_CANDIDATE_PAIRS.csv")
    write_json({"data_firewall": firewall, "current_phase_test_not_accessed": True, "historical_test_contact_exists": True}, "outputs", "STAGE0_DATA_ACCESS_AUDIT.json")
    write_json(decision, "outputs", "STAGE0_PHYSICAL_FEASIBILITY_SUMMARY.json")
    text = report(status, support, loss_contract)
    write_text(text, "reports", "STAGE0_PHYSICAL_FEASIBILITY_REPORT.md")
    write_text(text, "reports", "01_STAGE0_PHYSICS_FEASIBILITY.md")
    print(json.dumps({"status": status, "pair_count": support["pair_count"], "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

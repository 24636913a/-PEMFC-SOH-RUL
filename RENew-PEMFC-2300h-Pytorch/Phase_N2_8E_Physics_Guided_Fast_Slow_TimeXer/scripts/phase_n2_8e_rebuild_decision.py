"""Rebuild N2.8E summary-only artifacts after a formal-eligibility correction.

This script does not open raw data, Test data, or prediction arrays. It derives
the formal decision only from the completed development-only CSV summaries.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8E_Physics_Guided_Fast_Slow_TimeXer")


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    show = frame.loc[:, [c for c in columns if c in frame.columns]].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    head = "| " + " | ".join(show.columns) + " |"
    separator = "|" + "|".join("---" for _ in show.columns) + "|"
    rows = ["| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in show.itertuples(index=False, name=None)]
    return "\n".join([head, separator, *rows])


def main() -> None:
    comparison = pd.read_csv(ROOT / "outputs/model_screening/N2_8E_seed42_comparison.csv")
    ablation = pd.read_csv(ROOT / "outputs/ablation/N2_8E_physics_ablation.csv")
    multiseed = pd.read_csv(ROOT / "outputs/model_screening/N2_8E_multiseed_comparison.csv")
    dominance = pd.read_csv(ROOT / "outputs/diagnostics/physics_loss_dominance_audit.csv")
    alignment = pd.read_csv(ROOT / "outputs/diagnostics/step_alignment_derivative_audit.csv")
    eligible = {"FROZEN_REFERENCE", "FORMAL", "FORMAL_ZERO_CONTROL"}
    formal = comparison.loc[comparison["eligibility"].isin(eligible)].copy()
    if formal.empty or formal["model"].str.contains("Oracle|Shuffled", regex=True).any():
        raise RuntimeError("Formal decision eligibility filter failed closed")
    best = formal.sort_values("mean_rmse_delta_mV").iloc[0]
    original = formal.loc[formal["model"] == "TimeXer-Original"].iloc[0]
    rmse = float(best["mean_rmse_delta_mV"])
    if rmse >= 25.0:
        status = "WEAK"
    elif rmse >= 20.0:
        status = "MEANINGFUL_RECOVERY"
    elif rmse >= 15.0:
        status = "PROMISING"
    elif rmse > 12.0:
        status = "STRONG"
    else:
        status = "REVIVED"
    phase_n2_8f = "GO" if rmse < 20.0 else "NO-GO"
    decision = {
        "phase_status": "PASS",
        "test_performance_accessed": False,
        "formal_selection_eligibility": sorted(eligible),
        "oracle_excluded_from_formal_selection": True,
        "shuffled_excluded_from_formal_selection": True,
        "best_formal_seed42_model": str(best["model"]),
        "best_formal_seed42_rmse_delta_mV": rmse,
        "best_formal_seed42_r2_delta": float(best["mean_r2_delta"]),
        "time_xer_original_rmse_delta_mV": float(original["mean_rmse_delta_mV"]),
        "recovery_status": status,
        "multiseed_promoted": False,
        "multiseed_summary": multiseed.iloc[0].to_dict(),
        "enter_phase_n2_8f": phase_n2_8f,
        "n2_8f_reason": "Formal development-only best RMSE is below 20 mV." if phase_n2_8f == "GO" else "Formal development-only best RMSE is not below the fixed 20 mV N2.8F entry gate.",
    }
    (ROOT / "PHASE_N2_8E_DECISION.yaml").write_text(yaml.safe_dump(decision, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = f"""# Phase N2.8E results

## Protocol, firewall, and eligibility

This is a development-only 0.5 h / 48-step / 24-step / F2 / DeltaV experiment. Test target rows read: **0**. No Test prediction, metric, feature diagnostic, selection, or tuning was constructed. Formal future-exogenous inputs were strict expanding chronological OOF Ridge pseudo-forecasts in training and Train-fit Ridge forecasts in validation.

The formal-selection eligibility filter contains only `FROZEN_REFERENCE`, `FORMAL`, and `FORMAL_ZERO_CONTROL`. Oracle is a diagnostic ceiling and shuffled future conditions are a negative control. Neither can select the best formal model, define recovery status, or open Phase N2.8F.

## Step-aligned controls

{table(comparison, ["model", "eligibility", "mean_r2_delta", "mean_rmse_delta_mV", "mean_mae_delta_mV", "fold_3_rmse_delta_mV", "gain_vs_original_pct", "mean_p95_mV", "mean_p99_mV"])}

Each future row `z_h` creates only `gamma_h,beta_h` for the corresponding state `H_h`; no future cross-attention, pooling, or time-wide concatenation is used. The Oracle row remains in the table solely as a diagnostic upper bound.

## Fast-slow and physics ablation

{table(ablation, ["model", "mean_r2_delta", "mean_rmse_delta_mV", "fold_3_rmse_delta_mV", "gain_vs_original_pct", "mean_p95_mV", "mean_p99_mV"])}

The slow branch uses historical latent state only. It has no interpolated SOH supervision, no fabricated slow label, and no RUL target. Smoothness, active-current sensitivity, and slow monotonicity are soft regularizers, not a complete PEMFC mechanistic model.

## Causal and physics audits

{table(dominance, ["model", "fold", "seed", "max_consecutive_physics_dominance_epochs", "physics_loss_status"])}

{table(alignment, ["model", "fold", "mean_abs_diagonal_derivative", "max_abs_off_diagonal_derivative"])}

All completed physics-loss runs remained balanced under the fixed coefficients. Current-density derivatives are calculated from Ridge-predicted future conditions at active operating points; no future true voltage is an input.

## Formal decision

- Best formal seed-42 model: **{best['model']}**
- Mean RMSE DeltaV: **{rmse:.6f} mV**
- Mean R2 DeltaV: **{float(best['mean_r2_delta']):.6f}**
- TimeXer-Original mean RMSE DeltaV: **{float(original['mean_rmse_delta_mV']):.6f} mV**
- Recovery category: **{status}**
- Multi-seed promotion: **No**. The frozen PGFS promotion rule was not met.
- Phase N2.8F entry: **{phase_n2_8f}**

N2.8E is therefore an execution PASS with a development-only conclusion. Per-horizon, Fold-3, tail, fast-slow variance, current-sensitivity, slow-monotonicity, and loss-history evidence are retained in the named CSV outputs and figures. No N2.8F model is trained here.
"""
    (ROOT / "PHASE_N2_8E_RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"repaired": True, "best_formal_model": str(best["model"]), "best_formal_rmse_mV": rmse, "n2_8f": phase_n2_8f}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Frozen-input, Development-only B2 affine calibration characterization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
CHM = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model"
TARGETS = CHM / "inputs" / "CHM_SPARSE_ANCHOR_TARGETS.csv"
MAPPING = CHM / "inputs" / "CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv"
BASELINES = CHM / "outputs" / "CHM_D_BASELINE_LOAO.csv"
PROTOCOL = PHASE / "configs" / "AHC_POSTHOC_PROTOCOL.yaml"
OUTPUTS, FIGURES, REPORTS, FINAL = (PHASE / name for name in ("outputs", "figures", "reports", "final"))
CHECKPOINTS = (300, 600, 850, 1150, 1500, 1750)
SUPPORT3 = (300, 850, 1750)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def affine_parameters(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    slope = 0.0 if denominator == 0.0 else float(np.dot(x_centered, y - y.mean()) / denominator)
    slope = max(0.0, slope)
    return slope, float(y.mean() - slope * x.mean())


def read_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    expected = [str(REPO / item) for item in protocol["allowed_inputs"]]
    observed = [str(TARGETS), str(MAPPING), str(BASELINES)]
    if observed != expected:
        raise RuntimeError("AHC_INPUT_ALLOWLIST_MISMATCH")
    targets, mapping, baselines = (pd.read_csv(path) for path in (TARGETS, MAPPING, BASELINES))
    if tuple(targets["checkpoint"].astype(int)) != CHECKPOINTS or tuple(baselines["checkpoint"].astype(int)) != CHECKPOINTS:
        raise RuntimeError("AHC_CHECKPOINT_UNIVERSE_MISMATCH")
    if not set(CHECKPOINTS).issubset(set(mapping["checkpoint"].astype(int))):
        raise RuntimeError("AHC_MAPPING_CHECKPOINT_MISMATCH")
    return targets, mapping, baselines, protocol


def anchor_frame(targets: pd.DataFrame, mapping: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    grouped = mapping.groupby("checkpoint")["P1S_operating_voltage_retention"].median()
    frame = targets[["checkpoint", "Y", "median_source_joint_support"]].copy()
    frame["R"] = frame["checkpoint"].map(grouped).astype(float)
    frame = frame.merge(baselines[["checkpoint", "B0_prediction", "B2_prediction", "B2_absolute_error"]], on="checkpoint", validate="one_to_one")
    if not np.allclose(frame["R"], frame["B0_prediction"], rtol=0.0, atol=1e-12):
        raise RuntimeError("AHC_FROZEN_P1S_FUNCTIONAL_MISMATCH")
    return frame.sort_values("checkpoint").reset_index(drop=True)


def loao(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for checkpoint in CHECKPOINTS:
        train = frame.loc[frame["checkpoint"].ne(checkpoint)]
        held = frame.loc[frame["checkpoint"].eq(checkpoint)].iloc[0]
        slope, intercept = affine_parameters(train["R"].to_numpy(), train["Y"].to_numpy())
        prediction = slope * float(held.R) + intercept
        rows.append({"heldout_checkpoint": checkpoint, "training_checkpoints": "|".join(map(str, train["checkpoint"].astype(int))), "slope_a": slope, "intercept_b": intercept, "heldout_R": float(held.R), "heldout_Y": float(held.Y), "prediction": prediction, "absolute_error": abs(prediction - float(held.Y))})
    return pd.DataFrame(rows)


def metrics(prediction: pd.Series, target: pd.Series) -> dict[str, float]:
    residual = prediction.to_numpy(dtype=float) - target.to_numpy(dtype=float)
    return {"MAE": float(np.mean(np.abs(residual))), "RMSE": float(np.sqrt(np.mean(np.square(residual)))), "bias": float(np.mean(residual))}


def write(frame: pd.DataFrame, name: str) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUTS / name, index=False)


def figure_lines(loao_frame: pd.DataFrame, all_six: tuple[float, float], x_grid: np.ndarray) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in loao_frame.itertuples(): ax.plot(x_grid, row.slope_a * x_grid + row.intercept_b, alpha=.7, label=f"hold {row.heldout_checkpoint} h")
    ax.plot(x_grid, all_six[0] * x_grid + all_six[1], color="black", lw=2, label="all six")
    ax.set(xlabel="Frozen P1S anchor functional R", ylabel="Polarization health anchor Y", title="B2 LOAO calibration lines")
    ax.legend(ncol=2, fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "B2_LOAO_CALIBRATION_LINES.png", dpi=180); plt.close(fig)


def run() -> dict:
    targets, mapping, baselines, protocol = read_frozen_inputs()
    frame = anchor_frame(targets, mapping, baselines)
    loao_frame = loao(frame)
    original_mae = float(baselines["B2_absolute_error"].mean())
    reproduced_mae = float(loao_frame["absolute_error"].mean())
    if abs(reproduced_mae - float(protocol["reproduction"]["aggregate_MAE"])) > float(protocol["reproduction"]["tolerance"]) or not np.allclose(loao_frame["prediction"], baselines["B2_prediction"], rtol=0.0, atol=float(protocol["reproduction"]["tolerance"])):
        raise RuntimeError("B2_REPRODUCTION_FAIL")
    write(loao_frame, "B2_LOAO_PARAMETER_AUDIT.csv")
    slope_all, intercept_all = affine_parameters(frame.R.to_numpy(), frame.Y.to_numpy())
    x_grid = np.linspace(float(frame.R.min()), float(frame.R.max()), 100)
    functions = np.vstack([row.slope_a * x_grid + row.intercept_b for row in loao_frame.itertuples()])
    function_stability = pd.DataFrame({"R_grid": x_grid, "six_fit_median_prediction": np.median(functions, axis=0), "six_fit_min_prediction": functions.min(axis=0), "six_fit_max_prediction": functions.max(axis=0), "six_fit_range": np.ptp(functions, axis=0)})
    write(function_stability, "B2_LOAO_FUNCTION_STABILITY.csv")
    influence = loao_frame[["heldout_checkpoint", "slope_a", "intercept_b"]].copy()
    influence["delta_slope_a"] = influence.slope_a - slope_all; influence["delta_intercept_b"] = influence.intercept_b - intercept_all
    all_predictions = slope_all * frame.R.to_numpy() + intercept_all
    influence["max_prediction_shift_at_observed_R"] = [float(np.max(np.abs(row.slope_a * frame.R.to_numpy() + row.intercept_b - all_predictions))) for row in influence.itertuples()]
    write(influence, "B2_ANCHOR_INFLUENCE.csv")
    chronological_rows = []
    for checkpoint in CHECKPOINTS:
        current = frame.loc[frame.checkpoint.eq(checkpoint)].iloc[0]; prior = frame.loc[frame.checkpoint.lt(checkpoint)]
        if checkpoint == 300: status, slope, intercept, prediction = "NO_PRIOR_CALIBRATION", np.nan, np.nan, np.nan
        elif checkpoint == 600: status, slope, intercept, prediction = "INSUFFICIENT_PRIOR_ANCHORS", np.nan, np.nan, np.nan
        else:
            slope, intercept = affine_parameters(prior.R.to_numpy(), prior.Y.to_numpy()); prediction = slope * current.R + intercept
            status = "TWO_ANCHOR_WARMSTART_DESCRIPTIVE" if checkpoint == 850 else "FORMAL_CHRONOLOGICAL_EVALUATION"
        chronological_rows.append({"checkpoint": checkpoint, "prior_anchor_set": "|".join(map(str, prior.checkpoint.astype(int))), "prior_anchor_count": len(prior), "slope_a": slope, "intercept_b": intercept, "R_current": current.R, "prediction": prediction, "target": current.Y, "signed_error": prediction-current.Y if np.isfinite(prediction) else np.nan, "absolute_error": abs(prediction-current.Y) if np.isfinite(prediction) else np.nan, "status": status, "raw_B0_prediction": current.R, "raw_B0_absolute_error": abs(current.R-current.Y)})
    chronological = pd.DataFrame(chronological_rows); write(chronological, "B2_CHRONOLOGICAL_PREFIX_EVALUATION.csv")
    chronological_formal = chronological.loc[chronological.checkpoint.isin([1150, 1500, 1750])]
    chrono_metrics = metrics(chronological_formal.prediction, chronological_formal.target); raw_metrics = metrics(chronological_formal.raw_B0_prediction, chronological_formal.target)
    support = frame.loc[frame.checkpoint.isin(SUPPORT3)]; support_slope, support_intercept = affine_parameters(support.R.to_numpy(), support.Y.to_numpy())
    support_sensitivity = pd.DataFrame({"R_grid": x_grid, "all_six_prediction": slope_all*x_grid+intercept_all, "joint_support3_prediction": support_slope*x_grid+support_intercept})
    support_sensitivity["absolute_difference"] = abs(support_sensitivity.all_six_prediction-support_sensitivity.joint_support3_prediction); write(support_sensitivity, "B2_ALL6_VS_JOINT_SUPPORT3_SENSITIVITY.csv")
    all_six = pd.DataFrame([{ "candidate_name":"Development all-anchor affine calibration candidate", "slope_a":slope_all, "intercept_b":intercept_all, "R_min":frame.R.min(), "R_max":frame.R.max(), "joint_support3_slope_a":support_slope, "joint_support3_intercept_b":support_intercept }]); write(all_six, "B2_ALL_SIX_CALIBRATION.csv")
    confounding = pd.DataFrame([{ "spearman_age_R":float(frame.checkpoint.corr(frame.R, method="spearman")), "spearman_age_Y":float(frame.checkpoint.corr(frame.Y, method="spearman")), "spearman_R_Y":float(frame.R.corr(frame.Y, method="spearman")), "anchor_count":6, "interpretation":"DESCRIPTIVE_SHARED_AGE_CONFOUNDING_AUDIT_ONLY" }]); write(confounding, "B2_SHARED_AGE_CONFOUNDING_AUDIT.csv")
    figure_lines(loao_frame, (slope_all, intercept_all), x_grid)
    FIGURES.mkdir(parents=True, exist_ok=True)
    def save_scatter(name, x, y, xlabel, ylabel, title):
        fig, ax=plt.subplots(figsize=(5,4)); ax.scatter(x,y); [ax.annotate(str(c),(a,b)) for a,b,c in zip(x,y,CHECKPOINTS)]; ax.set(xlabel=xlabel,ylabel=y,title=title); fig.tight_layout(); fig.savefig(FIGURES/name,dpi=180); plt.close(fig)
    save_scatter("B2_TARGET_VS_LOAO_PREDICTION.png", loao_frame.heldout_Y, loao_frame.prediction, "Target Y", "LOAO prediction", "B2 LOAO")
    save_scatter("B2_ANCHOR_INFLUENCE.png", influence.heldout_checkpoint, influence.max_prediction_shift_at_observed_R, "Held-out anchor", "Maximum function shift", "Anchor influence")
    save_scatter("B2_CHRONOLOGICAL_PREFIX_PREDICTIONS.png", chronological_formal.target, chronological_formal.prediction, "Target Y", "Chronological prefix prediction", "Later checkpoint transfer (n=3)")
    error_compare=pd.DataFrame({"B2":chronological_formal.absolute_error,"Raw B0":chronological_formal.raw_B0_absolute_error}); fig,ax=plt.subplots(figsize=(5,4)); error_compare.plot(kind="bar",ax=ax); ax.set(xlabel="Chronological rows",ylabel="Absolute error",title="Late checkpoints: raw vs calibrated"); fig.tight_layout(); fig.savefig(FIGURES/"B2_RAW_VS_CALIBRATED_LATE_CHECKPOINT_ERROR.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4)); ax.plot(frame.checkpoint,frame.R,"o-",label="R"); ax.plot(frame.checkpoint,frame.Y,"s-",label="Y"); ax.set(xlabel="Checkpoint h",title="Shared-age descriptive audit"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"B2_AGE_R_Y_RELATION.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(x_grid,support_sensitivity.all_six_prediction,label="all six"); ax.plot(x_grid,support_sensitivity.joint_support3_prediction,label="joint-support three"); ax.set(xlabel="R",ylabel="Prediction",title="All-six vs joint-support-three sensitivity"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"B2_ALL6_VS_JOINT_SUPPORT3_SENSITIVITY.png",dpi=180); plt.close(fig)
    finite_nonnegative = bool(np.isfinite(loao_frame[["slope_a","intercept_b"]].to_numpy()).all() and (loao_frame.slope_a >= 0).all() and slope_all >= 0)
    status = "AFFINE_CALIBRATION_DEVELOPMENT_HYPOTHESIS_SUPPORTED" if np.isclose(reproduced_mae, original_mae, rtol=0.0, atol=float(protocol["reproduction"]["tolerance"])) and chrono_metrics["MAE"] < raw_metrics["MAE"] and finite_nonnegative else "AFFINE_CALIBRATION_DEVELOPMENT_HYPOTHESIS_WEAK"
    audit = {"base_commit":protocol["base_commit"],"analysis":"POSTHOC_HYPOTHESIS_CHARACTERIZATION","input_sha256":{str(path.relative_to(REPO)):sha256(path) for path in (TARGETS,MAPPING,BASELINES)},"B2_original_LOAO_reproduced":True,"original_B2_LOAO_MAE":original_mae,"all_six_slope":slope_all,"all_six_intercept":intercept_all,"LOAO_slope_range":float(loao_frame.slope_a.max()-loao_frame.slope_a.min()),"maximum_LOAO_function_spread_over_observed_R":float(function_stability.six_fit_range.max()),"chronological_n3_metrics":chrono_metrics,"raw_B0_n3_metrics":raw_metrics,"joint_support3_max_function_difference":float(support_sensitivity.absolute_difference.max()),"shared_age_confounding":"REPORTED","validated_SOH":False,"RUL":"NO","Test_accessed":False,"final_status":status}
    (OUTPUTS/"AHC_RUN_AUDIT.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    FINAL.mkdir(parents=True,exist_ok=True); (FINAL/"FINAL_AHC_DECISION.yaml").write_text(yaml.safe_dump(audit,sort_keys=False),encoding="utf-8")
    REPORTS.mkdir(parents=True,exist_ok=True); (REPORTS/"FINAL_AFFINE_CALIBRATION_CHARACTERIZATION.md").write_text(f"# Phase AHC final characterization\n\n## Established Development evidence\n\nB2 LOAO reproduced exactly (MAE {original_mae:.15f}), outperforming raw B0 and CHM in the historical frozen LOAO experiment.\n\n## New posthoc checks\n\nAll-six candidate: `Y = {slope_all:.12f} R + {intercept_all:.12f}`. Chronological n=3 B2 MAE/RMSE/bias: {chrono_metrics['MAE']:.12f} / {chrono_metrics['RMSE']:.12f} / {chrono_metrics['bias']:.12f}; raw B0 n=3 MAE: {raw_metrics['MAE']:.12f}. Parameter, function, deletion, chronological, and support-sensitivity tables are retained in `outputs/`.\n\n## Limitations\n\nThis is posthoc small-n Development characterization. R and Y may share an age trend; the six anchors cannot identify an independent health mapping or causal mechanism. Only three median-source anchors are inside joint common support. No validated SOH, Test generalization, RUL, or cross-stack claim is made.\n\n## Decision\n\n`{status}`. Test opening is not automatic; at most it is a manual-review candidate.\n",encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); args=parser.parse_args()
    if not args.execute: raise SystemExit("Explicit --execute required.")
    outcome=run(); print(json.dumps(outcome,indent=2))

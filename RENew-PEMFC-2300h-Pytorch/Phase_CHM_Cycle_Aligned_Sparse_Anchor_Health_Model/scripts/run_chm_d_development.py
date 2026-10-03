"""Execute the frozen Development-only CHM-D experiment once authority passes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))

from chm_baselines import baseline_metrics, run_baseline_loao
from chm_bootstrap import bootstrap_weights, optimizer_seed, seed_tuple
from chm_data import ANCHORS, CONDITION_COLUMNS, CHMRuntimeFailure, FrozenCHMData, load_frozen_chm_data, target_for_checkpoint, verify_runtime_authority
from chm_evaluation import CHMFitFailure, PROFILE_ORDER, SEEDS, aggregate_metrics, fit_chm, fit_outer_prediction, outer_training_anchors, select_profile_for_outer_fold


OUTPUTS = PHASE / "outputs"
FIGURES = PHASE / "figures"
REPORTS = PHASE / "reports"
FINAL = PHASE / "final"
BASE_HEAD = "a3e56dd5bc55a6f7ab4b4aa373adbb190714be73"


def repo_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PHASE.parents[1], text=True).strip()


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / name
    frame.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, Any], name: str) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def reference_scaler_audit(data: FrozenCHMData) -> dict[str, Any]:
    return {
        "reference_population": "earliest_10_legal_complete_cycles",
        "reference_complete_hou_ids": data.hou.loc[data.hou["hou_kind"].eq("COMPLETE_CYCLE_HOU"), "hou_id"].head(10).tolist(),
        "condition_median": data.scaler.median,
        "condition_scale_IQR_or_std": data.scaler.scale,
        "constant_reference_features_excluded_from_network": list(data.scaler.constant_features),
        "model_input_feature_names": list(data.scaler.input_feature_names),
        "forbidden_age_or_time_inputs_used": False,
    }


def outer_loao(data: FrozenCHMData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Any], dict[int, str | None]]:
    inner_records: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    outer_fits: list[Any] = []
    selected: dict[int, str | None] = {}
    for heldout in ANCHORS:
        profile, records, scores = select_profile_for_outer_fold(data, heldout)
        inner_records.extend(records)
        selected[heldout] = profile
        selection_rows.append({"outer_heldout_checkpoint": heldout, "A_BALANCED_score": scores["A_BALANCED"], "B_ANCHOR_EMPHASIS_score": scores["B_ANCHOR_EMPHASIS"], "C_OBSERVATION_EMPHASIS_score": scores["C_OBSERVATION_EMPHASIS"], "selected_profile": profile if profile else "OUTER_FOLD_PROFILE_SELECTION_INVALID"})
        if profile is None:
            prediction_rows.append({"checkpoint": heldout, "target": target_for_checkpoint(data, heldout), "pred_seed42": np.nan, "pred_seed2024": np.nan, "pred_seed3407": np.nan, "official_prediction_three_seed_median": np.nan, "signed_error": np.nan, "absolute_error": np.nan, "squared_error": np.nan, "selected_profile": "OUTER_FOLD_PROFILE_SELECTION_INVALID", "legal_prediction": False})
            continue
        try:
            fits, row = fit_outer_prediction(data, heldout, profile)
            outer_fits.extend([(heldout, fit) for fit in fits])
            prediction_rows.append(row)
        except CHMFitFailure as exc:
            prediction_rows.append({"checkpoint": heldout, "target": target_for_checkpoint(data, heldout), "pred_seed42": np.nan, "pred_seed2024": np.nan, "pred_seed3407": np.nan, "official_prediction_three_seed_median": np.nan, "signed_error": np.nan, "absolute_error": np.nan, "squared_error": np.nan, "selected_profile": profile, "legal_prediction": False, "illegal_reason": f"ILLEGAL_OUTER_PREDICTION:{exc}"})
    return pd.DataFrame(inner_records), pd.DataFrame(selection_rows), pd.DataFrame(prediction_rows), outer_fits, selected


def monotonicity_audit(data: FrozenCHMData, outer_fits: list[Any], final_fits: list[Any] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = data.hou["hou_kind"].eq("COMPLETE_CYCLE_HOU")
    for heldout, fit in outer_fits:
        rows.append({"fit_scope": "outer_LOAO", "outer_heldout_checkpoint": heldout, "seed": fit.seed, "profile": fit.profile, "complete_cycle_latent_increase_count": fit.monotonic_increase_count, "monotonic_pass": fit.monotonic_increase_count == 0})
    if final_fits:
        for fit in final_fits:
            states = fit.states[complete.to_numpy()]
            rows.append({"fit_scope": "final_all_six", "outer_heldout_checkpoint": np.nan, "seed": fit.seed, "profile": fit.profile, "complete_cycle_latent_increase_count": int(np.count_nonzero(np.diff(states) > 0.0)), "monotonic_pass": bool(np.all(np.diff(states) <= 0.0))})
    return pd.DataFrame(rows)


def final_profile(selected: dict[int, str | None]) -> tuple[str | None, dict[str, int]]:
    labels = list(selected.values())
    if len(labels) != 6 or any(label not in PROFILE_ORDER for label in labels):
        return None, {profile: 0 for profile in PROFILE_ORDER}
    counts = {profile: labels.count(profile) for profile in PROFILE_ORDER}
    highest = max(counts.values())
    return next(profile for profile in PROFILE_ORDER if counts[profile] == highest), counts


def run_outer_bootstrap(data: FrozenCHMData, selected: dict[int, str | None]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    interval_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for fold_index, heldout in enumerate(ANCHORS):
        profile = selected[heldout]
        if profile is None:
            continue
        training = outer_training_anchors(heldout)
        predictions: list[float] = []
        for replicate in range(200):
            stream = 101
            try:
                weights, multiplicity = bootstrap_weights(data, 4, stream, fold_index, replicate)
                fit = fit_chm(data, training, profile, optimizer_seed(replicate), weights, multiplicity)
                prediction = fit.anchor_predictions[heldout]
                if not np.isfinite(prediction):
                    raise CHMFitFailure("nonfinite_anchor_prediction")
                predictions.append(prediction)
                replicate_rows.append({"scope": "outer_LOAO_block4", "outer_heldout_checkpoint": heldout, "replicate_index": replicate, "legal": True, "prediction": prediction})
            except Exception as exc:
                failures.append({"scope": "outer_LOAO_block4", "outer_heldout_checkpoint": heldout, "replicate_index": replicate, "RNG_seed_tuple": str(seed_tuple(stream, fold_index, replicate)), "optimizer_seed": optimizer_seed(replicate), "failure_reason": str(exc)})
                replicate_rows.append({"scope": "outer_LOAO_block4", "outer_heldout_checkpoint": heldout, "replicate_index": replicate, "legal": False, "prediction": np.nan})
        if predictions:
            p2, p50, p97 = np.percentile(predictions, [2.5, 50, 97.5]).tolist()
        else:
            p2 = p50 = p97 = np.nan
        interval_rows.append({"checkpoint": heldout, "legal_replicates": len(predictions), "scheduled_replicates": 200, "minimum_legal_replicates": 190, "P2_5": p2, "P50": p50, "P97_5": p97, "target": target_for_checkpoint(data, heldout), "coverage": bool(p2 <= target_for_checkpoint(data, heldout) <= p97) if len(predictions) >= 190 else False})
    return pd.DataFrame(interval_rows), pd.DataFrame(replicate_rows), failures


def run_final_bootstrap(data: FrozenCHMData, profile: str, block_length: int, scheduled: int, stream: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    trajectories: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    for replicate in range(scheduled):
        try:
            weights, multiplicity = bootstrap_weights(data, block_length, stream, 0, replicate)
            fit = fit_chm(data, ANCHORS, profile, optimizer_seed(replicate), weights, multiplicity)
            trajectories.append(fit.states)
        except Exception as exc:
            failures.append({"scope": f"final_all_six_block{block_length}", "outer_heldout_checkpoint": np.nan, "replicate_index": replicate, "RNG_seed_tuple": str(seed_tuple(stream, 0, replicate)), "optimizer_seed": optimizer_seed(replicate), "failure_reason": str(exc)})
    if not trajectories:
        return pd.DataFrame(), failures
    array = np.vstack(trajectories)
    return pd.DataFrame({"hou_id": data.hou["hou_id"], "P2_5": np.percentile(array, 2.5, axis=0), "P50": np.percentile(array, 50, axis=0), "P97_5": np.percentile(array, 97.5, axis=0), "legal_replicates": len(trajectories), "block_length": block_length}), failures


def prefix_select_and_fit(data: FrozenCHMData, checkpoint: int) -> dict[str, Any]:
    prior = tuple(anchor for anchor in ANCHORS if anchor < checkpoint)
    if checkpoint in (300, 600, 850):
        status = "COLD_START_NO_PRIOR_NONBOL_ANCHOR" if checkpoint == 300 else "INSUFFICIENT_PRIOR_ANCHORS_FOR_PREFIX_MODEL_SELECTION"
        return {"checkpoint": checkpoint, "prior_anchor_set": "|".join(map(str, prior)), "status": status, "prediction": np.nan, "target": target_for_checkpoint(data, checkpoint), "absolute_error": np.nan}
    scores: dict[str, list[float]] = {profile: [] for profile in PROFILE_ORDER}
    for profile in PROFILE_ORDER:
        for inner_heldout in prior:
            inner_train = tuple(anchor for anchor in prior if anchor != inner_heldout)
            for seed in SEEDS:
                fit = fit_chm(data, inner_train, profile, seed)
                scores[profile].append(abs(fit.anchor_predictions[inner_heldout] - target_for_checkpoint(data, inner_heldout)))
    means = {profile: float(np.mean(values)) for profile, values in scores.items()}
    lowest = min(means.values())
    profile = next(item for item in PROFILE_ORDER if means[item] <= lowest + 1e-8)
    fits = [fit_chm(data, prior, profile, seed) for seed in SEEDS]
    prediction = float(np.median([fit.anchor_predictions[checkpoint] for fit in fits]))
    target = target_for_checkpoint(data, checkpoint)
    return {"checkpoint": checkpoint, "prior_anchor_set": "|".join(map(str, prior)), "status": "LEGAL_PREFIX_FIT", "selected_profile_from_prior_anchors_only": profile, "prediction": prediction, "target": target, "absolute_error": abs(prediction-target)}


def plot_results(loao: pd.DataFrame, baselines: pd.DataFrame, final_trajectory: pd.DataFrame | None, uncertainty: pd.DataFrame | None, data: FrozenCHMData, s1_trajectory: pd.DataFrame | None, online: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    legal = loao[loao["legal_prediction"]].copy()
    fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(legal["target"], legal["official_prediction_three_seed_median"], color="#1f77b4"); lim = [min(legal[["target", "official_prediction_three_seed_median"]].min())-.002, max(legal[["target", "official_prediction_three_seed_median"]].max())+.002]; ax.plot(lim, lim, "k--"); ax.set(xlabel="Polarization target Y", ylabel="LOAO prediction", title="CHM LOAO prediction vs target"); fig.tight_layout(); fig.savefig(FIGURES / "F01_LOAO_PREDICTION_VS_TARGET.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(legal["checkpoint"].astype(str), legal["absolute_error"]); ax.set(xlabel="Held-out checkpoint (h)", ylabel="Absolute error", title="CHM LOAO absolute error"); fig.tight_layout(); fig.savefig(FIGURES / "F02_LOAO_ABSOLUTE_ERROR_BY_ANCHOR.png", dpi=180); plt.close(fig)
    mae = {"CHM": legal["absolute_error"].mean(), "B0": baselines["B0_absolute_error"].mean(), "B1": baselines["B1_absolute_error"].mean(), "B2": baselines["B2_absolute_error"].mean()}; fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(mae.keys(), mae.values()); ax.set(ylabel="LOAO MAE", title="Development LOAO MAE"); fig.tight_layout(); fig.savefig(FIGURES / "F03_CHM_VS_BASELINE_MAE.png", dpi=180); plt.close(fig)
    if final_trajectory is not None:
        complete = final_trajectory[final_trajectory["hou_kind"].eq("COMPLETE_CYCLE_HOU")]
        fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(complete["measurement_end_ophrs2_h"], complete["H_official_median"], label="CHM final latent H"); ax.set(xlabel="Operating hours", ylabel="Latent Health State H", title="Final Development latent trajectory"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "F04_FINAL_LATENT_HEALTH_TRAJECTORY.png", dpi=180); plt.close(fig)
        if uncertainty is not None and not uncertainty.empty:
            merged = final_trajectory.merge(uncertainty[["hou_id", "P2_5", "P50", "P97_5"]], on="hou_id"); c = merged[merged["hou_kind"].eq("COMPLETE_CYCLE_HOU")]; fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(c["measurement_end_ophrs2_h"], c["H_official_median"], label="Deterministic median"); ax.fill_between(c["measurement_end_ophrs2_h"], c["P2_5"], c["P97_5"], alpha=.25, label="Block-bootstrap interval"); ax.set(xlabel="Operating hours", ylabel="Latent Health State H", title="Final trajectory with bootstrap interval"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "F05_FINAL_TRAJECTORY_WITH_BOOTSTRAP_CI.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4)); ax.scatter(complete["P1S_retention"], complete["H_official_median"], s=12); ax.set(xlabel="P1S retention", ylabel="Latent Health State H", title="P1S retention vs latent H"); fig.tight_layout(); fig.savefig(FIGURES / "F06_P1S_RETENTION_VS_LATENT_HEALTH.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(complete["measurement_end_ophrs2_h"], complete["H_official_median"])
        for checkpoint, indices in data.anchor_indices.items():
            ax.scatter(final_trajectory.loc[list(indices), "measurement_end_ophrs2_h"], final_trajectory.loc[list(indices), "H_official_median"], label=str(checkpoint))
        ax.set(xlabel="Operating hours", ylabel="Latent Health State H", title="Sparse anchor mappings on latent trajectory")
        ax.legend(title="Checkpoint")
        fig.tight_layout()
        fig.savefig(FIGURES / "F07_ANCHOR_TARGETS_ON_LATENT_TRAJECTORY.png", dpi=180)
        plt.close(fig)
        if s1_trajectory is not None:
            joined = complete[["hou_id", "measurement_end_ophrs2_h", "H_official_median"]].merge(s1_trajectory[["hou_id", "H_S1_official_median"]], on="hou_id"); fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(joined["measurement_end_ophrs2_h"], joined["H_official_median"], label="Primary"); ax.plot(joined["measurement_end_ophrs2_h"], joined["H_S1_official_median"], label="S1", alpha=.8); ax.set(xlabel="Operating hours", ylabel="Latent Health State H", title="Primary vs S1 trajectories"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "F08_PRIMARY_VS_S1_TRAJECTORY.png", dpi=180); plt.close(fig)
    legal_online = online[online["status"].eq("LEGAL_PREFIX_FIT")]; fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(legal_online["target"], legal_online["prediction"]); ax.set(xlabel="Target", ylabel="Causal-prefix prediction", title="Causal online diagnostic (n=3)"); fig.tight_layout(); fig.savefig(FIGURES / "F09_CAUSAL_ONLINE_DIAGNOSTIC.png", dpi=180); plt.close(fig)


def build_report(result: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase CHM-D — Development Execution", "", "## 1. Execution integrity", "", f"- Base HEAD: `{result['base_head']}`", "- Authority: `FINAL_V2A`", "- Frozen SHA checks: PASS", "- Test/raw/polarization/EIS accessed: NO", "", "## 2–4. Baselines and nested LOAO", "", f"- CHM MAE: {result['metrics']['CHM']['MAE']:.8f}", f"- B0/B1/B2 MAE: {result['metrics']['B0']['MAE']:.8f} / {result['metrics']['B1']['MAE']:.8f} / {result['metrics']['B2']['MAE']:.8f}", "- Six official CHM estimates are three-seed medians; Spearman (n=6) is descriptive only.", "", "## 5–8. Gates and qualification", "", *[f"- {gate}: {status}" for gate, status in result["gates"].items()], f"- Primary Development qualification: `{result['primary_status']}`", "", "## 9. Final all-six latent trajectory", "", "Development-only Cycle-Aligned Latent Health State. It is not validated SOH.", "", "## 10. S1 secondary robustness", "", f"- Status: `{result['S1']['status']}`", "- S1 is a prespecified secondary robustness diagnostic, not a primary gate.", "", "## 11. Causal-online diagnostic", "", "Only 1150/1500/1750 h have legal prefix fits; aggregate n=3 is secondary only.", "", "## 12. Limitations", "", "- Only six sparse Development anchors; PAA/PAV ancestry remains exploratory.", "- Three of six median sources are inside full joint support.", "- LOCAL_WINDOW_SCATTER_PROXY is not repeat variance.", "- S1 is not an independent validation set; Test remains unopened.", "- Latent health is not validated SOH and no RUL conclusion is made.", ""]
    (REPORTS / "FINAL_CHM_D_DEVELOPMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> dict[str, Any]:
    runtime = verify_runtime_authority()  # must precede all optimizer creation
    data = load_frozen_chm_data()
    write_json(reference_scaler_audit(data), "REFERENCE_SCALER_AUDIT.json")
    baselines = run_baseline_loao(data)
    write_csv(baselines, "CHM_D_BASELINE_LOAO.csv")
    inner, selections, loao, outer_fits, selected = outer_loao(data)
    write_csv(inner, "CHM_D_INNER_PROFILE_SELECTION.csv")
    write_csv(selections, "CHM_D_OUTER_PROFILE_SELECTION.csv")
    write_csv(loao, "CHM_D_LOAO_PREDICTIONS.csv")
    legal = loao[loao["legal_prediction"]].copy()
    metrics = {"CHM": aggregate_metrics(legal) if len(legal) == 6 else {"MAE": np.nan, "RMSE": np.nan, "bias": np.nan, "Spearman_n6_descriptive_only": np.nan, "calibration_slope_target_on_prediction": np.nan, "calibration_intercept_target_on_prediction": np.nan}, "B0": baseline_metrics(baselines, "B0"), "B1": baseline_metrics(baselines, "B1"), "B2": baseline_metrics(baselines, "B2")}
    profile, counts = final_profile(selected)
    gate1 = len(legal) == 6 and profile is not None
    gate2 = bool(gate1 and metrics["CHM"]["MAE"] < metrics["B0"]["MAE"])
    gate3 = bool(gate1 and metrics["CHM"]["MAE"] <= min(metrics["B1"]["MAE"], metrics["B2"]["MAE"]) + 1e-6)
    monotonic = monotonicity_audit(data, outer_fits)
    gate4 = bool(gate1 and len(monotonic) == 18 and monotonic["monotonic_pass"].all())
    final_fits: list[Any] | None = None
    final_trajectory: pd.DataFrame | None = None
    s1_frame: pd.DataFrame | None = None
    s1_result: dict[str, Any] = {"status": "NOT_RUN", "trajectory_Spearman": np.nan, "median_abs_difference": np.nan, "terminal_difference": np.nan}
    if profile is not None:
        final_fits = [fit_chm(data, ANCHORS, profile, seed) for seed in SEEDS]
        final_trajectory = data.hou[["hou_id", "hou_kind", "raw_cycle_id", "measurement_end_ophrs2_h", "measurement_end_datetime", "P1S_operating_voltage_retention"]].copy().rename(columns={"P1S_operating_voltage_retention": "P1S_retention"})
        for fit in final_fits:
            final_trajectory[f"H_seed{fit.seed}"] = fit.states
        final_trajectory["H_official_median"] = np.median(np.vstack([fit.states for fit in final_fits]), axis=0)
        write_csv(final_trajectory, "CHM_D_FINAL_LATENT_TRAJECTORY.csv")
        monotonic = monotonicity_audit(data, outer_fits, final_fits)
        s1_fits = [fit_chm(data, (300, 850, 1750), profile, seed) for seed in SEEDS]
        s1_frame = pd.DataFrame({"hou_id": data.hou["hou_id"], "H_S1_seed42": s1_fits[0].states, "H_S1_seed2024": s1_fits[1].states, "H_S1_seed3407": s1_fits[2].states})
        s1_frame["H_S1_official_median"] = np.median(np.vstack([fit.states for fit in s1_fits]), axis=0)
        primary_complete = final_trajectory.loc[final_trajectory["hou_kind"].eq("COMPLETE_CYCLE_HOU"), "H_official_median"].to_numpy()
        s1_complete = s1_frame.loc[data.hou["hou_kind"].eq("COMPLETE_CYCLE_HOU"), "H_S1_official_median"].to_numpy()
        rho = float(pd.Series(primary_complete).corr(pd.Series(s1_complete), method="spearman"))
        median_difference = float(np.median(np.abs(primary_complete-s1_complete)))
        terminal_difference = float(abs(primary_complete[-1]-s1_complete[-1]))
        supported = rho >= .95 and median_difference <= .01 and terminal_difference <= .01
        s1_result = {"status": "S1_ROBUSTNESS_SUPPORTED" if supported else "S1_ROBUSTNESS_CONCERN", "trajectory_Spearman": rho, "median_abs_difference": median_difference, "terminal_difference": terminal_difference}
    write_csv(monotonic, "CHM_D_MONOTONICITY_AUDIT.csv")
    gates = {"Gate_1": "PASS" if gate1 else "FAIL", "Gate_2": "PASS" if gate2 else "FAIL", "Gate_3": "PASS" if gate3 else "FAIL", "Gate_4": "PASS" if gate4 else "FAIL"}
    bootstrap_triggered = gate1 and gate2 and gate3 and gate4
    outer_intervals = pd.DataFrame(); final_uncertainty = pd.DataFrame(); failures: list[dict[str, Any]] = []
    gate5 = False
    if bootstrap_triggered and profile is not None:
        outer_intervals, _, failed_outer = run_outer_bootstrap(data, selected); failures.extend(failed_outer)
        write_csv(outer_intervals, "CHM_D_OUTER_BOOTSTRAP_INTERVALS.csv")
        legal_counts = outer_intervals["legal_replicates"].tolist()
        gate5 = bool(len(outer_intervals) == 6 and min(legal_counts) >= 190 and int(outer_intervals["coverage"].sum()) >= 5)
        primary_uncertainty, failed_primary = run_final_bootstrap(data, profile, 4, 500, 201); failures.extend(failed_primary)
        sensitivity_uncertainty, failed_sensitivity = run_final_bootstrap(data, profile, 8, 100, 301); failures.extend(failed_sensitivity)
        final_uncertainty = primary_uncertainty
        if len(primary_uncertainty):
            write_csv(primary_uncertainty, "CHM_D_FINAL_TRAJECTORY_UNCERTAINTY.csv")
        if len(sensitivity_uncertainty):
            write_csv(sensitivity_uncertainty, "CHM_D_FINAL_TRAJECTORY_UNCERTAINTY_BLOCK8_SENSITIVITY.csv")
        final_block4_legal = int(primary_uncertainty["legal_replicates"].iloc[0]) if len(primary_uncertainty) else 0
        final_block8_legal = int(sensitivity_uncertainty["legal_replicates"].iloc[0]) if len(sensitivity_uncertainty) else 0
        gate5 = bool(gate5 and final_block4_legal >= 475 and final_block8_legal >= 95)
    gates["Gate_5"] = "PASS" if gate5 else "FAIL"
    failure_columns = ["scope", "outer_heldout_checkpoint", "replicate_index", "RNG_seed_tuple", "optimizer_seed", "failure_reason"]
    write_csv(pd.DataFrame(failures, columns=failure_columns), "CHM_D_BOOTSTRAP_FAILURES.csv")
    online = pd.DataFrame([prefix_select_and_fit(data, checkpoint) for checkpoint in ANCHORS])
    write_csv(online, "CHM_D_CAUSAL_ONLINE_DIAGNOSTIC.csv")
    online_legal = online[online["status"].eq("LEGAL_PREFIX_FIT")]
    online_metrics = {"checkpoints": [1150, 1500, 1750], "MAE": float(online_legal["absolute_error"].mean()), "RMSE": float(np.sqrt(np.mean((online_legal["prediction"]-online_legal["target"]).square())))}
    primary_status = "DEVELOPMENT_HEALTH_MODEL_QUALIFIED" if all(value == "PASS" for value in gates.values()) else "DEVELOPMENT_HEALTH_MODEL_NOT_QUALIFIED"
    result = {"phase": "Phase_CHM_D_Development_Execution", "base_head": BASE_HEAD, "execution_head": repo_head(), "execution_integrity": {"authority_version": runtime["authority"]["authority_version"], "protocol_hash_pass": True, "supplement_hash_pass": True, "frozen_input_hash_pass": True, "CHM_D_test_raw_accessed": False, "CHM_D_test_polarization_accessed": False, "CHM_D_test_EIS_accessed": False, "CHM_D_test_used_for_selection": False}, "outer_LOAO": {"legal_predictions": int(len(legal)), **metrics["CHM"]}, "baselines": {"B0_MAE": metrics["B0"]["MAE"], "B0_RMSE": metrics["B0"]["RMSE"], "B1_MAE": metrics["B1"]["MAE"], "B1_RMSE": metrics["B1"]["RMSE"], "B2_MAE": metrics["B2"]["MAE"], "B2_RMSE": metrics["B2"]["RMSE"]}, "gates": gates, "final_profile": {"majority_vote_counts": counts, "selected": profile, "outer_target_used_for_final_profile": False}, "uncertainty": {"bootstrap_triggered": bootstrap_triggered, "outer_legal_replicates": outer_intervals.set_index("checkpoint")["legal_replicates"].to_dict() if len(outer_intervals) else {}, "all_six_block4_legal": int(final_uncertainty["legal_replicates"].iloc[0]) if len(final_uncertainty) else 0, "all_six_block8_legal": 100 - sum(1 for failure in failures if failure["scope"] == "final_all_six_block8") if bootstrap_triggered else 0}, "S1": s1_result, "causal_online": online_metrics, "primary_status": primary_status, "latent_state_is_validated_SOH": False, "RUL": "NOT_RUN", "Test": "SEALED_FOR_CHM_D", "automatic_start": False, "NEXT_CANDIDATE_PHASE": "Phase_THV_Test_Health_Validation" if primary_status == "DEVELOPMENT_HEALTH_MODEL_QUALIFIED" else "NONE"}
    write_yaml(result, FINAL / "FINAL_DECISION.yaml")
    write_json({"metrics": metrics, "gates": gates, "bootstrap_triggered": bootstrap_triggered}, "CHM_D_DETERMINISTIC_METRICS.json")
    if final_trajectory is not None:
        write_csv(s1_frame, "CHM_D_S1_TRAJECTORY.csv")
    plot_results(loao, baselines, final_trajectory, final_uncertainty, data, s1_frame, online)
    build_report(result)
    return result


def summary(result: dict[str, Any]) -> None:
    print("Phase CHM-D Development Execution\n=================================")
    print(f"Base HEAD: {result['base_head']}\nAuthority: FINAL_V2A\nFrozen hash validation: PASS\nTest accessed: NO\nHOU: 152\nComplete cycles: 146\nSparse anchors: 6")
    print(f"Outer LOAO legal: {result['outer_LOAO']['legal_predictions']} / 6")
    for checkpoint, profile in result["final_profile"].get("majority_vote_counts", {}).items(): print(f"Profile votes {checkpoint}: {profile}")
    print(f"Majority-vote final profile: {result['final_profile']['selected']}")
    print(f"CHM MAE: {result['outer_LOAO']['MAE']:.8f}\nB0 Raw P1S MAE: {result['baselines']['B0_MAE']:.8f}\nB1 Isotonic MAE: {result['baselines']['B1_MAE']:.8f}\nB2 Nonnegative affine MAE: {result['baselines']['B2_MAE']:.8f}\nCHM RMSE: {result['outer_LOAO']['RMSE']:.8f}\nAggregate Spearman n=6: {result['outer_LOAO']['Spearman_n6_descriptive_only']:.8f}")
    for gate, status in result["gates"].items(): print(f"{gate}: {status}")
    print(f"Bootstrap triggered: {'YES' if result['uncertainty']['bootstrap_triggered'] else 'NO'}\nPrimary Development qualification: {result['primary_status']}\nS1: {result['S1']['status']}\nCausal online n=3 MAE: {result['causal_online']['MAE']:.8f}\nLatent health called validated SOH: NO\nRUL: NOT_RUN\nTest: NOT_OPENED\nAutomatic next phase: NO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run the authorized Development-only experiment.")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Explicit --execute required; this runner will not start implicitly.")
    try:
        summary(execute())
    except CHMRuntimeFailure as exc:
        raise SystemExit(f"CHM_PROTOCOL_RUNTIME_MISMATCH: {exc}")

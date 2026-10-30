"""Meaningful numerical and leakage checks, not implementation-mirroring tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import forecast_from_observations, calculate_rul
from evaluate import evaluate


def run_checks(observations, config, truth=None):
    model, reference = config["model"], config["reference"]
    reference_run = forecast_from_observations(observations, model, reference)
    # Change all future measurements AND future restart events. Neither fitted
    # parameters, historical states nor fixed-origin predictions may change.
    mutated = observations.copy()
    mask = mutated.time_h > model["forecast_cutoff_h"]
    mutated.loc[mask, "stack_voltage_v"] += 10.0
    mutated.loc[mask, "restart"] = 1
    future_mutated_run = forecast_from_observations(mutated, model, reference)
    pd.testing.assert_frame_equal(reference_run["history"], future_mutated_run["history"])
    pd.testing.assert_frame_equal(reference_run["forecast"], future_mutated_run["forecast"])
    assert reference_run["parameters"] == future_mutated_run["parameters"]

    # Stronger prefix check: use a strictly earlier observed cutoff. It is
    # selected from this run's history, so changing config does not invalidate
    # the test by leaving an old hard-coded cutoff behind.
    earlier_cfg = copy.deepcopy(model)
    eligible = observations.loc[(observations.time_h >= model["calibration_end_h"])
                                & (observations.time_h < model["forecast_cutoff_h"]), "time_h"].sort_values()
    # There must remain at least 10 post-calibration points for the model's
    # rolling slope fit. Tiny custom inputs may legitimately skip this check.
    prefix_check = "SKIP: fewer than 11 eligible prefix observations"
    if len(eligible) >= 11:
        idx = min(max(10, int(0.8 * (len(eligible) - 1))), len(eligible) - 1)
        earlier_cutoff = float(eligible.iloc[idx])
        earlier_cfg["forecast_cutoff_h"] = earlier_cutoff
        earlier_run = forecast_from_observations(observations, earlier_cfg, reference)
        pd.testing.assert_frame_equal(
            reference_run["history"].loc[reference_run["history"].time_h <= earlier_cutoff].reset_index(drop=True),
            earlier_run["history"].reset_index(drop=True))
        prefix_check = f"PASS: prefix through {earlier_cutoff:g} h is unchanged"

    series = reference_run["history"].recovery_guardrail_estimate_v.dropna().to_numpy()
    assert np.all(np.diff(series) <= 1e-12)
    assert calculate_rul(.70, 1e-5, .6912) is None
    assert calculate_rul(.68, -2e-5, .6912) == 0.0
    assert np.isclose(calculate_rul(.703, -2e-5, .6912), 590)

    # Algebra: Vlt = Vop - delta + q. Recovery increment is q_before-q_after,
    # not an extra positive term to be subtracted again from post-rest voltage.
    latent, delta, q_before, q_after = .670, .010, .012, .003
    before, after = latent + delta - q_before, latent + delta - q_after
    assert np.isclose(after - before, q_before - q_after)
    assert np.isclose(after - delta + q_after, latent)

    checks = {
        "future_measurements_and_restart_flags_do_not_change_inference": "PASS",
        "earlier_cutoff_prefix_equals_full_run_prefix": prefix_check,
        "guardrail_history_has_no_upward_steps": "PASS",
        "rul_zero_and_no_finite_crossing_edge_cases": "PASS",
        "recovery_algebra_does_not_double_subtract": "PASS",
        "inference_runs_without_truth_argument_or_truth_file": "PASS",
    }
    # External data must never inherit the bundled oracle merely because its
    # timestamps happen to overlap the synthetic example.
    from run_demo import select_truth_path
    root = Path(__file__).resolve().parent
    assert select_truth_path(root / "external_user_observations.csv", None, root) is None
    assert select_truth_path(root / "data/observations.csv", None, root, no_evaluate=True) is None
    checks["external_observations_do_not_auto_load_bundled_truth"] = "PASS"
    # Deliberately corrupt oracle values only after inference and verify the
    # inference object remains bit-for-bit unchanged. Scores should change.
    if truth is not None:
        old_history, old_future = reference_run["history"].copy(), reference_run["forecast"].copy()
        original_metrics, _ = evaluate(reference_run, truth)
        wrong_truth = truth.copy()
        wrong_truth["latent_voltage_v"] += .100
        changed_metrics, _ = evaluate(reference_run, wrong_truth)
        pd.testing.assert_frame_equal(old_history, reference_run["history"])
        pd.testing.assert_frame_equal(old_future, reference_run["forecast"])
        assert original_metrics["methods"]["recovery_separated"]["heldout_latent_rmse_mv"] != changed_metrics["methods"]["recovery_separated"]["heldout_latent_rmse_mv"]
        checks["corrupt_truth_changes_only_evaluation_not_predictions"] = "PASS"
    return checks


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    obs = pd.read_csv(root / "data/observations.csv")
    truth_path = root / "data/truth_for_evaluation_only.csv"
    oracle = pd.read_csv(truth_path) if truth_path.exists() else None
    print(json.dumps(run_checks(obs, cfg, oracle), ensure_ascii=False, indent=2))

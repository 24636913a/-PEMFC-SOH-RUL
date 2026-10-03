"""Frozen CHM fitting, nested LOAO selection, and deterministic metrics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from chm_data import ANCHORS, FrozenCHMData, target_for_checkpoint, training_anchor_frame
from chm_losses import anchor_loss, anchor_predictions, observation_loss, prepare_smoothness_design, smoothness_loss
from chm_model import CHMModel


PROFILES: dict[str, dict[str, float]] = {
    "A_BALANCED": {"lambda_anchor": 1.0, "lambda_obs": 1.0, "lambda_smooth": 0.1},
    "B_ANCHOR_EMPHASIS": {"lambda_anchor": 3.0, "lambda_obs": 1.0, "lambda_smooth": 0.1},
    "C_OBSERVATION_EMPHASIS": {"lambda_anchor": 1.0, "lambda_obs": 3.0, "lambda_smooth": 0.1},
}
PROFILE_ORDER = tuple(PROFILES)
SEEDS = (42, 2024, 3407)


class CHMFitFailure(RuntimeError):
    """A deterministic/bootstrapped fit produced an illegal numerical state."""


@dataclass
class FitResult:
    profile: str
    seed: int
    training_checkpoints: tuple[int, ...]
    states: np.ndarray
    anchor_predictions: dict[int, float]
    epochs: int
    final_training_loss: float
    monotonic_increase_count: int


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def default_complete_weights(data: FrozenCHMData) -> dict[int, int]:
    return {
        int(cycle): 1
        for cycle in data.hou.loc[data.hou["hou_kind"].eq("COMPLETE_CYCLE_HOU"), "raw_cycle_id"].astype(int).tolist()
    }


def _check_finite_model(model: CHMModel) -> None:
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise CHMFitFailure("nonfinite_model_parameters")


def _loss_inputs(data: FrozenCHMData, training_checkpoints: Iterable[int]) -> tuple[list[tuple[int, int, int]], torch.Tensor, torch.Tensor]:
    anchor_frame = training_anchor_frame(data, tuple(int(value) for value in training_checkpoints))
    groups = [data.anchor_indices[int(checkpoint)] for checkpoint in anchor_frame["checkpoint"].tolist()]
    targets = torch.tensor(anchor_frame["Y"].astype(float).to_numpy(), dtype=torch.float64)
    raw_weights = anchor_frame["reliability_weight_raw"].astype(float).to_numpy()
    if not np.isfinite(raw_weights).all() or np.any(raw_weights <= 0):
        raise CHMFitFailure("invalid_frozen_anchor_reliability_weight")
    weights = torch.tensor(raw_weights / raw_weights.mean(), dtype=torch.float64)
    return groups, targets, weights


def fit_chm(
    data: FrozenCHMData,
    training_checkpoints: tuple[int, ...] | list[int],
    profile: str,
    seed: int,
    observation_weights: np.ndarray | None = None,
    complete_weights: dict[int, int] | None = None,
) -> FitResult:
    """Fit one frozen full-sequence CHM instance using only supplied train anchors."""
    if profile not in PROFILES or seed not in SEEDS:
        raise CHMFitFailure("unregistered_profile_or_seed")
    configure_determinism(seed)
    train = tuple(sorted(int(value) for value in training_checkpoints))
    groups, anchor_targets, reliability_weights = _loss_inputs(data, train)
    complete_weights = default_complete_weights(data) if complete_weights is None else dict(complete_weights)
    expected_cycle_ids = set(default_complete_weights(data))
    if set(complete_weights) != expected_cycle_ids or any(weight < 0 for weight in complete_weights.values()):
        raise CHMFitFailure("invalid_complete_cycle_bootstrap_weights")
    obs_np = np.ones(len(data.hou), dtype=np.float64) if observation_weights is None else np.asarray(observation_weights, dtype=np.float64)
    if obs_np.shape != (len(data.hou),) or not np.isfinite(obs_np).all() or np.any(obs_np < 0) or float(obs_np.sum()) <= 0:
        raise CHMFitFailure("loss_normalization_denominator_invalid")
    partial_mask = data.hou["hou_kind"].eq("PREANCHOR_PARTIAL_HOU").to_numpy()
    if not np.all(obs_np[partial_mask] == 1):
        raise CHMFitFailure("partial_HOU_observation_weight_not_one")
    model = CHMModel(data.model_inputs.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    inputs = torch.tensor(data.model_inputs, dtype=torch.float64)
    retention = torch.tensor(data.hou["P1S_operating_voltage_retention"].astype(float).to_numpy(), dtype=torch.float64)
    obs_weights = torch.tensor(obs_np, dtype=torch.float64)
    profile_weights = PROFILES[profile]
    smoothness_design = prepare_smoothness_design(data.continuous_complete_triples, complete_weights)
    checkpoint_loss: float | None = None
    final_loss = math.nan
    epochs = 0
    try:
        for epoch in range(1, 5001):
            optimizer.zero_grad(set_to_none=True)
            states = model.latent_trajectory(inputs)
            calibrated = model.observation_calibration(retention)
            predictions = anchor_predictions(states, groups)
            loss_anchor = anchor_loss(predictions, anchor_targets, reliability_weights)
            loss_obs = observation_loss(states, calibrated, obs_weights)
            loss_smooth = smoothness_loss(states, smoothness_design)
            total = profile_weights["lambda_anchor"] * loss_anchor + profile_weights["lambda_obs"] * loss_obs + profile_weights["lambda_smooth"] * loss_smooth
            if not torch.isfinite(total):
                raise CHMFitFailure("nonfinite_training_loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            _check_finite_model(model)
            final_loss = float(total.detach().cpu())
            epochs = epoch
            if epoch >= 500 and epoch % 200 == 0:
                if checkpoint_loss is not None:
                    relative_improvement = (checkpoint_loss - final_loss) / max(abs(checkpoint_loss), 1e-12)
                    if relative_improvement < 1e-5:
                        break
                checkpoint_loss = final_loss
    except CHMFitFailure:
        raise
    except Exception as exc:  # exact exception is preserved as a numerical failure record
        raise CHMFitFailure(f"optimizer_exception:{type(exc).__name__}") from exc
    with torch.no_grad():
        states_tensor = model.latent_trajectory(inputs)
        if not torch.isfinite(states_tensor).all():
            raise CHMFitFailure("nonfinite_HOU_latent_states")
        states_np = states_tensor.detach().cpu().numpy()
        predictions_all = {
            checkpoint: float(torch.median(states_tensor[torch.tensor(data.anchor_indices[checkpoint])]).cpu())
            for checkpoint in ANCHORS
        }
    complete_indices = data.hou.index[data.hou["hou_kind"].eq("COMPLETE_CYCLE_HOU")].to_numpy()
    increase_count = int(np.count_nonzero(np.diff(states_np[complete_indices]) > 0.0))
    return FitResult(profile=profile, seed=seed, training_checkpoints=train, states=states_np, anchor_predictions=predictions_all, epochs=epochs, final_training_loss=final_loss, monotonic_increase_count=increase_count)


def select_profile_for_outer_fold(data: FrozenCHMData, outer_heldout: int) -> tuple[str | None, list[dict[str, Any]], dict[str, float]]:
    """Nested 5x3 inner LOAO; the outer target never enters a fit or score."""
    outer_train = tuple(checkpoint for checkpoint in ANCHORS if checkpoint != outer_heldout)
    records: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    for profile in PROFILE_ORDER:
        errors: list[float] = []
        invalid = False
        for inner_heldout in outer_train:
            inner_train = tuple(checkpoint for checkpoint in outer_train if checkpoint != inner_heldout)
            for seed in SEEDS:
                try:
                    fit = fit_chm(data, inner_train, profile, seed)
                    prediction = fit.anchor_predictions[inner_heldout]
                    target = target_for_checkpoint(data, inner_heldout)
                    error = abs(prediction - target)
                    records.append({"outer_heldout_checkpoint": outer_heldout, "inner_heldout_checkpoint": inner_heldout, "profile": profile, "seed": seed, "prediction": prediction, "target": target, "absolute_error": error, "fit_status": "LEGAL", "epochs": fit.epochs, "final_training_loss": fit.final_training_loss})
                    errors.append(error)
                except CHMFitFailure as exc:
                    invalid = True
                    records.append({"outer_heldout_checkpoint": outer_heldout, "inner_heldout_checkpoint": inner_heldout, "profile": profile, "seed": seed, "prediction": np.nan, "target": target_for_checkpoint(data, inner_heldout), "absolute_error": np.nan, "fit_status": f"PROFILE_INVALID_FOR_OUTER_SELECTION:{exc}", "epochs": np.nan, "final_training_loss": np.nan})
        scores[profile] = float(np.mean(errors)) if not invalid and len(errors) == 15 else math.inf
    legal = [profile for profile in PROFILE_ORDER if math.isfinite(scores[profile])]
    if not legal:
        return None, records, scores
    lowest = min(scores[profile] for profile in legal)
    selected = next(profile for profile in PROFILE_ORDER if profile in legal and scores[profile] <= lowest + 1e-8)
    return selected, records, scores


def fit_outer_prediction(data: FrozenCHMData, outer_heldout: int, profile: str) -> tuple[list[FitResult], dict[str, Any]]:
    training = outer_training_anchors(outer_heldout)
    fits: list[FitResult] = []
    for seed in SEEDS:
        fits.append(fit_chm(data, training, profile, seed))
    seed_predictions = [fit.anchor_predictions[outer_heldout] for fit in fits]
    official = float(np.median(seed_predictions))
    target = target_for_checkpoint(data, outer_heldout)
    return fits, {
        "checkpoint": outer_heldout,
        "target": target,
        "pred_seed42": seed_predictions[0],
        "pred_seed2024": seed_predictions[1],
        "pred_seed3407": seed_predictions[2],
        "official_prediction_three_seed_median": official,
        "signed_error": official - target,
        "absolute_error": abs(official - target),
        "squared_error": (official - target) ** 2,
        "selected_profile": profile,
        "legal_prediction": True,
    }


def outer_training_anchors(outer_heldout: int) -> tuple[int, ...]:
    if outer_heldout not in ANCHORS:
        raise CHMFitFailure("illegal_outer_heldout_checkpoint")
    return tuple(checkpoint for checkpoint in ANCHORS if checkpoint != outer_heldout)


def aggregate_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    target = predictions["target"].astype(float).to_numpy()
    pred = predictions["official_prediction_three_seed_median"].astype(float).to_numpy()
    residual = pred - target
    slope, intercept = (np.polyfit(pred, target, 1) if len(pred) >= 2 and np.ptp(pred) > 0 else (np.nan, np.nan))
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual.square()))),
        "bias": float(np.mean(residual)),
        "Spearman_n6_descriptive_only": float(pd.Series(pred).corr(pd.Series(target), method="spearman")),
        "calibration_slope_target_on_prediction": float(slope),
        "calibration_intercept_target_on_prediction": float(intercept),
    }

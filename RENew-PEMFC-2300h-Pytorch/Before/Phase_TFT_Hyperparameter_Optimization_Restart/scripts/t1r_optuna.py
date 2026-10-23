"""Optuna search and study-object closure contracts for an authorized T1-R run.

Nothing in this module creates a study at import time. The fake studies used in
tests implement the small ``.trials`` protocol consumed by :func:`audit_study`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable


class OptunaContractError(RuntimeError):
    pass


ALLOWED_STATES = {"COMPLETE", "FAILED", "PRUNED", "RUNNING", "WAITING"}


def _state_name(trial: Any) -> str:
    state = getattr(trial, "state", None)
    return getattr(state, "name", str(state)).upper()


def _trial_number(trial: Any) -> int:
    return int(getattr(trial, "number"))


def fixed_trial_zero(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg["search"]["fixed_trial_0"])


def assert_new_study_path(path: Path, phase_root: Path, old_phase_root: Path) -> None:
    resolved, root = path.resolve(), phase_root.resolve()
    if root not in resolved.parents or old_phase_root.resolve() in resolved.parents or "T1R" not in path.name:
        raise OptunaContractError("STUDY_PATH_ISOLATION_FAILURE")


def audit_study(study: Any, cfg: dict[str, Any], *, expected_study_prefix: str = "T1R") -> dict[str, Any]:
    """Fail closed from the persisted study object; never accept caller totals."""
    trials = list(getattr(study, "trials", []))
    search = cfg["search"]
    budget, minimum = int(search["attempted_trials_per_outer_fold"]), int(search["minimum_completed_trials_per_outer_fold"])
    study_name = str(getattr(study, "study_name", ""))
    if not study_name.startswith(expected_study_prefix):
        raise OptunaContractError("OLD_OR_FOREIGN_STUDY_NAMESPACE")
    numbers = [_trial_number(t) for t in trials]
    if len(trials) != budget or len(set(numbers)) != len(numbers) or sorted(numbers) != list(range(budget)):
        raise OptunaContractError("TRIAL_BUDGET_OR_IDENTITY_INVALID")
    states = [_state_name(t) for t in trials]
    if any(s not in ALLOWED_STATES for s in states):
        raise OptunaContractError("TRIAL_STATE_INVALID")
    if any(s in {"RUNNING", "WAITING"} for s in states):
        raise OptunaContractError("UNRESOLVED_TRIAL_STATE")
    completed = [t for t, state in zip(trials, states) if state == "COMPLETE"]
    if len(completed) < minimum:
        raise OptunaContractError("INSUFFICIENT_COMPLETED_TRIALS")
    trial0 = next(t for t in trials if _trial_number(t) == 0)
    expected0 = fixed_trial_zero(cfg)
    params0 = dict(getattr(trial0, "user_attrs", {}).get("resolved_params", {}))
    if _state_name(trial0) != "COMPLETE" or any(params0.get(k) != v for k, v in expected0.items()):
        raise OptunaContractError("FIXED_TRIAL_ZERO_IDENTITY_FAILURE")
    for trial in completed:
        value = getattr(trial, "value", None)
        attrs = dict(getattr(trial, "user_attrs", {}))
        resolved = attrs.get("resolved_params", {})
        required_params = {"learning_rate", "weight_decay", "dropout", "hidden_size", "hidden_continuous_size", "batch_size"}
        if value is None or not math.isfinite(float(value)) or "best_epoch" not in attrs or "run_namespace" not in attrs or required_params - set(resolved):
            raise OptunaContractError("COMPLETE_TRIAL_METADATA_INVALID")
    for trial, state in zip(trials, states):
        if state in {"FAILED", "PRUNED"} and not dict(getattr(trial, "user_attrs", {})).get("failure_reason"):
            raise OptunaContractError("TERMINATED_TRIAL_REASON_MISSING")
    best = getattr(study, "best_trial", None)
    if best is None or _state_name(best) != "COMPLETE":
        raise OptunaContractError("BEST_TRIAL_MISSING")
    best_attrs = dict(getattr(best, "user_attrs", {}))
    return {"pass": True, "study_name": study_name, "attempted": len(trials), "completed": len(completed), "best_trial_id": _trial_number(best), "best_params": dict(best_attrs["resolved_params"]), "best_epoch": int(best_attrs["best_epoch"])}


def create_authorized_study(*, optuna_module: Any, storage_path: Path, fold: str, cfg: dict[str, Any], test_only: bool = False) -> Any:
    """Create a new isolated study. Called only after the runner authorization gate."""
    if not test_only: assert_new_study_path(storage_path, Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2] / "Phase_TFT_Hyperparameter_Optimization")
    study_name = f"{cfg['search']['study_name_prefix']}_{fold}"
    sampler = optuna_module.samplers.TPESampler(seed=int(cfg["search"]["sampler_seed"]))
    return optuna_module.create_study(study_name=study_name, direction="minimize", sampler=sampler, storage=f"sqlite:///{storage_path}", load_if_exists=False)


def suggest_trial_params(trial: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """The only future trial-parameter route: trial 0 fixed, 1--15 TPE space."""
    if _trial_number(trial) == 0:
        params = fixed_trial_zero(cfg)
        trial.set_user_attr("resolved_params", params)
        return params
    search = cfg["search"]
    width_index = trial.suggest_int("width_index", 0, len(search["width_combinations"]) - 1)
    width = search["width_combinations"][int(width_index)]
    params = {"learning_rate": trial.suggest_float("learning_rate", **search["learning_rate"]), "weight_decay": trial.suggest_float("weight_decay", **search["weight_decay"]), "dropout": trial.suggest_categorical("dropout", list(search["dropout"])), "hidden_size": int(width[0]), "hidden_continuous_size": int(width[1]), "batch_size": trial.suggest_categorical("batch_size", list(search["batch_size"]))}
    trial.set_user_attr("resolved_params", params)
    return params


def run_search(study: Any, cfg: dict[str, Any], objective_factory: Callable[[int], Callable[[Any], float]]) -> dict[str, Any]:
    """Future authorized search implementation; the objective owns model construction."""
    budget = int(cfg["search"]["attempted_trials_per_outer_fold"])
    if list(getattr(study, "trials", [])):
        raise OptunaContractError("NONEMPTY_NEW_T1R_STUDY")
    study.optimize(objective_factory(0), n_trials=1)
    study.optimize(objective_factory(1), n_trials=budget - 1)
    return audit_study(study, cfg, expected_study_prefix=str(cfg["search"]["study_name_prefix"]))

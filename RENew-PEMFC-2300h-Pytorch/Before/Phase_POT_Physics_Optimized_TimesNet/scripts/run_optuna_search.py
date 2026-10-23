"""Run the preregistered Fold-1-inner Optuna-TPE searches for POT M3/M4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import yaml

from pot_common import (
    PHASE, POTFailure, build_frozen_folds, load_inherited_protocols, load_protocol,
    safe_read_development, write_csv, write_json,
)
from pot_training import HyperParameters, METHODS, method_windows, select_inner_epoch


def _suggest(trial: optuna.Trial, cfg: dict[str, Any]) -> HyperParameters:
    space = cfg["optuna_tpe"]["search_space"]
    return HyperParameters(
        learning_rate=float(trial.suggest_float("learning_rate", float(space["learning_rate"]["low"]), float(space["learning_rate"]["high"]), log=True)),
        weight_decay=float(trial.suggest_float("weight_decay", float(space["weight_decay"]["low"]), float(space["weight_decay"]["high"]), log=True)),
        dropout=float(trial.suggest_categorical("dropout", list(space["dropout"]["choices"]))),
        batch_size=int(trial.suggest_categorical("batch_size", list(space["batch_size"]["choices"]))),
    )


def _signature(hp: HyperParameters) -> str:
    return json.dumps({"learning_rate": hp.learning_rate, "weight_decay": hp.weight_decay, "dropout": hp.dropout, "batch_size": hp.batch_size}, sort_keys=True)


def _trial_rows(study: optuna.Study, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {"method": method, "trial": trial.number, "state": trial.state.name, "value_inner_deltaV_RMSE_mV": trial.value}
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        row.update({f"user_{k}": v for k, v in trial.user_attrs.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trial") if rows else pd.DataFrame()


def _run_method(method: str, outer_train: Any, cfg: dict[str, Any], bs_cfg: dict[str, Any], common: Any) -> dict[str, Any]:
    search = cfg["optuna_tpe"]
    train = method_windows(method, outer_train)
    fit, stopper = common.split_inner_fit_stopper(train, float(cfg["inner_stopping"]["stopper_fraction"]))
    if set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)):
        raise POTFailure("OPTUNA_TARGET_ROW_ISOLATION_FAIL")
    storage_path = PHASE / str(search["resume_storage"]).format(method=method)
    storage = f"sqlite:///{storage_path.as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=int(search["sampler_seed"]), multivariate=False)
    study = optuna.create_study(study_name=f"POT_{method}", direction="minimize", sampler=sampler, storage=storage, load_if_exists=True)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    seen = {_signature(HyperParameters(**{key: trial.params[key] for key in ("learning_rate", "weight_decay", "dropout", "batch_size")})) for trial in completed if set(trial.params) == {"learning_rate", "weight_decay", "dropout", "batch_size"}}

    def objective(trial: optuna.Trial) -> float:
        hp = _suggest(trial, cfg)
        signature = _signature(hp)
        if signature in seen:
            trial.set_user_attr("deduplicated", True)
            raise optuna.TrialPruned("DUPLICATE_HYPERPARAMETER_SIGNATURE")
        seen.add(signature)
        selected = select_inner_epoch(
            method, fit, stopper, hp,
            int(search["trial_seed_base"]) + int(trial.number), bs_cfg, common,
            int(search["max_epochs_per_trial"]), int(search["patience_per_trial"]), float(search["gradient_clip_norm"]),
        )
        trial.set_user_attr("selected_epoch", int(selected["selected_epoch"]))
        trial.set_user_attr("epochs_ran", int(selected["epochs_ran"]))
        trial.set_user_attr("inner_time_s", float(selected["inner_time_s"]))
        trial.set_user_attr("input_dimension", int(METHODS[method]["input_dimension"]))
        trial.set_user_attr("outer_validation_accessed", False)
        return float(selected["best_inner_RMSE_mV"])

    remaining = int(search["n_trials_per_method"]) - len(completed)
    if remaining < 0:
        raise POTFailure("OPTUNA_TRIAL_BUDGET_EXCEEDED")
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) != int(search["n_trials_per_method"]):
        raise POTFailure("OPTUNA_COMPLETED_TRIAL_COUNT_MISMATCH")
    write_csv(_trial_rows(study, method), "outputs", f"OPTUNA_{method}_TRIALS.csv")
    best = study.best_trial
    hp = HyperParameters(
        learning_rate=float(best.params["learning_rate"]), weight_decay=float(best.params["weight_decay"]),
        dropout=float(best.params["dropout"]), batch_size=int(best.params["batch_size"]),
    )
    return {"method": method, "best_trial": int(best.number), "best_inner_deltaV_RMSE_mV": float(best.value), "selected_epoch": int(best.user_attrs["selected_epoch"]), "hyperparameters": hp.__dict__, "storage_relative_path": str(storage_path.relative_to(PHASE)).replace("\\", "/"), "completed_trials": len(completed), "outer_validation_accessed": False}


def _write_resolved_config(cfg: dict[str, Any], protocol_sha: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = {
        "status": "FROZEN_AFTER_FOLD1_INNER_ONLY_OPTUNA_BEFORE_OUTER_EVALUATION",
        "protocol_sha256": protocol_sha,
        "optimizer": cfg["formal_training"]["optimizer"],
        "formal_training": cfg["formal_training"],
        "search_results": {item["method"]: item for item in results},
        "outer_validation_used_for_selection": False,
        "test_accessed": False,
    }
    path = PHASE / "configs" / "RESOLVED_OPTUNA_CONFIGS.yaml"
    digest_path = PHASE / "configs" / "RESOLVED_OPTUNA_CONFIGS.sha256"
    payload = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    if path.exists():
        expected = digest_path.read_text(encoding="utf-8").split()[0]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise POTFailure("RESOLVED_OPTUNA_CONFIG_DRIFT")
        if path.read_text(encoding="utf-8") != payload:
            raise POTFailure("RESOLVED_OPTUNA_CONFIG_ALREADY_DIFFERENT")
    else:
        path.write_text(payload, encoding="utf-8")
        digest_path.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "  RESOLVED_OPTUNA_CONFIGS.yaml\n", encoding="utf-8")
    return resolved


def main() -> None:
    cfg, protocol_sha = load_protocol()
    stage0 = json.loads((PHASE / "outputs" / "STAGE0_PHYSICAL_FEASIBILITY_SUMMARY.json").read_text(encoding="utf-8"))
    if stage0["status"] not in {"PHYSICS_LOSS_FEASIBLE", "PHYSICS_LOSS_NOT_ESTABLISHED"}:
        raise POTFailure("STAGE0_RESULT_MISSING_OR_INVALID")
    bs_cfg, _df_cfg, common = load_inherited_protocols(cfg)
    frame, firewall = safe_read_development(cfg, bs_cfg)
    folds = build_frozen_folds(frame, cfg, bs_cfg, common)
    outer_train = next(train for fold, train, _validation in folds if fold == cfg["optuna_tpe"]["search_fold"])
    results = [_run_method(method, outer_train, cfg, bs_cfg, common) for method in cfg["optuna_tpe"]["methods"]]
    resolved = _write_resolved_config(cfg, protocol_sha, results)
    write_json({"status": "COMPLETE", "protocol_sha256": protocol_sha, "stage0_status": stage0["status"], "search_results": results, "data_firewall": firewall, "outer_validation_used_for_selection": False, "test_accessed": False}, "outputs", "OPTUNA_SEARCH_AUDIT.json")
    print(json.dumps({"status": "COMPLETE", "methods": [r["method"] for r in results], "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

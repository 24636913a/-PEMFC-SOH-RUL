"""T1-R primary state-machine runner.

The real dependency adapter remains token- and file-authorized.  Tests inject a
fully isolated adapter into this same dispatcher; no synthetic path can touch a
formal SQLite study, RealHyFC data, checkpoints, or Test artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from t1r_common import PHASE, T1RFailure, load_protocol, load_t0_model, safe_development
from t1r_decision import INSUFFICIENT, PENDING, multiseed_decision, seed42_decision
from t1r_evaluation import load_frozen_t0_metrics, load_frozen_t0_query_universe, validate_multiseed_identity, validate_pair_rows
from t1r_optuna import assert_new_study_path, audit_study


FOLDS = ("fold_1", "fold_2", "fold_3")
AUTHORIZATION = PHASE / "configs" / "T1R_EXECUTION_AUTHORIZATION.yaml"


class PipelineFailure(T1RFailure):
    pass


class DependencyAdapter(Protocol):
    def study(self, fold: str) -> Any: ...
    def freeze_selected(self, audits: dict[str, dict[str, Any]]) -> None: ...
    def evaluate(self, seed: int, selected: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]: ...
    def t0_queries(self) -> list[dict[str, Any]]: ...
    def metrics(self, rows: list[dict[str, Any]], model: str) -> dict[str, float]: ...


def runner_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_isolation(cfg: dict[str, Any]) -> None:
    path = str(cfg["search"]["database_relative_path"])
    if "Phase_TFT_Hyperparameter_Optimization/" in path or "T1R" not in cfg["search"]["study_name_prefix"]:
        raise PipelineFailure("OLD_T1_STUDY_ISOLATION_FAILURE")
    assert_new_study_path(PHASE / "outputs" / "optuna_T1R_fold_1.db", PHASE, PHASE.parent / "Phase_TFT_Hyperparameter_Optimization")


def verify_execution_authorization(token: str, protocol_sha: str) -> dict[str, Any]:
    """Validate the future explicit permission, rejecting omission and stale code."""
    if token != "EXECUTE_T1R_SEPARATE_TRAINING":
        raise PipelineFailure("EXECUTION_TOKEN_INVALID")
    if not AUTHORIZATION.is_file():
        raise PipelineFailure("NO_SEPARATE_T1R_EXECUTION_AUTHORIZATION")
    try:
        import yaml
        auth = yaml.safe_load(AUTHORIZATION.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PipelineFailure("AUTHORIZATION_FILE_UNREADABLE") from exc
    validate_authorization_payload(auth, protocol_sha)
    return auth


def validate_authorization_payload(auth: dict[str, Any], protocol_sha: str) -> None:
    """Pure production validation, also exercised with tampered payload tests."""
    required = {"protocol_sha256", "runner_sha256", "allowed_phase", "study_namespace", "development_scope"}
    if not isinstance(auth, dict) or required - set(auth):
        raise PipelineFailure("AUTHORIZATION_FILE_SCHEMA_INVALID")
    if auth["protocol_sha256"] != protocol_sha or auth["runner_sha256"] != runner_sha256():
        raise PipelineFailure("AUTHORIZATION_IDENTITY_MISMATCH")
    if auth["allowed_phase"] != PHASE.name or auth["study_namespace"] != "T1R" or auth["development_scope"] != "DEVELOPMENT_ONLY":
        raise PipelineFailure("AUTHORIZATION_SCOPE_MISMATCH")


def assert_development_only_paths(paths: list[str | Path]) -> None:
    """Whitelist concrete inputs; the project directory name itself is irrelevant."""
    allowed_files = {str((PHASE.parent / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv").resolve()), str((PHASE.parent / "Phase_TFT_Original_Baseline" / "outputs" / "TFT_SEED42_PREDICTIONS.csv").resolve()), str((PHASE.parent / "Phase_TFT_Original_Baseline" / "outputs" / "TFT_SEED42_PER_FOLD_RESULTS.csv").resolve())}
    for path in paths:
        value = str(Path(path).resolve())
        if value not in allowed_files:
            raise PipelineFailure("TEST_OR_UNREGISTERED_PATH_REFERENCE")


def check_authorization(token: str) -> None:
    """Backward-compatible public fail-closed authorization check."""
    _cfg, protocol_sha = load_protocol()
    verify_execution_authorization(token, protocol_sha)


def _fold_metrics(rows: list[dict[str, Any]], adapter: DependencyAdapter) -> dict[str, dict[str, float]]:
    return {model: adapter.metrics(rows, model) for model in ("B", "C")}


def execute_pipeline(*, cfg: dict[str, Any], protocol_sha: str, adapter: DependencyAdapter, authorization_verified: bool, synthetic: bool = False) -> dict[str, Any]:
    """One dispatch path for formal execution and isolated fake E2E verification."""
    if not authorization_verified:
        raise PipelineFailure("EXECUTION_AUTHORIZATION_NOT_VERIFIED")
    states = ["PROTOCOL_VERIFIED", "DATA_AND_T0_AUTHORITY_VERIFIED", "EXECUTION_AUTHORIZATION_VERIFIED"]
    audits: dict[str, dict[str, Any]] = {}
    for i, fold in enumerate(FOLDS, 1):
        audits[fold] = audit_study(adapter.study(fold), cfg, expected_study_prefix=str(cfg["search"]["study_name_prefix"]))
        states.append(f"FOLD{i}_OPTUNA_SEARCH")
    states += ["TRIAL_CLOSURE_AUDIT"]
    adapter.freeze_selected(audits)
    states += ["SELECTED_CONFIGURATIONS_FROZEN"]
    selected = {fold: {"trial_id": a["best_trial_id"], "params": a["best_params"], "best_epoch": a["best_epoch"]} for fold, a in audits.items()}
    t0 = adapter.t0_queries()
    seed42_rows, _ = adapter.evaluate(42, selected)
    seed42_identity = validate_pair_rows(seed42_rows, protocol_sha, t0)
    seed42_metrics = _fold_metrics(seed42_rows, adapter)
    a = load_frozen_t0_metrics(PHASE.parent / "Phase_TFT_Original_Baseline" / "outputs" / "TFT_SEED42_PER_FOLD_RESULTS.csv") if not synthetic else {"rmse": float(cfg["comparators"]["A_frozen_t0"]["RMSE_mV"]), "fold3_rmse": float(cfg["comparators"]["A_frozen_t0"]["fold3_RMSE_mV"]), "p99": 10.0, "target_equal_rmse": 10.0}
    decision42 = seed42_decision(a, seed42_metrics["B"], seed42_metrics["C"], cfg=cfg, identity_audited=bool(seed42_identity["pass"]))
    states += ["SEED42_BC_OUTER_EVALUATION", "SEED42_DECISION"]
    result: dict[str, Any] = {"states": states, "trial_audits": audits, "selected": selected, "seed42": decision42, "synthetic": synthetic, "model_training_performed": False if synthetic else None, "test_accessed": False}
    if decision42["status"] != PENDING:
        result["final_status"] = decision42["status"]
        if hasattr(adapter,"persist_result"): adapter.persist_result(result)
        return result
    all_rows = list(seed42_rows)
    metrics_by_seed = {42: {**seed42_metrics["C"], "b_rmse": seed42_metrics["B"]["rmse"]}}
    for seed in (2024, 3407):
        rows, _ = adapter.evaluate(seed, selected)
        all_rows.extend(rows)
        identity = validate_pair_rows(rows, protocol_sha, t0)
        seed_metrics = _fold_metrics(rows, adapter)
        if not identity["pass"]:
            raise PipelineFailure("MULTISEED_PAIR_IDENTITY_FAILURE")
        metrics_by_seed[seed] = {**seed_metrics["C"], "b_rmse": seed_metrics["B"]["rmse"]}
    states += ["MULTISEED_EVALUATION"]
    multi_identity = validate_multiseed_identity(all_rows, protocol_sha, t0)
    states += ["MULTISEED_IDENTITY_AUDIT"]
    final = multiseed_decision(decision42, multi_identity, metrics_by_seed, cfg=cfg)
    states += ["FINAL_DECISION"]
    result.update({"multiseed_identity": multi_identity, "multiseed_metrics": metrics_by_seed, "final_decision": final, "final_status": final["status"]})
    if hasattr(adapter,"persist_result"): adapter.persist_result(result)
    return result


def static_preflight() -> dict[str, Any]:
    cfg, protocol_sha = load_protocol()
    validate_isolation(cfg)
    _dev, data_audit = safe_development(cfg)
    model, _ = load_t0_model()
    return {"status": "T1R_FULL_PIPELINE_READY", "protocol_sha256": protocol_sha, "data_audit": data_audit, "canonical_tft_loaded": model.__name__, "model_training_performed": False, "test_accessed": False, "authorization_file_present": AUTHORIZATION.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "execute"), default="static")
    parser.add_argument("--authorization", default="PREPARATION_ONLY")
    args = parser.parse_args()
    if args.mode == "static":
        print(json.dumps(static_preflight(), sort_keys=True))
        return
    cfg, protocol_sha = load_protocol()
    validate_isolation(cfg)
    verify_execution_authorization(args.authorization, protocol_sha)
    safe_development(cfg); load_t0_model()
    # This remains unreachable in this no-training acceptance phase because no
    # valid authorization file is created or shipped.
    from t1r_real_adapter import RealAdapter
    execute_pipeline(cfg=cfg, protocol_sha=protocol_sha, adapter=RealAdapter(cfg, protocol_sha), authorization_verified=True)


if __name__ == "__main__":
    main()

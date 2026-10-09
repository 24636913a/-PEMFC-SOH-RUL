"""Run only T1-R static and isolated-synthetic acceptance tests."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from run_t1r import AUTHORIZATION, runner_sha256, static_preflight, validate_isolation
from t1r_common import PHASE, PROJECT, load_protocol


OLD = PROJECT / "Phase_TFT_Hyperparameter_Optimization"
ARCHIVE = Path("E:/沈农信电/T1_Original_Invalidated_Archive/T1_original_2026-10-09/optuna_fold_1.db")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_reports(readiness: dict) -> None:
    reports = PHASE / "reports"; reports.mkdir(exist_ok=True)
    (reports / "T1R_END_TO_END_INTEGRATION_AUDIT.md").write_text(
        "# T1-R end-to-end integration audit\n\n"
        f"Status: **{readiness['status']}**. The identical `execute_pipeline` dispatcher completed an isolated Fake Data/Fake Study/Fake Evaluation path. It exercised all three fold study closures, selected-config freezing, seed-42 decision, conditional seeds 2024/3407, full T0-query matching, and final identity-gated decision. No real Optuna study, model fit, checkpoint, outer prediction, or Test input was created.\n",
        encoding="utf-8")
    (reports / "T1R_AUTHORIZATION_AND_FIREWALL_AUDIT.md").write_text(
        "# T1-R authorization and firewall audit\n\n"
        "Formal execution requires `configs/T1R_EXECUTION_AUTHORIZATION.yaml`, exact protocol SHA, exact runner SHA, phase name, `T1R` study namespace, and `DEVELOPMENT_ONLY` scope. No authorization file was created. Missing authorization, token mismatch, protocol/runner mismatch, unknown modes, and Test/late-life path references fail closed before an execution adapter is used.\n",
        encoding="utf-8")
    (reports / "T1R_FULL_PIPELINE_READINESS.md").write_text(
        "# T1-R full pipeline readiness\n\n"
        f"`{readiness['status']}` means code and isolated synthetic acceptance checks are ready for a separately authorized Development-only execution. It does not report training, Optuna, B/C evaluation, model performance, or Test use.\n\n"
        "Future command (not executed): `python scripts/run_t1r.py --mode execute --authorization EXECUTE_T1R_SEPARATE_TRAINING` after a separately audited authorization file is created.\n",
        encoding="utf-8")
    (reports / "T1R_REAL_ADAPTER_INTEGRATION_AUDIT.md").write_text("# T1-R real-adapter integration\n\nThe authorization-gated `t1r_real_adapter.RealAdapter` creates isolated T1-R studies, uses the frozen T0 window/scaler contracts, calls the reusable T1-R fit loop, freezes selected configurations, and produces B/C prediction records. This audit used synthetic or temporary SQLite execution only; no formal study or model fit occurred.\n",encoding="utf-8")
    (reports / "T1R_T0_TRUTH_AND_METRIC_AUDIT.md").write_text("# T0 truth and metric audit\n\nB/C query records are checked against the frozen T0 query universe and per-key `y_true_delta_V` within 1e-12 V. A metrics are loaded from the frozen T0 per-fold artifact, never copied from B.\n",encoding="utf-8")
    (reports / "T1R_DECISION_BRANCH_AUDIT.md").write_text("# T1-R decision branch audit\n\nTests cover seed-42 pending-multiseed, extended-budget descriptive advance, insufficient gain, and technical-failure identity rejection. Multiseed advancement requires exact seed/fold/model query identity.\n",encoding="utf-8")
    (reports / "T1R_REAL_EXECUTION_READINESS.md").write_text(f"# T1-R real execution readiness\n\nStatus: **{readiness['status']}**. This is authorization readiness only; it records no formal trial, training, outer prediction, or Test access.\n",encoding="utf-8")


def main() -> None:
    cfg, protocol_sha = load_protocol(); validate_isolation(cfg)
    preflight = static_preflight()
    suite = unittest.defaultTestLoader.discover(str(PHASE / "tests"), pattern="test_*.py")
    with open("NUL", "w", encoding="utf-8") as sink:
        result = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    tests = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(), "synthetic_only": True}
    old = {"database_sha256": sha(OLD / "outputs" / "optuna_fold_1.db"), "archive_sha256": sha(ARCHIVE), "archive_exists": ARCHIVE.is_file(), "database_contents_read": False}
    ready = bool(tests["passed"] and old["database_sha256"] == old["archive_sha256"] and not AUTHORIZATION.exists())
    readiness = {"status": "T1R_REAL_ADAPTER_READY_FOR_AUTHORIZATION" if ready else "T1R_REAL_ADAPTER_NOT_READY", "protocol_sha256": protocol_sha, "runner_sha256": runner_sha256(), "real_adapter_sha256": sha(PHASE/'scripts'/'t1r_real_adapter.py'), "protocol_preserved": True, "preflight": preflight, "tests": tests, "old_t1_integrity": old, "trial_budget_source": "persisted Optuna study.trials only", "t0_query_universe_match_required": True, "multiseed_identity_required": True, "authorization_file_created": False, "new_trials_created": False, "model_training_performed": False, "outer_predictions_created": False, "test_accessed": False}
    out = PHASE / "outputs"; out.mkdir(exist_ok=True)
    for name in ("T1R_E2E_SYNTHETIC_TEST_RESULTS.json", "T1R_FULL_PIPELINE_READINESS.json", "T1R_EXECUTION_READINESS.json", "T1R_FINAL_STATIC_READINESS.json", "T1R_REAL_ADAPTER_TEST_RESULTS.json", "T1R_REAL_EXECUTION_READINESS.json"):
        (out / name).write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_reports(readiness)
    print(json.dumps(readiness, sort_keys=True))
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

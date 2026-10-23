"""Durable, phase-global state machine for the formal AHC-TV Test opening.

The ledger is deliberately outside the runtime artifact directory.  Its
exclusive creation consumes the future Test opening before Reader M can touch
metadata, and it remains in place after every later failure.  The public CLI
does not expose the dependency-injection hooks in this module; they exist only
for synthetic state-machine tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_STARTED = "EXECUTION_STARTED_BEFORE_FIRST_TEST_IO"
STATUS_STAGE_P = "STAGE_P_FROZEN"
STATUS_TARGET_STARTED = "TARGET_UNBLINDING_STARTED"
STATUS_TARGETS = "TARGETS_UNBLINDED"
STATUS_DECISION = "DECISION_FROZEN"
STATUS_COMPLETED = "COMPLETED"
STATUS_ABORTED = "ABORTED_AFTER_TEST_EXECUTION_STARTED"

FORWARD_TRANSITIONS = {
    STATUS_STARTED: STATUS_STAGE_P,
    STATUS_STAGE_P: STATUS_TARGET_STARTED,
    STATUS_TARGET_STARTED: STATUS_TARGETS,
    STATUS_TARGETS: STATUS_DECISION,
    STATUS_DECISION: STATUS_COMPLETED,
}


class OneShotLedgerError(RuntimeError):
    """A phase-global one-shot invariant was violated."""


def canonical_runtime_root(phase: Path) -> Path:
    return Path(phase) / "runtime"


def canonical_run_dir(phase: Path) -> Path:
    return canonical_runtime_root(phase) / "ONE_SHOT_TEST_RUN"


def ledger_path(phase: Path) -> Path:
    return canonical_runtime_root(phase) / "AHC_TV_ONE_SHOT_EXECUTION_LEDGER.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(directory: Path) -> None:
    """Persist directory entry changes where the platform exposes directory FDs."""
    if os.name == "nt":
        return
    descriptor = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _replace_ledger(path: Path, payload: dict[str, Any]) -> None:
    """The sole allowed mutable file operation: an audited forward transition."""
    if not path.exists():
        raise OneShotLedgerError("AHC_TV_ONE_SHOT_LEDGER_MISSING")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("r+", encoding="utf-8", newline="\n") as handle:
        handle.seek(0)
        handle.write(encoded)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def read_ledger(phase: Path) -> dict[str, Any]:
    path = ledger_path(phase)
    if not path.exists():
        raise OneShotLedgerError("AHC_TV_ONE_SHOT_LEDGER_MISSING")
    return json.loads(path.read_text(encoding="utf-8"))


def create_execution_ledger(*, phase: Path, authorization_marker: str, base_commit: str | None = None) -> dict[str, Any]:
    """Create the permanent execution record using O_EXCL-equivalent mode ``x``."""
    phase = Path(phase)
    run_dir = canonical_run_dir(phase)
    ledger = ledger_path(phase)
    if ledger.exists():
        raise OneShotLedgerError("AHC_TV_ONE_SHOT_ALREADY_CONSUMED")
    if run_dir.exists():
        # A partial directory without a ledger is not safe to reuse.  It does
        # not consume the authorization, but it blocks Test I/O fail-closed.
        raise OneShotLedgerError("AHC_TV_ONE_SHOT_RUNTIME_PATH_ALREADY_EXISTS")
    if base_commit is None:
        repo = phase.parents[1]
        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    payload: dict[str, Any] = {
        "status": STATUS_STARTED,
        "base_commit": base_commit,
        "authorization_marker": authorization_marker,
        "started_at_utc": _utc_now(),
        "target_unblinded": False,
        "stage_p_frozen": False,
        "completed": False,
        "transition_history": [{"status": STATUS_STARTED, "at_utc": _utc_now()}],
    }
    try:
        _write_exclusive_json(ledger, payload)
    except FileExistsError as exc:
        raise OneShotLedgerError("AHC_TV_ONE_SHOT_ALREADY_CONSUMED") from exc
    return payload


def transition_execution_ledger(*, phase: Path, expected_status: str, next_status: str) -> dict[str, Any]:
    """Advance exactly one prescribed state; backward/skip transitions fail closed."""
    if FORWARD_TRANSITIONS.get(expected_status) != next_status:
        raise OneShotLedgerError("AHC_TV_ONE_SHOT_INVALID_STATE_TRANSITION")
    path = ledger_path(phase)
    payload = read_ledger(phase)
    if payload.get("status") != expected_status:
        raise OneShotLedgerError("AHC_TV_ONE_SHOT_INVALID_STATE_TRANSITION")
    payload["status"] = next_status
    payload["stage_p_frozen"] = bool(payload.get("stage_p_frozen") or next_status in {STATUS_STAGE_P, STATUS_TARGET_STARTED, STATUS_TARGETS, STATUS_DECISION, STATUS_COMPLETED})
    payload["target_unblinded"] = bool(payload.get("target_unblinded") or next_status in {STATUS_TARGETS, STATUS_DECISION, STATUS_COMPLETED})
    payload["completed"] = bool(next_status == STATUS_COMPLETED)
    payload.setdefault("transition_history", []).append({"status": next_status, "at_utc": _utc_now()})
    _replace_ledger(path, payload)
    return payload


def abort_execution_ledger(*, phase: Path, stage: str, exception: BaseException, target_unblinded: bool) -> dict[str, Any]:
    """Make a post-start failure permanent without permitting a retry."""
    path = ledger_path(phase)
    payload = read_ledger(phase)
    if payload.get("status") == STATUS_ABORTED:
        return payload
    if payload.get("status") not in {*FORWARD_TRANSITIONS, STATUS_COMPLETED}:
        raise OneShotLedgerError("AHC_TV_ONE_SHOT_INVALID_STATE_TRANSITION")
    if payload.get("status") == STATUS_COMPLETED:
        raise OneShotLedgerError("AHC_TV_ONE_SHOT_INVALID_STATE_TRANSITION")
    payload["status"] = STATUS_ABORTED
    payload["target_unblinded"] = bool(payload.get("target_unblinded") or target_unblinded)
    payload["completed"] = False
    payload["aborted_at_utc"] = _utc_now()
    payload["failure"] = {
        "stage": stage,
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "target_unblinded": bool(payload["target_unblinded"]),
    }
    payload.setdefault("transition_history", []).append({"status": STATUS_ABORTED, "at_utc": _utc_now()})
    _replace_ledger(path, payload)
    return payload


def write_exclusive_text(path: Path, text: str) -> None:
    """Durably create a formal artifact once; no overwrite path exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    _write_exclusive_json(Path(path), value)


def write_exclusive_sha256_sidecar(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    write_exclusive_text(path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n")
    return digest


def load_frozen_results_for_offline_postprocessing(*, phase: Path) -> list[dict[str, Any]]:
    """Read only the once-frozen result table; never call a polarization reader."""
    path = canonical_run_dir(phase) / "FROZEN_TEST_TARGETS_AND_ERRORS.csv"
    sidecar = path.with_name(path.name + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise OneShotLedgerError("AHC_TV_FROZEN_TARGET_RESULTS_UNAVAILABLE")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != observed:
        raise OneShotLedgerError("AHC_TV_FROZEN_TARGET_RESULTS_HASH_MISMATCH")
    import csv
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

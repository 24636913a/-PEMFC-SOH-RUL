"""Identity-first B/C evaluation contracts used before any T1-R decision."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import pandas as pd


class EvaluationIdentityError(RuntimeError):
    pass


FIELDS = {"seed", "model", "fold", "origin_row", "target_row", "horizon_step", "y_true_delta_V", "y_pred_delta_V", "protocol_sha256", "run_id"}


def load_frozen_t0_query_universe(t0_prediction_path: Path) -> list[dict[str, Any]]:
    """Load only frozen Development query identities and truths from the T0 artifact."""
    needed = ["fold", "origin_row", "target_row", "horizon_step", "y_true_delta_V"]
    frame = pd.read_csv(t0_prediction_path, usecols=needed)
    if frame.empty or set(frame.fold.astype(str)) != {"fold_1", "fold_2", "fold_3"}:
        raise EvaluationIdentityError("FROZEN_T0_QUERY_UNIVERSE_INVALID")
    return frame.to_dict(orient="records")


def query_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (str(row["fold"]), int(row["origin_row"]), int(row["target_row"]), int(row["horizon_step"]))


def _finite(row: dict[str, Any]) -> bool:
    try:
        return all(math.isfinite(float(row[k])) for k in ("y_true_delta_V", "y_pred_delta_V"))
    except (KeyError, TypeError, ValueError):
        return False


def validate_pair_rows(rows: list[dict[str, Any]], protocol_sha: str, t0_query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(FIELDS - set(row) for row in rows) or any(row["protocol_sha256"] != protocol_sha or not _finite(row) for row in rows):
        raise EvaluationIdentityError("PAIR_FIELDS_PROTOCOL_OR_FINITE_FAILURE")
    models = {row["model"] for row in rows}
    if models != {"B", "C"}:
        raise EvaluationIdentityError("PAIR_MODEL_SET_INVALID")
    frozen = {query_key(r) for r in t0_query_rows}
    if not frozen:
        raise EvaluationIdentityError("FROZEN_T0_QUERY_UNIVERSE_MISSING")
    by_model = {model: [r for r in rows if r["model"] == model] for model in ("B", "C")}
    truth_by_key: dict[tuple[Any, ...], float] = {}
    for model_rows in by_model.values():
        keys = [query_key(r) for r in model_rows]
        if len(keys) != len(set(keys)) or set(keys) != frozen:
            raise EvaluationIdentityError("PAIR_QUERY_UNIVERSE_MISMATCH")
        for row in model_rows:
            key, truth = query_key(row), float(row["y_true_delta_V"])
            if key in truth_by_key and truth_by_key[key] != truth:
                raise EvaluationIdentityError("PAIR_TRUTH_MISMATCH")
            truth_by_key[key] = truth
    counts = {fold: len({k[1] for k in frozen if k[0] == fold}) for fold in ("fold_1", "fold_2", "fold_3")}
    required = {"fold_1": 233, "fold_2": 245, "fold_3": 466}
    if counts != required:
        raise EvaluationIdentityError("FROZEN_T0_ORIGIN_COUNT_MISMATCH")
    return {"pass": True, "paired_queries": len(frozen), "origins_by_fold": counts}


def validate_multiseed_identity(rows: list[dict[str, Any]], protocol_sha: str, t0_query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(seed, model, fold) for seed in (42, 2024, 3407) for model in ("B", "C") for fold in ("fold_1", "fold_2", "fold_3")}
    group_keys = [(int(r.get("seed", -1)), str(r.get("model")), str(r.get("fold"))) for r in rows]
    groups = set(group_keys)
    if groups != expected:
        raise EvaluationIdentityError("MULTISEED_IDENTITY_INCOMPLETE_OR_DUPLICATE")
    # A group is allowed many horizon rows but every exact seed/model/fold set must
    # represent the frozen T0 query universe, which is checked seed-wise below.
    for seed in (42, 2024, 3407):
        validate_pair_rows([r for r in rows if int(r["seed"]) == seed], protocol_sha, t0_query_rows)
    return {"pass": True, "record_groups": len(groups), "seeds": [42, 2024, 3407]}


def validate_multiseed_summary(rows: list[dict[str, Any]], protocol_sha: str) -> dict[str, Any]:
    """Legacy summary validator retained for report-only callers; never a final gate."""
    expected = {(s, m, f) for s in (42, 2024, 3407) for m in ("B", "C") for f in ("fold_1", "fold_2", "fold_3")}
    seen = [(r.get("seed"), r.get("model"), r.get("fold")) for r in rows]
    if set(seen) != expected or len(seen) != len(expected) or any(r.get("protocol_sha256") != protocol_sha for r in rows):
        raise EvaluationIdentityError("MULTISEED_IDENTITY_INCOMPLETE_OR_DUPLICATE")
    return {"pass": True, "records": len(rows)}

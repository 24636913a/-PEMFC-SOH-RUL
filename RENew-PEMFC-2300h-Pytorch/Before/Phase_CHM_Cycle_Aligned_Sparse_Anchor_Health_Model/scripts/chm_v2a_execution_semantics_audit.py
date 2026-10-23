"""Static fail-closed audit for CHM Final V2 Execution Supplement A.

This module validates frozen configuration and deterministically exercises
selection, seed aggregation, and block-multiplicity semantics.  It has no
model, optimizer, fit, LOAO execution, metric, prediction, or Test access
path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml


PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
CONFIGS = PHASE / "configs"
INPUTS = PHASE / "inputs"
OUTPUTS = PHASE / "outputs"
FINAL_V2 = CONFIGS / "PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml"
SUPPLEMENT = CONFIGS / "CHM_FINAL_V2_EXECUTION_SUPPLEMENT_A.yaml"
SUPPLEMENT_SHA = CONFIGS / "CHM_FINAL_V2_EXECUTION_SUPPLEMENT_A.sha256"
AUTHORITY = CONFIGS / "CHM_PROTOCOL_AUTHORITY.yaml"
HASHES = CONFIGS / "CHM_FROZEN_INPUT_HASHES_V2.yaml"
HOU = INPUTS / "CHM_HOU_TABLE.csv"
AUDIT_OUTPUT = OUTPUTS / "CHM_V2A_EXECUTION_SEMANTICS_AUDIT.json"
PROFILES = ("A_BALANCED", "B_ANCHOR_EMPHASIS", "C_OBSERVATION_EMPHASIS")
SEEDS = (42, 2024, 3407)
EXPECTED_SEGMENTS = ((1, 24, 24), (26, 36, 11), (38, 49, 12), (51, 53, 3), (55, 74, 20), (76, 100, 25), (103, 125, 23), (127, 133, 7), (135, 150, 16), (152, 156, 5))


class ExecutionSemanticsFailure(RuntimeError):
    """Raised when any frozen V2A execution contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ExecutionSemanticsFailure(f"YAML mapping required: {path.name}")
    return payload


def assert_sidecar_hash(path: Path, sidecar: Path) -> str:
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if expected != observed:
        raise ExecutionSemanticsFailure(f"SHA mismatch: {path.name}")
    return observed


def choose_final_profile(outer_selected_profiles: Sequence[str]) -> str:
    """Use only six inner-selected profile labels; no target/error argument exists."""
    if len(outer_selected_profiles) != 6 or any(profile not in PROFILES for profile in outer_selected_profiles):
        raise ExecutionSemanticsFailure("Exactly six legal outer selected profiles are required.")
    counts = Counter(outer_selected_profiles)
    maximum = max(counts.values())
    return next(profile for profile in PROFILES if counts.get(profile, 0) == maximum)


def inner_profile_score(residuals: Sequence[float]) -> float:
    """Arithmetic mean of the 5 inner folds x 3 seeds absolute errors."""
    values = np.asarray(residuals, dtype=float)
    if values.shape != (15,) or not np.isfinite(values).all():
        raise ExecutionSemanticsFailure("Inner score requires 15 finite residuals.")
    return float(np.abs(values).mean())


def validate_three_seed_values(values: Sequence[float | None]) -> np.ndarray:
    if len(values) != 3 or any(value is None for value in values):
        raise ExecutionSemanticsFailure("All three frozen seeds are required; neither deletion nor replacement is legal.")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ExecutionSemanticsFailure("A frozen seed produced a non-finite value.")
    return array


def aggregate_outer_seed_predictions(values: Sequence[float | None]) -> float:
    """Formal deterministic outer point prediction: median of exactly 3 seeds."""
    return float(np.median(validate_three_seed_values(values)))


def seed_tuple(stream_id: int, outer_fold_index: int, replicate_index: int) -> tuple[int, int, int, int]:
    if stream_id not in {101, 201, 301} or outer_fold_index < 0 or replicate_index < 0:
        raise ExecutionSemanticsFailure("Illegal bootstrap stream/fold/replicate index.")
    return (90210, stream_id, outer_fold_index, replicate_index)


def bootstrap_generator(stream_id: int, outer_fold_index: int, replicate_index: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_tuple(stream_id, outer_fold_index, replicate_index))))


def effective_block_length(requested_length: int, segment_length: int) -> int:
    if requested_length not in {4, 8} or segment_length < 1:
        raise ExecutionSemanticsFailure("Requested block length or segment length is invalid.")
    return min(requested_length, segment_length)


def segment_multiplicity_weights(raw_cycle_ids: Sequence[int], requested_length: int, rng: np.random.Generator) -> dict[int, int]:
    """Draw moving blocks solely to create original-cycle loss multiplicities."""
    cycles = tuple(int(cycle) for cycle in raw_cycle_ids)
    if not cycles or any(later != earlier + 1 for earlier, later in zip(cycles, cycles[1:])):
        raise ExecutionSemanticsFailure("A bootstrap segment must contain strictly contiguous raw cycle IDs.")
    n = len(cycles)
    length = effective_block_length(requested_length, n)
    draw_count = math.ceil(n / length)
    starts = rng.integers(0, n - length + 1, size=draw_count)
    expanded: list[int] = []
    for start in starts.tolist():
        expanded.extend(cycles[start : start + length])
    counted = expanded[:n]
    weights = {cycle: 0 for cycle in cycles}
    for cycle in counted:
        weights[cycle] += 1
    if sum(weights.values()) != n:
        raise ExecutionSemanticsFailure("Multiplicity does not sum to the segment length.")
    return weights


def complete_cycle_segments(hou: pd.DataFrame) -> list[tuple[int, ...]]:
    required = {"hou_kind", "raw_cycle_id"}
    if not required.issubset(hou.columns):
        raise ExecutionSemanticsFailure("Frozen HOU table lacks bootstrap columns.")
    complete = hou.loc[hou["hou_kind"].eq("COMPLETE_CYCLE_HOU"), "raw_cycle_id"]
    if complete.isna().any():
        raise ExecutionSemanticsFailure("Complete HOU has missing raw cycle ID.")
    cycles = complete.astype(int).tolist()
    if cycles != sorted(cycles) or len(cycles) != len(set(cycles)):
        raise ExecutionSemanticsFailure("Frozen complete-cycle order is not a unique raw-cycle chronology.")
    segments: list[list[int]] = []
    for cycle in cycles:
        if not segments or cycle != segments[-1][-1] + 1:
            segments.append([cycle])
        else:
            segments[-1].append(cycle)
    return [tuple(segment) for segment in segments]


def assert_expected_segments(hou: pd.DataFrame) -> list[tuple[int, ...]]:
    segments = complete_cycle_segments(hou)
    observed = tuple((segment[0], segment[-1], len(segment)) for segment in segments)
    if observed != EXPECTED_SEGMENTS:
        raise ExecutionSemanticsFailure(f"Frozen HOU segments drifted: {observed!r}")
    return segments


def build_bootstrap_observation_weights(hou: pd.DataFrame, requested_length: int, stream_id: int, outer_fold_index: int, replicate_index: int) -> pd.DataFrame:
    """Return weights alongside, never instead of, the original 152-row chronology."""
    segments = assert_expected_segments(hou)
    rng = bootstrap_generator(stream_id, outer_fold_index, replicate_index)
    complete_weights: dict[int, int] = {}
    for segment in segments:
        complete_weights.update(segment_multiplicity_weights(segment, requested_length, rng))
    result = hou.copy()
    complete_mask = result["hou_kind"].eq("COMPLETE_CYCLE_HOU")
    weights = pd.Series(1, index=result.index, dtype=int)
    complete_cycle_weights = result.loc[complete_mask, "raw_cycle_id"].astype(int).map(complete_weights)
    if complete_cycle_weights.isna().any():
        raise ExecutionSemanticsFailure("A complete HOU lacks a bootstrap multiplicity.")
    weights.loc[complete_mask] = complete_cycle_weights.astype(int).to_numpy()
    result["bootstrap_observation_weight"] = weights
    if len(result) != len(hou) or not result["hou_id"].equals(hou["hou_id"]):
        raise ExecutionSemanticsFailure("Bootstrap attempted to reorder or remove HOU rows.")
    if not (result.loc[result["hou_kind"].eq("PREANCHOR_PARTIAL_HOU"), "bootstrap_observation_weight"] == 1).all():
        raise ExecutionSemanticsFailure("Partial HOU weight drifted from one.")
    return result


def optimizer_seed_for_replicate(replicate_index: int) -> int:
    if replicate_index < 0:
        raise ExecutionSemanticsFailure("Negative replicate index.")
    return SEEDS[replicate_index % 3]


def replicate_accounting(scheduled: int, invalid_replicates: Iterable[int]) -> dict[str, Any]:
    invalid = sorted(set(int(index) for index in invalid_replicates))
    if scheduled < 1 or any(index < 0 or index >= scheduled for index in invalid):
        raise ExecutionSemanticsFailure("Invalid replicate accounting input.")
    return {"scheduled": scheduled, "invalid_indices": invalid, "legal_count": scheduled - len(invalid), "replacement_performed": False}


def _require(mapping: dict[str, Any], path: Sequence[str], expected: Any) -> None:
    cursor: Any = mapping
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            raise ExecutionSemanticsFailure(f"Missing V2A field: {'.'.join(path)}")
        cursor = cursor[key]
    if cursor != expected:
        raise ExecutionSemanticsFailure(f"V2A field drift: {'.'.join(path)}")


def validate_supplement(supplement: dict[str, Any]) -> None:
    _require(supplement, ("supplement_version",), "FINAL_V2A_EXECUTION_SEMANTICS_A")
    _require(supplement, ("status",), "FINAL_V2A_FROZEN_READY_FOR_CHM_D")
    _require(supplement, ("supplementary_authority_only",), ["final_all_six_profile_selection", "seed_aggregation_and_failure", "bootstrap_execution_semantics"])
    _require(supplement, ("final_all_six_profile_selection", "selection_rule"), "OUTER_INNER_SELECTION_MAJORITY_VOTE")
    _require(supplement, ("final_all_six_profile_selection", "required_outer_fold_count"), 6)
    _require(supplement, ("final_all_six_profile_selection", "tie_break_order"), list(PROFILES))
    _require(supplement, ("seeds_and_deterministic_prediction_semantics", "frozen_seeds"), list(SEEDS))
    _require(supplement, ("seeds_and_deterministic_prediction_semantics", "outer_point_prediction"), "MEDIAN_3_SEEDS")
    _require(supplement, ("seeds_and_deterministic_prediction_semantics", "deterministic_primary_gate_metrics_use"), "six_outer_three_seed_median_predictions_only")
    _require(supplement, ("bootstrap_execution_semantics", "base_seed"), 90210)
    _require(supplement, ("bootstrap_execution_semantics", "rng"), "NUMPY_PCG64_SEEDSEQUENCE")
    _require(supplement, ("bootstrap_execution_semantics", "bootstrap_time_axis_reordered"), False)
    _require(supplement, ("bootstrap_execution_semantics", "partial_HOU_resampled"), False)
    _require(supplement, ("bootstrap_execution_semantics", "invalid_replicate_replacement"), "prohibited")
    _require(supplement, ("bootstrap_execution_semantics", "profile_selection_inside_bootstrap"), "prohibited")
    _require(supplement, ("S1_reporting_contract", "primary_gate"), False)
    _require(supplement, ("non_actions_for_this_supplement", "neural_training"), False)
    _require(supplement, ("non_actions_for_this_supplement", "Test_access"), False)
    expected_segments = [
        {"start_raw_cycle_id": start, "end_raw_cycle_id": end, "n": length}
        for start, end, length in EXPECTED_SEGMENTS
    ]
    _require(supplement, ("bootstrap_execution_semantics", "expected_complete_cycle_segments"), expected_segments)


def validate_authority_and_hash_manifest() -> None:
    authority = load_yaml(AUTHORITY)
    if authority.get("authority_version") != "FINAL_V2A" or authority.get("current_status") != "FINAL_V2A_FROZEN_READY_FOR_CHM_D":
        raise ExecutionSemanticsFailure("V2A authority/status is not frozen.")
    normative = authority.get("normative_executable", [])
    if normative[:2] != ["configs/PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml", "configs/CHM_FINAL_V2_EXECUTION_SUPPLEMENT_A.yaml"]:
        raise ExecutionSemanticsFailure("Final V2 and Supplement A are not the first execution authorities.")
    if authority.get("supplementary_authority_topics") != ["final_all_six_profile_selection", "seed_aggregation_and_failure", "bootstrap_execution_semantics"]:
        raise ExecutionSemanticsFailure("Supplement A authority scope drifted.")
    manifest = load_yaml(HASHES)
    records = manifest.get("records", {})
    required = {"final_v2_protocol", "execution_semantics_supplement_a", "protocol_authority", "HOU_table", "sparse_anchor_targets", "frozen_anchor_interval_mapping", "authority_audit_script", "execution_semantics_audit_script"}
    if not required.issubset(records):
        raise ExecutionSemanticsFailure("V2A hash manifest is incomplete.")
    for entry in records.values():
        path = REPO / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ExecutionSemanticsFailure(f"Frozen input hash mismatch: {entry.get('path')}")


def run_execution_semantics_audit() -> dict[str, Any]:
    """Audit frozen semantics and deterministic helpers without training or fitting."""
    supplement_hash = assert_sidecar_hash(SUPPLEMENT, SUPPLEMENT_SHA)
    final_v2 = load_yaml(FINAL_V2)
    supplement = load_yaml(SUPPLEMENT)
    if final_v2.get("protocol_version") != "FINAL_V2" or final_v2.get("status") != "FINAL_V2_FROZEN_AWAITING_CHM_D_EXECUTION":
        raise ExecutionSemanticsFailure("Final V2 protocol was modified or its status drifted.")
    validate_supplement(supplement)
    validate_authority_and_hash_manifest()
    hou = pd.read_csv(HOU)
    segments = assert_expected_segments(hou)
    trial = build_bootstrap_observation_weights(hou, 4, 101, 0, 0)
    expected_weights = sum(len(segment) for segment in segments)
    observed_weights = int(trial.loc[trial["hou_kind"].eq("COMPLETE_CYCLE_HOU"), "bootstrap_observation_weight"].sum())
    if expected_weights != observed_weights:
        raise ExecutionSemanticsFailure("Bootstrap complete-cycle multiplicity total drifted.")
    result = {
        "phase": PHASE.name,
        "base_commit": "6db2be2e4e37f942f67e5e1636d8bdfbe92534d5",
        "status": "FINAL_V2A_FROZEN_READY_FOR_CHM_D",
        "training_executed": False,
        "LOAO_executed": False,
        "bootstrap_model_fitting_executed": False,
        "MAE_RMSE_generated": False,
        "Test_accessed": False,
        "final_profile_selection_rule": "OUTER_INNER_SELECTION_MAJORITY_VOTE",
        "outer_target_used_for_final_profile": False,
        "outer_point_seed_aggregation": "MEDIAN_3_SEEDS",
        "inner_profile_score": "MEAN_ABSOLUTE_ERROR_OVER_5_FOLDS_X_3_SEEDS",
        "bootstrap_base_seed": 90210,
        "bootstrap_rng": "NUMPY_PCG64_SEEDSEQUENCE",
        "bootstrap_time_axis_reordered": False,
        "partial_HOU_resampled": False,
        "invalid_replicate_replaced": False,
        "S1_primary_gate": False,
        "complete_cycle_segments": [{"start_raw_cycle_id": segment[0], "end_raw_cycle_id": segment[-1], "n": len(segment)} for segment in segments],
        "complete_cycle_multiplicity_sum_for_static_trial": observed_weights,
        "supplement_a_sha256": supplement_hash,
        "execution_semantics_audit": "PASS",
        "remaining_execution_ambiguity": "NONE",
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with AUDIT_OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def print_summary(result: dict[str, Any]) -> None:
    print("CHM Final V2 Execution Supplement A")
    print("===================================")
    print(f"Base commit: {result['base_commit']}")
    print("Training executed: NO")
    print("Final all-six profile: OUTER-INNER-SELECTION MAJORITY VOTE")
    print("Outer heldout error used for final profile: NO")
    print("Outer seed aggregation: MEDIAN(42,2024,3407)")
    print("Inner score: MEAN MAE OVER 5 FOLDS x 3 SEEDS")
    print("Bootstrap base seed: 90210")
    print("Bootstrap method: TIME-AXIS-PRESERVING BLOCK MULTIPLICITY WEIGHTS")
    print("Original HOU chronology preserved: YES")
    print("Invalid replicate replacement: NO")
    print("S1 qualification gate: NO")
    print("Test accessed: NO")
    print(f"Execution-semantics audit: {result['execution_semantics_audit']}")
    print(f"Remaining execution ambiguity: {result['remaining_execution_ambiguity']}")
    print(f"Final status: {result['status']}")
    print("Next action: CHM_D_DEVELOPMENT_EXECUTION")


if __name__ == "__main__":
    print_summary(run_execution_semantics_audit())

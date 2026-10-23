"""Static consistency checks for the non-executable CHM candidate protocol.

This module reads only frozen RCR/PAA development artifacts. It has no model,
optimizer, raw-data loader, polarization loader, or Test-file access path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


EXPECTED_PARTIAL_IDS = (
    "PARTIAL_PRE_300",
    "PARTIAL_PRE_600",
    "PARTIAL_PRE_850",
    "PARTIAL_PRE_1150",
    "PARTIAL_PRE_1500",
    "PARTIAL_PRE_1750",
)
ALL_ANCHORS = (300, 600, 850, 1150, 1500, 1750)


class StaticProtocolFailure(RuntimeError):
    """Raised when a frozen source and the candidate protocol disagree."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def phase_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_paths() -> dict[str, Path]:
    repo = repository_root()
    return {
        "candidate": phase_root() / "configs" / "PREREGISTERED_CHM_PROTOCOL_CANDIDATE_V1.yaml",
        "rcr_catalog": repo / "RENew-PEMFC-2300h-Pytorch" / "Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit" / "outputs" / "RAW_CYCLE_CATALOG.csv",
        "paa_complete": repo / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs" / "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv",
        "paa_partial": repo / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs" / "PREANCHOR_PARTIAL_CYCLE_INTERVALS.csv",
        "paa_alignment": repo / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAA_P1_StateToken_PreAnchor_Alignment" / "outputs" / "P1S_ANCHOR_ALIGNMENT.csv",
    }


def load_candidate() -> dict:
    path = source_paths()["candidate"]
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["status"] != "TECHNICALLY_RESOLVED_AWAITING_FINAL_FREEZE":
        raise StaticProtocolFailure("Candidate status is not technically resolved and awaiting final freeze.")
    if config["execution_authorized"]:
        raise StaticProtocolFailure("Candidate must not authorize execution.")
    return config


def reject_chm_test_path(path: str | Path) -> None:
    """CHM must fail closed before any Test path can be opened."""
    value = str(path).lower().replace("\\", "/")
    forbidden = ("/test/", "test_", "2000h", "2300h", "after-air-air", "afterairstart", "/eis/")
    if any(token in value for token in forbidden):
        raise StaticProtocolFailure(f"CHM_TEST_ACCESS_PROHIBITED: {path}")


def frozen_complete_cycles() -> pd.DataFrame:
    catalog = pd.read_csv(source_paths()["rcr_catalog"])
    complete = catalog.loc[catalog["complete_cycle"].astype(bool)].copy()
    if len(complete) != 146:
        raise StaticProtocolFailure(f"Expected 146 frozen complete cycles, found {len(complete)}.")
    if complete["raw_cycle_id"].duplicated().any():
        raise StaticProtocolFailure("Frozen complete-cycle IDs are duplicated.")
    return complete.sort_values("raw_cycle_id", kind="stable").reset_index(drop=True)


def frozen_partial_hou() -> pd.DataFrame:
    partial = pd.read_csv(source_paths()["paa_partial"])
    partial = partial.loc[partial["stable_window"].eq("middle_50")].copy()
    observed = tuple(sorted(partial["cycle_interval_id"].astype(str)))
    if observed != tuple(sorted(EXPECTED_PARTIAL_IDS)):
        raise StaticProtocolFailure(f"Partial HOU IDs changed: {observed}.")
    if len(partial) != 6:
        raise StaticProtocolFailure("Exactly six frozen partial HOU are required.")
    return partial


def known_measured_hou_time_order() -> pd.DataFrame:
    """Audit time order of all PAA-measured HOU currently available.

    PAA's existing complete-measurement artifact stops at its own pre-anchor
    firewall and therefore is not the future 146-cycle CHM table. This static
    audit validates the measurement-time ordering rule on all frozen PAA HOU
    already available; future execution must create the remaining complete HOU
    with the same rule before any fit.
    """
    complete = pd.read_csv(source_paths()["paa_complete"])
    complete = complete.loc[complete["stable_window"].eq("middle_50")].copy()
    partial = frozen_partial_hou().copy()
    records = pd.concat([complete, partial], ignore_index=True)
    records["measurement_end_datetime"] = pd.to_datetime(
        records["measurement_end_datetime"], errors="coerce"
    )
    if records["measurement_end_datetime"].isna().any():
        raise StaticProtocolFailure("Known PAA HOU contains missing measurement_end_datetime.")
    if records["measurement_end_datetime"].duplicated().any():
        raise StaticProtocolFailure("HOU time order is not strict because timestamps are duplicated.")
    return records.sort_values("measurement_end_datetime", kind="stable").reset_index(drop=True)


def assert_strict_datetime_order(records: pd.DataFrame) -> None:
    values = pd.to_datetime(records["measurement_end_datetime"], errors="coerce")
    if values.isna().any() or not values.is_monotonic_increasing or values.duplicated().any():
        raise StaticProtocolFailure("HOU must be strictly ordered by measurement_end_datetime.")


def selected_anchor_ids() -> dict[int, tuple[str, str, str]]:
    alignment = pd.read_csv(source_paths()["paa_alignment"])
    summaries = alignment.loc[alignment["record_type"].eq("anchor_summary")]
    result: dict[int, tuple[str, str, str]] = {}
    for row in summaries.itertuples(index=False):
        checkpoint = int(row.checkpoint_h)
        ids = tuple(str(row.selected_cycle_interval_ids).split("|"))
        if checkpoint not in ALL_ANCHORS or len(ids) != 3:
            raise StaticProtocolFailure(f"Anchor {checkpoint} does not have exactly three frozen HOU IDs.")
        result[checkpoint] = ids
    if tuple(sorted(result)) != ALL_ANCHORS:
        raise StaticProtocolFailure("Frozen PAA anchor set changed.")
    return result


def anchor_prediction_from_three_hou_states(states: Iterable[float]) -> float:
    values = np.asarray(list(states), dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise StaticProtocolFailure("Anchor prediction requires exactly three finite frozen HOU states.")
    return float(np.median(values))


def outer_training_anchor_ids(heldout_checkpoint_h: int) -> tuple[int, ...]:
    if heldout_checkpoint_h not in ALL_ANCHORS:
        raise StaticProtocolFailure("Outer holdout must be one of the six non-BoL anchors.")
    return tuple(anchor for anchor in ALL_ANCHORS if anchor != heldout_checkpoint_h)


def online_prefix_anchor_ids(checkpoint_h: int) -> tuple[int, ...]:
    if checkpoint_h not in ALL_ANCHORS:
        raise StaticProtocolFailure("Online checkpoint must be one of the six non-BoL anchors.")
    return tuple(anchor for anchor in ALL_ANCHORS if anchor < checkpoint_h)


def attach_block_weights_without_reordering(
    ordered_hou_ids: Iterable[str], integer_weights: dict[str, int]
) -> pd.DataFrame:
    """Attach bootstrap weights to the original HOU order; never resample rows."""
    ids = list(ordered_hou_ids)
    if len(ids) != len(set(ids)):
        raise StaticProtocolFailure("HOU IDs must be unique before bootstrap weighting.")
    weights = [integer_weights.get(hou_id, 0) for hou_id in ids]
    if any(int(weight) != weight or weight < 0 for weight in weights):
        raise StaticProtocolFailure("Block weights must be non-negative integers.")
    return pd.DataFrame({"hou_id": ids, "bootstrap_weight": weights})


def run_static_audit() -> dict[str, int | str]:
    config = load_candidate()
    complete = frozen_complete_cycles()
    partial = frozen_partial_hou()
    known = known_measured_hou_time_order()
    assert_strict_datetime_order(known)
    anchors = selected_anchor_ids()
    if "PARTIAL_PRE_600" not in anchors[600]:
        raise StaticProtocolFailure("600 h anchor must include PARTIAL_PRE_600.")
    if config["anchor_functional"]["formal_prediction"] != "median_of_three_frozen_HOU_latent_states":
        raise StaticProtocolFailure("Candidate anchor functional is not the frozen median-of-three-HOU rule.")
    return {
        "candidate_status": config["status"],
        "complete_cycle_HOU": len(complete),
        "partial_preanchor_HOU": len(partial),
        "known_timestamped_HOU_checked": len(known),
        "anchor_mapping": "MEDIAN_OF_3_FROZEN_HOU_STATES",
    }


if __name__ == "__main__":
    for key, value in run_static_audit().items():
        print(f"{key}: {value}")

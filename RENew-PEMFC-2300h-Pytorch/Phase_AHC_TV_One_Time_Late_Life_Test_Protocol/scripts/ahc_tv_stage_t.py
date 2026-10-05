"""Stage T: the sole paired target reader for a completed AHC-TV Stage P run."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from ahc_tv_alignment import (
    PHASE,
    TARGET_COLUMNS,
    assert_runtime_input_allowed,
    authorize_and_verify_target_unblinding,
    read_yaml,
    require_manual_test_authorization,
)
from ahc_tv_stage_p import bind_both_polarization_sources


class StageTError(RuntimeError):
    """The atomic two-checkpoint target unblinding could not proceed."""


def _log(events: list[str] | None, event: str) -> None:
    if events is not None:
        events.append(event)


def _target_from_table(table: pd.DataFrame, *, bol_voltage: float) -> float:
    if tuple(table.columns) != TARGET_COLUMNS:
        raise StageTError("TARGET_READER_COLUMN_ISOLATION_FAILED")
    density = pd.to_numeric(table["j.S.act [A/cm2]"], errors="coerce")
    active = table.loc[:, TARGET_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    voltage = active.mean(axis=1, skipna=False)
    selected = voltage.loc[density.sub(1.0).abs().le(0.01) & voltage.notna()]
    if selected.empty:
        raise StageTError("TEST_TARGET_JSTAR_ROWS_UNAVAILABLE")
    return float(selected.median() / float(bol_voltage))


def _read_one_target(
    path: Path,
    *,
    read_csv: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    return read_csv(
        assert_runtime_input_allowed(Path(path)),
        sep=";", encoding="utf-8-sig", usecols=list(TARGET_COLUMNS),
    )


def unblind_both_test_targets_once(
    *,
    polarization_sources: Mapping[int, Path] | Iterable[Path],
    frozen_prediction_path: Path,
    authorization_token: str,
    phase: Path = PHASE,
    event_log: list[str] | None = None,
    read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
) -> dict:
    """Read 2000 and 2300 target columns only after one unified gate passes.

    Binding both source paths, validating the frozen prediction/provenance, and
    replaying Development all occur before either target source is opened.
    Thus a 2000 result cannot cause an operator-controlled pause or change
    before the paired 2300 target is read.
    """
    require_manual_test_authorization(authorization_token)
    phase = Path(phase)
    sources = bind_both_polarization_sources(polarization_sources, phase=phase)
    prediction = authorize_and_verify_target_unblinding(
        frozen_prediction_path, authorization_token=authorization_token, phase=phase
    )
    alignment_inputs = read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    bol_voltage = float(alignment_inputs["polarization_target_normalization"]["V_pol_BoL_jstar_1_0"])
    targets: dict[int, float] = {}
    for checkpoint in (2000, 2300):
        target_table = _read_one_target(sources[checkpoint], read_csv=read_csv)
        _log(event_log, f"TARGET_{checkpoint}_READ")
        targets[checkpoint] = _target_from_table(target_table, bol_voltage=bol_voltage)
    frame = prediction.copy()
    frame["target_Y"] = frame["checkpoint"].map(targets).astype(float)
    frame["AE_P1S"] = (frame["P1S_affine_prediction"] - frame["target_Y"]).abs()
    frame["AE_Age"] = (frame["nominal_age_prediction"] - frame["target_Y"]).abs()
    frame["AE_Raw_P1S"] = (frame["raw_P1S_prediction"] - frame["target_Y"]).abs()
    by_checkpoint = frame.set_index("checkpoint")
    rules = {
        "A": bool(by_checkpoint.loc[2000, "AE_P1S"] < by_checkpoint.loc[2000, "AE_Age"]),
        "B": bool(by_checkpoint.loc[2300, "AE_P1S"] < by_checkpoint.loc[2300, "AE_Age"]),
        "C": bool(frame["AE_P1S"].mean() < frame["AE_Age"].mean()),
    }
    decision = (
        "LATE_LIFE_TEST_SUPPORTS_INCREMENTAL_P1S_CALIBRATION"
        if all(rules.values())
        else "LATE_LIFE_TEST_DOES_NOT_SUPPORT_INCREMENTAL_P1S_CALIBRATION"
    )
    _log(event_log, "DECISION_COMPUTED")
    return {"target_results": frame, "rules": rules, "decision": decision}

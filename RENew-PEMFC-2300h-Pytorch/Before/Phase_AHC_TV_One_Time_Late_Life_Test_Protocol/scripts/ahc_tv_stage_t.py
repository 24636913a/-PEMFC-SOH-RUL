"""Stage T: the sole paired target reader for a completed AHC-TV Stage P run."""
from __future__ import annotations

import warnings
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


def extract_jstar_voltage_from_table(table: pd.DataFrame) -> tuple[float, int]:
    """Frozen 10-active-cell nanmean/nanmedian extractor used by Stage T.

    It mirrors the Development polarization reader: one missing active cell
    does not invalidate a row, while an all-missing active-cell row contributes
    no voltage.  The returned count is the number of finite j*=1.0 local rows.
    """
    if tuple(table.columns) != TARGET_COLUMNS:
        raise StageTError("TARGET_READER_COLUMN_ISOLATION_FAILED")
    density = pd.to_numeric(table["j.S.act [A/cm2]"], errors="coerce")
    active = table.loc[:, TARGET_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    cells = active.to_numpy(dtype=float)
    all_missing = np.isnan(cells).all(axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        voltage = np.nanmean(cells, axis=1)
    voltage[all_missing] = np.nan
    voltage = pd.Series(voltage, index=table.index, dtype=float)
    selected = voltage.loc[density.sub(1.0).abs().le(0.01) & voltage.notna()]
    if selected.empty:
        raise StageTError("TEST_TARGET_JSTAR_ROWS_UNAVAILABLE")
    return float(np.nanmedian(selected.to_numpy(dtype=float))), int(len(selected))


def _target_from_table(table: pd.DataFrame, *, bol_voltage: float) -> float:
    voltage, _ = extract_jstar_voltage_from_table(table)
    return float(voltage / float(bol_voltage))


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
    before_first_target_read: Callable[[], None] | None = None,
    synthetic_test_mode: bool = False,
) -> dict:
    """Read 2000 and 2300 target columns only after one unified gate passes.

    Binding both source paths, validating the frozen prediction/provenance, and
    replaying Development all occur before either target source is opened.
    Thus a 2000 result cannot cause an operator-controlled pause or change
    before the paired 2300 target is read.
    """
    require_manual_test_authorization(authorization_token)
    phase = Path(phase)
    sources = bind_both_polarization_sources(
        polarization_sources, phase=phase, synthetic_test_mode=synthetic_test_mode
    )
    prediction = authorize_and_verify_target_unblinding(
        frozen_prediction_path, authorization_token=authorization_token, phase=phase
    )
    alignment_inputs = read_yaml(phase / "configs" / "AHC_TV_FROZEN_ALIGNMENT_INPUTS.yaml")
    bol_voltage = float(alignment_inputs["polarization_target_normalization"]["V_pol_BoL_jstar_1_0"])
    targets: dict[int, float] = {}
    if before_first_target_read is not None:
        before_first_target_read()
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

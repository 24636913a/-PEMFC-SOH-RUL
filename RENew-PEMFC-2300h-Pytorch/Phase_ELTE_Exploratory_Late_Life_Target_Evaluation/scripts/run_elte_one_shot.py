"""One-shot, explicitly exploratory 2000/2300 h target evaluation.

This runner has exactly one target-I/O path.  It first freezes all predictions
and provenance, creates its own ledger, then reads both nominal targets in one
uninterrupted loop.  An existing ledger is a permanent fail-closed stop.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Importing frozen helpers must not refresh tracked bytecode inside their phase.
sys.dont_write_bytecode = True


REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "RENew-PEMFC-2300h-Pytorch"
PHASE = PROJECT / "Phase_ELTE_Exploratory_Late_Life_Target_Evaluation"
AHC = PROJECT / "Phase_AHC_TV_One_Time_Late_Life_Test_Protocol"
POSTMORTEM = PROJECT / "Phase_AHC_TV_Target_Blind_Alignment_Postmortem"
CNP1S = PROJECT / "Phase_CNP1S_Condition_Normalized_P1S_Development_Study"
CONFIG = PHASE / "configs" / "ELTE_FROZEN_PROTOCOL.yaml"
PREDICTIONS = PHASE / "inputs" / "FROZEN_ELTE_PREDICTIONS_BEFORE_TARGET_OPENING.csv"
PROVENANCE = PHASE / "inputs" / "ELTE_PREDICTION_PROVENANCE.json"
LEDGER = PHASE / "runtime" / "ELTE_ONE_SHOT_LEDGER.json"
RESULTS = PHASE / "outputs" / "ELTE_TARGET_RESULTS.csv"
SUMMARY = PHASE / "outputs" / "ELTE_SUMMARY.json"
EXPOSURE = PHASE / "runtime" / "PROJECT_TARGET_EXPOSURE.json"
REPORT = PHASE / "reports" / "EXPLORATORY_LATE_LIFE_TARGET_EVALUATION.md"
FIGURE = PHASE / "figures" / "Development_to_LateLife_Health_Trajectory.png"
AUTHORIZATION = "OPEN_EXPLORATORY_LATE_LIFE_TEST"
BASE_TRACKED_COMMIT = "184433d2f392c164920e39c61cc9ed106e04f4e6"

DATA_ROOT = Path(
    "E:/沈农信电/课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命/数据/"
    "《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据"
)


class ELTEError(RuntimeError):
    """Raised when the one-shot contract cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_sha256_sidecar(path: Path) -> str:
    digest = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ELTEError(f"YAML_MAPPING_REQUIRED:{path}")
    return loaded


def _git(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _assert_start_state() -> str:
    head = _git(["git", "rev-parse", "HEAD"])
    remote = _git(["git", "rev-parse", "origin/main"])
    status = _git(["git", "status", "--short"])
    if head != remote or head != BASE_TRACKED_COMMIT:
        raise ELTEError("ELTE_BASE_COMMIT_OR_REMOTE_MISMATCH")
    allowed_untracked = (
        "RENew-PEMFC-2300h-Pytorch/Phase_ELTE_Exploratory_Late_Life_Target_Evaluation/",
        "RENew-PEMFC-2300h-Pytorch/Phase_IVRA_Independent_Validation_Resource_Audit/",
    )
    for line in filter(None, status.splitlines()):
        path = line[3:].replace("\\", "/")
        if not (line.startswith("?? ") and any(path.startswith(prefix) for prefix in allowed_untracked)):
            raise ELTEError(f"TRACKED_OR_UNRELATED_WORKTREE_CHANGE:{line}")
    if not _git(["git", "merge-base", "--is-ancestor", BASE_TRACKED_COMMIT, head]) == "":
        raise ELTEError("BASE_ANCESTRY_CHECK_FAILED")
    if not (AHC / "runtime" / "AHC_TV_ONE_SHOT_EXECUTION_LEDGER.json").is_file():
        raise ELTEError("HISTORICAL_AHC_LEDGER_MISSING")
    if LEDGER.exists():
        raise ELTEError("ELTE_ONE_SHOT_ALREADY_CONSUMED")
    return head


def _load_stage_t_symbols() -> tuple[Any, Any, Any]:
    """Import frozen source binding and target extraction without running AHC-TV."""
    scripts = AHC / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from ahc_tv_stage_p import verify_frozen_polarization_source_identity
        from ahc_tv_stage_t import TARGET_COLUMNS, extract_jstar_voltage_from_table
    finally:
        sys.path.pop(0)
    return verify_frozen_polarization_source_identity, TARGET_COLUMNS, extract_jstar_voltage_from_table


def _selected_intervals(checkpoint: int, protocol: dict[str, Any]) -> pd.DataFrame:
    table = pd.read_csv(POSTMORTEM / "outputs" / f"AHC_TV_ALIGNMENT_POSTMORTEM_INTERVALS_{checkpoint}.csv")
    wanted = list(protocol["selection"][checkpoint]["interval_ids"])
    chosen = table.loc[table["cycle_interval_id"].isin(wanted)].copy()
    if set(chosen["cycle_interval_id"]) != set(wanted) or len(chosen) != 3:
        raise ELTEError(f"FROZEN_INTERVAL_SET_MISMATCH:{checkpoint}")
    chosen["selected_rank"] = pd.to_numeric(chosen["selected_rank"], errors="coerce")
    chosen = chosen.sort_values("selected_rank")
    if chosen["selected_rank"].tolist() != [1.0, 2.0, 3.0]:
        raise ELTEError(f"FROZEN_INTERVAL_RANK_MISMATCH:{checkpoint}")
    if chosen["cycle_interval_id"].tolist() != wanted:
        raise ELTEError(f"FROZEN_INTERVAL_ORDER_MISMATCH:{checkpoint}")
    support_count = int(chosen["in_frozen_joint_common_support"].astype(bool).sum())
    expected = int(protocol["selection"][checkpoint]["expected_support_count"])
    if support_count != expected:
        raise ELTEError(f"FROZEN_SUPPORT_COUNT_MISMATCH:{checkpoint}")
    return chosen


def _nonnegative_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.allclose(np.var(x), 0.0):
        raise ELTEError("CN_AFFINE_FIT_NOT_IDENTIFIABLE")
    slope = max(0.0, float(np.cov(x, y, ddof=0)[0, 1] / np.var(x)))
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def _cn_prediction(chosen: pd.DataFrame, transform: dict[str, Any], affine: tuple[float, float]) -> tuple[list[float], float, float]:
    variables = list(transform["condition_variables"])
    mean = transform["condition_scaler_mean"]
    scale = transform["condition_scaler_std"]
    coef = transform["condition_coefficients_standardized"]
    reference = transform["reference_condition"]
    def g(row: Any) -> float:
        def value(variable: str) -> float:
            return float(row[variable]) if isinstance(row, dict) else float(getattr(row, variable))
        return float(sum(((value(var) - float(mean[var])) / float(scale[var])) * float(coef[var]) for var in variables))
    g_ref = g(reference)
    normalized_ref = float(transform["normalized_reference_voltage_V"])
    cn = [float((float(row.P1S_voltage_V) - (g(row) - g_ref)) / normalized_ref) for row in chosen.itertuples(index=False)]
    r_cn = float(np.median(cn))
    return cn, r_cn, float(affine[0] * r_cn + affine[1])


def _freeze_predictions(base_commit: str, protocol: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    models = _read_yaml(AHC / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml")
    transform_path = CNP1S / "configs" / "FROZEN_CN_P1S_TRANSFORM.yaml"
    transform = _read_yaml(transform_path)
    if transform.get("model_type") != "C2_partial_linear_Ridge" or bool(transform.get("test_validated", True)):
        raise ELTEError("CN_TRANSFORM_NOT_FROZEN_SECONDARY_ARTIFACT")
    anchors = pd.read_csv(CNP1S / "outputs" / "CN_P1S_ANCHOR_COMPARISON.csv")
    anchors = anchors.loc[anchors["checkpoint"].isin([300, 600, 850, 1150, 1500, 1750])].copy()
    if anchors["checkpoint"].tolist() != [300, 600, 850, 1150, 1500, 1750]:
        raise ELTEError("CN_DEVELOPMENT_ANCHOR_SET_MISMATCH")
    cn_affine = _nonnegative_affine(anchors["R_CN_Ridge"].to_numpy(float), anchors["Y"].to_numpy(float))
    raw = models["P1S_affine"]
    age = models["nominal_age_affine"]
    rows: list[dict[str, Any]] = []
    provenance_intervals: dict[str, list[dict[str, Any]]] = {}
    for checkpoint in (2000, 2300):
        chosen = _selected_intervals(checkpoint, protocol)
        retention = chosen["P1S_operating_voltage_retention"].to_numpy(float).tolist()
        r_raw = float(np.median(retention))
        cn_values, r_cn, cn_prediction = _cn_prediction(chosen, transform, cn_affine)
        rows.append({
            "checkpoint": checkpoint,
            "interval_1": chosen.iloc[0]["cycle_interval_id"],
            "interval_2": chosen.iloc[1]["cycle_interval_id"],
            "interval_3": chosen.iloc[2]["cycle_interval_id"],
            "support_count": int(chosen["in_frozen_joint_common_support"].astype(bool).sum()),
            "support_status": protocol["selection"][checkpoint]["support_status"],
            "R1": retention[0], "R2": retention[1], "R3": retention[2], "R_median": r_raw,
            "prediction_Raw_P1S_B2": float(raw["slope"]) * r_raw + float(raw["intercept"]),
            "prediction_Age": float(age["intercept_alpha"]) + float(age["slope_beta"]) * checkpoint,
            "prediction_Raw_B0": r_raw,
            "CN_R1": cn_values[0], "CN_R2": cn_values[1], "CN_R3": cn_values[2], "R_CN": r_cn,
            "prediction_CN_Ridge": cn_prediction,
        })
        provenance_intervals[str(checkpoint)] = chosen[["cycle_interval_id", "selected_rank", "P1S_operating_voltage_retention", "P1S_voltage_V", "in_frozen_joint_common_support"]].to_dict(orient="records")
    prediction = pd.DataFrame(rows)
    provenance = {
        "base_commit": base_commit,
        "authorization": AUTHORIZATION,
        "nature": "EXPLORATORY_ONLY",
        "target_values_read": False,
        "selected_intervals": provenance_intervals,
        "raw_B2_coefficients": {"a": float(raw["slope"]), "b": float(raw["intercept"])},
        "age_coefficients": {"alpha": float(age["intercept_alpha"]), "beta": float(age["slope_beta"])},
        "BoL_voltage_V": float(protocol["targets"]["bol_voltage_V"]),
        "CN_transform_sha256": _sha256(transform_path),
        "CN_affine_coefficients": {"a_CN": cn_affine[0], "b_CN": cn_affine[1]},
        "CN_ridge_alpha": float(transform["ridge_alpha"]),
        "CN_reference_voltage_V": float(transform["normalized_reference_voltage_V"]),
    }
    return prediction, provenance


def _create_ledger(base_commit: str, prediction_sha: str, provenance_sha: str) -> dict[str, Any]:
    ledger = {
        "authorization": AUTHORIZATION,
        "nature": "EXPLORATORY_ONLY",
        "base_commit": base_commit,
        "target_opening": "CONSUMED_ON_FIRST_TARGET_IO",
        "predictions_frozen_before_target": True,
        "frozen_prediction_sha256": prediction_sha,
        "prediction_provenance_sha256": provenance_sha,
        "target_opening_started": True,
        "target_read_events": [],
        "targets_unblinded": False,
        "completed": False,
    }
    _atomic_json(LEDGER, ledger)
    return ledger


def _preflight_target_sources(protocol: dict[str, Any]) -> tuple[dict[int, Path], Any, Any]:
    """Bind both exact target paths before the ELTE ledger is consumed."""
    verify_identity, columns, extract = _load_stage_t_symbols()
    paths = {checkpoint: DATA_ROOT / protocol["targets"]["exact_nominal_sources"][checkpoint] for checkpoint in (2000, 2300)}
    for checkpoint, path in paths.items():
        if not path.is_file():
            raise ELTEError(f"TARGET_SOURCE_MISSING:{checkpoint}")
        verify_identity(checkpoint, path, phase=AHC)
    return paths, columns, extract


def _read_paired_targets(paths: dict[int, Path], columns: Any, extract: Any, ledger: dict[str, Any]) -> dict[int, tuple[float, int]]:
    """Read both already-bound targets through the frozen Stage-T extractor."""
    target_values: dict[int, tuple[float, int]] = {}
    # The next loop is the sole target I/O path.  Do not insert decision points.
    for checkpoint in (2000, 2300):
        table = pd.read_csv(paths[checkpoint], sep=";", encoding="utf-8-sig", usecols=list(columns))
        voltage, row_count = extract(table)
        target_values[checkpoint] = (float(voltage), int(row_count))
        ledger["target_read_events"].append(f"TARGET_{checkpoint}_READ")
        _atomic_json(LEDGER, ledger)
    ledger["targets_unblinded"] = True
    ledger["project_target_exposure"] = {"2000": "OPENED", "2300": "OPENED"}
    _atomic_json(LEDGER, ledger)
    _atomic_json(EXPOSURE, {"2000_2300_POLARIZATION_TARGETS_OPENED": True, "nature": "EXPLORATORY_ONLY"})
    return target_values


def _evaluate(prediction: pd.DataFrame, targets: dict[int, tuple[float, int]], bol: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = prediction.copy()
    result["target_voltage_V"] = result["checkpoint"].map({key: value[0] for key, value in targets.items()})
    result["target_jstar_row_count"] = result["checkpoint"].map({key: value[1] for key, value in targets.items()}).astype(int)
    result["target_Y"] = result["target_voltage_V"] / bol
    for method in ("Raw_P1S_B2", "Age", "Raw_B0", "CN_Ridge"):
        result[f"AE_{method}"] = (result[f"prediction_{method}"] - result["target_Y"]).abs()
        result[f"AE_{method}_mV"] = result[f"AE_{method}"] * bol * 1000.0
    raw = result.set_index("checkpoint")
    e1 = bool(raw.loc[2000, "AE_Raw_P1S_B2"] < raw.loc[2000, "AE_Age"])
    e2 = bool(raw.loc[2300, "AE_Raw_P1S_B2"] < raw.loc[2300, "AE_Age"])
    e3 = bool(result["AE_Raw_P1S_B2"].mean() < result["AE_Age"].mean())
    if e1 and e2 and e3:
        signal, direction = "STRONG_P1S_EXPLORATORY_LATE_LIFE_SIGNAL", "YES"
    elif not e1 and not e2:
        signal, direction = "WEAK_P1S_EXPLORATORY_LATE_LIFE_SIGNAL", "NO"
    else:
        signal, direction = "MIXED_P1S_EXPLORATORY_LATE_LIFE_SIGNAL", "MIXED"
    cn_mae = float(result["AE_CN_Ridge"].mean())
    raw_mae = float(result["AE_Raw_P1S_B2"].mean())
    cn_statement = "CN_P1S_SHOWS_EXPLORATORY_LATE_LIFE_IMPROVEMENT" if cn_mae < raw_mae else "CN_P1S_NO_MEANINGFUL_EXPLORATORY_IMPROVEMENT" if np.isclose(cn_mae, raw_mae, atol=1e-12) else "CN_P1S_EXPLORATORY_DEGRADATION"
    summary = {
        "nature": "EXPLORATORY_ONLY", "confirmatory_test": False, "targets_now_unblinded": True,
        "raw_p1s_b2_mae": raw_mae, "raw_p1s_b2_mae_mV": float(result["AE_Raw_P1S_B2_mV"].mean()),
        "age_mae": float(result["AE_Age"].mean()), "age_mae_mV": float(result["AE_Age_mV"].mean()),
        "raw_b0_mae": float(result["AE_Raw_B0"].mean()), "raw_b0_mae_mV": float(result["AE_Raw_B0_mV"].mean()),
        "cn_ridge_mae": cn_mae, "cn_ridge_mae_mV": float(result["AE_CN_Ridge_mV"].mean()),
        "E1": e1, "E2": e2, "E3": e3, "exploratory_signal": signal,
        "cn_p1s_exploratory_result": cn_statement, "direction_worth_deeper_study": direction,
        "post_target_tuning": False,
    }
    return result, summary


def _plot(results: pd.DataFrame) -> None:
    anchors = pd.read_csv(CNP1S / "outputs" / "CN_P1S_ANCHOR_COMPARISON.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 1.5, "figure.dpi": 150})
    fig, axis = plt.subplots(figsize=(10.5, 6.0))
    ages = anchors["checkpoint"].to_numpy(float)
    axis.plot(ages, anchors["Y"], color="#0F4D92", marker="o", linewidth=2.4, label="Development polarization health")
    axis.plot(ages, anchors["prediction_Raw_B2"], color="#B64342", marker="s", linestyle="--", linewidth=1.8, label="Raw P1S B2 prediction")
    late = results.sort_values("checkpoint")
    axis.scatter(late["checkpoint"], late["target_Y"], color="#0F4D92", marker="D", s=75, zorder=5, label="Late-life target (exploratory)")
    axis.plot(late["checkpoint"], late["prediction_Raw_P1S_B2"], color="#B64342", marker="s", linestyle="--", linewidth=1.8)
    axis.plot(late["checkpoint"], late["prediction_Age"], color="#42949E", marker="^", linestyle=":", linewidth=2.0, label="Frozen age baseline")
    axis.plot(late["checkpoint"], late["prediction_CN_Ridge"], color="#9A4D8E", marker="P", linestyle="-.", linewidth=1.7, label="CN-P1S Ridge (secondary)")
    for row in late.itertuples(index=False):
        offset = (0, 13) if int(row.checkpoint) == 2000 else (0, -22)
        axis.annotate("EXPLORATORY TARGET OPENED", (row.checkpoint, row.target_Y), xytext=offset, textcoords="offset points", ha="center", fontsize=8, color="#555555")
    axis.set_xlabel("Nominal ageing checkpoint (h)")
    axis.set_ylabel("Normalized polarization health Y")
    axis.set_xticks([300, 600, 850, 1150, 1500, 1750, 2000, 2300])
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _report(results: pd.DataFrame, summary: dict[str, Any], provenance: dict[str, Any]) -> None:
    rows = results.set_index("checkpoint")
    def line(checkpoint: int, method: str) -> str:
        return f"{float(rows.loc[checkpoint, f'prediction_{method}']):.9f} (AE {float(rows.loc[checkpoint, f'AE_{method}_mV']):.3f} mV)"
    content = [
        "# Exploratory Late-Life Target Evaluation", "",
        "## 1. Why this is exploratory rather than confirmatory", "",
        "The historical AHC-TV one-shot remains aborted after Test execution started, with target unblinding false. This new phase is an explicitly authorized, one-shot exploratory opening; the 2000/2300 targets were protocol-exposed and are not a pristine independent Test. No post-target tuning was performed.", "",
        "## 2. Frozen pre-target predictions", "",
        f"Predictions and provenance were SHA256-frozen before target I/O. Raw B2 coefficients were a={provenance['raw_B2_coefficients']['a']:.12f}, b={provenance['raw_B2_coefficients']['b']:.12f}; the nominal-age coefficients were alpha={provenance['age_coefficients']['alpha']:.12f}, beta={provenance['age_coefficients']['beta']:.12f}. CN-P1S is secondary only.", "",
        "## 3. Target extraction", "",
        "The frozen Stage-T semantics were used: active-cell nanmean over U1–U5 and U7–U11, U6 excluded, j*=1.0 A/cm² within ±0.01 A/cm², then nanmedian across valid rows. Both nominal sources were opened consecutively after prediction freeze.", "",
        "## 4. 2000 h result", "",
        f"Support is 0/3 (`OUT_OF_FROZEN_DEVELOPMENT_JOINT_SUPPORT`). Target Y={rows.loc[2000, 'target_Y']:.9f}; target voltage={rows.loc[2000, 'target_voltage_V']:.9f} V. Raw P1S B2: {line(2000, 'Raw_P1S_B2')}; age: {line(2000, 'Age')}; raw B0: {line(2000, 'Raw_B0')}; CN Ridge: {line(2000, 'CN_Ridge')}. This is an out-of-frozen-support exploratory extrapolation.", "",
        "## 5. 2300 h result", "",
        f"Support is 2/3 (`FROZEN_ALIGNMENT_COMPATIBLE`). Target Y={rows.loc[2300, 'target_Y']:.9f}; target voltage={rows.loc[2300, 'target_voltage_V']:.9f} V. Raw P1S B2: {line(2300, 'Raw_P1S_B2')}; age: {line(2300, 'Age')}; raw B0: {line(2300, 'Raw_B0')}; CN Ridge: {line(2300, 'CN_Ridge')}. This is alignment-compatible exploratory evidence, not a pristine Test.", "",
        "## 6. Raw P1S vs Age", "",
        f"E1/E2/E3 are {summary['E1']}/{summary['E2']}/{summary['E3']}. Raw P1S B2 MAE={summary['raw_p1s_b2_mae']:.9f} ({summary['raw_p1s_b2_mae_mV']:.3f} mV); age MAE={summary['age_mae']:.9f} ({summary['age_mae_mV']:.3f} mV).", "",
        "## 7. CN-P1S secondary comparison", "",
        f"CN Ridge MAE={summary['cn_ridge_mae']:.9f} ({summary['cn_ridge_mae_mV']:.3f} mV): `{summary['cn_p1s_exploratory_result']}`. Its prior Development result remains `CONDITION_NORMALIZED_P1S_NO_CLEAR_DEVELOPMENT_GAIN`; it was neither promoted nor retuned.", "",
        "## 8. Error in normalized Y and mV", "",
        "All absolute errors are saved in `outputs/ELTE_TARGET_RESULTS.csv`; mV errors are normalized-Y error × frozen BoL voltage × 1000.", "",
        "## 9. Development-to-late-life trajectory", "",
        "See `figures/Development_to_LateLife_Health_Trajectory.png`. Opened late-life targets are explicitly marked exploratory.", "",
        "## 10. Scientific interpretation", "",
        "A good 2000 h estimate cannot invalidate the support gate because it is only one out-of-support point. A good 2300 h estimate is stronger same-stack exploratory evidence but remains non-confirmatory.", "",
        "## 11. Is the direction worth deeper study?", "",
        f"`{summary['exploratory_signal']}`; decision: `{summary['direction_worth_deeper_study']}`. Any next study requires a separately authorized protocol and a new clean validation resource.", "",
        "## 12. What evidence is still missing?", "",
        "A clean independent Test, prospectively frozen success/failure criteria, and no target-exposed method refinement are still missing. This phase must not trigger automatic SOH, RUL, or model training.", "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ELTE exploratory one-shot target opening.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate every pre-target dependency without creating a ledger or reading targets.")
    args = parser.parse_args()
    for directory in (PHASE / "inputs", PHASE / "runtime", PHASE / "outputs", PHASE / "reports", PHASE / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    base_commit = _assert_start_state()
    protocol = _read_yaml(CONFIG)
    target_paths, target_columns, target_extractor = _preflight_target_sources(protocol)
    prediction, provenance = _freeze_predictions(base_commit, protocol)
    if args.preflight_only:
        print(json.dumps({
            "status": "ELTE_PRE_TARGET_PREFLIGHT_PASS",
            "base_commit": base_commit,
            "targets_read": False,
            "ledger_created": False,
            "prediction_rows_validated": int(len(prediction)),
            "provenance_fields_validated": sorted(provenance),
        }, ensure_ascii=False, indent=2))
        return
    prediction.to_csv(PREDICTIONS, index=False, float_format="%.15g")
    prediction_sha = _write_sha256_sidecar(PREDICTIONS)
    _atomic_json(PROVENANCE, provenance)
    provenance_sha = _write_sha256_sidecar(PROVENANCE)
    ledger = _create_ledger(base_commit, prediction_sha, provenance_sha)
    targets = _read_paired_targets(target_paths, target_columns, target_extractor, ledger)
    results, summary = _evaluate(prediction, targets, float(protocol["targets"]["bol_voltage_V"]))
    results.to_csv(RESULTS, index=False, float_format="%.15g")
    _atomic_json(SUMMARY, summary)
    _plot(results)
    _report(results, summary, provenance)
    ledger["completed"] = True
    ledger["result_paths"] = {"results": str(RESULTS.relative_to(REPO)), "summary": str(SUMMARY.relative_to(REPO))}
    _atomic_json(LEDGER, ledger)
    print(json.dumps({"status": "ELTE_COMPLETED", "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Development-only condition-normalized P1S study.

The program deliberately consumes only frozen Development artifacts.  It does
not resolve, open, or reference the 2000/2300 polarization source files and it
does not import any AHC-TV one-shot or Stage-T module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler


BASE_COMMIT = "d83aa909a687915ef24ebae8cddaae374111b99d"
PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
PAA = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_PAA_P1_StateToken_PreAnchor_Alignment"
CHM = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model"
AHC = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_AHC_Sparse_Anchor_Affine_Health_Calibration"
OUTPUTS = PHASE / "outputs"
FIGURES = PHASE / "figures"
REPORTS = PHASE / "reports"
CONFIGS = PHASE / "configs"

PROTOCOL = CONFIGS / "CNP1S_DEVELOPMENT_PROTOCOL.yaml"
COMPLETE = PAA / "outputs" / "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv"
PARTIAL = PAA / "outputs" / "PREANCHOR_PARTIAL_CYCLE_INTERVALS.csv"
REFERENCE = PAA / "outputs" / "P1S_REFERENCE_COMPLETE_CYCLES.csv"
ALIGNMENT = PAA / "outputs" / "P1S_ANCHOR_ALIGNMENT.csv"
TARGETS = CHM / "inputs" / "CHM_SPARSE_ANCHOR_TARGETS.csv"
B2_AUTHORITY = CHM / "outputs" / "CHM_D_BASELINE_LOAO.csv"
FROZEN_MODELS = REPO / "RENew-PEMFC-2300h-Pytorch" / "Phase_AHC_TV_One_Time_Late_Life_Test_Protocol" / "configs" / "FROZEN_DEVELOPMENT_MODELS.yaml"

ANCHORS = (300, 600, 850, 1150, 1500, 1750)
CONDITIONS = (
    "current_density",
    "anode_inlet_temperature_C",
    "anode_inlet_pressure_kPag",
    "anode_inlet_RH_pct",
    "anode_hydrogen_flow_nl_min",
    "anode_stoichiometry",
    "cathode_stoichiometry",
)
ALPHA_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEVELOPMENT_BOUNDARY_H = 1819.997738
TOLERANCE = 1e-10
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B",
    "red_strong": "#B64342", "neutral": "#767676", "teal": "#42949E",
}


class CNP1SError(RuntimeError):
    """A frozen Development-only input or leakage guard failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 12,
        "axes.linewidth": 1.8, "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "svg.fonttype": "none",
    })


def _assert_target_blind_inputs(paths: tuple[Path, ...]) -> None:
    banned = {"polcurve_nominal_rhfc_2000h.csv", "polcurve_nominal_rhfc_2300h.csv"}
    for path in paths:
        if not path.exists():
            raise CNP1SError(f"FROZEN_DEVELOPMENT_INPUT_MISSING:{path}")
        if path.name.lower() in banned or "one_time_late_life_test_protocol/runtime" in path.as_posix().lower():
            raise CNP1SError(f"TEST_INPUT_REJECTED:{path}")


def _numeric(data: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    result = data.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def _load_intervals() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    complete = pd.read_csv(COMPLETE)
    partial = pd.read_csv(PARTIAL)
    reference = pd.read_csv(REFERENCE)
    required = {
        "stable_window", "cycle_interval_id", "measurement_end_ophrs2_h", "measurement_end_datetime",
        "P1S_voltage_V", "P1S_operating_voltage_retention", *CONDITIONS,
    }
    for name, frame in (("complete", complete), ("partial", partial), ("reference", reference)):
        absent = required - set(frame.columns)
        if absent:
            raise CNP1SError(f"P1S_{name.upper()}_SCHEMA_INVALID:" + ",".join(sorted(absent)))
    complete = complete.loc[complete["stable_window"].eq("middle_50")].copy()
    partial = partial.loc[partial["stable_window"].eq("middle_50")].copy()
    data = pd.concat([complete, partial], ignore_index=True, sort=False)
    data = _numeric(data, ["measurement_end_ophrs2_h", "P1S_voltage_V", "P1S_operating_voltage_retention", *CONDITIONS])
    data["measurement_end_datetime"] = pd.to_datetime(data["measurement_end_datetime"], errors="raise")
    if data["cycle_interval_id"].duplicated().any() or not data["measurement_end_ophrs2_h"].lt(DEVELOPMENT_BOUNDARY_H).all():
        raise CNP1SError("DEVELOPMENT_P1S_INTERVAL_UNIVERSE_INVALID")
    data = data.sort_values(["measurement_end_ophrs2_h", "cycle_interval_id"], kind="mergesort").reset_index(drop=True)
    reference = _numeric(reference, ["measurement_end_ophrs2_h", "P1S_voltage_V", "P1S_operating_voltage_retention", *CONDITIONS])
    reference_ids = reference["cycle_interval_id"].astype(str).tolist()
    if len(reference) != 10 or reference_ids != data.head(10)["cycle_interval_id"].astype(str).tolist():
        raise CNP1SError("EARLIEST_TEN_REFERENCE_INTERVALS_MISMATCH")
    reference_voltage = float(reference["reference_voltage_V"].iloc[0])
    if not np.isfinite(reference_voltage) or not np.allclose(reference["P1S_voltage_V"].median(), reference_voltage, atol=TOLERANCE, rtol=0.0):
        raise CNP1SError("FROZEN_P1S_REFERENCE_VOLTAGE_MISMATCH")
    return data, reference, complete, reference_voltage


def _load_anchor_inputs(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alignment = pd.read_csv(ALIGNMENT)
    targets = pd.read_csv(TARGETS)
    b2 = pd.read_csv(B2_AUTHORITY)
    target_sources = set(targets["source_polarization_file"].astype(str).str.lower())
    if {"polcurve_nominal_rhfc_2000h.csv", "polcurve_nominal_rhfc_2300h.csv"} & target_sources:
        raise CNP1SError("TEST_TARGET_SOURCE_REFERENCE_REJECTED")
    summaries = alignment.loc[alignment["record_type"].eq("anchor_summary")].copy()
    rows = alignment.loc[alignment["record_type"].eq("anchor_interval")].copy()
    summaries = _numeric(summaries, ["checkpoint_h", "precharacterization_cutoff_h", "P1S_anchor_retention"])
    rows = _numeric(rows, ["checkpoint_h", "selection_rank", "measurement_end_ophrs2_h", "P1S_operating_voltage_retention", *CONDITIONS])
    targets = _numeric(targets, ["checkpoint", "Y"])
    b2 = _numeric(b2, ["checkpoint", "target", "B2_prediction", "B2_absolute_error"])
    if tuple(targets["checkpoint"].astype(int)) != ANCHORS or tuple(b2["checkpoint"].astype(int)) != ANCHORS:
        raise CNP1SError("DEVELOPMENT_ANCHOR_SET_INVALID")
    expected_ids = set(intervals["cycle_interval_id"].astype(str))
    if not set(rows["cycle_interval_id"].astype(str)).issubset(expected_ids):
        raise CNP1SError("FROZEN_ANCHOR_INTERVAL_NOT_IN_DEVELOPMENT_UNIVERSE")
    return summaries, rows, targets.merge(b2[["checkpoint", "B2_prediction", "B2_absolute_error"]], on="checkpoint", validate="one_to_one")


def _build_features(age: np.ndarray, conditions: np.ndarray, *, spline: SplineTransformer | None = None, scaler: StandardScaler | None = None, fit: bool) -> tuple[np.ndarray, SplineTransformer, StandardScaler]:
    if fit:
        spline = SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="linear")
        scaler = StandardScaler()
        age_basis = spline.fit_transform(age.reshape(-1, 1))
        condition_basis = scaler.fit_transform(conditions)
    else:
        if spline is None or scaler is None:
            raise CNP1SError("MODEL_TRANSFORM_NOT_FITTED")
        age_basis = spline.transform(age.reshape(-1, 1))
        condition_basis = scaler.transform(conditions)
    return np.hstack([age_basis, condition_basis]), spline, scaler


def _fit_condition_model(data: pd.DataFrame, *, kind: str, alpha: float | None = None) -> dict[str, Any]:
    age = data["measurement_end_ophrs2_h"].to_numpy(dtype=float)
    conditions = data.loc[:, CONDITIONS].to_numpy(dtype=float)
    x, spline, scaler = _build_features(age, conditions, fit=True)
    y = data["P1S_voltage_V"].to_numpy(dtype=float)
    if kind == "OLS":
        estimator: LinearRegression | Ridge = LinearRegression()
    elif kind == "Ridge":
        if alpha is None:
            raise CNP1SError("RIDGE_ALPHA_REQUIRED")
        estimator = Ridge(alpha=float(alpha), fit_intercept=True)
    else:
        raise CNP1SError(f"UNKNOWN_CONDITION_MODEL:{kind}")
    estimator.fit(x, y)
    return {
        "kind": kind, "alpha": None if alpha is None else float(alpha), "spline": spline,
        "scaler": scaler, "estimator": estimator, "age_feature_count": int(x.shape[1] - len(CONDITIONS)),
        "fitting_row_count": int(len(data)),
    }


def _predict(model: dict[str, Any], data: pd.DataFrame) -> np.ndarray:
    age = data["measurement_end_ophrs2_h"].to_numpy(dtype=float)
    conditions = data.loc[:, CONDITIONS].to_numpy(dtype=float)
    x, _, _ = _build_features(age, conditions, spline=model["spline"], scaler=model["scaler"], fit=False)
    return np.asarray(model["estimator"].predict(x), dtype=float)


def _condition_effect(model: dict[str, Any], conditions: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = conditions.loc[:, CONDITIONS].to_numpy(dtype=float) if isinstance(conditions, pd.DataFrame) else np.asarray(conditions, dtype=float)
    standardized = model["scaler"].transform(values)
    coefficients = np.asarray(model["estimator"].coef_, dtype=float)
    return standardized @ coefficients[-len(CONDITIONS):]


def _normalized_voltage(model: dict[str, Any], data: pd.DataFrame, reference_condition: np.ndarray) -> np.ndarray:
    return data["P1S_voltage_V"].to_numpy(dtype=float) - (_condition_effect(model, data) - _condition_effect(model, reference_condition.reshape(1, -1)))


def _select_ridge_alpha(prefix: pd.DataFrame, *, label: str) -> tuple[float, pd.DataFrame]:
    data = prefix.sort_values("measurement_end_ophrs2_h", kind="mergesort").reset_index(drop=True)
    validation_count = max(5, int(np.ceil(0.20 * len(data))))
    if len(data) - validation_count < 14:
        raise CNP1SError(f"EARLY_PREFIX_INSUFFICIENT:{label}")
    train = data.iloc[:-validation_count].copy()
    validation = data.iloc[-validation_count:].copy()
    records: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        model = _fit_condition_model(train, kind="Ridge", alpha=alpha)
        mae = float(np.mean(np.abs(_predict(model, validation) - validation["P1S_voltage_V"].to_numpy(dtype=float))))
        records.append({
            "prefix_label": label, "selection_target": "P1S_voltage_reconstruction_MAE",
            "inner_train_rows": int(len(train)), "inner_validation_rows": int(len(validation)),
            "alpha": float(alpha), "validation_MAE_V": mae,
        })
    table = pd.DataFrame(records).sort_values(["validation_MAE_V", "alpha"], kind="mergesort").reset_index(drop=True)
    return float(table.iloc[0]["alpha"]), table


def _nonnegative_affine(x: np.ndarray, y: np.ndarray, x_value: float) -> tuple[float, float, float]:
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    slope = 0.0 if denominator == 0 else float(np.dot(centered, y - y.mean()) / denominator)
    slope = max(0.0, slope)
    intercept = float(y.mean() - slope * x.mean())
    return float(slope * x_value + intercept), slope, intercept


def _loao(values: pd.Series, targets: pd.Series, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for heldout in ANCHORS:
        training = [checkpoint for checkpoint in ANCHORS if checkpoint != heldout]
        prediction, slope, intercept = _nonnegative_affine(
            values.loc[training].to_numpy(dtype=float), targets.loc[training].to_numpy(dtype=float), float(values.loc[heldout])
        )
        y = float(targets.loc[heldout])
        rows.append({
            "model": model_name, "checkpoint": heldout, "prediction": prediction, "target_Y": y,
            "absolute_error": abs(prediction - y), "squared_error": (prediction - y) ** 2,
            "slope_a": slope, "intercept_b": intercept,
        })
    return pd.DataFrame(rows)


def _chronological(values: pd.Series, targets: pd.Series, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for heldout in (1150, 1500, 1750):
        training = [checkpoint for checkpoint in ANCHORS if checkpoint < heldout]
        prediction, slope, intercept = _nonnegative_affine(
            values.loc[training].to_numpy(dtype=float), targets.loc[training].to_numpy(dtype=float), float(values.loc[heldout])
        )
        y = float(targets.loc[heldout])
        rows.append({
            "model": model_name, "target_checkpoint": heldout, "training_checkpoints": "|".join(map(str, training)),
            "prediction": prediction, "target_Y": y, "absolute_error": abs(prediction - y),
            "squared_error": (prediction - y) ** 2, "slope_a": slope, "intercept_b": intercept,
        })
    return pd.DataFrame(rows)


def _age_residuals(data: pd.DataFrame, value_column: str) -> np.ndarray:
    age = data["measurement_end_ophrs2_h"].to_numpy(dtype=float)
    spline = SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="linear")
    x = spline.fit_transform(age.reshape(-1, 1))
    estimator = LinearRegression().fit(x, data[value_column].to_numpy(dtype=float))
    return data[value_column].to_numpy(dtype=float) - estimator.predict(x)


def _correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.corr(right, method=method))


def _condition_decoupling(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model, column in (("Raw", "V_raw"), ("CN_OLS", "V_CN_OLS"), ("CN_Ridge", "V_CN_Ridge")):
        residual = pd.Series(_age_residuals(data, column), index=data.index)
        for variable in CONDITIONS:
            condition = data[variable]
            spearman = _correlation(residual, condition, "spearman")
            pearson = _correlation(residual, condition, "pearson")
            records.append({
                "model": model, "variable": variable,
                "Spearman_residual_vs_condition": spearman,
                "Pearson_residual_vs_condition": pearson,
                "absolute_Spearman": float(abs(spearman)),
            })
    return pd.DataFrame(records)


def _local_scatter(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = data.sort_values("measurement_end_ophrs2_h", kind="mergesort").reset_index(drop=True)
    for model, column in (("Raw", "V_raw"), ("CN_OLS", "V_CN_OLS"), ("CN_Ridge", "V_CN_Ridge")):
        for start in range(0, len(ordered) - 2):
            window = ordered.iloc[start:start + 3]
            values = window[column].to_numpy(dtype=float)
            median = float(np.median(values))
            rows.append({
                "model": model, "window_size_intervals": 3,
                "start_interval_id": str(window.iloc[0]["cycle_interval_id"]),
                "end_interval_id": str(window.iloc[-1]["cycle_interval_id"]),
                "start_ophrs2_h": float(window.iloc[0]["measurement_end_ophrs2_h"]),
                "end_ophrs2_h": float(window.iloc[-1]["measurement_end_ophrs2_h"]),
                "local_MAD_V": float(np.median(np.abs(values - median))),
                "local_std_V": float(np.std(values, ddof=0)),
            })
    return pd.DataFrame(rows)


def _trend_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, column in (("Raw", "R_raw"), ("CN_OLS", "R_CN_OLS"), ("CN_Ridge", "R_CN_Ridge")):
        x = data["measurement_end_ophrs2_h"].to_numpy(dtype=float)
        y = data[column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({
            "model": model, "retention_column": column,
            "Spearman_retention_vs_OpHrs2": float(pd.Series(y).corr(pd.Series(x), method="spearman")),
            "linear_drift_slope_per_h": float(slope), "linear_intercept": float(intercept),
        })
    return pd.DataFrame(rows)


def _plot_development_trend(data: pd.DataFrame, path: Path) -> None:
    _style()
    fig, axis = plt.subplots(figsize=(10.0, 4.5))
    for column, label, color in (
        ("R_raw", "Raw P1S", PALETTE["neutral"]),
        ("R_CN_OLS", "CN-P1S OLS", PALETTE["teal"]),
        ("R_CN_Ridge", "CN-P1S Ridge", PALETTE["blue_main"]),
    ):
        axis.plot(data["OpHrs2_h"], data[column], marker="o", markersize=3.2, linewidth=1.6, label=label, color=color)
    axis.set(xlabel="Operating age, OpHrs2 (h)", ylabel="P1S retention", title="Development P1S before and after condition normalization")
    axis.legend(ncol=3, loc="best", fontsize=9)
    fig.tight_layout(pad=1.0); fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _plot_anchor_relationship(anchors: pd.DataFrame, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), sharey=True)
    spec = (("R_raw", "Raw P1S", PALETTE["neutral"]), ("R_CN_OLS", "CN-P1S OLS", PALETTE["teal"]), ("R_CN_Ridge", "CN-P1S Ridge", PALETTE["blue_main"]))
    for axis, (column, title, color) in zip(axes, spec):
        axis.scatter(anchors[column], anchors["Y"], color=color, s=46, edgecolor="black", linewidth=0.5)
        for row in anchors.itertuples(index=False):
            axis.annotate(str(int(row.checkpoint)), (getattr(row, column), row.Y), xytext=(3, 3), textcoords="offset points", fontsize=8)
        axis.set_title(title); axis.set_xlabel("Anchor retention")
    axes[0].set_ylabel("Frozen Development polarization-health anchor Y")
    fig.tight_layout(pad=1.0); fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _plot_condition_sensitivity(decoupling: pd.DataFrame, path: Path) -> None:
    _style()
    pivot = decoupling.pivot(index="variable", columns="model", values="absolute_Spearman").reindex(CONDITIONS)
    labels = [name.replace("anode_", "A.").replace("cathode_", "C.").replace("_", " ") for name in CONDITIONS]
    x = np.arange(len(labels)); width = 0.25
    fig, axis = plt.subplots(figsize=(11.0, 4.5))
    for index, (model, color) in enumerate((("Raw", PALETTE["neutral"]), ("CN_OLS", PALETTE["teal"]), ("CN_Ridge", PALETTE["blue_main"]))):
        axis.bar(x + (index - 1) * width, pivot[model], width, label=model, color=color, edgecolor="black", linewidth=0.7)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("|Spearman| after age-trend removal")
    axis.set_title("Development residual operating-condition sensitivity")
    axis.legend(ncol=3, fontsize=9)
    fig.tight_layout(pad=1.0); fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _plot_local_scatter(scatter: pd.DataFrame, path: Path) -> None:
    _style()
    labels = ["Raw", "CN-OLS", "CN-Ridge"]
    data = [scatter.loc[scatter["model"].eq(model), "local_std_V"].to_numpy(dtype=float) * 1e3 for model in ("Raw", "CN_OLS", "CN_Ridge")]
    fig, axis = plt.subplots(figsize=(6.5, 4.4))
    boxes = axis.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], (PALETTE["neutral"], PALETTE["teal"], PALETTE["blue_main"])):
        box.set(facecolor=color, alpha=0.78, edgecolor="black")
    axis.set(ylabel="3-interval local standard deviation (mV)", title="Development local P1S scatter")
    fig.tight_layout(pad=1.0); fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _markdown_table(data: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    if data.empty:
        return "No rows."
    table = data.loc[:, columns].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    head = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([head, divider, *body])


def _report(summary: dict[str, Any], anchors: pd.DataFrame, loao: pd.DataFrame, decoupling: pd.DataFrame, scatter: pd.DataFrame, trend: pd.DataFrame) -> str:
    ridge_loao = loao.loc[loao["model"].eq("CN_Ridge")]
    local = scatter.groupby("model", as_index=False).agg(median_local_MAD_V=("local_MAD_V", "median"), median_local_std_V=("local_std_V", "median"))
    condition_focus = decoupling.loc[decoupling["variable"].isin(("anode_stoichiometry", "cathode_stoichiometry", "anode_inlet_RH_pct", "anode_inlet_temperature_C"))]
    return "\n".join([
        "# Condition-Normalized P1S Development Study", "",
        "## 1. Research question", "", "Can a partial condition model reduce P1S operating-condition sensitivity while retaining the frozen sparse Development polarization-health relationship?", "",
        "## 2. Data boundary", "", "`DEVELOPMENT_ONLY`; 2000/2300 polarization targets were not accessed. The consumed AHC-TV one-shot ledger was not read for execution or modified. No Stage T, Test AE/MAE, or Rules A/B/C calculation occurred.", "",
        "## 3. Why hard support was insufficient", "", "The preceding target-blind audit found the P1S state observable at late life but the frozen seven-dimensional hard box brittle to condition drift. This study retains all legal Development P1S intervals rather than deleting out-of-box rows.", "",
        "## 4. Condition-normalization formulation", "", "For each chronology-safe prefix, the partial model is `V_P1S = f(age) + G(z) + epsilon`, with `f(age)` a train-prefix cubic spline and `z` the seven frozen operating variables. Normalization is `V_CN = V_raw - [G(z) - G(z_ref)]`. Age is not supplied to the later `R -> Y` affine mapping.", "",
        "## 5. OLS comparator", "", "C1 uses the registered partial-linear OLS condition model and the same nonnegative affine anchor calibration as every other candidate.", "",
        "## 6. Ridge primary model", "", f"C2 selects alpha by chronological inner validation P1S-voltage reconstruction MAE only. Final-Development selected alpha: `{summary['selected_final_ridge_alpha']}`.", "",
        "## 7. Chronology-safe evaluation", "", "Each anchor correction uses only intervals ending before that anchor's frozen characterization cutoff. Its fixed existing latest-three interval IDs are retained; no support-based replacement was made.", "",
        "## 8. LOAO health-anchor results", "", _markdown_table(anchors, ["checkpoint", "Y", "R_raw", "R_CN_OLS", "R_CN_Ridge", "AE_Raw_B2", "AE_CN_OLS", "AE_CN_Ridge"]), "", f"Raw B2 LOAO MAE was reproduced at `{summary['Raw_B2_LOAO_MAE']:.15f}`. OLS / Ridge CN-P1S LOAO MAE: `{summary['OLS_CN_P1S_LOAO_MAE']:.15f}` / `{summary['Ridge_CN_P1S_LOAO_MAE']:.15f}`.", "",
        "## 9. Condition decoupling", "", _markdown_table(condition_focus, ["model", "variable", "Spearman_residual_vs_condition", "Pearson_residual_vs_condition", "absolute_Spearman"]), "", "Residual correlations remove a low-frequency age spline before relating voltage to operating conditions.", "",
        "## 10. Local scatter", "", _markdown_table(local, ["model", "median_local_MAD_V", "median_local_std_V"], digits=9), "",
        "## 11. Trend preservation", "", _markdown_table(trend, ["model", "Spearman_retention_vs_OpHrs2", "linear_drift_slope_per_h"]), "",
        "## 12. Comparison with Raw B2", "", "The Raw comparator is the frozen six-anchor B2 authority. Its exact LOAO reproduction was verified before any CN comparison; no new raw baseline was searched.", "",
        "## 13. Limitations", "", "Only six sparse Development health anchors exist. The condition transform is exploratory and Development-frozen; it is neither Test-validated SOH nor RUL, and it has not been applied to 2000/2300.", "",
        "## 14. Decision", "", f"Final status: `{summary['final_development_status']}`. Gates A/B/C/D: `{summary['gates']['A']}` / `{summary['gates']['B']}` / `{summary['gates']['C']}` / `{summary['gates']['D']}`.", "",
        "## 15. Next validation requirement", "", "Any future target-blind transform-applicability audit or independent validation must be separately authorized. This study does not reopen, reinterpret, or rerun the consumed AHC-TV Test.", "",
    ]).rstrip() + "\n"


def main() -> None:
    _assert_target_blind_inputs((PROTOCOL, COMPLETE, PARTIAL, REFERENCE, ALIGNMENT, TARGETS, B2_AUTHORITY, FROZEN_MODELS))
    protocol = _yaml(PROTOCOL)
    if protocol.get("base_commit") != BASE_COMMIT or tuple(protocol.get("development_anchors_h", [])) != ANCHORS:
        raise CNP1SError("FROZEN_PROTOCOL_INVALID")
    intervals, reference, _, frozen_raw_reference_voltage = _load_intervals()
    summaries, anchor_rows, targets = _load_anchor_inputs(intervals)
    target_y = targets.set_index("checkpoint")["Y"].astype(float)
    reference_condition = reference.loc[:, CONDITIONS].median().to_numpy(dtype=float)
    reference_payload = {
        "source_artifact": str(REFERENCE.relative_to(REPO)), "number_of_intervals": int(len(reference)),
        "interval_ids": reference["cycle_interval_id"].astype(str).tolist(), "reference_condition": {name: float(value) for name, value in zip(CONDITIONS, reference_condition)},
        "frozen_raw_reference_voltage_V": frozen_raw_reference_voltage,
    }
    prefix_models: dict[int, dict[str, dict[str, Any]]] = {}
    anchor_values: list[dict[str, Any]] = []
    alpha_records: list[pd.DataFrame] = []
    for checkpoint in ANCHORS:
        summary_row = summaries.loc[summaries["checkpoint_h"].astype(int).eq(checkpoint)]
        selected = anchor_rows.loc[anchor_rows["checkpoint_h"].astype(int).eq(checkpoint)].sort_values("selection_rank", kind="mergesort")
        if len(summary_row) != 1 or len(selected) != 3:
            raise CNP1SError(f"FROZEN_ANCHOR_ALIGNMENT_INVALID:{checkpoint}")
        cutoff = float(summary_row.iloc[0]["precharacterization_cutoff_h"])
        prefix = intervals.loc[intervals["measurement_end_ophrs2_h"].lt(cutoff)].copy()
        if len(prefix) < 20:
            raise CNP1SError(f"EARLY_PREFIX_INSUFFICIENT:{checkpoint}")
        ols = _fit_condition_model(prefix, kind="OLS")
        alpha, alpha_table = _select_ridge_alpha(prefix, label=f"anchor_{checkpoint}")
        alpha_records.append(alpha_table.assign(checkpoint=checkpoint))
        ridge = _fit_condition_model(prefix, kind="Ridge", alpha=alpha)
        prefix_models[checkpoint] = {"OLS": ols, "Ridge": ridge}
        ids = selected["cycle_interval_id"].astype(str).tolist()
        selected_intervals = intervals.set_index("cycle_interval_id", drop=False).loc[ids].reset_index(drop=True)
        raw_anchor = float(np.median(selected_intervals["P1S_voltage_V"].to_numpy(dtype=float) / frozen_raw_reference_voltage))
        frozen_anchor = float(summary_row.iloc[0]["P1S_anchor_retention"])
        if abs(raw_anchor - frozen_anchor) > TOLERANCE:
            raise CNP1SError(f"RAW_ANCHOR_RETENTION_REPRODUCTION_FAIL:{checkpoint}")
        row: dict[str, Any] = {
            "checkpoint": checkpoint, "Y": float(target_y.loc[checkpoint]), "cutoff_ophrs2_h": cutoff,
            "fixed_selected_interval_ids": "|".join(ids), "R_raw": raw_anchor,
            "prefix_condition_model_rows": int(len(prefix)), "Ridge_alpha": alpha,
        }
        for name, model in (("OLS", ols), ("Ridge", ridge)):
            normalized_selected = _normalized_voltage(model, selected_intervals, reference_condition)
            normalized_reference = _normalized_voltage(model, reference, reference_condition)
            normalized_ref = float(np.median(normalized_reference))
            row[f"V_CN_reference_{name}"] = normalized_ref
            row[f"R_CN_{name}"] = float(np.median(normalized_selected / normalized_ref))
        anchor_values.append(row)
    anchors = pd.DataFrame(anchor_values).sort_values("checkpoint").reset_index(drop=True)
    raw_values = anchors.set_index("checkpoint")["R_raw"]
    ols_values = anchors.set_index("checkpoint")["R_CN_OLS"]
    ridge_values = anchors.set_index("checkpoint")["R_CN_Ridge"]
    raw_loao = _loao(raw_values, target_y, "Raw_B2")
    authority = targets.set_index("checkpoint")
    if not np.allclose(raw_loao["prediction"].to_numpy(dtype=float), authority.loc[list(ANCHORS), "B2_prediction"].to_numpy(dtype=float), rtol=0.0, atol=TOLERANCE):
        raise CNP1SError("RAW_B2_LOAO_PREDICTION_REPRODUCTION_FAIL")
    if not np.allclose(raw_loao["absolute_error"].to_numpy(dtype=float), authority.loc[list(ANCHORS), "B2_absolute_error"].to_numpy(dtype=float), rtol=0.0, atol=TOLERANCE):
        raise CNP1SError("RAW_B2_LOAO_ERROR_REPRODUCTION_FAIL")
    raw_b2_mae = float(raw_loao["absolute_error"].mean())
    if abs(raw_b2_mae - 0.0004768699068203319) > TOLERANCE:
        raise CNP1SError("RAW_B2_LOAO_MAE_AUTHORITY_MISMATCH")
    ols_loao = _loao(ols_values, target_y, "CN_OLS")
    ridge_loao = _loao(ridge_values, target_y, "CN_Ridge")
    loao = pd.concat([raw_loao, ols_loao, ridge_loao], ignore_index=True)
    prediction_columns = {
        "Raw_B2": ("prediction", "absolute_error"), "CN_OLS": ("prediction", "absolute_error"), "CN_Ridge": ("prediction", "absolute_error"),
    }
    for model, frame in (("Raw_B2", raw_loao), ("CN_OLS", ols_loao), ("CN_Ridge", ridge_loao)):
        mapping = frame.set_index("checkpoint")
        anchors[f"prediction_{model}"] = anchors["checkpoint"].map(mapping["prediction"])
        anchors[f"AE_{model}"] = anchors["checkpoint"].map(mapping["absolute_error"])
    anchors = anchors.rename(columns={"AE_Raw_B2": "AE_Raw_B2", "AE_CN_OLS": "AE_CN_OLS", "AE_CN_Ridge": "AE_CN_Ridge"})
    chronological = pd.concat([
        _chronological(raw_values, target_y, "Raw_B2"),
        _chronological(ols_values, target_y, "CN_OLS"),
        _chronological(ridge_values, target_y, "CN_Ridge"),
    ], ignore_index=True)
    final_alpha, final_alpha_table = _select_ridge_alpha(intervals, label="final_development")
    alpha_records.append(final_alpha_table.assign(checkpoint="FINAL_DEVELOPMENT"))
    final_ols = _fit_condition_model(intervals, kind="OLS")
    final_ridge = _fit_condition_model(intervals, kind="Ridge", alpha=final_alpha)
    development = intervals.copy()
    development["interval_id"] = development["cycle_interval_id"].astype(str)
    development["OpHrs2_h"] = development["measurement_end_ophrs2_h"]
    development["V_raw"] = development["P1S_voltage_V"]
    development["R_raw"] = development["V_raw"] / frozen_raw_reference_voltage
    for name, model in (("OLS", final_ols), ("Ridge", final_ridge)):
        voltage = _normalized_voltage(model, development, reference_condition)
        ref_voltage = float(np.median(_normalized_voltage(model, reference, reference_condition)))
        development[f"V_CN_{name}"] = voltage
        development[f"R_CN_{name}"] = voltage / ref_voltage
    final_columns = ["interval_id", "OpHrs2_h", "V_raw", "R_raw", "V_CN_OLS", "R_CN_OLS", "V_CN_Ridge", "R_CN_Ridge", *CONDITIONS]
    development_output = development.loc[:, final_columns].copy()
    decoupling = _condition_decoupling(development)
    scatter = _local_scatter(development)
    trend = _trend_metrics(development)
    decoupling_index = decoupling.set_index(["model", "variable"])
    raw_anode = float(decoupling_index.loc[("Raw", "anode_stoichiometry"), "absolute_Spearman"])
    ridge_anode = float(decoupling_index.loc[("CN_Ridge", "anode_stoichiometry"), "absolute_Spearman"])
    raw_cathode = float(decoupling_index.loc[("Raw", "cathode_stoichiometry"), "absolute_Spearman"])
    ridge_cathode = float(decoupling_index.loc[("CN_Ridge", "cathode_stoichiometry"), "absolute_Spearman"])
    raw_anchor_spearman = float(raw_values.corr(target_y, method="spearman"))
    ridge_anchor_spearman = float(ridge_values.corr(target_y, method="spearman"))
    gates = {
        "A": "PASS" if float(ridge_loao["absolute_error"].mean()) <= raw_b2_mae + TOLERANCE else "FAIL",
        "B": "PASS" if ridge_anode < raw_anode - TOLERANCE else "FAIL",
        "C": "PASS" if ridge_cathode < raw_cathode - TOLERANCE else "FAIL",
        "D": "PASS" if ridge_anchor_spearman + TOLERANCE >= raw_anchor_spearman else "FAIL",
    }
    final_status = "CONDITION_NORMALIZED_P1S_DEVELOPMENT_SUPPORTED" if all(value == "PASS" for value in gates.values()) else "CONDITION_NORMALIZED_P1S_NO_CLEAR_DEVELOPMENT_GAIN"
    alpha_selection = pd.concat(alpha_records, ignore_index=True)
    reference_payload["final_OLS_normalized_reference_voltage_V"] = float(np.median(_normalized_voltage(final_ols, reference, reference_condition)))
    reference_payload["final_Ridge_normalized_reference_voltage_V"] = float(np.median(_normalized_voltage(final_ridge, reference, reference_condition)))
    input_paths = (PROTOCOL, COMPLETE, PARTIAL, REFERENCE, ALIGNMENT, TARGETS, B2_AUTHORITY, FROZEN_MODELS)
    summary = {
        "base_commit": BASE_COMMIT, "phase": "DEVELOPMENT_ONLY", "targets_2000_2300_accessed": False,
        "one_shot_ledger_modified": False, "one_shot_runner_executed": False,
        "Raw_B2_LOAO_MAE": raw_b2_mae, "OLS_CN_P1S_LOAO_MAE": float(ols_loao["absolute_error"].mean()),
        "Ridge_CN_P1S_LOAO_MAE": float(ridge_loao["absolute_error"].mean()),
        "Raw_anchor_Spearman": raw_anchor_spearman, "Ridge_CN_P1S_anchor_Spearman": ridge_anchor_spearman,
        "anode_stoichiometry_residual_dependence": {"Raw": raw_anode, "CN_Ridge": ridge_anode, "improved": bool(ridge_anode < raw_anode - TOLERANCE)},
        "cathode_stoichiometry_residual_dependence": {"Raw": raw_cathode, "CN_Ridge": ridge_cathode, "improved": bool(ridge_cathode < raw_cathode - TOLERANCE)},
        "local_scatter_median_std_V": {row.model: float(row.median_local_std_V) for row in scatter.groupby("model", as_index=False).agg(median_local_std_V=("local_std_V", "median")).itertuples(index=False)},
        "selected_final_ridge_alpha": final_alpha, "gates": gates, "final_development_status": final_status,
        "true_independent_test_executed": False, "age_used_in_health_mapping": False,
        "frozen_input_sha256": {str(path.relative_to(REPO)): _sha256(path) for path in input_paths},
    }
    CONFIGS.mkdir(parents=True, exist_ok=True); OUTPUTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "CN_P1S_REFERENCE_CONDITION.json").write_text(json.dumps(reference_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    development_output.to_csv(OUTPUTS / "CN_P1S_DEVELOPMENT_INTERVALS.csv", index=False)
    anchors.to_csv(OUTPUTS / "CN_P1S_ANCHOR_COMPARISON.csv", index=False)
    loao.to_csv(OUTPUTS / "CN_P1S_LOAO_RESULTS.csv", index=False)
    chronological.to_csv(OUTPUTS / "CN_P1S_CHRONOLOGICAL_RESULTS.csv", index=False)
    decoupling.to_csv(OUTPUTS / "CN_P1S_CONDITION_DECOUPLING.csv", index=False)
    scatter.to_csv(OUTPUTS / "CN_P1S_LOCAL_SCATTER.csv", index=False)
    trend.to_csv(OUTPUTS / "CN_P1S_TREND_PRESERVATION.csv", index=False)
    alpha_selection.to_csv(OUTPUTS / "CN_P1S_RIDGE_ALPHA_SELECTION.csv", index=False)
    transform = {
        "model_type": "C2_partial_linear_Ridge", "condition_variables": list(CONDITIONS),
        "condition_scaler_mean": {name: float(value) for name, value in zip(CONDITIONS, final_ridge["scaler"].mean_)},
        "condition_scaler_std": {name: float(value) for name, value in zip(CONDITIONS, final_ridge["scaler"].scale_)},
        "age_spline": {"degree": 3, "n_knots": 5, "expanded_knots_h": [float(value) for value in final_ridge["spline"].bsplines_[0].t]},
        "ridge_alpha": final_alpha, "intercept_V": float(final_ridge["estimator"].intercept_),
        "age_basis_coefficients": [float(value) for value in final_ridge["estimator"].coef_[:final_ridge["age_feature_count"]]],
        "condition_coefficients_standardized": {name: float(value) for name, value in zip(CONDITIONS, final_ridge["estimator"].coef_[-len(CONDITIONS):])},
        "reference_condition": reference_payload["reference_condition"],
        "normalized_reference_voltage_V": reference_payload["final_Ridge_normalized_reference_voltage_V"],
        "development_boundary_ophrs2_h": DEVELOPMENT_BOUNDARY_H,
        "fitting_row_count": int(len(intervals)), "fitting_max_interval_end_ophrs2_h": float(intervals["measurement_end_ophrs2_h"].max()),
        "input_artifact_sha256": summary["frozen_input_sha256"], "test_validated": False,
        "artifact_sha256_sidecar": "FROZEN_CN_P1S_TRANSFORM.yaml.sha256",
    }
    transform_path = CONFIGS / "FROZEN_CN_P1S_TRANSFORM.yaml"
    transform_path.write_text(yaml.safe_dump(transform, sort_keys=False, allow_unicode=True), encoding="utf-8")
    transform_hash = _sha256(transform_path)
    transform_path.with_name(transform_path.name + ".sha256").write_text(f"{transform_hash}  {transform_path.name}\n", encoding="utf-8")
    summary["frozen_transform_created"] = True
    summary["frozen_transform_sha256"] = transform_hash
    (OUTPUTS / "CN_P1S_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_development_trend(development, FIGURES / "Fig1_Raw_vs_CN_P1S_Development.png")
    _plot_anchor_relationship(anchors, FIGURES / "Fig2_Anchor_Health_Relationship.png")
    _plot_condition_sensitivity(decoupling, FIGURES / "Fig3_Condition_Sensitivity_Before_After.png")
    _plot_local_scatter(scatter, FIGURES / "Fig4_Local_Scatter_Before_After.png")
    (REPORTS / "CONDITION_NORMALIZED_P1S_DEVELOPMENT_STUDY.md").write_text(_report(summary, anchors, loao, decoupling, scatter, trend), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

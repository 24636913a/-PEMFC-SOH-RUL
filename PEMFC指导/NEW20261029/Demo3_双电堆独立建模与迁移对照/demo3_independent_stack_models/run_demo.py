#!/usr/bin/env python3
"""Independent stack health models, synthetic data, and an honest transfer baseline.

Only observation CSVs and beginning-of-life calibration enter forecasts.
Ground-truth files are opened AFTER forecasts and are used only for evaluation.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fuelcell_demo3_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ("independent", "pooled_common_rate", "cross_stack_rate")
COLORS = {"independent": "#478F8B", "pooled_common_rate": "#8A8E92", "cross_stack_rate": "#C47758"}
LABELS = {"independent": "Independent model", "pooled_common_rate": "Pooled common rate", "cross_stack_rate": "Other stack rate"}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def generate_data(config: dict, data_dir: Path) -> None:
    """Generate observations and separate truth; never invoked by estimation itself."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["random_seed"])
    jref = config["reference_current_density_A_cm2"]
    tref = config["reference_temperature_C"]
    for sid, spec in config["stacks"].items():
        t = np.arange(0.0, spec["simulation_end_h"] + 0.5 * spec["sampling_interval_h"], spec["sampling_interval_h"])
        phase = t / spec["sampling_interval_h"]
        if spec["load_profile"] == "constant_current":
            j = np.full(len(t), jref)
            temp = tref + 0.8 * np.sin(phase / 31.0)
        else:
            j = np.clip(jref + 0.23 * np.sin(phase / 8.0) + 0.12 * np.sin(phase / 2.3) + rng.normal(0.0, 0.03, len(t)), 0.35, 1.3)
            temp = tref + 3.0 * np.sin(phase / 27.0) + rng.normal(0.0, 0.3, len(t))
        soh = 1.0 - spec["synthetic_soh_loss_rate_per_h"] * t
        latent_cell_v = spec["reference_cell_voltage_V"] * soh
        # Measured-condition terms are independently calibrated in this teaching scenario.
        condition = -config["initial_polarization_slope_V_per_A_cm2"] * (j - jref) + config["temperature_coefficient_V_per_C"] * (temp - tref)
        innovation = rng.normal(0.0, config["measurement_noise_cell_V"], len(t))
        noise = np.empty_like(innovation)
        noise[0] = innovation[0]
        for k in range(1, len(t)):
            noise[k] = 0.35 * noise[k-1] + np.sqrt(1.0 - 0.35**2) * innovation[k]
        obs = pd.DataFrame({
            "time_h": t,
            "stack_voltage_V": spec["n_cells"] * (latent_cell_v + condition + noise),
            "current_density_A_cm2": j,
            "temperature_C": temp,
            "n_cells": spec["n_cells"],
        })
        truth = pd.DataFrame({"time_h": t, "true_soh": soh, "true_reference_cell_voltage_V": latent_cell_v})
        obs.to_csv(data_dir / f"{sid}_observations.csv", index=False, float_format="%.12g")
        truth.to_csv(data_dir / f"{sid}_truth.csv", index=False, float_format="%.12g")
    write_json(data_dir / "provenance.json", {
        "synthetic": True,
        "random_seed": config["random_seed"],
        "description_zh": "完全由 generate_data 生成；FC1_like/RealHyFC_like 只是情景名，不是任何公开或私有电堆的实测记录。5/10片与3倍SOH衰减率均为人为设定。",
        "rate_ratio": 3.0,
        "calibration": "V0, initial polarization slope and temperature coefficient are assumed known from separate beginning-of-life measurements.",
        "config": config,
    })


def load_observations(config: dict, data_dir: Path) -> dict[str, pd.DataFrame]:
    required = ["time_h", "stack_voltage_V", "current_density_A_cm2", "temperature_C", "n_cells"]
    obs = {}
    for sid, spec in config["stacks"].items():
        path = data_dir / f"{sid}_observations.csv"
        if not path.exists():
            raise ValueError(f"Missing {path}; use --regenerate-data only if synthetic regeneration is intended.")
        frame = pd.read_csv(path)
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name}: nonfinite input")
        if not frame["time_h"].is_monotonic_increasing or frame["time_h"].duplicated().any():
            raise ValueError(f"{path.name}: time_h must increase strictly")
        if len(frame) < 4 or (frame["n_cells"] <= 0).any():
            raise ValueError(f"{path.name}: invalid length or cell count")
        if not np.all(frame["n_cells"].to_numpy() == spec["n_cells"]):
            raise ValueError(f"{path.name}: n_cells differs from config; verify physical units")
        if (frame["time_h"] <= spec["train_end_h"]).sum() < 3:
            raise ValueError(f"{path.name}: need at least 3 historical observations")
        obs[sid] = frame
    return obs


def standardize(frame: pd.DataFrame, config: dict, sid: str) -> pd.DataFrame:
    """Reference-condition correction with pre-existing calibration, no fitted future stats."""
    out = frame.copy()
    out["cell_voltage_V"] = out["stack_voltage_V"] / out["n_cells"]
    out["reference_cell_voltage_V"] = (
        out["cell_voltage_V"]
        + config["initial_polarization_slope_V_per_A_cm2"] * (out["current_density_A_cm2"] - config["reference_current_density_A_cm2"])
        - config["temperature_coefficient_V_per_C"] * (out["temperature_C"] - config["reference_temperature_C"])
    )
    out["observed_soh"] = out["reference_cell_voltage_V"] / config["stacks"][sid]["reference_cell_voltage_V"]
    out["raw_voltage_ratio"] = out["cell_voltage_V"] / config["stacks"][sid]["reference_cell_voltage_V"]
    return out


def fit_nonincreasing_line(t: np.ndarray, y: np.ndarray, origin: float) -> dict:
    centered = t - t.mean()
    denom = float(centered @ centered)
    if denom <= 0.0:
        raise ValueError("Cannot estimate a slope from identical timestamps")
    raw_slope = float(centered @ (y - y.mean()) / denom)
    slope = min(raw_slope, 0.0)
    at_origin = float(y.mean() + slope * (origin - t.mean()))
    return {"slope_soh_per_h": slope, "unconstrained_slope_soh_per_h": raw_slope, "soh_at_origin": at_origin, "origin_h": float(origin), "n_training": len(t)}


def build_forecasts(config: dict, observations: dict[str, pd.DataFrame]):
    """No truth parameter. All slope/state fits use time <= each configured train_end_h."""
    standardized = {sid: standardize(obs, config, sid) for sid, obs in observations.items()}
    fits = {}
    training = {}
    pooled_num, pooled_den = 0.0, 0.0
    for sid, spec in config["stacks"].items():
        hist = standardized[sid].loc[standardized[sid]["time_h"] <= spec["train_end_h"]].copy()
        t = hist["time_h"].to_numpy()
        y = hist["observed_soh"].to_numpy()
        fits[sid] = fit_nonincreasing_line(t, y, spec["train_end_h"])
        training[sid] = hist
        tx, yx = t - t.mean(), y - y.mean()
        pooled_num += float(tx @ yx)
        pooled_den += float(tx @ tx)
    shared_slope = min(pooled_num / pooled_den, 0.0)
    all_forecasts = {}
    for sid, spec in config["stacks"].items():
        others = [other for other in fits if other != sid]
        if len(others) != 1:
            raise ValueError("This two-stack teaching demo needs exactly two stacks")
        other = others[0]
        # All methods use the same target-history level at origin: only the slope differs.
        slopes = {"independent": fits[sid]["slope_soh_per_h"], "pooled_common_rate": shared_slope, "cross_stack_rate": fits[other]["slope_soh_per_h"]}
        t_test = standardized[sid].loc[standardized[sid]["time_h"] > spec["train_end_h"], "time_h"].to_numpy()
        pred = pd.DataFrame({"time_h": t_test})
        pred["observed_soh"] = standardized[sid].loc[standardized[sid]["time_h"] > spec["train_end_h"], "observed_soh"].to_numpy()
        for method, slope in slopes.items():
            pred[method] = fits[sid]["soh_at_origin"] + slope * (t_test - spec["train_end_h"])
        all_forecasts[sid] = {"predictions": pred, "slopes": slopes, "origin_h": spec["train_end_h"], "soh_at_origin": fits[sid]["soh_at_origin"], "source_stack_for_transfer": other}
    return standardized, fits, all_forecasts


def rmse(x, y) -> float | None:
    a, b = np.asarray(x), np.asarray(y)
    return float(np.sqrt(np.mean((a-b)**2))) if len(a) else None


def predicted_rul(soh_at_origin: float, slope: float, threshold: float) -> float | None:
    if soh_at_origin <= threshold:
        return 0.0
    if slope >= 0.0:
        return None  # No finite crossing under a flat projected trajectory.
    return float((threshold - soh_at_origin) / slope)


def observed_truth_crossing(truth: pd.DataFrame, threshold: float, origin: float) -> dict:
    """Crossing from recorded truth horizon, with explicit right censoring."""
    t = truth["time_h"].to_numpy(dtype=float)
    s = truth["true_soh"].to_numpy(dtype=float)
    at = np.flatnonzero(s <= threshold)
    if not len(at):
        return {"status": "right_censored", "crossing_h": None, "rul_h": None, "rul_lower_bound_h": max(float(t[-1] - origin), 0.0)}
    k = int(at[0])
    if k == 0:
        crossing = float(t[0])
    else:
        crossing = float(t[k-1] + (threshold - s[k-1]) * (t[k] - t[k-1]) / (s[k] - s[k-1]))
    return {"status": "observed_in_truth", "crossing_h": crossing, "rul_h": max(crossing - origin, 0.0), "rul_lower_bound_h": None}


def load_optional_truth(config: dict, data_dir: Path) -> dict[str, pd.DataFrame | None]:
    truth = {}
    for sid in config["stacks"]:
        path = data_dir / f"{sid}_truth.csv"
        if not path.exists():
            truth[sid] = None
            continue
        frame = pd.read_csv(path)
        if not {"time_h", "true_soh"}.issubset(frame.columns):
            raise ValueError(f"Invalid optional truth file: {path}")
        if not frame["time_h"].is_monotonic_increasing or frame["time_h"].duplicated().any():
            raise ValueError(f"Truth times must be strictly increasing: {path}")
        truth[sid] = frame
    return truth


def evaluate(config: dict, forecasts: dict, truths: dict) -> tuple[pd.DataFrame, dict]:
    rows, crossing_info = [], {}
    for sid, forecast in forecasts.items():
        pred = forecast["predictions"]
        truth = truths.get(sid)
        crossing = observed_truth_crossing(truth, config["threshold_soh"], forecast["origin_h"]) if truth is not None else {"status": "unavailable", "crossing_h": None, "rul_h": None, "rul_lower_bound_h": None}
        crossing_info[sid] = crossing
        merged = pred.merge(truth[["time_h", "true_soh"]], on="time_h", how="inner") if truth is not None else None
        for method, slope in forecast["slopes"].items():
            rul = predicted_rul(forecast["soh_at_origin"], slope, config["threshold_soh"])
            true_rul = crossing["rul_h"]
            err = rul - true_rul if rul is not None and true_rul is not None else None
            rows.append({"stack": sid, "method": method,
                "test_n": len(pred), "train_end_h": forecast["origin_h"],
                "soh_at_origin": forecast["soh_at_origin"], "fitted_soh_loss_rate_per_h": -slope,
                "test_observation_rmse_soh": rmse(pred[method], pred["observed_soh"]),
                "test_latent_rmse_soh": rmse(merged[method], merged["true_soh"]) if merged is not None else None,
                "predicted_rul_h": rul, "true_rul_h": true_rul, "truth_crossing_status": crossing["status"],
                "rul_signed_error_h": err, "rul_absolute_error_h": abs(err) if err is not None else None})
    return pd.DataFrame(rows), crossing_info


def forecast_signature(forecasts: dict, sid: str, method: str) -> np.ndarray:
    f = forecasts[sid]
    return np.r_[f["soh_at_origin"], f["slopes"][method], f["predictions"][method].to_numpy()]


def self_checks(config: dict, observations: dict, forecasts: dict) -> dict:
    checks = {}
    # Future values must not influence any training fit or future predictions at fixed timestamps.
    future_modified = {sid: frame.copy() for sid, frame in observations.items()}
    for sid, frame in future_modified.items():
        mask = frame["time_h"] > config["stacks"][sid]["train_end_h"]
        frame.loc[mask, "stack_voltage_V"] += 1.234
        frame.loc[mask, "current_density_A_cm2"] += 0.432
    _, _, future_forecasts = build_forecasts(config, future_modified)
    for sid in forecasts:
        for method in METHODS:
            np.testing.assert_array_equal(forecast_signature(forecasts, sid, method), forecast_signature(future_forecasts, sid, method))
    checks["future_observation_perturbation_invariance"] = "PASS"

    # Independent predictions must not depend on the other stack's training data.
    for target in observations:
        changed = {sid: frame.copy() for sid, frame in observations.items()}
        for other, frame in changed.items():
            if other != target:
                frame["stack_voltage_V"] += frame["n_cells"] * (-0.00002 * frame["time_h"])
        _, _, altered = build_forecasts(config, changed)
        np.testing.assert_array_equal(forecast_signature(forecasts, target, "independent"), forecast_signature(altered, target, "independent"))
        assert not np.allclose(forecast_signature(forecasts, target, "cross_stack_rate"), forecast_signature(altered, target, "cross_stack_rate"))
    checks["independent_model_no_other_stack_training"] = "PASS"
    checks["transfer_baseline_responds_to_source_change"] = "PASS"

    # Evaluation on deliberately false truth cannot mutate stored forecasts.
    before = {sid: {m: forecast_signature(forecasts, sid, m).copy() for m in METHODS} for sid in forecasts}
    false_truth = {sid: pd.DataFrame({"time_h": o["time_h"], "true_soh": np.full(len(o), 0.10)}) for sid, o in observations.items()}
    evaluate(config, forecasts, false_truth)
    for sid in forecasts:
        for method in METHODS:
            np.testing.assert_array_equal(before[sid][method], forecast_signature(forecasts, sid, method))
    checks["truth_only_evaluation_does_not_mutate_forecasts"] = "PASS"
    changed_generator = copy.deepcopy(config)
    changed_generator["random_seed"] += 999
    for spec in changed_generator["stacks"].values():
        spec["synthetic_soh_loss_rate_per_h"] = 0.5
        spec["simulation_end_h"] = 999999.0
    _, _, changed_generator_forecasts = build_forecasts(changed_generator, observations)
    for sid in forecasts:
        for method in METHODS:
            np.testing.assert_array_equal(before[sid][method], forecast_signature(changed_generator_forecasts, sid, method))
    checks["simulation_truth_parameters_do_not_enter_estimation"] = "PASS"
    assert np.isclose(predicted_rul(0.98, -0.00002, 0.96), 1000.0)
    assert predicted_rul(0.98, 0.0, 0.96) is None
    assert predicted_rul(0.95, -0.00002, 0.96) == 0.0
    censored = observed_truth_crossing(pd.DataFrame({"time_h": [0.0, 100.0], "true_soh": [1.0, 0.98]}), 0.96, 50.0)
    assert censored["status"] == "right_censored" and censored["rul_h"] is None and censored["rul_lower_bound_h"] == 50.0
    checks["rul_units_flat_case_and_censoring"] = "PASS"
    for sid, frame in observations.items():
        std = standardize(frame, config, sid)
        scaled = frame.copy()
        scaled["stack_voltage_V"] *= 2.0
        scaled["n_cells"] *= 2.0
        np.testing.assert_allclose(std["observed_soh"], standardize(scaled, config, sid)["observed_soh"], rtol=0.0, atol=0.0)
    checks["series_cell_count_scale_invariance"] = "PASS"
    for f in forecasts.values():
        assert all(slope <= 0.0 for slope in f["slopes"].values())
        for method in METHODS:
            assert np.all(np.diff(f["predictions"][method]) <= 1e-15)
    checks["nonincreasing_predicted_health"] = "PASS"
    return checks


def make_plots(config: dict, standardized: dict, forecasts: dict, truths: dict, metrics: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 8.1), constrained_layout=True)
    for row, (sid, frame) in enumerate(standardized.items()):
        ax0, ax1 = axs[row]
        ax0.plot(frame["time_h"], frame["stack_voltage_V"], color="#A1A4A7", lw=0.8, label="Measured stack voltage")
        ax0.set(xlabel="Operating time (h)", ylabel="Stack voltage (V)", title=f"{sid}: {config['stacks'][sid]['n_cells']} series cells")
        ax0.axvline(config["stacks"][sid]["train_end_h"], color="#444444", ls="--", lw=0.9)
        ax1.plot(frame["time_h"], frame["raw_voltage_ratio"] * 100, color="#D0D2D3", lw=0.7, label="Raw voltage ratio")
        ax1.plot(frame["time_h"], frame["observed_soh"] * 100, color=COLORS["independent"], lw=0.8, label="Condition-corrected SOH")
        if truths[sid] is not None:
            ax1.plot(truths[sid]["time_h"], truths[sid]["true_soh"] * 100, color="#252525", lw=1.4, label="Synthetic latent truth")
        ax1.set(xlabel="Operating time (h)", ylabel="SOH (%)", title="Cell-count + reference-condition normalization")
        ax0.legend(frameon=False, fontsize=8)
        ax1.legend(frameon=False, fontsize=8)
    fig.suptitle("SYNTHETIC example: normalize units and conditions before modeling", fontsize=14)
    fig.savefig(out / "01_units_and_conditions.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)
    for ax, (sid, f) in zip(axs, forecasts.items()):
        frame = standardized[sid]
        ax.plot(frame["time_h"], frame["observed_soh"] * 100, color="#CBD0D1", lw=0.8, label="Condition-corrected observations")
        if truths[sid] is not None:
            ax.plot(truths[sid]["time_h"], truths[sid]["true_soh"] * 100, color="#242424", lw=1.6, label="Synthetic latent truth")
        pred = f["predictions"]
        for method in METHODS:
            times = np.r_[f["origin_h"], pred["time_h"]]
            values = np.r_[f["soh_at_origin"], pred[method]]
            ax.plot(times, values * 100, color=COLORS[method], lw=1.7, ls="-" if method == "independent" else "--", label=LABELS[method])
        ax.axvline(f["origin_h"], color="#555555", ls=":", lw=1.2)
        ax.axhline(config["threshold_soh"] * 100, color="#735E50", ls=":", lw=1.2, label="Illustrative 4% loss threshold")
        ax.set(xlabel="Operating time (h)", ylabel="SOH (%)", title=sid)
        ax.legend(frameon=False, fontsize=7.7, loc="lower left")
    fig.suptitle("Time-held-out forecasts: the same target level, different learned aging rates", fontsize=13)
    fig.savefig(out / "02_forecast_comparison.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    x = np.arange(len(config["stacks"]))
    width = 0.24
    stacks = list(config["stacks"])
    for i, method in enumerate(METHODS):
        selected = metrics.loc[metrics["method"] == method].set_index("stack").reindex(stacks)
        axs[0].bar(x + (i-1)*width, selected["test_observation_rmse_soh"] * 100, width, color=COLORS[method], label=LABELS[method])
        vals = pd.to_numeric(selected["rul_absolute_error_h"], errors="coerce").to_numpy(dtype=float)
        axs[1].bar(x + (i-1)*width, vals, width, color=COLORS[method], label=LABELS[method])
    axs[0].set(ylabel="Held-out observation RMSE (percentage points)", title="Includes measurement noise")
    axs[1].set(ylabel="RUL absolute error (h)", title="Evaluated only when truth crossing is available")
    observation_max = float(metrics["test_observation_rmse_soh"].max()) * 100
    if np.isfinite(observation_max) and observation_max > 0:
        axs[0].set_ylim(0, observation_max * 1.45)
    rul_errors = pd.to_numeric(metrics["rul_absolute_error_h"], errors="coerce")
    if rul_errors.notna().any() and float(rul_errors.max()) > 0:
        axs[1].set_ylim(0, float(rul_errors.max()) * 1.45)
    for ax in axs:
        ax.set_xticks(x, stacks)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(out / "03_metrics.png", dpi=180)
    plt.close(fig)


def write_result_notes(config: dict, metrics: pd.DataFrame, checks: dict, crossings: dict, out: Path) -> None:
    lines = ["# Demo 3：分别建模的运行结果", "", "本文件由 run_demo.py 根据本次运行的实际输出生成。所有随包数据均为固定种子的模拟数据。", "", "## 主要结果", "", "SOH RMSE 的单位为百分点（原始 SOH 比例乘 100）；RUL 单位为小时。", "", "| 电堆情景 | 方法 | 持出观测 RMSE / 百分点 | 潜态 RMSE / 百分点 | 预测 RUL / h | 真实 RUL / h | RUL 绝对误差 / h |", "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(v, scale=1.0):
        return "不可评价" if pd.isna(v) else f"{float(v)*scale:.4f}"
    for row in metrics.to_dict("records"):
        lines.append(f"| {row['stack']} | {row['method']} | {fmt(row['test_observation_rmse_soh'], 100)} | {fmt(row['test_latent_rmse_soh'], 100)} | {fmt(row['predicted_rul_h'])} | {fmt(row['true_rul_h'])} | {fmt(row['rul_absolute_error_h'])} |")
    lines += ["", "## 怎样解释", "", "- 这些数字验证程序在给定合成情景中的行为，不能作为 FC1 或 RealHyFC 真实数据的实验结论。", "- 分别建模只使用对应电堆截点之前的数据。两个迁移/共享对照在起点采用与独立模型完全相同的目标堆健康水平，仅替换衰减斜率。", "- 真实 SOH 的下降速率在模拟中固定为 3∶1；独立模型已匹配模拟器的线性形式，因此此例没有验证复杂非线性老化的建模能力。", "- 电池数归一化、工况修正改善可比性；它们不会使不同电堆的衰减速度相等。", "- 0.96 SOH 表示相对参考电压下降 4%，是本次演示阈值；不是电压只剩 4%，也不是通用报废标准。", "- RUL 以各自训练截点为起点。在未来观测未覆盖阈值时，真 RUL 记为右删失，绝对误差保持空值；没有 truth 时只跳过真值评价。", "- 拟合状态与 RUL 都使用同一参考工况电压口径；不能把此预测直接作为变载下最低瞬时运行电压的到达时间。", "", "## 检查结果", ""]
    lines += [f"- `{name}`：{status}" for name, status in checks.items()]
    lines += ["", "## 阈值到达记录", "", "```json", json.dumps(crossings, ensure_ascii=False, indent=2), "```", "", "## 文件索引", "", "- `metrics.csv` / `metrics.json`：各方法误差与 RUL。", "- `fitted_models.json`：训练范围、估计斜率、预测起点。", "- `*_standardized.csv`：单位与参考工况换算结果。", "- `*_predictions.csv`：严格持出段的预测；truth 列仅为事后评价追加。", "- `01_units_and_conditions.png`：电堆电压与标准化健康量。", "- `02_forecast_comparison.png`：分别建模、共享斜率和直接迁移对照。", "- `03_metrics.png`：误差对照。", "- `checks.json`：自动检查。", "- `RUN_LOG.txt` / `environment.json`：运行日志与实际软件版本。", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--regenerate-data", action="store_true", help="Explicitly overwrite synthetic CSVs in --data-dir using config.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not 0 < config["threshold_soh"] < 1:
        raise ValueError("threshold_soh must lie between 0 and 1")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = []
    def say(message):
        text = str(message)
        print(text)
        log.append(text)
    if args.regenerate_data:
        generate_data(config, args.data_dir)
        say("Regenerated explicitly synthetic data using the configured fixed seed.")
    observations = load_observations(config, args.data_dir)
    standardized, fits, forecasts = build_forecasts(config, observations)
    # Only NOW read truth. No latent labels enter fitting or forecasting.
    truths = load_optional_truth(config, args.data_dir)
    metrics, crossings = evaluate(config, forecasts, truths)
    checks = self_checks(config, observations, forecasts)
    for sid, frame in standardized.items():
        frame.to_csv(out / f"{sid}_standardized.csv", index=False, float_format="%.12g")
        pred = forecasts[sid]["predictions"].copy()
        if truths[sid] is not None:
            pred = pred.merge(truths[sid][["time_h", "true_soh"]], on="time_h", how="left")
        pred.to_csv(out / f"{sid}_predictions.csv", index=False, float_format="%.12g")
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.12g")
    metric_records = json.loads(metrics.to_json(orient="records"))
    write_json(out / "metrics.json", {"synthetic": True, "threshold_soh": config["threshold_soh"], "metrics": metric_records, "truth_crossings": crossings})
    write_json(out / "fitted_models.json", {"independent_fits": fits, "forecast_parameters": {sid: {key: value for key, value in f.items() if key != "predictions"} for sid, f in forecasts.items()}})
    rate_comparison = []
    for sid, fit in fits.items():
        truth = truths[sid]
        true_rate = None
        if truth is not None and len(truth) >= 2:
            true_rate = float(-(truth["true_soh"].iloc[-1] - truth["true_soh"].iloc[0]) / (truth["time_h"].iloc[-1] - truth["time_h"].iloc[0]))
        rate_comparison.append({"stack": sid, "estimated_loss_rate_soh_per_h": -fit["slope_soh_per_h"], "truth_average_loss_rate_soh_per_h": true_rate, "train_end_h": fit["origin_h"], "n_training": fit["n_training"]})
    pd.DataFrame(rate_comparison).to_csv(out / "rate_comparison.csv", index=False, float_format="%.12g")
    write_json(out / "checks.json", checks)
    write_json(out / "environment.json", {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib")}})
    write_json(out / "used_config.json", config)
    make_plots(config, standardized, forecasts, truths, metrics, out)
    write_result_notes(config, metrics, checks, crossings, out)
    say("Demo 3: independent stack models, pooled rate, and source-rate transfer")
    say("Data provenance: SYNTHETIC, not measured FC1/RealHyFC records")
    say(f"Threshold: SOH = {config['threshold_soh']:.4f}; loss = {(1-config['threshold_soh'])*100:.1f}%")
    say(metrics[["stack", "method", "test_latent_rmse_soh", "predicted_rul_h", "true_rul_h", "rul_absolute_error_h"]].to_string(index=False))
    say(f"Meaningful self-checks: {len(checks)} passed")
    say("Predictions use no future observation values and no truth labels.")
    say("Done. See RESULTS.md, plots, predictions and metrics in the output directory.")
    (out / "RUN_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

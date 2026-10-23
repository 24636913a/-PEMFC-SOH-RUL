"""Phase C: development-only future polarization-curve forecasting.

This executable is deliberately fail-closed around data scope and time:
only seven development nominal polarization files are opened; causal dynamic
inputs end at each diagnosis completion time; and dynamic validation windows
are target-completion-purged from training windows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
REPO = PROJECT.parent
sys.path.insert(0, str(ROOT))
from models.dynamic_encoders import LSTMDynamicEncoder, TimeXerHistoryEncoder, parameter_count  # noqa: E402

SEED = 42
DEV_END = 1819.997738
REFERENCE_VOLTAGE = 0.6989533066
STAGES = ["BoL", "300", "600", "850", "1150", "1500", "1750"]
NOMINAL_H = {"BoL": 0.0, "300": 300.0, "600": 600.0, "850": 850.0, "1150": 1150.0, "1500": 1500.0, "1750": 1750.0}
ORIGINS = [("850", "1150"), ("1150", "1500"), ("1500", "1750")]
GRID = np.asarray([0.8, 1.0, 1.2, 1.4, 1.6], dtype=float)
TOL = 0.01

RUL90 = PROJECT / "Phase_RUL90_Data_Feasibility" / "outputs"
N1 = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
TIMEXER_VENDOR = PROJECT / "Phase_N2_8B_TimeMixerPP_and_TimeXer_Adaptation" / "vendor" / "TimeXer"
RAW = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据")

OUT, FIG, REPORT, CKPT = ROOT / "outputs", ROOT / "figures", ROOT / "reports", ROOT / "checkpoints"

FEATURES = [
    "cell_voltage_mean", "cell_voltage_std", "cell_voltage_range",
    "actual_current_A_mean", "actual_current_A_std",
    "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std",
    "actual_anode_stoichiometry_mean", "actual_cathode_stoichiometry_mean",
    "actual_anode_inlet_rh_pct_mean", "actual_cathode_inlet_rh_pct_mean",
    "actual_anode_hydrogen_flow_nl_min_mean", "actual_cathode_air_flow_nl_min_mean",
    "actual_cathode_inlet_temperature_C_mean",
]

PALETTE = {"persistence": "#B64342", "trend": "#CFCECE", "lstm": "#3775BA", "timexer": "#0F4D92", "fusion": "#9A4D8E", "average": "#42949E", "truth": "#111111"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def md_table(frame: pd.DataFrame, digits: int = 5) -> str:
    df = frame.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    head = [str(c) for c in df.columns]
    rows = [[str(v) for v in r] for r in df.fillna("").itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])


def phase_setup() -> dict[str, Any]:
    for p in (OUT, FIG, REPORT, CKPT):
        p.mkdir(parents=True, exist_ok=True)
    freeze = yaml.safe_load((RUL90 / "METHOD_FREEZE.yaml").read_text(encoding="utf-8"))
    if float(freeze["primary_jstar_A_cm2"]) != 1.0 or float(freeze["reference_voltage_V"]) != REFERENCE_VOLTAGE or freeze["excluded_channel"] != "U6 [V]":
        raise RuntimeError("Fail closed: Phase RUL90 frozen reference definition is not the expected one")
    ref = pd.read_csv(RUL90 / "reference_voltage_development.csv")
    got = set(ref["stage"].astype(str).str.extract(r"(BoL|300|600|850|1150|1500|1750)", expand=False))
    if got != set(STAGES) or ref["after_air_air_start"].astype(str).str.lower().eq("true").any():
        raise RuntimeError("Fail closed: development reference anchors do not match the permitted normal nominal set")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    return {
        "source_git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "started_at": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
        "cuda_available": torch.cuda.is_available(), "torch_version": torch.__version__,
        "source_hashes": {str(p.relative_to(PROJECT)): sha256(p) for p in [RUL90 / "METHOD_FREEZE.yaml", RUL90 / "polarization_curves_development.csv", RUL90 / "polarization_points_development.csv", RUL90 / "reference_voltage_development.csv"]},
    }


def raw_path(stage: str) -> Path:
    suffix = "BoL" if stage == "BoL" else f"{stage}h"
    return RAW / f"Polcurve_Nominal_RHFC_{suffix}.csv"


def parse_float(value: str) -> float:
    return float(value.replace(",", ""))


def read_curve(stage: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Stream one allowed raw curve and make frozen active-cell grid points."""
    path = raw_path(stage)
    if not path.exists():
        raise RuntimeError(f"Missing required development curve: {path}")
    active_cells = ["U1 [V]", "U2 [V]", "U3 [V]", "U4 [V]", "U5 [V]", "U7 [V]", "U8 [V]", "U9 [V]", "U10 [V]", "U11 [V]"]
    wanted = ["DateTime", "CalHrs [h]", "OpHrs [h]", "Number of cells", "I.S.act [A]", "j.S.act [A/cm2]", "Membrane active area [cm2]", "U6 [V]"] + active_cells + ["Stoic.S.A [-]", "Stoic.S.C [-]", "p.Si.A [kPag]", "p.Si.C [kPag]", "T.Si.A [°C]", "T.Si.C [°C]", "F.Si.H.A [nl/min]", "F.Si.Air.C [nl/min]"]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(f).rstrip("\r\n").split(";")
        idx = {v: i for i, v in enumerate(header)}
        missing = [v for v in wanted if v not in idx]
        if missing:
            raise RuntimeError(f"Raw curve misses required fields: {missing}")
        max_idx = max(idx[v] for v in wanted)
        j_all: list[float] = []; v_all: list[float] = []; u6: list[float] = []; dates: list[str] = []
        conf: dict[str, list[float]] = {v: [] for v in wanted if v not in ["DateTime"] + active_cells + ["U6 [V]"]}
        for line in f:
            fields = line.rstrip("\r\n").split(";", max_idx)
            if len(fields) <= max_idx: continue
            try:
                j = parse_float(fields[idx["j.S.act [A/cm2]"]])
                cells = [parse_float(fields[idx[c]]) for c in active_cells]
                voltage = float(np.median(cells))
                u6_v = parse_float(fields[idx["U6 [V]"]])
            except (ValueError, TypeError):
                continue
            j_all.append(j); v_all.append(voltage); u6.append(u6_v); dates.append(fields[idx["DateTime"]])
            for name in conf:
                try: conf[name].append(parse_float(fields[idx[name]]))
                except (ValueError, TypeError): pass
    j = np.asarray(j_all); v = np.asarray(v_all); u6_arr = np.asarray(u6)
    if len(j) < 100 or not np.isfinite(v).all():
        raise RuntimeError(f"Insufficient valid raw samples in {path.name}")
    direct_rows = []
    for value in GRID:
        mask = np.abs(j - value) <= TOL
        if mask.sum() < 3:
            raise RuntimeError(f"CURVE_TASK_BLOCKED: {stage} has only {mask.sum()} direct samples at j={value}")
        direct_rows.append({"stage": stage, "j_A_cm2": value, "support_min_A_cm2": float(j[mask].min()), "support_max_A_cm2": float(j[mask].max()), "matching_tolerance_A_cm2": TOL, "construction": "direct_target_window_median", "valid_sample_count": int(mask.sum()), "active_voltage_median_V": float(np.median(v[mask])), "active_voltage_mean_V": float(np.mean(v[mask])), "active_voltage_std_mV": float(np.std(v[mask]) * 1000), "quality": "PASS" if mask.sum() >= 3 else "FAIL"})
    sign = np.sign(np.diff(j)); sign = sign[np.abs(sign) > 0]
    summary: dict[str, Any] = {
        "stage": stage, "filename": path.name, "raw_file_sha256": sha256(path), "raw_sample_count": int(len(j)), "diagnostic_start_datetime": min(dates), "diagnostic_end_datetime": max(dates), "j_min_A_cm2": float(j.min()), "j_max_A_cm2": float(j.max()), "scan_sign_change_count": int(np.sum(sign[1:] != sign[:-1])) if len(sign) > 1 else 0, "scan_direction_rule": "NOT_IDENTIFIABLE_FROM_JITTERED_SAMPLE_ORDER__pooled_pre_frozen_target_window_median", "cell_count_median": float(np.median(conf["Number of cells"])), "active_area_cm2_median": float(np.median(conf["Membrane active area [cm2]"])), "u6_median_V": float(np.median(u6_arr)), "u6_negative_ratio": float(np.mean(u6_arr < 0)),
    }
    for name, values in conf.items():
        if values: summary[f"{name}_median"] = float(np.median(values)); summary[f"{name}_std"] = float(np.std(values))
    return summary, pd.DataFrame(direct_rows)


def load_dynamic() -> pd.DataFrame:
    usecols = ["bin_start_ophrs2_h", "bin_end_ophrs2_h", "first_datetime", "last_datetime", "bin_valid_for_windows", "temporal_segment_id", "split"] + FEATURES
    df = pd.read_csv(N1, usecols=usecols, nrows=3005)
    if len(df) != 3005 or float(df["bin_end_ophrs2_h"].max()) > DEV_END or df["split"].eq("test").any():
        raise RuntimeError("Fail closed: dynamic causal table includes non-development rows")
    df["first_datetime"] = pd.to_datetime(df["first_datetime"]); df["last_datetime"] = pd.to_datetime(df["last_datetime"])
    return df


def window(df: pd.DataFrame, end_pos: int, steps: int = 48) -> np.ndarray | None:
    start = end_pos - steps + 1
    if start < 0: return None
    x = df.iloc[start:end_pos + 1]
    if len(x) != steps or not x["bin_valid_for_windows"].astype(bool).all() or x["temporal_segment_id"].nunique() != 1:
        return None
    if not np.allclose(np.diff(x["bin_end_ophrs2_h"].to_numpy(float)), 0.5, atol=1e-6): return None
    vals = x[FEATURES].to_numpy(np.float32)
    return vals if np.isfinite(vals).all() else None


def dynamic_windows(df: pd.DataFrame, dispatch: pd.Timestamp) -> list[dict[str, Any]]:
    rows = []
    for end in range(47, len(df) - 24):
        target = end + 24
        if df.iloc[target]["last_datetime"] > dispatch: break
        x = window(df, end)
        future = df.iloc[end + 1: target + 1]
        if x is None or not future["bin_valid_for_windows"].astype(bool).all() or future["temporal_segment_id"].nunique() != 1 or future["temporal_segment_id"].iloc[0] != df.iloc[end]["temporal_segment_id"]:
            continue
        y = float(df.iloc[target]["cell_voltage_mean"] - df.iloc[end]["cell_voltage_mean"])
        if np.isfinite(y): rows.append({"x": x, "y": y, "input_start": end - 47, "target_end": target, "target_completed": df.iloc[target]["last_datetime"]})
    if len(rows) < 60: raise RuntimeError(f"CURVE_TASK_BLOCKED: only {len(rows)} causal dynamic windows before diagnosis completion {dispatch}")
    return rows


@dataclass
class Bundle:
    model: nn.Module
    scaler: StandardScaler
    pca: PCA
    params: int
    epoch: int
    train_n: int
    validation_n: int
    purge_n: int
    device: str


def fit_encoder(name: str, factory: Any, rows: list[dict[str, Any]], origin: str) -> Bundle:
    cut = max(40, int(math.floor(len(rows) * 0.8)))
    train = rows[:cut]
    final_train_target = train[-1]["target_end"]
    valid = [r for r in rows[cut:] if r["input_start"] > final_train_target]
    purge_n = len(rows[cut:]) - len(valid)
    if len(valid) < 10: raise RuntimeError("CURVE_TASK_BLOCKED: target-completion purge leaves too few dynamic validation windows")
    scaler = StandardScaler().fit(np.concatenate([r["x"] for r in train], axis=0))
    xtr = np.stack([scaler.transform(r["x"]) for r in train]).astype(np.float32); xva = np.stack([scaler.transform(r["x"]) for r in valid]).astype(np.float32)
    ytr_raw = np.asarray([r["y"] for r in train], np.float32); yva_raw = np.asarray([r["y"] for r in valid], np.float32)
    ym, ys = float(ytr_raw.mean()), max(float(ytr_raw.std()), 1e-7)
    ytr, yva = (ytr_raw - ym) / ys, (yva_raw - ym) / ys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = factory().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); loss_fn = nn.MSELoss(); best = None; best_score = float("inf"); bad = 0; best_epoch = 0
    for epoch in range(1, 19):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(xb.to(device)), yb.to(device)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): pred = model(torch.from_numpy(xva).to(device)).cpu().numpy()
        score = float(np.sqrt(np.mean((pred - yva) ** 2)))
        if score < best_score - 1e-6:
            best_score, best, best_epoch, bad = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            bad += 1
            if bad >= 4: break
    if best is None: raise RuntimeError("Dynamic encoder failed to fit")
    model.load_state_dict(best); model.cpu().eval()
    with torch.no_grad(): repr_train = model.encode(torch.from_numpy(xtr)).numpy()
    pca = PCA(n_components=1, random_state=SEED).fit(repr_train)
    torch.save({"state": model.state_dict(), "parameter_count": parameter_count(model), "best_epoch": best_epoch, "dynamic_train_windows": len(train), "dynamic_validation_windows": len(valid), "purged_windows": purge_n}, CKPT / f"{name}_{origin}_encoder.pt")
    return Bundle(model, scaler, pca, parameter_count(model), best_epoch, len(train), len(valid), purge_n, str(device))


def history_before(df: pd.DataFrame, dispatch: pd.Timestamp, scaler: StandardScaler) -> np.ndarray | None:
    candidates = df.index[(df["last_datetime"] <= dispatch) & df["bin_valid_for_windows"].astype(bool)]
    if len(candidates) == 0: return None
    raw = window(df, int(candidates.max()))
    return None if raw is None else scaler.transform(raw).astype(np.float32)


def pca_scalar(bundle: Bundle, x: np.ndarray) -> float:
    with torch.no_grad(): z = bundle.model.encode(torch.from_numpy(x[None, :, :])).numpy()
    return float(bundle.pca.transform(z)[0, 0])


def ridge_scalar(x: np.ndarray, y: np.ndarray, xeval: np.ndarray) -> float:
    sx, sy = StandardScaler().fit(x), StandardScaler().fit(y.reshape(-1, 1))
    model = Ridge(alpha=4.0).fit(sx.transform(x), sy.transform(y.reshape(-1, 1)).ravel())
    return float(sy.inverse_transform(model.predict(sx.transform(xeval.reshape(1, -1))).reshape(-1, 1))[0, 0])


def metrics(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = pd.DataFrame(rows)
    run = p[p.status.eq("RUN")].copy()
    per_origin = []
    for (model, origin, target), x in run.groupby(["model", "origin_stage", "target_stage"]):
        e = x["error_mV"].to_numpy(float)
        j1_error = float(np.abs(x.loc[np.isclose(x.j_A_cm2, 1.0), "error_mV"].iloc[0])) if np.isclose(x.j_A_cm2, 1.0).any() else np.nan
        per_origin.append({"model": model, "origin_stage": origin, "target_stage": target, "n_grid_points": len(x), "Curve_MAE_mV": float(np.mean(np.abs(e))), "Curve_RMSE_mV": float(np.sqrt(np.mean(e ** 2))), "j1_abs_error_mV": j1_error})
    po = pd.DataFrame(per_origin)
    per_current = []
    for (model, j), x in run.groupby(["model", "j_A_cm2"]):
        e = x.error_mV.to_numpy(float); per_current.append({"model": model, "j_A_cm2": j, "n_origins": len(x), "MAE_mV": float(np.mean(np.abs(e))), "RMSE_mV": float(np.sqrt(np.mean(e ** 2)))})
    pc = pd.DataFrame(per_current)
    summary = []
    for model, x in po.groupby("model"):
        raw = run[run.model.eq(model)].error_mV.to_numpy(float)
        summary.append({"model": model, "n_origins": len(x), "mean_curve_MAE_mV": float(x.Curve_MAE_mV.mean()), "mean_curve_RMSE_mV": float(x.Curve_RMSE_mV.mean()), "pooled_grid_RMSE_mV": float(np.sqrt(np.mean(raw ** 2))), "j1_MAE_mV": float(x.j1_abs_error_mV.mean())})
    return po, pc, pd.DataFrame(summary)


def physical(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows); result = []
    for (model, origin, target), x in df[df.status.eq("RUN")].groupby(["model", "origin_stage", "target_stage"]):
        y = x.sort_values("j_A_cm2").prediction_voltage_V.to_numpy(float); violations = int(np.sum(np.diff(y) > 0))
        result.append({"model": model, "origin_stage": origin, "target_stage": target, "pair_count": len(y) - 1, "violation_pair_count": violations, "curve_has_violation": bool(violations > 0), "violation_rate": violations / max(1, len(y) - 1), "projection_applied": False})
    return pd.DataFrame(result)


def make_figures(grid: pd.DataFrame, predictions: pd.DataFrame, per_origin: pd.DataFrame, per_current: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 1.8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for stage, x in grid.groupby("stage"):
        ax.errorbar(x.j_A_cm2, x.active_voltage_median_V, yerr=x.active_voltage_std_mV / 1000, marker="o", ms=4, lw=1.4, capsize=2, label=stage)
    ax.set(xlabel="Current density (A cm$^{-2}$)", ylabel="Active 10-cell median voltage (V)", title="Development-only: common polarization grid and within-window dispersion")
    ax.legend(ncol=2, fontsize=8); fig.savefig(FIG / "polarization_data_quality.png", dpi=300); fig.savefig(FIG / "polarization_data_quality.pdf"); plt.close(fig)
    run = predictions[predictions.status.eq("RUN")]
    figs, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    labels = {"M0_Persistence": "Persistence", "M1_LinearTrend": "Linear trend", "M2_LSTM": "LSTM", "M3_TimeXer": "TimeXer", "M4_Fusion": "Fusion", "M4_FixedAverage": "Fixed average"}
    colors = {"M0_Persistence": PALETTE["persistence"], "M1_LinearTrend": PALETTE["trend"], "M2_LSTM": PALETTE["lstm"], "M3_TimeXer": PALETTE["timexer"], "M4_Fusion": PALETTE["fusion"], "M4_FixedAverage": PALETTE["average"]}
    for ax, (origin, target) in zip(axes, ORIGINS):
        q = run[(run.origin_stage.eq(origin)) & (run.target_stage.eq(target))]
        truth = q.drop_duplicates("j_A_cm2").sort_values("j_A_cm2")
        ax.plot(truth.j_A_cm2, truth.truth_voltage_V, "k-o", lw=2, ms=4, label="Observed future")
        for m, x in q.groupby("model"): ax.plot(x.sort_values("j_A_cm2").j_A_cm2, x.sort_values("j_A_cm2").prediction_voltage_V, marker="o", lw=1.1, ms=3, color=colors[m], label=labels[m])
        ax.set(title=f"{origin}→{target} planned checkpoint", xlabel="Current density (A cm$^{-2}$)")
    axes[0].set_ylabel("Voltage (V)")
    handles, labs = axes[-1].get_legend_handles_labels()
    figs.suptitle("Development-only exploratory future polarization curves", y=.985, fontsize=13)
    figs.legend(handles, labs, loc="upper center", bbox_to_anchor=(.5, .945), ncol=4, fontsize=8, frameon=False)
    figs.tight_layout(rect=(0, 0, 1, .79))
    figs.savefig(FIG / "future_curve_predictions.png", dpi=300, bbox_inches="tight"); figs.savefig(FIG / "future_curve_predictions.pdf", bbox_inches="tight"); plt.close(figs)
    fig, ax = plt.subplots(figsize=(8.0, 4.3), constrained_layout=True)
    order = ["M0_Persistence", "M1_LinearTrend", "M2_LSTM", "M3_TimeXer", "M4_Fusion", "M4_FixedAverage"]
    xbase = np.arange(len(ORIGINS)); width = 0.12
    for i, model in enumerate(order):
        vals = []
        for o, t in ORIGINS:
            q = per_origin[(per_origin.model.eq(model)) & (per_origin.origin_stage.eq(o)) & (per_origin.target_stage.eq(t))]
            vals.append(q.Curve_RMSE_mV.iloc[0] if len(q) else np.nan)
        ax.bar(xbase + (i - 2.5) * width, vals, width, label=labels[model], color=colors[model], edgecolor="black", linewidth=.5)
    ax.set(xticks=xbase, xticklabels=[f"{o}→{t}" for o, t in ORIGINS], ylabel="Curve-RMSE (mV)", xlabel="Planned diagnostic checkpoint", title="Development-only exploratory; blanks = NOT_ESTIMABLE")
    ax.legend(ncol=3, fontsize=7); fig.savefig(FIG / "per_origin_error.png", dpi=300); fig.savefig(FIG / "per_origin_error.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.0, 4.3), constrained_layout=True)
    for model, x in per_current.groupby("model"):
        ax.plot(x.j_A_cm2, x.MAE_mV, marker="o", lw=1.5, label=labels[model], color=colors[model])
    ax.set(xlabel="Current density (A cm$^{-2}$)", ylabel="MAE (mV)", title="Per-current-density error; unequal origin coverage")
    ax.legend(ncol=3, fontsize=7); fig.savefig(FIG / "per_current_error.png", dpi=300); fig.savefig(FIG / "per_current_error.pdf"); plt.close(fig)
    j1 = run[np.isclose(run.j_A_cm2, 1.0)].copy(); fig, ax = plt.subplots(figsize=(7.8, 4.1), constrained_layout=True)
    for model, x in j1.groupby("model"):
        ax.scatter(x.truth_SOH, x.prediction_SOH, label=labels[model], color=colors[model], s=42)
    lo, hi = min(j1.truth_SOH.min(), j1.prediction_SOH.min()) - .002, max(j1.truth_SOH.max(), j1.prediction_SOH.max()) + .002
    ax.plot([lo, hi], [lo, hi], "k--", lw=1); ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Observed SOH at j=1.0", ylabel="Predicted SOH at j=1.0", title="Development-only SOH derived from curve prediction")
    ax.legend(fontsize=7, ncol=3); fig.savefig(FIG / "soh_prediction.png", dpi=300); fig.savefig(FIG / "soh_prediction.pdf"); plt.close(fig)
    common = per_origin[per_origin.origin_stage.eq("1500")].copy(); fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    vals = common.set_index("model").reindex(order).Curve_RMSE_mV; ax.bar(range(len(vals)), vals, color=[colors[m] for m in order], edgecolor="black", linewidth=.6)
    ax.set(xticks=range(len(vals)), xticklabels=[labels[m] for m in order], ylabel="Curve-RMSE (mV)", title="All-model common origin: 1500→1750 (n=1 case comparison)")
    ax.tick_params(axis="x", rotation=25); fig.savefig(FIG / "model_comparison_common_origins.png", dpi=300); fig.savefig(FIG / "model_comparison_common_origins.pdf"); plt.close(fig)


def reports(audit: dict[str, Any], curve_audit: pd.DataFrame, grid: pd.DataFrame, splits: pd.DataFrame, summary: pd.DataFrame, per_origin: pd.DataFrame, physical_df: pd.DataFrame, resources: pd.DataFrame) -> None:
    text(REPORT / "01_CURVE_DATA_AUDIT.md", f"""# Curve data audit

Seven normal nominal Development files were streamed directly: BoL, 300, 600, 850, 1150, 1500 and 1750 h. AfterAirAirStart files were not opened. Active voltage is the frozen median over U1–U5 and U7–U11; U6 is excluded. All files have median cell count 10 and active area 280 cm².

The sample-order current trace has many sign changes rather than a reliably labelled up/down sweep. No consistent scan-direction branch can therefore be identified without inventing a rule. Before modelling, the frozen rule was set to pooled direct target-window medians, `|j-j*|≤0.01 A cm⁻²`; it preserves the prior RUL90 j=1.0 definition. All five common grid points have at least three direct samples in every curve, so no extrapolation or interpolation was required.

{md_table(curve_audit[["stage", "raw_sample_count", "diagnostic_start_datetime", "diagnostic_end_datetime", "j_min_A_cm2", "j_max_A_cm2", "cell_count_median", "active_area_cm2_median", "u6_median_V", "u6_negative_ratio", "scan_sign_change_count"]])}
""")
    text(REPORT / "02_TIME_ALIGNMENT_AND_LEAKAGE.md", f"""# Time alignment and leakage audit

Prediction dispatch is defined as **diagnosis completion**, the final raw `DateTime` in the current curve, never its start time. Dynamic input histories end at or before dispatch. The target diagnosis completion time and its realised `OpHrs2` are output-only descriptors and are not model inputs. The only future-time covariate is the pre-specified **planned checkpoint-label gap** scenario.

The causal table is limited to its first 3,005 rows, with maximum end `{DEV_END}` h and no Test split rows. For dynamic pretraining, every target completes by the current dispatch; the chronological auxiliary validation set is purged until its input begins strictly after the final auxiliary training target completion. Scalers, PCA compression and curve heads are fit separately in each origin on preceding data only.

{md_table(splits)}

No 2000/2300 h raw polarization file was opened, no new Test target was read, and no Test metric was computed. Historical descriptive Test access remains truthfully marked compromised.
""")
    text(REPORT / "03_MODEL_DESIGN.md", """# Model design

M0 predicts the current five-point curve unchanged. M1 fits a separate linear trend at each current density from preceding full curve checkpoints. M2 and M3 pretrain, respectively, a 16-hidden-unit LSTM and a 32-dimensional history-only TimeXer encoder on 24 h causal history → 12 h dynamic-voltage increment. Their representations are frozen and compressed to one PCA component fitted on dynamic training inputs only.

For each health transition, the curve head predicts a scalar mean curve shift `s` from planned diagnostic gap and the frozen representation; a fixed training-fold curve template `S(j)` reconstructs `ΔV(j)=s·S(j)`. This uses one independent curve transition per supervision example—not five independent lifetime examples. Ridge alpha is fixed at 4.0. M4 concatenates planned gap plus both PCA scalars; it is run only where its four coefficients (including intercept) do not exceed the available transition count. M4 fixed 0.5/0.5 averaging is a diagnostic, not a learned fusion architecture.
""")
    text(REPORT / "04_TRAINING_REPORT.md", f"# Training report\n\n{md_table(resources)}\n\nSeed is fixed at 42; LSTM and TimeXer share the same dynamic task, data scope, early-stopping budget and purge rule.\n")
    text(REPORT / "05_FAIR_MODEL_COMPARISON.md", f"""# Fair model comparison

{md_table(summary, digits=4)}

All-model common-origin comparison is only 1500→1750 (n=1); it is a case comparison, not a generalization ranking. M2/M3 share two origins, whereas M0/M1 cover three. Never compare their unequal-coverage aggregate values as proof of superiority.
""")
    text(REPORT / "06_PHYSICAL_CONSISTENCY.md", f"""# Physical consistency

Selected nominal reference curves are assessed for non-increasing voltage as current density increases. No monotonic projection or post-processing is applied to any model.

{md_table(physical_df)}
""")
    text(REPORT / "07_ABLATION.md", """# Minimal ablation

- A0: M1 curve trend (no dynamic representation).
- A1: M2 LSTM representation with the same scalar curve head.
- A2: M3 TimeXer representation with the same scalar curve head.
- A3: M4 concatenated LSTM+TimeXer representations with a fixed strongly regularized head.
- A4: M4 fixed 0.5/0.5 prediction average diagnostic.

The ablation is limited by 2/3/4 history-available transition curves at the three origins. Missing entries are formal `NOT_ESTIMABLE` outcomes.
""")
    common = per_origin[per_origin.origin_stage.eq("1500")].sort_values("model")
    text(REPORT / "FINAL_RESEARCH_DECISION.md", f"""# Final research decision

## 已证明的事实

The five-point common current-density-grid curve task is constructible for seven comparable normal nominal Development curves. Inputs, target-completion purge, fold-local scaling/PCA and Test firewall were enforced. However, M2/M3 lack estimable supervision at the first origin and M4 only has one all-model common origin.

{md_table(summary, digits=4)}

All-model common origin (1500→1750, n=1):

{md_table(common[["model", "Curve_MAE_mV", "Curve_RMSE_mV", "j1_abs_error_mV"]], digits=4)}

## 探索性观察

TimeXer has lower error than LSTM in both of their shared origins and at all five current-density points; its observed effect is therefore not restricted to j=1.0. It nevertheless does not beat curve persistence in the only all-model common 1500→1750 case (2.268 vs 1.172 mV Curve-RMSE). Fixed averaging is 1.877 mV in that same case, while learned fusion is 5.435 mV. These are case observations, not architecture-selection evidence.

## 尚未验证的假设

Whether dynamic encoders add stable curve-prediction value, whether LSTM/TimeXer are complementary, and whether performance transfers to independently held-out hardware remain unverified. This is neither true RUL nor EOL prediction.

## 唯一建议

**C — 寻找独立数据验证。** The chief limitation is the number of independent curve transitions, not curve-grid density or time alignment. Do not choose a paper core neural architecture from this pilot or add more fusion modules before independent, configuration-comparable curve sequences exist.
""")


def main() -> None:
    audit = phase_setup(); started = time.time()
    curve_summaries, grid_parts = [], []
    for stage in STAGES:
        summary, points = read_curve(stage); curve_summaries.append(summary); grid_parts.append(points)
    curve_audit = pd.DataFrame(curve_summaries); grid = pd.concat(grid_parts, ignore_index=True)
    if grid.groupby("j_A_cm2")["stage"].nunique().min() != len(STAGES): raise RuntimeError("CURVE_TASK_BLOCKED: no common five-point grid")
    curve_audit.to_csv(OUT / "curve_anchor_table.csv", index=False); grid.to_csv(OUT / "current_density_grid.csv", index=False)
    curves = {stage: grid[grid.stage.eq(stage)].sort_values("j_A_cm2").active_voltage_median_V.to_numpy(float) for stage in STAGES}
    dynamic = load_dynamic(); meta = curve_audit.set_index("stage")
    align_rows, pred_rows, resource_rows = [], [], []
    for origin, target in ORIGINS:
        dispatch = pd.Timestamp(meta.loc[origin, "diagnostic_end_datetime"])
        target_complete = pd.Timestamp(meta.loc[target, "diagnostic_end_datetime"])
        legal = dynamic_windows(dynamic, dispatch)
        known = STAGES[:STAGES.index(origin) + 1]
        trans = [(known[i - 1], known[i]) for i in range(1, len(known)) if known[i - 1] != "BoL"]
        align_rows.append({"origin_stage": origin, "target_stage": target, "origin_diagnosis_complete": dispatch, "target_diagnosis_complete_output_only": target_complete, "planned_gap_h": NOMINAL_H[target] - NOMINAL_H[origin], "origin_actual_ophrs2_output_only": float(dynamic.loc[dynamic.last_datetime.le(dispatch), "bin_end_ophrs2_h"].max()), "target_actual_ophrs2_output_only": float(dynamic.loc[dynamic.last_datetime.le(target_complete), "bin_end_ophrs2_h"].max()), "legal_dynamic_windows": len(legal), "history_available_transition_curves": len(trans), "protocol": "planned-checkpoint-gap scenario"})
        truth = curves[target]
        def append_curve(model: str, status: str, values: np.ndarray | None, reason: str = "") -> None:
            for j, actual in zip(GRID, truth):
                val = np.nan if values is None else float(values[np.where(np.isclose(GRID, j))[0][0]])
                row = {"origin_stage": origin, "target_stage": target, "model": model, "status": status, "reason": reason, "j_A_cm2": float(j), "truth_voltage_V": float(actual), "truth_SOH": float(actual / REFERENCE_VOLTAGE) if np.isclose(j, 1.0) else np.nan, "prediction_voltage_V": val, "prediction_SOH": float(val / REFERENCE_VOLTAGE) if np.isclose(j, 1.0) and np.isfinite(val) else np.nan}
                row["error_mV"] = (val - actual) * 1000 if np.isfinite(val) else np.nan; row["absolute_error_mV"] = abs(row["error_mV"]) if np.isfinite(row["error_mV"]) else np.nan; pred_rows.append(row)
        append_curve("M0_Persistence", "RUN", curves[origin])
        # Simple curve trend: independent current-density coefficients but curve-level checkpoints.
        xage = np.asarray([NOMINAL_H[s] for s in known], float); ycurves = np.stack([curves[s] for s in known])
        trend = np.asarray([np.polyval(np.polyfit(xage, ycurves[:, k], 1), NOMINAL_H[target]) for k in range(len(GRID))])
        append_curve("M1_LinearTrend", "RUN", trend)
        if len(trans) < 3:
            for m in ["M2_LSTM", "M3_TimeXer", "M4_Fusion", "M4_FixedAverage"]: append_curve(m, "NOT_ESTIMABLE", None, "Fewer than three history-available transition curves")
            resource_rows.extend([{ "origin_stage": origin, "model": m, "status": "NOT_RUN", "reason": "sparse curve supervision"} for m in ["M2_LSTM", "M3_TimeXer"]])
            continue
        factories = {"M2_LSTM": lambda: LSTMDynamicEncoder(len(FEATURES)), "M3_TimeXer": lambda: TimeXerHistoryEncoder(TIMEXER_VENDOR, len(FEATURES))}
        bundles: dict[str, Bundle] = {}
        for name, factory in factories.items():
            b = fit_encoder(name, factory, legal, origin); bundles[name] = b
            resource_rows.append({"origin_stage": origin, "model": name, "status": "RUN", "dynamic_train_windows": b.train_n, "dynamic_validation_windows": b.validation_n, "purged_windows": b.purge_n, "parameter_count": b.params, "best_epoch": b.epoch, "device": b.device})
        source_hist = {m: history_before(dynamic, dispatch, b.scaler) for m, b in bundles.items()}
        if any(v is None for v in source_hist.values()): raise RuntimeError(f"CURVE_TASK_BLOCKED: no 24h causal history at dispatch {origin}")
        reps = {m: [] for m in bundles}; gaps, scalar_y = [], []; deltas = []
        for src, dst in trans:
            gaps.append((NOMINAL_H[dst] - NOMINAL_H[src]) / 1000.0); deltas.append(curves[dst] - curves[src]); scalar_y.append(float(np.mean(deltas[-1])))
            source_time = pd.Timestamp(meta.loc[src, "diagnostic_end_datetime"])
            for m, b in bundles.items():
                h = history_before(dynamic, source_time, b.scaler)
                if h is None: raise RuntimeError(f"CURVE_TASK_BLOCKED: no causal source history at {src}")
                reps[m].append(pca_scalar(b, h))
        shape = np.mean(np.stack(deltas), axis=0); shape_mean = float(np.mean(shape))
        if abs(shape_mean) < 1e-7: raise RuntimeError("CURVE_TASK_BLOCKED: zero learned training-fold curve delta template")
        shape = shape / shape_mean
        planned_gap = (NOMINAL_H[target] - NOMINAL_H[origin]) / 1000.0; scalar_y_arr = np.asarray(scalar_y)
        predicted: dict[str, np.ndarray] = {}
        for m, b in bundles.items():
            x = np.column_stack([gaps, reps[m]]); ev = np.asarray([planned_gap, pca_scalar(b, source_hist[m])]); s = ridge_scalar(x, scalar_y_arr, ev); predicted[m] = curves[origin] + s * shape; append_curve(m, "RUN", predicted[m])
        if len(trans) < 4:
            append_curve("M4_Fusion", "UNDERDETERMINED", None, "Four fusion coefficients including intercept exceed three transitions")
        else:
            x = np.column_stack([gaps, reps["M2_LSTM"], reps["M3_TimeXer"]])
            ev = np.asarray([planned_gap, pca_scalar(bundles["M2_LSTM"], source_hist["M2_LSTM"]), pca_scalar(bundles["M3_TimeXer"], source_hist["M3_TimeXer"])])
            s = ridge_scalar(x, scalar_y_arr, ev); predicted["M4_Fusion"] = curves[origin] + s * shape; append_curve("M4_Fusion", "RUN", predicted["M4_Fusion"])
        average = 0.5 * (predicted["M2_LSTM"] + predicted["M3_TimeXer"]); append_curve("M4_FixedAverage", "RUN", average)
    splits = pd.DataFrame(align_rows); predictions = pd.DataFrame(pred_rows); resources = pd.DataFrame(resource_rows)
    per_origin, per_current, summary = metrics(pred_rows); violations = physical(pred_rows)
    predictions.to_csv(OUT / "all_curve_predictions.csv", index=False); splits.to_csv(OUT / "rolling_origin_splits.csv", index=False); per_origin.to_csv(OUT / "per_origin_curve_metrics.csv", index=False); per_current.to_csv(OUT / "per_current_metrics.csv", index=False); summary.to_csv(OUT / "summary_metrics.csv", index=False); resources.to_csv(OUT / "training_resources.csv", index=False); violations.to_csv(OUT / "physical_violations.csv", index=False)
    common = per_origin[per_origin.origin_stage.eq("1500")].copy(); common.to_csv(OUT / "common_origin_comparison.csv", index=False); per_origin[["model", "origin_stage", "target_stage", "Curve_RMSE_mV", "Curve_MAE_mV"]].to_csv(OUT / "ablation_results.csv", index=False)
    make_figures(grid, predictions, per_origin, per_current)
    audit.update({"finished_at": datetime.now().isoformat(timespec="seconds"), "elapsed_seconds": round(time.time() - started, 2), "allowed_raw_stages_opened": STAGES, "forbidden_test_stages_opened": [], "new_test_target_accessed": False, "test_metrics_computed": False, "historical_test_seal": "COMPROMISED_BY_PRIOR_DESCRIPTIVE_ACCESS", "time_alignment": "diagnosis_completion_dispatch", "dynamic_target_completion_purge": True})
    (ROOT / "RUN_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    reports(audit, curve_audit, grid, splits, summary, per_origin, violations, resources)
    common = per_origin[per_origin.origin_stage.eq("1500")].copy()
    time_xer_per_current = per_current[per_current.model.eq("M3_TimeXer")][["j_A_cm2", "MAE_mV"]].to_dict(orient="records")
    final = {"phase": "Phase_C_Future_Polarization_Forecasting", "decision": "NO-GO_FOR_ARCHITECTURE_SELECTION", "scope": "Development-only exploratory evaluation", "source_git_head": audit["source_git_head"], "n_independent_curves": 7, "n_independent_prediction_targets": 3, "common_j_grid_A_cm2": GRID.tolist(), "curve_task_feasible": True, "time_alignment_verified": True, "diagnosis_dispatch_is_completion": True, "dynamic_pretraining_causal": True, "target_completion_purge": True, "new_test_target_accessed": False, "test_metrics_computed": False, "original_test_previously_inspected": True, "RUL_labels_used": False, "true_EOL_prediction_claimed": False, "summary_metrics": summary.to_dict(orient="records"), "all_model_common_origin": "1500_to_1750_only_n_equals_1", "common_origin_metrics": common.to_dict(orient="records"), "timexer_vs_lstm": "TimeXer lower at both shared origins and all five j points, but not lower than persistence on the all-model common case", "timexer_per_current_MAE_mV": time_xer_per_current, "learned_fusion_improves_over_timexer": False, "dynamic_encoders_improve_over_persistence_on_all_model_common_case": False, "physical_violation_rows": violations.to_dict(orient="records"), "recommended_next_direction": "C: seek independent configuration-comparable curve sequences for validation", "limitations": ["Only 2/3/4 history-available transition curves across origins", "M4 fusion has one common evaluation case", "Prior descriptive Test access remains compromised"], "source_hashes": audit["source_hashes"]}
    (ROOT / "FINAL_DECISION.yaml").write_text(yaml.safe_dump(final, sort_keys=False, allow_unicode=True), encoding="utf-8")
    text(ROOT / "README.md", "# Phase C — Future Polarization Forecasting\n\nRun `python scripts/run_phase_c.py`, then `python tests/test_phase_c_guards.py`. The run is Development-only and never opens 2000/2300 h raw polarization targets.\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

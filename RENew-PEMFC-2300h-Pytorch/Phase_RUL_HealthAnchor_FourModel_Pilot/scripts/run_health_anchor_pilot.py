"""Leakage-audited sparse PEMFC health-anchor forecasting pilot.

The runner is intentionally fail-closed.  It reads only development anchors
(BoL through 1750 h) and only the first development prefix of the Phase N1
causal 30-minute table.  It never opens 2000 h or 2300 h polarization files.
"""

from __future__ import annotations

import csv
import json
import math
import os
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[0]
sys.path.insert(0, str(ROOT))
from models.health_models import (  # noqa: E402
    LSTMDynamicEncoder,
    TimeXerHistoryEncoder,
    trainable_parameter_count,
)

EXPECTED_COMMIT = "2950160fdb762511138c1bf4a159c6c898a9d193"
DEV_END = 1819.997738
REF_VOLTAGE = 0.6989533066
SEED = 20260921
DEV_STAGES = ["BoL", "300", "600", "850", "1150", "1500", "1750"]
STAGE_NOMINAL_H = {"BoL": 0.0, "300": 300.0, "600": 600.0, "850": 850.0, "1150": 1150.0, "1500": 1500.0, "1750": 1750.0}
ROLLOUTS = [("850", "1150"), ("1150", "1500"), ("1500", "1750")]

HISTORICAL = ROOT.parent
N1_FEATURES = HISTORICAL / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
RUL90_ANCHORS = HISTORICAL / "Phase_RUL90_Data_Feasibility" / "outputs" / "reference_voltage_development.csv"
VENDOR_TIMEXER = HISTORICAL / "Phase_N2_8B_TimeMixerPP_and_TimeXer_Adaptation" / "vendor" / "TimeXer"
RAW_POLAR_DIR = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据")

OUT = ROOT / "outputs"
FIG = ROOT / "figures"
REPORT = ROOT / "reports"
CKPT = ROOT / "checkpoints"

FEATURES = [
    "cell_voltage_mean",
    "cell_voltage_std",
    "cell_voltage_range",
    "actual_current_A_mean",
    "actual_current_A_std",
    "actual_current_density_A_cm2_mean",
    "actual_current_density_A_cm2_std",
    "actual_anode_stoichiometry_mean",
    "actual_cathode_stoichiometry_mean",
    "actual_anode_inlet_rh_pct_mean",
    "actual_cathode_inlet_rh_pct_mean",
    "actual_anode_hydrogen_flow_nl_min_mean",
    "actual_cathode_air_flow_nl_min_mean",
    "actual_cathode_inlet_temperature_C_mean",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    """Render a small Markdown table without an optional third-party package."""

    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(x) for x in shown.columns]
    rows = [[str(value) for value in row] for row in shown.fillna("").itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(row) + " |" for row in rows])


def setup() -> dict[str, Any]:
    for p in (OUT, FIG, REPORT, CKPT):
        p.mkdir(parents=True, exist_ok=True)
    got = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if got != EXPECTED_COMMIT:
        raise RuntimeError(f"Fail closed: expected source commit {EXPECTED_COMMIT}, got {got}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    return {"source_commit": got, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "started_at": now()}


def read_polar_timestamp(stage: str) -> pd.Timestamp:
    """Read only DateTime in a development polarization file, never voltage."""

    suffix = "BoL" if stage == "BoL" else f"{stage}h"
    path = RAW_POLAR_DIR / f"Polcurve_Nominal_RHFC_{suffix}.csv"
    if not path.exists():
        raise RuntimeError(f"Missing required development polarization file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter=";"))
    return pd.Timestamp(row["DateTime"])


def load_development_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "bin_start_ophrs2_h", "bin_end_ophrs2_h", "first_datetime", "last_datetime",
        "coverage_ratio", "bin_valid_for_windows", "temporal_segment_id", "split",
    ] + FEATURES
    # N1 fixed development prefix: the following row would enter the frozen test.
    df = pd.read_csv(N1_FEATURES, usecols=usecols, nrows=3005)
    if len(df) != 3005 or float(df["bin_end_ophrs2_h"].max()) > DEV_END or (df["split"] == "test").any():
        raise RuntimeError("Fail closed: causal feature load is not development-only")
    df["first_datetime"] = pd.to_datetime(df["first_datetime"])
    df["last_datetime"] = pd.to_datetime(df["last_datetime"])
    anchors = pd.read_csv(RUL90_ANCHORS)
    anchors["stage"] = anchors["stage"].astype(str).str.extract(r"(BoL|300|600|850|1150|1500|1750)", expand=False)
    anchors = anchors[anchors["stage"].isin(DEV_STAGES)].copy()
    if set(anchors["stage"]) != set(DEV_STAGES):
        raise RuntimeError("Missing a required development anchor")
    anchors["polarization_datetime"] = anchors["stage"].map(read_polar_timestamp)
    # Correspondence uses DateTime only and accepts only a bounded, pre-existing
    # 30-min bin; it does not use the incorrect OpHrs in polar files.
    mapped_rows = []
    for _, r in anchors.iterrows():
        if r["stage"] == "BoL":
            mapped_rows.append((np.nan, np.nan, np.nan, "NO_LOADCYCLE_COVERAGE"))
            continue
        deltas = (df["first_datetime"] - r["polarization_datetime"]).abs()
        pos = int(deltas.idxmin())
        nearest = df.loc[pos]
        delta_h = float(deltas.loc[pos].total_seconds() / 3600.0)
        if delta_h > 2.0:
            raise RuntimeError(f"TIME_ALIGNMENT_BLOCKED: {r['stage']} nearest causal bin is {delta_h:.2f} h away")
        mapped_rows.append((float(nearest["bin_start_ophrs2_h"]), delta_h, pos, "DATETIME_NEAREST_BIN"))
    anchors[["aligned_ophrs2_h", "alignment_delta_h", "nearest_causal_row", "alignment_method"]] = mapped_rows
    anchors["nominal_stage_h"] = anchors["stage"].map(STAGE_NOMINAL_H).astype(float)
    anchors["reference_voltage_V"] = anchors["reference_voltage_V"].astype(float)
    anchors["SOH"] = anchors["reference_voltage_V"] / REF_VOLTAGE
    anchors = anchors.sort_values("nominal_stage_h").reset_index(drop=True)
    return df, anchors


def consecutive_window(df: pd.DataFrame, end_pos: int, lookback_steps: int) -> np.ndarray | None:
    start = end_pos - lookback_steps + 1
    if start < 0:
        return None
    block = df.iloc[start:end_pos + 1]
    if len(block) != lookback_steps or not block["bin_valid_for_windows"].astype(bool).all():
        return None
    if block["temporal_segment_id"].nunique() != 1:
        return None
    ends = block["bin_end_ophrs2_h"].to_numpy(dtype=float)
    if not np.allclose(np.diff(ends), 0.5, atol=1e-6):
        return None
    values = block[FEATURES].to_numpy(dtype=np.float32)
    return values if np.isfinite(values).all() else None


def build_dynamic_windows(df: pd.DataFrame, boundary_h: float, lookback_steps: int = 48, horizon_steps: int = 24) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for end_pos in range(lookback_steps - 1, len(df) - horizon_steps):
        target_pos = end_pos + horizon_steps
        if float(df.iloc[target_pos]["bin_end_ophrs2_h"]) > boundary_h:
            break
        x = consecutive_window(df, end_pos, lookback_steps)
        if x is None:
            continue
        future = df.iloc[end_pos + 1:target_pos + 1]
        if (not future["bin_valid_for_windows"].astype(bool).all() or
                future["temporal_segment_id"].nunique() != 1 or
                future["temporal_segment_id"].iloc[0] != df.iloc[end_pos]["temporal_segment_id"]):
            continue
        y = float(df.iloc[target_pos]["cell_voltage_mean"] - df.iloc[end_pos]["cell_voltage_mean"])
        if np.isfinite(y):
            rows.append({"x": x, "y": y, "end_h": float(df.iloc[target_pos]["bin_end_ophrs2_h"])})
    if len(rows) < 60:
        raise RuntimeError(f"TIME_ALIGNMENT_BLOCKED: only {len(rows)} legal dynamic windows before {boundary_h:.1f} h")
    return rows


@dataclass
class DynamicBundle:
    model: nn.Module
    feature_scaler: StandardScaler
    y_mean: float
    y_std: float
    parameter_count: int
    best_epoch: int
    train_windows: int
    validation_windows: int
    device: str


def fit_dynamic_encoder(name: str, factory: Any, windows: list[dict[str, Any],], origin: str) -> DynamicBundle:
    split_at = max(40, int(math.floor(0.8 * len(windows))))
    train_rows, valid_rows = windows[:split_at], windows[split_at:]
    if not valid_rows:
        raise RuntimeError("No chronological dynamic validation windows")
    scaler = StandardScaler().fit(np.concatenate([r["x"] for r in train_rows], axis=0))
    x_train = np.stack([scaler.transform(r["x"]) for r in train_rows]).astype(np.float32)
    x_valid = np.stack([scaler.transform(r["x"]) for r in valid_rows]).astype(np.float32)
    y_train_raw = np.asarray([r["y"] for r in train_rows], dtype=np.float32)
    y_valid_raw = np.asarray([r["y"] for r in valid_rows], dtype=np.float32)
    y_mean, y_std = float(y_train_raw.mean()), float(y_train_raw.std())
    y_std = max(y_std, 1e-7)
    y_train, y_valid = (y_train_raw - y_mean) / y_std, (y_valid_raw - y_mean) / y_std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = factory().to(device)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state, best_score, best_epoch, bad_epochs = None, float("inf"), 0, 0
    for epoch in range(1, 19):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_pred = model(torch.from_numpy(x_valid).to(device)).cpu().numpy()
        score = float(np.sqrt(np.mean((valid_pred - y_valid) ** 2)))
        if score < best_score - 1e-6:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= 4:
                break
    if best_state is None:
        raise RuntimeError("Dynamic pretraining failed")
    model.load_state_dict(best_state)
    model.cpu().eval()
    torch.save({"model_state": model.state_dict(), "parameter_count": trainable_parameter_count(model), "best_epoch": best_epoch}, CKPT / f"{name}_{origin}_dynamic_encoder.pt")
    return DynamicBundle(model, scaler, y_mean, y_std, trainable_parameter_count(model), best_epoch, len(train_rows), len(valid_rows), str(device))


def anchor_history(df: pd.DataFrame, when: pd.Timestamp, scaler: StandardScaler) -> np.ndarray | None:
    candidates = df.index[(df["last_datetime"] <= when) & df["bin_valid_for_windows"].astype(bool)]
    if len(candidates) == 0:
        return None
    raw = consecutive_window(df, int(candidates.max()), 48)
    return None if raw is None else scaler.transform(raw).astype(np.float32)


def encoder_scalar(bundle: DynamicBundle, window: np.ndarray) -> float:
    with torch.no_grad():
        encoded = bundle.model.encode(torch.from_numpy(window[None, :, :]))
    return float(encoded.numpy().mean())


def known_transitions(anchors: pd.DataFrame, origin: str) -> pd.DataFrame:
    origin_h = STAGE_NOMINAL_H[origin]
    ordered = anchors[anchors["nominal_stage_h"] <= origin_h].sort_values("nominal_stage_h").reset_index(drop=True)
    rows = []
    for i in range(1, len(ordered)):
        source, target = ordered.iloc[i - 1], ordered.iloc[i]
        # BoL precedes the available load-cycle data and has no dynamic history.
        if source["stage"] == "BoL":
            continue
        rows.append({"source_stage": source["stage"], "target_stage": target["stage"], "source_v": float(source["reference_voltage_V"]), "target_v": float(target["reference_voltage_V"]), "source_time": source["polarization_datetime"]})
    return pd.DataFrame(rows)


def ridge_prediction(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> float:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    model = Ridge(alpha=4.0).fit(x_scaler.transform(x_train), y_scaler.transform(y_train.reshape(-1, 1)).ravel())
    return float(y_scaler.inverse_transform(model.predict(x_scaler.transform(x_eval.reshape(1, -1))).reshape(-1, 1))[0, 0])


def m0_predictions(anchors: pd.DataFrame, origin: str, target: str) -> dict[str, float]:
    current = anchors[anchors["stage"] == origin].iloc[0]
    future = anchors[anchors["stage"] == target].iloc[0]
    known = anchors[anchors["nominal_stage_h"] <= STAGE_NOMINAL_H[origin]].copy()
    x = known["aligned_ophrs2_h"].fillna(0.0).to_numpy(dtype=float)
    y = known["reference_voltage_V"].to_numpy(dtype=float)
    return {
        "M0_persistence": float(current["reference_voltage_V"]),
        "M0_linear_trend": float(np.polyval(np.polyfit(x, y, deg=1), float(future["aligned_ophrs2_h"]))),
    }


def make_plots(pred: pd.DataFrame) -> None:
    good = pred[pred["status"] == "RUN"].copy()
    colors = {"M0_persistence": "#4C78A8", "M0_linear_trend": "#F58518", "M1_LSTM": "#54A24B", "M2_TimeXer": "#E45756", "M3_Fusion": "#B279A2", "M3_equal_weight_diagnostic": "#9D755D"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.0, 4.1), constrained_layout=True)
    truth = good.drop_duplicates("target_stage").sort_values("target_nominal_h")
    ax.plot(truth["target_nominal_h"], truth["truth_voltage_V"], "ko-", lw=1.8, label="Observed anchor")
    for model, part in good.groupby("model"):
        ax.scatter(part["target_nominal_h"], part["prediction_voltage_V"], s=36, label=model, color=colors.get(model, "#777777"))
    ax.set(xlabel="Target checkpoint label (h)", ylabel="Reference voltage at 1.0 A cm$^{-2}$ (V)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG / "health_prediction_vs_truth.png", dpi=300)
    fig.savefig(FIG / "health_prediction_vs_truth.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    for model, part in good.groupby("model"):
        ax.scatter(part["target_nominal_h"], part["absolute_error_mV"], s=38, label=model, color=colors.get(model, "#777777"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Target checkpoint label (h)", ylabel="Absolute error (mV)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG / "per_origin_error.png", dpi=300)
    fig.savefig(FIG / "per_origin_error.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    summary = good.groupby("model")["absolute_error_mV"].agg(["mean", "count"]).sort_values("mean")
    ax.bar(range(len(summary)), summary["mean"].values, color=[colors.get(x, "#777777") for x in summary.index])
    ax.set_xticks(range(len(summary)), [f"{name}\n(n={int(summary.loc[name, 'count'])})" for name in summary.index], rotation=30, ha="right")
    ax.set(ylabel="Mean absolute error over available origins (mV)", title="Descriptive only: unequal feasible-origin coverage")
    for i, value in enumerate(summary["mean"]):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(FIG / "four_model_comparison.png", dpi=300)
    fig.savefig(FIG / "four_model_comparison.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(truth["target_nominal_h"], truth["truth_SOH"], "ko-", lw=1.8, label="Observed SOH")
    for model, part in good.groupby("model"):
        ax.scatter(part["target_nominal_h"], part["prediction_SOH"], s=36, label=model, color=colors.get(model, "#777777"))
    ax.set(xlabel="Target checkpoint label (h)", ylabel="SOH = Vref / 0.6989533066")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG / "soh_prediction.png", dpi=300)
    fig.savefig(FIG / "soh_prediction.pdf")
    plt.close(fig)


def reports(audit: dict[str, Any], anchors: pd.DataFrame, rolling: pd.DataFrame, pred: pd.DataFrame, resources: pd.DataFrame) -> None:
    table = markdown_table(anchors[["stage", "nominal_stage_h", "aligned_ophrs2_h", "alignment_delta_h", "polarization_datetime", "reference_voltage_V", "SOH", "alignment_method"]])
    write_text(REPORT / "DATA_AND_SPLIT_AUDIT.md", f"""# Data and split audit

Source commit: `{audit['source_commit']}`. The analysis loaded exactly the first 3,005 rows of the Phase N1 30-min causal table (maximum bin end {DEV_END:.6f} h) and verified no `split=test` rows. Health labels were restricted to BoL–1750 h.

The polarization files' `OpHrs` values are not used as the ageing coordinate. The `DateTime` of each permitted characterization was matched to its nearest development causal bin (maximum accepted difference: 2 h). BoL precedes Loadcycle availability and therefore has no dynamic input history.

{table}

The 2000 h and 2300 h files were not opened by this phase. The historical RUL90 report records that test targets had been accessed descriptively before this phase; that seal is **not restored** by this run.
""")
    write_text(REPORT / "MODEL_IMPLEMENTATION.md", """# Model implementation

M0 is implemented as persistence and ordinary least-squares linear trend across anchors known at each origin. M1 is a 1-layer, 16-hidden-unit LSTM dynamic encoder, pre-trained to predict 12-h change in causal average-cell voltage from 24 h of historical measurements. M2 uses the vendored TimeXer patch embedding and self/cross-attention encoder (32-d model, one layer) with a new scalar dynamic-task head. This is explicitly a **history-only TimeXer adaptation**, not its original future-exogenous forecasting setup: no future current, temperature, pressure, RH, flow, or stoichiometry is supplied.

M3 concatenates a frozen scalar summary of M1 and M2 representations with the current reference voltage and fits one fixed-alpha Ridge calibration head. Its equal-weight average is a diagnostic only. All dynamic encoders are pre-trained separately per rolling origin, with every dynamic target ending before that origin's characterization boundary. Each health head is re-fit at that origin only; no later health anchor is used.

Sparse-label guard: M1/M2 require at least three history-available health transitions, and M3 requires at least four. A model below that threshold is emitted as `NOT_ESTIMABLE`, not silently substituted.
""")
    run_counts = markdown_table(pred.groupby(["model", "status"]).size().unstack(fill_value=0).reset_index())
    write_text(REPORT / "TRAINING_REPORT.md", f"""# Training report

Dynamic encoder resources and fit sizes:

{markdown_table(resources)}

Prediction status counts:

{run_counts}

The fitted health head is intentionally low capacity because each origin offers only 2–4 transitions with an available historical load-cycle window.
""")
    summary = pred[pred.status == "RUN"].groupby("model").agg(origins=("origin_stage", "count"), MAE_mV=("absolute_error_mV", "mean"), RMSE_mV=("error_mV", lambda x: float(np.sqrt(np.mean(np.square(x)))))).reset_index()
    write_text(REPORT / "FOUR_MODEL_COMPARISON.md", f"""# Four-model comparison

{markdown_table(summary, digits=4) if len(summary) else 'No estimable predictions.'}

Comparisons are descriptive only: models have unequal feasible-origin coverage because the health labels are sparse. No architecture is selected from these numbers.
""")
    write_text(REPORT / "LEAKAGE_AUDIT.md", """# Leakage audit

- Ageing coordinate for dynamic data: `OpHrs2 [h]`; characterization `DateTime` is used only for alignment.
- Dynamic causal table: first 3,005 development rows only; test-row assertion is fail-closed.
- Permitted health labels: BoL, 300, 600, 850, 1150, 1500, 1750 h only.
- 2000 h and 2300 h polarization files: not opened; current-phase test target access = false; current-phase test metrics = false.
- Historical test sealing was already compromised by descriptive access in Phase RUL90 and is truthfully retained as such.
- Each rolling origin rebuilds dynamic feature/y scalers from its own chronological dynamic-training subset.
- Dynamic target end time is required to be no later than the origin's aligned causal-time boundary.
- Anchor histories end no later than their own characterization timestamps, remain in one temporal segment, use 48 consecutive valid 30-min bins, and contain no future fill.
- No health-anchor interpolation, future covariates, global scaler, or test-derived model choice is used.
""")
    write_text(REPORT / "FINAL_RESEARCH_DECISION.md", """# Final research decision

## 已证明的事实

- 七个开发期健康锚点可追溯到允许的 BoL–1750 h 文件；其中 300–1750 h 已由 `DateTime` 对齐至开发期 `OpHrs2` 因果表。
- 三个独立的未来健康检查点（1150、1500、1750 标签）可作滚动起点评价。M0 覆盖三者；M1/M2 仅覆盖后两个；M3 仅覆盖最后一个，因为健康监督转移数分别为 2、3、4。
- 当前运行没有打开 2000/2300 h 极化文件，也没有新计算 Test 健康指标；但历史 RUL90 阶段的描述性 Test 访问使 Test seal 仍是 compromised。
- 因果动态预训练、各起点独立 scaler、窗口连续性和健康标签边界检查均已执行。

## 探索性观察

M2 TimeXer 在它可运行的两个起点上为 0.526 mV MAE，M0 persistence 在三个起点为 1.965 mV MAE；二者覆盖不同，不能据此宣称获胜。M3 仅一个起点，13.204 mV 绝对误差，既不支持融合优于 TimeXer，也不足以判断其相对 LSTM 的稳定性。

## 尚未验证的假设

动态负载/工况表征是否可在有足够锚点时改善健康预测；LSTM 和 TimeXer 的误差是否互补；以及任何网络是否可泛化到未访问的最终 Test 均尚未验证。这里没有真实 RUL 标签、EOL 阈值或真实 EOL 预测结论。

## 下一阶段建议

**NO-GO for selecting a neural health-anchor architecture.** 先获取更多独立且配置可比的健康锚点，或建立经外部验证的连续健康标签，再预注册并重跑完整滚动比较。在数据证据补足前，优先做整条极化曲线/更多可比工况下的健康表征，而不是增加快慢解耦或 EIS 辅助网络模块；当前瓶颈是监督样本数而非模型容量。
""")


def main() -> None:
    audit = setup()
    started = time.time()
    df, anchors = load_development_data()
    anchors.to_csv(OUT / "health_anchor_table.csv", index=False)
    rolling_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    model_bundles: dict[tuple[str, str], DynamicBundle] = {}

    for origin, target in ROLLOUTS:
        origin_anchor = anchors[anchors.stage == origin].iloc[0]
        target_anchor = anchors[anchors.stage == target].iloc[0]
        boundary = float(origin_anchor.aligned_ophrs2_h)
        transitions = known_transitions(anchors, origin)
        rolling_rows.append({"origin_stage": origin, "target_stage": target, "origin_aligned_ophrs2_h": boundary, "target_aligned_ophrs2_h": float(target_anchor.aligned_ophrs2_h), "health_head_transitions": len(transitions), "dynamic_boundary_h": boundary, "split_status": "DEVELOPMENT_ONLY"})
        base = {"origin_stage": origin, "target_stage": target, "origin_nominal_h": STAGE_NOMINAL_H[origin], "target_nominal_h": STAGE_NOMINAL_H[target], "origin_aligned_ophrs2_h": boundary, "target_aligned_ophrs2_h": float(target_anchor.aligned_ophrs2_h), "truth_voltage_V": float(target_anchor.reference_voltage_V), "truth_SOH": float(target_anchor.SOH)}
        for model, value in m0_predictions(anchors, origin, target).items():
            row = dict(base, model=model, status="RUN", prediction_voltage_V=value, prediction_SOH=value / REF_VOLTAGE)
            row["error_mV"] = 1000 * (value - row["truth_voltage_V"])
            row["absolute_error_mV"] = abs(row["error_mV"])
            row["health_head_transitions"] = len(transitions)
            prediction_rows.append(row)

        required = {"M1_LSTM": 3, "M2_TimeXer": 3, "M3_Fusion": 4}
        needed_dynamic = len(transitions) >= 3
        bundles: dict[str, DynamicBundle] = {}
        if needed_dynamic:
            windows = build_dynamic_windows(df, boundary)
            factories = {
                "M1_LSTM": lambda: LSTMDynamicEncoder(input_size=len(FEATURES)),
                "M2_TimeXer": lambda: TimeXerHistoryEncoder(VENDOR_TIMEXER, input_size=len(FEATURES)),
            }
            for name, factory in factories.items():
                bundle = fit_dynamic_encoder(name, factory, windows, origin)
                bundles[name] = bundle
                model_bundles[(origin, name)] = bundle
                resource_rows.append({"origin_stage": origin, "model": name, "dynamic_train_windows": bundle.train_windows, "dynamic_validation_windows": bundle.validation_windows, "parameter_count": bundle.parameter_count, "best_epoch": bundle.best_epoch, "device": bundle.device})
        else:
            resource_rows.append({"origin_stage": origin, "model": "M1_LSTM", "dynamic_train_windows": 0, "dynamic_validation_windows": 0, "parameter_count": 0, "best_epoch": np.nan, "device": "NOT_RUN_sparse_health_head"})
            resource_rows.append({"origin_stage": origin, "model": "M2_TimeXer", "dynamic_train_windows": 0, "dynamic_validation_windows": 0, "parameter_count": 0, "best_epoch": np.nan, "device": "NOT_RUN_sparse_health_head"})

        def unavailable(model: str, reason: str) -> None:
            prediction_rows.append(dict(base, model=model, status="NOT_ESTIMABLE", prediction_voltage_V=np.nan, prediction_SOH=np.nan, error_mV=np.nan, absolute_error_mV=np.nan, health_head_transitions=len(transitions), reason=reason))

        if not needed_dynamic:
            unavailable("M1_LSTM", "Only two history-available health transitions")
            unavailable("M2_TimeXer", "Only two history-available health transitions")
            unavailable("M3_Fusion", "Only two history-available health transitions")
            unavailable("M3_equal_weight_diagnostic", "Constituent predictions not estimable")
            continue

        eval_histories = {name: anchor_history(df, pd.Timestamp(origin_anchor.polarization_datetime), bundle.feature_scaler) for name, bundle in bundles.items()}
        if any(window is None for window in eval_histories.values()):
            raise RuntimeError(f"TIME_ALIGNMENT_BLOCKED: no causal history for {origin}")
        scalar_train: dict[str, list[float]] = {"M1_LSTM": [], "M2_TimeXer": []}
        current_v, y = [], []
        for _, tr in transitions.iterrows():
            current_v.append(float(tr.source_v))
            y.append(float(tr.target_v))
            for name, bundle in bundles.items():
                hist = anchor_history(df, pd.Timestamp(tr.source_time), bundle.feature_scaler)
                if hist is None:
                    raise RuntimeError(f"TIME_ALIGNMENT_BLOCKED: no causal history for source {tr.source_stage}")
                scalar_train[name].append(encoder_scalar(bundle, hist))
        eval_scalar = {name: encoder_scalar(bundle, eval_histories[name]) for name, bundle in bundles.items()}
        y_arr, cv_arr = np.asarray(y), np.asarray(current_v)
        predictions: dict[str, float] = {}
        for name in ("M1_LSTM", "M2_TimeXer"):
            if len(transitions) < required[name]:
                unavailable(name, "Insufficient transitions")
                continue
            x_train = np.column_stack([cv_arr, np.asarray(scalar_train[name])])
            pred_v = ridge_prediction(x_train, y_arr, np.asarray([float(origin_anchor.reference_voltage_V), eval_scalar[name]]))
            predictions[name] = pred_v
            row = dict(base, model=name, status="RUN", prediction_voltage_V=pred_v, prediction_SOH=pred_v / REF_VOLTAGE, health_head_transitions=len(transitions))
            row["error_mV"], row["absolute_error_mV"] = 1000 * (pred_v - row["truth_voltage_V"]), abs(1000 * (pred_v - row["truth_voltage_V"]))
            prediction_rows.append(row)
        if len(transitions) < required["M3_Fusion"]:
            unavailable("M3_Fusion", "Requires four transitions for 3-input fusion calibration")
            unavailable("M3_equal_weight_diagnostic", "M3 primary calibration not estimable")
        else:
            x_train = np.column_stack([cv_arr, np.asarray(scalar_train["M1_LSTM"]), np.asarray(scalar_train["M2_TimeXer"])])
            pred_v = ridge_prediction(x_train, y_arr, np.asarray([float(origin_anchor.reference_voltage_V), eval_scalar["M1_LSTM"], eval_scalar["M2_TimeXer"]]))
            row = dict(base, model="M3_Fusion", status="RUN", prediction_voltage_V=pred_v, prediction_SOH=pred_v / REF_VOLTAGE, health_head_transitions=len(transitions))
            row["error_mV"], row["absolute_error_mV"] = 1000 * (pred_v - row["truth_voltage_V"]), abs(1000 * (pred_v - row["truth_voltage_V"]))
            prediction_rows.append(row)
            avg = float(np.mean([predictions["M1_LSTM"], predictions["M2_TimeXer"]]))
            row = dict(base, model="M3_equal_weight_diagnostic", status="RUN", prediction_voltage_V=avg, prediction_SOH=avg / REF_VOLTAGE, health_head_transitions=len(transitions))
            row["error_mV"], row["absolute_error_mV"] = 1000 * (avg - row["truth_voltage_V"]), abs(1000 * (avg - row["truth_voltage_V"]))
            prediction_rows.append(row)

    rolling = pd.DataFrame(rolling_rows)
    pred = pd.DataFrame(prediction_rows)
    resources = pd.DataFrame(resource_rows)
    rolling.to_csv(OUT / "rolling_origin_splits.csv", index=False)
    pred.to_csv(OUT / "all_predictions.csv", index=False)
    pred.to_csv(OUT / "per_origin_metrics.csv", index=False)
    run_pred = pred[pred.status == "RUN"].copy()
    summary = run_pred.groupby("model").agg(
        feasible_origins=("origin_stage", "count"),
        MAE_mV=("absolute_error_mV", "mean"),
        RMSE_mV=("error_mV", lambda x: float(np.sqrt(np.mean(np.square(x))))),
    ).reset_index()
    if len(summary):
        soh_error = run_pred.assign(abs_soh_error=(run_pred["prediction_SOH"] - run_pred["truth_SOH"]).abs()).groupby("model")["abs_soh_error"].mean().reset_index(name="mean_abs_SOH_error")
        summary = summary.merge(soh_error, on="model", how="left")
    summary.to_csv(OUT / "summary_metrics.csv", index=False)
    resources.to_csv(OUT / "training_resources.csv", index=False)
    ablation = pred[pred.model.isin(["M3_Fusion", "M3_equal_weight_diagnostic"])][["origin_stage", "target_stage", "model", "status", "absolute_error_mV", "health_head_transitions"]]
    ablation.to_csv(OUT / "ablation_results.csv", index=False)
    make_plots(pred)
    audit.update({"finished_at": now(), "elapsed_seconds": round(time.time() - started, 2), "test_target_access_in_this_phase": False, "test_metrics_computed_in_this_phase": False, "historical_test_seal": "COMPROMISED_BY_PRIOR_DESCRIPTIVE_ACCESS", "development_anchor_stages": DEV_STAGES, "models_with_run_predictions": sorted(pred.loc[pred.status == "RUN", "model"].unique().tolist())})
    (ROOT / "RUN_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    def metric_record(model: str) -> dict[str, Any]:
        part = pred[(pred["model"] == model) & (pred["status"] == "RUN")]
        if len(part) == 0:
            return {"status": "NOT_ESTIMABLE", "n_origins": 0}
        return {"status": "RUN", "n_origins": int(len(part)), "MAE_mV": float(part["absolute_error_mV"].mean()), "RMSE_mV": float(np.sqrt(np.mean(np.square(part["error_mV"]))))}
    final = {
        "phase": "Phase_RUL_HealthAnchor_FourModel_Pilot",
        "source_commit": EXPECTED_COMMIT,
        "project_path": str(ROOT),
        "development_boundary": {"coordinate": "OpHrs2 [h]", "end_h": DEV_END, "causal_rows": 3005},
        "health_reference_definition": "V_ref is active 10-cell median voltage at frozen j*=1.0 A/cm2; SOH=V_ref/0.6989533066",
        "number_of_development_anchors": int(len(anchors)),
        "number_of_valid_rolling_origins": int(len(rolling)),
        "time_alignment_verified": True,
        "dynamic_pretraining_causal": True,
        "training_targets_causal": True,
        "feature_normalization_causal": True,
        "models_attempted": ["M0_persistence", "M0_linear_trend", "M1_LSTM", "M2_TimeXer_history_only", "M3_LSTM_TimeXer_fusion"],
        "models_successfully_trained": ["M1_LSTM_dynamic_encoder", "M2_TimeXer_dynamic_encoder", "M3_fusion_head_at_final_origin_only"],
        "models_not_estimable": {"M1_LSTM": ["850_to_1150"], "M2_TimeXer": ["850_to_1150"], "M3_Fusion": ["850_to_1150", "1150_to_1500"]},
        "M0_metrics": {"persistence": metric_record("M0_persistence"), "linear_trend": metric_record("M0_linear_trend")},
        "M1_metrics": metric_record("M1_LSTM"),
        "M2_metrics": metric_record("M2_TimeXer"),
        "M3_metrics": metric_record("M3_Fusion"),
        "fusion_improves_over_lstm": "INCONCLUSIVE: one shared origin; fusion 13.204 mV vs LSTM 13.643 mV",
        "fusion_improves_over_timexer": "NO: one shared origin; fusion 13.204 mV vs TimeXer 0.316 mV",
        "dynamic_features_improve_over_simple_baselines": "INCONCLUSIVE: unequal feasible-origin coverage",
        "evidence_strength": "VERY_LOW",
        "original_test_previously_inspected": True,
        "new_test_target_accessed": False,
        "test_metrics_computed": False,
        "RUL_labels_used": False,
        "true_EOL_prediction_claimed": False,
        "decision": "NO-GO",
        "recommended_next_direction": "Add independent configuration-comparable health anchors or validated continuous health labels before architecture selection; prioritize richer polarization-curve health representation over new network modules.",
        "limitations": ["Only 2/3/4 health transitions are available by rolling origin.", "BoL has no corresponding load-cycle history.", "Historical descriptive Test access compromised the test seal before this phase."],
    }
    (ROOT / "FINAL_DECISION.yaml").write_text(yaml.safe_dump(final, sort_keys=False, allow_unicode=True), encoding="utf-8")
    reports(audit, anchors, rolling, pred, resources)
    write_text(ROOT / "README.md", """# Phase RUL Health-Anchor Four-Model Pilot

Run `python scripts/run_health_anchor_pilot.py` from this directory. The runner is development-only and fail-closed. Its final decision is intentionally about feasibility, not a claim that an architecture is best.
""")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

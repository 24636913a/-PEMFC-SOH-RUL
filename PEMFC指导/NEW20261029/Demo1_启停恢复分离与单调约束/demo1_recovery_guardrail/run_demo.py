#!/usr/bin/env python3
"""Run a reproducible recovery/guardrail demonstration entirely offline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import tempfile

# A writable cache also works in read-only-home teaching/sandbox environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fuelcell_demo1_mpl_cache"))

import matplotlib
import numpy as np
import pandas as pd
import scipy

from model import forecast_from_observations, METHODS
from evaluate import evaluate
from plots import make_plots


NAMES_ZH = {"raw_voltage": "原始电压直接外推", "recovery_separated": "恢复分离", "recovery_guardrail": "恢复分离＋单调护栏"}


def select_truth_path(observation_path: Path, explicit_truth: Path | None, root: Path,
                      no_evaluate: bool = False) -> Path | None:
    """Never silently attach bundled oracle values to external observations."""
    if no_evaluate:
        return None
    if explicit_truth is not None:
        if not explicit_truth.exists():
            raise FileNotFoundError(f"Explicit truth file does not exist: {explicit_truth}")
        return explicit_truth
    if observation_path.resolve() == (root / "data/observations.csv").resolve():
        bundled_truth = root / "data/truth_for_evaluation_only.csv"
        return bundled_truth if bundled_truth.exists() else None
    return None


def write_report(metrics: dict, parameters: dict, events: pd.DataFrame, output: Path):
    provenance = ("全部为固定种子的模拟数据；不是 FC1 实测结果，也不是神经网络或论文性能声明。"
                  if metrics["is_synthetic"] else "使用外部观测；数据来源、参考条件及适用性需由使用者确认。")
    lines = ["# Demo 1 运行结果：启停恢复分离与单调护栏", "",
             "> " + provenance, "",
             "## 实验设置", "",
             f"- 0–{parameters['calibration_last_used_h']:.0f} h：只用历史观测校准恢复龄模型。",
             f"- {parameters['calibration_last_used_h']:.0f}–{metrics['forecast_cutoff_h']:.0f} h：参数冻结，逐点因果估计长期状态。",
             f"- 在 {metrics['forecast_cutoff_h']:.0f} h 固定预测起点，之后不再读取未来观测或未来重启标记。",
             f"- 阈值 = {1-metrics['voltage_loss_fraction']:.3f} × {metrics['reference_voltage_v']:.3f} V = {metrics['threshold_v']:.4f} V，表示参考电压损失 {100*metrics['voltage_loss_fraction']:g}%。该阈值仅为教学设定。",
             f"- 所有方法用相同 EWMA 系数与 {metrics['forecast_fit_window_h']:g} h 尾部窗口线性外推，原始电压方法故意省略恢复分离，作为问题基线。", "",
             "## 从历史估计出的参数", "",
             f"- 可逆损失饱和幅值：{1000*parameters['reversible_amplitude_fit_v']:.4f} mV。",
             f"- 时间常数：{parameters['reversible_tau_fit_h']:.1f} h。",
             f"- 初始校准阶段的长期电压下降率：{1e6*parameters['decay_fit_v_per_h']:.4f} μV/h。",
             "- 以上参数通过观测拟合；未使用 truth 文件，也未把生成器中的真值参数传入模型。", "",
             "## 三种方法对比", ""]
    if metrics["evaluation_available"]:
        crossing = metrics['true_latent_threshold_time_h']
        truth_line = ("评估真值在已提供的评估时间范围内尚未达到阈值，真实 RUL 未知。" if crossing is None
                      else f"评估真值：长期状态在 {crossing:.1f} h 达到阈值，因此当前真实 RUL = {metrics['true_rul_h']:.1f} h。")
        lines += [truth_line, "",
                  "| 方法 | 历史潜态 RMSE / mV | 留出未来潜态 RMSE / mV | RUL / h | RUL 绝对误差 / h | 历史向上步数 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in METHODS:
            m = metrics["methods"][name]
            rul = "无有限交点" if m["rul_h"] is None else f"{m['rul_h']:.2f}"
            error = "不适用" if m["rul_absolute_error_h"] is None else f"{m['rul_absolute_error_h']:.2f}"
            lines.append(f"| {NAMES_ZH[name]} | {m['history_latent_rmse_mv']:.4f} | {m['heldout_latent_rmse_mv']:.4f} | {rul} | {error} | {m['history_upward_steps']} |")
        lines += ["", f"历史评分覆盖 {metrics['history_evaluation_window_h'][0]:g}–{metrics['history_evaluation_window_h'][1]:g} h；未来评分仅包含预测起点之后的 {metrics['heldout_n']} 个留出样本，均按单电池电压计算。原始电压基线被当作长期状态代理，其偏差正是本例要展示的问题。", "",
                  "RUL 使用直线的解析交点，可能超出图示预测范围；是否落在图示范围内记录于 metrics.json 的 crossing_within_plotted_horizon。", "",
                  "单调投影消除了潜态估计的向上波动，但不保证每项误差都会下降。本次运行中请同时比较 RUL 误差、历史 RMSE 和未来 RMSE；不要只挑有利指标。"]
    else:
        lines += ["未提供 truth 文件：已完成真实可用的观测推断；跳过需要真值的 RMSE 和 RUL 误差。", "",
                  "| 方法 | 当前估计 / V | 外推斜率 / μV/h | RUL / h | 历史向上步数 |", "|---|---:|---:|---:|---:|"]
        for name in METHODS:
            m = metrics["methods"][name]
            rul = "无有限交点" if m["rul_h"] is None else f"{m['rul_h']:.2f}"
            lines.append(f"| {NAMES_ZH[name]} | {m['current_estimate_v']:.6f} | {1e6*m['slope_v_per_h']:.4f} | {rul} | {m['history_upward_steps']} |")
    lines += ["", "## 恢复事件检查", ""]
    chosen = events.loc[np.isclose(events.restart_time_h, 809)] if len(events) else events
    if not len(chosen) and len(events):
        chosen = events.tail(1)
    if len(chosen):
        e = chosen.iloc[0]
        lines += [f"- 选取已观测的 {e.restart_time_h:g} h 重启事件。",
                  f"- 重启前后观测电压：{e.raw_voltage_before_v:.6f} → {e.raw_voltage_after_v:.6f} V。",
                  f"- 相邻样本的观测跃升：{1000*e.observed_jump_v:.4f} mV，包含测量噪声及相邻采样间隔内的长期衰减。",
                  f"- 从可逆损失模型估计的恢复量：{1000*e.estimated_recovered_loss_v:.4f} mV。",
                  "- 观测跃升需要与可逆状态和工况变化一起解释，不能直接判定长期老化逆转。"]
    else:
        lines += ["校准后至预测起点之间没有可分析的已观测重启事件。"]
    lines += ["", "随包数据中的 809 h 只是按需求设置的模拟事件，并未声称原始 FC1 数据在该时刻发生相同现象。", "",
              "## 文件导航", "",
              "- `01_recovery_decomposition.png`：原始电压、剩余可逆损失、三种状态估计。",
              "- `02_fixed_origin_forecast.png`：固定预测起点后的外推与阈值交点。",
              "- `03_restart_at_809h.png`：重启前后的局部放大。",
              "- `history_estimates.csv`：逐点因果估计；校准结束前的状态估计为空。",
              "- `future_forecasts.csv`：固定起点的未来预测，未回填未来观测。",
              "- `event_recovery_estimates.csv`：恢复增量估计与单独评估用真值。",
              "- `metrics.json`、`model_parameters.json`：未四舍五入的数值与拟合参数。",
              "- `checks.json`：因果性、未来隔离、truth 隔离、单调性、RUL 和恢复代数检查。", "",
              "## 解释边界", "",
              "这是可解释的小模型验证流程，不是部署用寿命预测器。生成机制和恢复模型在此例中相互匹配；真实的部分恢复、工况变化、未知重启时刻、非线性长期退化会破坏这些简化。真实数据应采用独立诊断或恢复后基准确认长期状态，并对模型失配、估计不确定性和实际业务阈值重新验证。", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--observations", type=Path, default=root / "data/observations.csv")
    parser.add_argument("--truth", type=Path, default=None,
                        help="Explicit evaluation oracle. Bundled truth is automatic ONLY for bundled observations.")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--regenerate-data", action="store_true", help="Regenerate bundled synthetic data first.")
    parser.add_argument("--no-evaluate", action="store_true", help="Do not read truth, even if present.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip optional regression/leakage checks.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.regenerate_data:
        from generate_data import generate
        if args.observations.resolve() != (root / "data/observations.csv").resolve():
            parser.error("--regenerate-data is only allowed for the bundled observations path.")
        generate(config, root / "data")
    observations = pd.read_csv(args.observations)
    # INFERENCE FINISHES HERE. Only modeling/reference config enters the model.
    result = forecast_from_observations(observations, config["model"], config["reference"])
    # Oracle file is opened only after predictions have already been computed.
    truth = None
    truth_path = select_truth_path(args.observations, args.truth, root, args.no_evaluate)
    if truth_path is not None:
        truth = pd.read_csv(truth_path)
    metrics, events = evaluate(result, truth)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result["history"].to_csv(output / "history_estimates.csv", index=False, float_format="%.12g")
    result["forecast"].to_csv(output / "future_forecasts.csv", index=False, float_format="%.12g")
    events.to_csv(output / "event_recovery_estimates.csv", index=False, float_format="%.12g")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "model_parameters.json").write_text(json.dumps(result["parameters"], indent=2, ensure_ascii=False), encoding="utf-8")
    make_plots(result, observations, truth, output)
    write_report(metrics, result["parameters"], events, output)
    if not args.skip_checks:
        from self_check import run_checks
        checks = run_checks(observations, config, truth)
        (output / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
               "platform": platform.system(), "synthetic_seed": config["synthetic"]["seed"]}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    log = ["Demo 1 completed.", metrics["data_status"],
           f"Forecast cutoff: {metrics['forecast_cutoff_h']:.0f} h; threshold: {metrics['threshold_v']:.4f} V",
           f"Truth used for evaluation only: {truth is not None}"]
    for name in METHODS:
        m = metrics["methods"][name]
        log.append(f"{name}: RUL={m['rul_h']} h, upward_steps={m['history_upward_steps']}")
    log.append(f"Results: {output.name}/")
    (output / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()

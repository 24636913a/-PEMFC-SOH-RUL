#!/usr/bin/env python3
"""Run the complete sparse-diagnostic demonstration, offline and reproducibly."""
from pathlib import Path
import argparse
import json
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"fuelcell_demo2_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from estimator import estimate

ROOT=Path(__file__).resolve().parent
METHODS={"raw_ratio":"Raw Vop / V0(ref)",
         "initial_curve_standardized_soh":"Initial-curve standardization",
         "kalman_without_anchors_soh":"Same filter, no later anchors",
         "anchor_fusion_soh":"Causal sparse-anchor fusion"}
COLORS={"raw_ratio":"#9c9c9c","initial_curve_standardized_soh":"#d18b69",
        "kalman_without_anchors_soh":"#b78e37","anchor_fusion_soh":"#398c8b"}


def predict_from_files(input_dir,estimator_config):
    """Truth-free production entry point. File inputs are explicit and limited."""
    input_dir=Path(input_dir)
    operation=pd.read_csv(input_dir/"operation.csv")
    diagnostics=pd.read_csv(input_dir/"diagnostics.csv")
    fused,updates,metadata=estimate(operation,diagnostics,estimator_config,True)
    unanchored,_,_=estimate(operation,diagnostics,estimator_config,False)
    predictions=fused[["time_h","raw_ratio","initial_curve_standardized_soh"]].copy()
    predictions["kalman_without_anchors_soh"]=unanchored.estimated_soh
    predictions["anchor_fusion_soh"]=fused.estimated_soh
    return operation,diagnostics,predictions,fused,updates,metadata


def error_metrics(values,truth):
    e=(np.asarray(values)-np.asarray(truth))*100 # Percentage points, not relative percent.
    return {"rmse_percentage_points":float(np.sqrt(np.mean(e*e))),
            "mae_percentage_points":float(np.mean(np.abs(e))),
            "mean_error_percentage_points":float(np.mean(e)),"samples":int(len(e))}


def evaluate(predictions,diagnostics,updates,truth):
    joined=predictions.merge(truth,on="time_h",validate="one_to_one")
    if len(joined)!=len(predictions): raise ValueError("Truth and operation timestamps do not match")
    times=sorted(diagnostics.time_h.unique())
    is_anchor=joined.time_h.isin(times)
    metric={"metric_unit":"SOH percentage points (pp)","scope":"Synthetic data, not real-dataset accuracy",
            "overall":{},"non_anchor_samples":{},"intervals_are_causal":True}
    for method in METHODS:
        metric["overall"][method]=error_metrics(joined[method],joined.true_soh)
        metric["non_anchor_samples"][method]=error_metrics(joined.loc[~is_anchor,method],joined.loc[~is_anchor,"true_soh"])
    interval_rows=[]
    for a,b in zip(times[:-1],times[1:]):
        # Exclude BOTH endpoints: no anchor-time error can flatter the interval score.
        select=(joined.time_h>a)&(joined.time_h<b)
        for method in METHODS:
            e=error_metrics(joined.loc[select,method],joined.loc[select,"true_soh"])
            interval_rows.append({"start_anchor_h":float(a),"next_anchor_h":float(b),"method":method,**e})
    updates=updates.merge(truth[["time_h","true_dc_v","true_dr_ohm_cm2"]],on="time_h",how="left",validate="one_to_one")
    updates["dc_error_before_mv"]=(updates.dc_before_update_v-updates.true_dc_v)*1000
    updates["dc_error_after_mv"]=(updates.dc_after_update_v-updates.true_dc_v)*1000
    metric["anchor_update_loss_rmse_mv"]={"before":float(np.sqrt(np.mean(updates.dc_error_before_mv**2))),
              "after":float(np.sqrt(np.mean(updates.dc_error_after_mv**2)))}
    return joined,metric,pd.DataFrame(interval_rows),updates


def save_figure(fig,out,name):
    fig.savefig(out/f"{name}.png",dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{name}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def plots(operation,diagnostics,predictions,states,updates,truth,intervals,out,is_synthetic=True):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.labelcolor":"#222222","text.color":"#222222",
                         "axes.grid":True,"grid.color":"#ededed","grid.linewidth":.7})
    times=sorted(diagnostics.time_h.unique())
    session_counts=diagnostics.groupby("session_id").size()
    n_sessions=len(session_counts)
    point_label=str(int(session_counts.iloc[0])) if session_counts.nunique()==1 else "variable"
    fig,axes=plt.subplots(4,1,figsize=(11.2,12.6),sharex=True,gridspec_kw={"height_ratios":[1,1,1,1.65]})
    axes[0].plot(operation.time_h,operation.current_density_a_cm2,color="#6c7a7b",lw=.9)
    axes[0].set_ylabel("Current density\n(A cm$^{-2}$)")
    axes[0].set_title(("Synthetic " if is_synthetic else "")+"dynamic-load operation and causal health estimation",loc="left",weight="bold")
    axes[1].plot(operation.time_h,operation.voltage_cell_v,color="#9b9b9b",lw=.8,label="Observed operating voltage")
    if truth is not None: axes[1].plot(truth.time_h,truth.true_reference_voltage_v,color="#222222",lw=1.6,label="True long-term voltage at reference current")
    axes[1].set_ylabel("Single-cell voltage (V)");axes[1].legend(fontsize=9,loc="lower left")
    axes[2].plot(predictions.time_h,predictions.raw_ratio*100,lw=.7,color=COLORS["raw_ratio"],label=METHODS["raw_ratio"])
    if truth is not None: axes[2].plot(truth.time_h,truth.true_soh*100,color="#222222",lw=1.5,ls="--",label="Latent truth")
    axes[2].set_ylabel("Raw ratio (%)");axes[2].legend(ncol=2,loc="lower left",fontsize=8)
    for method in list(METHODS)[1:]:
        axes[3].plot(predictions.time_h,predictions[method]*100,lw=1.3,color=COLORS[method],alpha=.9,label=METHODS[method])
    if truth is not None: axes[3].plot(truth.time_h,truth.true_soh*100,color="#222222",lw=1.8,ls="--",label="Latent truth (evaluation only)")
    for ax in axes:
        for t in times: ax.axvline(t,color="#b75d38",ls=":",lw=.7,alpha=.55)
    axes[3].set_ylabel("Standardized SOH (%)");axes[3].set_xlabel("Operating time (h)")
    axes[3].legend(ncol=2,loc="lower left",fontsize=8)
    fig.text(.12,.008,f"Vertical dotted lines = {n_sessions} diagnostic sessions; {point_label} current points per scan.",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.025,1,1));save_figure(fig,out,"01_dynamic_load_and_health")

    fig,axes=plt.subplots(1,2,figsize=(11.2,4.4))
    cmap=plt.get_cmap("viridis")
    for k,(sid,scan) in enumerate(diagnostics.groupby("session_id",sort=True)):
        axes[0].plot(scan.current_density_a_cm2,scan.voltage_cell_v,".-",ms=3,lw=1,color=cmap(k/max(n_sessions-1,1)),label=f"{int(scan.time_h.iloc[0])} h")
    axes[0].set_xlabel("Current density (A cm$^{-2}$)");axes[0].set_ylabel("Controlled diagnostic voltage (V)")
    axes[0].set_title(f"{n_sessions} time-separated polarization scans",loc="left",weight="bold")
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].plot(states.time_h,states.estimated_dc_v*1000,color="#398c8b",label="Estimated reference loss")
    axes[1].scatter(updates.time_h,updates.diagnostic_dc_v*1000,color="#ce815e",s=28,label="Arrived diagnostic anchor",zorder=4)
    if truth is not None: axes[1].plot(truth.time_h,truth.true_dc_v*1000,color="#222222",ls="--",label="True reference loss")
    axes[1].set_xlabel("Operating time (h)");axes[1].set_ylabel("Reference-current loss d_c (mV)")
    axes[1].set_title("Anchor correction and propagation",loc="left",weight="bold");axes[1].legend(fontsize=8)
    fig.tight_layout();save_figure(fig,out,"02_diagnostic_scans_and_loss")

    if truth is not None:
        fig,axes=plt.subplots(1,2,figsize=(11.2,4.4))
        x=np.arange(len(updates));w=.36
        axes[0].bar(x-w/2,updates.dc_error_before_mv,width=w,color="#cc8663",label="Before anchor update")
        axes[0].bar(x+w/2,updates.dc_error_after_mv,width=w,color="#398c8b",label="After anchor update")
        axes[0].set_xticks(x,[str(int(t)) for t in updates.time_h]);axes[0].axhline(0,color="#444444",lw=.8)
        axes[0].set_xlabel("Diagnostic availability time (h)");axes[0].set_ylabel("Reference-loss error (mV)")
        axes[0].set_title(f"Corrections at {len(updates)} later diagnostics",loc="left",weight="bold");axes[0].legend(fontsize=8)
        for method in ["initial_curve_standardized_soh","kalman_without_anchors_soh","anchor_fusion_soh"]:
            d=intervals[intervals.method==method]
            axes[1].plot(np.arange(len(d)),d.rmse_percentage_points,"o-",lw=1.3,ms=4,color=COLORS[method],label=METHODS[method])
        first=intervals[intervals.method=="anchor_fusion_soh"]
        axes[1].set_xticks(np.arange(len(first)),[f"{int(a)}–\n{int(b)}" for a,b in zip(first.start_anchor_h,first.next_anchor_h)],fontsize=8)
        axes[1].set_xlabel("Open interval between diagnostics (h)");axes[1].set_ylabel("SOH RMSE (percentage points)")
        axes[1].set_title("Errors between anchors (endpoints excluded)",loc="left",weight="bold");axes[1].legend(fontsize=8)
        fig.tight_layout();save_figure(fig,out,"03_anchor_updates_and_interval_error")


def write_report(out,metrics,metadata,checks,truth_available,diagnostics):
    session_counts=diagnostics.groupby("session_id").size()
    n_sessions=len(session_counts)
    point_description=str(int(session_counts.iloc[0])) if session_counts.nunique()==1 else f"{int(session_counts.min())}–{int(session_counts.max())}"
    last_time=float(diagnostics.time_h.max())
    lines=["# Demo 2：稀疏诊断锚点校准——本次运行结果","",
       ("**全部数据为固定随机种子的模拟数据，不是 FC1 或 RealHyFC 实测结果。**" if metadata.get("data_kind")=="SYNTHETIC_DATA" else "**本次使用外部输入数据。来源、测量协议和标签可靠性需由数据提供者确认。**"),"",
       f"- 运行得到的初始参考电压：{metadata['fitted_initial_reference_voltage_v']:.6f} V / 单电池。",
       f"- 初始诊断仅用于建立基准；在线使用后续锚点：{metadata['post_initial_anchor_count_used']} 次。",
       f"- {n_sessions} 次诊断表示 {n_sessions} 个不同时间的极化扫描，每次扫描 {point_description} 个电流密度点。",
       "- 推断不读取 truth 文件；未来诊断不会反向修正过去的在线结果。",""]
    if truth_available:
        lines += ["## SOH 误差（单位：百分点 pp）","","| 方法 | 全部样本 RMSE | 非诊断样本 RMSE | 非诊断样本 MAE |","|---|---:|---:|---:|"]
        chinese={"raw_ratio":"原始电压直接相除","initial_curve_standardized_soh":"初始曲线+温度标准化",
                 "kalman_without_anchors_soh":"相同滤波器、无后续锚点","anchor_fusion_soh":"因果稀疏锚点融合"}
        for m in METHODS:
            a=metrics["overall"][m];b=metrics["non_anchor_samples"][m]
            lines.append(f"| {chinese[m]} | {a['rmse_percentage_points']:.4f} | {b['rmse_percentage_points']:.4f} | {b['mae_percentage_points']:.4f} |")
        before=metrics["anchor_update_loss_rmse_mv"]["before"];after=metrics["anchor_update_loss_rmse_mv"]["after"]
        lines += ["",f"{metadata['post_initial_anchor_count_used']} 次后续锚点的参考损失估计 RMSE：更新前 {before:.3f} mV，更新后 {after:.3f} mV。",
          "这只描述模拟数据中的即时校正效果；`interval_metrics.csv` 单独报告锚点之间、且排除两端诊断时刻的误差。",
          f"最后一次 {last_time:g} h 诊断只影响其到达及之后的样本，不给此前区间带来改善。",""]
    else:
        lines += ["未发现 truth 文件：已完成在线估计和作图；无真实标签的准确率指标被跳过。",""]
    lines += ["## 可重复性与检查","",f"自动检查：{checks['passed']} / {checks['total']} 通过。详见 `checks.json`。", "",
       "## 如何解读","",
       "1. 原始比值受负载影响，不能直接当作固定参考工况下的健康度。",
       "2. 仅用初始极化曲线和已校准温度系数，可以去掉大部分可解释工况效应；它不能自动知道老化斜率变化与未建模偏差。",
       "3. 无后续锚点的滤波器与融合滤波器完全同型、同参；区别仅是后续诊断是否到达并被吸收。这是主要消融对照。",
       "4. 锚点提供参考损失与斜率信息，帮助区分长期退化和慢变偏差；两个锚点之间仍依赖状态演化假设。",
       "5. 本例不证明某种网络无法训练，也不证明八次体检对所有电堆足够。复杂模型能否识别，需要数据量、先验与独立验证共同判断。",
       "6. 当前 demo 给出在线 SOH，不给出经过验证的 RUL；协方差仅为模型内部不确定度，不是经覆盖率验证的置信区间。", "",
       "## 输出文件","","- `predictions.csv`：四种方法在线 SOH。",
       "- `state_estimates.csv`：长期损失、斜率、慢变偏差和模型内部标准差。",
       "- `anchor_updates.csv`：每次后续诊断的更新前后值。",
       "- `interval_metrics.csv`：锚点之间的误差，排除两个端点。",
       "- 三组 PNG / SVG：动态工况与健康度、诊断扫描与损失、更新及区间误差。",
       "- `metrics.json`、`model_metadata.json`、`checks.json`、`environment.json`：机器可读运行记录。", ""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results")
    parser.add_argument("--config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--regenerate-data",action="store_true",help="Regenerate the bundled synthetic CSVs before estimation")
    parser.add_argument("--skip-demo-fixture-check",action="store_true",help="Use general shape checks instead of synthetic metadata counts (for custom real data)")
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding="utf-8"))
    if args.regenerate_data:
        from simulator import generate_data
        generate_data(config,args.input_dir)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    operation,diagnostics,predictions,states,updates,metadata=predict_from_files(args.input_dir,config["estimator"])
    data_metadata_path=args.input_dir/"metadata.json"
    data_metadata=json.loads(data_metadata_path.read_text(encoding="utf-8")) if data_metadata_path.exists() else {}
    metadata["data_kind"]=data_metadata.get("kind","EXTERNAL_INPUT_UNVERIFIED_PROVENANCE")
    predictions.to_csv(out/"predictions.csv",index=False,float_format="%.10g")
    states.to_csv(out/"state_estimates.csv",index=False,float_format="%.10g")
    # Evaluation begins only after predictions have already been made and saved.
    truth_path=args.input_dir/"truth_for_evaluation_only.csv"
    truth=pd.read_csv(truth_path) if truth_path.exists() else None
    if truth is not None:
        joined,metrics,intervals,updates=evaluate(predictions,diagnostics,updates,truth)
        metrics["scope"]=("Synthetic data, not real-dataset accuracy" if metadata["data_kind"]=="SYNTHETIC_DATA" else "External input compared to provided labels; provenance and label validity not independently verified")
        joined.to_csv(out/"evaluation_with_truth.csv",index=False,float_format="%.10g")
        intervals.to_csv(out/"interval_metrics.csv",index=False,float_format="%.10g")
    else:
        metrics={"truth_available":False,"accuracy_metrics_skipped":True};intervals=None
        # Remove only known truth-dependent products from an earlier run in the
        # same output directory, so stale accuracy files cannot be mistaken for
        # results from the present unlabeled dataset.
        for stale_name in ["evaluation_with_truth.csv","interval_metrics.csv",
                           "03_anchor_updates_and_interval_error.png","03_anchor_updates_and_interval_error.svg"]:
            (out/stale_name).unlink(missing_ok=True)
    updates.to_csv(out/"anchor_updates.csv",index=False,float_format="%.10g")
    plots(operation,diagnostics,predictions,states,updates,truth,intervals,out,is_synthetic=metadata["data_kind"]=="SYNTHETIC_DATA")
    from verify import run_checks
    checks=run_checks(config,args.input_dir,check_fixture=not args.skip_demo_fixture_check)
    import sys
    environment={"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__,
                 "command":"python run_demo.py","random_seed":data_metadata.get("random_seed")}
    for name,value in [("metrics",metrics),("model_metadata",metadata),("checks",checks),("environment",environment)]:
        (out/f"{name}.json").write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    write_report(out,metrics,metadata,checks,truth is not None,diagnostics)
    print(json.dumps({"output_directory":str(out),"checks":checks,"metrics":metrics},ensure_ascii=False,indent=2))
    if checks["passed"]!=checks["total"]: raise SystemExit("Verification failed")

if __name__=="__main__":main()

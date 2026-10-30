# %% 1. 设置：只读已有处理结果，不重新拟合或分离
"""在VS Code运行本文件即可补画电压曲线；旧输出目录不变。"""
from pathlib import Path
from datetime import datetime
import importlib.util
import json
import sys
import uuid

import matplotlib.pyplot as plt
import pandas as pd

CODE_DIRECTORY = Path(
    r"E:\沈农信电\ PEMFC-SOH+RUL\PEMFC指导\NEW20261029\数据处理结果\启停恢复分离与单调约束\代码文件"
)
if "__file__" in globals():
    CODE_DIRECTORY = Path(__file__).resolve().parent

# 指定要补图的真实运行目录，不挑选拟合效果更好的运行。
EXISTING_RUN_NAME = "20261030_153112_259671_cdbe1c"
SHOW_FIGURE = True


# %% 2. 独立补图入口
def plot_existing_run():
    module_path = CODE_DIRECTORY / "RealHyFC_启停恢复处理.py"
    spec = importlib.util.spec_from_file_location("realhyfc_recovery_plot_helpers", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载绘图函数：{module_path}")
    module = importlib.util.module_from_spec(spec)
    # 主程序由__main__保护；此处只加载定义，不调用run/fit_recovery/process_states。
    spec.loader.exec_module(module)
    source = CODE_DIRECTORY.parent / "输出结果" / EXISTING_RUN_NAME
    old_summary = json.loads((source / "运行记录/完成摘要.json").read_text(encoding="utf-8-sig"))
    if old_summary.get("status") != "COMPLETED":
        raise ValueError("只能补画完整保存的结果，不能把失败或部分运行当完整结果。")
    cfg = json.loads((source / "运行记录/本次配置.json").read_text(encoding="utf-8-sig"))
    # 科学参数沿用那次运行，仅补充显示参数；不利用旧成绩修改模型。
    plot_cfg = json.loads((CODE_DIRECTORY / "处理配置.json").read_text(encoding="utf-8-sig"))
    cfg["raw_voltage_plot_bucket_rows"] = plot_cfg.get("raw_voltage_plot_bucket_rows", 600)
    cfg["show_figures"] = SHOW_FIGURE
    csv_path = source / "表格/恢复分离与单调约束逐平台结果.csv"
    fit_path = source / "运行记录/恢复模型参数与资格.json"
    config_path = source / "运行记录/本次配置.json"
    checksum_table = pd.read_csv(source / "运行记录/输出SHA256.csv")
    for path in (csv_path, fit_path, config_path):
        relative = path.relative_to(source).as_posix()
        matched = checksum_table.loc[checksum_table.path.str.replace("\\", "/", regex=False).eq(relative)]
        if len(matched) != 1 or module.digest(path) != str(matched.iloc[0].sha256).lower():
            raise ValueError(f"已有结果SHA不一致，停止补图：{relative}")
    obs = pd.read_csv(csv_path)
    fit = json.loads(fit_path.read_text(encoding="utf-8-sig"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:6]
    out = CODE_DIRECTORY.parent / "输出结果" / ("电压曲线示意图_" + stamp)
    for folder in ("图片", "绘图数据", "表格", "运行记录"):
        (out / folder).mkdir(parents=True, exist_ok=False)
    try:
        module.log("读取原始缓存和旧处理结果；不执行任何参数拟合。")
        root = module.find_repository(CODE_DIRECTORY)
        meta, _, arrays, _ = module.load_contract(root, cfg, out)
        raw = module.raw_voltage_plot_points(arrays, meta["columns"], cfg)
        plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                             "axes.unicode_minus": False, "font.size": 10})
        figure = module.voltage_curve_schematic(raw, obs, fit, cfg, int(meta["rows"]))
        figure.savefig(out / "图片/06_电压曲线示意图.png", dpi=cfg["plot_dpi"], bbox_inches="tight")
        figure.savefig(out / "图片/06_电压曲线示意图.pdf", bbox_inches="tight")
        module.save_csv(out / "绘图数据/06_电压曲线示意图_原始电压.csv", raw)
        module.save_csv(out / "绘图数据/06_电压曲线示意图_固定负载曲线.csv", obs)
        module.save_json(out / "运行记录/补图来源.json", dict(
            source_run=str(source), source_predictions_sha256=module.digest(csv_path),
            source_parameters_sha256=module.digest(fit_path),
            plot_helper_sha256=module.digest(module_path),
            plot_entry_sha256=module.digest(CODE_DIRECTORY / "仅绘制已有结果电压曲线.py"),
            Python=sys.version, original_rows=int(meta["rows"]), plotted_raw_rows=len(raw),
            plotted_platforms=len(obs), decomposition_status=fit["status"],
            raw_bucket_rows=cfg["raw_voltage_plot_bucket_rows"],
            no_refitting=True, no_new_decomposition=True))
        module.save_csv(out / "运行记录/输出SHA256.csv", pd.DataFrame([
            dict(path=p.relative_to(out).as_posix(), sha256=module.digest(p))
            for p in sorted(out.rglob("*")) if p.is_file()]))
        module.log(f"补图完成，分离状态仍为{fit['status']}；输出：{out}")
        if SHOW_FIGURE:
            plt.show()
        plt.close(figure)
        return out
    except Exception:
        import traceback
        (out / "运行记录/补图失败.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    RESULT_DIRECTORY = plot_existing_run()

# -*- coding: utf-8 -*-
"""
生成消融对比柱状图

读取 消融实验outputs/消融结果汇总.txt, 自动生成各模型 R² 对比柱状图。
输出: 消融实验outputs/消融对比柱状图.png

用法: 消融脚本全部跑完后, 运行本脚本一次即可。
"""
import sys, re, warnings
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\New-PEMFC-Pytorch\消融实验outputs")
SUMMARY_FILE = OUTPUT_DIR / "消融结果汇总.txt"


def parse_summary():
    """解析汇总文件 → {模型名: {'FC1': (min,mean,max), 'FC2': ...}}"""
    results = {}
    if not SUMMARY_FILE.exists():
        print(f"[警告] 未找到汇总文件: {SUMMARY_FILE}")
        print("        请先运行消融脚本, 结果会自动记录到该文件。")
        return results
    for line in SUMMARY_FILE.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        # 格式: [FC1] 模型名: Min=... Mean=... Max=...  [倒数第N次]
        m = re.match(r'\[(FC\d)\]\s+(.+?):\s*(.+)', line)
        if not m:
            continue
        fc = m.group(1)
        model = m.group(2).strip()
        vals = {}
        for kv in m.group(3).split():
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
        if all(k in vals for k in ('Min', 'Mean', 'Max')):
            results.setdefault(model, {})[fc] = (vals['Min'], vals['Mean'], vals['Max'])
    return results


def model_color(model):
    """根据模型类型分配颜色: 本文主模型=红, 前人=橙, 组件消融=蓝, 通用=灰"""
    if 'PINN-TFT-MT' in model or '主模型' in model:
        return '#c0392b'   # 红 (本文主模型)
    if 'ChouWang' in model or 'RF-TCN' in model:
        return '#e67e22'   # 橙 (前人)
    if 'minus' in model or '纯MSE' in model or 'standard-GRN' in model:
        return '#2980b9'   # 蓝 (组件消融)
    return '#7f8c8d'       # 灰 (通用基线)


def plot(results):
    if not results:
        print("无数据可画图。")
        return

    # 找出出现的工况, 只画有数据的工况
    fcs = ['FC1', 'FC2']
    present_fcs = [fc for fc in fcs if any(fc in v for v in results.values())]

    fig, axes = plt.subplots(1, len(present_fcs), figsize=(7 * len(present_fcs), 6),
                             squeeze=False)
    for ax, fc in zip(axes[0], present_fcs):
        # 该工况下各模型, 按 Mean 降序
        items = [(m, v[fc][1]) for m, v in results.items() if fc in v]
        items.sort(key=lambda x: x[1], reverse=True)
        if not items:
            continue
        models = [it[0] for it in items]
        means = [it[1] for it in items]
        colors = [model_color(m) for m in models]

        bars = ax.bar(range(len(models)), means, color=colors, edgecolor='black', linewidth=0.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean R²')
        ax.set_title(f'{fc} 消融对比 (Mean R²)')
        ax.grid(axis='y', alpha=0.3)
        # 柱顶标注数值
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

    fig.suptitle('PEMFC 退化状态估计 — 消融实验对比 (Loop 50)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = OUTPUT_DIR / '消融对比柱状图.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片已保存到 {out_png}")


if __name__ == '__main__':
    print("读取汇总文件:", SUMMARY_FILE)
    data = parse_summary()
    print(f"解析到 {len(data)} 个模型:", list(data.keys()))
    plot(data)

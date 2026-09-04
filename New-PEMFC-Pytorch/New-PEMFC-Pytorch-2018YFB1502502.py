# ================================================================
# PEMFC 16590h 长时寿命数据集 — 退化趋势分析与基础建模
# ================================================================
# 数据来源: 2018YFB1502502-001, 中科院大连化物所
# 数据概况: 188个采样点, 16590h跨度, 电压0.73→0.47V (35%衰减)
# 特征: Time_h (时间), CurrentDensity_mAcm2 (电流密度)
# 目标: Voltage_V (电压)
# ================================================================

import sys, os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ================================================================
# 0. 配置
# ================================================================
import random
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\2018YFB1502502-001-质子交换膜燃料电池寿命测试数据集\16590小时寿命测试数据.xlsx")
OUTPUT_DIR = Path("./outputs")
date_str = datetime.now().strftime('%Y%m%d')
n = 1
while (OUTPUT_DIR / f"{date_str}_{n}_16590h分析").exists():
    n += 1
RUN_DIR = OUTPUT_DIR / f"{date_str}_{n}_16590h分析"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# 1. 数据加载
# ================================================================
print("\n" + "="*60)
print("16590h 数据集加载")
print("="*60)

df = pd.read_excel(DATA_PATH)
df.columns = ['Time_h', 'CurrentDensity_mAcm2', 'Voltage_V']
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna().sort_values('Time_h').reset_index(drop=True)

print(f"有效样本: {len(df)}")
print(f"时间跨度: {df.Time_h.min():.1f} ~ {df.Time_h.max():.1f} h ({df.Time_h.max()/24:.0f} 天)")
print(f"电流密度: {df.CurrentDensity_mAcm2.min():.0f} ~ {df.CurrentDensity_mAcm2.max():.0f} mA/cm²")
print(f"电压范围: {df.Voltage_V.min():.4f} ~ {df.Voltage_V.max():.4f} V")
print(f"电压衰减: {(df.Voltage_V.max() - df.Voltage_V.min()):.4f} V ({(df.Voltage_V.max()-df.Voltage_V.min())/df.Voltage_V.max()*100:.1f}%)")

# 电流密度分组统计
print(f"\n电流密度分组:")
for cd, grp in df.groupby(pd.cut(df.CurrentDensity_mAcm2, bins=[0, 260, 500, 750, 1100])):
    if len(grp) > 0:
        print(f"  {cd} mA/cm²: {len(grp):3d} 点, 电压 {grp.Voltage_V.min():.4f}~{grp.Voltage_V.max():.4f}")

# ================================================================
# 2. 退化趋势可视化
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 原始电压-时间 (电流颜色标注)
ax = axes[0, 0]
sc = ax.scatter(df.Time_h, df.Voltage_V, c=df.CurrentDensity_mAcm2,
                cmap='viridis', s=15, alpha=0.8)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Voltage Degradation over 16590h (color=current density)')
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label='Current Density (mA/cm²)')

# 图2: 固定电流段 (800±5 mA/cm²) 的纯退化曲线
ax = axes[0, 1]
mask_800 = (df.CurrentDensity_mAcm2 > 790) & (df.CurrentDensity_mAcm2 < 810)
df_800 = df[mask_800].copy()
ax.scatter(df_800.Time_h, df_800.Voltage_V, s=20, c='steelblue', alpha=0.7)
# 趋势线
if len(df_800) > 5:
    z = np.polyfit(df_800.Time_h, df_800.Voltage_V, 1)
    t_line = np.linspace(df_800.Time_h.min(), df_800.Time_h.max(), 100)
    ax.plot(t_line, np.polyval(z, t_line), 'r--', linewidth=2, label=f'Linear trend ({z[0]*1000:.3f} mV/h)')
ax.set_xlabel('Time (h)')
ax.set_ylabel('Voltage (V)')
ax.set_title(f'Fixed Current (~800 mA/cm²): {len(df_800)} points')
ax.legend()
ax.grid(True, alpha=0.3)

# 图3: 每小时退化速率 (μV/h)
ax = axes[1, 0]
df_sorted = df.sort_values('Time_h')
time_gap = df_sorted.Time_h.diff().values[1:]
dv = df_sorted.Voltage_V.diff().values[1:]
rate_uv_per_h = (dv / time_gap) * 1e6  # μV/h
ax.plot(df_sorted.Time_h.values[1:], rate_uv_per_h, 'o-', markersize=3, linewidth=0.5, alpha=0.6)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Degradation Rate (μV/h)')
ax.set_title('Instantaneous Degradation Rate')
ax.grid(True, alpha=0.3)

# 图4: 电压分布 (按电流分组)
ax = axes[1, 1]
current_groups = {'~1000': (980, 1020), '~800': (790, 810),
                  '~500-600': (490, 610), '~200-250': (190, 260)}
colors = ['steelblue', 'darkorange', 'green', 'red']
for (label, (lo, hi)), c in zip(current_groups.items(), colors):
    mask = (df.CurrentDensity_mAcm2 > lo) & (df.CurrentDensity_mAcm2 < hi)
    if mask.sum() > 0:
        grp = df[mask]
        ax.scatter(grp.Time_h, grp.Voltage_V, s=15, label=f'{label} mA/cm² ({mask.sum()}pts)', c=c, alpha=0.7)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Voltage by Current Density Group')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RUN_DIR / 'degradation_analysis.png', dpi=150)
plt.show()
print(f"\n退化分析图已保存至 {RUN_DIR / 'degradation_analysis.png'}")

# ================================================================
# 3. 各电流段退化速率分析
# ================================================================
print("\n" + "="*60)
print("各电流段退化速率分析 (线性拟合)")
print("="*60)

current_bins = [(790, 810, '~800 mA/cm²'), (590, 610, '~600 mA/cm²'),
                (490, 510, '~500 mA/cm²'), (390, 410, '~400 mA/cm²'),
                (240, 260, '~250 mA/cm²'), (190, 210, '~200 mA/cm²')]

degradation_rates = []

for lo, hi, label in current_bins:
    mask = (df.CurrentDensity_mAcm2 > lo) & (df.CurrentDensity_mAcm2 < hi)
    if mask.sum() >= 5:
        grp = df[mask].sort_values('Time_h')
        z = np.polyfit(grp.Time_h, grp.Voltage_V, 1)
        rate_uvh = abs(z[0]) * 1e6
        r2_lin = r2_score(grp.Voltage_V, np.polyval(z, grp.Time_h))
        print(f"  {label}: {len(grp):3d}点, 退化={rate_uvh:.2f} μV/h, "
              f"年化={abs(z[0])*8760*1000:.2f} mV/年, R²_lin={r2_lin:.3f}")
        degradation_rates.append({
            'label': label, 'rate_uvh': rate_uvh, 'rate_v_yr': abs(z[0])*8760,
            'points': len(grp), 'r2': r2_lin,
            'time': grp.Time_h.values, 'voltage': grp.Voltage_V.values,
            'fit': np.polyval(z, grp.Time_h)
        })

# ================================================================
# 4. 与 IEEE 2014 交叉验证
# ================================================================
print("\n" + "="*60)
print("与 IEEE 2014 FC1 数据集对比")
print("="*60)

# IEEE 2014 FC1 的典型退化速率 (3.35V → 3.22V over 1000h ≈ 130 μV/h)
# 归一化: 130 μV/h / 3.35V = 38.8 ppm/h
print("IEEE 2014 FC1 (1000h):")
print("  电压范围: 3.35 → 3.22 V")
print("  退化速率: ~130 μV/h (~39 ppm/h)")
print("  特征: 高密度采样, 24传感器, 固定工况")

print(f"\n16590h (本数据集):")
print(f"  电压范围: 0.73 → 0.47 V")
if degradation_rates:
    best_rate = degradation_rates[0]  # 800 mA/cm²
    print(f"  800mA段退化: {best_rate['rate_uvh']:.2f} μV/h "
          f"({best_rate['rate_uvh']/df.Voltage_V.max()*1e6/1000:.1f} ppm/h)")
print("  特征: 稀疏采样, 2传感器, 多电流工况")

print("\n⚠ 注意: 两个数据集电堆规格和测试条件不同，退化速率不可直接对比绝对值。")
print("  16590h数据的核心价值: 验证退化趋势的长期线性性 (16590h ≫ 1000h)。")

# ================================================================
# 5. 可视化: 各电流段退化
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 各电流段退化曲线 + 线性拟合
ax = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(degradation_rates)))
for i, dr in enumerate(degradation_rates):
    ax.scatter(dr['time'], dr['voltage'], s=15, alpha=0.6,
              color=colors[i], label=f"{dr['label']} ({dr['rate_uvh']:.1f}μV/h)")
    ax.plot(dr['time'], dr['fit'], '--', color=colors[i], alpha=0.5, linewidth=1)
ax.set_xlabel('Time (h)'); ax.set_ylabel('Voltage (V)')
ax.set_title('Degradation by Current Density Group (with linear fit)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 图2: 退化速率随电流变化
ax = axes[0, 1]
currents = [int(d['label'].split()[0][1:]) for d in degradation_rates]
rates = [d['rate_uvh'] for d in degradation_rates]
ax.bar(range(len(currents)), rates, tick_label=[f"{c}" for c in currents],
       color='steelblue', alpha=0.7)
ax.set_xlabel('Current Density (mA/cm²)')
ax.set_ylabel('Degradation Rate (μV/h)')
ax.set_title('Degradation Rate vs Current Density')
ax.grid(True, alpha=0.3)

# 图3: 全数据退化概览
ax = axes[1, 0]
sc = ax.scatter(df.Time_h, df.Voltage_V, c=df.CurrentDensity_mAcm2,
                cmap='viridis', s=20, alpha=0.8)
# 全局趋势线
z_global = np.polyfit(df.Time_h, df.Voltage_V, 1)
t_line = np.linspace(df.Time_h.min(), df.Time_h.max(), 200)
ax.plot(t_line, np.polyval(z_global, t_line), 'r--', linewidth=2,
        label=f'Global trend: {abs(z_global[0])*1e6:.1f} μV/h')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Voltage (V)')
ax.set_title('Full 16590h Degradation Overview')
ax.legend(); ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label='Current Density (mA/cm²)')

# 图4: 归一化比较 (各段起始电压归一化)
ax = axes[1, 1]
for i, dr in enumerate(degradation_rates):
    V0 = dr['voltage'][0]
    norm_v = (dr['voltage'] - V0) / V0 * 100  # 百分比变化
    norm_t = dr['time'] - dr['time'].min()
    ax.scatter(norm_t, norm_v, s=15, alpha=0.6, color=colors[i],
              label=f"{dr['label']} ({dr['rate_uvh']:.1f}μV/h)")
    # 线性拟合
    z_norm = np.polyfit(norm_t, norm_v, 1)
    ax.plot(norm_t, np.polyval(z_norm, norm_t), '--', color=colors[i], alpha=0.4, linewidth=1)
ax.set_xlabel('Time from start (h)'); ax.set_ylabel('Voltage Change (%)')
ax.set_title('Normalized Degradation (all groups start from 0%)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RUN_DIR / 'degradation_rates.png', dpi=150)
plt.show()

# ================================================================
# 6. RUL 估算
# ================================================================
print("\n" + "="*60)
print("RUL 估算 (基于800 mA/cm²段)")
print("="*60)

if degradation_rates:
    dr = degradation_rates[0]  # 800 mA/cm²
    V_start = dr['voltage'][0]
    V_fail_90 = V_start * 0.90
    V_fail_80 = V_start * 0.80
    rate_vh = dr['rate_uvh'] / 1e6

    print(f"参考电流: 800 mA/cm²")
    print(f"初始电压: {V_start:.4f} V")
    print(f"退化速率: {dr['rate_uvh']:.2f} μV/h ({dr['rate_v_yr']*1000:.2f} mV/年)")
    print(f"失效阈值 (90%): {V_fail_90:.4f} V → "
          f"{(V_start-V_fail_90)/rate_vh:.0f} h ({(V_start-V_fail_90)/rate_vh/8760:.1f} 年)")
    print(f"失效阈值 (80%): {V_fail_80:.4f} V → "
          f"{(V_start-V_fail_80)/rate_vh:.0f} h ({(V_start-V_fail_80)/rate_vh/8760:.1f} 年)")
    print(f"已运行: {df.Time_h.max():.0f} h")
    print(f"按10%衰减标准, 当前已消耗寿命的 {df.Time_h.max()/((V_start-V_fail_90)/rate_vh)*100:.0f}%")

print(f"\n全部输出已保存至 {RUN_DIR}/")
print("="*60)
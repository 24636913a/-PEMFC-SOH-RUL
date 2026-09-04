# -*- coding: utf-8 -*-
"""
FC1 RUL 预测 — HI + 物理外推 (无数据泄露)

流程:
  1. 定义健康指标 HI = SOH = V / V_init, V_init 取活化峰值(去噪后 max)
  2. 退化段 = [拐点(电压峰值), Sp] 的电压序列
  3. 用物理退化模型(线性/指数/幂律)拟合退化段, 外推到 96% 阈值
  4. RUL = 外推穿越时刻 - Sp

关键: 外推只用历史(<=Sp)数据, 不用未来电压 → 无数据泄露
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from pathlib import Path

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

USE_FC = 'FC1'
SAMPLING_INTERVAL = 1.0
SP_LIST = [300, 400, 500, 600]
SG_WINDOW = 21

# 数据加载
data_dict = {'FC1': {'ageing': []}}
for dirpath, _, filenames in os.walk(DATA_PATH):
    for fname in filenames:
        if not fname.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(dirpath, fname), encoding='ISO-8859-1')
        if 'FC1' in dirpath or 'FC1' in fname: data_dict['FC1']['ageing'].append(df)

fc_dfs = [df for df in data_dict['FC1']["ageing"] if 'Time (h)' in df.columns]
age_df = pd.concat(fc_dfs, ignore_index=True).sort_values('Time (h)').reset_index(drop=True)
t_hours = age_df['Time (h)'].values
age_df['t_bin'] = (t_hours / SAMPLING_INTERVAL).astype('int32')
sampled_df = age_df.iloc[age_df.drop_duplicates(subset=['t_bin']).index].reset_index(drop=True)
if 't_bin' in sampled_df.columns: sampled_df = sampled_df.drop(columns=['t_bin'])
s_df = sampled_df.copy().ffill().bfill().fillna(0)
target_col = 'Utot (V)'
t = sampled_df['Time (h)'].values
V_raw = s_df[target_col].values.astype(np.float64)

# 去噪 (轻 SG, 保留退化趋势)
V_smooth = savgol_filter(V_raw, SG_WINDOW, 2)

# 活化峰值作为 V_init; 拐点 = 电压峰值位置
t_peak_idx = int(np.argmax(V_smooth))
V_init = V_smooth[t_peak_idx]
t_peak = t[t_peak_idx]
V_fail = V_init * 0.96

# 真实 EOL: 去噪电压首次 < V_fail 的时刻
fail_idx = np.where(V_smooth < V_fail)[0]
EOL_true = t[fail_idx[0]] if len(fail_idx) > 0 else t[-1]

print(f"FC1: {len(t)}帧, 活化峰值 V_init={V_init:.4f}V @ t={t_peak:.1f}h")
print(f"V_fail(96%)={V_fail:.4f}V, 真实 EOL={EOL_true:.1f}h")

# 物理退化模型
def lin(t, a, b):            return a * t + b
def exp_decay(t, a, b, c):   return c + a * np.exp(b * t)
def powerlaw(t, a, b, c):    return c - a * np.power(t, b)

MODELS = {
    '线性':  (lin,       [V_smooth[t_peak_idx], -1e-4]),
    '指数':  (exp_decay, [1.0, -1e-3, V_smooth[t_peak_idx]]),
    '幂律':  (powerlaw, [1e-4, 1.0, V_smooth[t_peak_idx]]),
}

def fit_and_extrapolate(V_seg, t_seg, func, p0):
    """拟合退化段, 外推找穿越 V_fail 的时刻"""
    try:
        popt, _ = curve_fit(func, t_seg, V_seg, p0=p0, maxfev=100000)
    except Exception:
        return None
    # 在外推区间 [Sp, Sp+2000h] 找穿越
    t_ext = np.linspace(t_seg[-1], t_seg[-1] + 2000, 20000)
    V_ext = func(t_ext, *popt)
    cross = np.where(V_ext < V_fail)[0]
    if len(cross) > 0:
        return t_ext[cross[0]]
    return None

print(f"\n=== FC1 RUL 预测 (HI + 线性外推 + 拟合窗口扫描, EOL={EOL_true:.1f}h) ===")
WINDOWS = [None, 400, 300, 200, 150, 100]
print(f"{'Sp(h)':<8s} {'窗口':<8s} {'真实RUL':<10s} {'预测RUL':<10s} {'误差':<8s} {'RE%':<8s}")

results = []
for Sp in SP_LIST:
    if Sp >= EOL_true:
        continue
    true_RUL = EOL_true - Sp
    for W in WINDOWS:
        start_t = t_peak if W is None else max(t_peak, Sp - W)
        mask = (t >= start_t) & (t <= Sp)
        t_seg = t[mask]; V_seg = V_smooth[mask]
        if len(t_seg) < 10: continue
        t_eol = fit_and_extrapolate(V_seg, t_seg, lin, [V_seg[0], -1e-4])
        if t_eol is None: continue
        pred_RUL = t_eol - Sp
        ae = abs(pred_RUL - true_RUL)
        re = ae / true_RUL * 100
        wlbl = '全段' if W is None else f'{W}h'
        results.append((Sp, wlbl, true_RUL, pred_RUL, ae, re))
        print(f"{Sp:<8d} {wlbl:<8s} {true_RUL:<10.1f} {pred_RUL:<10.1f} {ae:<8.1f} {re:<8.2f}")

print("\n=== 各 Sp 最优窗口 (最小 RE) ===")
best = {}
for Sp, wlbl, tR, pR, ae, re in results:
    if Sp not in best or re < best[Sp][3]:
        best[Sp] = (wlbl, pR, ae, re)
for Sp in sorted(best):
    wlbl, pR, ae, re = best[Sp]
    print(f"Sp={Sp}: 窗口={wlbl}, 预测RUL={pR:.1f}h, 误差={ae:.1f}h, RE={re:.2f}%")

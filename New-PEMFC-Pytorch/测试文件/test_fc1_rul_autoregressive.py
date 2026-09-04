# -*- coding: utf-8 -*-
"""
FC1 RUL 预测 — 自回归时序预测 (无泄露)

正确范式: 用历史电压预测未来电压, 自回归滚动, 不喂未来真实数据。

训练: 用 [0, Sp] 数据学一步预测 f: x_t(含t时刻电压) -> v_{t+1}
预测: 从 Sp 出发, 滚动预测电压, 只更新电压列(其他传感器准静态假设不变),
      直到预测电压 < V_fail 穿越阈值。

关键: 预测阶段只用 Sp 时刻的真实传感器 + 模型自回归, 不碰未来。
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from pathlib import Path
import matplotlib; matplotlib.use('Agg')

seed_value = 42
np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")
USE_FC = 'FC1'
SAMPLING_INTERVAL = 1.0
SP_LIST = [300, 400, 500, 600]
SG_WINDOW = 21
MAX_STEPS = 1500  # 自回归最大步数 (h)

# ============ 数据加载 ============
data_dict = {'FC1': {'ageing': []}, 'FC2': {'ageing': []}}
for dirpath, _, filenames in os.walk(DATA_PATH):
    for fname in filenames:
        if not fname.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(dirpath, fname), encoding='ISO-8859-1')
        if 'FC1' in dirpath or 'FC1' in fname: data_dict['FC1']['ageing'].append(df)
        elif 'FC2' in dirpath or 'FC2' in fname: data_dict['FC2']['ageing'].append(df)

fc_dfs = [df for df in data_dict[USE_FC]["ageing"] if 'Time (h)' in df.columns]
age_df = pd.concat(fc_dfs, ignore_index=True).sort_values('Time (h)').reset_index(drop=True)
t_hours = age_df['Time (h)'].values
age_df['t_bin'] = (t_hours / SAMPLING_INTERVAL).astype('int32')
sampled_df = age_df.iloc[age_df.drop_duplicates(subset=['t_bin']).index].reset_index(drop=True)
if 't_bin' in sampled_df.columns: sampled_df = sampled_df.drop(columns=['t_bin'])
meas_cols = [c for c in sampled_df.columns if c != 'Time (h)']
s_df = sampled_df[meas_cols].copy().ffill().bfill().fillna(0)
meas_cols = [c for c in meas_cols if s_df[c].std() > 1e-10]
smoothed_df = pd.DataFrame({'Time (h)': sampled_df['Time (h)']})
for col in meas_cols:
    smoothed_df[col] = savgol_filter(s_df[col].values, SG_WINDOW, 2)

target_col = 'Utot (V)'
all_cols = list(meas_cols)
target_idx = all_cols.index(target_col)
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values)
time_h = smoothed_df['Time (h)'].values
n_features = len(all_cols)

# EOL 定义: 活化峰值 (去噪后 max)
V_smooth = data_values[:, target_idx].astype(np.float64)
V_init = V_smooth.max()
V_fail = V_init * 0.96
fail_idx = np.where(V_smooth < V_fail)[0]
EOL_true = time_h[fail_idx[0]] if len(fail_idx) > 0 else time_h[-1]
print(f"FC1: {N}帧, V_init(活化峰值)={V_init:.4f}V, V_fail={V_fail:.4f}V, EOL={EOL_true:.1f}h")

# ============ 模型: LSTM 一步预测 (L=1) ============
class OneStepLSTM(nn.Module):
    def __init__(self, n_feat, hidden=96):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, 1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ============ 训练一步预测: f(x_t) -> v_{t+1} ============
def train_onestep(Sp):
    mask = time_h <= Sp
    tr_data = data_values[mask]
    tr_time = time_h[mask]
    n_tr = len(tr_data)
    # 训练对: x_i = 传感器(t_i), y = 电压(t_{i+1})
    X = tr_data[:-1]          # (n_tr-1, n_feat)
    y = tr_data[1:, target_idx]  # (n_tr-1,)
    sX = MinMaxScaler(); sy = MinMaxScaler()
    X_s = sX.fit_transform(X).reshape(-1, 1, n_features)
    y_s = sy.fit_transform(y.reshape(-1, 1)).ravel()
    tr_ld = DataLoader(TensorDataset(
        torch.tensor(X_s, dtype=torch.float32),
        torch.tensor(y_s, dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)
    model = OneStepLSTM(n_features).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    mse = nn.MSELoss()
    for ep in range(300):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = mse(model(bx), by); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
    return model, sX, sy

# ============ 自回归 RUL 预测 ============
def predict_autoregressive(model, sX, sy, Sp):
    # 起点: Sp 时刻的真实传感器 (历史, 已知)
    sp_idx = np.where(time_h <= Sp)[0][-1]
    x = data_values[sp_idx].copy()  # (n_feat,) 含 Sp 真实电压
    pred_V = [float(x[target_idx])]
    model.eval()
    with torch.no_grad():
        for k in range(MAX_STEPS):
            x_in = torch.tensor(sX.transform(x.reshape(1, -1)).reshape(1, 1, n_features),
                                dtype=torch.float32).to(device)
            v_next = model(x_in).cpu().numpy()[0, 0]
            v_next = sy.inverse_transform([[v_next]])[0, 0]
            # 只更新电压列, 其他传感器保持不变 (准静态假设)
            x = x.copy()
            x[target_idx] = v_next
            pred_V.append(v_next)
            if v_next < V_fail:
                break
    # 穿越时刻
    t_eol = Sp + (len(pred_V) - 1) * SAMPLING_INTERVAL
    return t_eol, np.array(pred_V)

print(f"\n=== FC1 RUL 自回归预测 (无泄露, EOL={EOL_true:.1f}h) ===")
print(f"{'Sp(h)':<8s} {'真实RUL':<10s} {'预测RUL':<10s} {'误差(h)':<10s} {'RE%':<8s} {'穿越?':<8s}")
results = []
for Sp in SP_LIST:
    if Sp >= EOL_true:
        print(f"{Sp:<8d}  Sp >= EOL, 跳过"); continue
    true_RUL = EOL_true - Sp
    model, sX, sy = train_onestep(Sp)
    t_eol, pred_V = predict_autoregressive(model, sX, sy, Sp)
    pred_RUL = t_eol - Sp
    abs_err = abs(pred_RUL - true_RUL)
    rel_err = abs_err / true_RUL * 100
    crossed = pred_V[-1] < V_fail
    results.append((Sp, true_RUL, pred_RUL, abs_err, rel_err, crossed))
    print(f"{Sp:<8d} {true_RUL:<10.1f} {pred_RUL:<10.1f} {abs_err:<10.1f} {rel_err:<8.2f} {'是' if crossed else '否':<8s}")
    print(f"    [诊断] 预测电压末值={pred_V[-1]:.4f}V (V_fail={V_fail:.4f}), 预测步数={len(pred_V)-1}")

print("\n=== 汇总 ===")
crossed_all = [r[5] for r in results]
print(f"穿越阈值: {sum(crossed_all)}/{len(crossed_all)}")
if results:
    re_mean = np.mean([r[4] for r in results])
    print(f"平均 RE = {re_mean:.2f}%")

# ================================================================
# PEMFC RUL 消融实验 — 独立文件
# ================================================================
# 用途: 对 DLinear / ResMLP / LSTM / TSMixer / TFT 进行公平消融对比
# 每个模型使用独立调参后的最优学习率，Loop 50 评估
#
# 使用: 修改下面 USE_FC 和 MODEL_NAME，然后直接运行
#   USE_FC    = 'FC1' 或 'FC2'
#   MODEL_NAME = 'dlinear' | 'resmlp' | 'lstm' | 'tsmixer' | 'tft'
#
# DLinear 特殊处理: 需要 L=10 序列输入 (L=1 下退化为线性回归)
# 其他模型: L=1 + lag 增强 (lag=1,2,4)
# ================================================================

#%% ================================================================
# Section 1: 环境配置
# ================================================================
import sys, os, pickle, random
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# ── 🔧 修改这里 ──────────────────────────────────────────────────
USE_FC = 'FC2'           # 'FC1' (准静态) 或 'FC2' (纹波)
MODEL_NAME = 'dlinear'    # 'dlinear' | 'resmlp' | 'lstm' | 'tsmixer' | 'tft'
# ─────────────────────────────────────────────────────────────────

assert USE_FC in ('FC1', 'FC2'), "USE_FC 必须是 'FC1' 或 'FC2'"
assert MODEL_NAME in ('dlinear', 'resmlp', 'lstm', 'tsmixer', 'tft'), \
    "MODEL_NAME 必须是 'dlinear'|'resmlp'|'lstm'|'tsmixer'|'tft'"

# FC1/FC2 各自的最优采样间隔
SAMPLING_INTERVAL = {'FC1': 1.0, 'FC2': 0.5}[USE_FC]

# 各模型独立调参后的最优学习率 (公平消融)
_MODEL_LR = {
    'dlinear': 0.01,
    'resmlp':  0.001,
    'lstm':    0.0002,
    'tsmixer': 0.01,
    'tft':     0.0003,
}

# 共享超参 (FC1/FC2 通用)
HIDDEN_SIZE = 96
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 100

# LSTM/TFT 的层数
NUM_LAYERS = 1

# DLinear 特殊: 需要 L=10
DLINEAR_SEQ_LEN = 10

# 固定随机种子 (基础种子, Loop 时递增)
BASE_SEED = 42

# 输出路径
OUTPUT_BASE = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\New-PEMFC-Pytorch\消融实验outputs")

# 硬件
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 设置初始种子
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据路径
DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

print(f"\n{'='*60}")
print(f"消融实验: {MODEL_NAME.upper()} | {USE_FC} | {SAMPLING_INTERVAL}h采样")
print(f"学习率: {_MODEL_LR[MODEL_NAME]} | hidden={HIDDEN_SIZE} | dropout={DROPOUT}")
print(f"{'='*60}")


#%% ================================================================
# Section 2: 数据加载
# ================================================================
data_dict = {"FC1": {"ageing": []}, "FC2": {"ageing": []}}
csv_files = list(DATA_PATH.rglob("*.csv"))

for file in csv_files:
    filename = file.name
    try:
        df = pd.read_csv(file, encoding="ISO-8859-1")
    except:
        continue

    if "FC1" in filename:
        stack = "FC1"
    elif "FC2" in filename:
        stack = "FC2"
    else:
        continue

    if "Ageing" in filename:
        data_dict[stack]["ageing"].append(df)

print(f"FC1 ageing: {len(data_dict['FC1']['ageing'])} 文件, "
      f"FC2 ageing: {len(data_dict['FC2']['ageing'])} 文件")


#%% ================================================================
# Section 3: 数据预处理
# ================================================================

IS_DLINEAR = (MODEL_NAME == 'dlinear')
MAX_LAG = 4
LAGS = [1, 2, 4]

print(f"\n预处理模式: {'DLinear L=' + str(DLINEAR_SEQ_LEN) if IS_DLINEAR else 'L=1 + lag=' + str(LAGS)}")

# ── 3.1 构建原始数据 ──
fc_dfs = [df for df in data_dict[USE_FC]["ageing"] if 'Time (h)' in df.columns]
age_df = pd.concat(fc_dfs, ignore_index=True).sort_values('Time (h)').reset_index(drop=True)
print(f"{USE_FC} 原始数据: {len(age_df)} 行")

# ── 3.2 降采样 ──
t_hours = age_df['Time (h)'].values
if SAMPLING_INTERVAL < 1.0:
    age_df['t_bin'] = (t_hours / SAMPLING_INTERVAL).astype('int32')
    sampled_idx = age_df.drop_duplicates(subset=['t_bin']).index
else:
    sampled_idx = age_df.astype({'Time (h)': 'int32'}).drop_duplicates(subset=['Time (h)']).index
sampled_df = age_df.iloc[sampled_idx].reset_index(drop=True)
if 't_bin' in sampled_df.columns:
    sampled_df = sampled_df.drop(columns=['t_bin'])
print(f"降采样至 {SAMPLING_INTERVAL}h: {len(sampled_df)} 行")

# ── 3.3 SG 滤波 ──
meas_cols = [c for c in sampled_df.columns if c != 'Time (h)']
s_df = sampled_df[meas_cols].ffill().bfill().fillna(0)
valid_cols = [c for c in meas_cols if s_df[c].std() > 1e-10]
meas_cols = valid_cols

smoothed_df = pd.DataFrame()
smoothed_df['Time (h)'] = sampled_df['Time (h)']
for col in meas_cols:
    smoothed_df[col] = savgol_filter(s_df[col].values, window_length=21, polyorder=2)
print(f"SG滤波完成: {len(meas_cols)} 特征")

# ── 3.4 特征 & 目标 ──
target_col = 'Utot (V)'
all_cols = [c for c in meas_cols]
target_idx = all_cols.index(target_col)
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values)
n_features_orig = len(all_cols)

# ── 3.5 构建样本 ──
if IS_DLINEAR:
    # DLinear: L=DLINEAR_SEQ_LEN 滑动窗口 (无 lag 标记)
    L = DLINEAR_SEQ_LEN
    X_list, y_list = [], []
    for i in range(N - L):
        X_list.append(data_values[i:i+L])       # (L, n_feat)
        y_list.append(data_values[i+L, target_idx])
    X_raw = np.array(X_list)  # (n_samp, L, n_feat)
    y_raw = np.array(y_list)
    n_features = n_features_orig  # 无 lag marker
    print(f"滑动窗口 L={L}: {len(X_raw)} 样本")
else:
    # L=1 + lag 增强
    X_sec, y_sec = [], []
    for lag in LAGS:
        marker = lag / 10.0
        X_lag = data_values[MAX_LAG - lag : N - lag]
        y_lag = data_values[MAX_LAG : N, target_idx]
        lag_col = np.full((len(X_lag), 1), marker, dtype=np.float32)
        X_sec.append(np.concatenate([X_lag, lag_col], axis=1))
        y_sec.append(y_lag)
    X_raw_flat = np.concatenate(X_sec, axis=0)
    y_raw = np.concatenate(y_sec, axis=0)
    n_features = X_raw_flat.shape[1]  # 24 + 1 = 25
    n_per_lag = N - MAX_LAG
    print(f"Lag增强 (lag={LAGS}): 有效样本={n_per_lag}, 增强后={len(y_raw)}")

# ── 3.6 时序划分 (50/10/40) ──
if IS_DLINEAR:
    n_total = len(X_raw)
    train_end = int(n_total * 0.5)
    val_end = int(n_total * 0.6)
    X_train_raw = X_raw[:train_end]
    X_val_raw = X_raw[train_end:val_end]
    X_test_raw = X_raw[val_end:]
    y_train_raw = y_raw[:train_end]
    y_val_raw = y_raw[train_end:val_end]
    y_test_raw = y_raw[val_end:]
else:
    # 每个 lag 段独立划分 (防止数据泄漏)
    n_per_lag = N - MAX_LAG
    te_lag = int(n_per_lag * 0.5)
    ve_lag = int(n_per_lag * 0.6)
    X_tr_p, X_vl_p, X_ts_p = [], [], []
    y_tr_p, y_vl_p, y_ts_p = [], [], []
    for k in range(len(LAGS)):
        s = k * n_per_lag
        X_tr_p.append(X_raw_flat[s : s+te_lag])
        X_vl_p.append(X_raw_flat[s+te_lag : s+ve_lag])
        X_ts_p.append(X_raw_flat[s+ve_lag : s+n_per_lag])
        y_tr_p.append(y_raw[s : s+te_lag])
        y_vl_p.append(y_raw[s+te_lag : s+ve_lag])
        y_ts_p.append(y_raw[s+ve_lag : s+n_per_lag])
    X_train_raw = np.concatenate(X_tr_p)
    X_val_raw = np.concatenate(X_vl_p)
    X_test_raw = np.concatenate(X_ts_p)
    y_train_raw = np.concatenate(y_tr_p)
    y_val_raw = np.concatenate(y_vl_p)
    y_test_raw = np.concatenate(y_ts_p)

print(f"划分: 训练 {len(X_train_raw)} | 验证 {len(X_val_raw)} | 测试 {len(X_test_raw)}")

# ── 3.7 MinMax 归一化 ──
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

if IS_DLINEAR:
    # DLinear: reshape to (n, L, n_feat), then flatten for scaler
    ns_tr, L_dl, nf = X_train_raw.shape
    X_train_flat = scaler_X.fit_transform(X_train_raw.reshape(-1, nf)).reshape(ns_tr, L_dl, nf)
    X_val_flat   = scaler_X.transform(X_val_raw.reshape(-1, nf)).reshape(len(X_val_raw), L_dl, nf)
    X_test_flat  = scaler_X.transform(X_test_raw.reshape(-1, nf)).reshape(len(X_test_raw), L_dl, nf)
    X_train = X_train_flat
    X_val   = X_val_flat
    X_test  = X_test_flat
else:
    X_train_flat = scaler_X.fit_transform(X_train_raw)
    X_val_flat   = scaler_X.transform(X_val_raw)
    X_test_flat  = scaler_X.transform(X_test_raw)
    X_train = X_train_flat.reshape(-1, 1, n_features)
    X_val   = X_val_flat.reshape(-1, 1, n_features)
    X_test  = X_test_flat.reshape(-1, 1, n_features)

y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
y_val   = scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel()
y_test  = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# 失效阈值
initial_voltage = np.percentile(y_train_raw, 95)
failure_threshold = initial_voltage * 0.96
print(f"初始电压: {initial_voltage:.4f}V, 失效阈值(96%): {failure_threshold:.4f}V")


#%% ================================================================
# Section 4: 模型定义
# ================================================================

class DLinearModel(nn.Module):
    """DLinear: 时序分解 + 线性映射 (需 L>=5)"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        seq_len = DLINEAR_SEQ_LEN
        k = min(5, seq_len)
        if k % 2 == 0:
            k -= 1
        self.decomp = nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2)
        self.trend_linear = nn.Linear(seq_len, output_size)
        self.residual_linear = nn.Linear(seq_len, output_size)

    def forward(self, x):
        # x: (batch, L, n_feat)
        x_t = x.permute(0, 2, 1)  # (batch, n_feat, L)
        trend = self.decomp(x_t)
        residual = x_t - trend
        t_out = self.trend_linear(trend)
        r_out = self.residual_linear(residual)
        return (t_out + r_out).mean(dim=1)


class ResMLPModel(nn.Module):
    """残差MLP (L=1 + lag增强)"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm_mid = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.head = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        h = self.norm_in(x)
        residual = self.proj(h)
        h = F.silu(self.fc1(h))
        h = self.dropout(h)
        h = F.silu(self.fc2(h))
        h = h + residual
        h = self.norm_mid(h)
        h = F.silu(self.fc3(h))
        return self.head(h)


class LSTMBaseline(nn.Module):
    """LSTM 基线"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        return self.fc(out)


class GatedResidualNetwork(nn.Module):
    """GRN (TFT 组件)"""
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_hidden, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        a = self.proj(x)
        h = F.elu(self.fc1(x))
        g = torch.sigmoid(self.gate(h))
        glu = g * self.fc2(h)
        return self.ln(a + self.dropout(glu))


class TSMixerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual


class TSMixerModel(nn.Module):
    """TSMixer: All-MLP"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSMixerBlock(input_size, hidden_size, dropout)
            for _ in range(max(1, num_layers))
        ])
        self.head = nn.Linear(input_size, output_size)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class TFTModel(nn.Module):
    """TFT: VS + LSTM + GRN + MultiheadAttention"""
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 nhead=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        if hidden_size % nhead != 0:
            nhead = 2 if hidden_size % 2 == 0 else 1

        self.feat_weights = nn.Parameter(torch.zeros(input_size))
        self.feat_bias = nn.Parameter(torch.zeros(input_size))

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.grn = GatedResidualNetwork(hidden_size, dim_feedforward, hidden_size, dropout)
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        x_weighted = x * w.unsqueeze(0).unsqueeze(0)
        if self.training:
            x_weighted = x_weighted + torch.randn_like(x_weighted) * 0.01
        lstm_out, _ = self.lstm(x_weighted)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out)
        attn_out, _ = self.self_attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        out = attn_out[:, -1, :]
        return self.fc(out)


# ── 模型工厂 ──
def make_model(input_size):
    """根据 MODEL_NAME 创建模型"""
    if MODEL_NAME == 'dlinear':
        return DLinearModel(input_size, HIDDEN_SIZE, NUM_LAYERS, 1)
    elif MODEL_NAME == 'resmlp':
        return ResMLPModel(input_size, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT)
    elif MODEL_NAME == 'lstm':
        return LSTMBaseline(input_size, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT)
    elif MODEL_NAME == 'tsmixer':
        return TSMixerModel(input_size, HIDDEN_SIZE, 2, 1, DROPOUT)
    elif MODEL_NAME == 'tft':
        return TFTModel(input_size, HIDDEN_SIZE, NUM_LAYERS, 1,
                        nhead=2, dim_feedforward=128, dropout=DROPOUT)


#%% ================================================================
# Section 5: 训练 + Loop 50 评估
# ================================================================

# DataLoaders
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),     batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),   batch_size=BATCH_SIZE, shuffle=False)

print(f"\nDataLoader: train={len(train_loader)} batches, val={len(val_loader)}, test={len(test_loader)}")


def train_one_run(model, seed):
    """单次训练, 返回最佳模型和训练历史"""
    # 设置种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    lr = _MODEL_LR[MODEL_NAME]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    patience_cnt = 0
    history = {'train_loss': [], 'val_loss': []}

    # SWA
    swa_start = 40
    swa_model = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        history['train_loss'].append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item()
        avg_val = val_loss / len(val_loader)
        history['val_loss'].append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

        # SWA
        if epoch >= swa_start:
            if swa_model is None:
                swa_model = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_model:
                    swa_model[k] = swa_model[k] * (swa_count/(swa_count+1)) + \
                                   model.state_dict()[k].cpu() * (1/(swa_count+1))
                swa_count += 1

    # 恢复 best, 再叠加 SWA
    model.load_state_dict(best_state)
    if swa_model is not None:
        swa_state = {k: v.to(device) for k, v in swa_model.items()}
        model.load_state_dict(swa_state)

    return model, history


def evaluate(model):
    """评估模型, 返回原始电压尺度的预测值和真实值"""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds.append(model(bx).cpu().numpy())
            trues.append(by.numpy())
    yp_norm = np.concatenate(preds).ravel()
    yt_norm = np.concatenate(trues).ravel()

    yp = scaler_y.inverse_transform(yp_norm.reshape(-1, 1)).ravel()
    yt = scaler_y.inverse_transform(yt_norm.reshape(-1, 1)).ravel()

    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    return yt, yp, r2, rmse, mae


# ── Loop 50 ──
N_LOOPS = 50
N_ENSEMBLE = 5

print(f"\n{'='*60}")
print(f"Loop {N_LOOPS} 评估开始: {MODEL_NAME.upper()} | {USE_FC}")
print(f"{'='*60}")

loop_results = []
for li in range(N_LOOPS):
    seed = BASE_SEED + li * 10
    m = make_model(n_features)
    m, hist = train_one_run(m, seed)
    yt, yp, r2, rmse, mae = evaluate(m)
    loop_results.append({'r2': r2, 'rmse': rmse, 'mae': mae, 'y_true': yt, 'y_pred': yp})
    print(f"  [{li+1:2d}/{N_LOOPS}] seed={seed}  R²={r2:.4f}  RMSE={rmse:.6f}V  MAE={mae:.6f}V")

# ── 统计 ──
r2_list = [r['r2'] for r in loop_results]
r2_min, r2_mean, r2_max = np.min(r2_list), np.mean(r2_list), np.max(r2_list)

# Top-K 集成
loop_results.sort(key=lambda x: x['r2'], reverse=True)
top_k = loop_results[:N_ENSEMBLE]
y_true_ens = top_k[0]['y_true']
y_pred_ens = np.mean([r['y_pred'] for r in top_k], axis=0)
r2_ensemble = r2_score(y_true_ens, y_pred_ens)

print(f"\n{'='*60}")
print(f"📊 {MODEL_NAME.upper()} | {USE_FC} | {SAMPLING_INTERVAL}h | Loop {N_LOOPS} 结果")
print(f"{'='*60}")
print(f"  Min R²:  {r2_min:.4f}")
print(f"  Mean R²: {r2_mean:.4f}")
print(f"  Max R²:  {r2_max:.4f}")
print(f"  Top-{N_ENSEMBLE} 集成 R²: {r2_ensemble:.4f}")
print(f"  RMSE range: [{np.min([r['rmse'] for r in loop_results]):.6f}, "
      f"{np.max([r['rmse'] for r in loop_results]):.6f}] V")


#%% ================================================================
# Section 6: 保存结果
# ================================================================

# 输出目录
date_str = datetime.now().strftime('%Y%m%d')
n = 1
while (OUTPUT_BASE / f"{date_str}_{n}_{MODEL_NAME}_{USE_FC}").exists():
    n += 1
RUN_DIR = OUTPUT_BASE / f"{date_str}_{n}_{MODEL_NAME}_{USE_FC}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n输出目录: {RUN_DIR}")

# 保存汇总
summary = {
    'model': MODEL_NAME,
    'fc': USE_FC,
    'sampling_interval': SAMPLING_INTERVAL,
    'learning_rate': _MODEL_LR[MODEL_NAME],
    'hidden_size': HIDDEN_SIZE,
    'dropout': DROPOUT,
    'batch_size': BATCH_SIZE,
    'n_loops': N_LOOPS,
    'n_ensemble': N_ENSEMBLE,
    'r2_min': r2_min,
    'r2_mean': r2_mean,
    'r2_max': r2_max,
    'r2_ensemble': r2_ensemble,
    'r2_all': r2_list,
    'is_dlinear_l10': IS_DLINEAR,
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_test': len(X_test),
}

with open(RUN_DIR / 'ablation_summary.pkl', 'wb') as f:
    pickle.dump(summary, f)

# 保存文本报告
report = f"""消融实验报告
============
模型: {MODEL_NAME.upper()}
数据集: {USE_FC}
采样间隔: {SAMPLING_INTERVAL}h
输入模式: {'L=' + str(DLINEAR_SEQ_LEN) + ' 滑动窗口' if IS_DLINEAR else 'L=1 + lag=' + str(LAGS)}

超参数:
  学习率: {_MODEL_LR[MODEL_NAME]}
  hidden_size: {HIDDEN_SIZE}
  dropout: {DROPOUT}
  batch_size: {BATCH_SIZE}
  epochs max: {EPOCHS}
  patience: {PATIENCE}

数据:
  训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}

Loop {N_LOOPS} 结果:
  Min R²:  {r2_min:.4f}
  Mean R²: {r2_mean:.4f}
  Max R²:  {r2_max:.4f}
  Top-{N_ENSEMBLE} 集成 R²: {r2_ensemble:.4f}

各次 R²:
  {', '.join(f'{r:.4f}' for r in r2_list)}
"""

with open(RUN_DIR / 'report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)

# 快速可视化: R² 分布直方图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(r2_list, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
for v, label, color in [(r2_min, f'Min={r2_min:.3f}', 'red'),
                          (r2_mean, f'Mean={r2_mean:.3f}', 'orange'),
                          (r2_max, f'Max={r2_max:.3f}', 'green')]:
    ax.axvline(x=v, color=color, linestyle='--', linewidth=2, label=label)
ax.set_xlabel('R²')
ax.set_ylabel('Count')
ax.set_title(f'{MODEL_NAME.upper()} R² Distribution ({N_LOOPS} runs, {USE_FC})')
ax.legend()
ax.grid(alpha=0.3)

# 最佳模型预测 vs 真实
ax = axes[1]
best = loop_results[0]
time_axis = np.arange(len(best['y_true'])) * SAMPLING_INTERVAL
ax.plot(time_axis, best['y_true'], 'b-', linewidth=1.5, alpha=0.7, label='True')
ax.plot(time_axis, best['y_pred'], 'r--', linewidth=1.5, alpha=0.7, label=f'Pred (R²={best["r2"]:.4f})')
ax.set_xlabel(f'Test Time (hours)')
ax.set_ylabel('Stack Voltage (V)')
ax.set_title(f'{MODEL_NAME.upper()} Best Model Prediction ({USE_FC})')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RUN_DIR / 'results_summary.png', dpi=150)
plt.show()

print(f"\n✅ 消融实验完成! 结果保存至 {RUN_DIR}/")
print(f"   - ablation_summary.pkl")
print(f"   - report.txt")
print(f"   - results_summary.png")

# ================================================================
# PEMFC 跨工况迁移: FC1↔FC2
# ================================================================
# 训练在FC1, 测试FC2; 训练在FC2, 测试FC1
# ================================================================
import sys, os, pickle, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

seed_value = 42
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

# ================================================================
# 加载数据
# ================================================================
csv_files = list(DATA_PATH.rglob("*.csv"))
data_dict = {"FC1": {"ageing": []}, "FC2": {"ageing": []}}
for file in csv_files:
    try:
        df = pd.read_csv(file, encoding="ISO-8859-1")
    except:
        continue
    if "FC1" in file.name and "Ageing" in file.name:
        data_dict["FC1"]["ageing"].append(df)
    elif "FC2" in file.name and "Ageing" in file.name:
        data_dict["FC2"]["ageing"].append(df)

# ================================================================
# 预处理函数
# ================================================================
def preprocess(fc_key, sampling=1.0):
    from scipy.signal import savgol_filter

    dfs = [df for df in data_dict[fc_key]["ageing"] if 'Time (h)' in df.columns]
    age_df = pd.concat(dfs, ignore_index=True).sort_values('Time (h)').reset_index(drop=True)

    # 降采样
    sampled_idx = age_df.astype({'Time (h)': 'int32'}).drop_duplicates(subset=['Time (h)']).index
    sampled_df = age_df.iloc[sampled_idx].reset_index(drop=True)

    # SG滤波
    meas_cols = [c for c in sampled_df.columns if c != 'Time (h)']
    s_df = sampled_df[meas_cols].ffill().bfill().fillna(0)
    valid_cols = [c for c in meas_cols if s_df[c].std() > 1e-10]

    smoothed_df = pd.DataFrame()
    smoothed_df['Time (h)'] = sampled_df['Time (h)']
    for col in valid_cols:
        smoothed_df[col] = savgol_filter(s_df[col].values, window_length=21, polyorder=2)

    target_col = 'Utot (V)'
    all_cols = [c for c in valid_cols]
    target_idx = all_cols.index(target_col)
    data_values = smoothed_df[all_cols].values.astype(np.float32)
    N = len(data_values)

    # Lag增强
    LAGS = [1, 2, 4]
    MAX_LAG = 4
    X_sections, y_sections = [], []
    for lag in LAGS:
        lag_marker = lag / 10.0
        X_lag = data_values[MAX_LAG-lag : N-lag]
        y_lag = data_values[MAX_LAG : N, target_idx]
        lag_col = np.full((len(X_lag), 1), lag_marker, dtype=np.float32)
        X_sections.append(np.concatenate([X_lag, lag_col], axis=1))
        y_sections.append(y_lag)

    X_raw = np.concatenate(X_sections, axis=0)
    y_raw = np.concatenate(y_sections, axis=0)

    # 时序划分
    n_per_lag = N - MAX_LAG
    train_end = int(n_per_lag * 0.5)
    val_end   = int(n_per_lag * 0.6)

    X_tr, y_tr = [], []; X_va, y_va = [], []; X_ts, y_ts = [], []
    for k in range(len(LAGS)):
        s = k * n_per_lag
        X_tr.append(X_raw[s : s+train_end])
        X_va.append(X_raw[s+train_end : s+val_end])
        X_ts.append(X_raw[s+val_end : s+n_per_lag])
        y_tr.append(y_raw[s : s+train_end])
        y_va.append(y_raw[s+train_end : s+val_end])
        y_ts.append(y_raw[s+val_end : s+n_per_lag])

    return (np.concatenate(X_tr), np.concatenate(y_tr),
            np.concatenate(X_va), np.concatenate(y_va),
            np.concatenate(X_ts), np.concatenate(y_ts))

# ================================================================
# TFT 模型 (与主文件相同)
# ================================================================
class GatedResidualNetwork(nn.Module):
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
        h = torch.nn.functional.elu(self.fc1(x))
        g = torch.sigmoid(self.gate(h))
        return self.ln(a + self.dropout(g * self.fc2(h)))

class TFTModel(nn.Module):
    def __init__(self, input_size, hidden_size=80, num_layers=1, output_size=1,
                 nhead=4, dim_feedforward=128, dropout=0.15):
        super().__init__()
        if hidden_size % nhead != 0:
            nhead = 2 if hidden_size % 2 == 0 else 1
        self.feat_weights = nn.Parameter(torch.zeros(input_size))
        self.feat_bias   = nn.Parameter(torch.zeros(input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.grn = GatedResidualNetwork(hidden_size, dim_feedforward, hidden_size, dropout)
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        xw = x * w.unsqueeze(0).unsqueeze(0)
        if self.training:
            xw = xw + torch.randn_like(xw) * 0.01
        lstm_out, _ = self.lstm(xw)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out)
        attn_out, _ = self.self_attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        return self.fc(attn_out[:, -1, :])

# ================================================================
# 训练函数
# ================================================================
def train_one(model, X_tr, y_tr, X_va, y_va, lr=2e-4, epochs=400, patience=80):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25)
    criterion = nn.MSELoss()
    best_val = float('inf'); best_state = None; patience_cnt = 0
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience: break
    model.load_state_dict(best_state)
    return model

# ================================================================
# 主体: FC1 -> FC2 和 FC2 -> FC1
# ================================================================
print("跨工况迁移实验")
print("="*60)

for train_fc, test_fc in [('FC1', 'FC2'), ('FC2', 'FC1')]:
    print(f"\n{'='*60}")
    print(f"训练: {train_fc}, 测试: {test_fc}")
    print(f"{'='*60}")

    # 加载训练数据
    X_tr_raw, y_tr_raw, X_va_raw, y_va_raw, X_ts_raw, y_ts_raw = preprocess(train_fc)
    X_test_raw, y_test_raw, _, _, _, _ = preprocess(test_fc)
    X_test_raw_ts = X_test_raw; y_test_raw_ts = y_test_raw  # test set for target FC

    # 归一化 (用训练集 fit)
    sc_X = MinMaxScaler(feature_range=(0, 1))
    sc_y = MinMaxScaler(feature_range=(0, 1))
    X_tr = torch.tensor(sc_X.fit_transform(X_tr_raw).reshape(-1, 1, 25), dtype=torch.float32)
    y_tr = torch.tensor(sc_y.fit_transform(y_tr_raw.reshape(-1, 1)).ravel(), dtype=torch.float32).view(-1, 1)
    X_va = torch.tensor(sc_X.transform(X_va_raw).reshape(-1, 1, 25), dtype=torch.float32)
    y_va = torch.tensor(sc_y.transform(y_va_raw.reshape(-1, 1)).ravel(), dtype=torch.float32).view(-1, 1)

    # 训练
    model = TFTModel(25, 80, 1, 1).to(device)
    model = train_one(model, X_tr.to(device), y_tr.to(device), X_va.to(device), y_va.to(device))

    # 在训练域测试集评估
    X_ts = torch.tensor(sc_X.transform(X_ts_raw).reshape(-1, 1, 25), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred_tr = sc_y.inverse_transform(model(X_ts).cpu().numpy().reshape(-1,1)).ravel()
        true_tr = sc_y.inverse_transform(y_ts_raw.reshape(-1,1)).ravel()
    r2_same = r2_score(true_tr, pred_tr)
    print(f"同域测试 ({train_fc}->{train_fc}): R2={r2_same:.4f}")

    # 跨工况测试: 用训练域的 scaler 转换测试域数据
    X_test = torch.tensor(sc_X.transform(X_test_raw_ts).reshape(-1, 1, 25), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred_cross = sc_y.inverse_transform(model(X_test).cpu().numpy().reshape(-1,1)).ravel()
        # 获取测试域的真实电压 (原始值, 不需要 scaler)
        true_cross_v = y_test_raw_ts  # 原始电压
    r2_cross = r2_score(true_cross_v, pred_cross)
    rmse_cross = np.sqrt(mean_squared_error(true_cross_v, pred_cross))
    print(f"跨域测试 ({train_fc}->{test_fc}): R2={r2_cross:.4f}, RMSE={rmse_cross:.4f}V")

print("\n跨工况迁移实验完成。")

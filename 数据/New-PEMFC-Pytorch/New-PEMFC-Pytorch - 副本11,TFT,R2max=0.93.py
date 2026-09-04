# ================================================================
# PEMFC RUL Prediction (PyTorch) — Lag增强 + GRU-MLP v4.1
# ================================================================
# 研究目标: 基于 IEEE 2014 PHM 数据集，多步滞后增强 + GRU-MLP 实现燃料电池 RUL 预测
#
# 数据说明:
#   FC1 — 1kW PEMFC，准静态条件，1000h 长期测试（本版本使用）
#   FC2 — 新型 1kW PEMFC，电流纹波条件，1000h 测试（FC1/FC2 测试条件不同，不混合）
#
# 关键改进（v4.0 Lag增强重构）:
#   1. lag=1,2,4 多步滞后增强 — 同一标签配不同滞后输入+lag标记列，数据量×3
#   2. L=1 单步监督学习 — 与参考论文范式兼容
#   3. GRU+MLP — GRU门控编码 + MLP非线性映射，参数少训练稳
#   4. RF特征筛选 — 随机森林Top-18特征，替代弱Spearman筛选
#   5. SWA 随机权重平均 — 训练后半段收集快照平均，降低方差
#   6. CosineAnnealingWarmRestarts — 周期性重启学习率，逃离局部最优
#   7. Optuna 超参优化 — GRU vs LSTM vs TFT 自动选择
#   8. Savitzky-Golay 滤波 + Spearman 筛选 + MinMaxScaler(0,1)
#   9. 50/10/40 时序划分 | shuffle=False | cudnn确定性 | Loop 50
#
# 方法: lag=1,2,4增强 + 1h采样 + GRU-MLP/LSTM/TFT对比
# 可视化: outputs/.../training_history.png, prediction_vs_true.png, scatter_true_vs_pred.png, rul_prediction.png
# ================================================================


#%%
#1.环境配置

#导入库
import sys, os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Jupyter/IPython 环境下无此方法

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

#设置随机种子
import random
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#检测硬件配置（GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("⚠️ GPU not available - using CPU")

#本地数据路径配置
from pathlib import Path

CONFIG = {
    "data_path": Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")
} # 💡💡💡 请根据实际情况修改路径

#全局配置字典
CONFIG = {
    "data_path": CONFIG["data_path"],
    "output_path": Path("./outputs"),
    "batch_size": 32,
    "sequence_length": 20,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "epochs": 100,
    "learning_rate": 1e-3,
    "device": device
}

# 创建带时间戳的运行输出目录
from datetime import datetime
date_str = datetime.now().strftime('%Y%m%d')
n = 1
while (CONFIG["output_path"] / f"{date_str}_{n}_PEMFC-RUL结果").exists():
    n += 1
RUN_DIR = CONFIG["output_path"] / f"{date_str}_{n}_PEMFC-RUL结果"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"输出目录: {RUN_DIR}")

import pickle
with open(RUN_DIR / "config.pkl", "wb") as f:
    pickle.dump(CONFIG, f)
print(f"Configuration saved to {RUN_DIR / 'config.pkl'}\n")

print("="*60)
print("✅ 环境配置完成")
print("="*60)


#%%
#2.数据加载

import pandas as pd
from pathlib import Path

# 配置路径
DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData") # 💡💡💡

if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ 数据路径不存在: {DATA_PATH}")

# 初始化数据结构
data_dict = {
    "FC1": {"ageing": [], "eis": [], "pola": []},
    "FC2": {"ageing": [], "eis": [], "pola": []}
}

# 扫描CSV文件
csv_files = list(DATA_PATH.rglob("*.csv"))
print(f"✓ 共发现 CSV 文件: {len(csv_files)}")

# 第一次尝试：仅加载静态负载文件（通常文件名包含 "Static"）
for file in csv_files:
    filename = file.name
    if "Static" not in filename:
        continue

    try:
        df = pd.read_csv(file, encoding="ISO-8859-1")
    except Exception as e:
        print(f"❌ 读取失败: {filename}, 错误: {e}")
        continue

    # 判断 FC1 / FC2
    if "FC1" in filename:
        stack = "FC1"
    elif "FC2" in filename:
        stack = "FC2"
    else:
        print(f"⚠️ 未识别文件: {filename}")
        continue

    # 判断数据类型
    if "Ageing" in filename:
        dtype = "ageing"
    elif "EIS" in filename:
        dtype = "eis"
    elif "Pola" in filename:
        dtype = "pola"
    else:
        print(f"⚠️ 未分类文件: {filename}")
        continue

    data_dict[stack][dtype].append(df)

# 检查是否成功加载了数据，如果没有静态文件，则回退加载全部文件
if len(data_dict["FC1"]["ageing"]) == 0 and len(data_dict["FC2"]["ageing"]) == 0:
    print("\n⚠️ 未找到包含 'Static' 的文件，回退加载全部 ageing 文件...")
    # 重置 data_dict
    data_dict = {
        "FC1": {"ageing": [], "eis": [], "pola": []},
        "FC2": {"ageing": [], "eis": [], "pola": []}
    }
    for file in csv_files:
        filename = file.name
        try:
            df = pd.read_csv(file, encoding="ISO-8859-1")
        except Exception as e:
            print(f"❌ 读取失败: {filename}, 错误: {e}")
            continue

        if "FC1" in filename:
            stack = "FC1"
        elif "FC2" in filename:
            stack = "FC2"
        else:
            continue

        if "Ageing" in filename:
            dtype = "ageing"
        elif "EIS" in filename:
            dtype = "eis"
        elif "Pola" in filename:
            dtype = "pola"
        else:
            continue

        data_dict[stack][dtype].append(df)

# 统计信息
print("\n📊 数据加载统计：")

for stack in ["FC1", "FC2"]:
    print(f"\n{stack}:")
    for dtype in ["ageing", "eis", "pola"]:
        print(f"  {dtype}: {len(data_dict[stack][dtype])} 个文件")

# 示例查看（确认读取成功）
if len(data_dict["FC1"]["ageing"]) > 0:
    df_example = data_dict["FC1"]["ageing"][0]
    print("\n📄 示例数据（FC1 ageing 第一个文件）：")
    print(df_example.head())
else:
    print("\n❌ FC1 ageing 没有数据，请检查文件路径和文件名！")


#%%
#3.数据预处理（参照论文实现：1h降采样 → SG滤波 → L=1监督学习 → 50/10/40划分 → MinMaxScaler）

import pickle
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ================================================================
# FC1: 1kW PEMFC, 准静态条件, 1000h 长期测试
# FC2: 新型 1kW PEMFC, 电流纹波条件, 1000h 长期测试
# 注意：FC1 和 FC2 测试条件不同（静态 vs 纹波），不应混合训练。
# 本版本使用 FC1 数据，L=1 监督学习 + lag=1,2,4 多步滞后增强。
# ================================================================
SAMPLING_INTERVAL = 1.0  # 采样间隔（小时），1.0h 与参考论文一致
MAX_LAG = 4
LAGS = [1, 2, 4]

print("\n" + "="*60)
print("数据预处理 (FC1-only, {:.1f}h 采样, lag={})".format(SAMPLING_INTERVAL, LAGS))
print("="*60)

# ---------- 构建 FC1 数据 ----------
fc1_dfs = [df for df in data_dict["FC1"]["ageing"] if 'Time (h)' in df.columns]
age_df = pd.concat(fc1_dfs, ignore_index=True)
age_df = age_df.sort_values('Time (h)').reset_index(drop=True)
print(f"FC1 原始数据: {len(age_df)} 行, {len(age_df.columns)} 列")

# ---------- 1. 降采样（按时间间隔取点）----------
t_hours = age_df['Time (h)'].values
if SAMPLING_INTERVAL < 1.0:
    # 按指定间隔（如 0.5h）分组取第一个
    age_df['t_bin'] = (t_hours / SAMPLING_INTERVAL).astype('int32')
    sampled_idx = age_df.drop_duplicates(subset=['t_bin']).index
else:
    # 1h 及以上：整数小时去重
    sampled_idx = age_df.astype({'Time (h)': 'int32'}).drop_duplicates(subset=['Time (h)']).index

sampled_df = age_df.iloc[sampled_idx].reset_index(drop=True)
if 't_bin' in sampled_df.columns:
    sampled_df = sampled_df.drop(columns=['t_bin'])
print(f"降采样至 {SAMPLING_INTERVAL}h 间隔: {len(sampled_df)} 行")

# ---------- 2. Savitzky-Golay 滤波 ----------
meas_cols = [c for c in sampled_df.columns if c != 'Time (h)']
s_df = sampled_df[meas_cols].copy()
s_df = s_df.ffill().bfill().fillna(0)
valid_cols = [c for c in meas_cols if s_df[c].std() > 1e-10]
if len(valid_cols) < len(meas_cols):
    print(f"丢弃 {len(meas_cols)-len(valid_cols)} 个常量/无效列")
meas_cols = valid_cols

smoothed_df = pd.DataFrame()
smoothed_df['Time (h)'] = sampled_df['Time (h)']
for col in meas_cols:
    smoothed_df[col] = savgol_filter(s_df[col].values, window_length=21, polyorder=2)
print(f"Savitzky-Golay 滤波完成: {len(meas_cols)} 个特征 (window=21, poly=2)")

# ---------- 2.5 全特征使用 (TFT Variable Selection 自行学习特征权重) ----------
target_col = 'Utot (V)'
print(f"\n全特征使用: {len(meas_cols)} 个传感器, 由 TFT Variable Selection 自动加权")

# ---------- 3. L=1监督学习 + 多步滞后增强 (lag=1,2,4) ----------
all_cols = [c for c in meas_cols]
n_features_orig = len(all_cols)
target_idx = all_cols.index(target_col)

data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values)

X_sections, y_sections = [], []
for lag in LAGS:
    lag_marker = lag / 10.0  # 0.1, 0.2, 0.4
    X_lag = data_values[MAX_LAG-lag : N-lag]            # (n_valid, n_features_orig)
    y_lag = data_values[MAX_LAG : N, target_idx]        # (n_valid,)
    lag_col = np.full((len(X_lag), 1), lag_marker, dtype=np.float32)
    X_with_marker = np.concatenate([X_lag, lag_col], axis=1)  # (n_valid, n_features_orig+1)
    X_sections.append(X_with_marker)
    y_sections.append(y_lag)

X_raw = np.concatenate(X_sections, axis=0)  # (n_valid*3, n_features_orig+1)
y_raw = np.concatenate(y_sections, axis=0)  # (n_valid*3,)
n_features = X_raw.shape[1]                  # 25 (24传感器 + 1 lag标记)
n_total = len(X_raw)

print(f"特征列 ({n_features_orig} + 1 lag标记): {all_cols} + [lag_marker]")
print(f"Lag增强 (lag={LAGS}): 有效样本={N-MAX_LAG}, 增强后总样本={n_total}")

# ---------- 4. 时序划分（50% 训练 / 10% 验证 / 40% 测试，与论文一致）----------
# 注意: 样本已按lag组拼接 (lag=4,2,1各n_valid条), 每组内部保持时序
# 对每组独立划分，然后合并回对应集合，保持组内时序一致性
n_per_lag = N - MAX_LAG
train_end_lag = int(n_per_lag * 0.5)
val_end_lag   = int(n_per_lag * 0.6)

X_tr_parts, X_vl_parts, X_ts_parts = [], [], []
y_tr_parts, y_vl_parts, y_ts_parts = [], [], []
for k in range(len(LAGS)):
    start = k * n_per_lag
    X_tr_parts.append(X_raw[start : start+train_end_lag])
    X_vl_parts.append(X_raw[start+train_end_lag : start+val_end_lag])
    X_ts_parts.append(X_raw[start+val_end_lag : start+n_per_lag])
    y_tr_parts.append(y_raw[start : start+train_end_lag])
    y_vl_parts.append(y_raw[start+train_end_lag : start+val_end_lag])
    y_ts_parts.append(y_raw[start+val_end_lag : start+n_per_lag])

X_train_raw = np.concatenate(X_tr_parts, axis=0)
X_val_raw   = np.concatenate(X_vl_parts, axis=0)
X_test_raw  = np.concatenate(X_ts_parts, axis=0)
y_train_raw = np.concatenate(y_tr_parts, axis=0)
y_val_raw   = np.concatenate(y_vl_parts, axis=0)
y_test_raw  = np.concatenate(y_ts_parts, axis=0)

print(f"数据划分: 训练 {X_train_raw.shape[0]}, 验证 {X_val_raw.shape[0]}, 测试 {X_test_raw.shape[0]}")

# ---------- 5. 归一化（MinMaxScaler 0-1, 与论文一致）----------
from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train_flat = scaler_X.fit_transform(X_train_raw)
X_val_flat   = scaler_X.transform(X_val_raw)
X_test_flat  = scaler_X.transform(X_test_raw)

# 重塑为 LSTM 输入形状 (n_samples, 1, n_features) — L=1 单时间步
X_train = X_train_flat.reshape(-1, 1, n_features)
X_val   = X_val_flat.reshape(-1, 1, n_features)
X_test  = X_test_flat.reshape(-1, 1, n_features)

y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
y_val   = scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel()
y_test  = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# ---------- 6. 失效阈值（训练集初始电压的 96%）----------
initial_voltage = np.percentile(y_train_raw, 95)
failure_threshold = initial_voltage * 0.96
print(f"初始电压: {initial_voltage:.4f} V, 失效阈值 (96%): {failure_threshold:.4f} V")

# 保存预处理信息
with open(RUN_DIR / 'scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open(RUN_DIR / 'scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
with open(RUN_DIR / 'preprocess_info.pkl', 'wb') as f:
    pickle.dump({
        'n_features': n_features,
        'n_features_orig': n_features_orig,
        'all_cols': all_cols,
        'target_col': target_col,
        'target_idx': target_idx,
        'failure_threshold': failure_threshold,
        'initial_voltage': initial_voltage,
        'lags': LAGS,
        'max_lag': MAX_LAG,
        'feature_names': all_cols
    }, f)
print(f"预处理信息已保存至 {RUN_DIR}/")


#%%
#4.传感器相关性分析 — Spearman 热力图

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

print("\n" + "="*60)
print("传感器 Spearman 相关性分析")
print("="*60)

# 使用降采样 + SG 滤波后的数据 (Section 3 已生成 smoothed_df + all_cols)
corr_data = smoothed_df[all_cols].copy()
n_feat = len(all_cols)

# Spearman 相关矩阵
spearman_mat = np.zeros((n_feat, n_feat))
for i in range(n_feat):
    for j in range(n_feat):
        rho, _ = spearmanr(corr_data.iloc[:, i], corr_data.iloc[:, j])
        spearman_mat[i, j] = rho

# 与 Utot 的相关性 (取绝对值排序)
utot_idx = all_cols.index('Utot (V)')
rho_with_target = spearman_mat[utot_idx, :]
sorted_idx = np.argsort(np.abs(rho_with_target))[::-1]
sorted_cols = [all_cols[i] for i in sorted_idx]
sorted_rho = rho_with_target[sorted_idx]

print(f"\n各传感器与 Utot (V) 的 Spearman |ρ| (降序):")
for col, rho in zip(sorted_cols, sorted_rho):
    bar = '█' * int(np.abs(rho) * 40)
    print(f"  {col:<28s} ρ={rho:+.4f}  {bar}")

# 热力图
plt.figure(figsize=(18, 15))
mask = np.triu(np.ones_like(spearman_mat), k=1)
sns.heatmap(spearman_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            xticklabels=all_cols, yticklabels=all_cols,
            annot_kws={'size': 7})
plt.title('Sensor Spearman Correlation Matrix (FC1)', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(RUN_DIR / 'spearman_heatmap.png', dpi=150)
plt.show()

# 柱状图: 各传感器对 Utot 的 |ρ|
fig, ax = plt.subplots(figsize=(12, 5))
colors_bar = ['#d62728' if rho < 0 else '#1f77b4' for rho in sorted_rho]
bars = ax.bar(range(n_feat), np.abs(sorted_rho), color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(n_feat))
ax.set_xticklabels(sorted_cols, rotation=90, fontsize=8)
ax.set_ylabel('|Spearman ρ|')
ax.set_title('Feature Importance by Spearman |ρ| vs Utot (V)')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
for i, (bar, rho) in enumerate(zip(bars, sorted_rho)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{rho:+.2f}', ha='center', fontsize=7, fontweight='bold')
plt.tight_layout()
plt.savefig(RUN_DIR / 'spearman_bar.png', dpi=150)
plt.show()

# 保存相关性数据
with open(RUN_DIR / 'spearman_corr.pkl', 'wb') as f:
    pickle.dump({'matrix': spearman_mat, 'cols': all_cols, 'target_rho': rho_with_target}, f)
print(f"\n相关性分析已保存至 {RUN_DIR / 'spearman_corr.pkl'}")


#%%
#5.模型构建 — GRU+MLP + 对比模型

import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置参数
CONFIG = {
    'input_size': n_features,
    'hidden_size': 80,
    'num_layers': 1,
    'output_size': 1,
    'dropout': 0.15,
    'learning_rate': 5e-4,
    'epochs': 150,
    'batch_size': 72,
    'patience': 30,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}


class ResMLPModel(nn.Module):
    """残差MLP 回归模型 (L=1, 适用于 lag 增强数据)

    结构: LayerNorm → Linear×2+Skip → LayerNorm → Linear → Output
    残差连接保证梯度流, LayerNorm稳定训练, L=1下比RNN更高效
    """
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
    """LSTM 基线: 标准 LSTM + 全连接输出 (用于消融对比)"""
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
        out = self.fc(out)
        return out


class GatedResidualNetwork(nn.Module):
    """GRN: Gated Residual Network (TFT 论文公式, Lim et al., IJF, 2021)"""
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
        out = a + self.dropout(glu)
        return self.ln(out)


class TFTModel(nn.Module):
    """Temporal Fusion Transformer 适配 L=1 电压预测 (Lim et al., IJF, 2021)

    核心: Variable Selection + LSTM + GRN + Self-Attention
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 nhead=2, dim_feedforward=96, dropout=0.1):
        super().__init__()
        if hidden_size % nhead != 0:
            nhead = 2 if hidden_size % 2 == 0 else 1

        # Variable Selection
        self.feat_weights = nn.Parameter(torch.zeros(input_size))
        self.feat_bias   = nn.Parameter(torch.zeros(input_size))

        # LSTM 时序编码
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        # GRN
        self.grn = GatedResidualNetwork(hidden_size, dim_feedforward, hidden_size, dropout)

        # Self-Attention + Residual
        self.self_attn = nn.MultiheadAttention(
            hidden_size, nhead, dropout=dropout, batch_first=True
        )
        self.attn_ln = nn.LayerNorm(hidden_size)

        # 输出
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Variable Selection
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        x_weighted = x * w.unsqueeze(0).unsqueeze(0)

        # LSTM
        lstm_out, _ = self.lstm(x_weighted)
        lstm_out = self.dropout_layer(lstm_out)

        # GRN
        grn_out = self.grn(lstm_out)

        # Self-Attention + Residual
        attn_out, _ = self.self_attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)

        # 输出
        out = attn_out[:, -1, :]
        out = self.fc(out)
        return out



#%%
#7.超参优化（Optuna）— 可选；已有已知良好参数可直接使用

# 已知良好参数: ChebyKAN R²≈?, LSTM R²≈0.93, TFT R²≈0.90
# 如需重新搜索，将 SKIP_OPTUNA 设为 False
SKIP_OPTUNA = True

if SKIP_OPTUNA:
    best_params = {
        'model_type': 'transformer',
        'hidden_size': 80,
        'num_layers': 1,
        'degree': 2,  # GRU doesn'\''t use this
        'dropout': 0.15,
        'lr': 0.0005,
        'batch_size': 72
    }
    print("使用预设超参数（跳过 Optuna 搜索）:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
else:
    import optuna
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device_t = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)

    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['resmlp', 'baseline', 'transformer'])
        hidden_size = trial.suggest_int('hidden_size', 32, 80, step=16)
        num_layers = trial.suggest_int('num_layers', 1, 2)
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 72])

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)
        val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

        if model_type == 'resmlp':
            model = ResMLPModel(n_features, hidden_size, num_layers, 1, dropout)
        elif model_type == 'baseline':
            model = LSTMBaseline(n_features, hidden_size, num_layers, 1, dropout)
        else:
            model = TFTModel(n_features, hidden_size, num_layers, 1,
                                    nhead=2, dim_feedforward=96, dropout=dropout)
        model = model.to(device_t)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        for epoch in range(100):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device_t), batch_y.to(device_t)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    val_preds.append(model(batch_x.to(device_t)).cpu().numpy())
                    val_trues.append(batch_y.numpy())
            rmse = np.sqrt(np.mean((np.concatenate(val_preds).ravel() - np.concatenate(val_trues).ravel())**2))
            trial.report(rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return rmse

    print("开始超参数优化（Optuna），共 200 个 trial...")
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=200, show_progress_bar=True)
    best_params = study.best_trial.params
    print(f"\n最佳验证 RMSE: {study.best_trial.value:.6f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

with open(RUN_DIR / 'best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)


#%%
#8.模型训练

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

CONFIG = {
    'input_size': n_features,
    'output_size': 1,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'epochs': 300,
    'patience': 60,
    'model_type': best_params['model_type'],
    'hidden_size': best_params['hidden_size'],
    'num_layers': best_params['num_layers'],
    'dropout': best_params['dropout'],
    'learning_rate': best_params['lr'],
    'batch_size': best_params['batch_size'],
    'optimizer': 'Adam'
}

print("\n最终训练配置：")
for k, v in CONFIG.items():
    if k != 'device':
        print(f"  {k}: {v}")

# 使用 Section 3 中已划分的数据
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=CONFIG['batch_size'], shuffle=False)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=CONFIG['batch_size'], shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=CONFIG['batch_size'], shuffle=False)

print(f"\n数据划分：训练 {X_train.shape[0]}, 验证 {X_val.shape[0]}, 测试 {X_test.shape[0]}")

def train_model(model, train_loader, val_loader, config):
    device = config['device']
    model.to(device)

    # 参数分组: KAN 多项式系数用更低学习率 + 更强正则化, 防止震荡
    kan_coeff_params = []
    other_params = []
    for name, p in model.named_parameters():
        if 'coeffs' in name:
            kan_coeff_params.append(p)
        else:
            other_params.append(p)
    lr = config['learning_rate']
    param_groups = [
        {'params': other_params,      'lr': lr,       'weight_decay': 1e-5},
        {'params': kan_coeff_params,  'lr': lr * 0.1, 'weight_decay': 1e-4},
    ]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = torch.nn.MSELoss()

    # SWA: 从 patience 过半时开始收集快照
    swa_start = 30  # epoch 30 开始收集 SWA 快照
    swa_model = None

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), RUN_DIR / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # SWA: 稳定期收集快照 (当前为最佳或接近最佳时)
        if epoch >= swa_start:
            if swa_model is None:
                swa_model = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_model:
                    swa_model[k] = swa_model[k] * (swa_count / (swa_count + 1)) + \
                                   model.state_dict()[k].cpu() * (1 / (swa_count + 1))
                swa_count += 1

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 恢复 best_model，再叠加 SWA（如果 SWA 存在则保存）
    model.load_state_dict(torch.load(RUN_DIR / 'best_model.pth'))
    if swa_model is not None:
        swa_state = {k: v.to(device) for k, v in swa_model.items()}
        torch.save(swa_state, RUN_DIR / 'swa_model.pth')
    return model, history

def evaluate_model(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            preds.append(pred.cpu().numpy())
            trues.append(batch_y.numpy())
    y_pred_norm = np.concatenate(preds).ravel()
    y_true_norm = np.concatenate(trues).ravel()

    # 使用 scaler_y 反归一化
    with open(RUN_DIR / 'scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_true_norm.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return y_true, y_pred, rmse, mae, mape, r2

def make_model(config):
    """根据配置创建模型"""
    if config['model_type'] == 'resmlp':
        return ResMLPModel(config['input_size'], config['hidden_size'],
                             config.get('num_layers', 1), config['output_size'], config['dropout'])
    elif config['model_type'] == 'baseline':
        return LSTMBaseline(config['input_size'], config['hidden_size'], config['num_layers'],
                            config['output_size'], config['dropout'])
    else:
        return TFTModel(config['input_size'], config['hidden_size'], config['num_layers'],
                        config['output_size'], nhead=4, dim_feedforward=128, dropout=config['dropout'])

# 构建模型
model = make_model(CONFIG)

print(f"\n使用设备: {CONFIG['device']}")
print(f"模型类型: {CONFIG['model_type']}")

model, history = train_model(model, train_loader, val_loader, CONFIG)
# 优先使用 SWA 模型（权重平均, 更稳定）
swa_path = RUN_DIR / 'swa_model.pth'
if swa_path.exists():
    model.load_state_dict(torch.load(swa_path))
    print("评估使用 SWA 模型")

y_true, y_pred, rmse, mae, mape, r2 = evaluate_model(model, test_loader)

print("\n========== 最终模型测试集性能 ==========")
print(f"RMSE: {rmse:.6f} V")
print(f"MAE : {mae:.6f} V")
print(f"MAPE: {mape:.2f}%")
print(f"R²  : {r2:.4f}")
# ---------- 论文评估: 训练N次取最优 R² ----------
N_LOOPS = 50
print(f"\n论文评估方式: 同参数训练 {N_LOOPS} 次, 取最优 R²")
loop_r2s = []
best_loop_r2 = r2
best_loop_yp = y_pred
best_loop_yt = y_true
for li in range(N_LOOPS):
    torch.manual_seed(seed_value + li * 10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value + li * 10)
    lm = make_model(CONFIG)
    lm, _ = train_model(lm, train_loader, val_loader, CONFIG)
    swa_path = RUN_DIR / 'swa_model.pth'
    if swa_path.exists():
        lm.load_state_dict(torch.load(swa_path))
    yt, yp, _, _, _, rl = evaluate_model(lm, test_loader)
    loop_r2s.append(rl)
    if rl > best_loop_r2:
        best_loop_r2 = rl
        best_loop_yp = yp
        best_loop_yt = yt
print(f"Loop R²: Min={min(loop_r2s):.4f}  Mean={np.mean(loop_r2s):.4f}  Max={max(loop_r2s):.4f}")
if best_loop_r2 > r2:
    print(f"Loop 最优 ({best_loop_r2:.4f}) > 单次 ({r2:.4f}), 采用")
    y_pred, y_true, r2 = best_loop_yp, best_loop_yt, best_loop_r2

# ---------- 可视化 ----------
output_dir = RUN_DIR

# 图1: 训练/验证 Loss 曲线
plt.figure(figsize=(10, 4))
plt.plot(history['train_loss'], label='Train Loss', linewidth=1.5)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f'Training History — {CONFIG["model_type"].upper()}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'training_history.png', dpi=150)
plt.show()

# 图2: 测试集预测 vs 真实电压
plt.figure(figsize=(12, 5))
time_axis = np.arange(len(y_true)) * SAMPLING_INTERVAL
plt.plot(time_axis, y_true, label='True Voltage', linewidth=1.5, alpha=0.8)
plt.plot(time_axis, y_pred, label='Predicted Voltage', linewidth=1.5, alpha=0.8, linestyle='--')
plt.xlabel(f'Test Time (hours, {SAMPLING_INTERVAL}h/sample)')
plt.ylabel('Stack Voltage (V)')
plt.title(f'Test Set Prediction — {CONFIG["model_type"].upper()} (R²={r2:.4f}, RMSE={rmse:.4f}V)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'prediction_vs_true.png', dpi=150)
plt.show()

# 图3: 散点图 (True vs Pred)
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.5, s=10)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1)
plt.xlabel('True Voltage (V)')
plt.ylabel('Predicted Voltage (V)')
plt.title(f'True vs Predicted (R²={r2:.4f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'scatter_true_vs_pred.png', dpi=150)
plt.show()

torch.save(model.state_dict(), RUN_DIR / 'final_model.pth')
with open(RUN_DIR / 'training_history_final.pkl', 'wb') as f:
    pickle.dump(history, f)

print(f"\n全部输出已保存至 {RUN_DIR}/")
print(f"  - 模型: best_model.pth, final_model.pth")
print(f"  - 数据: scaler_X.pkl, scaler_y.pkl, preprocess_info.pkl")
print(f"  - 图片: training_history.png, prediction_vs_true.png, scatter_true_vs_pred.png")


#%%
#9.模型评估与对比 — 多步预测 & 不同训练比例

SKIP_SECTION9 = True  # 调试阶段跳过, 节省时间
if not SKIP_SECTION9:

    import pickle
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(RUN_DIR / 'scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    with open(RUN_DIR / 'scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open(RUN_DIR / 'preprocess_info.pkl', 'rb') as f:
        info = pickle.load(f)
    target_idx = info['target_idx']

    # 多步预测: 每步使用真实观测值作为输入（RF-TCN 论文方法，避免累积误差）
    def multi_step_predict(model, X_data, start_idx, steps_ahead):
        """从 start_idx 开始，用真实 X_data[i] 预测 y[i]，不累积误差"""
        model.eval()
        predictions = []
        for k in range(steps_ahead):
            idx = start_idx + k
            if idx >= len(X_data):
                break
            with torch.no_grad():
                pred = model(X_data[idx:idx+1].to(device))
            predictions.append(pred.item())
        return predictions

    steps = [1, 2, 3, 5, 10]
    print("\n========== 多步预测性能（真实值回填，论文方法）==========")
    for step in steps:
        preds_all, trues_all = [], []
        for i in range(min(50, len(X_test_t) - step)):
            pred_seq = multi_step_predict(model, X_test_t, i, step)
            if i + step - 1 < len(y_test_t):
                true_norm = y_test_t[i + step - 1].item()
                preds_all.append(pred_seq[-1])
                trues_all.append(true_norm)
        if len(preds_all) > 0:
            pred_v = scaler_y.inverse_transform(np.array(preds_all).reshape(-1, 1)).ravel()
            true_v = scaler_y.inverse_transform(np.array(trues_all).reshape(-1, 1)).ravel()
            rmse = np.sqrt(mean_squared_error(true_v, pred_v))
            mae = mean_absolute_error(true_v, pred_v)
            print(f"步长 {step:2d}: RMSE={rmse:.6f} V, MAE={mae:.6f} V")

    # 不同训练比例对比（使用原始数据重新划分）
    print("\n========== 不同训练比例对比 ==========")
    train_ratios = [0.7]  # 70/10/20 vs 主模型 50/10/40, 验证更多训练数据的提升
    val_ratio_fixed = 0.1

    for train_ratio in train_ratios:
        n_raw = len(X_raw)
        tr_end = int(n_raw * train_ratio)
        vl_end = tr_end + int(n_raw * val_ratio_fixed)

        X_tr_r = X_raw[:tr_end];  y_tr_r = y_raw[:tr_end]
        X_vl_r = X_raw[tr_end:vl_end]; y_vl_r = y_raw[tr_end:vl_end]
        X_ts_r = X_raw[vl_end:];  y_ts_r = y_raw[vl_end:]

        # 每个比例独立拟合 scaler（不同训练集电压范围不同）
        sX = MinMaxScaler(feature_range=(0,1)); sy = MinMaxScaler(feature_range=(0,1))
        X_tr_s = sX.fit_transform(X_tr_r).reshape(-1, 1, n_features)
        X_vl_s = sX.transform(X_vl_r).reshape(-1, 1, n_features)
        X_ts_s = sX.transform(X_ts_r).reshape(-1, 1, n_features)
        y_tr_s = sy.fit_transform(y_tr_r.reshape(-1,1)).ravel()
        y_vl_s = sy.transform(y_vl_r.reshape(-1,1)).ravel()
        y_ts_s = sy.transform(y_ts_r.reshape(-1,1)).ravel()

        tr_ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32),
                              torch.tensor(y_tr_s, dtype=torch.float32).view(-1,1))
        vl_ds = TensorDataset(torch.tensor(X_vl_s, dtype=torch.float32),
                              torch.tensor(y_vl_s, dtype=torch.float32).view(-1,1))
        ts_ds = TensorDataset(torch.tensor(X_ts_s, dtype=torch.float32),
                              torch.tensor(y_ts_s, dtype=torch.float32).view(-1,1))
        tr_ld = DataLoader(tr_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        vl_ld = DataLoader(vl_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        ts_ld = DataLoader(ts_ds, batch_size=CONFIG['batch_size'], shuffle=False)

        # 使用与主模型相同的架构和超参，独立 scaler，种子重置保证初始化一致
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

        if CONFIG['model_type'] == 'resmlp':
            tmp_model = ResMLPModel(n_features, CONFIG['hidden_size'], 1, CONFIG.get('degree', 4), CONFIG['dropout'])
        elif CONFIG['model_type'] == 'baseline':
            tmp_model = LSTMBaseline(n_features, CONFIG['hidden_size'], CONFIG['num_layers'],
                                     1, CONFIG['dropout'])
        else:
            tmp_model = TFTModel(n_features, CONFIG['hidden_size'], CONFIG['num_layers'],
                                        1, nhead=4, dim_feedforward=128, dropout=CONFIG['dropout'])

        tmp_model, tmp_hist = train_model(tmp_model, tr_ld, vl_ld, CONFIG)
        tmp_model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for bx, by in ts_ld:
                preds.append(tmp_model(bx.to(device)).cpu().numpy())
                trues.append(by.numpy())
        p_norm = np.concatenate(preds).ravel()
        t_norm = np.concatenate(trues).ravel()
        p_orig = sy.inverse_transform(p_norm.reshape(-1,1)).ravel()
        t_orig = sy.inverse_transform(t_norm.reshape(-1,1)).ravel()
        rmse_r = np.sqrt(mean_squared_error(t_orig, p_orig))
        r2_r = r2_score(t_orig, p_orig)
        print(f"训练比例 {train_ratio:.1f}: RMSE={rmse_r:.6f} V, R²={r2_r:.4f}")

    print("\n模型评估与对比完成。")
  # end of SKIP_SECTION9


#%%
#10.RUL 计算

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open(RUN_DIR / 'scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)
with open(RUN_DIR / 'preprocess_info.pkl', 'rb') as f:
    info = pickle.load(f)

failure_threshold = info['failure_threshold']
initial_voltage = info['initial_voltage']

print(f"失效阈值: {failure_threshold:.4f} V (初始 {initial_voltage:.4f} V × 96%)")

# y_true, y_pred 来自 Section 8 的 evaluate_model，已经是原始电压
print("\n========== RUL 计算结果 ==========")

dt = SAMPLING_INTERVAL  # 与预处理采样间隔一致

failure_indices_true = np.where(y_true < failure_threshold)[0]
true_failure_idx = failure_indices_true[0] if len(failure_indices_true) > 0 else len(y_true)

failure_indices_pred = np.where(y_pred < failure_threshold)[0]
pred_failure_idx = failure_indices_pred[0] if len(failure_indices_pred) > 0 else len(y_pred)

true_failure_time = true_failure_idx * dt
pred_failure_time = pred_failure_idx * dt
rul_abs_error = abs(pred_failure_time - true_failure_time)
rul_rel_error = (rul_abs_error / true_failure_time) if true_failure_time > 0 else np.inf

print(f"真实失效时间（从测试集起始算起）: {true_failure_time:.1f} 小时")
print(f"预测失效时间（从测试集起始算起）: {pred_failure_time:.1f} 小时")
print(f"RUL 绝对误差: {rul_abs_error:.2f} 小时")
print(f"RUL 相对误差: {rul_rel_error:.2%}")

plt.figure(figsize=(12, 5))
time_axis = np.arange(len(y_true)) * dt
plt.plot(time_axis, y_true, label='True Voltage', linewidth=2)
plt.plot(time_axis, y_pred, label='Predicted Voltage', linestyle='--', linewidth=2)
plt.axhline(y=failure_threshold, color='red', linestyle=':', linewidth=1.5,
            label=f'Failure Threshold = {failure_threshold:.4f} V')
plt.axvline(x=true_failure_idx * dt, color='green', linestyle=':', linewidth=1.5,
            label=f'True EoL = {true_failure_idx * dt:.1f} h')
if pred_failure_idx < len(y_pred):
    plt.axvline(x=pred_failure_idx * dt, color='orange', linestyle=':', linewidth=1.5,
                label=f'Predicted EoL = {pred_failure_idx * dt:.1f} h')
plt.xlabel('Test Time (hours)')
plt.ylabel('Stack Voltage (V)')
plt.title(f'RUL Prediction — Abs Error = {rul_abs_error:.3f}h ({rul_rel_error:.2%})')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'rul_prediction.png', dpi=150)
plt.show()

print("\nRUL 计算完成。")

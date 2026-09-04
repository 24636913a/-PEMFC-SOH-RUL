## 项目说明
#PEMFC RUL Prediction (PyTorch)

## 研究目标
#基于 IEEE 2014 PHM 数据集，构建 LSTM + Transformer + PINN模型，实现燃料电池寿命预测（RUL）。

## 代码框架
#1.环境配置
#2.数据加载
#3.数据预处理
#   - 重采样
#   - 特征工程（Spearman / Pearson / ACF，特征筛选）
#   - 平滑
#   - 归一化
#   - 序列构造
#4.降维（PCA）
#5.模型构建
#   - LSTM（baseline）
#   - LSTM + Transformer
#6.物理约束建模（PINN）
#   - 单调性约束
#   - 平滑约束
#7.超参优化（Optuna）
#8.模型训练
#9.模型评估与对比
#   - MAE / MSE / R²
#   - 单步 vs 多步
#   - 不同训练比例
#10. RUL计算


#%%
#1.环境配置

#导入库
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
import os, random
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

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
    "data_path": Path(r"E:\作业\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")
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

CONFIG["output_path"].mkdir(exist_ok=True)
import pickle
with open(CONFIG["output_path"]/"config.pkl", "wb") as f:
    pickle.dump(CONFIG, f)
print(f"Configuration saved to {CONFIG['output_path']/ 'config.pkl'}")

#保存配置
config_file = CONFIG["output_path"] / "config.pkl"
with open(config_file, "wb") as f:
    pickle.dump(CONFIG, f)
print(f"✓ Configuration saved to {config_file}\n")

print("="*60)
print("✅ 环境配置完成")
print("="*60)


#%%
#2.数据加载

import pandas as pd
from pathlib import Path

# 配置路径
DATA_PATH = Path(r"E:\作业\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData") # 💡💡💡

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
#3.数据预处理
#   - 重采样
#   - 特征工程（Spearman / Pearson / ACF，特征筛选）
#   - 平滑
#   - 归一化
#   - 序列构造

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt 

# 从 data_dict 构建 age_df
# 以 FC1 为例，将所有 ageing 文件按时间拼接
df_list = []
for df in data_dict["FC1"]["ageing"]:
    if 'Time (h)' in df.columns:
        df_list.append(df)
    else:
        print("警告：缺少 'Time (h)' 列，跳过该文件")

if df_list:
    age_df = pd.concat(df_list, ignore_index=True)
    age_df = age_df.sort_values('Time (h)').reset_index(drop=True)
    print(f"✓ 构建 FC1 ageing 数据，共 {len(age_df)} 行")
    print("列名：", age_df.columns.tolist())
else:
    raise ValueError("没有找到有效的 ageing 数据")

# 重采样、平滑、归一化、滑动窗口
# 注意：请根据实际列名修改下面的 target_col 和 feature_cols
target_col = 'Utot (V)'          # 你的健康指标（电压或 SOH）💡💡💡
feature_cols = ['I (A)',
    'TinAIR (°C)',
    'ToutWAT (°C)',
    'PinH2 (mbara)',
    'DinH2 (l/mn)',
    'DinAIR (l/mn)',
    'HrAIRFC (%)']  # 根据实际调整💡💡💡

# 设置时间索引并重采样
age_df['Time'] = pd.to_datetime(age_df['Time (h)'], unit='h')
age_df = age_df.set_index('Time').sort_index()
age_df = age_df[~age_df.index.duplicated(keep='first')]
age_resampled = age_df.resample('0.5h').ffill()  # 每0.5小时采样 💡💡💡

# ---------- 先平滑目标电压，再做特征工程 ----------
print("\n--- 平滑目标电压 ---")
raw = age_resampled[target_col].values
smooth = np.copy(raw)
window = 15
for i in range(len(raw)):
    start = max(0, i - window)
    smooth[i] = np.mean(raw[start:i+1])
age_resampled[f'{target_col}_smooth'] = smooth
target_smooth_col = f'{target_col}_smooth'          # 用于后续相关性分析的目标列

# ---------- 特征工程（基于平滑后的电压）----------
print("\n--- 开始特征工程（基于平滑后的电压）---")
feat_cols_raw = [col for col in feature_cols if col in age_resampled.columns]
df_corr = age_resampled[feat_cols_raw + [target_smooth_col]].dropna()

# 计算 Pearson 相关系数（与平滑电压）
pearson_dict = {}
for col in feat_cols_raw:
    r_pear, _ = pearsonr(df_corr[col], df_corr[target_smooth_col])
    pearson_dict[col] = r_pear
pearson_series = pd.Series(pearson_dict).sort_values(ascending=False)
print("\n=== 与平滑电压的 Pearson 相关系数 ===")
print(pearson_series.round(4))

# 计算 ACF 并画图（基于平滑电压）
utot_series = df_corr[target_smooth_col].values
max_lag = 50
acf_vals = acf(utot_series, nlags=max_lag, fft=False)
plt.figure(figsize=(10,4))
plt.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
plt.title(f'Autocorrelation of {target_smooth_col} (resampled & smoothed)')
plt.xlabel('Lag (hours)')
plt.ylabel('ACF')
plt.grid(True)
plt.show()
print("\n=== 自相关系数（前20阶）===")
for lag in range(1, 21):
    print(f"Lag {lag:2d}: {acf_vals[lag]:.4f}")

# 根据 ACF 建议序列长度
threshold_acf = 0.3
lags_below = np.where(acf_vals[1:] < threshold_acf)[0]
if len(lags_below) > 0:
    suggested_L = lags_below[0] + 1
    print(f"\n基于 ACF 阈值 {threshold_acf}，建议序列长度 L = {suggested_L}")
else:
    suggested_L = 20
    print(f"\n所有滞后自相关系数均 >= {threshold_acf}，建议使用默认 L=20")

# 特征筛选（Top‑K 基于 Pearson 绝对值）
top_k = 5
selected_by_target = pearson_series.abs().sort_values(ascending=False).head(top_k).index.tolist()
print(f"\n基于 Pearson 绝对值选出的 Top-{top_k} 特征: {selected_by_target}")

# 去除冗余特征（互相关 > 0.9）
if len(selected_by_target) > 1:
    sub_df = df_corr[selected_by_target]
    corr_matrix = sub_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high_corr_cols = upper.index[upper[col] > 0.9].tolist()
        for high_col in high_corr_cols:
            if pearson_series[col] < pearson_series[high_col]:
                to_drop.add(col)
            else:
                to_drop.add(high_col)
    final_features = [col for col in selected_by_target if col not in to_drop]
else:
    final_features = selected_by_target

print(f"\n最终推荐的特征列（去除冗余后）: {final_features}")

# 使用筛选后的特征作为后续的 feature_cols
feature_cols = final_features
print(f"将使用筛选后的特征列: {feature_cols}")

# 序列长度
L = suggested_L   # 启用 ACF 建议的序列长度
print(f"序列长度 L = {L}")

# ---------- 构建最终特征集 ----------
existing_features = [col for col in feature_cols if col in age_resampled.columns]
existing_features.append(target_smooth_col)
existing_features = list(dict.fromkeys(existing_features))
print(f"使用的最终特征列: {existing_features}")

features = age_resampled[existing_features].values

# 先划分再归一化（防止数据泄露）
n_total = len(features)
train_end = int(n_total * 0.7)
val_end = int(n_total * 0.85)

train_raw = features[:train_end]
val_raw   = features[train_end:val_end]
test_raw  = features[val_end:]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_raw)
val_scaled   = scaler.transform(val_raw)
test_scaled  = scaler.transform(test_raw)
features_scaled = np.vstack([train_scaled, val_scaled, test_scaled])

# ---------- 计算平滑电压失效阈值（基于训练集平滑电压，单位：V）----------
smooth_idx = existing_features.index(target_smooth_col)
train_smooth_voltage = train_raw[:, smooth_idx]          # 原始平滑电压值
initial_smooth_voltage = np.percentile(train_smooth_voltage, 95)
failure_threshold_smooth = initial_smooth_voltage * 0.96   # 4% 衰减
print(f"平滑电压初始值: {initial_smooth_voltage:.4f} V，失效阈值: {failure_threshold_smooth:.4f} V")

# 构造滑动窗口序列
X, y = [], []
for i in range(len(features_scaled) - L):
    X.append(features_scaled[i:i+L])
    y.append(features_scaled[i+L, smooth_idx])
X = np.array(X)
y = np.array(y)

print(f"输入形状: {X.shape}")
print(f"输出形状: {y.shape}")

# 保存预处理信息
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('preprocess_info.pkl', 'wb') as f:
    pickle.dump({
        'existing_features': existing_features,
        'target_col': target_col,
        'failure_threshold_smooth': failure_threshold_smooth,
        'initial_smooth_voltage': initial_smooth_voltage
    }, f)
print("预处理信息（scaler, existing_features, target_col, 失效阈值）已保存。")


#%%
#4.降维（PCA）
#from sklearn.decomposition import PCA

#pca = PCA(n_components=0.95)  # 保留 95% 方差
#principal_components = pca.fit_transform(df_selected_norm)

#print("\nPCA 主成分数：", pca.n_components_)
#print("各成分解释的方差比：", pca.explained_variance_ratio_)


#%%
#5.模型构建
#   - LSTM（baseline）
#   - LSTM + Transformer

import torch
import torch.nn as nn

# 配置参数（可在此处修改）
CONFIG = {
    'input_size': X.shape[2],          # 特征数
    'hidden_size': 128,
    'num_layers': 2,
    'output_size': 1,
    'dropout': 0.2,
    'learning_rate': 5e-4,
    'epochs': 300,
    'batch_size': 32,
    'patience': 50,
    'lambda_monotonic': 1.0,
    'lambda_smooth': 0.1,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}    #💡💡💡

# 1) LSTM Baseline
class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 2) LSTM + Transformer（修复 nhead 整除）
class LSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 nhead=4, dim_feedforward=128, dropout=0.2):
        super().__init__()
        # 保证 nhead 能整除 hidden_size
        if hidden_size % nhead != 0:
            nhead = 2 if hidden_size % 2 == 0 else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        trans_out = self.transformer(lstm_out)
        out = trans_out[:, -1, :]
        out = self.fc(out)
        return out

# 3) PINN LSTM (物理约束通过同一预测层施加，避免断层)
class PINNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_main = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        main_out = self.fc_main(last_step)
        # 辅助输出用于物理约束，共享同一预测层
        aux_out = self.fc_main(lstm_out)
        return main_out, aux_out

print("模型定义完成。可用类: LSTMBaseline, LSTMTransformer, PINNLSTM")


#%%
#6.物理约束建模（PINN）
#   - 单调性约束
#   - 平滑约束

import torch

def compute_physical_loss(reconstructed_seq, lambda_mono, lambda_smooth):
    seq = reconstructed_seq.squeeze(-1)
    diff = seq[:, 1:] - seq[:, :-1]
    monotonic_penalty = torch.mean(torch.relu(diff + 1e-4))
    diff2 = diff[:, 1:] - diff[:, :-1]
    smooth_penalty = torch.mean(torch.abs(diff2))
    return monotonic_penalty, smooth_penalty

print("物理约束损失函数定义完成。")


#%%
#7.超参优化（Optuna）

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

n_total = len(X)
train_ratio, val_ratio = 0.7, 0.15
train_end = int(n_total * train_ratio)
val_end = train_end + int(n_total * val_ratio)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG = {
    'input_size': X.shape[2],
    'output_size': 1,
    'device': device,
    'batch_size': 32
}

def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['baseline', 'transformer', 'pinn'])
    hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    
    lambda_mono = None
    lambda_smooth = None
    if model_type == 'pinn':
        lambda_mono = trial.suggest_float('lambda_monotonic', 0.01, 0.5, log=True)
        lambda_smooth = trial.suggest_float('lambda_smooth', 0.001, 0.1, log=True)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    
    if model_type == 'baseline':
        model = LSTMBaseline(CONFIG['input_size'], hidden_size, num_layers, CONFIG['output_size'], dropout)
        use_physics = False
    elif model_type == 'transformer':
        model = LSTMTransformer(CONFIG['input_size'], hidden_size, num_layers, CONFIG['output_size'])
        use_physics = False
    else:
        model = PINNLSTM(CONFIG['input_size'], hidden_size, num_layers, CONFIG['output_size'], dropout)
        use_physics = True
    
    model = model.to(device)
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    
    criterion = nn.MSELoss()
    n_epochs = 50  # 增加每个 trial 的训练轮数，提升搜索质量
    
    for epoch in range(n_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            if use_physics:
                main_pred, aux_pred = model(batch_x)
                mse_loss = criterion(main_pred, batch_y)
                mono_pen, smooth_pen = compute_physical_loss(aux_pred, lambda_mono, lambda_smooth)
                loss = mse_loss + lambda_mono * mono_pen + lambda_smooth * smooth_pen
            else:
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                if use_physics:
                    pred, _ = model(batch_x)
                else:
                    pred = model(batch_x)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(batch_y.numpy())
        val_pred = np.concatenate(val_preds).ravel()
        val_true = np.concatenate(val_trues).ravel()
        rmse = np.sqrt(np.mean((val_true - val_pred)**2))
        
        trial.report(rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return rmse

print("开始超参数优化（Optuna），共 30 个 trial...")
study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\n========== 最佳超参数 ==========")
best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")
print(f"最佳验证 RMSE: {best_trial.value:.6f}")

with open('best_params.pkl', 'wb') as f:
    pickle.dump(best_trial.params, f)
print("最佳参数已保存到 best_params.pkl")


#%%
#8.模型训练

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

with open('best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

print("加载的最佳超参数：")
for k, v in best_params.items():
    print(f"  {k}: {v}")

try:
    CONFIG
except NameError:
    CONFIG = {}

CONFIG['input_size'] = X.shape[2]
CONFIG['output_size'] = 1
CONFIG['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG['epochs'] = 500          # 延长训练，配合早停
CONFIG['patience'] = 30         # 稍微增加耐心

CONFIG['model_type'] = best_params['model_type']
CONFIG['hidden_size'] = best_params['hidden_size']
CONFIG['num_layers'] = best_params['num_layers']
CONFIG['dropout'] = best_params['dropout']
CONFIG['learning_rate'] = best_params['lr']
CONFIG['batch_size'] = best_params['batch_size']
CONFIG['optimizer'] = best_params['optimizer']

if CONFIG['model_type'] == 'pinn':
    CONFIG['lambda_monotonic'] = best_params.get('lambda_monotonic', 0.1)
    CONFIG['lambda_smooth'] = best_params.get('lambda_smooth', 0.01)
else:
    CONFIG['lambda_monotonic'] = 0.0
    CONFIG['lambda_smooth'] = 0.0

print("\n最终训练配置：")
for k, v in CONFIG.items():
    if k != 'device':
        print(f"  {k}: {v}")

# 数据划分
n_total = len(X)
train_ratio, val_ratio = 0.7, 0.15
train_end = int(n_total * train_ratio)
val_end = train_end + int(n_total * val_ratio)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=CONFIG['batch_size'], shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=CONFIG['batch_size'], shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=CONFIG['batch_size'], shuffle=False)

print(f"\n数据划分：训练 {X_train.shape[0]}, 验证 {X_val.shape[0]}, 测试 {X_test.shape[0]}")

# 训练函数
def train_model(model, train_loader, val_loader, config, use_physics=False):
    device = config['device']
    model.to(device)

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()          # MSE 与 RMSE 优化目标一致

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    if use_physics:
        history['monotonic'] = []
        history['smooth'] = []

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        if use_physics:
            epoch_mono = 0.0
            epoch_smooth = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            if use_physics:
                main_pred, aux_pred = model(batch_x)
                mse_loss = criterion(main_pred, batch_y)
                mono_pen, smooth_pen = compute_physical_loss(
                    aux_pred, config['lambda_monotonic'], config['lambda_smooth']
                )
                loss = mse_loss + config['lambda_monotonic'] * mono_pen + config['lambda_smooth'] * smooth_pen
                epoch_mono += mono_pen.item()
                epoch_smooth += smooth_pen.item()
            else:
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
                if use_physics:
                    main_pred, _ = model(batch_x)
                    loss = criterion(main_pred, batch_y)
                else:
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        if use_physics:
            history['monotonic'].append(epoch_mono / len(train_loader))
            history['smooth'].append(epoch_smooth / len(train_loader))

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            if use_physics:
                print(f"  Monotonic: {history['monotonic'][-1]:.6f} | Smooth: {history['smooth'][-1]:.6f}")

    model.load_state_dict(torch.load('best_model.pth'))
    return model, history

# 评估函数（输出原始尺度）
def evaluate_model(model, test_loader, use_physics=False):
    device = next(model.parameters()).device
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            if use_physics:
                main_pred, _ = model(batch_x)
            else:
                main_pred = model(batch_x)
            preds.append(main_pred.cpu().numpy())
            trues.append(batch_y.numpy())
    y_pred_norm = np.concatenate(preds).ravel()
    y_true_norm = np.concatenate(trues).ravel()
    
    # 反标准化到原始电压
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('preprocess_info.pkl', 'rb') as f:
        info = pickle.load(f)
    smooth_col = f'{info["target_col"]}_smooth'
    voltage_idx = info['existing_features'].index(smooth_col)
    v_mean = scaler.mean_[voltage_idx]
    v_std = scaler.scale_[voltage_idx]
    
    y_pred = y_pred_norm * v_std + v_mean
    y_true = y_true_norm * v_std + v_mean
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return y_true, y_pred, rmse, mae, mape, r2

# 构建模型
use_physics = (CONFIG['model_type'] == 'pinn')
if CONFIG['model_type'] == 'baseline':
    model = LSTMBaseline(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_layers'],
                         CONFIG['output_size'], CONFIG['dropout'])
elif CONFIG['model_type'] == 'transformer':
    model = LSTMTransformer(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_layers'],
                            CONFIG['output_size'], nhead=4, dim_feedforward=128, dropout=CONFIG['dropout'])
else:
    model = PINNLSTM(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_layers'],
                     CONFIG['output_size'], CONFIG['dropout'])

print(f"\n使用设备: {CONFIG['device']}")
print(f"模型类型: {CONFIG['model_type']}, 使用物理约束: {use_physics}")

model, history = train_model(model, train_loader, val_loader, CONFIG, use_physics)

y_true, y_pred, rmse, mae, mape, r2 = evaluate_model(model, test_loader, use_physics)

print("\n========== 最终模型测试集性能 ==========")
print(f"RMSE: {rmse:.6f} V")
print(f"MAE : {mae:.6f} V")
print(f"MAPE: {mape:.2f}%")
print(f"R²  : {r2:.4f}")

plt.figure(figsize=(10,5))
plt.plot(y_true, label='True Voltage', alpha=0.7)
plt.plot(y_pred, label='Predicted Voltage', alpha=0.7)
plt.legend()
plt.title(f'{CONFIG["model_type"].upper()} Voltage Prediction on Test Set')
plt.xlabel('Sample Index')
plt.ylabel('Stack Voltage (V)')
plt.grid(True)
plt.show()

torch.save(model.state_dict(), 'final_model.pth')
with open('training_history_final.pkl', 'wb') as f:
    pickle.dump(history, f)

print("\n最终模型已保存为 final_model.pth，训练历史保存为 training_history_final.pkl")


#%%
#9.模型评估与对比
#   - MAE / MSE / R²
#   - 单步 vs 多步
#   - 不同训练比例

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq_len = X.shape[1]
n_features = X.shape[2]

with open('preprocess_info.pkl', 'rb') as f:
    info = pickle.load(f)
existing_features = info['existing_features']
target_col = info['target_col']
voltage_col_name = f'{target_col}_smooth'
voltage_idx = existing_features.index(voltage_col_name)

# 加载 standard 偏差用于多步预测单位还原
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
voltage_std = scaler.scale_[voltage_idx]

def multi_step_predict(model, initial_seq, steps_ahead, use_physics=False, voltage_idx=-1):
    model.eval()
    current_seq = initial_seq.clone().unsqueeze(0).to(device)
    predictions = []
    for _ in range(steps_ahead):
        with torch.no_grad():
            if use_physics:
                pred, _ = model(current_seq)
            else:
                pred = model(current_seq)
        next_val = pred.item()
        predictions.append(next_val)
        new_step = current_seq[0, -1, :].clone()
        new_step[voltage_idx] = next_val
        current_seq = torch.cat([current_seq[:, 1:, :], new_step.unsqueeze(0).unsqueeze(0)], dim=1)
    return predictions

# 多步预测对比
steps = [1, 2, 3, 5, 10]
multi_rmse = {}
multi_mae = {}

print("\n========== 多步预测性能（还原至真实尺度 V） ==========")
for step in steps:
    preds_all = []
    trues_all = []
    for i in range(min(50, len(X_test_t))):
        init_seq = X_test_t[i].to(device)
        pred_sequence = multi_step_predict(model, init_seq, step, use_physics, voltage_idx)
        # 正确的未来索引：步长1 -> y_test[i+0]; 步长step -> y_test[i+step-1]
        if i + step - 1 < len(y_test_t):
            true_val = y_test_t[i + step - 1].item()
            preds_all.append(pred_sequence[-1])
            trues_all.append(true_val)
    if len(preds_all) > 0:
        rmse = np.sqrt(mean_squared_error(trues_all, preds_all)) * voltage_std
        mae = mean_absolute_error(trues_all, preds_all) * voltage_std
        multi_rmse[step] = rmse
        multi_mae[step] = mae
        print(f"步长 {step:2d}: RMSE={rmse:.6f} V, MAE={mae:.6f} V")

# 不同训练比例对比
train_ratios = [0.6, 0.7, 0.8]
val_ratio_fixed = 0.15
results_ratio = {}

print("\n========== 不同训练比例对比 ==========")
for train_ratio in train_ratios:
    test_ratio = 1 - train_ratio - val_ratio_fixed
    if test_ratio <= 0:
        continue
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio_fixed)
    
    X_train_r = X[:train_end]
    y_train_r = y[:train_end]
    X_val_r = X[train_end:val_end]
    y_val_r = y[train_end:val_end]
    X_test_r = X[val_end:]
    y_test_r = y[val_end:]
    
    train_ds = TensorDataset(torch.tensor(X_train_r, dtype=torch.float32),
                             torch.tensor(y_train_r, dtype=torch.float32).view(-1,1))
    val_ds = TensorDataset(torch.tensor(X_val_r, dtype=torch.float32),
                           torch.tensor(y_val_r, dtype=torch.float32).view(-1,1))
    test_ds = TensorDataset(torch.tensor(X_test_r, dtype=torch.float32),
                            torch.tensor(y_test_r, dtype=torch.float32).view(-1,1))
    train_loader_r = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader_r = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader_r = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    tmp_config = CONFIG.copy()
    tmp_config['epochs'] = 100   # 不同比例训练时也用足够轮数
    if CONFIG['model_type'] == 'baseline':
        tmp_model = LSTMBaseline(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_layers'],
                                 CONFIG['output_size'], CONFIG['dropout'])
    elif CONFIG['model_type'] == 'transformer':
        tmp_model = LSTMTransformer(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_layers'],
                                    CONFIG['output_size'], nhead=4, dim_feedforward=128, dropout=CONFIG['dropout'])
    else:
        tmp_model = PINNLSTM(CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_layers'],
                             CONFIG['output_size'], CONFIG['dropout'])
    use_physics_tmp = (CONFIG['model_type'] == 'pinn')
    
    _, _ = train_model(tmp_model, train_loader_r, val_loader_r, tmp_config, use_physics_tmp)
    y_true_r, y_pred_r, rmse_r, mae_r, mape_r, r2_r = evaluate_model(tmp_model, test_loader_r, use_physics_tmp)
    results_ratio[train_ratio] = {'rmse': rmse_r, 'mae': mae_r, 'r2': r2_r}
    print(f"训练比例 {train_ratio:.1f}: RMSE={rmse_r:.6f} V, R²={r2_r:.4f}")

print("\n模型评估与对比完成。")


#%%
#10.RUL计算
import numpy as np
import matplotlib.pyplot as plt

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('preprocess_info.pkl', 'rb') as f:
    info = pickle.load(f)
existing_features = info['existing_features']
target_col = info['target_col']
voltage_col_name = f'{target_col}_smooth'
voltage_idx = existing_features.index(voltage_col_name)

voltage_mean = scaler.mean_[voltage_idx]
voltage_std = scaler.scale_[voltage_idx]

# 使用动态计算的平滑电压失效阈值
failure_threshold_smooth = info['failure_threshold_smooth']
failure_threshold_norm = (failure_threshold_smooth - voltage_mean) / voltage_std
print(f"失效阈值（平滑电压）: {failure_threshold_smooth:.4f} V")
print(f"失效阈值（归一化）: {failure_threshold_norm:.4f}")

# 获取测试集预测（注意避免双重反标准化）
if 'y_pred' not in locals():
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            if use_physics:
                main_pred, _ = model(batch_x)
            else:
                main_pred = model(batch_x)
            preds.append(main_pred.cpu().numpy())
            trues.append(batch_y.numpy())
    y_pred_norm = np.concatenate(preds).ravel()
    y_true_norm = np.concatenate(trues).ravel()
    # 此时是归一化值，需要逆变换
    y_pred_orig = y_pred_norm * voltage_std + voltage_mean
    y_true_orig = y_true_norm * voltage_std + voltage_mean
else:
    # 来自 evaluate_model 的 y_pred/y_true 已经是原始电压，直接使用
    y_pred_orig = y_pred
    y_true_orig = y_true

# RUL 计算（基于平滑电压阈值）
failure_indices_true = np.where(y_true_orig < failure_threshold_smooth)[0]
true_failure_time = failure_indices_true[0] if len(failure_indices_true) > 0 else len(y_true_orig)

failure_indices_pred = np.where(y_pred_orig < failure_threshold_smooth)[0]
pred_failure_time = failure_indices_pred[0] if len(failure_indices_pred) > 0 else len(y_pred_orig)

dt = 0.5  # 采样间隔（小时）
print("\n========== RUL 计算结果 ==========")
print(f"真实失效时间（从测试集起始算起）: {true_failure_time * dt:.1f} 小时")
print(f"预测失效时间（从测试集起始算起）: {pred_failure_time * dt:.1f} 小时")
rul_absolute_error = abs(pred_failure_time - true_failure_time) * dt
rul_relative_error = (abs(pred_failure_time - true_failure_time) / true_failure_time) if true_failure_time > 0 else np.inf
print(f"RUL 绝对误差: {rul_absolute_error:.2f} 小时")
print(f"RUL 相对误差: {rul_relative_error:.2%}")

plt.figure(figsize=(12, 5))
plt.plot(y_true_orig, label='True Voltage', linewidth=2)
plt.plot(y_pred_orig, label='Predicted Voltage', linestyle='--', linewidth=2)
plt.axhline(y=failure_threshold_smooth, color='red', linestyle=':', label=f'Failure Threshold = {failure_threshold_smooth:.4f} V')
plt.axvline(x=true_failure_time, color='green', linestyle=':', 
            label=f'True EoL at {true_failure_time * dt:.1f}h')
if pred_failure_time < len(y_pred_orig):
    plt.axvline(x=pred_failure_time, color='orange', linestyle=':', 
                label=f'Predicted EoL at {pred_failure_time * dt:.1f}h')
plt.xlabel('Time (hours from test start)')
plt.ylabel('Stack Voltage (V)')
plt.title('Voltage Degradation and RUL Prediction')
plt.legend()
plt.grid(True)
plt.show()

print("\nRUL 计算完成。")
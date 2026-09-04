#%%
#5.TimeMixer 多尺度时序分解模型

class TimeMixerModel(nn.Module):
    """TimeMixer: Past-Decomposable Mixing + Future-Predictor Mixing

    输入: (batch, L=10, 31特征)
    输出: dict {tau: (batch,)} for each future step
    5个降采样尺度, PDM分解 + FPM跨尺度融合
    """
    def __init__(self, seq_len, n_features, d_model=64, d_ff=128, dropout=0.1,
                 output_steps=None, use_physics=False):
        super().__init__()
        if output_steps is None:
            output_steps = [1]
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.use_physics = use_physics
        self.output_steps = output_steps
        self.n_scales = min(5, seq_len)

        self.input_proj = nn.Linear(n_features, d_model)
        self.dropout_in = nn.Dropout(dropout)

        # PDM: 多尺度降采样 + MLP
        self.scale_downsamples = nn.ModuleList()
        self.scale_mlps = nn.ModuleList()
        for s in range(self.n_scales):
            ds_factor = 2 ** s
            ds_len = max(1, seq_len // ds_factor)
            if ds_len < 1:
                break
            self.scale_downsamples.append(
                nn.AvgPool1d(kernel_size=ds_factor, stride=ds_factor))
            self.scale_mlps.append(nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            ))

        # FPM: 跨尺度注意力
        self.cross_scale_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.fpm_ln = nn.LayerNorm(d_model)

        # 各 future step 输出头
        self.heads = nn.ModuleDict()
        for tau in output_steps:
            self.heads[str(tau)] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

    def forward(self, x):
        # x: (B, L, 31) -> (B, L, d_model)
        batch = x.shape[0]
        x_proj = self.input_proj(x)
        x_proj = self.dropout_in(x_proj)

        # PDM
        scale_outputs = []
        for s in range(self.n_scales):
            x_s = x_proj.permute(0, 2, 1)
            x_s = self.scale_downsamples[s](x_s)
            x_s = x_s.permute(0, 2, 1)
            x_s = self.scale_mlps[s](x_s)
            scale_outputs.append(x_s)

        # FPM: 跨尺度融合
        min_len = min(s.shape[1] for s in scale_outputs)
        aligned = [s[:, :min_len, :] for s in scale_outputs]
        kv = torch.stack(aligned, dim=1).mean(dim=1)
        q = scale_outputs[-1][:, :min_len, :]
        fused, _ = self.cross_scale_attn(q, kv, kv)
        fused = self.fpm_ln(fused + q)

        # 全局池化 + 多头输出
        global_feat = fused.mean(dim=1)

        outputs = {}
        for tau in self.output_steps:
            outputs[tau] = self.heads[str(tau)](global_feat).squeeze(-1)

        return outputs


#%%
#6.对照模型（用于消融）

class SimpleMLP(nn.Module):
    """纯MLP基线, L=10展平为向量"""
    def __init__(self, seq_len, n_features, d_model=64, output_steps=None):
        super().__init__()
        self.output_steps = output_steps or [1]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Linear(d_model, len(self.output_steps)),
        )

    def forward(self, x):
        out = self.net(x)
        return {tau: out[:, i] for i, tau in enumerate(self.output_steps)}


class LSTMSeqModel(nn.Module):
    """单层LSTM + 多头输出"""
    def __init__(self, seq_len, n_features, hidden=96, output_steps=None):
        super().__init__()
        self.output_steps = output_steps or [1]
        self.lstm = nn.LSTM(n_features, hidden, 1, batch_first=True)
        self.heads = nn.ModuleDict({
            str(tau): nn.Linear(hidden, 1) for tau in self.output_steps
        })

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        feat = h[-1]
        return {str(tau): self.heads[str(tau)](feat).squeeze(-1)
                for tau in self.output_steps}


#%%
#7.超参数与模型工厂

CONFIG = {
    'input_size': n_features,
    'seq_len': SEQUENCE_LENGTH,
    'output_size': len(FUTURE_STEPS),
    'output_steps': FUTURE_STEPS,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'epochs': 500,
    'patience': 100,
    'model_type': 'timemixer',
    'use_physics': True,
    'd_model': 64,
    'd_ff': 128,
    'dropout': 0.10,
    'learning_rate': 3e-4,
    'batch_size': 64,
}

SKIP_OPTUNA = True
_MODEL_LR = {'timemixer': 0.0003, 'lstm': 0.0002, 'mlp': 0.001}

if SKIP_OPTUNA:
    print("使用预设超参数（跳过Optuna搜索）")
    print(f"model={CONFIG['model_type']}, d_model={CONFIG['d_model']}, lr={CONFIG['learning_rate']}")

def make_model(config):
    mt = config.get('model_type', 'timemixer')
    seq_len = config.get('seq_len', SEQUENCE_LENGTH)
    n_feat = config.get('input_size', n_features)
    out_steps = config.get('output_steps', FUTURE_STEPS)

    if mt == 'timemixer':
        return TimeMixerModel(
            seq_len, n_feat,
            d_model=config.get('d_model', 64),
            d_ff=config.get('d_ff', 128),
            dropout=config.get('dropout', 0.1),
            output_steps=out_steps,
            use_physics=config.get('use_physics', True))
    elif mt == 'lstm':
        return LSTMSeqModel(
            seq_len, n_feat,
            hidden=config.get('hidden_size', 96),
            output_steps=out_steps)
    elif mt == 'mlp':
        return SimpleMLP(
            seq_len, n_feat,
            d_model=config.get('d_model', 64),
            output_steps=out_steps)
    else:
        return TimeMixerModel(
            seq_len, n_feat,
            d_model=config.get('d_model', 64),
            d_ff=config.get('d_ff', 128),
            dropout=config.get('dropout', 0.1),
            output_steps=out_steps,
            use_physics=config.get('use_physics', True))


#%%
#8.数据加载与多步目标

from torch.utils.data import DataLoader, TensorDataset

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

y_multi_t = {tau: torch.tensor(y_multi_dict[tau], dtype=torch.float32).view(-1, 1)
             for tau in FUTURE_STEPS}


class MultiStepDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_dict):
        self.X = X
        self.y_dict = y_dict

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        targets = torch.cat([self.y_dict[tau][idx] for tau in FUTURE_STEPS], dim=-1)
        return self.X[idx], targets


train_ds = MultiStepDataset(X_train_t, y_multi_t)
val_ds = MultiStepDataset(X_val_t, y_multi_t)
test_ds = MultiStepDataset(X_test_t, y_multi_t)

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=False)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"\n训练 {len(train_ds)}, 验证 {len(val_ds)}, 测试 {len(test_ds)}")
print(f"多步预测目标: tau={FUTURE_STEPS}")


#%%
#9.训练与评估函数

def train_model(model, train_loader, val_loader, config):
    device = config['device']
    model.to(device)
    use_physics = config.get('use_physics', False)
    out_steps = config.get('output_steps', [1])

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss()

    step_weights = {s: 1.0 / (0.5 + i * 0.25) for i, s in enumerate(out_steps)}
    total_w = sum(step_weights.values())
    step_weights = {k: v / total_w for k, v in step_weights.items()}

    lambda_mono = 0.1
    lambda_smooth = 0.05
    swa_start = 40
    swa_model = None
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y_all in train_loader:
            batch_x = batch_x.to(device)
            batch_y_all = batch_y_all.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)

            L_mse = 0.0
            for i, tau in enumerate(out_steps):
                L_mse += step_weights[tau] * criterion(outputs[tau], batch_y_all[:, i])

            if use_physics:
                pred_main = outputs[out_steps[0]]
                diff = pred_main[1:] - pred_main[:-1]
                L_mono = torch.mean(torch.clamp(diff, min=0) ** 2)
                if len(pred_main) >= 3:
                    sd = pred_main[2:] - 2 * pred_main[1:-1] + pred_main[:-2]
                    L_smooth = torch.mean(sd ** 2)
                else:
                    L_smooth = torch.tensor(0.0, device=device)
                loss = L_mse + lambda_mono * L_mono + lambda_smooth * L_smooth
            else:
                loss = L_mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += L_mse.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y_all in val_loader:
                batch_x = batch_x.to(device)
                batch_y_all = batch_y_all.to(device)
                outputs = model(batch_x)
                L_val = 0.0
                for i, tau in enumerate(out_steps):
                    L_val += step_weights[tau] * criterion(outputs[tau], batch_y_all[:, i])
                val_loss += L_val.item()
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

        if epoch >= swa_start:
            if swa_model is None:
                swa_model = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_model:
                    swa_model[k] = swa_model[k] * (swa_count / (swa_count + 1)) + \
                                   model.state_dict()[k].cpu() * (1 / (swa_count + 1))
                swa_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

    model.load_state_dict(torch.load(RUN_DIR / 'best_model.pth'))
    if swa_model is not None:
        swa_state = {k: v.to(device) for k, v in swa_model.items()}
        torch.save(swa_state, RUN_DIR / 'swa_model.pth')
    return model, history


def evaluate_model(model, test_loader, config):
    """评估 tau=1h 预测性能（主指标）"""
    device = config['device']
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y_all in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            tau0 = config['output_steps'][0]
            preds.append(outputs[tau0].cpu().numpy())
            trues.append(batch_y_all[:, 0].cpu().numpy())
    y_pred_norm = np.concatenate(preds).ravel()
    y_true_norm = np.concatenate(trues).ravel()

    with open(RUN_DIR / 'scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_true_norm.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return y_true, y_pred, rmse, mae, mape, r2

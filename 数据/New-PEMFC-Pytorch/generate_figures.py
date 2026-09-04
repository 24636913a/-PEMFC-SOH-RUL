# ================================================================
# 论文图表生成: Fig 2 (预处理流程图) + Fig 3 (TFT架构图) + Fig 12 (消融柱状图)
# 用法: python generate_figures.py
# 输出: outputs/paper_figures/
# ================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

RUN_DIR = Path('./outputs/paper_figures')
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# Fig 2: 数据预处理流程图 (5 Steps)
# ================================================================
def draw_fig2_preprocessing():
    fig, ax = plt.subplots(1, 1, figsize=(10, 16))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 22)
    ax.axis('off')
    ax.set_facecolor('white')

    box_w, box_h = 8.2, 1.8
    x0 = 1.0

    colors = ['#E8E8E8', '#D6EAF8', '#D6EAF8', '#FDEBD0', '#D5F5E3', '#E8DAEF', '#FADBD8']

    titles = [
        'Raw Data: IEEE 2014 PHM Data Challenge',
        'Step 1: Temporal Downsampling',
        'Step 2: Savitzky-Golay Filtering',
        'Step 3: L=1 Single-Step Learning + Multi-Lag Augmentation',
        'Step 4: Chronological 50/10/40 Split (Per-Lag)',
        'Step 5: MinMax Normalization [0, 1]',
        'Model Input'
    ]

    details = [
        'FC2 (Primary): 2 CSV, 127,370 rows (~30s sampling)\nFC1 (Cross-Validation): 3 CSV, 143,862 rows\n24 sensors per stack  |  Target: Utot (V)\nFC1/FC2 processed independently',
        'FC2: 0.5h interval — ripple condition needs higher resolution\nFC1: 1.0h interval — quasi-static, denser sampling adds no info\nFC2: 127,370 → 2,042 rows   |   FC1: 143,862 → 1,155 rows\n\nAdjacent ~30s points nearly identical; degradation discernible at hourly scale',
        'Window length = 21, Polynomial order = 2\nApplied to each of 24 sensors independently\nSuppresses measurement noise (±0.005 V)\nPreserves degradation slope changes (unlike moving average)',
        'For each time t, construct 3 samples:\n  X[t−1] → y[t] (lag=1, marker=0.1)    X[t−2] → y[t] (lag=2, marker=0.2)    X[t−4] → y[t] (lag=4, marker=0.4)\n25th feature = lag marker column (lag/10)\nFC2: 2,038 → 6,114 (×3)   |   FC1: 1,151 → 3,453 (×3)\nAvoids SNR < 1 in sliding window; lag marker lets model distinguish time distance',
        'Each lag group split independently, then merged\nTrain: first 50%  |  Validation: middle 10%  |  Test: last 40%\nshuffle = False (preserves temporal order)\nFC2: Train=3,057  |  Val=609  |  Test=2,448\n\nPer-lag split prevents data leakage across lag copies',
        'X: MinMaxScaler(0,1) — 25 features unified\ny: MinMaxScaler(0,1) — Utot only\nFit on training set, transform val & test\n\nMinMax chosen over Z-score:\n  Sensors span orders of magnitude (V~3, T~50, flow~20)\n  Z-score would let large-valued sensors dominate gradients',
        'X = (N, 1, 25)\n25 = 24 sensors + 1 lag marker\n1 = single time step\nN = 6,114 (FC2) / 3,453 (FC1)'
    ]

    y_positions = [20.5, 17.8, 15.0, 11.8, 8.5, 5.5, 2.8]

    for i, (y, title, detail, c) in enumerate(zip(y_positions, titles, details, colors)):
        rect = FancyBboxPatch((x0, y), box_w, box_h,
                              boxstyle="round,pad=0.1", facecolor=c,
                              edgecolor='#333333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x0 + 0.35, y + box_h - 0.3, title, fontsize=11.5, fontweight='bold',
                va='top', color='#1a1a1a')
        ax.text(x0 + 0.35, y + box_h - 0.8, detail, fontsize=8, va='top',
                color='#333333', linespacing=1.3)

        if i < len(y_positions) - 1:
            next_top = y_positions[i+1] + box_h
            ax.annotate('', xy=(x0 + box_w/2, y), xytext=(x0 + box_w/2, next_top),
                       arrowprops=dict(arrowstyle='->', color='#555555',
                                       lw=2.5, connectionstyle='arc3,rad=0'))

    # 右侧设计理由注释
    side_notes = [
        (17.8, '≈ 10 Hz raw → hourly\ndownsampling'),
        (15.0, 'SG polynomial fit\npreserves inflection points\n(voltage drop rate changes)'),
        (11.8, 'Three time-distance views\n1h / 2h / 4h lookback\n×3 data expansion'),
        (8.5, 'Simulates real deployment:\ntrain on history,\npredict future degradation'),
        (5.5, 'Retains physical magnitude\ndifferences across sensors\n→ VS can selectively weight'),
    ]
    for y, text in side_notes:
        ax.text(x0 + box_w + 0.35, y + box_h/2, text, fontsize=7.5,
                va='center', color='#666666', style='italic')

    # 标题
    ax.text(5.1, 22.5, 'Preprocessing Pipeline for PEMFC Degradation Data',
            ha='center', fontsize=13, fontweight='bold', color='#1a1a1a')

    plt.tight_layout()
    fig.savefig(RUN_DIR / 'fig2_preprocessing_pipeline.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Fig 2 (Preprocessing Pipeline) saved.')


# ================================================================
# Fig 3: TFT 架构图
# ================================================================
def draw_fig3_tft():
    fig, ax = plt.subplots(1, 1, figsize=(9, 11.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 17)
    ax.axis('off')
    ax.set_facecolor('white')

    components = [
        {
            'name': '(1) Variable Selection',
            'y': 13.2, 'h': 2.2,
            'color': '#D5F5E3',
            'formula': 'w = σ(α + β)     x̃ = x ⊙ w     Train: x̃ += ε·N(0, 0.01²)',
            'detail': '25 weights (24 sensors + 1 lag), learned via SGD backprop\nSoft-selects informative sensors, suppresses irrelevant ones\n→ Trained weights = feature importance ranking',
        },
        {
            'name': '(2) LSTM Encoder',
            'y': 10.2, 'h': 2.2,
            'color': '#D6EAF8',
            'formula': 'f, i, o = σ(Wx̃ + Uh + b)     c = f ⊙ c + i ⊙ tanh(Wx̃ + Uh + b)',
            'detail': '1 layer, hidden_size = 96, batch_first = True\nshuffle = False → hidden state propagates degradation history across batch\nGate mechanisms capture long-term trends in voltage decline',
        },
        {
            'name': '(3) GRN (Gated Residual Network)',
            'y': 6.8, 'h': 2.5,
            'color': '#FDEBD0',
            'formula': 'z = ELU(W₁h)     g = σ(W_g·z)     out = LN( proj(h) + Drop(g ⊙ W₂z) )',
            'detail': 'Internal FF dim = 128, Dropout = 0.10\nResidual bypass: even if gate saturates, skip connection passes raw signal\n→ Key to training stability: FC1 Min R² from −2.21 → 0.79 with GRN',
        },
        {
            'name': '(4) MultiheadAttention (4 heads)',
            'y': 3.5, 'h': 2.5,
            'color': '#E8DAEF',
            'formula': 'Q, K, V = Linear(h)×3     Attn = softmax(QKᵀ/√24)·V     h’ = LN(h + MultiHead(h))',
            'detail': '4 heads × d_k = 24, batch_first = True\nL=1: attention operates on feature dimension → captures cross-sensor interactions\n(e.g., U1-U5 synchronous decline, temperature ↔ voltage coupling)',
        },
        {
            'name': 'Output Layer',
            'y': 1.6, 'h': 1.2,
            'color': '#FADBD8',
            'formula': 'ŷ = W_fc · h[-1, :] + b_fc     FC(96 → 1)',
            'detail': 'Take last timestep → scalar Utot prediction',
        },
    ]

    box_w = 7.5
    x0 = 1.25

    for comp in components:
        y, h = comp['y'], comp['h']
        rect = FancyBboxPatch((x0, y), box_w, h,
                              boxstyle="round,pad=0.15", facecolor=comp['color'],
                              edgecolor='#444444', linewidth=1.8)
        ax.add_patch(rect)

        ax.text(x0 + 0.3, y + h - 0.25, comp['name'], fontsize=11.5, fontweight='bold',
                va='top', color='#1a1a1a')
        ax.text(x0 + 0.3, y + h - 0.75, comp['formula'], fontsize=8.5,
                va='top', color='#2C3E50', family='monospace')
        ax.text(x0 + 0.3, y + h - 1.3, comp['detail'], fontsize=7.8, va='top',
                color='#444444', linespacing=1.3)

    # 组件间箭头
    for i in range(len(components) - 1):
        y_from = components[i]['y']
        y_to = components[i+1]['y'] + components[i+1]['h']
        ax.annotate('', xy=(x0 + box_w/2, y_from),
                   xytext=(x0 + box_w/2, y_to),
                   arrowprops=dict(arrowstyle='->', color='#444444', lw=2.5))

    # 输入标注
    top_y = components[0]['y'] + components[0]['h']
    ax.text(x0 + box_w/2, top_y + 0.6, 'Input: x in R^25 (24 sensors + lag marker)',
            ha='center', fontsize=10, fontweight='bold', color='#1a1a1a')
    ax.annotate('', xy=(x0 + box_w/2, top_y + 0.15),
               xytext=(x0 + box_w/2, top_y + 0.45),
               arrowprops=dict(arrowstyle='->', color='#444444', lw=2.5))

    # 输出标注
    bot_y = components[-1]['y']
    ax.text(x0 + box_w/2, bot_y - 0.6, 'Output: y_hat in R  (scalar voltage)',
            ha='center', fontsize=10, fontweight='bold', color='#1a1a1a')
    ax.annotate('', xy=(x0 + box_w/2, bot_y - 0.1),
               xytext=(x0 + box_w/2, bot_y - 0.4),
               arrowprops=dict(arrowstyle='->', color='#444444', lw=2.5))

    # Loss
    ax.text(x0 + box_w/2, bot_y - 1.15, 'L = MSE(ŷ, y_true)     Optimizer: Adam(lr=3e-4, wd=1e-5)',
            ha='center', fontsize=9, fontweight='bold', color='#C0392B')

    # 左侧形状标注
    shapes_left = [
        (13.2+2.2/2, '(batch,1,25)'),   # VS
        (10.2+2.2/2, '(batch,1,25)'),   # LSTM
        (6.8+2.5/2, 'h: (batch,1,96)'),  # GRN
        (3.5+2.5/2, '(batch,1,96)'),    # Attn
        (1.6+1.2/2, '(batch,1,96)'),    # Out
    ]
    for y, text in shapes_left:
        ax.text(x0 - 0.5, y, text, fontsize=7, color='#888888', va='center', ha='right')

    # 右侧关键说明
    side_notes = [
        (13.2+2.2/2, '→ learned importance'),
        (10.2+2.2/2, '→ temporal memory'),
        (6.8+2.5/2, '→ noise suppression'),
        (3.5+2.5/2, '→ cross-sensor interaction'),
    ]
    for y, text in side_notes:
        ax.text(x0 + box_w + 0.4, y, text, fontsize=7.5, color='#666666', va='center',
                style='italic')

    ax.text(5, 16.2, 'Temporal Fusion Transformer (TFT) Architecture',
            ha='center', fontsize=13, fontweight='bold', color='#1a1a1a')
    ax.text(5, 15.5, 'Adapted for PEMFC Single-Step Voltage Prediction (L=1 + Lag Augmentation)',
            ha='center', fontsize=9, color='#666666')

    plt.tight_layout()
    fig.savefig(RUN_DIR / 'fig3_tft_architecture.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Fig 3 (TFT Architecture) saved.')


# ================================================================
# Fig 12: 消融实验双栏柱状图 (FC1 + FC2)
# ================================================================
def draw_fig12_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = ['DLinear\n(L=10)', 'ResMLP\n(L=1)', 'LSTM\n(L=1)', 'TSMixer\n(L=1)', 'TFT\n(L=1)']
    x = np.arange(len(models))
    w = 0.25

    # ---- FC1 ----
    fc1_min  = [0.65, -12.88, -2.21,  0.21, 0.79]
    fc1_mean = [0.91, -10.00, -0.12,  0.70, 0.90]
    fc1_max  = [0.98,  -7.72,  0.77,  0.91, 0.93]

    ax1.bar(x - w, fc1_min,  w, label='Min R²', color='#E74C3C', edgecolor='white', linewidth=0.3)
    ax1.bar(x,     fc1_mean, w, label='Mean R²', color='#3498DB', edgecolor='white', linewidth=0.3)
    ax1.bar(x + w, fc1_max,  w, label='Max R²', color='#2ECC71', edgecolor='white', linewidth=0.3)

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
    ax1.axvspan(3.5, 4.5, alpha=0.08, color='orange')
    ax1.text(4, 1.2, 'Proposed', ha='center', fontsize=9, fontweight='bold', color='#E67E22')

    # 给负数模型加注释
    for val, label, offset in [(-12.88, 'ResMLP', -13.5), (-2.21, '', -3.5)]:
        if label:
            ax1.annotate(f'LayerNorm\ndestroys\nMinMax data',
                        xy=(1, -12.88), xytext=(2.5, -14),
                        fontsize=7.5, color='#C0392B',
                        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2))

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel('R² Score', fontsize=10)
    ax1.set_title('FC1 (Quasi-Static, 1h sampling)', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax1.set_ylim(-16, 2)
    ax1.grid(axis='y', alpha=0.15, linewidth=0.5)

    # ---- FC2 ----
    fc2_min  = [-4.49, 0.21, 0.83, 0.95, 0.89]
    fc2_mean = [-4.33, 0.34, 0.96, 0.97, 0.95]
    fc2_max  = [-4.15, 0.49, 0.98, 0.99, 0.98]

    ax2.bar(x - w, fc2_min,  w, label='Min R²', color='#E74C3C', edgecolor='white', linewidth=0.3)
    ax2.bar(x,     fc2_mean, w, label='Mean R²', color='#3498DB', edgecolor='white', linewidth=0.3)
    ax2.bar(x + w, fc2_max,  w, label='Max R²', color='#2ECC71', edgecolor='white', linewidth=0.3)

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
    ax2.axvspan(3.5, 4.5, alpha=0.08, color='orange')
    ax2.text(4, 1.2, 'Proposed', ha='center', fontsize=9, fontweight='bold', color='#E67E22')

    # DLinear 标注
    ax2.annotate('AvgPool1d\ndecomposition\nfails on ripple data',
                xy=(0, -4.49), xytext=(2.2, -3),
                fontsize=7.5, color='#C0392B',
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2))

    # TSMixer 高亮
    ax2.annotate('Best stability\nMin R² = 0.95',
                xy=(3, 0.95), xytext=(3, -1.5),
                fontsize=8, color='#27AE60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5),
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', edgecolor='#27AE60', alpha=0.8))

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8)
    ax2.set_title('FC2 (Ripple Current, 0.5h sampling)', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax2.set_ylim(-6, 2)
    ax2.grid(axis='y', alpha=0.15, linewidth=0.5)

    fig.suptitle('Ablation Study: Five Models under Two PEMFC Operating Conditions (Loop 50)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(RUN_DIR / 'fig12_ablation_study.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Fig 12 (Ablation Study) saved.')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    draw_fig2_preprocessing()
    draw_fig3_tft()
    draw_fig12_ablation()
    print(f'\nAll figures saved to {RUN_DIR.resolve()}/')

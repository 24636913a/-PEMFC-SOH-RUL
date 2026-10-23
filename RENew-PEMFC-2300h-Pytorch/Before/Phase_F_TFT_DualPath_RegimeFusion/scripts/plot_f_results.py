"""Publication-style Development exploratory figures from frozen caches only."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
OUT, FIG = PHASE / 'outputs', PHASE / 'figures'
GROUPS = ['F0', 'F1', 'F2']
COLORS = ['#B64342', '#3775BA', '#8BCF8B']


def save(fig, name):
    fig.tight_layout(pad=2)
    fig.savefig(FIG / (name + '.png'), dpi=300, facecolor='white')
    fig.savefig(FIG / (name + '.pdf'), facecolor='white')
    plt.close(fig)


def run():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 12, 'axes.linewidth': 1.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none'})
    m = pd.read_csv(OUT / 'metrics/F_SEED42_METRICS.csv')
    for split, number in (('validation', 1), ('assessment', 2)):
        s = m.loc[m.split == split].set_index('experiment_group').loc[GROUPS]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        bars = ax.bar(GROUPS, s.RMSE_mV, color=COLORS, edgecolor='black', linewidth=1.2)
        ax.set_ylim(0, s.RMSE_mV.max() * 1.18)
        for b, value in zip(bars, s.RMSE_mV):
            ax.text(b.get_x()+b.get_width()/2, value, f'{value:.3f}', ha='center', va='bottom')
        ax.set(ylabel=f'{split.title()} RMSE (mV)', title=f'Seed42 — Development exploratory\n{split.title()}: F0 / F1 / F2')
        save(fig, f'Fig{number}_{split.title()}_RMSE')
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for i, group in enumerate(GROUPS):
        h = pd.read_csv(OUT / f'training_logs/{group}_epochs_seed42.csv')
        axes[i, 0].plot(h.epoch, h.train_standardized_MSE, color=COLORS[i], lw=2)
        axes[i, 1].plot(h.epoch, h.validation_RMSE_mV, color=COLORS[i], lw=2)
        axes[i, 0].set(xlabel='Epoch', ylabel='Train standardized MSE', title=group)
        axes[i, 1].set(xlabel='Epoch', ylabel='Validation RMSE (mV)', title=group)
    fig.suptitle('Seed42 — Development exploratory convergence')
    save(fig, 'Fig3_Training_Convergence')
    frames = {g: pd.read_csv(OUT / f'predictions/{g}_assessment_seed42.csv') for g in GROUPS}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    limit = max(max(f.y_true_delta_V.abs().max(), f.y_pred_delta_V.abs().max()) for f in frames.values()) * 1000 * 1.03
    for i, g in enumerate(GROUPS):
        f = frames[g]
        axes[i].scatter(f.y_true_delta_V*1000, f.y_pred_delta_V*1000, alpha=.22, s=5, color=COLORS[i], rasterized=True)
        axes[i].plot([-limit, limit], [-limit, limit], 'k--', lw=1)
        axes[i].set(xlabel='True ΔV (mV)', ylabel='Predicted ΔV (mV)', title=g,
                    xlim=(-limit, limit), ylim=(-limit, limit))
    fig.suptitle('Assessment — Seed42 — Development exploratory (all queries)')
    save(fig, 'Fig4_Assessment_DeltaV_Scatter')
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, g in enumerate(GROUPS):
        t = pd.read_csv(OUT / f'metrics/{g}_assessment_TARGET_SSE.csv')
        ax.plot(t.target_age_h, t.mean_absolute_error_mV, label=g, color=COLORS[i], lw=1.6)
    ax.set(xlabel='Target operating age (h)', ylabel='Mean absolute error per target (mV)',
           title='Assessment — Seed42 — Development exploratory'); ax.legend(ncol=3)
    ax.set_ylim(bottom=0)
    save(fig, 'Fig5_Assessment_Error_Over_Time')
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, g in enumerate(GROUPS):
        t = pd.read_csv(OUT / f'metrics/{g}_assessment_HORIZON_RMSE.csv')
        ax.plot(t.horizon_step*.5, t.RMSE_mV, label=g, color=COLORS[i], lw=2)
    ax.set(xlabel='Forecast horizon (h)', ylabel='Assessment RMSE (mV)',
           title='Seed42 — Development exploratory'); ax.set_ylim(bottom=0); ax.legend(ncol=3)
    save(fig, 'Fig6_Assessment_Horizon_RMSE')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, split in zip(axes, ('validation', 'assessment')):
        t = pd.read_csv(OUT / f'predictions/F2_{split}_GATES.csv')
        ax.hist(t.gate, bins=np.linspace(0, 1, 26), color=COLORS[2], edgecolor='black')
        ax.axvline(.5, ls='--', color='black', label='Initial gate = 0.5')
        ax.set(xlabel='LSTM fusion weight g (dimensionless)', ylabel='Origin count', title=split.title(), xlim=(0, 1)); ax.legend()
    fig.suptitle('F2 gate — Seed42 — Development exploratory')
    save(fig, 'Fig7_F2_Gate_Distribution')
    s = m.loc[m.split == 'assessment'].set_index('experiment_group').loc[GROUPS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(GROUPS, s.parameter_count, color=COLORS, edgecolor='black')
    axes[1].bar(GROUPS, s.training_time_s / 60, color=COLORS, edgecolor='black')
    axes[0].set(ylabel='Trainable parameter count', ylim=(0, s.parameter_count.max()*1.1))
    axes[1].set(ylabel='Total training time (min)', ylim=(0, s.training_time_s.max()/60*1.1))
    fig.suptitle('Seed42 — Development exploratory parameter/compute cost')
    save(fig, 'Fig8_Parameters_and_Training_Cost')


if __name__ == '__main__':
    run()

# %% [markdown]
# # RealHyFC F2：交互式流程展示
# 按顺序 Run Cell，或 Run All Cells。只读取历史搜索/训练记录并执行冻结权重推理。
# 修改下一单元的 SEED 为 42、103、193、202、210、212 或 "all"。

# %% 1. 种子、环境及真实进度记录
SEED = 202

import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone
from contextlib import contextmanager

import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import Image, Markdown, display


def locate_program_folder():
    bases = [Path.cwd(), *Path.cwd().parents]
    if '__file__' in globals():
        bases.insert(0, Path(__file__).resolve().parent)
    for base in bases:
        for candidate in [base, base / 'REAL_Paper_File/F2主程序',
                          base / 'RENew-PEMFC-2300h-Pytorch/REAL_Paper_File/F2主程序']:
            if candidate.name == 'F2主程序' and (candidate / 'RealHyFC_F2_Main.py').is_file():
                return candidate.resolve()
    raise FileNotFoundError('请先在仓库、研究根目录或F2主程序文件夹中打开交互窗口。')


FOLDER = locate_program_folder()
CORE_FILE = FOLDER / 'RealHyFC_F2_Main.py'
spec = importlib.util.spec_from_file_location('realhyfc_interactive_frozen_core', CORE_FILE)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)  # __name__ is not __main__: no CLI, training or inference here.
plt = core.plt
if SEED == 'all':
    seeds = list(core.SUPPORTED_SEEDS)
elif isinstance(SEED, int) and not isinstance(SEED, bool) and SEED in core.SUPPORTED_SEEDS:
    seeds = [SEED]
else:
    raise ValueError('SEED仅允许42、103、193、202、210、212或"all"')
run = core.OUTPUT_ROOT / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + f'_F2_interactive_seed{SEED}')
run.mkdir(parents=True, exist_ok=False)
events, image_files, source_hashes = [], [], {}


def progress(message):
    print(message, flush=True)
    events.append(dict(sequence=len(events) + 1, utc=datetime.now(timezone.utc).isoformat(), message=message))
    core.save_json(run, 'presentation/progress.json', events)


def protect(path):
    path = Path(path).resolve()
    source_hashes[str(path)] = core.sha256(path)
    return path


def show_png(path):
    path = Path(path)
    core.require(path.is_file(), 'DISPLAY_IMAGE_MISSING')
    display(Image(filename=str(path)))  # Works with Agg: actual image/png display_data in IPython.
    image_files.append(str(path.relative_to(run)))


def save_and_show(fig, name, data):
    path = core.output_path(run, f'presentation/{name}.png')
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)
    core.save_csv(run, f'presentation/{name}.csv', data)
    show_png(path)


protect(CORE_FILE)
protect(core.ROOT / 'RealHyFC_F2_Main.py')
protect(core.SEED_ASSETS)
protect(core.SOURCE)
protect(core.FROZEN_DATA / 'SCALER.json')
environment = core.configure_environment('cuda')
self_checks = core.integration_self_checks()
display(Markdown('**只读历史展示 + 冻结权重推理**。不启动训练、Optuna搜索或模型选择。'))
print(f'种子：{seeds}；Python {environment["python"]}；GPU：{environment["GPU"]}')
print(f'本次独立输出：{run}')

# %% 2. 数据读取、四区间及合法窗口
progress('正在读取RealHyFC数据并核对源文件身份')
frame, source = core.load_full_realhyfc()
windows, exclusions = {}, []
for partition, (start, end) in core.chronological_partitions(len(frame)).items():
    windows[partition], excluded = core.build_legal_windows(frame, start, end)
    excluded['partition'] = partition
    exclusions.append(excluded)
window_audit = core.verify_window_identity(windows, frame)
core.save_csv(run, 'WINDOW_COUNTS_AND_IDENTITIES.csv', window_audit)
core.save_csv(run, 'EXCLUDED_WINDOWS.csv', pd.concat(exclusions, ignore_index=True))
progress('已完成合法窗口构建及冻结索引逐点核验')
print(f'实际分箱记录：{len(frame)}；输入48×14，预测24步ΔV；每步0.5运行小时')
display(pd.DataFrame(window_audit)[['partition', 'records', 'origins', 'queries', 'unique_targets']])
display(Markdown('正式显示指标仅使用 **Development Validation：205个起点、4920个查询/种子**。'
                 'Train不混入评价；历史Assessment和后段保留探索性身份。'))
plot_frame = frame[['bin_end_ophrs2_h', 'cell_voltage_mean', 'temporal_segment_id']].copy()
fig, ax = plt.subplots(figsize=(12, 4))
for index, (partition, (start, end)) in enumerate(core.chronological_partitions(len(frame)).items()):
    part = plot_frame.iloc[start:end]
    # Plot each continuous segment separately, without connecting across interruptions.
    first = True
    breaks = part.temporal_segment_id.ne(part.temporal_segment_id.shift()) | part.bin_end_ophrs2_h.diff().sub(.5).abs().gt(1e-6)
    for _, section in part.groupby(breaks.cumsum()):
        ax.plot(section.bin_end_ophrs2_h, section.cell_voltage_mean, lw=.8,
                color=f'C{index}', label=partition if first else None)
        first = False
ax.set(xlabel='Actual operating time (h)', ylabel='Raw binned cell voltage (V)',
       title=f'RealHyFC: {len(frame)} real half-hour bins; original chronological partitions')
ax.grid(alpha=.3); ax.legend(fontsize=8)
save_and_show(fig, '01_raw_voltage', plot_frame)

# %% 3. Train-only标准化：参数仅来自合法Train历史
progress('正在核对Train-only标准化参数')
scaler_preview = core.TrainOnlyScaler().fit(windows['TRAIN'].x_raw)
scaler_authority = core.read_json(core.FROZEN_DATA / 'SCALER.json')
core.require(np.array_equal(scaler_preview.mean, scaler_authority['mean']) and
             np.array_equal(scaler_preview.scale, scaler_authority['scale']), 'INTERACTIVE_SCALER_IDENTITY_MISMATCH')
stats = pd.DataFrame(dict(feature=core.FEATURES, train_mean=scaler_preview.mean,
                         train_standard_deviation=scaler_preview.scale))
display(stats)
print('z = (x − Train mean) / Train standard deviation；目标尺度来自原始Train电压。')
print('使用1380个合法Train历史窗口拟合；后续仅对历史电压作因果EWMA，再标准化和Top10掩码。')
core.save_csv(run, 'presentation/train_only_scaler.csv', stats)
progress('已完成Train-only标准化参数核验')

# %% 4. Spearman历史特征检查：不使用Validation选特征
mask = scaler_authority['Top10_indices'] if 'Top10_indices' in scaler_authority else [0, 3, 13, 11, 4, 9, 6, 7, 10, 8]
scores = core.spearman_train_audit(windows['TRAIN'], mask)
progress('已完成Spearman特征检查，Top10与冻结配置一致')
display(scores[['feature', 'mean_abs_spearman', 'selected_top10']])
print('固定Top10：', [core.FEATURES[index] for index in mask])
fig, ax = plt.subplots(figsize=(10, 6))
ordered = scores.iloc[::-1]
ax.barh(ordered.feature, ordered.mean_abs_spearman,
        color=['#1f77b4' if selected else '#bdbdbd' for selected in ordered.selected_top10])
ax.set(xlabel='Mean |Spearman rho| across 24 DeltaV targets (Train only)',
       title='Frozen Train-only feature ranking; blue = retained Top10')
ax.tick_params(axis='y', labelsize=8); ax.grid(axis='x', alpha=.25)
save_and_show(fig, '02_spearman_top10', scores)
core.save_csv(run, 'TRAIN_ONLY_SPEARMAN_AUDIT.csv', scores)

# %% 5. 因果EWMA alpha=0.3：真实窗口滤波对比
progress('正在执行并核对冻结预处理：因果EWMA alpha=0.3、Train scaler及Top10掩码')
scaler, prepared, frozen_scores = core.preprocess_frozen(windows)
core.require(np.array_equal(scaler.mean, scaler_preview.mean) and np.array_equal(scaler.scale, scaler_preview.scale),
             'DISPLAY_AND_INFERENCE_SCALERS_DIFFER')
pd.testing.assert_frame_equal(scores, frozen_scores, check_exact=True)
example = windows['TRAIN'].x_raw[:1]
filtered = core.causal_ewma(example)
example_time = windows['TRAIN'].origin_age_h[0] - np.arange(47, -1, -1) * .5
filter_data = pd.DataFrame(dict(age_h=example_time, raw_voltage_V=example[0, :, 0],
                                causal_EWMA_voltage_V=filtered[0, :, 0]))
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(filter_data.age_h, filter_data.raw_voltage_V, lw=1.5, alpha=.8, label='Raw history voltage')
ax.plot(filter_data.age_h, filter_data.causal_EWMA_voltage_V, lw=1.5, ls='--', label='Causal EWMA, alpha=0.3')
ax.set(xlabel='Actual history operating time (h)', ylabel='Cell voltage (V)',
       title='First legal Train history: real 48-step window; no future samples')
ax.grid(alpha=.3); ax.legend()
save_and_show(fig, '03_causal_ewma', filter_data)
progress('已完成因果EWMA处理；四区间预处理张量与冻结数组完全一致')

# %% 6. 历史超参数记录：基础TFT与后来F2试验分别显示
progress('正在读取当前冻结超参数与既有Optuna历史（只读，不创建试验）')
provenance = core.load_historical_hyperparameters()
parameters = provenance['parameters']
protect(core.HYPERPARAMETER_SOURCE)
display(Markdown('**当前六个原始F2 checkpoint实际配置**：继承基础TFT历史Fold 2、Trial 13。'
                 '后来F2专属调参试验单独展示，其配置未替换这六个checkpoint。'))
display(pd.DataFrame([parameters]).T.rename(columns={0: 'actual_frozen_value'}))
optuna_root = core.ROOT / 'Phase_TFT_Hyperparameter_Optimization_Restart/outputs'
base_trials = []
for fold in ['fold_1', 'fold_2', 'fold_3']:
    database = protect(optuna_root / f'optuna_T1R_{fold}.db')
    # SQLite read-only URI: no Optuna load/create/optimize API, no schema writes.
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        trials = pd.read_sql_query('SELECT t.number, t.state, v.value AS standardized_inner_stop_RMSE '
                                   'FROM trials t LEFT JOIN trial_values v ON t.trial_id=v.trial_id '
                                   'AND v.objective=0 ORDER BY t.number', connection)
    trials['fold'] = fold
    base_trials.append(trials)
base_trials = pd.concat(base_trials, ignore_index=True)
fig, ax = plt.subplots(figsize=(10, 4))
for fold, trials in base_trials.groupby('fold', sort=False):
    valid = trials.loc[trials.state == 'COMPLETE']
    ax.plot(valid.number, valid.standardized_inner_stop_RMSE, marker='o', ms=3, label=fold)
fold2 = base_trials.loc[(base_trials.fold == 'fold_2') & (base_trials.number == 13)].iloc[0]
ax.scatter([13], [fold2.standardized_inner_stop_RMSE], marker='*', s=140, c='red', label='Inherited Fold 2 Trial 13')
ax.set(xlabel='Historical trial number', ylabel='Standardized inner-stop RMSE (not mV)',
       title='Historical base-TFT Optuna search; three separate studies, no new optimization')
ax.grid(alpha=.3); ax.legend(fontsize=8)
save_and_show(fig, '04_base_TFT_optuna_history', base_trials)
display(base_trials)
refinement = core.ROOT / 'Phase_SCI_F2_Hyperparameter_Refinement'
f2_trial_file = protect(refinement / 'outputs/OPTUNA_TRIALS.csv')
f2_trials = core.read_csv(f2_trial_file)
refinement_summary = core.read_csv(protect(refinement / 'SUMMARY.csv'))
refinement_selection = core.read_json(protect(refinement / 'configs/SELECTED_CONFIGURATION.json'))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(f2_trials.number, f2_trials.RMSE_mV, marker='o', label='Historical F2 inner-TUNE RMSE')
selected_trial = f2_trials.loc[f2_trials.number == refinement_selection['trial_number']].iloc[0]
ax.scatter([selected_trial.number], [selected_trial.RMSE_mV], c='red', marker='*', s=140,
           label='Historical selected trial; not used by these six checkpoints')
ax.set(xlabel='Historical F2 trial number', ylabel='Inner-TUNE DeltaV RMSE (mV)',
       title='Later six-trial F2 refinement: Train-internal search, not a current optimization')
ax.grid(alpha=.3); ax.legend(fontsize=8)
save_and_show(fig, '05_F2_refinement_history', f2_trials)
display(f2_trials)
display(Markdown('**后续F2调参结果未取得整体改善**：以下是该阶段原始三种子对照，'
                 '不是当前六个筛选实例的总体结果。不能把两次搜索的不同目标或单位混合比较。'))
display(refinement_summary)
core.save_csv(run, 'presentation/historical_F2_refinement_summary.csv', refinement_summary)
progress('已读取冻结超参数与两类历史搜索记录；当前配置保持不变')

# %% 7. 当前种子的真实历史训练记录：并非实时训练
assets, histories = {}, {}
for seed in seeds:
    asset = core.historical_checkpoint(seed)
    assets[seed] = asset
    for key in ['checkpoint', 'reference_prediction', 'training_history', 'protocol_source']:
        protect(core.ROOT.parent / asset[key])
    history = core.read_csv(core.ROOT.parent / asset.training_history)
    histories[seed] = history
    progress(f'已读取Seed {seed}历史训练记录：{len(history)}个Epoch，最佳Epoch={asset.best_epoch}（未训练）')
    loss_data = history[['epoch', 'train_standardized_MSE', 'validation_standardized_RMSE']].copy()
    loss_data['validation_standardized_MSE'] = loss_data.validation_standardized_RMSE ** 2
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_data.epoch, loss_data.train_standardized_MSE, label='Historical Train Loss', lw=1.5)
    ax.plot(loss_data.epoch, loss_data.validation_standardized_MSE, label='Historical Validation Loss', lw=1.5)
    ax.axvline(asset.best_epoch, color='gray', ls='--', lw=1, label=f'Historical best epoch {asset.best_epoch}')
    ax.set(xlabel='Historical epoch', ylabel='Standardized MSE Loss',
           title=f'F2 Seed {seed}: saved training log only; no new training')
    ax.legend(); ax.grid(alpha=.3)
    save_and_show(fig, f'06_training_history_seed{seed}', loss_data)

# %% 8. 冻结权重真实GPU推理及图像直接显示
@contextmanager
def prediction_progress():
    original = core.predict_standardized
    def observed_predict(model, x, batch_size, device):
        progress(f'正在进行GPU预测：{len(x)}个历史窗口，batch_size={batch_size}')
        result = original(model, x, batch_size, device)  # exact original inference implementation
        progress(f'已完成GPU预测：{len(x)}个窗口 × 24步')
        return result
    core.predict_standardized = observed_predict
    try:
        yield
    finally:
        core.predict_standardized = original


results, audits = [], []
for seed in seeds:
    progress(f'正在加载Seed {seed}权重并核对模型身份')
    asset = assets[seed]
    checkpoint_file = core.ROOT.parent / asset.checkpoint
    model = core.initialize_f2(seed, parameters, core.torch.device('cuda'))
    checkpoint = core.load_checkpoint(model, checkpoint_file, scaler, seed, asset)
    with prediction_progress():
        frames, parity = core.predict_partitions(model, seed, windows, prepared, scaler, parameters,
                                                run, core.torch.device('cuda'), asset, checkpoint)
    for partition in core.PARTITIONS:
        results.append(dict(model='F2', seed=seed, partition=partition, **core.calculate_metrics(frames[partition])))
    core.save_result_bundle(run, seed, checkpoint_file, scaler, provenance, frames['VALIDATION'])
    core.plot_outputs(run, seed, frames, histories[seed], checkpoint['best_epoch'], 'evaluate', 'VALIDATION')
    audits.append(dict(seed=seed, best_epoch=checkpoint['best_epoch'], checkpoint_sha256=core.sha256(checkpoint_file),
                       optimizer_steps=0, training_performed=False, **parity))
    progress(f'已生成Seed {seed}预测图像；Validation 4920点与历史冻结预测完全一致')
    display(Markdown(f'**Seed {seed}：Development Validation**。时间曲线固定取第24步，'
                     '不拼接重叠窗口；散点包含全部24步查询。'))
    show_png(run / f'seed{seed}/prediction_vs_true.png')
    show_png(run / f'seed{seed}/scatter_true_vs_pred.png')
    del model, checkpoint
    core.torch.cuda.empty_cache()

# %% 9. 三项指标、结果保存与只读完整性验收
summary = core.evaluation_summary(results, 'VALIDATION', seeds)
core.save_csv(run, 'F2_PARTITION_METRICS.csv', results)
core.save_json(run, 'evaluation_results.json', summary)
core.save_csv(run, 'evaluation_results.csv', [{**summary, 'seeds': ','.join(map(str, seeds))}])
for path, expected in source_hashes.items():
    core.require(core.sha256(Path(path)) == expected, 'HISTORICAL_SOURCE_MODIFIED:' + path)
core.save_json(run, 'FROZEN_RUN_CONFIGURATION.json', dict(seeds=seeds, displayed_evaluation_partition='VALIDATION',
    environment=environment, hyperparameters=provenance, seed_population='Validation-selected instances, not an unbiased population',
    plot_rule='fixed step24, original raw voltage recovery; all24 queries used for metrics',
    UI='VS Code Python Interactive / Jupyter image/png display with Agg backend', source_hashes=source_hashes))
progress('已完成指标复算与历史文件SHA检查，输出最终评价指标')
display(Markdown('### 最终结果：Development Validation，R²沿用原始ΔV定义\n'
                 'all为各种子指标的等权平均，不是预测集成。'))
print(f'RMSE: {summary["RMSE_V"]:.6f} V')
print(f'MAE : {summary["MAE_V"]:.6f} V')
print(f'R²  : {summary["R2"]:.4f}')
print(f'结果与PNG已保存：{run}')
core.save_json(run, 'INTERACTIVE_ACCEPTANCE.json', dict(status='PASS', seeds=seeds, self_checks=self_checks,
    seed_audits=audits, ipython_active=get_ipython() is not None, requested_image_displays=len(image_files),
    images=image_files, historical_sources_unchanged=len(source_hashes), new_training=False, new_Optuna_search=False,
    pre_existing_program_changes_preserved=True, summary=summary))
core.save_csv(run, 'ARTIFACT_MANIFEST.csv', [dict(path=path.relative_to(run).as_posix(), sha256=core.sha256(path))
    for path in sorted(run.rglob('*')) if path.is_file() and path.name != 'ARTIFACT_MANIFEST.csv'])

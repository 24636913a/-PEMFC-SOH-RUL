"""Summarize completed, independently verified runs without changing the protocol."""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parents[1]
ROOT = FOLDER.parents[1]
RECEIPTS = FOLDER / 'fresh_training_acceptance'


def csv(path): return pd.read_csv(path, float_precision='round_trip')


if __name__ == '__main__':
    report = json.loads((RECEIPTS / 'ACCEPTANCE.json').read_text(encoding='utf-8'))
    assert report['status'] == 'PASS', 'Do not summarize incomplete training as completed results'
    source = ROOT / 'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
    frame = pd.read_csv(source)
    all_run = FOLDER / next(row['run'] for row in report['cohorts'] if row['cohort'] == 'all')
    first_run = FOLDER / next(row['run'] for row in report['cohorts'] if row['cohort'] == 202)
    means = csv(all_run / 'HOLDOUT_MEAN_METRICS.csv')
    details = csv(all_run / 'HOLDOUT_PER_SEED_METRICS.csv')
    paired = csv(all_run / 'TOP11_MINUS_TOP10.csv')
    points = csv(all_run / 'seed202/top10/predictions_TEST.csv').drop_duplicates('target_row').copy()
    protocol = json.loads((all_run / 'PROTOCOL_BEFORE_TRAINING.json').read_text(encoding='utf-8'))
    edges = protocol['descriptive_voltage_groups']['edges_V']
    points['voltage_group'] = np.searchsorted(edges, points.true_voltage_V)
    points['target_zero_current_fraction'] = frame.zero_current_fraction.to_numpy()[points.target_row]
    groups = []
    for group in range(4):
        block = points.loc[points.voltage_group == group]
        groups.append(dict(voltage_group=group, unique_target_times=len(block),
            true_voltage_mean_V=float(block.true_voltage_V.mean()),
            current_A_median=float(block.target_actual_current_A.median()),
            current_density_median_A_cm2=float(block.target_actual_current_density_A_cm2.median()),
            current_density_P10_A_cm2=float(block.target_actual_current_density_A_cm2.quantile(.1)),
            current_density_P90_A_cm2=float(block.target_actual_current_density_A_cm2.quantile(.9)),
            mean_zero_current_fraction=float(block.target_zero_current_fraction.mean())))
    associations = dict(unique_holdout_target_times=len(points),
        Spearman_voltage_current_density=float(points.true_voltage_V.corr(points.target_actual_current_density_A_cm2,method='spearman')),
        Train_Pearson_mean_current_vs_mean_current_density=float(frame.iloc[:2690].actual_current_A_mean.corr(
            frame.iloc[:2690].actual_current_density_A_cm2_mean)),
        voltage_group_count=4, grouping_fitted_only_on_train=True,
        observed_current_only_used_for_post_hoc_coloring=True,
        groups_are_not_proof_of_exactly_four_physical_states=True)
    pd.DataFrame(groups).to_csv(RECEIPTS / 'VOLTAGE_LOAD_GROUPS.csv', index=False, float_format='%.17g')
    (RECEIPTS / 'LOAD_ASSOCIATION.json').write_text(json.dumps(associations,indent=2)+'\n',encoding='utf-8',newline='\n')
    tables = []
    for cohort, run in [('Seed202_gate',first_run),('six_seed_all',all_run)]:
        for record in csv(run / 'HOLDOUT_MEAN_METRICS.csv').to_dict('records'):
            tables.append(dict(cohort=cohort, **record))
    pd.DataFrame(tables).to_csv(RECEIPTS / 'FINAL_MEAN_METRICS.csv', index=False, float_format='%.17g')
    train_records = []
    for cohort in report['cohorts']:
        run = FOLDER / cohort['run']
        for ledger_file in sorted(run.glob('seed*/top*/TRAINING_LEDGER.json')):
            row = json.loads(ledger_file.read_text(encoding='utf-8'))
            train_records.append(dict(cohort=str(cohort['cohort']),seed=row['seed'],feature_mode=row['feature_mode'],
                best_epoch=row['best_epoch'],stop_epoch=row['stop_epoch'],optimizer_steps=row['optimizer_steps'],
                training_minutes=row['training_seconds']/60,checkpoint_sha256=row['best_checkpoint_sha256']))
    pd.DataFrame(train_records).to_csv(RECEIPTS / 'ALL_14_TRAINING_RECORDS.csv',index=False,float_format='%.17g')
    header='| Seed | 特征 | RMSE（V） | MAE（V） | R²（ΔV） |\n|---|---|---:|---:|---:|\n'
    rows=''.join(f'| {r.seed} | {r.feature_mode} | {r.RMSE_V:.9f} | {r.MAE_V:.9f} | {r.R2:.9f} |\n' for r in details.itertuples())
    mean_text=''.join(f'| 六种子等权平均 | {r.feature_mode} | {r.RMSE_V:.9f} | {r.MAE_V:.9f} | {r.R2:.9f} |\n' for r in means.itertuples())
    difference = paired.loc[paired.seed.astype(str)=='mean'].iloc[0]
    win_rmse=int((paired.loc[paired.seed.astype(str)!='mean','RMSE_V_top11_minus_top10']<0).sum())
    win_mae=int((paired.loc[paired.seed.astype(str)!='mean','MAE_V_top11_minus_top10']<0).sum())
    group_table=pd.DataFrame(groups).to_string(index=False)
    text=f'''# 新协议真实训练与评价总结

本轮完成14次独立训练：先Seed202的Top10/Top11完整验收，再从头执行all模式12次训练。全部使用450轮上限、patience40；没有旧权重热启动、没有旧预测回放，没有根据后留结果调参或改掩码。

数据3843行，Train[0,2690)、Validation[2690,3074)、后留[3074,3843)，合法窗口1750/218/527。后留12648个查询/模型。该区间受过历史研究反馈，不是独立盲测。新协议结果与旧协议分开保存。

## all模式逐种子与等权平均

{header}{rows}{mean_text}
Top11减Top10的六种子平均差：RMSE {difference.RMSE_V_top11_minus_top10:.9f} V；MAE {difference.MAE_V_top11_minus_top10:.9f} V；R² {difference.R2_top11_minus_top10:.9f}。Top11的RMSE在{win_rmse}/6个种子更低，MAE在{win_mae}/6个种子更低。这里只作探索性描述，不选择最优种子，不把六个历史筛选种子视为独立寿命样本或无偏随机种子总体。

## 四个电压分组与负载

分组中心与边界只使用Train的真实电压拟合，k=4预先固定，仅用于描述，不是新增预测模型。着色使用目标时刻实际电流密度，只用于事后诊断，未进入预测器。

```text
{group_table}
```

按唯一后留目标时间统计，避免把重叠查询当成独立工况事件。电压与实际电流密度的Spearman相关为{associations['Spearman_voltage_current_density']:.6f}。应结合各组负载中位数与P10/P90判断关联及重叠；四组聚集本身既不能证明模型异常，也不能证明恰好四种物理状态或不可逆退化。所有原始散点均保留。

从低电压组到高电压组，平均零电流占比依次为{', '.join(f"{row['mean_zero_current_fraction']*100:.2f}%" for row in groups)}。这说明分箱电压聚集与负载占空和聚合混合有关，不能把半小时均值的跨组关系误当作同一稳态极化曲线。Top11新增的平均电流与Top10已有平均电流密度，在Train中的Pearson相关为{associations['Train_Pearson_mean_current_vs_mean_current_density']:.9f}；追加该通道不等于增加一个独立传感器，误差变化应按实际输入表示的受控实验解释。

## 实现与显示证据

主入口F2_Train_Interactive.py内嵌原F2全部必要模型类，参数量332436；没有旧项目Python依赖。Train-only scaler、固定掩码、因果EWMA alpha=0.3、48→24和ΔV保持一致。训练过程中有真实backward/step及权重变化证据，训练曲线由本次Epoch日志持续刷新。已执行Notebook保存真实IPython显示及更新输出；未声称做过VS Code GUI点击验收。

完整训练记录、均值、配对差值和负载统计均有CSV；ACCEPTANCE.json记录独立新checkpoint重放、预处理/指标复算及历史保护检查。

## 旧文件保留与后续归档建议

旧RealHyFC_F2_Main.py、F2_VSCode_Interactive.py及Notebook、Seed Assets、测试、历史权重和结果全部保留。建议以后仅将旧入口标为“历史冻结回放”，保留新入口为正式训练；若需移动归档，应另列文件清单、相对路径依赖和备份SHA后请用户确认。本轮不执行删除、覆盖或移动归档。
'''
    (RECEIPTS / 'FINAL_RESULTS.md').write_text(text,encoding='utf-8',newline='\n')
    print(pd.DataFrame(tables)[['cohort','feature_mode','RMSE_V','MAE_V','R2']].to_string(index=False))
    print('FINAL_RESEARCH_RESULTS_WRITTEN_NO_PROTOCOL_CHANGE')

# P3 — 完整450 Epoch敏感性实验

## 范围与身份

POST_P2_EXPLORATORY_EPOCH_SENSITIVITY。Base c9b7ecce5e76c9fcdcc74dac414ea0c7ad1d85c0，Protocol b513da7468ddf2b22d7b6d92c90d0549d6c14e2571787bb9f416e617710c7467，code identity a3dcc80324d6461724b6e7f0c86d9e609f3cfdc4f74e0f5931a128572fdb0d3b。
九次新鲜初始化训练均450 epoch，总4050，无重试。P2/F2/所有旧阶段只读，旧权重没有打开。
与P2对应初始权重、核心/门控及CPU/CUDA/Python/NumPy/loader RNG身份完全一致；本轮三组每seed实际450轮origin顺序完全一致。
CUDA_STRICT_BITWISE_DETERMINISM_NOT_CLAIMED：初始身份一致不代表与P2独立运行逐步bitwise一致。P2与P3没有拼接或从旧权重续训。

## 唯一训练改变

不因patience40停止；其余H/P1/P2配置和48→24 ΔV任务固定。Best-of-450为主，Last-450为强制敏感性。
best监测标准化Validation RMSE < best−1e−12。first hypothetical p40记录本条P3轨迹首次stale>=40时刻，并非P2实际stop。
AdamW、batch32、clip1、无scheduler/AMP、E→C eval/no_grad/batch32预测路径全部保留。

## 完整九训练、两种保存规则

| checkpoint_role | model | seed | RMSE_mV | MAE_mV | endpoint_12h_RMSE_mV | target_time_equal_RMSE_mV | P95_abs_error_mV | P99_abs_error_mV | max_abs_error_mV | best_epoch | selected_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best | G0 | 42 | 8.426761 | 6.133565 | 7.632165 | 8.648812 | 16.071430 | 23.437698 | 130.884835 | 437 | 437 |
| best | G0 | 52 | 4.521650 | 3.469026 | 4.743246 | 4.886244 | 9.042881 | 13.400761 | 21.945482 | 414 | 414 |
| best | G0 | 62 | 7.394893 | 3.453704 | 20.110940 | 22.888979 | 8.895256 | 14.281684 | 304.916823 | 380 | 380 |
| best | G1 | 42 | 7.708970 | 5.099357 | 20.923101 | 19.208659 | 13.226109 | 19.213468 | 283.597939 | 445 | 445 |
| best | G1 | 52 | 4.349331 | 3.181819 | 4.442866 | 4.818952 | 8.511870 | 13.883474 | 40.524751 | 437 | 437 |
| best | G1 | 62 | 8.346741 | 3.914094 | 24.555798 | 22.354177 | 10.060047 | 16.926554 | 308.080764 | 432 | 432 |
| best | G2 | 42 | 6.058107 | 3.177588 | 21.569211 | 14.441072 | 8.306960 | 13.655300 | 301.958682 | 407 | 407 |
| best | G2 | 52 | 8.033946 | 5.651349 | 8.395523 | 9.255397 | 14.430053 | 23.886399 | 124.052197 | 324 | 324 |
| best | G2 | 62 | 8.412698 | 3.183495 | 29.297300 | 26.667221 | 7.901004 | 13.004231 | 308.134382 | 395 | 395 |
| last450 | G0 | 42 | 9.451447 | 4.878363 | 29.638494 | 26.880425 | 12.184763 | 16.098079 | 304.742980 | 437 | 450 |
| last450 | G0 | 52 | 6.187082 | 3.816607 | 11.318363 | 6.182124 | 9.850244 | 15.385900 | 146.305570 | 414 | 450 |
| last450 | G0 | 62 | 8.309131 | 4.615403 | 20.527997 | 23.355936 | 11.483251 | 15.523980 | 309.755904 | 380 | 450 |
| last450 | G1 | 42 | 8.471016 | 5.673670 | 21.871457 | 19.571805 | 14.944430 | 21.542958 | 284.487061 | 445 | 450 |
| last450 | G1 | 52 | 6.590278 | 3.777610 | 19.855600 | 18.412427 | 10.247085 | 16.458308 | 269.774221 | 437 | 450 |
| last450 | G1 | 62 | 11.110552 | 3.945493 | 43.694917 | 27.969576 | 10.233895 | 16.550022 | 459.009681 | 432 | 450 |
| last450 | G2 | 42 | 24.284085 | 6.821858 | 23.963240 | 29.628665 | 17.296439 | 53.807438 | 449.854408 | 407 | 450 |
| last450 | G2 | 52 | 9.304464 | 4.356757 | 29.899210 | 27.189271 | 11.332868 | 17.989385 | 307.619513 | 324 | 450 |
| last450 | G2 | 62 | 10.591337 | 4.427306 | 29.682471 | 27.612160 | 12.016221 | 22.306586 | 304.590213 | 395 | 450 |


## 三Seed等权均值与样本SD（ddof=1）

| checkpoint_role | model | mean | sample_SD | min | max |
| --- | --- | --- | --- | --- | --- |
| best | G0 | 6.781101 | 2.023618 | 4.521650 | 8.426761 |
| best | G1 | 6.801681 | 2.147604 | 4.349331 | 8.346741 |
| best | G2 | 7.501584 | 1.264350 | 6.058107 | 8.412698 |
| last450 | G0 | 7.982553 | 1.656505 | 6.187082 | 9.451447 |
| last450 | G1 | 8.723948 | 2.270727 | 6.590278 | 11.110552 |
| last450 | G2 | 14.726629 | 8.301972 | 9.304464 | 24.284085 |


4920预测查询等权RMSE为正式指标；205 origins、251 unique target rows，重叠查询不是独立样本。
target-time-equal是每target先平均平方误差再开根号，不是预测值平均集成。
所有2247/2248及极端查询保留。24个horizon和Top1/5/10 target SSE详表在outputs，不删除异常。

## 六个研究问题

1. G2在P3首次假设pat40停止之后产生新最优的seed数：2/3。逐seed：[{"seed": 42, "hypothetical_stop_epoch": 238, "best_at_hypothetical_stop": {"best_epoch": 198, "best_standardized_RMSE": 0.06383570283651352}, "full450_best_epoch": 407, "new_best_after_hypothetical_stop": true}, {"seed": 52, "hypothetical_stop_epoch": 273, "best_at_hypothetical_stop": {"best_epoch": 233, "best_standardized_RMSE": 0.07148974388837814}, "full450_best_epoch": 324, "new_best_after_hypothetical_stop": true}, {"seed": 62, "hypothetical_stop_epoch": 435, "best_at_hypothetical_stop": {"best_epoch": 395, "best_standardized_RMSE": 0.06149489805102348}, "full450_best_epoch": 395, "new_best_after_hypothetical_stop": false}]。
2. G2 Best-of-450优于P2实际best的seed：[42, 52, 62]，平均P3−P2正式RMSE=-1.641850 mV。
3. 公平450预算下Best参考门槛状态：NO_CONSISTENT_MEAN_GAIN_OVER_BOTH_CONTROLS；Last参考状态：NO_CONSISTENT_MEAN_GAIN_OVER_BOTH_CONTROLS。各组均值与每seed完整公开，不挑保存规则。
4. 是否三个seed均改善：True（相对P2）；是否同时赢两对照三seed，见下面固定门槛表，不用个别seed代替稳定性。
5. 长终点、target-time-equal、极端误差是否同步，以下P3−P2负值才是改善：

| seed | P3_best_minus_P2_endpoint_12h_RMSE_mV | P3_best_minus_P2_target_time_equal_RMSE_mV | P3_best_minus_P2_P99_abs_error_mV | P3_best_minus_P2_max_abs_error_mV |
| --- | --- | --- | --- | --- |
| 42 | -0.088187 | -7.832151 | -8.796472 | 12.271622 |
| 52 | -21.285511 | -17.689621 | 6.174899 | -185.127558 |
| 62 | -0.480319 | -0.213091 | -3.167438 | -1.266952 |


6. 本实验只能说明冻结预算及已复用Validation下的经验敏感性；更长搜索轨迹本身增加best选择机会，不能证明门控学到电化学机理。
优化盆地、延迟gate学习或CUDA数值差异等机制没有独立识别，均只是假设。不能把Best的选优收益说成Last也一致稳定。

## P2工程门槛：仅参照，不修改历史判定

{
  "best": {
    "status": "NO_CONSISTENT_MEAN_GAIN_OVER_BOTH_CONTROLS",
    "all_predefined_criteria_PASS": false,
    "comparisons": {
      "G0": {
        "mean_RMSE_gain_mV": -0.7204823787207317,
        "required_gain_mV": 0.5,
        "mean_gain_PASS": false,
        "paired_RMSE_wins": 1,
        "endpoint_each_seed_PASS": false,
        "target_time_equal_each_seed_PASS": false,
        "mean_MAE_PASS": true,
        "mean_P99_PASS": true
      },
      "G1": {
        "mean_RMSE_gain_mV": -0.6999026822334828,
        "required_gain_mV": 0.5,
        "mean_gain_PASS": false,
        "paired_RMSE_wins": 1,
        "endpoint_each_seed_PASS": false,
        "target_time_equal_each_seed_PASS": false,
        "mean_MAE_PASS": true,
        "mean_P99_PASS": true
      }
    },
    "group_means": {
      "G0": {
        "RMSE_mV": 6.78110128896197,
        "MAE_mV": 4.352098265491584,
        "P99_abs_error_mV": 17.0400477976849
      },
      "G1": {
        "RMSE_mV": 6.801680985449219,
        "MAE_mV": 4.065089954840296,
        "P99_abs_error_mV": 16.674498620753507
      },
      "G2": {
        "RMSE_mV": 7.501583667682702,
        "MAE_mV": 4.004143874629332,
        "P99_abs_error_mV": 16.848643625271595
      }
    },
    "historical_condition_independent_value_supported": false,
    "statistical_significance_claim": false,
    "electrochemical_causality_claim": false,
    "historical_F2_automatically_replaced": false,
    "additional_model_or_training_authorized": false,
    "physics_F2": "PAUSED",
    "SOH_RUL_completed": false,
    "Test_accessed": false,
    "confirmation_eligibility": "NOT_ESTABLISHED_REUSED_DEVELOPMENT_VALIDATION",
    "final_model_selection": "RESEARCH_OWNER_DECISION_REQUIRED"
  },
  "last450": {
    "status": "NO_CONSISTENT_MEAN_GAIN_OVER_BOTH_CONTROLS",
    "all_predefined_criteria_PASS": false,
    "comparisons": {
      "G0": {
        "mean_RMSE_gain_mV": -6.744075714785483,
        "required_gain_mV": 0.5,
        "mean_gain_PASS": false,
        "paired_RMSE_wins": 0,
        "endpoint_each_seed_PASS": false,
        "target_time_equal_each_seed_PASS": false,
        "mean_MAE_PASS": false,
        "mean_P99_PASS": false
      },
      "G1": {
        "mean_RMSE_gain_mV": -6.002680445305177,
        "required_gain_mV": 0.5,
        "mean_gain_PASS": false,
        "paired_RMSE_wins": 1,
        "endpoint_each_seed_PASS": false,
        "target_time_equal_each_seed_PASS": false,
        "mean_MAE_PASS": false,
        "mean_P99_PASS": false
      }
    },
    "group_means": {
      "G0": {
        "RMSE_mV": 7.982553151001078,
        "MAE_mV": 4.436791386881331,
        "P99_abs_error_mV": 15.669319634325838
      },
      "G1": {
        "RMSE_mV": 8.723948420481383,
        "MAE_mV": 4.465590647137435,
        "P99_abs_error_mV": 18.18376299304268
      },
      "G2": {
        "RMSE_mV": 14.72662886578656,
        "MAE_mV": 5.201973731931715,
        "P99_abs_error_mV": 31.367802619934206
      }
    },
    "historical_condition_independent_value_supported": false,
    "statistical_significance_claim": false,
    "electrochemical_causality_claim": false,
    "historical_F2_automatically_replaced": false,
    "additional_model_or_training_authorized": false,
    "physics_F2": "PAUSED",
    "SOH_RUL_completed": false,
    "Test_accessed": false,
    "confirmation_eligibility": "NOT_ESTABLISHED_REUSED_DEVELOPMENT_VALIDATION",
    "final_model_selection": "RESEARCH_OWNER_DECISION_REQUIRED"
  }
}

P2正式 NO_CONSISTENT_MEAN_GAIN_OVER_BOTH_CONTROLS 保留。不自动替换原F2，不升级为独立确认性证据。
O1历史正面结果：F2整体正式Validation RMSE优势可以报告，但为 F2_ADVANTAGE_METRIC_DEPENDENT，非全指标全seed一致；原J冻结结构与L确认性Test不合格限制继续有效。

## 数据与访问

仅N1 nrows3007后age<1819.997738得3005。Train/Validation范围[0,2103)/[2103,2404)，1380/205窗。
不做全N1 hash；沿用允许Development fingerprint、Train-only未滤波scaler、Top10/因果EWMA0.3。
没有Assessment、Test、raw、极化或其它新标签访问，Python审计/CSV入口防线及纯合成测试证据保留。
测试只验证工程防线，不生成科研模型结果。

## 产物与备份

18checkpoint原件和18逐文件SHA一致副本。副本在P3/backups/checkpoints，独立目录但同E卷，并非跨设备灾备。
没有删除唯一副本；权重和batch journal本地保留、Git忽略，清单记录path/bytes/SHA。
全部18预测缓存重读重算、checkpoint精确重载、450history与共同采样完整性PASS。
图表用固定颜色、清晰分栏、300dpi PNG及PDF；训练图展示上限60mV，未截断原始日志或排名数据。

## 完成后明确结论与图表核验

**有限支持长训练增加 G2 的最佳权重机会，不支持“跑满450轮就使动态门控稳定胜出”。**
同一条P3轨迹内，Seed42首次假设早停238轮之前的最佳监测RMSE为8.732928 mV，完整450轮最佳为6.058107 mV（407轮）；Seed52对应9.780025→8.033946 mV（273轮假设停止，324轮最佳）；Seed62仍为8.412698 mV（435轮假设停止，395轮最佳），之后没有新最优。这是本轨迹内的预算敏感性证据，不应与独立P2运行作逐步因果拼接。

与P2实际Best比较，G2三个seed均改善，平均9.143434→7.501584 mV（约17.96%）；但G0/G1的P3 Best均值分别6.781101/6.801681 mV，G2仍高0.720482/0.699903 mV。G2只在Seed42同时胜过两对照，Seed52/62没有，原P2判定不变。

G2长终点和target-time-equal相对P2在三个seed均改善，但极端指标并未同步：Seed42 max AE增加12.271622 mV，Seed52 P99增加6.174899 mV。Last-450 G2三seed均值14.726629 mV，明显高于自己的Best 7.501584 mV和Last两对照7.982553/8.723948 mV；因此不能把选优结果解释成终态权重稳定或动态融合独立有效。

五张图已逐张视觉检查；仅调整P2/P3比较图的文字标注偏移以避免重叠，不改变任何绘图数据、预测、训练代码、协议或指标。报告图采用scientific-figure-making的分栏、统一颜色、300dpi PNG及PDF规范。

大型逐次文件访问事件日志与batch journal完整留在本地并Git忽略；SHA/大小仍在P3_LOCAL_ARTIFACT_AND_BACKUP_MANIFEST.json，精简访问摘要另见audits/P3_RUNTIME_ACCESS_SUMMARY.json。18份权重及18份逐文件核验副本仍分别保存，同E卷，不冒充跨设备灾备。

## 停止

此处完成并停止。不调门槛，不增加融合比例、seed/backbone/Physics-F2，不进入SOH/RUL，不自动论文写作。

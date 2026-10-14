# CPU实际结果

| horizon | model | evaluation_class | n | MAE_SOH_pp | RMSE_SOH_pp | P95_abs_SOH_pp | max_abs_SOH_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | M0_REFERENCE_PERSISTENCE | Exploratory calendar replay | 3 | 0.2811724894230168 | 0.3268026154702232 | 0.4813093703819759 | 0.5116961246122042 |
| H1 | M1_RIDGE96_M2_NC | Exploratory calendar replay | 3 | 0.0797299804370414 | 0.0887214022892251 | 0.1128768492402754 | 0.114368499870654 |
| H3 | M0_REFERENCE_PERSISTENCE | Exploratory calendar replay | 3 | 0.2811724894230168 | 0.3268026154702232 | 0.4813093703819759 | 0.5116961246122042 |
| H3 | M1_RIDGE96_M2_NC | Exploratory calendar replay | 3 | 0.0323853745939906 | 0.0429048773460137 | 0.0658932752912144 | 0.0707141263833754 |
| M0_ALL_EVENTS | M0_REFERENCE_PERSISTENCE | Exploratory calendar replay | 6 | 0.5115824219424747 | 0.7028055500486265 | 1.2800014087760796 | 1.5361031701640382 |

M0全6事件均值不能与M1三事件比较；正式呈现的是H1/H3各自共同的1150/1500/1750三个事件。三事件既少且来自单堆，还存在测量参考和校准假设，不能称独立确认或统计显著。

本轮新校准命名M2_NC，复用aR+b形式，输入对为各历史参考前“最近完整周期”的R。它不同于原AHC三条selected记录的中位数（含partial记录）；改变算子是为保证校准和预测输入一致。原AHC参数/成绩全部保留，只作历史参照。每个事件分别拟合过去3/4/5对，H1/H3参数相同，不调参；保存6份参数记录。96h模型、alpha、权重没有重拟合。

该实验不以真实future P1S作M1输入。target真实R只保存在原缓存用于误差审计。独立程序用矩阵最小二乘复算仿射，用缓存系数/标准化复算Ridge及所有误差。

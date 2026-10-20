# RealHyFC 长期趋势研究：24/48h试运行完成，停止扩大训练

先读 [通俗总结](reports/04_长期预测最终研究总结.md)。本轮真实运行4次Seed202 GPU训练，原生F2未超过LSTM；48h均未通过扩展门槛，六种子/后留/RUL均未执行。条件校正趋势是探索性统计代理，不是SOH或不可逆退化标签。

## VS Code

打开 F2_LongTerm_Interactive.py，选择项目Python3.11/CUDA，使用每个`# %%`的Run Cell或Run All Cells。默认MODE='analyze'只读展示真实CSV和图像；改为'pilot'才在新目录从头运行4个模型，显示实际Epoch/Loss和新曲线。不得为筛选较好成绩反复运行。该入口不import有顶层训练副作用的旧脚本。

## 文件索引

- configs：数据/目标资格规则与后续训练协议，两者均先于对应数据检查/训练冻结。
- scripts/audit_data.py：3843行时钟/有效性；45个窗口配置及全部剔除记录。
- scripts/audit_targets.py：仅前3074行构建目标，Train-only OLS；无后留数值。
- scripts/native_models.py：完整原生F2与普通LSTM定义，不依赖旧项目Python模块。
- scripts/train_longterm.py --pilot：真实GPU训练、反向传播、best保存；每次新目录。
- scripts/verify_independent.py：独立时钟/标签/指标复算与checkpoint逐点回放；--history检查历史保护。
- scripts/render_figures.py：基于CSV生成全部PNG/PDF；scripts/write_reports.py生成五份报告。
- scripts/verify_interactive.py：真实IPython代码单元及PNG显示负载验收，非VS Code GUI截图。
- outputs/qualification_20261020_v1：源数据读取前的缺失历史DLL错误，不删除。
- outputs/qualification_20261020_v2：正式连续性/目标审计、注册SHA、历史保护与失败说明。
- outputs/ACTIVE_PILOT.json：本轮4次训练目录身份；其子目录保留权重、逐点预测、日志及输入/标签/窗口SHA。
- tables：各指标、逐点预测及原始绘图数据。没有六种子平均/后留成绩。
- literature：已上传原文的设计借鉴、SHA及RealHyFC研究终止元数据。

## 图像用途（300dpi PNG + PDF + CSV）

|图|用途|
|---|---|
|F1|完整记录时钟、连续有效段与冻结边界，后留仅元数据|
|F2|短中长期候选的合法起点与不重叠目标覆盖，未训练跨度不补成绩|
|F3|真实原始电压、A/B因果趋势，Train+Validation，不跨断点|
|F4|A/B与历史零电流混合的相关性，非因果识别|
|F5|两模型与保持法24/48h真实pilot指标|
|F6|固定模型每一步预测误差|
|F7|首个共同起点与最差两模型平均SSE起点，完整轨迹不拼接|
|F8|仅固定末步的误差随真实目标运行小时变化|
|F9|所有重叠查询残差，保留尾部|
|F10|本轮F2的真实历史门控，不宣称独立因果收益|
|F11|本轮四次训练Loss，不回放历史训练|
|F12|真实目标运行状态与末步误差，未来状态仅事后诊断|

## 复核命令

从本目录执行 `python -B -m unittest discover -s tests -v`、`python -B scripts/verify_independent.py`。图表复绘不训练：`python -B scripts/render_figures.py`。只在需要新的预登记研究时执行`python -B scripts/train_longterm.py --pilot`；它不会续训或覆盖任何旧权重。

本研究不同于旧12h原始动态电压任务，RMSE不能跨目标直接排名。后留已历史查看，非独立盲测；本轮没有后留推理。没有真实EOL标签，不训练真实RUL。历史文件不删除、不覆盖、不移动。

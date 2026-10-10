# Phase G execution blocked

{
  "status": "PHASE_G_EXECUTION_BLOCKED",
  "ledger": {
    "status": "PHASE_G_EXECUTION_BLOCKED",
    "trajectories": {
      "G-E-A": {
        "status": "BLOCKED",
        "returncode": 1
      }
    },
    "max_epochs": 48,
    "test_accessed": false,
    "new_assessment_inference": false
  },
  "no_extra_training": true,
  "new_assessment_inference": false,
  "test_accessed": false
}

Strict deterministic execution was not silently relaxed. See trajectory TECHNICAL_FAILURE.json; no historical cause can be inferred. No further runs authorized.

## A. 历史事实

历史 Epoch 1–3 的 Train MSE 与 Validation 标准化 RMSE 完全相同。Epoch 4 首次出现差异：Train MSE 绝对差 `6.773255289971303e-09`；Validation 标准化 RMSE 绝对差 `2.980232238769531e-07`。后者不是 mV，其约为 `0.0000407705 mV`。完整逐 epoch 外连接表保留了 E2 的全部日志和 F0 的全部日志；一方停止后的空值不是缺失结果的补造。

E2/F0 最佳 epoch 分别为 287/165。安全 `weights_only=True` 加载通过；参数名、形状和 323483 参数量一致，初始化摘要及目标尺度一致。最终权重差异符合不同最佳 epoch，不能据此认定错误。既存 Assessment 缓存的 9072 个预测键、真值逐项一致；缓存重算 RMSE 为 E2 `7.785820667999225 mV`、F0 `21.413261037197774 mV`。它们只是既有结果，不是本轮模型推理。

代码比对显示 forward/MSE/backward/clip/AdamW/Validation 与早停数学顺序相同。E2 在内存保存 CPU clone，最终写盘；F0 在每次改善写盘，并增加梯度有限检查、ID/哈希/日志。F0 明确 CPU threads=1；E2 未保存实际历史线程数。E2 的 Python/CUDA/cuDNN/GPU/TF32/严格确定性开关及首批次哈希缺少独立历史证据，标记 `HISTORICAL_RUNTIME_EVIDENCE_UNAVAILABLE` / `HISTORICAL_FIRST_BATCH_FINGERPRINT_UNAVAILABLE`。

## B. 本轮受控执行

协议在执行前冻结，SHA256 为 `21e27f8fb99d93f77d1ff1e098476a4d0050ccad442bb077a8380ca12961febb`。7 项防线测试全部通过；仅 `nrows=3007 → age<1819.997738 → 3005 rows` 读取，未完整哈希 N1，未构造 Assessment 窗口。Train/Validation 分别 1380/205 窗口，Top10/scaler/数据指纹与历史 F0 对齐；当前 G-E-A 初始化 digest 与历史 E2/F0 一致。

受控环境为 Python3.11.9、Torch2.6.0+cu124、pytorch-forecasting1.8.0、CUDA12.4、cuDNN90100、RTX3070 Laptop、CPU threads=1。导入 Torch 前配置 CUBLAS `:4096:8`；严格确定性 `True`、warn_only=False；cuDNN deterministic=True、benchmark=False、两类 TF32=False。

G-E-A 在 epoch1 / batch0（从零计数）的 Backward 报 `upsample_linear1d_backward_out_cuda does not have a deterministic implementation`。Forward 已执行，Backward 未完成，Optimizer Step 尚未发生；完成 epoch=0、参数更新=0。其余三条轨迹未启动。

|比较|是否一致|最早分叉 batch|说明|
|---|---|---|---|
|G-E-A vs G-E-B|无法评估|不可用|G-E-A 首批次阻断，G-E-B 未运行|
|G-F-A vs G-F-B|无法评估|不可用|均未运行|
|G-E-A vs G-F-A|无法评估|不可用|未形成可比轨迹|

这不是“路径重复分叉”或“受控环境全部一致”。未关闭确定性、未用 warn_only、未切换 CPU、未修改历史模型插值实现，也未追加非确定性尝试。

## C. 原因分层

已证实：历史第4 epoch 分叉；现有 CUDA 运算阻断严格确定性 Backward。上游 `TimeDistributedInterpolation.interpolate` 确实包含 `F.interpolate(mode='linear', align_corners=True)`，与报错算子类型吻合，但没有采集原历史运算的因果干预证据。

有证据支持但未确定：历史运行可能受底层数值/执行路径敏感性影响，显式线程控制及状态保存/同步时机是候选差异。

尚无法判断：历史第4 epoch 分叉是否由该算子、CPU线程、CUDA调度或其他状态造成。不能把“该算子不支持严格确定性”写成“已证实该算子造成历史分叉”。历史首个分叉 batch、forward/backward/step 位置与 RNG 提前分叉均不可定位。

## 后续研究交接

1. E2 的低 RMSE 尚无充分独立重复/多 Seed 稳定性证据；本轮没有复现 epoch287 最优成绩。
2. F0/F1/F2 仍为单 Seed 的 Development 架构探索，Assessment 已被复用；F2 对 F1 的改善不能推导为对 F0 的整体优势，历史跨阶段复现性限制仍存在。
3. 如需稳定性结论，应另行明确授权独立多 Seed / 可执行的确定性方案；本轮不启动，也不自行选择替代路径。
4. 当前证据不足以扩展 TFT 架构创新或推翻 `TIMESNET_FREEZE` / `BACKBONE_SEARCH_CLOSED`。

正式状态：`PHASE_G_EXECUTION_BLOCKED`。停止，无 Test、新 Assessment 推理、额外训练、SOH 或 RUL。

---
name: pemfc
description: PEMFC退化预测论文的核心项目状态、最终模型方案、双任务分工、关键实验结论
metadata: 
  node_type: memory
  type: project
  originSessionId: fd9c4c25-3b86-4695-96ab-02bce7eba91f
  modified: 2026-09-03T09:49:59.488Z
---

# PEMFC 退化预测论文 — 项目状态（截至 2026-09）

## 项目背景
- 课题：质子交换膜燃料电池（PEMFC）退化状态估计与剩余寿命预测
- 开题报告题目（暂定）：质子交换膜燃料电池健康状态估计与剩余寿命预测研究
- 数据集：IEEE PHM 2014（FC1 准静态 ~1154h + FC2 电流纹波 ~1020h，24 传感器）
- 导师否定了"纯 TFT 用于 PEMFC"——因为 Chou & Wang 2025 (Renewable Energy) 已发表 TFT 做 PEMFC RUL
- 参考论文文件夹：`参考论文/`（含 Chou & Wang TFT、RF-TCN、M-LSTM 等）

## 最终模型方案
- **主模型 = Adaptive GRN + MCLSTM TFT**（L=1 + lag 增强）
- 四个组件改进的定论：
  - **Adaptive GRN**：✅ 核心创新（GRN 门控受电压波动动态得分调制）
  - **MCLSTM**：⚠️ 降级为探索性设计（LSTM cell state 单调约束，非 GPILSTM）——对状态估计微益、对 RUL 有害，论文定位为「物理约束向内部状态迁移的探索/negative result」，不与 Adaptive GRN 并列为主创新
  - **Adaptive VS**：❌ 负结果（两工况都微降，已放弃）
  - Attention / LSTM 结构：⚠️ 已推翻「不动」——改为 Condition-aware LSTM（见下）

## 编码器方向（当前最关键决策）
- 消融核对发现：主模型（LSTM 编码器）全面落后 ChouWang-BiLSTM-TFT（FC1 0.9069 vs 0.9383；FC2 0.9767 vs 0.9849）
- 根因：L=1 下 LSTM cell state 退化——初始 c_0=0、遗忘门失效，退化为带门控 MLP；BiLSTM 只是「2 倍宽度非线性映射」
- 否决：BiLSTM（Chou & Wang 已做）、Mamba（2025 已有 KF-Mamba PEMFC）、TCN/GRU（Sahajpal 已系统比较）
- 选定：Physics-/Condition-aware LSTM——退化状态向量 d 初始化 cell state(c_0≠0) + 退化门控 α/β 调制遗忘/输入门，让 cell state 真正承载退化状态
- 备选：sLSTM(xLSTM，更强门控非线性，但需手写、L=1 下长期记忆发挥不出)
- 三设计闭环：Condition-aware LSTM(状态更新) + Adaptive GRN(信息融合) + MCLSTM(状态约束)
- 下一步：写 test_condition_lstm.py，FC1+Loop20 验证 Min R² 是否稳超 ChouWang 0.8496

## 双任务分工（重要）
- **FC2 = 退化状态估计**（主任务）：纹波信息密度高，Adaptive GRN 在这里最强
- **FC1 = 寿命预测**（RUL）：teacher forcing 在线监测（one-step-ahead，对标 Sahajpal 2023/RF-TCN 口径，**非** Chou & Wang 的真未来递归滚动），挑退化段+强去噪，副本22 双头 λ_cell=0 架构统一版平均 RE≈1.22%（单头版 0.68%）；EOL 活化峰值定义=807h
- FC2 不适合做 RUL：EOL=393.5h 落在训练集里，测试集全程高于阈值

## 关键实验结论
1. L=1 是最优范式：TimeMixer L=5/10/20 全负，FT-Transformer FC1 Loop50 崩到 -6.94
2. 去噪工况依赖（核心发现）：FC2 去噪有害（纹波是信息），FC1 去噪有益（波动是噪声）
3. MCLSTM 对状态估计微益、对 RUL 有害（cell 约束把预测钉在阈值上方）
4. 评估口径差异：Chou & Wang 用 RE%+单次，你坚持 Loop 50 + R²，复现出他们 Mean R²=-1.31
5. RUL 三范式对比：自回归(无泄露真外推)=RE 369%崩(SNR<1 学不到退化率)；物理外推(无泄露)=RE≈24%；teacher forcing 在线监测(one-step-ahead，对标 Sahajpal/RF-TCN，非 Chou&Wang 真未来递归)=挑退化段 Sp=400/500/600 平均 RE≈0.68%
6. RUL 最终选 teacher forcing 在线监测（领域惯例，论文诚实标注"滑动窗口+真实观测值回填"），副本22(FC1版最终版Ⅱ)双头 TFTModel+λ_cell=0 架构统一版，平均 RE≈1.22%
7. 特征泄露判定：ΔUtot/SOH 用当前 y 算（技术上算泄露），但删掉后 FC2 Mean R² 0.9767→0.9677、FC1 0.907→0.821 均净降；结论=保留 6 衍生特征，主文件不删（FC2 含特征即定稿）
8. 前端滤波探索（全负，已关闭）：普通 EKF、VB-AKF、Multi-scale SG、Adaptive SG、SG+VB-AKF 结合，在 FC1/FC2 全部无稳定正增益（Mean 均 ≤+0.005 或为负，Min 多翻负）。关键教训：Loop 20 正增益会在 Loop 50 缩水/翻负（VB-AKF Loop15 +0.03→Loop50 -0.006；Adaptive SG FC2 Loop50 +0.005/FC1 Loop50 -0.014）。结论=SG(21) 固定窗口是唯一最优前端，主文件无需前端改动
9. 消融代码完整：30 个脚本（8通用模型+2前人+5 TFT变体，×2工况），全部 Loop 50 + 500epochs + SWA + 记录格式 [FCx] 模型名，结果自动追加到消融结果汇总.txt。生成器=`生成消融脚本.py`（统一模板+15模型，可一键重生成）
10. 消融核对（500epochs+SWA，2026-09-03）：主模型完全复现主文件成绩（FC1 0.9069/FC2 0.9767），但全面落后 ChouWang-BiLSTM-TFT（FC1 0.9383/FC2 0.9849）。物理约束 FC1 有害（纯MSE 0.9414>主模型0.9069）；MCLSTM 两工况≈0。FC2 TFT-minus-VS 漏跑（FC1 跑两次）待补。→ 触发编码器方向决策（见上）

## 两工况 Loop 50 成绩（最终 Adaptive GRN+MCLSTM，含 6 衍生特征）
- FC2：Min R²=0.9458, Mean=0.9767, Max=0.9939
- FC1：Min R²=0.7786, Mean=0.9069, Max=0.9688

## 代码位置
- 主代码：`代码/New-PEMFC-Pytorch/New-PEMFC-Pytorch.py`（定稿状态）；副本21=FC2版最终版Ⅲ（状态估计），副本22=FC1版最终版Ⅱ（状态评估+teacher forcing RUL，含运行时间输出）
- 消融脚本：`消融实验代码/`（30个脚本，Loop 50 + 500epochs + SWA，由 `生成消融脚本.py` 一键生成）
- 测试脚本：`测试文件/`（L>1扫描、去噪扫描、FC1 RUL、HI+物理外推 test_fc1_rul_physics.py、特征泄露验证 test_feature_leak.py）
- 消融结果：`消融实验outputs/消融结果汇总.txt`（旧 300epochs 结果备份为 消融结果汇总_300epochs备份.txt）
- 论文大纲：`文章章节/论文大纲.md`（v6 问题驱动版，Adaptive GRN 唯一主创新，MCLSTM 降级）
- 参考论文总结：`参考论文/参考论文总结_2026-09-02.md`（15 篇唯一，含 Li 2023 补入）

## 相关
- [[导师反馈-论文结构]] — 导师要求论文结构对标参考文献
- [[Chou-Wang-TFT对比]] — 与已发表 TFT 论文的差异化分析

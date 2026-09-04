---
name: chou-wang-tft
description: 与 Chou & Wang 2025 TFT 论文的差异化分析、评估口径对比、数据集L取值争议
metadata: 
  node_type: memory
  type: project
  originSessionId: fd9c4c25-3b86-4695-96ab-02bce7eba91f
  modified: 2026-08-15T08:18:16.486Z
---

# 与 Chou & Wang (2025) TFT 论文的对比

## 那篇论文做了什么
- 首次将 TFT 用于 PEMFC RUL，唯一架构改动是 LSTM→BiLSTM（其余组件保留）
- 训练段用 Bacon-Watts 拐点检测挑"退化加速段"（FC2 场景1 仅 55 个样本）
- 预处理：median 滤波 + SG + RF 特征选择 + NNI AutoML
- 评估只用 RE%（相对误差），报单次结果，不报 Loop 50

## 关键差异（你的优势）
1. 你用 L=1+lag，他用滑动窗口 L>1
2. 你用 Loop 50 Min/Mean/Max R²，他报单次 RE%
3. 你做了 8 模型×2工况消融，他没有
4. 你做四重可解释性 + MC Dropout，他都没做

## 重要发现：他们的"成功"是评估口径造成的
- 复现他们的 L=10+τ=1 路线，用 Loop 50 + R² 口径：Mean R²=-1.31，Min=-14.35
- 他们的 RE=0.006%~0.066% 靠的是：松指标(RE%) + 单次 + 挑退化段 + 纹波可预测性
- 结论：你的评估严谨性(Loop 50)是比"精度更高"更核心的贡献

## 数据集 L 取值争议
- 所有参考文献都没明确写序列窗口 L 是多少（领域通病）
- 你无法"对齐他们的 L"，只能靠自己的 L 扫描实验
- 这是可写进论文的点："现有工作普遍未报告序列窗口设置依据"

## 相关
- [[PEMFC论文项目状态]] — 项目主状态

# 任务级区域资格

冻结任务：`PEMFC_DYNAMIC_MULTISTEP_DELTA_V_FORECASTING`；RealHyFC；0.5h；48×14历史；24步未来ΔV。不是 SOH/RUL 或长期单调退化。

| 区域 / 任务 | 分类 | 证据与允许表述 |
| --- | --- | --- |
| Train/Validation/Assessment <1819.997738h，F2 | NOT_ELIGIBLE_AS_CONFIRMATORY_TEST | 已用于训练、选择和复用；H/I2/J/K；仍可称 Development 结果 |
| 既有 late-life >=1819.997738h，F2 | TARGET_OR_METRIC_EXPOSED；NOT_ELIGIBLE_AS_CONFIRMATORY_TEST | RUL90 明确 full voltage/current 描述；AHC raw读取/后验支持诊断。后验路线影响无法排除；未证明 F2 用 Test监督训练 |
| AHC 当时 nominal 2000/2300 polarization | 原工作流目标盲，但原确认性 one-shot 已消费 | 原 ledger StageP aborted；不得重试；不能推广为 dynamic-voltage-blind |
| 当前2000/2300极化健康任务 | TARGET_OR_METRIC_EXPOSED；POSTHOC_SELECTION_INFLUENCED；NOT_ELIGIBLE_AS_CONFIRMATORY_TEST | ELTE exploratory protocol 在失败/支持诊断后制定；两目标及误差已打开 |
| 仅文件名/metadata 访问本身 | UNKNOWN_ELIGIBILITY | 不足以证明数值/选择暴露或独立 |
| 尚未识别的外部数据/新stack | UNKNOWN_ELIGIBILITY | 未提供合格 source/访问史/同任务协议，不自动宣布 independent |

## 为什么不是“时间更晚就独立”

Chronological ordering 保证方向，不保证未参与研究反馈；同一 stack 也只评估 within-stack temporal generalization，三个Seed、多cell、重叠窗口都不是独立堆寿命样本。

特征接触但目标保持盲时，可能仍支持特定 target-blind 描述，但需要记录影响、事先冻结用途。这里不是只有 filename 或工况：full动态电压已有明确访问，健康目标后来也打开。把 Test 改名 temporal holdout 不消除使用史，只能诚实标注 post-hoc/exploratory。

综合状态 `PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE` 针对**已识别的既有晚期候选区域**。不代表所有 PEMFC 数据都不合格，不代表 F2 失败，不宣称完整 supervised leakage。没有已证实的另一个独立 RealHyFC 子区间；由于 full raw access，不能通过就地划子区间洗白。

本轮只能 BLOCKED；不能进入 READY_FOR_EXPLICIT_AUTHORIZATION。若以后选择探索性 holdout，需单独公开其已暴露性质并明确授权；若要确认性验证，优先新stack/外部独立动态数据，先冻结源访问史和信息集合，再另行授权。

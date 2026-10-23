# Phase P — 全部成果路径

阶段绝对目录：`E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_P_Final_Innovation_and_Experiment_Planning`。

## 七份科研报告

1. [P_CURRENT_RESEARCH_STATUS.md](P_CURRENT_RESEARCH_STATUS.md)
2. [P_GATE_INDEPENDENT_ABLATION_PLAN.md](P_GATE_INDEPENDENT_ABLATION_PLAN.md)
3. [P_PHYSICS_LITE_CANDIDATE_REVIEW.md](P_PHYSICS_LITE_CANDIDATE_REVIEW.md)
4. [P_PHYSICS_LITE_EQUATION_AND_INFORMATION_AUDIT.md](P_PHYSICS_LITE_EQUATION_AND_INFORMATION_AUDIT.md)
5. [P_MATCHED_EXPERIMENT_DESIGN.md](P_MATCHED_EXPERIMENT_DESIGN.md)
6. [P_FINAL_GO_NO_GO_DECISION.md](P_FINAL_GO_NO_GO_DECISION.md)
7. [P_SUPERVISOR_PRESENTATION_PREPARATION.md](P_SUPERVISOR_PRESENTATION_PREPARATION.md)

## 提案与可复核审计

- [README.md](README.md)
- [protocol/P_PROPOSED_GATE_PROTOCOL.json](protocol/P_PROPOSED_GATE_PROTOCOL.json)：三组九次，G1同型固定上下文global gate；明确未获训练授权。
- [protocol/P_PROPOSED_GATE_PROTOCOL.sha256](protocol/P_PROPOSED_GATE_PROTOCOL.sha256)
- [evidence/P_STATIC_SOURCE_MANIFEST.json](evidence/P_STATIC_SOURCE_MANIFEST.json)：19个静态仓库依据和1个原PDF身份；不读取权重/预测cache。
- [outputs/P_STATIC_PLANNING_AUDIT.json](outputs/P_STATIC_PLANNING_AUDIT.json)
- [outputs/P_DOCUMENT_GUARD_TEST_RESULTS.json](outputs/P_DOCUMENT_GUARD_TEST_RESULTS.json)
- [outputs/P_DOCUMENT_GUARD_TEST_LOG.txt](outputs/P_DOCUMENT_GUARD_TEST_LOG.txt)
- [scripts/run_p_static_audit.py](scripts/run_p_static_audit.py)
- [tests/test_p_document_guards.py](tests/test_p_document_guards.py)
- [.gitignore](.gitignore) / [.gitattributes](.gitattributes)
- [P_DELIVERABLE_INDEX.md](P_DELIVERABLE_INDEX.md)

`runtime/`只有内部原论文公式页渲染，不提交，不作为本项目原创图；没有新预测、模型训练、物理拟合、PPT或论文正文产物。

本轮19静态来源哈希与基准版本核对，九checkpoint身份仅查询历史manifest。技术审计和7项文档防线测试不构成模型性能/梯度实测或真实实验运行验收。

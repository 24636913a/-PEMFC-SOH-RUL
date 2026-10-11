# Phase L — 任务级独立评价资格与复现冻结

Base: `1fe9ce87ecc33ac297ff0e4a14cd4ca7a8ada3a6`。本阶段只新增文档、身份清单及合成防火墙测试，不打开 Test、不加载模型、不推理、不训练。

最终科学状态：`PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE`。

范围限定：现有 RealHyFC 晚期区域（名义 Development 边界 1819.997738 h 之后）不能作为当前 F2 的 pristine confirmatory Test。没有证明 F2 曾用 Test 标签训练；实际证据是既有目标相关电压访问、晚期诊断反馈与后续探索性健康目标揭盲，无法排除项目路线层面的后验影响。外部独立数据资格尚未识别，不按合格处理。

## 运行

在本目录运行：

```powershell
python -B tests/test_l_synthetic_firewall.py
python -B scripts/audit_l_readonly.py
```

`--freeze` 仅用于本阶段第一次冻结（要求 base HEAD、tracked tree clean）；提交后复核不要求 HEAD 永远等于旧 base，要求 base 在祖先链且冻结输入/代码 SHA 一致。不要重新运行历史阶段 runner。

协议文件使用 JSON 编码，符合 YAML 1.2；SHA 为本地字节身份。来源 Git 文本只允许 CRLF/LF 等价核验；checkpoint 永远精确字节 SHA。全部来源见 manifests；历史代码只读取、未 import。所有九个模型作为完整冻结集合，无最佳 Seed 选择。

最终判定见 [决策报告](reports/L_FINAL_EVALUATION_READINESS_DECISION.md)，执行审计见 `outputs/L_FINAL_READONLY_INTEGRITY_AUDIT.json`。PASS 指本轮只读审计及合成测试通过，绝不等于 Test 资格通过。

完成后停止。新的探索性或独立评价均需以后单独、明确授权；本轮状态机 BLOCKED。

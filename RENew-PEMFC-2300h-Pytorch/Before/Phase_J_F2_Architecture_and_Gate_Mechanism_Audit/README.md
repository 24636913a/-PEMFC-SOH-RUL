# Phase J — F2 架构冻结与工况门控机制验证

Base: `427d4dc402f083a5b9e094b12981c5e231df8100`. Protocol SHA256: `0244f05fd777ccd615fda7eb9533d129323dc27fbb58212e091130011d0d9b24`. Code identity SHA256: `67b0d3588f75db56d435ddc5277d38324708f660ac10cc1850a5908d771612c8`.

Validation only, 205 origins/seed, 615 gates; all 4920 queries per seed retained. New training/Assessment inference/Test/seed selection: NO.

本阶段固定已有F2定义，仅以Validation开展被动门控采集和唯一g=.5诊断。不训练、不更改网络、不推理Assessment、不打开Test。

状态：PHASE_J_F2_ARCHITECTURE_FROZEN；PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED。

执行历史与可复核入口（从本目录，python -B）：

九项 parity/full/fixed 推理阶段已经全部完成，执行 token 已消费。原 `run_j_audit.py --execute` 随后在报告接口处 BLOCKED：H 主指标表没有 `endpoint_RMSE_mV` 字段，终端编码错误又掩盖了原 KeyError。原脚本、冻结协议、BLOCKED ledger 和日志均保留；禁止再次运行该推理入口或重新冻结协议。`audit_j_readonly.py` 的历史接口也不作为有效复核入口。

单独注册的缓存补全仅读取已保存结果及哈希锁定的 H Validation CSV，不构造模型、不执行 forward、不重复任何推理。它逐条验证三个 Seed 全部 205 origins 的键、真值和预测完全一致，并独立重算指标。可重复运行：

```text
python -B cache_completion/complete_saved_cache_audit.py --complete
python -B -m unittest discover -s cache_completion -p test_saved_cache_audit.py -v
python -B scripts/analyze_j_gate_behavior.py
python -B scripts/plot_j_figures.py
python -B scripts/write_j_reports.py
```

原始 26 项回归测试与缓存补全 5 项测试均已实际通过，日志位于 outputs/integrity_audit。缓存补全 manifest SHA256：`ac2652eff04eb5ff72360446a5a629932fcd21101000741e55576b54b22023ff`；完整执行代码身份 SHA256：`8d461f63bdcd4212634b962f2b875ed3f49a954a5515fafec51df5d997bad43c`。原 BLOCKED wrapper 与有效缓存完成记录分别呈现，不重写历史。

本目录的 `.gitattributes` 禁止 Git 自动换行转换，保证 SHA 冻结的代码、协议及证据字节在 Windows checkout 后不被 core.autocrlf 改写。SVG 仅机械移除生成器的行末空格，XML 验证通过，图形几何、数据、PNG/PDF 与执行代码不变；最终交付清单记录规范化后的 SVG 哈希。

统计和图像脚本可从缓存重算；报告生成器写出自动草稿，最终交付中的解释性补充见报告及 `J_CACHE_ONLY_INTERFACE_CORRECTION.md`。协议只冻结一次，执行token禁止已进入的parity/full/fixed stage重复。输入仅合法N1逻辑prefix、Train分组及Validation205origin；新图6组均PNG/PDF/SVG，架构另有Mermaid可编辑源。完整文件清单见outputs/integrity_audit/J_DELIVERY_MANIFEST.json。方法草稿与图像采用academic-paper写作约束和scientific-figure-making样式，均以真实源码/数据为证据，不主张全球首次创新。

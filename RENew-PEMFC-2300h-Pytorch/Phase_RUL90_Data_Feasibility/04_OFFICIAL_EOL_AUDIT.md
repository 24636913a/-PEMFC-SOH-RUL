# 官方 EOL 审计（Official EOL audit）

`OFFICIAL_EOL_DEFINITION = NOT_FOUND`。原始说明/PDF 中存在 durability lower-voltage-limit 的文字证据（见 `outputs/eol_evidence.csv`），但未发现把 **90% 初始参考电压**定义为官方 EOL、给出官方 EOL time label，或明确说明“2300 h 停止即电池 failure”的结构化原始标签。因此测试结束不能自动等于失效，也不能定义 `RUL=2300-t`。

论文 PDF 第 4 页第 14–20 行说明研究目标是完成超过 2000 operating hours，并在约 200 cycles 后结束 durability test 以做完整 characterization；这支持“预定/研究性结束（EoT）”，而不是已记录 failure。第 8 页第 15–17 行的 500 mV lower voltage limit 出现在 60/50 kPaG 压力下、1.8 A/cm² 的局部 sensitivity measurement 无法继续测量的语境。它是条件特定的运行限值，不是本阶段预注册的 90% reference-voltage EOL，也不能自动证明 run-to-failure。证据摘录：

| source_file | page | line | evidence_text | interpretation |
| --- | --- | --- | --- | --- |
| IJHE_2026_Fensterle_Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing.pdf | 1 | 47 | ered the most critical, as it can ultimately lead to cell failure, whereas | evidence only; does not by itself prove official EOL |
| IJHE_2026_Fensterle_Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing.pdf | 1 | 49 | mance losses [4]. Membrane failure is mainly caused by the formation of | evidence only; does not by itself prove official EOL |
| IJHE_2026_Fensterle_Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing.pdf | 8 | 16 | because the cell voltage fell below 500 mV, which was defined as the | evidence only; does not by itself prove official EOL |
| IJHE_2026_Fensterle_Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing.pdf | 8 | 17 | lower voltage limit for the durability test. | evidence only; does not by itself prove official EOL |
| IJHE_2026_Fensterle_Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing.pdf | 14 | 37 | [5]Bender G, Felt W, Ulsh M. Detecting and localizing failure points in proton | evidence only; does not by itself prove official EOL |

# Phase E — TFT 全特征与因果滤波联合对照实验

本阶段在固定的 RealHyFC Development 单次时间顺序 7:1:2 划分上，比较 Top10 与 Raw14 两种输入掩码，以及 Raw、单侧因果 SG(7,2) 和逐窗口初始化 EWMA(α=0.3) 三种输入处理。

六组模型均为独立 Seed42 训练的 CanonicalStrictHistoryTFT，均采用同一套仅由未滤波合法 Train 历史拟合的公共输入 scaler 与 ΔV 尺度。Assessment 已在历史阶段使用，因此所有结果均为 `DEVELOPMENT_EXPLORATORY_REUSED_ASSESSMENT`，不构成独立 Test 结论。

运行：`python scripts/run_e_comparison.py --audit`，随后 `python scripts/run_e_comparison.py --run`。运行前会验证协议哈希、3007 行前缀访问、3005 Development 行、固定窗口数、因果滤波和公共 scaler；所有检查 fail-closed。

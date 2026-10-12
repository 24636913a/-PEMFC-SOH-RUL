# Phase P1 — 门控消融实验技术验收

仅工程预检查，`training_authorized=false`。不读取真实时序、Assessment 点预测、Test 或历史权重；不执行优化器更新、正式训练或论文正文写作。

G0 直接使用冻结 F1 固定0.5融合；G1 只覆盖冻结F2的门控上下文为固定ones(5)；G2直接使用冻结F2。共同核心与G1/G2门控按Seed42/52/62 fresh共享初始化。

在仓库根目录运行：

```powershell
python -B RENew-PEMFC-2300h-Pytorch/Phase_P1_Gate_Ablation_Engineering_Preflight/scripts/run_p1_preflight.py --freeze-engineering-draft
```

首次冻结后重复验收省略 `--freeze-engineering-draft`；源码、协议或冻结来源身份变化会拒绝继续。显式refreeze只能修复技术草案并留存上一版，不能放宽科学门槛或授权训练。全部测试从这个带IO/optimizer防线的入口执行，不直接运行裸pytest。

检查使用真实 Canonical TFT/F1/F2、合成输入65×48×14、合成梯度（不step）、1380个合成索引两次DataLoader排列、9个新合成checkpoint保存/重载。临时权重和预测在`runtime/`，Git忽略；证据哈希、初始化/RNG/实际排列和测试日志在`outputs/`，报告在`reports/`。

技术PASS只能用于申请未来九次正式训练授权。独立授权前本入口始终拒绝真实训练，不存在自动切换。H正常CUDA环境不宣称严格位级训练确定性。原Phase P/H/F及历史九个checkpoint保持只读。

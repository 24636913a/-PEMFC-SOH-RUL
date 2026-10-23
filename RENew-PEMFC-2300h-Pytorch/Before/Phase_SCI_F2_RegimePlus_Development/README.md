# F2-RegimePlus Development探索性实验

基准`36572a6678a9e64df689c8c6ec282c7233b6ac36`；协议SHA256 `8f6361457e2f1110560357d2cfc880d952ca47bed4218186c4bb99727cb25966`。本阶段是重复使用Development Validation的探索性研究，未评价新模型的Assessment或后段，不是独立Test。

**NO_CONFIRMED_GAIN | NO-GO**

三seed平均MAE从4.274253降至4.066576mV（改善4.859%），但12运行小时终点RMSE从15.173052升至29.446082mV（恶化94.068%）。相对原F2，三个seed整体RMSE均恶化；对A1则42、62改善，52恶化。较小的seed离散度不能抵消三个实例共同的尾部失败。

| variant | seed | RMSE_mV | MAE_mV | endpoint_RMSE_mV |
| --- | --- | --- | --- | --- |
| B0 | 42 | 6.265169 | 4.704244 | 7.146145 |
| A1 | 42 | 10.260655 | 5.348725 | 30.307095 |
| B1 | 42 | 8.807123 | 3.685795 | 29.474650 |
| B0 | 52 | 7.034632 | 5.033394 | 8.555617 |
| A1 | 52 | 6.996165 | 4.840483 | 6.307080 |
| B1 | 52 | 9.260209 | 4.231135 | 29.515126 |
| B0 | 62 | 8.436399 | 3.085123 | 29.817395 |
| A1 | 62 | 9.518811 | 4.534154 | 29.804950 |
| B1 | 62 | 9.248544 | 4.282799 | 29.348472 |

实际执行一个三特征门控候选与等容量归零特征消融，固定42/52/62共六次新GPU训练；B0历史预测逐点复现。历史模型及论文结果不替换。完整证据见01—05报告和outputs，冻结判据见configs。

复现（本目录，既有Python/PyTorch CUDA环境）：
```powershell
python -B scripts/experiment.py prepare # 首次专用，已冻结时拒绝重复
python -B scripts/experiment.py run # 已成功训练跳过；失败现场拒绝静默重跑
python -B scripts/audit_inputs.py
python -B scripts/analyze.py
python -B scripts/verify.py
python -B scripts/protect_history.py
python -B scripts/seal.py
```
模型定义只读导入冻结RealHyFC_F2_Main.py，SHA写入协议；新门控及完整训练逻辑位于scripts/experiment.py。只从有界Train/Validation缓存取输入、标签。没有调用旧完整数据读取或旧评价入口。

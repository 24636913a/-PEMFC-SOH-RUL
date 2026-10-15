# Independent weak-physics polarization branch
本轮真实24次CPU A/B训练；单一RealHyFC Development，非新backbone搜索。结果：短暂激活，无清楚物理增量，不启动TFT/SOH融合。

首次执行顺序（已有完整输出时不再运行prepare/train）：
```
python -B scripts/prepare.py
python -B tests/test_autograd.py
python -B scripts/train.py
python -B scripts/analyze.py
python -B scripts/verify_independent.py
python -B scripts/render_figures.py
python -B scripts/write_reports.py
```
依赖Python3.11、PyTorch2.6、NumPy、pandas、matplotlib；CPU、float64、单线程。正式协议冻结后不得改网格、模型、lambda或预算。已完成RunID直接跳过；技术中断仅从同Run检查点继续。

TaskA重建context是同次测量Train锚点电压，TaskB是过去起点P1S保持率；分别训练不可混用。B有测量完成可用假设，不认证严格在线发布。H系列mV-equivalent与本轮真实电压mV不能混称。

论文子目录是存档镜像，脚本来源路径按本研究Phase定位；复现请回到原Phase目录，勿在镜像内启动训练。历史全部只读，所有原始Test文件均不可读。本轮结束后停止。

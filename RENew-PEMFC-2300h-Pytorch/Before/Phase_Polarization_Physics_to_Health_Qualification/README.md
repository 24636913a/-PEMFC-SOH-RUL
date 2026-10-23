# Polarization physics → health qualification
探索性Development；物理诊断可用、健康物理训练接口未合格。原始Test未读取；原结果只读。

在本目录执行（Python3.11，numpy/pandas/scipy/scikit-learn/matplotlib）：
```
python -B scripts/run_qualification.py freeze  # 仅首次；已有冻结文件时禁止
python -B scripts/run_qualification.py run     # 仅未完成时；单文件/拟合检查点续做
python -B scripts/run_h2.py                   # 仅未完成时；一轮预注册CPU实验
python -B scripts/synthesize.py
python -B scripts/verify_independent.py
python -B -m unittest discover -s tests -v
python -B scripts/plot_figures.py
```
已有完整结果时，仅运行独立核验、测试或绘图；不要删除完成标志重新拟合。上游源数据以SHA256校验，所有读取由白名单防火墙约束。源Raw文件7个，导出的选定通道CSV保留原始行身份供核验；不是独立寿命样本。
REAL_Paper_File中的同名材料是存档镜像，复现应回到仓库研究主目录下的原Phase目录执行；脚本历史来源路径按原Phase定位，不能在论文镜像内直接启动实验。

# Demo 3｜两台电堆独立建模：从这里开始

## 先看已经生成的结果

完整解压后，双击 `results/report.html`，可在浏览器中查看本次结果图、中文解读和指标表，不需要安装 Python，也不需要联网。图像点击后可查看原图。目录中的 `results/RESULTS.md` 是同一次运行的文字说明。

## 重新运行

在本文件所在目录打开终端，使用 Python 3.12（本次验证环境）：

```bash
python -m pip install -r requirements.txt
python run_demo.py
python build_report.py
```

第一次安装依赖可能需要联网；依赖准备好后，演示运行使用本地 CSV。无需 GPU、模型权重、API 密钥或额外数据下载。默认运行读取随附数据；只有显式加 `--regenerate-data` 才重新生成模拟数据。

若要保留原始结果：

```bash
python run_demo.py --output-dir rerun_results
python build_report.py --output-dir rerun_results
```

不要直接在 ZIP 预览界面内运行；先解压整个文件夹。

## 到哪里找代码和数据

- `README.md`：算法公式、输入字段、参数和换用真实数据的步骤。
- `run_demo.py` 与同目录 Python 模块：可编辑源代码。
- `config.json`：参数配置。
- `data/`：模拟观测、诊断或事件记录，以及独立保存的评估真值。
- `results/`：已执行的结果图、逐点 CSV、指标、验证记录和环境信息。
- `requirements-tested.txt`：本次实际版本；常规安装使用 `requirements.txt`。
- `RESEARCH_NOTES.md`：三个问题的共同口径、实验设计和真实数据迁移注意点。
- `REFERENCES.md`：已核对的原始论文与数据记录。
- `MANIFEST.sha256`：交付文件校验清单；重新运行会更新结果文件，因此其哈希可能改变。

## 数值是什么性质

所有数据和结果均为模拟演示。`FC1_like`、`RealHyFC_like` 只是情景名称。5/10 片、3 倍速率和诊断日程等按各包配置设定，不能当作真实数据结论。4% 损失阈值表示 SOH 为 0.96。`truth` 只用于事后评价，不能作为部署输入。

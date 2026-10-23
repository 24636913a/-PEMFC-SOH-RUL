# Seed 202秒级F2训练：中断记录保留

本目录是用户运行的原始训练现场，状态为`FAILED_OR_INTERRUPTED`。保留原始状态、错误栈、配置、预处理身份、源码快照、Epoch日志、训练图和已有`best_model.pt`。

日志仅记录1个已完成Epoch，后续训练以`KeyboardInterrupt`中断；该checkpoint是当时保存的部分成果，不是已完成450轮训练或正式最终模型。本次没有重新训练、推理、评价或更换权重，`holdout_evaluated=false`保持原样。

上传范围及SHA256见`UPLOAD_MANIFEST.json`；checkpoint通过Git LFS保存。所有原始上传文件按原字节保留，不进行换行转换。

`cache/`下12个派生缓存文件合计约1.26 GB，继续保留本地，不上传Git。已有缓存SHA沿用`source_manifest.json`的原始记录，本次没有重新散列这些缓存；未提供原SHA的索引缓存仅登记路径与大小。缓存不是从GitHub恢复本地完整运行环境的自动保证。

原始源码、模型、数据和其他历史实验均未改变。本次只归档中断现场，不将其写成成功实验或正式后留评价结果。

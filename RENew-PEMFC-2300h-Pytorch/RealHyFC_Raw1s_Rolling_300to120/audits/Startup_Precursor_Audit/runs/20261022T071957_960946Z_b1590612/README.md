# 启动前兆审计：20261022T071957_960946Z_b1590612

这是用户实际运行完成的结果，原始run_manifest.json状态为COMPLETED。本次仅上传文件，没有重新运行数据审计、训练或推理，也没有修改原始结果。

先阅读 [AUDIT_REPORT.md](AUDIT_REPORT.md)，指标见 [precursor_evaluation.csv](precursor_evaluation.csv)，数值汇总见 [audit_summary.json](audit_summary.json)。图片、PDF、事件清单、主要指标和源代码快照可在本目录直接查看。

## 完整原始目录

**FULL_RUN_ARCHIVE.zip** 使用Git LFS保存，约967 MB。它包含上传前全部**68个原始文件，共5,107,495,009字节**，包括所有cache、2.20 GB的event_contexts.csv及其他大CSV，没有省略原始文件。

为避免重复占用仓库，原始cache及超过10 MiB的CSV只在完整ZIP中保存；其余原始文件同时作为普通文件上传，方便在线查看。直接浏览列表以UPLOAD_ORIGINAL_FILES_SHA256.csv中的directly_browsable字段为准。所有68个原始文件在本地原路径仍保持不变。

ZIP SHA256：`c1395c1d2334ed3c4f5997af129507964f69778a80b85033b4ef43c45c6644e0`。

已经逐个解压流式核验ZIP内68个文件的SHA256，与上传前原始文件完全一致；核验记录为UPLOAD_INTEGRITY.json。完整原始文件清单、字节数和SHA256见UPLOAD_ORIGINAL_FILES_SHA256.csv。

下载完整ZIP后，解压到**新的空目录**即可恢复原始目录结构。克隆仓库时需要安装Git LFS，并执行`git lfs pull`获取真实压缩包；Git指针文件不等于完整ZIP。

原有run_manifest.json等历史身份记录保持原样。这个README、上传清单、完整ZIP、校验记录和LFS属性是上传时新增的交付文件，不属于原始68个分析产物。

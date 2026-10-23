# 模型及checkpoint身份

9个新P2 checkpoint只在P2/outputs/checkpoints，本地安全保留、Git忽略，无独立备份（已如实记录），从未加载/覆盖历史九权重。模型run_id、seed、protocol/code SHA、初始参数/RNG、输入fingerprints、最佳epoch、重载严格相同和缓存真值/键均审计。

完整本地文件清单、bytes、SHA、备份状态：outputs/audits/P2_LOCAL_ARTIFACT_MANIFEST.json。P1及H/F原始源码只读。初始核心共享，G1/G2门控57参数共享且g=0.5；G0332379，G1/G2332436。G1固定ones5、G2原历史五工况，核心decoder不变。

产物完整性：PASS；全部预测重新读取重算一致。

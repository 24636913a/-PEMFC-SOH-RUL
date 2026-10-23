# 原 H 协议与运行语义

状态：PASS。

直接 AST 核验原 H：`validation_standardized_RMSE < best - 1e-12`。监控标准化 RMSE，不是 mV RMSE；每组自身最佳 checkpoint；max450、patience40、clip1.0，无 scheduler。AdamW lr=0.0009605449687795866、weight_decay=6.95942232671908e-6、batch32 保持不变。本次没有构造优化器或执行 epoch。

Raw F2 48×14→24 ΔV、冻结 Top10、标准化空间零掩码、逐窗口因果 EWMA alpha0.3、Train-only scaler、Train [0,2103)、Validation [2103,2404)、1380/205 窗口与 H 原协议相同。这些仅从配置/源码核对，本次不运行真实数据构建器。

P 的早停描述没有明确空间，P1 只在新 draft 明确 H 的真实 standardized 语义，不修改 P 历史文件。将来 N1 必须 nrows3007 后年龄过滤为3005，禁止直接 nrows3005 或完整文件哈希；此处读取次数为0。

实际调用 AST 提取的 H.configure_normal/environment，环境与 H 全部字段一致；记录于执行审计。原 E.predict→C.predict 使用 eval + no_grad + batch32；没有 inference_mode、AMP、requires_grad 改写或 TF32 改动。

# P2 冻结前 Git 文本表示误报记录

首次 `--freeze` 在任何P2协议、ledger、N1读取或模型训练之前退出，提示 `FROZEN_SOURCE_CHANGED: .../E_COMMON_SCALER.json`。

只读核对发现：工作区完整字节SHA256仍为H冻结值 `969233228590c11f3c14f4eb364acbcbc638b70fa35a44ba72926ecb7a616ab4`，F数据审计字节SHA仍为 `e4aa7472464f6268c53de18f7059e09668539fe960a85f547ec69e63cb5273b4`，`git diff`为空；该历史JSON的Git属性为`text: auto`。原因是Git blob LF与冻结工作区CRLF表示不一致，不是历史scaler内容变化。

仅修复新P2身份检查：Git历史文本核对只规范化CRLF→LF；原H登记过的每个文件仍独立严格校验完整工作区字节SHA。完整SHA记录继续保留原字节，不能以文本规范化hash冒充历史hash，也不能把内容变化解释为换行变化。新增生产函数回归测试验证内容修改、H字节SHA不符仍失败。

未修改任何旧文件、训练参数或科学门槛；没有协议重冻结、ledger删除、失败组重启或额外Seed。该修复发生在第一次有效协议冻结及真实训练之前。

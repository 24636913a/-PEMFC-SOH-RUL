# 执行恢复与Git审计

## 起点现场（北京时间2026-10-14 09:27起）

本地main HEAD及缓存origin/main均为8997b8c881f206680235361b36f892f0edc2af62。git ls-remote两次连接重置；随后认证GitHub CLI branches/main请求成功，实时远程SHA相同。没有拉取、合并或覆盖其它成果。

中断前本目录已有`.gitignore`、4脚本、1测试、SYNTHETIC_TEST_RESULTS.json及忽略的matplotlib缓存；最后文件更新时间01:11。没有协议/资格决策/训练ledger/checkpoint/九份报告，本阶段尚未提交。进程查询未发现python训练进程；nvidia-smi只有显示应用。本轮从资格冻结继续，未重新运行已通过的合成测试。

仓库历史tracked diff与暂存区起初为空；另有多个旧N2阶段node_modules未跟踪依赖文件，保持原样、没有加入本阶段提交。恢复期间未修改历史tracked文件。提交范围只能是本阶段目录。

## 实际后续执行

1. --freeze只执行一次，SHA=39003814b9af339765b90177cac8a4bf1586f65f1e90ba28aadaac9a66ff4ed8。
2. --qualify只执行一次；约17秒GPU前向检查、9个seed/role组合，0 optimizer step。控制台记录保留；没有技术失败或重启。
3. 仅候选准入失败，formal_runs_started=0。没有运行--run，没有本阶段训练成绩或新权重。
4. 报告生成只读资格JSON、历史指标汇总和文档；新增REPORT_SOURCE_MANIFEST记录其身份，独立于执行前冻结manifest。新增报告脚本不参与模型/资格执行，不修改执行身份。
5. 用phase-local .gitattributes固定LF，避免core.autocrlf=true在提交时改变冻结脚本/JSON哈希；自检须验证工作区与Git index字节身份一致。

原执行脚本把BASE设为参考SHA，因本次本地/实时远程确实相同才freeze成功；未来仓库更新不能修改BASE并重冻结本次协议。Windows绝对源路径用于本机身份审计；迁移机器应单独审定路径/环境，不宣称直接可移植复现。

## 读写与资料限制

唯一真实时序表读取为nrows=3007有界Development读取，过滤3005，再仅1380 Train窗口。对象指纹非混合全文件SHA。checkpoint读取6个旧G2 best/last，仅Train推理。无新Validation推理、Assessment/Test、raw、晚期极化/健康目标；2000/2300百分数只引用指令里的既有审计摘要。

SOH报告读取范围见REPORT_SOURCE_MANIFEST；历史Test曝光边界仍继承，不称全部资料从未暴露。文献查询只补适用条件，未下载RealHyFC数据。NETL官方手册已读；starvation原始实验论文全文403，只使用检索摘要证据。

## 可复现入口与成果保护

`python -B scripts/run_soft_physics_qualification.py --freeze`、`--qualify`均为一次性入口，已有协议/决策会拒绝重跑。`--run`没有在物理资格失败时放行的训练worker。已完成结果应阅读/验证，不删除后重跑。报告生成脚本也拒绝覆盖已有报告。

最终提交前自检保存在outputs/RECOVERY_FINAL_AUDIT.json；实际提交/推送状态另存忽略的outputs/GIT_TRANSPORT_RECEIPT.local.json，并可由Git提交历史与实时远程验证，交接回复给出实际SHA。网络失败时保留本地提交、不强推、不重做训练。报告不写入自身commit SHA，避免为回填SHA制造重复提交。

## 字节码缓存拒绝与交付验证修复

运行日志保留9次历史__pycache__/*.pyc读取拒绝；Python随后读取已allowlist并核验SHA的.py源文件，无历史写入。交付验证器最初把任何DENIED都视为失败，人工检查确认均为上述缓存后，仅对此类已认证源对应的只读.pyc拒绝设审计例外；未改生产防线或重跑资格审计。

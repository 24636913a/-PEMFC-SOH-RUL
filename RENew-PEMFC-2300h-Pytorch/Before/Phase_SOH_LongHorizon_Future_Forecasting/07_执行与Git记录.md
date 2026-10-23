# 执行和成果身份

起点main本地/实时远程均为6ce263d65d9e89d7912672986d976c7116b03426，工作区干净。所有新文件位于本独立阶段；历史AHC/PHI/CTAE/CMRF和模型权重未修改。没有GPU、raw重建、晚期目标打开或模型搜索。

执行：`python -B -m unittest discover -s tests -v` → `python -B scripts/run_soh_experiment.py --freeze` → `--run` → `python -B scripts/verify_saved_results_reference_diagnostic.py` → `python -B scripts/make_figures_and_reports.py`。freeze绑定8个小型来源文件全SHA、BoL单记录逻辑指纹和全部脚本/测试；首次运行ledger已完成，禁止删除后重跑。

测试和控制台保存outputs，必要测试14项。第一次预冻结测试发现N1 CSV含UTF-8 BOM，修正reader为utf-8-sig后14项通过；保留TEST_CONSOLE.log失败现场及TEST_CONSOLE_ATTEMPT2.log通过记录，没有正式实验重跑。独立重算使用不同OLS实现，不只调用原方法。输入时间/周期ID及每个原点的参数保存JSON。模型数学/数据访问由固定源码和protocol哈希记录；报告/图表均使用已保存结果，没有重新拟合。

读取审计见DATA_ACCESS_AUDIT.json；未读取PHI全六点B2/SG/isotonic/PCHIP列作为输入，只消耗周期身份列。N1混合SOH表只解析第一BoL记录，不做全文件SHA。source SHA在configs/FROZEN_SOH_PROTOCOL.json，历史正式代码/报告作为只读依据，未调用旧runner。

输出CSV和PNG/PDF由真实预测生成，最终文件哈希清单见outputs/DELIVERABLE_MANIFEST.json。提交前检查历史tracked diff为空、图表视觉QA及Git暂存字节身份；提交/推送回执本地outputs/GIT_RECEIPT.local.json（忽略），最终回复提供实际SHA，避免为了回填自身SHA重复提交。禁止强推，远程更新需先审定。

## 独立审计修复和最终图表验收

冻结的verify_saved_results.py已经独立核对全部12个预测和指标，但在最后把用户近似M1参考误差容差设得过严（0.0002 pp），因实际差0.000229 pp而停止。保留原冻结脚本与失败日志，新增审计专用verify_saved_results_reference_diagnostic.py，仅将近似值改为诊断字段，不修改模型、数据或预测。新审计的来源/自身SHA在AUDIT_REPAIR_NOTE.json；全部执行前冻结代码保持原SHA，正式CPU预测仅运行一次。

四张PNG逐张视觉检查通过：标题、图例、轴标签无遮挡；颜色/纹理一致；观察点与拟合/外推线有清楚区别；PDF同步生成。报告之后补充误差归因及近似自检差异，只使用保存的误差和参数，没有重新拟合。最终源/代码/产物核验在FINAL_INTEGRITY_AUDIT.json。

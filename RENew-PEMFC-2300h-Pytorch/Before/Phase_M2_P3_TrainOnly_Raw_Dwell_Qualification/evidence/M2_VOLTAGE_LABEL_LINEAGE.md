# Voltage label lineage

raw有效10个cell → 瞬时cell mean → middle50% dwell median为拟议辅助标签。已输出真实measurement logical record/time起止与source identity。它同源于N1 mean/min/max/std/range及EWMA后继，因此主F2 voltage通道不能直接复用于辅助可见view。

M1九个非电压通道view原封不动；T/RH/P/flow只用于资格，未新增辅助输入；未暴露raw cell、exported mean、P1S/B2、EWMA或voltage-derived cache。M1列级隔离仍保留。

Stage1 physical gate未通过，故Stage3 actual auxiliary input-window/label dwell联接尚未执行。这里的真实来源恢复不等于complete temporal isolation PASS。没有建立辅助网络或使用新dwell作为伪SOH训练标签。

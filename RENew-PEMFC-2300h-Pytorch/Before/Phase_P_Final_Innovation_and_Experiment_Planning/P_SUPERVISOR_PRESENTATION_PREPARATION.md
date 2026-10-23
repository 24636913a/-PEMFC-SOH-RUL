# 导师汇报材料提纲（只规划，不制作PPT/论文）

## 汇报顺序：先结果和边界，再创新判断

**1. 研究问题**：RealHyFC动态负载下，过去24h实际观测能否预测未来12h动态电压增量？未来actual工况未知。不是当前SOH/RUL或已完成PINN。

材料：H冻结协议、O信息集合报告；自有任务示意可依据源码重绘，不能拿竞争论文原图作为原创。

**2. 当前模型**：Canonical TFT→同VSN输入的LSTM/因果depthwise CNN→encoder特征融合→保留decoder/attention/head。F0/F1/F2职责、parameter counts以及57参数gate控制对象。

材料：F frozen models源码；O `evidence/O_F2_CODE_AND_CHECKPOINT_IDENTITY.md`；J已有架构图。不要使用早期未实际实现的Adaptive GRN/MCLSTM方案替代当前模型。

**3. 与已发表工作的差异**：P1是BiLSTM替换TFT两侧；P2是原TFT协变量；P3独立输出目标误差选择；P4已有dilCNN及双条件路径。F2具体融合位置/信息契约可区分，但双路径/多尺度/gate概念不是首次。

材料：O全文账本、O-01/02/05。题目先限定动态多步电压，不以physics/SOH/RUL提升包装。

**4. 原始与改进结果**：H Validation等Seed正式RMSE F0 9.067563、F1 9.155598、F2 7.245400mV；F2相对F0约20.10%。同时显示三Seed独立数据、SD、endpoint、target等权、P99/max和完整horizon，而非只放最好Seed。

材料：O1九模型总表/等Seed表和六组原创图。表名必须Validation；Assessment只能注明已复用Development，不能作为新的Test成绩。

**5. 模块证据与待批准最后实验**：F1独立收益不稳；J learned/fixed同权重诊断均值仅差−.002019mV，不是正式从头消融。展示拟议G0/G1/G2、参数/初始化公平、九次预算和预定停止规则。

材料：J fixed gate报告、P_GATE/P_MATCHED文档。新实验栏目前全部写“尚未执行，待授权”；若获准完成后，展示全九结果，无论正/负。若G2不优全局G1，明确削弱工况自适应机制主张。

**6. 物理增强有没有真实增量？**：B1数据辅助不等于物理知识；B2固定状态dV/dj<0有理论依据，但M2支持/未来接口不足；B3teacher不合格。展示六项准入矩阵，给出当前NO-GO，不许在汇报图中画已实现Physics-F2。

材料：P物理候选和方程审计、M2/N最终决策。成本低≠科学有效，也不需要为论文强行挂PINN。

**7. 研究限制和导师决策**：增量但可辩护；独立gate效益待证；Seed62低multiplicity异常仍保留；Validation/Assessment反复反馈；独立确认Test不合格；没有真实健康寿命验证。

材料：O1 Seed62及最终决策、L确认性资格、O创新报告。请导师明确：是否认可“严格信息边界+整体预测实证”定位；是否接受门控作为组成而非独立核心贡献；需要怎样的独立验证强度。本轮不替导师批准投稿或写稿。

## 汇报前材料完成条件

已有：冻结源码/身份、九模型Validation完整稳健性表、六组原创图、全文竞争对照、J门控诊断、M2/N NO-GO。

若负责人批准门控实验，还需：9/9训练和原始日志、共同初始化审计、完整new预测/metrics、epoch/时间/参数、事前门槛判定、最终模型版本和贡献清单。没有授权就保留“未执行”，可选择按现有F2有限主张汇报。

不需要本轮新增物理结果、真实RUL或新Test，也不需要进一步缓存后验诊断。导师认可前论文正文保持未开始。

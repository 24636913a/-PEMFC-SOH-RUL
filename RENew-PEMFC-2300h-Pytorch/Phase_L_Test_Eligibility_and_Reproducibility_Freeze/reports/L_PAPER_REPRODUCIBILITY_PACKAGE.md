# 论文证据包（已有结果，不是新实验）

主张：冻结历史工况感知双路径TFT（F2）用于RealHyFC动态工况多步ΔV预测。不是不可逆健康轨迹、SOH或RUL模型。所有数字均源自H/I2/J已有Development报告；L不打开缓存、不重新forward。

## Task / preprocessing / architecture

0.5h；48×14→24ΔV；Train-only Spearman Top10固定；仅历史voltage EWMAα=.3；14slots zero-mask；common Train scaler及targetscale冻结。TPE是配置工具，不独立算法创新。H_CONFIG optimizer AdamW、lr0.0009605449687795866、weight_decay6.95942232671908e−6、batch32、max450、patience40、clip1、hidden64、continuous32、dropout.1。

F0=Canonical TFT；F1=原post-LSTM encoder + causal depthwise kernel3/dilation1和4/leftpadding2和8，经pointwise/dropout/residualLayerNorm，固定.5融合；F2只替换为历史5变量mean→5→8ELU→1sigmoid的gate。gate不使用未来真实工况。参数量F0=323483、F1=332379、F2=332436（J_ID）。这些工程差异不等于两个独立已证明因果收益。

Development：nrows3007→age<1819.997738→3005；Train[0,2103)、Validation[2103,2404)、Assessment[2404,3005)。Validation205origins、4920queries；Assessment378origins、9072queries。这里只引述历史数量，不据此推算Test数量。

## 已保存成绩：每Seed query-weighted RMSE，mV

| scope | model | Seed42 | Seed52 | Seed62 | 等Seed均值 | sample SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Development Validation | F0 | 8.307161 | 10.381544 | 8.513985 | 9.067563 | 1.142630 |
| Development Validation | F1 | 9.212743 | 8.005451 | 10.248600 | 9.155598 | 1.122666 |
| Development Validation | F2 | 6.265169 | 7.034632 | 8.436399 | 7.245400 | 1.100853 |
| reused Development Assessment | F0 | 18.912760 | 19.255238 | 27.068482 | 21.745493 | 4.613023 |
| reused Development Assessment | F1 | 18.469522 | 38.157043 | 12.052070 | 22.892878 | 13.603013 |
| reused Development Assessment | F2 | 17.661515 | 13.764316 | 5.165326 | 12.197052 | 6.393819 |

H/I2来源：不是三fold数值，不是Test，也不挑最优Seed。F2对F0/F1都是3/3 direction，但absolute Assessment variability仍大；F1对F0 Validation仅1/3、Assessment2/3改善，不能声称双路径单独稳定收益。

## J门控诊断与负证据

J learned gate和fixed.5在Validation等SeedRMSE分别7.245400/7.247419mV，差约−.002019mV；Seed42方向与52/62混合。gate随历史输入变化且影响数值，但其独立、稳定精度增益和电化学物理机制未证实。原J wrapper报告接口BLOCKED，后续注册缓存补全没有重推理；两段执行历史均保留（J_REPORT）。

必须原样保留：

- PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS
- C1_D0_E0_REPRODUCIBILITY_UNRESOLVED
- PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED
- TIMESNET_FREEZE
- BACKBONE_SEARCH_CLOSED
- EOL_NOT_ESTABLISHED

L新增限制：现有late-life不具确认性独立资格；没有新Test performance。不能把历史三foldTimesNet/TFT、EMD、单splitF0/F1/F2拼成公平总榜；当前task信息集合/target/split/seed预算不统一。Sparse health证据不转成dense真实SOH，500mV条件限值不转成确认EOL。

## 复现交付与建议论文结构

任务和访问史 → frozen preprocessing/scaler → F0/F1/F2结构及代码SHA → H多SeedValidation → I-R执行语义限制 → I2复用Assessment → J门控诊断 → 当前Test资格/单stack限制。全部九weights身份在Lmanifest；源码实际执行环境见L环境契约，原始历史协议不覆盖。

可报告“复用Development上经验优势”；不可报告“独立Test泛化证明”“动态gate已验证物理因果”“完整SOH/RUL已解决”。本轮不编造外部文献/新评价，AI辅助材料需人工核查后用于投稿。

# F2冻结源码与九Checkpoint身份

基准 `9723ed75910594155b4c0ae4888d9bbe8519ec11`。本轮只静态读取源码/配置/历史manifest，不import模型，不load、不hash、不forward权重。SHA见 `outputs/O_STATIC_SOURCE_MANIFEST.json`。**权重SHA只引用L/J/I2历史身份，并非本轮权重字节重新校验。**

## 源码映射

以下路径均相对 `RENew-PEMFC-2300h-Pytorch/`：

- Canonical：`Phase_TFT_Original_Baseline/models/canonical_tft.py`，类 `CanonicalStrictHistoryTFT`。
- 原生forward镜像和融合位置：`Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py`，`forward` / `encode_history_branch` / `fuse_encoder`。
- 卷积分支：同目录 `causal_multiscale_branch.py`，`CausalMultiScaleHistoryBranch`。
- gate：同目录 `regime_aware_dualpath_tft.py`，`RegimeAwareDualPathTFT`。
- 特征：同目录 `frozen_feature_schema.py`。
- 实际H输入：`Phase_H_Determinism_MultiSeed_Stability/scripts/h_common.py`。
- 沿用E变换：`Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py`。
- 冻结超参：`Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml`。

## 张量、机制及增量

|模块|输入→输出|计算/事实|相对F0增量与证据|
|---|---|---|---|
|Raw输入|B×48×14|mean/std/range/min/max单体V；I mean/std/p2p；j mean/std/p2p；load transitions；operation transitions；zero fraction|相同输入schema|
|Top10|维度仍14|零mask索引1/2/5/12；保留0/3/13/11/4/9/6/7/10/8|不要写成14项均非零有效|
|EWMA|窗口V序列|alpha .3，首点初始化，历史单向递推；只作用电压通道|共同冻结处理；无future平滑|
|scaler|14变量标准化|H `prepare`沿用合法Train历史拟合并与E保存mean/scale逐项相等|当前O不执行拟合；不能把H历史写成仅load无fit|
|encoder VSN|B×48×15→B×48×64|14slots+相对历史索引，原生VSN|保留|
|LSTM路径|B×48×64→同形|原生LSTM后gate+addnorm|保留|
|CNN路径|同encoder VSN→B×48×64|depthwise k3 d1/d4，左pad2/8；concat→pointwise1×1→dropout/residual/LayerNorm|8,896参数；只历史|
|F1融合|两路同形→同形|0.5 LSTM+0.5 CNN|固定融合；独立增益不稳|
|F2 gate|五项历史48步均值→B×1×1|Linear5→8、ELU、Linear8→1、sigmoid；全窗口标量广播|57参数；对象是编码路径，不是变量VSN|
|F2融合|同形→同形|w LSTM+(1-w) CNN；以源码实际权重方向为准|一个联合模型内特征级融合|
|后续TFT|encoder融合+原decoder|static enrichment、masked interpretable attention、position GRN、gated output|保留，不改decoder LSTM|
|decoder|B×24×1 known索引|horizon index h/24；不读取未来actual|原生known接口的严格实例|
|输出|B×24×1|标准化ΔV MSE点预测；Vhat=Vorigin+sV*uhat，sV=.13680318714138442V|非quantile/非SOH/RUL|

gate五项顺序：load_transition_count、zero_current_fraction、j_std、I_std、j_mean。均从模型历史encoder_cont取得，无未来目标。相对时间索引不是绝对operating age。

参数量历史authority：F0=323,483；F1=332,379；F2=332,436。CNN分支新增8,896，F2较F1新增57；非本轮实例化统计。F2比F0更大，现有F1对照未完全隔离训练轨迹/容量；不能以参数增加很少即宣称capacity排除。

## 九Checkpoint历史身份

来源：`Phase_L_Test_Eligibility_and_Reproducibility_Freeze/manifests/L_FROZEN_MODEL_MANIFEST.json`，并对照 `Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/checkpoint_identity/J_NINE_CHECKPOINT_IDENTITY.json` 与 `Phase_I2_Frozen_Assessment_Evaluation/outputs/checkpoint_manifest/I2_CHECKPOINT_MANIFEST.json`。权重路径均为 `Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-Fx-SeedN_best.pt`。

|身份|best epoch|历史SHA256|
|---|---:|---|
|F0/42|277|82d3cbfa50a41ac381907a06598bb62ff4cffd58a5e39aabda26330271797528|
|F0/52|213|8d9a9450c8d868577c67e2b8c75c7b0c2bbba336eb95816624ad74792a4bab25|
|F0/62|353|10ab47ca4415465b4d11c14cdeca0bc33cee954038d8d0cd78346595eebb26b1|
|F1/42|204|9eb9ffa33685e0ab00d082c04fbe566944ec8b7d7ded28f434c0de8bd9f6c599|
|F1/52|125|9eea6aef92052b0e6b641fa99d7554d990a1e35baa8c8db3fa93d37f1e4d2fd3|
|F1/62|176|35ee1029482df179c79955d2258d770d2675162c5aec5d1a465456e82743c1c0|
|F2/42|204|0b28f576e1f5172875203849ad452a40e702040d0c3609b12074a506dbeea974|
|F2/52|223|8100c0199fe43e967a082fb78b6222b9a2ec73d40bedb845098e4fe3b7d3d4ed|
|F2/62|324|bd727a697fc99cd6fd3391bd6ae347b3d077ec19941db28b4ad62cd223cd36f5|

本轮只认证这些manifest和源码文件内容/来源提交一致。若需要验证当前权重实际字节，属于另行授权；本轮没有进行。历史 `PHASE_J_F2_ARCHITECTURE_FROZEN` 不变。

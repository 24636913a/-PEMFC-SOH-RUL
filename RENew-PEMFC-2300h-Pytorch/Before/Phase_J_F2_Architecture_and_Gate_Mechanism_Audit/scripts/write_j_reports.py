"""Scope-limited Chinese methods draft and evidence-linked reports, no fabricated references."""
import j_common as j

def table(frame):
    cols=list(frame.columns)
    lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for row in frame.itertuples(index=False,name=None):
        lines.append('|'+'|'.join(f'{x:.9g}' if isinstance(x,float) else str(x) for x in row)+'|')
    return '\n'.join(lines)

def write_reports():
    cfg=j.load_protocol();summary=j.read_json(j.OUT/'J_SUMMARY.json');proof=j.read_json(j.OUT/'integrity_audit/J_FINAL_READONLY_AUDIT.json')
    stats=j.read_csv(j.OUT/'gate_diagnostics/J_GATE_STATISTICS.csv');cmp=j.read_csv(j.OUT/'fixed_gate_diagnostic/J_FIXED_GATE_COMPARISON.csv')
    corr=j.read_csv(j.OUT/'gate_diagnostics/J_GATE_CONTEXT_SPEARMAN.csv');pairs=j.read_csv(j.OUT/'gate_diagnostics/J_PAIRED_SEED_GATE_DIFFERENCES.csv')
    hist=j.read_csv(j.OUT/'architecture/J_HISTORICAL_ABLATION_EVIDENCE.csv')
    params=j.read_csv(j.OUT/'architecture/J_PARAMETER_COUNTS.csv')
    common=f"Base: `{j.BASE}`. Protocol SHA256: `{j.digest(j.PROTOCOL)}`. Code identity SHA256: `{cfg['code_identity']['sha256']}`.\n\nValidation only, 205 origins/seed, 615 gates; all 4920 queries per seed retained. New training/Assessment inference/Test/seed selection: NO."
    j.text_write(j.PHASE/'reports/J_ARCHITECTURE_FREEZE_AUDIT.md',f'''# F2 结构冻结审计

{common}

`PHASE_J_F2_ARCHITECTURE_FROZEN`

{table(params)}

F1 相比 F0 新增 8896 个卷积分支参数，F2 相比 F1 新增 57 个门控参数。F0 保持原生 strict-history TFT；F1 在历史编码表示上等权融合；F2 用历史工况门控替换该等权融合。不是最终预测值的加权平均。

卷积分支直接消费 encoder VSN 的 [B,48,64]，而非原始14变量或 post-LSTM 表示。两条 depthwise 路径 k=3,d=1/4，分别仅左补2/8点，感受野3/9点（最早到当前跨度1/4 h）。没有额外卷积激活函数。拼接128通道后 pointwise 映射64，Dropout(.1)、VSN残差、LayerNorm。门控的LSTM侧为 post-LSTM GLU+AddNorm 后的表示。

原生 LSTM encoder 的 hidden/cell 继续初始化原生 decoder。融合不改这个状态交接，也不改 decoder residual。融合后的历史与原生decoder拼接后才进入 static enrichment、attention、position-wise GRN 和输出头。Decoder只接确定性相对步长；原始14个物理占位槽为零且不进入 decoder VSN。

超参数（历史冻结而非本轮重搜）：48→24，0.5 h，Raw F2 14槽，hidden64，continuous32，attention4，LSTM1，dropout .1，batch32；AdamW lr={cfg['hyperparameters']['learning_rate']}, weight_decay={cfg['hyperparameters']['weight_decay']}。本轮不创建 optimizer。使用原Train scaler、Top10与电压EWMA α=.3。

身份：九个主实验 checkpoint 9/9 唯一身份核验；三个 F2 固定32 Validation参考逐元素精确相等；被动hook和恢复正常门控均3/3 bitwise exact。完整参数/buffer及requires_grad保持，grad=None。原Phase F enabled full-forward证据来自 `F_IMPLEMENTATION_AUDIT.md`，15项历史合成测试；不将其eval一致性夸大为训练bitwise确定性。

源码权威：T0 `canonical_tft.py`，F `dualpath_strict_history_tft.py`、`causal_multiscale_branch.py`、`regime_aware_dualpath_tft.py`、`frozen_feature_schema.py`。逐文件SHA见冻结协议 sources，逐参数名/shape见 J_PARAMETER_SCHEMA.csv。上游 pytorch-forecasting 1.8.0 mirrored forward 版本敏感，源代码SHA必须保持。

无新增模型模块、无推理延迟或GPU效率宣称。结构身份可以冻结，与门控物理解释是否成立是不同问题。
''')
    j.text_write(j.PHASE/'reports/J_GATE_BEHAVIOR_REPORT.md',f'''# 工况门控行为

{common}

门控证据状态：`{summary['gate_status']}`。

{table(stats)}

SD采用ddof=1。低门控g<.1，中间.4≤g≤.6，高门控g>.9；诊断性dynamic定义SD>.001且range>.01，均在结果产生前冻结，属于PROJECT_ANALYSIS_DESIGN，不用于调参/筛选。g为LSTM权重，1−g为Conv权重；一个origin仅一个g，广播到48×64历史表示，不为24个horizon分别生成门控。

{table(pairs)}

三个Seed的门控绝对值及排序可能不同；只报告配对分布差异，不选择最佳Seed或把不同权重函数视为同一物理状态。是否饱和必须依据低/高比例逐Seed判断，接近1偏LSTM、接近0偏Conv。低中高区间不覆盖全部[0,1]，三项比例不必加和为1。

{table(corr)}

相关关系说明数学输入依赖。标准化工况均值从masked 48步输入被动采集，使用保存的scaler反变换，并独立与未滤波Raw工况窗口均值校验。五项工况不接受电压EWMA。原始单位与标准化值分列，未混淆。

Train1380个合法历史窗口确定每变量1/3、2/3分位阈值，冻结后应用Validation；退化/重复分位记录degenerate，空组不强制补样本。完整分层见 J_TRAIN_DEFINED_STRATA.csv。重叠窗口不是独立统计样本，相关/分组均描述性，不报告独立性p值或电化学机制。
''')
    j.text_write(j.PHASE/'reports/J_FIXED_GATE_DIAGNOSTIC.md',f'''# 唯一固定门控反事实诊断

{common}

{table(cmp)}

差值固定为 learned−fixed，负值表示当前checkpoint学习门控的RMSE更低。预测差异统计衡量数值依赖。全部指标由持久化CSV重新读取，并用独立标量计算核验（48项，最大差异{proof['max_independent_metric_difference']:.3g}）。原H完整Validation指标保持，未删除目标、重加权选择模型或挑Seed。

仅临时forward输出hook返回 `torch.full_like(out,0.5)`，finally移除；三个模型恢复后的全205预测与正常缓存np.array_equal。没有改变文件权重、state_dict、buffer、requires_grad或梯度。每项操作及独立进程PID见execution/state审计。

固定.5的F2不是独立训练F1。其其余参数已与学习门控共同训练，推理时替换改变训练条件一致性。因此结果最多支持当前checkpoint对门控的局部预测依赖，不是严格控制训练过程的结构消融，更不是物理因果机制识别。
''')
    method=r'''# 可用于论文的方法架构草稿

## 整体方法与信息集

本仓库采用工况感知双路径严格历史TFT预测动态运行电压变化。输入仅为预测起点及此前已观测的多变量历史，主输出是24步ΔV，而非直接SOH或RUL。方法描述以冻结实现为限，不主张文献范围中的首次提出。

令采样间隔δ=0.5 h，历史长度L=48、预测长度H=24。原始单体平均电压记为V，历史输入为

\[
X_t=[x_{t-47},\ldots,x_t]\in\mathbb R^{48\times14},\quad
y_{t,h}=\Delta V_{t,h}=V_{t+h}-V_t,\ h=1,\ldots,24.
\]

## Train限定特征处理

历史Train窗中第j变量的起点值与每个horizon的ΔV计算Spearman相关，冻结的排序分数为

\[
S_j=\frac{1}{H}\sum_{h=1}^{H}|\rho_S(x_{t,j},y_{t,h})|.
\]

实现对undefined相关使用nanmean，稳定按Raw索引解决同分，电压为必留变量。只消费历史已经冻结的Top10（0,3,13,11,4,9,6,7,10,8），本轮不重算排名。掩码在标准化空间置零未选槽位，保留14槽模型接口。

每个历史窗内仅对电压独立进行因果EWMA，首点以原始电压初始化：

\[
\bar V_1=V_1,\qquad \bar V_\ell=0.3V_\ell+0.7\bar V_{\ell-1}.
\]

其他13变量保持原值。输入scaler由未滤波合法Train历史拟合并保存，当前直接复用：

\[
\tilde x_{\ell,j}=m_j\frac{x^{\mathrm{filtered}}_{\ell,j}-\mu_j}{\sigma_j},\qquad
\tilde y_{t,h}=\frac{\Delta V_{t,h}}{\sigma_V}.
\]

目标没有额外减均值，σV为原输入电压scale。不要把标准化目标误写为原始电压V。

## TFT变量选择与历史编码

原生TFT对14个连续槽位和确定性历史相对坐标进行prescaling与变量选择，形成E∈R^{48×64}。可概括为

\[
E_\ell=\sum_j w_{\ell,j}\mathrm{GRN}_j(\tilde x_{\ell,j}),\qquad \sum_jw_{\ell,j}=1.
\]

该式仅概括源码VSN，不替代实际GRN/GLU实现。无真实静态协变量，静态embedding为零，保留原生static-context网络的已训练常量变换。原生LSTM历史编码与其后GLU、AddNorm产生H_LSTM∈R^{48×64}。编码器最终hidden/cell仍交给原生decoder，与后续fusion分开。

## 多尺度因果卷积

卷积分支同样消费E，而不是消费H_LSTM。对每个通道c，k=3的depthwise卷积写为

\[
C_{d,\ell,c}=b_{d,c}+\sum_{u=0}^{2}w_{d,u,c}E_{\ell-du,c},\quad d\in\{1,4\}.
\]

负索引用左侧零padding，右侧不补值，因此仅访问过去/当前位置。感受野跨度为3/9个采样点；对应最早到当前时间跨度1/4 h。d=4分支采样的是0、4、8步间隔，不等同于9点连续平均。

\[
H_{\mathrm{Conv}}=\mathrm{LayerNorm}\{E+\mathrm{Dropout}[\mathrm{PWConv}(C_1\Vert C_4)]\}.
\]

通道64+64拼接128，经kernel1 pointwise压回64。没有另加卷积激活函数。

## 固定与工况感知融合

F1历史融合为

\[
H_{F1}=0.5H_{\mathrm{LSTM}}+0.5H_{\mathrm{Conv}}.
\]

F2按Raw索引J=(11,13,9,6,8)，从标准化masked历史输入取工况上下文

\[
z_{t,r}=\frac1{48}\sum_{\ell=1}^{48}\tilde x_{t,\ell,J_r},\quad z_t\in\mathbb R^5.
\]

五变量为load_transition_count、zero_current_fraction、current-density std、current std、current-density mean。z不是原始电流单位；因这五项均被保留且未被电压滤波，物理窗口均值可按μJ+σJz恢复并独立核验。

\[
g_t=\mathrm{sigmoid}\{W_2\mathrm{ELU}(W_1z_t+b_1)+b_2\},
\quad W_1\in\mathbb R^{8\times5},\ b_1\in\mathbb R^8,
\quad W_2\in\mathbb R^{1\times8},\ b_2\in\mathbb R.
\]

\[
H_{F2}=g_tH_{\mathrm{LSTM}}+(1-g_t)H_{\mathrm{Conv}}.
\]

单个g在历史时间和特征维广播，不是每个horizon一个g，也不是最终预测混合。初始化输出层零使初始g=.5，训练后的g必须实际采集，不假设保持初值。

## 原生解码与预测输出

Decoder只输入h/24的相对坐标，未来14个物理槽为零且不进入decoder VSN。原LSTM encoder的hidden/cell初始化decoder，随后decoder GLU/AddNorm保持。融合历史与decoder表示拼接，经过static enrichment GRN，原生interpretable attention（Q为decoder，K/V为完整历史+decoder，因果decoder mask）、post-attention GateNorm、position-wise GRN、pre-output GateNorm及逐步scalar projection：

\[
\widehat{\Delta V}_{t,1:24}=\sigma_V\,f_{\theta}(\tilde X_t),\qquad
\hat V_{t+h}=V_t+\widehat{\Delta V}_{t,h}.
\]

pre-output GateNorm仍使用未被融合改动的decoder residual。没有新增SOH/RUL头、未来真实工况或物理退化微分方程。

## 评价、复杂度与证据边界

\[
\mathrm{RMSE}=1000\sqrt{N^{-1}\sum_{(t,h)}(\widehat{\Delta V}_{t,h}-\Delta V_{t,h})^2},\quad
\mathrm{MAE}=1000N^{-1}\sum_{(t,h)}|\widehat{\Delta V}_{t,h}-\Delta V_{t,h}|.
\]

上式单位mV，预测查询等权，重叠窗口不独立。目标时刻等权只作为既有敏感性指标，先平均同一target的平方误差，再跨target平均开根，不能先平均预测。三Seed算术均值与pooled RMSE不同。

F0/F1/F2参数量323483/332379/332436。卷积分支新增8896，门控新增5×8+8+8×1+1=57。没有测量延迟或效率提升宣称。

Spearman、EWMA、Optuna-TPE是成熟的筛选、预处理和配置优化工具，不称为新算法。结构贡献是保留原TFT并增加双路径历史表示及历史工况自适应融合。F1无稳定增益，不能声称卷积必然提高精度。F2的历史Validation和复用Assessment三Seed方向性改善支持限定的经验预测优势。门控相关与推理时固定替换仅解释数学/数值依赖，不能单独证明不可逆退化的物理机理。
'''
    j.text_write(j.PHASE/'reports/J_METHOD_ARCHITECTURE_DRAFT.md',method+'\n\n'+common+'\n')
    j.text_write(j.PHASE/'reports/J_INNOVATION_EVIDENCE_MATRIX.md','''# 创新声明与证据对应

|创新声明|对应实现|已有实验证据|尚缺证据|允许的论文表述|
|---|---|---|---|---|
|Spearman特征筛选|历史Train起点与24步ΔV平均绝对相关，Top10|冻结代码和数据身份|独立数据泛化|成熟特征筛选策略，不是新算法|
|因果EWMA|窗内电压α=.3|历史消融、当前因果扰动回归测试|独立Test稳定性|因果预处理，不是首次提出|
|Optuna-TPE|历史T1-R冻结参数|历史搜索与身份|本轮不做新搜索|训练配置优化工具|
|双路径历史编码|encoder VSN→原LSTM、因果depthwise/pointwise Conv|源码、15项历史F工程测试、当前因果/参数审计；F1无稳定收益|双路径单独稳定增益|双路径提供异质历史表示，不宣称必然提高预测精度|
|历史工况自适应融合|五维mean上下文→5→8→1门控，表示级融合|历史F2对F0/F1 Validation与复用Assessment3/3；本轮实际g与固定.5诊断|独立Test、训练匹配机制实验、物理干预证据|工况感知融合具有限定经验支持，局部checkpoint数值依赖不等于物理机制|
|完整SOH/RUL方法|当前没有SOH/RUL头|动态ΔV预测结果|健康监督、轨迹映射、EOL、右删失、不确定性与独立验证|当前仅冻结动态多步电压预测核心|

本文没有开展文献全球新颖性搜索，禁止“首次提出”“已识别真实退化机制”等绝对声明。
''')
    j.text_write(j.PHASE/'reports/J_SCIENTIFIC_LIMITATIONS.md','''# 必须保留的限制

'''+ '\n'.join('- '+x for x in j.LIMITS)+'''

F2 Assessment SD=6.3938192307 mV，大于F0的4.6130226020 mV；Seed62在均值收益中贡献明显。F1固定融合并未获得稳定收益。F2三个seed均胜出不等于训练稳定性已解决。F2误差集中于少数目标，历史I2报告五目标SSE占比37.54%–84.70%，最大SSE目标row2952/1793.5 h、multiplicity19。这里只引用已保存报告，没有再读取Assessment输入或进行Assessment forward。

原Development Assessment存在历史选择复用，非独立Test；Validation也曾用于训练/选择。重叠窗口和同数据三个seed不能提供独立样本显著性。门控与工况的数学依赖不证明物理因果机理，固定.5替换有训练/推理条件差异。C1/D0/E0历史复现未决，不能与不同协议TimesNet直接排名。

当前动态ΔV输出不是独立SOH ground truth，不完成RUL。未来需要明确健康监督和参考条件、动态电压与健康状态映射、未来健康轨迹、观测/外推EOL与右删失说明、阈值和不确定性校准及独立验证。无自动下一阶段、正式Test打开或额外架构训练。
''')
    j.text_write(j.PHASE/'reports/J_FINAL_ARCHITECTURE_DECISION.md',f'''# Phase J 最终结论

{common}

Architecture: `{summary['architecture_status']}`。

Gate evidence: `{summary['gate_status']}`。

## 原有结构消融证据（仅读取已保存结果）

{table(hist)}

## 当前门控诊断

{table(stats[['seed','mean','SD','min','max','P5','P25','P75','P95','low_lt_0p1_fraction','middle_0p4_to_0p6_fraction','high_gt_0p9_fraction']])}

{table(cmp)}

## 七项研究交接

1. F2可以作为TFT分支的冻结动态ΔV预测架构：真实源码、9个checkpoint身份、参数量、三个F2精确重载及只读状态均通过。
2. 门控是否动态：以预注册SD/range和实际记录逐Seed回答，完整值见上表；不把不同Seed的函数强行解释为同一物理状态。
3. 是否影响预测：三个相同checkpoint固定.5对照的预测差异和RMSE差值见上表，结论仅限当前Validation局部数值/预测依赖。
4. 物理机制尚未证明。相关与反事实输出替换不足以识别不可逆退化或电化学因果效应。
5. 本轮不需要、也不授权继续改变TFT结构。即使门控证据不足仍可冻结结构并保留解释限制。
6. 距完整SOH/RUL仍缺健康标签与参考条件、状态/轨迹映射、可验证EOL或删失处理、阈值及不确定性、独立评估。不能把动态ΔV模型直接写成完整寿命预测。
7. 可以进入后续论文实验设计和独立Test方案准备，但必须承认历史复用及复现限制；这是规划准备，不是授权打开Test或继续模型选择。

创新一有结构/工程证据，但F1不支持必然的独立增益。创新二有三Seed历史预测优势及本轮限定门控诊断；允许“历史工况感知融合的经验预测作用”，不允许“真实物理机制已验证”。

New training NO; architecture modification NO; hyperparameter search NO; new Assessment inference NO; formal Test NO; seed selection NO; historical checkpoint modification NO。完成后停止。
''')
    j.text_write(j.PHASE/'README.md',f'''# Phase J — F2 架构冻结与工况门控机制验证

{common}

本阶段固定已有F2定义，仅以Validation开展被动门控采集和唯一g=.5诊断。不训练、不更改网络、不推理Assessment、不打开Test。

状态：{summary['architecture_status']}；{summary['gate_status']}。

运行（从本目录，python -B；历史推理入口禁止重复）：

```text
python -B scripts/audit_j_architecture.py --freeze
python -B -m unittest discover -s tests -v
python -B scripts/run_j_audit.py --execute
python -B scripts/audit_j_readonly.py
python -B scripts/analyze_j_gate_behavior.py
python -B scripts/plot_j_figures.py
python -B scripts/write_j_reports.py
```

协议只冻结一次，执行token禁止已进入的parity/full/fixed stage重复。统计/只读审计可从缓存复核。输入仅合法N1逻辑prefix、Train分组及Validation205origin；新图6组均PNG/PDF/SVG，架构另有Mermaid可编辑源。完整文件清单见outputs/integrity_audit/J_DELIVERY_MANIFEST.json。方法草稿与图像采用academic-paper写作约束和scientific-figure-making样式，均以真实源码/数据为证据，不主张全球首次创新。
''')
    print('J seven reports and README generated from persisted audited records')

if __name__=='__main__':write_reports()

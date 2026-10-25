"""原始图、来源索引和中文核验报告；没有预测误差或健康特征拟合。"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from 核验基础 import table, sha, save_json


def create_figures(cfg, out, fc1, real, polar, gaps, fc_summary, real_summary):
    os.environ['MPLCONFIGDIR'] = str(out/'绘图缓存')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10,
                         'axes.grid': True, 'grid.alpha': 0.18})
    image_dir=out/'图片'; sources=[]
    def finish(fig,name,data,note):
        fig.tight_layout(); fig.savefig(image_dir/name,dpi=cfg['绘图DPI'],bbox_inches='tight');plt.close(fig)
        sources.append(dict(图片=name,绘图CSV=data,说明=note,DPI=cfg['绘图DPI']))
    def lines(ax,frame,y,color):
        for _, group in frame.groupby('连续段',sort=False):
            ax.plot(group.运行小时,group[y],lw=0.45,color=color,marker='.' if len(group)<2 else None,ms=1.5)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    lines(axes[0],fc1,'平均单片电压_V','#176a86');lines(axes[1],fc1,'电堆总电压_V','#5d7690')
    axes[0].set(title=f'FC1 原始运行观测（{fc_summary["rows"]:,}条；无滤波/插值）',ylabel='平均单片电压 (V)')
    axes[1].set(xlabel='实际累计运行坐标 Time (h)',ylabel='电堆总电压 (V)')
    finish(fig,'FC1原始运行电压.png','FC1原始运行绘图点.csv','按源文件及间断分段；横轴为原Time(h)，不补点')
    fig,ax=plt.subplots(figsize=(12,4.8));lines(ax,real,'平均单片电压_V','#176a86')
    ax.set(title=f'RealHyFC 原始十单片均值（全量{real_summary["rows"]:,}条统计；绘图{len(real):,}点）',
           xlabel='实际 OpHrs2 累计运行小时 (h)',ylabel='平均单片电压 (V)')
    finish(fig,'RealHyFC原始运行电压.png','RealHyFC原始运行绘图点.csv','固定行间隔抽样及端点/断点两侧，所有纵坐标均为原始均值；U6不参与')
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True);lines(axes[0],real,'电流_A','#ba7628')
    axes[0].set(title='RealHyFC 原始电流与实际日历观测间断',ylabel='实际电流 (A)')
    gap=pd.DataFrame(gaps);gap=gap.loc[(gap.电堆=='RealHyFC')&(gap.时间差秒>1.05)]
    if len(gap):
        axes[1].scatter(gap.运行小时,gap.时间差秒,s=14,color='#a7443d');axes[1].set_yscale('log')
    axes[1].set(xlabel='实际 OpHrs2 累计运行小时 (h)',ylabel='日历观测间隔 (s，对数)')
    finish(fig,'RealHyFC运行电流与数据间断.png','RealHyFC原始运行绘图点.csv;运行观测间断明细.csv','间隔是原DateTime差，不是相同大小的运行小时间隔')
    def curves(ax,frame):
        groups=list(frame.groupby('文件名',sort=False));palette=plt.get_cmap('viridis')
        for k,(file,g) in enumerate(groups):
            color=palette(k/max(1,len(groups)-1));label_done=False
            for direction,marker in [(0,'o'),(-1,'v'),(1,'^')]:
                part=g.loc[g.扫描方向==direction]
                if part.empty:continue
                label=('BoL' if 'BoL' in file else f'{g.标称老化小时.iloc[0]:g} h（标称）') if not label_done else None
                ax.scatter(part.电流密度_A_cm2,part.平均单片电压_V,s=2,alpha=0.35,color=color,marker=marker,label=label)
                label_done=True
        ax.set(xlabel='原实测电流密度 (A/cm²)',ylabel='平均单片电压 (V)')
        ax.legend(fontsize=8,ncol=2,markerscale=3)
        ax.text(0.01,0.01,'▼降载  ▲升载  ○未定；方向为描述性滞回确认，不平滑原始点',transform=ax.transAxes,fontsize=8)
    for stack in ('FC1','RealHyFC'):
        frame=polar.loc[(polar.电堆==stack)&(~polar.启停后)]
        if frame.empty:continue
        fig,ax=plt.subplots(figsize=(10,6));curves(ax,frame)
        ax.set_title(stack+' 原始极化测量（标称检查点；非独立寿命样本）')
        finish(fig,stack+'原始极化曲线.png','极化原始测量点.csv','保留全部测量点及顺序方向标签；未选稳态点、未插值、未拟合')
    pairs=[]
    for age in (1500,2300):
        frame=polar.loc[(polar.电堆=='RealHyFC')&(polar.标称老化小时==age)]
        if frame.启停后.nunique()==2:pairs.append((age,frame))
    if pairs:
        fig,axes=plt.subplots(1,len(pairs),figsize=(7*len(pairs),5),squeeze=False,sharey=True)
        for ax,(age,frame) in zip(axes[0],pairs):
            for after,color,label in [(False,'#787878','名义测试'),(True,'#177d93','Air/AirStart后测试')]:
                part=frame.loc[frame.启停后==after]
                ax.scatter(part.电流密度_A_cm2,part.平均单片电压_V,s=2,alpha=0.4,color=color,label=label)
            ax.set(title=f'{age} h（文件标称）',xlabel='实测电流密度 (A/cm²)',ylabel='平均单片电压 (V)')
            ax.legend(markerscale=3,fontsize=9)
        fig.suptitle('RealHyFC 启停前后原始极化对照：描述性配对，不等于退化修复或因果证据')
        finish(fig,'RealHyFC启停前后极化对比.png','极化原始测量点.csv','两次测试先后顺序由原DateTime核实；不宣称条件完全相同')
    for row in sources:
        row['图片SHA256']=sha(image_dir/row['图片'])
        row['源CSV_SHA256']=';'.join(sha(out/'表格'/name) for name in row['绘图CSV'].split(';'))
    table(out/'表格/图片与原始来源索引.csv',sources)
    return sources


def code_review(root, report_path, identities):
    text='''# 数据读取代码审查报告

本次只静态阅读历史源码，没有导入或执行旧训练入口。根README也已阅读。历史输出不改写。

## FC1既有读取程序

来源：`New-PEMFC-Pytorch/New-PEMFC-Pytorch - 副本22,FC1版,最终版Ⅱ.py`。

- 入口在旧脚本顶层，按旧绝对路径递归搜索CSV；ISO-8859-1解码、默认逗号分隔。当前仓库实际找到的是FC1 Excel副本，旧CSV入口不能直接当作当前数据入口。
- 先挑选名称包含Static的CSV，未找到时回退FC1/FC2所有Ageing/EIS/Pola文件；按文件名分组。
- 老化表按Time(h)排序，然后按整数小时取第一条，属于降采样；排序前没有原始次序审计，不宜用已排序结果验证原始时序。
- 在全时域表上ffill、bfill、fillna(0)，再进行居中Savitzky–Golay(21,2)滤波。bfill使用后续记录，居中滤波可能跨越后续训练边界，因此不能直接复用为严格因果的历史输入。
- 旧目标Utot(V)是电堆总电压；新阶段图分别标识Utot和五单片均值，不能混用电压层级。
- 该版随后使用50/10/40时序切分，MinMaxScaler在Train拟合。Train-only scaler是正确隔离环节，但不能抵消此前全局回填/居中平滑带来的信息交叉。
- 本次不判断或重算旧模型成绩，仅指出后续新因果协议不得照搬上述前处理。

## RealHyFC既有读取程序

来源：`RENew-PEMFC-2300h-Pytorch/Before/REAL_Paper_File/F2主程序/raw1s/data.py`及`features.py`。

- 17份Loadcycle由冻结清单登记，UTF-8-SIG、分号分隔、分块读取，读取前验证大小和SHA256。
- 十个有效单片为U1–U5及U7–U11，U6独立；缺一单片不能用其余单片的nanmean代替定义。
- DateTime要求严格递增；950–1050 ms、OpHrs2不回退、原传感器有限决定连续段，文件边界强制断开。不插值或通过电压大小筛除困难样本。
- 原窗口300条历史+120条目标均位于同分区、同连续段；24候选仪器值先转float32缓存。
- 30点统计只用当前历史窗口，窗口首条特征初始化为0，不借用未来记录。该阶段没有EWMA或旧半小时Top10掩码。
- Train-only Spearman/标准化取预定最多10000个Train合法起点；Spearman使用Train未来标签是监督特征筛选，不是在线未来输入。后续任务仍必须冻结训练区间后再拟合，不能用本轮全量描述统计拟合模型预处理。
- 原numeric把同时含逗号和点的字符串视为千位分组。当前原始OpHrs确有`1,415.16`等形式；新程序只接受明确千位格式，并记录无法解析值，不推测含糊混合格式。

## 本阶段独立实现

不调用旧模型、旧scaler、预测缓存或加工标签。Excel按原行顺序读取；RealHyFC按历史文件顺序实际分块复核，保留所有原始记录用于统计。没有平滑、插值、归一化、特征拟合或模型评价。

FC1字段`J (A/cmｲ)`及温度字段的全角字符来自现有Excel副本；原字段名原样登记，显示单位结合字段语境和I/J数值一致性核验。FC1有效面积仅可从I/J作观测推断，缺独立规格证据；不能把任意50或100 cm²硬编码为已确认物理参数。

RealHyFC极化表部分只有OpHrs，未给OpHrs2。两个计时器不能视为相同，也不以固定偏移或文件名标称小时补算。以原始日历时钟找相邻运行记录，缺直接OpHrs2时仅报告单调小时边界。测量结束和结果发布是不同的可用时间。

本轮显式获准描述性读取2000/2300 h曲线；不重选历史模型、不改变历史Test资格，也不把描述性全数据统计当作未来模型的全局拟合输入。
'''
    report_path.write_text(text,encoding='utf-8')
    save_json(report_path.with_name('历史代码只读身份记录.json'),identities)


def make_report(path,cfg,fc,real,integrity,coverage,alignment,figures,output,code_version):
    blocks=[]
    if real['rows']!=8604654 or not real['historical_sha_all_match'] or not real['all_rows_match']:
        blocks.append('RealHyFC重新读取的行数或历史SHA身份不一致')
    clock_bad=sum(int(r.get('日历时间倒序',0))+int(r.get('跨文件重叠',0))+int(r.get('超容限小时回退',0))+int(r.get('无效时间',0)) for r in integrity if r['电堆']=='RealHyFC')
    if clock_bad:blocks.append('RealHyFC时间身份需进一步处理')
    fc_bad=sum(int(r.get('小时回退',0)) for r in integrity if r['电堆']=='FC1')
    if fc_bad:blocks.append('FC1运行时间有回退，不能默认为连续排序后的真值')
    qualified={s:sum(r['覆盖完整共同区间'] and not r['启停后'] for r in coverage if r['电堆']==s) for s in ('FC1','RealHyFC')}
    if any(v<2 for v in qualified.values()):blocks.append('共同区间可支持的极化检查点不足')
    status='BLOCKED' if blocks else 'CONDITIONAL_PASS'
    brief=[f'# 阶段0数据核验报告：{status}', '',
        f'FC1实际读取{fc["source_files"]}份老化表、{fc["rows"]:,}条运行记录，累计运行坐标{fc["first_hour"]:.6f}–{fc["last_hour"]:.6f} h。',
        f'RealHyFC实际重新读取{real["source_files"]}份Loadcycle、{real["rows"]:,}条记录，OpHrs2={real["first_hour"]:.6f}–{real["last_hour"]:.6f} h；历史SHA和行数核对结果为{real["historical_sha_all_match"] and real["all_rows_match"]}。', '',
        '运行与极化原始数据可以读取，但**数值电流密度覆盖并不等于跨电堆工况已可比**。FC1现有极化表没有诊断日历时间和同时温压，原始CSV未找到；RealHyFC若干曲线没有直接OpHrs2，结果发布时间也未知。', '',
        '## 1. 实际数据及完整性', '',
        f'- FC1绝对日历时钟未记录；全局Time(h)重复数={fc["global_duplicate_hours"]}。当前源为Excel副本，不冒称已核验原发布CSV。',
        f'- RealHyFC原DateTime覆盖{real["first_datetime"]}至{real["last_datetime"]}；不填补为0–2300 h。',
        '- 逐文件缺失、重复、回退、零电流及电流密度换算差异见`老化运行数据完整性检查.csv`。负电压、零电流和回升仅登记，不据此判为损伤或修复，不剔除。',
        '- 文件边界及实际日历空档见`运行观测间断明细.csv`；绘图按连续段断线，不把断点连接成连续测量。',
        '- FC1检查Utot减五单片之和；RealHyFC目标为十有效单片严格均值，U6诊断独立，均值定义没有变更。', '',
        '## 2. 原始图与抽样', '',
        f'已生成{len(figures)}张真实PNG（{cfg["绘图DPI"]}dpi）。FC1绘图按原记录，RealHyFC固定每{cfg["RealHyFC作图行间隔"]}行取点，额外保留块端点/断点两侧；全量统计不抽样。',
        '全部原始绘图坐标和来源SHA见`图片与原始来源索引.csv`及各绘图CSV。没有平滑、插值或模型生成曲线。极化每条曲线保留原始点，扫描方向仅是0.05 A/cm²幅度的描述性滞回确认，不是硬件扫描精确动作时刻。', '',
        '## 3. 极化覆盖与统一参考点', '',
        f'在原始观测上覆盖0.3–1.0 A/cm²的非after曲线：FC1={qualified["FC1"]}条，RealHyFC={qualified["RealHyFC"]}条；具体每条及升降载分段见覆盖表。',
        '0.9 A/cm²只作为候选参考。精确实测点、邻域测量支持和同一方向分段括距均已报告；没有在本阶段插值或构造SOH。选择分支、稳态平台和插值规则仍需阶段1预先注册。',
        'FC1表内I/J支持有效面积的数值推断，RealHyFC部分曲线明确记录280 cm²。推断值不等于独立厂家规格确认；两套电堆单片数、压力单位和工作温度不同，不能直接把电压差解释成健康差。',
        'FC1运行压力为mbara（绝压），RealHyFC为kPag（表压）；没有本地大气压证据时不能直接数值比较或暗中换算。', '',
        '## 4. 时间对齐与少样本适配', '',
        'FC1的全部极化时间资格为NOMINAL_ONLY：按Txxx文件名找运行时间邻点只是标称定位，不是诊断真实时间的恢复。不能把该时间作为已证实的在线标签发布时间。',
        'RealHyFC极化的真实DateTime与运行记录逐点找最近前/后记录；缺OpHrs2时仅报告运行小时上下界，未用名义300/600 h或OpHrs固定偏移替代实际小时。BoL早于首份运行记录，不能硬配为同一起点。',
        f'RealHyFC按实测OpHrs2计数的原始运行观测：{real["adaptation_raw_rows"]}。这些记录量说明观测支持，不是模型适配已验证。',
        '重点曲线（物理结束资格仍不等于发布时间资格）：', '']
    for r in alignment:
        if r['电堆']=='RealHyFC' and r['标称老化小时'] in (0,300,600):
            brief.append(f'- {r["文件名"]}：日历{r["实际测试开始"]}至{r["实际测试结束"]}；OpHrs2边界[{r["累计小时下界"]}, {r["累计小时上界"]}]；前300h={r["前300h物理结束资格"]}，前600h={r["前600h物理结束资格"]}；发布时刻未知。')
    brief += ['', '只有独立确认诊断发布时间、实际训练截止和可用标签后，才能冻结前300/600 h少样本适配协议。当前对齐表不能自动作为因果训练标签。历史周期ID仅在原RCR日期区间覆盖整条曲线时登记，未匹配则保持未知。', '',
        '## 5. 启停配对与物理边界', '',
        '1500/2300标称曲线如存在before/after文件，按原日期核查先后并画原点对照；不假设相同硬件状态、完全一致工况或不可逆退化被修复。具体温压在覆盖表逐文件列出，FC1极化同时条件没有测量字段，不能由附近运行值冒充。', '',
        '## 6. 阶段验收与下一阶段', '',
        ('阻断项：'+'；'.join(blocks)) if blocks else '没有发现阻止读取和描述性可视化的数据身份错误。主要限制是标签时钟、原始来源溯源和跨电堆物理条件资格。',
        '可以有条件进入**离线、描述性的二维极化健康特征提取**：保留每条曲线、方向、测试身份，不合并为独立寿命样本；先冻结稳态点和参考支撑规则。',
        '目前不能据此宣布已具备严格在线跨电堆迁移、已验证SOH精度或真实RUL标签。若坚持因果少样本预测，诊断发布时间与FC1真实测量时刻是必须补齐的证据。',
        f'代码版本：{code_version}。所有新成果位于PEMFC指导，旧代码/权重/结果不修改。未执行训练、预测精度、SOH/RUL或特征拟合；未commit/push。', '']
    path.write_text('\n'.join(brief),encoding='utf-8')
    return status,blocks

"""Stage-one synthesis of directly checked, graded paper extractions; no experiment data."""
from common import *
KEYS=['research_target','dataset_conditions','input_information','forecast_mode','horizon_physical_time','main_model','innovation','comparison_models','ablation','condition_analysis','robustness_generalization','metrics','figure_types','section_structure','borrowable_practices','not_directly_transferable','locators']
LABELS=['研究问题与目标','数据及工况','输入信息','单步/多步方式','跨度及物理时间','主要模型','方法创新','比较模型','模块消融','不同工况分析','稳健性及泛化','评价指标','核心图表','章节结构','本项目可借鉴','不能直接照搬','定位汇总']
MAPPING=dict(zip(KEYS,['research_target','dataset_conditions','input_information','forecast_mode','horizon_physical_time','model_structure','claimed_innovation','comparators','ablations','condition_analysis','robustness_generalization','metrics','figures_tables','section_organization','adoptable_practices','nontransferable_practices','specific_locator_summary']))
def main():
    meta=read(P/'literature/verified_metadata.json');a=read(P/'literature/core_P1_P5_evidence.json');b=read(P/'literature/P6_aux_evidence.json')
    papers={r['paper_id']:r for r in a['papers']}
    for r in b['papers']:
        k={'AUX_JIANG':'A1','AUX_YI':'A2','AUX_KO':'A3'}.get(r['paper_id'],r['paper_id']);papers[k]=r
    records=[];wide=[];md=[];overview=[];figs=[];source=[]
    intro='# 已发表PEMFC论文横向比较矩阵\n\n核验日期2026-10-17；定向六篇核心+三篇辅助，另核对原TFT与RealHyFC数据论文。不是系统综述，不支持全领域排他性首创。7篇本地全文本轮直接重核并查看关键图表；Tu(P3)及Yakut(P5)未取得正文，仅官方摘要/有限片段，UNKNOWN不是未做。所有页码为PDF从1起的页序。每字段等级与定位保存在长表CSV/原JSON；不把不同论文RMSE直接排名。\n\n'
    for m in meta['records']:
        if m['id'] not in papers:continue
        k=m['id'];r=papers[k];f=r.get('fields_by_key') or {key:r['fields'][MAPPING[key]] for key in KEYS};assert len(f)==17
        is_full=k not in ['P3','P5'];status='本轮原PDF方法/实验全文重核+关键页面视觉检查' if is_full else '官方摘要/有限片段；全文不可访问'
        local=r.get('source_relative_to_repository',r.get('local_pdf'));digest=r.get('source_sha256',r.get('local_pdf_sha256'))
        if local:assert (REPO/local).is_file() and sha(REPO/local)==digest,('LOCAL_PDF_IDENTITY',k)
        overview.append(dict(ID=k,作者='; '.join(m['authors']),年份=m['bibliographic_year'],期刊=m['journal'],DOI=m['doi'],证据状态=status))
        source.append(dict(id=k,doi=m['doi'],local_pdf=local,sha256=digest,publisher=m['publisher_url'],metadata=m['raw_crossref_path'],access=status))
        rows=[];ww={'paper_id':k,'doi':m['doi'],'title':m['title'],'access_status':status}
        for i,(key,label) in enumerate(zip(KEYS,LABELS),1):
            cell=f[key];value=cell['value'];grade=cell.get('evidence_grade','UNKNOWN');loc=cell.get('locator','UNKNOWN')
            url=cell.get('source_url',m['publisher_url']);records.append(dict(paper_id=k,doi=m['doi'],field_number=i,field=key,value=value,evidence_grade=grade,locator=loc,source_url=url))
            rows.append({'核对项':label,'已核实内容/未知边界':value,'证据等级与定位':grade+'; '+loc});ww[key]=value
        author_label=m['authors'][0]+('等' if len(m['authors'])>1 else '')
        wide.append(ww);md.append(f"## {k} · {author_label} ({m['bibliographic_year']})\n\n*{m['title']}*，{m['journal']} {m.get('volume') or ''}，{m.get('article_number') or m.get('pages') or ''}。[DOI]({m['doi_url']})。\n\n证据：{status}。\n\n"+mdtable(pd.DataFrame(rows)))
        figs.append(f"## {k}\n\n**主要结果/消融应学什么：** {f['comparison_models']['value']} {f['ablation']['value']}\n\n**图表：** {f['figure_types']['value']}\n\n**章节组织：** {f['section_structure']['value']}\n\n**本稿映射：** {f['borrowable_practices']['value']}\n\n**边界：** {f['not_directly_transferable']['value']}\n\n**定位：** {f['figure_types'].get('locator','UNKNOWN')}；{f['section_structure'].get('locator','UNKNOWN')}。")
    assert len(wide)==9 and len(records)==153
    cs('outputs/evidence_index/LITERATURE_FIELD_EVIDENCE.csv',records);cs('outputs/evidence_index/LITERATURE_COMPARISON_WIDE.csv',wide);js('outputs/evidence_index/LITERATURE_SOURCE_INDEX.json',source)
    text('literature/01_已发表PEMFC论文横向比较矩阵.md',intro+mdtable(pd.DataFrame(overview))+'\n\n'+'\n\n'.join(md)+'\n\n## 本项目任务身份\n\nRealHyFC的10-cell耐久记录；24h历史输入（48×0.5h），同一起点直接输出未来12h（24步）原始ΔV。推理decoder只接收确定的相对步长，不接未来真实电流、温度、压力或其他实际工况。信息边界可区别具体实现，但原TFT本就区分已知/未知变量，不能声称首创未知未来预测。我们的误差来自Development，既有Assessment被复用，筛选六实例不是随机总体。上述文献不同数据/采样/目标/滚动方式/输入信息条件不能直接比较绝对误差。\n')
    text('literature/02_参考文献核心实验与图表清单.md','# 核心实验与图表清单\n\n先参照原文的问题—结果逻辑，再用本项目自己的真实CSV重绘；不复制原论文插图或结果数字作为本稿实验。仅摘要可得的两篇不作为图号/章节结构证据。\n\n'+'\n\n'.join(figs)+'\n\n## 本文图表取舍\n\n正文以任务/结构、三Seed公平比较、24步误差、历史工况分组和独立训练消融为主；六个优秀实例、十次门控置换、目标SSE与极端窗口放补充材料或有限讨论。P6三层结果组织（模型/跨度/消融）可借鉴，P4原生/改进TFT对照可借鉴；不会因参考文献包含RUL或物理模型而扩展本文。\n')
    text('literature/03_本项目创新点与重叠风险.md',"# 候选创新与重叠风险\n\n文献搜索限用户给定9篇及两项基础来源，日期2026-10-17。以下是与最近邻方法的精确差异，不是全领域首创证明。\n\n|候选主张|最近邻/重叠|本稿可保留的差异|证据限制|\n|---|---|---|---|\n|双路径历史编码|P1已有电压与ARIMA条件卷积路径；P3已有TCN-LSTM；A1已有组合预测|F2在同一encoder VSN表示上并行保留原LSTM与因果depthwise CNN，post-LSTM encoder位置融合|不是泛称CNN+LSTM首次；独立卷积分支收益需F1/F0，不能以整体F2胜出替代|\n|TFT用于PEMFC并利用协变量|P2已做到TFT、工况信息及解释性；P4已有改进TFT与native对照|明确部署输入契约、固定48→24原始ΔV与自有实现|不是PEMFC首个TFT，不是首次未来未知预测|\n|历史工况门控|原TFT已有VSN/GRN/GLU等门控；A1公式的输出择优与F2不同|5项历史统计控制encoder两条路径特征混合，可复算真实g|数值依赖不等于独立精度提升；P2训练消融未通过，六筛选实例方向不一致|\n|多步预测|P1滚动块预测与真实测量更新；P4递归及多小时实验；P5明确单/多步|同起点直接24步输出，只用历史实测输入|逐horizon评价是固定模型诊断，不是重新训练多种输出长度的对照|\n|透明评价|P6模型/跨度/消融结构；多文献已有多数据或重复实验|同信息的H三Seed、Train固定工况、端点/尾部/target-time-equal及筛选全披露|不是跨stack泛化证据，不宣称显著性；单数据且评价复用是主要审稿风险|\n\n## 本轮更正与证据冲突\n\nA2 Yi的原PDF p2§2为90kW、400cell、1374h自建台架，取极化0.7 A/cm²点序列，并非Phase O记录的PHM。本轮只更正新材料，不改历史。A3 Ko按原文与Crossref为4作者，不采用二手聚合虚加作者。P6正式表格与摘要概括范围存在差异，事实以具体表/任务为准，不搬其摘要区间作跨论文排名。P4存在native TFT与特征选择对照，不能把文献写成完全缺少消融。\n\n## 可以写进工作稿的贡献边界\n\n聚焦一个冻结任务上的双路径TFT实现、清楚的历史信息接口及可追溯实证。将门控描述为被实现并接受对照检验的设计，不写成已证实的独立精度提升或电化学因果机制。精度数字和贡献等级必须分别由本项目正式结果核验。SCI分区不能替代方法新意、可信对照和评价独立性；本阶段不提供录用保证。\n")
    refs=['# 核验参考文献\n\n[P1]–[P6]核心；[A1]–[A3]辅助；[TFT]原方法；[DATA_PAPER]数据来源。实际证据等级及页码见矩阵/索引，不将元数据核验等同全文阅读。\n']
    for m in meta['records']:
        refs.append(f"## {m['id']}\n\n{'; '.join(m['authors'])}. ({m['bibliographic_year']}). {m['title']}. *{m['journal']}*, {m.get('volume') or ''}, {m.get('article_number') or m.get('pages') or ''}. [DOI]({m['doi_url']}).\n")
    text('literature/REFERENCES.md','\n'.join(refs))
    js('outputs/audits/LITERATURE_PHASE_COMPLETE.json',dict(date='2026-10-17',papers=9,fields=153,local_full_text_rechecked=7,abstract_only=['P3','P5'],systematic_review=False,
        files={f:sha(P/f) for f in ['literature/01_已发表PEMFC论文横向比较矩阵.md','literature/02_参考文献核心实验与图表清单.md','literature/03_本项目创新点与重叠风险.md','literature/REFERENCES.md']},experiment_computation_started=False))
    print('LITERATURE_FIRST_STAGE_COMPLETE_9_PAPERS_153_FIELDS')
if __name__=='__main__':main()

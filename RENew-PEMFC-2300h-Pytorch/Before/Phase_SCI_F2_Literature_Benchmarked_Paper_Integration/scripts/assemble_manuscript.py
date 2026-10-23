"""Concatenate the six reviewed Chinese working chapters; keep sources unchanged."""
from common import *
def main():
    verified_config();files=sorted((P/'manuscript').glob('SCI_0[1-6]_*.md'));assert len(files)==6
    disclosure='''# 工作稿声明与待确认事项

## Data and code availability

RealHyFC原始耐久性数据来自[Fensterle等数据记录](https://doi.org/10.5281/zenodo.20715542)，官方元数据标记CC BY4.0。对应原始数据论文为[10.1016/j.ijhydene.2026.156060](https://doi.org/10.1016/j.ijhydene.2026.156060)。本文仅使用已冻结合法Development接口；本阶段没有读取或发布新的Test数值。代码、缓存复算、图源数据及SHA索引保存于本项目独立整合阶段，发布状态以本阶段Git回执为准。不能把数据许可自动解释成所有第三方软件或论文PDF的许可。

## AI assistance

AI辅助工具用于定向文献定位与整理、代码编写、既有缓存数值复核、原创图表生成和中文工作稿起草。文献已按实际可访问范围分级；不同代理的内部检查不等于真人同行评议。研究负责人应审阅实验解释与全文、核验引用并对最终投稿内容负责。AI工具不列为作者。

## Author contributions / funding / conflicts / ethics

本工作稿尚未确认作者名单和CRediT分工、基金项目及编号、利益冲突真值；这些信息须由研究负责人和实际作者填写，不能默认“无资助”或“无冲突”。本阶段未开展人体或动物实验，仅处理公开电化学实验资料和既有预测缓存。正式伦理声明以作者单位与目标期刊要求核定。本文件是待审核声明，不是已签署投稿声明。

## Scientific scope

本文为单RealHyFC记录上反复使用Development的探索性点预测研究，不构成独立确认性Test、跨堆泛化、因果机理或SOH/RUL验证。两篇论文尚无全文；其未知细节不是否定证据。门控设计和数值依赖可以报告，独立稳定精度收益不能报告为已证实。
'''
    text('manuscript/SCI_07_声明与参考文献说明.md',disclosure)
    blocks=['# 基于历史双路径时间融合Transformer的PEMFC动态电压多步预测\n\n中文可追溯工作稿，日期2026-10-17。供研究负责人审核，不是投稿终版。正文六章与自有图表/原始CSV对应，章末工作稿溯源不属于拟投稿正文。\n']
    blocks += [f.read_text(encoding='utf-8') for f in files]
    blocks += [disclosure,(P/'literature/REFERENCES.md').read_text(encoding='utf-8')]
    text('manuscript/SCI_完整中文工作稿.md','\n\n---\n\n'.join(blocks).rstrip()+'\n')
    cs('outputs/evidence_index/MANUSCRIPT_CHAPTER_INDEX.csv',[dict(chapter=i,path=str(f.relative_to(P)).replace('\\','/'),sha256=sha(f),characters=len(f.read_text(encoding='utf-8'))) for i,f in enumerate(files,1)])
    print('SIX_CHAPTER_WORKING_MANUSCRIPT_ASSEMBLED')
if __name__=='__main__':main()

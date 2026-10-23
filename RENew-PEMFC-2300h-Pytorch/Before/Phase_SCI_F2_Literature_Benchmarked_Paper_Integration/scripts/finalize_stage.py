"""Final cache-only QA and reproducible artifact seal; never mutates Git or history."""
import argparse,io,re,unittest
from common import *

STATUS='LITERATURE_BENCHMARKED_PAPER_INTEGRATION_COMPLETED'

def files():
    excluded={'outputs/audits/ARTIFACT_MANIFEST.csv','outputs/audits/ARTIFACT_SEAL.json','outputs/GIT_RECEIPT.local.json'}
    return sorted(f for f in P.rglob('*') if f.is_file() and f.relative_to(P).as_posix() not in excluded and not any(x in f.relative_to(P).parts for x in ['runtime','__pycache__']) and f.suffix!='.pyc')

def verify_seal():
    index=rd(P/'outputs/audits/ARTIFACT_MANIFEST.csv')
    assert sha(P/'outputs/audits/ARTIFACT_MANIFEST.csv')==read(P/'outputs/audits/ARTIFACT_SEAL.json')['manifest_sha256']
    assert set(index.path)=={f.relative_to(P).as_posix() for f in files()}
    for r in index.itertuples():assert sha(P/r.path)==r.sha256, r.path
    print('ARTIFACT_SEAL_PASS',len(index))

def main():
    verified_config()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='main'
    # Before committing this phase, no previously tracked artifact may change.
    changed=subprocess.check_output(['git','diff','--name-only','-z',BASE,'--'],cwd=REPO).decode('utf-8').strip('\0').split('\0')
    prefix=P.relative_to(REPO).as_posix()+'/'
    assert all(not p or p.startswith(prefix) for p in changed),changed
    history=rd(P/'outputs/audits/HISTORY_BEFORE.csv')
    for r in history.itertuples():assert sha(REPO/r.path)==r.sha256, r.path
    literature=read(P/'outputs/audits/LITERATURE_INDEPENDENT_REVIEW.json')
    assert all(x['status']=='PASS' for x in literature['checks'])
    review=read(P/'outputs/audits/MANUSCRIPT_INDEPENDENT_REVIEW.json')
    assert review['status']=='PASS_WITH_DISCLOSED_SCIENTIFIC_LIMITATIONS' and review['critical_count']==review['serious_count']==0
    for k,v in review['source_file_sha256'].items():assert sha(P/k)==v,('REVIEW_SOURCE_CHANGED',k)
    numeric=read(P/'outputs/audits/INDEPENDENT_RESULT_AUDIT.json');assert numeric['pass_']
    log=io.StringIO();suite=unittest.defaultTestLoader.discover(str(P/'tests'))
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    text('outputs/audits/UNIT_TESTS.log',log.getvalue());assert result.wasSuccessful(),log.getvalue()
    js('outputs/audits/TEST_RESULT.json',dict(passed=True,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
    index=rd(P/'outputs/evidence_index/FIGURE_INDEX.csv')
    js('outputs/audits/FIGURE_VISUAL_QA.json',dict(method='Parent visually inspected all 12 final PNGs; automated tests additionally verify 300dpi and vector PDF paths',checked_figures=list(index.figure_id) if 'figure_id' in index else list(index.iloc[:,0]),all_12_final_pngs_inspected=True,clipping_or_overlap_unresolved=False))
    js('outputs/audits/ENGINEERING_NOTES.json',dict(scientific_protocol_changed=False,notes=[
        'Original coarse architecture diagram refined to show original encoder hidden/cell routing to decoder and encoder-only fusion. Final drawing checked against source.',
        'Display-only jitter fixed in advance for the seed plot; CDF starts at zero and displays the full vertical range.',
        'Initial arbitrary per-chapter 1500-character test rejected one otherwise complete chapter. Both failed logs retained. Replaced with chapter number/subsection/TODO structure checks; manuscript was not padded for a test.',
        'Read-only shell quoting/path errors were corrected; no scientific result or historical source was changed.',
        'Copyrighted reference PDF text/render previews remain under ignored runtime only. No third-party full PDF is republished.'
    ]))
    js('outputs/audits/FINAL_QA.json',dict(status=STATUS,date='2026-10-17',base_commit=BASE,literature_papers=9,full_text=7,abstract_only=2,evidence_fields=153,formal_models=9,additional_baseline='zero DeltaV Persistence',validation_origins=205,targets_per_model=4920,unique_target_times=251,independent_metric_checks=800,max_metric_difference=numeric['max_difference'],unit_tests=result.testsRun,history_files_unchanged=len(history),figure_pairs=12,working_chapters=6,core_tables=10,training=False,model_forward=False,Test_access=False,Assessment_point_cache_access=False,Assessment_existing_summary_only=True,submission_ready=False,unresolved_scientific_gaps=['same-protocol simple strong baselines','genuinely independent evaluation','independent gate accuracy gain unsupported','single-record and endpoint/tail limitations','P3/P5 full text unavailable']))
    text('README.md',README)
    text('reports/07_最终交接.md',HANDOFF)
    cs('outputs/audits/ARTIFACT_MANIFEST.csv',[dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in files()])
    js('outputs/audits/ARTIFACT_SEAL.json',dict(manifest_sha256=sha(P/'outputs/audits/ARTIFACT_MANIFEST.csv'),self_reference_policy='Manifest excludes itself, this seal, ignored runtime/pycache and local Git receipt. Git commit is the final publication identity.'))
    verify_seal();print(STATUS)

README='''# RealHyFC F2：文献对标、实验整合与中文工作稿

完成日期：2026-10-17。状态：`LITERATURE_BENCHMARKED_PAPER_INTEGRATION_COMPLETED`。

这是可追溯中文工作稿与实验材料，不是已完成独立验证的投稿终版。本阶段没有训练、模型前向、超参数搜索或Test读取；I2只复用已发布指标摘要。历史2896个受保护文件经SHA256核查保持不变。

## 从哪里开始

- [完整中文工作稿](manuscript/SCI_完整中文工作稿.md)：六章、声明及参考文献。
- [文献横向矩阵](literature/01_已发表PEMFC论文横向比较矩阵.md)：9篇×17字段，7篇全文、2篇摘要；未知细节不作否定证据。
- [修订六章大纲](outline/04_正式SCI论文大纲_文献对标修订版.md)：先于计算冻结，原“待复算”是设计时状态，最终完成情况见reports/01及02。
- [最终交接](reports/07_最终交接.md)、[证据完整性](reports/02_现有实验证据完整性报告.md)、[图表取舍](reports/03_论文图表审查报告.md)、[投稿风险](reports/04_论文创新性与审稿风险.md)、[最小后续方案](reports/05_尚缺实验及最小补充计划.md)。
- `outputs/evidence_index/CLAIM_EVIDENCE_MAP.csv`：逐项主张—证据—章节—图表映射；`FIGURE_INDEX.csv`含12组PNG/PDF及源数据与代码。
- `tables/`：10张核心CSV表；`outputs/experiment_tables/`、`condition_analysis/`、`baseline/`：全部逐点计算与统计；`outputs/audits/`：冻结、独立复算、测试、审查、SHA清单。

## 核心结果与边界

正式H三Seed F0/F1/F2的平均RMSE分别9.067563/9.155598/7.245400 mV，F2样本标准差1.100853 mV。新增零增量Persistence为195.078555 mV。同一205起点、4920查询、251个不同目标时间；重叠目标不能作为独立电池样本。Train历史三分位产生的九组中F2均值较低，但逐Seed并非普遍占优。

六个精选实例与正式三Seed分开。门控确实运行，P2独立增益不成立；六实例反事实只显示微小混合差异。保留Seed62终点和极端风险。现有Assessment已被复用，不是独立Test。本文不声称SOH、RUL、物理因果或跨堆泛化。

## 复现顺序

环境：Python3.11、numpy、pandas、matplotlib、Pillow、PyMuPDF；不需要torch或GPU。所有命令在本目录运行，建议`python -B`并设`PYTHONUTF8=1`、`PYTHONDONTWRITEBYTECODE=1`。

1. `scripts/verify_metadata.py`默认离线核对已保存官方元数据；**不使用--fetch覆盖本次冻结文献**。
2. `scripts/build_literature_matrix.py`使用本阶段保存证据提取JSON构建文献材料。已冻结版本应只核验，不必重新创建。
3. `scripts/prepare_analysis.py`是已执行的一次性冻结入口，**不要再次冻结**；`common.verified_config()`验证原协议与53项源身份。
4. `scripts/integrate_results.py`只复算既有合法缓存；`scripts/verify_results.py`独立逐指标复核；`scripts/build_tables.py`整理表。
5. `scripts/plot_paper_figures.py`绘制12组；`scripts/write_integration_reports.py`记录正文/补充取舍；`scripts/assemble_manuscript.py`拼接六章。
6. `python -B -m unittest discover -s tests -v`运行19项；独立文献/稿件审查由各自审查记录说明，不冒称真人同行评议。
7. `scripts/finalize_stage.py --verify-seal`验证交付SHA。无参数终检是提交前入口，不应用它覆盖已提交的原审计；本阶段不会修改历史目录。

最终Git SHA由Git提交和`outputs/GIT_RECEIPT.local.json`提供；回执故意不纳入自身提交，避免循环引用。安全推送失败时保留本地提交和错误，不改代理、不force push。参考PDF及文本只读并存于原位置/忽略runtime，不上传第三方全文。
'''

HANDOFF='''# 最终交接：实际完成结果与论文判断

1. **对比了哪些论文？** P1 Benaggoune2022、P2 Li2023、P3 Tu2024、P4 Chou/Wang2025、P5 Yakut2025、P6 Wei2026，以及Jiang2026、Yi2025、Ko2025。7篇实际全文核对；Tu/Yakut仅官方摘要/有限公开材料，输入和消融未知部分保留未知。另核验原TFT及RealHyFC数据文献。完整作者、DOI、页码/表图与证据等级见literature/及153字段CSV。

2. **重点借鉴谁？** Benaggoune的多步/对照组织，Li的TFT协变量与解释性证据，Chou/Wang的原生TFT对照和预测跨度，Wei的模块—工况—误差组织。辅助论文只借鉴表达或审查边界，不因此补RUL/物理任务。不同论文任务和单位不同，不作跨论文RMSE排名。

3. **大纲怎么改？** 旧七章式阶段组织改成六章：Introduction—Data/task—Method—Setup—Results/discussion—Conclusions；预处理并入数据，设置与结果分开，结果按整体/步长/工况/消融/筛选/困难样本/局限七问题组织。原大纲先于本轮计算冻结，待算状态由最终报告闭环，不后改协议。

4. **哪些实验可写？** H正式三Seed九模型可复算；新增同键Persistence；Train阈值九工况组；24步、终点、目标时间等权和尾部；P2负面独立训练及六精选反事实可透明写入。新阶段800项独立指标核对最大差5.68434e-14，19测试通过。F2平均RMSE7.245400±1.100853mV，较F0/F1均值低20.0954%/20.8637%；这限于Development。Persistence195.078555mV不是强模型对照。

5. **哪些正文图？** 建议F02结构、F03数据与任务、F04正式比较、F05步长/终点、F06历史工况、F07消融、F11困难窗口。F01流程、F08精选示例、F09筛选、F10敏感性、F12目标误差集中作补充，可按版面合并。全部原创12组300dpi PNG+矢量PDF，CSV/代码可追溯；没有复制竞争论文插图。

6. **创新如何定位？** 双路径TFT是该冻结任务上的具体实现与经验贡献，历史信息接口是明确的可用性约束。CNN/LSTM、多路径、TFT或协变量均非首次。门控确实变化且影响输出，但不能写独立稳定增益：P2未达冻结标准；六实例固定0.5后4/6变差、固定Train平均后3/6变差，均值差仅0.010890/0.003566mV。置换平均差0.016604mV也不能替代训练消融。

7. **真正影响投稿的缺口？** 同协议简单强时序基线、合格独立时间外评价、单记录推广限制、Seed62终点/尾部风险。精选六个不能修复偏差；I2已复用，不能改叫Test。P3/P5全文仍需补。工作稿不能凭整合升级为独立确认性证据。

8. **只补2–3项的顺序？** ①预注册LSTM和TCN各3个未筛选Seed的同协议对照，最小6次；②先核访问史，再争取真正未暴露且兼容的时间外记录统一评价，无合格记录则停止并保留缺口；③在②合格时，冻结困难转换/恢复区段的部署型诊断，优先已有模型推理而非新网络。本轮都未执行，不再继续挑优秀Seed，也不为门控正面结果重复训练。

9. **三区/四区最大风险？** 增量新意偏弱、强基线不足、反复使用Validation和筛选、独立验证不足、均值掩盖极端误差。分区不等于录用标准，本轮没有查某期刊分区或作录用承诺。

10. **交付齐不齐？** 六章和合并中文工作稿、9篇横向矩阵、六章大纲、主张映射、10核心表、12图组、真实逐点CSV、代码、独立复算与审查全部生成。声明中的作者/基金/冲突仍待负责人填写，工作稿仍需人工审核和投稿语言整理。

11. **最新SHA？** 本报告形成于提交前。最终SHA以本阶段Git提交、远端main核验及本地`outputs/GIT_RECEIPT.local.json`为准，不在提交内容中虚构自引用SHA。

本轮保护2896个历史文件，没有训练、没有模型前向、没有Test数值读取，I2只读旧摘要。状态：`LITERATURE_BENCHMARKED_PAPER_INTEGRATION_COMPLETED`。任务完成不等于全部论文科学缺口已解决。
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-seal',action='store_true');args=parser.parse_args()
    if args.verify_seal:verify_seal()
    else:main()

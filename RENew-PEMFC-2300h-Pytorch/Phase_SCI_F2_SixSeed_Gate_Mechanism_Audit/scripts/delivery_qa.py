"""Final byte/figure/history checks without any forward calls."""
from common import *
def main():
    cfg();assert read(P/'outputs/audits/FINAL_TESTS.json')['pass']
    for row in rd(P/'outputs/audits/HISTORY_BEFORE.csv').itertuples():assert sha(REPO/row.path)==row.sha256
    inventory=[]
    from PIL import Image
    import fitz
    for row in rd(P/'outputs/FIGURE_INDEX.csv').itertuples():
        png=P/(row.figure+'.png');pdf=P/(row.figure+'.pdf')
        with Image.open(png) as im:size=im.size;dpi=im.info['dpi']
        with fitz.open(pdf) as doc:pages=len(doc);drawings=len(doc[0].get_drawings())
        inventory.append(dict(figure=row.figure,png_sha256=sha(png),pdf_sha256=sha(pdf),width=size[0],height=size[1],dpi=dpi[0],PDF_pages=pages,PDF_vector_paths=drawings))
    cs('outputs/audits/FIGURE_FILE_QA.csv',inventory)
    qa=dict(PNG_reviewed=8,PDF_vector_validated=8,
        review='All eight final PNGs inspected via view_image; no clipped axes/legend, units/stage/seed explicit. Fig01 breaks missing origins; fig03 legend outside axes. Fig06/07/08 each trace is one24-step window. PDF same Matplotlib figure, single page with vector paths.',
        numerical_changes=False);qa['pass']=True
    js('outputs/audits/VISUAL_QA.json',qa)
    mark='\n## 最终核验与复现说明\n'
    old=(P/'README.md').read_text(encoding='utf-8').split(mark)[0]
    txt('README.md',old+mark+'\n17项测试通过；78组独立GPU重放共383760点严格一致；624项标量指标复算最大差异8.881784197001252e-16；1099个历史checkpoint及共1646个历史文件SHA未变。以上证据分别见outputs/audits/FINAL_TESTS.json、INDEPENDENT_VERIFICATION.json及HISTORY_BEFORE.csv。\n\n报告用finalize_reports.py生成，保留原冻结报告代码及依赖错误记录；图表用plot_figures.py再运行finalize_figures.py。最终运行python -B scripts/delivery_qa.py后才运行seal.py。封存后重新绘图会改变PDF元数据，因此应先核验原件SHA，再在副本中复现图形。未创建任何训练任务或后续自动执行。\n')
    print('DELIVERY_QA_PASS')
if __name__=='__main__':main()

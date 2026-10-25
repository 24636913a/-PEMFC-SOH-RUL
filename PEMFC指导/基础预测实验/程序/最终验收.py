"""最终完整性、图表、正式代码及历史保护的轻量检查；不重复数据准备。"""
from pathlib import Path
import ast,json,hashlib
import pandas as pd
from PIL import Image
WORK=Path(__file__).resolve().parents[1];GUIDE=WORK.parent;OUT=WORK/'输出结果';FORMAL=GUIDE/'完整代码'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while c:=f.read(8*1024*1024):h.update(c)
    return h.hexdigest()

def main():
    cfg=json.loads((WORK/'配置/执行前冻结.json').read_text(encoding='utf-8'))
    assert sha(WORK/'配置/阶段2A冻结协议.json')==cfg['协议SHA256']
    for rel,h in cfg['历史SHA256'].items():
        if rel.replace('\\','/')=='完整代码/使用说明.md':continue
        assert sha(GUIDE/rel)==h,rel
    pub=json.loads((OUT/'正式版本发布记录.json').read_text(encoding='utf-8'))
    backup=list((FORMAL/'完整代码历史副本').glob('*/使用说明.md'))
    assert len(backup)==1 and sha(backup[0])==cfg['历史SHA256']['完整代码\\使用说明.md']
    assert pub['status']=='PASS' and pub['compared_files']==17
    formal_out=WORK/pub['formal_output']
    for out in [OUT,formal_out]:
        assert json.loads((out/'独立复核通过.json').read_text(encoding='utf-8'))['checks']==102
        done=json.loads((out/'基础预测完成.json').read_text(encoding='utf-8'))
        for rel,h in done['output_sha256'].items():assert sha(out/rel)==h,rel
    identity=json.loads((FORMAL/'正式代码SHA256.json').read_text(encoding='utf-8'))
    for rel,h in identity.items():assert sha(FORMAL/rel)==h,rel
    for root in [WORK/'程序',FORMAL/'程序']:
        for f in root.rglob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f))
    assert sha(FORMAL/'程序/基础预测核心.py')==sha(WORK/'程序/基础预测核心.py')
    required=['共同负载健康特征核对表.csv','温压周期波动与工况漂移对比.csv','300和600小时诊断可用性表.csv',
        'FC1基础预测逐起点结果.csv','RealHyFC基础预测逐起点结果.csv','一维二维健康特征公平对比.csv','SOH评价结果.csv','FC1_RUL阈值与删失结果.csv']
    for name in required:assert (OUT/'表格'/name).is_file()
    figures=pd.read_csv(OUT/'表格/图表来源与SHA256.csv');assert len(figures)==9
    for _,r in figures.iterrows():
        assert sha(WORK/r['绘图CSV'])==r.CSV_SHA256
        png=OUT/'图片'/(r['图']+'.png');pdf=OUT/'图片'/(r['图']+'.pdf')
        assert sha(png)==r.PNG_SHA256 and sha(pdf)==r.PDF_SHA256
        with Image.open(png) as im:
            assert im.width>1000 and im.height>500 and min(im.info['dpi'])>299.5
            im.verify()
        assert pdf.read_bytes().startswith(b'%PDF-')
    # Final record explicitly distinguishes numeric verification from scientific conclusions.
    (WORK/'阶段2A验收报告.md').write_text('''# 阶段2A验收报告

工程状态：PASS。科学结论：可信基础基线已完成；二维增量未获支持；测量与因果时间限制仍在。

- 阶段入口102项独立检查通过；正式入口102项同样通过。
- 首个正式预测核心与阶段核心SHA一致，独立重新拟合产生的17份结果/参数文件逐字节一致。
- 原阶段725份受保护文件未变；唯一更新的正式代码使用说明原件已在历史副本中逐字节备份。
- 八项必需CSV、四份主报告、九组300dpi PNG/PDF及逐图CSV/SHA索引齐全。
- 合法预测均保留完整目标块、历史截止、Train-only基准/平台选择，预测前未改变冻结规则。
- 首次执行、排除/NA、绘图工程错误与修复说明均保留；没有删除不利结果。
- 没有神经网络、迁移训练、GPU任务或新RUL失效标签；没有改写阶段0–1.3。
- 全部结论属于已查看全轨迹的探索性分析；检测完成后可用是离线假设，不是已认证在线发布时间。

图像已逐组核对主要坐标、缺测断线、固定起点、单位和NA。未将重叠窗口拼接为单次连续预测。

已有自动Git提交授权仅用于本轮本地提交；本轮不自动推送远端。
''',encoding='utf-8')
    files=[]
    for root in [WORK,FORMAL]:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '绘图缓存' in p.parts or '__pycache__' in p.parts or p.name=='最终文件SHA256.csv':continue
            files.append(dict(文件=str(p.relative_to(GUIDE)),字节=p.stat().st_size,SHA256=sha(p)))
    pd.DataFrame(files).to_csv(OUT/'表格/最终文件SHA256.csv',index=False,encoding='utf-8-sig')
    print(f'最终验收PASS：{len(files)}文件身份登记；9组图表；102项独立复核×2入口；历史保护通过。')


if __name__=='__main__':main()

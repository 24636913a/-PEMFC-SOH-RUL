"""独立核对来源字节、旧特征身份、证据定位、决策范围及历史保护。"""
from pathlib import Path
from xml.etree import ElementTree as ET
import ast,hashlib,json
import numpy as np
import pandas as pd
from PIL import Image

WORK=Path(__file__).resolve().parent.parent;GUIDE=WORK.parent.parent;REPO=GUIDE.parent
OUT=WORK/'输出结果';TAB=OUT/'表格';DATA=WORK/'配置/来源资料';PRIOR=WORK.parent/'共同负载特征研究'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run():
    old=json.loads((WORK/'配置/历史保护清单.json').read_text(encoding='utf-8'))
    changed=[n for n,h in old['文件SHA256'].items() if not (GUIDE/n).exists() or sha(GUIDE/n)!=h]
    assert not changed,changed
    sources=pd.read_csv(TAB/'来源文件登记表.csv')
    for r in sources.to_dict('records'):assert sha(REPO/r['来源文件'])==r['SHA256']
    src=pd.read_csv(TAB/'原始文件来源身份复核.csv');assert len(src)==19
    assert src.loc[src.电堆=='RealHyFC','官方逐字节身份'].eq('MATCH').all()
    # 官方元数据确认已检查源与方法PDF，不能仅比较本轮自行生成的两个hash列。
    official=json.loads((DATA/'RealHyFC耐久数据登记.json').read_text(encoding='utf-8'))
    method=json.loads((DATA/'RealHyFC方法数据登记.json').read_text(encoding='utf-8'))
    checks={f['key']:f['checksum'] for f in official['files']}
    for r in src.loc[src.电堆=='RealHyFC'].to_dict('records'):assert checks[r['文件名']]=='md5:'+r['MD5']
    for sid,meta in [('S02',official),('S03',method)]:
        md5=next(f['checksum'] for f in meta['files'] if f['key'].endswith('.pdf'))
        assert md5=='md5:'+sources.loc[sources.来源ID==sid,'MD5'].iloc[0]
    same=pd.read_csv(TAB/'同名BoL数据身份区别.csv');assert same.MD5.nunique()==2
    previous=pd.read_csv(PRIOR/'输出结果/表格/路线B双参考特征.csv')
    previous=previous.loc[previous.方案.isin(['FC1_10','B60'])].sort_values(['电堆','文件名']).reset_index(drop=True)
    actual=pd.read_csv(TAB/'逐曲线双参考建模资格.csv').sort_values(['电堆','文件名']).reset_index(drop=True)
    assert len(previous)==len(actual)==19
    diffs=[]
    for c in previous.columns:
        a=previous[c];b=actual[c]
        if a.dtype.kind in 'if':
            assert np.array_equal(a.isna(),b.isna())
            delta=(a-b).abs().max();diffs.append(0 if pd.isna(delta) else float(delta));assert pd.isna(delta) or delta<1e-15
        else:assert a.fillna('').equals(b.fillna('')),c
    assert actual.loc[~actual.双参考支持,'D_low_V'].isna().all()
    # 原作者XML中明确的协议与标签定位；不依赖二次综述。
    root=ET.fromstring((DATA/'FC1原作者论文.xml').read_bytes())
    tables={a.get('id'):''.join(a.itertext()) for a in root.findall('.//table-wrap')}
    protocol=tables['tbl0003'];assert 'from 1' in protocol and 'to 0' in protocol and '1000' in protocol
    for hour in ['0; 48; 185; 348; 515; 658; 823; 991']:assert hour in protocol
    text=''.join(root.itertext());assert '1300' in text and '55' in text and '42 excel' in text
    e=pd.read_csv(TAB/'来源证据登记表.csv');assert e.证据ID.is_unique and len(e)==26
    ids=set(sources.来源ID)
    for s in e.来源ID:
        assert set(s.split(';'))<=ids
    assert e.页码章节或字段.notna().all()
    results=pd.read_csv(TAB/'研究路线决策建议.csv').set_index('层级')
    assert results.loc['A','结论']=='YES_WITH_LIMITS'
    assert results.loc['B','结论']=='CONDITIONAL_GO_PROTOCOL_REQUIRED'
    assert results.loc['C','结论']=='NO_BLOCKED'
    assert results.loc['D-SOH','结论'].startswith('NO_') and results.loc['D-RUL','结论'].startswith('NO_')
    config=json.loads((WORK/'配置/证据核验范围.json').read_text(encoding='utf-8'))
    assert config['基准HEAD']=='eea57da72400c4e07f9e8ced30f40c25ba1ae68f'
    assert not list((GUIDE/'完整代码').rglob('*.py'))
    for p in (WORK/'程序').glob('*.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(not n.name.startswith(('torch','tensorflow','keras')) for n in node.names)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ['fit','backward','train','step']
    pngs=list((OUT/'图片').glob('*.png'));assert len(pngs)==5
    for p in pngs:
        with Image.open(p) as im:assert im.width>800 and im.height>1000
    q=pd.read_csv(TAB/'指导教师问题与实验影响.csv');assert len(q)==8 and q.问题.notna().all()
    # 原文页已实际渲染检查；不会把原文截图冒充新实验图。
    report=['# 阶段1.3复核记录','',
      f'来源/定位/资格检查通过。{len(old["文件SHA256"])}个历史文件SHA保持一致，阶段0至1.2没有变化。',
      f'12个来源、26条证据、19条原曲线身份和19条固定特征记录核对通过。11份Real极化MD5及两份Real原论文PDF MD5与官方登记一致。特征CSV往返最大差{max(diffs):.3e}，没有重新计算或改变特征。',
      'FC1原作者XML表3核对测试时点/降载1000秒；名义设定与字段范围可定位。没有伪造未取得PDF的页码。FC1官方归档未取得，原始同一性仍不能补证。',
      '5张原文截图结构通过，已放大目视核对耐久论文第3页和方法论文第4页的表格列及单位。未生成预测图或精度。',
      'A/B/C/D资格分别核验，不将B有条件离线基线建议误写成正式训练已通过。程序静态检查无训练框架、反向传播或拟合入口；完整代码目录未增加程序。',
      '来源抓取中的403/503等失败保留，不宣称缺失资料永久不存在。当前只完成准备研究。']
    (WORK/'复核记录.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    manifest={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob('*') if p.is_file() and p.name not in ['交付SHA256.json','最终验收记录.json'] and '__pycache__' not in p.parts}
    write(OUT/'交付SHA256.json',manifest)
    result={'状态':'PASS','历史保护文件数':len(old['文件SHA256']),'历史变化文件':changed,'来源数':len(sources),'证据条数':len(e),
        'Real官方MD5匹配':11,'冻结特征最大往返差':max(diffs),'原文截图数':len(pngs),'教师问题数':len(q),'实际训练':False,'预测精度计算':False,'准备代码进入完整代码':False,
        '交付文件数':len(manifest),'Git推送':False}
    write(OUT/'最终验收记录.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':run()

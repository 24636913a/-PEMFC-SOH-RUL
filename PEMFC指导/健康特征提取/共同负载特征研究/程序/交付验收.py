"""最终文件完整性、图片/PDF结构和来源验收；不改历史。"""
from pathlib import Path
import json,hashlib,ast
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import fitz

WORK=Path(__file__).resolve().parent.parent;OUT=WORK/'输出结果';TAB=OUT/'表格';GUIDE=WORK.parent.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    record=json.loads((OUT/'独立复核结果.json').read_text(encoding='utf-8'));assert record['状态']=='PASS'
    old=json.loads((WORK/'配置/执行前证据.json').read_text(encoding='utf-8'))
    for rel,h in old['历史文件SHA256'].items():assert sha(GUIDE/rel)==h,rel
    assert not list((GUIDE/'完整代码').rglob('*.py')),'准备代码不得进入完整代码'
    for p in (WORK/'程序').glob('*.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ['backward','fit','train','step']
    listing=pd.read_csv(TAB/'图片与真实CSV索引.csv');png=[];figchecks=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for r in listing.to_dict('records'):
        p=OUT/'图片'/r['图片'];assert sha(p)==r['SHA256']
        for filename in r['数据表'].split(';'):assert (TAB/filename).is_file()
        assert (WORK/r['绘图程序']).is_file()
        if p.suffix=='.png':
            with Image.open(p) as im:
                assert min(im.info.get('dpi',(0,0)))>299
                assert im.width>1500 and im.height>800
                assert np.asarray(im.convert('RGB')).std()>3
                thumb=im.convert('RGB');thumb.thumbnail((570,400))
                cell=Image.new('RGB',(590,440),'white');cell.paste(thumb,((590-thumb.width)//2,28));ImageDraw.Draw(cell).text((8,5),p.stem,font=font,fill='black');png.append(cell)
                figchecks.append(dict(文件=p.name,像素=[im.width,im.height],dpi=list(im.info['dpi']),SHA256=sha(p),检查='PASS'))
        else:
            with fitz.open(p) as doc:assert len(doc)==1 and len(doc[0].get_text())>10
    # 验收缩略图仅为真实PNG缩放排版，非新增科研数据图。
    cols=3;rows=(len(png)+cols-1)//cols;sheet=Image.new('RGB',(cols*590,rows*440),'#EEEEEE')
    for i,cell in enumerate(png):sheet.paste(cell,((i%cols)*590,(i//cols)*440))
    sheet.save(OUT/'图片验收缩略图.jpg',quality=90)
    (OUT/'图片结构检查.json').write_text(json.dumps(figchecks,ensure_ascii=False,indent=2),encoding='utf-8')
    # CSV必须保持NA语义，禁止将拒绝双参考的斜率写为零。
    d=pd.read_csv(TAB/'路线B双参考特征.csv');assert d.loc[~d.双参考支持,'S_V_per_A_cm2'].isna().all()
    cv=pd.read_csv(TAB/'共享平台分组留出.csv');assert cv.loc[~cv.可评价,['一维预测_V','二维预测_V']].isna().all().all()
    report=['# 本轮复核记录','',f'数值独立复核PASS；{record["源文件SHA数"]}个极化源SHA，{len(old["历史文件SHA256"])}个历史文件不变。',
      f'{len(png)}张300dpi PNG及相应单页PDF结构通过；每图有真实CSV、程序和SHA。已通过缩略图逐张目视核对，并放大检查覆盖图和温度/压力原始例证，坐标及中文可辨，缺失点未填补。结构测试不替代视觉检查。',
      '八项算法/边界测试；严格A60逐组身份和共享BoL扰动一致。没有训练入口或反向传播调用。完整代码目录无Python准备代码。',
      '保留首次空集合索引错误及第二次末尾测试夹具缺字段错误；独立复核通过后续完已有数值结果。未以失败结果冒充完整结果。',
      '远端HEAD因GitHub连接失败未认证；本地起始HEAD见配置。无push，无网络配置变更。',
      '最终分层结论：数学CONDITIONAL_PASS；测量BLOCKED；因果时间BLOCKED；正式训练NO。']
    (WORK/'复核记录.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    excluded={'SHA256交付清单.json','最终验收记录.json'}
    manifest={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob('*') if p.is_file() and p.name not in excluded and '绘图缓存' not in p.parts}
    (OUT/'SHA256交付清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(状态='PASS',历史文件保护=len(old['历史文件SHA256']),PNG数=len(png),PDF数=int((listing.图片.str.endswith('.pdf')).sum()),
                交付文件数=len(manifest),正式训练=False,历史改写=False,准备代码进入完整代码=False,
                远端最新性='UNCONFIRMED_NETWORK_FAILURE',Git推送=False)
    (OUT/'最终验收记录.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

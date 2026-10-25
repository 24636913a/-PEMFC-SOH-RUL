"""读取论文、字段和既有表，形成可定位证据；不提取新特征或预测。"""
from pathlib import Path
from xml.etree import ElementTree as ET
from datetime import datetime,timezone
import hashlib,json,sys
import fitz
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent.parent;GUIDE=WORK.parent.parent;REPO=GUIDE.parent
DATA=WORK/'配置/来源资料';OUT=WORK/'输出结果';TAB=OUT/'表格'
PRIOR=WORK.parent/'共同负载特征研究'

def hashes(path):
    s=hashlib.sha256();m=hashlib.md5()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):s.update(chunk);m.update(chunk)
    return s.hexdigest(),m.hexdigest()

def write_csv(name,data):pd.DataFrame(data).to_csv(TAB/name,index=False,encoding='utf-8-sig')

def write_json(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run():
    source=[];claims=[];pages={}
    urls={
      'S01':('FC1原作者数据论文','10.1016/j.dib.2022.108601',DATA/'FC1原作者论文.xml','https://pmc.ncbi.nlm.nih.gov/articles/PMC9679439/','CC BY-NC-ND 4.0，原XML未改写'),
      'S02':('RealHyFC耐久实验原论文','10.1016/j.ijhydene.2026.156060',REPO/'参考论文/IJHE_2026_Fensterle_Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing.pdf','https://doi.org/10.1016/j.ijhydene.2026.156060','CC BY 4.0，使用本地原PDF'),
      'S03':('RealHyFC直接引用的方法原论文','10.1002/fuce.70057',DATA/'RealHyFC原方法论文.pdf','https://zenodo.org/records/19068126','CC BY 4.0，原PDF未改写'),
      'S04':('Greenlight厂商通用规格','',DATA/'G200厂商通用手册.pdf','https://www.greenlightinnovation.com/files/product-documents/5e4d8a71966b1/FCATS_Stack_Small-web.pdf','厂商公开手册，仅证据核对；不是该试验台校准证书'),
      'S05':('RealHyFC耐久原数据官方登记','10.5281/zenodo.20715542',DATA/'RealHyFC耐久数据登记.json','https://zenodo.org/records/20715542','官方元数据'),
      'S06':('RealHyFC方法原数据官方登记','10.5281/zenodo.19068126',DATA/'RealHyFC方法数据登记.json','https://zenodo.org/records/19068126','官方元数据；不等同同一个耐久堆'),
      'S07':('RealHyFC本地EIS说明','',REPO/'数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据/ReadMe_EIS_measurements.txt','','原作者数据随附说明，内容范围为EIS命名'),
      'S08':('阶段0实际字段与时钟核验','',GUIDE/'数据处理/输出结果/表格/运行数据与极化测试对应表.csv','','既有核验，不重跑全量审计'),
      'S09':('阶段1.2双参考冻结结果','',PRIOR/'输出结果/表格/路线B双参考特征.csv','','旧结果只读，公式不变'),
      'S10':('阶段0采样间隔检查','',GUIDE/'数据处理/输出结果/表格/老化运行数据完整性检查.csv','','旧结果只读'),
      'S11':('阶段1.1温度记录周期复核','',WORK.parent/'方法复核/输出结果/表格/全Nyquist温度频谱复核.csv','','只读已有统计，不重算频谱'),
      'S12':('阶段1.2严格支持与压力屏障','',PRIOR/'输出结果/表格/共同负载实际支持表.csv','','旧结果只读')}
    for sid,(name,doi,p,url,license_) in urls.items():
        s,m=hashes(p);source.append(dict(来源ID=sid,资料=name,DOI=doi,来源文件=str(p.relative_to(REPO)),URL=url,SHA256=s,MD5=m,字节=p.stat().st_size,性质=license_))
        if p.suffix=='.pdf':
            with fitz.open(p) as doc:
                pages[sid]=[pg.get_text() for pg in doc]
                # 原文按页保留，供定位；不是由摘要生成的内容。
                (OUT/(sid+'_原文分页文本.txt')).write_text('\n\n'.join(f'PDF第{i+1}页\n{t}' for i,t in enumerate(pages[sid])),encoding='utf-8')
    root=ET.fromstring((DATA/'FC1原作者论文.xml').read_bytes())
    doi=next(p.text for p in root.findall('.//article-id') if p.attrib.get('pub-id-type')=='doi')
    assert doi=='10.1016/j.dib.2022.108601'
    sections={s.findtext('title'):''.join(s.itertext()) for s in root.findall('./body/sec')}
    tables={t.attrib.get('id'):''.join(t.itertext()) for t in root.findall('.//table-wrap')}
    assert 'descending current ramp' in sections['Experimental Design, Materials and Methods']
    assert '1000' in tables['tbl0003'] and '991' in tables['tbl0003']
    assert '1300' in sections['Data Description'] and '55' in sections['Data Description']
    # PDF表格需人工视觉核对，本脚本只做相关页锚点/来源身份检查。
    assert 'Table 3' in pages['S02'][2] and 'G200' in pages['S02'][1]
    assert '8 h' in pages['S03'][2] and '50' in pages['S03'][3]
    write_csv('来源文件登记表.csv',source)
    def add(code,topic,status,sid,loc,fact,limits,impact):
        claims.append(dict(证据ID=code,问题=topic,证据判断=status,来源ID=sid,页码章节或字段=loc,
            核验事实=fact,不能推出=limits,影响的后续实验=impact))
    add('F01','FC1标称时刻依据','已证实（文献协议）','S01','Data Description (1)(c); Table 3',
        '原作者定义Txxx为老化时刻，并列0/48/185/348/515/658/823/991 h。','不是本地每条曲线的实际起止日历时间，也不是结果发布时间。','离线老化时点解释有原作者依据；严格在线截止仍受限')
    add('F02','FC1真实时钟','缺少证据','S08','FC1时钟资格、实际测试开始/结束、结果可用时间',
        '既有本地极化表无Time/DateTime和诊断发布字段；原始阶段0 NOMINAL_ONLY保持。','不能将文件名、Excel创建时间、下载时间当测量结束或发布。','严格时间因果少样本协议')
    add('F03','FC1极化扫描','已证实（文献协议）','S01','Experimental Design, Materials and Methods (2)(b); Table 3',
        '极化采用1至0 A/cm²的约1000秒连续降载；最低流量对应0.20 A/cm²。','不能视为每个电流密度5分钟稳态停留，也不能从密集点数恢复采样时钟。','统一定义可保留，跨堆测量协议不能宣称相同')
    add('F04','FC1稳定期含义','已证实（适用范围）','S01','Experimental Design, Materials and Methods (2)(b)',
        '15分钟稳定期出现在极化后的0.70/0.45/0.20电流密度EIS测量序列。','不能移植成极化曲线上每点15分钟稳定证据。','拒绝用EIS协议替代极化稳态认证')
    add('F05','FC1温度设定','已证实（名义实验设定）','S01','Data Description (1)(a)',
        '氢气/空气入口设定28/42°C；冷却液出口作为堆温度控制量，维持55°C。','不证明每条极化同时实测都相同；55°C出口不等于Real70°C入口。','堆内比较有协议依据；跨堆仍有热条件差异')
    add('F06','FC1压力与湿度设定','已证实（名义实验设定）','S01','Data Description (1)(a); Table 1',
        '入口压力设定1300 mbar绝压；氢气/空气RH为0/50%，化学计量比2/4。','不是入口压力允许波动，也不能直接与Real出口kPag数值比较。','同条件解释和跨堆观测差异登记')
    add('F07','FC1 Excel来源形式','部分证实','S01','Data Description (1); 本地FC1目录42文件',
        '原作者明确42份Excel发布形式；本地FC1目录也为42文件。文中举例文件名同时使用.csv。','文件数/名称吻合不是本地数据与作者归档逐字节一致证明。','不再以没有CSV单独否定Excel形式，但仍需作者归档校验')
    add('F08','FC1采样声明与本地间隔','已证实存在差异，原因未知','S01;S10','Specifications Table 1Hz; 既有正时间差中位秒',
        '原文声明采集频率1Hz；本地3份老化表既有时间差中位约30.202–30.203秒。','不能据此擅自重采样、指认文件损坏、推断极化采样率或静默恢复缺失点。','后续运行数据融合前必须核对导出/抽样链')
    add('F09','FC1极化同步工况','缺少证据（且原文说明未保存）','S01;S08','Data Description (1)(c); 本地极化表列名',
        '作者说明极化文件只保存5单片/堆电压、电流、密度；本地列名与此相容。','老化运行文件含温压不代表可在未知诊断时刻准确配对。','测量可比性仍不能从邻近运行值补齐')
    add('F10','FC1独立重复测量','缺少证据','S01;S09','Table 3; 阶段1.2各日期记录',
        '8条曲线对应不同老化阶段，未找到同一健康状态同条件独立复测。','每条曲线大量点、分块和不同老化日期不是独立重复。','无法认证跨日期重测误差')
    add('R01','Real冷却液测量含义','已证实（字段和方法）','S02;S03;S08','S02 PDF2–3页; S03 PDF3–4页; T.Si.CL/T.So.CL',
        'T.Si.CL/T.So.CL为冷却液入口/出口记录；名义入口70°C，流量按高负载10K温差选择。','入口记录不等于膜或堆内局部温度；温差目标不是允许波动。','保留真实热条件差异，不做温度修正')
    add('R02','Real温度传感器准确度','缺少证据','S02;S03;S04','两论文实验方法及误差段; G200手册2页',
        '核阅资料未给出该试验台T.Si.CL传感器型号、校准与准确度。','CDDM校准、EIS误差和通用电压测量±1mV不能代替冷却液温度准确度。','无法量化温度系统误差对D的传播')
    add('R03','Real允许温控波动','缺少证据','S02;S03','S02 Table 1–3; S03 Table 1–2',
        '名义设定和敏感性试验温度档位有依据，末60秒允许峰峰波动未给出。','温度敏感性55–75°C范围不是平台允许摆动；未认证取消1°C限制。','本轮不得搜索或替换温度阈值')
    add('R04','Real入口出口压力','已证实（字段及名义设定）','S02;S03;S08','S02 PDF3页Table 1; S03 PDF4页Table 1; p.Si.A/p.So.A/p.Si.C/p.So.C',
        '气体名义压力160/150 kPaG对应阳极/阴极出口；Si/So对应入口/出口。','不能把入口与出口差当仪器错误，也不能与FC1绝压直接混用。','压力筛选与跨堆对比必须保留位置及压力基准')
    add('R05','Real压力允许波动','缺少证据','S02;S03;S04','实验方法/压力敏感性档位/厂商通用规格',
        '未找到该数据测量条件允许的入口压力峰峰公差。','压力敏感性变化范围、试验台量程不是3kPa公差认证；不放宽原规则。','2300h启停后低负载拒绝保持')
    add('R06','Real方法中的50kPaG','已证实（位置限定）','S03','PDF4页Table 1 Coolant行 Inlet pressure列',
        '50 kPaG属于冷却液入口名义压力。','不是阳极/阴极入口压力设定，更不是压力允许波动。','避免错列、跨介质引用参数')
    add('R07','Real极化平台协议','已证实（文献协议）','S02;S03','S02 PDF2–3页Table 3; S03 PDF3页2.2.1',
        '先降后升，16负载点；每点5分钟或低负载2分钟，极化前1A/cm²预调理30分钟。','停留时长不是全部日期实测温压都等价的证明。','与FC1连续扫描区别明确，不能混成同一个稳态协议')
    add('R08','Real恢复与休息协议','已证实（方法论文范围）','S03','PDF3页2.2 Characterization Methods',
        '方法论文称表征序列之间至少休息8小时；break-in后至少休息4小时。','不能给每个耐久老化文件补出未记录的休息时间；作者意图排除恢复不等于实测证明完全消除。','启停后、方向与预调理史仍需单列')
    add('R09','Real方法与耐久硬件身份','部分证实，不能互换','S02;S03;S05;S06','S02 PDF2页2.1; S03 PDF2页2.1; 两官方数据集条目',
        '方法论文研究10片堆；耐久论文说明BoL完整EIS敏感性另用8片新堆。两个数据集同名BoL极化文件MD5不同。','相同项目、MEA批次和文件名不证明同一试验堆；不得借敏感性数据校正本任务D。','跨试验校准、传感器与硬件身份限制')
    add('R10','Real四秒温度记录周期','原始观测已证实，原因未知','S11','最强全带周期_s; 原69平台队列',
        '既有记录呈约4.07秒主要周期，未重算或调整。','没有控制器、传感器和同步链证据，不能判定是仪器故障或真实内部热波动。','数值分辨率/周期不得当传感器精度或物理原因')
    add('T01','Real时钟关系','部分证实','S08','实际测试开始/结束; 累计小时下界/上界; 原OpHrs2字段',
        '有实际DateTime及已核验运行小时约束；部分早期曲线只可定区间。','不能按标称300/600代替精确运行小时。','保留区间，不据文件名提前使用诊断')
    add('T02','严格300/600小时可用性','已证实边界，发布时间仍未知','S08','300h/600h行的运行小时区间及物理结束资格',
        '300h曲线为312.15411–327.81354h；600h为566.16980–615.66968h。','300h不能进入严格300h截止；600h不能默认严格600h已完成/可得。','可另行按诊断事件编号离线研究，但不能冒称严格小时在线适配')
    add('T03','诊断结果真实可用时间','缺少证据','S01;S02;S03;S08','论文采集描述; 本地字段; 时钟资格',
        '未找到逐诊断处理完成、质检和向预测者开放的时间记录。','测量结束、表格保存/下载时间以及论文公开日期不能默认为真实在线可用时刻。','严格因果迁移BLOCKED；离线回放只能显式假设')
    add('T04','数据公开日期','已证实（仅数据发布）','S05;S06','metadata.publication_date及updated',
        '耐久数据登记公开日期2026-06-29，方法数据登记为2026-06-03。','这些日期不是2024/2025每次诊断的在线处理/结果发布时间。','分开记录公共来源发布与实验信息可用性')
    add('H01','双参考数学资格','沿用阶段1.2，不重新计算','S09;S12','方案FC1_10/B60; 双参考支持',
        '使用固定0.45与0.90、各自BoL；17条主曲线数学可计算，2300h启停后B60双参考拒绝。','不能用数学支持替代相同测量协议、不可逆退化或独立重复性证据。','保留原数学CONDITIONAL_PASS和缺失记录')
    add('H02','正式SOH和RUL终点','缺少本轮可核实证据','S01;S02;S08','FC1表征/记录截止; Real耐久过程与时钟表',
        '本轮未核实与既定参考健康定义一致、可验证的物理EOL或明确删失事件表。','991/1154/2300标签不是自动EOL；论文说数据可用于RUL不等于每条记录有真实EOL。','仅可提出条件阈值寿命研究；真实RUL评价继续受限')
    write_csv('来源证据登记表.csv',claims)
    # 逐源文件身份和官方MD5；仅字节哈希，不重新读数值、不改任何原始数据。
    official=json.loads((DATA/'RealHyFC耐久数据登记.json').read_text(encoding='utf-8'))
    method=json.loads((DATA/'RealHyFC方法数据登记.json').read_text(encoding='utf-8'))
    officialmap={p['key']:p for p in official['files']};methodmap={p['key']:p for p in method['files']}
    src=pd.read_csv(PRIOR/'输出结果/表格/源文件身份复核.csv');sourcecheck=[];headers=[]
    for r in src.to_dict('records'):
        path=REPO/r['原始路径'];s,m=hashes(path);assert s==r['SHA256']
        expected=officialmap.get(r['文件名'],{}).get('checksum','')
        if r['电堆']=='RealHyFC':assert expected=='md5:'+m,(r['文件名'],m,expected)
        sourcecheck.append(dict(电堆=r['电堆'],文件名=r['文件名'],原始路径=r['原始路径'],SHA256=s,MD5=m,与阶段1_2一致=True,
            官方MD5=expected,官方逐字节身份='MATCH' if expected=='md5:'+m else 'NOT_AVAILABLE_FC1',
            文件形式依据='S01说明Excel，但本地尚无官方checksum' if r['电堆']=='FC1' else 'S05官方元数据'))
        frame=pd.read_excel(path,nrows=0,engine='openpyxl') if r['电堆']=='FC1' else pd.read_csv(path,sep=';',nrows=0)
        relevant=list(frame.columns) if r['电堆']=='FC1' else [f for f in frame.columns if f.startswith(('T.S','p.S')) or f in ['DateTime','OpHrs2','OpHrs','I.S.act [A]','j.S.act [A/cm2]']]
        headers.append(dict(电堆=r['电堆'],文件名=r['文件名'],实际字段=';'.join(relevant),完整字段数=len(frame.columns),DateTime字段存在='DateTime' in frame.columns,
            OpHrs2字段存在=any('OpHrs2' in c for c in frame.columns),发布时间字段存在=any('release' in c.lower() or 'publish' in c.lower() for c in frame.columns)))
    write_csv('原始文件来源身份复核.csv',sourcecheck);write_csv('极化文件实际字段核验.csv',headers)
    metadata=[]
    for name in ['Polcurve_Nominal_RHFC_BoL.csv']:
        for label,mp in [('耐久试验',officialmap),('方法试验',methodmap)]:metadata.append(dict(文件名=name,来源=label,MD5=mp[name]['checksum'],字节=mp[name]['size'],能否仅凭同名互换='NO'))
    write_csv('同名BoL数据身份区别.csv',metadata)
    # 只读历史特征、时钟与协议；不重新拟合、不新增未来预测误差。
    t=pd.read_csv(PRIOR/'输出结果/表格/实际时间资格.csv');d=pd.read_csv(PRIOR/'输出结果/表格/路线B双参考特征.csv')
    d=d.loc[d.方案.isin(['FC1_10','B60'])].merge(t[['文件名','时钟资格','累计小时下界','累计小时上界','结果发布时间']],on='文件名')
    d['数学资格']=np.where(d.双参考支持,'CONDITIONAL_PASS_MATHEMATICAL_ONLY','BLOCKED_SUPPORT')
    d['日期比较资格']='OFFLINE_PROTOCOL_QUALIFIED_ONLY_ACTUAL_CONDITIONS_UNVERIFIED'
    d['跨堆测量等价']='NOT_ESTABLISHED';d['同条件独立重测']='NOT_IDENTIFIED';d['严格因果可用']='BLOCKED'
    write_csv('逐曲线双参考建模资格.csv',d)
    interval=pd.read_csv(GUIDE/'数据处理/输出结果/表格/老化运行数据完整性检查.csv')
    write_csv('FC1采样声明与本地记录差异.csv',interval.loc[interval.电堆=='FC1',['文件名','行数','起始运行小时','末尾运行小时','正时间差中位秒']].assign(原论文采集声明='1Hz，S01 Specifications Table',差异原因='UNKNOWN；不修改原始时间'))
    # 证据页原样渲染，避免文本抽取错列；不生成预测曲线。
    renders=[('S02',[2,3]),('S03',[3,4]),('S04',[2])]
    for sid,nums in renders:
        with fitz.open(urls[sid][2]) as doc:
            for num in nums:doc[num-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/'图片'/f'{sid}_原文第{num}页.png')
    # 冻结资料目录的文件列表只说明本轮检索范围，不以未发现证明永久不存在。
    coverage=[dict(来源ID='S01',核阅范围='完整XML正文、表1–3、数据访问和许可，未取得PDF',缺失检索='未见极化逐事件日历、传感器准确度、结果释放日志'),
              dict(来源ID='S02',核阅范围='15页全文，重点PDF2–4页方法、表1–3与硬件说明',缺失检索='未见冷却液传感器准确度/温控公差/逐诊断发布时间'),
              dict(来源ID='S03',核阅范围='15页全文，重点PDF2–4页方法及表格',缺失检索='补充预调理/休息协议，仍未见该测温链准确度及公差'),
              dict(来源ID='S04',核阅范围='2页完整厂商通用规格',缺失检索='不是本试验台配置/校准证书；未给冷却液温控公差'),
              dict(来源ID='S05/S06',核阅范围='官方JSON完整metadata、files与checksums；不下载大型原始数据',缺失检索='元数据未列逐事件释放记录；两个BoL不相同')]
    for row in coverage:
        if row['来源ID'] in ['S02','S03']:row['核阅范围']='15页全文文本提取与关键词检索；重点完整阅读/核对PDF2–4页实验方法和表格'
    write_csv('核阅范围与未找到证据.csv',coverage)
    keywords=['temperature sensor','thermocouple','pt100','thermistor','accuracy','uncertainty','tolerance','stability','rest period','dwell','release','timestamp','supplement']
    hits=[]
    for sid,pp in pages.items():
        for word in keywords:
            found=[i+1 for i,t in enumerate(pp) if word in t.lower()]
            hits.append(dict(来源ID=sid,关键词=word,出现页=';'.join(map(str,found)),命中页数=len(found),说明='全文文本检索辅助；未匹配不证明资料永久不存在，命中须结合测量对象解释'))
    write_csv('全文关键词位置复核.csv',hits)
    write_json(OUT/'证据核验运行记录.json',dict(UTC=datetime.now(timezone.utc).isoformat(),来源数=len(source),证据条数=len(claims),
       旧特征记录数=len(d),源身份复核数=len(sourcecheck),Real官方MD5匹配数=sum(r['官方逐字节身份']=='MATCH' for r in sourcecheck),
       训练=False,预测精度计算=False,公式更改=False,温压阈值搜索=False,原始运行数据全量重读=False))
    print('来源、字段、固定特征资格核验完成：',len(claims),'条证据；Real官方身份匹配11份。',flush=True)

if __name__=='__main__':run()

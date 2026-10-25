"""只获取论文/官方元数据；不下载原始运行数据，不执行任何模型。"""
from pathlib import Path
import concurrent.futures,datetime,hashlib,json,subprocess
import requests

WORK=Path(__file__).resolve().parent.parent;GUIDE=WORK.parent.parent;REPO=GUIDE.parent
DEST=WORK/'配置/来源资料';OUT=WORK/'输出结果'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def snapshot():
    p=WORK/'配置/历史保护清单.json'
    if p.exists():return
    old={str(q.relative_to(GUIDE)):sha(q) for q in GUIDE.rglob('*') if q.is_file() and not q.is_relative_to(WORK) and '__pycache__' not in q.parts and '绘图缓存' not in q.parts and not q.name.startswith('~$')}
    data={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'文件SHA256':old}
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def fetch(name,url):
    path=DEST/name
    if path.exists():return {'文件':name,'URL':url,'状态':'已有来源只读复用','SHA256':sha(path)}
    try:
        response=requests.get(url,timeout=(8,25),headers={'User-Agent':'PEMFC-academic-evidence-audit/1.0'},stream=True)
        response.raise_for_status();parts=[];size=0
        for chunk in response.iter_content(65536):
            parts.append(chunk);size+=len(chunk)
            if size>12*1024*1024:raise RuntimeError('超过文献/元数据大小上限，停止下载')
        path.write_bytes(b''.join(parts))
        return {'文件':name,'URL':url,'实际URL':response.url,'状态':'取得','字节':size,'SHA256':sha(path),'Content-Type':response.headers.get('Content-Type'),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except Exception as e:return {'文件':name,'URL':url,'状态':'失败，不能据此认定资料不存在','错误':str(e)}

def main():
    snapshot()
    sources=[('FC1原作者论文.xml','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9679439/fullTextXML'),
             ('FC1原作者论文网页.html','https://europepmc.org/articles/PMC9679439'),
             ('FC1官方门户.html','https://search-data.ubfc.fr/FR-18008901306731-2021-07-19_IEEE-PHM-Data-Challenge-2014.html'),
             ('RealHyFC耐久数据登记.json','https://zenodo.org/api/records/20715542'),
             ('RealHyFC方法数据登记.json','https://zenodo.org/api/records/19068126')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda pair:fetch(*pair),sources))
    (OUT/'来源获取记录.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in results:print(r['文件'],r['状态'],r.get('字节',''),flush=True)

if __name__=='__main__':main()

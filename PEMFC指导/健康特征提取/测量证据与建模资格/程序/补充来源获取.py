"""有限补充原作者方法文献和厂商规格，仅用于证据审查。"""
from pathlib import Path
import concurrent.futures,json
from 来源获取 import fetch,WORK,DEST,OUT

def main():
    meta=json.loads((DEST/'RealHyFC方法数据登记.json').read_text(encoding='utf-8'))
    method=next(f['links']['self'] for f in meta['files'] if f['key'].endswith('.pdf'))
    urls=[('FC1原作者论文.pdf','https://europepmc.org/articles/PMC9679439?pdf=render'),
          ('RealHyFC原方法论文.pdf',method),
          ('G200厂商通用手册.pdf','https://www.greenlightinnovation.com/files/product-documents/5e4d8a71966b1/FCATS_Stack_Small-web.pdf')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:r=list(pool.map(lambda pair:fetch(*pair),urls))
    (OUT/'补充来源获取记录.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    for a in r:print(a['文件'],a['状态'],a.get('字节',''),flush=True)

if __name__=='__main__':main()

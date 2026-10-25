"""迁移后按预存SHA查找原审计代码证据，绝不将改动后的源码冒充旧版本。"""
from pathlib import Path
import hashlib,json

GUIDE=next(p for p in Path(__file__).resolve().parents if p.name=='PEMFC指导')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def resolve_evidence(relative,expected_sha):
    current=GUIDE/relative
    if current.is_file() and digest(current)==expected_sha:return current
    index=GUIDE/'健康特征提取/方法复核/配置/准备代码路径索引.json'
    rows=json.loads(index.read_text(encoding='utf-8'))
    direct=[r for r in rows if r['原路径']==relative and r['SHA256']==expected_sha]
    same=[r for r in rows if r['SHA256']==expected_sha and Path(r['原路径']).name==Path(relative).name]
    for row in direct+same:
        archived=(GUIDE/row['归档路径']).resolve()
        if archived.is_relative_to(GUIDE.resolve()) and archived.is_file() and digest(archived)==expected_sha:return archived
    raise FileNotFoundError('无法核对原证据（不修改旧SHA或跳过检查）：'+relative)

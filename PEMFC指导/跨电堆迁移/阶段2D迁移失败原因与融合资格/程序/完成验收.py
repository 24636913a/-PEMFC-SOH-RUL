"""完成记录与本轮文件SHA；不重新执行模型或原阶段审计。"""
import hashlib,json
from pathlib import Path
import pandas as pd
TASK=Path(__file__).resolve().parents[1];GUIDE=TASK.parents[1];REPO=GUIDE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    report=json.loads((TASK/'输出结果/独立检查通过.json').read_text(encoding='utf-8'));assert report['status']=='PASS'
    snapshot=pd.read_csv(TASK/'配置/历史文件SHA256.csv');assert all(sha(REPO/r.path)==r.sha256 for r in snapshot.itertuples())
    for name in ['负迁移原因分析报告.md','共同健康表征可比性评估.md','二维观测融合数据充分性报告.md','第三档下一阶段决策报告.md']:assert (TASK/name).is_file()
    cfg=json.loads((TASK/'配置/阶段2D审计协议.json').read_text(encoding='utf-8'));reg=json.loads((TASK/'配置/预登记SHA256.json').read_text(encoding='utf-8'));assert sha(TASK/'配置/阶段2D审计协议.json')==reg['protocol_sha256']
    fig=json.loads((TASK/'输出结果/表格/图片来源索引.json').read_text(encoding='utf-8'));assert len(fig)==7 and all(x['script_sha256']==sha(TASK/x['script']) for x in fig)
    entry=dict(status='STAGE_2D_AUDIT_COMPLETED',engineering='PASS',science='二维联合融合与网络仍不合格；一维全寿命仅数学门槛可达，不等于早期在线适配',head=cfg['head'],historical_files_unchanged=len(snapshot),formal_code_modified=False,retrained_models=0,diagnostic_files=19,figures_png_pdf=7,independent_checks=report['checks'],github_push=False)
    (TASK/'输出结果/阶段2D验收记录.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (TASK/'验收与历史保护记录.md').write_text(f'''# 阶段2D验收与历史保护

工程验收PASS。{report['checks']}项独立检查通过，全部19个真实诊断身份、配对二分搜索、两参考数值、冻结2538条预测传播代数、单位和标签均已核验。7组300dpi PNG/PDF有真实CSV与脚本SHA；人工检查配对上限、误差量级及双参考关系图，末端标签重叠已调整，仅改标签位置。

{len(snapshot)}个既有历史文件SHA保持一致，包括完整代码2A/2B/2C及此前研究记录。未重训任何模型，未调用旧运行入口；未生成实际未来时钟替代预测成绩；没有阈值或迁移权重搜索。历史拒绝及NA保留。所有新程序只在本任务目录，不更新完整代码。

全部主要结果由只读冻结CSV/JSON得到，真实源码读取/AST检查未执行模型。诊断配对独立检查不导入生成脚本；速率只做描述或与既有参数核对，不是重训迁移模型。完整寿命关系没有拟合运行—诊断未来预测映射。

科学结论与工程PASS分开：全寿命主序列FC1 7名义、Real 8完成假设配对，均未达到9对二维门槛；严格早期更少；发布时间和测量条件仍未认证。16单元网络暂停。全寿命一维数学门槛可达到，只支持受限离线研究，不自动允许严格在线迁移。

本轮未发生计算失败或替换历史成绩。只读资料核查有未获证据项，均按未知记录；不把未知说成不存在。Git沿用此前自动本地提交授权，本轮不推送远端。
''',encoding='utf-8')
    files=[dict(path=p.relative_to(TASK).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(TASK.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='本轮文件SHA256.csv']
    pd.DataFrame(files).to_csv(TASK/'输出结果/本轮文件SHA256.csv',index=False,encoding='utf-8-sig');print(json.dumps(entry,ensure_ascii=False))
if __name__=='__main__':main()

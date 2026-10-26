"""阶段2D：预登记只读审计范围与历史文件SHA。不运行预测模型。"""
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
TASK=Path(__file__).resolve().parents[1];GUIDE=TASK.parents[1];REPO=GUIDE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    config=TASK/'配置/阶段2D审计协议.json'
    if config.exists():raise FileExistsError('审计协议已冻结，不覆盖')
    for d in ['配置','输出结果/表格','输出结果/图片','输出结果/绘图数据']:(TASK/d).mkdir(parents=True,exist_ok=True)
    files=[]
    for p in sorted(GUIDE.rglob('*')):
        if p.is_file() and TASK not in p.parents and '__pycache__' not in p.parts and '绘图缓存' not in p.parts and not p.name.startswith('~$'):
            files.append(dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    pd.DataFrame(files).to_csv(TASK/'配置/历史文件SHA256.csv',index=False,encoding='utf-8-sig')
    cfg=dict(version='阶段2D只读诊断_v1',head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),utc=datetime.now(timezone.utc).isoformat(),
      scope='仅冻结观测/预测/参数和已有支持表；不调用2A/2B/2C运行入口或拟合函数；不解析原始860万运行CSV；不修改完整代码',
      points=[.45,.90],schemes={'FC1':'FC1_10','RealHyFC':'B60'},strict_comparator='Real A60保留原拒绝；FC1仍NOMINAL_ONLY',
      cutoffs_h=[300.,600.,629.2038],full_life='仅描述数据上限，不用全寿命拟合观测映射或预测器',
      pairing='完全沿用2B：诊断完成前最近合格观测且<=48日历小时；FC1只可名义小时同规则，不能认证真实时间；不跨扫描，不补BoL运行点',
      independence='主降载名义诊断序列单列；启停后诊断单列，不混入主序列；同时统计文件数、可配对测试数、唯一运行锚点数、独立电堆数。相同运行锚点不能按两个独立老化配对计数。即便唯一锚点仍来自同一寿命，非统计独立重复。',
      minimum=dict(one_dim_pairs=6,two_dim_pairs=9,standardized_condition_max=100,paired_diagnostic_transitions=6),
      descriptive='只计算固定既有时间边界内OLS描述斜率、相邻原始ID变化、相关性、秩/条件数；不据描述结果选择负载、窗口、阈值或迁移权重',
      time_audit='冻结状态与速率的传播差额q*(名义lead-真实lead)及可逆项差，仅事后审计；不输出真实未来时钟替代预测，不重排新模型成绩',
      attribution='源信息增量delta_v=-base*(q_source-q_target)*(已知过去传播+名义未来时长)；C为该增量的0.25倍；MSE差=2mean(e_target*delta)+mean(delta^2)，只是数值代数非物理因果分解',
      network='仅核对原方案8->16->2共178个网络权重、标量5->16->1共113个；不构建或训练网络',
      stop='形成路线B/A/C推荐及证据边界即停止，不新增预测实验',
      git='沿用用户此前自动Git授权仅本地提交本阶段；不推送；旧阶段和完整代码不动')
    config.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (TASK/'配置/预登记SHA256.json').write_text(json.dumps(dict(protocol_sha256=sha(config),utc=cfg['utc'],history_files=len(files)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (TASK/'.gitattributes').write_text('* -text\n',encoding='utf-8');(TASK/'.gitignore').write_text('__pycache__/\n',encoding='utf-8')
    print('FROZEN',len(files),sha(config))
if __name__=='__main__':main()

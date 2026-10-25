"""阶段1入口：重新读取极化源文件，继承阶段0安全读取能力，不读取全量运行CSV。"""
from pathlib import Path
import argparse, datetime, hashlib, json, sys, traceback, shutil
import numpy as np
import pandas as pd
import yaml


def find_guide():
    for p in Path(__file__).resolve().parents:
        if (p/'数据处理/数据核验报告.md').exists():return p
    raise RuntimeError('找不到PEMFC指导根目录')


GUIDE=find_guide()
if not (Path(__file__).parent/'核验基础.py').exists():
    sys.path.insert(0,str(GUIDE/'数据处理/程序'))
from 核验基础 import sha, save_json, table, read_polar
from 极化特征计算 import grid,variants,prepare,representatives,interpolate,solve,block_draws,unit_tests


def compute(cfg,out):
    tables=GUIDE/cfg['输入表目录'];root=GUIDE.parent
    registry=pd.read_csv(tables/'数据源文件登记表.csv')
    coverage=pd.read_csv(tables/'极化曲线覆盖范围.csv')
    time=pd.read_csv(tables/'运行数据与极化测试对应表.csv')
    # 必须完整解析阶段0扫描身份表，独立重建扫描后再次逐条核对。
    scans=pd.read_csv(tables/'极化扫描分段与参考点支持.csv')
    datasets={}; profiles={}; audits=[]; membership=[];rep_rows=[]; identity=[]
    for row in coverage.to_dict('records'):
        stack,name=row['电堆'],row['文件名'];source=root/row['原始路径']
        original=registry.loc[registry.原始路径==row['原始路径']].iloc[0]
        digest=sha(source)
        if digest != original.SHA256:raise RuntimeError('原始极化SHA发生变化：'+name)
        frame,_=read_polar(source,stack)
        if len(frame)!=row['原始行数']:raise RuntimeError('极化行数与阶段0不符：'+name)
        raw,selected=prepare(frame,stack,cfg)
        known=scans.loc[(scans.电堆==stack)&(scans.文件名==name)&scans.覆盖共同区间&(scans.方向=='降载')]
        expected=int(known.iloc[0]['分段']) if len(known) else None
        if selected!=expected:raise RuntimeError('扫描分段身份不一致：'+name)
        meta=time.loc[(time.电堆==stack)&(time.文件名==name)].iloc[0].to_dict()
        datasets[(stack,name)]=meta
        identity.append(dict(电堆=stack,文件名=name,原始路径=row['原始路径'],SHA256=digest,实际行数=len(frame),扫描分段=selected,与阶段0身份一致=True))
        print('读取与扫描核对:',stack,name,'segment=',selected,flush=True)
        for variant in variants(stack):
            reps,checks,members=representatives(raw,selected,stack,variant,cfg)
            profiles[(stack,name,variant)]=reps
            audits.extend(dict(电堆=stack,文件名=name,方案=variant,**x) for x in checks)
            membership.extend(dict(电堆=stack,文件名=name,方案=variant,**x) for x in members)
            rep_rows.extend(dict(电堆=stack,文件名=name,方案=variant,**{k:v for k,v in p.items() if k!='block'},
                                 分块代表值=';'.join(format(float(x),'.17g') for x in p['block'])) for p in reps)
    table(out/'表格/极化源身份复核.csv',identity)
    table(out/'表格/平台及分组筛选明细.csv',audits)
    table(out/'表格/代表点原始行成员.csv',membership)
    table(out/'表格/合格代表点及条件.csv',rep_rows)
    g=grid(cfg);fits=[];points=[];brackets=[];bootstrap=[]
    for (stack,name),meta in datasets.items():
        baseline=cfg['基准'][stack]
        for variant in variants(stack):
            reps=profiles[(stack,name,variant)]; bol=profiles[(stack,baseline,variant)]
            v,support=interpolate(reps,g,cfg['插值']['最大括距'])
            b,bsupport=interpolate(bol,g,cfg['插值']['最大括距'])
            brackets.extend(dict(电堆=stack,文件名=name,方案=variant,来源类型=typ,**r) for typ,s in [('本次',support),('BoL',bsupport)] for r in s)
            loss=b-v;mask=np.isfinite(loss);x=g[mask];y=loss[mask]
            base=dict(电堆=stack,文件名=name,方案=variant,标称老化小时=meta['标称老化小时'],启停后=meta['启停后'],
                      基准文件=baseline,扫描方向='降载',扫描依据='最早完整降载段',时钟资格=meta['时钟资格'],
                      对齐状态=meta['对齐状态'],实际测试开始=meta['实际测试开始'],实际测试结束=meta['实际测试结束'],
                      累计小时下界=meta['累计小时下界'],累计小时上界=meta['累计小时上界'],
                      前300h物理结束资格=meta['前300h物理结束资格'],前600h物理结束资格=meta['前600h物理结束资格'],
                      结果可用时间=meta['结果可用时间'],在线适配资格='UNCONFIRMED_RELEASE_TIME',
                      稳态资格='NO_CLOCK_SCAN_DESCRIPTOR' if stack=='FC1' else 'OBSERVED_STABILITY_SCREEN',
                      原始路径=meta['原始路径'],完整共同网格受支持=bool(mask.all()),有效网格数=int(mask.sum()),
                      参考点支持=bool(interpolate(reps,np.array([.9]),cfg['插值']['最大括距'])[1][0]['支持'] and interpolate(bol,np.array([.9]),cfg['插值']['最大括距'])[1][0]['支持']))
            enough=len(x)>=cfg['拟合']['最少网格数'] and np.ptp(x)>=cfg['拟合']['最小电流跨度']-1e-12
            if not enough:
                base.update(计算状态='REJECTED_SUPPORT',拟合质量='NA',l_1=np.nan,l_2=np.nan)
                fits.append(base)
                points.extend(dict(电堆=stack,文件名=name,方案=variant,j=float(q),本次电压_V=v[i],BoL电压_V=b[i],损失_V=loss[i],纳入拟合=False,二维重构_V=np.nan,一维重构_V=np.nan,权重=np.nan) for i,q in enumerate(g))
                continue
            f=solve(x,y,cfg['共同网格']['j_ref'])
            base.update({k:value for k,value in f.items() if k not in ['pred','residual','weights']},计算状态='COMPUTED')
            # 相同BoL每次使用同一随机流，明确包含共享参考的不确定性，BoL自身不重复扰动。
            seed=cfg['稳定性']['扰动种子']+(0 if stack=='FC1' else 10000)
            db=block_draws(bol,g,cfg,np.random.default_rng(seed))
            dv=db if name==baseline else block_draws(reps,g,cfg,np.random.default_rng(seed+int(meta['标称老化小时'])+int(meta['启停后'])+1))
            X=np.column_stack([np.ones(len(x)),x/.9]);w=f['weights']
            projection=np.linalg.inv(X.T@(w[:,None]*X))@(X.T*w)
            draws=(db[:,mask]-dv[:,mask])@projection.T
            interval=np.quantile(draws,[.025,.975],axis=0)
            base.update(l1条件扰动下限_V=float(interval[0,0]),l1条件扰动上限_V=float(interval[1,0]),
                        l2条件扰动下限_V=float(interval[0,1]),l2条件扰动上限_V=float(interval[1,1]),
                        l2扰动符号保持=bool(interval[0,1]*interval[1,1]>0),
                        不确定性身份='条件分块扰动范围；不是独立测试置信区间')
            bootstrap.extend(dict(电堆=stack,文件名=name,方案=variant,扰动编号=k,l_1=float(bet[0]),l_2=float(bet[1])) for k,bet in enumerate(draws))
            k=0
            for i,q in enumerate(g):
                rec=dict(电堆=stack,文件名=name,方案=variant,j=float(q),本次电压_V=v[i],BoL电压_V=b[i],损失_V=loss[i],纳入拟合=bool(mask[i]),二维重构_V=np.nan,一维重构_V=np.nan,权重=np.nan)
                if mask[i]:rec.update(二维重构_V=float(f['pred'][k]),一维重构_V=f['一维c'],权重=float(w[k]));k+=1
                points.append(rec)
            # 只描述同负载与BoL的工况差异；不据此调整电压。
            for field,label in [('temp','温差_C'),('pa','阳极压差_kPa'),('pc','阴极压差_kPa')]:
                rv=interpolate(reps,g,cfg['插值']['最大括距'],[p[field] for p in reps])[0]
                rb=interpolate(bol,g,cfg['插值']['最大括距'],[p[field] for p in bol])[0]
                z=np.abs((rv-rb)[mask]);base['相对BoL最大'+label]=float(np.nanmax(z)) if np.isfinite(z).any() else np.nan
            fits.append(base)
    feature=pd.DataFrame(fits); curves=pd.DataFrame(points); comparisons=[]
    for (stack,name),meta in datasets.items():
        parent=feature.loc[(feature.电堆==stack)&(feature.文件名==name)&(feature.方案=='主规则')]
        if parent.iloc[0].计算状态!='COMPUTED':continue
        delta=[]
        for variant in variants(stack)[1:]:
            a=curves.loc[(curves.电堆==stack)&(curves.文件名==name)&(curves.方案=='主规则')]
            b=curves.loc[(curves.电堆==stack)&(curves.文件名==name)&(curves.方案==variant)]
            common=np.isfinite(a.损失_V.to_numpy())&np.isfinite(b.损失_V.to_numpy())
            x=g[common]
            if len(x)>=cfg['拟合']['最少网格数'] and np.ptp(x)>=cfg['拟合']['最小电流跨度']-1e-12:
                fa=solve(x,a.损失_V.to_numpy()[common],.9);fb=solve(x,b.损失_V.to_numpy()[common],.9)
                d=max(abs(fa['l_1']-fb['l_1']),abs(fa['l_2']-fb['l_2']))
                delta.append(d);comparisons.append(dict(电堆=stack,文件名=name,方案=variant,共同网格数=len(x),
                    主l1_V=fa['l_1'],变体l1_V=fb['l_1'],主l2_V=fa['l_2'],变体l2_V=fb['l_2'],最大参数变化_V=d,
                    最小j=float(x.min()),最大j=float(x.max())))
        ix=parent.index[0];r=feature.loc[ix];scale=max(abs(r.l_1),abs(r.l_2),cfg['稳定性']['相对尺度下限_V'])
        maxdelta=max(delta) if delta else np.nan
        stable=bool(len(delta)==len(variants(stack))-1 and maxdelta<=cfg['稳定性']['最大参数绝对改变量_V'] and maxdelta/scale<=cfg['稳定性']['最大参数相对改变量'])
        feature.loc[ix,'处理规则最大参数变化_V']=maxdelta;feature.loc[ix,'参数稳定性']=stable
        c=cfg['可靠性判据'];gain=r.负载留组一维RMSE_V-r.负载留组二维RMSE_V
        extra=bool(gain>=c['负载留组绝对改善至少_V'] and gain/max(r.负载留组一维RMSE_V,1e-15)>=c['负载留组相对改善至少'])
        identifiable=bool(r.条件数<=c['加权设计矩阵条件数上限'] and abs(r.参数相关系数)<=c['参数相关系数绝对值上限'])
        supported=bool(stable and extra and identifiable and r.l2扰动符号保持 and r.二维RMSE_V<=c['二维重构RMSE上限_V'])
        quality='BASELINE_IDENTITY' if name==cfg['基准'][stack] else 'CONDITIONAL_2D_INFORMATION' if supported else '2D_NOT_ROBUSTLY_SUPPORTED'
        feature.loc[ix,'拟合质量']=quality;feature.loc[ix,'负载留组额外信息']=extra;feature.loc[ix,'数值可辨识']=identifiable
    table(out/'表格/全部方案二维特征.csv',feature)
    for stack in ['FC1','RealHyFC']:
        table(out/f'表格/{stack}二维健康特征.csv',feature.loc[(feature.电堆==stack)&(feature.方案=='主规则')])
    table(out/'表格/共同网格损失及一二维重构.csv',curves)
    table(out/'表格/局部插值左右支持.csv',brackets)
    table(out/'表格/同支持区间处理敏感性.csv',comparisons)
    table(out/'表格/条件分块扰动参数.csv',bootstrap)
    return feature,curves,pd.DataFrame(comparisons)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--redraw-from',type=Path,help='只读已有计算CSV；新目录重绘与重判，不重算特征、不覆盖原运行')
    args=parser.parse_args()
    stage=GUIDE/'健康特征提取'
    localcfg=Path(__file__).parent/'配置/健康特征冻结规则.yaml'
    cfgpath=localcfg if localcfg.exists() else stage/'配置/健康特征冻结规则.yaml'
    cfg=yaml.safe_load(cfgpath.read_text(encoding='utf-8-sig'))
    frozen=json.loads((stage/'配置/计算前冻结记录.json').read_text(encoding='utf-8'))
    if sha(cfgpath)!=frozen['config_sha256']:raise RuntimeError('冻结配置被修改，停止')
    checks=unit_tests()
    if args.check_only:
        print('PASS',checks);return
    out=stage/'输出结果'
    if (out/'运行记录.json').exists():out=out/'历史核验'/datetime.datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    for d in ['表格','图片']: (out/d).mkdir(parents=True,exist_ok=True)
    started=datetime.datetime.now().astimezone().isoformat()
    try:
        if args.redraw_from:
            source=args.redraw_from.resolve()
            previous=json.loads((source/'运行记录.json').read_text(encoding='utf-8'))
            if previous['config_sha256']!=sha(cfgpath):raise RuntimeError('重绘来源配置不一致')
            log_variance=[]
            for item in json.loads((source/'结果SHA256.json').read_text(encoding='utf-8')):
                current=sha(source/item['file'])
                if current!=item['sha256']:
                    if item['file']=='执行日志.txt':
                        log_variance.append(dict(file=item['file'],recorded=item['sha256'],current=current,
                            reason='首轮外部Tee在主程序生成SHA后追加最终stdout；不涉及计算CSV、图像或配置'))
                    else:raise RuntimeError('重绘来源结果SHA不一致：'+item['file'])
            for p in (source/'表格').glob('*.csv'):shutil.copy2(p,out/'表格'/p.name)
            feature=pd.read_csv(out/'表格/全部方案二维特征.csv');curves=pd.read_csv(out/'表格/共同网格损失及一二维重构.csv')
            sensitivity=pd.read_csv(out/'表格/同支持区间处理敏感性.csv')
            save_json(out/'重绘来源记录.json',dict(source=str(source),source_record_sha256=sha(source/'运行记录.json'),
                 reason='修正无资格曲线的空图显示与两堆整体资格判定，未修改提取规则或数值',recomputed_features=False,
                 source_execution_log_variance=log_variance))
        else:feature,curves,sensitivity=compute(cfg,out)
        from 特征报告绘图 import deliver
        result=deliver(stage,out,cfg,feature,curves,sensitivity)
        protected=[]
        for item in frozen['protected_files']:
            checked=GUIDE/item['path'];current=sha(checked)
            # 正式完整代码合法更新后，按已核对归档中的阶段0副本验证旧代码。
            if current!=item['sha256'] and item['path'].startswith('完整代码/'):
                rel=Path(item['path']).relative_to('完整代码')
                matching=[p/rel for p in (GUIDE/'完整代码/完整代码历史副本').iterdir() if p.is_dir() and (p/rel).is_file() and sha(p/rel)==item['sha256']]
                if matching:checked=matching[0];current=sha(checked)
            protected.append(dict(**item,checked_path=str(checked),unchanged=current==item['sha256']))
        if not all(x['unchanged'] for x in protected):raise RuntimeError('阶段0/正式代码发生变化')
        save_json(out/'历史成果保护核验.json',protected)
        record=dict(status=result,started=started,finished=datetime.datetime.now().astimezone().isoformat(),
                    config_sha256=sha(cfgpath),tests=checks,source_files=19,training=False,prediction_metrics=False,
                    code=[dict(file=p.name,sha256=sha(p)) for p in sorted(Path(__file__).parent.glob('*.py'))],
                    protected_files_unchanged=True)
        save_json(out/'运行记录.json',record)
        save_json(out/'结果SHA256.json',[dict(file=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['结果SHA256.json','执行日志.txt']])
        print(result, '结果:',out,flush=True)
    except Exception:
        save_json(out/'故障记录.json',dict(started=started,error=traceback.format_exc()))
        raise


if __name__=='__main__':main()

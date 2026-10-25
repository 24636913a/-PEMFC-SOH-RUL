"""阶段0原始测量审计。只生成描述性统计与时间对应，不拟合健康或预测模型。"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from 核验基础 import (CELLS_REAL, CELLS_FC1, REAL_FIELDS, Stat, col, numeric, mean_cells,
                  table, read_polar, scan_directions)


def nominal(path, stack):
    if stack == 'FC1':
        return float(re.search(r'_T(\d+)', path.name).group(1))
    return 0.0 if 'BoL' in path.name else float(re.search(r'_(\d+)h', path.name).group(1))


def bracket(j, reference):
    values = np.unique(j[np.isfinite(j)])
    below = values[values <= reference]; above = values[values >= reference]
    return dict(参考点下侧最近实测j=float(below[-1]) if len(below) else np.nan,
                参考点上侧最近实测j=float(above[0]) if len(above) else np.nan,
                参考点括距=float(above[0]-below[-1]) if len(below) and len(above) else np.nan,
                参考点可括定=bool(len(below) and len(above)))


def polarization(root, cfg, registry, out):
    fc = root/cfg['FC1目录']; real = root/cfg['RealHyFC目录']
    files = [('FC1', p) for p in sorted(fc.glob('FC1_Pola*')) if p.suffix.lower() in ('.xlsx', '.csv')]
    files += [('RealHyFC', p) for p in sorted(real.glob('Polcurve_Nominal*.csv'))]
    records = []; point_frames = []; direction_rows = []
    for stack, path in files:
        item = registry.start(path, stack, '极化测量')
        frame, encoding = read_polar(path, stack)
        names = CELLS_FC1 if stack == 'FC1' else CELLS_REAL
        voltage, cells = mean_cells(frame, names)
        jname = next((n for n in frame if n.startswith('J (A/cm')), None) if stack == 'FC1' else 'j.S.act [A/cm2]'
        if jname is None or jname not in frame:
            raise ValueError('缺少实测电流密度字段：'+path.name)
        j = col(frame, jname); current = col(frame, 'I (A)' if stack == 'FC1' else 'I.S.act [A]')
        direction, sid = scan_directions(j, cfg['扫描方向确认幅度'])
        dates = pd.to_datetime(frame['DateTime'], errors='coerce', format='mixed') if 'DateTime' in frame else None
        start = dates.min().isoformat() if dates is not None and dates.notna().any() else None
        end = dates.max().isoformat() if dates is not None and dates.notna().any() else None
        valid = np.isfinite(j) & np.isfinite(voltage); subset = j[valid]
        lo, hi = cfg['共同电流密度区间']; reference = cfg['参考电流密度']
        area_mask = np.isfinite(current) & np.isfinite(j) & (np.abs(j) > 0.05)
        area = current[area_mask]/j[area_mask]
        area = area[np.isfinite(area)]
        stat = Stat(); stat.add(col(frame, 'Ustack (V)')-cells.sum(axis=1) if stack == 'FC1' else col(frame, 'U.S.AveCell [V]')-voltage)
        record = dict(电堆=stack, 文件名=path.name, 原始路径=path.relative_to(root).as_posix(),
            标称老化小时=nominal(path, stack), 启停后='afterAirAirStart' in path.name,
            实际测试开始=start, 实际测试结束=end, 原始行数=len(frame), 有效j及电压行数=int(valid.sum()),
            缺失单片记录=int((~np.isfinite(cells).all(axis=1)).sum()),
            最小电流密度=float(subset.min()) if len(subset) else np.nan,
            最大电流密度=float(subset.max()) if len(subset) else np.nan,
            共同区间点数=int((valid & (j >= lo) & (j <= hi)).sum()),
            覆盖完整共同区间=bool(len(subset) and subset.min() <= lo and subset.max() >= hi),
            参考邻域点数=int((valid & (np.abs(j-reference) <= cfg['参考邻域半宽'])).sum()),
            距参考点最近偏差=float(np.abs(subset-reference).min()) if len(subset) else np.nan,
            精确参考点实测存在=bool(np.any(np.isclose(subset, reference, atol=1e-9, rtol=0))),
            升载点数=int((direction == 1).sum()), 降载点数=int((direction == -1).sum()),
            方向未定点数=int((direction == 0).sum()), 扫描分段数=int(len(np.unique(sid))),
            I除以j面积中位_cm2=float(np.median(area)) if len(area) else np.nan,
            文件显式面积_cm2=float(np.nanmedian(col(frame, 'Membrane active area [cm2]'))) if 'Membrane active area [cm2]' in frame else np.nan,
            面积依据='I/J数值一致性推得；缺独立规格证据' if stack == 'FC1' else '显式面积字段和I/J独立核对；缺列者单列',
            累计运行小时字段='OpHrs2 [h]' if 'OpHrs2 [h]' in frame else 'NO_CANONICAL_OPHRS2',
            测试开始OpHrs2=col(frame, 'OpHrs2 [h]')[0], 测试结束OpHrs2=col(frame, 'OpHrs2 [h]')[-1],
            测试开始OpHrs=col(frame, 'OpHrs [h]')[0], 测试结束OpHrs=col(frame, 'OpHrs [h]')[-1],
            冷却液入口温度中位_C=float(np.nanmedian(col(frame, 'T.Si.CL [°C]'))) if 'T.Si.CL [°C]' in frame else np.nan,
            阳极入口压力中位_kPag=float(np.nanmedian(col(frame, 'p.Si.A [kPag]'))) if 'p.Si.A [kPag]' in frame else np.nan,
            阴极入口压力中位_kPag=float(np.nanmedian(col(frame, 'p.Si.C [kPag]'))) if 'p.Si.C [kPag]' in frame else np.nan,
            同时温压条件可核对=stack == 'RealHyFC', 结果发布时间='UNCONFIRMED',
            时钟资格='NOMINAL_ONLY' if dates is None else 'MEASUREMENT_CLOCK_KNOWN_RELEASE_UNKNOWN',
            单位='j:A/cm²; V:平均单片V; 原始点不平滑',
            堆电压或原记录均值一致性最大绝对差_V=stat.record()['max_abs'], **bracket(subset, reference))
        for part in np.unique(sid):
            mask = valid & (sid == part)
            if not mask.any():
                continue
            values = j[mask]
            direction_rows.append(dict(电堆=stack, 文件名=path.name, 分段=int(part),
                方向='升载' if direction[mask][-1] > 0 else '降载' if direction[mask][-1] < 0 else '未定',
                原始点数=int(mask.sum()), 最小j=float(values.min()), 最大j=float(values.max()),
                覆盖共同区间=bool(values.min() <= lo and values.max() >= hi), **bracket(values, reference)))
        points = pd.DataFrame(dict(电堆=stack, 文件名=path.name, 原始行号=np.arange(len(frame)),
            标称老化小时=record['标称老化小时'], 启停后=record['启停后'], 电流密度_A_cm2=j,
            平均单片电压_V=voltage, 电流_A=current, 扫描方向=direction, 扫描分段=sid))
        points['实际测量时间'] = dates.astype('string').to_numpy() if dates is not None else None
        points['原OpHrs2_h'] = col(frame, 'OpHrs2 [h]')
        for raw, label in [('T.Si.CL [°C]', '冷却液入口_C'), ('p.Si.A [kPag]', '阳极入口_kPag'),
                           ('p.Si.C [kPag]', '阴极入口_kPag'), ('F.Si.H.A [nl/min]', '氢流量_nl_min'),
                           ('F.Si.Air.C [nl/min]', '空气流量_nl_min')]:
            points[label] = col(frame, raw)
        point_frames.append(points); records.append(record)
        registry.finish(item, path, len(frame), list(frame.columns), 编码及分隔=encoding,
                        起始时间=start, 结束时间=end, 字段单位=record['单位'], 时钟资格=record['时钟资格'])
    if not records:
        raise RuntimeError('未找到可读极化源文件')
    all_points = pd.concat(point_frames, ignore_index=True)
    table(out/'表格/极化原始测量点.csv', all_points)
    table(out/'表格/极化曲线覆盖范围.csv', records)
    table(out/'表格/极化扫描分段与参考点支持.csv', direction_rows)
    return records, all_points


def fc1_operation(root, cfg, registry, out):
    base = root/cfg['FC1目录']
    files = sorted([p for p in base.glob('FC1_Ageing*') if p.suffix.lower() in ('.xlsx', '.csv')],
                   key=lambda p: int(re.search(r'part(\d+)', p.name).group(1)))
    if not files:
        raise RuntimeError('没有FC1老化文件，不能制造运行图')
    integrity = []; plots = []; gaps = []; joined = []; field_maps = []
    previous = None; segment = -1; offset = 0
    for path in files:
        item = registry.start(path, 'FC1', '老化运行')
        frame = pd.read_excel(path, engine='openpyxl') if path.suffix.lower() == '.xlsx' else pd.read_csv(path, encoding='ISO-8859-1')
        hours = col(frame, 'Time (h)'); voltage, cells = mean_cells(frame, CELLS_FC1)
        jname = next(n for n in frame if n.startswith('J (A/cm'))
        current = col(frame, 'I (A)'); density = col(frame, jname)
        dt = np.diff(hours, prepend=previous if previous is not None else hours[0])*3600
        finite_time = np.isfinite(hours)
        bad = ~np.isfinite(voltage) | ~finite_time | (dt <= 0) | (dt > cfg['FC1间断阈值秒'])
        bad[0] = True
        segments = segment+np.cumsum(bad); segment = int(segments[-1])
        for i in np.flatnonzero((dt > cfg['FC1间断阈值秒']) | (dt < 0) | ((dt == 0) & (np.arange(len(dt)) > 0))):
            gaps.append(dict(电堆='FC1', 文件名=path.name, 原始行号=int(i), 运行小时=hours[i],
                             时间差秒=dt[i], 类型='运行小时间断或回退；无绝对日历时钟'))
        if previous is not None:
            gaps.append(dict(电堆='FC1', 文件名=path.name, 原始行号=0, 运行小时=hours[0],
                             时间差秒=dt[0], 类型='文件边界'))
        residual = col(frame, 'Utot (V)')-cells.sum(axis=1)
        ratio = current[(np.abs(density) > 0.05) & np.isfinite(current)]/density[(np.abs(density) > 0.05) & np.isfinite(current)]
        ratio = ratio[np.isfinite(ratio)]
        record = dict(电堆='FC1', 文件名=path.name, 行数=len(frame), 起始运行小时=hours[0],
            末尾运行小时=hours[-1], 最小运行小时=float(np.nanmin(hours)), 最大运行小时=float(np.nanmax(hours)),
            无效时间=int((~finite_time).sum()), 单片缺失记录=int((~np.isfinite(cells).all(axis=1)).sum()),
            电流缺失=int((~np.isfinite(current)).sum()), 密度缺失=int((~np.isfinite(density)).sum()),
            相邻重复时刻=int((dt[1:] == 0).sum()), 小时回退=int((dt < 0).sum()),
            长间断数=int((dt > cfg['FC1间断阈值秒']).sum()),
            正时间差中位秒=float(np.median(dt[dt > 0])) if np.any(dt > 0) else np.nan,
            完整重复记录=int(frame.duplicated().sum()), 零电流记录=int((np.abs(current) <= 0.1).sum()),
            Utot减单片和最大绝对差_V=float(np.nanmax(np.abs(residual))),
            I除以J面积中位_cm2=float(np.median(ratio)) if len(ratio) else np.nan,
            平均单片电压最小_V=float(np.nanmin(voltage)), 平均单片电压最大_V=float(np.nanmax(voltage)),
            绝对日历时间='NOT_RECORDED', 数据来源资格='LOCAL_EXCEL_COPY_ORIGINAL_CSV_NOT_FOUND' if path.suffix.lower() == '.xlsx' else 'CSV')
        integrity.append(record)
        view = pd.DataFrame(dict(文件名=path.name, 原始行号=np.arange(len(frame)), 全局行号=offset+np.arange(len(frame)),
            运行小时=hours, 平均单片电压_V=voltage, 电堆总电压_V=col(frame, 'Utot (V)'),
            电流_A=current, 电流密度_A_cm2=density, 连续段=segments))
        joined.append(view)
        take = (np.arange(len(frame)) % cfg['FC1作图行间隔'] == 0) | bad
        plots.append(view.loc[take]); offset += len(frame); previous = hours[-1]
        field_maps.extend(dict(电堆='FC1', 文件名=path.name, 原字段=str(name),
            说明='J字段平方与温度符号有原始字符编码损坏；原字符串保留，电流密度按原J数值使用' if '\uff72' in name or '\uff70' in name else '原始字段') for name in frame.columns)
        registry.finish(item, path, len(frame), list(frame.columns), 编码及分隔='Excel OOXML/openpyxl' if path.suffix.lower()=='.xlsx' else 'ISO-8859-1 CSV',
                        起始运行小时=hours[0], 末尾运行小时=hours[-1], 字段单位='Time:h; I:A; 原J:A/cm²; U:V; 压力:mbara')
    combined = pd.concat(joined, ignore_index=True)
    summary = dict(rows=len(combined), first_hour=float(np.nanmin(combined.运行小时)),
        last_hour=float(np.nanmax(combined.运行小时)),
        global_duplicate_hours=int(combined.运行小时.duplicated().sum()), source_files=len(files),
        calendar_time_known=False, original_csv_found=any(p.suffix.lower()=='.csv' for p in files))
    plotted = pd.concat(plots, ignore_index=True)
    table(out/'表格/FC1原始运行绘图点.csv', plotted)
    table(out/'表格/FC1原始字段与单位登记.csv', field_maps)
    return integrity, plotted, gaps, summary, combined


def real_operation(root, cfg, registry, out, polar):
    authority = json.loads((root/cfg['历史运行清单']).read_text(encoding='utf-8-sig'))
    sources = sorted(authority['source_files'], key=lambda row: row['first_datetime'])
    if len(sources) != 17:
        raise RuntimeError('历史权威清单不是17份，不擅自调整数据范围')
    targets = {pd.Timestamp(e[k]).value for e in polar if e['电堆']=='RealHyFC' for k in ('实际测试开始','实际测试结束') if e[k]}
    neighbors = {t: dict(before=None, after=None) for t in targets}
    integrity = []; plotted = []; gaps = []; voltage_stats = Stat(); u6_stats = []; fields_missing = []
    previous = None; segment = -1; global_offset = 0; hashes_match = True
    horizon_rows = {'前300h': 0, '前600h': 0, '600h后': 0}
    real_area = cfg['RealHyFC面积记录参考值']; lower_dt, upper_dt = cfg['RealHyFC连续间隔秒']
    for info in sources:
        path = root/info['path']; item = registry.start(path, 'RealHyFC', 'Loadcycle运行')
        file_size_sha_match = item['SHA256'] == info['sha256'] and item['字节数'] == info['bytes']
        hashes_match &= file_size_sha_match
        n = 0; duplicates = backwards = age_backwards = age_tolerant = missing_time = gap_count = 0
        missing_voltage = missing_current = missing_density = negative_cells = zero_current = overlaps = 0
        density_residual = Stat(); u6 = Stat(); local_voltage = Stat(); sampled = []
        first = last = None; first_calendar = last_calendar = None; field_count = {}
        for frame in pd.read_csv(path, sep=';', encoding='utf-8-sig', chunksize=cfg['分块行数'],
                                 usecols=lambda name: name in REAL_FIELDS, low_memory=False):
            required = set(CELLS_REAL+['DateTime','OpHrs2 [h]','OpHrs [h]','I.S.act [A]','j.S.act [A/cm2]','U6 [V]'])
            if not required.issubset(frame.columns):
                raise ValueError('Loadcycle必要字段缺失：'+str(required-set(frame.columns)))
            dates = pd.to_datetime(frame['DateTime'], errors='coerce', format='mixed')
            ts = dates.to_numpy(dtype='datetime64[ns]').astype(np.int64); time_valid=dates.notna().to_numpy()
            hours = col(frame,'OpHrs2 [h]'); op1=col(frame,'OpHrs [h]')
            current=col(frame,'I.S.act [A]'); j=col(frame,'j.S.act [A/cm2]')
            horizon_rows['前300h'] += int((np.isfinite(hours)&(hours>=0)&(hours<=300)).sum())
            horizon_rows['前600h'] += int((np.isfinite(hours)&(hours>=0)&(hours<=600)).sum())
            horizon_rows['600h后'] += int((np.isfinite(hours)&(hours>600)).sum())
            voltage,cells=mean_cells(frame,CELLS_REAL); u6.add(col(frame,'U6 [V]'))
            local_voltage.add(voltage); voltage_stats.add(voltage)
            density_residual.add(j-current/real_area)
            dt = np.diff(ts, prepend=previous['ns'] if previous else ts[0]).astype(float)/1e9
            valid_pair = time_valid & np.r_[previous['time_valid'] if previous else False, time_valid[:-1]]
            dt[~valid_pair] = np.nan
            da = np.diff(hours, prepend=previous['hours'] if previous else hours[0])
            valid_obs = np.isfinite(cells).all(axis=1)&time_valid&np.isfinite(hours)&np.isfinite(current)&np.isfinite(j)
            connected = valid_obs & (dt>=lower_dt)&(dt<=upper_dt)&(da>=-cfg['运行小时回退容限'])
            connected[1:] &= valid_obs[:-1]
            if previous:
                connected[0] &= previous['valid']
            if n==0:
                connected[0]=False
                if previous:
                    overlaps += int(bool(time_valid[0] and previous['time_valid'] and ts[0]<=previous['ns']))
            starts=~connected; ids=segment+np.cumsum(starts); segment=int(ids[-1])
            problem=(dt>upper_dt)|(dt<=0)|(da < -cfg['运行小时回退容限'])
            problem[0] |= n==0 and previous is not None
            for k in np.flatnonzero(problem):
                gaps.append(dict(电堆='RealHyFC', 文件名=path.name, 原始行号=int(n+k),
                    运行小时=hours[k], 日历时间=str(dates.iloc[k]), 时间差秒=dt[k], 小时差=da[k],
                    类型='文件边界' if k==0 and n==0 else '时间间断/重复/倒序/运行小时回退'))
            duplicates += int(np.nansum(dt==0)); backwards += int(np.nansum(dt<0))
            age_backwards += int((da<0).sum()); age_tolerant += int((da < -cfg['运行小时回退容限']).sum())
            gap_count += int((dt>upper_dt).sum()); missing_time += int((~time_valid).sum())
            missing_voltage += int((~np.isfinite(cells).all(axis=1)).sum())
            missing_current += int((~np.isfinite(current)).sum()); missing_density += int((~np.isfinite(j)).sum())
            negative_cells += int((cells<0).any(axis=1).sum()); zero_current += int((np.abs(current)<=0.1).sum())
            for name in frame.columns:
                if name != 'DateTime':
                    field_count[name]=field_count.get(name,0)+int((~np.isfinite(numeric(frame[name]))).sum())
            view=pd.DataFrame(dict(文件名=path.name, 原始行号=n+np.arange(len(frame)),
                全局行号=global_offset+np.arange(len(frame)), 日历时间=dates.astype('string').to_numpy(),
                运行小时=hours, 原OpHrs=op1, 平均单片电压_V=voltage, 电流_A=current,
                电流密度_A_cm2=j, 连续段=ids))
            take=((n+np.arange(len(frame)))%cfg['RealHyFC作图行间隔']==0)|starts
            bad_index=np.flatnonzero(starts); take[np.maximum(bad_index-1,0)]=True; take[[0,-1]]=True
            sampled.append(view.loc[take])
            finite_indices=np.flatnonzero(time_valid)
            for target,state in neighbors.items():
                eligible=finite_indices[ts[finite_indices]<=target]
                if len(eligible):
                    k=int(eligible[np.argmax(ts[eligible])])
                    if state['before'] is None or ts[k]>state['before']['ns']:
                        state['before']=dict(ns=int(ts[k]),DateTime=str(dates.iloc[k]),hours=float(hours[k]),
                            op1=float(op1[k]),file=path.name,row=int(n+k),global_row=int(global_offset+k))
                eligible=finite_indices[ts[finite_indices]>=target]
                if len(eligible):
                    k=int(eligible[np.argmin(ts[eligible])])
                    if state['after'] is None or ts[k]<state['after']['ns']:
                        state['after']=dict(ns=int(ts[k]),DateTime=str(dates.iloc[k]),hours=float(hours[k]),
                            op1=float(op1[k]),file=path.name,row=int(n+k),global_row=int(global_offset+k))
            if first is None:
                first=float(hours[0]); first_calendar=str(dates.iloc[0])
            last=float(hours[-1]); last_calendar=str(dates.iloc[-1])
            previous=dict(ns=int(ts[-1]),hours=float(hours[-1]),time_valid=bool(time_valid[-1]),valid=bool(valid_obs[-1]))
            n+=len(frame); global_offset+=len(frame)
        plots=pd.concat(sampled,ignore_index=True); plotted.append(plots)
        integrity.append(dict(电堆='RealHyFC',文件名=path.name,行数=n,历史行数=info['rows'],历史行数一致=n==info['rows'],
            历史SHA及大小一致=file_size_sha_match,起始运行小时=first,末尾运行小时=last,
            起始日历时间=first_calendar,末尾日历时间=last_calendar,相邻重复时刻=duplicates,
            日历时间倒序=backwards,跨文件重叠=overlaps,小时回退=age_backwards,
            超容限小时回退=age_tolerant,长间断数=gap_count,无效时间=missing_time,
            单片缺失记录=missing_voltage,电流缺失=missing_current,密度缺失=missing_density,
            存在负单片的记录=negative_cells,零电流记录=zero_current,
            j减I除280最大绝对差=float(density_residual.max_abs),
            平均单片电压最小_V=local_voltage.record()['minimum'],平均单片电压最大_V=local_voltage.record()['maximum']))
        u6_stats.append(dict(文件名=path.name,**u6.record(),用于平均电压=False,异常物理原因='NOT_ESTABLISHED'))
        fields_missing.extend(dict(文件名=path.name,字段=k,缺失或无法解析值=v) for k,v in field_count.items())
        registry.finish(item,path,n,list(frame.columns),编码及分隔='UTF-8-SIG; ; ; 分块读取',
            起始时间=first_calendar,结束时间=last_calendar,起始运行小时=first,末尾运行小时=last,
            字段单位='DateTime:源文件日历(时区未注明); OpHrs2:h; I:A; j:A/cm²; U:V; p:kPag')
    sample=pd.concat(plotted,ignore_index=True)
    table(out/'表格/RealHyFC原始运行绘图点.csv',sample)
    table(out/'表格/U6独立诊断.csv',u6_stats)
    table(out/'表格/RealHyFC逐字段缺失检查.csv',fields_missing)
    summary=dict(rows=global_offset,source_files=len(sources),first_hour=integrity[0]['起始运行小时'],
        last_hour=integrity[-1]['末尾运行小时'],first_datetime=integrity[0]['起始日历时间'],
        last_datetime=integrity[-1]['末尾日历时间'],historical_sha_all_match=hashes_match,
        all_rows_match=all(row['历史行数一致'] for row in integrity),voltage=voltage_stats.record(),
        adaptation_raw_rows=horizon_rows,
        duplicate_check_scope='若全局无倒序，相邻相等检查覆盖全部重复；有倒序则不宣称完整全局唯一性')
    return integrity,sample,gaps,summary,neighbors


def align_polar(root,cfg,polar,neighbors,fc1_all):
    cycles=None; cycle_path=root/cfg['历史周期表']
    if cycle_path.exists():
        cycles=pd.read_csv(cycle_path)
        cycles['_start']=pd.to_datetime(cycles.cycle_start,format='mixed')
        cycles['_end']=pd.to_datetime(cycles.cycle_end,format='mixed')
    rows=[]
    for p in polar:
        record=dict(p)
        record.update(前一运行文件=None,前一运行行号=None,前一运行时间=None,前一运行小时=np.nan,
            后一运行文件=None,后一运行行号=None,后一运行时间=None,后一运行小时=np.nan,
            距前一运行记录秒=np.nan,距后一运行记录秒=np.nan,对应循环编号=None,
            循环依据='没有直接周期身份', 结果可用时间='UNCONFIRMED_RELEASE_TIME')
        if p['电堆']=='FC1':
            target=p['标称老化小时']; before=fc1_all.loc[fc1_all.运行小时<=target]; after=fc1_all.loc[fc1_all.运行小时>=target]
            for group,label,index in ((before,'前一',-1),(after,'后一',0)):
                if len(group):
                    r=group.iloc[index];record[label+'运行文件']=r.文件名;record[label+'运行行号']=int(r.原始行号);record[label+'运行小时']=r.运行小时
            record.update(对齐状态='NOMINAL_ONLY',对齐依据='仅按文件名标称小时寻找运行邻点；不是实测诊断时刻对齐',
                累计小时下界=np.nan,累计小时上界=np.nan,前300h物理结束资格='NOMINAL_ONLY',前600h物理结束资格='NOMINAL_ONLY',
                不确定因素='极化表无Time/DateTime/温压；绝对时刻和发布时刻未确认')
        else:
            start=pd.Timestamp(p['实际测试开始']);end=pd.Timestamp(p['实际测试结束'])
            before=neighbors[start.value]['before'];after=neighbors[end.value]['after']
            for point,label in ((before,'前一'),(after,'后一')):
                if point:
                    record[label+'运行文件']=point['file'];record[label+'运行行号']=point['row'];record[label+'运行时间']=point['DateTime'];record[label+'运行小时']=point['hours']
            record['距前一运行记录秒']=(start.value-before['ns'])/1e9 if before else np.nan
            record['距后一运行记录秒']=(after['ns']-end.value)/1e9 if after else np.nan
            direct=np.isfinite(p['测试开始OpHrs2']) and np.isfinite(p['测试结束OpHrs2'])
            low=p['测试开始OpHrs2'] if direct else before['hours'] if before else np.nan
            high=p['测试结束OpHrs2'] if direct else after['hours'] if after else np.nan
            record.update(累计小时下界=low,累计小时上界=high,
                对齐状态='DIRECT_OPHRS2' if direct else 'CALENDAR_ALIGNED_OPHRS2_BOUNDED' if before and after else 'CALENDAR_ONLY_ONE_SIDED',
                对齐依据='原始DateTime精确找前/后运行观测；缺OpHrs2时仅给单调时钟边界，不用OpHrs固定偏移补算',
                不确定因素='测量结束只给物理最早可用时刻；诊断发布时间未知；标称小时不等于实测累计小时')
            for cutoff in (300,600):
                record[f'前{cutoff}h物理结束资格']=('YES_PHYSICAL_ONLY_RELEASE_UNKNOWN' if np.isfinite(high) and high<=cutoff
                    else 'NO' if np.isfinite(low) and low>cutoff else 'UNCONFIRMED')
            if cycles is not None:
                match=cycles.loc[(cycles['_start']<=start)&(cycles['_end']>=end)]
                if len(match)==1:
                    record['对应循环编号']=int(match.iloc[0].raw_cycle_id)
                    record['循环依据']='历史RCR日期区间覆盖整条曲线；保留历史完整性标志='+str(match.iloc[0].complete_cycle)
        rows.append(record)
    return rows

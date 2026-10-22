"""Explicit user-run pipeline. No import-time execution, raw ingestion, or model dependency."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import traceback
import uuid
import numpy as np
import pandas as pd
from .io_contract import VARIABLES, PHASES, FAMILIES, load_frozen, read_json, save, sha
from .features import phase_features, choose_controls, stream_features
from .rules import fit_train_rules, assess, event_classification, tri_any


class Rates:
    def __init__(self):
        self.counts = defaultdict(lambda: [0, 0, 0, 0])

    def add(self, partition, variable, phase, name, value):
        counts = self.counts[(partition, variable, phase, name)]
        counts[0] += 1
        counts[1 if value is True else 2 if value is False else 3] += 1

    def frame(self):
        columns = ['partition', 'variable_or_family', 'phase', 'criterion', 'total', 'associated',
                   'not_associated', 'unsupported_or_uncalibrated', 'fraction_among_supported', 'fraction_of_all']
        return pd.DataFrame([dict(partition=key[0], variable_or_family=key[1], phase=key[2],
                                  criterion=key[3], total=n, associated=yes, not_associated=no,
                                  unsupported_or_uncalibrated=missing,
                                  fraction_among_supported=yes/(yes+no) if yes+no else np.nan,
                                  fraction_of_all=yes/n if n else np.nan)
                             for key, (n, yes, no, missing) in sorted(self.counts.items())], columns=columns)


def add_variable_rates(rates, row):
    for criterion in ('raw_numerical_change', 'material_step', 'background_unusual'):
        rates.add(row['partition'], row['variable'], row['phase'], criterion, row[criterion])


def add_family_rates(rates, rows):
    if not rows:
        return
    lookup = {(r['variable'], r['phase']): r['background_unusual'] for r in rows}
    for family in FAMILIES:
        variables = [v for v, spec in VARIABLES.items() if spec[1] == family]
        for phase in PHASES:
            state = tri_any(lookup.get((v, phase)) for v in variables)
            rates.add(rows[0]['partition'], family, phase, 'family_background_unusual', state)


def control_records(rows, partition):
    return [dict(id=f'CONTROL_{partition}_{int(row)}', partition=partition, anchor_row=int(row)) for row in rows]


def protection(root, source, declared, before):
    records = []
    for item in declared['files']:
        actual = sha(source/item['file'])
        records.append(dict(group='frozen_audit', file=item['file'], expected=item['sha256'], actual=actual,
                            unchanged=actual == item['sha256']))
    if before.exists():
        for item in pd.read_csv(before).itertuples():
            actual = sha(root/item.path)
            records.append(dict(group='formal_F2', file=item.path, expected=item.sha256, actual=actual,
                                unchanged=actual == item.sha256))
    return dict(all_unchanged=all(r['unchanged'] for r in records), records=records,
                cache_verification='metadata SHA and byte sizes; full cache hashing intentionally not repeated')


def write_report(run, source, classified, rates, old, lag, cfg):
    parts = classified.groupby('partition', sort=False).size()
    unknown = classified.no_specific_association_in_observed_fields
    lines = ['# 电压突变观测关联细化（用户本次运行结果）', '',
             '这是一项事后观测关联分析。全部物理原因仍为UNDETERMINED；没有模型训练或因果识别。', '',
             f'冻结事件保留 {len(classified)} / {cfg["expected_frozen_events"]}；来源：`{source}`。', '',
             '分区事件数：'+', '.join(f'{key}={value}' for key, value in parts.items()), '',
             f'没有突出关联（全部已支持检验均阴性）：{int(unknown.eq(True).sum())}；'
             f'无法完整判定：{int(unknown.isna().sum())}。二者不能混为一类。', '',
             '## 旧比例为何接近100%', '',
             '旧D在整个±180秒中只要出现一次≥5 A加载电流跳变就成立；旧E用整个窗口温压极差与0.1℃/1 kPag比较。'
             '混合事件前后和缺少普通时刻对照，使常见变化也被登记为关联。', '',
             '新版将[-120,0)秒、上一采样→锚点、(0,120]秒分开。'
             '数值变化、Train数值变化幅度代理及超出Train普通背景分位数分别统计。'
             '普通对照排除冻结事件段及两侧121秒。阈值文件先冻结，再应用Validation/后留与事件。', '',
             '## 如何看输出', '',
             '- `event_observational_classification.csv`：每个冻结事件恰好一行，多标签、NA与未知均保留。',
             '- `variable_associations.csv`：9变量×3阶段逐项证据、幅度、时间与缺失原因。',
             '- `association_rates_by_phase.csv`与`ordinary_control_rates.csv`：对照事件与普通时刻的命中率和有效分母。',
             '- `legacy_comparison.csv`：旧D/E及新分阶段家族标签；两套定义不同，不能声称类别直接等价。',
             '- `frozen_train_rules.json`：实际Train控制样本数、q99、数值幅度代理及NA理由。', '',
             '即使新比例仍很高，也不调阈值追求更低的比例。单变量多指标OR、变量家族OR和重叠上下文均会影响比例；'
             'q99不是事件检验的显著性，也不是家族1%假阳性保证。', '',
             '## 时间解释', '',
             '所有阶段相对原冻结anchor_row：正向/混合事件沿用第一正触发，纯负向沿用第一触发。'
             '因此“before”只表示锚点以前，可能已包含同段更早触发或邻近事件，相关污染字段已保存。'
             '它不能自动升级成独立启动前兆。AFTER_ONLY标签更不能用于恢复前预测。', '',
             '原长停留段约65秒、短停留段约7秒为电压恢复后到再次加载的间隔；以下直接引用原汇总，'
             '不重算、不用作当前前兆阈值。actual_load定义下长停留中位数可为66秒，应保留原差异。', '',
             lag.to_csv(index=False), '',
             '## 限制', '',
             '没有阀门/控制器硬件状态，因此温度、压力和气流变化时间不等于硬件动作时间；'
             '负载电流与电压瞬时反向仅为符号相容。共同控制、恢复和传感器噪声均可能产生关联。'
             '普通对照不是与事件同工况的随机干预，也不能用于推断因果风险比。', '',
             '缓存只读、按局部片段取值；只校验既有manifest和文件大小，没有再次扫描/散列整个原始数据或缓存。'
             '独立核验和语义测试须由用户运行。本报告仅在当前运行的完整性核验完成后生成。', '']
    (run/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def run(stage, source_override=None, progress=print):
    stage = Path(stage).resolve()
    research = next(p for p in stage.parents if p.name == 'RENew-PEMFC-2300h-Pytorch')
    root = research.parent
    cfg = read_json(stage/'configs/PROTOCOL.json')
    declared = read_json(stage/'configs/FROZEN_SOURCE_IDENTITIES.json')
    source = Path(source_override).resolve() if source_override else research/declared['source_run_relative_to_research']
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]
    result = stage/'runs'/run_id
    result.mkdir(parents=True, exist_ok=False)
    if result == source or source in result.parents or result in source.parents:
        raise RuntimeError('Source and new output directories must be disjoint')
    def log(message):
        text = datetime.now(timezone.utc).isoformat()+' '+str(message)
        with (result/'run.log').open('a', encoding='utf-8') as stream:
            stream.write(text+'\n')
        progress(text)
    manifest = dict(status='STARTED', source=str(source), result=str(result), analysis_kind='observational association only',
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    training=False, raw_csv_read=False)
    save(result/'run_manifest.json', manifest)
    try:
        log('检查冻结事件、原分区及已有只读缓存；不读取Loadcycle原始CSV')
        events, old, bounds, schema, arrays, positions = load_frozen(source, declared, cfg)
        save(result/'protocol.json', cfg); save(result/'source_identities.json', declared)
        initial = protection(root, source, declared, stage/'docs/F2_PROTECTION_BEFORE.csv')
        save(result/'source_protection_before.json', initial)
        if not initial['all_unchanged']:
            raise RuntimeError('Historical protected files changed; stop without modifying them')
        code = result/'code_snapshot'; code.mkdir()
        identities = []
        for item in sorted(stage.rglob('*')):
            if item.is_file() and 'runs' not in item.relative_to(stage).parts and '__pycache__' not in item.parts:
                if item.suffix not in ('.py', '.json', '.md', '.csv', '.txt'):
                    continue
                relative = item.relative_to(stage); target = code/relative
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(item, target)
                identities.append(dict(file=relative.as_posix(), sha256=sha(item)))
        save(result/'code_identity.json', identities)
        save(result/'cache_access.json', dict(mode='r', full_cache_scan=False, original_declared=schema['cache_files'],
             variables={v: dict(column=s[0], present=s[0] in positions, unit=s[2]) for v, s in VARIABLES.items()}))
        log('按固定日历网格选Train普通对照；排除全部原事件上下文')
        train_rows, selection = choose_controls(arrays, events, 'TRAIN', bounds['TRAIN'], cfg)
        selections = [selection]
        train_anchors = control_records(train_rows, 'TRAIN')
        pd.DataFrame(train_anchors, columns=['id', 'partition', 'anchor_row']).to_csv(result/'train_control_index.csv', index=False)
        stream_features(train_anchors, arrays, positions, bounds, cfg, result/'train_control_features.csv', log)
        frozen = fit_train_rules(result/'train_control_features.csv', cfg)
        save(result/'frozen_train_rules.json', frozen)
        rules_sha = sha(result/'frozen_train_rules.json')
        save(result/'threshold_freeze.json', dict(sha256=rules_sha, at_utc=datetime.now(timezone.utc).isoformat(),
             stage='before event classification and validation/holdout control extraction',
             train_control_index_sha256=sha(result/'train_control_index.csv')))
        log('Train阈值已冻结；开始逐批细化全部15578个原事件')
        event_rates = Rates(); classified = []; batch = []; features_batch = []
        first_ns = arrays['time_ns'][events.first_row.to_numpy(np.int64)]
        last_ns = arrays['time_ns'][events.last_row.to_numpy(np.int64)]
        event_partitions = events.partition.to_numpy(); event_segments = events.segment.to_numpy()
        width = int(cfg['phase_seconds']*1e9)
        for number, event in enumerate(events.to_dict('records'), 1):
            anchor = int(event['anchor_row']); t = int(arrays['time_ns'][anchor])
            comparable = (event_partitions == event['partition']) & (event_segments == event['segment'])
            identity = dict(id=event['event_id'], event_id=event['event_id'], partition=event['partition'],
                            anchor_row=anchor, anchor_DateTime=event['anchor_DateTime'])
            features = phase_features(arrays, positions, anchor, bounds[event['partition']], cfg, identity)
            assessed = [assess(row, frozen) for row in features]
            for row in assessed:
                add_variable_rates(event_rates, row)
            identity.update(first_row=event['first_row'], last_row=event['last_row'], direction=event['direction'],
                segment=event['segment'], partition_first_row=bounds[event['partition']][0],
                source_file=event['source_file'], first_source_row=event['first_source_row'],
                anchor_file_id=int(arrays['file_id'][anchor]), anchor_source_row=int(arrays['source_row'][anchor]),
                anchor_OpHrs2_h=float(arrays['values'][anchor, positions['OpHrs2 [h]']]) if 'OpHrs2 [h]' in positions else np.nan,
                before_contains_own_earlier_trigger=int(event['first_row']) < anchor,
                anchor_seconds_after_event_first_trigger=(t-int(arrays['time_ns'][int(event['first_row'])]))/1e9,
                other_event_segments_in_before=int(np.sum(comparable & (first_ns < t) & (last_ns >= t-width)))-int(int(event['first_row']) < anchor),
                other_event_segments_in_after=int(np.sum(comparable & (first_ns <= t+width) & (last_ns > t)))-int(int(event['last_row']) > anchor),
                legacy_low_zero_recovery_proxy=event['C_low_zero_voltage_recovery_proxy'])
            record = event_classification(identity, assessed, arrays, positions)
            for family in FAMILIES:
                for phase in PHASES:
                    event_rates.add(event['partition'], family, phase, 'family_background_unusual', record[f'{phase}_{family}'])
            classified.append(record); batch.extend(assessed); features_batch.extend(features)
            if number % cfg['batch_anchors'] == 0 or number == len(events):
                for name, rows in (('variable_associations.csv', batch), ('event_phase_features.csv', features_batch)):
                    target = result/name
                    pd.DataFrame(rows).to_csv(target, mode='a', index=False, header=not target.exists())
                batch.clear(); features_batch.clear(); log(f'事件 {number}/{len(events)} 已保存')
        classified = pd.DataFrame(classified)
        if len(classified) != cfg['expected_frozen_events'] or classified.event_id.duplicated().any():
            raise RuntimeError('Output lost or duplicated frozen events')
        expected = events[['event_id', 'partition', 'anchor_row']].reset_index(drop=True)
        pd.testing.assert_frame_equal(classified[expected.columns].reset_index(drop=True), expected, check_dtype=False)
        classified.to_csv(result/'event_observational_classification.csv', index=False)
        rates = event_rates.frame(); rates.to_csv(result/'association_rates_by_phase.csv', index=False)
        log('应用冻结规则评价普通对照；不重新拟合阈值')
        ordinary = Rates()
        for partition in bounds:
            if partition == 'TRAIN':
                path = result/'train_control_features.csv'
            else:
                rows, selection = choose_controls(arrays, events, partition, bounds[partition], cfg)
                selections.append(selection); anchors = control_records(rows, partition)
                pd.DataFrame(anchors, columns=['id', 'partition', 'anchor_row']).to_csv(result/f'{partition.lower()}_control_index.csv', index=False)
                path = result/f'{partition.lower()}_control_features.csv'
                stream_features(anchors, arrays, positions, bounds, cfg, path, log)
            pending = []; pending_id = None
            for frame in pd.read_csv(path, chunksize=cfg['fit_csv_chunksize']):
                for row in frame.to_dict('records'):
                    if pending_id is not None and row['id'] != pending_id:
                        add_family_rates(ordinary, pending); pending.clear()
                    assessed = assess(row, frozen)
                    add_variable_rates(ordinary, assessed)
                    pending_id = row['id']; pending.append(assessed)
            add_family_rates(ordinary, pending)
        ordinary.frame().to_csv(result/'ordinary_control_rates.csv', index=False)
        save(result/'control_selection.json', selections)
        legacy = []
        for partition, group in old.groupby('partition'):
            for tag in ('D_loaded_current_step_proxy', 'E_temperature_pressure_association_proxy'):
                raw = group[tag]
                if raw.dtype == object:
                    flags = raw.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
                else:
                    flags = raw
                legacy.append(dict(partition=partition, criterion=tag, phase='legacy_plus_minus_180s', total=len(group),
                                   associated=int(flags.eq(True).sum()), fraction_of_all=float(flags.eq(True).mean()),
                                   identity='original broad-window definition; NOT equivalent to new phase labels'))
        pd.concat([pd.DataFrame(legacy), rates.loc[rates.criterion.eq('family_background_unusual')]], ignore_index=True).to_csv(result/'legacy_comparison.csv', index=False)
        if sha(result/'frozen_train_rules.json') != rules_sha:
            raise RuntimeError('Frozen Train rules changed during application')
        final = protection(root, source, declared, stage/'docs/F2_PROTECTION_BEFORE.csv')
        save(result/'source_protection_after.json', final)
        if not final['all_unchanged']:
            raise RuntimeError('Historical protected source changed during analysis')
        lag = pd.read_csv(source/'restart_lag_distribution.csv')
        write_report(result, source, classified, rates, old, lag, cfg)
        output_hashes = [dict(file=p.relative_to(result).as_posix(), sha256=sha(p)) for p in sorted(result.rglob('*'))
                         if p.is_file() and p.name not in ('run_manifest.json', 'run.log', 'output_sha256.json')]
        save(result/'output_sha256.json', output_hashes)
        manifest.update(status='COMPLETED', events=len(classified), thresholds_sha256=rules_sha,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        save(result/'run_manifest.json', manifest); log('完成；未改变任何历史文件；物理原因仍为未知')
        return result
    except BaseException as error:
        (result/'ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        manifest.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=str(error))
        save(result/'run_manifest.json', manifest)
        log('已保留失败/中断证据；不改阈值、不重建原始数据、不覆盖旧结果')
        raise

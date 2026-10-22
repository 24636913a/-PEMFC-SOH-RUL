"""Explicit user-run orchestration. No training imports, CUDA, old program execution."""
from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import platform
import subprocess
import traceback
import uuid
import numpy as np
import pandas as pd
from .data_io import dump, ingest, partitions, sha, date, col
from .events import detect, classify, startup_timelines, cycle_mapping
from .precursors import build_features, event_snapshots, observable_onsets, evaluate
from .report import generate


def run_audit(audit, overrides):
    audit = Path(audit).resolve()
    research = next(p for p in audit.parents if p.name == 'RENew-PEMFC-2300h-Pytorch')
    repo = research.parent
    cfg = json.loads((audit/'configs/PROTOCOL.json').read_text(encoding='utf-8'))
    cfg.update(overrides)
    if cfg.get('run_mode') != 'AUDIT' or tuple(cfg['prediction_horizons']) != (30, 60, 120):
        raise ValueError('Only the declared AUDIT and fixed 30/60/120s evaluation is supported')
    if set(cfg['precursor_seconds']) != {0, 1, 5, 10, 30, 60, 120} or cfg['context_seconds'] < 120:
        raise ValueError('Keep all preregistered offsets and >=120 seconds of context')
    if cfg['warning_evaluation_stride'] != 1:
        raise ValueError('All legal observed origins are evaluated; no silent downsampling')
    result = audit/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8])
    result.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('startup_audit.'+result.name); logger.setLevel(logging.INFO); logger.propagate = False
    handler = logging.FileHandler(result/'run.log', encoding='utf-8'); logger.addHandler(handler)
    state = dict(status='STARTED', completed_stages=[], started_utc=datetime.now(timezone.utc).isoformat(),
                 execution_owner='user', device='CPU', models_created=0, formal_F2_modified_by_audit=False)
    formal = research/'REAL_Paper_File/F2主程序'
    protected = {str(p.relative_to(repo)): sha(p) for p in formal.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    dump(result/'config.json', cfg); dump(result/'run_manifest.json', state)

    def log(message):
        logger.info(message); print(message, flush=True)

    def stage(name):
        state['completed_stages'].append(name)
        dump(result/'run_manifest.json', state)

    try:
        state['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        state['environment'] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
        references = json.loads((audit/'configs/REFERENCE_IDENTITIES.json').read_text(encoding='utf-8-sig'))
        for reference in references:
            if '/outputs/' in reference['path'] and sha(research/reference['path']) != reference['sha256']:
                raise RuntimeError('Historical RCR source identity changed; preserve evidence instead of silently realigning')
        state['read_only_reference_identities'] = references
        sources = [audit/'Audit_Startup_Precursors_Interactive.py', *sorted(Path(__file__).parent.glob('*.py')),
                   *sorted((audit/'configs').glob('*.json'))]
        state['code_and_config_sha256'] = {str(p.relative_to(audit)): sha(p) for p in sources}
        snapshot = result/'source_snapshot'; snapshot.mkdir()
        for source in sources:
            dest = snapshot/source.relative_to(audit); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(source.read_bytes())
        dump(result/'run_manifest.json', state)
        log('CPU原始秒级审计开始；不加载任何F2模型，不操作其他运行进程。')
        arrays, metadata = ingest(repo, audit, result, cfg, log)
        state.update(raw_rows=metadata['rows'], source_files=metadata['source_files'], source_manifest_sha256=sha(result/'source_manifest.json'))
        stage('raw_ingest_and_identity')
        split = []
        for name, (lo, hi) in partitions(metadata['rows']).items():
            split.append(dict(partition=name, first_row=lo, end_row_exclusive=hi, records=hi-lo,
                              first_DateTime=date(arrays, lo), last_DateTime=date(arrays, hi-1),
                              first_OpHrs2_h=float(col(arrays, 'OpHrs2 [h]')[lo]),
                              last_OpHrs2_h=float(col(arrays, 'OpHrs2 [h]')[hi-1]),
                              role='Train thresholds' if name == 'TRAIN' else 'frozen-rule description'))
        pd.DataFrame(split).to_csv(result/'partition_table.csv', index=False)
        events, threshold = detect(arrays, result, cfg, log); stage('all_voltage_events')
        classification, targets = classify(arrays, events, cfg, result); stage('event_state_classification')
        timeline = startup_timelines(arrays, events, threshold, cfg, result); stage('restart_timelines')
        cycle_mapping(arrays, events, timeline, research, cfg, result); stage('RCR_template_alignment')
        features, legal, cuts = build_features(arrays, result, cfg, log); stage('Train_only_signal_threshold_freeze')
        snapshots = event_snapshots(arrays, events, targets, cfg, result)
        observable_onsets(arrays, events, features, legal, cuts, cfg, result); stage('causal_event_snapshots')
        metrics = evaluate(arrays, events, targets, features, legal, cuts, cfg, result, log); stage('full_stride1_warning_evaluation')
        if cfg['make_figures']:
            from .figures import generate as make_figures
            make_figures(arrays, events, classification, snapshots, timeline, result, cfg)
            stage('figures_and_plot_CSVs')
        generate(result, metadata, events, classification, snapshots, timeline, metrics, cfg); stage('actual_CSV_report')
        changes = [p for p, expected in protected.items() if not (repo/p).is_file() or sha(repo/p) != expected]
        dump(result/'formal_F2_readonly_check.json', dict(before=protected, changed_during_audit=changes,
             audit_write_scope=str(result), note='Any concurrent external edits are not attributed to this read-only audit'))
        state.update(status='COMPLETED', finished_utc=datetime.now(timezone.utc).isoformat(),
                     protected_files_changed_during_run=changes,
                     artifacts={str(p.relative_to(result)): dict(bytes=p.stat().st_size, sha256=sha(p))
                                for p in result.rglob('*') if p.is_file() and 'cache' not in p.relative_to(result).parts and p.name not in ('run_manifest.json', 'run.log')})
        dump(result/'run_manifest.json', state)
        log(f'审计完成：{result}\n请先阅读AUDIT_REPORT.md；结果为观测关联与固定规则诊断，不是物理因果结论。')
        return result
    except BaseException:
        error = traceback.format_exc()
        (result/'failure.txt').write_text(error, encoding='utf-8')
        state.update(status='INCOMPLETE_OR_FAILED', error=error, finished_utc=datetime.now(timezone.utc).isoformat())
        dump(result/'run_manifest.json', state)
        (result/'INCOMPLETE_REPORT.md').write_text('本次审计未完整执行，不生成最终研究结论。已完成阶段：'+str(state['completed_stages'])+'\n\n'+error, encoding='utf-8')
        log(f'审计中断或失败；已保存真实现场：{result}')
        raise
    finally:
        handler.close(); logger.removeHandler(handler)

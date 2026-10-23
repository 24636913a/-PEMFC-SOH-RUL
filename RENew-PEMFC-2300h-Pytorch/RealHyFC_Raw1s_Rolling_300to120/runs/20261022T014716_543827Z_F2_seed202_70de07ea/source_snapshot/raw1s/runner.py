"""Only execute via explicit user Run All Cells. Never evaluates Holdout."""
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import Windows, freeze_windows, prepare_raw, save_json, sha
from .features import fit_preprocessing
from .metrics import Metrics
from .models import create_model, MODEL_PARAMETERS
from .plots import final_plots, loss_plot


def identity(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validation(model, loader, prep, cache, cfg, device):
    model.eval(); metric = Metrics(cfg['horizon']); total = count = 0; example = None
    with torch.inference_mode():
        for index, (x, objective, truth, base, origins) in enumerate(loader):
            output = model(x.to(device))
            if not torch.isfinite(output).all():
                raise RuntimeError('Nonfinite model output')
            target = objective.to(device)
            total += float(torch.sum((output-target)**2).item()); count += target.numel()
            prediction = output.double().cpu().numpy()*prep['target_scale'] + prep['target_mean']
            if cfg['target_mode'] == 'DeltaV':
                prediction += base.numpy()[:, None]
            metric.update(truth.numpy(), prediction)
            if example is None:
                origin = int(origins[0]); targets = origin + np.arange(1, cfg['horizon']+1)
                times = np.asarray(cache['time_ns'][targets])
                example = pd.DataFrame(dict(origin_row=origin, target_row=targets,
                    origin_DateTime=str(np.datetime64(int(cache['time_ns'][origin]), 'ns')),
                    target_DateTime=times.astype('datetime64[ns]').astype(str),
                    elapsed_seconds=(times-int(cache['time_ns'][origin]))/1e9,
                    target_OpHrs2_h=np.asarray(cache['age'][targets]),
                    true_V=truth[0].numpy(), predicted_V=prediction[0]))
            if index % cfg['progress_batches'] == 0:
                print(f'Validation逐秒完整评价 {index+1}/{len(loader)} batches', flush=True)
    return total/count, metric, example


def run(settings, entry_file):
    cfg = dict(settings)
    if (cfg['lookback'], cfg['horizon'], tuple(cfg['split'])) != (300, 120, (.7, .1, .2)):
        raise ValueError('This protocol is fixed to raw1s 300→120 and chronological 70/10/20')
    if cfg['model'] not in ('F2', 'LSTM') or cfg['run_mode'] != 'TRAIN':
        raise ValueError('Only one selected F2/LSTM fresh TRAIN run is implemented; Holdout has no entry')
    if cfg['target_mode'] != 'V':
        raise ValueError('DeltaV is reserved in dataset code; this registered run permits V only')
    positive_integers = ('train_stride', 'max_epochs', 'patience', 'batch_size',
                         'gradient_accumulation', 'preprocess_origins',
                         'progress_batches', 'plot_every_epochs', 'chunk_rows')
    if cfg['feature_count'] != 10 or any(not isinstance(cfg[key], int) or cfg[key] < 1 for key in positive_integers):
        raise ValueError('Invalid frozen feature budget or training settings')
    entry = Path(entry_file).resolve()
    research = next(p for p in entry.parents if p.name == 'RENew-PEMFC-2300h-Pytorch')
    workspace = research/'RealHyFC_Raw1s_Rolling_300to120'
    cfg.update(repo=str(research.parent), source_authority=str(workspace/'reports/LEGACY_RAW_SOURCE_AUTHORITY.json'))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + f"_{cfg['model']}_seed{cfg['seed']}_" + uuid.uuid4().hex[:8]
    result = workspace/'runs'/run_id
    result.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('raw1s.'+run_id); logger.setLevel(logging.INFO); logger.propagate = False
    handler = logging.FileHandler(result/'run.log', encoding='utf-8'); logger.addHandler(handler)
    manifest = dict(status='STARTED', run_id=run_id, started_utc=datetime.now(timezone.utc).isoformat(),
                    execution_owner='user', holdout_evaluated=False, target_mode=cfg['target_mode'])
    save_json(result/'run_manifest.json', manifest)
    save_json(result/'config.json', cfg)
    try:
        manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=research.parent, text=True).strip()
        manifest['git_status'] = subprocess.check_output(['git', 'status', '--short'], cwd=research.parent, text=True)
        sources = [entry, *sorted((entry.parent/'raw1s').glob('*.py'))]
        manifest['code_sha256'] = {str(p.relative_to(entry.parent)): sha(p) for p in sources}
        snapshot = result/'source_snapshot'; snapshot.mkdir()
        for path in sources:
            dest = snapshot/path.relative_to(entry.parent); dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
        save_json(result/'run_manifest.json', manifest)
        logger.info('Configuration frozen before data preparation: %s', json.dumps(cfg))
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; no automatic CPU fallback')
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)
        manifest['environment'] = dict(python=platform.python_version(), torch=str(torch.__version__),
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
            strict_determinism=False, bitwise_reproducibility_promised=False)
        print('开始真实原始CSV读取；不是旧缓存回放。全量逐秒训练/Validation可能耗时很长。', flush=True)
        cache, n = prepare_raw(cfg, result)
        parts = freeze_windows(cache, n, cfg, result)
        logger.info('All partition/window identities frozen before model construction: %s', sha(result/'split_and_windows.json'))
        print(f'已冻结{n:,}条原始记录的7:1:2边界及逐秒窗口；不评价Holdout。', flush=True)
        prep = fit_preprocessing(cache, parts['TRAIN'], cfg, result)
        save_json(result/'preprocess_info.json', prep)
        manifest['preprocess_sha256'] = sha(result/'preprocess_info.json')
        manifest['split_sha256'] = sha(result/'split_and_windows.json')
        # Fixed index stride chosen BEFORE training; Validation is ALWAYS stride 1.0
        train_origins = parts['TRAIN'][::cfg['train_stride']]
        manifest['training_origin_count'] = len(train_origins)
        manifest['training_origin_sha256'] = hashlib.sha256(np.asarray(train_origins).tobytes()).hexdigest()
        generator = torch.Generator().manual_seed(cfg['seed'])
        train = DataLoader(Windows(cache, train_origins, cfg, prep), batch_size=cfg['batch_size'],
                           shuffle=True, generator=generator, num_workers=0, pin_memory=True)
        val = DataLoader(Windows(cache, parts['VALIDATION'], cfg, prep), batch_size=cfg['batch_size'],
                         shuffle=False, num_workers=0, pin_memory=True)
        model = create_model(cfg).to('cuda')
        manifest['model_parameters'] = MODEL_PARAMETERS
        manifest['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        manifest['initial_state_sha256'] = identity(model)
        save_json(result/'run_manifest.json', manifest)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        history = []; best = float('inf'); best_epoch = None; stale = 0; updates = 0
        checkpoint = result/'best_model.pt'
        logger.info('Fresh model and optimizer; %s parameters; no checkpoint loaded', manifest['trainable_parameters'])
        print(f"开始{cfg['model']} Seed{cfg['seed']}真实训练，参数{manifest['trainable_parameters']:,}。", flush=True)
        started = time.perf_counter()
        for epoch in range(1, cfg['max_epochs']+1):
            model.train(); loss_sum = 0.0; seen = 0
            accumulation = cfg['gradient_accumulation']
            optimizer.zero_grad(set_to_none=True)
            group_n = 0
            for batch, (x, objective, _, _, _) in enumerate(train):
                output = model(x.to('cuda')); target = objective.to('cuda')
                squared = torch.sum((output-target)**2)
                if not torch.isfinite(squared):
                    raise RuntimeError('Nonfinite training loss')
                # Accumulate SUM gradients, normalize by actual query count;
                # correct also for final short microbatch/accumulation group.
                squared.backward(); group_n += target.numel()
                loss_sum += float(squared.detach().item()); seen += target.numel()
                if (batch+1) % accumulation == 0 or batch+1 == len(train):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_n)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step(); updates += 1; optimizer.zero_grad(set_to_none=True); group_n = 0
                if batch % cfg['progress_batches'] == 0:
                    print(f'Epoch {epoch} Train {batch+1}/{len(train)} batches', flush=True)
            val_loss, score, _ = validation(model, val, prep, cache, cfg, 'cuda')
            current = score.result()['RMSE_mV']
            improved = current < best
            if improved:
                best, best_epoch, stale = current, epoch, 0
                temporary = checkpoint.with_suffix('.pt.writing')
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, config=cfg, preprocessing=prep,
                                split_sha256=manifest['split_sha256'], updates=updates,
                                validation_RMSE_mV=current, code_sha256=manifest['code_sha256']), temporary)
                temporary.replace(checkpoint)
            else:
                stale += 1
            row = dict(epoch=epoch, train_loss=loss_sum/seen, validation_loss=val_loss,
                       validation_RMSE_mV=current, best_epoch=best_epoch, optimizer_updates=updates,
                       last_gradient_norm=float(norm), elapsed_seconds=time.perf_counter()-started)
            history.append(row)
            pd.DataFrame(history).to_csv(result/'training_history.csv', index=False)
            logger.info('Epoch %s', json.dumps(row))
            print(f'Epoch {epoch}: Train Loss={row["train_loss"]:.7g} Validation Loss={val_loss:.7g} Best Epoch={best_epoch} stale={stale}/{cfg["patience"]}', flush=True)
            if epoch == 1 or epoch % cfg['plot_every_epochs'] == 0 or stale >= cfg['patience']:
                loss_plot(history, result)
            if stale >= cfg['patience']:
                break
        manifest['final_state_sha256'] = identity(model)
        manifest['optimizer_updates'] = updates
        if updates == 0 or manifest['final_state_sha256'] == manifest['initial_state_sha256']:
            raise RuntimeError('No parameter update evidence')
        # Only this run's freshly trained best checkpoint is loaded.
        stored = torch.load(checkpoint, map_location='cuda', weights_only=False)
        model.load_state_dict(stored['state_dict'], strict=True)
        _, score, example = validation(model, val, prep, cache, cfg, 'cuda')
        metric = score.result()
        if not np.isclose(metric['RMSE_mV'], best, atol=1e-8, rtol=1e-8):
            raise RuntimeError('Best checkpoint Validation replay differs; retain failure for user inspection')
        metric.update(model=cfg['model'], seed=cfg['seed'], partition='VALIDATION', best_epoch=best_epoch, holdout_evaluated=False)
        save_json(result/'metrics.json', metric)
        pd.DataFrame([metric]).to_csv(result/'metrics.csv', index=False)
        loss_plot(history, result, final=True)
        final_plots(metric, score.horizon_rmse(), example, cfg, result)
        manifest.update(status='COMPLETED_VALIDATION_ONLY', best_epoch=best_epoch, stop_epoch=history[-1]['epoch'],
                        best_checkpoint_sha256=sha(checkpoint), finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(result/'run_manifest.json', manifest)
        logger.info('Completed Validation only: %s', json.dumps(metric))
        r2_text = 'NA' if metric['R2'] is None else f"{metric['R2']:.6f}"
        print(f"RMSE: {metric['RMSE_mV']:.6f} mV\nMAE : {metric['MAE_mV']:.6f} mV\nR²  : {r2_text}\n保存路径: {result}")
        return result
    except BaseException:
        error = traceback.format_exc()
        (result/'failure.txt').write_text(error, encoding='utf-8')
        manifest.update(status='FAILED_OR_INTERRUPTED', failure=error, finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(result/'run_manifest.json', manifest)
        logger.error(error)
        print(f'运行失败或中断，现场保留：{result}', flush=True)
        raise
    finally:
        handler.close(); logger.removeHandler(handler)

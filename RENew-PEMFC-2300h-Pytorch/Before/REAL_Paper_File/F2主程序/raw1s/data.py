"""User-run raw CSV streaming, immutable per-run mmap and exact stride-one index."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .features import CELLS, EXTRA_FIELDS, numeric, raw_base, window_features


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


DTYPES = dict(x='float32', voltage='float64', time_ns='int64', age='float64',
              segment='int32', valid='uint8', file_id='int16', source_row='int64', u6='float32')


def prepare_raw(cfg, run):
    """Never uses legacy derived arrays. User execution verifies all 17 payloads."""
    repo = Path(cfg['repo'])
    authority = json.loads(Path(cfg['source_authority']).read_text(encoding='utf-8-sig'))
    sources = sorted(authority['source_files'], key=lambda f: f['first_datetime'])
    if len(sources) != 17:
        raise RuntimeError('Exactly 17 original Loadcycle files required')
    cache_dir = run / 'cache'; cache_dir.mkdir(exist_ok=False)
    streams = {key: (cache_dir / (key + '.bin')).open('wb') for key in DTYPES}
    n = 0; segment = -1; previous = None; identities = []; u6_stats = []
    try:
        for fid, info in enumerate(sources):
            source = (repo / info['path']).resolve()
            if not source.is_relative_to(repo.resolve()) or not source.name.startswith('Loadcycle_'):
                raise RuntimeError('Unexpected raw source path')
            if source.stat().st_size != info['bytes'] or sha(source) != info['sha256']:
                raise RuntimeError(f'Raw/LFS identity mismatch: {source}')
            print(f'读取原始秒级数据 {fid + 1}/17: {source.name}', flush=True)
            offset = 0; finite_u6 = 0; bad_u6 = 0; min_u6 = None; max_u6 = None
            required = set(CELLS + EXTRA_FIELDS + ('DateTime', 'I.S.act [A]', 'j.S.act [A/cm2]', 'U6 [V]'))
            for frame in pd.read_csv(source, sep=';', encoding='utf-8-sig', chunksize=cfg['chunk_rows'],
                                     usecols=lambda name: name in required, low_memory=False):
                if not frame.columns.is_unique or not required.issubset(frame.columns):
                    raise RuntimeError(f'Raw schema mismatch: {source.name}')
                x, voltage, valid = raw_base(frame)
                dates = pd.to_datetime(frame['DateTime'], errors='raise')
                if dates.isna().any():
                    raise RuntimeError('Missing DateTime; no interpolation allowed')
                ts = dates.to_numpy(dtype='datetime64[ns]').astype(np.int64)
                age = numeric(frame['OpHrs2 [h]'])
                dt = np.diff(ts, prepend=previous[0] if previous else ts[0] - 1_000_000_000)
                # Refuse reordered/duplicate records, rather than quietly sort raw data.
                if np.any(dt <= 0):
                    raise RuntimeError('DateTime is not strictly chronological; preserve evidence and inspect source ordering')
                da = np.diff(age, prepend=previous[1] if previous else age[0])
                connected = (dt >= 950_000_000) & (dt <= 1_050_000_000) & (da >= -1e-7) & valid
                connected[1:] &= valid[:-1]
                connected[0] &= offset > 0 and previous is not None and previous[2]
                starts = ~connected
                ids = segment + np.cumsum(starts).astype(np.int32)
                segment = int(ids[-1])
                u6 = numeric(frame['U6 [V]'])
                good = u6[np.isfinite(u6)]
                finite_u6 += len(good); bad_u6 += int((~np.isfinite(u6)).sum())
                if len(good):
                    min_u6 = float(good.min()) if min_u6 is None else min(min_u6, float(good.min()))
                    max_u6 = float(good.max()) if max_u6 is None else max(max_u6, float(good.max()))
                arrays = dict(x=x, voltage=voltage, time_ns=ts, age=age, segment=ids,
                              valid=valid, file_id=np.full(len(frame), fid),
                              source_row=np.arange(offset, offset + len(frame)), u6=u6)
                for key, value in arrays.items():
                    np.asarray(value, dtype=DTYPES[key]).tofile(streams[key])
                previous = (int(ts[-1]), float(age[-1]), bool(valid[-1]))
                offset += len(frame); n += len(frame)
            identities.append(dict(path=info['path'], sha256=info['sha256'], bytes=info['bytes'], rows=offset))
            u6_stats.append(dict(file=source.name, finite=finite_u6, missing=bad_u6, min_V=min_u6, max_V=max_u6,
                                 target_included=False, cause='NOT_ESTABLISHED'))
    finally:
        for stream in streams.values():
            stream.close()
    pd.DataFrame(u6_stats).to_csv(run / 'u6_diagnostics.csv', index=False)
    cache = {key: np.memmap(cache_dir / (key + '.bin'), mode='r', dtype=dtype,
                            shape=(n, 24) if key == 'x' else (n,)) for key, dtype in DTYPES.items()}
    save_json(run / 'source_manifest.json', dict(rows=n, files=identities, cached_arrays=DTYPES,
              source_authority_sha256=sha(cfg['source_authority']), cache_sha256={k: sha(cache_dir / (k + '.bin')) for k in DTYPES}))
    return cache, n


def freeze_windows(cache, n, cfg, run):
    """Entire 300 history and 120 labels inside each partition: disjoint support.

    Counts/identities are prepared for Holdout without computing any performance.
    Rejections are summarized, with full source-row validity/segment arrays kept.
    """
    edges = (0, int(n * .7), int(n * .8), n)
    parts = {}; contract = []
    l, h = cfg['lookback'], cfg['horizon']
    for name, left, right in zip(('TRAIN', 'VALIDATION', 'EXPLORATORY_HOLDOUT'), edges[:-1], edges[1:]):
        index_path = run / 'cache' / (name + '_origins.bin')
        legal = 0; invalid = 0
        with index_path.open('wb') as stream:
            for start in range(left + l - 1, max(left + l - 1, right - h), cfg['chunk_rows']):
                origins = np.arange(start, min(right - h, start + cfg['chunk_rows']), dtype=np.int64)
                first = origins - l + 1; last = origins + h
                # Monotone segment IDs: endpoint equality certifies every interior row.
                ok = (cache['segment'][first] == cache['segment'][last]) & cache['valid'][first] & cache['valid'][last]
                origins[ok.astype(bool)].tofile(stream)
                legal += int(ok.sum()); invalid += int((~ok.astype(bool)).sum())
        if not legal:
            raise RuntimeError(f'No legal {name} windows; do not alter continuity/split to force results')
        parts[name] = np.memmap(index_path, dtype='int64', mode='r', shape=(legal,))
        contract.append(dict(partition=name, start_row=left, end_row_exclusive=right, records=right-left,
                             start_DateTime=str(np.datetime64(int(cache['time_ns'][left]), 'ns')),
                             last_DateTime=str(np.datetime64(int(cache['time_ns'][right-1]), 'ns')),
                             start_OpHrs2_h=float(cache['age'][left]), last_OpHrs2_h=float(cache['age'][right-1]),
                             legal_origins=legal, queries=legal*h, invalid_internal_windows=invalid,
                             boundary_excluded=min(right-left, l+h-1), origin_index_sha256=sha(index_path),
                             index_file=str(index_path.relative_to(run)), stride=1, support='entire input and target in partition'))
    save_json(run / 'split_and_windows.json', dict(partitions=contract, lookback=l, horizon=h,
              continuity='same source file, 950.0.1050ms, finite candidate sensors, nondecreasing OpHrs2',
              overlapping_queries_are_independent_trials=False, holdout_evaluated=False))
    pd.DataFrame(contract).to_csv(run / 'partition_table.csv', index=False)
    return parts


class Windows(Dataset):
    def __init__(self, cache, origins, cfg, prep):
        self.cache, self.origins, self.cfg, self.prep = cache, origins, cfg, prep
        self.mask = np.zeros(24, dtype=np.float32)
        self.mask[prep['selected_indices']] = 1.0

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index]); h = self.cfg['horizon']; l = self.cfg['lookback']
        x = window_features(self.cache['x'][origin-l+1:origin+1])
        x = ((x - np.asarray(self.prep['mean'])) / np.asarray(self.prep['scale'])) * self.mask
        y = np.array(self.cache['voltage'][origin+1:origin+h+1], dtype=np.float64)
        base = float(self.cache['voltage'][origin])
        objective = y - base if self.cfg['target_mode'] == 'DeltaV' else y
        objective = (objective - self.prep['target_mean']) / self.prep['target_scale']
        return (torch.from_numpy(x.astype(np.float32)), torch.from_numpy(objective.astype(np.float32)),
                torch.from_numpy(y), base, origin)

"""Stream only original Loadcycle CSV into this audit run's own read-only mmap."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

CELLS = tuple(f'U{i} [V]' for i in range(1, 12) if i != 6)
CORE = ('I.S.act [A]', 'I.S.set [A]', 'I.S.set.internal [A]',
        'j.S.act [A/cm2]', 'operation_point', 'T.Si.CL [°C]', 'T.So.CL [°C]',
        'p.Si.A [kPag]', 'p.Si.C [kPag]', 'OpHrs2 [h]')
OPTIONAL = ('T.Si.A [°C]', 'T.Si.C [°C]', 'T.So.A [°C]', 'T.So.C [°C]',
            'F.Si.H.A [nl/min]', 'F.Si.Air.C [nl/min]', 'F.Si.C.total [nl/min]')
COLUMNS = ('mean_voltage_V', *CELLS, 'U6 [V]', *CORE, *OPTIONAL)
INDEX = {name: i for i, name in enumerate(COLUMNS)}
SIGNALS = dict(coolant_in='T.Si.CL [°C]', coolant_out='T.So.CL [°C]',
               anode_p='p.Si.A [kPag]', cathode_p='p.Si.C [kPag]')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def number(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    text = series.astype('string').str.strip().str.replace('\ufeff', '', regex=False)
    both = text.str.contains(',', regex=False, na=False) & text.str.contains('.', regex=False, na=False)
    text = text.mask(both, text.str.replace(',', '', regex=False))
    return pd.to_numeric(text.str.replace(',', '.', regex=False), errors='coerce').to_numpy(dtype=float, na_value=np.nan)


def ingest(repo, audit, run, cfg, log):
    authority_path = audit/'configs/RAW_SOURCE_AUTHORITY.json'
    authority = json.loads(authority_path.read_text(encoding='utf-8-sig'))
    files = sorted(authority['source_files'], key=lambda f: f['first_datetime'])
    if len(files) != 17:
        raise RuntimeError('Expected exactly 17 original Loadcycle sources')
    cache = run/'cache'; cache.mkdir(exist_ok=False)
    dtypes = dict(values='float64', time_ns='int64', segment='int32',
                  file_id='int16', source_row='int64', delta_V='float64')
    streams = {k: (cache/(k+'.bin')).open('wb') for k in dtypes}
    total = 0; segment_id = -1; prev = None; identities = []; schema = []; u6_records = []
    try:
        for fid, source in enumerate(files):
            path = (repo/source['path']).resolve()
            if not path.is_relative_to(repo.resolve()) or not path.name.startswith('Loadcycle_'):
                raise RuntimeError('Source outside allowed raw Loadcycle files')
            if path.stat().st_size != source['bytes'] or sha(path) != source['sha256']:
                raise RuntimeError(f'Raw source/LFS identity mismatch: {path}')
            log(f'读取原始文件 {fid+1}/17：{path.name}')
            row = 0; required = set(CELLS+CORE+('U6 [V]', 'DateTime'))
            u6_count = u6_missing = invalid_voltage_rows = breaks = 0
            u6_min = u6_max = sample_min = sample_max = None
            for frame in pd.read_csv(path, sep=';', encoding='utf-8-sig', chunksize=cfg['chunk_rows'],
                                     usecols=lambda name: name in required or name in OPTIONAL, low_memory=False):
                if not frame.columns.is_unique or not required.issubset(frame.columns):
                    raise RuntimeError(f'Required raw schema missing in {path.name}')
                if row == 0:
                    schema.append(dict(file=path.name, optional_present=[c for c in OPTIONAL if c in frame],
                                       optional_missing=[c for c in OPTIONAL if c not in frame]))
                ts = pd.to_datetime(frame.DateTime, errors='raise')
                if ts.isna().any():
                    raise RuntimeError('Missing DateTime; no interpolation allowed')
                ns = ts.to_numpy(dtype='datetime64[ns]').astype(np.int64)
                x = np.full((len(frame), len(COLUMNS)), np.nan)
                for name in COLUMNS[1:]:
                    if name in frame:
                        x[:, INDEX[name]] = number(frame[name])
                x[:, 0] = x[:, 1:11].mean(1)
                voltage_ok = np.isfinite(x[:, 1:11]).all(1)
                dt = np.diff(ns, prepend=prev[0] if prev else ns[0]-1_000_000_000)
                if np.any(dt <= 0):
                    bad = np.flatnonzero(dt <= 0)
                    pd.DataFrame(dict(source_file=path.name, source_row=row+bad,
                                      DateTime=ns[bad].astype('datetime64[ns]').astype(str), delta_ns=dt[bad])).to_csv(run/'invalid_datetime_records.csv', index=False)
                    raise RuntimeError('DateTime duplicate/reversal; do not silently reorder raw records')
                age = x[:, INDEX['OpHrs2 [h]']]
                age_delta = np.diff(age, prepend=prev[2] if prev else age[0])
                connected = ((dt >= cfg['minimum_sample_dt_seconds']*1e9) &
                             (dt <= cfg['maximum_sample_dt_seconds']*1e9) & voltage_ok &
                             np.isfinite(age) & (age_delta >= -1e-7))
                connected[1:] &= voltage_ok[:-1]
                connected[0] &= row > 0 and prev is not None and prev[3]
                internal_dt = dt if row else dt[1:]
                if len(internal_dt):
                    sample_min = int(internal_dt.min()) if sample_min is None else min(sample_min, int(internal_dt.min()))
                    sample_max = int(internal_dt.max()) if sample_max is None else max(sample_max, int(internal_dt.max()))
                invalid_voltage_rows += int((~voltage_ok).sum()); breaks += int((~connected).sum())
                u6 = x[:, INDEX['U6 [V]']]; good_u6 = u6[np.isfinite(u6)]
                u6_count += len(good_u6); u6_missing += int((~np.isfinite(u6)).sum())
                if len(good_u6):
                    u6_min = float(good_u6.min()) if u6_min is None else min(u6_min, float(good_u6.min()))
                    u6_max = float(good_u6.max()) if u6_max is None else max(u6_max, float(good_u6.max()))
                ids = segment_id + np.cumsum(~connected).astype(np.int32)
                segment_id = int(ids[-1])
                delta = np.diff(x[:, 0], prepend=prev[1] if prev else np.nan)
                delta[~connected] = np.nan
                arrays = dict(values=x, time_ns=ns, segment=ids, file_id=np.full(len(frame), fid),
                              source_row=np.arange(row, row+len(frame)), delta_V=delta)
                for key, value in arrays.items():
                    np.asarray(value, dtype=dtypes[key]).tofile(streams[key])
                prev = (int(ns[-1]), float(x[-1, 0]), float(age[-1]), bool(voltage_ok[-1]))
                row += len(frame); total += len(frame)
                dump(run/'ingest_progress.json', dict(status='PARTIAL_RAW_INGEST', total_rows_written=total,
                     file_id=fid, file=path.name, source_rows_written=row, cache_dtypes=dtypes, columns=COLUMNS))
            if row != source['rows']:
                raise RuntimeError('Raw row count differs from historical source authority')
            identities.append(dict(file_id=fid, path=source['path'], sha256=source['sha256'], rows=row,
                                   internal_dt_min_ns=sample_min, internal_dt_max_ns=sample_max,
                                   continuity_breaks_including_file_start=breaks, invalid_voltage_rows=invalid_voltage_rows))
            u6_records.append(dict(file=path.name, finite_rows=u6_count, missing_rows=u6_missing,
                                   minimum_V=u6_min, maximum_V=u6_max, target_included=False, physical_cause='NOT_ESTABLISHED'))
    finally:
        for stream in streams.values():
            stream.close()
    arrays = {key: np.memmap(cache/(key+'.bin'), mode='r', dtype=dtype,
              shape=(total, len(COLUMNS)) if key == 'values' else (total,)) for key, dtype in dtypes.items()}
    metadata = dict(rows=total, source_files=identities, source_authority_sha256=sha(authority_path),
                    columns=COLUMNS, dtypes=dtypes, schema=schema,
                    cache_files={k: dict(path='cache/'+k+'.bin', sha256=sha(cache/(k+'.bin'))) for k in dtypes},
                    U6_role='diagnostic_only; physical cause unknown',
                    row_indexing='global_row and source_row are zero based; CSV header excluded',
                    continuity='same file, finite 10 cell voltages, 950..1050ms and nondecreasing finite OpHrs2')
    dump(run/'source_manifest.json', metadata)
    pd.DataFrame(u6_records).to_csv(run/'u6_diagnostics.csv', index=False)
    return arrays, metadata


def partitions(n):
    return dict(TRAIN=(0, int(.7*n)), VALIDATION=(int(.7*n), int(.8*n)),
                EXPLORATORY_HOLDOUT=(int(.8*n), n))


def col(arrays, name):
    return arrays['values'][:, INDEX[name]]


def date(arrays, row):
    return None if row is None else str(np.datetime64(int(arrays['time_ns'][int(row)]), 'ns'))


def match_past(arrays, desired_ns, reference_rows, cfg):
    """As-of only: never select a record after the requested timestamp."""
    rows = np.searchsorted(arrays['time_ns'], desired_ns, side='right')-1
    clipped = np.clip(rows, 0, len(arrays['time_ns'])-1)
    ok = ((rows >= 0) & (desired_ns-arrays['time_ns'][clipped] <= cfg['match_tolerance_seconds']*1e9) &
          (arrays['segment'][clipped] == arrays['segment'][reference_rows]))
    return clipped, ok

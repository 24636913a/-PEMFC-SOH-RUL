"""Frozen event identity and read-only existing mmap. Never reads raw CSVs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

VARIABLES = {
    'actual_current': ('I.S.act [A]', 'CURRENT', 'A'),
    'set_current': ('I.S.set [A]', 'COMMAND', 'A'),
    'internal_set_current': ('I.S.set.internal [A]', 'COMMAND', 'A'),
    'coolant_in': ('T.Si.CL [°C]', 'COOLANT_TEMP', '°C'),
    'coolant_out': ('T.So.CL [°C]', 'COOLANT_TEMP', '°C'),
    'anode_pressure': ('p.Si.A [kPag]', 'INLET_PRESSURE', 'kPag'),
    'cathode_pressure': ('p.Si.C [kPag]', 'INLET_PRESSURE', 'kPag'),
    'hydrogen_flow': ('F.Si.H.A [nl/min]', 'GAS_FLOW', 'nl/min'),
    'air_flow': ('F.Si.Air.C [nl/min]', 'GAS_FLOW', 'nl/min'),
}
PHASES = ('before', 'instant', 'after')
FAMILIES = ('CURRENT', 'COMMAND', 'COOLANT_TEMP', 'INLET_PRESSURE', 'GAS_FLOW')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_frozen(source, declared, cfg):
    if read_json(source/'run_manifest.json')['status'] != 'COMPLETED':
        raise RuntimeError('Source audit must be completed; never analyze a changing partial run')
    for item in declared['files']:
        if sha(source/item['file']) != item['sha256']:
            raise RuntimeError('Frozen source identity mismatch: '+item['file'])
    catalog = pd.read_csv(source/'event_catalog.csv')
    old = pd.read_csv(source/'event_classification.csv')
    parts = pd.read_csv(source/'partition_table.csv')
    if len(catalog) != cfg['expected_frozen_events'] or catalog.event_id.duplicated().any() or old.event_id.duplicated().any():
        raise RuntimeError('Frozen 15578-event identity is not intact; do not redetect or remove events')
    joined = catalog.merge(old[['event_id', 'partition', 'anchor_row', 'anchor_DateTime', 'C_low_zero_voltage_recovery_proxy']],
                           on=['event_id', 'partition'], how='outer', indicator=True, validate='one_to_one')
    if not joined['_merge'].eq('both').all():
        raise RuntimeError('Catalog and old classification event/partition keys differ')
    expected = joined.first_positive_row.fillna(joined.first_row).to_numpy(dtype=np.int64)
    if not np.array_equal(expected, joined.anchor_row.to_numpy(dtype=np.int64)):
        raise RuntimeError('Historical anchors disagree; stop rather than choose a more favorable anchor')
    joined = joined.drop(columns='_merge').sort_values('event_id').reset_index(drop=True)
    schema = read_json(source/'source_manifest.json'); n = int(schema['rows'])
    bounds = {row.partition: (int(row.first_row), int(row.end_row_exclusive)) for row in parts.itertuples()}
    ordered = [bounds[name] for name in ('TRAIN', 'VALIDATION', 'EXPLORATORY_HOLDOUT')]
    if ordered[0][0] != 0 or ordered[-1][1] != n or ordered[0][1] != ordered[1][0] or ordered[1][1] != ordered[2][0]:
        raise RuntimeError('Original partition table has changed or is not contiguous')
    arrays = {}
    # Size checks and original declared hashes only. No full cache rehash/scan.
    for name in ('values', 'time_ns', 'segment', 'file_id', 'source_row'):
        dtype = np.dtype(schema['dtypes'][name]); path = source/schema['cache_files'][name]['path']
        shape = (n, len(schema['columns'])) if name == 'values' else (n,)
        if not path.is_file() or path.stat().st_size != int(np.prod(shape))*dtype.itemsize:
            raise RuntimeError(f'Existing cache unavailable/wrong size: {name}. Restore the original ZIP to a separate directory; do not rebuild raw data.')
        arrays[name] = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    positions = {name: i for i, name in enumerate(schema['columns'])}
    if 'mean_voltage_V' not in positions:
        raise RuntimeError('Saved mean-voltage definition unavailable')
    for event in joined.itertuples():
        lo, hi = bounds[event.partition]; row = int(event.anchor_row)
        if not (lo <= int(event.first_row) <= row <= int(event.last_row) < hi):
            raise RuntimeError('Frozen anchor/event crosses its original partition')
        if arrays['segment'][row] != event.segment:
            raise RuntimeError('Frozen cache segment differs from saved event identity')
        for field, position in (('anchor_DateTime', row), ('first_DateTime', int(event.first_row)),
                                ('last_DateTime', int(event.last_row))):
            expected_ns = int(np.datetime64(getattr(event, field), 'ns').astype('int64'))
            if int(arrays['time_ns'][position]) != expected_ns:
                raise RuntimeError('Exact saved timestamp/cache mismatch: '+field)
    return joined, old, bounds, schema, arrays, positions

"""User-run independent output audit. Never imports/reruns the classification pipeline."""
from pathlib import Path
import argparse
import json
import hashlib
import pandas as pd


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(output):
    output = Path(output).resolve()
    manifest = json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'COMPLETED':
        raise AssertionError('Only COMPLETED output may be presented as completed analysis')
    source = Path(manifest['source'])
    original = pd.read_csv(source/'event_classification.csv')
    final = pd.read_csv(output/'event_observational_classification.csv')
    keys = ['event_id', 'partition', 'anchor_row', 'anchor_DateTime']
    pd.testing.assert_frame_equal(original[keys].sort_values('event_id').reset_index(drop=True),
                                  final[keys].sort_values('event_id').reset_index(drop=True), check_dtype=False)
    assert len(final) == 15578 and not final.event_id.duplicated().any()
    assert final.physical_cause_status.eq('UNDETERMINED').all()
    freeze = json.loads((output/'threshold_freeze.json').read_text(encoding='utf-8'))
    assert file_sha(output/'frozen_train_rules.json') == freeze['sha256']
    rules = json.loads((output/'frozen_train_rules.json').read_text(encoding='utf-8'))
    assert rules['fitted_partition'] == 'TRAIN'
    controls = pd.read_csv(output/'train_control_index.csv')
    assert controls.empty or controls.partition.eq('TRAIN').all()
    assert file_sha(output/'train_control_index.csv') == freeze['train_control_index_sha256']
    counts = pd.Series(dtype='int64'); rows = 0; seen = {}
    variables = ('actual_current', 'set_current', 'internal_set_current', 'coolant_in', 'coolant_out',
                 'anode_pressure', 'cathode_pressure', 'hydrogen_flow', 'air_flow')
    phases = ('before', 'instant', 'after')
    for frame in pd.read_csv(output/'variable_associations.csv', chunksize=30000):
        assert set(frame.phase.unique()) <= {'before', 'instant', 'after'}
        assert (frame.anchor_row == frame.anchor_row.astype('int64')).all()
        group = frame.groupby('event_id').size()
        counts = counts.add(group, fill_value=0)
        for item in frame[['event_id', 'variable', 'phase']].itertuples(index=False):
            bit = 1 << (variables.index(item.variable)*3+phases.index(item.phase))
            assert not (seen.get(item.event_id, 0) & bit), 'Duplicated event-variable-phase key'
            seen[item.event_id] = seen.get(item.event_id, 0) | bit
        rows += len(frame)
    assert rows == 15578*9*3 and counts.eq(27).all()
    assert set(seen) == set(final.event_id) and all(mask == (1 << 27)-1 for mask in seen.values())
    for filename in ('source_protection_before.json', 'source_protection_after.json'):
        assert json.loads((output/filename).read_text(encoding='utf-8'))['all_unchanged']
    for item in json.loads((output/'output_sha256.json').read_text(encoding='utf-8')):
        assert file_sha(output/item['file']) == item['sha256'], item['file']
    report = dict(status='PASS', events=len(final), variable_phase_rows=rows,
                  verification='independent output identity, frozen Train rules, counts, file hashes; no classification rerun')
    (output/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='核验用户已完成的分类细化输出；不运行分析')
    parser.add_argument('output_directory')
    verify(parser.parse_args().output_directory)

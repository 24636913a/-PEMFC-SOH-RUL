"""Bounded Train access and mathematical audit utilities. No historical imports."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = '293242599cf3b385d6e52d54f35f482f12059489'
RESEARCH = 'RENew-PEMFC-2300h-Pytorch'
N1 = RESEARCH + '/Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
RAW14 = ['cell_voltage_mean', 'cell_voltage_std', 'cell_voltage_range', 'cell_voltage_min', 'cell_voltage_max',
         'actual_current_A_mean', 'actual_current_A_std', 'actual_current_A_peak_to_peak',
         'actual_current_density_A_cm2_mean', 'actual_current_density_A_cm2_std',
         'actual_current_density_A_cm2_peak_to_peak', 'load_transition_count',
         'operation_point_transitions', 'zero_current_fraction']
AUX_NONVOLTAGE = RAW14[5:]
CONDITIONS = ['actual_anode_inlet_temperature_C_mean', 'actual_anode_outlet_temperature_C_mean',
 'actual_cathode_inlet_temperature_C_mean', 'actual_cathode_outlet_temperature_C_mean',
 'actual_anode_inlet_rh_pct_mean', 'actual_cathode_inlet_rh_pct_mean',
 'actual_anode_inlet_pressure_kPag_mean', 'actual_anode_outlet_pressure_kPag_mean',
 'actual_cathode_inlet_pressure_kPag_mean', 'actual_cathode_outlet_pressure_kPag_mean',
 'actual_anode_hydrogen_flow_nl_min_mean', 'actual_cathode_air_flow_nl_min_mean',
 'actual_anode_stoichiometry_mean', 'actual_cathode_stoichiometry_mean']
METADATA = ['resolution_hours', 'bin_start_ophrs2_h', 'bin_end_ophrs2_h',
 'first_datetime', 'last_datetime', 'coverage_ratio', 'raw_sample_count',
 'bin_valid_for_windows', 'temporal_segment_id', 'gap_before_flag',
 'startup_flag', 'shutdown_flag', 'internal_datetime_gap_count',
 'number_of_operation_points', 'operation_point_mode', 'source_file_names']
SAFE_COLUMNS = list(dict.fromkeys(METADATA + RAW14 + CONDITIONS))

class AuditFailure(RuntimeError):
    pass

def require(ok, message):
    if not ok:
        raise AuditFailure(message)

def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()

def metadata_hash(repo, relative, allowlist):
    """Whole-file hashes only for explicitly frozen NON-data metadata/code/weights."""
    require(relative in allowlist and relative != N1, 'FULL_HASH_OR_UNALLOWLISTED_ACCESS_DENIED')
    path = (repo / relative).resolve()
    require(path.is_relative_to(repo.resolve()), 'SOURCE_ESCAPE_DENIED')
    require(path.is_file(), 'MISSING_SOURCE:' + relative)
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()

def record_bytes(stream):
    """Only read(1), no buffered reader or read-ahead after terminating LF.

    Quote parity handles CSV escaped quotes and quoted embedded newlines. A final
    record without LF is rejected; accepting it would require an extra EOF probe.
    The in-memory csv parser subsequently validates field structure strictly.
    """
    buf = bytearray()
    quoted = False
    while True:
        byte = stream.read(1)
        require(bool(byte), 'MISSING_TERMINATED_CSV_RECORD')
        buf.extend(byte)
        require(len(buf) <= 1048576, 'OVERSIZED_CSV_RECORD')
        if byte == b'"':
            quoted = not quoted
        if byte == b'\n' and not quoted:
            return bytes(buf)

def bounded_csv_stream(stream, count):
    require(isinstance(count, int) and count > 0, 'INVALID_RECORD_BUDGET')
    records = [record_bytes(stream)]
    for _ in range(count):
        records.append(record_bytes(stream))
    payload = b''.join(records)
    rows = list(csv.reader(io.StringIO(payload.decode('utf-8-sig')), strict=True))
    require(len(rows) == count + 1, 'CSV_LOGICAL_RECORD_COUNT_MISMATCH')
    header = rows[0]
    require(len(header) == len(set(header)), 'DUPLICATE_COLUMN')
    require(all(len(row) == len(header) for row in rows[1:]), 'CSV_SCHEMA_MISMATCH')
    return header, rows[1:], len(payload)

def load_train(repo, access):
    require(access['mixed_source'] == N1 and access['record_count'] == 2103,
            'TRAIN_ACCESS_CONFIGURATION_MISMATCH')
    path = (repo / N1).resolve()
    require(path.is_relative_to(repo.resolve()), 'SOURCE_ESCAPE_DENIED')
    # FileIO is unbuffered. pandas never sees the physical path.
    with path.open('rb', buffering=0) as stream:
        header, records, consumed = bounded_csv_stream(stream, 2103)
        require(stream.tell() == consumed, 'READ_AHEAD_DETECTED')
    require(set(SAFE_COLUMNS).issubset(header), 'REQUIRED_SOURCE_COLUMN_MISSING')
    frame = pd.DataFrame(records, columns=header)[SAFE_COLUMNS].copy()
    strings = {'first_datetime', 'last_datetime', 'source_file_names'}
    for col in frame.columns:
        if col not in strings:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
    boundary = validate_train_identity(frame, access)
    payload = frame.to_csv(index=False, lineterminator='\n', float_format='%.17g', na_rep='<NA>').encode('utf-8')
    boundary.update(logical_records_read=2103, bytes_consumed=consumed,
                    allowed_train_fingerprint_sha256=digest_bytes(payload),
                    fingerprint_columns=SAFE_COLUMNS,
                    serialization='UTF-8, LF, index=false, header=true, %.17g, <NA>, fixed SAFE_COLUMNS',
                    full_source_hash=False, next_record_probe=False,
                    physical_read_method='FileIO buffering=0; read(1); exactly header+2103 CSV records')
    return frame, boundary

def validate_train_identity(frame, access):
    require(len(frame) == 2103, 'TRAIN_ROW_COUNT_MISMATCH')
    start = frame.bin_start_ophrs2_h.to_numpy(float)
    end = frame.bin_end_ophrs2_h.to_numpy(float)
    require(np.isfinite(start).all() and np.isfinite(end).all(), 'TRAIN_AGE_MISSING')
    require(np.all(np.diff(start) > 0), 'TRAIN_ORDER_OR_DUPLICATE_MISMATCH')
    require(np.allclose(end-start, .5, atol=1e-9, rtol=0), 'BIN_WIDTH_MISMATCH')
    require(np.all(frame.resolution_hours == .5), 'RESOLUTION_MISMATCH')
    require(start[0] == access['train_start_h'] and end[-1] == access['train_end_h'],
            'FROZEN_BOUNDARY_RECORD_MISMATCH')
    require(np.all(end <= 1310) and np.all(end < 1819.997738), 'TRAIN_TIME_BOUNDARY_BREACH')
    segment = frame.temporal_segment_id.to_numpy(float)
    require(np.isfinite(segment).all() and np.all(np.diff(segment) >= 0), 'SEGMENT_IDENTITY_MISMATCH')
    require(np.all(segment == segment.astype(int)), 'SEGMENT_NOT_INTEGER')
    # Original loader keeps age<1819.997738, then reset_index, with no valid-row
    # removal. N1 is globally sorted before export; every consumed row survives.
    require(np.all(end < 1819.997738), 'PREFIX_FILTER_NOT_IDENTITY')
    valid = frame.bin_valid_for_windows.to_numpy(float) == 1
    features = frame[RAW14].to_numpy(float)
    legal = 0
    for origin in range(47, 2103-24):
        block = slice(origin-47, origin+25)
        if (valid[block].all() and np.all(segment[block] == segment[origin])
            and np.allclose(np.diff(end[block]), .5, atol=1e-6, rtol=0)
            and np.isfinite(features[block]).all()):
            legal += 1
    require(legal == 1380, 'F2_TRAIN_WINDOW_IDENTITY_MISMATCH')
    return dict(pass_=True, f2_rows=[0,2103], physical_csv_data_records=[0,2103],
                source_prefix_to_f2_train='identity: no rejected prefix records',
                start_h=float(start[0]), end_h=float(end[-1]), legal_f2_train_windows=legal,
                segments_preserved=True, segments_recalculated=False,
                validation_read=False, assessment_read=False, test_read=False)

def no_voltage_view(frame):
    """Independent candidate auxiliary view, never the unchanged F2 input view.

    No voltage at ANY time is exposed, including EWMA/neighbor descendants. No
    age, cycle/curve IDs, scaler-derived voltage references, or cached latent.
    """
    require(set(AUX_NONVOLTAGE).issubset(frame.columns), 'AUX_SCHEMA_MISMATCH')
    return frame[AUX_NONVOLTAGE].copy()

def order_loss(low, high):
    return np.maximum(0, np.asarray(high)-np.asarray(low))**2

def scientific_pair_gate(raw_dwell_verified, state_comparable, condition_support, amplitude_independent):
    return bool(raw_dwell_verified and state_comparable and condition_support and amplitude_independent)

def eligible_metadata(frame):
    """Necessary bin flags only. This function DOES NOT certify any raw dwell."""
    stages = [
      ('coverage_ge_0p8', frame.coverage_ratio.ge(.8)),
      ('bin_valid', frame.bin_valid_for_windows.eq(1)),
      ('positive_current_no_zero', frame.actual_current_density_A_cm2_mean.gt(0) & frame.zero_current_fraction.eq(0)),
      ('no_load_or_operation_transition', frame.load_transition_count.eq(0) & frame.operation_point_transitions.eq(0)),
      ('one_operation_point', frame.number_of_operation_points.eq(1)),
      ('no_startup_shutdown_internal_gap', frame.startup_flag.eq(0) & frame.shutdown_flag.eq(0) & frame.internal_datetime_gap_count.eq(0)),
      ('not_segment_return_bin', frame.gap_before_flag.eq(0)),
      ('all_14_condition_variables_finite', pd.Series(np.isfinite(frame[CONDITIONS].to_numpy(float)).all(axis=1), index=frame.index)),
    ]
    mask = pd.Series(True, index=frame.index)
    audit = []
    for name, criterion in stages:
        before = int(mask.sum())
        mask &= criterion
        audit.append(dict(stage=name, denominator_before=before, exclusions=before-int(mask.sum()),
                          remaining=int(mask.sum()), evidence_level='BIN_METADATA_ONLY_NOT_DWELL'))
    return mask, audit

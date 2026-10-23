"""Window-local features: no 29-row external warmup and no future instruments."""
import numpy as np
import pandas as pd

FEATURE_NAMES = (
    'mean10_V', 'std10_V', 'range10_V', 'min10_V', 'max10_V',
    'actual_current_A', 'current_past30_std_A', 'current_past30_ptp_A',
    'current_density_A_cm2', 'density_past30_std_A_cm2',
    'density_past30_ptp_A_cm2', 'past1s_load_change',
    'past1s_operation_point_change', 'zero_current_flag', 'set_current_A',
    'internal_set_current_A', 'anode_inlet_T_C', 'cathode_inlet_T_C',
    'coolant_inlet_T_C', 'coolant_outlet_T_C', 'anode_inlet_p_kPag',
    'cathode_inlet_p_kPag', 'operation_point', 'OpHrs2_h',
)
GATE_NAMES = ('past1s_load_change', 'zero_current_flag',
              'density_past30_std_A_cm2', 'current_past30_std_A',
              'current_density_A_cm2')
# Required voltage state and native gate context; remaining FOUR slots are
# selected by Train-only Spearman. This is NOT the old 30-minute Top10 mask.
REQUIRED = ('mean10_V', *GATE_NAMES)
EXTRA_FIELDS = ('I.S.set [A]', 'I.S.set.internal [A]', 'T.Si.A [°C]',
                'T.Si.C [°C]', 'T.Si.CL [°C]', 'T.So.CL [°C]',
                'p.Si.A [kPag]', 'p.Si.C [kPag]', 'operation_point', 'OpHrs2 [h]')
CELLS = tuple(f'U{i} [V]' for i in range(1, 12) if i != 6)


def numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    text = series.astype('string').str.strip().str.replace('\ufeff', '', regex=False)
    both = text.str.contains(',', regex=False, na=False) & text.str.contains('.', regex=False, na=False)
    text = text.mask(both, text.str.replace(',', '', regex=False))
    return pd.to_numeric(text.str.replace(',', '.', regex=False), errors='coerce').to_numpy(dtype=float, na_value=np.nan)


def raw_base(frame):
    cells = np.column_stack([numeric(frame[c]) for c in CELLS])
    x = np.zeros((len(frame), 24), dtype=np.float32)
    x[:, :5] = np.column_stack((cells.mean(1), cells.std(1), np.ptp(cells, axis=1), cells.min(1), cells.max(1)))
    x[:, 5] = numeric(frame['I.S.act [A]'])
    x[:, 8] = numeric(frame['j.S.act [A/cm2]'])
    x[:, 13] = np.abs(x[:, 5]) <= .1
    x[:, 14:] = np.column_stack([numeric(frame[c]) for c in EXTRA_FIELDS])
    # No voltage-magnitude or jump filter: legitimate extreme observations stay.
    valid = np.isfinite(cells).all(1) & np.isfinite(x).all(1)
    return x, cells.mean(1), valid


def window_features(base):
    """First observation initializes every statistic; only supplied 300 rows."""
    x = np.array(base, dtype=np.float64, copy=True)
    n = len(x)
    ends = np.arange(1, n + 1)
    starts = np.maximum(0, ends - 30)
    counts = ends - starts
    for source, std_slot, ptp_slot in ((5, 6, 7), (8, 9, 10)):
        a = x[:, source]
        s = np.r_[0.0, np.cumsum(a)]
        q = np.r_[0.0, np.cumsum(a * a)]
        mean = (s[ends] - s[starts]) / counts
        x[:, std_slot] = np.sqrt(np.maximum(0.0, (q[ends] - q[starts]) / counts - mean * mean))
        x[:, ptp_slot] = [np.ptp(a[left:right]) for left, right in zip(starts, ends)]
    x[:, 11] = np.r_[0.0, np.abs(np.diff(x[:, 5])) > .5]
    x[:, 12] = np.r_[0.0, np.diff(x[:, 22]) != 0]
    return x


def fit_preprocessing(cache, train_origins, cfg, run):
    """Fixed evenly spaced Train origins; no error-based sampling or tuning.

    Spearman: rho(mean of each feature over 300 history rows,
                  mean of raw mean10 voltage over the next 120 rows).
    Constant features rank last; ties resolved by candidate schema order.
    Mandatory native-gate fields remain even if redundant; otherwise remove
    perfect rank duplicates against already selected features (Train only).
    """
    count = min(cfg['preprocess_origins'], len(train_origins))
    subset = train_origins[np.unique(np.linspace(0, len(train_origins) - 1, count).astype(np.int64))]
    summaries, responses = [], []
    sums = np.zeros(24); squares = np.zeros(24); rows = 0
    ysum = ysq = 0.0; yn = 0
    for index, origin in enumerate(subset):
        x = window_features(cache['x'][origin - cfg['lookback'] + 1:origin + 1])
        y = np.asarray(cache['voltage'][origin + 1:origin + 1 + cfg['horizon']], dtype=float)
        summaries.append(x.mean(0)); responses.append(y.mean())
        sums += x.sum(0); squares += (x * x).sum(0); rows += len(x)
        if cfg['target_mode'] == 'DeltaV':
            y = y - float(cache['voltage'][origin])
        ysum += y.sum(); ysq += np.dot(y, y); yn += y.size
        if index % 1000 == 0:
            print(f'仅Train特征与标准化拟合 {index + 1}/{len(subset)}', flush=True)
    table = pd.DataFrame(summaries, columns=FEATURE_NAMES)
    response = pd.Series(responses)
    scores = table.apply(lambda column: column.corr(response, method='spearman'))
    selected = [FEATURE_NAMES.index(name) for name in REQUIRED]
    decisions = {i: 'required_native_gate_or_voltage' for i in selected}
    ranks = sorted(range(24), key=lambda i: (-abs(scores.iloc[i]) if np.isfinite(scores.iloc[i]) else float('inf'), i))
    for i in ranks:
        if i in selected:
            continue
        if not np.isfinite(scores.iloc[i]):
            decisions[i] = 'constant_or_undefined'; continue
        duplicate = any(abs(table.iloc[:, i].corr(table.iloc[:, j], method='spearman')) >= 1 - 1e-12 for j in selected)
        if duplicate:
            decisions[i] = 'perfect_rank_duplicate'; continue
        if len(selected) < cfg['feature_count']:
            selected.append(i); decisions[i] = 'spearman_selected'
        else:
            decisions[i] = 'below_fixed_budget'
    if len(selected) != cfg['feature_count']:
        raise RuntimeError('Insufficient nonconstant nonduplicate Train features; do not silently change selection budget')
    mean = sums / rows
    scale = np.sqrt(np.maximum(0.0, squares / rows - mean * mean))
    scale[scale < 1e-8] = 1.0
    ymean = ysum / yn
    yscale = max(float(np.sqrt(max(0.0, ysq / yn - ymean * ymean))), 1e-8)
    pd.DataFrame([dict(index=i, feature=name, rho=None if not np.isfinite(scores.iloc[i]) else float(scores.iloc[i]),
                       selected=i in selected, decision=decisions.get(i)) for i, name in enumerate(FEATURE_NAMES)]).to_csv(run / 'feature_selection.csv', index=False)
    return dict(mean=mean.tolist(), scale=scale.tolist(), selected_indices=selected,
                selected_names=[FEATURE_NAMES[i] for i in selected], target_mean=ymean,
                target_scale=yscale, fitting_origins=len(subset), fitting_rows=rows,
                sample_rule='evenly spaced legal TRAIN origins, np.linspace endpoint inclusive',
                spearman_definition='history300 feature mean versus future120 raw voltage mean',
                mandatory=list(REQUIRED), ewma=False, external_warmup_rows=0)

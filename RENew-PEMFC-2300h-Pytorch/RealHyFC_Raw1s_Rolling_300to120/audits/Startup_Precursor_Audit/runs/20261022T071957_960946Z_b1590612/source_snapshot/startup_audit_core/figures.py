"""Truthful 300dpi audit figures; no neural models, synthetic curves or smoothing."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data_io import SIGNALS, INDEX, col
from .events import context_indices

COLORS = dict(TRAIN='#2878B5', VALIDATION='#55A868', EXPLORATORY_HOLDOUT='#D95319')


def save(fig, path):
    for ax in fig.axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        if get_ipython() is not None:
            display(Image(filename=str(path)))
    except ImportError:
        pass


def overview(arrays, events, run, cfg):
    sample = np.arange(0, len(arrays['time_ns']), 60)
    frame = pd.DataFrame(dict(global_row=sample, DateTime=np.asarray(arrays['time_ns'][sample]).astype('datetime64[ns]').astype(str),
                              voltage_V=np.asarray(col(arrays, 'mean_voltage_V')[sample])))
    frame.to_csv(run/'voltage_background_plot_data.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(pd.to_datetime(frame.DateTime), frame.voltage_V, s=1, color='#888888', label='Background: fixed 60-row plotting sample')
    points = []
    for part, group in events.groupby('partition'):
        rows = group.first_row.to_numpy(dtype=np.int64)
        times = np.asarray(arrays['time_ns'][rows]).astype('datetime64[ns]')
        voltage = col(arrays, 'mean_voltage_V')[rows]
        ax.scatter(times, voltage, s=8, color=COLORS[part], label=f'{part}: {len(rows)} episodes')
        points.extend(dict(event_id=int(event_id), DateTime=str(t), voltage_V=float(v), partition=part)
                      for event_id, t, v in zip(group.event_id, times, voltage))
    pd.DataFrame(points).to_csv(run/'voltage_event_plot_data.csv', index=False)
    ax.set(xlabel='Original DateTime', ylabel='Mean cell voltage (V)', title='All detected voltage episodes | retrospective CPU audit')
    ax.legend(frameon=False, fontsize=8); fig.autofmt_xdate()
    save(fig, run/'voltage_event_overview.png')


def examples(arrays, classification, run, cfg):
    chosen = []; decisions = []
    for part in COLORS:
        subset = classification.loc[(classification.partition == part) & classification.C_low_zero_voltage_recovery_proxy.fillna(False).astype(bool)]
        for event in subset.itertuples(index=False):
            supported = event.context_before_seconds >= 120 and event.context_after_seconds >= 120
            selected = supported and sum(p == part for p, _ in chosen) < cfg['plot_examples_per_partition']
            decisions.append(dict(event_id=event.event_id, partition=part, selected=selected,
                                  reason='first chronological qualified recovery; never chosen by signal strength' if selected else 'context incomplete or fixed example quota reached'))
            if selected:
                chosen.append((part, event))
    pd.DataFrame(decisions).to_csv(run/'example_selection.csv', index=False)
    fig, axes = plt.subplots(max(1, len(chosen)), 4, figsize=(17, max(3.5, 2.6*len(chosen))), squeeze=False)
    if not chosen:
        for ax in axes.flat:
            ax.text(.5, .5, 'No qualified recovery example', ha='center', transform=ax.transAxes)
        save(fig, run/'startup_precursor_examples.png'); return
    plot_rows = []
    for panel, (part, event) in enumerate(chosen):
        row = int(event.anchor_row); index = context_indices(arrays, row, cfg['context_seconds'])
        if len(index) > cfg['plot_max_event_points']:
            index = index[np.unique(np.linspace(0, len(index)-1, cfg['plot_max_event_points']).astype(int))]
        elapsed = (arrays['time_ns'][index]-int(arrays['time_ns'][row]))/1e9
        fields = [('mean_voltage_V',), ('I.S.act [A]', 'I.S.set [A]', 'I.S.set.internal [A]'),
                  ('T.Si.CL [°C]', 'T.So.CL [°C]'), ('p.Si.A [kPag]', 'p.Si.C [kPag]')]
        for j, names in enumerate(fields):
            ax = axes[panel, j]
            for field in names:
                values = np.asarray(col(arrays, field)[index])
                ax.plot(elapsed, values, label=field, linewidth=1)
                plot_rows.extend(dict(event_id=event.event_id, partition=part, global_row=int(r),
                                      elapsed_seconds=float(t), variable=field, value=float(v)) for r, t, v in zip(index, elapsed, values))
            ax.axvline(0, color='black', linestyle='--', linewidth=.7)
            ax.set(xlabel='Seconds relative to first positive trigger', ylabel=('V', 'A', '°C', 'kPag')[j],
                   title=f'Event {event.event_id} | {part}')
            ax.legend(fontsize=7, frameon=False)
    pd.DataFrame(plot_rows).to_csv(run/'startup_precursor_examples_plot_data.csv', index=False)
    save(fig, run/'startup_precursor_examples.png')


def leadtime(snapshots, run):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7)); records = []
    table = snapshots.loc[snapshots.low_zero_recovery_proxy.fillna(False).astype(bool)]
    for ax, (signal, field) in zip(axes.flat, SIGNALS.items()):
        for part in COLORS:
            subset = table[(table.signal == signal) & (table.partition == part)]
            baseline = subset[subset.seconds_before == 120].set_index('event_id').value
            local = []
            for offset, group in subset.groupby('seconds_before'):
                paired = group.set_index('event_id').value-baseline
                values = paired.dropna().to_numpy()
                if len(values):
                    row = dict(signal=signal, partition=part, seconds_before=int(offset), n=len(values),
                               median=float(np.median(values)), q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)))
                    records.append(row); local.append(row)
            if local:
                frame = pd.DataFrame(local).sort_values('seconds_before', ascending=False)
                x = -frame.seconds_before.to_numpy()
                ax.plot(x, frame['median'], marker='o', color=COLORS[part], label=part)
                ax.fill_between(x, frame.q25.to_numpy(float), frame.q75.to_numpy(float), color=COLORS[part], alpha=.15)
        ax.set(xlabel='Seconds relative to first positive trigger', ylabel='Change from -120s ('+('kPag' if signal.endswith('_p') else '°C')+')',
               title=f'{field} | median and IQR, low/zero recovery proxies')
        if ax.lines:
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.text(.5, .5, 'No paired qualified observations', ha='center', transform=ax.transAxes)
    pd.DataFrame(records).to_csv(run/'temperature_pressure_leadtime_plot_data.csv', index=False)
    save(fig, run/'temperature_pressure_leadtime.png')


def lags(timeline, run):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); records = []
    qualified = timeline.loc[timeline.comparable.fillna(False).astype(bool)]
    for ax, column in zip(axes, ('command_lag_seconds', 'actual_load_lag_seconds')):
        for group, table in qualified.groupby('stop_group'):
            values = table[column].dropna().to_numpy()
            if len(values):
                ax.hist(values, bins=40, alpha=.5, label=f'{group}: n={len(values)}')
                records.extend(dict(zero_command_spell_id=int(row.zero_command_spell_id), stop_group=group,
                                    lag_type=column, lag_seconds=float(getattr(row, column)))
                               for row in table.itertuples(index=False) if pd.notna(getattr(row, column)))
        ax.axvline(65, color='black', linestyle='--', label='65 s: descriptive reference only')
        ax.set(xlabel='Seconds after voltage recovery', ylabel='Observed command-zero spells', title=column)
        ax.legend(frameon=False, fontsize=8)
    pd.DataFrame(records).to_csv(run/'restart_lag_plot_data.csv', index=False)
    save(fig, run/'restart_lag_distribution.png')


def generate(arrays, events, classification, snapshots, timeline, run, cfg):
    overview(arrays, events, run, cfg)
    examples(arrays, classification, run, cfg)
    leadtime(snapshots, run)
    lags(timeline, run)

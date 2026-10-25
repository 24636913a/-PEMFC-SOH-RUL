"""只读原始来源、文件身份与数值统计；无模型、滤波或插值。"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

CELLS_REAL = [f'U{i} [V]' for i in range(1, 12) if i != 6]
CELLS_FC1 = [f'U{i} (V)' for i in range(1, 6)]
REAL_FIELDS = ['DateTime', 'OpHrs [h]', 'OpHrs2 [h]', 'CalHrs [h]',
               'Number of cells', 'I.S.act [A]', 'I.S.set [A]', 'j.S.act [A/cm2]',
               'Membrane active area [cm2]', 'U.S.AveCell [V]', 'U6 [V]',
               'T.Si.CL [°C]', 'T.So.CL [°C]', 'T.Si.A [°C]', 'T.Si.C [°C]',
               'p.Si.A [kPag]', 'p.Si.C [kPag]', 'F.Si.H.A [nl/min]',
               'F.Si.Air.C [nl/min]', 'RH.Si.A  [%]', 'RH.Si.C [%]', 'operation_point', *CELLS_REAL]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def table(path, records, columns=None):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records, columns=columns)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def numeric(series):
    """明确区分美式千位分组和单独逗号小数；含糊格式保留NA。"""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').to_numpy(float)
    text = series.astype('string').str.strip().str.replace('\ufeff', '', regex=False)
    grouped = text.str.fullmatch(r'[+-]?\d{1,3}(,\d{3})+(\.\d+)?', na=False)
    text = text.mask(grouped, text.str.replace(',', '', regex=False))
    comma = text.str.contains(',', regex=False, na=False) & ~text.str.contains('.', regex=False, na=False)
    text = text.mask(comma, text.str.replace(',', '.', regex=False))
    return pd.to_numeric(text, errors='coerce').to_numpy(float, na_value=np.nan)


def col(frame, name):
    return numeric(frame[name]) if name in frame else np.full(len(frame), np.nan)


def mean_cells(frame, names):
    missing = set(names)-set(frame.columns)
    if missing:
        raise ValueError('缺少指定单片通道：'+str(sorted(missing)))
    values = np.column_stack([numeric(frame[name]) for name in names])
    # 不用nanmean掩盖某一个单片缺测。
    return values.mean(axis=1), values


class Stat:
    def __init__(self):
        self.n = 0; self.missing = 0; self.total = 0.0
        self.minimum = np.inf; self.maximum = -np.inf; self.max_abs = 0.0

    def add(self, values):
        values = np.asarray(values, float); valid = values[np.isfinite(values)]
        self.missing += len(values)-len(valid)
        if len(valid):
            self.n += len(valid); self.total += float(valid.sum())
            self.minimum = min(self.minimum, float(valid.min())); self.maximum = max(self.maximum, float(valid.max()))
            self.max_abs = max(self.max_abs, float(np.abs(valid).max()))

    def record(self):
        return clean(dict(finite=self.n, missing=self.missing,
                          minimum=self.minimum if self.n else np.nan,
                          maximum=self.maximum if self.n else np.nan,
                          mean=self.total/self.n if self.n else np.nan,
                          max_abs=self.max_abs if self.n else np.nan))


class Registry:
    def __init__(self, root, logger):
        self.root = root; self.rows = []; self.logger = logger

    def start(self, path, stack, kind):
        path = path.resolve(); stat = path.stat()
        with path.open('rb') as stream:
            if stream.read(120).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise RuntimeError('仅有Git LFS指针，没有实际数据载荷：'+str(path))
        digest = sha(path); rel = path.relative_to(self.root).as_posix()
        lfs = 'NOT_TRACKED_OR_GIT_UNAVAILABLE'; expected = None
        result = subprocess.run(['git', 'cat-file', '-s', 'HEAD:'+rel], cwd=self.root, capture_output=True, text=True)
        if result.returncode == 0:
            lfs = 'GIT_REGULAR_PAYLOAD'
            if int(result.stdout.strip()) <= 512:
                data = subprocess.check_output(['git', 'show', 'HEAD:'+rel], cwd=self.root).decode('utf-8')
                if data.startswith('version https://git-lfs.github.com/spec/v1'):
                    expected = next(x.split(':', 1)[1] for x in data.splitlines() if x.startswith('oid sha256:'))
                    lfs = 'LFS_PAYLOAD_SHA_VERIFIED' if digest == expected else 'LFS_SHA_MISMATCH'
        item = dict(电堆=stack, 数据类型=kind, 文件名=path.name, 原始路径=rel,
                    字节数=stat.st_size, SHA256=digest, Git_LFS状态=lfs, LFS_SHA256=expected,
                    读取成功=False, 实际行数=0, 异常原因='', _mtime_ns=stat.st_mtime_ns)
        self.rows.append(item)
        if lfs == 'LFS_SHA_MISMATCH':
            raise RuntimeError('Git LFS载荷SHA不符：'+rel)
        self.logger('读取 '+stack+' / '+path.name)
        return item

    def finish(self, item, path, rows, fields, **metadata):
        after = path.stat()
        if after.st_size != item['字节数'] or after.st_mtime_ns != item['_mtime_ns']:
            raise RuntimeError('读取过程中源文件被外部修改，停止：'+str(path))
        item.update(读取成功=True, 实际行数=int(rows), 关键字段=';'.join(fields), **metadata)

    def public(self):
        return [{k: v for k, v in item.items() if not k.startswith('_')} for item in self.rows]


def read_polar(path, stack):
    if stack == 'FC1':
        return pd.read_excel(path, engine='openpyxl'), 'Excel OOXML/openpyxl'
    return pd.read_csv(path, sep=';', encoding='utf-8-sig', usecols=lambda c: c in REAL_FIELDS,
                       low_memory=False), 'UTF-8-SIG; semicolon; explicit numeric parsing'


def scan_directions(j, excursion):
    """滞回确认的描述性扫描方向；不平滑j/V，不称准确物理扫描转折时刻。"""
    out = np.zeros(len(j), np.int8); segment = np.zeros(len(j), np.int32)
    low = high = None; direction = 0; sid = 0
    for i, value in enumerate(j):
        if not np.isfinite(value):
            low = high = None; direction = 0; sid += 1
        elif low is None:
            low = high = value
        elif direction == 0:
            low = min(low, value); high = max(high, value)
            if value-low >= excursion:
                direction = 1
            elif high-value >= excursion:
                direction = -1
        elif direction > 0:
            high = max(high, value)
            if high-value >= excursion:
                direction = -1; low = value; sid += 1
        else:
            low = min(low, value)
            if value-low >= excursion:
                direction = 1; high = value; sid += 1
        out[i] = direction; segment[i] = sid
    return out, segment


def self_test():
    result = numeric(pd.Series(['1,415.16', '0,25', 'bad', '1.2,3']))
    assert result[0] == 1415.16 and result[1] == 0.25 and np.isnan(result[2:]).all()
    values = pd.DataFrame({n: [1.0, np.nan if i == 0 else 1.0] for i, n in enumerate(CELLS_FC1)})
    mean, _ = mean_cells(values, CELLS_FC1)
    assert mean[0] == 1.0 and np.isnan(mean[1])
    direction, _ = scan_directions(np.array([1.0, 0.99, 0.90, 0.2, 0.3, 0.4]), 0.05)
    assert direction[2] == -1 and direction[-1] == 1
    return dict(status='PASS', tests=['locale_numeric', 'missing_cell_not_hidden', 'scan_direction_hysteresis'],
                model_training=False, prediction_metrics=False)

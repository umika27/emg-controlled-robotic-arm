"""Strict ingestion: retain repeated timestamps, reject decreasing timestamps."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import stat
import warnings
import zipfile
import numpy as np
import pandas as pd
from .sampling import analyze_sampling

CHANNELS = [f'channel{i}' for i in range(1, 9)]
COLUMNS = ['time', *CHANNELS, 'class']


@dataclass
class Recording:
    subject_id: str
    recording_id: str
    timestamps: np.ndarray
    samples: np.ndarray
    labels: np.ndarray
    duplicate_rows: int = 0
    missing_labels: int = 0


def safe_extract(archive, destination):
    root = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            target = (root / info.filename).resolve()
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename or not target.is_relative_to(root):
                raise ValueError(f'Unsafe ZIP path: {info.filename}')
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('ZIP symlinks are forbidden')
        z.extractall(root)


def discover(raw_dir='data/raw', project_root='.'):
    root = Path(raw_dir)
    files = sorted(root.glob('**/*raw*.txt'))
    if not files:
        archives = sorted(Path(project_root).glob('*.zip'))
        if len(archives) != 1:
            raise FileNotFoundError('Expected one dataset ZIP in project root or extracted data in raw_dir.')
        safe_extract(archives[0], root)
        files = sorted(root.glob('**/*raw*.txt'))
    if not files:
        raise FileNotFoundError('No raw recordings found.')
    return files


def read_recording(path, missing_label='error'):
    path = Path(path)
    try:
        frame = pd.read_csv(path, sep=r'\s+')
        if list(frame.columns) != COLUMNS:
            raise ValueError(f'Expected columns {COLUMNS}, got {list(frame.columns)}')
        frame = frame.apply(pd.to_numeric, errors='raise')
        missing = int(frame['class'].isna().sum())
        if missing and missing_label == 'unmarked':
            warnings.warn(f'{path}: {missing} missing label(s) retained as unmarked; source unchanged.')
            frame['class'] = frame['class'].fillna(0)
        elif missing_label not in ('error', 'unmarked'):
            raise ValueError('missing_label policy must be error or unmarked.')
        array = frame.to_numpy(dtype=float)
        if len(array) < 2 or not np.isfinite(array).all():
            raise ValueError('Recording needs at least two finite rows; missing data are rejected.')
        if (np.diff(array[:, 0]) < 0).any() or array[-1, 0] <= array[0, 0]:
            raise ValueError('Timestamps must be nondecreasing with positive duration.')
        y = array[:, -1]
        if not np.isin(y, np.arange(8)).all():
            raise ValueError('Labels must be integers from 0 through 7.')
        return Recording(path.parent.name, path.stem, array[:, 0], array[:, 1:9], y.astype(int), int(frame.duplicated().sum()), missing)
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f'{path}: {exc}') from exc


def inventory(files, missing_label='error'):
    rows, counts, summaries = [], [], []
    for path in files:
        r = read_recording(path, missing_label=missing_label)
        rows.append(dict(subject_id=r.subject_id, recording_id=r.recording_id,
                         rows=len(r.labels), duration_s=(r.timestamps[-1]-r.timestamps[0])/1000,
                         duplicate_rows=r.duplicate_rows, missing_labels=r.missing_labels, **analyze_sampling(r.timestamps, r.samples)))
        for label, n in zip(*np.unique(r.labels, return_counts=True)):
            counts.append(dict(subject_id=r.subject_id, recording_id=r.recording_id, label=int(label), count=int(n)))
        for i, channel in enumerate(CHANNELS):
            x = r.samples[:, i]
            summaries.append(dict(subject_id=r.subject_id, recording_id=r.recording_id, channel=channel,
                                  mean=x.mean(), std=x.std(), min=x.min(), max=x.max()))
    return pd.DataFrame(rows), pd.DataFrame(counts), pd.DataFrame(summaries)

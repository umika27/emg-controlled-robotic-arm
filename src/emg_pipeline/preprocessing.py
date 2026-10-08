from pathlib import Path
from time import perf_counter
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from .dataset import discover, read_recording
from .labels import validate_labels
from .sampling import resample_recording
from .filtering import design_filter, filter_valid_segments
from .windows import make_windows
from .splitting import split_groups


def prepare(config):
    start = perf_counter()
    fs = config['sampling']['target_hz']
    labels = validate_labels(config['labels'])
    sos = design_filter(fs, **config['filter'])
    arrays, frames, selected = [], [], []
    present_subjects = {p.parent.name for p in discover(config['raw_dir'])}
    missing = set(config.get('subjects') or []) - present_subjects
    if missing:
        raise ValueError(f'Requested subjects not found: {sorted(missing)}')
    for path in tqdm(discover(config['raw_dir']), desc='Preparing recordings'):
        if config.get('subjects') is not None and path.parent.name not in config['subjects']:
            continue
        r = read_recording(path, missing_label=config.get('missing_label', 'error'))
        selected.append(f'{r.subject_id}/{r.recording_id}')
        t, x, y, valid = resample_recording(r, fs, config['sampling']['max_gap_ms'], return_valid=True)
        x = filter_valid_segments(x, valid, sos, config['filter']['mode'])
        a, m = make_windows(x, t, y, r.subject_id, r.recording_id, fs, labels, **config['window'])
        if not len(a):
            raise ValueError(f'{path} produced no windows; revise labels or window duration.')
        arrays.append(a)
        frames.append(m)
    if not arrays:
        raise ValueError('No recordings selected. Subject IDs must use two digits.')
    x, meta = np.concatenate(arrays), pd.concat(frames, ignore_index=True)
    if set(labels)-set(meta.label):
        raise ValueError(f'Requested classes unavailable: {set(labels)-set(meta.label)}')
    masks, assignment = split_groups(meta, seed=config['seed'], **config['split'])
    meta['split'] = ''
    for key, mask in masks.items():
        meta.loc[mask, 'split'] = key
    out = Path(config['processed_dir']); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'windows.npz', windows=x)
    meta.to_csv(out/'windows.csv', index=False)
    manifest = dict(config=config, split_groups=assignment, selected_recordings=selected,
                    windows=len(x), preprocessing_seconds=perf_counter()-start)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return x, meta, manifest


def load_prepared(config):
    out = Path(config['processed_dir'])
    manifest = json.loads((out/'manifest.json').read_text())
    keys = ['seed', 'labels', 'subjects', 'sampling', 'filter', 'window', 'split', 'raw_dir', 'missing_label']
    if any(manifest['config'].get(k) != config.get(k) for k in keys):
        raise ValueError('Prepared data configuration differs. Run prepare-data again.')
    return np.load(out/'windows.npz')['windows'], pd.read_csv(out/'windows.csv', dtype={'subject_id': str}), manifest

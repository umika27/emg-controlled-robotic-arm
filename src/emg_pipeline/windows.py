import numpy as np
import pandas as pd


def window_sizes(fs, seconds=.2, overlap=.5):
    if fs <= 0 or seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError('Invalid window duration, rate, or overlap.')
    n = round(fs*seconds)
    if n < 3:
        raise ValueError('Windows require at least three samples.')
    return n, max(1, round(n*(1-overlap)))


def make_windows(x, t, y, subject, recording, fs, labels, seconds=.2, overlap=.5):
    n, hop = window_sizes(fs, seconds, overlap)
    boundaries = np.r_[0, np.flatnonzero(np.diff(y) != 0)+1, len(y)]
    windows, metadata = [], []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if y[a] not in labels:
            continue
        for start in range(a, b-n+1, hop):
            windows.append(x[start:start+n].T)
            metadata.append(dict(subject_id=subject, recording_id=recording, label=int(y[a]),
                                 start_ms=float(t[start]), end_ms=float(t[start+n-1]),
                                 duration_ms=1000*n/fs, sample_span_ms=float(t[start+n-1]-t[start]), start_index=start))
    return np.asarray(windows, dtype=np.float32).reshape(-1, 8, n), pd.DataFrame(metadata)

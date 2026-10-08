"""Timestamp rates are row rates, not proof of independent sensor acquisitions."""
import numpy as np


def analyze_sampling(timestamps, samples=None):
    t = np.asarray(timestamps, float)
    d = np.diff(t)
    if len(t) < 2 or not np.isfinite(t).all() or (d < 0).any() or not (d > 0).any():
        raise ValueError('Need finite nondecreasing timestamps with positive duration.')
    p = d[d > 0]
    result = dict(empirical_row_hz=1000*(len(t)-1)/(t[-1]-t[0]),
                  median_positive_dt_ms=float(np.median(p)), repeated_timestamps=int((d == 0).sum()),
                  max_gap_ms=float(d.max()), dt_cv=float(p.std()/p.mean()))
    if samples is not None:
        same = np.all(np.diff(samples, axis=0) == 0, axis=1)
        result['repeated_vector_fraction'] = float(same.mean())
        result['vector_change_hz'] = float((~same).sum()*1000/(t[-1]-t[0]))
    return result


def resample_recording(recording, fs, max_gap_ms=50, return_valid=False):
    """Previous-value hold, no future samples. Conflicting duplicate labels become 0.

    Long timestamp gaps become unmarked. Labels are never interpolated.
    """
    if fs <= 0 or max_gap_ms <= 0:
        raise ValueError('Sampling rate and maximum gap must be positive.')
    t, x, y = recording.timestamps, recording.samples, recording.labels
    unique, first, counts = np.unique(t, return_index=True, return_counts=True)
    last = first + counts - 1
    labels = y[last].copy()
    for k in np.flatnonzero(counts > 1):
        if len(np.unique(y[first[k]:last[k]+1])) > 1:
            labels[k] = 0
    grid = np.arange(t[0], t[-1] + 1e-9, 1000/fs)
    idx = np.searchsorted(unique, grid, side='right') - 1
    out_y = labels[idx].copy()
    valid = grid - unique[idx] <= max_gap_ms
    out_y[~valid] = 0
    result = (grid, x[last[idx]], out_y)
    return (*result, valid) if return_valid else result

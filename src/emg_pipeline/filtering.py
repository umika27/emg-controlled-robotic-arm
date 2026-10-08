import warnings
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi, sosfiltfilt


def design_filter(fs, kind='bandpass', low_hz=5, high_hz=40, order=4, notch_hz=None, **_):
    if fs <= 0 or order < 1:
        raise ValueError('Positive sampling rate and filter order required.')
    if kind == 'bandpass':
        cutoff = [low_hz, high_hz]
        if low_hz >= high_hz:
            raise ValueError('Band-pass low cutoff must be below high cutoff.')
    elif kind == 'highpass':
        cutoff = low_hz
    elif kind == 'lowpass':
        cutoff = high_hz
    else:
        raise ValueError('Filter kind must be bandpass, highpass, or lowpass.')
    if np.any(np.asarray(cutoff) <= 0) or np.any(np.asarray(cutoff) >= fs/2):
        raise ValueError(f'Cutoffs must be strictly between 0 and Nyquist ({fs/2:g} Hz).')
    sos = butter(order, cutoff, btype=kind, fs=fs, output='sos')
    if notch_hz is not None:
        if not 0 < notch_hz < fs/2:
            raise ValueError('Notch must be below Nyquist.')
        b, a = iirnotch(notch_hz, 30, fs=fs)
        sos = np.vstack([sos, tf2sos(b, a)])
    return sos


class CausalFilter:
    def __init__(self, sos):
        self.sos = sos
        self.state = None

    def process(self, samples):
        x = np.asarray(samples, float)
        if x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all():
            raise ValueError('Expected finite [samples, 8] input.')
        if len(x) == 0:
            return x.copy()
        if self.state is None:
            self.state = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        out, self.state = sosfilt(self.sos, x, axis=0, zi=self.state)
        return out


def filter_signal(samples, sos, mode='causal'):
    x = np.asarray(samples, float)
    if x.ndim != 2 or x.shape[1] != 8 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite [samples, 8] input.')
    if mode == 'causal':
        return CausalFilter(sos).process(x)
    if mode != 'zero_phase':
        raise ValueError('Mode must be causal or zero_phase.')
    pad = min(len(x)-1, 3*(2*len(sos)+1))
    if pad < 3*(2*len(sos)+1):
        warnings.warn('Short recording: reduced zero-phase padding; inspect boundary transients.')
    return sosfiltfilt(sos, x, axis=0, padlen=pad)


def filter_valid_segments(samples, valid, sos, mode='causal'):
    """Reset at acquisition outages; ordinary unmarked samples remain continuous."""
    x = np.asarray(samples)
    result = np.zeros_like(x, dtype=float)
    boundaries = np.r_[0, np.flatnonzero(np.diff(valid.astype(int)) != 0)+1, len(valid)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if valid[a]:
            result[a:b] = filter_signal(x[a:b], sos, mode)
    return result

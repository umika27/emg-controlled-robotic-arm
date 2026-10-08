import numpy as np
import pandas as pd
from scipy.signal import periodogram


def extract_features(windows, fs, noise_threshold=2e-5, spectral=False):
    x = np.asarray(windows, float)
    if x.ndim != 3 or x.shape[1] != 8 or x.shape[2] < 3 or not np.isfinite(x).all() or noise_threshold < 0:
        raise ValueError('Features require finite [windows, 8, samples>=3] and nonnegative threshold.')
    d = np.diff(x, axis=2)
    values = dict(rms=np.sqrt(np.mean(x*x, axis=2)), mav=np.mean(abs(x), axis=2),
                  variance=np.var(x, axis=2, ddof=1), waveform_length=np.sum(abs(d), axis=2),
                  zero_crossings=np.sum((x[:, :, :-1]*x[:, :, 1:] < 0) & (abs(d) >= noise_threshold), axis=2),
                  slope_sign_changes=np.sum((d[:, :, :-1]*d[:, :, 1:] < 0) &
                    (np.minimum(abs(d[:, :, :-1]), abs(d[:, :, 1:])) >= noise_threshold), axis=2))
    if spectral:
        f, p = periodogram(x, fs=fs, axis=2, detrend='constant', scaling='density')
        power = p.sum(axis=2)
        values['mean_frequency_hz'] = np.divide((p*f).sum(axis=2), power, out=np.zeros_like(power), where=power > 0)
        indices = np.argmax(np.cumsum(p, axis=2) >= power[:, :, None]/2, axis=2)
        values['median_frequency_hz'] = np.where(power > 0, f[indices], 0)
        values['band_power_signal_squared'] = p.sum(axis=2)*(f[1]-f[0])
    return pd.DataFrame({f'ch{c+1}_{name}': value[:, c] for name, value in values.items() for c in range(8)})

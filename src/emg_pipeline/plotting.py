from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, sosfreqz
from .labels import LABELS


def save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def signal_plots(t, raw, filtered, y, sos, fs, directory):
    out = Path(directory)
    # A selected 10-second interval containing supervised gestures.
    marked = np.flatnonzero(y > 0)
    start = max(0, (int(marked[0]) if len(marked) else 0)-round(fs))
    stop = min(len(t), start+round(fs*10))
    fig, axes = plt.subplots(8, 1, figsize=(12, 12), sharex=True)
    for c, ax in enumerate(axes):
        ax.plot(t[start:stop]/1000, raw[start:stop, c], alpha=.65, label='raw (resampled)')
        ax.plot(t[start:stop]/1000, filtered[start:stop, c], label='filtered', lw=.7)
        ax.set_ylabel(f'ch{c+1}\nsource units')
    axes[0].legend(); axes[-1].set_xlabel('Time (s)'); axes[0].set_title('Raw and filtered EMG')
    save(fig, out/'signals.png')
    fig, ax = plt.subplots(figsize=(9, 4))
    for array, label in [(raw, 'raw'), (filtered, 'filtered')]:
        f, p = welch(array, fs, axis=0, nperseg=min(512, len(array)))
        ax.semilogy(f, np.maximum(p.mean(axis=1), 1e-30), label=label)
    ax.set(xlabel='Frequency (Hz)', ylabel='Mean channel PSD (source units²/Hz)', title='Welch PSD'); ax.legend()
    save(fig, out/'psd.png')
    f, response = sosfreqz(sos, worN=1024, fs=fs)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(f, 20*np.log10(np.maximum(abs(response), 1e-8)))
    axes[0].set(ylabel='Magnitude (dB)', ylim=(-80, 5), title='Causal SOS frequency response')
    axes[1].plot(f, np.unwrap(np.angle(response))); axes[1].set(xlabel='Frequency (Hz)', ylabel='Phase (rad)')
    save(fig, out/'filter_response.png')


def windows_plot(meta, directory):
    row = meta.iloc[0]
    selected = meta[(meta.subject_id == row.subject_id) & (meta.recording_id == row.recording_id)]
    selected = selected[selected.start_ms < row.start_ms+3000]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, item in enumerate(selected.itertuples()):
        ax.plot([item.start_ms/1000, (item.start_ms+item.duration_ms)/1000], [i, i], lw=3)
    ax.set(xlabel='Time (s)', ylabel='Window index', title='Overlapping windows within one gesture segment')
    save(fig, Path(directory)/'windows.png')


def distribution_plot(counts, path):
    values = counts.groupby('label')['count'].sum()
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar([LABELS[int(k)] for k in values.index], values.values)
    ax.set(ylabel='Raw rows (includes repeated samples)', title='Dataset label distribution'); ax.tick_params(axis='x', rotation=25)
    save(fig, path)


def feature_plot(features, meta, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, name in zip(axes.flat, ['rms', 'mav', 'variance', 'waveform_length', 'zero_crossings', 'slope_sign_changes']):
        labels = sorted(meta.label.unique())
        ax.boxplot([features.loc[(meta.label == label).to_numpy(), f'ch1_{name}'] for label in labels],
                   tick_labels=[LABELS[int(label)] for label in labels], showfliers=False)
        ax.set_title(f'Channel 1 {name.replace("_", " ")}')
        unit = 'Count/window' if name in ['zero_crossings', 'slope_sign_changes'] else ('Source units²' if name == 'variance' else 'Source units')
        ax.set_ylabel(unit); ax.tick_params(axis='x', rotation=25)
    save(fig, path)


def confusion_plot(result, path):
    fig, ax = plt.subplots(figsize=(6, 5)); matrix = np.array(result['confusion_matrix'])
    img = ax.imshow(matrix, cmap='Blues'); fig.colorbar(img, ax=ax)
    names = [LABELS.get(label, str(label)) for label in result['labels']]
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
           xlabel='Predicted gesture', ylabel='True gesture', title='Held-out confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha='center', va='center')
    save(fig, path)


def training_plot(history, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in ['train_loss', 'validation_loss']:
        ax.plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    ax.set(xlabel='Epoch', ylabel='Weighted cross entropy', title='CNN training'); ax.legend(); save(fig, path)


def prediction_plot(meta, path):
    first = meta.iloc[0]
    subset = meta[(meta.subject_id == first.subject_id) & (meta.recording_id == first.recording_id)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(subset.end_ms/1000, subset.label, s=9, label='true')
    ax.scatter(subset.end_ms/1000, subset.prediction, s=5, alpha=.6, label='predicted')
    ax.set(xlabel='Window end time (s)', ylabel='Gesture label', title=f'Held-out subject {first.subject_id}, {first.recording_id}')
    ax.legend(); save(fig, path)

import copy
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from emg_pipeline.dataset import read_recording, safe_extract, Recording, discover
from emg_pipeline.labels import validate_labels
from emg_pipeline.sampling import analyze_sampling, resample_recording
from emg_pipeline.filtering import design_filter, filter_signal, CausalFilter
from emg_pipeline.windows import make_windows
from emg_pipeline.features import extract_features
from emg_pipeline.normalization import ChannelNormalizer
from emg_pipeline.splitting import split_groups
from emg_pipeline.models.cnn import GestureCNN
from emg_pipeline.streaming import StreamingProcessor
from emg_pipeline.training import train_baseline, train_cnn
from emg_pipeline.preprocessing import prepare
from emg_pipeline.inference import Predictor
from emg_pipeline.evaluation import evaluate_artifact


def test_archive_safety(tmp_path):
    for name in ['../escape.txt', '/absolute.txt', 'nested/../../escape.txt', 'a\\evil.txt']:
        path = tmp_path/'bad.zip'
        with zipfile.ZipFile(path, 'w') as z: z.writestr(name, 'bad')
        with pytest.raises(ValueError): safe_extract(path, tmp_path/'out')
    assert not (tmp_path/'escape.txt').exists()


def test_parse(tmp_path):
    frame = pd.DataFrame(np.zeros((4, 10)), columns=['time', *[f'channel{i}' for i in range(1, 9)], 'class'])
    frame['time'] = [0, 1, 1, 2]; frame['class'] = 1
    path = tmp_path/'record.txt'; frame.to_csv(path, sep='\t', index=False)
    r = read_recording(path); assert r.duplicate_rows == 1
    for column, value in [('time', -1), ('class', 1.5), ('channel1', np.nan)]:
        bad = frame.astype(float).copy(); bad.loc[2, column] = value; bad.to_csv(path, sep='\t', index=False)
        with pytest.raises(ValueError): read_recording(path)
    frame.rename(columns={'time': 'wrong'}).to_csv(path, sep='\t', index=False)
    with pytest.raises(ValueError): read_recording(path)


@pytest.mark.parametrize('labels', [[0, 1], [1, 1], [], [8]])
def test_labels(labels):
    with pytest.raises(ValueError): validate_labels(labels)


def test_sampling():
    r = Recording('01', 'a', np.array([0, 0, 5, 100.]), np.arange(32).reshape(4, 8), np.array([1, 2, 2, 3]))
    assert analyze_sampling(r.timestamps)['repeated_timestamps'] == 1
    t, x, y = resample_recording(r, 200, 20)
    assert y[0] == 0 and y[1] == 2 and y[10] == 0 and y[-1] == 3
    assert np.array_equal(x[0], r.samples[1])
    with pytest.raises(ValueError): analyze_sampling([0, -1])


@pytest.mark.parametrize('kwargs', [{'high_hz': 100}, {'low_hz': 0}, {'notch_hz': 100}, {'low_hz': 90, 'high_hz': 80}])
def test_nyquist(kwargs):
    with pytest.raises(ValueError): design_filter(200, **kwargs)


def test_filter_and_chunk_invariance():
    rng = np.random.default_rng(0); x = rng.normal(size=(300, 8)); sos = design_filter(200)
    full = filter_signal(x, sos)
    processor = CausalFilter(sos)
    chunks = np.concatenate([processor.process(part) for part in np.array_split(x, 13)])
    np.testing.assert_allclose(chunks, full, atol=1e-12)
    assert np.isfinite(filter_signal(x[:3], sos, mode='zero_phase')).all()
    x[0, 0] = np.nan
    with pytest.raises(ValueError): filter_signal(x, sos)


def test_windows_boundaries():
    x = np.ones((100, 8)); t = np.arange(100)*5.; y = np.r_[np.ones(45), np.full(55, 2)]
    windows, meta = make_windows(x, t, y, '01', 'a', 200, [1, 2])
    assert windows.shape == (2, 8, 40)
    for row in meta.itertuples():
        assert len(np.unique(y[row.start_index:row.start_index+40])) == 1
        assert row.duration_ms == 200 and row.sample_span_ms == 195


def test_features():
    x = np.ones((2, 8, 40))*2
    f = extract_features(x, 200, spectral=True)
    np.testing.assert_allclose(f.ch1_rms, 2)
    assert (f.ch1_variance == 0).all() and (f.ch1_waveform_length == 0).all()
    assert (f.ch1_mean_frequency_hz == 0).all()
    pattern = np.tile([1., -1., 1., -1.], (1, 8, 1))
    f = extract_features(pattern, 200, noise_threshold=.1)
    assert f.ch1_zero_crossings.iloc[0] == 3 and f.ch1_slope_sign_changes.iloc[0] == 2


def test_training_only_normalization():
    train = np.ones((3, 8, 40)); test = np.ones((2, 8, 40))*100
    normalizer = ChannelNormalizer().fit(train)
    assert (normalizer.transform(train) == 0).all()
    assert (normalizer.transform(test) == 99).all()
    restored = ChannelNormalizer.from_state(normalizer.state())
    np.testing.assert_array_equal(restored.transform(test), normalizer.transform(test))


def test_splits():
    meta = pd.DataFrame({'subject_id': np.repeat(['01', '02', '03', '04', '05'], 4),
                         'recording_id': ['a']*20, 'label': [1, 2]*10})
    masks, assignment = split_groups(meta)
    assert sum(mask.astype(int) for mask in masks.values()).tolist() == [1]*20
    assert not set(assignment['train']) & set(assignment['test'])
    assert split_groups(meta)[1] == assignment


def test_cnn_contract():
    model = GestureCNN(4)
    assert model(torch.zeros(3, 8, 40)).shape == (3, 4)
    with pytest.raises(ValueError): model(torch.zeros(3, 40, 8))


class DummyPredictor:
    def __init__(self):
        self.config = yaml.safe_load(Path('configs/default.yaml').read_text())
        self.windows = []
    def predict(self, window):
        self.windows.append(window.copy())
        return {'label': 2, 'confidence': .9}


def test_streaming_chunk_invariance():
    rng = np.random.default_rng(1)
    t = np.r_[np.arange(0, 600, 1.), np.arange(650, 1400, 1.)]
    x = rng.normal(size=(len(t), 8))
    a, b = DummyPredictor(), DummyPredictor()
    first = StreamingProcessor(a); second = StreamingProcessor(b)
    out_a = first.process_chunk(x, t)+first.flush()
    out_b = []
    for start in range(0, len(t), 37): out_b.extend(second.process_chunk(x[start:start+37], t[start:start+37]))
    out_b.extend(second.flush())
    np.testing.assert_allclose(a.windows, b.windows, atol=1e-12)
    assert [(r['label'], r['end_ms']) for r in out_a] == [(r['label'], r['end_ms']) for r in out_b]
    assert out_a[0]['label'] is None and out_a[2]['label'] == 2
    with pytest.raises(ValueError): second.process_chunk(x[:2], [-1, 0])


def test_streaming_matches_offline():
    r = Recording('01', 'a', np.repeat(np.arange(300)*2., 2), np.random.default_rng(3).normal(size=(600, 8)), np.ones(600, int))
    t, x, y = resample_recording(r, 200)
    predictor = DummyPredictor(); p = StreamingProcessor(predictor)
    p.process_chunk(r.samples, r.timestamps); p.flush()
    offline = filter_signal(x, design_filter(200))
    expected, _ = make_windows(offline, t, y, '01', 'a', 200, [1])
    np.testing.assert_allclose(predictor.windows, expected, atol=1e-6)


def test_real_subset_end_to_end(tmp_path):
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    files = discover(cfg['raw_dir'])
    cfg.update(subjects=['01', '02', '03'], processed_dir=str(tmp_path/'processed'), report_dir=str(tmp_path/'reports'))
    cfg['training'].update(epochs=1, trees=8, batch_size=512)
    x, meta, manifest = prepare(cfg)
    assert set(meta.subject_id) == {'01', '02', '03'}
    train_baseline(cfg); train_cnn(cfg)
    for name in ['baseline.pkl', 'cnn.pt']:
        predictor = Predictor(tmp_path/'processed'/name)
        prediction = predictor.predict(x[0])
        assert prediction['label'] in cfg['labels'] and 0 <= prediction['confidence'] <= 1
        with pytest.raises(ValueError): predictor.predict(np.zeros((40, 8)))
    result = evaluate_artifact(cfg, 'baseline')
    assert 0 <= result['accuracy'] <= 1


def test_missing_label_policy(tmp_path):
    frame = pd.DataFrame(np.zeros((3, 10)), columns=['time', *[f'channel{i}' for i in range(1, 9)], 'class'])
    frame['time'] = [0, 5, 10]; frame['class'] = [1., 1., np.nan]
    path = tmp_path/'r.txt'; frame.to_csv(path, sep='\t', index=False)
    with pytest.raises(ValueError): read_recording(path)
    with pytest.warns(UserWarning, match='unmarked'):
        r = read_recording(path, missing_label='unmarked')
    assert r.missing_labels == 1 and r.labels[-1] == 0


def test_streaming_gap_offline_parity():
    from emg_pipeline.filtering import filter_valid_segments
    t = np.r_[np.arange(0, 400, 2.), np.arange(800, 1400, 2.)]
    r = Recording('01', 'r', t, np.random.default_rng(8).normal(size=(len(t), 8)), np.ones(len(t), int))
    grid, raw, y, valid = resample_recording(r, 200, return_valid=True)
    filtered = filter_valid_segments(raw, valid, design_filter(200))
    predictor = DummyPredictor(); processor = StreamingProcessor(predictor, stable_windows=1)
    emitted = processor.process_chunk(r.samples, r.timestamps)+processor.flush()
    for window, item in zip(predictor.windows, emitted):
        start = np.searchsorted(grid, item['start_ms'])
        np.testing.assert_allclose(window, filtered[start:start+40].T, atol=1e-6)


def test_filter_attenuation_and_spectral_units():
    t = np.arange(2000)/200
    wave = np.sin(2*np.pi*20*t) + np.sin(2*np.pi*95*t)
    x = np.tile(wave[:, None], (1, 8))
    filtered = filter_signal(x, design_filter(200))
    spectrum = abs(np.fft.rfft(filtered[400:, 0]))
    frequency = np.fft.rfftfreq(1600, 1/200)
    assert spectrum[np.argmin(abs(frequency-95))] < spectrum[np.argmin(abs(frequency-20))]*.05
    feature = extract_features(np.tile(np.sin(2*np.pi*20*t)[None, None], (1, 8, 1)), 200, spectral=True)
    assert feature.ch1_mean_frequency_hz.iloc[0] == pytest.approx(20)
    assert feature.ch1_median_frequency_hz.iloc[0] == pytest.approx(20)
    assert feature.ch1_band_power_signal_squared.iloc[0] == pytest.approx(.5)


def test_activation_training_calibration():
    from emg_pipeline.models.baseline import ActivationThreshold
    x = np.r_[np.ones((20, 8, 40))*.01, np.ones((20, 8, 40))*.2]
    y = np.r_[np.ones(20, int), np.full(20, 2)]
    model = ActivationThreshold().fit(x, y)
    assert np.array_equal(model.predict(x), y != 1)
    before = model.threshold
    model.predict(np.ones((3, 8, 40))*100)
    assert model.threshold == before

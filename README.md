# EMG-Controlled Assistive Robotic Arm: gesture pipeline

A CPU-compatible Python research pipeline for eight-channel EMG gesture recognition. Predictions contain gesture labels and confidence; robot control belongs to a future, separate layer.

## macOS quick start

Install Python 3.11 or newer (this workspace was executed with Python 3.13). Put `emg+data+for+gestures.zip` in the repository root. From a fresh checkout:

```sh
cd emg-processing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
emg inspect-data
emg plot-signal
emg prepare-data
emg extract-features
emg train-baseline
emg evaluate --model baseline
emg train-cnn
emg evaluate --model cnn
emg replay --model cnn --chunk-size 128
pytest -q
```

`emg run-all` runs inspection, plotting, preparation, both training jobs, evaluation and both replays. Global config precedes the command: `emg --config configs/smoke.yaml run-all`. All paths are relative to the working directory. `requirements-lock.txt` records the actual tested environment; use it to reproduce dependencies on a compatible Python/macOS architecture. No CUDA required.

## Dataset and timing

The archive README describes 36 subjects, two series per subject, eight MYO channels, static gestures performed for 3 seconds with 3-second pauses, and labels 0 unmarked, 1 rest, 2 fist, 3 wrist flexion, 4 wrist extension, 5 radial deviation, 6 ulnar deviation and 7 extended palm. Not all subjects performed label 7. The README provides milliseconds for timestamps but no amplitude units or nominal sampling rate. The associated paper is [Lobov et al., Sensors 2018](https://doi.org/10.3390/s18041122); this implementation does not infer an acquisition rate from a device name.

Inspection reads every recording strictly and writes row counts, durations, timestamp differences, repeated timestamps, repeated channel vectors, per-subject labels and channel summaries. Exact duplicate rows are counted and retained for audit; nonnumeric, missing, infinite, decreasing timestamps, invalid labels and malformed columns fail explicitly. The explicit `missing_label: unmarked` policy retains missing labels as class 0 and audits their counts; set `error` for strict rejection. This archive has one missing final label in subject 34, first recording. Equal timestamps are accepted as acquisition/logging ambiguity. ZIP extraction rejects traversal and symlinks. Original class 0 remains available for inspection.

**The 200 Hz setting is an explicit analysis grid, not a verified nominal acquisition rate.** The empirical row rate measures stored rows per elapsed timestamp. It can overstate independent sensor acquisitions because channel vectors repeat. Previous-value hold maps the nonuniform timestamps to the grid causally; the last vector at a duplicate timestamp wins, conflicting labels become unmarked. Holds older than 50 ms are unmarked and streaming resets state after outages. This policy is reproducible but is not proof of ideal anti-aliasing; independent sensor timing and aliasing must be assessed before physiological frequency interpretation. Empirical vector-change rates are approximately 91–111 Hz; they are not independently verified acquisitions. The conservative 40 Hz upper cutoff is below half the minimum observed vector-change rate as well as below the analysis-grid Nyquist frequency. No quantitative SNR improvement is asserted.

## Processing and experiments

Defaults: causal fourth-order 5–40 Hz Butterworth band-pass on a 200 Hz grid; 200 ms/40-sample windows, 50% overlap; labels 1–4; seeded 60/20/20 approximate subject split. Every cutoff must be strictly below Nyquist. Optional `notch_hz: 50` or `60` requires justified mains interference, and is disabled initially. High-pass, low-pass and band-pass are configurable.

`filter.mode: zero_phase` supports offline forward/backward filtering with reduced padding and a warning for short sequences. It uses future data and squares the causal magnitude response. Causal filtering uses persistent SOS state and initial steady-state conditions. Both have startup/boundary effects; causal phase delay depends on frequency, so it is not a single fixed latency. Streaming refuses zero-phase-trained artifacts.

Windows cannot cross recording boundaries or label transitions: each constant-label segment has its own hop origin. Mixed labels and class 0 are excluded; incomplete trailing windows are omitted. Metadata records subject, recording, start/end timestamps, start index, nominal duration and observed sample span. A 40-sample window at 200 Hz has 200 ms duration and a 195 ms first-to-last sample span.

Features per channel: RMS, MAV, sample variance, waveform length, zero-crossing count and slope-sign-change count. Count thresholds use original source amplitude units (`2e-5` default). Optional periodogram mean/median frequency (Hz) and total Nyquist-band power (source units²) use correct frequency bins and zero-energy guards; 200 ms spectral estimates are coarse. CNN input is `[batch, 8, samples]`, with no FFT.

Subject splits hold out all recordings of each subject. Set `split.unit: recording` only for an explicit within-subject/recording-generalization experiment; this holds out complete subject/recording pairs but does not guarantee every subject appears in each split. Group assignments are saved exactly. Missing split classes produce warnings; CNN training fails if any configured class is missing in training. Explicit `subjects` selection requires two-digit quoted strings; defaults include all subjects. To use all gestures set `labels: [1, 2, 3, 4, 5, 6, 7]`; availability varies by subject.

The RMS activation threshold maximizes training binary balanced accuracy over a fixed quantile grid. It detects rest versus activation and cannot distinguish active gestures. The multiclass baseline uses training-fitted StandardScaler and a class-weighted RandomForest. CNN channel statistics are fitted only on training windows, with weighted cross entropy, deterministic seeds, validation-loss early stopping and the best checkpoint. Test labels do not determine preprocessing statistics or checkpoint selection.

## Artifacts and interpretation

- `data/raw/`: extracted immutable source data (ignored).
- `data/processed/`: windows, provenance CSV, split/config manifest, named features, baseline/activation pickle artifacts and CNN checkpoint (ignored).
- `reports/metrics/`: dataset inventory, distributions, evaluation JSON, held-out prediction CSVs, training history and live replay CSVs.
- `reports/figures/`: raw/filtered channels, PSD, filter response, window boundaries, feature distributions, class distribution, confusion matrices, held-out predictions and learning curves.
- `reports/experiment.md`: measured experimental report; `reports/verification.md`: execution handoff.

CNN checkpoints include weights, label map, config/filter settings and channel statistics. Baseline artifacts include the fitted scaler, classifier, label map, feature names and config. **Load only trusted local model files**: pickle loading executes Python objects. Prepared configuration is checked against the requested configuration before training/evaluation. Each config should use distinct output directories to preserve experiments.

Evaluation reports accuracy, balanced accuracy, macro F1, per-class precision/recall/F1 and confusion matrices. Threshold scores are reported separately as binary metrics. Timing separates total preparation wall time, CPU batch prediction time and nominal window accumulation. Batch time per window is throughput, not single-window latency. Replay records actual per-window predictor time, window duration and additional stability delay; no hardware end-to-end timing is claimed. Overlapping test windows are correlated; use subject-level bootstrap confidence intervals in future confirmatory work.

## Streaming interface

```python
from emg_pipeline.inference import Predictor
from emg_pipeline.streaming import StreamingProcessor
p = StreamingProcessor(Predictor('data/processed/cnn.pt'))
predictions = p.process_chunk(samples, timestamps)  # [N,8], [N] milliseconds
predictions += p.flush()  # only at end of replay
```

Each call returns a list of zero or more dictionaries (one per completed window), with `label`, `confidence`, `status`, `start_ms`, `end_ms`, timing and stability metadata. `label=None`, `status='unknown'` is the fallback until three consecutive above-threshold predictions agree. Stable rest has `label=1`, `status='rest'`. Confidence is the raw maximum model probability and is not calibrated. No fallback emits a motor command. A pending source sample is finalized only once a newer timestamp arrives, allowing duplicate timestamps split across chunks to use the same last-value policy. `flush()` finalizes the last observed grid time.

Filters and buffers persist across chunks, never across separate replay processors. Outages clear state and votes; backward or invalid timestamp chunks fail without accepting those chunks. Replay uses one saved test recording, resamples causally, and writes end-time true labels for inspection. Streaming windows may cross real gesture transitions because transition labels are unavailable live; this differs from supervised windows. Compare replay with offline results and calibrate rejection/stability on validation recordings before hardware integration.

## Tests and scope

`pytest -q` covers ingestion/security, label validation, timing, Nyquist guards, filters, transition-safe windows, feature calculations, training-only normalization, group splitting, CNN shape, streaming chunk invariance/offline parity, inference and a one-epoch real-data smoke experiment using subjects 01–03. The smoke test writes only temporary artifacts and does not require full training. Full source ZIP or extracted data is needed for that test.

This is a functioning research pipeline, not a validated clinical system. Remaining work includes independently verifying acquisition timing, evaluating sensitivity to resampling/filter settings, confidence calibration, all-label experiments, subject-level uncertainty and future hardware control design.

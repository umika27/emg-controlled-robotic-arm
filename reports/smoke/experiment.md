# EMG gesture recognition experiment

Eight-channel time-domain classification; outputs are gesture labels, never motor commands.

## Methodology

Configuration:

```yaml
missing_label: unmarked
seed: 42
raw_dir: data/raw
processed_dir: data/processed/smoke
report_dir: reports/smoke
labels:
- 1
- 2
- 3
- 4
subjects:
- '01'
- '02'
- '03'
- '04'
- '05'
- '06'
sampling:
  target_hz: 200
  max_gap_ms: 50
filter:
  mode: causal
  kind: bandpass
  low_hz: 5
  high_hz: 40
  order: 4
  notch_hz: null
window:
  seconds: 0.2
  overlap: 0.5
features:
  noise_threshold: 2.0e-05
  spectral: false
split:
  unit: subject
  validation: 0.2
  test: 0.2
training:
  epochs: 2
  batch_size: 256
  learning_rate: 0.001
  patience: 4
  threads: 4
  trees: 30
streaming:
  confidence: 0.6
  stable_windows: 3
```

Nondecreasing millisecond timestamps are resampled by causal previous-value hold. Conflicting duplicate labels and gaps over the configured limit are unmarked. No claim is made that row rate equals independent sensor rate. The README does not specify nominal sampling frequency. The 200 Hz default is an analysis grid and needs independent acquisition-rate validation.

Missing source labels are retained as unmarked under the explicit configuration policy and audited in inventory.csv. Filtering resets at acquisition outages and remains continuous within valid portions of each recording. Window extraction resets at every label transition and excludes label 0. Overlapping windows remain in the same subject or complete-recording split. Channel statistics, feature scaling and activation threshold use training data only. CNN selection uses validation loss; test data never select checkpoints.

Prepared 1527 windows. Preprocessing elapsed: 0.653 seconds.

Exact group assignment: `{"test": ["04"], "validation": ["03"], "train": ["06", "05", "02", "01"]}`

Window counts by split and label:

```text
split       label
test        1         62
            2         63
            3         66
            4         68
train       1        254
            2        253
            3        265
            4        256
validation  1         60
            2         59
            3         61
            4         60
```

## Measured results

Dataset inventory: 36 subjects, 72 recordings, 4,237,908 source rows. See metrics/inventory.csv for timing and source-quality details.

baseline: accuracy=0.8069, balanced accuracy=0.8107, macro F1=0.7795; CPU batch inference=0.0647 ms/window.
cnn: accuracy=0.4093, balanced accuracy=0.3971, macro F1=0.2658; CPU batch inference=0.0482 ms/window.

Binary activation threshold: test balanced accuracy=1.0000. This is a separate binary task and cannot be compared directly with gesture accuracy.

## Limitations and next steps

Overlapping windows are correlated; metrics are window-level, without subject bootstrap intervals. Source amplitude units are not specified by the README. Previous-value downsampling assumes held sensor values and does not recover lost high-frequency information; acquisition-rate and aliasing validation remain necessary. No quantitative SNR improvement is claimed.

Causal filters introduce frequency-dependent delay and startup transients. Zero-phase filtering uses future data and has a squared magnitude response; zero-phase artifacts are rejected by streaming. Replay buffers can cross unknown gesture transitions, unlike supervised training windows. Stability adds hop-based delay; confidence is uncalibrated. Replay is a functional simulation, not a hardware latency measurement.

Future work: validate independent sensor timing, assess all available labels and subject confidence intervals, calibrate rejection on validation data, then integrate a separately reviewed robot-control interface. No PID or hardware drivers exist.

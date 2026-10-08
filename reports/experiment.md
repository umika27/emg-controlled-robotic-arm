# EMG gesture recognition experiment

Eight-channel time-domain classification; outputs are gesture labels, never motor commands.

## Methodology

Configuration:

```yaml
missing_label: unmarked
seed: 42
raw_dir: data/raw
processed_dir: data/processed
report_dir: reports
labels:
- 1
- 2
- 3
- 4
subjects: null
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
  epochs: 15
  batch_size: 256
  learning_rate: 0.001
  patience: 4
  threads: 4
  trees: 150
streaming:
  confidence: 0.6
  stable_windows: 3
```

Nondecreasing millisecond timestamps are resampled by causal previous-value hold. Conflicting duplicate labels and gaps over the configured limit are unmarked. No claim is made that row rate equals independent sensor rate. The README does not specify nominal sampling frequency. The 200 Hz default is an analysis grid and needs independent acquisition-rate validation.

Missing source labels are retained as unmarked under the explicit configuration policy and audited in inventory.csv. Filtering resets at acquisition outages and remains continuous within valid portions of each recording. Window extraction resets at every label transition and excludes label 0. Overlapping windows remain in the same subject or complete-recording split. Channel statistics, feature scaling and activation threshold use training data only. CNN selection uses validation loss; test data never select checkpoints.

Prepared 9454 windows. Preprocessing elapsed: 3.955 seconds.

Exact group assignment: `{"test": ["28", "19", "27", "06", "05", "24", "36"], "validation": ["08", "26", "33", "11", "23", "21", "34"], "train": ["17", "07", "25", "30", "29", "10", "31", "20", "04", "01", "16", "18", "32", "13", "22", "12", "15", "35", "03", "02", "14", "09"]}`

Window counts by split and label:

```text
split       label
test        1         418
            2         433
            3         433
            4         451
train       1        1472
            2        1418
            3        1498
            4        1484
validation  1         493
            2         450
            3         449
            4         455
```

## Measured results

Dataset inventory: 36 subjects, 72 recordings, 4,237,908 source rows. See metrics/inventory.csv for timing and source-quality details.

baseline: accuracy=0.9648, balanced accuracy=0.9648, macro F1=0.9647; CPU batch inference=0.0211 ms/window.
cnn: accuracy=0.9135, balanced accuracy=0.9137, macro F1=0.9131; CPU batch inference=0.0566 ms/window.

Binary activation threshold: test balanced accuracy=0.9600. This is a separate binary task and cannot be compared directly with gesture accuracy.

## Limitations and next steps

Overlapping windows are correlated; metrics are window-level, without subject bootstrap intervals. Source amplitude units are not specified by the README. Previous-value downsampling assumes held sensor values and does not recover lost high-frequency information; acquisition-rate and aliasing validation remain necessary. No quantitative SNR improvement is claimed.

Causal filters introduce frequency-dependent delay and startup transients. Zero-phase filtering uses future data and has a squared magnitude response; zero-phase artifacts are rejected by streaming. Replay buffers can cross unknown gesture transitions, unlike supervised training windows. Stability adds hop-based delay; confidence is uncalibrated. Replay is a functional simulation, not a hardware latency measurement.

Future work: validate independent sensor timing, assess all available labels and subject confidence intervals, calibrate rejection on validation data, then integrate a separately reviewed robot-control interface. No PID or hardware drivers exist.

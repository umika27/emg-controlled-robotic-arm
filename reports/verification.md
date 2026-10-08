# Technical handoff

Implemented every requested module and CLI command, including strict/audited ingestion, timestamp diagnostics, causal resampling, valid SOS filters, transition-safe provenance windows, named features, group splits, training-only normalization, binary activation and multiclass baselines, CPU CNN training/checkpointing, evaluation, stateful live replay, plots, configuration, documentation and automated tests. No hardware drivers or PID are included.

## Executed commands

From `/Users/umikasood/emg-processing`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas scipy matplotlib scikit-learn torch pytest PyYAML tqdm
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/emg inspect-data
.venv/bin/emg run-all
.venv/bin/emg extract-features
.venv/bin/emg evaluate --model cnn
.venv/bin/emg --config configs/smoke.yaml run-all
.venv/bin/pytest -q
.venv/bin/python -m compileall -q src scripts
.venv/bin/python scripts/provenance.py
```

Verification runs used `XDG_CACHE_HOME=/tmp/emg-cache MPLCONFIGDIR=/tmp/emg-mpl` to avoid unwritable host font caches. Editable installation initially failed because sandbox DNS blocked fetching setuptools; the authorized retry succeeded. Initial test expectations and a configuration-key typo were corrected; final results below describe the final code, not those failed attempts.

## Actual verification

Final pytest: **24 passed, 0 failed**, with one expected warning for reduced padding on a deliberately short signal. Includes a temporary, one-epoch, three-subject real-data smoke test. Full `run-all` and six-subject/two-epoch smoke `run-all` both exited successfully.

Inventory: 36 subjects, 72 recordings, 4,237,908 source rows. Prepared 9,454 supervised windows for labels 1–4, using causal 5–40 Hz filtering, 200 Hz analysis grid and subject-separated splits.

| Task/model | Test accuracy | Test balanced accuracy | Test macro F1 |
|---|---:|---:|---:|
| baseline multiclass | 0.9648 | 0.9648 | 0.9647 |
| cnn multiclass | 0.9135 | 0.9137 | 0.9131 |
| Activation binary (separate task) | 0.9429 | 0.9600 | 0.9272 |

CNN ran 14 epochs and retained the minimum-validation-loss checkpoint. Exact splits and per-class metrics are in the experimental report and JSON artifacts.

## Produced artifacts

- `data/raw/EMG_data_for_gestures-master/`: source README and all 72 recordings.
- `data/processed/`: windows.npz, windows.csv, manifest.json, features.csv, activation.pkl, baseline.pkl, cnn.pt.
- `reports/metrics/`: inventory.csv, class_counts.csv, subject_class_counts.csv, channel_summary.csv, dataset_summary.json, provenance.json, activation.json, baseline/cnn validation and test JSON, CNN history, held-out predictions and replay CSVs.
- `reports/figures/`: 11 plots covering signals, PSD, response, boundaries, distributions, confusion matrices, predictions and training.
- `reports/smoke/` and `data/processed/smoke/`: separate smoke outputs.
- `reports/experiment.md`, README.md, requirements-lock.txt and configurations.

## Known limitations and next steps

The source has one missing final label in subject 34; it is retained as class 0 with a warning and inventory count. Nominal independent sensor acquisition rate and amplitude units are unspecified in the README. Row rate is 821–973 Hz while vector-change rate is 91–111 Hz; the latter is not a verified acquisition rate. Previous-value hold and the conservative upper cutoff are documented research choices, not proof of alias-free physiological reconstruction.

Default measured results cover gestures 1–4 only. Labels 5–7 are supported but all-label training has not been executed. Label 7 occurs in only two subjects, so subject-separated three-way class coverage cannot be guaranteed. Window-level scores use correlated overlapping windows and do not include subject bootstrap intervals. Confidence probabilities remain uncalibrated; replay may mix transitions and include unmarked intervals. No hardware end-to-end timing or quantitative SNR gain is claimed.

Next experiments: independently validate timing and resampling, evaluate all labels with explicit missing-class reporting, compute subject confidence intervals, tune/calibrate rejection on validation recordings, and integrate a separately reviewed hardware control layer.

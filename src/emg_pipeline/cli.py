import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
from .dataset import discover, inventory, read_recording
from .preprocessing import prepare, load_prepared
from .sampling import resample_recording
from .filtering import design_filter, filter_signal
from .features import extract_features
from .training import train_baseline, train_cnn
from .evaluation import evaluate_artifact
from .inference import Predictor
from .streaming import StreamingProcessor
from . import plotting


def inspect(config):
    files = discover(config['raw_dir'])
    rows, counts, channels = inventory(files, missing_label=config.get('missing_label', 'error'))
    out = Path(config['report_dir']); (out/'metrics').mkdir(parents=True, exist_ok=True)
    for name, frame in [('inventory', rows), ('class_counts', counts), ('channel_summary', channels)]:
        frame.to_csv(out/'metrics'/f'{name}.csv', index=False)
    counts.groupby(['subject_id', 'label'])['count'].sum().unstack(fill_value=0).to_csv(out/'metrics'/'subject_class_counts.csv')
    summary = dict(subjects=rows.subject_id.nunique(), recordings=len(rows), raw_rows=int(rows.rows.sum()),
                   empirical_row_hz_range=[float(rows.empirical_row_hz.min()), float(rows.empirical_row_hz.max())],
                   nominal_rate_hz=None, nominal_rate_note='Dataset README gives no nominal rate; 200 Hz is an explicit analysis grid, not an inferred acquisition rate.',
                   source_readme=str(next(Path(config['raw_dir']).glob('**/README.txt'))))
    (out/'metrics'/'dataset_summary.json').write_text(json.dumps(summary, indent=2))
    plotting.distribution_plot(counts, out/'figures'/'class_distribution.png')
    print(json.dumps(summary, indent=2)); return summary


def plot_signal(config):
    files = discover(config['raw_dir'])
    if config.get('subjects') is not None:
        files = [p for p in files if p.parent.name in config['subjects']]
    if not files:
        raise ValueError('No selected recordings.')
    r = read_recording(files[0], missing_label=config.get('missing_label', 'error')); fs = config['sampling']['target_hz']
    t, x, y = resample_recording(r, fs, config['sampling']['max_gap_ms'])
    sos = design_filter(fs, **config['filter'])
    plotting.signal_plots(t, x, filter_signal(x, sos, config['filter']['mode']), y, sos, fs, Path(config['report_dir'])/'figures')


def replay(config, kind, chunk_size):
    if chunk_size <= 0:
        raise ValueError('Chunk size must be positive.')
    _, meta, _ = load_prepared(config)
    first = meta[meta.split == 'test'].iloc[0]
    path = next(p for p in discover(config['raw_dir']) if p.parent.name == first.subject_id and p.stem == first.recording_id)
    r = read_recording(path, missing_label=config.get('missing_label', 'error'))
    predictor = Predictor(Path(config['processed_dir'])/('cnn.pt' if kind == 'cnn' else 'baseline.pkl'))
    processor = StreamingProcessor(predictor); predictions = []
    for start in range(0, len(r.samples), chunk_size):
        predictions.extend(processor.process_chunk(r.samples[start:start+chunk_size], r.timestamps[start:start+chunk_size]))
    predictions.extend(processor.flush())
    frame = pd.DataFrame(predictions)
    if len(frame):
        indices = np.searchsorted(r.timestamps, frame.end_ms, side='right')-1
        frame['true_label_at_end'] = r.labels[indices]
        frame['subject_id'] = r.subject_id; frame['recording_id'] = r.recording_id
    frame.to_csv(Path(config['report_dir'])/'metrics'/f'{kind}_replay.csv', index=False)
    print(f'Replayed {r.subject_id}/{r.recording_id}: {len(frame)} windows; unknown/rest are safe gesture-level fallbacks.')


def report(config):
    out = Path(config['report_dir']); x, meta, manifest = load_prepared(config)
    lines = ['# EMG gesture recognition experiment', '',
        'Eight-channel time-domain classification; outputs are gesture labels, never motor commands.', '',
        '## Methodology', '',
        'Configuration:', '', '```yaml', yaml.safe_dump(config, sort_keys=False).strip(), '```', '',
        'Nondecreasing millisecond timestamps are resampled by causal previous-value hold. Conflicting duplicate labels and gaps over the configured limit are unmarked. No claim is made that row rate equals independent sensor rate. The README does not specify nominal sampling frequency. The 200 Hz default is an analysis grid and needs independent acquisition-rate validation.', '',
        'Missing source labels are retained as unmarked under the explicit configuration policy and audited in inventory.csv. Filtering resets at acquisition outages and remains continuous within valid portions of each recording. Window extraction resets at every label transition and excludes label 0. Overlapping windows remain in the same subject or complete-recording split. Channel statistics, feature scaling and activation threshold use training data only. CNN selection uses validation loss; test data never select checkpoints.', '',
        f'Prepared {len(x)} windows. Preprocessing elapsed: {manifest["preprocessing_seconds"]:.3f} seconds.', '',
        f'Exact group assignment: `{json.dumps(manifest["split_groups"])}`', '',
        'Window counts by split and label:', '', '```text', meta.groupby(['split', 'label']).size().to_string(), '```', '',
        '## Measured results', '']
    summary_path = out/'metrics'/'dataset_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        lines += [f'Dataset inventory: {summary["subjects"]} subjects, {summary["recordings"]} recordings, {summary["raw_rows"]:,} source rows. See metrics/inventory.csv for timing and source-quality details.', '']
    for kind in ('baseline', 'cnn'):
        path = out/'metrics'/f'{kind}_test.json'
        if path.exists():
            result = json.loads(path.read_text())
            lines.append(f'{kind}: accuracy={result["accuracy"]:.4f}, balanced accuracy={result["balanced_accuracy"]:.4f}, macro F1={result["macro_f1"]:.4f}; CPU batch inference={result["inference_ms_per_window"]:.4f} ms/window.')
    path = out/'metrics'/'activation.json'
    if path.exists():
        result = json.loads(path.read_text())
        if 'test' in result:
            lines += ['', f'Binary activation threshold: test balanced accuracy={result["test"]["balanced_accuracy"]:.4f}. This is a separate binary task and cannot be compared directly with gesture accuracy.']
    lines += ['', '## Limitations and next steps', '',
        'Overlapping windows are correlated; metrics are window-level, without subject bootstrap intervals. Source amplitude units are not specified by the README. Previous-value downsampling assumes held sensor values and does not recover lost high-frequency information; acquisition-rate and aliasing validation remain necessary. No quantitative SNR improvement is claimed.', '',
        'Causal filters introduce frequency-dependent delay and startup transients. Zero-phase filtering uses future data and has a squared magnitude response; zero-phase artifacts are rejected by streaming. Replay buffers can cross unknown gesture transitions, unlike supervised training windows. Stability adds hop-based delay; confidence is uncalibrated. Replay is a functional simulation, not a hardware latency measurement.', '',
        'Future work: validate independent sensor timing, assess all available labels and subject confidence intervals, calibrate rejection on validation data, then integrate a separately reviewed robot-control interface. No PID or hardware drivers exist.', '']
    (out/'experiment.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Leakage-free EMG gesture pipeline')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('command', choices=['inspect-data', 'plot-signal', 'prepare-data', 'extract-features', 'train-baseline', 'train-cnn', 'evaluate', 'replay', 'run-all'])
    parser.add_argument('--model', choices=['cnn', 'baseline'], default='cnn')
    parser.add_argument('--chunk-size', type=int, default=128)
    args = parser.parse_args(); logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        from .configuration import validate_config
        config = validate_config(yaml.safe_load(Path(args.config).read_text()))
        command = args.command
        if command in ('inspect-data', 'run-all'): inspect(config)
        if command in ('plot-signal', 'run-all'): plot_signal(config)
        if command in ('prepare-data', 'run-all'):
            _, meta, _ = prepare(config); plotting.windows_plot(meta, Path(config['report_dir'])/'figures')
        if command == 'extract-features':
            x, meta, _ = load_prepared(config)
            f = extract_features(x, config['sampling']['target_hz'], **config['features'])
            f.to_csv(Path(config['processed_dir'])/'features.csv', index=False)
            plotting.feature_plot(f, meta, Path(config['report_dir'])/'figures'/'features.png')
        if command in ('train-baseline', 'run-all'): train_baseline(config)
        if command in ('train-cnn', 'run-all'): train_cnn(config)
        if command == 'evaluate':
            print(json.dumps(evaluate_artifact(config, args.model), indent=2)); report(config)
        if command == 'replay': replay(config, args.model, args.chunk_size)
        if command == 'run-all':
            for kind in ('baseline', 'cnn'):
                evaluate_artifact(config, kind); replay(config, kind, args.chunk_size)
            report(config)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        logging.error('%s', exc); sys.exit(2)


if __name__ == '__main__':
    main()

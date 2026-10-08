from pathlib import Path
from time import perf_counter
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score


def metrics(y, prediction, labels):
    return dict(accuracy=float(accuracy_score(y, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=labels, average='macro', zero_division=0)),
                confusion_matrix=confusion_matrix(y, prediction, labels=labels).tolist(),
                labels=list(map(int, labels)),
                per_class=classification_report(y, prediction, labels=labels, output_dict=True, zero_division=0))


def evaluate_artifact(config, kind='cnn'):
    from .inference import Predictor
    from .preprocessing import load_prepared
    from .plotting import confusion_plot, prediction_plot
    x, meta, manifest = load_prepared(config)
    mask = (meta.split == 'test').to_numpy()
    predictor = Predictor(Path(config['processed_dir'])/f'{kind}.pt' if kind == 'cnn' else Path(config['processed_dir'])/'baseline.pkl')
    start = perf_counter()
    probability = predictor.probabilities(x[mask])
    seconds = perf_counter()-start
    pred = np.array(predictor.labels)[probability.argmax(axis=1)]
    result = metrics(meta.loc[mask, 'label'], pred, predictor.labels)
    result.update(inference_seconds=seconds, windows=int(mask.sum()), inference_ms_per_window=seconds/mask.sum()*1000,
                  window_latency_ms=config['window']['seconds']*1000,
                  preprocessing_seconds=manifest['preprocessing_seconds'], split_groups=manifest['split_groups'],
                  timing_scope='CPU batch prediction including feature extraction or normalization; no hardware I/O')
    out = Path(config['report_dir']); (out/'metrics').mkdir(parents=True, exist_ok=True)
    (out/'metrics'/f'{kind}_test.json').write_text(json.dumps(result, indent=2))
    held = meta.loc[mask].copy(); held['prediction'] = pred
    held.to_csv(out/'metrics'/f'{kind}_predictions.csv', index=False)
    confusion_plot(result, out/'figures'/f'{kind}_confusion.png')
    prediction_plot(held, out/'figures'/f'{kind}_held_out.png')
    return result

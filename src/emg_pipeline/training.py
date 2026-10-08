from pathlib import Path
import copy
import json
import pickle
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .features import extract_features
from .normalization import ChannelNormalizer
from .models.baseline import traditional_model, ActivationThreshold
from .models.cnn import GestureCNN
from .evaluation import metrics
from .preprocessing import load_prepared


def seed_everything(seed, threads=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def train_baseline(config):
    x, meta, _ = load_prepared(config)
    train = (meta.split == 'train').to_numpy()
    valid = (meta.split == 'validation').to_numpy()
    f = extract_features(x, config['sampling']['target_hz'], **config['features'])
    y = meta.label.to_numpy()
    missing = set(config['labels']) - set(y[train])
    if missing:
        raise ValueError(f'Training split missing classes {sorted(missing)}')
    model = traditional_model(config['seed'], config['training']['trees']).fit(f.loc[train], y[train])
    artifact = dict(model=model, labels=model.classes_.tolist(), config=config, feature_names=f.columns.tolist())
    out = Path(config['processed_dir'])
    with (out/'baseline.pkl').open('wb') as stream:
        pickle.dump(artifact, stream)
    report = Path(config['report_dir'])/'metrics'; report.mkdir(parents=True, exist_ok=True)
    (report/'baseline_validation.json').write_text(json.dumps(metrics(y[valid], model.predict(f.loc[valid]), config['labels']), indent=2))
    f.to_csv(out/'features.csv', index=False)
    if 1 in config['labels']:
        activation = ActivationThreshold().fit(x[train], y[train])
        result = {'task': 'binary rest(0) versus activation(1)', 'threshold': activation.threshold}
        for split in ('validation', 'test'):
            mask = (meta.split == split).to_numpy()
            result[split] = metrics((y[mask] != 1).astype(int), activation.predict(x[mask]), [0, 1])
        (report/'activation.json').write_text(json.dumps(result, indent=2))
        with (out/'activation.pkl').open('wb') as stream:
            pickle.dump(activation, stream)
    else:
        (report/'activation.json').write_text(json.dumps({'skipped': 'Rest label 1 is excluded by explicit configuration.'}))
    from .plotting import feature_plot
    feature_plot(f, meta, Path(config['report_dir'])/'figures'/'features.png')
    return artifact


def train_cnn(config):
    seed_everything(config['seed'], config['training']['threads'])
    x, meta, _ = load_prepared(config)
    train = (meta.split == 'train').to_numpy(); valid = (meta.split == 'validation').to_numpy()
    labels = sorted(config['labels']); lookup = {label: i for i, label in enumerate(labels)}
    y = np.array([lookup[label] for label in meta.label])
    counts = np.bincount(y[train], minlength=len(labels))
    if (counts == 0).any():
        raise ValueError('Training split missing a configured class; change split or subset.')
    normalizer = ChannelNormalizer().fit(x[train])
    dataset = TensorDataset(torch.from_numpy(normalizer.transform(x[train])), torch.from_numpy(y[train]))
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(config['seed']))
    vx, vy = torch.from_numpy(normalizer.transform(x[valid])), torch.from_numpy(y[valid])
    model = GestureCNN(len(labels))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(y[train])/(len(labels)*counts), dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    best, stale, history, state = float('inf'), 0, [], None
    for epoch in range(config['training']['epochs']):
        model.train(); total = 0
        for bx, by in loader:
            optimizer.zero_grad(); loss = loss_fn(model(bx), by); loss.backward(); optimizer.step()
            total += loss.item()*len(by)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(batch) for batch in vx.split(config['training']['batch_size'])])
            val_loss = loss_fn(logits, vy).item()
        row = dict(epoch=epoch+1, train_loss=total/len(dataset), validation_loss=val_loss,
                   validation_accuracy=float((logits.argmax(1) == vy).float().mean()))
        history.append(row); print(row, flush=True)
        if val_loss < best:
            best, stale, state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config['training']['patience']:
            break
    if state is None:
        raise ValueError('Training epochs must be positive.')
    torch.save(dict(state_dict=state, labels=labels, config=config, normalization=normalizer.state()),
               Path(config['processed_dir'])/'cnn.pt')
    report = Path(config['report_dir']); (report/'metrics').mkdir(parents=True, exist_ok=True)
    (report/'metrics'/'cnn_history.json').write_text(json.dumps(history, indent=2))
    model.load_state_dict(state)
    with torch.no_grad():
        pred = torch.cat([model(batch) for batch in vx.split(config['training']['batch_size'])]).argmax(1).numpy()
    (report/'metrics'/'cnn_validation.json').write_text(json.dumps(metrics(meta.loc[valid, 'label'], np.array(labels)[pred], labels), indent=2))
    from .plotting import training_plot
    training_plot(history, report/'figures'/'training.png')
    return history

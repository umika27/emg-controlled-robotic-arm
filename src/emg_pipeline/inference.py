from pathlib import Path
import pickle
import numpy as np
import torch
from .models.cnn import GestureCNN
from .normalization import ChannelNormalizer
from .features import extract_features
from .windows import window_sizes


class Predictor:
    """Predict already-filtered windows. Load only locally trusted artifacts."""
    def __init__(self, path):
        path = Path(path)
        self.kind = 'cnn' if path.suffix == '.pt' else 'baseline'
        if self.kind == 'cnn':
            artifact = torch.load(path, map_location='cpu', weights_only=True)
            torch.set_num_threads(artifact['config']['training']['threads'])
            self.model = GestureCNN(len(artifact['labels']))
            self.model.load_state_dict(artifact['state_dict']); self.model.eval()
            self.normalizer = ChannelNormalizer.from_state(artifact['normalization'])
        else:
            with path.open('rb') as stream:
                artifact = pickle.load(stream)
            self.model = artifact['model']
        self.labels, self.config = artifact['labels'], artifact['config']
        self.samples, _ = window_sizes(self.config['sampling']['target_hz'], **self.config['window'])

    def probabilities(self, windows):
        x = np.asarray(windows)
        if x.ndim != 3 or x.shape[1:] != (8, self.samples) or not np.isfinite(x).all() or not len(x):
            raise ValueError(f'Expected finite nonempty [batch, 8, {self.samples}] filtered windows.')
        if self.kind == 'baseline':
            f = extract_features(x, self.config['sampling']['target_hz'], **self.config['features'])
            return self.model.predict_proba(f)
        with torch.no_grad():
            return torch.cat([torch.softmax(self.model(batch), dim=1) for batch in
                       torch.from_numpy(self.normalizer.transform(x)).split(256)]).numpy()

    def predict(self, window):
        p = self.probabilities(np.asarray(window)[None])[0]
        index = int(p.argmax())
        return dict(label=int(self.labels[index]), confidence=float(p[index]))

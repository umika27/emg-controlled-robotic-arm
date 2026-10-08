import numpy as np
from sklearn.preprocessing import StandardScaler


class ChannelNormalizer:
    def fit(self, train):
        x = np.asarray(train)
        if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Need finite nonempty training windows.')
        self.mean = x.mean(axis=(0, 2), keepdims=True)
        self.scale = x.std(axis=(0, 2), keepdims=True)
        self.scale[self.scale < 1e-12] = 1
        return self

    def transform(self, x):
        x = np.asarray(x)
        if not np.isfinite(x).all():
            raise ValueError('Normalization input must be finite.')
        return ((x-self.mean)/self.scale).astype(np.float32)

    def state(self):
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.mean, obj.scale = np.array(state['mean']), np.array(state['scale'])
        if not np.isfinite(obj.mean).all() or not np.isfinite(obj.scale).all() or (obj.scale <= 0).any():
            raise ValueError('Invalid normalizer state.')
        return obj

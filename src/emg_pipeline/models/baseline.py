import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


class ActivationThreshold:
    """Mean channel RMS threshold; calibrated exclusively on training labels."""
    def fit(self, x, labels):
        scores = self.scores(x)
        y = np.asarray(labels) != 1
        if len(np.unique(y)) != 2:
            raise ValueError('Activation calibration requires rest and active training examples.')
        candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
        quality = [balanced_accuracy_score(y, scores > threshold) for threshold in candidates]
        self.threshold = float(candidates[np.argmax(quality)])
        return self

    @staticmethod
    def scores(x):
        return np.sqrt(np.mean(np.asarray(x)**2, axis=2)).mean(axis=1)

    def predict(self, x):
        return (self.scores(x) > self.threshold).astype(int)


def traditional_model(seed=42, trees=150):
    return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=trees, class_weight='balanced',
                         random_state=seed, n_jobs=4, min_samples_leaf=2))

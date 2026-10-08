"""Fail early on inconsistent or invalid experimental settings."""
import math
from .labels import validate_labels
from .windows import window_sizes
from .filtering import design_filter


def validate_config(config):
    validate_labels(config['labels'])
    fs = config['sampling']['target_hz']
    if not math.isfinite(fs) or fs <= 0 or config['sampling']['max_gap_ms'] <= 0:
        raise ValueError('Sampling values must be finite and positive.')
    window_sizes(fs, **config['window'])
    design_filter(fs, **config['filter'])
    if config['filter']['mode'] not in ['causal', 'zero_phase']:
        raise ValueError('filter.mode must be causal or zero_phase.')
    subjects = config.get('subjects')
    if subjects is not None and (not subjects or len(set(subjects)) != len(subjects) or any(not isinstance(s, str) or len(s) != 2 or not s.isdigit() for s in subjects)):
        raise ValueError('subjects must contain distinct quoted two-digit IDs.')
    train = config['training']
    if any(train[k] < 1 for k in ['epochs', 'batch_size', 'patience', 'threads', 'trees']) or train['learning_rate'] <= 0:
        raise ValueError('Training counts and learning rate must be positive.')
    if config['features']['noise_threshold'] < 0:
        raise ValueError('Noise threshold must be nonnegative.')
    return config

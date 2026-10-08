import warnings
import numpy as np


def split_groups(metadata, seed=42, unit='subject', validation=.2, test=.2):
    if unit not in ('subject', 'recording') or validation <= 0 or test <= 0 or validation+test >= 1:
        raise ValueError('Invalid split unit or fractions.')
    groups = (metadata.subject_id.astype(str) if unit == 'subject' else
              metadata.subject_id.astype(str)+'/'+metadata.recording_id.astype(str))
    unique = np.array(sorted(groups.unique()))
    if len(unique) < 3:
        raise ValueError('At least three groups required for train/validation/test.')
    np.random.default_rng(seed).shuffle(unique)
    nt, nv = max(1, round(len(unique)*test)), max(1, round(len(unique)*validation))
    if nt+nv >= len(unique):
        raise ValueError('Fractions leave no training groups.')
    assignments = dict(test=unique[:nt].tolist(), validation=unique[nt:nt+nv].tolist(), train=unique[nt+nv:].tolist())
    masks = {k: groups.isin(v).to_numpy() for k, v in assignments.items()}
    expected = set(metadata.label.unique())
    for k, mask in masks.items():
        missing = expected - set(metadata.loc[mask, 'label'])
        if missing:
            warnings.warn(f'{k} missing classes {sorted(missing)}')
    return masks, assignments

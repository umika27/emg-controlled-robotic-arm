LABELS = {0: 'unmarked', 1: 'rest', 2: 'fist', 3: 'wrist flexion',
          4: 'wrist extension', 5: 'radial deviation', 6: 'ulnar deviation', 7: 'extended palm'}


def validate_labels(labels):
    values = list(labels)
    if not values or len(set(values)) != len(values) or any(x not in LABELS or x == 0 for x in values):
        raise ValueError('Choose distinct supervised labels from 1 through 7.')
    return values

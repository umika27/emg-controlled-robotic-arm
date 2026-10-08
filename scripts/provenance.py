"""Capture immutable source archive checksums and execution environment."""
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import version

result = {'python': platform.python_version(), 'platform': platform.platform(),
          'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn', 'torch', 'pytest', 'PyYAML']},
          'archives': {path.name: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() for path in Path('.').glob('*.zip')}}
Path('reports/metrics/provenance.json').write_text(json.dumps(result, indent=2))

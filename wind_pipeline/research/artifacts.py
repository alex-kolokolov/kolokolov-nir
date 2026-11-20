from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from .config import ResearchConfig


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(temporary, path)


def code_hash(config: ResearchConfig):
    base = config.root / 'wind_pipeline'
    files = [base / 'research_data.py', base / 'models/gru.py']
    files += [p for p in (base / 'research').glob('*.py') if p.stem != 'report']
    return object_hash({str(p.relative_to(config.root)): file_hash(p) for p in sorted(files)})


def environment():
    packages = ('numpy', 'pandas', 'pyarrow', 'torch', 'catboost', 'scikit-learn', 'matplotlib')
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in packages}}


def verify_run(config: ResearchConfig) -> dict:
    path = config.output / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError('Prepared manifest is absent; run research --step prepare first')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['config_hash'] != object_hash(config.payload):
        raise ValueError('Config changed: select a new output_dir for a separate experiment')
    if manifest['code_hash'] != code_hash(config):
        raise ValueError('Pipeline code changed since preparation; use a new experiment directory')
    if manifest['source_sha256'] != file_hash(config.source):
        raise ValueError('Source data changed since preparation')
    if manifest['environment']['packages'] != environment()['packages']:
        raise ValueError('Dependency versions changed since preparation; use a new experiment directory')
    for farm, item in manifest['prepared'].items():
        for filename, expected in item['files'].items():
            if file_hash(config.output / 'prepared' / farm / filename) != expected:
                raise ValueError(f'Prepared artifact changed: {farm}/{filename}')
    return manifest


def model_path(config, source, target, method, seed, days=0):
    return config.output / 'models' / source / target / f'{method}_{days}d_seed{seed}'


def log_event(config, **event):
    with (config.output / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')

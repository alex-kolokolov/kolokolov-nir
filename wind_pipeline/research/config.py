from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..research_data import FEATURES


@dataclass(frozen=True)
class ResearchConfig:
    """Validated experiment specification; paths are relative to the project root."""
    path: Path
    root: Path
    payload: dict

    @property
    def output(self) -> Path:
        return (self.root / self.payload['output_dir']).resolve()

    @property
    def source(self) -> Path:
        return (self.root / self.payload['data_path']).resolve()

    def __getitem__(self, key):
        return self.payload[key]


def load_research_config(path: str | Path) -> ResearchConfig:
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[2]
    data = json.loads(path.read_text(encoding='utf-8'))
    required = ('name', 'data_path', 'raw_sources', 'output_dir', 'farms', 'rated_power_kw', 'lookback',
                'horizons', 'train_stride', 'evaluation_stride', 'train_start', 'train_end',
                'validation_end', 'adaptation_start', 'adaptation_days', 'test_start',
                'test_end', 'seeds', 'catboost', 'gru', 'features', 'preprocessing', 'methods',
                'device', 'bootstrap_repetitions', 'primary_horizon_minutes', 'primary_adaptation_days')
    if set(data) != set(required):
        raise ValueError(f'Config fields: missing={set(required)-set(data)}, unknown={set(data)-set(required)}')
    dates = [pd.Timestamp(data[k]) for k in ('train_start', 'train_end', 'validation_end',
                                          'adaptation_start', 'test_start', 'test_end')]
    if not (dates[0] < dates[1] < dates[2] <= dates[3] < dates[4] < dates[5]):
        raise ValueError('Training, validation, adaptation and test dates overlap or are unordered')
    if dates[3] + pd.Timedelta(days=max(data['adaptation_days'])) > dates[4]:
        raise ValueError('Adaptation labels would overlap the test interval')
    if len(data['farms']) != 2 or len(set(data['farms'])) != 2:
        raise ValueError('This cross-farm protocol requires two distinct farms')
    if not data['seeds'] or len(set(data['seeds'])) != len(data['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    if not data['adaptation_days'] or min(data['adaptation_days']) < 1:
        raise ValueError('Adaptation days must be positive')
    features = data['features']
    if not features or features[0] != 'power_pu' or not set(features) <= set(FEATURES) or len(set(features)) != len(features):
        raise ValueError('Invalid features; power_pu must be first')
    if not set(data['methods']) <= {'persistence', 'catboost', 'gru'} or not data['methods']:
        raise ValueError('Unknown or empty methods')
    if 'persistence' not in data['methods']:
        raise ValueError('Persistence is required as the reference')
    if data['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu, or cuda')
    if min(data['lookback'], data['train_stride'], data['evaluation_stride']) < 1 or data['lookback'] < 2:
        raise ValueError('Invalid window dimensions')
    if not data['horizons'] or min(data['horizons']) < 1 or len(set(data['horizons'])) != len(data['horizons']):
        raise ValueError('Horizons must be positive and unique')
    if data['rated_power_kw'] <= 0:
        raise ValueError('Rated power must be positive')
    if data['primary_horizon_minutes'] not in [h*10 for h in data['horizons']]:
        raise ValueError('Primary horizon must be one of the forecast horizons')
    if data['primary_adaptation_days'] not in data['adaptation_days']:
        raise ValueError('Primary adaptation days must be included')
    if set(r['farm'] for r in data['raw_sources']['records']) != set(data['farms']):
        raise ValueError('Raw-source farms must match the experiment farms')
    for record in data['raw_sources']['records']:
        if not record['files']:
            raise ValueError('Raw file list cannot be empty')
        for item in record['files']:
            if Path(item['name']).name != item['name'] or len(item['md5']) != 32:
                raise ValueError('Each raw file needs a safe basename and a pinned MD5 checksum')
    if data['gru']['epochs'] < 1 or data['gru']['adaptation_epochs'] < 1 or data['gru']['batch_size'] < 1:
        raise ValueError('Epochs and batch size must be positive')
    if data['catboost']['iterations'] < 1 or data['bootstrap_repetitions'] < 100:
        raise ValueError('Invalid training or bootstrap iteration count')
    config = ResearchConfig(path, root, data)
    if not config.source.is_relative_to(config.output) or config.source.suffix != '.parquet':
        raise ValueError('data_path must be a new parquet inside this experiment output_dir')
    return config

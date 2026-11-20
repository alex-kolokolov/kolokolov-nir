from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TransferConfig:
    root: Path
    payload: dict
    cubico: dict

    def __getitem__(self, key):
        return self.payload[key]

    @property
    def output(self):
        return self.root / self['output_dir']

    @property
    def external(self):
        return self['external_archive']['farm']

    @property
    def farms(self):
        return (*self['source_farms'], self.external)

    @property
    def raw_directory(self):
        return self.root / self.cubico['raw_sources']['directory']

    @property
    def archives(self):
        known = [r for r in self.cubico['raw_sources']['records'] if r['farm'] in self['source_farms']]
        return [*known, self['external_archive']]

    @property
    def horizons(self):
        return tuple(self['horizons_minutes'])

    @property
    def variants(self):
        return ('core', 'scada', 'weather') if self['weather'] else ('core', 'scada')

    @property
    def device(self):
        import torch

        if self['device'] == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        return 'cuda' if self['device'] != 'cpu' and torch.cuda.is_available() else 'cpu'


def load_transfer_config(path):
    root = Path.cwd()
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    cubico = json.loads((root / raw['cubico_config']).read_text(encoding='utf-8'))
    if raw.get('pipeline') != 'transfer':
        raise ValueError('Expected pipeline=transfer')
    config = TransferConfig(root, raw, cubico)
    if len(raw['source_farms']) < 2 or len(set(config.farms)) != len(config.farms):
        raise ValueError('Use at least two unique source farms and a different external farm')
    if {x['farm'] for x in config.archives} != set(config.farms):
        raise ValueError('Raw archive metadata is missing for a configured farm')
    steps = [*config.horizons, raw['history_minutes'], raw['train_stride_minutes'],
             raw['adaptation_stride_minutes'], raw['evaluation_stride_minutes']]
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 10 or x % 10 for x in steps):
        raise ValueError('Histories, horizons and strides must be positive multiples of 10 minutes')
    if sorted(set(config.horizons)) != list(config.horizons) or raw['primary_horizon_minutes'] not in config.horizons:
        raise ValueError('Horizons must be increasing and include the primary horizon')
    if raw['history_minutes'] < 120:
        raise ValueError('At least 120 minutes of context is required for the feature summaries')
    start, source_val, adapt, test, end = map(pd.Timestamp, (raw['train_start'], raw['source_validation_start'],
                                             raw['adaptation_start'], raw['test_start'], raw['test_end']))
    development = raw['development']
    dev_val = pd.Timestamp(development['source_validation_start'])
    folds = list(map(pd.Timestamp, development['adaptation_starts']))
    if (not folds or folds != sorted(set(folds)) or not start < dev_val < folds[0] < source_val < adapt < test < end):
        raise ValueError('Expected ordered development, source validation, adaptation and test periods')
    if development['adaptation_days'] <= 0 or development['evaluation_days'] <= 0:
        raise ValueError('Development support and evaluation must have positive durations')
    span = pd.Timedelta(days=development['adaptation_days'] + development['evaluation_days'])
    if any(a + span > b for a, b in zip(folds, [*folds[1:], source_val])):
        raise ValueError('Development evaluation periods must be disjoint and end before final validation')
    days = raw['adaptation_days']
    if (not days or sorted(set(days)) != days or min(days) <= 0 or raw['primary_adaptation_days'] not in days
            or adapt + pd.Timedelta(days=max(days)) > test):
        raise ValueError('Each adaptation budget must end before the test and include the primary budget')
    if raw['device'] not in ('auto', 'cpu', 'cuda') or raw['threads'] < 1 or not raw['seeds']:
        raise ValueError('Specify a valid compute device, seed list and thread count')
    if len(set(raw['seeds'])) != len(raw['seeds']) or any(not 0 < x <= 1 for x in raw['ses_alphas']):
        raise ValueError('Seeds must be unique and SES coefficients in (0, 1]')
    if not raw['ses_alphas'] or not 0 < raw['ramp_threshold_pu'] <= 1:
        raise ValueError('Specify SES candidates and a positive ramp threshold')
    if raw['weather']:
        weather = raw['weather']
        if (pd.Timestamp(weather['start']) > start or not 6 <= weather['publication_delay_hours'] <= 24
                or weather['download_workers'] < 1 or not 0 < weather['minimum_coverage'] <= 1
                or max(config.horizons) + 60 * (weather['publication_delay_hours'] + 12) > 42 * 60):
            raise ValueError('IFS archive configuration cannot cover the requested horizons and training dates')
    return config

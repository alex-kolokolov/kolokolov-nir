"""Verify the completed training, evaluation and report against their manifests."""
from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import file_hash, object_hash
from . import analysis
from .experiment import checked_file, model_jobs, require_training


def verify_training(config, identity):
    selection, training = require_training(config, identity)
    checked_file(config.output, 'development_scores.csv', selection['scores_sha256'])
    expected = {stem: (family, variant, method, days, seed, horizon)
                for family, variant, method, days, seed, horizon, stem, _, _ in
                model_jobs(config, None, None, None, None, selection, fit=False)}
    fields = ('family', 'variant', 'method', 'days', 'seed', 'horizon')
    models = training['models']
    actual = {model['stem']: tuple(model[name] for name in fields) for model in models}
    if len(models) != len(expected) or actual != expected:
        raise ValueError('Training manifest does not cover the configured models')
    for model in models:
        extension = '.pt' if model['family'] == 'gru' else '.cbm'
        checked_file(config.output / 'models', model['stem'] + extension, model['sha256'])
    return training


def verify_evaluation(config, training):
    directory = config.output / 'predictions'
    saved = json.loads((config.output / 'evaluation.json').read_text(encoding='utf-8'))
    if (saved.get('identity') != training['identity'] or saved.get('training') != object_hash(training)
            or saved.get('farm') != config.external or saved.get('start') != config['test_start']
            or saved.get('end_exclusive') != config['test_end']):
        raise ValueError('Evaluation manifest does not match the training run and test period')

    fields = ('variant', 'method', 'days', 'seed')
    expected = {('baseline', method, 0, -1, horizon)
                for method in ('persistence', 'ses') for horizon in config.horizons}
    for model in training['models']:
        horizons = config.horizons if model['family'] == 'gru' else (model['horizon'],)
        expected.update((*[model[name] for name in fields], horizon) for horizon in horizons)

    predictions = saved.get('predictions', [])
    actual = {tuple(entry[name] for name in (*fields, 'horizon')) for entry in predictions}
    if len(predictions) != len(expected) or actual != expected:
        raise ValueError('Evaluation manifest does not cover the configured predictions')
    for entry in predictions:
        variant, method, days, seed, horizon = (entry[name] for name in (*fields, 'horizon'))
        filename = f'{variant}__{method}__{days}d__{horizon}m__seed{seed}.parquet'
        if entry.get('file') != filename or not 0 < entry['windows'] <= saved['origins']:
            raise ValueError(f'Invalid prediction entry for {filename}')
        checked_file(directory, filename, entry['sha256'])
    return saved


def verify_report(config, evaluation):
    metadata = json.loads((config.output / 'report_metadata.json').read_text(encoding='utf-8'))
    analysis_hash = file_hash(Path(analysis.__file__))
    if (metadata.get('evaluation') != object_hash(evaluation) or metadata.get('analysis_code') != analysis_hash
            or set(metadata.get('files', {})) != set(analysis.REPORT_FILES)):
        raise ValueError('Report manifest does not match the current analysis and evaluation')
    checked_file(config.output / 'source_snapshot', 'analysis.py', analysis_hash)
    for name, digest in metadata['files'].items():
        checked_file(config.output, name, digest)

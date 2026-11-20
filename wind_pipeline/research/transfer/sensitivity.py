"""Replay delayed weather and compare GRU initializations on the saved test split."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ..artifacts import file_hash, object_hash, write_json
from .analysis import OBSERVATION, block_interval, forecast_metrics
from .data import FarmSeries
from .experiment import cache_model, require_training
from .models import TensorStore, fit_local_gru, gru_predict, seed_torch
from .scada import TURBINE
from .verification import verify_evaluation
from .weather import align_forecasts


def saved_prediction(config, evaluation, variant, method, days, horizon, seed):
    key = (variant, method, days, horizon, seed)
    entry = next(item for item in evaluation['predictions']
                 if tuple(item[field] for field in ('variant', 'method', 'days', 'horizon', 'seed')) == key)
    return pd.read_parquet(config.output / 'predictions' / entry['file'])


def test_rows(config, test, evaluation, horizon):
    index = config.horizons.index(horizon)
    observed = np.isfinite(test.targets[:, index])
    baseline = saved_prediction(config, evaluation, 'baseline', 'ses', 0, horizon, -1)
    expected = (test.times[observed], test.farm.turbines[TURBINE].to_numpy()[test.turbine_index[observed]],
                test.ratings[observed], test.targets[observed, index], test.persistence[observed])
    for name, values in zip(OBSERVATION, expected, strict=True):
        actual = baseline[name].to_numpy(dtype='datetime64[m]' if name == 'origin' else None)
        if not np.array_equal(actual, values):
            raise ValueError(f'Test windows differ from saved {horizon} minute observations: {name}')
    return index, observed, baseline


def source_models(config, training, variant):
    fingerprints = {model['stem']: model['sha256'] for model in training['models']}
    identity = {'run': training['identity'], 'selection': training['selection'], 'source': None}
    for seed in config['seeds']:
        stem = f'final/{variant}/gru_zero_seed{seed}'
        model, _, sha = cache_model(config, stem, identity, 'gru', variant)
        if fingerprints.get(stem) != sha:
            raise ValueError(f'Unexpected pretrained checkpoint: {stem}')
        yield seed, model


def replay_weather(config, settings, farm, test, training, evaluation, predictions):
    horizon = settings['weather_horizon_minutes']
    later = settings['later_delay_hours']
    reference_delay = config['weather']['publication_delay_hours']
    if (horizon not in config.horizons or any(type(delay) is not int for delay in later)
            or sorted(set(later)) != later or later[0] <= reference_delay
            or max(config.horizons) + 60 * (later[-1] + 12) > 42 * 60):
        raise ValueError('Use increasing later delays that fit the archived IFS lead times and configured horizons')
    index, observed, baseline = test_rows(config, test, evaluation, horizon)
    positions = test.time_index[observed]
    origins = test.times[observed]
    reference_runs = farm.weather_runs[positions]
    baseline_mae = forecast_metrics(baseline, config['ramp_threshold_pu'])['mae_kw']
    forecasts = pd.read_parquet(config.output / 'ifs_forecasts.parquet', filters=[('farm', '==', farm.name)])
    models = list(source_models(config, training, 'weather'))
    rows = []
    for delay in (reference_delay, *later):
        weather, runs = align_forecasts(forecasts, farm.times.astype('datetime64[m]'), config.horizons, delay)
        used = runs[positions]
        if (not np.isfinite(weather[positions, index]).all() or np.isnat(used).any()
                or (used > reference_runs).any() or (used + np.timedelta64(delay, 'h') > origins).any()):
            raise ValueError(f'Insufficient causal IFS coverage on the shared test at {delay} hours')
        if delay == reference_delay:
            np.testing.assert_array_equal(used, reference_runs)
        scenario = replace(farm, weather=weather, weather_runs=runs)
        store = TensorStore(config, {farm.name: scenario})
        dataset = store.pack([test])
        errors = []
        for seed, model in models:
            seed_torch(config, seed)
            values = gru_predict(model, store, dataset, config['gru']['batch_size'])[observed, index]
            if delay == reference_delay:
                archived = saved_prediction(config, evaluation, 'weather', 'gru_zero', 0, horizon, seed)
                np.testing.assert_allclose(values, archived.prediction_pu, rtol=1e-4, atol=5e-6)
            frame = baseline[OBSERVATION].assign(prediction_pu=values, weather_run=pd.to_datetime(used),
                                                 scenario='weather_delay', delay_hours=delay, seed=seed, horizon=horizon)
            predictions.append(frame)
            errors.append(forecast_metrics(frame, config['ramp_threshold_pu'])['mae_kw'])
        mae = float(np.mean(errors))
        rows.append({'horizon_minutes': horizon, 'delay_hours': delay, 'windows': len(baseline),
                     'changed_run_pct': 100 * (used != reference_runs).mean(), 'mae_kw': mae,
                     'ses_mae_kw': baseline_mae, 'skill_vs_ses_pct': 100 * (1 - mae / baseline_mae)})
        print(f'IFS delay {delay} h: MAE {mae:.2f} kW on {len(baseline):,} windows', flush=True)
    return pd.DataFrame(rows)


def compare_initializations(config, farm, test, selection, training, evaluation, predictions):
    days = config['primary_adaptation_days']
    horizon = config['primary_horizon_minutes']
    index, observed, baseline = test_rows(config, test, evaluation, horizon)
    candidate = selection['variants']['scada']['scratch']
    protocol = {'updates': candidate['updates'], 'learning_rate': candidate['learning_rate'], 'penalty': 0.0}
    start = pd.Timestamp(config['adaptation_start'])
    local = farm.select(config, start, start + pd.Timedelta(days=days), config['adaptation_stride_minutes'])
    store = TensorStore(config, {farm.name: farm})
    local_inputs = store.pack([local])
    test_inputs = store.pack([test])
    metrics = []
    errors = {'random': [], 'pretrained': []}
    for seed, source in source_models(config, training, 'scada'):
        for initialization, base in (('random', None), ('pretrained', source)):
            model, _ = fit_local_gru(config, store, local_inputs, seed, 'scada', 'full', protocol, base)
            values = gru_predict(model, store, test_inputs, config['gru']['batch_size'])[observed, index]
            frame = baseline[OBSERVATION].assign(prediction_pu=values, scenario='initialization',
                                                 initialization=initialization, seed=seed, horizon=horizon)
            predictions.append(frame)
            scores = forecast_metrics(frame, config['ramp_threshold_pu'])
            metrics.append({'initialization': initialization, 'seed': seed,
                            'mae_kw': scores['mae_kw'], 'nmae_pct': scores['nmae_pct']})
            errors[initialization].append(np.abs(values.astype(float) - frame.actual_pu.to_numpy(dtype=float)))
            print(f'GRU {initialization}, seed {seed}: nMAE {scores["nmae_pct"]:.3f}%', flush=True)
    paired = np.mean(errors['pretrained'], axis=0), np.mean(errors['random'], axis=0)
    lo, hi, blocks = block_interval(*paired, baseline.origin, pd.Timestamp(config['test_start']),
                                   config['bootstrap_block_days'], config['bootstrap_repetitions'],
                                   config['development_seed'], .95)
    comparison = {'horizon_minutes': horizon, 'local_days': days, 'local_windows': len(local),
                  'test_windows': len(baseline), 'delta_nmae_pp': 100 * (paired[0] - paired[1]).mean(),
                  'ci95_low_pp': lo, 'ci95_high_pp': hi, 'blocks': blocks, 'optimizer': 'AdamW',
                  'batch_size': min(len(local), config['gru']['adaptation_batch_size']), **protocol}
    return pd.DataFrame(metrics), comparison


def run_sensitivity(config, identity, settings):
    if not isinstance(settings['later_delay_hours'], list) or not isinstance(settings['compare_initializations'], bool):
        raise ValueError('Expected a list of later delays and a boolean for the initialization comparison')
    selection, training = require_training(config, identity)
    evaluation = verify_evaluation(config, training)
    farm = FarmSeries.load(config, config.external)
    test = farm.select(config, config['test_start'], config['test_end'], config['evaluation_stride_minutes'])
    predictions, tables = [], {}
    metadata = {'settings': settings, 'evaluation': object_hash(evaluation), 'farm': farm.name,
                'seeds': config['seeds'], 'device': config.device}
    if settings['later_delay_hours']:
        if not config['weather']:
            raise ValueError('Weather delays require archived IFS inputs')
        tables['weather_delay.csv'] = replay_weather(config, settings, farm, test, training, evaluation, predictions)
    if settings['compare_initializations']:
        tables['initialization.csv'], metadata['initialization'] = compare_initializations(
            config, farm, test, selection, training, evaluation, predictions)
    if not predictions:
        raise ValueError('Enable at least one sensitivity comparison')

    directory = config.output / 'sensitivity'
    directory.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=directory) as staging:
        staged = Path(staging)
        for name, frame in tables.items():
            frame.to_csv(staged / name, index=False)
        pd.concat(predictions, ignore_index=True).to_parquet(staged / 'predictions.parquet', index=False)
        products = [*tables, 'predictions.parquet']
        write_json(staged / 'protocol.json', {**metadata, 'files': {name: file_hash(staged / name) for name in products}})
        for name in [*products, 'protocol.json']:
            os.replace(staged / name, directory / name)
    return directory

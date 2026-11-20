from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

from ..artifacts import file_hash, object_hash, write_json
from .models import (TensorStore, TransferGRU, cat_predict, fit_catboost,
                     fit_local_gru, fit_source_gru, gru_predict)
from .scada import TURBINE

MODES = ('head', 'full', 'scratch')
DAY = pd.Timedelta(days=1)


def selected_windows(config, series, farms, start, end, stride):
    return [series[name].select(config, start, end, config[stride]) for name in farms]


def checked_file(directory, name, expected):
    path = directory / name
    if not path.is_file() or file_hash(path) != expected:
        raise ValueError(f'Artifact integrity failed: {path}')
    return path


def cache_model(config, relative, identity, family, variant, fit=None):
    directory = config.output / 'models'
    stem = directory / relative
    suffix = '.pt' if family == 'gru' else '.cbm'
    model_path = stem.with_suffix(suffix)
    meta_path = stem.with_suffix('.json')
    if meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding='utf-8'))
        if saved['identity'] != object_hash(identity):
            raise ValueError(f'Model was saved with a different experiment fingerprint: {stem}')
        checked_file(directory, str(model_path.relative_to(directory)), saved['sha256'])
        if family == 'gru':
            model = TransferGRU(config, variant).to(config.device)
            model.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True))
        else:
            model = CatBoostRegressor()
            model.load_model(str(model_path))
        return model, saved['training'], saved['sha256']
    if fit is None:
        raise FileNotFoundError(f'Trained model is missing: {stem}; run research --step train')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Train {relative}', flush=True)
    model, training = fit()
    temporary = model_path.with_suffix(suffix + '.tmp')
    if family == 'gru':
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    else:
        model.save_model(str(temporary), format='cbm')
    os.replace(temporary, model_path)
    sha = file_hash(model_path)
    write_json(meta_path, {'identity': object_hash(identity), 'sha256': sha, 'training': training})
    return model, training, sha


def development_row(fold, target, variant, method, candidate, horizon, actual, prediction):
    good = np.isfinite(actual)
    if not good.any() or not np.isfinite(prediction[good]).all():
        raise ValueError(f'Invalid development predictions for {method}, {target}, horizon {horizon}')
    return {'fold': fold, 'target': target, 'variant': variant, 'method': method, 'candidate': candidate,
            'horizon_minutes': horizon, 'windows': int(good.sum()),
            'mae_pu': float(np.abs(actual[good] - prediction[good]).mean())}


def row_scores(rows, config, fold, window, variant, method, candidate, prediction):
    for j, horizon in enumerate(config.horizons):
        rows.append(development_row(fold, window.farm.name, variant, method, candidate, horizon,
                                    window.targets[:, j], prediction[:, j]))


def tune(config, series, store, identity):
    directory = config.output
    path = directory / 'selection.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError('Training inputs changed; use a separate output_dir')
        checked_file(directory, 'development_scores.csv', saved['scores_sha256'])
        return saved

    settings = config['development']
    seed = config['development_seed']
    selections, rows = {}, []
    batch = config['gru']['batch_size']
    for target in config['source_farms']:
        others = [f for f in config['source_farms'] if f != target]
        val_end = settings['adaptation_starts'][0]
        source = store.pack(selected_windows(config, series, others, config['train_start'],
                                              settings['source_validation_start'], 'train_stride_minutes'))
        validation = store.pack(selected_windows(config, series, others, settings['source_validation_start'],
                                                  val_end, 'evaluation_stride_minutes'))
        folds = []
        for date in settings['adaptation_starts']:
            start = pd.Timestamp(date)
            support_end = start + settings['adaptation_days'] * DAY
            evaluation_end = support_end + settings['evaluation_days'] * DAY
            support = series[target].select(config, start, support_end, config['adaptation_stride_minutes'])
            evaluation = series[target].select(config, support_end, evaluation_end, config['evaluation_stride_minutes'])
            folds.append((date, support, evaluation, store.pack([support]), store.pack([evaluation])))

        for variant in config.variants:
            base, base_training, base_sha = cache_model(config, f'development/{target}/source_{variant}', identity,
                'gru', variant, lambda: fit_source_gru(config, store, source, validation, seed, variant))
            for date, support, validation_window, support_tensors, test_tensors in folds:
                row_scores(rows, config, date, validation_window, variant, 'zero_shot', 'source',
                           gru_predict(base, store, test_tensors, batch))
                for mode in MODES:
                    for index, candidate in enumerate(config['gru'][f'{mode}_candidates']):
                        source_model = None if mode == 'scratch' else base
                        trained, _, _ = cache_model(config, f'development/{target}/{date}/{variant}_{mode}_{index}',
                            {'run': identity, 'source': None if source_model is None else base_sha}, 'gru', variant,
                            lambda: fit_local_gru(config, store, support_tensors, seed, variant, mode, candidate, source_model))
                        predictions = gru_predict(trained, store, test_tensors, batch)
                        row_scores(rows, config, date, validation_window, variant, mode, str(index), predictions)

                for j, horizon in enumerate(config.horizons):
                    candidates = config['catboost']['local_iterations']
                    cb, _, _ = cache_model(config, f'development/{target}/{date}/cat_{variant}_{horizon}', identity,
                        'catboost', variant, lambda: fit_catboost(config, [support], None, seed, variant, horizon, max(candidates)))
                    for iterations in candidates:
                        prediction = cat_predict(config, cb, validation_window, variant, horizon, iterations)
                        rows.append(development_row(date, target, variant, 'cat_local', str(iterations), horizon,
                                                    validation_window.targets[:, j], prediction))

        for date, _, window, _, _ in folds:
            for alpha in config['ses_alphas']:
                prediction = np.broadcast_to(window.ses(alpha)[:, None], window.targets.shape)
                row_scores(rows, config, date, window, 'baseline', 'SES', str(alpha), prediction)

    scores = pd.DataFrame(rows)
    scores_path = directory / 'development_scores.csv'
    scores.to_csv(scores_path, index=False)
    primary = config['primary_horizon_minutes']
    for variant in config.variants:
        subset = scores.loc[scores.variant.eq(variant)]
        settings = {}
        for mode in MODES:
            averaged = subset.loc[subset.method.eq(mode)].groupby('candidate').mae_pu.mean()
            winner = str(averaged.idxmin())
            settings[mode] = {'index': int(winner), 'mean_development_mae_pu': float(averaged[winner]),
                              **config['gru'][f'{mode}_candidates'][int(winner)]}
        settings['catboost'] = {}
        for horizon in config.horizons:
            table = subset.loc[subset.method.eq('cat_local') & subset.horizon_minutes.eq(horizon)]
            settings['catboost'][str(horizon)] = int(table.groupby('candidate').mae_pu.mean().idxmin())
        selections[variant] = settings

    ses = {}
    baseline = scores.loc[scores.method.eq('SES')]
    for horizon in config.horizons:
        values = baseline.loc[baseline.horizon_minutes.eq(horizon)].groupby('candidate').mae_pu.mean()
        ses[str(horizon)] = float(values.idxmin())
    development_summary = scores.loc[scores.horizon_minutes.eq(primary)].groupby(
        ['variant', 'method', 'candidate'], as_index=False).mae_pu.mean()
    saved = {'identity': identity, 'scores_sha256': file_hash(scores_path), 'variants': selections, 'ses': ses,
             'primary_horizon_minutes': primary, 'development_farms': config['source_farms'],
             'objective': 'Mean MAE per fold, farm and horizon',
             'primary_development_scores': development_summary.to_dict('records')}
    write_json(path, saved)
    print('Locked adaptation candidates and SES coefficients from the six development runs', flush=True)
    return saved


def model_jobs(config, store, source_windows, validation_windows, local_windows, selection, fit):
    source = store.pack(source_windows) if fit else None
    validation = store.pack(validation_windows) if fit else None
    adaptation = {days: store.pack([window]) for days, window in local_windows.items()} if fit else {}
    for seed in config['seeds']:
        for variant in config.variants:
            source_stem = f'final/{variant}/gru_zero_seed{seed}'
            yield ('gru', variant, 'gru_zero', 0, seed, None, source_stem,
                   (lambda: fit_source_gru(config, store, source, validation, seed, variant)) if fit else None, None)
            for days in config['adaptation_days']:
                for mode in MODES:
                    stem = f'final/{variant}/gru_{mode}_{days}d_seed{seed}'
                    yield ('gru', variant, f'gru_{mode}', days, seed, None, stem,
                           (lambda base: fit_local_gru(config, store, adaptation[days], seed, variant, mode,
                            selection['variants'][variant][mode], None if mode == 'scratch' else base)) if fit else None,
                           source_stem if mode != 'scratch' else None)
            for horizon in config.horizons:
                stem = f'final/{variant}/cat_zero_{horizon}m_seed{seed}'
                yield ('catboost', variant, 'cat_zero', 0, seed, horizon, stem,
                       (lambda: fit_catboost(config, source_windows, validation_windows, seed, variant, horizon,
                                                config['catboost']['iterations'])) if fit else None, None)
                for days in config['adaptation_days']:
                    stem = f'final/{variant}/cat_local_{horizon}m_{days}d_seed{seed}'
                    yield ('catboost', variant, 'cat_local', days, seed, horizon, stem,
                           (lambda: fit_catboost(config, [local_windows[days]], None, seed, variant, horizon,
                                                selection['variants'][variant]['catboost'][str(horizon)])) if fit else None, None)

        primary = config['primary_horizon_minutes']
        yield ('gru', 'core', 'gru_zero_mse', 0, seed, None, f'final/core/gru_zero_mse_seed{seed}',
               (lambda: fit_source_gru(config, store, source, validation, seed, 'core', squared=True)) if fit else None, None)
        yield ('catboost', 'core', 'cat_zero_rmse', 0, seed, primary, f'final/core/cat_zero_rmse_{primary}m_seed{seed}',
               (lambda: fit_catboost(config, source_windows, validation_windows, seed, 'core', primary,
                                     config['catboost']['iterations'], squared=True)) if fit else None, None)


def final_models(config, store, identity, selection, source=None, validation=None, locals_=None):
    pretrained = {}
    fit = source is not None
    for family, variant, method, days, seed, horizon, stem, callback, parent in model_jobs(
            config, store, source, validation, locals_, selection, fit):
        base, base_hash = pretrained[parent] if parent else (None, None)
        if family == 'gru' and days and fit:
            fn = lambda: callback(base)
        else:
            fn = callback
        model, meta, sha = cache_model(config, stem, {'run': identity, 'selection': object_hash(selection),
                                                    'source': base_hash}, family, variant, fn)
        if method == 'gru_zero':
            pretrained[stem] = model, sha
        yield family, variant, method, days, seed, horizon, model, stem, sha


def train(config, series, identity):
    sources = config['source_farms']
    store = TensorStore(config, series)
    selection = tune(config, series, store, identity)
    source = selected_windows(config, series, sources, config['train_start'], config['source_validation_start'],
                              'train_stride_minutes')
    validation = selected_windows(config, series, sources, config['source_validation_start'], config['adaptation_start'],
                                  'evaluation_stride_minutes')
    start = pd.Timestamp(config['adaptation_start'])
    locals_ = {days: series[config.external].select(config, start, start + days * DAY, config['adaptation_stride_minutes'])
               for days in config['adaptation_days']}
    inventory = []
    for family, variant, method, days, seed, horizon, _, stem, sha in final_models(
            config, store, identity, selection, source, validation, locals_):
        inventory.append({'stem': stem, 'sha256': sha, 'family': family, 'variant': variant, 'method': method,
                          'days': days, 'seed': seed, 'horizon': horizon})
    write_json(config.output / 'training.json', {'identity': identity, 'selection': object_hash(selection),
               'models': inventory, 'source_windows': {w.farm.name: len(w) for w in source},
               'validation_windows': {w.farm.name: len(w) for w in validation},
               'adaptation_windows': {days: len(window) for days, window in locals_.items()}})
    package_dir = Path(__file__).resolve().parent
    snapshot = config.output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for p in package_dir.glob('*.py'):
        shutil.copy2(p, snapshot / p.name)
    (config.output / 'requirements.lock.txt').write_text('\n'.join(
        f'{name}=={version}' for name, version in identity['packages'].items()) + '\n', encoding='utf-8')


def require_training(config, identity):
    training = json.loads((config.output / 'training.json').read_text(encoding='utf-8'))
    selection = json.loads((config.output / 'selection.json').read_text(encoding='utf-8'))
    if training['identity'] != identity or training['selection'] != object_hash(selection):
        raise ValueError('Saved training does not match the prepared data and locked protocol')
    return selection, training


def evaluate(config, farm, identity):
    selection, training = require_training(config, identity)
    test = farm.select(config, config['test_start'], config['test_end'], config['evaluation_stride_minutes'])
    predictions = config.output / 'predictions'
    predictions.mkdir(exist_ok=True)
    bundles = []
    times = pd.to_datetime(test.times)
    turbine_ids = farm.turbines[TURBINE].to_numpy()[test.turbine_index]

    def save(variant, method, days, seed, j, values):
        horizon = config.horizons[j]
        observed = np.isfinite(test.targets[:, j])
        if not observed.any() or not np.isfinite(values[observed]).all():
            raise ValueError(f'Invalid test predictions for {variant}/{method}/{horizon}')
        file = f'{variant}__{method}__{days}d__{horizon}m__seed{seed}.parquet'
        rows = pd.DataFrame({'origin': times[observed], 'turbine': turbine_ids[observed],
                             'rated_kw': test.ratings[observed], 'actual_pu': test.targets[observed, j],
                             'current_pu': test.persistence[observed], 'prediction_pu': values[observed]})
        if farm.weather_runs is not None:
            rows['weather_run'] = pd.to_datetime(farm.weather_runs[test.time_index[observed]])
        rows.to_parquet(predictions / file, index=False)
        bundles.append({'variant': variant, 'method': method, 'days': days, 'seed': seed, 'horizon': horizon,
                        'file': file, 'sha256': file_hash(predictions / file), 'windows': len(rows)})

    for j, horizon in enumerate(config.horizons):
        save('baseline', 'persistence', 0, -1, j, test.persistence)
        save('baseline', 'ses', 0, -1, j, test.ses(selection['ses'][str(horizon)]))

    store = TensorStore(config, {farm.name: farm})
    tensors = store.pack([test])
    fingerprints = {m['stem']: m['sha256'] for m in training['models']}
    for family, variant, method, days, seed, horizon, model, stem, sha in final_models(config, store, identity, selection):
        if fingerprints.get(stem) != sha:
            raise ValueError(f'Final model differs from the training manifest: {stem}')
        if family == 'catboost':
            j = config.horizons.index(horizon)
            save(variant, method, days, seed, j, cat_predict(config, model, test, variant, horizon))
        else:
            values = gru_predict(model, store, tensors, config['gru']['batch_size'])
            for j in range(len(config.horizons)):
                save(variant, method, days, seed, j, values[:, j])
    write_json(config.output / 'evaluation.json', {'identity': identity, 'training': object_hash(training),
               'farm': farm.name, 'predictions': bundles, 'origins': len(test),
               'start': config['test_start'], 'end_exclusive': config['test_end']})
    print(f'Evaluated {len(bundles)} sets of predictions on {farm.name}', flush=True)

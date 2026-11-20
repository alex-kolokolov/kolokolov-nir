from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..metrics import regression_metrics
from .artifacts import log_event, model_path, object_hash, verify_run, write_json
from .runner import load_farms
from .training import checkpoint_complete, predict_catboost, predict_gru


def paired_block_interval(error_gain: np.ndarray, times: pd.DatetimeIndex,
                          repetitions: int = 2000, seed: int = 42, confidence: float = 0.95):
    """Paired seven-day blocks; keep contemporaneous turbines in the same block.

    Input can be seed-averaged per-example loss differences. This is NOT a
    confidence interval for the population of independently trained models.
    """
    block = ((times - times.min()) // pd.Timedelta(days=7)).to_numpy()
    identifiers = np.unique(block)
    if len(identifiers) < 2:
        raise ValueError('At least two calendar blocks are needed for uncertainty estimation')
    sums = np.array([error_gain[block == i].sum() for i in identifiers])
    counts = np.array([(block == i).sum() for i in identifiers])
    selected = np.random.default_rng(seed).integers(
        0, len(identifiers), (repetitions, len(identifiers)))
    samples = sums[selected].sum(1) / counts[selected].sum(1)
    tail = (1-confidence)/2
    return np.quantile(samples, [tail, 1-tail]), len(identifiers)


def evaluate(config):
    manifest = verify_run(config)
    fingerprint = manifest['run_fingerprint']
    farms = load_farms(config, manifest)
    directory = config.output/'predictions'
    directory.mkdir(parents=True, exist_ok=True)
    metrics, turbine_metrics, paired, keys = [], [], [], []
    for source in config['farms']:
        for target in config['farms']:
            test = farms[target].select(config['test_start'], config['test_end'], config['evaluation_stride'])
            reference = np.repeat(test.persistence[:, None], len(config['horizons']), axis=1)
            tasks = [('persistence', -1, 0, None)]
            for seed in config['seeds']:
                if 'catboost' in config['methods']:
                    tasks.append(('catboost_source', seed, 0,
                                  model_path(config, source, source, 'catboost_source', seed)))
                if 'gru' in config['methods']:
                    tasks.append(('gru_source', seed, 0,
                                  model_path(config, source, source, 'gru_source', seed)))
                if source != target:
                    for days in config['adaptation_days']:
                        methods = ([] if 'catboost' not in config['methods'] else ['catboost_local'])
                        if 'gru' in config['methods']:
                            methods += ['gru_scratch', 'gru_finetune']
                        tasks += [(method, seed, days, model_path(config, source, target, method, seed, days))
                                  for method in methods]
            errors = {}
            identity = pd.DataFrame({'timestamp': test.times, 'turbine_id': test.turbines})
            test_hash = object_hash({'anchors': test.anchors.tolist(), 'farm': target,
                                     'prepared': manifest['prepared'][target]['files']})
            keys.append(dict(source=source, target=target, test_windows=len(test), test_identity_hash=test_hash))
            for method, seed, days, stem in tasks:
                if stem is None:
                    predictions = reference
                else:
                    if not checkpoint_complete(stem, fingerprint):
                        raise ValueError(f'Model training is incomplete: {stem}')
                    if method.startswith('catboost'):
                        predictions = predict_catboost(config, stem.with_suffix('.cbm'), test)
                    else:
                        predictions = predict_gru(config, stem.with_suffix('.pt'), test)
                if predictions.shape != test.targets.shape or not np.isfinite(predictions).all():
                    raise ValueError('Invalid or nonfinite predictions')
                errors.setdefault((method, days), []).append(np.abs(predictions-test.targets)*config['rated_power_kw'])
                base = dict(source=source, target=target, method=method, seed=seed,
                            adaptation_days=days, scenario='in_domain' if source == target else 'cross_farm',
                            test_windows=len(test), test_identity_hash=test_hash)
                detail = identity.copy()
                for j, horizon in enumerate(config['horizons']):
                    true = test.targets[:, j]
                    pred = predictions[:, j]
                    rated = np.full(len(test), config['rated_power_kw'])
                    values = regression_metrics(true, pred, rated)
                    baseline_mae = float((np.abs(reference[:, j]-true)*rated).mean())
                    values['skill_vs_persistence_pct'] = 100 * (1-values['mae_kw']/baseline_mae)
                    metrics.append({**base, 'horizon_minutes': horizon*10, **values})
                    for turbine in np.unique(test.turbines):
                        selected = test.turbines == turbine
                        turbine_metrics.append({**base, 'horizon_minutes': horizon*10,
                            'turbine_id': turbine, 'n': int(selected.sum()),
                            **regression_metrics(true[selected], pred[selected], rated[selected])})
                    detail[f'true_{horizon*10}m_kw'] = true*config['rated_power_kw']
                    detail[f'prediction_{horizon*10}m_kw'] = pred*config['rated_power_kw']
                filename = f'{source}__{target}__{method}__{days}d__seed{seed}.parquet'
                detail.to_parquet(directory/filename, index=False)
                print(f'Evaluated {source}->{target}: {method}, {days}d, seed {seed}', flush=True)
            averaged = {key: np.mean(value, axis=0) for key,value in errors.items()}
            comparisons = []
            for method, days in averaged:
                if method == 'persistence':
                    continue
                comparisons.append(((method, days), ('persistence', 0)))
                if method == 'gru_finetune':
                    comparisons += [((method, days), ('gru_scratch', days)),
                                    ((method, days), ('catboost_local', days)),
                                    ((method, days), ('gru_source', 0))]
            for candidate, baseline in comparisons:
                if baseline not in averaged:
                    continue
                difference = averaged[baseline] - averaged[candidate]
                for j, horizon in enumerate(config['horizons']):
                    primary = (candidate[0]=='gru_finetune' and baseline[0]=='catboost_local'
                               and candidate[1]==config['primary_adaptation_days']
                               and horizon*10==config['primary_horizon_minutes'])
                    confidence = 0.975 if primary else 0.95
                    interval, blocks = paired_block_interval(
                        difference[:, j], test.times, config['bootstrap_repetitions'], confidence=confidence)
                    paired.append(dict(source=source, target=target,
                        candidate=candidate[0], adaptation_days=candidate[1],
                        reference=baseline[0], reference_adaptation_days=baseline[1],
                        horizon_minutes=horizon*10, mean_mae_gain_kw=float(difference[:,j].mean()),
                        ci_low_kw=interval[0], ci_high_kw=interval[1], calendar_blocks=blocks,
                        confidence=confidence, primary=primary,
                        note='calendar-block CI of seed-mean paired errors; Bonferroni for two co-primary directions'))
    result = pd.DataFrame(metrics)
    result.to_csv(config.output/'metrics_by_seed.csv', index=False)
    pd.DataFrame(turbine_metrics).to_csv(config.output/'metrics_by_turbine.csv', index=False)
    pd.DataFrame(paired).to_csv(config.output/'paired_comparisons.csv', index=False)
    grouping = ['source','target','scenario','method','adaptation_days','horizon_minutes']
    summary = result.groupby(grouping).agg(
        mae_kw_mean=('mae_kw','mean'), mae_kw_std=('mae_kw','std'),
        rmse_kw_mean=('rmse_kw','mean'), nmae_pct_mean=('nmae_pct','mean'),
        r2_mean=('r2','mean'), skill_pct_mean=('skill_vs_persistence_pct','mean'),
        runs=('seed','size'), test_windows=('test_windows','first')).reset_index()
    summary.to_csv(config.output/'metrics_summary.csv', index=False)
    write_json(config.output/'evaluation.json', dict(run_fingerprint=fingerprint,
        completed=True, tests=keys, prediction_files=len(list(directory.glob('*.parquet'))),
        primary_comparison=dict(candidate='gru_finetune', reference='catboost_local',
            horizon_minutes=config['primary_horizon_minutes'], adaptation_days=config['primary_adaptation_days']),
        multiple_testing='two co-primary directions: 97.5% intervals (Bonferroni); remaining 95% intervals are exploratory'))
    log_event(config, stage='evaluate', completed=True)

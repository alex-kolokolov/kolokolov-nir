"""Paired evaluation on the external farm and a report generated from saved predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..artifacts import file_hash, object_hash, write_json
from .experiment import checked_file, require_training

GROUP = ['variant', 'method', 'days', 'horizon']
OBSERVATION = ['origin', 'turbine', 'rated_kw', 'actual_pu', 'current_pu']
REPORT_FILES = ('metrics_by_seed.csv', 'metrics_by_turbine.csv', 'metrics_summary.csv',
                'paired_comparisons.csv', 'report.md', 'adaptation_curves.png')


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def forecast_metrics(frame, threshold):
    actual = frame.actual_pu.to_numpy(dtype=float)
    current = frame.current_pu.to_numpy(dtype=float)
    prediction = frame.prediction_pu.to_numpy(dtype=float)
    rated = frame.rated_kw.to_numpy(dtype=float)
    error = prediction - actual
    change = actual - current
    predicted_change = prediction - current
    ramp = np.abs(change) >= threshold
    warning = np.abs(predicted_change) >= threshold
    correct = warning & ramp & (np.sign(predicted_change) == np.sign(change))
    return {'windows': len(frame), 'mae_kw': np.abs(error * rated).mean(),
            'nmae_pct': np.abs(error).mean() * 100, 'rmse_kw': np.sqrt(np.mean((error * rated) ** 2)),
            'bias_kw': (error * rated).mean(), 'ramp_nmae_pct': divide(np.abs(error[ramp]).sum() * 100, ramp.sum()),
            'other_nmae_pct': divide(np.abs(error[~ramp]).sum() * 100, (~ramp).sum()),
            'ramp_prevalence': ramp.mean(), 'ramp_ap': (average_precision_score(ramp, np.abs(predicted_change))
                                                        if ramp.any() else np.nan),
            'signed_ramp_recall': divide(correct.sum(), ramp.sum()),
            'signed_ramp_precision': divide(correct.sum(), warning.sum())}


def block_interval(a, b, times, start, days, repetitions, seed, confidence):
    """Resample calendar blocks with all contemporary turbines kept together."""
    block = (pd.DatetimeIndex(times) - pd.Timestamp(start)) // pd.Timedelta(days=days)
    _, indices = np.unique(block, return_inverse=True)
    counts = np.bincount(indices)
    if len(counts) < 2:
        return np.nan, np.nan, len(counts)
    delta = np.bincount(indices, weights=(a - b) * 100)
    draw = np.random.default_rng(seed).integers(len(counts), size=(repetitions, len(counts)))
    boot = delta[draw].sum(axis=1) / counts[draw].sum(axis=1)
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [tail, 1 - tail])
    return float(lo), float(hi), len(counts)


def load_scores(config, evaluation):
    folder = config.output / 'predictions'
    references, errors, aggregate, by_turbine = {}, {}, [], []
    threshold = config['ramp_threshold_pu']
    observation_columns = [*OBSERVATION, 'weather_run'] if config['weather'] else OBSERVATION
    for entry in evaluation['predictions']:
        keys = {name: entry[name] for name in [*GROUP, 'seed']}
        horizon = keys['horizon']
        file = checked_file(folder, entry['file'], entry['sha256'])
        rows = pd.read_parquet(file)
        identity = rows[observation_columns]
        if len(rows) != entry['windows'] or not np.isfinite(rows.prediction_pu).all():
            raise ValueError(f'Prediction integrity failed: {file}')
        if horizon not in references:
            if (rows.duplicated(['origin', 'turbine']).any()
                    or rows.origin.min() < pd.Timestamp(config['test_start'])
                    or (rows.origin + pd.Timedelta(minutes=horizon)).max() >= pd.Timestamp(config['test_end'])):
                raise ValueError(f'Evaluation origins or targets cross the declared period: {file}')
            if config['weather']:
                delay = pd.Timedelta(hours=config['weather']['publication_delay_hours'])
                if ((rows.weather_run + delay) > rows.origin).any():
                    raise ValueError(f'Forecast uses a weather run before its availability: {file}')
            references[horizon] = identity
        if not identity.equals(references[horizon]):
            raise ValueError(f'Forecast methods were scored on different observations at horizon {horizon}')
        aggregate.append({**keys, **forecast_metrics(rows, threshold)})
        for turbine, group in rows.groupby('turbine', sort=False):
            delta = (group.prediction_pu - group.actual_pu).abs()
            by_turbine.append({**keys, 'turbine': turbine, 'windows': len(group), 'nmae_pct': 100 * delta.mean(),
                               'mae_kw': (delta * group.rated_kw).mean()})
        key = tuple(keys[name] for name in GROUP)
        error = (rows.prediction_pu - rows.actual_pu).abs().to_numpy(dtype=float)
        if key in errors:
            accumulated, count = errors[key]
            errors[key] = accumulated + error, count + 1
        else:
            errors[key] = error, 1
    for key, (error, count) in errors.items():
        errors[key] = error / count
    return pd.DataFrame(aggregate), pd.DataFrame(by_turbine), references, errors


def paired_comparisons(config, references, errors):
    rows = []
    days = config['primary_adaptation_days']
    primary_horizon = config['primary_horizon_minutes']

    def compare(label, a, b, primary=False):
        horizon = a[-1]
        confidence = .975 if primary else .95
        lo, hi, block_count = block_interval(errors[a], errors[b], references[horizon].origin, config['test_start'],
            config['bootstrap_block_days'], config['bootstrap_repetitions'], config['development_seed'], confidence)
        rows.append({'comparison': label, 'variant_a': a[0], 'method_a': a[1], 'days_a': a[2],
                     'variant_b': b[0], 'method_b': b[1], 'days_b': b[2], 'horizon_minutes': horizon,
                     'delta_nmae_pp': 100 * (errors[a] - errors[b]).mean(), 'ci_low_pp': lo, 'ci_high_pp': hi,
                     'skill_pct': 100 * (1 - divide(errors[a].mean(), errors[b].mean())),
                     'confidence': confidence, 'blocks': block_count, 'primary': primary})

    for horizon in config.horizons:
        for variant in config.variants:
            head = (variant, 'gru_head', days, horizon)
            primary = variant == 'scada' and horizon == primary_horizon
            compare('Transfer effect at equal local data budget', head, (variant, 'gru_scratch', days, horizon), primary)
            compare('Adapted head versus SES', head, ('baseline', 'ses', 0, horizon), primary)
            compare('Adapted head versus persistence', head, ('baseline', 'persistence', 0, horizon))
            compare('Adapted head versus source without adaptation', head, (variant, 'gru_zero', 0, horizon))
            compare('Adapted head versus full fine-tuning', head, (variant, 'gru_full', days, horizon))
            compare('Adapted head versus local CatBoost', head, (variant, 'cat_local', days, horizon))
            compare('Source GRU versus SES', (variant, 'gru_zero', 0, horizon), ('baseline', 'ses', 0, horizon))
            compare('Source CatBoost versus SES', (variant, 'cat_zero', 0, horizon), ('baseline', 'ses', 0, horizon))
        for method, budget in [('gru_zero', 0), ('gru_head', days), ('cat_zero', 0), ('cat_local', days)]:
            compare('Additional SCADA', ('scada', method, budget, horizon), ('core', method, budget, horizon))
            if 'weather' in config.variants:
                compare('Operational weather', ('weather', method, budget, horizon), ('scada', method, budget, horizon))
    compare('GRU MAE versus MSE', ('core', 'gru_zero', 0, primary_horizon), ('core', 'gru_zero_mse', 0, primary_horizon))
    compare('CatBoost MAE versus RMSE', ('core', 'cat_zero', 0, primary_horizon), ('core', 'cat_zero_rmse', 0, primary_horizon))
    return pd.DataFrame(rows)


def adaptation_plot(config, summary, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = {'gru_zero': 'GRU без адаптации', 'gru_head': 'GRU: голова', 'gru_full': 'GRU: вся сеть',
             'gru_scratch': 'GRU с нуля', 'cat_local': 'Локальный CatBoost', 'ses': 'SES'}
    variants = ('scada', 'weather') if 'weather' in config.variants else ('scada',)
    fig, axes = plt.subplots(len(variants), len(config.horizons), squeeze=False,
                             figsize=(4.9 * len(config.horizons), 3.35 * len(variants)), layout='constrained')
    for i, variant in enumerate(variants):
        for j, horizon in enumerate(config.horizons):
            ax = axes[i, j]
            table = summary[summary.horizon.eq(horizon)]
            for method, label in names.items():
                selected = table[(table.variant.eq(variant) | table.variant.eq('baseline')) & table.method.eq(method)]
                if method in ('ses', 'gru_zero'):
                    ax.axhline(selected.nmae_pct_mean.item(), linestyle='--', linewidth=1.2, label=label)
                else:
                    ax.plot(selected.days, selected.nmae_pct_mean, marker='o', markersize=3, label=label)
            ax.set_title(f'{"SCADA + IFS" if variant == "weather" else "SCADA"} · {horizon} мин')
            ax.set_xticks(config['adaptation_days'])
            ax.set_xlabel('Дней данных целевого парка')
            ax.set_ylabel('nMAE, % номинала')
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'Внешний тест · {config.external}', fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_report(config, scores, paired, references, selection, training):
    primary = config['primary_horizon_minutes']
    days = config['primary_adaptation_days']
    labels = {'persistence': 'Persistence', 'ses': 'Экспоненциальное сглаживание', 'cat_zero': 'CatBoost без адаптации',
              'cat_local': 'CatBoost на данных нового парка', 'gru_zero': 'GRU без адаптации',
              'gru_head': 'GRU с адаптацией головы', 'gru_full': 'GRU с адаптацией всей сети', 'gru_scratch': 'GRU с нуля'}
    variant_labels = {'baseline': 'Мощность', 'core': 'Базовый SCADA', 'scada': 'Расширенный SCADA', 'weather': 'SCADA + IFS'}
    lines = [f'# Перенос прогноза мощности на {config.external}', '',
             f'Целевой парк в периоде с {config["test_start"]} по {config["test_end"]} UTC, правая граница исключена. '
             'Источник обучения – ' + ' и '.join(config['source_farms']) + '. Параметры адаптации и сглаживания '
             'выбраны до оценки целевого парка на календарных окнах разработки.', '',
             'Окна по горизонтам – ' + ', '.join(f'{h} мин: {len(v):,}' for h, v in references.items()) + '. '
             f'Число турбин в тесте – {references[primary].turbine.nunique()}. '
             f'Число окон при бюджете адаптации {days} дн. – {training["adaptation_windows"][str(days)]:,}. '
             'nMAE выражена в процентах паспортной мощности, MAE и RMSE – в кВт.', '',
             '## Основные сравнения', '',
             f'Расширенный SCADA, {primary} минут, {days} дней. Отрицательная разность означает меньшую ошибку '
            'для адаптации головы. Интервалы получены парным бутстрепом календарных блоков '
             f'по {config["bootstrap_block_days"]} дней с сохранением всех турбин в блоке.', '',
             '| Сравнение с адаптацией головы | Δ nMAE, п.п. | Парный 97,5% интервал, п.п. | Skill, % |',
             '| --- | ---: | ---: | ---: |']
    for row in paired[paired.primary].itertuples():
        lines.append(f'| {labels[row.method_b]} | {row.delta_nmae_pp:+.3f} | '
                     f'[{row.ci_low_pp:+.3f}, {row.ci_high_pp:+.3f}] | {row.skill_pct:+.2f} |')
    lines.extend(['', 'Остальные сравнения приведены с описательными 95% интервалами. '
                  'Интервалы оценивают временной разброс на одном внешнем парке.', '',
                  '## Ошибка на полном тесте', ''])
    for horizon in config.horizons:
        lines.extend([f'### {horizon} минут', '',
                      '| Входы | Метод | Дни | MAE, кВт | RMSE, кВт | nMAE, % | Skill к SES, % |',
                      '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
        for row in scores.loc[scores.horizon.eq(horizon) & scores.days.isin([0, days])].itertuples():
            if row.method not in labels:
                continue
            lines.append(f'| {variant_labels[row.variant]} | {labels[row.method]} | {row.days} | {row.mae_kw_mean:.2f} | '
                         f'{row.rmse_kw_mean:.2f} | {row.nmae_pct_mean:.3f} | {row.skill_vs_ses_pct:+.2f} |')
        lines.append('')
    lines.extend(['## Вклад входов и целевой функции', '',
                  '| Изменение | Метод | Горизонт, мин | Δ nMAE, п.п. | Парный 95% интервал, п.п. |',
                  '| --- | --- | ---: | ---: | ---: |'])
    names = {'Additional SCADA': 'Расширение SCADA', 'Operational weather': 'Добавление IFS',
             'GRU MAE versus MSE': 'MAE вместо MSE', 'CatBoost MAE versus RMSE': 'MAE вместо RMSE'}
    for row in paired[paired.comparison.isin(names)].itertuples():
        lines.append(f'| {names[row.comparison]} | {labels[row.method_a]} | {row.horizon_minutes} | '
                     f'{row.delta_nmae_pp:+.3f} | [{row.ci_low_pp:+.3f}, {row.ci_high_pp:+.3f}] |')
    prevalence = scores[scores.method.eq('persistence')].sort_values('horizon')
    share = ', '.join(f'{row.horizon} мин – {row.ramp_prevalence_mean:.1%}' for row in prevalence.itertuples())
    lines.extend(['', '## Резкие изменения', '',
                  f'Событие определяется как итоговое изменение на горизонте не менее {config["ramp_threshold_pu"]:.0%} номинала. '
                  f'Доля таких окон: {share}. Режимы определены по факту для диагностики. '
                  'Полная nMAE равна среднему двух ошибок с весами по долям окон. '
                  'AP оценивает ранжирование событий по модулю точечного прогноза изменения, '
                  'а recall требует правильного знака и амплитуды выше порога.', '',
                  '| Входы | Метод | Горизонт, мин | nMAE вне событий, % | nMAE на событиях, % | AP | Recall, % |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    subset = scores[scores.method.isin(['persistence', 'ses', 'gru_zero', 'gru_head', 'cat_zero'])
                    & scores.variant.ne('core') & scores.days.isin([0, days])]
    for row in subset.itertuples():
        lines.append(f'| {variant_labels[row.variant]} | {labels[row.method]} | {row.horizon} | '
                     f'{row.other_nmae_pct_mean:.2f} | {row.ramp_nmae_pct_mean:.2f} | '
                     f'{row.ramp_ap_mean:.3f} | {100 * row.signed_ramp_recall_mean:.2f} |')
    lines.extend(['', 'Разложение разности с экспоненциальным сглаживанием по режимам. '
                  'Вклад каждого режима равен разности nMAE внутри него, умноженной на долю окон. '
                  'Отрицательный вклад уменьшает общую ошибку.', '',
                  '| Входы | Метод | Дни | Горизонт, мин | Вклад вне событий, п.п. | Вклад событий, п.п. | Δ nMAE, п.п. |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    indexed = scores.set_index(GROUP)
    regimes = [('scada', 'gru_head', days, primary)]
    if config['weather']:
        regimes.append(('weather', 'gru_zero', 0, max(config.horizons)))
    for variant, method, budget, horizon in regimes:
        row = indexed.loc[(variant, method, budget, horizon), :]
        ses = indexed.loc[('baseline', 'ses', 0, horizon), :]
        rate = ses.ramp_prevalence_mean
        other = (1 - rate) * (row.other_nmae_pct_mean - ses.other_nmae_pct_mean) if rate < 1 else 0.0
        ramps = rate * (row.ramp_nmae_pct_mean - ses.ramp_nmae_pct_mean) if rate > 0 else 0.0
        lines.append(f'| {variant_labels[variant]} | {labels[method]} | {budget} | {horizon} | '
                     f'{other:+.3f} | {ramps:+.3f} | {row.nmae_pct_mean - ses.nmae_pct_mean:+.3f} |')
    lines.extend(['', '## Воспроизведение', '',
                  'На обучении и тесте используются действительные измерения отпуска мощности. '
                  'Для показателей потерь сохранено покрытие по наблюдениям. Время их публикации в источниках неизвестно. '
                  'Признаки контроллера берутся из завершившихся интервалов SCADA.', '',
                  'В `selection.json` записаны параметры, выбранные по исходным паркам. '
                  'Все ошибки по seed, турбинам, горизонту и бюджету адаптации хранятся в таблицах `metrics_*.csv`. '
                  'Прогнозы с исходными наблюдениями и временем прогона погоды находятся в `predictions/`.', '',
                  f'Коэффициенты SES по горизонтам – {selection["ses"]}.', ''])
    if config['weather']:
        lines.extend(['Data from ECMWF via [WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#ifs-hres). '
                      'Операционный IFS HRES, компоненты ветра на 10 м, сетка 1,5° × 1,5°, инициализации 00/12 UTC. '
                      f'Принята задержка доступности {config["weather"]["publication_delay_hours"]} ч после инициализации.', ''])
    return '\n'.join(lines)


def report(config, identity):
    selection, training = require_training(config, identity)
    directory = config.output
    evaluation = json.loads((directory / 'evaluation.json').read_text(encoding='utf-8'))
    if evaluation['identity'] != identity or evaluation['training'] != object_hash(training):
        raise ValueError('Saved evaluation belongs to a different training run')
    by_seed, by_turbine, references, errors = load_scores(config, evaluation)
    by_seed.to_csv(directory / 'metrics_by_seed.csv', index=False)
    by_turbine.to_csv(directory / 'metrics_by_turbine.csv', index=False)
    metrics = sorted(set(by_seed) - set([*GROUP, 'seed']))
    grouped = by_seed.groupby(GROUP)[metrics].agg(['mean', 'std'])
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    summary = grouped.reset_index()
    for reference in ('persistence', 'ses'):
        baseline = summary.loc[summary.method.eq(reference)].set_index('horizon').nmae_pct_mean
        summary[f'skill_vs_{reference}_pct'] = 100 * (1 - summary.nmae_pct_mean / summary.horizon.map(baseline).replace(0, np.nan))
    summary.to_csv(directory / 'metrics_summary.csv', index=False)
    paired = paired_comparisons(config, references, errors)
    paired.to_csv(directory / 'paired_comparisons.csv', index=False)
    path = directory / 'report.md'
    path.write_text(render_report(config, summary, paired, references, selection, training), encoding='utf-8')
    adaptation_plot(config, summary, directory / 'adaptation_curves.png')
    snapshot = directory / 'source_snapshot' / Path(__file__).name
    snapshot.parent.mkdir(exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(directory / 'report_metadata.json', {'evaluation': object_hash(evaluation),
               'analysis_code': file_hash(snapshot),
               'files': {name: file_hash(directory / name) for name in REPORT_FILES}})
    print(f'Report written to {path}', flush=True)

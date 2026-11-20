from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import file_hash, verify_run, write_json
from ..research_data import RAW_COLUMNS


def markdown_table(frame):
    columns = list(frame.columns)
    result = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False, name=None):
        result.append('| '+' | '.join(f'{v:.3f}' if isinstance(v, (float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(result)


def generate_report(config):
    manifest = verify_run(config)
    evaluation = json.loads((config.output/'evaluation.json').read_text(encoding='utf-8'))
    if evaluation['run_fingerprint'] != manifest['run_fingerprint']:
        raise ValueError('Evaluation belongs to another experiment')
    summary = pd.read_csv(config.output/'metrics_summary.csv')
    paired = pd.read_csv(config.output/'paired_comparisons.csv')
    splits = pd.read_csv(config.output/'splits.csv')
    raw = pd.read_parquet(config.source, columns=['farm', 'timestamp']+list(RAW_COLUMNS[3:]))
    quality = []
    for (farm, month), group in raw.groupby(['farm', pd.Grouper(key='timestamp', freq='MS')]):
        for signal in RAW_COLUMNS[3:]:
            missing = int(group[signal].isna().sum())
            quality.append(dict(farm=farm, month=str(month.date()), signal=signal,
                                rows=len(group), missing_rows=missing,
                                missing_pct=100*missing/len(group)))
    quality = pd.DataFrame(quality)
    quality.to_csv(config.output/'signal_quality.csv', index=False)
    recent_quality = quality[quality.month == quality.month.max()][['farm','month','signal','missing_pct']]
    del raw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    methods = {'catboost_local':'CatBoost local', 'gru_scratch':'GRU from scratch',
               'gru_finetune':'GRU fine-tuned'}
    horizons = [h*10 for h in config['horizons']]
    fig, axes = plt.subplots(2, len(horizons), figsize=(4*len(horizons),7), squeeze=False)
    for i, source in enumerate(config['farms']):
        target = next(f for f in config['farms'] if f != source)
        subset = summary[(summary.source==source)&(summary.target==target)]
        for j,horizon in enumerate(horizons):
            ax = axes[i,j]
            for method,label in methods.items():
                values = subset[(subset.method==method)&(subset.horizon_minutes==horizon)].sort_values('adaptation_days')
                if values.empty:
                    continue
                ax.errorbar(values.adaptation_days,values.mae_kw_mean,
                            yerr=values.mae_kw_std.fillna(0),marker='o',capsize=3,label=label)
            for method,label,color in [('persistence','Persistence','gray'),('gru_source','GRU zero-shot','purple')]:
                values=subset[(subset.method==method)&(subset.horizon_minutes==horizon)]
                if len(values):
                    ax.axhline(values.mae_kw_mean.iloc[0],color=color,linestyle='--',label=label)
            ax.set(title=f'{source} → {target}\n{horizon} min',xlabel='Local calendar days',ylabel='MAE, kW')
            ax.set_xticks(config['adaptation_days'])
            ax.grid(alpha=.2)
    handles, labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.savefig(config.output/'adaptation_curves.png',dpi=170)
    plt.close(fig)
    primary=paired[(paired.candidate=='gru_finetune')&(paired.reference=='catboost_local')&
                   (paired.adaptation_days==config['primary_adaptation_days'])&
                   (paired.horizon_minutes==config['primary_horizon_minutes'])]
    primary_text=[]
    practical_text=[]
    for row in primary.itertuples():
        if row.ci_low_kw > 0:
            outcome='наблюдается преимущество GRU после адаптации'
        elif row.ci_high_kw < 0:
            outcome='наблюдается преимущество локального CatBoost'
        else:
            outcome='интервал включает ноль; устойчивое различие этим тестом не установлено'
        primary_text.append(f'- {row.source} → {row.target}: разность MAE CatBoost − GRU '
            f'{row.mean_mae_gain_kw:.3f} кВт, интервал '
            f'[{row.ci_low_kw:.3f}; {row.ci_high_kw:.3f}] кВт; {outcome}.')
        domain=summary[(summary.source==row.source)&(summary.target==row.target)&
                       (summary.horizon_minutes==config['primary_horizon_minutes'])]
        neural=domain[(domain.method=='gru_finetune')&
                      (domain.adaptation_days==config['primary_adaptation_days'])].iloc[0]
        naive=domain[domain.method=='persistence'].iloc[0]
        mae_change=100*(neural.mae_kw_mean/naive.mae_kw_mean-1)
        rmse_change=100*(1-neural.rmse_kw_mean/naive.rmse_kw_mean)
        practical_text.append(f'- {row.source} → {row.target}: MAE GRU '
            f'{neural.mae_kw_mean:.2f} кВт против Persistence {naive.mae_kw_mean:.2f} кВт '
            f'({"хуже" if mae_change>0 else "лучше"} на {abs(mae_change):.2f}%); '
            f'RMSE {neural.rmse_kw_mean:.2f} против {naive.rmse_kw_mean:.2f} кВт '
            f'({"снижение" if rmse_change>0 else "рост"} на {abs(rmse_change):.2f}%).')
    baseline=summary[summary.method.isin(['persistence','catboost_source','gru_source'])][
        ['source','target','method','horizon_minutes','mae_kw_mean','mae_kw_std','rmse_kw_mean','nmae_pct_mean','skill_pct_mean']]
    seven=summary[(summary.source!=summary.target)&
                  (summary.adaptation_days==config['primary_adaptation_days'])][
        ['source','target','method','horizon_minutes','mae_kw_mean','mae_kw_std','rmse_kw_mean','skill_pct_mean']]
    audits=pd.DataFrame([item['audit'] for item in manifest['prepared'].values()])[
        ['farm','rows','turbines','negative_power_rows','invalid_power_rows','invalid_input_rows','time_gaps','valid_windows_stride1']]
    report=f'''# Перенос прогноза мощности между {' и '.join(config['farms'])}

Эксперимент `{config['name']}` выполнен единым настраиваемым пайплайном на заново
подготовленных исходных SCADA-данных. Использованы Persistence, CatBoost и новая
небольшая GRU. Старые нейросетевые веса и старые нормировки не использовались.

## Главный результат заранее выбранного сравнения

Сравниваются GRU после {config['primary_adaptation_days']} дней адаптации и
CatBoost, обученный на тех же местных данных, горизонт
{config['primary_horizon_minutes']} минут. Положительная разность означает
меньшую ошибку GRU. Результат приведён для обоих направлений, без выбора удобного
тестового объекта.

{chr(10).join(primary_text) if primary_text else 'Это сравнение отключено выбранным набором методов.'}

Победа над локальным CatBoost не означает победу над простым прогнозом.
Для той же заранее выбранной точки получен следующий практический контроль:

{chr(10).join(practical_text)}

MAE описывает обычный абсолютный промах, RMSE сильнее штрафует крупные ошибки.
Обе обучаемые модели оптимизируют квадратичную ошибку, поэтому улучшение RMSE
при ухудшении MAE возможно. Проценты изменения RMSE здесь описательные;
отдельная проверка значимости RMSE не проводилась.

Интервалы построены парным bootstrap календарных блоков по 7 дней; ошибки
предварительно усреднены по начальным значениям. Это неопределённость во времени
для имеющихся повторов, а не универсальная оценка всех возможных обучений.
Для двух основных направлений используются интервалы 97,5% с поправкой
Бонферрони; остальные интервалы имеют уровень 95%.
Остальные сравнения разведочные. Не следует выбирать из таблиц лучший результат
и объявлять его заранее подтверждённой гипотезой.

## Подготовка и качество данных

Данные повторно прочитаны из исходных версионированных архивов Cubico. Имена
столбцов проверяются; позиционные номера столбцов не используются. Контрольные
суммы исходников и паспортная мощность проверены. Пропуски и недоступные измерения
маскируются с сохранением временных меток. Входные окна и все цели обязаны идти
с шагом 10 минут внутри одной турбины.

{markdown_table(audits)}

Доступность отдельных датчиков проверена также помесячно. Ниже доля отсутствующих
значений в последнем месяце исходного периода; полный реестр находится
в `signal_quality.csv`. Отсутствие обязательного датчика сокращает покрытие
прогнозом даже при доступной мощности. Это отдельная характеристика данных,
её нельзя скрывать одной средней ошибкой модели.

{markdown_table(recent_quality)}

Целевая величина — неотрицательная генерируемая мощность. Малые отрицательные
измерения мощности заменены нулём; это не прогноз собственного потребления.
Значения выше {config['preprocessing']['maximum_power_pu']*100:.0f}% паспортной
мощности и другие физические ограничения определены в конфигурации.
Окна с отсутствующим обязательным входом исключаются одинаково для всех методов.
Это оценка на пригодных окнах; произвольные пропуски в работе сервиса не моделируются.
Список входов: {', '.join(config['features'])}. История — {config['lookback']*10/60:g} часов.
Все масштабы фиксированы физически; статистики по будущему ряду не рассчитываются.

Показанное число пригодных окон относится ко всему временному массиву при
шаге 10 минут. Реальное число обучающих и тестовых примеров после календарного
разбиения и выбора шага приведено далее.

## Разбиение

{markdown_table(splits)}

Начало входного окна и конец всех будущих целей находятся внутри своего
интервала. Обучающие цели не пересекают валидацию, адаптацию или тест.
Для всех объёмов местных данных тест одинаков. Прогнозы выдаются с шагом
{config['evaluation_stride']*10} минут. На обучении исходной модели шаг
{config['train_stride']*10} минут. Гиперпараметры зафиксированы до оценки теста.

## Качество без местной адаптации

Строки с одинаковыми source и target — прогноз на своей площадке; разные
source и target — перенос на другой парк. Стандартное отклонение считается по
повторным обучениям; у Persistence оно не определено, поскольку обучение отсутствует.
Skill — снижение MAE в процентах относительно Persistence; отрицательное значение
означает ухудшение.

{markdown_table(baseline.fillna('—'))}

## Сравнение после местного обучения

GRU scratch обучается только на местных данных; GRU finetune начинает с модели
другого парка. Объём местных меток одинаков. Для малых объёмов используется
заранее фиксированное число эпох без дополнительной локальной валидационной
выборки. Приведена основная точка кривой адаптации.

{markdown_table(seven.fillna('—'))}

![Зависимость ошибки от объёма местных данных](adaptation_curves.png)

График показывает среднее MAE и разброс между начальными значениями, а не
доверительный интервал. Полные метрики всех объёмов и горизонтов находятся
в `metrics_summary.csv`, отдельные обучения — в `metrics_by_seed.csv`,
турбины — в `metrics_by_turbine.csv`. Результаты парных сравнений сохранены
в `paired_comparisons.csv`.

## Воспроизводимость и границы вывода

Использованы начальные значения {config['seeds']}. Веса, журналы обучения,
пообъектные прогнозы, точная конфигурация, снимок кода и версии зависимостей
сохранены рядом с отчётом. Хеш эксперимента: `{manifest['run_fingerprint']}`.
GPU-обучение CatBoost не гарантирует побитового совпадения; статистическую
устойчивость нужно оценивать по повторам.

Рассмотрены два парка одного производителя, один календарный период адаптации
и один будущий квартал. Эти данные не доказывают перенос между производителями,
работу во все сезоны или преимущество над всеми возможными настройками CatBoost.
Параметры моделей здесь заданы заранее, широкого равного поиска гиперпараметров
не проводилось. При отрицательном результате гипотезу преимущества GRU считать
неподтверждённой и сохранять исходные числа.

Источники: [Kelmarsh](https://zenodo.org/records/7212475),
[Penmanshiel](https://zenodo.org/records/5946808).
'''
    (config.output/'report.md').write_text(report,encoding='utf-8')
    snapshot = config.output/'source_snapshot/wind_pipeline/research/report.py'
    shutil.copy2(Path(__file__), snapshot)
    write_json(config.output/'report_metadata.json', dict(
        run_fingerprint=manifest['run_fingerprint'], reporting_code_sha256=file_hash(Path(__file__)),
        metric_summary_sha256=file_hash(config.output/'metrics_summary.csv'),
        report_sha256=file_hash(config.output/'report.md'),
        plot_sha256=file_hash(config.output/'adaptation_curves.png')))
    print(config.output/'report.md',flush=True)

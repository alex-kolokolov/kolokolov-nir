"""Create a self-contained report from the saved evaluation tables."""
from pathlib import Path


LABELS = {
    "persistence": "Persistence",
    "ses": "Экспоненциальное сглаживание",
    "gru": "GRU",
    "blend_persistence": "Фиксированная смесь Persistence + GRU",
    "blend_ses": "Фиксированная смесь сглаживания + GRU",
    "activity_persistence": "Правило по прошлому часу + Persistence",
    "activity_ses": "Правило по прошлому часу + сглаживание",
    "classifier_persistence": "Классификатор + Persistence",
    "classifier_ses": "Классификатор + сглаживание",
    "classifier": "CatBoost",
    "activity": "Изменчивость за прошлый час",
    "never": "Без предупреждений",
}


def dec(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def write_report(config, detectors, forecasts, pairs, policies):
    spec = config.spec
    primary = spec["primary_horizon_minutes"]
    threshold = spec["event_threshold_pu"]
    rated = config.parent["rated_power_kw"]
    report = [
        "# Прогноз резких изменений мощности и перенос переключения между моделями",
        "",
        f"Разведочная проверка после первого эксперимента с переносом. Резкое изменение — "
        f"разница между мощностью в момент прогноза и в конце горизонта не менее "
        f"{threshold * 100:g}% паспортной мощности ({threshold * rated:g} кВт) по модулю. "
        "Такое определение обнаруживает изменение между двумя точками: всплеск, "
        "который успел исчезнуть до конца горизонта, в метку не входит.",
        "",
        "Проверяются две разные задачи: качество предупреждений о событии и точность "
        "использования этих предупреждений для маршрутизации прогноза. Ни одно решение "
        "на тесте не использует наблюдаемую будущую мощность.",
        "",
        "## Данные и настройка",
        "",
        f"- CatBoost классификатор обучен на исходном парке до "
        f"{config.parent['train_end']} на истории мощности и скорости ветра.",
        f"- На исходном парке с {spec['calibration_start']} по {spec['calibration_end']} "
        f"(правая граница исключена) выбраны коэффициент сглаживания, вес постоянной "
        "смеси и пороги для обоих правил. Порог предупреждения выбирался по F1, "
        "порог для переключения прогноза — по MAE на всех окнах. Это разные решения.",
        f"- Проверка выполнена на другом парке с {spec['test_start']} по "
        f"{spec['test_end']} (правая граница исключена). Апрель–июнь, использованные "
        "ранее для исследования гипотезы, в этом расчёте не участвуют.",
        f"- Ветки для резких изменений: три существующие GRU после "
        f"{spec['adaptation_days']} дней местной адаптации. На исходном парке при "
        "настройке используется GRU до адаптации; возможное изменение ошибок "
        "после адаптации является частью проверки переноса. MAE, RMSE и доли "
        "переключений усреднены по трём запускам (или всем seed, указанным в базовом "
        "эксперименте). Детектор обучен один раз на направление и горизонт.",
        "- В спокойной ветке простое экспоненциальное сглаживание продолжает "
        "текущую сглаженную оценку уровня мощности. Все методы видят одну "
        "выборку пригодных окон исходного пайплайна; пропуски датчиков за её "
        "пределами здесь не оценивались.",
        "",
        "## Результаты",
    ]
    for metadata in policies:
        source, target, horizon = (metadata[key] for key in
                                   ("source", "target", "horizon_minutes"))
        d = detectors.query("source == @source and target == @target and horizon_minutes == @horizon")
        f = forecasts.query("source == @source and target == @target and horizon_minutes == @horizon")
        p = pairs.query("source == @source and target == @target and horizon_minutes == @horizon")
        report.extend([
            "",
            f"### {source} → {target}, {horizon} мин",
            "",
            f"Оценено {metadata['test_windows']} окон, турбин: {metadata['test_turbines']}. "
            f"Фактический диапазон точек прогноза: {metadata['first_test_origin']} — "
            f"{metadata['last_test_origin']} UTC.",
            "",
            "| Детектор | AP | Precision | Recall | F1 | Ложные тревоги / без события |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in d.itertuples():
            ap = "—" if row.average_precision != row.average_precision else dec(row.average_precision, 3)
            report.append(
                f"| {LABELS[row.detector]} | {ap} | "
                f"{dec(row.precision * 100, 1)}% | {dec(row.recall * 100, 1)}% | "
                f"{dec(row.f1, 3)} | {dec(row.false_positive_rate * 100, 1)}% |"
            )
        report.extend([
            "",
            f"Доля событий: {dec(d.prevalence.iloc[0] * 100, 1)}% (это AP "
            "для одинакового балла на всех окнах). Precision — доля верных "
            "предупреждений, recall — доля найденных событий. Нулевые значения "
            "при отсутствии предупреждений показаны как ноль. Перекрывающиеся "
            "окна считаются по точкам прогноза, не как отдельные физические события.",
            "",
            "| Прогноз | MAE, кВт | RMSE, кВт | Использована GRU |",
            "| --- | ---: | ---: | ---: |",
        ])
        for row in f.itertuples():
            share = "—" if row.fast_branch_share != row.fast_branch_share else dec(row.fast_branch_share * 100, 1) + "%"
            report.append(f"| {LABELS[row.policy]} | {dec(row.mae_kw)} | {dec(row.rmse_kw)} | {share} |")
        report.extend([
            "",
            "| Сравнение | Разность MAE, кВт | Парный интервал, кВт |",
            "| --- | ---: | ---: |",
        ])
        for row in p.itertuples():
            report.append(
                f"| {LABELS[row.candidate]} минус {LABELS[row.reference]} | "
                f"{dec(row.mae_difference_kw)} | "
                f"[{dec(row.lo_kw)}; {dec(row.hi_kw)}] ({dec(row.confidence * 100, 1)}%) |"
            )
        main = p.query("primary")
        if not main.empty:
            row = next(main.itertuples())
            if row.hi_kw < 0:
                conclusion = "переключение превзошло фиксированную смесь"
            elif row.lo_kw > 0:
                conclusion = "фиксированная смесь превзошла переключение"
            else:
                conclusion = "интервал включает ноль; преимущества переключения не установлено"
            report.extend(["", "**Основное сравнение:** " + conclusion + "."])
    report.extend([
        "",
        "Все интервалы получены парным бутстрепом 7-дневных блоков по ошибкам, "
        "усреднённым по seed; турбины одного блока не разрываются. "
        f"Для двух выбранных основных сравнений ({primary} мин, "
        "классификатор со сглаживанием против постоянной смеси со сглаживанием) "
        "использованы интервалы 97,5%; прочие 95% интервалы — описательные. "
        "Оценка относится к двум паркам и одному календарному периоду.",
        "",
        "Скорость ветра берётся только из прошлого. Добавление архива NWP, "
        "распознавание направления изменения и проверка наступления или времени "
        "отдельного события выходят за рамки эксперимента. Предупреждение "
        "о крупном изменении не доказывает, что GRU правильно оценивает размер "
        "этого изменения; это проверяется отдельно по общему прогнозу.",
        "",
        "Параметры и метрики: `policy_parameters.json`, `event_metrics.csv`, "
        "`forecast_metrics.csv`, `paired_comparisons.csv`. "
        "Прогнозы, вероятности и решения по каждому окну записаны в `predictions/`.",
        "",
        "Метод оценки событий: [IEA Wind, Forecast Solution Evaluation]"
        "(https://iea-wind.org/wp-content/uploads/2022/07/IEAWind_Task36_Recommended_Practice_Part3_2nd_Edition_public.pdf). "
        "Сглаживание: [Hyndman и Athanasopoulos]"
        "(https://otexts.com/fpp3/exponential-smoothing.html#simple-exponential-smoothing).",
        "",
    ])
    destination = Path(config.output) / "report.md"
    temp = destination.with_suffix(".md.tmp")
    temp.write_text("\n".join(report), encoding="utf-8")
    temp.replace(destination)

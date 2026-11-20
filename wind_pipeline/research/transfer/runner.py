from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd
import numpy as np

from ..artifacts import file_hash, object_hash, write_json
from .config import load_transfer_config
from .data import FarmSeries, write_grid
from .scada import fetch_scada, metadata_and_coordinates, rebuild_farm
from .weather import HresArchive


def hashes(directory, files):
    return {name: file_hash(directory / name) for name in files}


def prepare_identity(config):
    module = Path(__file__).resolve().parent
    return {'config_hash': object_hash({'transfer': config.payload, 'cubico_archives': config.archives,
                                         'raw_directory': config.cubico['raw_sources']['directory'],
                                         'raw_availability': config.cubico['raw_sources']['availability_minimum']}),
            'preparation_code': hashes(module, ['data.py', 'scada.py', 'weather.py'])}


def verify_prepared(config):
    path = config.output / 'prepared.json'
    if not path.is_file():
        raise FileNotFoundError('Prepared data is missing; run research --step prepare')
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved['identity'] != prepare_identity(config):
        raise ValueError('Preparation inputs changed; use a different output_dir for the modified experiment')
    if saved['files'] != hashes(config.output, saved['files']):
        raise ValueError('Prepared files were modified or damaged')
    return saved


def weather_archive(config, coords):
    settings = config['weather']
    pinned = settings['locations']
    if set(pinned) != set(coords) or any(not np.allclose(pinned[farm], coords[farm], atol=0.0001, rtol=0)
                                         for farm in coords):
        raise ValueError('IFS point coordinates must agree with the pinned turbine inventories')
    return HresArchive(settings, config.root / settings['cache'], pinned)


def fetch(config):
    print('Loading source data', flush=True)
    fetch_scada(config)
    if config['weather']:
        _, coords = metadata_and_coordinates(config)
        weather_archive(config, coords).fetch(config['weather']['start'], config['test_end'])


def prepare(config):
    if (config.output / 'prepared.json').exists():
        prepared = verify_prepared(config)
        print('Using prepared datasets', flush=True)
        return prepared
    print('Preparing time series and forecast features', flush=True)
    config.output.mkdir(parents=True, exist_ok=True)
    inventory, coords = metadata_and_coordinates(config)
    forecast = None
    produced = ['scada_coverage.csv', 'signal_coverage.csv', 'raw_sources.json', 'config.resolved.json']
    if config['weather']:
        archive = weather_archive(config, coords)
        forecast, chunks = archive.station_table(config['weather']['start'], config['test_end'])
        forecast.to_parquet(config.output / 'ifs_forecasts.parquet', index=False)
        write_json(config.output / 'ifs_manifest.json', {'source': config['weather'], 'locations': archive.locations,
                    'attribution': 'Data from ECMWF via WeatherBench 2', 'chunks': chunks})
        produced.extend(['ifs_forecasts.parquet', 'ifs_manifest.json'])

    audits, sources, coverage = [], [], []
    for record in config.archives:
        farm = record['farm']
        directory = config.output / 'prepared' / farm
        frame, source, counts = rebuild_farm(config, record, inventory[farm])
        audits.append(write_grid(config, farm, frame, inventory[farm], forecast, directory))
        sources.extend(source)
        coverage.extend(counts)
        produced.extend(str(p.relative_to(config.output)) for p in directory.glob('*') if p.is_file())
        print(f'Prepared {farm}: {len(frame):,} SCADA observations', flush=True)
        del frame
        gc.collect()
    pd.DataFrame(coverage).to_csv(config.output / 'scada_coverage.csv', index=False)
    pd.DataFrame(audits).to_csv(config.output / 'signal_coverage.csv', index=False)
    write_json(config.output / 'raw_sources.json', sources)
    write_json(config.output / 'config.resolved.json', config.payload)
    prepared = {'identity': prepare_identity(config), 'files': hashes(config.output, sorted(produced)), 'farms': audits}
    write_json(config.output / 'prepared.json', prepared)
    return prepared


def training_code():
    return hashes(Path(__file__).resolve().parent, ['config.py', 'models.py', 'experiment.py'])


def training_identity(config, prepared):
    versions = {name: importlib.metadata.version(name) for name in (
        'numpy', 'pandas', 'torch', 'catboost', 'pyarrow', 'scikit-learn', 'matplotlib', 'requests', 'numcodecs')}
    return {'prepared': object_hash(prepared), 'code': training_code(),
            'python': platform.python_version(), 'packages': versions, 'device': config.device}


def saved_training_identity(config, prepared):
    saved = json.loads((config.output / 'training.json').read_text(encoding='utf-8'))['identity']
    if saved['prepared'] != object_hash(prepared) or saved['code'] != training_code():
        raise ValueError('Saved training does not match the prepared data or current training code')
    return saved


def verify_run(config, prepared):
    from .verification import verify_evaluation, verify_report, verify_training

    training = verify_training(config, saved_training_identity(config, prepared))
    evaluation = verify_evaluation(config, training)
    verify_report(config, evaluation)
    print(f'Verified {len(training["models"])} final models and {len(evaluation["predictions"])} prediction sets', flush=True)


def run_all(config, prepared):
    from .analysis import report
    from .experiment import evaluate, train
    from .verification import verify_evaluation, verify_report, verify_training

    identity = training_identity(config, prepared)
    if (config.output / 'training.json').is_file():
        training = verify_training(config, identity)
        print('Using saved forecast models', flush=True)
    else:
        print('Training and adapting forecast models', flush=True)
        train(config, {farm: FarmSeries.load(config, farm) for farm in config.farms}, identity)
        training = verify_training(config, identity)

    try:
        evaluation = verify_evaluation(config, training)
    except (FileNotFoundError, ValueError):
        print(f'Generating forecasts for {config.external}', flush=True)
        evaluate(config, FarmSeries.load(config, config.external), identity)
        evaluation = verify_evaluation(config, training)
    else:
        print(f'Using saved forecasts for {config.external}', flush=True)

    try:
        verify_report(config, evaluation)
    except (FileNotFoundError, ValueError):
        print('Building comparison tables and report', flush=True)
        report(config, identity)
        verify_report(config, evaluation)
    else:
        print(f"Report: {config.output / 'report.md'}", flush=True)


def run_transfer(path, step='all', sensitivity_config=None):
    config = load_transfer_config(path)
    if step == 'all':
        if not (config.output / 'prepared.json').is_file():
            fetch(config)
        run_all(config, prepare(config))
    elif step == 'fetch':
        fetch(config)
    elif step == 'prepare':
        prepare(config)
    elif step == 'verify':
        verify_run(config, verify_prepared(config))
    elif step in ('report', 'sensitivity'):
        identity = saved_training_identity(config, verify_prepared(config))
        if step == 'sensitivity':
            from .sensitivity import run_sensitivity

            settings_path = sensitivity_config or 'configs/research_transfer_sensitivity.json'
            return run_sensitivity(config, identity, json.loads(Path(settings_path).read_text(encoding='utf-8')))

        from .analysis import report

        report(config, identity)
    elif step in ('train', 'evaluate'):
        prepared = verify_prepared(config)
        identity = training_identity(config, prepared)
        if step == 'train':
            from .experiment import train

            train(config, {farm: FarmSeries.load(config, farm) for farm in config.farms}, identity)
        else:
            from .experiment import evaluate

            evaluate(config, FarmSeries.load(config, config.external), identity)
    return config.output

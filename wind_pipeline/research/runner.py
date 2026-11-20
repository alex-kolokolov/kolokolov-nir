from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..research_data import FarmData, RAW_COLUMNS, prepare_farm
from .artifacts import (code_hash, environment, file_hash, log_event, model_path,
                        object_hash, verify_run, write_json)
from .config import ResearchConfig, load_research_config


def prepare(config: ResearchConfig):
    config.output.mkdir(parents=True, exist_ok=True)
    if (config.output / 'manifest.json').exists():
        verify_run(config)
        print('Prepared data verified; reusing cache', flush=True)
        return
    from .acquisition import ingest
    ingest(config)
    raw = pd.read_parquet(config.source, columns=list(RAW_COLUMNS))
    prepared, splits = {}, []
    for farm in config['farms']:
        data = prepare_farm(raw, farm, rated_kw=config['rated_power_kw'],
                            lookback=config['lookback'], horizons=config['horizons'],
                            features=config['features'], preprocessing=config['preprocessing'])
        directory = config.output / 'prepared' / farm
        directory.mkdir(parents=True, exist_ok=True)
        # Only raw identity/time and numerical inputs are cached; no old target is copied.
        data.frame[['timestamp', 'farm', 'turbine_id']].to_parquet(directory/'identity.parquet', index=False)
        np.save(directory/'values.npy', data.values)
        np.save(directory/'anchors.npy', data.anchors)
        item = dict(audit=data.audit, files={name: file_hash(directory/name)
                    for name in ('identity.parquet', 'values.npy', 'anchors.npy')})
        prepared[farm] = item
        periods = [('train', config['train_start'], config['train_end'], config['train_stride']),
                   ('validation', config['train_end'], config['validation_end'], config['evaluation_stride']),
                   ('test', config['test_start'], config['test_end'], config['evaluation_stride'])]
        periods += [(f'adaptation_{days}d', config['adaptation_start'],
                     str(pd.Timestamp(config['adaptation_start'])+pd.Timedelta(days=days)),
                     config['evaluation_stride']) for days in config['adaptation_days']]
        for name, start, end, stride in periods:
            selection = data.select(start, end, stride)
            if len(selection) < 2:
                raise ValueError(f'Insufficient windows: {farm}, {name}')
            splits.append(dict(farm=farm, split=name, start=start, end=end, stride=stride,
                               windows=len(selection), first_anchor=str(selection.times.min()),
                               last_anchor=str(selection.times.max()), turbines=len(set(selection.turbines))))
        print(f'{farm}: {data.audit}', flush=True)
    manifest = dict(config=config.payload, config_hash=object_hash(config.payload),
                    code_hash=code_hash(config), source_sha256=file_hash(config.source),
                    environment=environment(), prepared=prepared,
                    target_definition='exported generation: clip negative power to zero; retain physical overshoot up to configured bound',
                    interval_policy='full context and every future target lie within the named split',
                    raw_manifest=json.loads((config.output/'raw_manifest.json').read_text(encoding='utf-8')),
                    source_definition='rebuilt from original version-pinned Cubico CSV archives; no legacy parquet input')
    manifest['run_fingerprint'] = object_hash({k: manifest[k] for k in
        ('config_hash', 'code_hash', 'source_sha256', 'environment')})
    pd.DataFrame(splits).to_csv(config.output/'splits.csv', index=False)
    write_json(config.output/'manifest.json', manifest)
    write_json(config.output/'config.resolved.json', config.payload)
    packages = manifest['environment']['packages']
    (config.output/'requirements.lock.txt').write_text(
        '\n'.join(f'{p}=={v}' for p,v in packages.items())+'\n', encoding='utf-8')
    # Save the actual source used in the run, including configuration and tests.
    for path in sorted((config.root/'wind_pipeline').rglob('*.py')) + [config.root/'main.py']:
        destination = config.output/'source_snapshot'/path.relative_to(config.root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for path in (config.root/'tests').glob('test_research*.py'):
        destination = config.output/'source_snapshot'/'tests'/path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    log_event(config, stage='prepare', completed=True)


def load_farms(config, manifest):
    result = {}
    for farm in config['farms']:
        directory = config.output/'prepared'/farm
        result[farm] = FarmData(farm=farm, frame=pd.read_parquet(directory/'identity.parquet'),
            values=np.load(directory/'values.npy', mmap_mode='r'),
            anchors=np.load(directory/'anchors.npy'), horizons=tuple(config['horizons']),
            lookback=config['lookback'], audit=manifest['prepared'][farm]['audit'])
    return result


def train(config):
    from .training import fit_catboost, fit_gru
    manifest = verify_run(config)
    farms = load_farms(config, manifest)
    fingerprint = manifest['run_fingerprint']
    for source in config['farms']:
        other = next(f for f in config['farms'] if f != source)
        training = farms[source].select(config['train_start'], config['train_end'], config['train_stride'])
        validation = farms[source].select(config['train_end'], config['validation_end'], config['evaluation_stride'])
        for seed in config['seeds']:
            print(f'\nSOURCE {source}, seed {seed}', flush=True)
            if 'catboost' in config['methods']:
                fit_catboost(config, training, validation,
                    stem=model_path(config, source, source, 'catboost_source', seed),
                    seed=seed, fingerprint=fingerprint)
            if 'gru' in config['methods']:
                fit_gru(config, training, validation,
                    stem=model_path(config, source, source, 'gru_source', seed),
                    seed=seed, fingerprint=fingerprint)
            for days in config['adaptation_days']:
                end = str(pd.Timestamp(config['adaptation_start'])+pd.Timedelta(days=days))
                local = farms[other].select(config['adaptation_start'], end, config['evaluation_stride'])
                if 'catboost' in config['methods']:
                    fit_catboost(config, local, None,
                        stem=model_path(config, source, other, 'catboost_local', seed, days),
                        seed=seed, fingerprint=fingerprint)
                if 'gru' in config['methods']:
                    fit_gru(config, local, None,
                        stem=model_path(config, source, other, 'gru_scratch', seed, days),
                        seed=seed, fingerprint=fingerprint)
                    fit_gru(config, local, None,
                        stem=model_path(config, source, other, 'gru_finetune', seed, days),
                        initial_checkpoint=model_path(config, source, source, 'gru_source', seed).with_suffix('.pt'),
                        seed=seed, fingerprint=fingerprint)
    log_event(config, stage='train', completed=True)


def run_research(path: str | Path, step='all'):
    config = load_research_config(path)
    if step in ('all', 'fetch'):
        from .acquisition import fetch
        fetch(config)
    if step in ('all', 'prepare'):
        prepare(config)
    if step in ('all', 'train'):
        train(config)
    if step in ('all', 'evaluate'):
        from .evaluation import evaluate
        evaluate(config)
    if step in ('all', 'report'):
        from .report import generate_report
        generate_report(config)
    return config.output

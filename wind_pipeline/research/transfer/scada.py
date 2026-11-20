"""Read original Cubico and RES SCADA into the same UTC end-of-interval schema."""
from __future__ import annotations

import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ..acquisition import download_one, md5, normalize_column
from ..artifacts import file_hash

TIME = 'timestamp'
TURBINE = 'turbine_id'
KEY = [TIME, TURBINE]
SCADA = ('power_kw', 'wind_ms', 'wind_direction_deg', 'rotor_rpm', 'ambient_c', 'nacelle_c',
         'wind_sd_ms', 'yaw_deg', 'pitch_deg', 'setpoint_kw', 'running_fraction',
         'downtime_loss_kwh', 'curtailment_loss_kwh')

CUBICO = dict(zip(SCADA, (
    'Power (kW)', 'Wind speed (m/s)', 'Wind direction (°)', 'Rotor speed (rpm)',
    'Ambient temperature (converter) (°C)', 'Nacelle temperature (°C)',
    'Wind speed, Standard deviation (m/s)', 'Nacelle position (°)', 'Blade angle (pitch position) A (°)',
    'Turbine Power setpoint (kW)', '', 'Lost Production to Downtime (kWh)',
    'Lost Production to Curtailment (Total) (kWh)',
), strict=True))

SIEMENS = {
    'tblSCTurGrid': {'power_kw': 'wtc_ActPower_mean'},
    'tblSCTurbine': {'wind_ms': 'wtc_AcWindSp_mean', 'wind_sd_ms': 'wtc_AcWindSp_stddev',
                    'wind_direction_deg': 'wtc_ActualWindDirection_mean', 'rotor_rpm': 'wtc_MainSRpm_mean',
                    'yaw_deg': 'wtc_NacelPos_mean', 'pitch_deg': 'wtc_PitcPosA_mean',
                    'setpoint_kw': 'wtc_PowerRef_endvalue'},
    'tblSCTurTemp': {'ambient_c': 'wtc_AmbieTmp_mean', 'nacelle_c': 'wtc_NacelTmp_mean'},
    'tblSCTurFlag': {'running_fraction': 'wtc_ScInOper_timeon'},
}


def static_metadata(config, record):
    files = [f for f in record['files'] if f['name'].lower().endswith('.csv')]
    if len(files) != 1:
        raise ValueError(f'Expected one turbine inventory for {record["farm"]}')
    path = config.raw_directory / record['farm'] / files[0]['name']
    if md5(path) != files[0]['md5']:
        raise ValueError(f'Turbine inventory checksum mismatch: {path}')
    frame = pd.read_csv(path, encoding='utf-8-sig')
    if record['farm'] == config.external:
        ids = frame['Turbine Name'].str.extract(r'^T(\d+)$', expand=False).astype(int)
        stations = frame['Station ID'].astype(int)
    else:
        frame = frame.dropna(how='all').reset_index(drop=True)
        ids = frame['Title'].str.strip().str.extract(r'\s(\d+)$', expand=False).astype(int)
        stations = ids
    data = pd.DataFrame({TURBINE: [f'{record["farm"]}_T{i:02d}' for i in ids], 'station': stations,
                         'latitude': frame.Latitude, 'longitude': frame.Longitude,
                         'rated_kw': frame['Rated power (kW)'], 'model': frame.Model})
    if (data[TURBINE].duplicated().any() or data.station.duplicated().any()
            or not np.isfinite(data[['latitude', 'longitude', 'rated_kw']]).all().all()
            or (data.rated_kw <= 0).any()):
        raise ValueError(f'Invalid turbine metadata: {path}')
    return data.sort_values(TURBINE).reset_index(drop=True)


def metadata_and_coordinates(config):
    meta = {record['farm']: static_metadata(config, record) for record in config.archives}
    coords = {farm: tuple(frame[['latitude', 'longitude']].mean()) for farm, frame in meta.items()}
    return meta, coords


def fetch_scada(config):
    # Reuse version-pinned, resumable Zenodo acquisition from research v1.
    class DownloadConfig:
        root = config.root

        def __getitem__(self, key):
            return config.cubico[key]

    jobs = [(record, item) for record in config.archives for item in record['files']]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda pair: download_one(DownloadConfig(), *pair), jobs))


def read_cubico(stream, farm, member, timestamp_offset, availability_minimum):
    text = io.TextIOWrapper(stream, encoding='utf-8-sig')
    for index, line in enumerate(text):
        if index > 1000:
            raise ValueError(f'Cannot locate Cubico header in {member}')
        if line.lower().startswith('# time zone:') and line.split(':', 1)[1].strip() != 'UTC':
            raise ValueError(f'Unexpected Cubico timezone in {member}')
        header = next(csv.reader([line]))
        known = {normalize_column(value): value for value in header}
        if 'date_and_time' in known and 'data_availability' in known:
            break
    else:
        raise ValueError(f'Cannot locate Cubico header in {member}')
    mapping = {known[normalize_column(label)]: name for name, label in CUBICO.items()
               if label and normalize_column(label) in known}
    if 'power_kw' not in mapping.values():
        raise ValueError(f'Missing power column in {member}')
    mapping[known['date_and_time']] = TIME
    mapping[known['data_availability']] = 'availability'
    frame = pd.read_csv(text, names=header, usecols=list(mapping), low_memory=False).rename(columns=mapping)
    match = re.fullmatch(rf'Turbine_Data_{re.escape(farm)}_0*(\d+)_.*\.csv', Path(member).name, re.I)
    if not match:
        raise ValueError(f'Cannot identify Cubico turbine in {member}')
    frame[TURBINE] = f'{farm}_T{int(match[1]):02d}'
    availability = pd.to_numeric(frame.pop('availability'), errors='coerce')
    keep = availability.ge(availability_minimum)
    for name in set(SCADA) & set(frame.columns):
        frame[name] = pd.to_numeric(frame[name], errors='coerce').where(keep).astype('float32')
    frame[TIME] = (pd.to_datetime(frame[TIME], utc=True, errors='raise').dt.tz_localize(None)
                   + pd.Timedelta(minutes=timestamp_offset))
    return frame


def read_siemens(stream, fields, inventory):
    keep = {'TimeStamp', 'StationId', *fields.values()}
    frames = pd.read_csv(stream, usecols=lambda name: name in keep, encoding='utf-8-sig', low_memory=False)
    frames = frames.rename(columns={v: k for k, v in fields.items()})
    frames[TIME] = pd.to_datetime(frames.pop('TimeStamp'), utc=True, errors='raise').dt.tz_localize(None)
    frames[TURBINE] = frames.pop('StationId').map(inventory.set_index('station')[TURBINE])
    frames = frames.dropna(subset=[TURBINE])
    for name in set(fields) & set(frames.columns):
        series = pd.to_numeric(frames[name], errors='coerce').astype('float32')
        frames[name] = series / 600 if name == 'running_fraction' else series
    return unique_observations(frames, 'Siemens table')


def unique_observations(frame, label):
    frame = frame.drop_duplicates()
    if frame.duplicated(KEY).any():
        raise ValueError(f'Conflicting duplicate turbine timestamps in {label}')
    return frame


def siemens_months(archive, inventory, start, end):
    members = {}
    for name in archive.namelist():
        match = re.fullmatch(r'(tblSC\w+)_(\d{4})_(\d{2})\.csv', Path(name).name)
        if match and match[1] in SIEMENS:
            month = pd.Timestamp(year=int(match[2]), month=int(match[3]), day=1)
            if month < end and month + pd.offsets.MonthBegin(1) > start:
                members.setdefault(month, {})[match[1]] = name
    for month, tables in sorted(members.items()):
        if 'tblSCTurGrid' not in tables:
            raise ValueError(f'Missing Siemens grid table for {month:%Y-%m}')
        frame = None
        for table, name in sorted(tables.items(), key=lambda row: row[0] != 'tblSCTurGrid'):
            with archive.open(name) as stream:
                selected = read_siemens(stream, SIEMENS[table], inventory)
            frame = selected if frame is None else frame.merge(selected, on=KEY, how='left', validate='1:1')
        if 'power_kw' not in frame:
            raise ValueError(f'Missing Siemens power in {month:%Y-%m}')
        yield f'{month:%Y-%m}', frame


def rebuild_farm(config, record, inventory):
    farm = record['farm']
    start = pd.Timestamp(config['adaptation_start'] if farm == config.external else config['train_start'])
    end = pd.Timestamp(config['test_end'] if farm == config.external else config['adaptation_start'])
    blocks, sources, coverage = [], [], []
    for item in record['files']:
        path = config.raw_directory / farm / item['name']
        if md5(path) != item['md5']:
            raise ValueError(f'SCADA checksum mismatch: {path}')
        sources.append({'farm': farm, 'record_id': record['record_id'], 'filename': path.name,
                        'md5': item['md5'], 'sha256': file_hash(path)})
        if path.suffix.lower() != '.zip':
            continue
        if farm != config.external and int(re.search(r'_SCADA_(\d{4})_', path.name)[1]) > end.year:
            continue
        with ZipFile(path) as archive:
            if farm == config.external:
                selection = siemens_months(archive, inventory, start, end)
            else:
                def cubico_members():
                    for member in archive.namelist():
                        if Path(member).name.lower().startswith('turbine_data_') and member.lower().endswith('.csv'):
                            with archive.open(member) as stream:
                                yield member, read_cubico(stream, farm, member,
                                    config['cubico_timestamp_offset_minutes'],
                                    config.cubico['raw_sources']['availability_minimum'])

                selection = cubico_members()
            for member, frame in selection:
                frame = frame.loc[frame[TIME].ge(start) & frame[TIME].lt(end), [*KEY, *[c for c in SCADA if c in frame]]]
                if len(frame):
                    blocks.append(frame)
                    coverage.append({'farm': farm, 'member': member, 'rows': len(frame),
                                     **{c: int(frame[c].count()) if c in frame else 0 for c in SCADA}})
        print(f'{farm}: read {item["name"]}', flush=True)
    if not blocks:
        raise ValueError(f'No original observations within configured dates for {farm}')
    frame = unique_observations(pd.concat(blocks, ignore_index=True), farm)
    frame = frame.reindex(columns=[*KEY, *SCADA]).sort_values(KEY).reset_index(drop=True)
    farm_turbines = set(frame[TURBINE])
    if farm_turbines != set(inventory[TURBINE]):
        raise ValueError(f'Turbine inventory and observations disagree for {farm}: '
                         f'{sorted(farm_turbines ^ set(inventory[TURBINE]))}')
    if (frame[TIME] != frame[TIME].dt.floor('10min')).any():
        raise ValueError(f'{farm} observations are off the ten-minute grid')
    return frame, sources, coverage
